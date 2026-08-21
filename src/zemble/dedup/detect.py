"""Scanning a workspace and turning its units into ranked clone classes."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pathspec import GitIgnoreSpec

from zemble.dedup.homes import judge_classes
from zemble.dedup.ignore import apply_ignores, find_ignore_files
from zemble.dedup.model import CloneClass, CloneKind, DupeReport, Lane, PairReason, Unit
from zemble.dedup.structure import check_pair
from zemble.dedup.units import extract_units
from zemble.embedding.base import Embedder
from zemble.index.file_walker import walk_files
from zemble.parallel import pool_context

_WORKER_CHUNK = 60
_MAX_FILE_BYTES = 2_000_000
#: How many pair reasons a logic class keeps; a class of 20 members has 190 pairs.
_MAX_REASONS = 6


@dataclass(frozen=True, slots=True)
class DupeOptions:
    """Everything a duplication run can be told to do."""

    kinds: tuple[CloneKind, ...] = (CloneKind.EXACT, CloneKind.RENAMED)
    min_tokens: int = 30
    min_statements: int = 6
    windows: bool = True
    min_files: int = 1
    logic_threshold: float = 0.92
    logic_top_k: int = 10
    embedder: str | None = None
    paths: tuple[str, ...] = ()
    #: Gitignore-style patterns, relative to the root, dropped before anything is parsed.
    exclude: tuple[str, ...] = ()
    #: Which lane to report; None reports production, mixed and test sections in that order.
    lane: Lane | None = None
    #: Whether `<root>/.zemble/dupes.ignore` is honoured.
    use_ignore_file: bool = True
    jobs: int | None = None

    @property
    def wants_logic(self) -> bool:
        """Whether this run has to embed anything."""
        return CloneKind.LOGIC in self.kinds


@dataclass
class _Scan:
    """The Java files one run will read."""

    jobs: list[tuple[str, str]] = field(default_factory=list)


def _excluder(patterns: Sequence[str]) -> GitIgnoreSpec | None:
    """Compile --exclude patterns into one gitignore spec matched against root-relative paths."""
    return GitIgnoreSpec.from_lines(patterns) if patterns else None


def _scan(root: Path, paths: Sequence[str], exclude: Sequence[str] = ()) -> _Scan:
    """Walk the workspace for Java files, honouring a path restriction and exclude patterns."""
    scan = _Scan()
    selected = [Path(path).resolve() for path in paths]
    excluded = _excluder(exclude)
    for file_path in walk_files(root, extensions=[".java"]):
        if selected and not any(_is_under(file_path, choice) for choice in selected):
            continue
        if excluded is not None and excluded.match_file(file_path.relative_to(root).as_posix()):
            continue
        try:
            if file_path.stat().st_size > _MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        scan.jobs.append((str(file_path), file_path.relative_to(root).as_posix()))
    scan.jobs.sort(key=lambda job: job[1])
    return scan


def _is_under(file_path: Path, choice: Path) -> bool:
    """Whether a file is the given path or lives under it."""
    return file_path == choice or choice in file_path.parents


def _extract_batch(payload: tuple[list[tuple[str, str]], dict[str, object]]) -> list[Unit]:
    """Worker entry point: extract the units of a batch of files."""
    jobs, options = payload
    units: list[Unit] = []
    for absolute, relative in jobs:
        try:
            source = Path(absolute).read_bytes()
        except OSError:
            continue
        try:
            units.extend(extract_units(source, relative, **options))  # type: ignore[arg-type]
        except Exception:  # a single unparseable file never fails the run
            continue
    return units


def _batches(jobs: list[tuple[str, str]], size: int) -> Iterator[list[tuple[str, str]]]:
    """Split the file list into worker-sized batches."""
    for start in range(0, len(jobs), size):
        yield jobs[start : start + size]


def collect_units(root: Path, options: DupeOptions) -> tuple[list[Unit], int]:
    """Extract every unit under a root.

    :param root: The workspace directory.
    :param options: The run's options.
    :return: The units and the number of files read.
    """
    scan = _scan(root, options.paths, options.exclude)
    extract_options: dict[str, object] = {
        "min_tokens": options.min_tokens,
        "min_statements": options.min_statements,
        "windows": options.windows,
        "include_text": options.wants_logic,
    }
    workers = options.jobs if options.jobs is not None else min(os.cpu_count() or 1, 8)
    batches = list(_batches(scan.jobs, _WORKER_CHUNK))
    units: list[Unit] = []
    if workers <= 1 or len(batches) <= 1:
        for batch in batches:
            units.extend(_extract_batch((batch, extract_options)))
        return units, len(scan.jobs)
    context = pool_context()
    if context is None:
        # No start method is safe here (see zemble.parallel): extract in this process.
        for batch in batches:
            units.extend(_extract_batch((batch, extract_options)))
        return units, len(scan.jobs)
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for result in pool.map(_extract_batch, ((batch, extract_options) for batch in batches)):
            units.extend(result)
    return units, len(scan.jobs)


# ---- clone classes -------------------------------------------------------


def _group(units: Iterable[Unit], key: str) -> list[list[Unit]]:
    """Bucket units by one hash attribute, keeping only buckets with more than one member."""
    buckets: dict[str, list[Unit]] = defaultdict(list)
    for unit in units:
        buckets[getattr(unit, key)].append(unit)
    return [members for members in buckets.values() if len(members) > 1]


def _clone_class(kind: CloneKind, members: Sequence[Unit], reasons: Sequence[PairReason] = ()) -> CloneClass:
    """Build a clone class from its members, ordered by location."""
    ordered = tuple(sorted(members, key=lambda unit: (unit.file_path, unit.start_line)))
    return CloneClass(
        kind=kind,
        members=ordered,
        tokens=min(unit.token_count for unit in ordered),
        reasons=tuple(reasons),
    )


def exact_classes(units: Sequence[Unit]) -> list[CloneClass]:
    """Build the exact clone classes: identical token streams, comments and whitespace gone."""
    return [_clone_class(CloneKind.EXACT, members) for members in _group(units, "exact_hash")]


def renamed_classes(units: Sequence[Unit]) -> list[CloneClass]:
    """Build the alpha-renamed clone classes.

    A group whose members all share one exact hash is pure exact duplication and is reported
    under `exact` alone; a renamed class must span at least two distinct exact streams.

    :param units: The units to group.
    :return: The renamed clone classes.
    """
    classes = []
    for members in _group(units, "renamed_hash"):
        if len({member.exact_hash for member in members}) < 2:
            continue
        classes.append(_clone_class(CloneKind.RENAMED, members))
    return classes


class _UnionFind:
    """Disjoint sets over unit indices."""

    def __init__(self, size: int) -> None:
        """Start with every index in its own set."""
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        """Return the representative of an index's set."""
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        """Merge two sets."""
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _logic_candidates(units: Sequence[Unit]) -> list[Unit]:
    """Keep the units logic mode may compare: whole bodies that carry their source text."""
    return [unit for unit in units if unit.is_body and unit.text]


def _neighbour_pairs(
    candidates: Sequence[Unit], vectors: np.ndarray, threshold: float, top_k: int
) -> Iterator[tuple[int, int, float]]:
    """Yield index pairs whose embeddings are at least `threshold` cosine-similar."""
    from vicinity.backends.basic import BasicBackend

    backend = BasicBackend.from_vectors(vectors, metric="cosine")
    k = min(top_k + 1, len(candidates))
    seen: set[tuple[int, int]] = set()
    for query, (indices, distances) in enumerate(backend.query(vectors, k)):
        for neighbour, distance in zip(np.asarray(indices).tolist(), np.asarray(distances).tolist()):
            if neighbour == query:
                continue
            similarity = 1.0 - float(distance)
            if similarity < threshold:
                continue
            pair = (min(query, neighbour), max(query, neighbour))
            if pair in seen:
                continue
            seen.add(pair)
            yield pair[0], pair[1], similarity


def logic_classes(
    units: Sequence[Unit], options: DupeOptions, embedder: Embedder | None = None
) -> tuple[list[CloneClass], list[str]]:
    """Build the logic clone classes: embedding candidates that survive the structural check.

    :param units: Every extracted unit; only whole bodies are considered.
    :param options: The run's options, naming the embedder and the similarity threshold.
    :param embedder: An explicit embedder; None loads the configured one.
    :return: The classes and any notes worth printing (timings, skipped work).
    """
    from zemble.embedding.registry import load_embedder

    candidates = _logic_candidates(units)
    notes: list[str] = []
    if len(candidates) < 2:
        return [], notes
    started = time.perf_counter()
    embedder = embedder or load_embedder(options.embedder)
    vectors = embedder.embed_documents([unit.text or "" for unit in candidates])
    embedded = time.perf_counter() - started
    finder = _UnionFind(len(candidates))
    reasons: dict[tuple[int, int], PairReason] = {}
    pairs = 0
    for left, right, _similarity in _neighbour_pairs(candidates, vectors, options.logic_threshold, options.logic_top_k):
        one, other = candidates[left], candidates[right]
        if one.exact_hash == other.exact_hash or one.renamed_hash == other.renamed_hash:
            continue
        verdict = check_pair(one, other)
        if not verdict.accepted:
            continue
        pairs += 1
        finder.union(left, right)
        reasons[(left, right)] = PairReason(left=one.location, right=other.location, reason=verdict.reason)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(candidates)):
        groups[finder.find(index)].append(index)
    classes = []
    for root, indices in groups.items():
        if len(indices) < 2:
            continue
        member_set = set(indices)
        chosen = [reason for (left, right), reason in reasons.items() if left in member_set and right in member_set]
        classes.append(_clone_class(CloneKind.LOGIC, [candidates[index] for index in indices], chosen[:_MAX_REASONS]))
    notes.append(
        f"logic: embedded {len(candidates)} bodies with {embedder.model_id} in {embedded:.1f}s, "
        f"{pairs} pair(s) survived the structural check"
    )
    return classes, notes


# ---- ranking -------------------------------------------------------------


def _covers(bigger: CloneClass, smaller: CloneClass) -> bool:
    """Whether every member of one class sits inside a member of another, equally copied one."""
    if len(bigger.members) < len(smaller.members):
        return False
    for member in smaller.members:
        if not any(
            other.file_path == member.file_path
            and other.start_line <= member.start_line
            and member.end_line <= other.end_line
            for other in bigger.members
        ):
            return False
    return True


def rank(classes: Sequence[CloneClass], min_files: int) -> list[CloneClass]:
    """Sort classes by weight and drop the ones a bigger class already contains.

    Window units make the same copied code appear at every window length; a class whose
    members all sit inside the members of a higher-ranked class says nothing new.

    :param classes: The classes of one kind.
    :param min_files: Smallest number of distinct files a class may span.
    :return: The surviving classes, best first.
    """
    ordered = sorted(classes, key=lambda clone: (-clone.score, clone.members[0].location))
    kept: list[CloneClass] = []
    for clone in ordered:
        if clone.files < min_files:
            continue
        if any(_covers(other, clone) for other in kept):
            continue
        kept.append(clone)
    return kept


def find_duplication(
    root: str | Path, options: DupeOptions | None = None, embedder: Embedder | None = None
) -> DupeReport:
    """Run duplication detection over a workspace.

    :param root: The workspace directory.
    :param options: The run's options, or None for the defaults.
    :param embedder: An explicit embedder for logic mode; None loads the configured one.
    :return: The report, its classes already ranked.
    :raises FileNotFoundError: If the root does not exist.
    """
    options = options or DupeOptions()
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"No such directory: {root}")
    started = time.perf_counter()
    units, files = collect_units(root_path, options)
    report = DupeReport(
        root=str(root_path),
        analyzed_files=files,
        units=len(units),
        body_units=sum(1 for unit in units if unit.is_body),
        min_tokens=options.min_tokens,
        min_statements=options.min_statements,
        kinds=options.kinds,
        lane=options.lane,
    )
    classes: list[CloneClass] = []
    if CloneKind.EXACT in options.kinds:
        classes.extend(rank(exact_classes(units), options.min_files))
    if CloneKind.RENAMED in options.kinds:
        classes.extend(rank(renamed_classes(units), options.min_files))
    if options.wants_logic:
        found, notes = logic_classes(units, options, embedder)
        classes.extend(rank(found, options.min_files))
        report.notes.extend(notes)
    if options.use_ignore_file:
        scanned = [kind.value for kind in options.kinds]
        ignores = find_ignore_files(root_path, (unit.file_path for unit in units))
        suppression = apply_ignores(classes, ignores, scanned)
        classes = list(suppression.kept)
        report.suppressed = list(suppression.suppressed)
        report.ignore_problems = list(suppression.problems)
    if options.lane is not None:
        classes = [clone for clone in classes if clone.lane is options.lane]
        report.suppressed = [clone for clone in report.suppressed if clone.lane is options.lane]
    report.classes = classes
    homes, notes = judge_classes(root_path, classes)
    report.homes = homes
    report.notes.extend(notes)
    report.elapsed_seconds = time.perf_counter() - started
    return report

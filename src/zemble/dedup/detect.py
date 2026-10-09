"""Scanning a workspace and turning its units into ranked clone classes."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import numpy as np

from zemble.dedup.homes import judge_classes
from zemble.dedup.ignore import apply_ignores, find_ignore_files
from zemble.dedup.languages import supported_extensions
from zemble.dedup.model import CloneClass, CloneKind, DupeReport, Lane, PairReason, Unit
from zemble.dedup.structure import MAX_SKELETON_DISTANCE, check_pair
from zemble.dedup.unitcache import (
    VECTOR_FOLDER_NAME,
    FileRow,
    UnitIndex,
    content_digest,
    extraction_signature,
    root_cache_folder,
)
from zemble.dedup.units import extract_units
from zemble.embedding.base import Embedder
from zemble.index.file_walker import compile_ignore, walk_files
from zemble.parallel import pool_context, pooled

_WORKER_CHUNK = 60
#: One file's full extraction: its root-relative path, content digest ("" unreadable) and units (None failed).
_Extracted = tuple[str, str, list[Unit] | None]
_Result = TypeVar("_Result")
#: One unit's two hashes as an index row stores them (`unitcache.HASH_BYTES` per unit).
_HASH_DTYPE = np.dtype([("exact", "S16"), ("renamed", "S16")])
_MAX_FILE_BYTES = 2_000_000
#: How many pair reasons a logic class keeps; a class of 20 members has 190 pairs.
_MAX_REASONS = 6
#: Bodies whose similarity to every body is computed at once (rows x bodies float32): a whole run, a focused one.
_WHOLE_BLOCK = 1024
_FOCUS_BLOCK = 512
#: Similarity margin the reverse scan keeps below the threshold: a pair is confirmed against the other
#: body's own row, and two matrix products of one pair may round apart in the last bits.
_SIMILARITY_SLACK = 1e-4
#: The local vector mirror is swept once it holds this many times the vectors a run reads, plus the floor.
_MIRROR_SLACK = 2
_MIRROR_FLOOR = 10_000
#: How many failing paths the report names before it only counts them.
_MAX_FAILED_EXAMPLES = 5


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
    #: Report only the classes with a member under these paths; the other members may lie anywhere scanned.
    focus: tuple[str, ...] = ()
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
    """The files one run will read, absolute path plus the path the report prints."""

    jobs: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class _Extraction:
    """What reading the workspace produced: units, files read, and files that would not parse."""

    units: list[Unit] = field(default_factory=list)
    files: int = 0
    failed: list[str] = field(default_factory=list)


def _selected_paths(root: Path, paths: Sequence[str]) -> list[Path]:
    """Resolve a `--paths` restriction against the scan root, never against the process CWD.

    An absolute path is taken as given; everything else is root-relative, so a CLI run from
    anywhere and an MCP call that only knows the workspace agree on what `src` means.
    """
    resolved = []
    for raw in paths:
        path = Path(raw)
        resolved.append(path.resolve() if path.is_absolute() else (root / path).resolve())
    return resolved


def _scan(root: Path, paths: Sequence[str], exclude: Sequence[str] = ()) -> _Scan:
    """Walk the workspace for supported files, honouring a path restriction and exclude patterns."""
    scan = _Scan()
    selected = _selected_paths(root, paths)
    excluded = compile_ignore(exclude)
    for file_path in walk_files(root, extensions=supported_extensions()):
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


def _extract_file(absolute: str, relative: str, options: dict[str, object]) -> tuple[str, list[Unit] | None]:
    """Read and extract one file, returning its content digest ("" when unreadable) and its units (None on failure).

    A failure never fails the run, but it is counted and reported: a missing grammar or an
    unreadable file must not read as a clean result.
    """
    try:
        source = Path(absolute).read_bytes()
    except OSError:
        return "", None
    digest = content_digest(source)
    try:
        return digest, extract_units(source, relative, **options)  # type: ignore[arg-type]
    except Exception:
        return digest, None


def _extract_full_batch(payload: tuple[list[tuple[str, str]], dict[str, object]]) -> list[_Extracted]:
    """Worker entry point: every file of a batch with its content digest and full units."""
    jobs, options = payload
    return [(relative, *_extract_file(absolute, relative, options)) for absolute, relative in jobs]


def _extract_row_batch(payload: tuple[list[tuple[str, str]], dict[str, object]]) -> list[FileRow]:
    """Worker entry point: every file of a batch as its compact unit-index row."""
    return [FileRow.of(*result) for result in _extract_full_batch(payload)]


def _batches(jobs: list[tuple[str, str]], size: int) -> Iterator[list[tuple[str, str]]]:
    """Split the file list into worker-sized batches."""
    for start in range(0, len(jobs), size):
        yield jobs[start : start + size]


def _extract_options(options: DupeOptions, *, include_text: bool) -> dict[str, object]:
    """The keyword arguments every file of a run is extracted with."""
    return {
        "min_tokens": options.min_tokens,
        "min_statements": options.min_statements,
        "windows": options.windows,
        "include_text": include_text,
    }


def _run_batches(
    worker: Callable[[tuple[list[tuple[str, str]], dict[str, object]]], list[_Result]],
    jobs: list[tuple[str, str]],
    extract_options: dict[str, object],
    workers: int | None,
) -> Iterator[_Result]:
    """Run a worker entry point over the files, in processes when there is more than one batch, in job order."""
    count = workers if workers is not None else min(os.cpu_count() or 1, 8)
    # A handful of files (a focus, the files sharing its hashes) is still spread over every worker.
    batches = list(_batches(jobs, max(1, min(_WORKER_CHUNK, -(-len(jobs) // count)))))
    context = pool_context() if count > 1 and len(batches) > 1 else None
    if context is None:
        # No start method is safe here (see zemble.parallel): extract in this process.
        for batch in batches:
            yield from worker((batch, extract_options))
        return
    with pooled(count, context) as pool:
        for results in pool.map(worker, ((batch, extract_options) for batch in batches)):
            yield from results


def collect_units(root: Path, options: DupeOptions) -> _Extraction:
    """Extract every unit under a root.

    A logic run also refreshes the root's unit index (`zemble.dedup.unitcache`), so a focused run
    after it starts warm.

    :param root: The workspace directory.
    :param options: The run's options.
    :return: The units, the number of files read and the files that would not parse.
    """
    scan = _scan(root, options.paths, options.exclude)
    extract_options = _extract_options(options, include_text=options.wants_logic)
    extraction = _Extraction(files=len(scan.jobs))
    rows: list[FileRow] = []
    for relative, digest, units in _run_batches(_extract_full_batch, scan.jobs, extract_options, options.jobs):
        if units is None:
            extraction.failed.append(relative)
        else:
            extraction.units.extend(units)
        if options.wants_logic and digest:
            rows.append(FileRow.of(relative, digest, units))
    if rows:
        _refresh_index(root, extract_options, rows, None if options.paths else [job[1] for job in scan.jobs])
    return extraction


def _refresh_index(
    root: Path, extract_options: dict[str, object], rows: list[FileRow], walked: list[str] | None
) -> None:
    """Store fresh rows in the root's unit index, pruning files a whole-root walk no longer found."""
    index = UnitIndex(root, extraction_signature(extract_options))
    try:
        index.store(rows)
        if walked is not None:
            index.retain(walked)
    finally:
        index.close()


@dataclass
class _Indexed:
    """A focused run's view of the workspace: one index row per scanned file, in scan order."""

    rows: list[FileRow]
    files: int
    failed: list[str]
    #: How many rows had to be extracted rather than read from the index.
    extracted: int


def _indexed_scan(root: Path, options: DupeOptions, scan: _Scan, extract_options: dict[str, object]) -> _Indexed:
    """Read every scanned file's row from the unit index, extracting (and storing) only the changed ones."""
    digests: dict[str, str] = {}
    for absolute, relative in scan.jobs:
        try:
            digests[relative] = content_digest(Path(absolute).read_bytes())
        except OSError:
            continue
    index = UnitIndex(root, extraction_signature(extract_options))
    try:
        found = index.load(digests)
        misses = [job for job in scan.jobs if job[1] not in found]
        fresh = list(_run_batches(_extract_row_batch, misses, extract_options, options.jobs))
        index.store(row for row in fresh if row.digest)
        if not options.paths:
            index.retain(relative for _, relative in scan.jobs)
    finally:
        index.close()
    by_path = {**found, **{row.path: row for row in fresh}}
    rows = [by_path[relative] for _, relative in scan.jobs]
    return _Indexed(
        rows=rows,
        files=len(scan.jobs),
        failed=[row.path for row in rows if row.failed],
        extracted=len(fresh),
    )


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


class _CosineSpace:
    """Cosine nearest-neighbour rows over one run's body vectors, the way vicinity's basic backend computes them.

    Whole and focused runs both read their rows here, so the two can only differ where two matrix
    products of the same row round apart.
    """

    def __init__(self, vectors: np.ndarray, top_k: int) -> None:
        """Normalize the stored side once; `top_k` neighbours plus the body itself are kept per row."""
        from vicinity.utils import normalize_or_copy

        self.vectors = vectors
        self.stored = normalize_or_copy(vectors)
        self.k = min(top_k + 1, len(vectors))

    def similarities(self, indices: Sequence[int]) -> np.ndarray:
        """The cosine similarity of each of these bodies to every body, one row each."""
        from vicinity.utils import normalize

        result: np.ndarray = normalize(self.vectors[list(indices)]).dot(self.stored.T)
        return result

    def nearest(self, similarities: np.ndarray) -> Iterator[list[tuple[int, float]]]:
        """Each row's `k` nearest bodies, nearest first, as (body, similarity)."""
        distances = 1 - similarities
        indices = np.argpartition(distances, kth=self.k - 1, axis=1)[:, : self.k]
        ordered = np.take_along_axis(indices, np.argsort(np.take_along_axis(distances, indices, axis=1)), axis=1)
        for row, columns in zip(np.take_along_axis(distances, ordered, axis=1).tolist(), ordered.tolist()):
            yield [(column, 1.0 - float(distance)) for column, distance in zip(columns, row)]


def _neighbour_pairs(
    candidates: Sequence[Unit], vectors: np.ndarray, threshold: float, top_k: int
) -> Iterator[tuple[int, int, float]]:
    """Yield index pairs whose embeddings are at least `threshold` cosine-similar."""
    space = _CosineSpace(vectors, top_k)
    seen: set[tuple[int, int]] = set()
    for start in range(0, len(candidates), _WHOLE_BLOCK):
        block = range(start, min(start + _WHOLE_BLOCK, len(candidates)))
        for query, row in zip(block, space.nearest(space.similarities(block))):
            for neighbour, similarity in row:
                if neighbour == query or similarity < threshold:
                    continue
                pair = (min(query, neighbour), max(query, neighbour))
                if pair in seen:
                    continue
                seen.add(pair)
                yield pair[0], pair[1], similarity


@dataclass
class _BodyVectors:
    """The vectors of one run's candidate bodies, and what fetching them cost."""

    vectors: np.ndarray
    model_id: str
    seconds: float
    #: How many vectors the local mirror did not hold yet.
    fetched: int


def _body_vectors(
    candidates: Sequence[Unit], options: DupeOptions, embedder: Embedder | None, root: Path | None
) -> _BodyVectors:
    """Vectors for every candidate body, read from the root's local mirror and fetched only where it misses.

    Buying vectors is a paid seam, so the mirror's misses pass the bill guard, which raises
    ``EmbeddingBudgetExceeded`` on a remote embedder whose bill is over budget.

    AIDEV-NOTE: an embedding SERVER keeps no local copy, and reading 57k vectors back from one took
    22 s on 2026-10-09; the per-root mirror (`unitcache.VECTOR_FOLDER_NAME`, the same sqlite store the
    embedding cache uses) is what lets a focused run answer in seconds. An embedder that already
    caches locally is used as it is.
    """
    from zemble.embedding.cache import CachingEmbedder
    from zemble.embedding.pricing import require_affordable_bill
    from zemble.embedding.registry import load_embedder

    started = time.perf_counter()
    embedder = embedder or load_embedder(options.embedder)
    texts = [unit.text or "" for unit in candidates]
    if root is None or isinstance(embedder, CachingEmbedder):
        # The default embedder is local, where the guard is a no-op.
        require_affordable_bill(embedder, texts)
        vectors = embedder.embed_documents(texts)
        return _BodyVectors(vectors, embedder.model_id, time.perf_counter() - started, len(texts))
    folder = root_cache_folder(root) / VECTOR_FOLDER_NAME
    mirror = CachingEmbedder(embedder, embedder.model_id, folder)
    try:
        missing = mirror.cache.pending(texts, embedder.dimensions)
        require_affordable_bill(embedder, missing)
        vectors = mirror.embed_documents(texts)
        oversized = mirror.cache.count() > _MIRROR_SLACK * len(set(texts)) + _MIRROR_FLOOR
    finally:
        mirror.cache.close()
    if oversized:
        from zemble.embedding.gc import collect_unused

        # Every vector this run read was stamped today, so a one-day grace keeps them all.
        collect_unused(folder, grace_days=1, dry_run=False)
    return _BodyVectors(vectors, embedder.model_id, time.perf_counter() - started, len(missing))


def _accepted(one: Unit, other: Unit) -> PairReason | None:
    """The reason a pair is a logic clone, or None: not an exact or renamed match, and structurally close."""
    if one.exact_hash == other.exact_hash or one.renamed_hash == other.renamed_hash:
        return None
    verdict = check_pair(one, other)
    return PairReason(left=one.location, right=other.location, reason=verdict.reason) if verdict.accepted else None


def _logic_groups(
    candidates: Sequence[Unit], members: Iterable[int], reasons: dict[tuple[int, int], PairReason]
) -> list[CloneClass]:
    """Union the accepted pairs into classes; each keeps its first pair reasons in the order they were found."""
    nodes = sorted(set(members))
    position = {index: offset for offset, index in enumerate(nodes)}
    finder = _UnionFind(len(nodes))
    for left, right in reasons:
        finder.union(position[left], position[right])
    groups: dict[int, list[int]] = defaultdict(list)
    for index in nodes:
        groups[finder.find(position[index])].append(index)
    classes = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        member_set = set(indices)
        chosen = [reason for (left, right), reason in reasons.items() if left in member_set and right in member_set]
        classes.append(_clone_class(CloneKind.LOGIC, [candidates[index] for index in indices], chosen[:_MAX_REASONS]))
    return classes


def logic_classes(
    units: Sequence[Unit], options: DupeOptions, embedder: Embedder | None = None, root: Path | None = None
) -> tuple[list[CloneClass], list[str]]:
    """Build the logic clone classes: embedding candidates that survive the structural check.

    :param units: Every extracted unit; only whole bodies are considered.
    :param options: The run's options, naming the embedder and the similarity threshold.
    :param embedder: An explicit embedder; None loads the configured one.
    :param root: The scanned root, whose local vector mirror is read; None embeds without one.
    :return: The classes and any notes worth printing (timings, skipped work).
    """
    candidates = _logic_candidates(units)
    notes: list[str] = []
    if len(candidates) < 2:
        return [], notes
    bodies = _body_vectors(candidates, options, embedder, root)
    reasons: dict[tuple[int, int], PairReason] = {}
    for left, right, _similarity in _neighbour_pairs(
        candidates, bodies.vectors, options.logic_threshold, options.logic_top_k
    ):
        reason = _accepted(candidates[left], candidates[right])
        if reason is not None:
            reasons[(left, right)] = reason
    classes = _logic_groups(candidates, range(len(candidates)), reasons)
    notes.append(
        f"logic: embedded {len(candidates)} bodies with {bodies.model_id} in {bodies.seconds:.1f}s, "
        f"{len(reasons)} pair(s) survived the structural check"
    )
    return classes, notes


class _FocusedNeighbours:
    """The nearest-neighbour rows a focused logic run reads, each computed once.

    A whole run unions every accepted pair of the workspace; a focused one only needs the classes
    holding a focus body, so it walks outward from the focus bodies. A pair is a candidate when
    either side has the other among its `top_k` neighbours at the threshold, so every reached body
    needs both directions: its own row (forward), and every body that has IT in its row (reverse),
    found in the same similarity row and confirmed against that body's own row.

    AIDEV-NOTE: each similarity block streams the whole vector matrix (233 MB on the workspace) and
    small blocks multiply slowly, so a walk step makes ONE pass: the rows of every body the step may
    need next (the bodies to confirm and the next frontier, which is a subset of them) at once.
    """

    def __init__(self, candidates: Sequence[Unit], vectors: np.ndarray, options: DupeOptions) -> None:
        """Prepare the space a whole run would query, over the same vectors."""
        self.threshold = options.logic_threshold
        self.space = _CosineSpace(vectors, options.logic_top_k)
        self.rows: dict[int, list[tuple[int, float]]] = {}
        #: Full similarity rows, kept only for the bodies of the next walk step.
        self.similar: dict[int, np.ndarray] = {}
        self.skeletons = np.array([len(unit.skeleton) for unit in candidates])

    def compute(self, indices: Iterable[int]) -> None:
        """Compute the neighbour row and the full similarity row of each of these bodies."""
        todo = sorted({index for index in indices if index not in self.similar})
        for start in range(0, len(todo), _FOCUS_BLOCK):
            block = todo[start : start + _FOCUS_BLOCK]
            similarities = self.space.similarities(block)
            for row, (index, nearest) in enumerate(zip(block, self.space.nearest(similarities))):
                self.rows[index] = nearest
                self.similar[index] = similarities[row].copy()

    def keep_only(self, indices: set[int]) -> None:
        """Release the similarity rows of every body that is not about to be walked."""
        self.similar = {index: row for index, row in self.similar.items() if index in indices}

    def near(self, index: int) -> list[int]:
        """Every other body similar enough that ITS row might hold this one.

        Bodies whose control flow cannot be within the structural limit are dropped here: the edit
        distance of two skeletons is at least the difference of their lengths.
        """
        near = np.flatnonzero(self.similar[index] >= self.threshold - _SIMILARITY_SLACK)
        near = near[np.abs(self.skeletons[near] - self.skeletons[index]) <= MAX_SKELETON_DISTANCE]
        return [other for other in near.tolist() if other != index]

    def forward(self, index: int) -> list[int]:
        """The bodies this one's own row offers at the threshold."""
        return [other for other, similarity in self.rows[index] if other != index and similarity >= self.threshold]

    def offers(self, holder: int, index: int) -> bool:
        """Whether a body's own row offers another at the threshold."""
        return any(other == index and similarity >= self.threshold for other, similarity in self.rows[holder])

    def found_at(self, left: int, right: int) -> tuple[int, int]:
        """Where a whole run yields a pair first: the earliest query, then the neighbour's rank in it."""
        spots = []
        for holder, other in ((left, right), (right, left)):
            for rank, (neighbour, similarity) in enumerate(self.rows[holder]):
                if neighbour == other and similarity >= self.threshold:
                    spots.append((holder, rank))
        return min(spots)


def focused_logic_classes(
    candidates: Sequence[Unit], focus: Sequence[int], vectors: np.ndarray, options: DupeOptions
) -> tuple[list[CloneClass], int, int]:
    """Build the logic classes holding a focus body, identical to what a whole run reports for them.

    :param candidates: Every candidate body of the workspace, in whole-run order.
    :param focus: Indices of the focus bodies.
    :param vectors: The candidates' vectors, row for row.
    :param options: The run's options.
    :return: The classes, the accepted pairs found and how many bodies the walk reached.
    """
    if len(candidates) < 2 or not focus:
        return [], 0, 0
    space = _FocusedNeighbours(candidates, vectors, options)
    reasons: dict[tuple[int, int], PairReason] = {}
    judged: dict[tuple[int, int], PairReason | None] = {}

    def judge(one: int, other: int) -> PairReason | None:
        pair = (min(one, other), max(one, other))
        if pair not in judged:
            judged[pair] = _accepted(candidates[pair[0]], candidates[pair[1]])
        return judged[pair]

    reached = set(focus)
    frontier = sorted(reached)
    space.compute(frontier)
    while frontier:
        found: set[tuple[int, int]] = set()
        pending: list[tuple[int, int]] = []
        for index in frontier:
            offered = set(space.forward(index))
            found.update((min(index, other), max(index, other)) for other in offered if judge(index, other))
            pending.extend(
                (index, other) for other in space.near(index) if other not in offered and judge(index, other)
            )
        # Every body the next step can reach is in `found` or `pending`: one pass serves both.
        space.compute(({other for pair in found for other in pair} | {other for _, other in pending}) - reached)
        found.update((min(index, other), max(index, other)) for index, other in pending if space.offers(other, index))
        for pair in found:
            reasons[pair] = judge(*pair)  # type: ignore[assignment]
        fresh = {index for pair in found for index in pair} - reached
        space.keep_only(fresh)
        reached |= fresh
        frontier = sorted(fresh)
    ordered = dict(sorted(reasons.items(), key=lambda item: space.found_at(*item[0])))
    return _logic_groups(candidates, reached, ordered), len(ordered), len(reached)


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


@dataclass
class _Found:
    """What one run's scan found, before ignore files, lanes and homes are applied."""

    report: DupeReport
    classes: list[CloneClass]
    #: Root-relative paths of every file that produced a unit: where ignore files are looked for.
    unit_paths: list[str]


def _whole_run(root: Path, options: DupeOptions, embedder: Embedder | None) -> _Found:
    """Compare every scanned unit with every other one."""
    extraction = collect_units(root, options)
    units = extraction.units
    report = _report(root, options, extraction.files, extraction.failed, len(units), sum(u.is_body for u in units))
    classes: list[CloneClass] = []
    if CloneKind.EXACT in options.kinds:
        classes.extend(rank(exact_classes(units), options.min_files))
    if CloneKind.RENAMED in options.kinds:
        classes.extend(rank(renamed_classes(units), options.min_files))
    if options.wants_logic:
        found, notes = logic_classes(units, options, embedder, root)
        classes.extend(rank(found, options.min_files))
        report.notes.extend(notes)
    return _Found(report, classes, [unit.file_path for unit in units])


def _report(root: Path, options: DupeOptions, files: int, failed: Sequence[str], units: int, bodies: int) -> DupeReport:
    """The report header every run shares."""
    return DupeReport(
        root=str(root),
        analyzed_files=files,
        failed_files=len(failed),
        failed_examples=sorted(failed)[:_MAX_FAILED_EXAMPLES],
        supported_extensions=tuple(supported_extensions()),
        units=units,
        body_units=bodies,
        min_tokens=options.min_tokens,
        min_statements=options.min_statements,
        kinds=options.kinds,
        lane=options.lane,
        focus=options.focus,
    )


def _touches(clone: CloneClass, files: set[str]) -> bool:
    """Whether a class has a member in one of these files."""
    return any(member.file_path in files for member in clone.members)


def _literal_universe(
    root: Path, options: DupeOptions, indexed: _Indexed, focus: set[str], extract_options: dict[str, object]
) -> list[Unit]:
    """Every unit sharing an exact or renamed hash that a focus unit holds and some other unit holds too.

    The index rows' hashes decide which hashes repeat, so only the files holding a repeated one
    (focus files included) are parsed again; a clean focus parses nothing.
    """
    rows = [row for row in indexed.rows if row.unit_count]
    if not rows:
        return []
    table = np.frombuffer(b"".join(row.hashes for row in rows), dtype=_HASH_DTYPE)
    owner = np.repeat(np.arange(len(rows)), [row.unit_count for row in rows])
    in_focus = np.array([row.path in focus for row in rows])[owner]
    exact, renamed = _repeated(table["exact"], in_focus), _repeated(table["renamed"], in_focus)
    hit = np.isin(table["exact"], exact) | np.isin(table["renamed"], renamed)
    jobs = [(str(root / rows[index].path), rows[index].path) for index in np.unique(owner[hit]).tolist()]
    exact_hex, renamed_hex = _hex_digests(exact), _hex_digests(renamed)
    return [
        unit
        for *_, units in _run_batches(_extract_full_batch, jobs, extract_options, options.jobs)
        for unit in units or ()
        if unit.exact_hash in exact_hex or unit.renamed_hash in renamed_hex
    ]


def _hex_digests(values: np.ndarray) -> set[str]:
    """Fixed-width digests back as the hex a unit carries.

    AIDEV-NOTE: numpy hands an `S16` value back with its trailing NUL bytes stripped, so a digest
    ending in 00 would come back short and match no unit without the padding.
    """
    return {value.ljust(_HASH_DTYPE["exact"].itemsize, b"\0").hex() for value in values.tolist()}


def _repeated(column: np.ndarray, in_focus: np.ndarray) -> np.ndarray:
    """The hashes a focus unit holds that occur at least twice in the whole column."""
    shared = column[np.isin(column, np.unique(column[in_focus]))]
    values, counts = np.unique(shared, return_counts=True)
    return values[counts >= 2]


def _focus_files(root: Path, focus: Sequence[str], scan: _Scan) -> set[str]:
    """The scanned files under the focus paths, matched by root-relative prefix (a workspace has 10k+ files)."""
    prefixes = []
    for choice in _selected_paths(root, focus):
        if choice == root:
            return {relative for _, relative in scan.jobs}
        if root in choice.parents:
            prefixes.append(choice.relative_to(root).as_posix())
    exact = set(prefixes)
    under = tuple(f"{prefix}/" for prefix in prefixes)
    return {relative for _, relative in scan.jobs if relative in exact or relative.startswith(under)}


def _focused_run(root: Path, options: DupeOptions, embedder: Embedder | None) -> _Found:
    """Report the classes with a member under the focus paths, every other scanned file read from the unit index.

    The answer for those classes is the whole run's: a class's other members may lie anywhere scanned.
    """
    scan = _scan(root, options.paths, options.exclude)
    focus = _focus_files(root, options.focus, scan)
    extract_options = _extract_options(options, include_text=True)
    indexed = _indexed_scan(root, options, scan, extract_options)
    rows = indexed.rows
    bodies = [body for row in rows for body in row.bodies]
    report = _report(root, options, indexed.files, indexed.failed, sum(row.unit_count for row in rows), len(bodies))
    report.notes.append(
        f"focus: {len(focus)} file(s); {indexed.extracted} of {indexed.files} file(s) extracted, "
        "the rest read from the unit index"
    )
    classes: list[CloneClass] = []
    if CloneKind.EXACT in options.kinds or CloneKind.RENAMED in options.kinds:
        universe = _literal_universe(root, options, indexed, focus, extract_options)
        if CloneKind.EXACT in options.kinds:
            classes.extend(rank([c for c in exact_classes(universe) if _touches(c, focus)], options.min_files))
        if CloneKind.RENAMED in options.kinds:
            classes.extend(rank([c for c in renamed_classes(universe) if _touches(c, focus)], options.min_files))
    if options.wants_logic:
        candidates = _logic_candidates(bodies)
        chosen_bodies = [index for index, unit in enumerate(candidates) if unit.file_path in focus]
        if len(candidates) >= 2 and chosen_bodies:
            vectors = _body_vectors(candidates, options, embedder, root)
            started = time.perf_counter()
            found, pairs, reached = focused_logic_classes(candidates, chosen_bodies, vectors.vectors, options)
            classes.extend(rank(found, options.min_files))
            report.notes.append(
                f"logic: read {len(candidates)} body vectors of {vectors.model_id} in {vectors.seconds:.1f}s "
                f"({vectors.fetched} not mirrored yet); from {len(chosen_bodies)} focus bodies the walk reached "
                f"{reached} in {time.perf_counter() - started:.1f}s, {pairs} pair(s) survived the structural check"
            )
    return _Found(report, classes, [row.path for row in rows if row.unit_count])


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
    found = _focused_run(root_path, options, embedder) if options.focus else _whole_run(root_path, options, embedder)
    report, classes = found.report, found.classes
    if options.use_ignore_file:
        scanned = [kind.value for kind in options.kinds]
        ignores = find_ignore_files(root_path, found.unit_paths)
        # A scan that saw only part of the root cannot know that an entry matches nothing anywhere.
        whole = not options.paths and not options.focus
        suppression = apply_ignores(classes, ignores, scanned, judge_stale=whole)
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

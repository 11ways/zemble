"""A lane's index: its base workspace's stores, minus the files the lane changed, plus a small delta.

Nothing of the base is copied. The overlay presents one row space - the base rows of the files the
lane holds unchanged, in base order, then the delta's rows - through the same four store
interfaces a built index has (chunks, BM25, vectors, symbol definitions), so every tool reads an
overlay exactly as it reads any index. Paths are lane-relative, which for a lane is base-relative
too; the overlay's root is the lane, so absolute paths name the lane's files.

AIDEV-NOTE: BM25 scores both segments against ONE corpus - the base's documents plus the delta's -
so a delta row ranks exactly as it would in the base. Base rows the lane hides still count in that
corpus, as rows outside a sub-tree view do: a handful of edited files cannot move the statistics.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import overload

import numpy as np
import numpy.typing as npt
import orjson

from zemble.chunking.capsule import RepoRelativePaths
from zemble.embedding.base import Embedder
from zemble.index.bm25 import BM25, CorpusStats
from zemble.index.chunk_store import ChunkList, file_paths_of, languages_of
from zemble.index.create import PlannedFile, plan_file, reused_file, write_planned
from zemble.index.dense import SelectableBasicBackend
from zemble.index.file_walker import walk_entries
from zemble.index.files import FileStatus, get_extensions, get_file_status
from zemble.index.generation import publish, staging_for, write_metadata
from zemble.index.index import ZembleIndex
from zemble.index.scope import require_declared_scope
from zemble.index.symbols import SymbolDefinitions
from zemble.index.types import FileManifestEntry, PersistencePath, PreviousIndex
from zemble.lane import Lane, LaneComparer
from zemble.types import Chunk

logger = logging.getLogger(__name__)

#: The metadata key naming the base workspace a stored delta overlays.
OVERLAY_OF_KEY = "overlay_of"


class OverlayChunks(Sequence[Chunk]):
    """The chunks of the kept base rows, then the delta's, without copying either."""

    def __init__(self, base: Sequence[Chunk], keep: npt.NDArray[np.int64], delta: Sequence[Chunk] | None) -> None:
        """Present *keep*'s rows of *base* followed by every row of *delta*."""
        self._base = base
        self._keep = keep
        self._delta = delta
        self._split = len(keep)
        self._file_paths: list[str] | None = None
        self._languages: list[str | None] | None = None

    def __len__(self) -> int:
        """The number of visible chunks."""
        return self._split + (len(self._delta) if self._delta is not None else 0)

    @overload
    def __getitem__(self, item: int) -> Chunk: ...

    @overload
    def __getitem__(self, item: slice) -> list[Chunk]: ...

    def __getitem__(self, item: int | slice) -> Chunk | list[Chunk]:
        """Materialize one visible chunk, or a list for a slice."""
        if isinstance(item, slice):
            return [self[index] for index in range(*item.indices(len(self)))]
        if item < 0:
            item += len(self)
        if not 0 <= item < len(self):
            raise IndexError(item)
        if item < self._split:
            return self._base[int(self._keep[item])]
        assert self._delta is not None
        return self._delta[item - self._split]

    def __iter__(self) -> Iterator[Chunk]:
        """Iterate over every visible chunk."""
        for index in range(len(self)):
            yield self[index]

    @property
    def file_paths(self) -> list[str]:
        """Every visible chunk's file path, in row order."""
        if self._file_paths is None:
            base = file_paths_of(self._base)
            delta = file_paths_of(self._delta) if self._delta is not None else []
            self._file_paths = [base[row] for row in self._keep.tolist()] + list(delta)
        return self._file_paths

    @property
    def languages(self) -> list[str | None]:
        """Every visible chunk's language, in row order."""
        if self._languages is None:
            base = languages_of(self._base)
            delta = languages_of(self._delta) if self._delta is not None else []
            self._languages = [base[row] for row in self._keep.tolist()] + list(delta)
        return self._languages


class OverlayBM25:
    """BM25 over the kept base rows and the delta, scored against their one combined corpus."""

    def __init__(self, base: BM25, keep: npt.NDArray[np.int64], delta: BM25 | None) -> None:
        """Score *keep*'s rows of *base*, then *delta*'s."""
        self._base = base
        self._keep = keep
        self._delta = delta

    @property
    def document_count(self) -> int:
        """The number of visible documents."""
        return len(self._keep) + (self._delta.document_count if self._delta is not None else 0)

    def get_scores(
        self, tokens: list[str], weight_mask: npt.NDArray[np.bool_] | None = None
    ) -> npt.NDArray[np.float32]:
        """Score every visible document; the mask, when given, is aligned with the visible rows."""
        delta = self._delta
        n_docs = self._base.document_count + (delta.document_count if delta is not None else 0)
        total = self._base.total_doc_length + (delta.total_doc_length if delta is not None else 0)
        from_base = CorpusStats(n_docs, total, delta.document_frequency if delta is not None else (lambda _term: 0))
        scores = self._base.get_scores(tokens, corpus=from_base)[self._keep]
        if delta is not None:
            from_delta = CorpusStats(n_docs, total, self._base.document_frequency)
            scores = np.concatenate([scores, delta.get_scores(tokens, corpus=from_delta)])
        if weight_mask is not None:
            scores = scores * weight_mask
        return scores


class OverlayDense:
    """Cosine search over the kept base rows and the delta's, merged into one ranking."""

    def __init__(
        self, base: SelectableBasicBackend, keep: npt.NDArray[np.int64], delta: SelectableBasicBackend | None
    ) -> None:
        """Search *keep*'s rows of *base*, then *delta*'s."""
        self._base = base
        self._keep = keep
        self._delta = delta
        self._split = len(keep)

    def _distances(self, x: npt.NDArray, selector: npt.NDArray[np.int_] | None) -> npt.NDArray:
        """Return cosine distances to the selected visible rows (all of them for None), in selector order."""
        if selector is None:
            # One contiguous product over the whole base, then the kept columns: cheaper than gathering rows.
            parts = [self._base._dist(x)[:, self._keep]]  # noqa: SLF001
            if self._delta is not None:
                parts.append(self._delta._dist(x))  # noqa: SLF001
            return np.concatenate(parts, axis=1)
        split = int(np.searchsorted(selector, self._split))
        parts = []
        if split:
            parts.append(self._base._selector_dist(x, self._keep[selector[:split]]))  # noqa: SLF001
        if split < len(selector):
            assert self._delta is not None
            parts.append(self._delta._selector_dist(x, selector[split:] - self._split))  # noqa: SLF001
        return np.concatenate(parts, axis=1) if parts else np.empty((len(x), 0))

    def query(self, vectors: npt.NDArray, k: int, selector: npt.NDArray[np.int_] | None = None) -> list:
        """Return, per query vector, the visible row indices of its k nearest rows and their distances.

        :raises ValueError: If k is less than 1.
        """
        if k < 1:
            raise ValueError(f"k should be >= 1, is now {k}")
        if selector is not None:
            selector = np.sort(np.asarray(selector, dtype=np.int64))
        distances = self._distances(vectors, selector)
        effective_k = min(k, distances.shape[1])
        if effective_k == 0:
            return [(np.empty(0, dtype=np.int64), np.empty(0)) for _ in range(len(vectors))]
        indices = np.argpartition(distances, kth=effective_k - 1, axis=1)[:, :effective_k]
        ordered = np.take_along_axis(indices, np.argsort(np.take_along_axis(distances, indices, axis=1)), axis=1)
        nearest = np.take_along_axis(distances, ordered, axis=1)
        rows = selector[ordered] if selector is not None else ordered
        return list(zip(rows, nearest))


class OverlaySymbols:
    """Symbol definitions of the kept base rows and the delta, in visible row numbers."""

    def __init__(
        self,
        base: SymbolDefinitions | None,
        keep: npt.NDArray[np.int64],
        base_rows: int,
        delta: SymbolDefinitions | None,
        delta_rows: int,
    ) -> None:
        """Map *base*'s rows through *keep* and offset *delta*'s past them."""
        self._base = base
        self._delta = delta
        self._split = len(keep)
        self._visible = np.full(base_rows, -1, dtype=np.int64)
        self._visible[keep] = np.arange(len(keep), dtype=np.int64)
        self.n_chunks = self._split + delta_rows

    def chunks_defining(self, names: Sequence[str]) -> npt.NDArray[np.int64] | None:
        """Return the ascending visible rows defining any of *names*, or None when the caller must scan."""
        found: list[npt.NDArray[np.int64]] = []
        if self._base is not None:
            rows = self._base.chunks_defining(names)
            if rows is None:
                return None
            mapped = self._visible[np.asarray(rows, dtype=np.int64)]
            found.append(mapped[mapped >= 0])
        if self._delta is not None:
            rows = self._delta.chunks_defining(names)
            if rows is None:
                return None
            found.append(np.asarray(rows, dtype=np.int64) + self._split)
        return np.unique(np.concatenate(found)) if found else np.empty(0, dtype=np.int64)


@dataclass
class OverlayPlan:
    """Which of a lane's files the base answers unchanged and which the delta holds."""

    #: Lane-relative path -> the file's modification time in the lane, for base-held files.
    shared: dict[str, int] = field(default_factory=dict)
    #: Lane-relative paths of files the delta holds.
    differing: list[str] = field(default_factory=list)


def plan_overlay(lane: Lane, base: ZembleIndex) -> OverlayPlan:
    """Walk the lane as its own build would and decide, file by file, where each one is answered from.

    A file is the base's when the base indexed it, the base's stores still describe the base's file
    (its modification time is the manifest's), and the lane's copy is byte-for-byte the base's.
    """
    comparer = LaneComparer(lane)
    manifest = base._manifest  # noqa: SLF001
    plan = OverlayPlan()
    for walked in walk_entries(lane.root, get_extensions(base.storage_content), ignore=list(base.exclude)):
        try:
            if get_file_status(walked.path, None, walked.stat) != FileStatus.VALID:
                continue
        except OSError:
            continue
        relative = walked.relative_path
        entry = manifest.get(relative)
        if entry is not None and _base_current(lane.base / relative, entry) and comparer.same(relative):
            plan.shared[relative] = walked.stat.st_mtime_ns
        else:
            plan.differing.append(relative)
    return plan


def _base_current(path: Path, entry: FileManifestEntry) -> bool:
    """Return whether the base's stores still describe the base's file at *path*."""
    try:
        return path.stat().st_mtime_ns == entry.mtime_ns
    except OSError:
        return False


class OverlayIndex(ZembleIndex):
    """An index answering for a lane from its base's stores plus the delta of what the lane changed."""

    lane: Lane
    base: ZembleIndex
    delta: ZembleIndex | None
    plan: OverlayPlan

    @property
    def delta_chunks(self) -> int:
        """How many rows the delta holds: what this overlay costs beyond its base."""
        return len(self.delta.chunks) if self.delta is not None else 0


def _delta_previous(delta: ZembleIndex | None) -> PreviousIndex | None:
    """Return the stores of a previous delta, which its files still unchanged are copied from."""
    if delta is None or not isinstance(delta.chunks, ChunkList):
        return None
    return PreviousIndex(
        chunks=delta.chunks,
        vectors=delta._semantic_index.vectors,  # noqa: SLF001
        manifest=delta._manifest,  # noqa: SLF001
        bm25_index=delta._bm25_index,  # noqa: SLF001
        definitions=delta._definitions,  # noqa: SLF001
    )


def _stored_delta(final: Path, lane: Lane, base: ZembleIndex) -> ZembleIndex | None:
    """Load the delta a previous process left for this lane on this base, or None when there is none to reuse."""
    try:
        metadata = orjson.loads(PersistencePath.from_path(final).metadata.read_bytes())
        if metadata.get(OVERLAY_OF_KEY) != str(lane.base) or metadata.get("embedder") != base.embedder.model_id:
            return None
        return ZembleIndex.load_from_disk(final, embedder=base.embedder)
    except (OSError, ValueError, KeyError, orjson.JSONDecodeError):
        return None


def _write_delta(
    lane: Lane, base: ZembleIndex, differing: Sequence[str], final: Path, prior_delta: ZembleIndex | None
) -> ZembleIndex | None:
    """Write and load the delta stores of the files the lane changed, or None when it changed none."""
    if not differing:
        shutil.rmtree(final, ignore_errors=True)
        return None
    require_declared_scope(lane.root)
    embedder: Embedder = base.embedder
    capsules = base._capsules  # noqa: SLF001
    prior = _delta_previous(prior_delta)
    repo_paths = RepoRelativePaths()
    planned: list[PlannedFile] = []
    for relative in sorted(differing):
        entry = prior.manifest.get(relative) if prior is not None else None
        absolute = lane.root / relative
        try:
            current = entry is not None and absolute.stat().st_mtime_ns == entry.mtime_ns
        except OSError:
            continue
        if current and entry is not None:
            planned.append(reused_file(relative, entry))
            continue
        one = plan_file(absolute, relative, None, capsules, repo_paths)
        if one is not None:
            planned.append(one)
    if not any(file.count for file in planned):
        shutil.rmtree(final, ignore_errors=True)
        return None
    staging = staging_for(final)
    try:
        written = write_planned(planned, embedder, staging, previous=prior, capsules=capsules, source=lane.root)
        write_metadata(
            staging,
            root=lane.root,
            embedder=embedder,
            content=base.storage_content,
            capsules=capsules,
            exclude=base.exclude,
            manifest=written.manifest,
            extra={OVERLAY_OF_KEY: str(lane.base)},
        )
        publish(staging, final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    delta = ZembleIndex.load_from_disk(final, embedder=embedder)
    if written.embedded:
        logger.info("overlay %s: embedded %d new chunk(s)", lane.root, written.embedded)
    return delta


def build_overlay(lane: Lane, base: ZembleIndex, final: Path, previous: OverlayIndex | None = None) -> OverlayIndex:
    """Plan a lane against its base, write the delta of what differs, and compose the overlay.

    :param lane: The lane and its base workspace.
    :param base: The base's index as stored: never a view, so its rows are its stores' rows.
    :param final: The folder the delta stores are published into.
    :param previous: The overlay this one replaces, whose unchanged delta rows are copied, not chunked.
    :return: The overlay, rooted at the lane.
    :raises ValueError: If *base* is a view of an index rather than the index itself.
    """
    if base._view is not None:  # noqa: SLF001
        raise ValueError("an overlay is built on a base index itself, never on a view of it")
    plan = plan_overlay(lane, base)
    prior_delta = previous.delta if previous is not None else _stored_delta(final, lane, base)
    delta = _write_delta(lane, base, plan.differing, final, prior_delta)
    manifest = base._manifest  # noqa: SLF001
    shared = sorted(plan.shared, key=lambda relative: manifest[relative].start)
    overlay_manifest: dict[str, FileManifestEntry] = {}
    ranges: list[npt.NDArray[np.int64]] = []
    row = 0
    for relative in shared:
        entry = manifest[relative]
        ranges.append(np.arange(entry.start, entry.end, dtype=np.int64))
        overlay_manifest[relative] = FileManifestEntry(mtime_ns=plan.shared[relative], start=row, count=entry.count)
        row += entry.count
    keep = np.concatenate(ranges) if ranges else np.empty(0, dtype=np.int64)
    if delta is not None:
        for relative, entry in delta._manifest.items():  # noqa: SLF001
            overlay_manifest[relative] = FileManifestEntry(
                mtime_ns=entry.mtime_ns, start=row + entry.start, count=entry.count
            )
    delta_rows = len(delta.chunks) if delta is not None else 0
    index = OverlayIndex(
        base.embedder,
        OverlayBM25(base._bm25_index, keep, delta._bm25_index if delta is not None else None),  # type: ignore[arg-type]  # noqa: SLF001
        OverlayDense(base._semantic_index, keep, delta._semantic_index if delta is not None else None),  # type: ignore[arg-type]  # noqa: SLF001
        OverlayChunks(base.chunks, keep, delta.chunks if delta is not None else None),
        root=lane.root,
        content=base.storage_content,
        loaded_from_disk=True,
        manifest=overlay_manifest,
        capsules=base._capsules,  # noqa: SLF001
        definitions=OverlaySymbols(  # type: ignore[arg-type]
            base._definitions,  # noqa: SLF001
            keep,
            len(base.chunks),
            delta._definitions if delta is not None else None,  # noqa: SLF001
            delta_rows,
        ),
        exclude=base.exclude,
    )
    index.lane = lane
    index.base = base
    index.delta = delta
    index.plan = plan
    logger.info(
        "overlay %s on %s: %d file(s) from the base, %d in the delta (%d chunks)",
        lane.root,
        lane.base,
        len(plan.shared),
        len(plan.differing),
        delta_rows,
    )
    return index


def overlay_folder(index_folder: Path) -> Path:
    """Return the folder a lane's delta stores live in, beside where its own index variant would."""
    return index_folder.with_name(index_folder.name.replace("index", "overlay", 1))


__all__ = [
    "OVERLAY_OF_KEY",
    "OverlayBM25",
    "OverlayChunks",
    "OverlayDense",
    "OverlayIndex",
    "OverlayPlan",
    "OverlaySymbols",
    "build_overlay",
    "overlay_folder",
    "plan_overlay",
]

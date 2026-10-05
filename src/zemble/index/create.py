"""Plan and write one index generation, streaming it to disk in memory bounded by one batch."""

from bisect import bisect_right
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from vicinity.backends.basic import BasicArgs

from zemble.chunking import chunk_source
from zemble.chunking.capsule import CapsuleOptions, RepoRelativePaths, embedding_text
from zemble.embedding.base import Embedder, normalize_rows
from zemble.embedding.pricing import require_affordable_bill
from zemble.index.bm25 import BM25Writer
from zemble.index.chunk_store import ChunkList, ChunkStoreWriter, load_chunks
from zemble.index.file_walker import WalkedFile, walk_entries
from zemble.index.files import (
    FileStatus,
    detect_language,
    get_extensions,
    get_file_status,
    read_file_text,
)
from zemble.index.scope import changed_indexed_paths, require_affordable_scope, require_declared_scope
from zemble.index.sparse import enrich_for_bm25
from zemble.index.symbols import SymbolWriter
from zemble.index.types import FileManifestEntry, PersistencePath, PreviousIndex, make_chunk_id
from zemble.tokens import tokenize
from zemble.types import Chunk, ContentType

#: Rows embedded per provider round, and rows copied per step out of a previous matrix. Both
#: bound a build's vector memory: neither the fresh nor the reused vectors are ever held whole.
_EMBED_ROWS = 2048
_COPY_ROWS = 16384


@dataclass
class PlannedFile:
    """One file's contribution to a build: its chunks, or the previous index's claim on them."""

    indexed_path: str
    mtime_ns: int
    previous_entry: FileManifestEntry | None
    reused: bool
    chunks: list[Chunk]
    count: int


def _reused_file(indexed_path: str, previous_entry: FileManifestEntry) -> PlannedFile:
    """Plan a file whose modification time did not move: its chunks stay in the previous stores."""
    return PlannedFile(indexed_path, previous_entry.mtime_ns, previous_entry, True, [], previous_entry.count)


def _indexed_path(walked: WalkedFile, root: Path, display_root: Path | None) -> str:
    """Return the path a chunk is stored under, reusing the walker's own relative path."""
    if display_root == root:
        return walked.relative_path
    return str(walked.path.relative_to(display_root) if display_root else walked.path)


def plan_files(
    path: Path,
    content: ContentType | Sequence[ContentType] = (ContentType.CODE,),
    display_root: Path | None = None,
    previous_manifest: dict[str, FileManifestEntry] | None = None,
    capsules: CapsuleOptions | None = None,
    exclude: Sequence[str] = (),
) -> Iterator[PlannedFile]:
    """Walk a tree and chunk every file a build would index, without embedding anything.

    This is the chunking half of :func:`write_index`, shared with the pre-flight
    report so both see exactly the same files, the same capsule text and the same
    mtime-based reuse decision.

    :param path: Resolved absolute path to walk.
    :param content: Content types to index.
    :param display_root: If set, chunk file paths are stored relative to this root.
    :param previous_manifest: A previous build's manifest, or None for a full build.
    :param capsules: Context-capsule knobs; None resolves the environment override.
    :param exclude: Extra gitignore-style patterns, relative to `path`, this build skips.
    :return: One :class:`PlannedFile` per indexable file, in walk order.
    """
    resolved_capsules = CapsuleOptions.resolve(capsules)
    normalized = (content,) if isinstance(content, ContentType) else content
    repo_paths = RepoRelativePaths()
    for walked in walk_entries(path, get_extensions(normalized), ignore=list(exclude)):
        try:
            if get_file_status(walked.path, None, walked.stat) != FileStatus.VALID:
                continue
            indexed_path = _indexed_path(walked, path, display_root)
            mtime_ns = walked.stat.st_mtime_ns
            previous_entry = previous_manifest.get(indexed_path) if previous_manifest is not None else None

            if previous_entry is not None and previous_entry.mtime_ns == mtime_ns:
                planned = _reused_file(indexed_path, previous_entry)
            else:
                file_chunks = chunk_source(
                    read_file_text(walked.path),
                    indexed_path,
                    detect_language(walked.path),
                    resolved_capsules,
                    repo_paths.path_for(walked.path, indexed_path),
                )
                planned = PlannedFile(indexed_path, mtime_ns, previous_entry, False, file_chunks, len(file_chunks))
        except OSError:
            continue
        yield planned


def plan_changed_files(
    path: Path,
    changed: Iterable[Path],
    content: ContentType | Sequence[ContentType] = (ContentType.CODE,),
    display_root: Path | None = None,
    previous_manifest: dict[str, FileManifestEntry] | None = None,
    capsules: CapsuleOptions | None = None,
    exclude: Sequence[str] = (),
) -> Iterator[PlannedFile]:
    """Plan a build from a known set of changed paths, without walking the tree.

    Every file the previous build indexed is planned as reused unless the change set names
    it; a named path is re-chunked, or dropped when it is gone, ignored, or not something
    the walk would have reached. The caller therefore has to name every path that moved -
    this is the watcher's answer, not a discovery pass - and the order is the previous
    build's order with new files appended, which is what keeps reused vector rows in place.

    :param path: Resolved absolute path the index covers.
    :param changed: The paths that were added, edited or removed.
    :param content: Content types to index.
    :param display_root: If set, chunk file paths are stored relative to this root.
    :param previous_manifest: The previous build's manifest; every entry not named as changed is reused.
    :param capsules: Context-capsule knobs; None resolves the environment override.
    :param exclude: The gitignore-style patterns the index was built with, which a changed path
        still has to survive: an excluded build must not grow the paths it excluded back.
    :return: One :class:`PlannedFile` per file the new index holds.
    """
    resolved_capsules = CapsuleOptions.resolve(capsules)
    normalized = (content,) if isinstance(content, ContentType) else content
    repo_paths = RepoRelativePaths()
    manifest = previous_manifest or {}

    touched = changed_indexed_paths(path, changed, normalized, display_root, exclude)

    for indexed_path, previous_entry in manifest.items():
        candidate = touched.pop(indexed_path, None)
        if candidate is None:
            yield _reused_file(indexed_path, previous_entry)
            continue
        planned = _plan_one(candidate, indexed_path, previous_entry, resolved_capsules, repo_paths)
        if planned is not None:
            yield planned

    for indexed_path in sorted(touched):
        planned = _plan_one(touched[indexed_path], indexed_path, None, resolved_capsules, repo_paths)
        if planned is not None:
            yield planned


def _plan_one(
    file_path: Path,
    indexed_path: str,
    previous_entry: FileManifestEntry | None,
    capsules: CapsuleOptions,
    repo_paths: RepoRelativePaths,
) -> PlannedFile | None:
    """Plan one named file, reusing its chunks when its modification time did not move."""
    try:
        stat = file_path.stat()
        if get_file_status(file_path, None, stat) != FileStatus.VALID:
            return None
        mtime_ns = stat.st_mtime_ns
        if previous_entry is not None and previous_entry.mtime_ns == mtime_ns:
            return _reused_file(indexed_path, previous_entry)
        file_chunks = chunk_source(
            read_file_text(file_path),
            indexed_path,
            detect_language(file_path),
            capsules,
            repo_paths.path_for(file_path, indexed_path),
        )
        return PlannedFile(indexed_path, mtime_ns, previous_entry, False, file_chunks, len(file_chunks))
    except OSError:
        return None


@dataclass(frozen=True)
class WrittenIndex:
    """What :func:`write_index` wrote: the manifest the generation's metadata records, and its size."""

    manifest: dict[str, FileManifestEntry]
    chunks: int
    embedded: int


class _EmbeddingTexts(Sequence[str]):
    """The embedding texts of a written chunk store's fresh rows, produced on demand, never held whole.

    The bill guard reads every text a build will embed before one is bought; this hands it them one
    at a time out of the mapped store, so judging a full build costs its hashes, not its texts.
    """

    def __init__(self, chunks: ChunkList, runs: list[tuple[int, int]]) -> None:
        """Cover the rows of *runs*, `(first row, count)` pairs in row order."""
        self._chunks = chunks
        self._runs = runs
        self._starts = list(np.cumsum([0, *(count for _start, count in runs)]).tolist())

    def __len__(self) -> int:
        """The number of fresh rows."""
        return self._starts[-1]

    def __getitem__(self, index: int) -> str:  # type: ignore[override]
        """Return the embedding text of the *index*-th fresh row."""
        if not 0 <= index < len(self):
            raise IndexError(index)
        run = bisect_right(self._starts, index) - 1
        return embedding_text(self._chunks[self._runs[run][0] + index - self._starts[run]])

    def __iter__(self) -> Iterator[str]:
        """Yield every text, run by run."""
        for start, count in self._runs:
            for row in range(start, start + count):
                yield embedding_text(self._chunks[row])

    def rows(self) -> Iterator[int]:
        """Yield the chunk row of every text, in the same order."""
        for start, count in self._runs:
            yield from range(start, start + count)


def _write_vectors(
    directory: Path,
    embedder: Embedder,
    total: int,
    runs: list[tuple[int, int, int]],
    previous_vectors: np.ndarray | None,
    chunks: ChunkList,
) -> int:
    """Write the vector matrix straight into a mapped file: reused rows copied, fresh rows embedded.

    AIDEV-NOTE: every fresh row is judged by the bill guard in ONE call before a single one is
    bought, incremental builds included, exactly as when the whole set was embedded at once; only
    the buying is batched. The texts are produced lazily, so the guard sees all of them without a
    build holding them all.

    :return: How many rows were embedded.
    """
    directory.mkdir(parents=True, exist_ok=True)
    BasicArgs().dump(directory / "arguments.json")
    fresh = _EmbeddingTexts(chunks, [(start, count) for start, count, previous in runs if previous < 0])
    require_affordable_bill(embedder, fresh)
    dimensions = previous_vectors.shape[1] if previous_vectors is not None else embedder.dimensions
    vectors = np.lib.format.open_memmap(
        directory / "vectors.npy", mode="w+", dtype=np.float32, shape=(total, dimensions)
    )
    try:
        for start, count, previous in runs:
            if previous < 0 or previous_vectors is None:
                continue
            for offset in range(0, count, _COPY_ROWS):
                step = min(_COPY_ROWS, count - offset)
                vectors[start + offset : start + offset + step] = previous_vectors[
                    previous + offset : previous + offset + step
                ]
        batch_rows: list[int] = []
        batch_texts: list[str] = []
        for row, text in zip(fresh.rows(), fresh, strict=True):
            batch_rows.append(row)
            batch_texts.append(text)
            if len(batch_texts) == _EMBED_ROWS:
                vectors[batch_rows] = normalize_rows(embedder.embed_documents(batch_texts))
                batch_rows, batch_texts = [], []
        if batch_texts:
            vectors[batch_rows] = normalize_rows(embedder.embed_documents(batch_texts))
        vectors.flush()
    finally:
        del vectors
    return len(fresh)


def _extend(runs: list[tuple[int, int, int]], start: int, count: int, previous: int) -> None:
    """Record rows *start*..+*count* as reused from *previous* (or fresh, -1), merging contiguous runs."""
    if count <= 0:
        return
    if runs:
        last_start, last_count, last_previous = runs[-1]
        contiguous = last_start + last_count == start
        if contiguous and (
            (previous < 0 and last_previous < 0) or (previous >= 0 and last_previous + last_count == previous)
        ):
            runs[-1] = (last_start, last_count + count, last_previous)
            return
    runs.append((start, count, previous))


def write_index(
    path: Path,
    embedder: Embedder,
    target: Path,
    content: ContentType | Sequence[ContentType] = (ContentType.CODE,),
    display_root: Path | None = None,
    previous: PreviousIndex | None = None,
    capsules: CapsuleOptions | None = None,
    changed_paths: Iterable[Path] | None = None,
    exclude: Sequence[str] = (),
) -> WrittenIndex:
    """Write every component of one index generation for a directory into *target*.

    Nothing is held whole: a file's chunks are written as soon as it is planned, a reused file is
    copied from the previous generation's mapped stores, and vectors are embedded in batches into a
    mapped matrix. What stays in memory is what is NEW (its BM25 postings as flat integers) plus a
    few integers per chunk, so an edit to one file costs one file, not a copy of the index.

    :param path: Resolved absolute path to index.
    :param embedder: The embedder to use for indexing.
    :param target: The directory the components are written into; the caller publishes it.
    :param content: Content types to index.
    :param display_root: If set, chunk file paths are stored relative to this root.
    :param previous: The current generation, whose unchanged files are carried over.
    :param capsules: Context-capsule knobs; None resolves the environment override, else the defaults.
    :param changed_paths: The exact paths that moved, from a watcher; None walks the whole tree.
        Only honoured together with `previous`, which is what the unnamed files are reused from.
    :param exclude: Extra gitignore-style patterns, relative to `path`, this build skips at walk time.
    :raises ValueError: if no items were found, no index can be created.
    :return: The manifest and counts of the written generation.
    """
    previous_manifest = previous.manifest if previous is not None else {}
    resolved_capsules = CapsuleOptions.resolve(capsules)
    normalized = (content,) if isinstance(content, ContentType) else tuple(content)

    # AIDEV-NOTE: the scope guards live HERE, at the one construction seam every lane passes,
    # rather than in ZembleIndex.from_path: the daemon's watcher rebuild calls this function
    # directly, so a guard installed above it refused the CLI while the daemon chunked the same
    # tree unguarded. Measured against `previous_manifest` - the manifest this build will really
    # reuse from - so the guard can never approve an incremental build the build then does in
    # full because the previous index turned out to be unusable, and against the change set on
    # the lane that has one, so the guard never walks a tree the build itself refuses to walk.
    changed = list(changed_paths) if previous is not None and changed_paths is not None else None
    require_declared_scope(path)
    require_affordable_scope(
        path, embedder, normalized, exclude, previous_manifest or None, changed=changed, display_root=display_root
    )

    plan = (
        plan_changed_files(
            path,
            changed,
            content,
            display_root=display_root,
            previous_manifest=previous_manifest,
            capsules=resolved_capsules,
            exclude=exclude,
        )
        if changed is not None
        else plan_files(
            path,
            content,
            display_root=display_root,
            previous_manifest=previous_manifest if previous is not None else None,
            capsules=resolved_capsules,
            exclude=exclude,
        )
    )
    stores = PersistencePath.from_path(target)
    chunk_writer = ChunkStoreWriter(stores.chunks)
    bm25_writer = BM25Writer(stores.bm25_index, previous.bm25_index if previous is not None else None)
    symbol_writer = SymbolWriter(stores.symbols, previous.definitions if previous is not None else None)
    manifest: dict[str, FileManifestEntry] = {}
    runs: list[tuple[int, int, int]] = []
    row = 0
    for planned in plan:
        manifest[planned.indexed_path] = FileManifestEntry(mtime_ns=planned.mtime_ns, start=row, count=planned.count)
        if planned.reused and previous is not None and planned.previous_entry is not None:
            first = planned.previous_entry.start
            chunk_writer.reuse(previous.chunks, first, planned.count)
            bm25_writer.reuse(first, planned.count)
            symbol_writer.reuse(first, planned.count)
            _extend(runs, row, planned.count, first)
        else:
            for slot, chunk in enumerate(planned.chunks):
                chunk_writer.append(chunk)
                bm25_writer.add(
                    make_chunk_id(planned.indexed_path, slot),
                    tokenize(enrich_for_bm25(chunk, resolved_capsules.in_bm25)),
                )
                symbol_writer.add(chunk.content)
            _extend(runs, row, planned.count, -1)
        row += planned.count
    chunk_writer.finish()
    if not row:
        raise ValueError(f"No supported files found under {path}.")
    bm25_writer.finish()
    symbol_writer.finish()
    embedded = _write_vectors(
        stores.semantic_index,
        embedder,
        row,
        runs,
        previous.vectors if previous is not None else None,
        load_chunks(stores.chunks),
    )
    return WrittenIndex(manifest=manifest, chunks=row, embedded=embedded)

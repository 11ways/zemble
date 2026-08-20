from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from vicinity.backends.basic import BasicArgs

from zemble.chunking import chunk_source
from zemble.chunking.capsule import CapsuleOptions, RepoRelativePaths
from zemble.embedding.base import Embedder
from zemble.index.bm25 import BM25
from zemble.index.dense import SelectableBasicBackend, embed_chunks
from zemble.index.file_walker import WalkedFile, ignored_prefix, walk_entries
from zemble.index.files import (
    FileStatus,
    detect_language,
    get_extensions,
    get_file_status,
    read_file_text,
)
from zemble.index.sparse import enrich_for_bm25
from zemble.index.types import FileManifestEntry, PreviousIndex, make_chunk_id
from zemble.tokens import tokenize
from zemble.types import Chunk, ContentType, EmbeddingMatrix


@dataclass
class PlannedFile:
    """One file's contribution to a build: its chunks, or the previous index's claim on them."""

    indexed_path: str
    mtime_ns: int
    previous_entry: FileManifestEntry | None
    reused: bool
    chunks: list[Chunk]
    count: int


def _reindex_file(
    bm25_index: BM25,
    indexed_path: str,
    file_chunks: list[Chunk],
    previous_entry: FileManifestEntry | None,
    capsules: CapsuleOptions,
) -> None:
    """Replace a file's BM25 postings: remove its old slots (if any), then add its new ones."""
    if previous_entry is not None:
        for slot in range(previous_entry.count):
            bm25_index.remove_document(make_chunk_id(indexed_path, slot))
    for slot, chunk in enumerate(file_chunks):
        bm25_index.add_document(make_chunk_id(indexed_path, slot), tokenize(enrich_for_bm25(chunk, capsules.in_bm25)))


def _indexed_path(walked: WalkedFile, root: Path, display_root: Path | None) -> str:
    """Return the path a chunk is stored under, reusing the walker's own relative path."""
    if display_root == root:
        return walked.relative_path
    return str(walked.path.relative_to(display_root) if display_root else walked.path)


def _has_same_vector_layout(
    manifest: dict[str, FileManifestEntry], previous_manifest: dict[str, FileManifestEntry]
) -> bool:
    """Return whether both manifests use the same chunk ranges."""
    return len(manifest) == len(previous_manifest) and all(
        (previous_entry := previous_manifest.get(indexed_path)) is not None
        and entry.start == previous_entry.start
        and entry.count == previous_entry.count
        for indexed_path, entry in manifest.items()
    )


def plan_files(
    path: Path,
    content: ContentType | Sequence[ContentType] = (ContentType.CODE,),
    display_root: Path | None = None,
    previous_manifest: dict[str, FileManifestEntry] | None = None,
    capsules: CapsuleOptions | None = None,
    previous_chunks: Sequence[Chunk] | None = None,
) -> Iterator[PlannedFile]:
    """Walk a tree and chunk every file a build would index, without embedding anything.

    This is the chunking half of :func:`create_index_from_path`, shared with the pre-flight
    report so both see exactly the same files, the same capsule text and the same
    mtime-based reuse decision.

    :param path: Resolved absolute path to walk.
    :param content: Content types to index.
    :param display_root: If set, chunk file paths are stored relative to this root.
    :param previous_manifest: A previous build's manifest, or None for a full build.
    :param capsules: Context-capsule knobs; None resolves the environment override.
    :param previous_chunks: The previous build's chunk list, when reused chunks are wanted back.
    :return: One :class:`PlannedFile` per indexable file, in walk order.
    """
    resolved_capsules = CapsuleOptions.resolve(capsules)
    normalized = (content,) if isinstance(content, ContentType) else content
    repo_paths = RepoRelativePaths()
    for walked in walk_entries(path, get_extensions(normalized)):
        try:
            if get_file_status(walked.path, None, walked.stat) != FileStatus.VALID:
                continue
            indexed_path = _indexed_path(walked, path, display_root)
            mtime_ns = walked.stat.st_mtime_ns
            previous_entry = previous_manifest.get(indexed_path) if previous_manifest is not None else None

            if previous_entry is not None and previous_entry.mtime_ns == mtime_ns:
                reused = list(previous_chunks[previous_entry.start : previous_entry.end]) if previous_chunks else []
                planned = PlannedFile(indexed_path, mtime_ns, previous_entry, True, reused, previous_entry.count)
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
    previous_chunks: Sequence[Chunk] | None = None,
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
    :param previous_chunks: The previous build's chunk list, when reused chunks are wanted back.
    :return: One :class:`PlannedFile` per file the new index holds.
    """
    resolved_capsules = CapsuleOptions.resolve(capsules)
    normalized = (content,) if isinstance(content, ContentType) else content
    extensions = {extension.lower() for extension in get_extensions(normalized)}
    repo_paths = RepoRelativePaths()
    manifest = previous_manifest or {}
    root_for_paths = display_root if display_root is not None else path

    touched: dict[str, Path] = {}
    for candidate in changed:
        try:
            indexed_path = str(candidate.relative_to(root_for_paths).as_posix())
            relative = candidate.relative_to(path).as_posix()
        except ValueError:
            continue
        if candidate.suffix.lower() not in extensions or ignored_prefix(path, relative) is not None:
            continue
        touched[indexed_path] = candidate

    for indexed_path, previous_entry in manifest.items():
        candidate = touched.pop(indexed_path, None)
        if candidate is None:
            reused = list(previous_chunks[previous_entry.start : previous_entry.end]) if previous_chunks else []
            yield PlannedFile(indexed_path, previous_entry.mtime_ns, previous_entry, True, reused, previous_entry.count)
            continue
        planned = _plan_one(candidate, indexed_path, previous_entry, previous_chunks, resolved_capsules, repo_paths)
        if planned is not None:
            yield planned

    for indexed_path in sorted(touched):
        planned = _plan_one(touched[indexed_path], indexed_path, None, None, resolved_capsules, repo_paths)
        if planned is not None:
            yield planned


def _plan_one(
    file_path: Path,
    indexed_path: str,
    previous_entry: FileManifestEntry | None,
    previous_chunks: Sequence[Chunk] | None,
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
            reused = list(previous_chunks[previous_entry.start : previous_entry.end]) if previous_chunks else []
            return PlannedFile(indexed_path, mtime_ns, previous_entry, True, reused, previous_entry.count)
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


def _assemble_vectors(
    total: int,
    placements: list[tuple[int, PlannedFile]],
    fresh_rows: list[int],
    fresh: EmbeddingMatrix | None,
    previous: PreviousIndex | None,
    manifest: dict[str, FileManifestEntry],
) -> EmbeddingMatrix:
    """Build the vector matrix for a build, copying every reused row out of the previous index.

    :param total: The number of chunks in the new index.
    :param placements: Each planned file with the row its chunks start at.
    :param fresh_rows: The rows that were embedded this build.
    :param fresh: The freshly embedded vectors, in ``fresh_rows`` order, or None when none were.
    :param previous: The previous index, or None for a full build.
    :param manifest: The new manifest, used to decide whether the layout is unchanged.
    :return: The full vector matrix.
    """
    if previous is None:
        # A full build embeds every row, so the fresh matrix is already the whole thing.
        return fresh if fresh is not None else np.empty((0, 0), dtype=np.float32)
    if _has_same_vector_layout(manifest, previous.manifest):
        embeddings = previous.vectors
    else:
        embeddings = np.empty((total, previous.vectors.shape[1]), dtype=np.float32)
        for start, planned in placements:
            if planned.reused and planned.previous_entry is not None:
                entry = planned.previous_entry
                embeddings[start : start + planned.count] = previous.vectors[entry.start : entry.end]
    if fresh is not None:
        embeddings[fresh_rows] = fresh
    return embeddings


def create_index_from_path(
    path: Path,
    embedder: Embedder,
    content: ContentType | Sequence[ContentType] = (ContentType.CODE,),
    display_root: Path | None = None,
    previous: PreviousIndex | None = None,
    capsules: CapsuleOptions | None = None,
    changed_paths: Iterable[Path] | None = None,
) -> tuple[BM25, SelectableBasicBackend, list[Chunk], dict[str, FileManifestEntry]]:
    """Create an index from a resolved directory, optionally reusing a previous index's unchanged files.

    :param path: Resolved absolute path to index.
    :param embedder: The embedder to use for indexing.
    :param content: Content types to index.
    :param display_root: If set, chunk file paths are stored relative to this root.
    :param previous: A previously built index to reuse unchanged files' chunks/embeddings/postings from.
    :param capsules: Context-capsule knobs; None resolves the environment override, else the defaults.
    :param changed_paths: The exact paths that moved, from a watcher; None walks the whole tree.
        Only honoured together with `previous`, which is what the unnamed files are reused from.
    :raises ValueError: if no items were found, no index can be created.
    :return: A BM25 index, semantic index, list of chunks, and file manifest.
    """
    # The previous index keeps serving: for_update hands back an index sharing its immutable
    # postings, so nothing here writes into the BM25 object a warm daemon is querying.
    bm25_index = previous.bm25_index.for_update() if previous is not None else BM25()
    previous_manifest = previous.manifest if previous is not None else {}
    resolved_capsules = CapsuleOptions.resolve(capsules)

    if previous is not None and changed_paths is not None:
        plan = list(
            plan_changed_files(
                path,
                changed_paths,
                content,
                display_root=display_root,
                previous_manifest=previous_manifest,
                capsules=resolved_capsules,
                previous_chunks=previous.chunks,
            )
        )
    else:
        plan = list(
            plan_files(
                path,
                content,
                display_root=display_root,
                previous_manifest=previous_manifest if previous is not None else None,
                capsules=resolved_capsules,
                previous_chunks=previous.chunks if previous is not None else None,
            )
        )

    chunks: list[Chunk] = []
    chunk_ids: list[str] = []
    manifest: dict[str, FileManifestEntry] = {}
    placements: list[tuple[int, PlannedFile]] = []
    fresh_rows: list[int] = []

    for planned in plan:
        start = len(chunks)
        placements.append((start, planned))
        chunks.extend(planned.chunks)
        chunk_ids.extend(make_chunk_id(planned.indexed_path, slot) for slot in range(planned.count))
        manifest[planned.indexed_path] = FileManifestEntry(mtime_ns=planned.mtime_ns, start=start, count=planned.count)
        if not planned.reused:
            fresh_rows.extend(range(start, start + planned.count))

    if not chunks:
        raise ValueError(f"No supported files found under {path}.")

    # AIDEV-NOTE: every changed chunk is embedded in ONE call, incremental builds included.
    # That is what lets the caching embedder see the whole pending set at once - which is
    # where the budget guard lives - and it costs a paid provider one batched pass instead
    # of one request per changed file.
    fresh = embed_chunks(embedder, [chunks[row] for row in fresh_rows]) if fresh_rows else None
    embeddings = _assemble_vectors(len(chunks), placements, fresh_rows, fresh, previous, manifest)

    # BM25 is mutated only once the vectors exist: a refused or failed embed must not leave a
    # warm daemon's live index half-updated, and the BM25 index here IS that live object.
    for _start, planned in placements:
        if not planned.reused:
            _reindex_file(bm25_index, planned.indexed_path, planned.chunks, planned.previous_entry, resolved_capsules)
    for indexed_path in previous_manifest.keys() - manifest.keys():
        _reindex_file(bm25_index, indexed_path, [], previous_manifest[indexed_path], resolved_capsules)

    bm25_index.set_doc_order(chunk_ids)
    semantic_index = SelectableBasicBackend(embeddings, BasicArgs())

    return bm25_index, semantic_index, chunks, manifest

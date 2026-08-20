import hashlib
import json
import logging
import os
import shutil
import sys
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import orjson

from zemble.chunking.capsule import CapsuleOptions
from zemble.embedding.pricing import CONFIRM_ENV
from zemble.index.bm25 import BM25
from zemble.index.chunk_store import load_chunks
from zemble.index.dense import SelectableBasicBackend
from zemble.index.file_walker import walk_entries
from zemble.index.files import FileStatus, get_extensions, get_file_status
from zemble.index.types import CACHE_FORMAT_VERSION, FileManifestEntry, PersistencePath, PreviousIndex, make_chunk_id
from zemble.types import ContentType
from zemble.utils import is_git_url

logger = logging.getLogger(__name__)

#: (stored, requested) embedder pairs already reported, so one rebuild logs one line.
_REPORTED_EMBEDDER_MISMATCHES: set[tuple[str, str]] = set()

if TYPE_CHECKING:
    from zemble.index import ZembleIndex


def cache_key(path: str) -> str:
    """Compute the sha256 cache key for a local path or git URL."""
    if is_git_url(path):
        data = path.encode("utf-8")
    else:
        normalized = Path(path).expanduser().resolve()
        data = str(normalized).encode("utf-8")
    return hashlib.new("sha256", data).hexdigest()


def find_index_from_cache_folder(path: str, content: Sequence[ContentType] = (ContentType.CODE,)) -> Path:
    """Find an exact content index in the cache for a project path."""
    cache_dir = resolve_cache_folder() / cache_key(path)
    scope = "-".join(sorted({content_type.value for content_type in content}))
    return cache_dir / ("index" if scope == ContentType.CODE.value else f"index-{scope}")


def _windows_cache_dir(name: str) -> Path:
    """Get the default windows cache dir."""
    env_base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
    base = Path(env_base) if env_base is not None else Path.home() / "AppData" / "Local"
    return base / name / "Cache"


def _macos_cache_dir(name: str) -> Path:
    """Get the default macOS cache dir."""
    return Path.home() / "Library" / "Caches" / name


def _linux_cache_dir(name: str) -> Path:
    """Get the default Linux cache dir."""
    env_base = os.getenv("XDG_CACHE_HOME")
    base = Path(env_base) if env_base else Path.home() / ".cache"
    return base / name


def _get_valid_user_cache_dir() -> Path | None:
    """Gets the user cache dir if it is set and is a valid path."""
    user_cache_location = os.getenv("ZEMBLE_CACHE_LOCATION")
    if user_cache_location is None:
        return None
    user_cache_dir = Path(user_cache_location)
    if not user_cache_dir.is_absolute():
        logger.warning("ZEMBLE_CACHE_LOCATION is not an absolute path: %s", user_cache_location)
        return None

    return user_cache_dir


def resolve_cache_folder() -> Path:
    """Resolves a cache folder, respects ZEMBLE_CACHE_LOCATION (highest precedence), XDG_CACHE_HOME."""
    name = "zemble"
    if user_cache_dir := _get_valid_user_cache_dir():
        cache_dir = user_cache_dir
    elif sys.platform == "win32":
        cache_dir = _windows_cache_dir(name)
    elif sys.platform == "darwin":
        cache_dir = _macos_cache_dir(name)
    else:
        cache_dir = _linux_cache_dir(name)

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def clear_cache(path: str) -> None:
    """Clear all exact content indexes for the given path."""
    shutil.rmtree(find_index_from_cache_folder(path).parent, ignore_errors=True)


def save_index_to_cache(index: "ZembleIndex", path: str) -> None:
    """Save an index to the cache folder if it was freshly built."""
    if not index.loaded_from_disk:
        index.save(find_index_from_cache_folder(path, index.content))


def _metadata_matches(metadata: dict, embedder_id: str, content: Sequence[ContentType], capsule_key: str) -> bool:
    """Return True if the stored metadata is compatible with the requested parameters.

    A cache built with a different embedder is never reused silently: mixing vector spaces
    produces plausible-looking nonsense, so the mismatch is logged by name and rebuilt.

    :param metadata: The stored metadata document.
    :param embedder_id: The normalized spec string of the requested embedder.
    :param content: The requested content types.
    :param capsule_key: The requested context-capsule configuration.
    :return: Whether the cache can be reused.
    """
    from zemble.chunking.chunking import _DESIRED_CHUNK_LENGTH_CHARS  # avoid circular import at module level

    try:
        content_type = tuple(ContentType(s) for s in metadata["content_type"])
        # chunk_size and cache_version are absent in indexes built before those fields were added;
        # treat that as a mismatch so old caches are transparently rebuilt in the current format.
        chunk_size_ok = metadata.get("chunk_size") == _DESIRED_CHUNK_LENGTH_CHARS
        version_ok = metadata.get("cache_version") == CACHE_FORMAT_VERSION
        # A capsule change rewrites embedding text, so a differently built index is never reused.
        capsule_ok = metadata.get("capsules") == capsule_key
        stored_embedder = metadata["embedder"]
        if version_ok and stored_embedder != embedder_id:
            if (stored_embedder, embedder_id) in _REPORTED_EMBEDDER_MISMATCHES:
                return False
            _REPORTED_EMBEDDER_MISMATCHES.add((stored_embedder, embedder_id))
            logger.warning(
                "Cached index was built with embedder %s but %s was requested; rebuilding the index.",
                stored_embedder,
                embedder_id,
            )
            return False
        return set(content_type) == set(content) and chunk_size_ok and version_ok and capsule_ok
    except (KeyError, ValueError):
        return False


def get_validated_cache(
    path: str, embedder_id: str, content: Sequence[ContentType], capsules: CapsuleOptions | None = None
) -> Path | None:
    """Validates the cache folder and returns the index path.

    :param path: Source path or git URL the index was built from.
    :param embedder_id: The normalized spec string of the requested embedder.
    :param content: The requested content types.
    :param capsules: The requested context-capsule configuration.
    :return: The reusable index path, or None if it must be rebuilt.
    """
    index_path = find_index_from_cache_folder(path, content)
    if not index_path.exists():
        return None

    persistence_path = PersistencePath.from_path(index_path)
    if persistence_path.non_existing():
        return None

    with open(persistence_path.metadata, encoding="utf-8") as f:
        metadata = json.load(f)
    if not _metadata_matches(metadata, embedder_id, content, CapsuleOptions.resolve(capsules).key):
        return None

    if is_git_url(str(path)):
        return index_path

    write_time = metadata["time"]
    extensions = get_extensions(content)

    path_as_path = Path(path).resolve()
    stored_files = metadata.get("files", {})
    current_files = set()
    for walked in walk_entries(path_as_path, extensions=extensions):
        file_status = get_file_status(walked.path, write_time, walked.stat)
        if file_status == FileStatus.NEWER:
            return None
        if file_status != FileStatus.VALID:
            continue
        current_files.add(walked.relative_path)

    if current_files != set(stored_files):
        return None

    return index_path


def has_cached_index(path: str, content: Sequence[ContentType] = (ContentType.CODE,)) -> bool:
    """Return whether a complete index folder exists for a path, without checking it for staleness."""
    return not PersistencePath.from_path(find_index_from_cache_folder(path, content)).non_existing()


def _ancestor_directories(path: Path) -> list[Path]:
    """Return the strict ancestors of a directory, nearest first."""
    return list(path.parents)


def find_ancestor_index_root(
    path: str,
    embedder_id: str,
    content: Sequence[ContentType] = (ContentType.CODE,),
    capsules: CapsuleOptions | None = None,
    loaded_roots: Collection[str] = (),
) -> str | None:
    """Return the nearest ancestor of *path* that already has a usable index of the same content.

    An ancestor held in memory counts without a disk check; one on disk has to validate the
    way any cache hit does, so a stale ancestor is never served.

    :param path: The requested directory.
    :param embedder_id: The normalized spec of the embedder that would answer the request.
    :param content: The requested content types.
    :param capsules: The requested context-capsule configuration.
    :param loaded_roots: Roots a caller already holds in memory for exactly this content.
    :return: The ancestor root, or None when the sub-path needs its own index.
    """
    for ancestor in _ancestor_directories(Path(path)):
        candidate = str(ancestor)
        if candidate in loaded_roots:
            return candidate
        if has_cached_index(candidate, content) and get_validated_cache(candidate, embedder_id, content, capsules):
            return candidate
    return None


def resolve_index_root(
    path: str,
    embedder_id: str,
    content: Sequence[ContentType] = (ContentType.CODE,),
    capsules: CapsuleOptions | None = None,
    loaded_roots: Collection[str] = (),
) -> tuple[str, str | None]:
    """Route a request for a path to the index that should answer it.

    A sub-directory of an already indexed tree is answered from that tree, filtered to the
    sub-directory: a workspace index holds the sub-repo's chunks already, and building a
    second index over them costs a full re-embed of the sub-tree for nothing.

    :param path: The requested local path or git URL.
    :param embedder_id: The normalized spec of the embedder that would answer the request.
    :param content: The requested content types.
    :param capsules: The requested context-capsule configuration.
    :param loaded_roots: Roots a caller already holds in memory for exactly this content.
    :return: The root to index, and the root-relative prefix to restrict it to (None = the whole root).
    """
    if is_git_url(path):
        return path, None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        return path, None
    # An index of exactly this path, in memory or on disk, keeps serving it: someone asked for
    # it deliberately, and it is already paid for.
    if str(resolved) in loaded_roots or has_cached_index(str(resolved), content):
        return path, None
    ancestor = find_ancestor_index_root(str(resolved), embedder_id, content, capsules, loaded_roots)
    if ancestor is None:
        return path, None
    prefix = resolved.relative_to(ancestor).as_posix()
    logger.info("serving %s from the %s index (subtree filter)", resolved, ancestor)
    return ancestor, prefix


def indexed_ancestor_hint(path: str, content: Sequence[ContentType] = (ContentType.CODE,)) -> str | None:
    """Return the "you are inside an indexed tree" advice for a refused build, or None.

    Staleness is deliberately not checked here: a stale ancestor index still means the answer
    to a refused sub-tree build is to search the ancestor, not to pay for a second index.

    :param path: The path whose build was refused.
    :param content: The content types that were requested.
    :return: One sentence naming the ancestor and the ways out, or None when there is no ancestor index.
    """
    if is_git_url(path):
        return None
    try:
        resolved = Path(path).expanduser().resolve()
    except OSError:  # pragma: no cover - resolve() only raises on exotic filesystems
        return None
    for ancestor in _ancestor_directories(resolved):
        if has_cached_index(str(ancestor), content):
            return (
                f"{resolved} is inside {ancestor}, which is already indexed: search {ancestor} instead, "
                f"or pass the workspace root; to index {resolved} on its own anyway set {CONFIRM_ENV}=1."
            )
    return None


def load_manifest_for_incremental(
    path: str, embedder_id: str, content: Sequence[ContentType], capsules: CapsuleOptions | None = None
) -> dict[str, FileManifestEntry] | None:
    """Load only the file manifest of a compatible cached index.

    The mtimes alone answer what a build would reuse, at the cost of one small JSON read
    instead of loading every chunk and the whole vector matrix.

    :param path: Source path used to locate the cached index.
    :param embedder_id: The normalized spec string of the requested embedder.
    :param content: Content types the cached index must support.
    :param capsules: The requested context-capsule configuration.
    :return: The manifest, or None when there is no reusable index.
    """
    try:
        persistence_path = PersistencePath.from_path(find_index_from_cache_folder(path, content))
        if persistence_path.non_existing():
            return None
        with open(persistence_path.metadata, encoding="utf-8") as f:
            metadata = json.load(f)
        if not _metadata_matches(metadata, embedder_id, content, CapsuleOptions.resolve(capsules).key):
            return None
        raw_manifest = metadata.get("files")
        if not raw_manifest:
            return None
        return {indexed_path: FileManifestEntry(**entry) for indexed_path, entry in raw_manifest.items()}
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.debug("Unable to read the cached manifest for %s", path, exc_info=True)
        return None


def load_previous_for_incremental(
    path: str, embedder_id: str, content: Sequence[ContentType], capsules: CapsuleOptions | None = None
) -> PreviousIndex | None:
    """Load compatible index state for incremental reuse.

    :param path: Source path used to locate the cached index.
    :param embedder_id: The normalized spec string of the requested embedder.
    :param content: Content types the cached index must support.
    :param capsules: The requested context-capsule configuration.
    :return: Previous index state, or None if the cache is unavailable or invalid.
    """
    try:
        manifest = load_manifest_for_incremental(path, embedder_id, content, capsules)
        if manifest is None:
            return None
        persistence_path = PersistencePath.from_path(find_index_from_cache_folder(path, content))

        chunks = list(load_chunks(persistence_path.chunks))

        # Incremental reindexing writes new rows into this matrix, so it cannot be a mapped view.
        vectors = SelectableBasicBackend.load(persistence_path.semantic_index, writable=True).vectors
        bm25_index = BM25.load(persistence_path.bm25_index)
        chunk_count = len(chunks)
        if not (chunk_count == vectors.shape[0] == len(bm25_index.doc_order)):
            return None
        expected_ids: list[str] = []
        next_start = 0
        for indexed_path, entry in manifest.items():
            if entry.start != next_start or any(
                chunk.file_path != indexed_path for chunk in chunks[entry.start : entry.end]
            ):
                return None
            expected_ids.extend(make_chunk_id(indexed_path, slot) for slot in range(entry.count))
            next_start += entry.count
        if next_start != chunk_count or bm25_index.doc_order != expected_ids:
            return None

        return PreviousIndex(chunks=chunks, vectors=vectors, manifest=manifest, bm25_index=bm25_index)
    except (OSError, orjson.JSONDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.debug("Unable to reuse incremental cache for %s", path, exc_info=True)
        return None

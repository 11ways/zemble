import hashlib
import json
import logging
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import orjson

from zemble.index.bm25 import BM25
from zemble.index.dense import SelectableBasicBackend
from zemble.index.file_walker import walk_files
from zemble.index.files import FileStatus, get_extensions, get_file_status
from zemble.index.types import CACHE_FORMAT_VERSION, FileManifestEntry, PersistencePath, PreviousIndex, make_chunk_id
from zemble.types import Chunk, ContentType
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


def _metadata_matches(metadata: dict, embedder_id: str, content: Sequence[ContentType]) -> bool:
    """Return True if the stored metadata is compatible with the requested parameters.

    A cache built with a different embedder is never reused silently: mixing vector spaces
    produces plausible-looking nonsense, so the mismatch is logged by name and rebuilt.

    :param metadata: The stored metadata document.
    :param embedder_id: The normalized spec string of the requested embedder.
    :param content: The requested content types.
    :return: Whether the cache can be reused.
    """
    from zemble.chunking.chunking import _DESIRED_CHUNK_LENGTH_CHARS  # avoid circular import at module level

    try:
        content_type = tuple(ContentType(s) for s in metadata["content_type"])
        # chunk_size and cache_version are absent in indexes built before those fields were added;
        # treat that as a mismatch so old caches are transparently rebuilt in the current format.
        chunk_size_ok = metadata.get("chunk_size") == _DESIRED_CHUNK_LENGTH_CHARS
        version_ok = metadata.get("cache_version") == CACHE_FORMAT_VERSION
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
        return set(content_type) == set(content) and chunk_size_ok and version_ok
    except (KeyError, ValueError):
        return False


def get_validated_cache(path: str, embedder_id: str, content: Sequence[ContentType]) -> Path | None:
    """Validates the cache folder and returns the index path.

    :param path: Source path or git URL the index was built from.
    :param embedder_id: The normalized spec string of the requested embedder.
    :param content: The requested content types.
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
    if not _metadata_matches(metadata, embedder_id, content):
        return None

    if is_git_url(str(path)):
        return index_path

    write_time = metadata["time"]
    extensions = get_extensions(content)

    path_as_path = Path(path).resolve()
    stored_files = metadata.get("files", {})
    current_files = []
    for file_path in walk_files(path_as_path, extensions=extensions):
        file_status = get_file_status(file_path, write_time)
        if file_status == FileStatus.NEWER:
            return None
        if file_status != FileStatus.VALID:
            continue
        current_files.append(str(file_path.relative_to(path_as_path)))

    if set(current_files) != set(stored_files):
        return None

    return index_path


def load_previous_for_incremental(path: str, embedder_id: str, content: Sequence[ContentType]) -> PreviousIndex | None:
    """Load compatible index state for incremental reuse.

    :param path: Source path used to locate the cached index.
    :param embedder_id: The normalized spec string of the requested embedder.
    :param content: Content types the cached index must support.
    :return: Previous index state, or None if the cache is unavailable or invalid.
    """
    try:
        index_path = find_index_from_cache_folder(path, content)
        persistence_path = PersistencePath.from_path(index_path)
        if persistence_path.non_existing():
            return None

        with open(persistence_path.metadata, encoding="utf-8") as f:
            metadata = json.load(f)
        if not _metadata_matches(metadata, embedder_id, content):
            return None

        raw_manifest = metadata.get("files")
        if not raw_manifest:
            return None
        manifest = {indexed_path: FileManifestEntry(**entry) for indexed_path, entry in raw_manifest.items()}

        with open(persistence_path.chunks, "rb") as f:
            chunks = [Chunk.from_dict(item) for item in orjson.loads(f.read())]

        vectors = SelectableBasicBackend.load(persistence_path.semantic_index).vectors
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

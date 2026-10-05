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
from zemble.index.chunk_store import file_paths_of, load_chunks
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


def exclude_digest(exclude: Sequence[str]) -> str:
    """Return the stable digest of a build's exclude patterns, or "" when it excluded nothing.

    Order and repetition are irrelevant to what a walk skips, so they are irrelevant here too.
    """
    normalized = sorted({pattern.strip() for pattern in exclude if pattern.strip()})
    if not normalized:
        return ""
    return hashlib.new("sha256", "\n".join(normalized).encode("utf-8")).hexdigest()[:16]


def cache_key(path: str, exclude: Sequence[str] = ()) -> str:
    """Compute the sha256 cache key for a local path or git URL.

    A build told to exclude paths indexes a different tree than the same root does without
    them, so it gets its own key - and only then: a plain build's key is what it always was.
    """
    if is_git_url(path):
        data = path.encode("utf-8")
    else:
        normalized = Path(path).expanduser().resolve()
        data = str(normalized).encode("utf-8")
    digest = exclude_digest(exclude)
    if digest:
        data += b"\x00exclude=" + digest.encode("ascii")
    return hashlib.new("sha256", data).hexdigest()


def find_index_from_cache_folder(
    path: str, content: Sequence[ContentType] = (ContentType.CODE,), exclude: Sequence[str] = ()
) -> Path:
    """Find an exact content index in the cache for a project path."""
    cache_dir = resolve_cache_folder() / cache_key(path, exclude)
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
    """Save an index to the cache folder if it was freshly built.

    The exclude patterns the index was built with come off the index itself, so a pruned
    build can never be written over the plain index of the same root.
    """
    if not index.loaded_from_disk:
        index.save(find_index_from_cache_folder(path, index.storage_content, index.exclude))
        retire_covered_indexes(path, index.storage_content, index.exclude)


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
    path: str,
    embedder_id: str,
    content: Sequence[ContentType],
    capsules: CapsuleOptions | None = None,
    exclude: Sequence[str] = (),
) -> Path | None:
    """Validates the cache folder and returns the index path.

    :param path: Source path or git URL the index was built from.
    :param embedder_id: The normalized spec string of the requested embedder.
    :param content: The requested content types.
    :param capsules: The requested context-capsule configuration.
    :param exclude: Extra gitignore-style patterns the build was told to skip.
    :return: The reusable index path, or None if it must be rebuilt.
    """
    index_path = find_index_from_cache_folder(path, content, exclude)
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
    for walked in walk_entries(path_as_path, extensions=extensions, ignore=list(exclude)):
        file_status = get_file_status(walked.path, write_time, walked.stat)
        if file_status == FileStatus.NEWER:
            return None
        if file_status != FileStatus.VALID:
            continue
        current_files.add(walked.relative_path)

    if current_files != set(stored_files):
        return None

    return index_path


def _ordered(content: Collection[ContentType]) -> tuple[ContentType, ...]:
    """Return content types in their declaration order, the spelling every key and folder uses."""
    return tuple(content_type for content_type in ContentType if content_type in content)


def stored_variants(folder: Path) -> list[tuple[tuple[ContentType, ...], Path, dict]]:
    """Return every complete index stored in one key folder: its content, its folder and its metadata.

    :param folder: A cache key folder, holding `index` and `index-<scope>` folders.
    :return: The complete, readable variants, narrowest first.
    """
    found = []
    for index_path in sorted(folder.glob("index*")) if folder.is_dir() else []:
        content = _content_of_folder(index_path.name)
        persistence_path = PersistencePath.from_path(index_path)
        if content is None or persistence_path.non_existing():
            continue
        try:
            with open(persistence_path.metadata, encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, ValueError):
            metadata = {}
        found.append((content, index_path, metadata if isinstance(metadata, dict) else {}))
    return sorted(found, key=lambda variant: len(variant[0]))


def _content_of_folder(name: str) -> tuple[ContentType, ...] | None:
    """Read the content selection back out of an index folder name `find_index_from_cache_folder` gave."""
    if name == "index":
        return (ContentType.CODE,)
    scope = name.removeprefix("index-")
    if scope == name:
        return None
    try:
        return _ordered({ContentType(value) for value in scope.split("-")})
    except ValueError:
        return None


def covering_content(
    path: str,
    embedder_id: str,
    content: Sequence[ContentType],
    capsules: CapsuleOptions | None = None,
    exclude: Sequence[str] = (),
) -> tuple[ContentType, ...]:
    """Return the content selection a request for *content* is stored and served as.

    One root keeps one index on disk: the widest compatible one that covers the request, or the
    request itself when nothing covers it. A code request on a root that already has a code+docs
    index is answered from that index, narrowed to code, instead of embedding and storing the
    same code a second time; a narrower index left beside it is never read again.

    :param path: Local path or git URL.
    :param embedder_id: The normalized spec of the embedder that would answer the request.
    :param content: The requested content types.
    :param capsules: The requested context-capsule configuration.
    :param exclude: The exclude patterns the index was built with.
    :return: The content types to load or build.
    """
    wanted = _ordered(content)
    if is_git_url(path):
        return wanted
    capsule_key = CapsuleOptions.resolve(capsules).key
    variants = stored_variants(find_index_from_cache_folder(path, wanted, exclude).parent)
    for stored, _index_path, metadata in reversed(variants):
        if set(wanted) <= set(stored) and _metadata_matches(metadata, embedder_id, stored, capsule_key):
            return stored
    return wanted


def covered_variants(folder: Path) -> list[Path]:
    """Return the index folders in a key folder that a wider compatible sibling already covers.

    A narrower index is covered when a sibling holds every content type it holds and was built
    with the same embedder, capsules, chunk size and format: requests for it are answered from
    the sibling, so nothing reads it again.
    """
    variants = stored_variants(folder)
    covered = []
    for content, index_path, metadata in variants:
        for wider, _wider_path, wider_metadata in variants:
            if not set(content) < set(wider):
                continue
            same_build = all(metadata.get(key) == wider_metadata.get(key) for key in _BUILD_IDENTITY)
            if same_build and wider_metadata.get("cache_version") == CACHE_FORMAT_VERSION:
                covered.append(index_path)
                break
    return covered


#: The metadata that has to agree before one stored index may answer for another.
_BUILD_IDENTITY = ("embedder", "capsules", "chunk_size", "cache_version", "exclude")


def index_component_files(index_path: Path) -> list[Path]:
    """Return what an index consists of, leaving out the symbol graph that shares its `index` folder."""
    persistence = PersistencePath.from_path(index_path)
    return [
        persistence.chunks,
        persistence.bm25_index,
        persistence.semantic_index,
        persistence.symbols,
        persistence.metadata,
    ]


def remove_index_components(index_path: Path) -> None:
    """Delete one stored index, metadata last so a half-removed one never reads as complete."""
    *stores, metadata = index_component_files(index_path)
    metadata.unlink(missing_ok=True)
    for store in stores:
        if store.is_dir():
            shutil.rmtree(store)
        else:
            store.unlink(missing_ok=True)


def retire_covered_indexes(path: str, content: Sequence[ContentType], exclude: Sequence[str] = ()) -> list[Path]:
    """Delete the narrower indexes of a root that the index just saved for *content* covers.

    :return: The index folders whose components were removed.
    """
    folder = find_index_from_cache_folder(path, content, exclude).parent
    retired = covered_variants(folder)
    for index_path in retired:
        remove_index_components(index_path)
        logger.info(
            "removed %s: the %s index of the same root covers it", index_path, "-".join(c.value for c in content)
        )
    return retired


def ancestor_answering(root: str, content: Sequence[ContentType], metadata: dict) -> str | None:
    """Return the nearest ancestor whose stored index would answer for a sub-root's index, or None.

    The ancestor has to be what routing would pick (a compatible index covering the content, see
    `resolve_index_root`) and has to hold files under the sub-root: an ancestor whose walk skipped
    it, a nested repository it excludes, answers nothing there and the sub-root's own index stays.

    :param root: The sub-root the index was built for.
    :param content: The content types it holds.
    :param metadata: Its stored metadata, whose embedder and capsules the ancestor must share.
    :return: The ancestor root, or None.
    """
    embedder = metadata.get("embedder")
    if not isinstance(embedder, str):
        return None
    capsule_key = str(metadata.get("capsules", ""))
    resolved = Path(root)
    for ancestor in _ancestor_directories(resolved):
        variants = stored_variants(find_index_from_cache_folder(str(ancestor), content).parent)
        for stored, _path, ancestor_metadata in variants:
            if not (
                set(content) <= set(stored) and _metadata_matches(ancestor_metadata, embedder, stored, capsule_key)
            ):
                continue
            prefix = f"{resolved.relative_to(ancestor).as_posix()}/"
            if any(path.startswith(prefix) for path in ancestor_metadata.get("files", {})):
                return str(ancestor)
    return None


def has_cached_index(
    path: str, content: Sequence[ContentType] = (ContentType.CODE,), exclude: Sequence[str] = ()
) -> bool:
    """Return whether a complete index covering *content* exists for a path, without checking it for staleness."""
    folder = find_index_from_cache_folder(path, content, exclude).parent
    return any(set(content) <= set(stored) for stored, _path, _metadata in stored_variants(folder))


def cached_index_compatible(
    path: str, embedder_id: str, content: Sequence[ContentType], capsules: CapsuleOptions | None = None
) -> bool:
    """Return whether a complete on-disk index covering *content* for *path* was built with these parameters.

    Freshness is deliberately NOT checked: a stale ancestor index is still the right index to
    load and refresh incrementally, which is far cheaper than building a second one over a
    sub-tree it already covers.
    """
    capsule_key = CapsuleOptions.resolve(capsules).key
    return any(
        set(content) <= set(stored) and _metadata_matches(metadata, embedder_id, stored, capsule_key)
        for stored, _path, metadata in stored_variants(find_index_from_cache_folder(path, content).parent)
    )


def _ancestor_directories(path: Path) -> list[Path]:
    """Return the strict ancestors of a directory, nearest first."""
    return list(path.parents)


def find_ancestor_index_root(
    path: str,
    embedder_id: str,
    content: Sequence[ContentType] = (ContentType.CODE,),
    capsules: CapsuleOptions | None = None,
    loaded_roots: Collection[str] = (),
    on_disk: bool = True,
) -> str | None:
    """Return the nearest ancestor of *path* that already has a usable index of the same content.

    An ancestor held in memory counts without a disk check; one on disk only has to be
    COMPATIBLE (same embedder, content types and capsules): a stale ancestor is loaded and
    refreshed incrementally by the normal load path, never rebuilt as a second index.

    :param path: The requested directory.
    :param embedder_id: The normalized spec of the embedder that would answer the request.
    :param content: The requested content types.
    :param capsules: The requested context-capsule configuration.
    :param loaded_roots: Roots a caller already holds in memory for exactly this content.
    :param on_disk: Whether an ancestor that is only on disk (compatible, possibly stale) may be returned.
    :return: The ancestor root, or None when the sub-path needs its own index.
    """
    for ancestor in _ancestor_directories(Path(path)):
        candidate = str(ancestor)
        if candidate in loaded_roots:
            return candidate
        if on_disk and cached_index_compatible(candidate, embedder_id, content, capsules):
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
    # Precedence: an index of exactly this path that is already IN MEMORY, then an ancestor
    # that is already in memory (free, no second resident index), then an index of exactly
    # this path on disk (someone built it deliberately), then an ancestor on disk, else build.
    if str(resolved) in loaded_roots:
        return path, None
    loaded_ancestor = find_ancestor_index_root(
        str(resolved), embedder_id, content, capsules, loaded_roots, on_disk=False
    )
    if loaded_ancestor is not None:
        return _subtree_of(resolved, loaded_ancestor)
    if has_cached_index(str(resolved), content):
        return path, None
    ancestor = find_ancestor_index_root(str(resolved), embedder_id, content, capsules, loaded_roots)
    if ancestor is None:
        return path, None
    return _subtree_of(resolved, ancestor)


def _subtree_of(resolved: Path, ancestor: str) -> tuple[str, str]:
    """Answer for a sub-path from an ancestor root, logging the routing once."""
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
    path: str,
    embedder_id: str,
    content: Sequence[ContentType],
    capsules: CapsuleOptions | None = None,
    exclude: Sequence[str] = (),
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
        persistence_path = PersistencePath.from_path(find_index_from_cache_folder(path, content, exclude))
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
    path: str,
    embedder_id: str,
    content: Sequence[ContentType],
    capsules: CapsuleOptions | None = None,
    exclude: Sequence[str] = (),
) -> PreviousIndex | None:
    """Load compatible index state for incremental reuse.

    :param path: Source path used to locate the cached index.
    :param embedder_id: The normalized spec string of the requested embedder.
    :param content: Content types the cached index must support.
    :param capsules: The requested context-capsule configuration.
    :return: Previous index state, or None if the cache is unavailable or invalid.
    """
    try:
        manifest = load_manifest_for_incremental(path, embedder_id, content, capsules, exclude)
        if manifest is None:
            return None
        persistence_path = PersistencePath.from_path(find_index_from_cache_folder(path, content, exclude))

        # Mapped, not materialized: a build reuses these chunks by reference, and the
        # verification below reads the path column rather than building 100k Chunk objects.
        chunks = load_chunks(persistence_path.chunks)

        # Mapped read-only: the build copies this matrix itself if it has a row to write.
        vectors = SelectableBasicBackend.load(persistence_path.semantic_index).vectors
        bm25_index = BM25.load(persistence_path.bm25_index)
        chunk_count = len(chunks)
        if not (chunk_count == vectors.shape[0] == len(bm25_index.doc_order)):
            return None
        stored_paths = file_paths_of(chunks)
        expected_ids: list[str] = []
        next_start = 0
        for indexed_path, entry in manifest.items():
            if entry.start != next_start or any(
                stored_path != indexed_path for stored_path in stored_paths[entry.start : entry.end]
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

"""Cache entries nothing will read again: what `zemble clear orphans` finds and removes."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from zemble.cache import (
    ROOT_MARKER,
    USED_MARKER,
    ancestor_answering,
    cache_key,
    covered_variants,
    index_component_files,
    marked_root,
    remove_index_components,
    stored_variants,
)
from zemble.daemon.protocol import process_alive
from zemble.embedding.gc import indexes_by_family, last_activity
from zemble.graph.store import (
    answering_ancestor,
    graph_covers,
    graph_files,
    graph_root_of,
    remove_graph,
    retired_graph_files,
    sweep_graph_folder,
)
from zemble.index.generation import LOCK_NAME, STAGING_PREFIX
from zemble.openfiles import held_open, open_files

#: A cache entry's folder name: the sha256 of its source.
KEY_DIR_NAME = re.compile(r"^[a-f0-9]{64}$")
#: What `zemble.index.columnar` writes beside a column before renaming it: `<name>.<pid>.<thread>.tmp[.npy]`.
_TEMP_LEFTOVER = re.compile(r".+\.\d+\.\d+\.tmp(\.npy)?$")
#: A file a build names after its process, `<name>.building-<pid>`, which the build renames when done.
_PID_LEFTOVER = re.compile(r".+\.building-(\d+)$")
#: A staging folder a build writes its next index generation into (`zemble.index.generation`),
#: or the temporary folder an older daemon rebuilt in.
_STAGING_LEFTOVER = re.compile(rf"^{re.escape(STAGING_PREFIX)}.+-(\d+)-[^-]+$|^rebuild-.+$")
#: A temporary file younger than this may still be about to be renamed by a live writer.
_TEMP_MIN_AGE_SECONDS = 3600
#: Days a git-URL entry or an unused embedder file is kept after it was last written.
DEFAULT_MAX_AGE_DAYS = 30
_SECONDS_PER_DAY = 86400


class OrphanKind(str, Enum):
    """Why a cache entry is an orphan; each member names itself in a report and says how it is removed."""

    label: str

    def __new__(cls, value: str, label: str) -> "OrphanKind":
        """Build a member from its wire value and the phrase a report names it by."""
        member = str.__new__(cls, value)
        member._value_ = value
        member.label = label
        return member

    INDEX_ROOT_GONE = ("index-root-gone", "index whose root is gone")
    GRAPH_ROOT_GONE = ("graph-root-gone", "graph whose root is gone")
    GIT_URL_STALE = ("git-url-stale", "stale git-URL index")
    EMBEDDER_UNUSED = ("embedder-unused", "embedder cache no index uses")
    TEMP_LEFTOVER = ("temp-leftover", "temporary leftover")
    STAGING_LEFTOVER = ("staging-leftover", "index generation a killed build left unpublished")
    RETIRED_GRAPH = ("retired-graph", "retired graph version or superseded graph.sqlite")
    INDEX_COVERED = ("index-covered", "index a wider index of the same root covers")
    INDEX_UNDER_ANCESTOR = ("index-under-ancestor", "sub-root index an ancestor's index answers for")
    GRAPH_UNDER_ANCESTOR = ("graph-under-ancestor", "sub-root graph an ancestor's graph answers for")
    NAMED_ROOT_GONE = ("named-root-gone", "cache folder whose root is gone")
    UNNAMED_STALE = ("unnamed-stale", "cache folder nothing names, unwritten for an hour")
    LEAST_RECENTLY_USED = ("least-recently-used", "cache entry least recently used past the cache's size bound")


#: The kinds whose root no longer exists, or that nothing names any more: what a daemon prunes on its own, on start
#: and periodically.
ROOT_GONE_KINDS = frozenset(
    {OrphanKind.INDEX_ROOT_GONE, OrphanKind.GRAPH_ROOT_GONE, OrphanKind.NAMED_ROOT_GONE, OrphanKind.UNNAMED_STALE}
)
#: What a build whose process is gone left behind: a daemon sweeps these at start and on every prune.
DEAD_WRITER_KINDS = frozenset({OrphanKind.TEMP_LEFTOVER, OrphanKind.STAGING_LEFTOVER})


@dataclass
class Orphan:
    """One removable cache entry: a whole key folder, a set of files, or a graph folder's retired files."""

    kind: OrphanKind
    target: Path
    size: int
    files: tuple[Path, ...] = ()
    detail: str = ""


def find_orphans(cache_folder: Path, *, max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> list[Orphan]:
    """List every orphan in a cache folder, deleting nothing.

    :param cache_folder: The zemble cache folder.
    :param max_age_days: Age past which a git-URL entry and an unused embedder file go.
    :return: The orphans, key folders first, in a stable order.
    """
    now = time.time()
    oldest = now - max_age_days * _SECONDS_PER_DAY
    orphans: list[Orphan] = []
    doomed: set[Path] = set()
    logged = logged_roots(cache_folder) if cache_folder.is_dir() else {}
    for entry in sorted(cache_folder.iterdir()) if cache_folder.is_dir() else []:
        if not (entry.is_dir() and KEY_DIR_NAME.match(entry.name)):
            continue
        orphan = _key_dir_orphan(entry, oldest, logged, now - _TEMP_MIN_AGE_SECONDS)
        if orphan is not None:
            orphans.append(orphan)
            doomed.add(entry)
            continue
        graph = entry / "index"
        retired = tuple(retired_graph_files(graph)) if graph.is_dir() else ()
        if retired:
            orphans.append(Orphan(OrphanKind.RETIRED_GRAPH, graph, sum(map(_size, retired)), retired))
        covered = covered_variants(entry)
        for index_path in covered:
            orphans.append(_index_components_orphan(OrphanKind.INDEX_COVERED, index_path))
        orphans.extend(_under_ancestor(entry, set(covered)))
    orphans.extend(_unused_embedders(cache_folder, oldest))
    orphans.extend(_temp_leftovers(cache_folder, doomed, now))
    return orphans


#: The cache's size bound when ZEMBLE_CACHE_MAX_GB names none. Measured 2026-11-24 on the workspace host: the key
#: folders the daemon served in the last day held 11.2 GB (the last three days 12.0 GB) of 17.5 GB.
DEFAULT_CACHE_MAX_GB = 12.0
CACHE_MAX_GB_ENV = "ZEMBLE_CACHE_MAX_GB"


def over_bound(cache_folder: Path, *, max_bytes: int, held_roots: Collection[str]) -> list[Orphan]:
    """List the cache entries past the size bound, least recently used first out, deleting nothing.

    Entries are key folders and embedder files. Newest use first, each entry counts while the total stays within
    *max_bytes*; once one does not fit, every less recently used entry goes, however small. An entry a process holds
    a file of open, one written within the hour, or a key folder of one of *held_roots* never goes and counts first.

    :param held_roots: The roots a running daemon serves, loads or watches.
    :return: The entries to remove; empty where /proc cannot say what is open.
    """
    opened = open_files()
    if opened is None or not cache_folder.is_dir():
        return []
    held_keys = {cache_key(root) for root in held_roots}
    fresh = time.time() - _TEMP_MIN_AGE_SECONDS
    entries: list[tuple[float, bool, Orphan]] = []
    for entry in sorted(cache_folder.iterdir()):
        if entry.is_dir() and KEY_DIR_NAME.match(entry.name):
            files = tuple(_files_under(entry))
            data_write = _newest_data_write(entry)
            try:
                used = (entry / USED_MARKER).stat().st_mtime
            except FileNotFoundError:
                used = data_write
            held = entry.name in held_keys or data_write >= fresh or _any_open(files, opened)
            orphan = Orphan(OrphanKind.LEAST_RECENTLY_USED, entry, sum(map(_size, files)), files)
            entries.append((max(used, data_write), held, orphan))
    directory = cache_folder / "embeddings"
    for path in sorted(directory.glob("*.sqlite")) if directory.is_dir() else []:
        files = tuple(candidate for candidate in _embedder_files(path) if candidate.exists())
        try:
            held = _any_open(files, opened)
            used = max(candidate.stat().st_mtime for candidate in files) if held else last_activity(path)
        except FileNotFoundError:
            continue
        orphan = Orphan(OrphanKind.LEAST_RECENTLY_USED, path, sum(map(_size, files)), files)
        entries.append((used, held or used >= fresh, orphan))
    entries.sort(key=lambda item: -item[0])
    total = sum(orphan.size for _used, held, orphan in entries if held)
    doomed: list[Orphan] = []
    for _used, held, orphan in entries:
        if held:
            continue
        if not doomed and total + orphan.size <= max_bytes:
            total += orphan.size
            continue
        doomed.append(orphan)
    return doomed


def _any_open(files: Collection[Path], opened: set[tuple[int, int]]) -> bool:
    """Return whether any of *files* is among the open (device, inode) pairs."""
    for path in files:
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        if (stat.st_dev, stat.st_ino) in opened:
            return True
    return False


def _embedder_files(path: Path) -> list[Path]:
    """An embedder cache file and the sqlite files beside it."""
    return [path, *(path.with_name(path.name + suffix) for suffix in ("-wal", "-shm"))]


def remove_orphan(orphan: Orphan) -> bool:
    """Remove one orphan the way its kind says, and return whether anything was removed.

    A retired graph is swept as its writer, so a folder another process is writing is skipped.
    """
    match orphan.kind:
        case (
            OrphanKind.INDEX_ROOT_GONE
            | OrphanKind.GRAPH_ROOT_GONE
            | OrphanKind.NAMED_ROOT_GONE
            | OrphanKind.GIT_URL_STALE
            | OrphanKind.STAGING_LEFTOVER
        ):
            shutil.rmtree(orphan.target)
            return True
        case OrphanKind.UNNAMED_STALE | OrphanKind.LEAST_RECENTLY_USED:
            return _remove_unless_in_use(orphan)
        case OrphanKind.EMBEDDER_UNUSED | OrphanKind.TEMP_LEFTOVER:
            # Checked again right before deleting: the scan may be minutes old.
            if any(held_open(path) for path in orphan.files):
                return False
            for path in orphan.files:
                path.unlink(missing_ok=True)
            return True
        case OrphanKind.RETIRED_GRAPH:
            removed = sweep_graph_folder(orphan.target)
            return bool(removed)
        case OrphanKind.INDEX_COVERED | OrphanKind.INDEX_UNDER_ANCESTOR:
            # Only the index's own stores go: the symbol graph shares the `index` folder.
            remove_index_components(orphan.target)
            _prune_empty(orphan.target)
            return True
        case OrphanKind.GRAPH_UNDER_ANCESTOR:
            removed = remove_graph(orphan.target)
            _prune_empty(orphan.target)
            return removed



def _remove_unless_in_use(orphan: Orphan) -> bool:
    """Remove an entry unless, over what it holds NOW, a process has a file of it open or wrote its data in the hour.

    The scan that chose it may be minutes old, and a reader may have opened or written one of its files since.
    """
    directory = orphan.target.is_dir()
    files = tuple(_files_under(orphan.target)) if directory else orphan.files
    opened = open_files()
    if opened is None or _any_open(files, opened):
        return False
    if directory and _newest_data_write(orphan.target) >= time.time() - _TEMP_MIN_AGE_SECONDS:
        return False
    try:
        if directory:
            shutil.rmtree(orphan.target)
        else:
            for path in files:
                path.unlink(missing_ok=True)
    except OSError:
        return False
    return True

def _prune_empty(folder: Path) -> None:
    """Remove a variant folder and the root folder above it once nothing but a publish lock is left."""
    for path in (folder, folder.parent):
        try:
            if [entry.name for entry in path.iterdir()] == [LOCK_NAME]:
                (path / LOCK_NAME).unlink()
            path.rmdir()
        except OSError:
            return


def _index_components_orphan(kind: OrphanKind, index_path: Path, detail: str = "") -> Orphan:
    """Describe one stored index to remove, sized by its own stores, never the graph beside it."""
    files = tuple(index_component_files(index_path))
    size = sum(_tree_size(path) if path.is_dir() else _size(path) for path in files)
    return Orphan(kind, index_path, size, files, detail)


def _under_ancestor(entry: Path, covered: set[Path]) -> list[Orphan]:
    """List a sub-root's indexes and graph that an ancestor root's own already answers for.

    Routing sends a sub-path to its ancestor whenever the sub-root has nothing of its own, so
    these are only ever read because they exist. The sub-root must still exist: one that is gone
    is `INDEX_ROOT_GONE` / `GRAPH_ROOT_GONE` already.
    """
    orphans: list[Orphan] = []
    for content, index_path, metadata in stored_variants(entry):
        root = metadata.get("root_path")
        if index_path in covered or not isinstance(root, str) or not Path(root).is_dir():
            continue
        exclude = metadata.get("exclude")
        if cache_key(root, exclude if isinstance(exclude, list) else []) != entry.name:
            continue
        ancestor = ancestor_answering(root, content, metadata)
        if ancestor is not None:
            detail = f"{root} (answered from {ancestor})"
            orphans.append(_index_components_orphan(OrphanKind.INDEX_UNDER_ANCESTOR, index_path, detail))
    graph = entry / "index"
    root = graph_root_of(graph) if graph.is_dir() else None
    if root is not None and Path(root).is_dir() and cache_key(root) == entry.name:
        ancestor = answering_ancestor(Path(root))
        if ancestor is not None and graph_covers(*ancestor):
            files = tuple(graph_files(graph))
            detail = f"{root} (answered from {ancestor[0]})"
            orphans.append(Orphan(OrphanKind.GRAPH_UNDER_ANCESTOR, graph, sum(map(_size, files)), files, detail))
    return orphans


def _key_dir_orphan(entry: Path, oldest: float, logged: dict[str, str], fresh: float) -> Orphan | None:
    """Classify one key folder: by its indexes' metadata when it has any, else by its graph or root marker.

    :param fresh: Writes since this time mean a writer may still be naming the folder.
    """
    metadata = _index_metadata(entry)
    return _index_orphan(entry, metadata, oldest) if metadata else _graph_orphan(entry, logged, fresh)


def _index_orphan(entry: Path, metadata: list[dict], oldest: float) -> Orphan | None:
    """Return the orphan a key folder with indexes is, or None while any of them may still be read.

    A local entry's key is its root (and exclude patterns) hashed, and it goes once that root is
    gone. A git-URL entry's key hashes the URL while its stored root is the clone, deleted right
    after the build, so it goes by age; one without a readable build time is kept.
    """
    local_gone: list[str] = []
    url_times: list[object] = []
    for meta in metadata:
        root = meta.get("root_path")
        if not isinstance(root, str) or not root:
            return None
        exclude = meta.get("exclude")
        if cache_key(root, exclude if isinstance(exclude, list) else []) == entry.name:
            if Path(root).exists():
                return None
            local_gone.append(root)
        elif Path(root).exists():
            return None
        else:
            url_times.append(meta.get("time"))
    if local_gone and not url_times:
        return Orphan(OrphanKind.INDEX_ROOT_GONE, entry, _tree_size(entry), detail=local_gone[0])
    if url_times and not local_gone and all(isinstance(stamp, (int, float)) and stamp < oldest for stamp in url_times):
        return Orphan(OrphanKind.GIT_URL_STALE, entry, _tree_size(entry))
    return None


def _graph_orphan(entry: Path, logged: dict[str, str], fresh: float) -> Orphan | None:
    """Return the orphan a key folder without index metadata is once the root naming it is gone, else None.

    Its graph names its root; else its root marker does; else the root the daemon logs served whose key
    it is - the dupes caches of roots that existed before markers did.
    """
    graph = entry / "index"
    root = graph_root_of(graph) if graph.is_dir() else None
    if root is not None:
        if cache_key(root) != entry.name or Path(root).exists():
            return None
        return Orphan(OrphanKind.GRAPH_ROOT_GONE, entry, _tree_size(entry), detail=root)
    named = marked_root(entry) or logged.get(entry.name)
    if named is None:
        # Nothing names its root, and nothing can tell whether it exists: once nothing wrote it for an hour, nothing
        # will read it again. A graph store that names nothing readable may be mid-build: it stays.
        if not any(graph.glob("graph-*.sqlite")) and _newest_write(entry) < fresh:
            return Orphan(OrphanKind.UNNAMED_STALE, entry, _tree_size(entry), tuple(_files_under(entry)))
        return None
    if cache_key(named) != entry.name or Path(named).exists():
        return None
    return Orphan(OrphanKind.NAMED_ROOT_GONE, entry, _tree_size(entry), detail=named)


#: An absolute path as the daemon log writes one: a root it served, watched or rebuilt.
_LOGGED_PATH = re.compile(r"(?<![\w/])(/[^\s:'\"(),]+)")


def logged_roots(cache_folder: Path) -> dict[str, str]:
    """Map the cache key of every absolute path the daemon logs name to that path.

    The logs are zemble's own record of the roots it served; a key folder nothing else names is
    recognised by its key appearing here.
    """
    found: dict[str, str] = {}
    for log in sorted(cache_folder.glob("daemon.log*")):
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for path in set(_LOGGED_PATH.findall(text)):
            found.setdefault(cache_key(path.rstrip(".")), path.rstrip("."))
    return found


def _index_metadata(entry: Path) -> list[dict]:
    """Return the readable metadata of a key folder's index and overlay folders; unreadable ones are skipped."""
    documents = []
    for folder in sorted([*entry.glob("index*"), *entry.glob("overlay*")]):
        try:
            document = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(document, dict):
            documents.append(document)
    return documents


def _unused_embedders(cache_folder: Path, oldest: float) -> list[Orphan]:
    """List embedder cache files no saved index reads, untouched since `oldest` and open nowhere."""
    directory = cache_folder / "embeddings"
    if not directory.is_dir():
        return []
    used = set(indexes_by_family(cache_folder))
    orphans = []
    for path in sorted(directory.glob("*.sqlite")):
        if path.stem in used:
            continue
        files = [path, *(path.with_name(path.name + suffix) for suffix in ("-wal", "-shm"))]
        present = tuple(candidate for candidate in files if candidate.exists())
        if any(held_open(candidate) for candidate in present) or last_activity(path) >= oldest:
            continue
        orphans.append(
            Orphan(OrphanKind.EMBEDDER_UNUSED, path, sum(_size(candidate) for candidate in present), present)
        )
    return orphans


def _temp_leftovers(cache_folder: Path, doomed: set[Path], now: float) -> list[Orphan]:
    """List temp files and staging folders a killed writer left behind, outside key folders already going."""
    orphans = []
    for directory, subdirectories, names in os.walk(cache_folder):
        here = Path(directory)
        if any(here == folder or folder in here.parents for folder in doomed):
            continue
        for name in sorted(subdirectories):
            match = _STAGING_LEFTOVER.match(name)
            if match is None:
                continue
            subdirectories.remove(name)
            path = here / name
            if _left_by_a_live_writer(path, match.group(1), now):
                continue
            orphans.append(Orphan(OrphanKind.STAGING_LEFTOVER, path, _tree_size(path), (path,)))
        for name in sorted(names):
            match = _PID_LEFTOVER.match(name)
            if not _TEMP_LEFTOVER.match(name) and match is None:
                continue
            path = here / name
            if _left_by_a_live_writer(path, match.group(1) if match else None, now) or held_open(path):
                continue
            orphans.append(Orphan(OrphanKind.TEMP_LEFTOVER, path, _size(path), (path,)))
    return orphans


def _left_by_a_live_writer(path: Path, pid: str | None, now: float) -> bool:
    """Return whether a leftover may still belong to a running writer: its process lives, or it is recent.

    A leftover named after its process is its writer's while that process lives and never after,
    however young: a graph build killed mid-refresh leaves a scratch file of up to 15 GB.
    """
    try:
        modified = path.stat().st_mtime
    except FileNotFoundError:
        return True
    if pid is not None:
        return process_alive(int(pid))
    return now - modified < _TEMP_MIN_AGE_SECONDS


def _size(path: Path) -> int:
    """Return a file's size, zero when it is gone."""
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _newest_write(folder: Path) -> float:
    """Return when anything under a folder was last written, the folder's own time when it is empty."""
    newest = folder.stat().st_mtime
    for directory, _subdirectories, names in os.walk(folder):
        for name in names:
            try:
                newest = max(newest, (Path(directory) / name).stat().st_mtime)
            except FileNotFoundError:
                continue
    return newest


#: Files a reader or a marker writes, which say nothing about a key folder's data being in use.
_NOT_DATA = re.compile(rf"(-shm|-readers|\.lock)$|^({re.escape(ROOT_MARKER)}|{re.escape(USED_MARKER)})$")


def _newest_data_write(folder: Path) -> float:
    """Return when a key folder's data was last written: sqlite shared memory, locks and markers left out."""
    newest = 0.0
    for path in _files_under(folder):
        if _NOT_DATA.search(path.name):
            continue
        try:
            newest = max(newest, path.stat().st_mtime)
        except FileNotFoundError:
            continue
    return newest


def _files_under(folder: Path) -> Iterator[Path]:
    """Every file under a folder."""
    for directory, _subdirectories, names in os.walk(folder):
        for name in names:
            yield Path(directory) / name


def _tree_size(folder: Path) -> int:
    """Return the bytes every file under a folder takes."""
    total = 0
    for directory, _subdirectories, names in os.walk(folder):
        total += sum(_size(Path(directory) / name) for name in names)
    return total


__all__ = [
    "DEAD_WRITER_KINDS",
    "DEFAULT_MAX_AGE_DAYS",
    "ROOT_GONE_KINDS",
    "CACHE_MAX_GB_ENV",
    "DEFAULT_CACHE_MAX_GB",
    "Orphan",
    "OrphanKind",
    "find_orphans",
    "over_bound",
    "logged_roots",
    "remove_orphan",
]

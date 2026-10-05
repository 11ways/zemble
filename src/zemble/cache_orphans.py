"""Cache entries nothing will read again: what `zemble clear orphans` finds and removes."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from zemble.cache import (
    ancestor_answering,
    cache_key,
    covered_variants,
    index_component_files,
    remove_index_components,
    stored_variants,
)
from zemble.embedding.gc import indexes_by_family, last_activity
from zemble.graph.store import (
    graph_covers,
    graph_files,
    graph_present,
    graph_root_of,
    remove_graph,
    retired_graph_files,
    sweep_graph_folder,
)
from zemble.openfiles import held_open

#: A cache entry's folder name: the sha256 of its source.
KEY_DIR_NAME = re.compile(r"^[a-f0-9]{64}$")
#: What `zemble.index.columnar` writes beside a column before renaming it: `<name>.<pid>.<thread>.tmp[.npy]`.
_TEMP_LEFTOVER = re.compile(r".+\.\d+\.\d+\.tmp(\.npy)?$")
#: A temporary file younger than this may still be about to be renamed by a live writer.
_TEMP_MIN_AGE_SECONDS = 3600
#: Days a git-URL entry or an unused embedder file is kept after it was last written.
DEFAULT_MAX_AGE_DAYS = 30
_SECONDS_PER_DAY = 86400


class OrphanKind(str, Enum):
    """Why a cache entry is an orphan; each member says how it is removed."""

    INDEX_ROOT_GONE = "index-root-gone"
    GRAPH_ROOT_GONE = "graph-root-gone"
    GIT_URL_STALE = "git-url-stale"
    EMBEDDER_UNUSED = "embedder-unused"
    TEMP_LEFTOVER = "temp-leftover"
    RETIRED_GRAPH = "retired-graph"
    INDEX_COVERED = "index-covered"
    INDEX_UNDER_ANCESTOR = "index-under-ancestor"
    GRAPH_UNDER_ANCESTOR = "graph-under-ancestor"

    @property
    def label(self) -> str:
        """A short phrase naming the kind in a report."""
        match self:
            case OrphanKind.INDEX_ROOT_GONE:
                return "index whose root is gone"
            case OrphanKind.GRAPH_ROOT_GONE:
                return "graph whose root is gone"
            case OrphanKind.GIT_URL_STALE:
                return "stale git-URL index"
            case OrphanKind.EMBEDDER_UNUSED:
                return "embedder cache no index uses"
            case OrphanKind.TEMP_LEFTOVER:
                return "temporary leftover"
            case OrphanKind.RETIRED_GRAPH:
                return "retired graph version or superseded graph.sqlite"
            case OrphanKind.INDEX_COVERED:
                return "index a wider index of the same root covers"
            case OrphanKind.INDEX_UNDER_ANCESTOR:
                return "sub-root index an ancestor's index answers for"
            case OrphanKind.GRAPH_UNDER_ANCESTOR:
                return "sub-root graph an ancestor's graph answers for"


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
    for entry in sorted(cache_folder.iterdir()) if cache_folder.is_dir() else []:
        if not (entry.is_dir() and KEY_DIR_NAME.match(entry.name)):
            continue
        orphan = _key_dir_orphan(entry, oldest)
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


def remove_orphan(orphan: Orphan) -> bool:
    """Remove one orphan the way its kind says, and return whether anything was removed.

    A retired graph is swept as its writer, so a folder another process is writing is skipped.
    """
    match orphan.kind:
        case OrphanKind.INDEX_ROOT_GONE | OrphanKind.GRAPH_ROOT_GONE | OrphanKind.GIT_URL_STALE:
            shutil.rmtree(orphan.target)
            return True
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


def _prune_empty(folder: Path) -> None:
    """Remove a variant folder and the root folder above it once nothing is left in them."""
    for path in (folder, folder.parent):
        try:
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
        for ancestor in Path(root).parents:
            if graph_present(str(ancestor)) and graph_covers(
                str(ancestor), Path(root).relative_to(ancestor).as_posix()
            ):
                files = tuple(graph_files(graph))
                detail = f"{root} (answered from {ancestor})"
                orphans.append(Orphan(OrphanKind.GRAPH_UNDER_ANCESTOR, graph, sum(map(_size, files)), files, detail))
                break
    return orphans


def _key_dir_orphan(entry: Path, oldest: float) -> Orphan | None:
    """Classify one key folder: by its indexes' metadata when it has any, else by its graph."""
    metadata = _index_metadata(entry)
    return _index_orphan(entry, metadata, oldest) if metadata else _graph_orphan(entry)


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


def _graph_orphan(entry: Path) -> Orphan | None:
    """Return the orphan a graph-only key folder is once the root its graph names is gone, else None."""
    graph = entry / "index"
    if not graph.is_dir():
        return None
    root = graph_root_of(graph)
    if root is None or cache_key(root) != entry.name or Path(root).exists():
        return None
    return Orphan(OrphanKind.GRAPH_ROOT_GONE, entry, _tree_size(entry), detail=root)


def _index_metadata(entry: Path) -> list[dict]:
    """Return the readable metadata documents of a key folder's index folders; unreadable ones are skipped."""
    documents = []
    for folder in sorted(entry.glob("index*")):
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
    """List column temp files a killed save left behind, outside key folders already going."""
    orphans = []
    for directory, _subdirectories, names in os.walk(cache_folder):
        here = Path(directory)
        if any(here == folder or folder in here.parents for folder in doomed):
            continue
        for name in sorted(names):
            if not _TEMP_LEFTOVER.match(name):
                continue
            path = here / name
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            if now - stat.st_mtime < _TEMP_MIN_AGE_SECONDS or held_open(path):
                continue
            orphans.append(Orphan(OrphanKind.TEMP_LEFTOVER, path, stat.st_size, (path,)))
    return orphans


def _size(path: Path) -> int:
    """Return a file's size, zero when it is gone."""
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _tree_size(folder: Path) -> int:
    """Return the bytes every file under a folder takes."""
    total = 0
    for directory, _subdirectories, names in os.walk(folder):
        total += sum(_size(Path(directory) / name) for name in names)
    return total


__all__ = ["DEFAULT_MAX_AGE_DAYS", "Orphan", "OrphanKind", "find_orphans", "remove_orphan"]

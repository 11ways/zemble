"""Sqlite storage and incremental build of the Java symbol graph.

The graph lives beside the search index (same cache folder) but is independent of
it: building the graph never requires an index, and clearing one does not corrupt
the other.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import sqlite3
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from zemble.cache import find_index_from_cache_folder
from zemble.graph.facts import (
    TREE_SITTER_SOURCE,
    FactsFile,
    FactsFileState,
    FactsOverlay,
    FactsPlan,
    SkipBucket,
    SourceContribution,
    discover_facts_files,
    map_facts_files,
    matches_facts_glob,
    plan_facts,
    read_facts_files,
    symbol_facts,
)
from zemble.graph.generic import extract_generic_file, language_parser
from zemble.graph.hwk import extract_hwk_file
from zemble.graph.java import FileExtraction, extract_java_file
from zemble.graph.lookup import (
    HIERARCHY_KINDS,
    FileContext,
    MemoryLookup,
    SqliteLookup,
    SymbolLookup,
    context_from_row,
    declaration_keys,
    symbol_from_row,
)
from zemble.graph.model import Edge, EdgeKind, Resolution, Symbol
from zemble.graph.resolve import Resolver
from zemble.index.file_walker import ignored_prefix, walk_files
from zemble.index.files import detect_language, get_extensions
from zemble.languages.catalog import SPECS
from zemble.openfiles import held_open
from zemble.parallel import pool_context, pooled
from zemble.types import ContentType

logger = logging.getLogger(__name__)
DEFAULT_WORKERS = min(10, os.cpu_count() or 2)

GRAPH_FORMAT_VERSION = 8
#: The file naming the current graph version (first line) and the one it replaced (second).
GRAPH_POINTER_NAME = "graph.current"
#: The file whose OS lock is held by the one process allowed to write a workspace's graph.
GRAPH_LOCK_NAME = "graph.lock"
#: The single-file store older zemble kept; it is moved into the versioned layout on first open.
LEGACY_GRAPH_DB_NAME = "graph.sqlite"
#: The sidecars sqlite keeps beside a store.
_DB_SIDECARS = ("-journal", "-wal", "-shm")
#: The sidecar whose shared flock marks a version as being read; see `_hold_version`.
READERS_SUFFIX = "-readers"
#: Every sidecar a version file may have: sqlite's own, then the readers lock.
_VERSION_SIDECARS = (*_DB_SIDECARS, READERS_SUFFIX)
#: A graph version file, `graph-<n>.sqlite`, or one of its sidecars.
_VERSION_FILE = re.compile(r"graph-(\d+)\.sqlite(" + "|".join(map(re.escape, _VERSION_SIDECARS)) + ")?")
#: How often a reader re-resolves the pointer when the version it named was retired under it.
_HOLD_ATTEMPTS = 8
#: How long a reader waits on sqlite's own locks; in WAL mode only recovery holds one that long.
_READ_TIMEOUT_SECONDS = 30.0
#: How long a writer waits on sqlite's own locks, which only a checkpoint or recovery takes.
_WRITE_TIMEOUT_SECONDS = 30.0
#: What sqlite says when the file it opened is not a graph any more. A store that reports
#: any of these is never repaired in place: it is rebuilt from source, which is cheap
#: because a graph is derived data.
_CORRUPTION_MARKERS = ("malformed", "is not a database", "corrupt", "encrypted")
#: Free-page share above which a store is compacted into a fresh generation. Deletes never
#: return pages to the filesystem on their own, and an incremental refresh is mostly deletes.
_COMPACT_FREE_FRACTION = 0.25
#: The share `zemble graph compact` reclaims: asked for explicitly, it need not wait for drift.
_EXPLICIT_COMPACT_FREE_FRACTION = 0.02
#: Stores below this many pages are left alone; compacting a small file buys nothing.
_COMPACT_MIN_PAGES = 4096
# Edge kinds that are computed from resolved symbols rather than extracted from source.
# They are always recomputed for a re-resolved file, never reloaded and re-inserted.
_DERIVED_KINDS = (EdgeKind.OVERRIDES.value, EdgeKind.TESTS.value, EdgeKind.EXERCISES.value)
#: Languages with a hand-written extractor; every other language with a spec and a grammar
#: goes through the grammar-driven one. A language absent from both is skipped and counted.
_HAND_WRITTEN = {"java": extract_java_file, "hwk": extract_hwk_file}
#: The languages an extractor exists for: the hand-written lanes first, then every spec.
GRAPH_LANGUAGES: tuple[str, ...] = (*_HAND_WRITTEN, *(language for language in SPECS if language not in _HAND_WRITTEN))


def extractor_for(file_path: Path) -> Callable[[bytes, str], FileExtraction] | None:
    """The extractor that reads a file, or None when its language has none."""
    language = detect_language(file_path)
    if language is None:
        return None
    hand_written = _HAND_WRITTEN.get(language)
    if hand_written is not None:
        return hand_written
    if language in SPECS and language_parser(language) is not None:
        return partial(extract_generic_file, language=language)
    return None


_WORKER_CHUNK = 40
#: Files one build step holds in memory: extracted ones before they are written out, and
#: re-resolved ones while their edges resolve. A build's memory is bounded by this, never by
#: the size of the workspace; `tests/test_graph_incremental.py` pins that the batch size
#: changes nothing about the graph.
_BATCH_FILES = 100
#: The disposable database a build stages its unresolved and resolved edges in, beside the
#: store it writes; named like every other build leftover so `zemble clear orphans` finds it.
_SCRATCH_NAME = "graph-scratch.building-{pid}"
_MAX_FILE_BYTES = 2_000_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY, mtime_ns INTEGER, size INTEGER, package TEXT, imports TEXT
);
CREATE TABLE IF NOT EXISTS symbols (
    id TEXT PRIMARY KEY, kind TEXT, name TEXT, qualified_name TEXT, file_path TEXT,
    start_line INTEGER, end_line INTEGER, container_id TEXT, modifiers TEXT,
    annotations TEXT, signature TEXT, is_test INTEGER, param_types TEXT,
    annotation_args TEXT
);
CREATE TABLE IF NOT EXISTS refs (key INTEGER PRIMARY KEY, text TEXT NOT NULL UNIQUE);
CREATE TABLE IF NOT EXISTS edge_rows (
    src INTEGER NOT NULL, dst INTEGER, dst_name TEXT, kind TEXT, line INTEGER,
    resolution TEXT, candidate_count INTEGER, arity INTEGER, receiver TEXT,
    receiver_type TEXT, is_new INTEGER, file INTEGER NOT NULL, source TEXT, origin_ref TEXT
);
CREATE VIEW IF NOT EXISTS edges AS
    SELECT s.text AS src_id, d.text AS dst_id, e.dst_name, e.kind, e.line, e.resolution,
           e.candidate_count, e.arity, e.receiver, e.receiver_type, e.is_new, f.text AS file_path,
           e.source, e.origin_ref
    FROM edge_rows e JOIN refs s ON s.key = e.src LEFT JOIN refs d ON d.key = e.dst JOIN refs f ON f.key = e.file;
CREATE TABLE IF NOT EXISTS facts_status (
    path TEXT PRIMARY KEY, tool TEXT, tool_version TEXT, generated_at TEXT, language TEXT,
    mtime_ns INTEGER, size INTEGER, files_declared INTEGER, files_fresh INTEGER,
    files_stale INTEGER, unmapped INTEGER, paths TEXT, template_paths TEXT,
    fresh_paths TEXT, contributions TEXT, parse_buckets TEXT, error TEXT,
    generated_templates TEXT
);
CREATE TABLE IF NOT EXISTS facts_symbols (
    ref TEXT, file_path TEXT, line INTEGER, facts_file TEXT, language TEXT
);
CREATE TABLE IF NOT EXISTS decl_keys (
    symbol_id TEXT, key TEXT, value TEXT, file_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_qualified ON symbols(qualified_name);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path);
CREATE INDEX IF NOT EXISTS idx_symbols_container ON symbols(container_id);
CREATE INDEX IF NOT EXISTS idx_edge_rows_dst_kind ON edge_rows(dst, kind);
CREATE INDEX IF NOT EXISTS idx_edge_rows_src_kind ON edge_rows(src, kind);
CREATE INDEX IF NOT EXISTS idx_edge_rows_dstname_kind ON edge_rows(dst_name, kind);
CREATE INDEX IF NOT EXISTS idx_edge_rows_file ON edge_rows(file);
CREATE INDEX IF NOT EXISTS idx_facts_symbols_ref ON facts_symbols(ref, language);
CREATE INDEX IF NOT EXISTS idx_facts_symbols_file ON facts_symbols(facts_file);
CREATE INDEX IF NOT EXISTS idx_decl_keys_value ON decl_keys(key, value);
CREATE INDEX IF NOT EXISTS idx_decl_keys_file ON decl_keys(file_path);
"""


@dataclass
class GraphStats:
    """What one graph build did."""

    root: str
    files_scanned: int = 0
    extracted_files: int = 0
    unchanged_files: int = 0
    removed_files: int = 0
    reresolved_files: int = 0
    skipped_by_language: dict[str, int] = field(default_factory=dict)
    symbols: int = 0
    edges: int = 0
    resolution_counts: dict[str, int] = field(default_factory=dict)
    facts: dict[str, object] = field(default_factory=dict)
    duration_seconds: float = 0.0
    #: Whether this build rewrote the whole store because the one on disk was malformed.
    rebuilt_from_corruption: bool = False
    #: Whether this build rewrote the whole store because deletes had left too much of it free.
    compacted: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready view of the build statistics."""
        return {
            "root": self.root,
            "files_scanned": self.files_scanned,
            "extracted_files": self.extracted_files,
            "unchanged_files": self.unchanged_files,
            "removed_files": self.removed_files,
            "reresolved_files": self.reresolved_files,
            "skipped_by_language": self.skipped_by_language,
            "symbols": self.symbols,
            "edges": self.edges,
            "resolution_counts": self.resolution_counts,
            "facts": self.facts,
            "duration_seconds": round(self.duration_seconds, 3),
            "rebuilt_from_corruption": self.rebuilt_from_corruption,
            "compacted": self.compacted,
        }


def graph_folder(path: str) -> Path:
    """Return the cache folder that holds the graph for a project path."""
    return find_index_from_cache_folder(path, (ContentType.CODE,))


def _graph_dir(path: str) -> Path:
    """Return a project's graph folder, creating it if needed."""
    folder = graph_folder(path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def graph_db_path(path: str) -> Path | None:
    """Return the sqlite file of a project's current graph version, or None before the first build.

    Creates the graph folder, and first moves a single-file store an older zemble left into the
    versioned layout. The file is not held: a writer may retire it once it stops being current.
    """
    folder = _graph_dir(path)
    pointer = _readable_pointer(folder)
    return None if pointer is None else folder / pointer.current


def connect(path: str) -> sqlite3.Connection:
    """Open a project's current graph version for reading, holding it against retirement until closed.

    :raises GraphStoreMissing: If no graph has been built for the path yet.
    """
    connection = _connect_current(_graph_dir(path))
    if connection is None:
        raise GraphStoreMissing(f"no symbol graph has been built for {path!r}")
    return connection


class _HeldConnection(sqlite3.Connection):
    """A read-only connection that owns the shared lock keeping its version file on disk."""

    _hold: int | None = None

    def close(self) -> None:
        """Close the connection, then release the version it held."""
        try:
            super().close()
        finally:
            self._release()

    def _release(self) -> None:
        """Drop the shared lock; the writer may retire the version from now on."""
        hold, self._hold = self._hold, None
        if hold is not None:
            os.close(hold)

    def __del__(self) -> None:
        """Release the version of a connection nobody closed."""
        self._release()


def _connect_current(folder: Path) -> sqlite3.Connection | None:
    """Open the current version read-only under a shared hold, or None before the first build.

    A writer may retire the version a pointer named between the read of that pointer and the
    hold; the hold then reports it gone and the pointer is read again.

    :raises GraphStoreCorrupt: If the pointer keeps naming versions that vanish.
    """
    for _ in range(_HOLD_ATTEMPTS):
        pointer = _readable_pointer(folder)
        if pointer is None:
            return None
        hold = _hold_version(folder, pointer.current)
        if hold is None:
            continue
        try:
            connection = open_db(folder / pointer.current, read_only=True)
        except BaseException:
            os.close(hold)
            raise
        assert isinstance(connection, _HeldConnection)
        connection._hold = hold
        return connection
    raise GraphStoreCorrupt(f"graph pointer in {folder} kept naming retired versions")


def _hold_version(folder: Path, name: str) -> int | None:
    """Take the shared lock marking a version as read, or None when it was retired meanwhile.

    AIDEV-NOTE: the lock is on a `-readers` sidecar, never on the sqlite file. Closing ANY
    descriptor of a sqlite file drops every POSIX lock sqlite holds on it in this process, so a
    flock descriptor on the database itself would break sqlite's own locking the moment it was
    released. The writer retires a version only while holding this lock exclusively, and unlinks
    the sidecar last; a reader that reached the old sidecar or a recreated one sees, after its
    lock is granted, that the path no longer names its inode or that the database is gone.

    :return: The descriptor holding the lock, which the caller closes to release it.
    """
    lock_path = folder / f"{name}{READERS_SUFFIX}"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        held = os.fstat(descriptor)
        try:
            named = os.stat(lock_path)
        except FileNotFoundError:
            named = None
        if named is None or (named.st_dev, named.st_ino) != (held.st_dev, held.st_ino):
            os.close(descriptor)
            return None
        if not (folder / name).is_file():
            os.close(descriptor)
            return None
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def open_db(db_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open one graph version file, read-only, or writable under the durability the store is written with.

    A read-only connection never writes and never waits on a writer: in WAL mode it reads the
    last committed snapshot while a refresh is still mid-transaction.
    """
    if read_only:
        connection = sqlite3.connect(
            f"{db_path.resolve().as_uri()}?mode=ro",
            uri=True,
            timeout=_READ_TIMEOUT_SECONDS,
            factory=_HeldConnection,
        )
        connection.row_factory = sqlite3.Row
        return connection
    connection = sqlite3.connect(db_path, timeout=_WRITE_TIMEOUT_SECONDS)
    connection.row_factory = sqlite3.Row
    _apply_durability(connection)
    connection.executescript(_SCHEMA)
    _migrate(connection)
    return connection


def _apply_durability(connection: sqlite3.Connection) -> None:
    """Write through a WAL and fsync every commit.

    AIDEV-NOTE: this replaced `synchronous=OFF` + `journal_mode=MEMORY`, which threw the
    rollback journal away: a build killed mid-write (three OOM kills in one week) left a torn
    file that still answered queries. The javaweb store reached 1.95 GB with 1.28 GB of pages
    reachable from neither a tree nor the freelist, an `edges` btree with out-of-order rowids,
    and 4,751 ignored "database disk image is malformed" lines in the daemon log.
    It was then rollback-journal (DELETE) mode, because a full build was renamed over the one
    live file and a `-wal`/`-shm` pair left pointing at a replaced inode is a torn read. That
    mode made every reader wait on a refresh's minutes-long transaction once it spilled past
    the page cache ("database is locked" after 5 s). Now no file is ever renamed over another:
    each full build writes a new version, `graph-<n>.sqlite`, and readers find it through the
    `graph.current` pointer (see `_install`), so WAL is safe and readers never wait on the
    writer. Only one process writes at a time (`_writer_lock`).
    """
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")


class GraphStoreCorrupt(RuntimeError):
    """The graph store on disk cannot be read and has to be rebuilt from source."""


class GraphStoreMissing(RuntimeError):
    """No graph has been built for a workspace yet."""


def is_corruption(exc: BaseException) -> bool:
    """Return whether an error says the store itself is unreadable rather than the query."""
    if isinstance(exc, GraphStoreCorrupt):
        return True
    if not isinstance(exc, sqlite3.DatabaseError):
        return False
    message = str(exc).lower()
    return any(marker in message for marker in _CORRUPTION_MARKERS)


# ---- the versioned layout -------------------------------------------------


@dataclass(frozen=True)
class _Pointer:
    """The version file readers open, and the one it replaced, kept for readers still on it."""

    current: str
    previous: str | None = None


def _version_name(number: int) -> str:
    """Name the file of one graph version; `_VERSION_FILE` is the pattern it must match."""
    return f"graph-{number}.sqlite"


def _is_version_name(name: str) -> bool:
    """Return whether a name is a version file itself rather than a sidecar or anything else."""
    match = _VERSION_FILE.fullmatch(name)
    return match is not None and match.group(2) is None


def _read_pointer(folder: Path) -> _Pointer | None:
    """Read which version is current, or None when nothing was published in this layout yet.

    :raises GraphStoreCorrupt: If the pointer is unreadable or names a version that is not there.
    """
    pointer_path = folder / GRAPH_POINTER_NAME
    try:
        names = pointer_path.read_text(encoding="ascii").split()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise GraphStoreCorrupt(f"graph pointer {pointer_path} is unreadable: {exc}") from exc
    if not 1 <= len(names) <= 2 or not all(_is_version_name(name) for name in names):
        raise GraphStoreCorrupt(f"graph pointer {pointer_path} names no graph version: {names!r}")
    if not (folder / names[0]).is_file():
        raise GraphStoreCorrupt(f"graph pointer {pointer_path} names {names[0]}, which is missing")
    return _Pointer(names[0], names[1] if len(names) == 2 else None)


def _readable_pointer(folder: Path) -> _Pointer | None:
    """Resolve the current version for a reader, migrating a legacy store under the writer lock first."""
    pointer = _read_pointer(folder)
    if pointer is not None or not (folder / LEGACY_GRAPH_DB_NAME).is_file():
        return pointer
    with _writer_lock(folder, wait=True):
        return _locked_pointer(folder)


def _locked_pointer(folder: Path) -> _Pointer | None:
    """Resolve the current version for the writer-lock holder, migrating a legacy store first."""
    pointer = _read_pointer(folder)
    if pointer is None and (folder / LEGACY_GRAPH_DB_NAME).is_file():
        pointer = _migrate_legacy(folder)
    return pointer


@contextmanager
def _writer_lock(folder: Path, *, wait: bool) -> Iterator[bool]:
    """Hold a workspace's single-writer lock for the block, yielding whether it was taken.

    AIDEV-NOTE: an flock on `graph.lock`, never a PID file: the kernel drops it when its holder
    dies, however it dies, so a killed build leaves nothing to clean up. It belongs to the open
    file description, so two threads of one process exclude each other as well. With
    `wait=False` a held lock yields False at once instead of blocking.
    """
    descriptor = os.open(folder / GRAPH_LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            taken = True
        except BlockingIOError:
            taken = False
        if not taken and wait:
            logger.info("graph: waiting for the process writing %s to finish", folder)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            taken = True
        try:
            yield taken
        finally:
            if taken:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _next_version(folder: Path) -> str:
    """Name a version above every one on disk, so a new build never reuses a published name."""
    numbers = [int(match.group(1)) for entry in os.listdir(folder) if (match := _VERSION_FILE.fullmatch(entry))]
    return _version_name(max(numbers, default=0) + 1)


def _discard(db_path: Path) -> None:
    """Delete a version file and every sidecar it may have, the readers lock last."""
    db_path.unlink(missing_ok=True)
    for suffix in _VERSION_SIDECARS:
        db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)


def _seal(db_path: Path) -> None:
    """Give a finished copy the store's journal mode and schema before any reader can resolve it."""
    open_db(db_path).close()


def _publish(folder: Path, pointer: _Pointer) -> None:
    """Point readers at a version in one atomic step: a fsynced temp file renamed over the pointer."""
    temp = folder / f"{GRAPH_POINTER_NAME}.tmp"
    with temp.open("w", encoding="ascii") as handle:
        handle.write("".join(f"{name}\n" for name in (pointer.current, pointer.previous) if name))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, folder / GRAPH_POINTER_NAME)
    directory = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _install(folder: Path, built: str, replaced: str | None) -> None:
    """Make a finished version current, then retire every other version no reader holds.

    The pointer still names the replaced version while it stays on disk, so a process running
    older code, which keeps exactly the version the pointer names second, never sweeps it.
    """
    _sweep(folder, _Pointer(built, replaced), publish=True)


def _sweep(folder: Path, pointer: _Pointer, *, publish: bool = False) -> _Pointer:
    """Retire every version but the current one that no reader holds, and a legacy store nobody has open.

    Only the writer-lock holder sweeps, so a version the pointer does not name is either retired
    or a killed build's leftover that was never published. The pointer is rewritten when it
    named a previous version that is gone now, so it never names a file that is not there.

    :param publish: Publish the pointer first even when nothing was retired.
    :return: The pointer as it now stands on disk.
    """
    if publish:
        _publish(folder, pointer)
    for name in sorted(_retired_versions(folder, pointer.current)):
        _retire(folder, name)
    _drop_legacy_if_unheld(folder)
    if pointer.previous is not None and not (folder / pointer.previous).is_file():
        pointer = _Pointer(pointer.current)
        _publish(folder, pointer)
    return pointer


def _retired_versions(folder: Path, current: str) -> dict[str, list[str]]:
    """Group every version file and sidecar on disk that is not the current version's, by version."""
    retired: dict[str, list[str]] = {}
    for entry in sorted(os.listdir(folder)):
        match = _VERSION_FILE.fullmatch(entry)
        if match is None:
            continue
        name = _version_name(int(match.group(1)))
        if name != current:
            retired.setdefault(name, []).append(entry)
    return retired


@contextmanager
def _exclusive_hold(folder: Path, name: str, *, create: bool = True) -> Iterator[bool]:
    """Hold a version's readers lock exclusively for the block, yielding False when a reader has it.

    :param create: Create a missing lock file to lock it. A writer about to delete must, since a
        reader creating it at the same moment would otherwise hold a lock nobody checked; a probe
        that deletes nothing passes False and counts a missing lock file as unheld.
    """
    lock_path = folder / f"{name}{READERS_SUFFIX}"
    try:
        descriptor = os.open(lock_path, os.O_RDWR | (os.O_CREAT if create else 0), 0o644)
    except FileNotFoundError:
        yield True
        return
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True
    finally:
        os.close(descriptor)


def _retire(folder: Path, name: str) -> bool:
    """Delete one retired version and its sidecars unless a reader holds it.

    :return: Whether it was deleted.
    """
    with _exclusive_hold(folder, name) as free:
        if not free:
            logger.info("graph: keeping %s in %s, a reader still holds it", name, folder)
            return False
        _discard(folder / name)
    return True


def retired_graph_files(folder: Path) -> list[Path]:
    """List the files a sweep of this graph folder would delete now, deleting nothing.

    A held version is probed with the same non-blocking exclusive lock the sweep takes, which is
    released at once; a legacy store is listed only once the versioned layout exists.
    """
    pointer = _read_pointer(folder)
    if pointer is None:
        return []
    files: list[Path] = []
    for name, entries in sorted(_retired_versions(folder, pointer.current).items()):
        with _exclusive_hold(folder, name, create=False) as free:
            if free:
                files.extend(folder / entry for entry in entries)
    if _legacy_unheld(folder):
        files.extend(_legacy_files(folder))
    return files


def sweep_graph_folder(folder: Path) -> list[Path] | None:
    """Delete what `retired_graph_files` lists, as the writer, or None when another process is writing.

    :return: The files that were there before the sweep and are gone after it.
    """
    with _writer_lock(folder, wait=False) as taken:
        if not taken:
            return None
        before = retired_graph_files(folder)
        pointer = _read_pointer(folder)
        if pointer is not None:
            _sweep(folder, pointer)
        return [path for path in before if not path.exists()]


def graph_files(folder: Path) -> list[Path]:
    """Return every file a graph folder's graph consists of: versions, sidecars, pointer, lock, legacy store."""
    names = [entry for entry in sorted(os.listdir(folder)) if _VERSION_FILE.fullmatch(entry)] if folder.is_dir() else []
    files = [folder / name for name in names]
    files += [path for path in (folder / GRAPH_POINTER_NAME, folder / GRAPH_LOCK_NAME) if path.exists()]
    return files + _legacy_files(folder)


def remove_graph(folder: Path) -> bool:
    """Delete a whole stored graph, as its writer, unless a writer or any reader has part of it.

    Every version is held exclusively before anything is deleted, so a reader is never left on a
    half-removed graph; the pointer goes first, so a new reader finds no graph rather than a
    missing version.

    :return: Whether the graph was removed.
    """
    with _writer_lock(folder, wait=False) as taken:
        if not taken or (_legacy_files(folder) and not _legacy_unheld(folder)):
            return False
        names = sorted(
            {
                _version_name(int(match.group(1)))
                for entry in os.listdir(folder)
                if (match := _VERSION_FILE.fullmatch(entry))
            }
        )
        with ExitStack() as holds:
            if not all(holds.enter_context(_exclusive_hold(folder, name)) for name in names):
                return False
            (folder / GRAPH_POINTER_NAME).unlink(missing_ok=True)
            for name in names:
                _discard(folder / name)
            for path in _legacy_files(folder):
                path.unlink(missing_ok=True)
        (folder / GRAPH_LOCK_NAME).unlink(missing_ok=True)
    return True


def _legacy_files(folder: Path) -> list[Path]:
    """Return the legacy store and whichever of its sidecars exist."""
    legacy = folder / LEGACY_GRAPH_DB_NAME
    candidates = [legacy, *(legacy.with_name(legacy.name + suffix) for suffix in _DB_SIDECARS)]
    return [path for path in candidates if path.exists()]


def _legacy_unheld(folder: Path) -> bool:
    """Return whether a legacy store exists that no process has open, sidecars included."""
    files = _legacy_files(folder)
    if not files or files[0].name != LEGACY_GRAPH_DB_NAME:
        return False
    return not any(held_open(path) for path in files)


def _drop_legacy_if_unheld(folder: Path) -> None:
    """Delete the legacy store a copy-migration had to leave behind, once nobody has it open.

    Called only by the writer, after the versioned layout exists.
    """
    if not _legacy_unheld(folder):
        return
    for path in _legacy_files(folder):
        path.unlink(missing_ok=True)
    logger.info("graph: deleted the superseded single-file store in %s", folder)


def _migrate_legacy(folder: Path) -> _Pointer:
    """Move the single-file store an older zemble left into the versioned layout, without a rebuild.

    AIDEV-NOTE: a rename when no process has `graph.sqlite` (or a journal of it) open: nothing
    can then be using it, and sealing turns it into the first WAL version in place. A process
    still running the older code opens `graph.sqlite` by name in rollback-journal mode, so while
    one holds it the file is copied through sqlite's backup API instead - turning a held inode
    into a WAL database under a second name would give one database two journals - and the
    original is deleted later by the first writer that finds it unheld (`_drop_legacy_if_unheld`).
    """
    name = _next_version(folder)
    target = folder / name
    _discard(target)
    legacy = folder / LEGACY_GRAPH_DB_NAME
    if _legacy_unheld(folder) and _legacy_files(folder) == [legacy]:
        os.rename(legacy, target)
        try:
            _seal(target)
        except BaseException:
            os.rename(target, legacy)
            raise
        verb = "moved"
    else:
        try:
            source = sqlite3.connect(legacy, timeout=_WRITE_TIMEOUT_SECONDS)
            try:
                copy = sqlite3.connect(target)
                try:
                    source.backup(copy)
                finally:
                    copy.close()
            finally:
                source.close()
            _seal(target)
        except BaseException:
            _discard(target)
            raise
        verb = "copied"
    pointer = _Pointer(name)
    _publish(folder, pointer)
    logger.info("graph: %s the single-file store in %s into version %s", verb, folder, name)
    return pointer


@dataclass
class CompactedGraph:
    """What `compact_stored_graphs` did to one graph folder."""

    folder: Path
    size_before: int
    size_after: int
    #: Why nothing was done, when nothing was.
    skipped: str | None = None


def _version_size(folder: Path, name: str) -> int:
    """Return the bytes one version file and its WAL take."""
    return sum(path.stat().st_size for path in (folder / name, folder / f"{name}-wal") if path.exists())


def compact_stored_graphs(cache_folder: Path) -> list[CompactedGraph]:
    """Bring every stored graph to the current format and give its freed pages back to the filesystem.

    A graph is only migrated when something opens it for writing, which for a checkout nobody
    edits is never; this does it for all of them, each under its writer lock. A graph another
    process is writing is skipped, and a replaced version a reader still holds stays on disk
    until the next writer finds it released.

    :param cache_folder: The zemble cache folder.
    :return: One report per graph folder, in name order.
    """
    reports: list[CompactedGraph] = []
    for folder in sorted(cache_folder.glob("*/index")):
        if not ((folder / GRAPH_POINTER_NAME).is_file() or (folder / LEGACY_GRAPH_DB_NAME).is_file()):
            continue
        with _writer_lock(folder, wait=False) as taken:
            if not taken:
                reports.append(CompactedGraph(folder, 0, 0, "another process is writing it"))
                continue
            try:
                pointer = _locked_pointer(folder)
            except GraphStoreCorrupt as exc:
                reports.append(CompactedGraph(folder, 0, 0, str(exc)))
                continue
            if pointer is None:
                continue
            before = _version_size(folder, pointer.current)
            open_db(folder / pointer.current).close()
            _compact_if_drifted(folder, pointer.current, _EXPLICIT_COMPACT_FREE_FRACTION)
            current = _sweep(folder, _read_pointer(folder) or pointer).current
            reports.append(CompactedGraph(folder, before, _version_size(folder, current)))
    return reports


def _migrate(connection: sqlite3.Connection) -> None:
    """Add the columns a graph built by an older zemble does not have yet."""
    edge_columns = {row["name"] for row in connection.execute("PRAGMA table_info(edges)")}
    if "source" not in edge_columns:
        connection.execute("ALTER TABLE edges ADD COLUMN source TEXT")
    if "origin_ref" not in edge_columns:
        connection.execute("ALTER TABLE edges ADD COLUMN origin_ref TEXT")
    if "candidates" in edge_columns:
        _count_candidates(connection, edge_columns)
    _normalize_edges(connection)
    status_columns = {row["name"] for row in connection.execute("PRAGMA table_info(facts_status)")}
    if "template_paths" not in status_columns:
        connection.execute("ALTER TABLE facts_status ADD COLUMN template_paths TEXT")
    for column, kind in _FACTS_STATUS_ADDITIONS:
        if column not in status_columns:
            connection.execute(f"ALTER TABLE facts_status ADD COLUMN {column} {kind}")  # noqa: S608
    symbol_columns = {row["name"] for row in connection.execute("PRAGMA table_info(symbols)")}
    if "annotation_args" not in symbol_columns:
        connection.execute("ALTER TABLE symbols ADD COLUMN annotation_args TEXT")
    _backfill_declaration_keys(connection)


def _count_candidates(connection: sqlite3.Connection, edge_columns: set[str]) -> None:
    """Replace a format-6 store's candidate lists by their counts, in one transaction.

    The lists were most of the file (1.6 GB of 2.6 GB on javaweb). Dropping the column rewrites
    the table, which leaves the old pages free; the build that opened the store then compacts
    them away (`_compact_if_drifted`), so the disk is given back by the same refresh.
    """
    logger.info("graph: replacing stored candidate lists by their counts (format %d)", GRAPH_FORMAT_VERSION)
    if "candidate_count" not in edge_columns:
        connection.execute("ALTER TABLE edges ADD COLUMN candidate_count INTEGER")
    connection.execute(
        "UPDATE edges SET candidate_count = COALESCE(json_array_length(candidates), 0) WHERE candidate_count IS NULL"
    )
    connection.execute("ALTER TABLE edges DROP COLUMN candidates")
    connection.commit()


#: Spell a format-7 row's file the way `_edge_row` does: the part of its source id before `#`.
_LEGACY_FILE = "COALESCE(file_path, substr(src_id, 1, instr(src_id || '#', '#') - 1))"


def _normalize_edges(connection: sqlite3.Connection) -> None:
    """Turn a format-7 `edges` table into `edge_rows` over interned `refs`, behind the `edges` view.

    Each edge spelled its source and destination symbol ids and its file as text (~400 bytes
    with them), and three indexes copied those strings again: 2.1 GB of a zenit workspace graph,
    0.6 GB once every id is stored once in `refs` and edges hold its key. Readers keep their SQL:
    `edges` is now a view with the old columns, and the plan reaches `edge_rows` through its
    integer indexes. The old table's pages are freed; the build that opened the store compacts
    them away (`_compact_if_drifted`), as `zemble graph compact` does for a store nobody writes.
    """
    kind = connection.execute("SELECT type FROM sqlite_master WHERE name = 'edges'").fetchone()
    if kind is None or kind[0] != "table":
        return
    logger.info("graph: storing edge endpoints as interned keys (format %d)", GRAPH_FORMAT_VERSION)
    connection.execute("INSERT OR IGNORE INTO refs (text) SELECT src_id FROM edges")
    connection.execute("INSERT OR IGNORE INTO refs (text) SELECT dst_id FROM edges WHERE dst_id IS NOT NULL")
    connection.execute(f"INSERT OR IGNORE INTO refs (text) SELECT {_LEGACY_FILE} FROM edges")  # noqa: S608
    connection.execute(
        f"INSERT INTO edge_rows SELECT s.key, d.key, e.dst_name, e.kind, e.line, e.resolution, "  # noqa: S608
        f"e.candidate_count, e.arity, e.receiver, e.receiver_type, e.is_new, f.key, e.source, e.origin_ref "
        f"FROM (SELECT rowid AS r, *, {_LEGACY_FILE} AS file FROM edges) e "
        f"JOIN refs s ON s.text = e.src_id LEFT JOIN refs d ON d.text = e.dst_id JOIN refs f ON f.text = e.file "
        f"ORDER BY e.r"
    )
    connection.execute("DROP TABLE edges")
    connection.executescript(_SCHEMA)
    connection.commit()


def insert_edges(connection: sqlite3.Connection, edges: Iterable[Edge]) -> None:
    """Store edges, interning their endpoints; the one way an edge reaches `edge_rows` outside a build."""
    with _scratch(connection):
        _stage(connection, edges, table="resolved")
        _publish_resolved(connection)


def _publish_resolved(connection: sqlite3.Connection) -> None:
    """Copy a build's resolved edges out of the scratch database into `edge_rows`, interning their endpoints."""
    for column in ("src_id", "dst_id", "file_path"):
        connection.execute(
            f"INSERT OR IGNORE INTO main.refs (text) SELECT {column} FROM scratch.resolved "  # noqa: S608
            f"WHERE {column} IS NOT NULL"
        )
    connection.execute(
        "INSERT INTO main.edge_rows SELECT s.key, d.key, r.dst_name, r.kind, r.line, r.resolution, "
        "r.candidate_count, r.arity, r.receiver, r.receiver_type, r.is_new, f.key, r.source, r.origin_ref "
        "FROM scratch.resolved r JOIN main.refs s ON s.text = r.src_id LEFT JOIN main.refs d ON d.text = r.dst_id "
        "JOIN main.refs f ON f.text = r.file_path ORDER BY r.rowid"
    )


def _prune_refs(connection: sqlite3.Connection) -> None:
    """Forget interned ids no edge uses any more: an incremental build only ever adds to `refs`."""
    connection.execute(
        "DELETE FROM refs WHERE key NOT IN (SELECT src FROM edge_rows) AND key NOT IN "
        "(SELECT dst FROM edge_rows WHERE dst IS NOT NULL) AND key NOT IN (SELECT file FROM edge_rows)"
    )
    connection.commit()


#: Meta key saying the `decl_keys` table has been filled for this graph.
_DECL_KEYS_META = "decl_keys_built"

#: Columns `facts_status` grew after format version 4, so an older graph is migrated rather
#: than rebuilt. A row written by the older zemble simply reports zero for them until the
#: facts file it describes is read again.
_FACTS_STATUS_ADDITIONS = (
    ("fresh_paths", "TEXT"),
    ("contributions", "TEXT"),
    ("parse_buckets", "TEXT"),
    ("error", "TEXT"),
    ("generated_templates", "TEXT"),
)


def _backfill_declaration_keys(connection: sqlite3.Connection) -> None:
    """Fill `decl_keys` from the symbol table when a graph predates it.

    The table is derived data, so an older graph is migrated in one pass over its symbols
    rather than rebuilt. A meta flag says the pass has run, because "the table is empty" is
    also the honest state of a workspace that declares no Hawkeye registration at all.
    """
    done = connection.execute("SELECT value FROM meta WHERE key = ?", (_DECL_KEYS_META,)).fetchone()
    if done is not None:
        return
    symbols = (symbol_from_row(row) for row in connection.execute("SELECT * FROM symbols"))
    connection.executemany(
        "INSERT INTO decl_keys (symbol_id, key, value, file_path) VALUES (?,?,?,?)",
        _declaration_rows(symbols),
    )
    connection.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (_DECL_KEYS_META, "1"))
    connection.commit()


def graph_present(path: str) -> bool:
    """Return True if a graph exists for a path in either layout, readable or not."""
    folder = graph_folder(path)
    return (folder / GRAPH_POINTER_NAME).is_file() or (folder / LEGACY_GRAPH_DB_NAME).is_file()


def graph_exists(path: str) -> bool:
    """Return True if a graph database with symbols already exists for a path."""
    folder = graph_folder(path)
    connection = None
    try:
        connection = _connect_current(folder)
        if connection is None:
            return False
        return connection.execute("SELECT 1 FROM symbols LIMIT 1").fetchone() is not None
    except (sqlite3.Error, GraphStoreCorrupt) as exc:
        if is_corruption(exc):
            # Loud, and false: a malformed store is not a graph, so the next build makes one.
            logger.error("graph store in %s is malformed (%s); it will be rebuilt from source", folder, exc)
        return False
    finally:
        if connection is not None:
            connection.close()


def graph_root_of(folder: Path) -> str | None:
    """Return the workspace root a graph folder was built from, migrating and writing nothing.

    :return: The root its metadata names, or None when no readable graph is there.
    """
    try:
        pointer = _read_pointer(folder)
    except GraphStoreCorrupt:
        return None
    hold = None
    if pointer is not None:
        hold = _hold_version(folder, pointer.current)
        db_path = folder / pointer.current
    else:
        db_path = folder / LEGACY_GRAPH_DB_NAME
    try:
        if not db_path.is_file():
            return None
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=_READ_TIMEOUT_SECONDS)
        try:
            row = connection.execute("SELECT value FROM meta WHERE key = 'root'").fetchone()
        finally:
            connection.close()
    except sqlite3.Error:
        return None
    finally:
        if hold is not None:
            os.close(hold)
    return str(row[0]) if row is not None and row[0] else None


def graph_ancestor(path: str) -> tuple[str, str] | None:
    """Return the nearest ancestor with a graph and the path's prefix under it, when the path has no graph of its own.

    The precedence is search's (`zemble.cache.resolve_index_root`): a graph of exactly this
    path wins, then the nearest ancestor's, else the path needs one of its own.

    :return: The ancestor root and the root-relative POSIX prefix, or None.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir() or graph_present(str(resolved)):
        return None
    for ancestor in resolved.parents:
        if graph_present(str(ancestor)):
            return str(ancestor), resolved.relative_to(ancestor).as_posix()
    return None


def graph_covers(root: str, prefix: str) -> bool:
    """Return whether a root's current graph holds any file under a root-relative prefix.

    A sub-tree the ancestor's walk ignores (a nested repository it excludes, a build folder)
    is absent from its graph, and is then answered by a graph of its own.
    """
    try:
        connection = connect(root)
    except GraphStoreMissing:
        return False
    try:
        # A key range on the primary key, not LIKE: `_` and `%` are ordinary characters in a path.
        row = connection.execute(
            "SELECT 1 FROM files WHERE path >= ? AND path < ? LIMIT 1", (f"{prefix}/", f"{prefix}0")
        ).fetchone()
        return row is not None
    finally:
        connection.close()


def resolve_graph_root(path: str) -> tuple[str, str | None]:
    """Route a graph request for a path to the graph that answers it.

    :return: The root whose graph to open, and the root-relative prefix to restrict it to (None = all of it).
    """
    ancestor = graph_ancestor(path)
    if ancestor is not None and graph_covers(*ancestor):
        return ancestor
    return path, None


# ---- serialisation ------------------------------------------------------


def _symbol_row(symbol: Symbol) -> tuple:
    """Flatten a symbol into a database row."""
    return (
        symbol.id,
        symbol.kind.value,
        symbol.name,
        symbol.qualified_name,
        symbol.file_path,
        symbol.start_line,
        symbol.end_line,
        symbol.container_id,
        json.dumps(symbol.modifiers),
        json.dumps(symbol.annotations),
        symbol.signature,
        int(symbol.is_test),
        json.dumps(symbol.param_types),
        json.dumps(symbol.annotation_args),
    )


#: The edge columns `_edge_row` produces, in order. Named in the INSERT so a column added
#: later is a compile-time-obvious edit here rather than a positional mismatch at runtime.
_EDGE_COLUMNS = (
    "src_id",
    "dst_id",
    "dst_name",
    "kind",
    "line",
    "resolution",
    "candidate_count",
    "arity",
    "receiver",
    "receiver_type",
    "is_new",
    "file_path",
    "source",
    "origin_ref",
)
_EDGE_COLUMNS_SQL = ", ".join(_EDGE_COLUMNS)
_EDGE_PLACEHOLDERS = ",".join("?" * len(_EDGE_COLUMNS))


def _edge_row(edge: Edge) -> tuple:
    """Flatten an edge into a database row."""
    return (
        edge.src_id,
        edge.dst_id,
        edge.dst_name,
        edge.kind.value,
        edge.line,
        edge.resolution.value,
        edge.ambiguity(),
        edge.arity,
        edge.receiver,
        edge.receiver_type,
        int(edge.is_new),
        edge.src_id.split("#", 1)[0],
        edge.source,
        edge.origin_ref,
    )


def edge_from_row(row: sqlite3.Row) -> Edge:
    """Rebuild an edge from a database row; a format-6 row still carrying its candidate list is counted."""
    if "candidate_count" in row.keys():
        count = row["candidate_count"] or 0
    else:
        count = len(json.loads(row["candidates"])) if row["candidates"] else 0
    return Edge(
        src_id=row["src_id"],
        dst_name=row["dst_name"],
        kind=EdgeKind(row["kind"]),
        line=row["line"],
        dst_id=row["dst_id"],
        resolution=Resolution(row["resolution"]),
        candidate_count=count,
        arity=row["arity"],
        receiver=row["receiver"],
        receiver_type=row["receiver_type"],
        is_new=bool(row["is_new"]),
        source=row["source"] or TREE_SITTER_SOURCE,
        origin_ref=row["origin_ref"],
    )


# ---- extraction ---------------------------------------------------------


def _extract_one(job: tuple[str, str]) -> FileExtraction | None:
    """Extract one file in a worker process, returning None when it cannot be read."""
    absolute, relative = job
    extract = extractor_for(Path(absolute))
    if extract is None:
        return None
    try:
        source = Path(absolute).read_bytes()
    except OSError:
        return None
    try:
        return extract(source, relative)
    except Exception:
        logger.warning("Failed to extract %s", relative, exc_info=True)
        return None


def _extract_serial(jobs: Sequence[tuple[str, str]]) -> list[FileExtraction]:
    """Extract a batch of files in this process."""
    return [result for job in jobs for result in [_extract_one(job)] if result is not None]


def _extract_many(jobs: Sequence[tuple[str, str]], workers: int) -> list[FileExtraction]:
    """Extract a handful of files at once; a build extracts through :func:`_extracted_batches`."""
    return [extraction for batch in _extracted_batches(jobs, workers) for extraction in batch]


def _extracted_batches(jobs: Sequence[tuple[str, str]], workers: int) -> Iterator[list[FileExtraction]]:
    """Extract files :data:`_BATCH_FILES` at a time, using one process pool for all of them.

    The start method comes from `zemble.parallel.pool_context` (fork only in a
    single-threaded process, else spawn, else none); when no method is safe, or the pool
    fails for any reason, extraction runs in this process rather than aborting.
    """
    batches = [jobs[start : start + _BATCH_FILES] for start in range(0, len(jobs), _BATCH_FILES)]
    context = pool_context() if len(jobs) >= _WORKER_CHUNK * 2 and workers > 1 else None
    if context is None:
        for batch in batches:
            yield _extract_serial(batch)
        return
    done = 0
    try:
        with pooled(workers, context) as pool:
            for batch in batches:
                extracted = [result for result in pool.map(_extract_one, batch, chunksize=_WORKER_CHUNK) if result]
                done += 1
                yield extracted
    except Exception:
        logger.warning("Parallel extraction unavailable; falling back to a single process", exc_info=True)
        for batch in batches[done:]:
            yield _extract_serial(batch)


@dataclass
class _Scan:
    """The result of walking the workspace once."""

    jobs: list[tuple[str, str]] = field(default_factory=list)
    stamps: dict[str, tuple[int, int]] = field(default_factory=dict)
    skipped: Counter = field(default_factory=Counter)
    scanned: int = 0


def _scan(root: Path) -> _Scan:
    """Walk the workspace, splitting extractable files from everything the graph has no reader for."""
    scan = _Scan()
    for file_path in walk_files(root, extensions=get_extensions((ContentType.CODE,))):
        scan.scanned += 1
        suffix = file_path.suffix.lower()
        if extractor_for(file_path) is None:
            scan.skipped[detect_language(file_path) or suffix] += 1
            continue
        try:
            stat = file_path.stat()
        except OSError:
            continue
        if stat.st_size > _MAX_FILE_BYTES:
            scan.skipped[f"{suffix} (too large)"] += 1
            continue
        relative = file_path.relative_to(root).as_posix()
        scan.jobs.append((str(file_path), relative))
        scan.stamps[relative] = (stat.st_mtime_ns, stat.st_size)
    return scan


def _scan_changed(root: Path, changed: Iterable[Path], stored: dict[str, tuple[int, int]]) -> _Scan:
    """Build the scan a walk would have produced, from a change set plus the stored file stamps.

    Every file the graph already holds keeps the stamp it was extracted with, so the build
    sees it as unchanged without stat-ing it; only the named paths are looked at. A named
    path that is gone stays out of the stamps, which is what makes the build delete it.
    """
    scan = _Scan()
    named: dict[str, Path] = {}
    for candidate in changed:
        try:
            relative = candidate.relative_to(root).as_posix()
        except ValueError:
            continue
        if extractor_for(candidate) is not None and ignored_prefix(root, relative) is None:
            named[relative] = candidate

    for relative, stamp in stored.items():
        if relative in named:
            continue
        scan.jobs.append((str(root / relative), relative))
        scan.stamps[relative] = stamp

    for relative, candidate in named.items():
        try:
            stat = candidate.stat()
        except OSError:
            continue
        if stat.st_size > _MAX_FILE_BYTES:
            scan.skipped[f"{candidate.suffix.lower()} (too large)"] += 1
            continue
        scan.jobs.append((str(candidate), relative))
        scan.stamps[relative] = (stat.st_mtime_ns, stat.st_size)
    scan.scanned = len(scan.stamps)
    return scan


def _stored_stamps(connection: sqlite3.Connection) -> dict[str, tuple[int, int]]:
    """Return the modification stamp of every file the graph currently holds."""
    return {row["path"]: (row["mtime_ns"], row["size"]) for row in connection.execute("SELECT * FROM files")}


# ---- build ---------------------------------------------------------------


def build_graph(
    path: str,
    *,
    force: bool = False,
    workers: int | None = None,
    changed_paths: Iterable[Path] | None = None,
) -> GraphStats:
    """Build or incrementally refresh the symbol graph for a workspace, after any other writer finishes.

    :param path: Local directory to index.
    :param force: Re-extract every file instead of only changed ones.
    :param workers: Extraction process count; defaults to the CPU count.
    :param changed_paths: The exact paths that moved, from a watcher; None walks the tree.
        The caller must name every path that moved, since nothing else is looked at.
    :return: Statistics describing the build.
    :raises ValueError: If the path is not a local directory.
    """
    root = _local_root(path)
    folder = _graph_dir(str(root))
    with _writer_lock(folder, wait=True):
        return _build_locked(root, folder, force=force, workers=workers, changed_paths=changed_paths)


def refresh_graph(path: str) -> GraphStats | None:
    """Incrementally refresh the symbol graph for a workspace unless another process is writing it.

    :param path: Local directory to index.
    :return: Statistics describing the refresh, or None when another writer held the lock; the
        current version stays readable meanwhile.
    :raises ValueError: If the path is not a local directory.
    """
    root = _local_root(path)
    folder = _graph_dir(str(root))
    with _writer_lock(folder, wait=False) as taken:
        if not taken:
            logger.info("graph: another process is writing the graph of %s; reading the current one", root)
            return None
        return _build_locked(root, folder, force=False, workers=None, changed_paths=None)


def _local_root(path: str) -> Path:
    """Resolve a workspace path, refusing anything that is not a local directory."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"{path!r} is not a local directory")
    return root


def _build_locked(
    root: Path,
    folder: Path,
    *,
    force: bool,
    workers: int | None,
    changed_paths: Iterable[Path] | None,
) -> GraphStats:
    """Run one build for the writer-lock holder, rebuilding from source when the store is unreadable."""
    started = time.perf_counter()
    workers = workers if workers is not None else DEFAULT_WORKERS
    try:
        # A forced build rebuilds from source, so a legacy store is not worth copying first.
        pointer = _read_pointer(folder) if force else _locked_pointer(folder)
        current = None
        if pointer is not None:
            # Whatever a killed build left is swept now, not only when the next version lands.
            _sweep(folder, pointer)
            current = pointer.current
        stats = _build_into(root, folder, current, force=force, workers=workers, changed_paths=changed_paths)
    except (sqlite3.DatabaseError, GraphStoreCorrupt) as exc:
        if not is_corruption(exc):
            raise
        logger.error("graph store in %s is malformed (%s); rebuilding it from source", folder, exc)
        stats = _build_into(root, folder, None, force=True, workers=workers, changed_paths=None)
        stats.rebuilt_from_corruption = True
    stats.duration_seconds = time.perf_counter() - started
    return stats


def _build_into(
    root: Path,
    folder: Path,
    current: str | None,
    *,
    force: bool,
    workers: int,
    changed_paths: Iterable[Path] | None,
) -> GraphStats:
    """Run one build, into a new version when it rewrites the store, in the current one otherwise.

    A build that reads the whole workspace anyway writes a brand-new version and publishes it
    with one pointer rename, so a kill leaves the current version untouched instead of a torn
    one - and the version is compact by construction. An incremental refresh writes the current
    version in place, because copying the store per saved file would cost more than the
    refresh; its WAL makes that safe against a kill and invisible to readers until it commits.
    A None `current` also drops the version it replaced, which is how a corrupt one goes.
    """
    name = current if current is not None and not force else _next_version(folder)
    whole = name != current
    target = folder / name
    if whole:
        _discard(target)
    connection = open_db(target)
    # A forced build re-reads everything, facts files included, so it walks like a cold one.
    named = None if changed_paths is None or force else list(changed_paths)
    scan = _scan(root) if named is None else _scan_changed(root, named, _stored_stamps(connection))
    stats = GraphStats(root=str(root), files_scanned=scan.scanned, skipped_by_language=dict(scan.skipped))
    for language, count in sorted(scan.skipped.items(), key=lambda item: -item[1]):
        logger.info("graph: skipping %d %s file(s): no graph extractor for %s", count, language, language)

    try:
        try:
            _run_build(connection, root, scan, stats, force=force, workers=workers, named_changes=named)
        finally:
            connection.commit()
            connection.close()
    except BaseException:
        if whole:
            _discard(target)
        raise
    if whole:
        _install(folder, name, current)
    else:
        stats.compacted = _compact_if_drifted(folder, name)
    return stats


def _compact_if_drifted(folder: Path, current: str, free_fraction: float = _COMPACT_FREE_FRACTION) -> bool:
    """Rewrite a version whose deleted rows have left too much of it free, as a new version.

    Sqlite never returns freed pages to the filesystem, and an incremental refresh is mostly
    deletes: the store only ever grows unless something compacts it. `VACUUM INTO` writes the
    compact copy as the next version, published the way a full build is, rather than rewriting
    the current one under its readers. The caller holds the writer lock.
    """
    connection = sqlite3.connect(folder / current, timeout=_WRITE_TIMEOUT_SECONDS)
    try:
        pages = connection.execute("PRAGMA page_count").fetchone()[0]
        free = connection.execute("PRAGMA freelist_count").fetchone()[0]
        if pages < _COMPACT_MIN_PAGES or free < pages * free_fraction:
            return False
        if connection.execute("SELECT 1 FROM sqlite_master WHERE name = 'edge_rows'").fetchone():
            _prune_refs(connection)
        name = _next_version(folder)
        target = folder / name
        _discard(target)
        logger.info("graph: compacting %s (%d of %d pages free)", folder / current, free, pages)
        try:
            connection.execute("VACUUM INTO ?", (str(target),))
        except BaseException:
            _discard(target)
            raise
    finally:
        connection.close()
    _seal(target)
    _install(folder, name, current)
    return True


def _run_build(
    connection: sqlite3.Connection,
    root: Path,
    scan: _Scan,
    stats: GraphStats,
    *,
    force: bool,
    workers: int,
    named_changes: list[Path] | None,
) -> None:
    """Do the two-pass build inside an open connection, a batch of files at a time."""
    known = _stored_stamps(connection)
    changed = [job for job in scan.jobs if force or known.get(job[1]) != scan.stamps[job[1]]]
    removed = sorted(set(known) - set(scan.stamps))
    stats.extracted_files = len(changed)
    stats.unchanged_files = len(scan.jobs) - len(changed)
    stats.removed_files = len(removed)

    touched = sorted({job[1] for job in changed} | set(removed))
    # When every file is re-resolved anyway, which names moved cannot add a single target.
    everything = set(scan.stamps) <= set(touched)
    before = {} if everything else _declaration_index(connection, touched)

    with _scratch(connection):
        _delete_files(connection, touched)
        after: dict[str, set[str]] = {}
        for batch in _extracted_batches(changed, workers):
            _insert_extractions(connection, batch, scan.stamps)
            _stage(connection, (edge for extraction in batch for edge in extraction.edges))
            if not everything:
                for name, ids in _index_extractions(batch).items():
                    after.setdefault(name, set()).update(ids)

        targets = (
            set(touched) if everything else set(touched) | _dependent_files(connection, _moved_names(before, after))
        )
        targets &= set(scan.stamps)
        _resolve_pass(connection, {job[1] for job in changed}, targets, root, stats, named_changes, workers)
    _write_meta(connection, root)
    _write_coverage(connection, stats.skipped_by_language)
    stats.symbols = connection.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
    stats.edges = connection.execute("SELECT COUNT(*) FROM edge_rows").fetchone()[0]
    stats.resolution_counts = {
        row["resolution"]: row["n"]
        for row in connection.execute("SELECT resolution, COUNT(*) AS n FROM edge_rows GROUP BY resolution")
    }


@contextmanager
def _scratch(connection: sqlite3.Connection) -> Iterator[None]:
    """Attach a disposable database holding the edges a build has extracted and resolved so far.

    Resolution needs every changed file's symbols in the store before the first edge resolves,
    and the store must not hold a re-resolved file's edges while it does (`SqliteLookup` reads
    the stored hierarchy). Both halves of that wait here, on disk, instead of in memory.

    sqlite attaches and detaches only outside a transaction, so this is entered before the
    build's first write and commits the build when it leaves: a refresh stays one transaction,
    and one that fails is rolled back rather than committed half done.
    """
    folder = Path(connection.execute("PRAGMA database_list").fetchone()["file"]).parent
    path = folder / _SCRATCH_NAME.format(pid=os.getpid())
    path.unlink(missing_ok=True)
    connection.execute("ATTACH DATABASE ? AS scratch", (str(path),))
    try:
        connection.execute("PRAGMA scratch.journal_mode=OFF")
        connection.execute("PRAGMA scratch.synchronous=OFF")
        for table in ("pending", "resolved"):
            connection.execute(f"CREATE TABLE scratch.{table} AS SELECT * FROM main.edges WHERE 0")  # noqa: S608
        connection.execute("CREATE INDEX scratch.pending_file ON pending (file_path)")
        yield
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()
    finally:
        connection.execute("DETACH DATABASE scratch")
        path.unlink(missing_ok=True)


def _stage(connection: sqlite3.Connection, edges: Iterable[Edge], table: str = "pending") -> None:
    """Write edges into one of the scratch tables."""
    connection.executemany(
        f"INSERT INTO scratch.{table} ({_EDGE_COLUMNS_SQL}) VALUES ({_EDGE_PLACEHOLDERS})",  # noqa: S608
        (_edge_row(edge) for edge in edges),
    )


def _staged(connection: sqlite3.Connection, paths: Sequence[str], *, hierarchy: bool) -> list[Edge]:
    """Read back the unresolved edges of some files, the supertype ones or all the others."""
    kinds = ",".join("?" * len(HIERARCHY_KINDS))
    edges: list[Edge] = []
    for chunk in _chunks(paths):
        placeholders = ",".join("?" * len(chunk))
        query = (  # noqa: S608
            f"SELECT * FROM scratch.pending WHERE file_path IN ({placeholders}) "
            f"AND kind {'IN' if hierarchy else 'NOT IN'} ({kinds}) ORDER BY rowid"
        )
        edges.extend(_reset(edge_from_row(row)) for row in connection.execute(query, [*chunk, *HIERARCHY_KINDS]))
    return edges


def _unstage(connection: sqlite3.Connection, paths: Sequence[str]) -> None:
    """Move the stored extracted edges of files about to be re-resolved into the scratch table.

    Derived edges are left behind and deleted: they are recomputed from the resolved ones.
    """
    derived = ",".join("?" * len(_DERIVED_KINDS))
    for chunk in _chunks(paths):
        placeholders = ",".join("?" * len(chunk))
        connection.execute(
            f"INSERT INTO scratch.pending SELECT * FROM main.edges "  # noqa: S608
            f"WHERE file_path IN ({placeholders}) AND kind NOT IN ({derived})",
            [*chunk, *_DERIVED_KINDS],
        )
    _delete_edges(connection, paths)


def _write_meta(connection: sqlite3.Connection, root: Path) -> None:
    """Record the graph format version and the root it was built from."""
    connection.executemany(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        [
            ("format_version", str(GRAPH_FORMAT_VERSION)),
            ("root", str(root)),
            ("language", ",".join(GRAPH_LANGUAGES)),
            ("built_at", str(time.time())),
        ],
    )


def _write_coverage(connection: sqlite3.Connection, skipped: dict[str, int]) -> None:
    """Record which languages the build had no extractor for."""
    connection.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", ("skipped_by_language", json.dumps(skipped))
    )


def _declaration_index(connection: sqlite3.Connection, paths: Sequence[str]) -> dict[str, set[str]]:
    """Map every written name declared in the given files to the symbol ids that carry it."""
    index: dict[str, set[str]] = {}
    for chunk in _chunks(paths):
        placeholders = ",".join("?" * len(chunk))
        query = f"SELECT name, qualified_name, id FROM symbols WHERE file_path IN ({placeholders})"  # noqa: S608
        for row in connection.execute(query, chunk):
            index.setdefault(row["name"], set()).add(row["id"])
            index.setdefault(row["qualified_name"], set()).add(row["id"])
    return index


def _index_extractions(extractions: Iterable[FileExtraction]) -> dict[str, set[str]]:
    """Build the same name -> symbol id map from freshly extracted files."""
    index: dict[str, set[str]] = {}
    for extraction in extractions:
        for symbol in extraction.symbols:
            index.setdefault(symbol.name, set()).add(symbol.id)
            index.setdefault(symbol.qualified_name, set()).add(symbol.id)
    return index


def _moved_names(before: dict[str, set[str]], after: dict[str, set[str]]) -> set[str]:
    """Return the names whose declaration set changed, so their users must re-resolve.

    Re-resolving every file that merely mentions a name declared in a touched file
    is correct but hopeless in practice: a common method name like `of` drags in
    thousands of files on every save. What actually invalidates a resolution is the
    name pointing somewhere else, which is exactly a change in this map. A rename, a
    move between packages and a file rename all change it; re-saving a file does not.
    """
    return {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}


def _delete_files(connection: sqlite3.Connection, paths: Sequence[str]) -> None:
    """Drop every row belonging to the given files."""
    for chunk in _chunks(paths):
        placeholders = ",".join("?" * len(chunk))
        for table in ("symbols", "decl_keys"):
            connection.execute(f"DELETE FROM {table} WHERE file_path IN ({placeholders})", chunk)  # noqa: S608
        _delete_edges(connection, chunk)
        connection.execute(f"DELETE FROM files WHERE path IN ({placeholders})", chunk)  # noqa: S608


def _dependent_files(connection: sqlite3.Connection, names: set[str]) -> set[str]:
    """Return files holding an edge whose written destination name was declared in a changed file.

    A rename or a move changes which symbol a name points at, so every file that
    wrote that name must be resolved again even though its own text did not change.
    """
    dependents: set[str] = set()
    for chunk in _chunks(sorted(names)):
        placeholders = ",".join("?" * len(chunk))
        query = f"SELECT DISTINCT file_path FROM edges WHERE dst_name IN ({placeholders})"  # noqa: S608
        dependents.update(row["file_path"] for row in connection.execute(query, chunk))
    return dependents


def _chunks(items: Sequence[str], size: int = 400) -> Iterator[Sequence[str]]:
    """Split a sequence into chunks small enough for a sqlite IN clause."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _declaration_rows(symbols: Iterable[Symbol]) -> list[tuple[str, str, str, str]]:
    """Flatten every Hawkeye registration key a batch of symbols declares into table rows."""
    return [
        (symbol.id, key.value, value, symbol.file_path) for symbol in symbols for key, value in declaration_keys(symbol)
    ]


def _insert_extractions(
    connection: sqlite3.Connection, extractions: Iterable[FileExtraction], stamps: dict[str, tuple[int, int]]
) -> None:
    """Insert the symbols and file records of freshly extracted files."""
    file_rows = []
    symbol_rows = []
    for extraction in extractions:
        mtime_ns, size = stamps.get(extraction.file_path, (0, 0))
        file_rows.append(
            (
                extraction.file_path,
                mtime_ns,
                size,
                extraction.package,
                json.dumps(
                    {
                        "explicit": extraction.imports.explicit,
                        "wildcards": extraction.imports.wildcards,
                        "static_members": extraction.imports.static_members,
                        "static_wildcards": extraction.imports.static_wildcards,
                    }
                ),
            )
        )
        symbol_rows.extend(_symbol_row(symbol) for symbol in extraction.symbols)
    connection.executemany("INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?)", file_rows)
    connection.executemany("INSERT OR REPLACE INTO symbols VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", symbol_rows)
    connection.executemany(
        "INSERT INTO decl_keys (symbol_id, key, value, file_path) VALUES (?,?,?,?)",
        _declaration_rows(symbol for extraction in extractions for symbol in extraction.symbols),
    )


def _load_contexts(connection: sqlite3.Connection) -> dict[str, FileContext]:
    """Load every file's package and imports."""
    return {row["path"]: context_from_row(row) for row in connection.execute("SELECT * FROM files")}


def _resolve_pass(
    connection: sqlite3.Connection,
    fresh: set[str],
    targets: set[str],
    root: Path,
    stats: GraphStats,
    named_changes: Iterable[Path] | None,
    workers: int,
) -> None:
    """Run pass 2 for the target files against the whole workspace symbol table, a batch at a time.

    The facts overlay is folded in here rather than afterwards, because the derived
    edges (overrides, exercises) must be derived from the edges the graph keeps, not
    from the extracted ones a facts file just replaced.

    Nothing here reads more of the workspace than the target files need. The facts files are
    parsed only when at least one of them must be mapped, and a file the build is not
    re-resolving keeps every edge it had. The supertype edges of every target are resolved
    first, because a call chain may climb through any of them; everything else resolves and
    is written per batch of files, through a lookup that forgets between batches.
    """
    known_files = {row["path"] for row in connection.execute("SELECT path FROM files")}
    plan = _plan_facts(connection, root, named_changes)
    targets |= plan.invalidated
    targets &= known_files

    def load_symbols() -> list[Symbol]:
        return [symbol_from_row(row) for row in connection.execute("SELECT * FROM symbols")]

    _write_facts_symbols_for_read(connection, root, plan)
    overlay, targets = _map_overlay_for(connection, root, plan, targets, known_files, load_symbols)
    stats.reresolved_files = len(targets)

    ordered = sorted(targets)
    _unstage(connection, ordered)
    recovered = _recover_extracted_edges(root, plan, overlay, targets - fresh, workers)
    if recovered:
        # A file whose facts coverage moved is re-read from source, not from degraded copies.
        rereads = sorted({extraction.file_path for extraction in recovered})
        for chunk in _chunks(rereads):
            placeholders = ",".join("?" * len(chunk))
            connection.execute(f"DELETE FROM scratch.pending WHERE file_path IN ({placeholders})", chunk)  # noqa: S608
        _stage(connection, (edge for extraction in recovered for edge in extraction.edges))
    del recovered

    lookup = _lookup_for(connection, overlay)
    resolver = Resolver(lookup)
    hierarchy = _staged(connection, ordered, hierarchy=True)
    resolver.resolve_hierarchy(hierarchy)
    # The hierarchy a call chain is walked through is the one the graph will KEEP, so a file
    # whose facts own its supertypes contributes the tool's edges here rather than the
    # extractor's guesses. Resolving calls against the guesses and then storing the facts
    # would leave a chain the stored graph does not have.
    resolver.index_hierarchy(_hierarchy_after_overlay(hierarchy, overlay, targets))
    supertypes: dict[str, list[Edge]] = {}
    for edge in hierarchy:
        supertypes.setdefault(_file_of_edge(edge), []).append(edge)
    del hierarchy

    for batch in _chunks(ordered, _BATCH_FILES):
        files = set(batch)
        members = _staged(connection, batch, hierarchy=False)
        resolver.resolve_members(members)
        pending = [edge for path in batch for edge in supertypes.pop(path, [])] + members
        pending = _apply_overlay(pending, overlay, files)
        target_symbols = _target_symbols(connection, files, lookup)
        # A covered file's overrides come from its facts; deriving them again would double them.
        derived = resolver.derive_overrides(
            [symbol for symbol in target_symbols if EdgeKind.OVERRIDES not in overlay.kinds_owned(symbol.file_path)]
        )
        derived += resolver.derive_tests(target_symbols)
        derived += resolver.derive_exercises(pending)
        _stage(connection, pending + derived, table="resolved")
        lookup.release()
    _publish_resolved(connection)
    _write_facts_status(connection, overlay, plan)
    stats.facts = _facts_stats(connection)


def _file_of_edge(edge: Edge) -> str:
    """The file an edge is stored under: the one its source symbol is declared in."""
    return edge.src_id.split("#", 1)[0]


def _recover_extracted_edges(
    root: Path, plan: FactsPlan, overlay: FactsOverlay, stored_targets: set[str], workers: int
) -> list[FileExtraction]:
    """Re-extract the target files whose facts coverage may have changed.

    A covered file's extracted edges are REPLACED by the overlay's, so they are not in the
    table to reload: a file that loses its facts would otherwise be re-resolved from degraded
    copies of the fact edges rather than from what its source actually says. Re-reading those
    files is the only honest answer, and it is bounded by what one facts file covers.
    """
    changed_coverage = (plan.invalidated | plan.moved_coverage(overlay)) & stored_targets
    if not changed_coverage:
        return []
    jobs = [(str(root / path), path) for path in sorted(changed_coverage) if extractor_for(root / path) is not None]
    return _extract_many(jobs, workers)


def _lookup_for(connection: sqlite3.Connection, overlay: FactsOverlay) -> SymbolLookup:
    """Reach the workspace's declarations through sqlite's indexes, or the table facts mapping already read.

    Both answers are the same. Materialising the whole symbol table cost a full build ~340 MiB
    on a 200k-symbol workspace and bought no speed over the indexed lookup, so it happens only
    when mapping a facts file already had to read every symbol.
    """
    symbols = overlay.materialised_symbols
    if symbols is None:
        return SqliteLookup(connection)
    return MemoryLookup(symbols, _load_contexts(connection), _stored_hierarchy(connection))


def _target_symbols(connection: sqlite3.Connection, targets: set[str], lookup: SymbolLookup) -> list[Symbol]:
    """Return every symbol declared in a file being re-resolved, in table order."""
    if isinstance(lookup, MemoryLookup):
        return [symbol for symbol in lookup.all_symbols() if symbol.file_path in targets]
    found: list[Symbol] = []
    for chunk in _chunks(sorted(targets)):
        placeholders = ",".join("?" * len(chunk))
        query = f"SELECT * FROM symbols WHERE file_path IN ({placeholders})"  # noqa: S608
        found.extend(symbol_from_row(row) for row in connection.execute(query, chunk))
    return found


def _stored_hierarchy(connection: sqlite3.Connection) -> dict[str, list[str]]:
    """Load the resolved supertype map of every file the graph still holds edges for."""
    hierarchy: dict[str, list[str]] = {}
    rows = connection.execute(
        "SELECT src_id, dst_id FROM edges WHERE kind IN ('extends', 'implements') AND dst_id IS NOT NULL"
    )
    for row in rows:
        parents = hierarchy.setdefault(row["src_id"], [])
        if row["dst_id"] not in parents:
            parents.append(row["dst_id"])
    return hierarchy


# ---- the facts overlay, incrementally ------------------------------------


def _plan_facts(connection: sqlite3.Connection, root: Path, named_changes: Iterable[Path] | None) -> FactsPlan:
    """Decide which facts files moved, without reading a single one of them."""
    states = _facts_states(connection)
    return plan_facts(root, _present_facts_files(root, states, named_changes), states)


def _facts_states(connection: sqlite3.Connection) -> dict[str, FactsFileState]:
    """Read back what the previous build recorded about every facts file."""
    states: dict[str, FactsFileState] = {}
    for row in connection.execute("SELECT * FROM facts_status"):
        if row["fresh_paths"] is None:
            # Written before format version 5, so it lacks half of what a plan reasons about.
            # Forgetting it costs one re-mapping of that facts file and nothing after that.
            continue
        generated = {
            source: (str(template), bool(stale))
            for source, (template, stale) in json.loads(row["generated_templates"] or "{}").items()
        }
        states[row["path"]] = FactsFileState(
            relative_path=row["path"],
            mtime_ns=row["mtime_ns"] or 0,
            size=row["size"] or 0,
            sources=frozenset(json.loads(row["paths"] or "[]")),
            template_paths=frozenset(json.loads(row["template_paths"] or "[]")),
            generated=generated,
            contributions={
                path: SourceContribution.from_json(payload)
                for path, payload in json.loads(row["contributions"] or "{}").items()
            },
        )
    return states


def _present_facts_files(
    root: Path, states: dict[str, FactsFileState], named_changes: Iterable[Path] | None
) -> dict[str, Path]:
    """Find the workspace's facts files, walking the tree only when nobody named the changes.

    Walking for `**/build/zemble/*.jsonl` means descending every directory in the workspace,
    which costs more than the whole refresh it precedes. A caller that named its change set
    already promised to name every path that moved, facts files included - the daemon's
    watcher does exactly that - so the graph's own record plus that change set is the answer.
    """
    if named_changes is None:
        return {path.relative_to(root).as_posix(): path for path in discover_facts_files(root)}
    present = {relative: root / relative for relative in states if (root / relative).is_file()}
    for candidate in named_changes:
        try:
            relative = candidate.relative_to(root).as_posix()
        except ValueError:
            continue
        if candidate.is_file() and matches_facts_glob(root, candidate):
            present[relative] = candidate
    return present


def _map_overlay_for(
    connection: sqlite3.Connection,
    root: Path,
    plan: FactsPlan,
    targets: set[str],
    known_files: set[str],
    load_symbols: Callable[[], list[Symbol]],
) -> tuple[FactsOverlay, set[str]]:
    """Read and map exactly the facts files, and the sources of them, this build depends on.

    Two rounds, because mapping is what reveals which templates a moved facts file speaks for,
    and a second facts file covering one of those templates must then be mapped as well. A
    file that did not move declares what it declared before, so a third round can find nothing.

    :return: The overlay and the target set, grown by what the moved facts files turned out
        to cover.
    """
    request = plan.mapping_request(targets)
    if not request:
        return FactsOverlay(root=root), targets
    overlay = FactsOverlay(root=root)
    declared = StoredDeclaredSymbols(connection, "java")
    _read_requested(connection, root, plan, overlay, request)
    map_facts_files(overlay, load_symbols, request, declared)
    grown = targets | (plan.moved_coverage(overlay) & known_files)
    second = plan.mapping_request(grown)
    _read_requested(connection, root, plan, overlay, second)
    map_facts_files(overlay, load_symbols, second, declared)
    return overlay, grown


class StoredDeclaredSymbols:
    """The `symbol` facts of the whole workspace, kept in the graph and asked one ref at a time.

    A ref written in one facts file can be answered by a `symbol` fact in another, so mapping
    needs all of them - and parsing every facts file to answer a handful of refs is exactly
    what an incremental build must not do. The winner for a duplicated ref is the last one a
    full read would have taken: the highest facts file by path, and its last fact.
    """

    def __init__(self, connection: sqlite3.Connection, language: str) -> None:
        """Prepare the lookup; nothing is read until a ref actually falls to this rung."""
        self._connection = connection
        self._language = language
        self._cache: dict[str, tuple[str, int] | None] = {}

    def get(self, ref: str) -> tuple[str, int] | None:
        """Return the file path and line a `symbol` fact gave a ref, or None."""
        if ref not in self._cache:
            row = self._connection.execute(
                "SELECT file_path, line FROM facts_symbols WHERE ref = ? AND language = ? "
                "ORDER BY facts_file DESC, rowid DESC LIMIT 1",
                (ref, self._language),
            ).fetchone()
            self._cache[ref] = (row["file_path"], row["line"]) if row is not None else None
        return self._cache[ref]


def _write_facts_symbols_for_read(connection: sqlite3.Connection, root: Path, plan: FactsPlan) -> None:
    """Make sure every facts file present on disk has its `symbol` facts in the table.

    A facts file the graph has never read - or one written before this table existed - is
    parsed here for its `symbol` facts alone, so the lookup speaks for the whole workspace
    however little of it this build maps.
    """
    for relative in sorted(plan.vanished):
        connection.execute("DELETE FROM facts_symbols WHERE facts_file = ?", (relative,))
    # A facts file the graph already has a version-5 status row for was read by a build that
    # would have written its `symbol` facts, so holding none of them is the truth about it
    # rather than a gap to fill again on every build.
    stored = {row["facts_file"] for row in connection.execute("SELECT DISTINCT facts_file FROM facts_symbols")}
    known = stored | plan.moved | set(plan.states)
    missing = [plan.present[relative] for relative in sorted(set(plan.present) - known)]
    if missing:
        _write_facts_symbols(connection, read_facts_files(root, missing).files)


def _write_facts_symbols(connection: sqlite3.Connection, files: Sequence[FactsFile]) -> None:
    """Replace the `symbol` facts the graph holds for the given parsed facts files."""
    for loaded in files:
        connection.execute("DELETE FROM facts_symbols WHERE facts_file = ?", (loaded.relative_path,))
        connection.executemany(
            "INSERT INTO facts_symbols (ref, file_path, line, facts_file, language) VALUES (?,?,?,?,?)",
            [
                (ref, path, line, loaded.relative_path, loaded.header.language)
                for ref, path, line in symbol_facts(loaded)
            ],
        )


def _read_requested(
    connection: sqlite3.Connection, root: Path, plan: FactsPlan, overlay: FactsOverlay, request: dict
) -> None:
    """Parse the facts files a mapping request names that the overlay does not hold yet.

    Their `symbol` facts go into the table straight away, because the very mapping that is
    about to run reads them back out of it.
    """
    have = {loaded.relative_path for loaded in overlay.files} | {path for path, _ in overlay.errors}
    missing = [plan.present[relative] for relative in sorted(set(request) - have) if relative in plan.present]
    if not missing:
        return
    more = read_facts_files(root, missing)
    overlay.files.extend(more.files)
    overlay.errors.extend(more.errors)
    _write_facts_symbols(connection, more.files)


#: The `facts_status` columns a status row carries, in the order `_write_facts_status`
#: produces them. Named in the INSERT so a column added later is an obvious edit here.
_FACTS_STATUS_COLUMNS = (
    "path",
    "tool",
    "tool_version",
    "generated_at",
    "language",
    "mtime_ns",
    "size",
    "files_declared",
    "files_fresh",
    "files_stale",
    "unmapped",
    "paths",
    "template_paths",
    "fresh_paths",
    "contributions",
    "parse_buckets",
    "error",
    "generated_templates",
)
_FACTS_STATUS_COLUMNS_SQL = ", ".join(_FACTS_STATUS_COLUMNS)
_FACTS_STATUS_PLACEHOLDERS = ",".join("?" * len(_FACTS_STATUS_COLUMNS))


def _write_facts_status(connection: sqlite3.Connection, overlay: FactsOverlay, plan: FactsPlan) -> None:
    """Record what every facts file this build read contributed, and forget the ones that are gone.

    Rows for facts files this build did not read are left exactly as they were: their edges are
    still in the graph, so their accounting is still the truth. A file read but only PARTLY
    mapped keeps the stored accounting of the sources it was not asked about, which is why that
    accounting is kept per source rather than as one total.
    """
    for relative in sorted(plan.vanished):
        connection.execute("DELETE FROM facts_status WHERE path = ?", (relative,))
    rows = [_status_row(loaded, plan, overlay) for loaded in overlay.files]
    rows.extend(_error_rows(overlay))
    connection.executemany(
        f"INSERT OR REPLACE INTO facts_status ({_FACTS_STATUS_COLUMNS_SQL}) "  # noqa: S608
        f"VALUES ({_FACTS_STATUS_PLACEHOLDERS})",
        rows,
    )


def _status_row(loaded: FactsFile, plan: FactsPlan, overlay: FactsOverlay) -> tuple:
    """Build one facts file's status row, merging this build's mapping with what was stored."""
    contributions = _merged_contributions(loaded, plan, overlay)
    unmapped = sum(entry.buckets.get(SkipBucket.UNMAPPED.value, 0) for entry in contributions.values())
    return (
        loaded.relative_path,
        loaded.header.tool,
        loaded.header.tool_version,
        loaded.header.generated_at,
        loaded.header.language,
        loaded.mtime_ns,
        loaded.size,
        len(loaded.sources),
        len(loaded.fresh_files),
        len(loaded.stale_files),
        unmapped,
        json.dumps(sorted(loaded.sources)),
        json.dumps(sorted(_merged_templates(loaded, plan, overlay))),
        json.dumps(sorted(loaded.fresh_files)),
        json.dumps({path: entry.to_json() for path, entry in sorted(contributions.items())}),
        json.dumps(dict(loaded.parse_buckets)),
        _mapper_error(overlay, loaded.relative_path),
        json.dumps({source: list(entry) for source, entry in sorted(_merged_generated(loaded, plan, overlay).items())}),
    )


def _stored_state(loaded: FactsFile, plan: FactsPlan) -> FactsFileState | None:
    """Return what the previous build recorded about a facts file, if anything."""
    return plan.states.get(loaded.relative_path)


def _merged_contributions(loaded: FactsFile, plan: FactsPlan, overlay: FactsOverlay) -> dict[str, SourceContribution]:
    """Keep the stored accounting of the sources this build did not map, and replace the rest."""
    stored = _stored_state(loaded, plan)
    mapped = overlay.mapped_sources.get(loaded.relative_path, set())
    merged = {
        path: entry
        for path, entry in (stored.contributions if stored else {}).items()
        if path not in mapped and path in loaded.sources and loaded.sources[path].fresh
    }
    merged.update(loaded.contributions)
    return merged


def _merged_templates(loaded: FactsFile, plan: FactsPlan, overlay: FactsOverlay) -> set[str]:
    """Union this build's mapped templates with the stored ones of the sources it did not map.

    A stored source's template counts when its recorded verdict was not stale, which is a
    slight over-approximation: a generated source whose every ref went unmapped reached a
    template and gave it nothing. Erring that way only widens what a later build re-resolves.
    """
    stored = _stored_state(loaded, plan)
    mapped = overlay.mapped_sources.get(loaded.relative_path, set())
    kept = {
        template
        for source, (template, stale) in (stored.generated if stored else {}).items()
        if template and not stale and source not in mapped and source in loaded.sources
    }
    return kept | loaded.template_paths


def _merged_generated(loaded: FactsFile, plan: FactsPlan, overlay: FactsOverlay) -> dict[str, tuple[str, bool]]:
    """Keep the generated-source verdicts of the sources this build did not map."""
    stored = _stored_state(loaded, plan)
    mapped = overlay.mapped_sources.get(loaded.relative_path, set())
    merged = {
        source: entry
        for source, entry in (stored.generated if stored else {}).items()
        if source not in mapped and source in loaded.sources
    }
    merged.update(loaded.generated_templates)
    return merged


def _mapper_error(overlay: FactsOverlay, relative_path: str) -> str | None:
    """Return the mapping error recorded against one readable facts file, if any."""
    found = [message for path, message in overlay.errors if path == relative_path]
    return found[0] if found else None


def _error_rows(overlay: FactsOverlay) -> list[tuple]:
    """Build a status row for every facts file that could not be read at all.

    It is kept in the table so the next build's `stat` comparison sees a refusal it already
    made, and so a build that reads nothing still reports the error it reported before.
    """
    readable = {loaded.relative_path for loaded in overlay.files}
    rows = []
    for relative, message in overlay.errors:
        if relative in readable:
            continue
        try:
            stat = (overlay.root / relative).stat()
        except OSError:
            continue
        rows.append(
            (
                relative,
                None,
                None,
                None,
                None,
                stat.st_mtime_ns,
                stat.st_size,
                0,
                0,
                0,
                0,
                "[]",
                "[]",
                "[]",
                "{}",
                "{}",
                message,
                "{}",
            )
        )
    return rows


def _facts_stats(connection: sqlite3.Connection) -> dict[str, object]:
    """Summarise every facts file the graph currently stands on, mapped this build or not."""
    declared: set[str] = set()
    fresh: set[str] = set()
    templates: set[str] = set()
    counted: Counter = Counter({bucket.value: 0 for bucket in SkipBucket})
    files = external = generated_mapped = edges = 0
    errors: list[dict[str, str]] = []
    for row in connection.execute("SELECT * FROM facts_status"):
        declared |= set(json.loads(row["paths"] or "[]"))
        fresh |= set(json.loads(row["fresh_paths"] or "[]"))
        templates |= set(json.loads(row["template_paths"] or "[]"))
        counted.update(json.loads(row["parse_buckets"] or "{}"))
        for payload in json.loads(row["contributions"] or "{}").values():
            contribution = SourceContribution.from_json(payload)
            edges += contribution.edges
            external += contribution.external_targets
            generated_mapped += contribution.generated_mapped
            counted.update(contribution.buckets)
        files += 1 if row["tool"] else 0
        if row["error"]:
            errors.append({"path": row["path"], "error": row["error"]})
    return {
        "facts_files": files,
        "errors": errors,
        "files_declared": len(declared),
        "files_fresh": len(fresh),
        "files_stale": len(declared) - len(fresh),
        "edges": edges,
        "external_targets": external,
        "skipped": dict(counted),
        "unmapped": counted[SkipBucket.UNMAPPED.value],
        "generated_mapped": generated_mapped,
        "generated_templates": len(templates),
    }


def _delete_edges(connection: sqlite3.Connection, paths: Sequence[str]) -> None:
    """Drop the edges of files that are about to be re-resolved."""
    for chunk in _chunks(paths):
        placeholders = ",".join("?" * len(chunk))
        connection.execute(
            f"DELETE FROM edge_rows WHERE file IN (SELECT key FROM refs WHERE text IN ({placeholders}))",  # noqa: S608
            chunk,
        )


def _reset(edge: Edge) -> Edge:
    """Strip a stored edge's resolution so it can be resolved again."""
    edge.dst_id = None
    edge.resolution = Resolution.UNRESOLVED
    edge.candidates = []
    edge.candidate_count = 0
    edge.source = TREE_SITTER_SOURCE
    edge.origin_ref = None
    return edge


def _hierarchy_after_overlay(pending: list[Edge], overlay: FactsOverlay, targets: set[str]) -> list[Edge]:
    """Return the supertype edges the graph will keep, extractor's and tool's together."""
    kinds = (EdgeKind.EXTENDS, EdgeKind.IMPLEMENTS)
    kept = [
        edge
        for edge in pending
        if edge.kind in kinds and edge.kind not in overlay.kinds_owned(edge.src_id.split("#", 1)[0])
    ]
    for file_path in sorted(overlay.covered_files & targets):
        kept.extend(edge for edge in overlay.edges[file_path] if edge.kind in kinds)
    return kept


def _apply_overlay(pending: list[Edge], overlay: FactsOverlay, targets: set[str]) -> list[Edge]:
    """Replace the extracted call and hierarchy edges of every fact-covered file.

    Replacement is per FILE, never per edge: mixing a tool's edges with the extractor's
    would mean an answer no one could grade. A file the facts do not cover, or whose
    content moved on since they were written, keeps every extracted edge it had. Which KINDS
    a file's facts own is the overlay's call: a template reached through a Hawkeye source map
    yields only its calls, because the generated class knows nothing about what the template
    extends or renders.
    """
    kept = [edge for edge in pending if edge.kind not in overlay.kinds_owned(edge.src_id.split("#", 1)[0])]
    for file_path in sorted(overlay.covered_files & targets):
        kept.extend(overlay.edges[file_path])
    return kept

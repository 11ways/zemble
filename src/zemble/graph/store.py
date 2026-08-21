"""Sqlite storage and incremental build of the Java symbol graph.

The graph lives beside the search index (same cache folder) but is independent of
it: building the graph never requires an index, and clearing one does not corrupt
the other.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
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
from zemble.graph.hwk import extract_hwk_file
from zemble.graph.java import FileExtraction, extract_java_file
from zemble.graph.lookup import (
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
from zemble.parallel import pool_context, pooled
from zemble.types import ContentType

logger = logging.getLogger(__name__)

GRAPH_FORMAT_VERSION = 5
GRAPH_DB_NAME = "graph.sqlite"
#: The languages an extractor exists for, in the order they were added.
GRAPH_LANGUAGES = ("java", "hwk")
# Edge kinds that are computed from resolved symbols rather than extracted from source.
# They are always recomputed for a re-resolved file, never reloaded and re-inserted.
_DERIVED_KINDS = (EdgeKind.OVERRIDES.value, EdgeKind.TESTS.value, EdgeKind.EXERCISES.value)
#: File suffix -> the extractor that reads it. A suffix absent here is skipped and counted.
_EXTRACTORS = {".java": extract_java_file, ".hwk": extract_hwk_file}
_WORKER_CHUNK = 40
#: Above this many files to re-resolve, materialising the whole symbol table once beats
#: asking sqlite for each name a resolution touches. Both lookups answer identically; this
#: only decides which is cheaper, and `tests/test_graph_incremental.py` pins that.
_MEMORY_LOOKUP_TARGETS = 400
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
CREATE TABLE IF NOT EXISTS edges (
    src_id TEXT, dst_id TEXT, dst_name TEXT, kind TEXT, line INTEGER,
    resolution TEXT, candidates TEXT, arity INTEGER, receiver TEXT,
    receiver_type TEXT, is_new INTEGER, file_path TEXT, source TEXT, origin_ref TEXT
);
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
CREATE INDEX IF NOT EXISTS idx_edges_dst_kind ON edges(dst_id, kind);
CREATE INDEX IF NOT EXISTS idx_edges_src_kind ON edges(src_id, kind);
CREATE INDEX IF NOT EXISTS idx_edges_dstname_kind ON edges(dst_name, kind);
CREATE INDEX IF NOT EXISTS idx_edges_file ON edges(file_path);
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
        }


def graph_folder(path: str) -> Path:
    """Return the cache folder that holds the graph for a project path."""
    return find_index_from_cache_folder(path, (ContentType.CODE,))


def graph_db_path(path: str) -> Path:
    """Return the sqlite path of a project's graph, creating its folder if needed."""
    folder = graph_folder(path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / GRAPH_DB_NAME


def connect(path: str, *, read_only: bool = False) -> sqlite3.Connection:
    """Open (and if needed create) the graph database for a project path."""
    db_path = graph_db_path(path)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    if not read_only:
        connection.executescript(_SCHEMA)
        _migrate(connection)
    return connection


def _migrate(connection: sqlite3.Connection) -> None:
    """Add the columns a graph built by an older zemble does not have yet."""
    edge_columns = {row["name"] for row in connection.execute("PRAGMA table_info(edges)")}
    if "source" not in edge_columns:
        connection.execute("ALTER TABLE edges ADD COLUMN source TEXT")
    if "origin_ref" not in edge_columns:
        connection.execute("ALTER TABLE edges ADD COLUMN origin_ref TEXT")
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


def graph_exists(path: str) -> bool:
    """Return True if a graph database with symbols already exists for a path."""
    db_path = graph_folder(path) / GRAPH_DB_NAME
    if not db_path.is_file():
        return False
    connection = None
    try:
        connection = sqlite3.connect(db_path)
        return connection.execute("SELECT 1 FROM symbols LIMIT 1").fetchone() is not None
    except sqlite3.Error:
        return False
    finally:
        if connection is not None:
            connection.close()


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
    "candidates",
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
        json.dumps(edge.candidates) if edge.candidates else None,
        edge.arity,
        edge.receiver,
        edge.receiver_type,
        int(edge.is_new),
        edge.src_id.split("#", 1)[0],
        edge.source,
        edge.origin_ref,
    )


def edge_from_row(row: sqlite3.Row) -> Edge:
    """Rebuild an edge from a database row."""
    return Edge(
        src_id=row["src_id"],
        dst_name=row["dst_name"],
        kind=EdgeKind(row["kind"]),
        line=row["line"],
        dst_id=row["dst_id"],
        resolution=Resolution(row["resolution"]),
        candidates=json.loads(row["candidates"]) if row["candidates"] else [],
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
    extract = _EXTRACTORS[Path(absolute).suffix.lower()]
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
    """Extract a batch of files, using a process pool when the batch is large enough.

    The start method comes from `zemble.parallel.pool_context` (fork only in a
    single-threaded process, else spawn, else none); when no method is safe, or the pool
    fails for any reason, extraction runs in this process rather than aborting.
    """
    if len(jobs) < _WORKER_CHUNK * 2 or workers <= 1:
        return _extract_serial(jobs)
    context = pool_context()
    if context is None:
        return _extract_serial(jobs)
    try:
        with pooled(workers, context) as pool:
            return [result for result in pool.map(_extract_one, jobs, chunksize=_WORKER_CHUNK) if result is not None]
    except Exception:
        logger.warning("Parallel extraction unavailable; falling back to a single process", exc_info=True)
        return _extract_serial(jobs)


@dataclass
class _Scan:
    """The result of walking the workspace once."""

    jobs: list[tuple[str, str]] = field(default_factory=list)
    stamps: dict[str, tuple[int, int]] = field(default_factory=dict)
    skipped: Counter = field(default_factory=Counter)
    scanned: int = 0


def _scan(root: Path) -> _Scan:
    """Walk the workspace, splitting Java files from everything the graph has no extractor for."""
    scan = _Scan()
    for file_path in walk_files(root, extensions=get_extensions((ContentType.CODE,))):
        scan.scanned += 1
        suffix = file_path.suffix.lower()
        if suffix not in _EXTRACTORS:
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
        if candidate.suffix.lower() in _EXTRACTORS and ignored_prefix(root, relative) is None:
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
    """Build or incrementally refresh the symbol graph for a workspace.

    :param path: Local directory to index.
    :param force: Re-extract every file instead of only changed ones.
    :param workers: Extraction process count; defaults to the CPU count.
    :param changed_paths: The exact paths that moved, from a watcher; None walks the tree.
        The caller must name every path that moved, since nothing else is looked at.
    :return: Statistics describing the build.
    :raises ValueError: If the path is not a local directory.
    """
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"{path!r} is not a local directory")
    started = time.perf_counter()
    workers = workers if workers is not None else min(10, (os.cpu_count() or 2))

    connection = connect(str(root))
    # A forced build re-reads everything, facts files included, so it walks like a cold one.
    named = None if changed_paths is None or force else list(changed_paths)
    scan = _scan(root) if named is None else _scan_changed(root, named, _stored_stamps(connection))
    stats = GraphStats(root=str(root), files_scanned=scan.scanned, skipped_by_language=dict(scan.skipped))
    for language, count in sorted(scan.skipped.items(), key=lambda item: -item[1]):
        logger.info("graph: skipping %d %s file(s): no graph extractor for %s", count, language, language)

    try:
        _run_build(connection, root, scan, stats, force=force, workers=workers, named_changes=named)
    finally:
        connection.commit()
        connection.close()
    stats.duration_seconds = time.perf_counter() - started
    return stats


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
    """Do the two-pass build inside an open connection."""
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA journal_mode=MEMORY")
    known = _stored_stamps(connection)
    changed = [job for job in scan.jobs if force or known.get(job[1]) != scan.stamps[job[1]]]
    removed = sorted(set(known) - set(scan.stamps))
    stats.extracted_files = len(changed)
    stats.unchanged_files = len(scan.jobs) - len(changed)
    stats.removed_files = len(removed)

    touched = sorted({job[1] for job in changed} | set(removed))
    before = _declaration_index(connection, touched)
    _delete_files(connection, touched)

    extractions = _extract_many(changed, workers)
    _insert_extractions(connection, extractions, scan.stamps)
    after = _index_extractions(extractions)

    targets = set(touched) | _dependent_files(connection, _moved_names(before, after))
    targets &= set(scan.stamps)
    _resolve_pass(connection, extractions, targets, root, stats, named_changes, workers)
    connection.commit()
    _write_meta(connection, root)
    _write_coverage(connection, stats.skipped_by_language)
    stats.symbols = connection.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
    stats.edges = connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    stats.resolution_counts = {
        row["resolution"]: row["n"]
        for row in connection.execute("SELECT resolution, COUNT(*) AS n FROM edges GROUP BY resolution")
    }


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
        for table in ("symbols", "edges", "decl_keys"):
            connection.execute(f"DELETE FROM {table} WHERE file_path IN ({placeholders})", chunk)  # noqa: S608
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
    extractions: list[FileExtraction],
    targets: set[str],
    root: Path,
    stats: GraphStats,
    named_changes: Iterable[Path] | None,
    workers: int,
) -> None:
    """Run pass 2 for the target files against the whole workspace symbol table.

    The facts overlay is folded in here rather than afterwards, because the derived
    edges (overrides, exercises) must be derived from the edges the graph keeps, not
    from the extracted ones a facts file just replaced.

    Nothing here reads more of the workspace than the target files need. The symbol table is
    materialised only when the targets are numerous enough to make that the cheaper answer or
    when a facts file has to be mapped; the facts files are parsed only when at least one of
    them must be mapped; and a file the build is not re-resolving keeps every edge it had.
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

    fresh = {extraction.file_path for extraction in extractions}
    pending: list[Edge] = [edge for extraction in extractions for edge in extraction.edges]
    stored_targets = sorted(targets - fresh)
    recovered = _recover_extracted_edges(root, plan, overlay, set(stored_targets), workers)
    pending.extend(edge for extraction in recovered for edge in extraction.edges)
    reloaded = [path for path in stored_targets if path not in {e.file_path for e in recovered}]
    for chunk in _chunks(reloaded):
        placeholders = ",".join("?" * len(chunk))
        derived_placeholders = ",".join("?" * len(_DERIVED_KINDS))
        query = (  # noqa: S608
            f"SELECT * FROM edges WHERE file_path IN ({placeholders}) AND kind NOT IN ({derived_placeholders})"
        )
        pending.extend(_reset(edge_from_row(row)) for row in connection.execute(query, [*chunk, *_DERIVED_KINDS]))
    _delete_edges(connection, stored_targets)

    lookup = _lookup_for(connection, targets, overlay)
    resolver = Resolver(lookup)
    resolver.resolve_hierarchy(pending)
    # The hierarchy a call chain is walked through is the one the graph will KEEP, so a file
    # whose facts own its supertypes contributes the tool's edges here rather than the
    # extractor's guesses. Resolving calls against the guesses and then storing the facts
    # would leave a chain the stored graph does not have.
    resolver.index_hierarchy(_hierarchy_after_overlay(pending, overlay, targets))
    resolver.resolve_members(pending)
    pending = _apply_overlay(pending, overlay, targets)

    target_symbols = _target_symbols(connection, targets, lookup)
    # A covered file's overrides come from its facts; deriving them again would double them.
    derived = resolver.derive_overrides(
        [symbol for symbol in target_symbols if EdgeKind.OVERRIDES not in overlay.kinds_owned(symbol.file_path)]
    )
    derived += resolver.derive_tests(target_symbols)
    derived += resolver.derive_exercises(pending)
    connection.executemany(
        f"INSERT INTO edges ({_EDGE_COLUMNS_SQL}) VALUES ({_EDGE_PLACEHOLDERS})",  # noqa: S608
        [_edge_row(edge) for edge in pending + derived],
    )
    _write_facts_status(connection, overlay, plan)
    stats.facts = _facts_stats(connection)


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
    jobs = [(str(root / path), path) for path in sorted(changed_coverage) if (root / path).suffix in _EXTRACTORS]
    return _extract_many(jobs, workers)


def _lookup_for(connection: sqlite3.Connection, targets: set[str], overlay: FactsOverlay) -> SymbolLookup:
    """Pick the cheaper way to reach the workspace's declarations for this build.

    Both answers are the same; only the cost differs. A refresh of a handful of files touches
    a few thousand names and is far better served by the indexes sqlite already keeps, while a
    build re-resolving a large part of the tree would ask for most of the table one row at a
    time and should read it once instead.
    """
    if len(targets) <= _MEMORY_LOOKUP_TARGETS:
        return SqliteLookup(connection)
    symbols = overlay.materialised_symbols
    if symbols is None:
        symbols = [symbol_from_row(row) for row in connection.execute("SELECT * FROM symbols")]
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
        connection.execute(f"DELETE FROM edges WHERE file_path IN ({placeholders})", chunk)  # noqa: S608


def _reset(edge: Edge) -> Edge:
    """Strip a stored edge's resolution so it can be resolved again."""
    edge.dst_id = None
    edge.resolution = Resolution.UNRESOLVED
    edge.candidates = []
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

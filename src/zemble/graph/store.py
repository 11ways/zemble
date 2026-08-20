"""Sqlite storage and incremental build of the Java symbol graph.

The graph lives beside the search index (same cache folder) but is independent of
it: building the graph never requires an index, and clearing one does not corrupt
the other.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import sqlite3
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from zemble.cache import find_index_from_cache_folder
from zemble.graph.facts import OVERLAY_KINDS, TREE_SITTER_SOURCE, FactsOverlay, load_overlay
from zemble.graph.java import FileExtraction, FileImports, extract_java_file
from zemble.graph.model import Edge, EdgeKind, Resolution, Symbol, SymbolKind
from zemble.graph.resolve import FileContext, Resolver
from zemble.index.file_walker import walk_files
from zemble.index.files import detect_language, get_extensions
from zemble.types import ContentType

logger = logging.getLogger(__name__)

GRAPH_FORMAT_VERSION = 2
GRAPH_DB_NAME = "graph.sqlite"
GRAPH_LANGUAGE = "java"
# Edge kinds that are computed from resolved symbols rather than extracted from source.
# They are always recomputed for a re-resolved file, never reloaded and re-inserted.
_DERIVED_KINDS = (EdgeKind.OVERRIDES.value, EdgeKind.TESTS.value, EdgeKind.EXERCISES.value)
_EXTRACTOR_EXTENSIONS = frozenset({".java"})
_WORKER_CHUNK = 40
_MAX_FILE_BYTES = 2_000_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY, mtime_ns INTEGER, size INTEGER, package TEXT, imports TEXT
);
CREATE TABLE IF NOT EXISTS symbols (
    id TEXT PRIMARY KEY, kind TEXT, name TEXT, qualified_name TEXT, file_path TEXT,
    start_line INTEGER, end_line INTEGER, container_id TEXT, modifiers TEXT,
    annotations TEXT, signature TEXT, is_test INTEGER, param_types TEXT
);
CREATE TABLE IF NOT EXISTS edges (
    src_id TEXT, dst_id TEXT, dst_name TEXT, kind TEXT, line INTEGER,
    resolution TEXT, candidates TEXT, arity INTEGER, receiver TEXT,
    receiver_type TEXT, is_new INTEGER, file_path TEXT, source TEXT
);
CREATE TABLE IF NOT EXISTS facts_status (
    path TEXT PRIMARY KEY, tool TEXT, tool_version TEXT, generated_at TEXT, language TEXT,
    mtime_ns INTEGER, size INTEGER, files_declared INTEGER, files_fresh INTEGER,
    files_stale INTEGER, unmapped INTEGER, paths TEXT
);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_qualified ON symbols(qualified_name);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path);
CREATE INDEX IF NOT EXISTS idx_symbols_container ON symbols(container_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst_kind ON edges(dst_id, kind);
CREATE INDEX IF NOT EXISTS idx_edges_src_kind ON edges(src_id, kind);
CREATE INDEX IF NOT EXISTS idx_edges_dstname_kind ON edges(dst_name, kind);
CREATE INDEX IF NOT EXISTS idx_edges_file ON edges(file_path);
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
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(edges)")}
    if "source" not in columns:
        connection.execute("ALTER TABLE edges ADD COLUMN source TEXT")


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
    )


def symbol_from_row(row: sqlite3.Row) -> Symbol:
    """Rebuild a symbol from a database row."""
    return Symbol(
        id=row["id"],
        kind=SymbolKind(row["kind"]),
        name=row["name"],
        qualified_name=row["qualified_name"],
        file_path=row["file_path"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        container_id=row["container_id"],
        modifiers=json.loads(row["modifiers"]),
        annotations=json.loads(row["annotations"]),
        signature=row["signature"],
        is_test=bool(row["is_test"]),
        param_types=json.loads(row["param_types"]),
    )


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
    )


# ---- extraction ---------------------------------------------------------


def _extract_one(job: tuple[str, str]) -> FileExtraction | None:
    """Extract one file in a worker process, returning None when it cannot be read."""
    absolute, relative = job
    try:
        source = Path(absolute).read_bytes()
    except OSError:
        return None
    try:
        return extract_java_file(source, relative)
    except Exception:
        logger.warning("Failed to extract %s", relative, exc_info=True)
        return None


def _extract_serial(jobs: Sequence[tuple[str, str]]) -> list[FileExtraction]:
    """Extract a batch of files in this process."""
    return [result for job in jobs for result in [_extract_one(job)] if result is not None]


def _extract_many(jobs: Sequence[tuple[str, str]], workers: int) -> list[FileExtraction]:
    """Extract a batch of files, using a process pool when the batch is large enough.

    `fork` is preferred where the platform has it: the other start methods re-import
    the host's `__main__`, which fails outright for an embedded or piped interpreter.
    Any pool failure falls back to extracting in this process rather than aborting.
    """
    if len(jobs) < _WORKER_CHUNK * 2 or workers <= 1:
        return _extract_serial(jobs)
    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork") if "fork" in methods else None
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
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
        if suffix not in _EXTRACTOR_EXTENSIONS:
            scan.skipped[detect_language(file_path) or suffix] += 1
            continue
        try:
            stat = file_path.stat()
        except OSError:
            continue
        if stat.st_size > _MAX_FILE_BYTES:
            scan.skipped["java (too large)"] += 1
            continue
        relative = file_path.relative_to(root).as_posix()
        scan.jobs.append((str(file_path), relative))
        scan.stamps[relative] = (stat.st_mtime_ns, stat.st_size)
    return scan


# ---- build ---------------------------------------------------------------


def build_graph(path: str, *, force: bool = False, workers: int | None = None) -> GraphStats:
    """Build or incrementally refresh the symbol graph for a workspace.

    :param path: Local directory to index.
    :param force: Re-extract every file instead of only changed ones.
    :param workers: Extraction process count; defaults to the CPU count.
    :return: Statistics describing the build.
    :raises ValueError: If the path is not a local directory.
    """
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"{path!r} is not a local directory")
    started = time.perf_counter()
    workers = workers if workers is not None else min(10, (os.cpu_count() or 2))

    scan = _scan(root)
    stats = GraphStats(root=str(root), files_scanned=scan.scanned, skipped_by_language=dict(scan.skipped))
    for language, count in sorted(scan.skipped.items(), key=lambda item: -item[1]):
        logger.info("graph: skipping %d %s file(s): no graph extractor for %s", count, language, language)

    connection = connect(str(root))
    try:
        _run_build(connection, root, scan, stats, force=force, workers=workers)
    finally:
        connection.commit()
        connection.close()
    stats.duration_seconds = time.perf_counter() - started
    return stats


def _run_build(
    connection: sqlite3.Connection, root: Path, scan: _Scan, stats: GraphStats, *, force: bool, workers: int
) -> None:
    """Do the two-pass build inside an open connection."""
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA journal_mode=MEMORY")
    known = {row["path"]: (row["mtime_ns"], row["size"]) for row in connection.execute("SELECT * FROM files")}
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
    _resolve_pass(connection, extractions, targets, root, stats)
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
            ("language", GRAPH_LANGUAGE),
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
        for table in ("symbols", "edges"):
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
    connection.executemany("INSERT OR REPLACE INTO symbols VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", symbol_rows)


def _load_contexts(connection: sqlite3.Connection) -> dict[str, FileContext]:
    """Load every file's package and imports."""
    contexts: dict[str, FileContext] = {}
    for row in connection.execute("SELECT path, package, imports FROM files"):
        raw = json.loads(row["imports"]) if row["imports"] else {}
        contexts[row["path"]] = FileContext(
            file_path=row["path"],
            package=row["package"] or "",
            imports=FileImports(
                explicit=raw.get("explicit", {}),
                wildcards=raw.get("wildcards", []),
                static_members=raw.get("static_members", {}),
                static_wildcards=raw.get("static_wildcards", []),
            ),
        )
    return contexts


def _resolve_pass(
    connection: sqlite3.Connection, extractions: list[FileExtraction], targets: set[str], root: Path, stats: GraphStats
) -> None:
    """Run pass 2 for the target files against the whole workspace symbol table.

    The facts overlay is folded in here rather than afterwards, because the derived
    edges (overrides, exercises) must be derived from the edges the graph keeps, not
    from the extracted ones a facts file just replaced.
    """
    symbols = [symbol_from_row(row) for row in connection.execute("SELECT * FROM symbols")]
    overlay = load_overlay(root, symbols)
    targets |= _facts_targets(connection, overlay) & set(
        row["path"] for row in connection.execute("SELECT path FROM files")
    )
    stats.reresolved_files = len(targets)
    stats.facts = overlay.stats()
    resolver = Resolver(symbols, _load_contexts(connection))

    fresh = {extraction.file_path for extraction in extractions}
    pending: list[Edge] = [edge for extraction in extractions for edge in extraction.edges]
    stored_targets = sorted(targets - fresh)
    for chunk in _chunks(stored_targets):
        placeholders = ",".join("?" * len(chunk))
        derived_placeholders = ",".join("?" * len(_DERIVED_KINDS))
        query = (  # noqa: S608
            f"SELECT * FROM edges WHERE file_path IN ({placeholders}) AND kind NOT IN ({derived_placeholders})"
        )
        pending.extend(_reset(edge_from_row(row)) for row in connection.execute(query, [*chunk, *_DERIVED_KINDS]))
    _delete_edges(connection, stored_targets)

    resolver.index_hierarchy(_stored_hierarchy(connection, targets))
    resolver.resolve_all(pending)
    pending = _apply_overlay(pending, overlay, targets)

    target_symbols = [symbol for symbol in symbols if symbol.file_path in targets]
    # A covered file's overrides come from its facts; deriving them again would double them.
    derived = resolver.derive_overrides([symbol for symbol in target_symbols if not overlay.covers(symbol.file_path)])
    derived += resolver.derive_tests(target_symbols)
    derived += resolver.derive_exercises(pending)
    connection.executemany(
        "INSERT INTO edges VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", [_edge_row(edge) for edge in pending + derived]
    )
    _write_facts_status(connection, overlay)


def _stored_hierarchy(connection: sqlite3.Connection, targets: set[str]) -> list[Edge]:
    """Load already-resolved supertype edges from files that are not being re-resolved."""
    query = "SELECT * FROM edges WHERE kind IN ('extends', 'implements') AND dst_id IS NOT NULL"
    return [edge_from_row(row) for row in connection.execute(query) if row["file_path"] not in targets]


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
    return edge


def _apply_overlay(pending: list[Edge], overlay: FactsOverlay, targets: set[str]) -> list[Edge]:
    """Replace the extracted call and hierarchy edges of every fact-covered file.

    Replacement is per FILE, never per edge: mixing a tool's edges with the extractor's
    would mean an answer no one could grade. A file the facts do not cover, or whose
    content moved on since they were written, keeps every extracted edge it had.
    """
    kept = [
        edge for edge in pending if not (edge.kind in OVERLAY_KINDS and overlay.covers(edge.src_id.split("#", 1)[0]))
    ]
    for file_path in sorted(overlay.covered_files & targets):
        kept.extend(overlay.edges[file_path])
    return kept


def _facts_targets(connection: sqlite3.Connection, overlay: FactsOverlay) -> set[str]:
    """Return the source files whose edges must be rebuilt because their facts moved.

    A facts file that appeared, changed or vanished invalidates every source file it
    declares - now or last time - even though none of those files was itself edited.
    """
    previous = {row["path"]: row for row in connection.execute("SELECT * FROM facts_status")}
    current = {loaded.relative_path: loaded for loaded in overlay.files}
    moved: set[str] = set()
    for path, loaded in current.items():
        row = previous.get(path)
        if row is None or (row["mtime_ns"], row["size"]) != (loaded.mtime_ns, loaded.size):
            moved |= set(loaded.sources)
            if row is not None:
                moved |= set(json.loads(row["paths"] or "[]"))
    for path, row in previous.items():
        if path not in current:
            moved |= set(json.loads(row["paths"] or "[]"))
    return moved


def _write_facts_status(connection: sqlite3.Connection, overlay: FactsOverlay) -> None:
    """Record what every facts file contributed, and forget the ones that are gone."""
    connection.execute("DELETE FROM facts_status")
    unmapped_per_file: Counter = Counter(entry.facts_file for entry in overlay.unmapped)
    rows = [
        (
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
            unmapped_per_file.get(loaded.relative_path, 0),
            json.dumps(sorted(loaded.sources)),
        )
        for loaded in overlay.files
    ]
    connection.executemany("INSERT OR REPLACE INTO facts_status VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)

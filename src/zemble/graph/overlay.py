"""A lane's symbol graph as an overlay: its own rows in a small store, its base's graph read beneath them.

A lane holds its base's files except the few it edited, so its graph stores only the rows of the
files it owns (changed, added, or re-resolved because a name they wrote moved) plus masks: which
base files' rows those replace, and which base files lie outside the checkouts the lane holds. A
connection to the overlay attaches the base's graph version as `base` and creates TEMP views named
like every graph table; sqlite resolves an unqualified name in `temp` before `main`, so every
reader and the ordinary build read one row space without knowing a second store exists.

Two modes, one vocabulary: a READ connection hides the base's out-of-scope files (a lane answers
for its own checkouts), a BUILD connection keeps them (they are the world the lane's edits resolve
against, as a full graph of the workspace with the lane's edits applied would) and turns writes into
overlay rows and masks through INSTEAD OF triggers.

Keys: the overlay's `edge_rows` speak the base's `refs` keys for every text the base interned, and
negative keys of the overlay's own for the rest, so a join or GROUP BY on keys sees one id per text.
Those keys are the base VERSION's, which is why an overlay records the version (and the build stamp)
it was made on and is rebuilt when either moves; the store owns that decision.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

#: Meta keys an overlay records about the base it was built on.
META_BASE = "overlay_base"
META_BASE_VERSION = "overlay_base_version"
META_BASE_BUILT_AT = "overlay_base_built_at"
META_SCOPE = "overlay_scope"
#: What the overlay's second pass has yet to resolve (moved names, files to map facts onto; JSON), absent if nothing.
META_PENDING = "overlay_pending_names"

#: The schema an overlay attaches its base under.
BASE_SCHEMA = "base"

OVERLAY_SCHEMA = """
CREATE TABLE IF NOT EXISTS overlay_masked_files (path TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS overlay_masked_edges (key INTEGER PRIMARY KEY);
CREATE TABLE IF NOT EXISTS overlay_masked_facts (path TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS overlay_masked_status (path TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS overlay_hidden (path TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS overlay_hidden_keys (key INTEGER PRIMARY KEY);
"""

#: The first rowid of an overlay row in a table readers order by `rowid`, so it sorts after every base row.
#: AIDEV-NOTE: given at insert, never computed in the view: an expression in a compound view's column
#: (`o.rowid + offset`) stops sqlite flattening the view into a join, and every lookup joined to the
#: layered `symbols` then scanned all of them (a lane's resolution pass took ten minutes).
_OVERLAY_ROWID_START = 1 << 62

#: The file a symbol id belongs to: ids are `<file>#<qualified name>`.
_FILE_OF_SYMBOL = "substr({column}, 1, instr({column}, '#') - 1)"


class Mode(Enum):
    """How a connection presents the overlay: read by a query, or written by a build."""

    READ = "read"
    BUILD = "build"

    @property
    def hides_out_of_scope(self) -> bool:
        """Whether the base's files outside the lane's checkouts are left out."""
        match self:
            case Mode.READ:
                return True
            case Mode.BUILD:
                return False

    @property
    def writable(self) -> bool:
        """Whether writes through the views land in the overlay."""
        match self:
            case Mode.READ:
                return False
            case Mode.BUILD:
                return True


@dataclass(frozen=True)
class _Layer:
    """One graph table presented as base rows (minus masks) plus overlay rows.

    :param key: The SQL expression, over the row alias, naming what the row is masked and hidden by.
    :param mask: The mask table a replaced row is recorded in, keyed like `key`.
    :param hidden: The table naming what lies outside the lane's scope, keyed like `key`.
    :param delete_owned: The WHERE clause, over `OLD`, removing the overlay rows a deleted row stands for.
    :param upsert: Whether an insert replaces the overlay row with the same primary key.
    :param rowid: Whether readers order the table by `rowid`, which a view must then spell as a column.
    :param hides_overlay: Whether overlay rows are hidden out of scope too: only edges are, since the
        overlay re-resolves the edges of base files outside the lane when a name they wrote moved.
    """

    table: str
    key: str
    mask: str
    hidden: str
    delete_owned: str
    upsert: bool = False
    rowid: bool = False
    hides_overlay: bool = False


_LAYERS = (
    _Layer("files", "{a}.path", "overlay_masked_files", "overlay_hidden", "path = OLD.path", upsert=True),
    _Layer(
        "symbols",
        "{a}.file_path",
        "overlay_masked_files",
        "overlay_hidden",
        "id = OLD.id",
        upsert=True,
        rowid=True,
    ),
    _Layer(
        "symbol_metrics",
        _FILE_OF_SYMBOL.format(column="{a}.symbol_id"),
        "overlay_masked_files",
        "overlay_hidden",
        "symbol_id = OLD.symbol_id",
        upsert=True,
    ),
    _Layer("decl_keys", "{a}.file_path", "overlay_masked_files", "overlay_hidden", "file_path = OLD.file_path"),
    _Layer(
        "edge_rows",
        "{a}.file",
        "overlay_masked_edges",
        "overlay_hidden_keys",
        "file = OLD.file",
        hides_overlay=True,
    ),
    _Layer(
        "facts_symbols",
        "{a}.facts_file",
        "overlay_masked_facts",
        "overlay_hidden",
        "facts_file = OLD.facts_file",
        rowid=True,
    ),
    _Layer("facts_status", "{a}.path", "overlay_masked_status", "overlay_hidden", "path = OLD.path", upsert=True),
)

#: An edge's texts, as the store's `edges` view spells them, over `e` and its endpoints `s`, `d` and `f`.
_EDGE_TEXTS = (
    "s.text AS src_id, {dst} AS dst_id, e.dst_name, e.kind, e.line, e.resolution, e.candidate_count, e.arity, "
    "e.receiver, e.receiver_type, e.is_new, f.text AS file_path, e.source, e.origin_ref"
)
#: The joins naming an edge's endpoints, for a table of edge rows and a table of interned texts.
_EDGE_JOINS = (
    "{rows} e JOIN {refs} s ON s.key = e.src LEFT JOIN {refs} d ON d.key = e.dst JOIN {refs} f ON f.key = e.file"
)
#: `edge_refs`: every edge's keys beside its texts, which a reader aggregating over edges groups and filters by.
_EDGE_KEYS = "e.src, e.dst, e.file, "


def _edge_arms(keys: str, base_filter: str, own_filter: str) -> str:
    """The arms of an overlay's edge text view: the base's rows, then its own with and without a destination.

    AIDEV-NOTE: the shape is what keeps every lookup on an index. A base row carries base keys only, so it
    is joined to the base's `refs` alone: joined to the layered `refs`, sqlite expands every join into one
    arm per side and rescans the base's edges in each (a lane's `_dependent_files` ran for minutes). The
    overlay's own rows do need the layered `refs`, but only through INNER joins, which sqlite flattens; a
    LEFT JOIN on it is materialized whole, so a row without a destination is an arm of its own.
    """
    texts = _EDGE_TEXTS.format(dst="d.text")
    base = _EDGE_JOINS.format(rows=f"{BASE_SCHEMA}.edge_rows", refs=f"{BASE_SCHEMA}.refs")
    own = "main.edge_rows e JOIN refs s ON s.key = e.src JOIN refs f ON f.key = e.file"
    return (
        f"SELECT {keys}{texts} FROM {base} WHERE {base_filter} "
        f"UNION ALL SELECT {keys}{texts} FROM {own} JOIN refs d ON d.key = e.dst WHERE {own_filter} "
        f"UNION ALL SELECT {keys}{_EDGE_TEXTS.format(dst='NULL')} FROM {own} WHERE e.dst IS NULL AND {own_filter}"
    )


def is_overlay(connection: sqlite3.Connection, schema: str = "main") -> bool:
    """Return whether a graph version is a lane's overlay rather than a whole graph."""
    try:
        row = connection.execute(f"SELECT value FROM {schema}.meta WHERE key = ?", (META_BASE,)).fetchone()  # noqa: S608
    except sqlite3.DatabaseError:
        return False
    return row is not None and bool(row[0])


def is_overlay_file(db_path: Path) -> bool:
    """Return whether a graph version file on disk is an overlay, reading it without migrating anything."""
    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error:
        return False
    try:
        return is_overlay(connection)
    finally:
        connection.close()


def recorded_base(connection: sqlite3.Connection) -> tuple[str, str, str] | None:
    """Return the base root, version and build stamp an overlay was made on, or None for a whole graph."""
    meta = dict(
        connection.execute(
            "SELECT key, value FROM main.meta WHERE key IN (?, ?, ?)",
            (META_BASE, META_BASE_VERSION, META_BASE_BUILT_AT),
        ).fetchall()
    )
    if not meta.get(META_BASE):
        return None
    return meta[META_BASE], meta.get(META_BASE_VERSION, ""), meta.get(META_BASE_BUILT_AT, "")


def _columns(connection: sqlite3.Connection, table: str, schema: str = BASE_SCHEMA) -> list[str]:
    """A table's columns in order; the base's are the reference every reader was written against."""
    return [row[1] for row in connection.execute(f"PRAGMA {schema}.table_info({table})")]  # noqa: S608


def install(connection: sqlite3.Connection, mode: Mode) -> None:
    """Create the TEMP views (and, for a build, the triggers) presenting base plus overlay as one graph.

    The connection's `main` is the overlay and its base is attached as `base`.
    """
    for layer in _LAYERS:
        columns = _columns(connection, layer.table)
        # A column a retired format added stays in an older base (`files.generated`); the overlay has none.
        own = set(_columns(connection, layer.table, "main"))
        _create_view(connection, layer, columns, own, mode)
        if mode.writable:
            _create_triggers(connection, layer, [column for column in columns if column in own])
    connection.execute(
        f"CREATE TEMP VIEW refs AS SELECT key, text FROM {BASE_SCHEMA}.refs UNION ALL SELECT key, text FROM main.refs"  # noqa: S608
    )
    if mode.writable:
        # A text the base interned keeps the base's key; a new one gets the next negative key of the overlay's own.
        connection.execute(
            "CREATE TEMP TRIGGER overlay_refs_insert INSTEAD OF INSERT ON refs BEGIN "
            "INSERT INTO main.refs (key, text) SELECT MIN(0, COALESCE((SELECT MIN(key) FROM main.refs), 0)) - 1, "
            f"NEW.text WHERE NOT EXISTS (SELECT 1 FROM {BASE_SCHEMA}.refs WHERE text = NEW.text) "
            "AND NOT EXISTS (SELECT 1 FROM main.refs WHERE text = NEW.text); END"
        )
    edges = next(layer for layer in _LAYERS if layer.table == "edge_rows")
    base_filter, own_filter = _filters(edges, mode, "e", "e")
    for view, keys in (("edges", ""), ("edge_refs", _EDGE_KEYS)):
        connection.execute(f"CREATE TEMP VIEW {view} AS {_edge_arms(keys, base_filter, own_filter)}")  # noqa: S608


def install_readers(connection: sqlite3.Connection) -> None:
    """Create the readers' `edge_refs` view over a whole graph, as `install` does over an overlay."""
    joins = _EDGE_JOINS.format(rows="main.edge_rows", refs="main.refs")
    texts = _EDGE_TEXTS.format(dst="d.text")
    connection.execute(f"CREATE TEMP VIEW edge_refs AS SELECT {_EDGE_KEYS}{texts} FROM {joins}")  # noqa: S608


def _filters(layer: _Layer, mode: Mode, base_alias: str, own_alias: str) -> tuple[str, str]:
    """The WHERE clauses keeping a layer's unmasked (and, for a reader, in-scope) base rows and its own rows."""
    base_key, own_key = layer.key.format(a=base_alias), layer.key.format(a=own_alias)
    base_filter = f"NOT EXISTS (SELECT 1 FROM main.{layer.mask} m WHERE m.{_key_column(layer)} = {base_key})"
    own_filter = "1"
    if mode.hides_out_of_scope:
        hidden = f"SELECT 1 FROM main.{layer.hidden} h WHERE h.{_key_column(layer)}"
        base_filter += f" AND NOT EXISTS ({hidden} = {base_key})"
        if layer.hides_overlay:
            own_filter = f"NOT EXISTS ({hidden} = {own_key})"
    return base_filter, own_filter


def _create_view(connection: sqlite3.Connection, layer: _Layer, columns: list[str], own: set[str], mode: Mode) -> None:
    """Create one layered view: unmasked base rows, then the overlay's."""
    listed = ", ".join(columns)
    base_filter, own_filter = _filters(layer, mode, "b", "o")
    base_rowid = ", b.rowid AS rowid" if layer.rowid else ""
    own_rowid = ", o.rowid AS rowid" if layer.rowid else ""
    base_columns = ", ".join(f"b.{column}" for column in columns)
    own_columns = ", ".join(f"o.{column}" if column in own else "NULL" for column in columns)
    connection.execute(
        f"CREATE TEMP VIEW {layer.table} ({listed}{', rowid' if layer.rowid else ''}) AS "  # noqa: S608
        f"SELECT {base_columns}{base_rowid} FROM {BASE_SCHEMA}.{layer.table} b WHERE {base_filter} "
        f"UNION ALL SELECT {own_columns}{own_rowid} FROM main.{layer.table} o WHERE {own_filter}"
    )


def _key_column(layer: _Layer) -> str:
    """The column of a mask or hidden table: integer keys for edges, paths for everything else."""
    return "key" if layer.mask == "overlay_masked_edges" else "path"


def _create_triggers(connection: sqlite3.Connection, layer: _Layer, columns: list[str]) -> None:
    """Turn inserts and deletes on a layered view into overlay rows and masks of the base rows they replace."""
    listed = ", ".join(columns)
    new_values = ", ".join(f"NEW.{column}" for column in columns)
    new_key, old_key = layer.key.format(a="NEW"), layer.key.format(a="OLD")
    # A negative edge-file key is the overlay's own text, which no base row can carry.
    guard = " WHERE {} > 0" if _key_column(layer) == "key" else ""
    insert = "INSERT OR REPLACE" if layer.upsert else "INSERT"
    if layer.rowid:
        listed = f"rowid, {listed}"
        start = f"COALESCE((SELECT MAX(rowid) FROM main.{layer.table}), {_OVERLAY_ROWID_START - 1}) + 1"
        new_values = f"{start}, {new_values}"
    connection.execute(
        f"CREATE TEMP TRIGGER overlay_{layer.table}_insert INSTEAD OF INSERT ON {layer.table} BEGIN "  # noqa: S608
        f"INSERT OR IGNORE INTO main.{layer.mask} ({_key_column(layer)}) SELECT {new_key}{guard.format(new_key)}; "
        f"{insert} INTO main.{layer.table} ({listed}) VALUES ({new_values}); END"
    )
    connection.execute(
        f"CREATE TEMP TRIGGER overlay_{layer.table}_delete INSTEAD OF DELETE ON {layer.table} BEGIN "  # noqa: S608
        f"INSERT OR IGNORE INTO main.{layer.mask} ({_key_column(layer)}) SELECT {old_key}{guard.format(old_key)}; "
        f"DELETE FROM main.{layer.table} WHERE {layer.delete_owned}; END"
    )


def record_hidden(connection: sqlite3.Connection, holds: Callable[[str], bool], kept: Iterable[str]) -> int:
    """Record the base's files and facts files outside the lane's scope, by path and by edge-file key.

    :param holds: Whether a base-relative path is in the lane's scope.
    :param kept: Paths the lane holds whatever the scope says (its own walk found them).
    :return: How many paths were hidden.
    """
    own = set(kept)
    paths = [
        path
        for (path,) in connection.execute(
            f"SELECT path FROM {BASE_SCHEMA}.files UNION SELECT path FROM {BASE_SCHEMA}.facts_status"  # noqa: S608
        )
        if path not in own and not holds(path)
    ]
    connection.execute("DELETE FROM main.overlay_hidden")
    connection.execute("DELETE FROM main.overlay_hidden_keys")
    connection.executemany("INSERT OR IGNORE INTO main.overlay_hidden (path) VALUES (?)", ((path,) for path in paths))
    connection.execute(
        f"INSERT OR IGNORE INTO main.overlay_hidden_keys (key) SELECT r.key FROM {BASE_SCHEMA}.refs r "  # noqa: S608
        "JOIN main.overlay_hidden h ON r.text = h.path"
    )
    return len(paths)


def owned_files(connection: sqlite3.Connection) -> dict[str, tuple[int, int]]:
    """The stamps of the files whose rows the overlay itself holds."""
    return {row[0]: (row[1], row[2]) for row in connection.execute("SELECT path, mtime_ns, size FROM main.files")}


def restore_unchanged(connection: sqlite3.Connection) -> int:
    """Drop a build's re-resolved edges of base files that came out exactly as the base's, unmasking the base's.

    AIDEV-NOTE: a file the lane did not change is re-resolved when it wrote a name the lane moved, and
    on the zenit workspace one lane's 682 moved names made 11k of 19k files such dependents: stored
    again, the overlay grew to 988 MB, a copy of its base. Nearly all resolve as before; only the
    ones that differ are the lane's to keep. Runs on the build's staged edges, before they are published.

    :return: How many files were left to the base.
    """
    connection.execute("CREATE INDEX IF NOT EXISTS scratch.resolved_file ON resolved (file_path)")
    columns = ", ".join(row[1] for row in connection.execute("PRAGMA scratch.table_info(resolved)"))
    candidates = connection.execute(
        f"SELECT DISTINCT r.file_path, b.key FROM scratch.resolved r JOIN {BASE_SCHEMA}.refs b ON b.text = r.file_path "  # noqa: S608
        "JOIN main.overlay_masked_edges m ON m.key = b.key "
        "WHERE NOT EXISTS (SELECT 1 FROM main.files o WHERE o.path = r.file_path)"
    ).fetchall()
    texts = _EDGE_TEXTS.format(dst="d.text")
    base_rows = (
        f"SELECT {texts} FROM {_EDGE_JOINS.format(rows=f'{BASE_SCHEMA}.edge_rows', refs=f'{BASE_SCHEMA}.refs')} "
        "WHERE e.file = ?"
    )
    unchanged = [
        (path, key)
        for path, key in candidates
        if Counter(
            map(tuple, connection.execute(f"SELECT {columns} FROM scratch.resolved WHERE file_path = ?", (path,)))
        )  # noqa: S608
        == Counter(map(tuple, connection.execute(base_rows, (key,))))
    ]
    for start in range(0, len(unchanged), 400):
        chunk = unchanged[start : start + 400]
        marks = ",".join("?" * len(chunk))
        connection.execute(f"DELETE FROM scratch.resolved WHERE file_path IN ({marks})", [path for path, _ in chunk])  # noqa: S608
        connection.execute(f"DELETE FROM main.overlay_masked_edges WHERE key IN ({marks})", [key for _, key in chunk])  # noqa: S608
    return len(unchanged)


__all__ = [
    "BASE_SCHEMA",
    "META_BASE",
    "META_BASE_BUILT_AT",
    "META_BASE_VERSION",
    "META_PENDING",
    "META_SCOPE",
    "OVERLAY_SCHEMA",
    "Mode",
    "install",
    "is_overlay",
    "owned_files",
    "record_hidden",
    "recorded_base",
]

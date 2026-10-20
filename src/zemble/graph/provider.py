"""The query seam over a symbol graph.

`GraphProvider` is deliberately free of sqlite and tree-sitter types: a later
compiler-grade provider (javac, via zenit-dev) answers the same questions with
better resolution and drops straight in behind it.
"""

from __future__ import annotations

import json
import os
import posixpath
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace
from itertools import chain
from typing import Protocol, runtime_checkable

from zemble.graph.facts import TREE_SITTER_SOURCE
from zemble.graph.model import (
    NAMED_KINDS,
    TYPE_KINDS,
    CallerTally,
    CallTally,
    Edge,
    EdgeKind,
    EdgeTally,
    FileDependency,
    Hit,
    Resolution,
    RowCounts,
    Symbol,
    is_test_path,
)
from zemble.graph.signature import SymbolQuery
from zemble.graph.store import (
    GRAPH_FORMAT_VERSION,
    GraphStoreStale,
    connect,
    edge_from_row,
    resolve_graph_root,
    stored_format,
    symbol_from_row,
)
from zemble.languages.metrics import FILE_METRIC_COLUMNS, CallableMetrics, FileMetrics

_HIERARCHY_KINDS = (EdgeKind.EXTENDS.value, EdgeKind.IMPLEMENTS.value)

#: How a hit's `grade` words each resolution an edge can have.
RESOLUTION_PHRASES = {
    Resolution.EXACT: "exact match",
    Resolution.UNIQUE_NAME: "by-name match",
    Resolution.AMBIGUOUS: "ambiguous",
    Resolution.UNRESOLVED: "unresolved",
}

# The same edge reads in opposite directions depending on which end the answer is.
_VERBS_OUT = {
    EdgeKind.CALLS: "calls",
    EdgeKind.EXTENDS: "extends",
    EdgeKind.IMPLEMENTS: "implements",
    EdgeKind.OVERRIDES: "overrides",
    EdgeKind.REFERENCES_TYPE: "references",
    EdgeKind.ANNOTATED_WITH: "annotated with",
    EdgeKind.IMPORTS: "imports",
    EdgeKind.TESTS: "tests",
    EdgeKind.EXERCISES: "exercises",
}

_VERBS = {
    EdgeKind.CALLS: "called from",
    EdgeKind.EXTENDS: "extended by",
    EdgeKind.IMPLEMENTS: "implemented by",
    EdgeKind.OVERRIDES: "overridden by",
    EdgeKind.REFERENCES_TYPE: "referenced by",
    EdgeKind.ANNOTATED_WITH: "annotated in",
    EdgeKind.IMPORTS: "imported by",
    EdgeKind.TESTS: "tested by",
    EdgeKind.EXERCISES: "exercised by",
}


@runtime_checkable
class GraphProvider(Protocol):
    """Relationship queries over a workspace's symbols."""

    def definition(self, name: str) -> list[Symbol]:
        """Find declarations matching a simple name, a qualified name or `Type.member`, optionally `(Param,...)`."""
        ...

    def callers(self, symbol_id: str) -> list[Hit]:
        """Find every call site that reaches a callable."""
        ...

    def unattributed_calls(self, symbol_id: str) -> int:
        """Count the ambiguous call sites that may mean this callable but name no destination."""
        ...

    def callees(self, symbol_id: str) -> list[Hit]:
        """Find every callable invoked from a symbol's body."""
        ...

    def references(self, symbol_id: str) -> list[Hit]:
        """Find every edge of any kind pointing at a symbol."""
        ...

    def implementations(self, type_id: str, *, limit: int | None = None) -> list[Hit]:
        """Find direct and transitive subtypes of a type."""
        ...

    def supertypes(self, type_id: str, *, limit: int | None = None) -> list[Hit]:
        """Find direct and transitive supertypes of a type."""
        ...

    def overrides_of(self, method_id: str) -> list[Hit]:
        """Find the supertype method a method overrides."""
        ...

    def overridden_by(self, method_id: str) -> list[Hit]:
        """Find the subtype methods that override a method."""
        ...

    def tests_of(self, symbol_id: str) -> list[Hit]:
        """Find the tests covering a symbol, naming matches before incidental use."""
        ...

    def symbols_in_file(self, file_path: str) -> list[Symbol]:
        """List every symbol declared in one file, outermost and earliest first."""
        ...

    def symbols_at(self, file_path: str, start_line: int, end_line: int) -> list[Symbol]:
        """List the symbols whose line span contains a region, innermost first."""
        ...

    def neighbors(
        self, symbol_id: str, hops: int = 1, kinds: Sequence[EdgeKind] | None = None, *, limit: int | None = None
    ) -> list[Hit]:
        """Walk outward from a symbol in both directions."""
        ...

    def metrics(self, symbol_id: str) -> CallableMetrics | None:
        """Return a callable's measurements, or None when it has none (no body, unmeasured language)."""
        ...

    def metrics_in_file(self, file_path: str) -> dict[str, CallableMetrics]:
        """Return the measurements of every measured callable in one file, keyed by symbol id."""
        ...

    def file_metrics(self, file_path: str) -> FileMetrics | None:
        """Return a file's line and comment-line counts, or None when its language is not measured."""
        ...

    def all_metrics(self) -> Iterator[tuple[Symbol, CallableMetrics]]:
        """Yield every measured callable with its measurements, by file and line."""
        ...

    def all_symbols(self) -> Iterator[Symbol]:
        """Yield every symbol, by file and line."""
        ...

    def file_measurements(self) -> Iterator[tuple[str, FileMetrics, bool]]:
        """Yield every measured file's path, line counts and whether it is test code, by path."""
        ...

    def inbound_tallies(self, kinds: Sequence[EdgeKind] | None = None) -> list[EdgeTally]:
        """Count, per symbol, the distinct symbols reaching it by edge kind, resolution and test-ness."""
        ...

    def caller_tallies(self, kinds: Sequence[EdgeKind]) -> Iterator[CallerTally]:
        """Count, per symbol, the distinct symbols reaching it over these kinds, by their container."""
        ...

    def call_tallies(self) -> Iterator[CallTally]:
        """Count the calls each symbol writes on a non-type receiver, per callee (see `CallTally`)."""
        ...

    def outgoing_edges(self, kinds: Sequence[EdgeKind]) -> list[Edge]:
        """Return every edge of these kinds leaving a symbol, at any resolution, resolved or not."""
        ...

    def source_files(self) -> list[tuple[str, bool]]:
        """Return the absolute path of every file the graph was built from, and whether it is test code."""
        ...

    def unresolved_name_tallies(self, kinds: Sequence[EdgeKind] | None = None) -> dict[tuple[EdgeKind, str], int]:
        """Count the edges with no destination by kind and written name, across the whole graph."""
        ...

    def row_counts(self) -> RowCounts:
        """Count the graph's symbols, edges, measured callables and files, for sizing a whole-graph read."""
        ...

    def file_dependencies(self, kinds: Sequence[EdgeKind]) -> list[FileDependency]:
        """Count the resolved edges from each file into each other file, by kind and resolution."""
        ...


def display_name(symbol: Symbol) -> str:
    """Return a short human label: `Type` for types, `Type.member` for members."""
    if symbol.kind in NAMED_KINDS:
        return symbol.name
    parts = symbol.qualified_name.rsplit(".", 2)
    return ".".join(parts[-2:]) if len(parts) >= 2 else symbol.qualified_name


def _grade(edge: Edge) -> str:
    """Word how an edge was resolved: its grade, the tool that wrote it, and the generated member it came through.

    An edge an external tool wrote names that tool instead of a resolution grade: the
    grade is always `exact` there, and which tool said so is the part worth reading. An edge
    that reached its source through a source map also names the generated member it was
    written about, because "this template calls it" is only half the story a reader needs.
    """
    phrase = RESOLUTION_PHRASES[edge.resolution]
    if edge.resolution is Resolution.AMBIGUOUS:
        phrase = f"ambiguous, {edge.ambiguity()} candidates"
    if edge.source != TREE_SITTER_SOURCE:
        phrase = edge.source
    if edge.origin_ref:
        phrase = f"{phrase} via {edge.origin_ref}"
    return phrase


def _reason(symbol: Symbol, edge: Edge, depth: int = 1, *, outgoing: bool = False) -> str:
    """Build the one-line sentence explaining why a hit is in the answer."""
    table = _VERBS_OUT if outgoing else _VERBS
    verb = table.get(edge.kind, edge.kind.value)
    depth_note = f", depth {depth}" if depth > 1 else ""
    return f"{verb} {display_name(symbol)} (line {edge.line}, {_grade(edge)}{depth_note})"


_METRICS_SELECT = "SELECT symbol_id, lines, statements, complexity, nesting FROM symbol_metrics"


def _metrics_from_row(row: sqlite3.Row) -> CallableMetrics:
    """Rebuild one callable's measurements from a row carrying the `symbol_metrics` columns."""
    return CallableMetrics(row["lines"], row["statements"], row["complexity"], row["nesting"])


#: The `files` columns `FileMetrics.from_row` reads.
_FILE_METRICS = ", ".join(FILE_METRIC_COLUMNS)

#: Whether a `files f` row is test code: any symbol it declares is (the extractors' own verdict).
_FILE_IS_TEST = "(SELECT MAX(s.is_test) FROM symbols s WHERE s.file_path = f.path)"


def _file_is_test(row: sqlite3.Row) -> bool:
    """Read `_FILE_IS_TEST`, falling back to the path for a file that declares nothing."""
    return bool(row["is_test"]) or is_test_path(row["path"])


def _under(column: str, prefix: str) -> tuple[str, tuple[str, ...]]:
    """A WHERE clause keeping the rows whose `column` text starts with `prefix`, and its parameters.

    Every text under `prefix` sorts between it and the same text with its last character bumped,
    so the clause is an index range scan; an empty prefix keeps every row.
    """
    if not prefix:
        return "1", ()
    return f"{column} >= ? AND {column} < ?", (prefix, prefix[:-1] + chr(ord(prefix[-1]) + 1))


def _kinds_clause(column: str, kinds: Sequence[EdgeKind] | None) -> tuple[str, tuple[str, ...]]:
    """A WHERE clause keeping the edges of these kinds (every kind for None), and its parameters."""
    if kinds is None:
        return "1", ()
    values = tuple(kind.value for kind in kinds)
    return f"{column} IN ({','.join('?' * len(values))})", values


class SqliteGraphProvider:
    """A `GraphProvider` backed by the sqlite graph built by `zemble.graph.store`."""

    def __init__(self, path: str) -> None:
        """Open the graph database of a workspace path.

        :raises GraphStoreStale: If the current version is in another format than this zemble reads: its rows
            lack columns the readers name (a format-10 store has no `doc_lines`), so it is refused, never read.
        """
        self.path = path
        self.connection: sqlite3.Connection = connect(path)
        stored = stored_format(self.connection)
        if stored != GRAPH_FORMAT_VERSION:
            self.connection.close()
            raise GraphStoreStale(
                f"the symbol graph of {path} is format {stored}, this zemble reads format {GRAPH_FORMAT_VERSION}"
            )
        self.traversal_truncated = False

    def close(self) -> None:
        """Close the underlying database connection."""
        self.connection.close()

    # ---- symbol lookup --------------------------------------------------

    def symbol(self, symbol_id: str) -> Symbol | None:
        """Load one symbol by id."""
        row = self.connection.execute("SELECT * FROM symbols WHERE id = ?", (symbol_id,)).fetchone()
        return symbol_from_row(row) if row is not None else None

    def _symbols(self, ids: Iterable[str]) -> dict[str, Symbol]:
        """Load several symbols by id in one query."""
        ids = list(dict.fromkeys(ids))
        found: dict[str, Symbol] = {}
        for start in range(0, len(ids), 400):
            chunk = ids[start : start + 400]
            placeholders = ",".join("?" * len(chunk))
            query = f"SELECT * FROM symbols WHERE id IN ({placeholders})"  # noqa: S608
            for row in self.connection.execute(query, chunk):
                found[row["id"]] = symbol_from_row(row)
        return found

    def symbols_in_file(self, file_path: str) -> list[Symbol]:
        """List every symbol declared in one file, outermost and earliest first."""
        rows = self.connection.execute(
            "SELECT * FROM symbols WHERE file_path = ? ORDER BY start_line, end_line DESC", (file_path,)
        ).fetchall()
        return [symbol_from_row(row) for row in rows]

    def symbols_at(self, file_path: str, start_line: int, end_line: int) -> list[Symbol]:
        """List the symbols whose line span contains a region, innermost first.

        A chunk rarely lines up with a declaration, so containment is the anchor test:
        every symbol whose span covers the whole region, narrowest span first.
        """
        rows = self.connection.execute(
            "SELECT * FROM symbols WHERE file_path = ? AND start_line <= ? AND end_line >= ?",
            (file_path, start_line, end_line),
        ).fetchall()
        symbols = [symbol_from_row(row) for row in rows]
        return sorted(symbols, key=lambda symbol: (symbol.end_line - symbol.start_line, symbol.start_line, symbol.id))

    def definition(self, name: str) -> list[Symbol]:
        """Find declarations matching a simple name, a qualified name or `Type.member`, each optionally `(Param,...)`.

        A signature keeps only the callables whose parameter types match (`zemble.graph.signature`).
        """
        query = SymbolQuery.parse(name)
        return [symbol for symbol in self._named(query.name) if query.accepts(symbol)]

    def _named(self, name: str) -> list[Symbol]:
        """Find declarations matching a simple name, a qualified name or `Type.member`, best match first."""
        # Both branches use an index: `qualified_name` for a full name, `name` for the
        # last segment of `Type.member`. A LIKE '%.x' suffix scan would not.
        rows = self.connection.execute(
            "SELECT * FROM symbols WHERE qualified_name = ? OR name = ?", (name, name)
        ).fetchall()
        symbols = [symbol_from_row(row) for row in rows]
        if "." in name:
            last = name.rsplit(".", 1)[-1]
            suffix = f".{name}"
            seen = {symbol.id for symbol in symbols}
            symbols += [
                symbol
                for row in self.connection.execute("SELECT * FROM symbols WHERE name = ?", (last,))
                for symbol in [symbol_from_row(row)]
                if symbol.qualified_name.endswith(suffix) and symbol.id not in seen
            ]
        order = {"qualified": 0, "suffix": 1, "simple": 2}

        def rank(symbol: Symbol) -> tuple[int, int, str]:
            if symbol.qualified_name == name:
                bucket = order["qualified"]
            elif symbol.qualified_name.endswith(f".{name}"):
                bucket = order["suffix"]
            else:
                bucket = order["simple"]
            return bucket, 0 if symbol.kind in NAMED_KINDS else 1, symbol.id

        return sorted(symbols, key=rank)

    def unattributed_calls(self, symbol_id: str, *, prefix: str = "") -> int:
        """Count the ambiguous call sites under a prefix that may mean this callable but name no destination.

        AIDEV-NOTE: an ambiguous edge keeps only its candidate COUNT (`Edge.candidates`), so "may mean
        this one" is read from what the call wrote: the callable's name and arity on a receiver that
        names its declaring type, or a bare call in the type's own file. It is a count to warn with,
        never a list of callers.
        """
        symbol = self.symbol(symbol_id)
        owner = self.symbol(symbol.container_id) if symbol is not None and symbol.container_id else None
        if symbol is None or owner is None:
            return 0
        under, under_parameters = _under("file_path", prefix)
        row = self.connection.execute(
            "SELECT COUNT(*) FROM edges WHERE dst_name = ? AND kind = ? AND dst_id IS NULL AND resolution = ? "
            "AND (arity = ? OR arity < 0) "
            "AND (receiver_type IN (?, ?) OR receiver IN (?, ?) OR (receiver IS NULL AND file_path = ?)) "
            f"AND {under}",  # noqa: S608 - `under` is a literal clause with placeholders
            (
                symbol.name,
                EdgeKind.CALLS.value,
                Resolution.AMBIGUOUS.value,
                symbol.arity,
                owner.name,
                owner.qualified_name,
                owner.name,
                owner.qualified_name,
                owner.file_path,
                *under_parameters,
            ),
        ).fetchone()
        return int(row[0])

    # ---- edge queries ---------------------------------------------------

    def _incoming(self, symbol_id: str, kinds: Sequence[str] | None = None) -> list[Edge]:
        """Load edges pointing at a symbol."""
        return self._edges("dst_id", symbol_id, kinds)

    def _outgoing(self, symbol_id: str, kinds: Sequence[str] | None = None) -> list[Edge]:
        """Load resolved edges leaving a symbol."""
        return self._edges("src_id", symbol_id, kinds)

    def _edges(self, column: str, symbol_id: str, kinds: Sequence[str] | None) -> list[Edge]:
        """Load edges on one side of a symbol, optionally filtered by kind."""
        return list(self._iter_edges(column, symbol_id, kinds))

    def _iter_edges(self, column: str, symbol_id: str, kinds: Sequence[str] | None) -> Iterator[Edge]:
        """Stream one adjacency list, closing its cursor even when a traversal stops early."""
        query = f"SELECT * FROM edges WHERE {column} = ?"  # noqa: S608
        params: list[object] = [symbol_id]
        if kinds:
            query += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        if column == "src_id":
            query += " AND dst_id IS NOT NULL"
        cursor = self.connection.execute(query, params)
        try:
            for row in cursor:
                yield edge_from_row(row)
        finally:
            cursor.close()

    def _hits(self, edges: Sequence[Edge], *, side: str, depth: int = 1) -> list[Hit]:
        """Turn edges into hits by loading the symbol on the requested side."""
        ids = [edge.src_id if side == "src" else (edge.dst_id or "") for edge in edges]
        symbols = self._symbols(ident for ident in ids if ident)
        hits: list[Hit] = []
        for edge, ident in zip(edges, ids):
            symbol = symbols.get(ident)
            if symbol is None:
                continue
            hits.append(
                Hit(
                    symbol=symbol,
                    edge_kind=edge.kind,
                    line=edge.line,
                    resolution=edge.resolution,
                    reason=_reason(symbol, edge, depth, outgoing=side == "dst"),
                    depth=depth,
                    source=edge.source,
                    grade=_grade(edge),
                )
            )
        return hits

    def callers(self, symbol_id: str) -> list[Hit]:
        """Find every call site that reaches a callable."""
        return self._sorted(self._hits(self._incoming(symbol_id, [EdgeKind.CALLS.value]), side="src"))

    def callees(self, symbol_id: str) -> list[Hit]:
        """Find every callable invoked from a symbol's body."""
        return self._sorted(self._hits(self._outgoing(symbol_id, [EdgeKind.CALLS.value]), side="dst"))

    def references(self, symbol_id: str) -> list[Hit]:
        """Find every edge of any kind pointing at a symbol."""
        return self._sorted(self._hits(self._incoming(symbol_id), side="src"))

    def implementations(self, type_id: str, *, limit: int | None = None, prefix: str = "") -> list[Hit]:
        """Find direct and transitive subtypes, bounding a capped walk while it runs."""
        return self._walk(type_id, 32, _HIERARCHY_KINDS, incoming=True, limit=limit, prefix=prefix)

    def supertypes(self, type_id: str, *, limit: int | None = None, prefix: str = "") -> list[Hit]:
        """Find direct and transitive supertypes, bounding a capped walk while it runs."""
        return self._walk(type_id, 32, _HIERARCHY_KINDS, incoming=False, limit=limit, prefix=prefix)

    def _walk(
        self,
        symbol_id: str,
        hops: int,
        kinds: Sequence[str] | None,
        *,
        incoming: bool | None,
        limit: int | None,
        prefix: str,
    ) -> list[Hit]:
        """Walk breadth-first with bounded results and visited nodes for capped requests.

        Subtree walks may pass through siblings, but cannot accumulate an entire ancestor
        graph to find a few in-folder hits. A stopped walk reports a lower bound, not a total.
        """
        self.traversal_truncated = False
        if limit is not None and limit < 1:
            raise ValueError("traversal limit must be positive")
        visit_limit = max(4096, limit * 32) if limit is not None else None
        hits: list[Hit] = []
        seen = {symbol_id}
        frontier = [symbol_id]
        for depth in range(1, max(1, hops) + 1):
            next_frontier: list[str] = []
            for current in frontier:
                outgoing = self._iter_edges("src_id", current, kinds)
                ingoing = self._iter_edges("dst_id", current, kinds)
                edges = ingoing if incoming is True else outgoing if incoming is False else chain(outgoing, ingoing)
                try:
                    for edge in edges:
                        side = "src" if incoming is True or (incoming is None and edge.dst_id == current) else "dst"
                        for hit in self._hits([edge], side=side, depth=depth):
                            if hit.symbol.id in seen:
                                continue
                            seen.add(hit.symbol.id)
                            next_frontier.append(hit.symbol.id)
                            if hit.symbol.file_path.startswith(prefix):
                                hits.append(hit)
                            if (limit is not None and len(hits) >= limit) or (
                                visit_limit is not None and len(seen) >= visit_limit
                            ):
                                self.traversal_truncated = True
                                return hits
                finally:
                    outgoing.close()
                    ingoing.close()
            frontier = next_frontier
            if not frontier:
                break
        return hits

    def overrides_of(self, method_id: str) -> list[Hit]:
        """Find the supertype method a method overrides."""
        return self._hits(self._outgoing(method_id, [EdgeKind.OVERRIDES.value]), side="dst")

    def overridden_by(self, method_id: str) -> list[Hit]:
        """Find the subtype methods that override a method."""
        return self._sorted(self._hits(self._incoming(method_id, [EdgeKind.OVERRIDES.value]), side="src"))

    def tests_of(self, symbol_id: str) -> list[Hit]:
        """Find the tests covering a symbol, naming matches before incidental use."""
        symbol = self.symbol(symbol_id)
        if symbol is None:
            return []
        type_id = symbol_id if symbol.kind in TYPE_KINDS else (symbol.container_id or symbol_id)
        named = self._hits(self._incoming(type_id, [EdgeKind.TESTS.value]), side="src")
        exercising = self._hits(self._incoming(symbol_id, [EdgeKind.EXERCISES.value]), side="src")
        seen: set[str] = set()
        ordered: list[Hit] = []
        for hit in named + exercising:
            if hit.symbol.id in seen:
                continue
            seen.add(hit.symbol.id)
            ordered.append(hit)
        return ordered

    def neighbors(
        self,
        symbol_id: str,
        hops: int = 1,
        kinds: Sequence[EdgeKind] | None = None,
        *,
        limit: int | None = None,
        prefix: str = "",
    ) -> list[Hit]:
        """Walk outward in both directions, stopping during traversal when a cap is reached."""
        return self._walk(
            symbol_id,
            hops,
            [kind.value for kind in kinds] if kinds else None,
            incoming=None,
            limit=limit,
            prefix=prefix,
        )

    @staticmethod
    def _sorted(hits: list[Hit]) -> list[Hit]:
        """Order hits by file then line so output is stable."""
        return sorted(hits, key=lambda hit: (hit.symbol.file_path, hit.line, hit.symbol.id))

    # ---- measurements -----------------------------------------------------

    def metrics(self, symbol_id: str) -> CallableMetrics | None:
        """Return a callable's measurements, or None when it has none (no body, unmeasured language)."""
        row = self.connection.execute(f"{_METRICS_SELECT} WHERE symbol_id = ?", (symbol_id,)).fetchone()  # noqa: S608
        return _metrics_from_row(row) if row is not None else None

    def metrics_in_file(self, file_path: str) -> dict[str, CallableMetrics]:
        """Return the measurements of every measured callable in one file, keyed by symbol id."""
        rows = self.connection.execute(
            f"{_METRICS_SELECT} WHERE symbol_id IN (SELECT id FROM symbols WHERE file_path = ?)",  # noqa: S608
            (file_path,),
        )
        return {row["symbol_id"]: _metrics_from_row(row) for row in rows}

    def file_metrics(self, file_path: str) -> FileMetrics | None:
        """Return a file's measurements, or None when its language is not measured."""
        row = self.connection.execute(
            f"SELECT {_FILE_METRICS} FROM files WHERE path = ? AND lines IS NOT NULL",  # noqa: S608
            (file_path,),
        ).fetchone()
        return FileMetrics.from_row(row) if row is not None else None

    def all_metrics(self, *, prefix: str = "") -> Iterator[tuple[Symbol, CallableMetrics]]:
        """Yield every measured callable with its measurements, by file and line, optionally under a path prefix."""
        where, parameters = _under("s.file_path", prefix)
        query = (
            "SELECT s.*, m.lines, m.statements, m.complexity, m.nesting "
            f"FROM symbol_metrics m JOIN symbols s ON s.id = m.symbol_id WHERE {where} "  # noqa: S608
            "ORDER BY s.file_path, s.start_line, s.id"
        )
        for row in self.connection.execute(query, parameters):
            yield symbol_from_row(row), _metrics_from_row(row)

    # ---- bulk reads: whole-graph aggregates, one query each -------------

    def all_symbols(self, *, prefix: str = "") -> Iterator[Symbol]:
        """Yield every symbol, by file and line, optionally under a path prefix."""
        where, parameters = _under("file_path", prefix)
        query = f"SELECT * FROM symbols WHERE {where} ORDER BY file_path, start_line, id"  # noqa: S608
        for row in self.connection.execute(query, parameters):
            yield symbol_from_row(row)

    def file_measurements(self, *, prefix: str = "") -> Iterator[tuple[str, FileMetrics, bool]]:
        """Yield every measured file's path, line counts and test-ness, by path, optionally under a prefix.

        A file is test code when any symbol it declares is, else when its path is under a test source set.
        """
        where, parameters = _under("f.path", prefix)
        query = (
            f"SELECT f.path, {_FILE_METRICS}, "  # noqa: S608
            f"{_FILE_IS_TEST} AS is_test "
            f"FROM files f WHERE f.lines IS NOT NULL AND {where} ORDER BY f.path"  # noqa: S608
        )
        for row in self.connection.execute(query, parameters):
            path = row["path"]
            yield path, FileMetrics.from_row(row), _file_is_test(row)

    def inbound_tallies(self, kinds: Sequence[EdgeKind] | None = None, *, prefix: str = "") -> list[EdgeTally]:
        """Count, per symbol under a prefix, the distinct symbols reaching it by kind, resolution and test-ness.

        Sources are counted wherever they live: a caller outside the prefix still calls.
        """
        kinds_where, kinds_parameters = _kinds_clause("e.kind", kinds)
        under, under_parameters = _under("d.text", prefix)
        query = (
            "SELECT d.text AS dst_id, e.kind, e.resolution, COALESCE(s.is_test, 0) AS from_test, "
            "COUNT(DISTINCT e.src) AS sources "
            "FROM edge_rows e JOIN refs d ON d.key = e.dst JOIN refs r ON r.key = e.src "
            "LEFT JOIN symbols s ON s.id = r.text "
            f"WHERE {kinds_where} AND {under} "  # noqa: S608
            "GROUP BY e.dst, e.kind, e.resolution, from_test"
        )
        return [
            EdgeTally(
                row["dst_id"],
                EdgeKind(row["kind"]),
                Resolution(row["resolution"]),
                bool(row["from_test"]),
                row["sources"],
            )
            for row in self.connection.execute(query, kinds_parameters + under_parameters)
        ]

    def caller_tallies(self, kinds: Sequence[EdgeKind], *, prefix: str = "") -> Iterator[CallerTally]:
        """Count, per symbol under a prefix, the distinct symbols reaching it, by container, resolution and test-ness.

        Aggregated in SQL, one row per (symbol, container) pair rather than per edge; sources are
        counted wherever they live.
        """
        kinds_where, kinds_parameters = _kinds_clause("e.kind", kinds)
        under, under_parameters = _under("d.text", prefix)
        query = (
            "SELECT d.text AS dst_id, COALESCE(s.container_id, r.text) AS owner_id, e.resolution, "
            "COALESCE(s.is_test, 0) AS from_test, COUNT(DISTINCT e.src) AS sources "
            "FROM edge_rows e JOIN refs d ON d.key = e.dst JOIN refs r ON r.key = e.src "
            "LEFT JOIN symbols s ON s.id = r.text "
            f"WHERE {kinds_where} AND {under} "  # noqa: S608
            "GROUP BY e.dst, owner_id, e.resolution, from_test"
        )
        for row in self.connection.execute(query, kinds_parameters + under_parameters):
            yield CallerTally(
                row["dst_id"], row["owner_id"], Resolution(row["resolution"]), bool(row["from_test"]), row["sources"]
            )

    def call_tallies(self, *, prefix: str = "") -> Iterator[CallTally]:
        """Count the calls each symbol under a prefix writes on a non-type receiver, per callee, aggregated in SQL.

        A receiver is a type name when the extractor marked it as one: its `receiver_type` is the
        receiver itself or the receiver's last dotted segment (`Str`, `com.x.Str`). A receiver is chained
        when it is itself a call's result (its text holds a parenthesis: `builder.add(x).named(y)`).
        """
        under, under_parameters = _under("r.text", prefix)
        query = (
            "SELECT r.text AS src_id, d.text AS dst_id, e.resolution, e.receiver IS NULL AS bare, "
            "instr(COALESCE(e.receiver, ''), '(') > 0 AS chained, COUNT(*) AS calls "
            "FROM edge_rows e JOIN refs r ON r.key = e.src JOIN refs d ON d.key = e.dst "
            "WHERE e.kind = ? AND COALESCE(e.is_new, 0) = 0 AND NOT (e.receiver_type IS NOT NULL AND "
            "(e.receiver = e.receiver_type "
            "OR substr(e.receiver, -length(e.receiver_type) - 1) = '.' || e.receiver_type)) "
            f"AND {under} "  # noqa: S608
            "GROUP BY e.src, e.dst, e.resolution, bare, chained"
        )
        for row in self.connection.execute(query, (EdgeKind.CALLS.value, *under_parameters)):
            resolution = Resolution(row["resolution"])
            yield CallTally(
                row["src_id"], row["dst_id"], resolution, bool(row["bare"]), bool(row["chained"]), row["calls"]
            )

    def outgoing_edges(self, kinds: Sequence[EdgeKind], *, prefix: str = "") -> list[Edge]:
        """Return every edge of these kinds leaving a symbol under a prefix, at any resolution, resolved or not."""
        kinds_where, kinds_parameters = _kinds_clause("kind", kinds)
        under, under_parameters = _under("src_id", prefix)
        query = f"SELECT * FROM edges WHERE {kinds_where} AND {under} ORDER BY src_id, line"  # noqa: S608
        return [edge_from_row(row) for row in self.connection.execute(query, kinds_parameters + under_parameters)]

    def source_files(self) -> list[tuple[str, bool]]:
        """Return the absolute path of every file the graph was built from, and its test-ness, sorted."""
        root = os.path.realpath(self.path)
        rows = self.connection.execute(f"SELECT f.path, {_FILE_IS_TEST} AS is_test FROM files f ORDER BY f.path")  # noqa: S608
        return [(os.path.join(root, row["path"]), _file_is_test(row)) for row in rows]

    def unresolved_name_tallies(self, kinds: Sequence[EdgeKind] | None = None) -> dict[tuple[EdgeKind, str], int]:
        """Count the edges with no destination (ambiguous or unresolved) by kind and written name.

        Always the whole graph: a name written anywhere may mean a symbol in any sub-directory.
        """
        kinds_where, kinds_parameters = _kinds_clause("kind", kinds)
        query = (
            "SELECT kind, dst_name, COUNT(*) AS edges FROM edge_rows "
            f"WHERE dst IS NULL AND dst_name IS NOT NULL AND {kinds_where} GROUP BY kind, dst_name"  # noqa: S608
        )
        return {
            (EdgeKind(row["kind"]), row["dst_name"]): row["edges"]
            for row in self.connection.execute(query, kinds_parameters)
        }

    def row_counts(self) -> RowCounts:
        """Count the whole graph's rows, four cheap queries."""

        def count(table: str) -> int:
            return int(self.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])  # noqa: S608

        return RowCounts(count("symbols"), count("edge_rows"), count("symbol_metrics"), count("files"))

    def file_dependencies(self, kinds: Sequence[EdgeKind], *, prefix: str = "") -> list[FileDependency]:
        """Count the resolved edges from each file under a prefix into each other file, by kind and resolution."""
        kinds_where, kinds_parameters = _kinds_clause("e.kind", kinds)
        under, under_parameters = _under("f.text", prefix)
        query = (
            "SELECT f.text AS source_path, substr(d.text, 1, instr(d.text, '#') - 1) AS target_path, "
            "e.kind, e.resolution, COUNT(*) AS edges "
            "FROM edge_rows e JOIN refs f ON f.key = e.file JOIN refs d ON d.key = e.dst "
            f"WHERE {kinds_where} AND {under} "  # noqa: S608
            "GROUP BY e.file, target_path, e.kind, e.resolution"
        )
        return [
            FileDependency(
                row["source_path"],
                row["target_path"],
                EdgeKind(row["kind"]),
                Resolution(row["resolution"]),
                row["edges"],
            )
            for row in self.connection.execute(query, kinds_parameters + under_parameters)
            if row["target_path"] and row["target_path"] != row["source_path"]
        ]

    # ---- coverage --------------------------------------------------------

    def meta(self) -> dict[str, str]:
        """Return the graph's stored metadata."""
        return {row["key"]: row["value"] for row in self.connection.execute("SELECT key, value FROM meta")}

    def coverage_note(self) -> str:
        """Explain what the graph does and does not cover, for empty answers."""
        meta = self.meta()
        covered = meta.get("language", "")
        raw = meta.get("skipped_by_language")
        skipped = json.loads(raw) if raw else {}
        count = len(covered.split(",")) if covered else 0
        summary = f"The graph covers {count} languages (every bundled grammar)."
        if not skipped:
            return summary
        listed = ", ".join(
            f"{language} ({files})" for language, files in sorted(skipped.items(), key=lambda item: -item[1])[:6]
        )
        return f"{summary[:-1]}; no grammar for: {listed}."


class SubtreeGraphProvider:
    """A sub-directory's view of an ancestor's graph: its symbols only, ids and paths relative to it.

    AIDEV-NOTE: this is how a sub-directory is answered without a graph of its own, the way an
    ancestor's search index answers it through `ZembleIndex.subtree`. Every Symbol leaving the
    view is rebased (`id`, `container_id`, `file_path` lose the prefix) and every id entering it
    gets the prefix back, so a caller sees exactly the spelling a graph built for the sub-directory
    would use. Edges and walks run over the whole ancestor graph and only their answers are
    filtered: a hierarchy chain through a sibling module still reaches in-folder symbols, and a
    call into a sibling module resolves against its real target instead of by name.
    """

    def __init__(self, graph: SqliteGraphProvider, prefix: str, path: str) -> None:
        """Wrap an open ancestor provider.

        :param graph: The provider on the ancestor's graph; this view closes it.
        :param prefix: The sub-directory, relative to the ancestor root, in POSIX form.
        :param path: The path the caller asked about.
        """
        self.graph = graph
        self.path = path
        self.prefix = prefix.strip("/")
        self._lead = f"{self.prefix}/"

    @property
    def connection(self) -> sqlite3.Connection:
        """The ancestor graph's connection; raw SQL over it is NOT filtered to the sub-directory."""
        return self.graph.connection

    @property
    def traversal_truncated(self) -> bool:
        """Whether the ancestor walk stopped before counting every reachable symbol."""
        return self.graph.traversal_truncated

    def close(self) -> None:
        """Close the ancestor provider."""
        self.graph.close()

    def _inbound(self, symbol_id: str) -> str:
        """Spell a view id the way the ancestor graph stores it."""
        return self._lead + symbol_id

    def _strip(self, value: str) -> str:
        """Drop the prefix from an ancestor-relative id or path."""
        return value[len(self._lead) :] if value.startswith(self._lead) else value

    def _rebased(self, symbol: Symbol | None) -> Symbol | None:
        """Return the symbol as the view spells it, or None when it lies outside the sub-directory."""
        if symbol is None or not symbol.file_path.startswith(self._lead):
            return None
        return replace(
            symbol,
            id=self._strip(symbol.id),
            file_path=self._strip(symbol.file_path),
            container_id=self._strip(symbol.container_id) if symbol.container_id is not None else None,
        )

    def _symbols(self, symbols: Iterable[Symbol]) -> list[Symbol]:
        """Keep and rebase the in-folder symbols, in their order."""
        return [rebased for symbol in symbols if (rebased := self._rebased(symbol)) is not None]

    def _hits(self, hits: Iterable[Hit]) -> list[Hit]:
        """Keep and rebase the hits whose symbol is in the sub-directory, in their order."""
        return [replace(hit, symbol=rebased) for hit in hits if (rebased := self._rebased(hit.symbol)) is not None]

    def symbol(self, symbol_id: str) -> Symbol | None:
        """Load one in-folder symbol by its view id."""
        return self._rebased(self.graph.symbol(self._inbound(symbol_id)))

    def symbols_in_file(self, file_path: str) -> list[Symbol]:
        """List every symbol declared in one file under the sub-directory."""
        return self._symbols(self.graph.symbols_in_file(self._lead + file_path))

    def symbols_at(self, file_path: str, start_line: int, end_line: int) -> list[Symbol]:
        """List the symbols whose line span contains a region of a file under the sub-directory."""
        return self._symbols(self.graph.symbols_at(self._lead + file_path, start_line, end_line))

    def definition(self, name: str) -> list[Symbol]:
        """Find in-folder declarations matching a simple name, a qualified name or `Type.member`, optionally signed."""
        return self._symbols(self.graph.definition(name))

    def callers(self, symbol_id: str) -> list[Hit]:
        """Find every in-folder call site that reaches a callable."""
        return self._hits(self.graph.callers(self._inbound(symbol_id)))

    def unattributed_calls(self, symbol_id: str) -> int:
        """Count the in-folder ambiguous call sites that may mean this callable but name no destination."""
        return self.graph.unattributed_calls(self._inbound(symbol_id), prefix=self._lead)

    def callees(self, symbol_id: str) -> list[Hit]:
        """Find every in-folder callable invoked from a symbol's body."""
        return self._hits(self.graph.callees(self._inbound(symbol_id)))

    def references(self, symbol_id: str) -> list[Hit]:
        """Find every in-folder edge of any kind pointing at a symbol."""
        return self._hits(self.graph.references(self._inbound(symbol_id)))

    def implementations(self, type_id: str, *, limit: int | None = None) -> list[Hit]:
        """Find the in-folder direct and transitive subtypes of a type."""
        return self._hits(self.graph.implementations(self._inbound(type_id), limit=limit, prefix=self._lead))

    def supertypes(self, type_id: str, *, limit: int | None = None) -> list[Hit]:
        """Find the in-folder direct and transitive supertypes of a type."""
        return self._hits(self.graph.supertypes(self._inbound(type_id), limit=limit, prefix=self._lead))

    def overrides_of(self, method_id: str) -> list[Hit]:
        """Find the in-folder supertype method a method overrides."""
        return self._hits(self.graph.overrides_of(self._inbound(method_id)))

    def overridden_by(self, method_id: str) -> list[Hit]:
        """Find the in-folder subtype methods that override a method."""
        return self._hits(self.graph.overridden_by(self._inbound(method_id)))

    def tests_of(self, symbol_id: str) -> list[Hit]:
        """Find the in-folder tests covering a symbol."""
        return self._hits(self.graph.tests_of(self._inbound(symbol_id)))

    def neighbors(
        self, symbol_id: str, hops: int = 1, kinds: Sequence[EdgeKind] | None = None, *, limit: int | None = None
    ) -> list[Hit]:
        """Walk through the ancestor, applying the result cap to in-folder symbols during traversal."""
        return self._hits(self.graph.neighbors(self._inbound(symbol_id), hops, kinds, limit=limit, prefix=self._lead))

    def metrics(self, symbol_id: str) -> CallableMetrics | None:
        """Return an in-folder callable's measurements by its view id."""
        return self.graph.metrics(self._inbound(symbol_id))

    def metrics_in_file(self, file_path: str) -> dict[str, CallableMetrics]:
        """Return the measurements of every measured callable in one file under the sub-directory, by view id."""
        return {
            self._strip(symbol_id): metric
            for symbol_id, metric in self.graph.metrics_in_file(self._lead + file_path).items()
        }

    def file_metrics(self, file_path: str) -> FileMetrics | None:
        """Return the line and comment-line counts of a file under the sub-directory."""
        return self.graph.file_metrics(self._lead + file_path)

    def all_metrics(self) -> Iterator[tuple[Symbol, CallableMetrics]]:
        """Yield every measured in-folder callable, rebased, with its measurements."""
        for symbol, metric in self.graph.all_metrics(prefix=self._lead):
            rebased = self._rebased(symbol)
            if rebased is not None:
                yield rebased, metric

    def all_symbols(self) -> Iterator[Symbol]:
        """Yield every in-folder symbol, rebased."""
        for symbol in self.graph.all_symbols(prefix=self._lead):
            rebased = self._rebased(symbol)
            if rebased is not None:
                yield rebased

    def file_measurements(self) -> Iterator[tuple[str, FileMetrics, bool]]:
        """Yield every measured in-folder file, its path rebased."""
        for path, metric, is_test in self.graph.file_measurements(prefix=self._lead):
            yield self._strip(path), metric, is_test

    def inbound_tallies(self, kinds: Sequence[EdgeKind] | None = None) -> list[EdgeTally]:
        """Count the sources reaching each in-folder symbol, wherever those sources live, ids rebased."""
        return [
            replace(tally, symbol_id=self._strip(tally.symbol_id))
            for tally in self.graph.inbound_tallies(kinds, prefix=self._lead)
        ]

    def caller_tallies(self, kinds: Sequence[EdgeKind]) -> Iterator[CallerTally]:
        """Count the callers of each in-folder symbol by container, ids rebased (`_view_id`)."""
        for tally in self.graph.caller_tallies(kinds, prefix=self._lead):
            yield replace(tally, symbol_id=self._strip(tally.symbol_id), owner_id=self._view_id(tally.owner_id))

    def call_tallies(self) -> Iterator[CallTally]:
        """Count the calls of each in-folder symbol per callee, ids rebased (`_view_id`)."""
        for tally in self.graph.call_tallies(prefix=self._lead):
            yield replace(tally, caller_id=self._strip(tally.caller_id), callee_id=self._view_id(tally.callee_id))

    def _view_id(self, value: str) -> str:
        """An ancestor id as the view spells it: rebased inside the folder, its path `../`-relative outside it.

        An id outside the folder never collides with a rebased one, as its file path would if left as it is.
        """
        if value.startswith(self._lead):
            return self._strip(value)
        path, separator, rest = value.partition("#")
        return posixpath.relpath(path, self.prefix) + separator + rest

    def outgoing_edges(self, kinds: Sequence[EdgeKind]) -> list[Edge]:
        """Return the edges leaving in-folder symbols, rebased; a destination outside the folder reads None."""
        return [
            replace(
                edge,
                src_id=self._strip(edge.src_id),
                dst_id=self._strip(edge.dst_id) if edge.dst_id and edge.dst_id.startswith(self._lead) else None,
            )
            for edge in self.graph.outgoing_edges(kinds, prefix=self._lead)
        ]

    def source_files(self) -> list[tuple[str, bool]]:
        """Return every file of the ancestor graph: code outside the folder may still reference code inside it."""
        return self.graph.source_files()

    def unresolved_name_tallies(self, kinds: Sequence[EdgeKind] | None = None) -> dict[tuple[EdgeKind, str], int]:
        """Count the destination-less edges of the whole ancestor graph: names carry no path to filter on."""
        return self.graph.unresolved_name_tallies(kinds)

    def row_counts(self) -> RowCounts:
        """Count the whole ancestor graph: a view's bulk reads tally edges and names across all of it."""
        return self.graph.row_counts()

    def file_dependencies(self, kinds: Sequence[EdgeKind]) -> list[FileDependency]:
        """Count the edges leaving in-folder files; a target outside the folder is spelled `../`-relative to it."""
        return [
            replace(
                dependency,
                source_path=self._strip(dependency.source_path),
                target_path=posixpath.relpath(dependency.target_path, self.prefix),
            )
            for dependency in self.graph.file_dependencies(kinds, prefix=self._lead)
        ]

    def meta(self) -> dict[str, str]:
        """Return the ancestor graph's stored metadata."""
        return self.graph.meta()

    def coverage_note(self) -> str:
        """Explain what the ancestor graph covers, naming the sub-directory it is filtered to."""
        return f"{self.graph.coverage_note()} Answered from the graph of {self.graph.path}, filtered to {self.prefix}/."


#: What `open_provider` hands back: a root's own graph, or an ancestor's filtered to a sub-directory.
AnyProvider = SqliteGraphProvider | SubtreeGraphProvider


def open_provider(path: str) -> AnyProvider:
    """Open the graph that answers for a path: its own, else an ancestor's filtered to it.

    The caller runs `zemble.graph.cli.ensure_graph` first; both route through
    `zemble.graph.store.resolve_graph_root`, so they agree on which graph that is.
    """
    root, prefix = resolve_graph_root(path)
    graph = SqliteGraphProvider(root)
    return graph if prefix is None else SubtreeGraphProvider(graph, prefix, path)

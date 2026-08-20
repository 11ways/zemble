"""The query seam over a symbol graph.

`GraphProvider` is deliberately free of sqlite and tree-sitter types: a later
compiler-grade provider (javac, via zenit-dev) answers the same questions with
better resolution and drops straight in behind it.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from zemble.graph.model import TYPE_KINDS, Edge, EdgeKind, Hit, Resolution, Symbol
from zemble.graph.store import connect, edge_from_row, symbol_from_row

_HIERARCHY_KINDS = (EdgeKind.EXTENDS.value, EdgeKind.IMPLEMENTS.value)

_RESOLUTION_PHRASES = {
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
        """Find declarations matching a simple name, a qualified name or `Type.member`."""
        ...

    def callers(self, symbol_id: str) -> list[Hit]:
        """Find every call site that reaches a callable."""
        ...

    def callees(self, symbol_id: str) -> list[Hit]:
        """Find every callable invoked from a symbol's body."""
        ...

    def references(self, symbol_id: str) -> list[Hit]:
        """Find every edge of any kind pointing at a symbol."""
        ...

    def implementations(self, type_id: str) -> list[Hit]:
        """Find direct and transitive subtypes of a type."""
        ...

    def supertypes(self, type_id: str) -> list[Hit]:
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

    def neighbors(self, symbol_id: str, hops: int = 1, kinds: Sequence[EdgeKind] | None = None) -> list[Hit]:
        """Walk outward from a symbol in both directions."""
        ...


def display_name(symbol: Symbol) -> str:
    """Return a short human label: `Type` for types, `Type.member` for members."""
    if symbol.kind in TYPE_KINDS:
        return symbol.name
    parts = symbol.qualified_name.rsplit(".", 2)
    return ".".join(parts[-2:]) if len(parts) >= 2 else symbol.qualified_name


def _reason(symbol: Symbol, edge: Edge, depth: int = 1, *, outgoing: bool = False) -> str:
    """Build the one-line sentence explaining why a hit is in the answer."""
    table = _VERBS_OUT if outgoing else _VERBS
    verb = table.get(edge.kind, edge.kind.value)
    phrase = _RESOLUTION_PHRASES[edge.resolution]
    if edge.resolution is Resolution.AMBIGUOUS:
        phrase = f"ambiguous, {len(edge.candidates)} candidates"
    depth_note = f", depth {depth}" if depth > 1 else ""
    return f"{verb} {display_name(symbol)} (line {edge.line}, {phrase}{depth_note})"


class SqliteGraphProvider:
    """A `GraphProvider` backed by the sqlite graph built by `zemble.graph.store`."""

    def __init__(self, path: str) -> None:
        """Open the graph database of a workspace path."""
        self.path = path
        self.connection: sqlite3.Connection = connect(path)

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

    def definition(self, name: str) -> list[Symbol]:
        """Find declarations matching a simple name, a qualified name or `Type.member`."""
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
            return bucket, 0 if symbol.kind in TYPE_KINDS else 1, symbol.id

        return sorted(symbols, key=rank)

    # ---- edge queries ---------------------------------------------------

    def _incoming(self, symbol_id: str, kinds: Sequence[str] | None = None) -> list[Edge]:
        """Load edges pointing at a symbol."""
        return self._edges("dst_id", symbol_id, kinds)

    def _outgoing(self, symbol_id: str, kinds: Sequence[str] | None = None) -> list[Edge]:
        """Load resolved edges leaving a symbol."""
        return self._edges("src_id", symbol_id, kinds)

    def _edges(self, column: str, symbol_id: str, kinds: Sequence[str] | None) -> list[Edge]:
        """Load edges on one side of a symbol, optionally filtered by kind."""
        query = f"SELECT * FROM edges WHERE {column} = ?"  # noqa: S608
        params: list[object] = [symbol_id]
        if kinds:
            query += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        if column == "src_id":
            query += " AND dst_id IS NOT NULL"
        return [edge_from_row(row) for row in self.connection.execute(query, params)]

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

    def implementations(self, type_id: str) -> list[Hit]:
        """Find direct and transitive subtypes of a type."""
        return self._walk_hierarchy(type_id, incoming=True)

    def supertypes(self, type_id: str) -> list[Hit]:
        """Find direct and transitive supertypes of a type."""
        return self._walk_hierarchy(type_id, incoming=False)

    def _walk_hierarchy(self, type_id: str, *, incoming: bool) -> list[Hit]:
        """Breadth-first walk of the type hierarchy, recording the depth of each hop."""
        hits: list[Hit] = []
        seen = {type_id}
        frontier = [type_id]
        depth = 0
        while frontier and depth < 32:
            depth += 1
            next_frontier: list[str] = []
            for current in frontier:
                edges = (
                    self._incoming(current, _HIERARCHY_KINDS) if incoming else self._outgoing(current, _HIERARCHY_KINDS)
                )
                for hit in self._hits(edges, side="src" if incoming else "dst", depth=depth):
                    if hit.symbol.id in seen:
                        continue
                    seen.add(hit.symbol.id)
                    hits.append(hit)
                    next_frontier.append(hit.symbol.id)
            frontier = next_frontier
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

    def neighbors(self, symbol_id: str, hops: int = 1, kinds: Sequence[EdgeKind] | None = None) -> list[Hit]:
        """Walk outward from a symbol in both directions."""
        kind_values = [kind.value for kind in kinds] if kinds else None
        hits: list[Hit] = []
        seen = {symbol_id}
        frontier = [symbol_id]
        for depth in range(1, max(1, hops) + 1):
            next_frontier: list[str] = []
            for current in frontier:
                found = self._hits(self._outgoing(current, kind_values), side="dst", depth=depth)
                found += self._hits(self._incoming(current, kind_values), side="src", depth=depth)
                for hit in found:
                    if hit.symbol.id in seen:
                        continue
                    seen.add(hit.symbol.id)
                    hits.append(hit)
                    next_frontier.append(hit.symbol.id)
            frontier = next_frontier
        return hits

    @staticmethod
    def _sorted(hits: list[Hit]) -> list[Hit]:
        """Order hits by file then line so output is stable."""
        return sorted(hits, key=lambda hit: (hit.symbol.file_path, hit.line, hit.symbol.id))

    # ---- coverage --------------------------------------------------------

    def meta(self) -> dict[str, str]:
        """Return the graph's stored metadata."""
        return {row["key"]: row["value"] for row in self.connection.execute("SELECT key, value FROM meta")}

    def coverage_note(self) -> str:
        """Explain what the graph does and does not cover, for empty answers."""
        raw = self.meta().get("skipped_by_language")
        skipped = json.loads(raw) if raw else {}
        if not skipped:
            return "The graph covers Java source only."
        listed = ", ".join(
            f"{language} ({count})" for language, count in sorted(skipped.items(), key=lambda item: -item[1])[:6]
        )
        return f"The graph covers Java source only; no graph extractor for: {listed}."

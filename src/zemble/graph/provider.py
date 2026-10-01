"""The query seam over a symbol graph.

`GraphProvider` is deliberately free of sqlite and tree-sitter types: a later
compiler-grade provider (javac, via zenit-dev) answers the same questions with
better resolution and drops straight in behind it.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace
from itertools import chain
from typing import Protocol, runtime_checkable

from zemble.graph.facts import TREE_SITTER_SOURCE
from zemble.graph.model import NAMED_KINDS, TYPE_KINDS, Edge, EdgeKind, Hit, Resolution, Symbol
from zemble.graph.store import connect, edge_from_row, resolve_graph_root, symbol_from_row

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


def display_name(symbol: Symbol) -> str:
    """Return a short human label: `Type` for types, `Type.member` for members."""
    if symbol.kind in NAMED_KINDS:
        return symbol.name
    parts = symbol.qualified_name.rsplit(".", 2)
    return ".".join(parts[-2:]) if len(parts) >= 2 else symbol.qualified_name


def _reason(symbol: Symbol, edge: Edge, depth: int = 1, *, outgoing: bool = False) -> str:
    """Build the one-line sentence explaining why a hit is in the answer.

    An edge an external tool wrote names that tool instead of a resolution grade: the
    grade is always `exact` there, and which tool said so is the part worth reading. An edge
    that reached its source through a source map also names the generated member it was
    written about, because "this template calls it" is only half the story a reader needs.
    """
    table = _VERBS_OUT if outgoing else _VERBS
    verb = table.get(edge.kind, edge.kind.value)
    phrase = _RESOLUTION_PHRASES[edge.resolution]
    if edge.resolution is Resolution.AMBIGUOUS:
        phrase = f"ambiguous, {len(edge.candidates)} candidates"
    if edge.source != TREE_SITTER_SOURCE:
        phrase = edge.source
    if edge.origin_ref:
        phrase = f"{phrase} via {edge.origin_ref}"
    depth_note = f", depth {depth}" if depth > 1 else ""
    return f"{verb} {display_name(symbol)} (line {edge.line}, {phrase}{depth_note})"


class SqliteGraphProvider:
    """A `GraphProvider` backed by the sqlite graph built by `zemble.graph.store`."""

    def __init__(self, path: str) -> None:
        """Open the graph database of a workspace path."""
        self.path = path
        self.connection: sqlite3.Connection = connect(path)
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
            return bucket, 0 if symbol.kind in NAMED_KINDS else 1, symbol.id

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
        """Find in-folder declarations matching a simple name, a qualified name or `Type.member`."""
        return self._symbols(self.graph.definition(name))

    def callers(self, symbol_id: str) -> list[Hit]:
        """Find every in-folder call site that reaches a callable."""
        return self._hits(self.graph.callers(self._inbound(symbol_id)))

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

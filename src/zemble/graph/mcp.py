"""MCP tools over the Java symbol graph, registered onto an existing FastMCP server."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from zemble.graph.cli import ensure_graph, select_symbol
from zemble.graph.model import EdgeKind, Hit, Symbol
from zemble.graph.provider import SqliteGraphProvider, display_name

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

_REPO_DESCRIPTION = (
    "Local directory path of the workspace to query. The Java symbol graph is built on first use "
    "and refreshed once per server process."
)
_SYMBOL_DESCRIPTION = "A simple name (`PageWindow`), a qualified name, or `Type.member` (`PageWindow.of`)."


def _symbol_json(symbol: Symbol) -> dict[str, Any]:
    """Render a symbol for the wire."""
    return {
        "id": symbol.id,
        "kind": symbol.kind.value,
        "qualified_name": symbol.qualified_name,
        "file_path": symbol.file_path,
        "line": symbol.start_line,
        "signature": symbol.signature,
        "is_test": symbol.is_test,
    }


def _hit_json(hit: Hit) -> dict[str, Any]:
    """Render a hit for the wire."""
    return {
        "qualified_name": hit.symbol.qualified_name,
        "kind": hit.symbol.kind.value,
        "file_path": hit.symbol.file_path,
        "line": hit.line,
        "edge_kind": hit.edge_kind.value,
        "resolution": hit.resolution.value,
        "depth": hit.depth,
        "source": hit.source,
        "reason": hit.reason,
    }


def _open(repo: str) -> SqliteGraphProvider:
    """Build the graph if needed and open a provider on it."""
    ensure_graph(repo)
    return SqliteGraphProvider(repo)


def answer(repo: str, symbol: str, method: str, **kwargs: Any) -> str:
    """Resolve a written name and run one provider query, as JSON."""
    provider = _open(repo)
    try:
        candidates = provider.definition(symbol)
        if method == "definition":
            if not candidates:
                return json.dumps({"error": f"No symbol named {symbol!r}.", "note": provider.coverage_note()})
            return json.dumps({"results": [_symbol_json(found) for found in candidates]})
        chosen, competing = select_symbol(candidates, symbol)
        if chosen is None:
            if not competing:
                return json.dumps({"error": f"No symbol named {symbol!r}.", "note": provider.coverage_note()})
            return json.dumps(
                {
                    "error": f"{symbol!r} is ambiguous; pass a qualified name.",
                    "candidates": [_symbol_json(found) for found in competing],
                }
            )
        hits = getattr(provider, method)(chosen.id, **kwargs)
        payload: dict[str, Any] = {
            "symbol": _symbol_json(chosen),
            "display": display_name(chosen),
            "results": [_hit_json(hit) for hit in hits],
        }
        if not hits:
            payload["note"] = provider.coverage_note()
        return json.dumps(payload)
    finally:
        provider.close()


async def _dispatch(repo: str, symbol: str, method: str, **kwargs: Any) -> str:
    """Answer through the warm daemon when there is one, else in this process.

    The daemon holds a graph it keeps fresh with its watcher, so the workspace scan
    `ensure_graph` would do here is skipped entirely.
    """
    from zemble.daemon import client
    from zemble.daemon.protocol import DaemonError

    args: dict[str, Any] = {"path": repo, "symbol": symbol, "command": method}
    if "hops" in kwargs:
        args["hops"] = kwargs["hops"]
        kinds = kwargs.get("kinds")
        args["kinds"] = [kind.value for kind in kinds] if kinds else None
    try:
        return str(await asyncio.to_thread(client.call, "graph", args))
    except DaemonError:
        logger.debug("Falling back to an in-process graph query for %s", repo, exc_info=True)
    return await asyncio.to_thread(answer, repo, symbol, method, **kwargs)


def register_graph_tools(server: FastMCP) -> None:
    """Register the symbol-graph tools on a FastMCP server."""

    @server.tool()
    async def graph_definition(
        symbol: Annotated[str, Field(description=_SYMBOL_DESCRIPTION)],
        repo: Annotated[str, Field(description=_REPO_DESCRIPTION)],
    ) -> str:
        """Find where a Java symbol is declared, with its exact file, line and signature.

        Use this instead of grepping for `class Foo` or `void bar(`.
        """
        return await _dispatch(repo, symbol, "definition")

    @server.tool()
    async def graph_callers(
        symbol: Annotated[str, Field(description=_SYMBOL_DESCRIPTION)],
        repo: Annotated[str, Field(description=_REPO_DESCRIPTION)],
    ) -> str:
        """List every call site of a Java method or constructor, with a reason per hit.

        Each result says how confidently it was resolved: `exact` means the declaring
        type was pinned down, `unique_name` means only one symbol in the workspace
        carries that name, `ambiguous` means several did.
        """
        return await _dispatch(repo, symbol, "callers")

    @server.tool()
    async def graph_implementations(
        symbol: Annotated[str, Field(description=_SYMBOL_DESCRIPTION)],
        repo: Annotated[str, Field(description=_REPO_DESCRIPTION)],
    ) -> str:
        """List the direct and transitive subtypes of a Java class or interface, with their depth."""
        return await _dispatch(repo, symbol, "implementations")

    @server.tool()
    async def graph_tests_of(
        symbol: Annotated[str, Field(description=_SYMBOL_DESCRIPTION)],
        repo: Annotated[str, Field(description=_REPO_DESCRIPTION)],
    ) -> str:
        """Find the tests covering a Java symbol: naming matches (FooTest) first, then tests that use it."""
        return await _dispatch(repo, symbol, "tests_of")

    @server.tool()
    async def graph_neighbors(
        symbol: Annotated[str, Field(description=_SYMBOL_DESCRIPTION)],
        repo: Annotated[str, Field(description=_REPO_DESCRIPTION)],
        hops: Annotated[int, Field(description="How far to walk outward.", ge=1, le=4)] = 1,
        kinds: Annotated[
            list[str] | None,
            Field(description=f"Only follow these edge kinds: {', '.join(kind.value for kind in EdgeKind)}."),
        ] = None,
    ) -> str:
        """Walk the graph outward from a symbol in both directions, to see what it is wired to."""
        selected = [EdgeKind(value) for value in kinds] if kinds else None
        return await _dispatch(repo, symbol, "neighbors", hops=hops, kinds=selected)

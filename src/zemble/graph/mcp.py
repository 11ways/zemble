"""MCP tools over the symbol graph, registered onto an existing FastMCP server."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from zemble.graph.cli import PROVIDER_METHODS, ensure_graph, select_symbol
from zemble.graph.model import EdgeKind, Hit, Symbol
from zemble.graph.provider import AnyProvider, display_name, open_provider
from zemble.mcp_repo import resolve_repo, with_default_note

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

_REPO_DESCRIPTION = with_default_note(
    "Local directory path of the workspace to query. The symbol graph (every bundled grammar; Java and "
    "Hawkeye templates with compiler-grade lanes) is built on first use "
    "and refreshed once per server process."
)
_SYMBOL_DESCRIPTION = "A simple name (`PageWindow`), a qualified name, or `Type.member` (`PageWindow.of`)."
_LIMIT_DESCRIPTION = "Results to return at most; the payload's `total` says how many exist."

#: Hits returned per graph answer unless the caller raises it; keeps a hot symbol
#: from flooding the client, and the cap is never silent (`total` + `truncated`).
DEFAULT_LIMIT = 50


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


def _open(repo: str, *, fresh: bool = False) -> AnyProvider:
    """Build the graph if needed and open a provider on it."""
    if not fresh:
        ensure_graph(repo)
    return open_provider(repo)


def _capped(items: list[Any], limit: int, render: Any) -> dict[str, Any]:
    """Render at most `limit` items, naming how many exist so a cap is never silent."""
    payload: dict[str, Any] = {"results": [render(item) for item in items[:limit]], "total": len(items)}
    if len(items) > limit:
        payload["truncated"] = f"showing {limit} of {len(items)}; raise `limit` to see the rest"
    return payload


def answer(
    repo: str, symbol: str, method: str, *, limit: int = DEFAULT_LIMIT, fresh: bool = False, **kwargs: Any
) -> dict[str, Any]:
    """Resolve a written name and run one provider query, as a payload object.

    Returned as an object rather than a JSON string: both callers (this module's tools and
    the daemon) hand it straight to a client that would otherwise decode JSON out of JSON.
    `method` arrives over the wire, so it fails closed against the query vocabulary
    instead of reaching getattr on the provider.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    if method not in PROVIDER_METHODS:
        return {"error": f"Unknown graph command {method!r}."}
    provider = _open(repo, fresh=True) if fresh else _open(repo)
    try:
        candidates = provider.definition(symbol)
        if method == "definition":
            if not candidates:
                return {"error": f"No symbol named {symbol!r}.", "note": provider.coverage_note()}
            return _capped(candidates, limit, _symbol_json)
        chosen, competing = select_symbol(candidates, symbol)
        if chosen is None:
            if not competing:
                return {"error": f"No symbol named {symbol!r}.", "note": provider.coverage_note()}
            return {
                "error": f"{symbol!r} is ambiguous; pass a qualified name.",
                "candidates": [_symbol_json(found) for found in competing],
            }
        traversal = method in {"neighbors", "implementations", "supertypes"}
        if traversal:
            # One lookahead hit proves truncation without materializing the rest of the graph.
            kwargs["limit"] = limit + 1
        hits = getattr(provider, method)(chosen.id, **kwargs)
        payload: dict[str, Any] = {
            "symbol": _symbol_json(chosen),
            "display": display_name(chosen),
            **_capped(hits, limit, _hit_json),
        }
        if traversal and provider.traversal_truncated:
            payload["total_exact"] = False
            payload["truncated"] = (
                f"showing {min(limit, len(hits))} of at least {len(hits)}; traversal stopped at its cap; "
                "raise `limit` to explore further"
            )
        if not hits:
            payload["note"] = provider.coverage_note()
        return payload
    finally:
        provider.close()


async def _dispatch(
    repo: str | None, symbol: str, method: str, *, limit: int = DEFAULT_LIMIT, **kwargs: Any
) -> dict[str, Any]:
    """Answer through the warm daemon when there is one, else in this process.

    The daemon holds a graph it keeps fresh with its watcher, so the workspace scan
    `ensure_graph` would do here is skipped entirely.
    """
    from zemble.daemon.protocol import DaemonError, failure_message
    from zemble.mcp import _daemon_call

    repo = resolve_repo(repo)
    args: dict[str, Any] = {"path": repo, "symbol": symbol, "command": method, "limit": limit}
    if "hops" in kwargs:
        args["hops"] = kwargs["hops"]
        kinds = kwargs.get("kinds")
        args["kinds"] = [kind.value for kind in kinds] if kinds else None
    try:
        remote = await _daemon_call("graph", args)
        if isinstance(remote, dict):
            return remote
        if remote is not None:
            return {"error": "Daemon returned an invalid graph payload; no in-process fallback."}
    except DaemonError as error:
        return {"error": failure_message(error)}
    return await asyncio.to_thread(answer, repo, symbol, method, limit=limit, **kwargs)


def register_graph_tools(server: FastMCP) -> None:
    """Register the symbol-graph tools on a FastMCP server."""

    @server.tool(structured_output=False)
    async def graph_definition(
        symbol: Annotated[str, Field(description=_SYMBOL_DESCRIPTION)],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        limit: Annotated[int, Field(description=_LIMIT_DESCRIPTION, ge=1, le=500)] = DEFAULT_LIMIT,
    ) -> dict[str, Any]:
        """Find where a symbol is declared, with its exact file, line and signature.

        Use this instead of grepping for `class Foo` or `void bar(`.
        """
        return await _dispatch(repo, symbol, "definition", limit=limit)

    @server.tool(structured_output=False)
    async def graph_callers(
        symbol: Annotated[str, Field(description=_SYMBOL_DESCRIPTION)],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        limit: Annotated[int, Field(description=_LIMIT_DESCRIPTION, ge=1, le=500)] = DEFAULT_LIMIT,
    ) -> dict[str, Any]:
        """List every call site of a function, method or constructor, with a reason per hit.

        Each result says how confidently it was resolved: `exact` means the declaring
        type was pinned down, `unique_name` means only one symbol in the workspace
        carries that name, `ambiguous` means several did.
        """
        return await _dispatch(repo, symbol, "callers", limit=limit)

    @server.tool(structured_output=False)
    async def graph_implementations(
        symbol: Annotated[str, Field(description=_SYMBOL_DESCRIPTION)],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        limit: Annotated[int, Field(description=_LIMIT_DESCRIPTION, ge=1, le=500)] = DEFAULT_LIMIT,
    ) -> dict[str, Any]:
        """List the direct and transitive subtypes of a class, interface, trait or protocol, with their depth.

        For every override of one method (`Type.member`), use `graph_overrides` instead.
        """
        return await _dispatch(repo, symbol, "implementations", limit=limit)

    @server.tool(structured_output=False)
    async def graph_overrides(
        symbol: Annotated[str, Field(description="The overridden method, as `Type.member` or a qualified name.")],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        limit: Annotated[int, Field(description=_LIMIT_DESCRIPTION, ge=1, le=500)] = DEFAULT_LIMIT,
    ) -> dict[str, Any]:
        """List every subtype method that overrides a method, with its file and line.

        The method-level counterpart of `graph_implementations`: `Shape.area` lists each
        concrete `area()` in the workspace, ready to be read as line spans.
        """
        return await _dispatch(repo, symbol, "overridden_by", limit=limit)

    @server.tool(structured_output=False)
    async def graph_tests_of(
        symbol: Annotated[str, Field(description=_SYMBOL_DESCRIPTION)],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        limit: Annotated[int, Field(description=_LIMIT_DESCRIPTION, ge=1, le=500)] = DEFAULT_LIMIT,
    ) -> dict[str, Any]:
        """Find the tests covering a symbol: naming matches (FooTest, test_foo) first, then tests that use it."""
        return await _dispatch(repo, symbol, "tests_of", limit=limit)

    @server.tool(structured_output=False)
    async def graph_neighbors(
        symbol: Annotated[str, Field(description=_SYMBOL_DESCRIPTION)],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        hops: Annotated[int, Field(description="How far to walk outward.", ge=1, le=4)] = 1,
        kinds: Annotated[
            list[str] | None,
            Field(description=f"Only follow these edge kinds: {', '.join(kind.value for kind in EdgeKind)}."),
        ] = None,
        limit: Annotated[int, Field(description=_LIMIT_DESCRIPTION, ge=1, le=500)] = DEFAULT_LIMIT,
    ) -> dict[str, Any]:
        """Walk the graph outward from a symbol in both directions, to see what it is wired to."""
        selected = [EdgeKind(value) for value in kinds] if kinds else None
        return await _dispatch(repo, symbol, "neighbors", limit=limit, hops=hops, kinds=selected)

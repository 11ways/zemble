"""MCP tools for evidence bundles, outlines and signatures."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from zemble.evidence.answers import explain_payload, outline_payload, signatures_payload
from zemble.graph.cli import ensure_graph
from zemble.graph.provider import SqliteGraphProvider
from zemble.index import ZembleIndex
from zemble.types import ContentType

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

IndexGetter = Callable[[str, Sequence[ContentType]], Awaitable[ZembleIndex]]

DEFAULT_MCP_BUDGET = 2500

# The vocabulary is derived from ContentType rather than spelled out again; the value
# is resolved by `zemble.mcp._resolve_content_selection`, which refuses an unknown one.
_CONTENT_DESCRIPTION = (
    f"Content to search: {', '.join(kind.value for kind in ContentType)} or all. "
    "Defaults to the content the server was configured with."
)

_REPO_DESCRIPTION = (
    "Local directory path of the workspace. Both the code index and the Java symbol graph are built "
    "on first use and refreshed once per server process."
)


def _open(repo: str) -> SqliteGraphProvider:
    """Build the graph if needed and open a provider on it."""
    ensure_graph(repo)
    return SqliteGraphProvider(repo)


async def _remote(cmd: str, args: dict[str, Any]) -> dict[str, Any] | None:
    """Ask the warm daemon for one evidence payload, or return None to answer here.

    Imported lazily: `zemble.mcp` imports this module, so the reverse import can only
    run once the server is being built.
    """
    from zemble.mcp import _daemon_call

    return await _daemon_call(cmd, args)


def _explain_here(index: ZembleIndex, repo: str, query: str, budget: int, top_k: int) -> dict[str, Any]:
    """Build an evidence bundle in this process."""
    provider = _open(repo)
    try:
        return explain_payload(index, provider, query, budget, top_k)
    finally:
        provider.close()


def _outline_here(repo: str, target: str, members: str | None) -> dict[str, Any]:
    """Outline a file or a type in this process."""
    provider = _open(repo)
    try:
        return outline_payload(provider, target, members)
    finally:
        provider.close()


def _signatures_here(repo: str, symbol: str) -> dict[str, Any]:
    """Describe a symbol and its exact callers in this process."""
    provider = _open(repo)
    try:
        return signatures_payload(provider, symbol)
    finally:
        provider.close()


def _as_json(payload: dict[str, Any], key: str) -> str:
    """Render an outline or signatures payload the way the tool's callers expect it."""
    if "error" in payload:
        return json.dumps(
            {"error": payload["error"], "candidates": [c["qualified_name"] for c in payload["candidates"]]}
        )
    return json.dumps(payload[key])


def register_evidence_tools(
    server: FastMCP, get_index: IndexGetter, default_content: Sequence[ContentType] = (ContentType.CODE,)
) -> None:
    """Register the evidence tools on a FastMCP server.

    :param server: The server to register on.
    :param get_index: Awaitable that returns the code index for a repo and content selection.
    :param default_content: Content types the server was configured with.
    """

    @server.tool()
    async def explain(
        query: Annotated[str, Field(description="What you want explained, in natural language or as a symbol name.")],
        repo: Annotated[str, Field(description=_REPO_DESCRIPTION)],
        budget: Annotated[
            int,
            Field(description="Token budget for the whole bundle.", ge=200, le=50_000),
        ] = DEFAULT_MCP_BUDGET,
        top_k: Annotated[int, Field(description="Search results to expand.", ge=1, le=50)] = 20,
        content: Annotated[
            str | None,
            Field(description=_CONTENT_DESCRIPTION),
        ] = None,
    ) -> str:
        """Get a budgeted evidence bundle instead of reading whole files.

        Searches, then follows the Java symbol graph one hop out of what it found:
        the enclosing type's outline, the tests that cover it, its callers, its
        callees and any sibling documentation. Every item says why it is there, and
        anything that did not fit is listed as a location so you know it exists.
        Prefer this over `search` plus several `Read` calls.
        """
        # Imported here: `zemble.mcp` imports this module, so the reverse import can
        # only run once the server is being built.
        from zemble.mcp import _resolve_content_selection

        selected = _resolve_content_selection(content, default_content)
        payload = await _remote(
            "explain",
            {
                "path": repo,
                "query": query,
                "budget": budget,
                "top_k": top_k,
                "content": [item.value for item in selected],
            },
        )
        if payload is None:
            index = await get_index(repo, selected)
            payload = await asyncio.to_thread(_explain_here, index, repo, query, budget, top_k)
        if not payload["bundle"]["items"]:
            return f"No evidence found for {query!r}."
        return str(payload["markdown"])

    @server.tool()
    async def outline(
        target: Annotated[
            str, Field(description="A workspace-relative file path, or a simple or qualified type name.")
        ],
        repo: Annotated[str, Field(description=_REPO_DESCRIPTION)],
        members: Annotated[str | None, Field(description="Only show members whose name matches this pattern.")] = None,
    ) -> str:
        """List what a Java file or type declares, signatures only, for a few hundred tokens.

        Use this before reading a file: it shows every member with its line range, so
        the next read can be a line span instead of a whole file.
        """
        payload = await _remote("outline", {"path": repo, "target": target, "members": members})
        if payload is None:
            payload = await asyncio.to_thread(_outline_here, repo, target, members)
        return _as_json(payload, "outline")

    @server.tool()
    async def signatures(
        symbol: Annotated[str, Field(description="A simple name, a qualified name, or `Type.member`.")],
        repo: Annotated[str, Field(description=_REPO_DESCRIPTION)],
    ) -> str:
        """Show a Java symbol's signature and the call sites the graph resolved exactly.

        Cheaper than `graph_callers` when all you need is whether something is used
        and from where; weaker resolutions are counted rather than listed.
        """
        payload = await _remote("signatures", {"path": repo, "symbol": symbol})
        if payload is None:
            payload = await asyncio.to_thread(_signatures_here, repo, symbol)
        return _as_json(payload, "signatures")


__all__ = ["DEFAULT_MCP_BUDGET", "register_evidence_tools"]

"""MCP tools for evidence bundles, outlines and signatures."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Annotated

from pydantic import Field

from zemble.evidence.bundle import build_bundle
from zemble.evidence.outline import OutlineError, outline, signatures
from zemble.graph.cli import ensure_graph, select_symbol
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


def _bundle_markdown(index: ZembleIndex, repo: str, query: str, budget: int, top_k: int) -> str:
    """Build a bundle and render it as markdown, with its cost stated."""
    provider = _open(repo)
    try:
        bundle = build_bundle(index, provider, query, budget, top_k=top_k)
    finally:
        provider.close()
    if not bundle.items:
        return f"No evidence found for {query!r}."
    return bundle.render()


def _outline_json(repo: str, target: str, members: str | None) -> str:
    """Render an outline as JSON."""
    provider = _open(repo)
    try:
        return json.dumps(outline(provider, target, members).to_dict())
    except OutlineError as error:
        return json.dumps({"error": error.message, "candidates": [s.qualified_name for s in error.candidates]})
    finally:
        provider.close()


def _signatures_json(repo: str, symbol: str) -> str:
    """Render a symbol's signature and exact callers as JSON."""
    provider = _open(repo)
    try:
        chosen, candidates = select_symbol(provider.definition(symbol), symbol)
        if chosen is None:
            message = (
                f"{symbol!r} is ambiguous; pass a qualified name."
                if candidates
                else f"No symbol named {symbol!r}. {provider.coverage_note()}"
            )
            return json.dumps({"error": message, "candidates": [s.qualified_name for s in candidates]})
        return json.dumps(signatures(provider, chosen).to_dict())
    finally:
        provider.close()


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

        index = await get_index(repo, _resolve_content_selection(content, default_content))
        return await asyncio.to_thread(_bundle_markdown, index, repo, query, budget, top_k)

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
        return await asyncio.to_thread(_outline_json, repo, target, members)

    @server.tool()
    async def signatures(
        symbol: Annotated[str, Field(description="A simple name, a qualified name, or `Type.member`.")],
        repo: Annotated[str, Field(description=_REPO_DESCRIPTION)],
    ) -> str:
        """Show a Java symbol's signature and the call sites the graph resolved exactly.

        Cheaper than `graph_callers` when all you need is whether something is used
        and from where; weaker resolutions are counted rather than listed.
        """
        return await asyncio.to_thread(_signatures_json, repo, symbol)


__all__ = ["DEFAULT_MCP_BUDGET", "register_evidence_tools"]

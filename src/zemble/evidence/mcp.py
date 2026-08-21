"""MCP tools for evidence bundles, outlines and signatures."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from zemble.daemon.protocol import CommandRefused
from zemble.evidence.answers import explain_payload, outline_payload, signatures_payload
from zemble.graph.cli import ensure_graph
from zemble.graph.provider import SqliteGraphProvider
from zemble.index import ZembleIndex
from zemble.types import ContentType

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

IndexGetter = Callable[..., Awaitable[ZembleIndex]]

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

# The two filters mean the same thing here as on `search`, worded for evidence bundles.
# They cannot import `zemble.mcp`'s wording: that module imports this one to register the tools.
_PATHS_DESCRIPTION = "Only draw evidence from these sub-paths of `repo` (repo-relative)."
_EXCLUDE_DESCRIPTION = (
    "Gitignore-style patterns, relative to `repo`, dropped from the search that seeds the bundle; "
    "on a repo with no index yet they also prune the walk that builds it."
)


def _open(repo: str) -> SqliteGraphProvider:
    """Build the graph if needed and open a provider on it."""
    ensure_graph(repo)
    return SqliteGraphProvider(repo)


async def _remote(cmd: str, args: dict[str, Any]) -> dict[str, Any] | None:
    """Ask the warm daemon for one evidence payload, or return None to answer here.

    Imported lazily: `zemble.mcp` imports this module, so the reverse import can only
    run once the server is being built. A deliberate refusal propagates as
    :class:`CommandRefused`: answering it in this process would refuse identically.
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


def _as_payload(payload: dict[str, Any], key: str) -> dict[str, Any]:
    """Return an outline or signatures payload as an object, never as a JSON string."""
    if "error" in payload:
        return {"error": payload["error"], "candidates": [c["qualified_name"] for c in payload["candidates"]]}
    return payload[key]


async def _explain_answer(
    get_index: IndexGetter,
    repo: str,
    query: str,
    budget: int,
    top_k: int,
    selected: Sequence[ContentType],
    paths: Sequence[str],
    exclude: Sequence[str],
) -> str:
    """Answer one `explain` call from the daemon, or here, and render it as markdown.

    A refusal - too broad a root, too big a build - is returned as its own text: it is the
    answer, and retrying it in this process would only pay for the same refusal again.
    """
    try:
        payload = await _remote(
            "explain",
            {
                "path": repo,
                "query": query,
                "budget": budget,
                "top_k": top_k,
                "content": [item.value for item in selected],
                "paths": list(paths),
                "exclude": list(exclude),
            },
        )
        if payload is None:
            index = await get_index(repo, selected, paths, exclude)
            payload = await asyncio.to_thread(_explain_here, index, repo, query, budget, top_k)
    except (CommandRefused, ValueError) as exc:
        return str(exc)
    if not payload["bundle"]["items"]:
        return f"No evidence found for {query!r}."
    return str(payload["markdown"])


def register_evidence_tools(
    server: FastMCP, get_index: IndexGetter, default_content: Sequence[ContentType] = (ContentType.CODE,)
) -> None:
    """Register the evidence tools on a FastMCP server.

    :param server: The server to register on.
    :param get_index: Awaitable that returns the code index for a repo and content selection.
    :param default_content: Content types the server was configured with.
    """

    @server.tool(structured_output=False)
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
        paths: Annotated[list[str] | None, Field(description=_PATHS_DESCRIPTION)] = None,
        exclude: Annotated[list[str] | None, Field(description=_EXCLUDE_DESCRIPTION)] = None,
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
        return await _explain_answer(
            get_index, repo, query, budget, top_k, selected, tuple(paths or ()), tuple(exclude or ())
        )

    @server.tool(structured_output=False)
    async def outline(
        target: Annotated[
            str, Field(description="A workspace-relative file path, or a simple or qualified type name.")
        ],
        repo: Annotated[str, Field(description=_REPO_DESCRIPTION)],
        members: Annotated[str | None, Field(description="Only show members whose name matches this pattern.")] = None,
    ) -> dict[str, Any]:
        """List what a Java file or type declares, signatures only, for a few hundred tokens.

        Use this before reading a file: it shows every member with its line range, so
        the next read can be a line span instead of a whole file.
        """
        try:
            payload = await _remote("outline", {"path": repo, "target": target, "members": members})
        except CommandRefused as exc:
            return {"error": str(exc)}
        if payload is None:
            payload = await asyncio.to_thread(_outline_here, repo, target, members)
        return _as_payload(payload, "outline")

    @server.tool(structured_output=False)
    async def signatures(
        symbol: Annotated[str, Field(description="A simple name, a qualified name, or `Type.member`.")],
        repo: Annotated[str, Field(description=_REPO_DESCRIPTION)],
    ) -> dict[str, Any]:
        """Show a Java symbol's signature and the call sites the graph resolved exactly.

        Cheaper than `graph_callers` when all you need is whether something is used
        and from where; weaker resolutions are counted rather than listed.
        """
        try:
            payload = await _remote("signatures", {"path": repo, "symbol": symbol})
        except CommandRefused as exc:
            return {"error": str(exc)}
        if payload is None:
            payload = await asyncio.to_thread(_signatures_here, repo, symbol)
        return _as_payload(payload, "signatures")


__all__ = ["DEFAULT_MCP_BUDGET", "register_evidence_tools"]

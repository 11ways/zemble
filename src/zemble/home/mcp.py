"""The MCP tool behind `zemble home`."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from zemble.graph.cli import ensure_graph
from zemble.graph.provider import SqliteGraphProvider
from zemble.home.answers import DEFAULT_TOP_K, home_payload
from zemble.home.cli import HOME_CONTENT
from zemble.home.config import ConfigError, HomeConfig
from zemble.index import ZembleIndex
from zemble.types import ContentType

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

IndexGetter = Callable[[str, Sequence[ContentType]], Awaitable[ZembleIndex]]

_REPO_DESCRIPTION = (
    "Local directory path of the workspace. Both the code index and the Java symbol graph are built "
    "on first use and refreshed once per server process."
)


def _here(index: ZembleIndex, repo: str, description: str, top_k: int) -> dict[str, Any]:
    """Answer in this process, over a freshly opened graph."""
    config = HomeConfig.load(repo)
    ensure_graph(repo)
    provider = SqliteGraphProvider(repo)
    try:
        return home_payload(index, provider, config, description, top_k)
    finally:
        provider.close()


def register_home_tool(server: FastMCP, get_index: IndexGetter) -> None:
    """Register the `home` tool on a FastMCP server.

    :param server: The server to register on.
    :param get_index: Awaitable that returns the index for a repo and content selection.
    """

    @server.tool()
    async def home(
        description: Annotated[str, Field(description="The feature you are about to build, in your own words.")],
        repo: Annotated[str, Field(description=_REPO_DESCRIPTION)],
        top_k: Annotated[int, Field(description="Code results to weigh.", ge=1, le=100)] = DEFAULT_TOP_K,
    ) -> str:
        """Check whether a capability already exists and which module should own it.

        Call this BEFORE designing a new mechanism. It reports the existing
        mechanisms that look like the description and who consumes them, the modules
        that could host it ranked with reasons, a verdict (extend what exists, build
        it in a named module, or genuinely uncertain), and the workspace's own rules,
        forbidden dependencies and skills for the modules involved.
        """
        # Imported here: `zemble.mcp` imports this module, so the reverse import can
        # only run once the server is being built.
        from zemble.mcp import _daemon_call

        args = {
            "path": repo,
            "description": description,
            "top_k": top_k,
            "content": [item.value for item in HOME_CONTENT],
        }
        try:
            payload = await _daemon_call("home", args)
            if payload is None:
                index = await get_index(repo, HOME_CONTENT)
                payload = await asyncio.to_thread(_here, index, repo, description, top_k)
        except ConfigError as error:
            return str(error)
        return str(payload["markdown"])


__all__ = ["register_home_tool"]

"""The MCP tool behind `zemble home`."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from zemble.daemon.protocol import CommandRefused
from zemble.graph.cli import ensure_graph
from zemble.graph.provider import SqliteGraphProvider
from zemble.home.answers import DEFAULT_TOP_K, home_payload
from zemble.home.cli import HOME_CONTENT
from zemble.home.config import ConfigError, HomeConfig
from zemble.index import ZembleIndex
from zemble.mcp_repo import resolve_repo, with_default_note
from zemble.types import ContentType
from zemble.workspace import resolve_home_root

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

IndexGetter = Callable[[str, Sequence[ContentType]], Awaitable[ZembleIndex]]

_REPO_DESCRIPTION = with_default_note(
    "Local directory path; home queries expand to the nearest ancestor declaring .zemble/home.toml. "
    "The answer identifies that workspace, and all evidence paths are relative to it. "
    "Both the code index and the Java symbol graph are built "
    "on first use and refreshed once per server process."
)


def _here(index: ZembleIndex, repo: str, description: str, top_k: int, requested: str) -> dict[str, Any]:
    """Answer in this process, over a freshly opened graph."""
    config = HomeConfig.load(repo)
    ensure_graph(repo)
    provider = SqliteGraphProvider(repo)
    try:
        return home_payload(index, provider, config, description, top_k, requested_root=requested)
    finally:
        provider.close()


def register_home_tool(server: FastMCP, get_index: IndexGetter) -> None:
    """Register the `home` tool on a FastMCP server.

    :param server: The server to register on.
    :param get_index: Awaitable that returns the index for a repo and content selection.
    """

    @server.tool(structured_output=False)
    async def home(
        description: Annotated[str, Field(description="The feature you are about to build, in your own words.")],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
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

        requested = resolve_repo(repo)
        repo = str(resolve_home_root(requested))
        args = {
            "path": repo,
            "requested_path": requested,
            "description": description,
            "top_k": top_k,
            "content": [item.value for item in HOME_CONTENT],
        }
        try:
            HomeConfig.load(repo)
            payload = await _daemon_call("home", args)
            if payload is None:
                index = await get_index(repo, HOME_CONTENT)
                payload = await asyncio.to_thread(_here, index, repo, description, top_k, requested)
        # A refusal is the answer, on both lanes: the daemon raises CommandRefused and the
        # in-process one arrives as the ValueError `zemble.mcp._get_index` wraps a refusal in.
        # An agent has to be able to read the reason and act on it, not a stack trace.
        except (ConfigError, CommandRefused, ValueError) as error:
            return str(error)
        return str(payload["markdown"])


__all__ = ["register_home_tool"]

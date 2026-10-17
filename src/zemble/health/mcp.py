"""The MCP tool behind `zemble health`."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Annotated

from pydantic import Field

from zemble.daemon.protocol import DaemonError, failure_message
from zemble.health.cli import in_process
from zemble.health.model import Signal
from zemble.health.report import DEFAULT_LIMIT
from zemble.home.config import ConfigError
from zemble.mcp_repo import resolve_repo, with_default_note
from zemble.nib.parts import nib_text_only
from zemble.utils import is_git_url

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

_REPO_DESCRIPTION = with_default_note(
    "Local directory path to report on; a sub-directory of a workspace with a graph is answered from "
    "that graph, paths relative to it. The symbol graph is built on first use."
)


def register_health_tool(server: FastMCP) -> None:
    """Register the `health` tool on a FastMCP server."""
    nib_text_only(server, "health")

    @server.tool(structured_output=False)
    async def health(
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        limit: Annotated[int, Field(description="Findings to list.", ge=1, le=500)] = DEFAULT_LIMIT,
        signals: Annotated[
            list[str] | None,
            Field(description=f"Signals to measure (default: all): {', '.join(signal.value for signal in Signal)}."),
        ] = None,
        paths: Annotated[
            list[str] | None,
            Field(description="Restrict the report to these sub-paths of `repo` (relative, or absolute)."),
        ] = None,
    ) -> str:
        """Report code-shape findings ranked by how extreme they are in this repo, plus a score.

        Size (long, deep, complex callables; large types and files), machinery (single-implementation
        interfaces, single-subclass abstract types, public callables with at most one caller,
        unreferenced private code) and layering (edges crossing a `.zemble/home.toml` forbidden
        dependency). Thresholds are absolute and documented; graph findings rest only on edges
        resolved exactly or by unique name. The score is not a target: address or explain the
        findings. This is a report, never a gate.
        """
        # Imported here: `zemble.mcp` imports this module, so the reverse import can
        # only run once the server is being built.
        from zemble.mcp import _daemon_call

        if is_git_url(resolve_repo(repo)):
            return "health reads a local directory only; clone the repository and pass its path."
        repo = os.path.abspath(resolve_repo(repo))
        args = {"path": repo, "signals": signals, "paths": paths or [], "limit": limit}
        try:
            payload = await _daemon_call("health", args)
            if payload is None:
                payload = await asyncio.to_thread(in_process, repo, signals, paths or [], limit)
        except DaemonError as error:
            return failure_message(error)
        except (ConfigError, ValueError) as error:
            return str(error)
        return str(payload["markdown"])


__all__ = ["register_health_tool"]

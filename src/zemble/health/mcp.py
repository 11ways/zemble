"""The MCP tool behind `zemble health`."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Annotated

from pydantic import Field

from zemble.daemon.protocol import CommandOutgrown, DaemonError, failure_message
from zemble.health.answers import HealthRequest
from zemble.health.baseline import BASELINE_RELATIVE_PATH
from zemble.health.cli import in_process
from zemble.health.history import DEFAULT_WINDOW_DAYS
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
        since: Annotated[
            str | None,
            Field(
                description="A git ref: list only the findings touching a line changed since it (uncommitted and "
                "untracked files included); the score stays the whole scope's."
            ),
        ] = None,
        copying: Annotated[
            bool | None,
            Field(
                description="Measure the duplication tier (clone_mass, cloned_body, pass_through, vocabulary_drift); "
                "default: on with `since`, `paths` or a sub-directory, off over a whole tree."
            ),
        ] = None,
        history: Annotated[
            bool,
            Field(description="Measure the history axis (hotspot, boundary_co_change) from the git log."),
        ] = True,
        history_window: Annotated[
            int, Field(description="Days of git history the history axis reads.", ge=1)
        ] = DEFAULT_WINDOW_DAYS,
        baseline: Annotated[
            bool,
            Field(
                description=f"Diff this run against `<repo>/{BASELINE_RELATIVE_PATH}`: "
                "resolved / remaining / new instead of the flat list."
            ),
        ] = False,
        save_baseline: Annotated[
            bool,
            Field(
                description=f"Write this run's finding keys to `<repo>/{BASELINE_RELATIVE_PATH}` "
                "(after the diff, so one call can diff and advance the baseline; not with `since`)."
            ),
        ] = False,
    ) -> str:
        """Report code-shape findings ranked by how extreme they are in this repo, plus a score.

        Size (long, deep, complex callables; large types and files), machinery (single-implementation
        interfaces, single-subclass abstract types, public callables with at most one caller,
        unreferenced private code, pass-through methods), layering (edges crossing a `.zemble/home.toml`
        forbidden dependency), copying (files and bodies inside exact or renamed clone classes),
        vocabulary (values declared in several places), consistency (files whose style reads different
        from their language's or their directory's other files) and history (complex files that keep changing,
        files in different modules that change together, from the git log). Thresholds are absolute and
        documented; graph findings rest only on edges resolved exactly or by unique name. `since` turns it
        into a branch delta. The score is not a target: address or explain the findings. This is a report,
        never a gate.
        """
        # Imported here: `zemble.mcp` imports this module, so the reverse import can
        # only run once the server is being built.
        from zemble.mcp import _daemon_call

        if is_git_url(resolve_repo(repo)):
            return "health reads a local directory only; clone the repository and pass its path."
        repo = os.path.abspath(resolve_repo(repo))
        fixed = os.path.join(repo, BASELINE_RELATIVE_PATH)
        request = HealthRequest(
            signals=tuple(signals) if signals is not None else None,
            paths=tuple(paths or ()),
            limit=limit,
            since=since,
            copying=copying,
            history=history,
            history_window=history_window,
            baseline=fixed if baseline else None,
            save_baseline=fixed if save_baseline else None,
            baseline_optional=True,
        )
        note = ""
        try:
            try:
                payload = await _daemon_call("health", request.to_args(repo))
            except CommandOutgrown as outgrown:
                note = f"Note: {outgrown} Answered in the MCP server's process from the stored graph.\n\n"
                payload = None
            if payload is None:
                payload = await asyncio.to_thread(in_process, repo, request)
        except DaemonError as error:
            return failure_message(error)
        except (ConfigError, ValueError) as error:
            return str(error)
        return note + str(payload["markdown"])


__all__ = ["register_health_tool"]

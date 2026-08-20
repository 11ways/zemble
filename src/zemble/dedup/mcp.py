"""The `dupes` MCP tool, registered onto an existing FastMCP server."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Annotated

from pydantic import Field

from zemble.dedup.detect import DupeOptions, find_duplication
from zemble.dedup.model import CloneKind
from zemble.dedup.report import report_json

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

_REPO_DESCRIPTION = "Local directory path of the workspace to scan for duplicated Java code."


def _run(repo: str, kind: str, paths: list[str] | None, limit: int, min_files: int) -> str:
    """Run one duplication scan and render it as JSON."""
    kinds = tuple(CloneKind) if kind == "all" else (CloneKind(kind),)
    options = DupeOptions(kinds=kinds, min_files=min_files, paths=tuple(paths or ()))
    report = find_duplication(repo, options)
    return json.dumps(report_json(report, limit))


def register_dupes_tool(server: FastMCP) -> None:
    """Register the duplication tool on a FastMCP server."""

    @server.tool()
    async def dupes(
        repo: Annotated[str, Field(description=_REPO_DESCRIPTION)],
        kind: Annotated[
            str,
            Field(description="Which duplication to report: exact, renamed, logic, or all."),
        ] = "renamed",
        paths: Annotated[
            list[str] | None, Field(description="Restrict the scan to these paths inside the workspace.")
        ] = None,
        limit: Annotated[int, Field(description="Clone classes returned per kind.", ge=1, le=200)] = 25,
        min_files: Annotated[int, Field(description="Only report classes spanning at least N files.", ge=1)] = 1,
    ) -> str:
        """Report duplicated Java code as clone classes ranked by weight.

        `exact` matches token streams with comments and whitespace removed, `renamed` also
        normalizes locals, parameters and lambda parameters (a differing literal or field name
        never matches), and `logic` reports embedding candidates that passed a control-flow and
        call-set check, with the reason stated per pair. This is a report, never a gate.
        """
        return await asyncio.to_thread(_run, repo, kind, paths, limit, min_files)

"""The `dupes` MCP tool, registered onto an existing FastMCP server."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field

from zemble.dedup.detect import DupeOptions, find_duplication
from zemble.dedup.model import CloneKind, Lane
from zemble.dedup.report import format_report, report_json

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

_REPO_DESCRIPTION = "Local directory path of the workspace to scan for duplicated Java code."

DupeFormat = Literal["text", "json"]


def _options(kind: str, lane: str, paths: list[str] | None, exclude: list[str] | None, min_files: int) -> DupeOptions:
    """Build the run options from the tool's arguments."""
    kinds = tuple(CloneKind) if kind == "all" else (CloneKind(kind),)
    return DupeOptions(
        kinds=kinds,
        min_files=min_files,
        paths=tuple(paths or ()),
        exclude=tuple(exclude or ()),
        lane=None if lane == "all" else Lane(lane),
    )


def _run(options: DupeOptions, repo: str, limit: int, output: DupeFormat, brief: bool) -> str | dict[str, Any]:
    """Run one duplication scan and render it the way the caller asked for.

    The JSON form is returned as an object, never as a string: a tool that hands back
    `json.dumps(...)` from a `-> str` signature makes the client parse JSON out of JSON.
    """
    report = find_duplication(repo, options)
    if output == "json":
        return report_json(report, limit)
    return format_report(report, limit, brief=brief)


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
        exclude: Annotated[
            list[str] | None,
            Field(description="Gitignore-style patterns, relative to the root, dropped before parsing."),
        ] = None,
        lane: Annotated[
            str,
            Field(description="Report one lane only: production, mixed, test, or all."),
        ] = "all",
        limit: Annotated[int, Field(description="Clone classes returned per section.", ge=1, le=200)] = 25,
        min_files: Annotated[int, Field(description="Only report classes spanning at least N files.", ge=1)] = 1,
        format: Annotated[
            DupeFormat,
            Field(description="`text` returns the report as the CLI prints it; `json` returns a structured object."),
        ] = "text",
        brief: Annotated[
            bool, Field(description="Text format only: one line per class, no member paths and no reasons.")
        ] = False,
    ) -> str | dict[str, Any]:
        """Report duplicated Java code as clone classes ranked by weight.

        `exact` matches token streams with comments and whitespace removed, `renamed` also
        normalizes locals, parameters and lambda parameters (a differing literal or field name
        never matches), and `logic` reports embedding candidates that passed a control-flow and
        call-set check, with the reason stated per pair. Classes are sectioned by lane, so test
        scaffolding can never outrank production duplication. This is a report, never a gate.
        """
        options = _options(kind, lane, paths, exclude, min_files)
        return await asyncio.to_thread(_run, options, repo, limit, format, brief)

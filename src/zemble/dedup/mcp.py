"""The `dupes` MCP tool, registered onto an existing FastMCP server."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field

from zemble.dedup.baseline import BASELINE_RELATIVE_PATH, load_baseline, save_baseline
from zemble.dedup.detect import DupeOptions, find_duplication
from zemble.dedup.languages import supported_extensions, supported_languages
from zemble.dedup.model import CloneKind, Lane
from zemble.dedup.report import baseline_diff_json, format_baseline_diff, format_report, report_json

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

_LANGUAGES = ", ".join(supported_languages())
_EXTENSIONS = ", ".join(supported_extensions())
_REPO_DESCRIPTION = f"Local directory path of the workspace to scan for duplicated code ({_LANGUAGES}; {_EXTENSIONS})."

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


def _run(
    options: DupeOptions,
    repo: str,
    limit: int,
    output: DupeFormat,
    brief: bool,
    against_baseline: bool,
    write_baseline: bool,
) -> str | dict[str, Any]:
    """Run one duplication scan and render it the way the caller asked for.

    The JSON form is returned as an object, never as a string: a tool that hands back
    `json.dumps(...)` from a `-> str` signature makes the client parse JSON out of JSON.
    The baseline lives at the fixed `<repo>/.zemble/dupes.baseline.json`; diffing loads it
    before `save_baseline` overwrites it, so one call can both diff and advance it.
    """
    report = find_duplication(repo, options)
    baseline_path = Path(report.root) / BASELINE_RELATIVE_PATH
    baseline = None
    notes: list[str] = []
    if against_baseline:
        try:
            baseline = load_baseline(baseline_path)
        except ValueError as error:
            if baseline_path.is_file():
                notes.append(f"baseline: {error}")
            else:
                notes.append(f"baseline: none at {BASELINE_RELATIVE_PATH}; pass save_baseline=true to write one")
    if write_baseline:
        save_baseline(baseline_path, report)
        notes.append(f"baseline: wrote {len(report.classes)} class key(s) to {BASELINE_RELATIVE_PATH}")
    if output == "json":
        payload = baseline_diff_json(report, baseline, limit) if baseline else report_json(report, limit)
        if notes:
            payload["baseline_notes"] = notes
        return payload
    if baseline:
        text = format_baseline_diff(report, baseline, limit=limit)
    else:
        text = format_report(report, limit, brief=brief)
    return text + "".join(f"{note}\n" for note in notes)


def register_dupes_tool(server: FastMCP) -> None:
    """Register the duplication tool on a FastMCP server."""

    @server.tool(structured_output=False)
    async def dupes(
        repo: Annotated[str, Field(description=_REPO_DESCRIPTION)],
        kind: Annotated[
            str,
            Field(description="Which duplication to report: exact, renamed, logic, or all."),
        ] = "renamed",
        paths: Annotated[
            list[str] | None,
            Field(description="Restrict the scan to these paths, relative to the workspace root (or absolute)."),
        ] = None,
        exclude: Annotated[
            list[str] | None,
            Field(description="Gitignore-style patterns, relative to the workspace root, dropped before parsing."),
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
        baseline: Annotated[
            bool,
            Field(
                description="Diff this run against `<repo>/.zemble/dupes.baseline.json`: "
                "resolved / changed / remaining / new instead of the flat report."
            ),
        ] = False,
        save_baseline: Annotated[
            bool,
            Field(
                description="Write this run's clone class keys to `<repo>/.zemble/dupes.baseline.json` "
                "(after the diff, so one call can diff and advance the baseline)."
            ),
        ] = False,
    ) -> str | dict[str, Any]:
        """Report duplicated code as clone classes ranked by weight.

        Every supported language is scanned in one pass (the `repo` argument names them). A run
        that walked no supported file says so instead of reporting a clean workspace, and files
        that failed to parse are counted and named rather than silently dropped.
        `exact` matches token streams with comments and whitespace removed, `renamed` also
        normalizes locals, parameters and lambda parameters (a differing literal or field name
        never matches), and `logic` reports embedding candidates that passed a control-flow and
        call-set check, with the reason stated per pair. Classes are sectioned by lane, so test
        scaffolding can never outrank production duplication. Classes spanning modules declared
        in `.zemble/home.toml` carry a home verdict. This is a report, never a gate.
        """
        options = _options(kind, lane, paths, exclude, min_files)
        return await asyncio.to_thread(_run, options, repo, limit, format, brief, baseline, save_baseline)

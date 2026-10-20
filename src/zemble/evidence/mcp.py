"""MCP tools for evidence bundles, outlines and signatures."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from zemble.daemon.protocol import DaemonError, failure_message
from zemble.evidence.answers import explain_payload, outline_payload, signatures_payload
from zemble.evidence.bundle import bundle_header, item_heading, omitted_lines, split_truncation
from zemble.evidence.outline import (
    ambiguous_line,
    callers_heading,
    kind_label,
    no_member_line,
    omitted_line,
    outline_header,
)
from zemble.graph.cli import ensure_graph
from zemble.graph.mcp import CallSite, call_site_parts
from zemble.graph.provider import AnyProvider, open_provider
from zemble.index import ZembleIndex
from zemble.mcp_repo import resolve_repo, with_default_note
from zemble.nib.parts import Part, file_lines, location_part, nib_answer, source_part, text_part
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

_REPO_DESCRIPTION = with_default_note(
    "Local directory path of the workspace. Both the code index and the symbol graph are built "
    "on first use and refreshed once per server process."
)

# The two filters mean the same thing here as on `search`, worded for evidence bundles.
# They cannot import `zemble.mcp`'s wording: that module imports this one to register the tools.
_PATHS_DESCRIPTION = "Only draw evidence from these sub-paths of `repo` (repo-relative)."
_EXCLUDE_DESCRIPTION = (
    "Gitignore-style patterns, relative to `repo`, dropped from the search that seeds the bundle; "
    "on a repo with no index yet they also prune the walk that builds it."
)


def _open(repo: str) -> AnyProvider:
    """Build the graph if needed and open a provider on it."""
    ensure_graph(repo)
    return open_provider(repo)


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


def _outline_here(repo: str, target: str, members: str | None, everything: bool) -> dict[str, Any]:
    """Outline a file or a type in this process."""
    provider = _open(repo)
    try:
        return outline_payload(provider, target, members, everything)
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
) -> str | dict[str, Any]:
    """Answer one `explain` call from the daemon, or here, as its payload or as the text that replaces it.

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
    except DaemonError as exc:
        return failure_message(exc)
    except ValueError as exc:
        return str(exc)
    if not payload["bundle"]["items"]:
        return f"No evidence found for {query!r}."
    return payload


def explain_parts(root: str, payload: dict[str, Any]) -> list[Part]:
    """Lay out an evidence bundle for nib: each item's heading as prose, then the lines it shows.

    An item whose text is zemble's own rendering (an outline) or a bare location stays prose.
    """
    bundle = payload["bundle"]
    header = bundle_header(
        bundle["query"],
        bundle["intent"],
        bundle["intent_rule"],
        bundle["order"],
        len(bundle["items"]),
        bundle["total_tokens"],
    )
    parts = [text_part("\n".join(header).rstrip("\n"))]
    for item in bundle["items"]:
        heading = item_heading(item["file_path"], item["start_line"], item["end_line"], item["reason"])
        text = item["text"]
        if item["presentation"] == "location":
            parts.append(text_part(f"{heading}\n`{text}`" if text else f"{heading}\n(location only)"))
            continue
        if "verbatim" not in item:
            # A daemon older than the field: whether the text is file lines is unknown, so it is not shown.
            parts.append(text_part(heading))
            continue
        if not item["verbatim"]:
            parts.append(text_part(f"{heading}\n{text}"))
            continue
        kept, dropped = split_truncation(text)
        if dropped:
            heading += f"\n(showing {len(kept.splitlines())} lines; {dropped} more not shown)"
        parts.append(text_part(heading))
        shown = source_part(root, item["file_path"], item["start_line"], kept)
        if shown is not None:
            parts.append(shown)
    if bundle["omitted"]:
        parts.append(text_part("\n".join(omitted_lines(bundle["omitted"], bundle["unlisted_omissions"])).rstrip("\n")))
    return parts


#: Lines past a declaration's first one searched for the line naming it.
_DECLARATION_SCAN = 20

#: The annotations or decorators a line opens with (`@Override`, `@Named("x")`, `@app.route("/")`).
_LEADING_ANNOTATIONS = re.compile(r"^\s*(?:@[\w.$]+(?:\([^()]*\))?\s*)*")


def declaration_line(lines: Sequence[str] | None, start: int, end: int, name: str) -> int:
    """The line that names a declaration: past the annotations or decorators it starts with, else `start`.

    AIDEV-NOTE: the graph records where a declaration starts, which for an annotated Java
    method is its first `@Annotation` line; a row of that line alone would not say what it
    declares. The name is looked for after a line's leading annotations, so `@Named("run")`
    does not count as naming `run`, while `@NonNull Foo run()` does.
    """
    if not lines or not name:
        return start
    named = re.compile(rf"(?<![\w$]){re.escape(name)}(?![\w$])")
    last = min(end, start + _DECLARATION_SCAN, len(lines))
    for number in range(start, last + 1):
        text = lines[number - 1]
        if named.search(text, _LEADING_ANNOTATIONS.match(text).end()):  # type: ignore[union-attr]
            return number
    return start


def outline_parts(root: str, payload: dict[str, Any]) -> list[Part] | None:
    """Lay out an outline for nib: its header, every declaration's naming line as one block, then what it left out.

    Only one line per declaration is located: the graph does not record where a
    signature ends, and a guessed span could cut or overrun it.
    """
    if "entries" not in payload:
        return None
    parts = [text_part("\n".join(outline_header(payload["package"], payload["file_path"])))]
    lines = file_lines(location_part(root, payload["file_path"], 1, 1)["path"])
    for entry in payload["entries"]:
        line = declaration_line(lines, entry["start_line"], entry["end_line"], entry["name"])
        parts.append(location_part(root, payload["file_path"], line, line))
    if payload["members_filter"] and not payload["entries"]:
        parts.append(text_part(no_member_line(payload["members_filter"])))
    # A daemon older than the field lists everything and sends no counts.
    omitted = payload.get("omitted") or {}
    note = omitted_line(omitted.get("private", 0), omitted.get("fields", 0))
    if note:
        parts.append(text_part(note))
    return parts


def signatures_parts(root: str, payload: dict[str, Any]) -> list[Part] | None:
    """Lay out a signatures answer for nib: the declaration's line, then its exact call sites by file and caller."""
    if "callers" not in payload:
        return None
    label = kind_label(payload["kind"], payload["signature"] or payload["qualified_name"])
    sites = [CallSite(caller["file_path"], caller["line"], caller["caller"]) for caller in payload["callers"]]
    parts = [
        text_part(label),
        location_part(root, payload["file_path"], payload["start_line"], payload["start_line"]),
        text_part(callers_heading(len(payload["callers"]))),
    ]
    parts += call_site_parts(root, sites)
    if payload["ambiguous_callers"]:
        parts.append(text_part(ambiguous_line(payload["ambiguous_callers"])))
    return parts


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
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
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
    ) -> Any:
        """Get a budgeted evidence bundle instead of reading whole files.

        Searches, then follows the symbol graph one hop out of what it found:
        the enclosing type's outline, the tests that cover it, its callers, its
        callees and any sibling documentation. Every item says why it is there, and
        anything that did not fit is listed as a location so you know it exists.
        Prefer this over `search` plus several `Read` calls.
        """
        # Imported here: `zemble.mcp` imports this module, so the reverse import can
        # only run once the server is being built.
        from zemble.mcp import _resolve_content_selection

        repo = resolve_repo(repo)
        selected = _resolve_content_selection(content, default_content)
        answer = await _explain_answer(
            get_index, repo, query, budget, top_k, selected, tuple(paths or ()), tuple(exclude or ())
        )
        if isinstance(answer, str):
            return nib_answer(server, repo, answer)
        return nib_answer(server, repo, str(answer["markdown"]), lambda root, _: explain_parts(root, answer))

    @server.tool(structured_output=False)
    async def outline(
        target: Annotated[
            str, Field(description="A workspace-relative file path, or a simple or qualified type name.")
        ],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        members: Annotated[
            str | None,
            Field(description="Only show members whose name matches this pattern, private ones and fields included."),
        ] = None,
        all: Annotated[  # noqa: A002 - the wire name callers write
            bool, Field(description="List private members and fields too, instead of counting them.")
        ] = False,
    ) -> Any:
        """List the API a file, type or module declares, one declaration line per member.

        Use this before reading a file: each member's line number lets the next read be a
        line span instead of a whole file. Private members and fields are counted, not
        listed, unless `all` is set or `members` names them.
        """
        repo = resolve_repo(repo)
        try:
            payload = await _remote("outline", {"path": repo, "target": target, "members": members, "all": all})
        except DaemonError as exc:
            return nib_answer(server, repo, {"error": failure_message(exc)})
        if payload is None:
            payload = await asyncio.to_thread(_outline_here, repo, target, members, all)
        if "error" in payload:
            return nib_answer(server, repo, _as_payload(payload, "outline"))
        found = payload["outline"]
        return nib_answer(server, repo, payload["text"], lambda root, _: outline_parts(root, found))

    @server.tool(structured_output=False)
    async def signatures(
        symbol: Annotated[
            str,
            Field(
                description=(
                    "A simple name, a qualified name, or `Type.member`; `Type.member(Param)` picks one overload."
                )
            ),
        ],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
    ) -> Any:
        """Show a symbol's signature and the call sites the graph resolved exactly.

        Cheaper than `graph_callers` when all you need is whether something is used
        and from where; weaker resolutions are counted rather than listed.
        """
        repo = resolve_repo(repo)
        try:
            payload = await _remote("signatures", {"path": repo, "symbol": symbol})
        except DaemonError as exc:
            return nib_answer(server, repo, {"error": failure_message(exc)})
        if payload is None:
            payload = await asyncio.to_thread(_signatures_here, repo, symbol)
        return nib_answer(server, repo, _as_payload(payload, "signatures"), signatures_parts)


__all__ = ["DEFAULT_MCP_BUDGET", "register_evidence_tools"]

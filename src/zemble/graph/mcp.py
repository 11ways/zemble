"""MCP tools over the symbol graph, registered onto an existing FastMCP server."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from zemble.graph.cli import PROVIDER_METHODS, ensure_graph, select_symbol
from zemble.graph.facts import TREE_SITTER_SOURCE
from zemble.graph.model import EdgeKind, Hit, Resolution, Symbol
from zemble.graph.provider import RESOLUTION_PHRASES, AnyProvider, display_name, open_provider
from zemble.mcp_repo import resolve_repo, with_default_note
from zemble.nib.parts import Part, PartsBuilder, location_part, nib_answer, text_part

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

_REPO_DESCRIPTION = with_default_note(
    "Local directory path of the workspace to query. The symbol graph (every bundled grammar; Java and "
    "Hawkeye templates with compiler-grade lanes) is built on first use "
    "and refreshed once per server process."
)
_SYMBOL_DESCRIPTION = "A simple name (`PageWindow`), a qualified name, or `Type.member` (`PageWindow.of`)."
_LIMIT_DESCRIPTION = "Results to return at most; the payload's `total` says how many exist."

#: Hits returned per graph answer unless the caller raises it; keeps a hot symbol
#: from flooding the client, and the cap is never silent (`total` + `truncated`).
DEFAULT_LIMIT = 50

#: Call sites a callers answer shows the line of; the rest are named by file, caller and line.
SHOWN_SITES = 10


def _symbol_json(symbol: Symbol) -> dict[str, Any]:
    """Render a symbol for the wire."""
    return {
        "id": symbol.id,
        "kind": symbol.kind.value,
        "qualified_name": symbol.qualified_name,
        "file_path": symbol.file_path,
        "line": symbol.start_line,
        "signature": symbol.signature,
        "is_test": symbol.is_test,
    }


def _hit_json(hit: Hit) -> dict[str, Any]:
    """Render a hit for the wire."""
    return {
        "qualified_name": hit.symbol.qualified_name,
        "display": display_name(hit.symbol),
        "kind": hit.symbol.kind.value,
        "file_path": hit.symbol.file_path,
        "line": hit.line,
        "edge_kind": hit.edge_kind.value,
        "resolution": hit.resolution.value,
        "grade": hit.grade,
        "depth": hit.depth,
        "source": hit.source,
        "reason": hit.reason,
    }


def _open(repo: str, *, fresh: bool = False) -> AnyProvider:
    """Build the graph if needed and open a provider on it."""
    if not fresh:
        ensure_graph(repo)
    return open_provider(repo)


def _capped(items: list[Any], limit: int, render: Any) -> dict[str, Any]:
    """Render at most `limit` items, naming how many exist so a cap is never silent."""
    payload: dict[str, Any] = {"results": [render(item) for item in items[:limit]], "total": len(items)}
    if len(items) > limit:
        payload["truncated"] = f"showing {limit} of {len(items)}; raise `limit` to see the rest"
    return payload


def answer(
    repo: str, symbol: str, method: str, *, limit: int = DEFAULT_LIMIT, fresh: bool = False, **kwargs: Any
) -> dict[str, Any]:
    """Resolve a written name and run one provider query, as a payload object.

    Returned as an object rather than a JSON string: both callers (this module's tools and
    the daemon) hand it straight to a client that would otherwise decode JSON out of JSON.
    `method` arrives over the wire, so it fails closed against the query vocabulary
    instead of reaching getattr on the provider.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    if method not in PROVIDER_METHODS:
        return {"error": f"Unknown graph command {method!r}."}
    provider = _open(repo, fresh=True) if fresh else _open(repo)
    try:
        candidates = provider.definition(symbol)
        if method == "definition":
            if not candidates:
                return {"error": f"No symbol named {symbol!r}.", "note": provider.coverage_note()}
            return _capped(candidates, limit, _symbol_json)
        chosen, competing = select_symbol(candidates, symbol)
        if chosen is None:
            if not competing:
                return {"error": f"No symbol named {symbol!r}.", "note": provider.coverage_note()}
            return {
                "error": f"{symbol!r} is ambiguous; pass a qualified name.",
                "candidates": [_symbol_json(found) for found in competing],
            }
        traversal = method in {"neighbors", "implementations", "supertypes"}
        if traversal:
            # One lookahead hit proves truncation without materializing the rest of the graph.
            kwargs["limit"] = limit + 1
        hits = getattr(provider, method)(chosen.id, **kwargs)
        payload: dict[str, Any] = {
            "symbol": _symbol_json(chosen),
            "display": display_name(chosen),
            **_capped(hits, limit, _hit_json),
        }
        if traversal and provider.traversal_truncated:
            payload["total_exact"] = False
            payload["truncated"] = (
                f"showing {min(limit, len(hits))} of at least {len(hits)}; traversal stopped at its cap; "
                "raise `limit` to explore further"
            )
        if not hits:
            payload["note"] = provider.coverage_note()
        return payload
    finally:
        provider.close()


def _line_part(root: str, entry: dict[str, Any]) -> Part | None:
    """Locate the one line a symbol or hit points at, when it has a file and a line."""
    if not entry.get("file_path") or not entry.get("line"):
        return None
    return location_part(root, entry["file_path"], entry["line"], entry["line"])


def _error_parts(payload: dict[str, Any]) -> list[Part]:
    """Lay out a refusal as prose: the message, then one line per competing symbol."""
    lines = [str(payload["error"])]
    candidates = payload.get("candidates") or []
    if candidates:
        lines.append(f"{len(candidates)} candidate(s):")
        lines += [
            f"  {found['kind']} {found['qualified_name']}  {found['file_path']}:{found['line']}" for found in candidates
        ]
    if payload.get("note"):
        lines.append(str(payload["note"]))
    return [text_part("\n".join(lines))]


def _queried_parts(root: str, payload: dict[str, Any]) -> list[Part]:
    """Name the queried symbol and locate its declaration line, when the answer has one."""
    symbol = payload.get("symbol")
    if symbol is None:
        return []
    label = f"{payload.get('display', symbol['qualified_name'])}: {symbol['kind']} {symbol['qualified_name']}"
    parts = [text_part(label)]
    located = _line_part(root, symbol)
    return parts + ([located] if located is not None else [])


def _footer_part(payload: dict[str, Any]) -> Part:
    """State the exact total, and the cap or coverage note when there is one."""
    footer = [f"total: {payload['total']}"]
    footer += [str(payload[key]) for key in ("truncated", "note") if payload.get(key)]
    return text_part("\n".join(footer))


def graph_parts(root: str, payload: Any) -> list[Part] | None:
    """Lay out a graph answer for nib: the queried symbol, then each hit's prose and its one line.

    Only the line is located, never a guessed span: the payload names no signature's end.
    """
    if isinstance(payload, dict) and "error" in payload:
        return _error_parts(payload)
    if not isinstance(payload, dict) or "results" not in payload:
        return None
    parts = _queried_parts(root, payload)
    for entry in payload["results"]:
        if "edge_kind" in entry:
            prose = (
                f"{entry['qualified_name']} ({entry['kind']}): {entry['edge_kind']}, {entry['resolution']}, "
                f"depth {entry['depth']}, {entry['source']}: {entry['reason']}"
            )
        else:
            test = ", test" if entry.get("is_test") else ""
            prose = f"{entry['qualified_name']} ({entry['kind']}{test}): {entry.get('signature') or ''}".rstrip()
        parts.append(text_part(prose))
        located = _line_part(root, entry)
        if located is not None:
            parts.append(located)
    parts.append(_footer_part(payload))
    return parts


@dataclass(frozen=True)
class CallSite:
    """One call site as a callers answer names it."""

    file_path: str
    line: int
    caller: str
    # How the call was resolved, when it was not an exact match.
    note: str = ""


def _by_file(sites: Sequence[CallSite]) -> list[tuple[str, list[CallSite]]]:
    """Group call sites by file, files and sites in the order they first appear."""
    grouped: dict[str, list[CallSite]] = {}
    for site in sites:
        grouped.setdefault(site.file_path, []).append(site)
    return list(grouped.items())


def _file_line(file_path: str, sites: Sequence[CallSite]) -> str:
    """Name one file's call sites by caller: `path: Caller L12, L30; Other L5 (ambiguous, 2 candidates)`."""
    by_caller: dict[str, list[CallSite]] = {}
    for site in sites:
        by_caller.setdefault(site.caller, []).append(site)
    callers = [
        f"{caller} " + ", ".join(f"L{site.line}" + (f" ({site.note})" if site.note else "") for site in found)
        for caller, found in by_caller.items()
    ]
    return f"{file_path}: " + "; ".join(callers)


def call_site_parts(root: str, sites: Sequence[CallSite], shown: int = SHOWN_SITES) -> list[Part]:
    """Lay out call sites grouped by file and caller: the first `shown` with their lines, the rest by name only.

    AIDEV-NOTE: one prose line per file, then that file's lines with no prose between them,
    so nib prints each file's lines as one block under one header instead of one per site.
    """
    parts: list[Part] = []
    for file_path, group in _by_file(sites[:shown]):
        parts.append(text_part(_file_line(file_path, group)))
        for line in sorted({site.line for site in group}):
            parts.append(location_part(root, file_path, line, line))
    rest = sites[shown:]
    if rest:
        listed = [f"{len(rest)} more call site(s), lines not shown:"]
        listed += [_file_line(file_path, group) for file_path, group in _by_file(rest)]
        parts.append(text_part("\n".join(listed)))
    return parts


def call_sites_heading(sites: Sequence[CallSite], results: Sequence[dict[str, Any]]) -> str:
    """Count call sites, callers and files; tally resolutions when any is not exact, and the tools that wrote edges."""
    callers = len({(site.file_path, site.caller) for site in sites})
    files = len({site.file_path for site in sites})
    heading = f"{len(sites)} call site(s) from {callers} caller(s) in {files} file(s)"
    resolutions = Counter(str(entry["resolution"]) for entry in results)
    tools = Counter(
        str(entry["source"]) for entry in results if entry.get("source", TREE_SITTER_SOURCE) != TREE_SITTER_SOURCE
    )
    tally = [f"{resolution} {count}" for resolution, count in resolutions.items()]
    if set(resolutions) == {Resolution.EXACT.value}:
        tally = []
    tally += [f"{count} from {tool}" for tool, count in tools.items()]
    return heading + (f" ({', '.join(tally)})" if tally else "")


def _hit_site(entry: dict[str, Any]) -> CallSite:
    """Read one callers hit as a call site; a daemon older than `grade` and `display` gets the plainer fields.

    An exact edge needs no note, whether tree-sitter matched it or a tool wrote it: the heading
    counts the tools. Anything else (a weaker grade, an edge reached through a generated member)
    keeps its wording.
    """
    exact = entry["resolution"] == Resolution.EXACT.value
    grade = entry.get("grade") or ("" if exact else entry["resolution"])
    if exact and grade in {RESOLUTION_PHRASES[Resolution.EXACT], entry.get("source")}:
        grade = ""
    return CallSite(
        file_path=entry["file_path"],
        line=entry["line"],
        caller=entry.get("display") or entry["qualified_name"],
        note=grade,
    )


def callers_parts(root: str, payload: Any) -> list[Part] | None:
    """Lay out a callers answer for nib: the queried symbol, a count, then its call sites grouped by file and caller."""
    if isinstance(payload, dict) and "error" in payload:
        return _error_parts(payload)
    if not isinstance(payload, dict) or "results" not in payload:
        return None
    results = payload["results"]
    sites = [_hit_site(entry) for entry in results]
    parts = _queried_parts(root, payload)
    if sites:
        heading = call_sites_heading(sites, results)
        if len(sites) > SHOWN_SITES:
            heading += f"; the lines of the first {SHOWN_SITES}"
        parts.append(text_part(heading))
        parts += call_site_parts(root, sites)
    parts.append(_footer_part(payload))
    return parts


def callers_text(payload: Any) -> Any:
    """Render a callers answer as plain text, every listed call site by file and caller; a refusal stays as it is."""
    if not isinstance(payload, dict) or "results" not in payload or "symbol" not in payload:
        return payload
    symbol = payload["symbol"]
    results = payload["results"]
    sites = [_hit_site(entry) for entry in results]
    lines = [
        f"{payload.get('display', symbol['qualified_name'])}: {symbol['kind']} {symbol['qualified_name']}"
        f"  {symbol['file_path']}:{symbol['line']}"
    ]
    if sites:
        lines.append(call_sites_heading(sites, results))
        lines += [_file_line(file_path, group) for file_path, group in _by_file(sites)]
    lines.append(_footer_part(payload)["text"])
    return "\n".join(lines)


async def _dispatch(
    repo: str | None, symbol: str, method: str, *, limit: int = DEFAULT_LIMIT, **kwargs: Any
) -> dict[str, Any]:
    """Answer through the warm daemon when there is one, else in this process.

    The daemon holds a graph it keeps fresh with its watcher, so the workspace scan
    `ensure_graph` would do here is skipped entirely.
    """
    from zemble.daemon.protocol import DaemonError, failure_message
    from zemble.mcp import _daemon_call

    repo = resolve_repo(repo)
    args: dict[str, Any] = {"path": repo, "symbol": symbol, "command": method, "limit": limit}
    if "hops" in kwargs:
        args["hops"] = kwargs["hops"]
        kinds = kwargs.get("kinds")
        args["kinds"] = [kind.value for kind in kinds] if kinds else None
    try:
        remote = await _daemon_call("graph", args)
        if isinstance(remote, dict):
            return remote
        if remote is not None:
            return {"error": "Daemon returned an invalid graph payload; no in-process fallback."}
    except DaemonError as error:
        return {"error": failure_message(error)}
    return await asyncio.to_thread(answer, repo, symbol, method, limit=limit, **kwargs)


def register_graph_tools(server: FastMCP) -> None:
    """Register the symbol-graph tools on a FastMCP server."""

    async def _answer(
        repo: str | None,
        symbol: str,
        method: str,
        build: PartsBuilder = graph_parts,
        render: Callable[[Any], Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Answer one graph query, as `render` words it, with nib's parts when the server answers for nib."""
        resolved = resolve_repo(repo)
        payload = await _dispatch(resolved, symbol, method, **kwargs)
        plain = payload if render is None else render(payload)
        return nib_answer(server, resolved, plain, lambda root, _: build(root, payload))

    @server.tool(structured_output=False)
    async def graph_definition(
        symbol: Annotated[str, Field(description=_SYMBOL_DESCRIPTION)],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        limit: Annotated[int, Field(description=_LIMIT_DESCRIPTION, ge=1, le=500)] = DEFAULT_LIMIT,
    ) -> Any:
        """Find where a symbol is declared, with its exact file, line and signature.

        Use this instead of grepping for `class Foo` or `void bar(`.
        """
        return await _answer(repo, symbol, "definition", limit=limit)

    @server.tool(structured_output=False)
    async def graph_callers(
        symbol: Annotated[str, Field(description=_SYMBOL_DESCRIPTION)],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        limit: Annotated[int, Field(description=_LIMIT_DESCRIPTION, ge=1, le=500)] = DEFAULT_LIMIT,
    ) -> Any:
        """List every call site of a function, method or constructor, grouped by file and caller.

        Each result says how confidently it was resolved: `exact` means the declaring
        type was pinned down, `unique_name` means only one symbol in the workspace
        carries that name, `ambiguous` means several did. `total` is always exact.
        """
        return await _answer(repo, symbol, "callers", build=callers_parts, render=callers_text, limit=limit)

    @server.tool(structured_output=False)
    async def graph_implementations(
        symbol: Annotated[str, Field(description=_SYMBOL_DESCRIPTION)],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        limit: Annotated[int, Field(description=_LIMIT_DESCRIPTION, ge=1, le=500)] = DEFAULT_LIMIT,
    ) -> Any:
        """List the direct and transitive subtypes of a class, interface, trait or protocol, with their depth.

        For every override of one method (`Type.member`), use `graph_overrides` instead.
        """
        return await _answer(repo, symbol, "implementations", limit=limit)

    @server.tool(structured_output=False)
    async def graph_overrides(
        symbol: Annotated[str, Field(description="The overridden method, as `Type.member` or a qualified name.")],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        limit: Annotated[int, Field(description=_LIMIT_DESCRIPTION, ge=1, le=500)] = DEFAULT_LIMIT,
    ) -> Any:
        """List every subtype method that overrides a method, with its file and line.

        The method-level counterpart of `graph_implementations`: `Shape.area` lists each
        concrete `area()` in the workspace, ready to be read as line spans.
        """
        return await _answer(repo, symbol, "overridden_by", limit=limit)

    @server.tool(structured_output=False)
    async def graph_tests_of(
        symbol: Annotated[str, Field(description=_SYMBOL_DESCRIPTION)],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        limit: Annotated[int, Field(description=_LIMIT_DESCRIPTION, ge=1, le=500)] = DEFAULT_LIMIT,
    ) -> Any:
        """Find the tests covering a symbol: naming matches (FooTest, test_foo) first, then tests that use it."""
        return await _answer(repo, symbol, "tests_of", limit=limit)

    @server.tool(structured_output=False)
    async def graph_neighbors(
        symbol: Annotated[str, Field(description=_SYMBOL_DESCRIPTION)],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        hops: Annotated[int, Field(description="How far to walk outward.", ge=1, le=4)] = 1,
        kinds: Annotated[
            list[str] | None,
            Field(description=f"Only follow these edge kinds: {', '.join(kind.value for kind in EdgeKind)}."),
        ] = None,
        limit: Annotated[int, Field(description=_LIMIT_DESCRIPTION, ge=1, le=500)] = DEFAULT_LIMIT,
    ) -> Any:
        """Walk the graph outward from a symbol in both directions, to see what it is wired to."""
        selected = [EdgeKind(value) for value in kinds] if kinds else None
        return await _answer(repo, symbol, "neighbors", limit=limit, hops=hops, kinds=selected)

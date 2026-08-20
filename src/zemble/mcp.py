from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from zemble.daemon import client as daemon_client
from zemble.daemon.protocol import DaemonError
from zemble.dedup.mcp import register_dupes_tool
from zemble.evidence.mcp import register_evidence_tools
from zemble.graph.mcp import register_graph_tools
from zemble.index import ZembleIndex
from zemble.index_cache import CACHE_MAX_SIZE, IndexCache
from zemble.types import ContentType
from zemble.utils import format_results, is_git_url, resolve_chunk

logger = logging.getLogger(__name__)

_REPO_DESCRIPTION = (
    "A local directory path or https:// or http:// git URL (e.g. https://github.com/org/repo) to index and "
    "search. The index is cached after the first call, so repeat queries are fast."
)

ContentSelection = Literal["code", "docs", "config", "all"]

#: Re-exported so `zemble.mcp` keeps naming the cache size it serves with.
_CACHE_MAX_SIZE = CACHE_MAX_SIZE


async def _daemon_call(cmd: str, args: dict[str, Any]) -> Any | None:
    """Try to answer one tool call from the warm daemon.

    Several agent sessions then share one RAM copy of a workspace index instead of
    each holding its own.

    :param cmd: Daemon command name.
    :param args: Command arguments.
    :return: The daemon's result, or None if this process must answer in-process.
    """
    try:
        return await asyncio.to_thread(daemon_client.call, cmd, args)
    except DaemonError as exc:
        logger.info("daemon unavailable (%s); answering in-process", exc)
        return None


def unsafe_repo_reason(repo: str) -> str | None:
    """Return why a repo argument is refused, or None when it is acceptable.

    Checked before the daemon is contacted as well: a rejected transport scheme must
    never become a clone inside another process.
    """
    if is_git_url(repo) and not repo.startswith(("https://", "http://")):
        return f"Only https://, http://, or local directory paths are accepted as `repo`. Got: {repo!r}"
    return None


async def _answer_remotely(cmd: str, repo: str, args: dict[str, Any]) -> str | None:
    """Refuse an unacceptable repo, or answer one tool call from the warm daemon.

    :param cmd: Daemon command name.
    :param repo: The repo argument as the caller wrote it.
    :param args: The rest of the command arguments.
    :return: The text to return from the tool, or None when this process must answer.
    """
    refusal = unsafe_repo_reason(repo)
    if refusal is not None:
        return refusal
    remote = await _daemon_call(cmd, {"path": repo, **args})
    return None if remote is None else json.dumps(remote)


async def _get_index(repo: str, cache: IndexCache, content: Sequence[ContentType]) -> ZembleIndex:
    """Return a cached index for a repo, rejecting unsafe git transport schemes."""
    reason = unsafe_repo_reason(repo)
    if reason is not None:
        raise ValueError(reason)
    try:
        return await cache.get(repo, content=content)
    except Exception as exc:
        raise ValueError(f"Failed to index {repo!r}: {exc}") from exc


def _resolve_content_selection(
    content: ContentSelection | None, default_content: Sequence[ContentType]
) -> tuple[ContentType, ...]:
    """Resolve an MCP content selection to exact index content types."""
    if content is None:
        return tuple(default_content)
    if content == "all":
        return tuple(ContentType)
    return (ContentType(content),)


def create_server(cache: IndexCache, default_content: Sequence[ContentType] = (ContentType.CODE,)) -> FastMCP:
    """Build and return a configured FastMCP server backed by the given cache."""
    server = FastMCP(
        "zemble",
        instructions=(
            "Instant code search for any local or remote git repository. "
            "Call `search` once with a focused query, it returns the file path and exact line. "
            "Navigate directly to that file at the given line; do not grep for the same content. "
            "Use `find_related` to discover similar code elsewhere in the same repo. "
            "When working in a local project, pass the project root as `repo`. "
            "For remote repos, pass an explicit https:// URL. Never guess or infer URLs."
        ),
    )

    @server.tool()
    async def search(
        query: Annotated[str, Field(description="Natural language or code query.")],
        repo: Annotated[str, Field(description=_REPO_DESCRIPTION)],
        top_k: Annotated[int, Field(description="Number of results to return.", ge=1)] = 5,
        max_snippet_lines: Annotated[
            int | None,
            Field(
                description=(
                    "Lines of source to include per result. "
                    "Default (10): function/class signature + first body lines, enough to confirm the location. "
                    "0: file path and line range only. None: full chunk (~10-20 lines). "
                    "If the snippet does not contain enough context to confirm you have the right location, "
                    "call again with max_snippet_lines=None."
                ),
                ge=0,
            ),
        ] = 10,
        content: Annotated[
            ContentSelection | None,
            Field(description="Content to search. Defaults to the MCP server's configured content."),
        ] = None,
    ) -> str:
        """Search once with a focused query describing what the code does or its name.

        Write queries using function/class names or behavior descriptions, not error messages.
        Returns file paths and line numbers — navigate directly there, do not repeat the search.
        Pass a git URL or local path as `repo`; indexes are cached for the session.
        """
        selected_content = _resolve_content_selection(content, default_content)
        remote = await _answer_remotely(
            "search",
            repo,
            {
                "query": query,
                "top_k": top_k,
                "max_snippet_lines": max_snippet_lines,
                "content": [item.value for item in selected_content],
            },
        )
        if remote is not None:
            return remote
        try:
            index = await _get_index(repo, cache, selected_content)
        except ValueError as exc:
            return str(exc)
        results = index.search(query, top_k=top_k, max_snippet_lines=max_snippet_lines)
        if not results:
            return json.dumps({"error": "No results found."})
        return json.dumps(format_results(query, results, max_snippet_lines))

    @server.tool()
    async def find_related(
        file_path: Annotated[
            str,
            Field(description="Path to the file as stored in the index (use file_path from a search result)."),
        ],
        line: Annotated[int, Field(description="Line number (1-indexed).")],
        repo: Annotated[str, Field(description=_REPO_DESCRIPTION)],
        top_k: Annotated[int, Field(description="Number of similar chunks to return.", ge=1)] = 5,
        max_snippet_lines: Annotated[
            int | None,
            Field(
                description=(
                    "Lines of source per result. "
                    "Default 10 = signature + first body lines. 0 = location only. None = full chunk."
                ),
                ge=0,
            ),
        ] = 10,
        content: Annotated[
            ContentSelection | None,
            Field(description="Content containing the related file. Defaults to the MCP server configuration."),
        ] = None,
    ) -> str:
        """Find code similar to a known location.

        Useful for discovering all implementations of an interface, all callers of a function,
        or all tests for a class. Use after `search` when you need related code beyond the primary result.
        Pass `file_path` and `line` from a prior search result.
        """
        selected_content = _resolve_content_selection(content, default_content)
        remote = await _answer_remotely(
            "find_related",
            repo,
            {
                "file_path": file_path,
                "line": line,
                "top_k": top_k,
                "max_snippet_lines": max_snippet_lines,
                "content": [item.value for item in selected_content],
            },
        )
        if remote is not None:
            return remote
        try:
            index = await _get_index(repo, cache, selected_content)
        except ValueError as exc:
            return str(exc)
        chunk = resolve_chunk(index.chunks, file_path, line)
        if chunk is None:
            return (
                f"No chunk found at {file_path}:{line}. "
                "Make sure the file is indexed and the line number is within a known chunk."
            )
        results = index.find_related(chunk, top_k=top_k, max_snippet_lines=max_snippet_lines)
        if not results:
            return json.dumps({"error": f"No related chunks found for {file_path}:{line}."})
        label = f"Chunks related to {file_path}:{line}"
        return json.dumps(format_results(label, results, max_snippet_lines))

    register_graph_tools(server)
    register_dupes_tool(server)
    register_evidence_tools(server, lambda repo, selected: _get_index(repo, cache, selected), default_content)
    return server


async def serve(
    content: Sequence[ContentType] = (ContentType.CODE,),
) -> None:
    """Start an MCP stdio server."""
    cache = IndexCache()
    init_task = asyncio.create_task(cache.load_embedder_once())
    server = create_server(cache, default_content=content)
    try:
        await server.run_stdio_async()
    finally:
        if not init_task.done():
            init_task.cancel()

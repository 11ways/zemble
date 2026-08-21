from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import threading
import time
from collections.abc import Sequence
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from zemble.daemon import client as daemon_client
from zemble.daemon.protocol import CommandRefused, DaemonError
from zemble.dedup.mcp import register_dupes_tool
from zemble.evidence.mcp import register_evidence_tools
from zemble.graph.mcp import register_graph_tools
from zemble.home.mcp import register_home_tool
from zemble.index import ScopeRefused, ZembleIndex
from zemble.index_cache import CACHE_MAX_SIZE, IndexCache
from zemble.runtime.mcp import StaleAwareFastMCP, register_status_tool
from zemble.types import ContentType
from zemble.utils import describe_unresolved_location, format_results, is_git_url

logger = logging.getLogger(__name__)

_REPO_DESCRIPTION = (
    "A local directory path or https:// or http:// git URL (e.g. https://github.com/org/repo) to index and "
    "search. The index is cached after the first call, so repeat queries are fast."
)

ContentSelection = Literal["code", "docs", "config", "all"]

PATHS_DESCRIPTION = (
    'Only answer from these sub-paths of `repo` (repo-relative, e.g. ["src/main"]). '
    "Applied to the results, not to the index: the index itself is unchanged and shared."
)
EXCLUDE_DESCRIPTION = (
    "Gitignore-style patterns, relative to `repo`, whose files are dropped before the top-k is "
    'taken (e.g. ["vendor/", "*.min.js"]). When `repo` has no index yet, these also prune the '
    "walk that builds it, which is how a build refused as too large is recovered from in-band."
)

#: Re-exported so `zemble.mcp` keeps naming the cache size it serves with.
_CACHE_MAX_SIZE = CACHE_MAX_SIZE


async def _daemon_call(cmd: str, args: dict[str, Any]) -> Any | None:
    """Try to answer one tool call from the warm daemon.

    Several agent sessions then share one RAM copy of a workspace index instead of
    each holding its own.

    A refusal is not an outage: it is re-raised so the caller reports it once, instead of
    rebuilding the very index the daemon just refused to build and being refused again.

    :param cmd: Daemon command name.
    :param args: Command arguments.
    :return: The daemon's result, or None if this process must answer in-process.
    :raises CommandRefused: If the daemon deliberately refused the command.
    """
    try:
        return await asyncio.to_thread(daemon_client.call, cmd, args)
    except CommandRefused:
        raise
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


async def _answer_remotely(cmd: str, repo: str, args: dict[str, Any]) -> str | dict[str, Any] | None:
    """Refuse an unacceptable repo, or answer one tool call from the warm daemon.

    The daemon's payload is returned as the object it is; encoding it into a string here
    would make the client parse JSON out of JSON.

    :param cmd: Daemon command name.
    :param repo: The repo argument as the caller wrote it.
    :param args: The rest of the command arguments.
    :return: The refusal text or the payload, or None when this process must answer.
    """
    refusal = unsafe_repo_reason(repo)
    if refusal is not None:
        return refusal
    try:
        return await _daemon_call(cmd, {"path": repo, **args})
    except CommandRefused as exc:
        return str(exc)


async def _get_index(
    repo: str,
    cache: IndexCache,
    content: Sequence[ContentType],
    paths: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> ZembleIndex:
    """Return a cached index for a repo, rejecting unsafe git transport schemes.

    `paths` and `exclude` filter the answer at query time; `exclude` additionally prunes the
    walk when this repo has no index yet, so an oversized tree can be indexed in one call.
    """
    reason = unsafe_repo_reason(repo)
    if reason is not None:
        raise ValueError(reason)
    try:
        index = await cache.get(repo, content=content, exclude=exclude)
    except ScopeRefused as exc:
        raise ValueError(str(exc)) from exc
    except Exception as exc:
        raise ValueError(f"Failed to index {repo!r}: {exc}") from exc
    filtered = index.filtered(paths, exclude)
    if filtered is None:
        raise ValueError(f"No indexed file under {repo} survives paths={list(paths)} exclude={list(exclude)}")
    return filtered


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
    server = StaleAwareFastMCP(
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

    @server.tool(structured_output=False)
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
        paths: Annotated[list[str] | None, Field(description=PATHS_DESCRIPTION)] = None,
        exclude: Annotated[list[str] | None, Field(description=EXCLUDE_DESCRIPTION)] = None,
    ) -> str | dict[str, Any]:
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
                "paths": paths or [],
                "exclude": exclude or [],
            },
        )
        if remote is not None:
            return remote
        try:
            index = await _get_index(repo, cache, selected_content, paths or (), exclude or ())
        except ValueError as exc:
            return str(exc)
        results = index.search(query, top_k=top_k, max_snippet_lines=max_snippet_lines)
        if not results:
            return {"error": "No results found."}
        return format_results(query, results, max_snippet_lines)

    @server.tool(structured_output=False)
    async def find_related(
        file_path: Annotated[
            str,
            Field(
                description=(
                    "File path relative to `repo`, exactly as `search`, `dupes` and the graph tools print it "
                    "(use `file_path` from a prior result)."
                )
            ),
        ],
        line: Annotated[int, Field(description="Any line inside the chunk (1-indexed); it need not be the first.")],
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
        paths: Annotated[list[str] | None, Field(description=PATHS_DESCRIPTION)] = None,
        exclude: Annotated[list[str] | None, Field(description=EXCLUDE_DESCRIPTION)] = None,
    ) -> str | dict[str, Any]:
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
                "paths": paths or [],
                "exclude": exclude or [],
            },
        )
        if remote is not None:
            return remote
        try:
            index = await _get_index(repo, cache, selected_content, paths or (), exclude or ())
        except ValueError as exc:
            return str(exc)
        chunk = index.chunk_at(file_path, line)
        if chunk is None:
            return {"error": describe_unresolved_location(index, file_path, line), "unresolved_location": True}
        results = index.find_related(chunk, top_k=top_k, max_snippet_lines=max_snippet_lines)
        if not results:
            return {"error": f"No related chunks found for {file_path}:{line}."}
        label = f"Chunks related to {file_path}:{line}"
        return format_results(label, results, max_snippet_lines)

    register_status_tool(server)
    register_graph_tools(server)
    register_dupes_tool(server)
    register_evidence_tools(
        server,
        lambda repo, selected, paths=(), exclude=(): _get_index(repo, cache, selected, paths, exclude),
        default_content,
    )
    register_home_tool(server, lambda repo, selected: _get_index(repo, cache, selected))
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
        # The stdio loop returns on stdin EOF; nothing else stops this process, so every
        # background task started here has to be settled before the loop is torn down.
        init_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await init_task
        await asyncio.get_running_loop().shutdown_default_executor()


def force_exit_if_threads_linger(timeout: float = 5.0) -> None:
    """Leave the process even when a library thread outlives the stdio loop, naming what lingered."""
    deadline = time.monotonic() + timeout
    for thread in _live_threads():
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    remaining = _live_threads()
    if not remaining:
        return
    logger.warning("exiting with %d live thread(s): %s", len(remaining), ", ".join(t.name for t in remaining))
    sys.stderr.flush()
    os._exit(0)


def _live_threads() -> list[threading.Thread]:
    """Return the non-daemon threads other than this one that would keep the interpreter alive."""
    current = threading.current_thread()
    return [t for t in threading.enumerate() if t is not current and t.is_alive() and not t.daemon]

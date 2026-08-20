"""The zemble daemon: one warm process per user holding indexes in RAM.

Started on demand by a zemble command that needs it, never at login and never by a
timer, and it exits by itself once it has been idle long enough.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import logging
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from zemble.cache import find_index_from_cache_folder, save_index_to_cache
from zemble.daemon import client
from zemble.daemon.protocol import (
    DEFAULT_IDLE_MINUTES,
    DEFAULT_MAX_INDEXES,
    decode,
    encode,
    lock_path,
    pid_path,
    runtime_directory,
    socket_path,
)
from zemble.daemon.watch import IgnoreRules, RootWatcher
from zemble.index import ZembleIndex
from zemble.index.create import create_index_from_path
from zemble.index.files import get_extensions
from zemble.index.symbols import SymbolDefinitions
from zemble.index.types import PersistencePath, PreviousIndex
from zemble.index_cache import CacheKey, IndexCache, compute_cache_key
from zemble.types import ContentType
from zemble.utils import format_results, is_git_url, resolve_chunk

logger = logging.getLogger(__name__)

#: Handlers take the daemon and the request arguments, and return anything JSON-encodable.
Handler = Callable[["Daemon", dict[str, Any]], Awaitable[Any]]

#: Java is watched on top of the index's own extensions so the symbol graph stays fresh.
_GRAPH_EXTENSIONS = frozenset({".java"})
#: A rebuilt index is written back to the on-disk cache at most this often.
_PERSIST_INTERVAL_SECONDS = 10.0
#: How often the idle check runs.
_IDLE_CHECK_SECONDS = 30.0


def _env_int(name: str, default: int) -> int:
    """Read a non-negative integer setting from the environment, falling back on nonsense."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring %s=%r: not an integer", name, raw)
        return default
    return value if value >= 0 else default


def _content_types(raw: Sequence[str] | None) -> tuple[ContentType, ...]:
    """Resolve a wire content selection to index content types."""
    if not raw:
        return (ContentType.CODE,)
    if "all" in raw:
        return tuple(ContentType)
    return tuple(ContentType(value) for value in raw)


def rebuild_index(previous_index: ZembleIndex, cache_key: CacheKey) -> tuple[ZembleIndex, dict[str, int]]:
    """Reindex a root incrementally from an in-memory index, returning the new index and what moved.

    :param previous_index: The index currently serving this root.
    :param cache_key: The root and content types being rebuilt.
    :return: The replacement index, and counts of added, changed and removed files.
    """
    root = Path(cache_key[0])
    content = cache_key[1]
    # AIDEV-NOTE: create_index_from_path consumes the PreviousIndex - it mutates the BM25
    # index in place and may write into `vectors`. The BM25 index is the live one (copying a
    # multi-hundred-thousand document index costs more than the rebuild), so the caller must
    # hold this root's lock across the rebuild AND the swap.
    previous = PreviousIndex(
        chunks=previous_index.chunks,
        vectors=previous_index._semantic_index.vectors.copy(),
        manifest=previous_index._manifest,
        bm25_index=previous_index._bm25_index,
    )
    before = dict(previous_index._manifest)
    bm25_index, semantic_index, chunks, manifest = create_index_from_path(
        root,
        embedder=previous_index.embedder,
        content=content,
        display_root=root,
        previous=previous,
        capsules=previous_index._capsules,
    )
    counts = {
        "added": len(manifest.keys() - before.keys()),
        "removed": len(before.keys() - manifest.keys()),
        "changed": sum(
            1 for path, entry in manifest.items() if path in before and before[path].mtime_ns != entry.mtime_ns
        ),
    }
    index = ZembleIndex(
        previous_index.embedder,
        bm25_index,
        semantic_index,
        chunks,
        root=root,
        content=content,
        manifest=manifest,
        capsules=previous_index._capsules,
    )
    return index, counts


class Daemon:
    """Holds the warm indexes, the watchers, and the command table's state."""

    def __init__(
        self,
        max_indexes: int | None = None,
        idle_minutes: int | None = None,
        watch: bool = True,
    ) -> None:
        """Create a daemon.

        :param max_indexes: Resident index limit; None reads ZEMBLE_DAEMON_MAX_INDEXES.
        :param idle_minutes: Idle shutdown delay in minutes, 0 to never exit; None reads the environment.
        :param watch: Whether loaded local roots are watched for changes.
        """
        self.max_indexes = (
            max_indexes if max_indexes is not None else _env_int("ZEMBLE_DAEMON_MAX_INDEXES", DEFAULT_MAX_INDEXES)
        )
        self.idle_minutes = (
            idle_minutes if idle_minutes is not None else _env_int("ZEMBLE_DAEMON_IDLE_MINUTES", DEFAULT_IDLE_MINUTES)
        )
        self.watch_enabled = watch
        self.cache = IndexCache(max_size=max(1, self.max_indexes), on_evict=self._on_evict)
        self.watchers: dict[CacheKey, RootWatcher] = {}
        self.locks: dict[CacheKey, asyncio.Lock] = {}
        self.pending: set[CacheKey] = set()
        self.rebuilding: set[CacheKey] = set()
        self.last_rebuild: dict[CacheKey, dict[str, Any]] = {}
        self._persisted_at: dict[CacheKey, float] = {}
        self.started_at = time.time()
        self.last_request_at = time.monotonic()
        self.requests = 0
        self.stop_event = asyncio.Event()

    # -- index access -------------------------------------------------------

    def lock_for(self, cache_key: CacheKey) -> asyncio.Lock:
        """Return the per-root lock serialising rebuilds against queries."""
        return self.locks.setdefault(cache_key, asyncio.Lock())

    async def index_for(self, args: dict[str, Any]) -> tuple[CacheKey, ZembleIndex]:
        """Resolve the requested root, returning its key and warm index.

        :param args: Request arguments carrying `path` and optional `content` and `ref`.
        :return: The cache key and the index.
        :raises ValueError: If no path was given.
        """
        path = args.get("path")
        if not path:
            raise ValueError("missing 'path'")
        if is_git_url(str(path)) and not str(path).startswith(("https://", "http://")):
            raise ValueError(f"Only https://, http://, or local directory paths are accepted. Got: {path!r}")
        content = _content_types(args.get("content"))
        ref = args.get("ref")
        await self.cache.load_embedder_once()
        cache_key = compute_cache_key(path, ref, content)
        index = await self.cache.get(path, ref=ref, content=content)
        self._ensure_watcher(cache_key, index)
        return cache_key, index

    def _ensure_watcher(self, cache_key: CacheKey, index: ZembleIndex) -> None:
        """Start watching a local root the first time it is served."""
        if not self.watch_enabled or cache_key in self.watchers or is_git_url(cache_key[0]):
            return
        root = Path(cache_key[0])
        if not root.is_dir():
            return
        extensions = set(get_extensions(index.content)) | _GRAPH_EXTENSIONS
        watcher = RootWatcher(root, IgnoreRules(root, extensions), lambda paths: self._on_change(cache_key, paths))
        self.watchers[cache_key] = watcher
        watcher.start()
        logger.info("watching %s (%d extensions)", root, len(extensions))

    def _on_evict(self, cache_key: CacheKey) -> None:
        """Stop the watcher of an index that left the cache."""
        watcher = self.watchers.pop(cache_key, None)
        if watcher is not None:
            watcher.stop()
            logger.info("stopped watching %s", cache_key[0])
        self.locks.pop(cache_key, None)
        self._persisted_at.pop(cache_key, None)

    # -- rebuilding ---------------------------------------------------------

    async def _on_change(self, cache_key: CacheKey, paths: set[Path]) -> None:
        """Handle one coalesced change set for a watched root."""
        if cache_key not in self.cache._tasks:
            return
        self.pending.add(cache_key)
        try:
            await self.rebuild(cache_key, java_changed=any(path.suffix == ".java" for path in paths))
        finally:
            self.pending.discard(cache_key)

    async def rebuild(self, cache_key: CacheKey, *, java_changed: bool = True) -> dict[str, Any]:
        """Reindex one root in place and swap the result in atomically.

        :param cache_key: The root and content types to rebuild.
        :param java_changed: Whether the symbol graph should be refreshed too.
        :return: What the rebuild did.
        """
        lock = self.lock_for(cache_key)
        self.rebuilding.add(cache_key)
        started = time.monotonic()
        try:
            async with lock:
                if self.cache.is_building(cache_key):
                    return {"skipped": "initial build in progress"}
                current = next((index for key, index in self.cache.loaded() if key == cache_key), None)
                if current is None:
                    return {"skipped": "not loaded"}
                index, counts = await asyncio.to_thread(rebuild_index, current, cache_key)
                elapsed = time.monotonic() - started
                self.cache.replace(cache_key, index, cooldown_seconds=elapsed * 3)
        finally:
            self.rebuilding.discard(cache_key)
        result: dict[str, Any] = {**counts, "ms": round(elapsed * 1000), "chunks": len(index.chunks)}
        logger.info(
            "rebuilt %s: %d added, %d changed, %d removed, %d chunks in %d ms",
            cache_key[0],
            counts["added"],
            counts["changed"],
            counts["removed"],
            len(index.chunks),
            result["ms"],
        )
        if await self._persist(cache_key, index):
            await self._reload_definitions(cache_key, index)
        if java_changed:
            result["graph_ms"] = await self._refresh_graph(cache_key[0])
        self.last_rebuild[cache_key] = result
        return result

    async def _persist(self, cache_key: CacheKey, index: ZembleIndex) -> bool:
        """Write a rebuilt index back to the on-disk cache, at most once per interval.

        :param cache_key: The root and content types that were rebuilt.
        :param index: The index to write.
        :return: Whether it was written this time.
        """
        now = time.monotonic()
        if now - self._persisted_at.get(cache_key, 0.0) < _PERSIST_INTERVAL_SECONDS:
            return False
        self._persisted_at[cache_key] = now
        try:
            await asyncio.to_thread(save_index_to_cache, index, cache_key[0])
        except Exception:
            logger.warning("Failed to persist rebuilt index for %s", cache_key[0], exc_info=True)
            return False
        return True

    async def _reload_definitions(self, cache_key: CacheKey, index: ZembleIndex) -> None:
        """Re-attach the symbol-definition lookup a rebuild invalidated.

        The lookup is only ever built at save time, so a rebuilt index carries none until its
        chunks have been written; until then symbol reranking falls back to its own scan.
        """
        symbols = PersistencePath.from_path(find_index_from_cache_folder(cache_key[0], cache_key[1])).symbols
        try:
            index._definitions = await asyncio.to_thread(SymbolDefinitions.load, symbols)
        except Exception:
            logger.debug("No symbol definitions to re-attach for %s", cache_key[0], exc_info=True)

    async def _refresh_graph(self, root: str) -> int | None:
        """Incrementally refresh the symbol graph for a root that already has one."""
        from zemble.graph.store import build_graph, graph_exists

        if not await asyncio.to_thread(graph_exists, root):
            return None
        started = time.monotonic()
        try:
            await asyncio.to_thread(build_graph, root)
        except Exception:
            logger.warning("Failed to refresh the symbol graph for %s", root, exc_info=True)
            return None
        return round((time.monotonic() - started) * 1000)

    # -- serving ------------------------------------------------------------

    async def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run one request through the command table and shape its response."""
        request_id = request.get("id")
        command = request.get("cmd")
        handler = COMMANDS.get(str(command))
        if handler is None:
            return {"id": request_id, "ok": False, "error": f"unknown command: {command!r}"}
        self.requests += 1
        self.last_request_at = time.monotonic()
        args = request.get("args")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return {"id": request_id, "ok": False, "error": "'args' must be an object"}
        try:
            result = await handler(self, args)
        except Exception as exc:
            logger.warning("Command %r failed", command, exc_info=True)
            return {"id": request_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            self.last_request_at = time.monotonic()
        return {"id": request_id, "ok": True, "result": result}

    async def serve_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Serve newline-delimited requests on one connection until the peer closes it."""
        try:
            while True:
                line = await reader.readline()
                if not line:
                    return
                try:
                    request = decode(line)
                except Exception as exc:
                    writer.write(encode({"id": None, "ok": False, "error": str(exc)}))
                    await writer.drain()
                    continue
                response = await self.handle(request)
                writer.write(encode(response))
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError):  # pragma: no cover - client vanished
            return
        finally:
            with contextlib.suppress(OSError):
                writer.close()

    async def idle_loop(self) -> None:
        """Exit once the daemon has answered nothing for the configured idle window."""
        if self.idle_minutes <= 0:
            return
        limit = self.idle_minutes * 60
        while not self.stop_event.is_set():
            await asyncio.sleep(min(_IDLE_CHECK_SECONDS, limit))
            if self.rebuilding or self.pending:
                continue
            if time.monotonic() - self.last_request_at >= limit:
                logger.info("idle for %d minute(s); shutting down", self.idle_minutes)
                self.stop_event.set()
                return

    def shutdown(self) -> None:
        """Stop every watcher and ask the serve loop to end."""
        for watcher in list(self.watchers.values()):
            watcher.stop()
        self.watchers.clear()
        self.stop_event.set()


# -- command table ----------------------------------------------------------
# AIDEV-NOTE: one entry per command; a new daemon command is one function plus one line here.


async def _cmd_ping(daemon: Daemon, args: dict[str, Any]) -> Any:
    """Answer that the daemon is alive."""
    return {"pong": True, "pid": os.getpid()}


async def _cmd_status(daemon: Daemon, args: dict[str, Any]) -> Any:
    """Report what the daemon holds and what it is doing."""
    indexes = []
    for cache_key, index in daemon.cache.loaded():
        indexes.append(
            {
                "root": cache_key[0],
                "content": [content.value for content in cache_key[1]],
                "embedder": index.embedder.model_id,
                "chunks": len(index.chunks),
                "files": index.stats.indexed_files,
                "last_used": daemon.cache.last_used.get(cache_key),
                "watching": cache_key in daemon.watchers,
                "rebuilding": cache_key in daemon.rebuilding,
                "last_rebuild": daemon.last_rebuild.get(cache_key),
            }
        )
    building = [
        {"root": key[0], "content": [content.value for content in key[1]]}
        for key in daemon.cache._tasks
        if daemon.cache.is_building(key)
    ]
    return {
        "pid": os.getpid(),
        "uptime_seconds": round(time.time() - daemon.started_at, 1),
        "rss_mb": _rss_mb(),
        "requests": daemon.requests,
        "idle_seconds": round(time.monotonic() - daemon.last_request_at, 1),
        "idle_minutes_limit": daemon.idle_minutes,
        "max_indexes": daemon.max_indexes,
        "socket": str(socket_path()),
        "indexes": indexes,
        "building": building,
        "pending_reindex": [key[0] for key in daemon.pending],
    }


async def _cmd_search(daemon: Daemon, args: dict[str, Any]) -> Any:
    """Search one root, returning the same payload shape the CLI and MCP print."""
    cache_key, index = await daemon.index_for(args)
    query = str(args.get("query", ""))
    max_snippet_lines = args.get("max_snippet_lines")
    async with daemon.lock_for(cache_key):
        results = await asyncio.to_thread(
            index.search,
            query,
            top_k=int(args.get("top_k", 5)),
            max_snippet_lines=max_snippet_lines,
        )
    if not results:
        return {"error": "No results found."}
    return format_results(query, results, max_snippet_lines)


async def _cmd_find_related(daemon: Daemon, args: dict[str, Any]) -> Any:
    """Find chunks similar to a location, or report that the location is not indexed."""
    cache_key, index = await daemon.index_for(args)
    file_path = str(args.get("file_path", ""))
    line = int(args.get("line", 0))
    max_snippet_lines = args.get("max_snippet_lines")
    chunk = resolve_chunk(index.chunks, file_path, line)
    if chunk is None:
        return {"error": f"No chunk found at {file_path}:{line}.", "chunk_missing": True}
    async with daemon.lock_for(cache_key):
        results = await asyncio.to_thread(
            index.find_related,
            chunk,
            top_k=int(args.get("top_k", 5)),
            max_snippet_lines=max_snippet_lines,
        )
    if not results:
        return {"error": f"No related chunks found for {file_path}:{line}."}
    return format_results(f"Chunks related to {file_path}:{line}", results, max_snippet_lines)


async def _cmd_stats(daemon: Daemon, args: dict[str, Any]) -> Any:
    """Report what one index holds."""
    _cache_key, index = await daemon.index_for(args)
    stats = index.stats
    return {
        "path": args.get("path"),
        "embedder": stats.embedder,
        "dimensions": stats.dimensions,
        "indexed_files": stats.indexed_files,
        "total_chunks": stats.total_chunks,
        "content": [content.value for content in index.content],
        "languages": stats.languages,
    }


async def _cmd_graph(daemon: Daemon, args: dict[str, Any]) -> Any:
    """Answer a symbol-graph question, or just guarantee the graph is fresh.

    The daemon's win here is freshness: it builds the graph once and its watcher keeps
    it current, so a client never pays the workspace scan `ensure_graph` would do.
    """
    from zemble.graph.cli import ensure_graph
    from zemble.graph.mcp import answer

    path = str(args.get("path", ""))
    if not path:
        raise ValueError("missing 'path'")
    command = str(args.get("command", "ensure"))
    if command == "ensure":
        await asyncio.to_thread(ensure_graph, path)
        return {"ensured": True}
    kinds = args.get("kinds")
    extra: dict[str, Any] = {}
    if command == "neighbors":
        from zemble.graph.model import EdgeKind

        extra = {"hops": int(args.get("hops", 1)), "kinds": [EdgeKind(kind) for kind in kinds] if kinds else None}
    return await asyncio.to_thread(answer, path, str(args.get("symbol", "")), command, **extra)


async def _cmd_refresh(daemon: Daemon, args: dict[str, Any]) -> Any:
    """Force a rebuild check for a root, loading it first if it is not resident."""
    cache_key, _index = await daemon.index_for(args)
    return await daemon.rebuild(cache_key)


async def _cmd_evict(daemon: Daemon, args: dict[str, Any]) -> Any:
    """Drop one root from memory, stopping its watcher."""
    path = args.get("path")
    if not path:
        raise ValueError("missing 'path'")
    cache_key = compute_cache_key(str(path), args.get("ref"), _content_types(args.get("content")))
    was_loaded = cache_key in daemon.cache._tasks
    daemon.cache.evict(cache_key)
    return {"evicted": was_loaded, "root": cache_key[0]}


async def _cmd_shutdown(daemon: Daemon, args: dict[str, Any]) -> Any:
    """Stop the daemon after this response has been written."""
    asyncio.get_running_loop().call_later(0.05, daemon.shutdown)
    return {"stopping": True, "pid": os.getpid()}


COMMANDS: dict[str, Handler] = {
    "ping": _cmd_ping,
    "status": _cmd_status,
    "search": _cmd_search,
    "find_related": _cmd_find_related,
    "stats": _cmd_stats,
    "graph": _cmd_graph,
    "refresh": _cmd_refresh,
    "evict": _cmd_evict,
    "shutdown": _cmd_shutdown,
}


def _rss_mb() -> float | None:
    """Return this process's resident set size in MB, where the platform reports one."""
    try:
        with open("/proc/self/statm", encoding="utf-8") as handle:
            pages = int(handle.read().split()[1])
    except (OSError, IndexError, ValueError):
        return None
    return round(pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024), 1)


class SocketInUse(RuntimeError):
    """Another daemon already owns this socket."""


def _acquire_lock() -> Any:
    """Take the single-daemon lock for this socket.

    :return: The open lock file, which must stay open for the daemon's lifetime.
    :raises SocketInUse: If another live daemon holds it.
    """
    runtime_directory(create=True)
    handle = open(lock_path(), "w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise SocketInUse(f"another zemble daemon already owns {socket_path()}") from exc
    return handle


async def run(max_indexes: int | None = None, idle_minutes: int | None = None, watch: bool = True) -> None:
    """Run the daemon in the foreground until it is told, or decides, to stop.

    :param max_indexes: Resident index limit; None reads the environment.
    :param idle_minutes: Idle shutdown delay; None reads the environment.
    :param watch: Whether to watch loaded roots.
    :raises SocketInUse: If another daemon is already listening on this socket.
    """
    # A daemon must never route its own work through a daemon client: that is a deadlock
    # on its own socket, and every shared code path (graph ensure, search) can reach one.
    client.disable_for_this_process("running inside the daemon")
    lock = _acquire_lock()
    path = socket_path(create_dir=True)
    if path.exists():
        # We hold the lock, so any socket file here belongs to a daemon that is gone.
        path.unlink()
    daemon = Daemon(max_indexes=max_indexes, idle_minutes=idle_minutes, watch=watch)
    server = await asyncio.start_unix_server(daemon.serve_connection, path=str(path))
    os.chmod(path, 0o600)
    pid_path().write_text(f"{os.getpid()}\n", encoding="utf-8")
    logger.info(
        "zemble daemon %d listening on %s (max_indexes=%d, idle_minutes=%d)",
        os.getpid(),
        path,
        daemon.max_indexes,
        daemon.idle_minutes,
    )
    idle_task = asyncio.create_task(daemon.idle_loop())
    prewarm = asyncio.create_task(daemon.cache.load_embedder_once())
    try:
        async with server:
            await daemon.stop_event.wait()
    finally:
        idle_task.cancel()
        prewarm.cancel()
        daemon.shutdown()
        with contextlib.suppress(OSError):
            path.unlink()
        with contextlib.suppress(OSError):
            pid_path().unlink()
        lock.close()
        logger.info("zemble daemon %d stopped", os.getpid())

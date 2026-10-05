"""The zemble daemon: one warm process per user holding indexes in RAM.

Started on demand by a zemble command that needs it, never at login and never by a
timer, and it exits by itself once it has been idle long enough.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import gc
import json
import logging
import os
import tempfile
import time
import traceback
import weakref
from collections.abc import Awaitable, Callable, Sequence
from copy import copy
from pathlib import Path
from typing import Any

from zemble.cache import find_index_from_cache_folder, resolve_cache_folder
from zemble.chunking.chunking import _DESIRED_CHUNK_LENGTH_CHARS
from zemble.daemon import client
from zemble.daemon.admission import AdmissionBusy, ReadAdmission, request_deadline
from zemble.daemon.graph_jobs import GraphJobs
from zemble.daemon.memory import (
    MEMORY_ENV,
    MIB,
    VIRTUAL_ENV,
    MemoryRefused,
    allocation_backstop,
    default_budget_mb,
    virtual_mb,
)
from zemble.daemon.protocol import (
    ACCEPTS_BUSY_FIELD,
    DEFAULT_IDLE_MINUTES,
    DEFAULT_MAX_INDEXES,
    CommandBusy,
    ErrorKind,
    decode,
    encode,
    identity_envelope,
    lock_path,
    pid_path,
    runtime_directory,
    socket_path,
)
from zemble.daemon.watch import IgnoreRules, RootWatcher
from zemble.embedding.base import Embedder
from zemble.graph.facts import matches_facts_glob
from zemble.index import ZembleIndex
from zemble.index.create import create_index_from_path
from zemble.index.files import get_extensions
from zemble.index.scope import TreeEstimate, estimate_tree, measure_work, require_declared_scope
from zemble.index.types import PreviousIndex
from zemble.index_cache import CacheKey, IndexCache, compute_cache_key
from zemble.refusal import Refused
from zemble.runtime.identity import identity, status_payload
from zemble.runtime.memory import release_free_heap
from zemble.types import ContentType
from zemble.userenv import load_user_env, user_env_path
from zemble.utils import describe_unresolved_location, format_results, is_git_url

logger = logging.getLogger(__name__)

#: Handlers take the daemon and the request arguments, and return anything JSON-encodable.
Handler = Callable[["Daemon", dict[str, Any]], Awaitable[Any]]

# AIDEV-NOTE: a deterministic "no" is not an outage. Answering the same request in the
# client's own process refuses identically, so the wire says REFUSED and the client stops
# instead of paying for a second full build to be told the same thing. The tuple is derived
# from the base class rather than listing the two it used to name: every refusal IS a
# `Refused` and every `Refused` carries the knob this module reports, so a third refusal type
# is caught and reported here without a second edit - and never as an AttributeError.
REFUSAL_TYPES: tuple[type[Refused], ...] = (Refused,)

#: Java is watched on top of the index's own extensions so the symbol graph stays fresh.
_GRAPH_EXTENSIONS = frozenset({".java"})
#: How often the idle check runs.
_IDLE_CHECK_SECONDS = 30.0
_QUIET_SECONDS = 2.0
_MAX_CHANGED_PATHS = 4096


def _work_reserve(estimate: TreeEstimate, dimensions: int) -> float:
    """Reserve parsing/postings plus three transient copies of estimated fresh embedding rows."""
    rows = estimate.bytes / _DESIRED_CHUNK_LENGTH_CHARS + estimate.files
    return (estimate.bytes * 12 + rows * dimensions * 12) / MIB


def _merge_changes(
    pending: dict[CacheKey, set[Path] | None], key: CacheKey, paths: Sequence[Path] | set[Path] | None
) -> None:
    """Coalesce paths into a capped set, with None representing one full rescan."""
    changes = pending.setdefault(key, set())
    if changes is None or paths is None:
        pending[key] = None
        return
    changes.update(paths)
    if len(changes) > _MAX_CHANGED_PATHS:
        pending[key] = None


class ResidentCache(IndexCache):
    """The daemon owns freshness and keeps only one mapped generation per root."""

    require_persistence = True

    def __init__(
        self,
        max_size: int,
        on_evict: Callable[[CacheKey], None],
        on_build: Callable[[CacheKey], None],
        watch_owned: bool,
    ) -> None:
        """Watch admitted roots before chunking so edits during a cold build are not lost."""
        super().__init__(max_size=max_size, on_evict=on_evict)
        self._on_build = on_build
        self.watch_owned = watch_owned

    async def _build_tracked(
        self, source: str, ref: str | None, embedder: Embedder, cache_key: CacheKey, exclude: Sequence[str] = ()
    ) -> ZembleIndex:
        """Start collecting changes before entering the construction thread."""
        self._on_build(cache_key)
        return await super()._build_tracked(source, ref, embedder, cache_key, exclude)

    async def _evict_if_stale(self, cache_key: CacheKey) -> None:
        """Leave freshness to the watcher instead of rebuilding a churning tree from a query."""
        if not self.watch_owned:
            await super()._evict_if_stale(cache_key)

    def _build_index(
        self, source: str, ref: str | None, embedder: Embedder, cache_key: CacheKey, exclude: Sequence[str] = ()
    ) -> ZembleIndex:
        """Discard construction dictionaries and vectors before returning the mapped stores."""
        index = super()._build_index(source, ref, embedder, cache_key, exclude)
        path = find_index_from_cache_folder(cache_key[0], index.content, index.exclude)
        embedder = index.embedder
        del index
        gc.collect()
        release_free_heap()
        return ZembleIndex.load_from_disk(path, embedder=embedder)


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


def _patterns(raw: object) -> tuple[str, ...]:
    """Read a wire list of paths or gitignore patterns, dropping blanks and refusing a non-list."""
    if raw is None:
        return ()
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise ValueError("'paths' and 'exclude' must be lists of strings")
    return tuple(item.strip() for item in raw if item.strip())


def _content_types(raw: Sequence[str] | None) -> tuple[ContentType, ...]:
    """Resolve a wire content selection to index content types."""
    if not raw:
        return (ContentType.CODE,)
    if "all" in raw:
        return tuple(ContentType)
    return tuple(ContentType(value) for value in raw)


def rebuild_index(
    previous_index: ZembleIndex, cache_key: CacheKey, changed_paths: Sequence[Path] | None = None
) -> tuple[ZembleIndex, dict[str, int]]:
    """Reindex a root incrementally from an in-memory index, returning the new index and what moved.

    :param previous_index: The index currently serving this root.
    :param cache_key: The root and content types being rebuilt.
    :param changed_paths: The paths a watcher saw move; None re-walks the whole tree.
    :return: The replacement index, and counts of added, changed and removed files.
    """
    root = Path(cache_key[0])
    content = cache_key[1]
    # AIDEV-NOTE: the serving generation stays immutable. Reused vectors are streamed into a
    # temporary mapping, not copied to anonymous heap; BM25 shares its immutable postings.
    # Admission reserves the replacement inside the process budget before any allocation.
    previous = PreviousIndex(
        chunks=previous_index.chunks,
        vectors=previous_index._semantic_index.vectors,
        manifest=previous_index._manifest,
        bm25_index=previous_index._bm25_index,
    )
    before = dict(previous_index._manifest)
    with tempfile.TemporaryDirectory(prefix="rebuild-", dir=resolve_cache_folder()) as temporary:
        bm25_index, semantic_index, chunks, manifest = create_index_from_path(
            root,
            embedder=previous_index.embedder,
            content=content,
            display_root=root,
            previous=previous,
            capsules=previous_index._capsules,
            changed_paths=changed_paths,
            exclude=previous_index.exclude,
            vector_path=Path(temporary) / "vectors.npy",
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
        exclude=previous_index.exclude,
    )
    return index, counts


def _mapped_rebuild(
    current: ZembleIndex, cache_key: CacheKey, changed_paths: Sequence[Path] | None
) -> tuple[ZembleIndex, dict[str, int]]:
    """Release construction storage before publishing a mapped replacement."""
    index, counts = rebuild_index(current, cache_key, changed_paths)
    path = find_index_from_cache_folder(cache_key[0], cache_key[1], index.exclude)
    index.save(path)
    del index
    gc.collect()
    release_free_heap()
    return ZembleIndex.load_from_disk(path, embedder=current.embedder), counts


class Daemon:
    """Holds the warm indexes, the watchers, and the command table's state."""

    def __init__(
        self,
        max_indexes: int | None = None,
        idle_minutes: int | None = None,
        watch: bool = True,
        max_rss_mb: int | None = None,
    ) -> None:
        """Create a daemon.

        :param max_indexes: Resident index limit; None reads ZEMBLE_DAEMON_MAX_INDEXES.
        :param idle_minutes: Idle shutdown delay in minutes, 0 to never exit; None reads the environment.
        :param watch: Whether loaded local roots are watched for changes.
        :param max_rss_mb: Process memory ceiling in MiB; None reads the env or RAM-derived default.
        :raises ValueError: If the memory ceiling is not positive.
        """
        self.max_indexes = (
            max_indexes if max_indexes is not None else _env_int("ZEMBLE_DAEMON_MAX_INDEXES", DEFAULT_MAX_INDEXES)
        )
        self.idle_minutes = (
            idle_minutes if idle_minutes is not None else _env_int("ZEMBLE_DAEMON_IDLE_MINUTES", DEFAULT_IDLE_MINUTES)
        )
        self.watch_enabled = watch
        self.max_rss_mb = max_rss_mb if max_rss_mb is not None else _env_int(MEMORY_ENV, default_budget_mb())
        if self.max_rss_mb <= 0:
            raise ValueError(f"{MEMORY_ENV} must be positive")
        self.max_virtual_mb = _env_int(VIRTUAL_ENV, self.max_rss_mb)
        if self.max_virtual_mb <= 0:
            raise ValueError(f"{VIRTUAL_ENV} must be positive")
        self.cache = ResidentCache(
            max_size=max(1, self.max_indexes),
            on_evict=self._on_evict,
            on_build=self._ensure_watcher,
            watch_owned=watch,
        )
        self._operation_lock = asyncio.Lock()
        self._load_lock = asyncio.Lock()
        self.reads = ReadAdmission(_env_int("ZEMBLE_DAEMON_READ_SLOTS", 4), _env_int("ZEMBLE_DAEMON_QUEUE_LIMIT", 32))
        from zemble.daemon.read_cache import ReadCache

        self.read_cache = ReadCache()
        self.graphs = GraphJobs(self._load_lock, self.max_rss_mb)
        self._request_tasks: set[asyncio.Task[Any]] = set()
        self._query_keys: dict[asyncio.Task[Any], CacheKey] = {}
        self._watch_tasks: dict[CacheKey, asyncio.Task[None]] = {}
        self._watch_changes: dict[CacheKey, set[Path] | None] = {}
        self._changed_at: dict[CacheKey, float] = {}
        self.watchers: dict[CacheKey, RootWatcher] = {}
        self.locks: dict[CacheKey, asyncio.Lock] = {}
        self.pending: set[CacheKey] = set()
        self.rebuilding: set[CacheKey] = set()
        self._rebuild_tasks: dict[CacheKey, asyncio.Task[dict[str, Any]]] = {}
        self._rebuild_changes: dict[CacheKey, set[Path] | None] = {}
        self._rebuild_graph: set[CacheKey] = set()
        self.last_rebuild: dict[CacheKey, dict[str, Any]] = {}
        #: Why a root's last rebuild did not happen, e.g. a refused paid embed. Kept until one succeeds.
        self.last_error: dict[CacheKey, dict[str, Any]] = {}
        self.started_at = time.time()
        self.last_request_at = time.monotonic()
        self.requests = 0
        self.stop_event = asyncio.Event()

    # -- index access -------------------------------------------------------

    def rebuild_lock_for(self, cache_key: CacheKey) -> asyncio.Lock:
        """Return the per-root lock serialising one rebuild against the next.

        Queries deliberately do not take it: a rebuild builds a new index beside the one
        being served and swaps it in, so nothing a query can reach is ever half-updated.
        """
        return self.locks.setdefault(cache_key, asyncio.Lock())

    async def index_for(self, args: dict[str, Any]) -> tuple[CacheKey, ZembleIndex]:
        """Resolve the requested root, returning its key and the warm index that answers it.

        A path inside a root the daemon already holds is answered from that root, filtered to
        the sub-tree: the returned key names the ancestor, and the index is a view of it that
        speaks paths relative to the REQUESTED path, as every other tool for that path does.
        `paths` and `exclude` narrow the answer further, at query time; only a root with no
        index yet is BUILT pruned.

        :param args: Request arguments carrying `path` and optional `content`, `ref`,
            `paths` and `exclude`.
        :return: The cache key and the index.
        :raises ValueError: If no path was given, or the filter keeps no indexed file.
        """
        path = _root_of(args)
        if is_git_url(str(path)) and not str(path).startswith(("https://", "http://")):
            raise ValueError(f"Only https://, http://, or local directory paths are accepted. Got: {path!r}")
        content = _content_types(args.get("content"))
        ref = args.get("ref")
        paths = _patterns(args.get("paths"))
        exclude = _patterns(args.get("exclude"))
        await self.cache.load_embedder_once()
        warm = self._warm_for(path, ref, content)
        if warm is not None and not self.watch_enabled:
            await self.cache._evict_if_stale(warm[0])
            if warm[0] not in self.cache._tasks:
                warm = None
        if warm is None:
            async with self._load_lock:
                cache_key, index = await self._resident_for(path, ref, content, exclude)
        else:
            cache_key, index = warm
        request_task = asyncio.current_task()
        if request_task in self._request_tasks:
            self._query_keys[request_task] = cache_key
        self._ensure_watcher(cache_key)
        drop = set(get_extensions(index.content)) - set(get_extensions(content))
        content_filter = [f"*{extension}" for extension in sorted(drop)]
        filtered = index.filtered(paths, (*exclude, *content_filter))
        if filtered is None:
            raise ValueError(f"No indexed file under {path} survives paths={list(paths)} exclude={list(exclude)}")
        if set(content) != set(filtered.content):
            filtered = copy(filtered)
            filtered._content = content
        return cache_key, filtered

    def _warm_for(
        self, path: str, ref: str | None, content: tuple[ContentType, ...]
    ) -> tuple[CacheKey, ZembleIndex] | None:
        """Serve a covering immutable generation without waiting for a background build."""
        requested = compute_cache_key(path, ref, content)[0]
        for key, index in sorted(self.cache.loaded(), key=lambda entry: len(entry[0][0]), reverse=True):
            if not set(content) <= set(key[1]):
                continue
            if key[0] == requested:
                view = index
            elif not is_git_url(path) and Path(requested).is_relative_to(Path(key[0])):
                view = index.subtree(Path(requested).relative_to(Path(key[0])).as_posix())
            else:
                continue
            if view is not None:
                self.cache._tasks.move_to_end(key)
                self.cache.last_used[key] = time.time()
                return key, view
        return None

    def _admit(self, reserve_mb: float, keep: CacheKey | None = None) -> bool:
        """Evict idle LRU stores until both resident and address-space headroom cover the work."""
        while (_rss_mb() or 0) + reserve_mb > self.max_rss_mb or virtual_mb() + reserve_mb > self.max_virtual_mb:
            victim = next(
                (
                    key
                    for key, _ in self.cache.loaded()
                    if key != keep and key not in self.rebuilding and key not in self._query_keys.values()
                ),
                None,
            )
            if victim is None:
                return False
            self.cache.evict(victim)
            gc.collect()
            release_free_heap()
        return True

    async def _resident_for(
        self, path: str, ref: str | None, content: tuple[ContentType, ...], exclude: tuple[str, ...]
    ) -> tuple[CacheKey, ZembleIndex]:
        """Share a covering root, or replace its narrower selection before admitting another load."""
        requested = compute_cache_key(path, ref, content)
        for key in [key for key, _ in self.cache.loaded()]:
            if key[0] == requested[0] or (not is_git_url(path) and Path(requested[0]).is_relative_to(Path(key[0]))):
                if set(content) <= set(key[1]):
                    # The cache handles path rebasing; its lookup needs the covering selection.
                    return await self.cache.get_with_key(path, ref=ref, content=key[1], exclude=exclude)
                if key[0] == requested[0]:
                    content = tuple(kind for kind in ContentType if kind in set(content) | set(key[1]))
                    exclude = self.cache._exclude_by_key.get(key, ())
                    self.cache.evict(key)
                    break
        gc.collect()
        release_free_heap()
        reserve = 128.0
        if not is_git_url(path):
            require_declared_scope(Path(path))
            estimate = await asyncio.to_thread(estimate_tree, Path(path), content, exclude)
            reserve += _work_reserve(estimate, self.cache.embedder.dimensions)
        if not self._admit(reserve):
            raise MemoryRefused(
                f"Loading {path} needs ~{reserve:.0f} MiB headroom within {self.max_rss_mb} MiB ({MEMORY_ENV})."
            )
        return await self.cache.get_with_key(path, ref=ref, content=content, exclude=exclude)

    def _ensure_watcher(self, cache_key: CacheKey) -> None:
        """Start watching a local root the first time it is served."""
        if not self.watch_enabled or cache_key in self.watchers or is_git_url(cache_key[0]):
            return
        root = Path(cache_key[0])
        if not root.is_dir():
            return
        extensions = set(get_extensions(cache_key[1])) | _GRAPH_EXTENSIONS
        rules = IgnoreRules(root, extensions, always=lambda path: matches_facts_glob(root, path))
        watcher = RootWatcher(root, rules, lambda paths: self._on_change(cache_key, paths))
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
        self.last_error.pop(cache_key, None)
        self.last_rebuild.pop(cache_key, None)
        self.pending.discard(cache_key)
        self._watch_changes.pop(cache_key, None)
        self._changed_at.pop(cache_key, None)
        if cache_key not in self.rebuilding:
            self._rebuild_changes.pop(cache_key, None)
            self._rebuild_graph.discard(cache_key)
            rebuild = self._rebuild_tasks.pop(cache_key, None)
            if rebuild is not None:
                rebuild.cancel()
        task = self._watch_tasks.pop(cache_key, None)
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    # -- rebuilding ---------------------------------------------------------

    async def _on_change(self, cache_key: CacheKey, paths: set[Path]) -> None:
        """Handle one coalesced change set for a watched root."""
        if cache_key not in self.cache._tasks:
            return
        self.pending.add(cache_key)
        self._changed_at[cache_key] = time.monotonic()
        _merge_changes(self._watch_changes, cache_key, paths)
        if cache_key not in self._watch_tasks:
            self._watch_tasks[cache_key] = asyncio.create_task(self._watch_rebuild(cache_key))

    async def _watch_rebuild(self, cache_key: CacheKey) -> None:
        """Wait for actual quiet, retaining one bounded change set per resident root."""
        try:
            while cache_key in self.cache._tasks:
                delay = self._changed_at[cache_key] + _QUIET_SECONDS - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                    continue
                paths = self._watch_changes.pop(cache_key, set())
                root = Path(cache_key[0])
                graph = paths is None or any(path.suffix == ".java" or matches_facts_glob(root, path) for path in paths)
                try:
                    result = await self.rebuild(
                        cache_key, java_changed=graph, changed_paths=None if paths is None else sorted(paths)
                    )
                except Exception:
                    logger.exception("Watcher rebuild failed for %s", cache_key[0])
                    result = {"deferred": "rebuild failed"}
                if "deferred" in result:
                    self._watch_changes[cache_key] = None
                    self._changed_at[cache_key] = time.monotonic() + 28
                elif cache_key not in self._watch_changes:
                    return
        finally:
            if self._watch_tasks.get(cache_key) is asyncio.current_task():
                self.pending.discard(cache_key)
                self._watch_tasks.pop(cache_key, None)
                self._watch_changes.pop(cache_key, None)
                self._changed_at.pop(cache_key, None)

    async def rebuild(
        self, cache_key: CacheKey, *, java_changed: bool = True, changed_paths: Sequence[Path] | None = None
    ) -> dict[str, Any]:
        """Join one complete rebuild per index, including persistence and graph work.

        A cancelled waiter cannot cancel the shared job. Watcher changes arriving during a
        job are coalesced into a follow-up pass, so joining never loses a named edit.
        """
        if cache_key not in self.cache._tasks:
            return {"skipped": "not loaded"}
        if java_changed:
            self._rebuild_graph.add(cache_key)
        task = self._rebuild_tasks.get(cache_key)
        if task is None:
            # Queries serve the LRU generation; staleness must not start a second build mid-job.
            self.cache._revalidate_after[cache_key] = float("inf")
            task = asyncio.create_task(self._run_rebuild(cache_key, java_changed, changed_paths))
            self._rebuild_tasks[cache_key] = task
            task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        elif changed_paths:
            _merge_changes(self._rebuild_changes, cache_key, changed_paths)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                return {"skipped": "evicted before rebuild"}
            raise

    async def _run_rebuild(
        self, cache_key: CacheKey, java_changed: bool, changed_paths: Sequence[Path] | None
    ) -> dict[str, Any]:
        """Own a rebuild until its last phase and release build-only objects before trimming."""
        result: dict[str, Any] = {}
        try:
            while True:
                if (
                    cache_key in self._watch_tasks
                    and time.monotonic() < self._changed_at.get(cache_key, 0) + _QUIET_SECONDS
                ):
                    _merge_changes(self._watch_changes, cache_key, changed_paths)
                    return {"skipped": "tree still changing"}
                self.rebuilding.add(cache_key)
                try:
                    async with self._load_lock:
                        result = await self._rebuild_once(
                            cache_key, java_changed=java_changed, changed_paths=changed_paths
                        )
                    if java_changed or cache_key in self._rebuild_graph:
                        self._rebuild_graph.discard(cache_key)
                        result["graph_ms"] = await self._refresh_graph(cache_key[0], changed_paths)
                finally:
                    self.rebuilding.discard(cache_key)
                if cache_key not in self._rebuild_changes:
                    return result
                changes = self._rebuild_changes.pop(cache_key)
                changed_paths = None if changes is None else sorted(changes)
                java_changed = True
        except MemoryError as exc:
            traceback.clear_frames(exc.__traceback__)
            result = {"deferred": "allocation backstop", "knob": MEMORY_ENV}
            self.last_error[cache_key] = {"refused": f"Rebuild deferred by {MEMORY_ENV}", **result}
            return result
        except Exception as exc:
            # Keep traceback locations for logging, not failed builds' matrices and buffers.
            traceback.clear_frames(exc.__traceback__)
            raise
        finally:
            if self._rebuild_tasks.get(cache_key) is asyncio.current_task():
                self._rebuild_tasks.pop(cache_key, None)
                self._rebuild_changes.pop(cache_key, None)
                self.rebuilding.discard(cache_key)
                self._rebuild_graph.discard(cache_key)
            if cache_key not in self.cache._tasks:
                self.last_rebuild.pop(cache_key, None)
                self.last_error.pop(cache_key, None)
            # AIDEV-NOTE: the phase helper has returned, so its old index, persistence buffers
            # and graph working set are gone. Trimming before those phases kept their peak RSS.
            gc.collect()
            release_free_heap()
            if cache_key in self.cache._tasks:
                # Start the cooldown AFTER collection too; a tiny/refused build must still
                # serve its LRU generation rather than instantly launching a second build.
                self.cache._revalidate_after[cache_key] = time.monotonic() + max(1.0, result.get("ms", 0) / 1000 * 3)

    async def _rebuild_once(
        self, cache_key: CacheKey, *, java_changed: bool, changed_paths: Sequence[Path] | None
    ) -> dict[str, Any]:
        """Build beside the serving generation, then drop both local generations before graph work."""
        lock = self.rebuild_lock_for(cache_key)
        current = index = None
        started = time.monotonic()
        try:
            async with lock:
                if self.cache.is_building(cache_key):
                    return {"skipped": "initial build in progress"}
                current = next((index for key, index in self.cache.loaded() if key == cache_key), None)
                if current is None:
                    return {"skipped": "not loaded"}
                # AIDEV-NOTE: admission reserves copying/normalization and persistence scratch;
                # RLIMIT_AS is the allocation backstop if this estimate misses a shape.
                vectors = current._semantic_index.vectors.nbytes
                frozen = current._bm25_index._frozen
                postings = 0 if frozen is None else frozen.posting_docs.nbytes + frozen.posting_tf.nbytes
                reserve = 128 + (vectors + postings * 12 + len(current.chunks) * 512) / MIB
                work = await asyncio.to_thread(
                    measure_work,
                    Path(cache_key[0]),
                    cache_key[1],
                    current.exclude,
                    current._manifest,
                    changed_paths,
                    Path(cache_key[0]),
                )
                reserve += _work_reserve(work, current.embedder.dimensions)
                if not self._admit(reserve, keep=cache_key):
                    result = {"deferred": "memory budget", "reserve_mb": round(reserve), "knob": MEMORY_ENV}
                    self.last_error[cache_key] = {"refused": f"Rebuild deferred by {MEMORY_ENV}", **result}
                    return result
                try:
                    index, counts = await asyncio.to_thread(_mapped_rebuild, current, cache_key, changed_paths)
                except REFUSAL_TYPES as exc:
                    # Nothing was chunked, embedded or swapped, so the index that was serving
                    # this root before is still the one serving it now.
                    logger.warning("refused to rebuild %s: %s", cache_key[0], exc)
                    refusal = {"refused": str(exc), "at": time.time(), "knob": exc.knob}
                    self.last_error[cache_key] = refusal
                    return refusal
                elapsed = time.monotonic() - started
                # The swap is the only step a query could observe, and it is one dict write
                # on this event loop: an in-flight search keeps answering from the old index.
                self.cache.replace(cache_key, index, cooldown_seconds=float("inf"))
                if cache_key not in self.cache._tasks:
                    return {"skipped": "evicted during rebuild"}
        finally:
            # Only in-flight searches may still own the replaced generation, never a graph wait.
            current = None
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
        index = None
        self.last_rebuild[cache_key] = result
        self.last_error.pop(cache_key, None)
        return result

    async def _refresh_graph(self, root: str, changed_paths: Sequence[Path] | None = None) -> int | None:
        """Incrementally refresh the symbol graph for a root that already has one.

        The watcher's change set is handed straight to the graph build, which then stats the
        named files instead of walking the workspace for them. The build waits for any other
        process writing the graph rather than skipping, so the change set always lands.
        """
        from zemble.graph.store import graph_present

        # The predicate is the FILE, not a readable graph: a malformed store is exactly the one
        # the watcher must keep driving, and `build_graph` rebuilds it from source.
        if not await asyncio.to_thread(graph_present, root):
            return None
        started = time.monotonic()
        try:
            await self.graphs.ensure(root, time.monotonic() + 900, refresh=True, changed_paths=changed_paths)
        except Exception:
            logger.error("The symbol graph for %s is now STALE: its refresh failed", root, exc_info=True)
            return None
        return round((time.monotonic() - started) * 1000)

    async def with_graph(self, root: str, work: Callable[[Any], Any]) -> Any:
        """Read a prepared immutable graph without borrowing the construction slot."""
        await self.graphs.ensure(root, time.monotonic() + 25)
        return await asyncio.to_thread(_with_graph, root, work, fresh=True)

    async def _execute_read(self, handler: Handler, args: dict[str, Any], *, memo: bool = True) -> Any:
        """Pin a request's serving generation until its actual work finishes."""
        task = asyncio.current_task()
        self._request_tasks.add(task)
        try:
            if memo and handler in {_cmd_home, _cmd_graph}:
                indexes = tuple(
                    (key[0], weakref.ref(index)) for key, index in sorted(self.cache.loaded(), key=lambda x: x[0][0])
                )
                generations = tuple(sorted(self.graphs.generations.items()))
                key = (handler.__name__, json.dumps(args, sort_keys=True), indexes, generations)
                return await self.read_cache.get(key, lambda: self._execute_read(handler, args, memo=False))
            return await handler(self, args)
        finally:
            self._query_keys.pop(task, None)
            self._request_tasks.discard(task)

    async def _bounded_prepare(self, command: str, args: dict[str, Any], deadline: float) -> None:
        """Bound cold waiters independently of immutable read execution slots."""
        preparing = getattr(self, "_preparing", 0)
        if preparing >= self.reads.slots + self.reads.queue_limit:
            raise AdmissionBusy("construction waiting room is full")
        self._preparing = preparing + 1
        try:
            await self._prepare_read(command, args, deadline)
        finally:
            self._preparing -= 1

    async def _prepare_read(self, command: str, args: dict[str, Any], deadline: float) -> None:
        """Await shared cold jobs outside the finite read execution pool."""
        from zemble.workspace import resolve_home_root

        prepared = {**args, "path": str(resolve_home_root(_root_of(args)))} if command == "home" else args
        if command in {"search", "find_related", "home", "explain", "stats", "architectural"}:
            async with asyncio.timeout_at(deadline):
                await self.index_for(prepared)
        if command in {"graph", "home", "explain", "outline", "signatures", "find_related", "architectural"}:
            await self.graphs.ensure(_root_of(prepared), deadline)

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
        request_task = asyncio.current_task()
        self._request_tasks.add(request_task)
        try:
            deadline = request_deadline(request)
            if command in {"ping", "status", "shutdown", "refresh"}:
                result = await handler(self, args)
            else:
                # AIDEV-NOTE: cold construction waiters own no read permit. A disconnected home
                # request joins a durable graph job without blocking unrelated warm searches.
                self.reads.check_room()
                await self._bounded_prepare(str(command), args, deadline)
                if not self._admit(32):
                    raise MemoryRefused(f"No query headroom within {self.max_rss_mb} MiB ({MEMORY_ENV}).")
                result = await self.reads.run(lambda: self._execute_read(handler, args), deadline)
        except TimeoutError:
            return {
                "id": request_id,
                "ok": False,
                "error": "construction deadline expired; retry",
                "kind": ErrorKind.BUSY.value if request.get(ACCEPTS_BUSY_FIELD) else ErrorKind.REFUSED.value,
                "retry_after_ms": 250,
            }
        except CommandBusy as exc:
            # AIDEV-NOTE: pre-fix clients fall back on unknown error kinds. REFUSED is their
            # known no-fallback lane; advertise BUSY only to a caller that declares support.
            kind = ErrorKind.BUSY if request.get(ACCEPTS_BUSY_FIELD) is True else ErrorKind.REFUSED
            message = str(exc) if kind is ErrorKind.BUSY else f"Daemon busy, retry: {exc}"
            return {
                "id": request_id,
                "ok": False,
                "error": message,
                "kind": kind.value,
                "retry_after_ms": getattr(exc, "retry_after_ms", 250),
                "admission": self.reads.status(),
            }
        except MemoryError:
            gc.collect()
            release_free_heap()
            return {
                "id": request_id,
                "ok": False,
                "error": f"Daemon allocation refused by {MEMORY_ENV}",
                "kind": ErrorKind.REFUSED.value,
            }
        except Exception as exc:
            refused = isinstance(exc, REFUSAL_TYPES)
            kind = ErrorKind.REFUSED if refused else ErrorKind.FAILED
            # A refusal carries no traceback worth printing, but it does carry the numbers and
            # the knob that decided it. Logging only "refused" left 322 log lines saying nothing.
            logger.warning("Command %r %s: %s", command, "refused" if refused else "failed", exc, exc_info=not refused)
            message = str(exc) if refused else f"{type(exc).__name__}: {exc}"
            return {"id": request_id, "ok": False, "error": message, "kind": kind.value}
        finally:
            self.last_request_at = time.monotonic()
            self._query_keys.pop(request_task, None)
            self._request_tasks.discard(request_task)
            # Concurrent queries can finish after the rebuild's trim; release their freed buffers too.
            release_free_heap()
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
                # Every answer names the code that produced it, so a client can warn about a
                # daemon still running the snapshot it started with (see zemble.runtime).
                writer.write(encode({**response, **identity_envelope()}))
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
        for task in self._watch_tasks.values():
            task.cancel()
        self.stop_event.set()


# -- command table ----------------------------------------------------------
# AIDEV-NOTE: one entry per command; a new daemon command is one function plus one line here.


async def _cmd_ping(daemon: Daemon, args: dict[str, Any]) -> Any:
    """Answer that the daemon is alive, naming the code snapshot it runs."""
    current = identity()
    return {
        "pong": True,
        "pid": os.getpid(),
        "zemble_version": current.zemble_version,
        "source_revision": current.source_revision,
    }


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
                "last_error": daemon.last_error.get(cache_key),
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
        "max_rss_mb": daemon.max_rss_mb,
        "virtual_mb": round(virtual_mb(), 1),
        "read_admission": daemon.reads.status(),
        "graph_construction": daemon.graphs.status(),
        "serving_virtual_limit_mb": daemon.max_virtual_mb,
        "memory_configuration": {
            "rss_mb": daemon.max_rss_mb,
            "virtual_mb": daemon.max_virtual_mb,
            "total_mb": daemon.graphs.total_mb,
            "construction_mb": daemon.graphs.build_mb,
        },
        "quiet_seconds": _QUIET_SECONDS,
        "socket": str(socket_path()),
        "runtime": _runtime_status(),
        "indexes": indexes,
        "building": building,
        "pending_reindex": sorted(
            {key[0] for key in daemon.pending | (set(daemon._rebuild_tasks) - daemon.rebuilding)}
        ),
    }


def _runtime_status() -> dict[str, Any]:
    """Return the identity of the code this daemon runs, flattened with its staleness flag."""
    payload = status_payload(identity())
    return {**payload["identity"], "stale": payload["stale"], "note": payload["note"]}


async def _cmd_search(daemon: Daemon, args: dict[str, Any]) -> Any:
    """Search one root, returning the same payload shape the CLI and MCP print."""
    _cache_key, index = await daemon.index_for(args)
    query = str(args.get("query", ""))
    max_snippet_lines = args.get("max_snippet_lines")
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
    _cache_key, index = await daemon.index_for(args)
    file_path = str(args.get("file_path", ""))
    line = int(args.get("line", 0))
    max_snippet_lines = args.get("max_snippet_lines")
    chunk = index.chunk_at(file_path, line)
    if chunk is None:
        return {"error": describe_unresolved_location(index, file_path, line), "unresolved_location": True}
    from zemble.evidence.related import related_payload

    return await daemon.with_graph(
        _root_of(args),
        lambda graph: related_payload(
            index, graph, Path(_root_of(args)), file_path, line, int(args.get("top_k", 5)), max_snippet_lines
        ),
    )


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
    from zemble.graph.mcp import answer

    path = str(args.get("path", ""))
    if not path:
        raise ValueError("missing 'path'")
    command = str(args.get("command", "ensure"))
    if command == "ensure":
        await daemon.graphs.ensure(path, time.monotonic() + 25)
        return {"ensured": True}
    kinds = args.get("kinds")
    extra: dict[str, Any] = {}
    if args.get("limit") is not None:
        extra["limit"] = int(args["limit"])
    if command == "neighbors":
        from zemble.graph.model import EdgeKind

        extra.update({"hops": int(args.get("hops", 1)), "kinds": [EdgeKind(kind) for kind in kinds] if kinds else None})
    await daemon.graphs.ensure(path, time.monotonic() + 25)
    return await asyncio.to_thread(answer, path, str(args.get("symbol", "")), command, fresh=True, **extra)


def _with_graph(root: str, work: Callable[[Any], Any], *, fresh: bool = False) -> Any:
    """Run one graph question against a fresh provider, building the graph if needed.

    Called inside a worker thread: `ensure_graph` and the sqlite provider are both blocking,
    and a provider is not shared across threads.
    """
    from zemble.graph.cli import ensure_graph
    from zemble.graph.provider import open_provider

    if not fresh:
        ensure_graph(root, allow_daemon=False)
    provider = open_provider(root)
    try:
        return work(provider)
    finally:
        provider.close()


def _root_of(args: dict[str, Any]) -> str:
    """Read the workspace root out of a request, refusing a missing one."""
    root = args.get("path")
    if not root:
        raise ValueError("missing 'path'")
    root = str(root)
    if not is_git_url(root) and not os.path.isabs(root):
        # The daemon's cwd is "/"; resolving a relative path here would index the whole filesystem.
        raise ValueError(f"daemon needs an absolute path, got {root!r}")
    return root


async def _cmd_explain(daemon: Daemon, args: dict[str, Any]) -> Any:
    """Build an evidence bundle over the warm index and the daemon's own symbol graph.

    The graph is opened for the REQUESTED path (its own, or an ancestor's filtered view), even
    when the index is an ancestor's view: both speak paths relative to that path, as `outline`
    and `dupes` for it do, so chunks and symbols join on one spelling.
    """
    from zemble.evidence.answers import DEFAULT_BUDGET, DEFAULT_TOP_K, explain_payload

    _cache_key, index = await daemon.index_for(args)
    root = _root_of(args)
    query = str(args.get("query", ""))
    budget = int(args.get("budget", DEFAULT_BUDGET))
    top_k = int(args.get("top_k", DEFAULT_TOP_K))
    return await daemon.with_graph(
        root,
        lambda graph: explain_payload(index, graph, query, budget, top_k),
    )


async def _cmd_outline(daemon: Daemon, args: dict[str, Any]) -> Any:
    """Outline a file or a type from the daemon's graph."""
    from zemble.evidence.answers import outline_payload

    root = _root_of(args)
    target = str(args.get("target", ""))
    members = args.get("members")
    return await daemon.with_graph(root, lambda graph: outline_payload(graph, target, members))


async def _cmd_signatures(daemon: Daemon, args: dict[str, Any]) -> Any:
    """Describe one symbol and its exactly resolved call sites."""
    from zemble.evidence.answers import signatures_payload

    root = _root_of(args)
    symbol = str(args.get("symbol", ""))
    return await daemon.with_graph(root, lambda graph: signatures_payload(graph, symbol))


async def _cmd_home(daemon: Daemon, args: dict[str, Any]) -> Any:
    """Answer where a capability belongs, over the warm index and the daemon's graph."""
    from zemble.home.answers import DEFAULT_TOP_K, home_payload
    from zemble.home.config import HomeConfig
    from zemble.workspace import resolve_home_root

    requested = str(args.get("requested_path", _root_of(args)))
    root = str(resolve_home_root(_root_of(args)))
    description = str(args.get("description", ""))
    top_k = int(args.get("top_k", DEFAULT_TOP_K))
    config = await asyncio.to_thread(HomeConfig.load, root)
    _cache_key, index = await daemon.index_for({**args, "path": root})
    return await daemon.with_graph(
        root,
        lambda graph: home_payload(index, graph, config, description, top_k, requested_root=requested),
    )


async def _cmd_refresh(daemon: Daemon, args: dict[str, Any]) -> Any:
    """Force a rebuild check for a root, loading it first if it is not resident."""
    async with daemon._operation_lock:
        cache_key, _index = await daemon.index_for(args)
    del _index
    return await daemon.rebuild(cache_key)


async def _cmd_evict(daemon: Daemon, args: dict[str, Any]) -> Any:
    """Drop one root from memory, stopping its watcher."""
    path = args.get("path")
    if not path:
        raise ValueError("missing 'path'")
    cache_key = compute_cache_key(str(path), args.get("ref"), _content_types(args.get("content")))
    keys = [key for key in daemon.cache._tasks if key[0] == cache_key[0]]
    was_loaded = bool(keys)
    for key in keys:
        daemon.cache.evict(key)
    return {"evicted": was_loaded, "root": cache_key[0]}


async def _cmd_shutdown(daemon: Daemon, args: dict[str, Any]) -> Any:
    """Stop the daemon after this response has been written."""
    asyncio.get_running_loop().call_later(0.05, daemon.shutdown)
    return {"stopping": True, "pid": os.getpid()}


async def _cmd_architectural(daemon: Daemon, args: dict[str, Any]) -> Any:
    """Reviewable architectural evidence alongside, never inside, literal clone classes."""
    from zemble.dedup.architectural import architectural_candidates

    _key, index = await daemon.index_for(args)
    files = sorted({index._public_chunk(index.chunks[row]).file_path for row in range(len(index.chunks))})
    return await daemon.with_graph(
        _root_of(args),
        lambda graph: architectural_candidates(
            Path(_root_of(args)), files, graph, int(args.get("limit", 100)), int(args.get("min_files", 1))
        ),
    )


COMMANDS: dict[str, Handler] = {
    "ping": _cmd_ping,
    "status": _cmd_status,
    "search": _cmd_search,
    "find_related": _cmd_find_related,
    "stats": _cmd_stats,
    "graph": _cmd_graph,
    "explain": _cmd_explain,
    "outline": _cmd_outline,
    "signatures": _cmd_signatures,
    "home": _cmd_home,
    "architectural": _cmd_architectural,
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
    # AIDEV-NOTE: an old MCP client inherits the loaded flag but not settings added since it
    # started. An explicit path bypasses that flag; genuine shell environment overrides still win.
    load_user_env(user_env_path())
    # A daemon must never route its own work through a daemon client: that is a deadlock
    # on its own socket, and every shared code path (graph ensure, search) can reach one.
    client.disable_for_this_process("running inside the daemon")
    daemon = Daemon(max_indexes=max_indexes, idle_minutes=idle_minutes, watch=watch)
    daemon.max_virtual_mb = allocation_backstop(daemon.max_virtual_mb)
    lock = _acquire_lock()
    path = socket_path(create_dir=True)
    if path.exists():
        # We hold the lock, so any socket file here belongs to a daemon that is gone.
        path.unlink()
    from zemble.graph import store

    previous_workers = store.DEFAULT_WORKERS
    store.DEFAULT_WORKERS = 1
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
        await daemon.graphs.close()
        await daemon.reads.drain()
        store.DEFAULT_WORKERS = previous_workers
        with contextlib.suppress(OSError):
            path.unlink()
        with contextlib.suppress(OSError):
            pid_path().unlink()
        lock.close()
        logger.info("zemble daemon %d stopped", os.getpid())

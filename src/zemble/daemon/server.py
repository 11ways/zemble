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
import time
import traceback
import weakref
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from zemble.cache import (
    compatible_cache,
    covering_content,
    find_index_from_cache_folder,
    load_manifest_for_incremental,
    resolve_index_root,
    retire_covered_indexes,
)
from zemble.daemon import client
from zemble.daemon.admission import AdmissionBusy, ReadAdmission, request_deadline
from zemble.daemon.graph_jobs import GraphJobs
from zemble.daemon.memory import (
    MEMORY_ENV,
    MIB,
    MemoryBusy,
    ReadOutgrown,
    allocation_backstop,
    data_limit_mb,
    default_budget_mb,
    private_mb,
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
from zemble.daemon.watch import RootWatcher, WatchSpec
from zemble.embedding.base import Embedder
from zemble.graph.facts import matches_facts_glob
from zemble.index import ZembleIndex
from zemble.index.chunk_store import ChunkList
from zemble.index.create import change_set_moves_anything
from zemble.index.files import get_extensions
from zemble.index.scope import TreeEstimate, estimate_tree, measure_work, require_declared_scope
from zemble.index.types import PreviousIndex
from zemble.index_cache import CacheKey, IndexCache, compute_cache_key
from zemble.refusal import Refused
from zemble.runtime.identity import identity, status_payload
from zemble.runtime.memory import release_free_heap
from zemble.types import ContentType
from zemble.userenv import drop_proxy_settings, load_user_env, user_env_path
from zemble.utils import describe_unresolved_location, format_results, format_stats, is_git_url

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
#: The rebuild answer for a root whose watcher saw a change within the quiet window; the watcher runs it later.
_STILL_CHANGING = "tree still changing"
_MAX_CHANGED_PATHS = 4096


#: Private memory a build needs whatever it writes: one embedding batch, one merge block of
#: postings, the vocabulary, and a few integers per chunk (see `zemble.index.create.write_index`).
_BUILD_BASE_MB = 192.0
#: Private bytes a build holds per byte of NEW source: its BM25 postings as flat integers.
_BUILD_BYTES_PER_SOURCE_BYTE = 3


def _work_reserve(estimate: TreeEstimate) -> float:
    """Reserve what a streaming build of *estimate*'s new source holds in private memory."""
    return _BUILD_BASE_MB + estimate.bytes * _BUILD_BYTES_PER_SOURCE_BYTE / MIB


def _load_work(
    path: str, content: tuple[ContentType, ...], exclude: tuple[str, ...], embedder_id: str, loaded: set[str]
) -> TreeEstimate:
    """Measure what loading *path* will chunk: only what the stored index answering it does not cover unchanged."""
    root, _prefix = resolve_index_root(path, embedder_id, content, None, loaded)
    stored = covering_content(root, embedder_id, content, None, ())
    manifest = load_manifest_for_incremental(root, embedder_id, stored)
    if manifest is None:
        return estimate_tree(Path(path), content, exclude)
    return estimate_tree(Path(root), stored, (), manifest)


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
        """Watch the root before chunking; a watched root with a stored index starts from it, unbuilt.

        AIDEV-NOTE: catching a stored index up before answering made a cold request wait for a scan
        of every file plus whatever changed while no daemon ran: 58 s on the zenit workspace, past
        a 60 s caller. The stored generation answers at once; the catch-up is the full rescan the
        watcher queues once its watches are registered, which waits for this load to finish.
        """
        self._on_build(cache_key)
        if self.watch_owned and not is_git_url(source):
            stored = await asyncio.to_thread(_load_stored, cache_key, embedder, exclude)
            if stored is not None:
                return stored
        return await super()._build_tracked(source, ref, embedder, cache_key, exclude)

    async def _evict_if_stale(self, cache_key: CacheKey) -> None:
        """Leave freshness to the watcher instead of rebuilding a churning tree from a query."""
        if not self.watch_owned:
            await super()._evict_if_stale(cache_key)


def _load_stored(cache_key: CacheKey, embedder: Embedder, exclude: Sequence[str]) -> ZembleIndex | None:
    """Load a root's compatible stored index as it is, or None when it has none worth loading."""
    stored = compatible_cache(cache_key[0], embedder.model_id, cache_key[1], exclude=exclude)
    if stored is None:
        return None
    try:
        return ZembleIndex.load_from_disk(stored, embedder=embedder)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("The stored index of %s is unreadable (%s); building it instead", cache_key[0], exc)
        return None


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
    current: ZembleIndex, cache_key: CacheKey, changed_paths: Sequence[Path] | None = None
) -> tuple[ZembleIndex, dict[str, int]]:
    """Write and publish the next generation of a served root, returning it loaded and what moved.

    A change set that moves nothing the index holds - a facts file, build output, a touched file -
    hands back *current* itself rather than writing a copy of it.

    :param current: The index currently serving this root; it is read, never written.
    :param cache_key: The root and content types being rebuilt.
    :param changed_paths: The paths a watcher saw move; None re-walks the whole tree.
    :return: The replacement index, and counts of added, changed and removed files.
    """
    if not isinstance(current.chunks, ChunkList):
        raise TypeError("a served index is always loaded from its stores")
    root = Path(cache_key[0])
    content = cache_key[1]
    if changed_paths is not None and not change_set_moves_anything(
        root, changed_paths, content, root, current._manifest, current.exclude
    ):
        return current, {"added": 0, "removed": 0, "changed": 0}
    previous = PreviousIndex(
        chunks=current.chunks,
        vectors=current._semantic_index.vectors,
        manifest=current._manifest,
        bm25_index=current._bm25_index,
        definitions=current._definitions,
    )
    before = dict(current._manifest)
    index, written = ZembleIndex.build(
        root,
        current.embedder,
        find_index_from_cache_folder(cache_key[0], content, current.exclude),
        content=content,
        capsules=current._capsules,
        exclude=current.exclude,
        previous=previous,
        changed_paths=changed_paths,
    )
    retire_covered_indexes(cache_key[0], content, current.exclude)
    # The writer's scratch is garbage now; hand it back before the graph phase can wait for minutes.
    gc.collect()
    release_free_heap()
    manifest = written.manifest
    counts = {
        "added": len(manifest.keys() - before.keys()),
        "removed": len(before.keys() - manifest.keys()),
        "changed": sum(
            1 for path, entry in manifest.items() if path in before and before[path].mtime_ns != entry.mtime_ns
        ),
    }
    return index, counts


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
        # AIDEV-NOTE: graph construction has a lock of its own; it takes the index construction lock
        # only while it lowers this process's data limit and starts its worker, so an index load
        # never waits out a whole graph build (minutes on a busy workspace) - it fits under the
        # lowered ceiling (`_admit`) or is asked again.
        self.graphs = GraphJobs(self._load_lock, self.max_rss_mb, make_room=self._admit)
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
        #: What the pass running for a root rescans: its named paths, or None for the whole tree.
        self._rebuild_scope: dict[CacheKey, Sequence[Path] | None] = {}
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
        narrowed = index.for_content(content)
        filtered = narrowed.filtered(paths, exclude) if narrowed is not None else None
        if filtered is None:
            raise ValueError(f"No indexed file under {path} survives paths={list(paths)} exclude={list(exclude)}")
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
        """Evict idle LRU stores until private-memory headroom covers the work under the current ceiling.

        The ceiling is the lower of the budget and the data limit, which a running graph construction lowers.
        """
        while private_mb() + reserve_mb > min(self.max_rss_mb, data_limit_mb()):
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

    def _short_of_memory(self, need_mb: float, what: str) -> CommandBusy | ReadOutgrown:
        """Name work that does not fit: busy while work that will end holds memory, else outgrown, never refused.

        Busy is asked again until it fits; outgrown is answered by the caller in a process of its own.
        """
        ceiling = min(self.max_rss_mb, data_limit_mb())
        own = asyncio.current_task()
        ending = (
            any(not job.done() for job in self.graphs.jobs.values())
            or bool(self.rebuilding)
            or any(self.cache.is_building(key) for key in list(self.cache._tasks))
            or any(task is not own for task in self._query_keys)
        )
        if ending:
            return MemoryBusy(f"{what} waits for ~{need_mb:.0f} MiB of private memory that running work holds")
        free = max(ceiling - private_mb(), 0.0)
        return ReadOutgrown(
            f"{what} needs ~{need_mb:.0f} MiB of private memory and this daemon has {free:.0f} MiB free under "
            f"its {ceiling:.0f} MiB ceiling ({MEMORY_ENV}); answer it in a process of its own."
        )

    def admit_read(self, need_mb: float, what: str) -> None:
        """Make private-memory room for one bulk read, or say that only another process can hold it.

        The ceiling is the lower of the budget and the current `RLIMIT_DATA`, which a running graph
        construction lowers to the serving reserve (`GraphJobs._build_once`).

        :raises ReadOutgrown: If evicting idle indexes cannot free `need_mb` under that ceiling.
        """
        ceiling = min(self.max_rss_mb, data_limit_mb())
        if self._admit(need_mb) and private_mb() + need_mb <= ceiling:
            return
        free = max(ceiling - private_mb(), 0.0)
        raise ReadOutgrown(
            f"{what} needs ~{need_mb:.0f} MiB of private memory and this daemon has {free:.0f} MiB free under "
            f"its {ceiling:.0f} MiB ceiling ({MEMORY_ENV}); answer it in a process of its own."
        )

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
        joined = await self._join_build(requested[0], path, content)
        if joined is not None:
            return joined
        gc.collect()
        release_free_heap()
        reserve = _BUILD_BASE_MB
        if not is_git_url(path):
            require_declared_scope(Path(path))
            estimate = await asyncio.to_thread(
                _load_work, path, content, exclude, self.cache.embedder.model_id, self.cache.loaded_roots(content)
            )
            reserve = _work_reserve(estimate)
        if not self._admit(reserve):
            raise self._short_of_memory(reserve, f"Loading {path}")
        return await self.cache.get_with_key(path, ref=ref, content=content, exclude=exclude)

    async def _join_build(
        self, requested: str, path: str, content: tuple[ContentType, ...]
    ) -> tuple[CacheKey, ZembleIndex] | None:
        """Wait for an in-flight build of a covering root instead of reserving a second load of it.

        That build was admitted when it started; a request arriving after its first waiter gave
        up would otherwise ask for the same headroom again and be refused by the build's own use.
        """
        for key in [key for key in self.cache._tasks if self.cache.is_building(key)]:
            if not set(content) <= set(key[1]):
                continue
            if key[0] == requested:
                prefix = None
            elif not is_git_url(path) and Path(requested).is_relative_to(Path(key[0])):
                prefix = Path(requested).relative_to(Path(key[0])).as_posix()
            else:
                continue
            index = await asyncio.shield(self.cache._tasks[key])
            view = index if prefix is None else index.subtree(prefix)
            if view is not None:
                return key, view
        return None

    def _ensure_watcher(self, cache_key: CacheKey) -> None:
        """Start watching a local root the first time it is served."""
        if not self.watch_enabled or cache_key in self.watchers or is_git_url(cache_key[0]):
            return
        root = Path(cache_key[0])
        if not root.is_dir():
            return
        extensions = tuple(sorted(set(get_extensions(cache_key[1])) | _GRAPH_EXTENSIONS))
        watcher = RootWatcher(
            WatchSpec(root, extensions, facts=True),
            lambda paths: self._on_change(cache_key, paths),
            # Edits made before the watches existed were never seen: one full rescan covers them.
            on_ready=lambda: self._note_change(cache_key, None),
        )
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
        self._note_change(cache_key, paths)

    def _note_change(self, cache_key: CacheKey, paths: set[Path] | None) -> None:
        """Queue a change set for a resident root's quiet-period rebuild; None means rescan everything."""
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
                loading = self.cache._tasks[cache_key]
                if not loading.done():
                    # A rebuild skips a root whose first load is running and its change set would be
                    # lost; apply it after that load instead. A failed load evicts the root.
                    with contextlib.suppress(Exception):
                        await asyncio.shield(loading)
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
        elif changed_paths is None:
            # Joining a pass over named paths would answer a full rescan with a partial one: it follows
            # that pass instead. A pass that already rescans everything is simply shared.
            if self._rebuild_scope.get(cache_key, ()) is not None:
                _merge_changes(self._rebuild_changes, cache_key, None)
        elif changed_paths:
            _merge_changes(self._rebuild_changes, cache_key, changed_paths)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                return {"skipped": "evicted before rebuild"}
            raise

    async def bring_current(self, cache_key: CacheKey) -> dict[str, Any]:
        """Rescan a resident root's index until a pass runs on a tree at rest, and return that pass's result.

        A pass is skipped while the watcher still sees changes; it is then retried after the quiet
        window rather than reported, because the caller asked for an index that is current. The
        symbol graph is not part of it: on the zenit workspace its refresh took 107 s of a 125 s
        `zemble index`, and `zemble graph build` is the command that asks for it.
        """
        while True:
            result = await self.rebuild(cache_key, java_changed=False, changed_paths=None)
            if result.get("skipped") != _STILL_CHANGING:
                return result
            await asyncio.sleep(_QUIET_SECONDS)

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
                    return {"skipped": _STILL_CHANGING}
                self.rebuilding.add(cache_key)
                self._rebuild_scope[cache_key] = changed_paths
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
                self._rebuild_scope.pop(cache_key, None)
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
                # AIDEV-NOTE: a rebuild streams its generation to disk, so its private memory is
                # bounded by what is new, not by the index; RLIMIT_DATA is the backstop if this misses.
                work = await asyncio.to_thread(
                    measure_work,
                    Path(cache_key[0]),
                    cache_key[1],
                    current.exclude,
                    current._manifest,
                    changed_paths,
                    Path(cache_key[0]),
                )
                reserve = _work_reserve(work)
                if not self._admit(reserve, keep=cache_key):
                    result = {"deferred": "memory budget", "reserve_mb": round(reserve), "knob": MEMORY_ENV}
                    self.last_error[cache_key] = {"refused": f"Rebuild deferred by {MEMORY_ENV}", **result}
                    return result
                try:
                    index, counts = await asyncio.to_thread(rebuild_index, current, cache_key, changed_paths)
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
            # The watcher waits its refresh out: no client is behind it, and a refresh it gave up on
            # would still run, with this log line claiming it had failed.
            await self.graphs.ensure(root, None, refresh=True, changed_paths=changed_paths)
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
        if command in {"graph", "home", "explain", "outline", "signatures", "health", "find_related", "architectural"}:
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
                    raise self._short_of_memory(32, "This query")
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
            # Outgrown, never refused: the request fits in a process of its own, and the client answers it there.
            return {
                "id": request_id,
                "ok": False,
                "error": f"Daemon allocation refused by {MEMORY_ENV}",
                "kind": ErrorKind.OUTGROWN.value,
            }
        except Exception as exc:
            return _failure_response(request_id, str(command), exc)
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
        "private_mb": round(private_mb(), 1),
        "read_admission": daemon.reads.status(),
        "graph_construction": daemon.graphs.status(),
        "memory_configuration": {
            "private_mb": daemon.max_rss_mb,
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
    return format_stats(str(args.get("path")), index)


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


def _failure_response(request_id: Any, command: str, exc: Exception) -> dict[str, Any]:
    """Shape one failed command as its error kind: outgrown, refused, or failed with a traceback logged."""
    if isinstance(exc, ReadOutgrown):
        logger.info("Command %r outgrown: %s", command, exc)
        return {"id": request_id, "ok": False, "error": str(exc), "kind": ErrorKind.OUTGROWN.value}
    refused = isinstance(exc, REFUSAL_TYPES)
    kind = ErrorKind.REFUSED if refused else ErrorKind.FAILED
    # A refusal carries no traceback worth printing, but it does carry the numbers and
    # the knob that decided it. Logging only "refused" left 322 log lines saying nothing.
    logger.warning("Command %r %s: %s", command, "refused" if refused else "failed", exc, exc_info=not refused)
    message = str(exc) if refused else f"{type(exc).__name__}: {exc}"
    return {"id": request_id, "ok": False, "error": message, "kind": kind.value}


def _with_graph(root: str, work: Callable[[Any], Any], *, fresh: bool = False) -> Any:
    """Run one graph question against a fresh provider, building the graph if needed.

    Called inside a worker thread: `ensure_graph` and the sqlite provider are both blocking,
    and a provider is not shared across threads.
    """
    from zemble.graph.cli import ensure_graph
    from zemble.graph.provider import open_provider
    from zemble.graph.store import GraphStoreStale, build_graph, resolve_graph_root

    if not fresh:
        ensure_graph(root, allow_daemon=False)
    try:
        provider = open_provider(root)
    except GraphStoreStale:
        # A version of an older format (the daemon came up before its first build migrated it) is migrated by a
        # build first; one still stale after it raises GraphStoreStale, which names the format: never a bad read.
        build_graph(resolve_graph_root(root)[0])
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
    everything = bool(args.get("all", False))
    return await daemon.with_graph(root, lambda graph: outline_payload(graph, target, members, everything))


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


async def _cmd_health(daemon: Daemon, args: dict[str, Any]) -> Any:
    """Report code-shape findings over the daemon's graph and the nearest declared workspace config.

    The duplication tier reads the per-root unit index (`zemble.dedup.unitcache`), never the search index.
    """
    from zemble.health.answers import HealthRequest, health_payload
    from zemble.home.config import HomeConfig
    from zemble.workspace import resolve_home_root

    root = _root_of(args)
    config = await asyncio.to_thread(HomeConfig.load, resolve_home_root(root))
    request = HealthRequest.from_args(args)
    await daemon.graphs.ensure(root, time.monotonic() + 25)
    daemon.admit_read(await asyncio.to_thread(_health_need_mb, root), "health")
    try:
        return await daemon.with_graph(root, lambda graph: health_payload(graph, config, request))
    except MemoryError as exc:
        raise ReadOutgrown(
            f"health outgrew this daemon's private memory mid-read ({MEMORY_ENV}); answer it in a process of its own."
        ) from exc


def _health_need_mb(root: str) -> float:
    """Estimate one health read's private memory from the graph's row counts; runs in a worker thread."""
    from zemble.graph.provider import open_provider
    from zemble.health.collect import read_estimate_mb

    provider = open_provider(root)
    try:
        return read_estimate_mb(provider.row_counts())
    finally:
        provider.close()


async def _cmd_refresh(daemon: Daemon, args: dict[str, Any]) -> Any:
    """Load a root, rescan it until a pass finds the tree at rest, and report it; `zemble index` asks this."""
    async with daemon._operation_lock:
        cache_key, _index = await daemon.index_for(args)
    del _index
    rebuild = None if is_git_url(cache_key[0]) else await daemon.bring_current(cache_key)
    _key, index = await daemon.index_for(args)
    return {**format_stats(str(args.get("path")), index), "root": cache_key[0], "rebuild": rebuild}


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
    "health": _cmd_health,
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
    # AIDEV-NOTE: the daemon outlives the process that spawned it, so a proxy from that
    # process's environment (an agent session's per-session proxy) dies under it and every
    # outbound request is refused. Drop inherited proxies before the user env file is read,
    # so a proxy the daemon really needs is configured there, where it is stable.
    dropped_proxies = drop_proxy_settings()
    # AIDEV-NOTE: an old MCP client inherits the loaded flag but not settings added since it
    # started. An explicit path bypasses that flag; genuine shell environment overrides still win.
    load_user_env(user_env_path())
    if dropped_proxies:
        logger.info("Ignoring inherited proxy settings %s; set a daemon proxy in %s", dropped_proxies, user_env_path())
    # A daemon must never route its own work through a daemon client: that is a deadlock
    # on its own socket, and every shared code path (graph ensure, search) can reach one.
    client.disable_for_this_process("running inside the daemon")
    daemon = Daemon(max_indexes=max_indexes, idle_minutes=idle_minutes, watch=watch)
    daemon.max_rss_mb = allocation_backstop(daemon.max_rss_mb)
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

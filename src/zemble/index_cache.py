"""In-memory cache of built indexes, shared by the MCP server and the daemon.

Lives outside both so one process-wide implementation serves the in-process path
and the daemon, and a fix to eviction or build deduplication is made once.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path

from zemble.cache import get_validated_cache, indexed_ancestor_hint, resolve_index_root, save_index_to_cache
from zemble.embedding.base import Embedder
from zemble.embedding.pricing import EmbeddingBudgetExceeded
from zemble.embedding.registry import load_embedder
from zemble.index import ZembleIndex
from zemble.types import ContentType
from zemble.utils import is_git_url

logger = logging.getLogger(__name__)

CACHE_MAX_SIZE = 10  # Max number of cached indexes to keep in memory
MIN_REVALIDATE_FACTOR = 3  # Don't recheck staleness sooner than this many times the last build's duration

CacheKey = tuple[str, tuple[ContentType, ...]]


def compute_cache_key(
    source: str,
    ref: str | None = None,
    content: Sequence[ContentType] = (ContentType.CODE,),
) -> CacheKey:
    """Compute the canonical key for an exact index variant."""
    is_git = is_git_url(source)
    source_key = (f"{source}@{ref}" if ref else source) if is_git else str(Path(source).resolve())
    normalized = tuple(content_type for content_type in ContentType if content_type in content)
    return source_key, normalized


class IndexCache:
    """Cache of indexed repos and local paths for the lifetime of a process."""

    def __init__(self, max_size: int = CACHE_MAX_SIZE, on_evict: Callable[[CacheKey], None] | None = None) -> None:
        """Initialise an empty cache.

        :param max_size: How many index variants to keep resident before evicting the least recently used.
        :param on_evict: Called with the key of every entry that leaves the cache, for teardown of attached state.
        """
        self._embedder: Embedder | None = None
        self._model_error: BaseException | None = None
        self._model_ready = asyncio.Event()
        self._tasks: OrderedDict[CacheKey, asyncio.Future[ZembleIndex]] = OrderedDict()  # ordered for LRU eviction
        self._revalidate_after: dict[CacheKey, float] = {}
        self._max_size = max_size
        self._on_evict = on_evict
        self.last_used: dict[CacheKey, float] = {}

    @property
    def embedder(self) -> Embedder | None:
        """The embedder every index in this cache was built with, once loaded."""
        return self._embedder

    async def load_embedder_once(self) -> None:
        """Load the default embedder into the cache, recording a load failure instead of raising."""
        if self._model_ready.is_set():
            return
        try:
            embedder = await asyncio.to_thread(load_embedder)
            # Touch dimensions so the model is really loaded, not merely constructed.
            await asyncio.to_thread(lambda: embedder.dimensions)
            self._embedder = embedder
        except Exception as exc:
            logger.exception("Failed to load embedding model")
            self._model_error = exc
        finally:
            self._model_ready.set()

    async def _await_model(self) -> Embedder:
        """Block until the embedder is installed; re-raise the load error if it failed."""
        await self._model_ready.wait()
        if self._model_error is not None:
            raise self._model_error
        assert self._embedder is not None
        return self._embedder

    def _compute_cache_key(
        self,
        source: str,
        ref: str | None = None,
        content: Sequence[ContentType] = (ContentType.CODE,),
    ) -> CacheKey:
        """Compute the canonical key for an exact index variant."""
        return compute_cache_key(source, ref, content)

    def _build_index(self, source: str, ref: str | None, embedder: Embedder, cache_key: CacheKey) -> ZembleIndex:
        """Build an index for the given source and cache it."""
        source_key, content = cache_key
        try:
            index = (
                ZembleIndex.from_git(source, ref=ref, embedder=embedder, content=content)
                if is_git_url(source)
                else ZembleIndex.from_path(source_key, embedder=embedder, content=content)
            )
        except EmbeddingBudgetExceeded as exc:
            hint = indexed_ancestor_hint(source_key, content)
            raise EmbeddingBudgetExceeded(f"{exc} {hint}" if hint else str(exc)) from exc
        try:
            save_index_to_cache(index, source_key)
        except Exception:
            logger.warning("Failed to save index cache for %r", source_key, exc_info=True)
        return index

    async def _build_tracked(
        self, source: str, ref: str | None, embedder: Embedder, cache_key: CacheKey
    ) -> ZembleIndex:
        """Build an index and, for local paths, record when its staleness cooldown ends.

        The cooldown write happens after the await, i.e. back on the event loop thread,
        regardless of which thread `_build_index` itself ran on.
        """
        start = time.monotonic()
        index = await asyncio.to_thread(self._build_index, source, ref, embedder, cache_key)
        if not is_git_url(source):
            finished = time.monotonic()
            self._revalidate_after[cache_key] = finished + (finished - start) * MIN_REVALIDATE_FACTOR
        return index

    def evict(self, cache_key: CacheKey) -> None:
        """Evict one exact index variant from memory."""
        existed = self._tasks.pop(cache_key, None) is not None
        self._revalidate_after.pop(cache_key, None)
        self.last_used.pop(cache_key, None)
        if existed and self._on_evict is not None:
            self._on_evict(cache_key)

    def loaded(self) -> list[tuple[CacheKey, ZembleIndex]]:
        """Return every key whose index is built and available right now, oldest use first."""
        ready = []
        for cache_key, task in self._tasks.items():
            if task.done() and not task.cancelled() and task.exception() is None:
                ready.append((cache_key, task.result()))
        return ready

    def is_building(self, cache_key: CacheKey) -> bool:
        """Return whether a build for this key is in flight."""
        task = self._tasks.get(cache_key)
        return task is not None and not task.done()

    def replace(self, cache_key: CacheKey, index: ZembleIndex, cooldown_seconds: float = 0.0) -> None:
        """Swap in a freshly rebuilt index for a key that is already resident.

        Silently ignores a key that has since been evicted, so a rebuild that finishes
        after an eviction never resurrects it. The cooldown keeps the on-disk staleness
        check off a just-rebuilt entry, whose cache file may not be written yet.
        """
        if cache_key not in self._tasks:
            return
        future: asyncio.Future[ZembleIndex] = asyncio.get_running_loop().create_future()
        future.set_result(index)
        self._tasks[cache_key] = future
        self._revalidate_after[cache_key] = time.monotonic() + cooldown_seconds

    async def _evict_if_stale(self, cache_key: CacheKey) -> None:
        """Evict a cached local-path entry whose on-disk cache no longer matches its files.

        Skipped while inside the cooldown window so repos that are slow to build aren't
        rebuilt faster than they can be served.
        """
        cached = self._tasks.get(cache_key)
        if (
            cached is None
            or is_git_url(cache_key[0])
            or not cached.done()
            or cached.cancelled()
            or cached.exception() is not None
        ):
            return
        if time.monotonic() < self._revalidate_after.get(cache_key, 0.0):
            return
        if self._embedder is None:
            return
        validated = await asyncio.to_thread(get_validated_cache, cache_key[0], self._embedder.model_id, cache_key[1])
        # Only evict if this entry hasn't already been replaced by a concurrent caller.
        if validated is None and self._tasks.get(cache_key) is cached:
            self.evict(cache_key)

    def loaded_roots(self, content: Sequence[ContentType]) -> set[str]:
        """Return the roots held in memory right now for exactly these content types."""
        wanted = tuple(content_type for content_type in ContentType if content_type in content)
        return {key[0] for key, _index in self.loaded() if key[1] == wanted}

    async def get(
        self,
        source: str,
        ref: str | None = None,
        content: Sequence[ContentType] = (ContentType.CODE,),
    ) -> ZembleIndex:
        """Return an index for the requested source, building and caching it on first access."""
        _cache_key, index = await self.get_with_key(source, ref, content)
        return index

    async def get_with_key(
        self,
        source: str,
        ref: str | None = None,
        content: Sequence[ContentType] = (ContentType.CODE,),
    ) -> tuple[CacheKey, ZembleIndex]:
        """Return the index answering a request, and the key of the root it was built from.

        A sub-directory of an indexed tree is served from that tree, filtered to the
        sub-directory, so the key names the ANCESTOR root while the index is a restricted
        view of it. Only when the ancestor holds nothing under the sub-directory does the
        sub-directory get an index of its own.
        """
        embedder = await self._await_model()
        root, prefix = await asyncio.to_thread(
            resolve_index_root, source, embedder.model_id, content, None, self.loaded_roots(content)
        )
        if prefix is None:
            return await self._get_exact(source, ref, content)
        cache_key, index = await self._get_exact(root, ref, content)
        view = index.subtree(prefix)
        if view is not None:
            return cache_key, view
        logger.info("the %s index holds nothing under %s; indexing it on its own", root, source)
        return await self._get_exact(source, ref, content)

    async def _get_exact(
        self,
        source: str,
        ref: str | None = None,
        content: Sequence[ContentType] = (ContentType.CODE,),
    ) -> tuple[CacheKey, ZembleIndex]:
        """Return the index built from exactly this source, building and caching it on first access.

        Local paths are revalidated against the on-disk cache on every call (subject to a
        cooldown scaled by build time), so an entry is rebuilt once its files change.
        """
        cache_key = self._compute_cache_key(source, ref, content)
        await self._evict_if_stale(cache_key)

        if cache_key not in self._tasks:
            embedder = await self._await_model()
            # Re-check after the await: another caller may have populated the entry.
            if cache_key not in self._tasks:
                if len(self._tasks) >= self._max_size:
                    evicted_key, _ = self._tasks.popitem(last=False)
                    self._revalidate_after.pop(evicted_key, None)
                    self.last_used.pop(evicted_key, None)
                    if self._on_evict is not None:
                        self._on_evict(evicted_key)
                self._tasks[cache_key] = asyncio.create_task(self._build_tracked(source, ref, embedder, cache_key))
        self._tasks.move_to_end(cache_key)
        self.last_used[cache_key] = time.time()
        task = self._tasks[cache_key]
        try:
            return cache_key, await asyncio.shield(task)
        except asyncio.CancelledError:  # pragma: no cover
            if task.done():
                self.evict(cache_key)
            raise
        except Exception:
            # Only evict if this task hasn't already been replaced by evict()+get().
            if self._tasks.get(cache_key) is task:
                self.evict(cache_key)
            raise

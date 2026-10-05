"""Memory ownership and single-flight regressions for complete daemon rebuilds."""

import asyncio
import gc
import json
import os
import subprocess
import sys
import threading
import weakref
from pathlib import Path

import pytest

from tests.conftest import FakeEmbedder
from zemble.daemon import server
from zemble.daemon.memory import MEMORY_ENV
from zemble.index.chunk_store import ChunkList
from zemble.types import ContentType


def daemon():
    """Construct a daemon backed only by the test embedder."""
    instance = server.Daemon(watch=False)
    instance.cache._embedder = FakeEmbedder(dimensions=8)
    instance.cache._model_ready.set()
    return instance


@pytest.mark.anyio
async def test_old_generation_is_released_before_graph_wait_and_trim_runs_afterwards(tmp_project, monkeypatch):
    """Neither local generation may survive eviction merely because a graph job is blocked."""
    instance = daemon()
    key, index = await instance.index_for({"path": str(tmp_project)})
    old = weakref.ref(index)
    del index
    (tmp_project / "new.py").write_text("def newly_added():\n    return 42\n")
    entered = asyncio.Event()
    finish = asyncio.Event()
    phases = []

    async def graph(self, root, changed_paths=None):
        gc.collect()
        assert old() is None, "the graph writer wait must not own the replaced generation"
        phases.append("graph")
        entered.set()
        await finish.wait()
        phases.append("graph done")
        return 1

    monkeypatch.setattr(server.Daemon, "_refresh_graph", graph)
    monkeypatch.setattr(server, "release_free_heap", lambda: phases.append("trim"))
    task = asyncio.create_task(instance.rebuild(key))
    await asyncio.wait_for(entered.wait(), 10)
    assert phases == ["trim", "graph"]
    assert key in instance.rebuilding, "status includes persistence and graph work"
    # Eviction cannot pin even the replacement through the graph wait's frame.
    replacement = weakref.ref(instance.cache.loaded()[0][1])
    instance.cache.evict(key)
    gc.collect()
    assert replacement() is None
    finish.set()
    await task
    assert phases == ["trim", "graph", "graph done", "trim"]
    assert not instance.rebuilding and not instance._rebuild_tasks


@pytest.mark.anyio
async def test_refresh_waiters_join_full_job_and_cancellation_does_not_spawn_another(tmp_project, monkeypatch):
    """Concurrent callers share graph/persistence work even when the initiating waiter disappears."""
    instance = daemon()
    key, index = await instance.index_for({"path": str(tmp_project)})
    del index
    entered = asyncio.Event()
    finish = asyncio.Event()
    builds = 0
    real = server.rebuild_index

    def rebuild(*args):
        nonlocal builds
        builds += 1
        return real(*args)

    async def graph(self, root, changed_paths=None):
        entered.set()
        await finish.wait()
        return 1

    monkeypatch.setattr(server, "rebuild_index", rebuild)
    monkeypatch.setattr(server.Daemon, "_refresh_graph", graph)
    first = asyncio.create_task(instance.rebuild(key))
    await asyncio.wait_for(entered.wait(), 10)
    others = [asyncio.create_task(instance.rebuild(key)) for _ in range(8)]
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert builds == 1
    assert all(not waiter.done() for waiter in others)
    finish.set()
    outcomes = await asyncio.gather(*others)
    assert builds == 1, "joining the graph phase cannot schedule eight sequential rebuilds"
    assert all(outcome == outcomes[0] for outcome in outcomes)
    assert not instance._rebuild_tasks


@pytest.mark.anyio
async def test_named_edit_arriving_during_graph_work_is_not_lost(tmp_project: Path, monkeypatch):
    """Joining a running rebuild coalesces a later watcher change instead of losing it."""
    instance = daemon()
    key, index = await instance.index_for({"path": str(tmp_project)})
    del index
    entered = asyncio.Event()
    finish = asyncio.Event()
    calls = 0

    async def graph(self, root, changed_paths=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await finish.wait()
        return 1

    monkeypatch.setattr(server.Daemon, "_refresh_graph", graph)
    first = asyncio.create_task(instance.rebuild(key))
    await asyncio.wait_for(entered.wait(), 10)
    moved = tmp_project / "late.py"
    moved.write_text("def late_symbol():\n    return 2\n")
    second = asyncio.create_task(instance.rebuild(key, changed_paths=[moved]))
    await asyncio.sleep(0)
    finish.set()
    await asyncio.gather(first, second)
    assert calls == 2
    assert "late.py" in instance.cache.loaded()[0][1].indexed_paths()


@pytest.mark.anyio
async def test_failed_build_traceback_drops_scratch_before_trimming(tmp_project, monkeypatch):
    """A failed build keeps its diagnostic traceback locations, not its temporary buffers."""
    instance = daemon()
    key, index = await instance.index_for({"path": str(tmp_project)})
    del index
    scratch = []
    trimmed = []

    class Buffer:
        pass

    def failed(*args):
        buffer = Buffer()
        buffer.data = bytearray(1024 * 1024)
        scratch.append(weakref.ref(buffer))
        raise RuntimeError("failed memory build")

    def trim():
        assert scratch and scratch[0]() is None, "traceback locals must be cleared before trimming"
        trimmed.append(True)

    monkeypatch.setattr(server, "rebuild_index", failed)
    monkeypatch.setattr(server, "release_free_heap", trim)
    with pytest.raises(RuntimeError, match="failed memory build"):
        await instance.rebuild(key)
    assert trimmed == [True]
    assert not instance._rebuild_tasks and not instance.rebuilding
    assert instance.cache.loaded(), "the serving generation survives a failed build"


@pytest.mark.anyio
async def test_budget_evicts_idle_lru_but_not_an_active_query(tmp_project, tmp_path_factory, monkeypatch):
    """Memory pressure evicts idle stores in use order, not a request's pinned generation."""
    instance = daemon()
    first, index = await instance.index_for({"path": str(tmp_project)})
    del index
    other = tmp_path_factory.mktemp("memory-other")
    (other / "other.py").write_text("def other(): return 1\n")
    second, index = await instance.index_for({"path": str(other)})
    del index
    instance.max_rss_mb = 150
    monkeypatch.setattr(server, "virtual_mb", lambda: 0)
    monkeypatch.setattr(server, "_rss_mb", lambda: 100 * len(instance.cache.loaded()))
    instance._query_keys[asyncio.current_task()] = second
    assert instance._admit(30)
    assert first not in instance.cache._tasks and second in instance.cache._tasks
    assert not instance._admit(60), "active generations cannot be evicted to meet a reservation"
    instance._query_keys.clear()
    assert instance._admit(60)
    assert not instance.cache.loaded()


@pytest.mark.anyio
async def test_rebuild_defers_without_allocating_and_old_generation_survives(tmp_project, monkeypatch):
    """A rebuild without headroom must not start copying or embedding anything."""
    instance = daemon()
    key, index = await instance.index_for({"path": str(tmp_project)})
    monkeypatch.setattr(instance, "_admit", lambda *args, **kwargs: False)
    monkeypatch.setattr(server, "rebuild_index", lambda *args: pytest.fail("build started without headroom"))
    result = await instance.rebuild(key)
    assert result["deferred"] == "memory budget"
    assert instance.cache.loaded()[0][1] is index
    assert instance.last_error[key]["knob"] == MEMORY_ENV
    assert not instance.rebuilding


@pytest.mark.anyio
async def test_a_request_during_a_build_joins_it_instead_of_reserving_another_load(tmp_project, monkeypatch):
    """Once the request that started a build gave up, the next one waits for that build, not for headroom."""
    (tmp_project / "src").mkdir()
    (tmp_project / "src" / "inner.py").write_text("def inner():\n    return 3\n")
    instance = daemon()
    started = threading.Event()
    release = threading.Event()
    real = instance.cache._build_index

    def gated(*args, **kwargs):
        started.set()
        release.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(instance.cache, "_build_index", gated)

    # 1. The first request starts the build and gives up waiting for it, as a deadline does.
    first = asyncio.create_task(instance.index_for({"path": str(tmp_project)}))
    await asyncio.to_thread(started.wait, 10)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    key = next(iter(instance.cache._tasks))
    assert instance.cache.is_building(key), "step 1: the build carries on without its first waiter"

    # 2. A request for a sub-path arriving now joins it, though no headroom is left for a second load.
    monkeypatch.setattr(instance, "_admit", lambda *args, **kwargs: False)
    second = asyncio.create_task(instance.index_for({"path": str(tmp_project / "src")}))
    await asyncio.sleep(0.05)
    assert not second.done(), "step 2: it waits for the build rather than being refused"
    release.set()
    joined_key, view = await asyncio.wait_for(second, 10)
    assert joined_key == key, "step 2: and answers from the root being built"
    assert view.indexed_paths() and all(path.startswith("src/") for path in view._file_mapping), "step 2: narrowed"


@pytest.mark.anyio
@pytest.mark.parametrize("larger_first", [False, True])
async def test_one_root_shares_code_and_docs_and_code_never_returns_docs(tmp_project, larger_first):
    """A covering selection owns the stores and narrower requests own only candidate selectors."""
    instance = daemon()
    code = {"path": str(tmp_project), "content": ["code"]}
    both = {"path": str(tmp_project), "content": ["code", "docs"]}
    if not larger_first:
        await instance.index_for(code)
    key, broad = await instance.index_for(both)
    narrow_key, narrow = await instance.index_for(code)
    assert narrow_key == key and len(instance.cache.loaded()) == 1
    assert narrow._semantic_index is broad._semantic_index
    assert narrow._bm25_index is broad._bm25_index
    assert narrow.chunks is broad.chunks
    assert narrow.content == (ContentType.CODE,)
    assert "README.md" in broad.indexed_paths()
    assert "README.md" not in narrow._file_mapping
    assert all(result.chunk.file_path.endswith(".py") for result in narrow.search("Test project"))
    _, filtered = await instance.index_for({**code, "exclude": ["!README.md"]})
    assert all(result.chunk.file_path.endswith(".py") for result in filtered.search("Test project"))
    assert (await server._cmd_evict(instance, code))["evicted"]
    assert not instance.cache.loaded()


@pytest.mark.anyio
async def test_content_upgrade_keeps_the_existing_build_scope_not_a_query_filter(tmp_project):
    """A caller's answer filter must not silently prune an upgraded resident root."""
    instance = daemon()
    await instance.index_for({"path": str(tmp_project), "content": ["code"]})
    await instance.index_for({"path": str(tmp_project), "content": ["code", "docs"], "exclude": ["utils.py"]})
    key, whole = await instance.index_for({"path": str(tmp_project), "content": ["code", "docs"]})
    assert "utils.py" in whole._file_mapping
    assert not whole.exclude


@pytest.mark.anyio
async def test_evicted_queued_rebuild_answers_without_leaving_root_history(tmp_project, monkeypatch):
    """Evicting queued work answers its waiters rather than breaking their socket request."""
    instance = daemon()
    key, index = await instance.index_for({"path": str(tmp_project)})
    del index
    await instance._load_lock.acquire()
    try:
        waiter = asyncio.create_task(instance.rebuild(key))
        await asyncio.sleep(0)
        instance.cache.evict(key)
        result = await waiter
    finally:
        instance._load_lock.release()
    assert result == {"skipped": "evicted before rebuild"}
    assert not instance._rebuild_tasks and not instance._rebuild_graph and not instance._rebuild_changes


@pytest.mark.anyio
async def test_continuous_churn_waits_for_quiet_and_deduplicates_changes(tmp_project, monkeypatch):
    """A watcher keeps consuming events rather than building at debounce intervals during churn."""
    instance = daemon()
    key, index = await instance.index_for({"path": str(tmp_project)})
    del index
    monkeypatch.setattr(server, "_QUIET_SECONDS", 0.05)
    calls = []

    async def rebuild(key, **kwargs):
        calls.append(kwargs["changed_paths"])
        return {"changed": 1}

    monkeypatch.setattr(instance, "rebuild", rebuild)
    path = tmp_project / "auth.py"
    for _ in range(10):
        await instance._on_change(key, {path})
        await asyncio.sleep(0.01)
    assert not calls
    assert instance.pending == {key}
    assert instance._watch_changes[key] == {path}
    await instance._watch_tasks[key]
    assert calls == [[path]]
    assert not instance.pending and not instance._watch_changes and not instance._changed_at


@pytest.mark.anyio
async def test_change_queue_overflow_is_one_full_rescan_and_eviction_releases_it(tmp_project, monkeypatch):
    """Overflow keeps a rescan bit instead of an unbounded set of paths or historical roots."""
    instance = daemon()
    key, index = await instance.index_for({"path": str(tmp_project)})
    del index
    monkeypatch.setattr(server, "_MAX_CHANGED_PATHS", 2)
    await instance._on_change(key, {tmp_project / f"file_{i}.py" for i in range(100)})
    assert instance._watch_changes[key] is None
    task = instance._watch_tasks[key]
    instance.cache.evict(key)
    await asyncio.gather(task, return_exceptions=True)
    assert not instance.pending and not instance._watch_changes and not instance._watch_tasks
    assert not instance.last_error and not instance.last_rebuild


@pytest.mark.anyio
async def test_all_roots_share_one_rebuild_slot(tmp_project, tmp_path_factory, monkeypatch):
    """Two roots cannot own vector or graph scratch at the same time."""
    instance = daemon()
    first, index = await instance.index_for({"path": str(tmp_project)})
    del index
    other = tmp_path_factory.mktemp("rebuild-other")
    (other / "other.py").write_text("def other(): return 1\n")
    second, index = await instance.index_for({"path": str(other)})
    del index
    entered = asyncio.Event()
    finish = asyncio.Event()
    calls = []

    async def build(key, **kwargs):
        calls.append(key)
        assert instance._load_lock.locked(), "index scratch owns the shared construction slot"
        entered.set()
        await finish.wait()
        return {"changed": 1}

    monkeypatch.setattr(instance, "_rebuild_once", build)
    a = asyncio.create_task(instance.rebuild(first))
    await entered.wait()
    b = asyncio.create_task(instance.rebuild(second))
    await asyncio.sleep(0)
    assert calls == [first]
    finish.set()
    await asyncio.gather(a, b)
    assert calls == [first, second]
    assert not instance.rebuilding and not instance._rebuild_tasks


@pytest.mark.anyio
async def test_repeated_rebuilds_return_to_mapped_stores_without_heap_deltas(tmp_project):
    """Persistence does not leave mutable postings, chunk generations, or anonymous vectors resident."""
    import numpy as np

    instance = daemon()
    key, index = await instance.index_for({"path": str(tmp_project)})
    assert isinstance(index.chunks, ChunkList)
    del index
    for i in range(3):
        path = tmp_project / "edit.py"
        path.write_text(f"def edited(): return {i}\n")
        await instance.rebuild(key, java_changed=False, changed_paths=[path])
        index = instance.cache.loaded()[0][1]
        assert isinstance(index.chunks, ChunkList)
        assert isinstance(index._semantic_index.vectors, np.memmap)
        assert index._bm25_index.delta_documents == 0
        assert not index._views
        del index


def test_allocation_backstop_refuses_a_real_oversized_allocation():
    """The kernel refuses an underestimated allocation before it can consume the configured memory."""
    code = """
import json
import sys
from zemble.daemon.memory import allocation_backstop, virtual_mb
budget = int(virtual_mb()) + 32
allocation_backstop(budget)
try:
    scratch = bytearray(128 * 1024 * 1024)
except MemoryError:
    sys.stdout.write(json.dumps({'refused': True, 'budget_mb': budget, 'virtual_mb': virtual_mb()}))
else:
    raise AssertionError('allocation backstop did not bind')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
    )
    payload = json.loads(result.stdout)
    assert payload["refused"] and payload["virtual_mb"] <= payload["budget_mb"]


def test_backstop_reports_a_stricter_inherited_limit(monkeypatch):
    """Status must describe the effective ceiling even when the parent supplied a stricter ulimit."""
    from zemble.daemon import memory

    monkeypatch.setattr(memory, "virtual_mb", lambda: 10)
    monkeypatch.setattr(memory.resource, "getrlimit", lambda kind: (512 * memory.MIB, 1024 * memory.MIB))
    applied = []
    monkeypatch.setattr(memory.resource, "setrlimit", lambda kind, value: applied.append(value))
    assert memory.allocation_backstop(4096) == 512
    assert applied == [(512 * memory.MIB, 1024 * memory.MIB)]


def test_ignore_cache_does_not_accumulate_directory_or_mtime_history(tmp_path, monkeypatch):
    """Ignore-file churn retains at most the configured number of compiled specs."""
    from collections import OrderedDict

    from zemble.index import file_walker

    monkeypatch.setattr(file_walker, "_SPEC_CACHE", OrderedDict())
    monkeypatch.setattr(file_walker, "_MAX_SPEC_CACHE", 2)
    for i in range(8):
        (tmp_path / ".gitignore").write_text(f"ignored_{i}.py\n")
        spec = file_walker._load_ignore_for_dir(str(tmp_path))
        assert spec[0].match_file(f"ignored_{i}.py")
        assert len(file_walker._SPEC_CACHE) <= 2


@pytest.mark.anyio
async def test_embedder_prewarm_and_first_query_load_one_model(monkeypatch):
    """Concurrent first callers must not allocate multiple model copies."""
    from zemble.index_cache import IndexCache

    calls = []

    def load():
        calls.append(True)
        return FakeEmbedder()

    monkeypatch.setattr("zemble.index_cache.load_embedder", load)
    cache = IndexCache()
    await asyncio.gather(*(cache.load_embedder_once() for _ in range(10)))
    assert calls == [True]
    assert cache.embedder is not None


def test_ignored_build_events_do_not_realpath_every_artifact(tmp_path, monkeypatch):
    """Binary build noise is rejected before expensive facts-path resolution."""
    from zemble.graph.facts import matches_facts_glob

    def no_resolve(*args, **kwargs):
        pytest.fail("ignored artifact reached realpath")

    monkeypatch.setattr(Path, "resolve", no_resolve)
    assert not matches_facts_glob(tmp_path, tmp_path / "build/classes/Noise.class")


def test_facts_metadata_caches_have_finite_capacity(tmp_path):
    """Facts config generations and declared paths must not retain historical roots forever."""
    from zemble.graph import facts

    for i in range(140):
        facts.facts_source_globs(tmp_path / f"root_{i}")
    assert facts._cached_source_globs.cache_info().currsize <= 128
    for i in range(8200):
        assert facts._relative_to_workspace(tmp_path, tmp_path, f"source_{i}.java") == f"source_{i}.java"
    assert facts._cached_relative.cache_info().currsize <= 8192

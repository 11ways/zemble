"""Memory ownership and single-flight regressions for complete daemon rebuilds."""

import asyncio
import gc
import weakref
from pathlib import Path

import pytest

from tests.conftest import FakeEmbedder
from zemble.daemon import server


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
    assert phases == ["graph"]
    assert key in instance.rebuilding, "status includes persistence and graph work"
    # Eviction cannot pin even the replacement through the graph wait's frame.
    replacement = weakref.ref(instance.cache.loaded()[0][1])
    instance.cache.evict(key)
    gc.collect()
    assert replacement() is None
    finish.set()
    await task
    assert phases == ["graph", "graph done", "trim"]
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

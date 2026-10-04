"""Concurrent immutable reads, bounded queueing and detached construction journeys."""

import asyncio
import time

import pytest

from tests.conftest import FakeEmbedder
from zemble.daemon import server
from zemble.daemon.admission import AdmissionBusy, ReadAdmission
from zemble.daemon.graph_jobs import GraphJobs


@pytest.mark.anyio
async def test_36_clients_queue_and_share_four_immutable_read_slots():
    """All admitted clients finish, while actual concurrency stays at the configured bound."""
    reads = ReadAdmission(4, 32)
    entered = asyncio.Event()
    finish = asyncio.Event()
    running = peak = calls = 0

    async def work():
        nonlocal running, peak, calls
        running += 1
        calls += 1
        peak = max(peak, running)
        if running == 4:
            entered.set()
        await finish.wait()
        running -= 1
        return 42

    tasks = [asyncio.create_task(reads.run(work, time.monotonic() + 5)) for _ in range(36)]
    await entered.wait()
    assert reads.status() == {"slots": 4, "queue_limit": 32, "active": 4, "queued": 32}
    finish.set()
    assert await asyncio.gather(*tasks) == [42] * 36
    assert peak == 4 and calls == 36
    assert reads.status()["active"] == reads.status()["queued"] == 0


@pytest.mark.anyio
async def test_queue_refusal_and_execution_deadline_keep_the_actual_lease():
    """A timed-out client cannot make its running work disappear from memory admission."""
    reads = ReadAdmission(1, 1)
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def work():
        entered.set()
        await finish.wait()
        return "complete"

    first = asyncio.create_task(reads.run(work, time.monotonic() + 0.05))
    await entered.wait()
    queued = asyncio.create_task(reads.run(work, time.monotonic() + 0.01))
    await asyncio.sleep(0)
    with pytest.raises(AdmissionBusy, match="full") as refusal:
        await reads.run(work, time.monotonic() + 1)
    assert refusal.value.retry_after_ms > 0
    with pytest.raises(AdmissionBusy, match="queue deadline"):
        await queued
    with pytest.raises(AdmissionBusy, match="execution deadline"):
        await first
    assert reads.status()["active"] == 1 and reads.status()["queued"] == 0
    finish.set()
    await reads.drain()
    assert not reads.active


@pytest.mark.anyio
async def test_cancelled_waiter_cannot_release_a_live_read_permit():
    """Cancellation removes a waiter, not the immutable work it already started."""
    reads = ReadAdmission(1, 1)
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def work():
        entered.set()
        await finish.wait()

    client = asyncio.create_task(reads.run(work, time.monotonic() + 5))
    await entered.wait()
    client.cancel()
    with pytest.raises(asyncio.CancelledError):
        await client
    assert len(reads.active) == 1
    finish.set()
    await reads.drain()


@pytest.mark.anyio
async def test_timed_out_cold_home_does_not_take_a_warm_search_slot(tmp_project, monkeypatch):
    """Cold graph preparation never monopolizes unrelated warm index reads."""
    instance = server.Daemon(watch=False)
    instance.cache._embedder = FakeEmbedder(dimensions=8)
    instance.cache._model_ready.set()
    await instance.index_for({"path": str(tmp_project)})
    finish = asyncio.Event()

    async def prepare(root, deadline, **kwargs):
        async with asyncio.timeout_at(deadline):
            await finish.wait()

    monkeypatch.setattr(instance.graphs, "ensure", prepare)
    response = await instance.handle(
        {"cmd": "home", "args": {"path": str(tmp_project)}, "deadline_ms": 5, "accepts_busy": True}
    )
    assert response["kind"] == "busy" and response["retry_after_ms"] > 0
    assert instance.reads.status()["active"] == 0
    search = await instance.handle({"cmd": "search", "args": {"path": str(tmp_project), "query": "function"}})
    assert search["ok"], "the expired graph waiter must not block a warm search"


@pytest.mark.anyio
async def test_graph_waiters_join_and_a_deadline_leaves_construction_observable(tmp_path, monkeypatch):
    """Construction survives its first caller and is joined by the next caller."""
    from zemble.graph import store

    monkeypatch.setattr(store, "graph_ancestor", lambda root: None)
    jobs = GraphJobs(asyncio.Lock(), 4096)
    entered = asyncio.Event()
    finish = asyncio.Event()
    count = 0

    async def build(root):
        nonlocal count
        count += 1
        jobs.states[root]["state"] = "building"
        entered.set()
        await finish.wait()
        jobs.ready.add(root)
        jobs.states[root]["state"] = "ready"
        jobs.jobs.pop(root, None)

    monkeypatch.setattr(jobs, "_build", build)
    first = asyncio.create_task(jobs.ensure(str(tmp_path), time.monotonic() + 0.01))
    await entered.wait()
    with pytest.raises(AdmissionBusy, match="pending"):
        await first
    assert jobs.status()["jobs"][0]["state"] == "building"
    second = asyncio.create_task(jobs.ensure(str(tmp_path), time.monotonic() + 5))
    await asyncio.sleep(0)
    assert count == 1
    finish.set()
    await second
    assert jobs.status()["jobs"][0]["state"] == "ready"


@pytest.mark.anyio
async def test_worker_budget_and_parent_limit_are_separate_and_restored(tmp_path, monkeypatch):
    """Aggregate reservation includes parent mappings, and teardown restores its old ceiling."""
    from zemble.daemon import graph_jobs

    limits = []
    monkeypatch.setattr(graph_jobs, "virtual_mb", lambda: 2600)
    monkeypatch.setattr(graph_jobs.resource, "getrlimit", lambda _: (4096 * 1024 * 1024, -1))
    monkeypatch.setattr(graph_jobs.resource, "setrlimit", lambda _, limit: limits.append(limit))
    jobs = GraphJobs(asyncio.Lock(), 4096)
    jobs.states[str(tmp_path)] = {"state": "queued"}

    class Process:
        pid = 123
        returncode = 0

        async def communicate(self, message):
            return b'{"peak_rss_mb": 2000}', b""

    async def spawn(*args, **kwargs):
        assert int(args[-1]) + limits[-1][0] / 1024 / 1024 <= jobs.total_mb
        return Process()

    monkeypatch.setattr(graph_jobs.asyncio, "create_subprocess_exec", spawn)
    await jobs._build(str(tmp_path))
    assert limits[-1] == (4096 * 1024 * 1024, -1)
    assert jobs.process is None and jobs.states[str(tmp_path)]["state"] == "ready"


@pytest.mark.anyio
async def test_published_graph_reads_do_not_wait_for_or_cancel_a_refresh(tmp_path, monkeypatch):
    """A held refresh must leave the previous published generation available to warm readers."""
    from zemble.graph import store

    monkeypatch.setattr(store, "graph_ancestor", lambda root: None)
    jobs = GraphJobs(asyncio.Lock(), 4096)
    root = str(tmp_path)
    jobs.ready.add(root)
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def build_once(path):
        jobs.states[path]["state"] = "building"
        entered.set()
        await finish.wait()
        jobs.states[path]["state"] = "ready"

    monkeypatch.setattr(jobs, "_build_once", build_once)
    refreshing = asyncio.create_task(jobs.ensure(root, time.monotonic() + 5, refresh=True))
    await entered.wait()
    await asyncio.wait_for(jobs.ensure(root, time.monotonic() + 0.01), 0.1)
    assert not refreshing.done(), "reading a published graph must not cancel its refresh"
    assert jobs.status()["jobs"][0]["state"] == "building"
    finish.set()
    await refreshing
    assert root in jobs.ready

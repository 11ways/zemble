"""The graph construction queue: a base workspace's refresh runs before lane jobs, and gone roots are dropped."""

import asyncio
import os

import pytest

from zemble.daemon import graph_jobs
from zemble.daemon.graph_jobs import GraphJobs


class _Workers:
    """Stand in for the graph worker process: record the root each spawn builds and finish it on demand."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.gates: list[asyncio.Event] = []
        self.spawned = asyncio.Condition()

    async def spawn(self, *args, **kwargs):
        gate = asyncio.Event()
        self.gates.append(gate)
        async with self.spawned:
            self.started.append(args[3])
            self.spawned.notify_all()

        class Process:
            pid = 1
            returncode = None

            async def communicate(self, message):
                await gate.wait()
                self.returncode = 0
                return b"{}", b""

        return Process()

    async def started_count(self, count: int) -> None:
        async with self.spawned:
            await asyncio.wait_for(self.spawned.wait_for(lambda: len(self.started) >= count), 2)

    async def run_in_turn(self, count: int) -> None:
        """Let *count* builds run one after the other, each finishing as soon as it starts."""
        for number in range(1, count + 1):
            await self.started_count(number)
            self.gates[number - 1].set()


@pytest.fixture
def workers(monkeypatch):
    """Patch the worker spawn and the memory limits it sets."""
    fake = _Workers()
    monkeypatch.setattr(graph_jobs, "private_mb", lambda: 100)
    monkeypatch.setattr(graph_jobs.resource, "getrlimit", lambda _: (graph_jobs.resource.RLIM_INFINITY, -1))
    monkeypatch.setattr(graph_jobs.resource, "setrlimit", lambda *_: None)
    monkeypatch.setattr(graph_jobs.asyncio, "create_subprocess_exec", fake.spawn)
    return fake


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A base workspace and two lanes of it, as graph roots; lane detection reads this table."""
    base, lane_a, lane_b = (tmp_path / name for name in ("base", "lane-a", "lane-b"))
    for folder in (base, lane_a, lane_b):
        folder.mkdir()
    lanes = {str(lane_a): str(base), str(lane_b): str(base)}
    monkeypatch.setattr(graph_jobs, "_lane_base", lanes.get, raising=False)
    return str(base), str(lane_a), str(lane_b)


async def _settle() -> None:
    """Let every runnable task reach its next wait."""
    for _ in range(50):
        await asyncio.sleep(0)


@pytest.mark.anyio
async def test_a_base_refresh_jumps_ahead_of_queued_lane_jobs(workers, workspace):
    """A base refresh queued after a lane job runs before it: the main workspace never waits out the lanes."""
    base, lane_a, lane_b = workspace
    jobs = GraphJobs(asyncio.Lock(), 4096)
    jobs._start(lane_a, refresh=True, changed_paths=None)
    await workers.started_count(1)
    jobs._start(lane_b, refresh=True, changed_paths=None)
    await _settle()
    jobs._start(base, refresh=True, changed_paths=None)
    await _settle()
    await workers.run_in_turn(3)
    await asyncio.gather(*list(jobs.jobs.values()))
    assert workers.started == [lane_a, base, lane_b]


@pytest.mark.anyio
async def test_a_lane_waits_out_its_stale_base_including_the_follow_up_pass(workers, workspace):
    """A lane queued behind its base's refresh starts only after the base's follow-up change set landed too."""
    base, lane_a, _ = workspace
    jobs = GraphJobs(asyncio.Lock(), 4096)
    jobs._start(base, refresh=True, changed_paths=["a.java"])
    await workers.started_count(1)
    jobs._start(lane_a, refresh=True, changed_paths=None)
    await _settle()
    jobs._start(base, refresh=True, changed_paths=["b.java"])
    await workers.run_in_turn(3)
    await asyncio.gather(*list(jobs.jobs.values()))
    assert workers.started == [base, base, lane_a], "the lane seeds from the refreshed base, never a stale one"


@pytest.mark.anyio
async def test_a_queued_job_for_a_removed_root_is_dropped_without_a_worker(workers, workspace):
    """A lane removed while its job waits in the queue never gets a worker started for it."""
    base, lane_a, _ = workspace
    jobs = GraphJobs(asyncio.Lock(), 4096)
    jobs.ready.add(lane_a)
    jobs._start(base, refresh=True, changed_paths=None)
    await workers.started_count(1)
    queued = jobs._start(lane_a, refresh=True, changed_paths=None)
    await _settle()
    os.rmdir(lane_a)
    workers.gates[0].set()
    with pytest.raises(FileNotFoundError):
        await asyncio.wait_for(asyncio.shield(queued), 2)
    assert workers.started == [base]
    assert jobs.states[lane_a]["state"] == "dropped"
    assert lane_a not in jobs.ready


@pytest.mark.anyio
async def test_a_cancelled_waiter_neither_runs_nor_keeps_the_turn(workers, workspace):
    """Cancelling a queued job hands the turn to the next waiter once the running job ends."""
    base, lane_a, lane_b = workspace
    jobs = GraphJobs(asyncio.Lock(), 4096)
    jobs._start(base, refresh=True, changed_paths=None)
    await workers.started_count(1)
    cancelled = jobs._start(lane_a, refresh=True, changed_paths=None)
    jobs._start(lane_b, refresh=True, changed_paths=None)
    await _settle()
    cancelled.cancel()
    await workers.run_in_turn(2)
    await asyncio.gather(*list(jobs.jobs.values()), return_exceptions=True)
    assert workers.started == [base, lane_b]
    assert jobs.states[lane_a]["state"] == "cancelled"

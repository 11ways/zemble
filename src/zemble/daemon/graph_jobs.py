"""Single-flight graph construction in a memory-bounded process beside the serving index."""

from __future__ import annotations

import asyncio
import json
import os
import resource
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from zemble.daemon.admission import AdmissionBusy
from zemble.daemon.memory import MIB, ReadOutgrown, private_mb

TOTAL_MEMORY_ENV = "ZEMBLE_DAEMON_TOTAL_MEMORY_MB"
GRAPH_MEMORY_ENV = "ZEMBLE_GRAPH_BUILD_MEMORY_MB"

#: Private memory the serving side keeps for concurrent reads while a worker builds, beyond what it holds.
SERVING_SCRATCH_MB = 192
#: The least the serving side keeps while a worker builds.
SERVING_FLOOR_MB = 512


def serving_reserve_mb(private: float, total_mb: int, build_mb: int) -> float:
    """The serving side's private-memory ceiling while a worker builds: what the construction budget leaves.

    Never below its own use plus scratch, nor the floor; the worker gets the rest of the total, so the
    aggregate is still bounded, and a whole-graph read (`health`, ~642 MiB on a 250k-symbol
    workspace) stays answerable in the daemon while another root builds.
    """
    return max(private + SERVING_SCRATCH_MB, SERVING_FLOOR_MB, total_mb - build_mb)


class GraphJobs:
    """Own one construction process and one joined job per canonical graph root."""

    def __init__(
        self, index_lock: asyncio.Lock, serving_mb: int, *, make_room: Callable[[float], bool] | None = None
    ) -> None:
        """Separate total, serving and construction limits, with a four-GiB default total.

        A build resolves a hundred files at a time, so a 200k-symbol workspace builds from
        nothing in ~200 MiB; the ceilings leave room for mapping facts files, which reads
        every symbol at once.

        :param index_lock: The daemon's index construction lock, held only while a worker starts.
        :param make_room: Evicts idle serving stores until the given MiB more fit; whether they do.
        """
        self.index_lock = index_lock
        self.lock = asyncio.Lock()
        self.make_room = make_room
        self.total_mb = int(os.environ.get(TOTAL_MEMORY_ENV, "4096"))
        self.build_mb = int(os.environ.get(GRAPH_MEMORY_ENV, "2048"))
        if serving_mb <= 0 or not 0 < self.build_mb <= self.total_mb:
            raise ValueError("serving/total memory must be positive and contain the construction budget")
        self.jobs: dict[str, asyncio.Task[Any]] = {}
        self.states: dict[str, dict[str, Any]] = {}
        self.ready: set[str] = set()
        self.generations: dict[str, int] = {}
        self.process: asyncio.subprocess.Process | None = None
        self.paths: dict[str, list[str] | None] = {}
        self.pending: dict[str, set[str] | None] = {}

    async def ensure(
        self, root: str, deadline: float | None, *, refresh: bool = False, changed_paths: Any = None
    ) -> None:
        """Join construction without letting a client timeout cancel shared work.

        A root that already has a published graph is never waited for: its first request starts
        the refresh and reads the published generation meanwhile, as every later one does.

        :param deadline: When this caller stops waiting (``time.monotonic()``); None waits it out.
        """
        from zemble.graph.store import graph_exists

        if str(Path(root).resolve()) in self.ready and not refresh:
            return
        root = await self._graph_root(root)
        if root in self.ready and not refresh:
            return
        if root not in self.ready and await asyncio.to_thread(graph_exists, root):
            # AIDEV-NOTE: a refresh of a large, busy workspace (javac facts rewritten by every build)
            # took 3 to 15+ minutes; making the first request after a daemon start wait for it
            # turned every graph question into "busy, retry". The published graph answers now.
            self.ready.add(root)
        task = self._start(root, refresh=refresh, changed_paths=changed_paths)
        if task is None or (root in self.ready and not refresh):
            return
        await self._join(root, task, deadline)

    @staticmethod
    async def _graph_root(path: str) -> str:
        """Resolve a path to the root whose graph answers for it: its own, or a covering ancestor's."""
        from zemble.graph.store import graph_ancestor, graph_covers

        root = str(Path(path).resolve())
        ancestor = await asyncio.to_thread(graph_ancestor, root)
        if ancestor is not None and await asyncio.to_thread(graph_covers, *ancestor):
            root = ancestor[0]
        return str(Path(root).resolve())

    def _start(self, root: str, *, refresh: bool, changed_paths: Any) -> asyncio.Task[Any] | None:
        """Start the root's construction, or fold this change set into the one already running.

        :return: The job to join, or None when the queue is full and a published graph answers.
        :raises AdmissionBusy: When the queue is full and the caller has nothing to read meanwhile.
        """
        task = self.jobs.get(root)
        if task is None:
            if sum(not task.done() for task in self.jobs.values()) >= 8:
                if root in self.ready and not refresh:
                    return None
                raise AdmissionBusy("graph construction queue is full")
            self.states[root] = {"root": root, "state": "queued", "queued_at": time.time()}
            # AIDEV-NOTE: refresh replaces a published graph beside its readers, just like
            # an index rebuild. Keeping the old ready generation avoids turning edits into outages.
            self.paths[root] = [str(path) for path in changed_paths] if changed_paths is not None else None
            task = asyncio.create_task(self._build(root))
            self.jobs[root] = task
            task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        elif refresh:
            paths = self.pending.setdefault(root, set())
            if paths is None or changed_paths is None:
                self.pending[root] = None
            else:
                paths.update(str(path) for path in changed_paths)
                if len(paths) > 4096:
                    self.pending[root] = None
        return task

    @staticmethod
    async def _join(root: str, task: asyncio.Task[Any], deadline: float | None) -> None:
        """Wait for a shared job until this caller's deadline, never cancelling it."""
        try:
            async with asyncio.timeout_at(deadline):
                await asyncio.shield(task)
        except TimeoutError as exc:
            raise AdmissionBusy(f"graph construction pending for {root}; inspect status and retry") from exc

    async def _build(self, root: str) -> None:
        """Drain a bounded follow-up change set before publishing this job as complete.

        AIDEV-NOTE: no lifetime. A job killed at a deadline discards everything it did, and the
        next attempt redoes the same work and dies the same way: the graph never moves again.
        Waiters are bounded by their own deadline, the worker by its private-memory limit.
        """
        try:
            while True:
                await self._build_once(root)
                if root not in self.pending:
                    return
                paths = self.pending.pop(root)
                self.paths[root] = sorted(paths) if paths is not None else None
        finally:
            self.jobs.pop(root, None)
            self.paths.pop(root, None)
            self.pending.pop(root, None)

    async def _build_once(self, root: str) -> None:
        """Reserve aggregate private memory, reap the worker and restore serving limits on every path.

        AIDEV-NOTE: one worker at a time (`self.lock`). The index construction lock is held only
        while this process's data limit drops to the serving reserve and the worker starts: no index
        load or rebuild is mid-allocation when the limit drops, and every one after it is admitted
        under the lowered limit, so none waits out the build.
        """
        state = self.states[root]
        started = time.monotonic()
        try:
            async with self.lock:
                async with self.index_lock:
                    old_soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
                    worker_limit = self._worker_limit()
                    if worker_limit < 512 and self.make_room is not None and self.make_room(512 + SERVING_SCRATCH_MB):
                        worker_limit = self._worker_limit()
                    if worker_limit < 512:
                        raise ReadOutgrown(
                            f"graph construction needs 512 MiB beside the {private_mb():.0f} MiB this daemon serves "
                            f"within {self.total_mb} MiB ({TOTAL_MEMORY_ENV}); answer it in a process of its own."
                        )
                    serving_limit = serving_reserve_mb(private_mb(), self.total_mb, self.build_mb)
                    parent_limit = int(serving_limit * MIB)
                    if old_soft != resource.RLIM_INFINITY:
                        parent_limit = min(parent_limit, old_soft)
                    resource.setrlimit(resource.RLIMIT_DATA, (parent_limit, hard))
                    try:
                        self.process = await asyncio.create_subprocess_exec(
                            sys.executable,
                            "-m",
                            "zemble.daemon.graph_worker",
                            root,
                            str(worker_limit),
                            stdin=asyncio.subprocess.PIPE,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            env={
                                **os.environ,
                                "OPENBLAS_NUM_THREADS": "1",
                                "OMP_NUM_THREADS": "1",
                                "MALLOC_ARENA_MAX": "2",
                            },
                        )
                    except BaseException:
                        resource.setrlimit(resource.RLIMIT_DATA, (old_soft, hard))
                        raise
                try:
                    state.update(
                        state="building",
                        pid=self.process.pid,
                        budget_mb=worker_limit,
                        started_at=time.time(),
                        serving_limit_mb=parent_limit / MIB,
                    )
                    message = json.dumps({"changed_paths": self.paths.get(root)}).encode()
                    stdout, stderr = await self.process.communicate(message)
                    if self.process.returncode:
                        # A failure, never a refusal: the client asks again, then answers in its own process.
                        error = stderr.decode()[-2000:]
                        raise RuntimeError(f"graph worker failed ({self.process.returncode}): {error}")
                    state.update(json.loads(stdout))
                    self.generations[root] = self.generations.get(root, 0) + 1
                    state["generation"] = self.generations[root]
                    self.ready.add(root)
                    state["state"] = "ready"
                finally:
                    if self.process is not None and self.process.returncode is None:
                        self.process.terminate()
                        await self.process.wait()
                    self.process = None
                    resource.setrlimit(resource.RLIMIT_DATA, (old_soft, hard))
        except BaseException as exc:
            state.update(state="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed", error=str(exc))
            raise
        finally:
            state["elapsed_seconds"] = round(time.monotonic() - started, 3)

    async def close(self) -> None:
        """Cancel joined jobs and wait for worker teardown before the daemon exits."""
        tasks = list(self.jobs.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def status(self) -> dict[str, Any]:
        """Expose completed/refused jobs as well as active construction."""
        return {
            "total_memory_mb": self.total_mb,
            "construction_memory_mb": self.build_mb,
            "jobs": list(self.states.values()),
        }

    def _worker_limit(self) -> int:
        """The private memory a worker gets now: what the total leaves beside the serving reserve, up to the budget."""
        return int(min(self.build_mb, self.total_mb - serving_reserve_mb(private_mb(), self.total_mb, self.build_mb)))

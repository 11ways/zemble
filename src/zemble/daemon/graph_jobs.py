"""Single-flight graph construction in a bounded-lifetime process beside the serving index."""

from __future__ import annotations

import asyncio
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

from zemble.daemon.admission import AdmissionBusy
from zemble.daemon.memory import MIB, MemoryRefused, virtual_mb

TOTAL_MEMORY_ENV = "ZEMBLE_DAEMON_TOTAL_MEMORY_MB"
GRAPH_MEMORY_ENV = "ZEMBLE_GRAPH_BUILD_MEMORY_MB"


class GraphJobs:
    """Own one construction process and one joined job per canonical graph root."""

    def __init__(self, construction_lock: asyncio.Lock, serving_mb: int) -> None:
        """Separate total, serving and construction limits, with an eight-GiB default total."""
        self.lock = construction_lock
        self.total_mb = int(os.environ.get(TOTAL_MEMORY_ENV, "8192"))
        self.build_mb = int(os.environ.get(GRAPH_MEMORY_ENV, "6144"))
        if serving_mb <= 0 or not 0 < self.build_mb <= self.total_mb:
            raise ValueError("serving/total memory must be positive and contain the construction budget")
        self.jobs: dict[str, asyncio.Task[Any]] = {}
        self.states: dict[str, dict[str, Any]] = {}
        self.ready: set[str] = set()
        self.generations: dict[str, int] = {}
        self.process: asyncio.subprocess.Process | None = None
        self.paths: dict[str, list[str] | None] = {}
        self.pending: dict[str, set[str] | None] = {}

    async def ensure(self, root: str, deadline: float, *, refresh: bool = False, changed_paths: Any = None) -> None:
        """Join construction without letting a client timeout cancel shared work."""
        from zemble.graph.store import graph_ancestor, graph_covers

        root = str(Path(root).resolve())
        if root in self.ready and not refresh:
            return
        ancestor = await asyncio.to_thread(graph_ancestor, root)
        if ancestor is not None and await asyncio.to_thread(graph_covers, *ancestor):
            root = ancestor[0]
        root = str(Path(root).resolve())
        if root in self.ready and not refresh:
            return
        task = self.jobs.get(root)
        if task is None:
            if sum(not task.done() for task in self.jobs.values()) >= 8:
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
        try:
            async with asyncio.timeout_at(deadline):
                await asyncio.shield(task)
        except TimeoutError as exc:
            raise AdmissionBusy(f"graph construction pending for {root}; inspect status and retry") from exc

    async def _build(self, root: str) -> None:
        """Drain a bounded follow-up change set before publishing this job as complete."""
        try:
            async with asyncio.timeout(900):
                while True:
                    await self._build_once(root)
                    if root not in self.pending:
                        return
                    paths = self.pending.pop(root)
                    self.paths[root] = sorted(paths) if paths is not None else None
        except TimeoutError:
            self.states[root].update(state="failed", error="graph construction deadline expired")
            raise
        finally:
            self.jobs.pop(root, None)
            self.paths.pop(root, None)
            self.pending.pop(root, None)

    async def _build_once(self, root: str) -> None:
        """Reserve aggregate address space, reap the worker and restore serving limits on every path."""
        state = self.states[root]
        started = time.monotonic()
        try:
            async with self.lock:
                old_soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                # AIDEV-NOTE: reserve bounded concurrent read scratch while the worker owns the rest;
                # summing per-process limits, not RSS alone, bounds the complete construction lifetime.
                serving_limit = max(virtual_mb() + 192, 512)
                worker_limit = int(min(self.build_mb, self.total_mb - serving_limit))
                if worker_limit < 512:
                    raise MemoryRefused("insufficient aggregate graph construction headroom")
                parent_limit = int(serving_limit * MIB)
                if old_soft != resource.RLIM_INFINITY:
                    parent_limit = min(parent_limit, old_soft)
                resource.setrlimit(resource.RLIMIT_AS, (parent_limit, hard))
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
                    state.update(
                        state="building",
                        pid=self.process.pid,
                        budget_mb=worker_limit,
                        started_at=time.time(),
                        serving_virtual_limit_mb=parent_limit / MIB,
                    )
                    message = json.dumps({"changed_paths": self.paths.get(root)}).encode()
                    stdout, stderr = await self.process.communicate(message)
                    if self.process.returncode:
                        error = stderr.decode()[-2000:]
                        raise MemoryRefused(f"graph worker failed ({self.process.returncode}): {error}")
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
                    resource.setrlimit(resource.RLIMIT_AS, (old_soft, hard))
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

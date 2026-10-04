"""Bound concurrent immutable reads and their waiting room without abandoning running work."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from zemble.daemon.protocol import CommandBusy


class AdmissionBusy(CommandBusy):
    """A bounded admission deadline or queue refused a retryable request."""

    retry_after_ms = 250


class ReadAdmission:
    """Keep permits with running tasks, including after their clients stop waiting."""

    def __init__(self, slots: int = 4, queue_limit: int = 32) -> None:
        """Declare execution and queue capacities independently."""
        if slots < 1 or queue_limit < 0:
            raise ValueError("admission capacities must be positive/nonnegative")
        self.slots = slots
        self.queue_limit = queue_limit
        self.queued = 0
        self.active: set[asyncio.Task[Any]] = set()
        self._permits = asyncio.Semaphore(slots)

    def check_room(self) -> None:
        """Refuse a full waiting room before allocating cold preparation state."""
        if len(self.active) >= self.slots and self.queued >= self.queue_limit:
            raise AdmissionBusy("read queue is full")

    async def run(self, work: Callable[[], Awaitable[Any]], deadline: float) -> Any:
        """Queue until the deadline, then keep the lease until actual execution finishes."""
        self.check_room()
        self.queued += 1
        try:
            async with asyncio.timeout_at(deadline):
                await self._permits.acquire()
        except TimeoutError as exc:
            raise AdmissionBusy("read queue deadline expired") from exc
        finally:
            self.queued -= 1

        async def execute() -> Any:
            try:
                return await work()
            finally:
                self._permits.release()
                self.active.discard(asyncio.current_task())

        task = asyncio.create_task(execute())
        self.active.add(task)
        task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        try:
            async with asyncio.timeout_at(deadline):
                return await asyncio.shield(task)
        except TimeoutError as exc:
            raise AdmissionBusy("read execution deadline expired") from exc

    def status(self) -> dict[str, int]:
        """Report actual active work rather than just connected clients."""
        return {"slots": self.slots, "queue_limit": self.queue_limit, "active": len(self.active), "queued": self.queued}

    async def drain(self) -> None:
        """Reap reads before shutdown closes the daemon's resources."""
        if self.active:
            await asyncio.gather(*self.active, return_exceptions=True)


def request_deadline(request: dict[str, Any]) -> float:
    """Use one absolute deadline for preparation, queueing and execution."""
    milliseconds = float(request.get("deadline_ms", 25000))
    if not 0 < milliseconds <= 900000:
        raise ValueError("deadline_ms must be in (0, 900000]")
    return time.monotonic() + milliseconds / 1000

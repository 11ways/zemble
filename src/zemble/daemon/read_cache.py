"""Bound repeated immutable graph questions by serving/publication generation."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Hashable
from typing import Any


class ReadCache:
    """Coalesce identical questions without retaining serving indexes or failed answers."""

    def __init__(self, capacity: int = 64) -> None:
        """Declare one bounded result store and reuse active admission slots for misses."""
        self.capacity = capacity
        self.values: OrderedDict[Hashable, Any] = OrderedDict()
        self.pending: dict[Hashable, asyncio.Task[Any]] = {}

    async def get(self, key: Hashable, work: Callable[[], Awaitable[Any]]) -> Any:
        """Memoize successful immutable answers; generation changes select another key."""
        if key in self.values:
            self.values.move_to_end(key)
            return self.values[key]
        task = self.pending.get(key)
        if task is None:
            task = asyncio.create_task(work())
            self.pending[key] = task
        try:
            value = await asyncio.shield(task)
        finally:
            if task.done():
                self.pending.pop(key, None)
        if not isinstance(value, dict) or "error" not in value:
            self.values[key] = value
            self.values.move_to_end(key)
            while len(self.values) > self.capacity:
                self.values.popitem(last=False)
        return value

"""Repeated questions coalesce while serving/publication epochs invalidate old results."""

import asyncio
import gc
import weakref

import pytest

from zemble.daemon.read_cache import ReadCache


@pytest.mark.anyio
async def test_identical_questions_join_but_new_generations_recompute():
    """Repeated network/reranking work happens once per immutable generation."""
    cache = ReadCache(2)
    entered = asyncio.Event()
    finish = asyncio.Event()
    calls = 0

    async def work():
        nonlocal calls
        calls += 1
        entered.set()
        await finish.wait()
        return {"result": calls}

    first = asyncio.create_task(cache.get(("question", 1), work))
    await entered.wait()
    second = asyncio.create_task(cache.get(("question", 1), work))
    await asyncio.sleep(0)
    assert calls == 1
    finish.set()
    assert await first == await second == {"result": 1}
    assert await cache.get(("question", 1), work) == {"result": 1}
    assert await cache.get(("question", 2), work) == {"result": 2}
    assert await cache.get(("question", 3), work) == {"result": 3}
    assert len(cache.values) == 2


@pytest.mark.anyio
async def test_failed_answers_and_serving_indexes_are_not_retained():
    """An availability failure is never memoized and weak epoch keys own no old index."""
    cache = ReadCache()
    calls = 0

    async def failed():
        nonlocal calls
        calls += 1
        return {"error": "unavailable"}

    await cache.get("bad", failed)
    await cache.get("bad", failed)
    assert calls == 2 and not cache.values

    class Index:
        pass

    index = Index()
    epoch = weakref.ref(index)

    async def answer():
        return {"result": 42}

    await cache.get(("question", epoch), answer)
    del index
    gc.collect()
    assert epoch() is None

"""Zemble never fails a query.

The client waits out busy and answers here what the daemon cannot; the daemon names memory it lacks busy or outgrown,
never refused, and never makes an index load wait out a graph build.
"""

from __future__ import annotations

import asyncio
import resource
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import FakeEmbedder
from zemble.daemon import client, graph_jobs, server
from zemble.daemon.graph_jobs import GraphJobs
from zemble.daemon.memory import MemoryBusy, ReadOutgrown
from zemble.daemon.protocol import CommandBusy, CommandFailed, CommandOutgrown, CommandRefused, ErrorKind
from zemble.types import ContentType


@pytest.fixture
def daemon_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the client talk to a (mocked) daemon."""
    monkeypatch.setattr(client, "_disabled_reason", None)
    monkeypatch.delenv("ZEMBLE_DAEMON", raising=False)


def _replies(monkeypatch: pytest.MonkeyPatch, *replies: Any) -> list[str]:
    """Make the daemon answer each call with the next reply, raising those that are errors; return what was asked."""
    asked: list[str] = []
    queue = list(replies)

    def call(cmd: str, args: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        asked.append(cmd)
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(client, "call", call)
    return asked


@pytest.mark.usefixtures("daemon_allowed")
def test_busy_is_waited_out_however_often_it_comes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A busy daemon is asked again until it answers, never reported."""
    asked = _replies(monkeypatch, *[CommandBusy("read queue is full")] * 6, {"answer": 1})
    assert client.answer("search", {}) == {"answer": 1}
    assert len(asked) == 7


@pytest.mark.usefixtures("daemon_allowed")
def test_what_the_daemon_cannot_answer_is_handed_to_this_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure is asked again once; a second one, or a read the daemon outgrew, is answered here."""
    _replies(monkeypatch, CommandFailed("boom"), {"answer": 1})
    assert client.answer("search", {}) == {"answer": 1}
    _replies(monkeypatch, CommandFailed("boom"), CommandFailed("boom"))
    with pytest.raises(client.AnswerHere, match="boom"):
        client.answer("search", {})
    _replies(monkeypatch, CommandOutgrown("needs 900 MiB"))
    with pytest.raises(client.AnswerHere, match="needs 900 MiB"):
        client.answer("health", {})


@pytest.mark.usefixtures("daemon_allowed")
def test_a_refusal_is_still_the_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deliberate refusal (scope, work ceiling, spending budget) is raised, never answered here."""
    _replies(monkeypatch, CommandRefused("over the embedding budget"))
    with pytest.raises(CommandRefused):
        client.answer("search", {})


def test_a_disabled_daemon_is_answered_here(monkeypatch: pytest.MonkeyPatch) -> None:
    """ZEMBLE_DAEMON=0 leaves the request to this process instead of refusing it."""
    monkeypatch.setattr(client, "_disabled_reason", None)
    monkeypatch.setenv("ZEMBLE_DAEMON", "0")
    with pytest.raises(client.AnswerHere, match="ZEMBLE_DAEMON=0"):
        client.answer("search", {})


def _daemon() -> server.Daemon:
    """A daemon backed only by the test embedder."""
    instance = server.Daemon(watch=False)
    instance.cache._embedder = FakeEmbedder(dimensions=8)
    instance.cache._model_ready.set()
    return instance


@pytest.mark.anyio
async def test_memory_a_daemon_lacks_is_busy_while_work_runs_and_outgrown_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Short memory is never refused: busy while work that will end holds it, else outgrown for another process."""
    instance = _daemon()
    monkeypatch.setattr(server, "private_mb", lambda: 4000.0)
    monkeypatch.setattr(server, "data_limit_mb", lambda: 4096.0)
    key = ("/root", (), None)
    # The asking request's own pinned index is no work that will end for it.
    instance._query_keys[asyncio.current_task()] = key
    assert isinstance(instance._short_of_memory(512, "Loading /p"), ReadOutgrown)
    instance.rebuilding.add(key)
    assert isinstance(instance._short_of_memory(512, "Loading /p"), MemoryBusy)


@pytest.mark.anyio
async def test_a_load_that_does_not_fit_is_outgrown_and_an_allocation_failure_too(
    tmp_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither a load past the ceiling nor a MemoryError reaches the client as a refusal."""
    instance = _daemon()
    monkeypatch.setattr(instance, "_admit", lambda *args, **kwargs: False)
    response = await instance.handle({"id": 1, "cmd": "search", "args": {"path": str(tmp_project), "query": "x"}})
    assert response["kind"] == ErrorKind.OUTGROWN.value, response

    async def exhausted(*args: Any) -> None:
        raise MemoryError

    monkeypatch.setattr(instance, "_bounded_prepare", exhausted)
    response = await instance.handle({"id": 2, "cmd": "search", "args": {"path": str(tmp_project), "query": "x"}})
    assert response["kind"] == ErrorKind.OUTGROWN.value, response


class _Worker:
    """A graph worker that answers once `finish` is set, with `code` as its exit status."""

    pid = 4242

    def __init__(self, finish: asyncio.Event, code: int) -> None:
        self.finish, self.code, self.returncode = finish, code, None

    async def communicate(self, message: bytes) -> tuple[bytes, bytes]:
        await self.finish.wait()
        self.returncode = self.code
        return b'{"symbols": 1}', b"worker stderr"

    def terminate(self) -> None:
        """Nothing to stop."""

    async def wait(self) -> int:
        return self.code


def _graph_jobs(monkeypatch: pytest.MonkeyPatch, finish: asyncio.Event, code: int = 0, **kwargs: Any) -> GraphJobs:
    """Graph jobs whose worker is `_Worker`, never touching the test process's data limit."""

    async def spawn(*args: Any, **spawn_kwargs: Any) -> _Worker:
        return _Worker(finish, code)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(resource, "setrlimit", lambda *args: None)
    return GraphJobs(asyncio.Lock(), 4096, **kwargs)


@pytest.mark.anyio
async def test_a_graph_build_holds_the_index_lock_only_while_its_worker_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An index load never waits out a graph build: the lock is free while the worker runs."""
    finish = asyncio.Event()
    jobs = _graph_jobs(monkeypatch, finish)
    monkeypatch.setattr(graph_jobs, "private_mb", lambda: 100.0)
    root = str(tmp_path)
    jobs.states[root] = {"root": root, "state": "queued"}
    build = asyncio.create_task(jobs._build_once(root))
    while jobs.states[root]["state"] != "building":
        await asyncio.sleep(0.01)
    assert not jobs.index_lock.locked(), "the index construction lock is free during the build"
    finish.set()
    await build
    assert jobs.states[root]["state"] == "ready"


@pytest.mark.anyio
async def test_a_failed_worker_is_a_failure_and_no_headroom_is_outgrown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crashed worker fails, which the client asks again; headroom eviction cannot make is outgrown, never refused."""
    finish = asyncio.Event()
    finish.set()
    root = str(tmp_path)
    monkeypatch.setattr(graph_jobs, "private_mb", lambda: 100.0)
    jobs = _graph_jobs(monkeypatch, finish, code=1)
    jobs.states[root] = {"root": root, "state": "queued"}
    with pytest.raises(RuntimeError, match="graph worker failed"):
        await jobs._build_once(root)
    asked: list[float] = []
    monkeypatch.setattr(graph_jobs, "private_mb", lambda: 3900.0)
    jobs = _graph_jobs(monkeypatch, finish, make_room=lambda mb: asked.append(mb) or False)
    jobs.states[root] = {"root": root, "state": "queued"}
    with pytest.raises(ReadOutgrown, match="answer it in a process of its own"):
        await jobs._build_once(root)
    assert asked, "idle serving stores were asked to make room first"


@pytest.mark.anyio
async def test_a_request_answered_here_borrows_its_index(tmp_project: Path, mock_embedder: FakeEmbedder) -> None:
    """What a daemon could not answer is answered here without keeping the index, which the daemon owns."""
    from zemble.index_cache import IndexCache
    from zemble.mcp import _get_index

    cache = IndexCache()
    cache._embedder = mock_embedder
    cache._model_ready.set()
    with client.request_scope():
        client.answer_here("daemon unavailable: not running")
        index = await _get_index(str(tmp_project), cache, [ContentType.CODE])
    assert index.search("authenticate", top_k=1), "the request was answered"
    assert not cache._tasks, "and the index was not kept"
    # A deliberate --no-daemon process keeps what it loads, as before.
    await _get_index(str(tmp_project), cache, [ContentType.CODE])
    assert cache._tasks

"""Zemble never fails a query, and a daemon-enabled client never keeps an index the daemon owns."""

import asyncio
import json
import logging
import socket
import threading

import pytest

from tests.conftest import FakeEmbedder
from zemble import cli, mcp
from zemble.daemon import client, server
from zemble.daemon.protocol import (
    ACCEPTS_BUSY_FIELD,
    CONSTRUCTION_WAIT_SECONDS,
    DEADLINE_FIELD,
    REVISION_FIELD,
    CommandBusy,
    CommandFailed,
    CommandOutgrown,
    CommandRefused,
    DaemonUnavailable,
    ErrorKind,
    decode,
    encode,
)
from zemble.index_cache import IndexCache


@pytest.fixture
def daemon_mode(monkeypatch, tmp_path):
    """Use a private runtime directory and remove the local opt-out supplied by unit fixtures."""
    monkeypatch.setattr(client, "_disabled_reason", None)
    monkeypatch.setattr(client, "_daemon_revision", None)
    monkeypatch.delenv("ZEMBLE_DAEMON", raising=False)
    monkeypatch.delenv("ZEMBLE_DAEMON_SOCKET", raising=False)
    monkeypatch.setenv("ZEMBLE_DAEMON_DIR", str(tmp_path / "run"))


#: What the daemon cannot answer, so the client answers the request in its own process.
ANSWERED_HERE = [
    DaemonUnavailable("startup failed"),
    DaemonUnavailable("connection lost: timed out"),
    CommandFailed("malformed message"),
    CommandFailed("unsupported command"),
    CommandOutgrown("needs 900 MiB of private memory"),
]

MCP_CASES = {
    "search": {"query": "shape"},
    "find_related": {"file_path": "auth.py", "line": 1},
    "explain": {"query": "shape"},
    "outline": {"target": "auth.py"},
    "signatures": {"symbol": "authenticate"},
    "home": {"description": "shape"},
    "graph_definition": {"symbol": "authenticate"},
    "graph_callers": {"symbol": "authenticate"},
}


class _AnsweredHere(Exception):
    """Raised by a stubbed local path, to prove the tool reached it."""


def _stub_local_paths(monkeypatch) -> list[str]:
    """Replace every tool's local path with a stub that records it ran; return the record."""
    ran: list[str] = []

    def local(name):
        def run(*args, **kwargs):
            ran.append(name)
            raise _AnsweredHere(name)

        return run

    monkeypatch.setattr("zemble.mcp._get_index", local("index"))
    monkeypatch.setattr("zemble.graph.mcp.answer", local("graph"))
    monkeypatch.setattr("zemble.evidence.mcp._open", local("open"))
    monkeypatch.setattr("zemble.home.mcp._here", local("home"))
    return ran


@pytest.mark.anyio
@pytest.mark.parametrize("error", ANSWERED_HERE, ids=lambda error: f"{type(error).__name__}-{error}")
async def test_every_mcp_tool_answers_here_what_its_daemon_cannot(daemon_mode, tmp_project, monkeypatch, error):
    """Every daemon-backed tool asks again, then answers in its own process, never reporting the error."""
    asked: list[str] = []

    def fail(cmd, *args, **kwargs):
        asked.append(cmd)
        raise error

    monkeypatch.setattr(client, "call", fail)
    ran = _stub_local_paths(monkeypatch)
    instance = mcp.create_server(IndexCache())
    for name, args in MCP_CASES.items():
        ran.clear()
        try:
            await instance.call_tool(name, {**args, "repo": str(tmp_project)})
        except Exception as raised:  # the stub's exception, wrapped by FastMCP
            assert "_AnsweredHere" in repr(raised.__cause__) or isinstance(raised.__cause__, _AnsweredHere), raised
        assert ran, f"{name} answered in its own process"
    assert asked, "the daemon was asked first"


@pytest.mark.anyio
async def test_every_mcp_tool_waits_out_busy_and_reports_a_refusal(daemon_mode, tmp_project, monkeypatch):
    """Busy is asked again; a refusal is the answer, reported with no local work."""
    replies: dict[str, list[Exception]] = {}

    def daemon(cmd, *args, **kwargs):
        queue = replies.setdefault(cmd, [CommandBusy("occupied"), CommandBusy("occupied")])
        raise queue.pop(0) if queue else CommandRefused("memory budget refuses this root")

    monkeypatch.setattr(client, "call", daemon)
    ran = _stub_local_paths(monkeypatch)
    cache = IndexCache()
    instance = mcp.create_server(cache)
    for name, args in MCP_CASES.items():
        replies.clear()
        result = await instance.call_tool(name, {**args, "repo": str(tmp_project)})
        assert "memory budget refuses this root" in result[0].text, (name, result)
    assert not ran and not cache._tasks and cache.embedder is None


@pytest.mark.parametrize("command", ["search", "stats", "find-related", "explain", "home", "outline", "signatures"])
def test_cli_reports_a_refusal_without_loading_indexes(daemon_mode, tmp_project, monkeypatch, capsys, command):
    """A refusal ends every query CLI command: answering it here would refuse identically."""
    root = str(tmp_project)
    args = {
        "search": ["query", root],
        "stats": [root],
        "find-related": ["auth.py", "1", root],
        "explain": [root, "query"],
        "home": [root, "query"],
        "outline": [root, "auth.py"],
        "signatures": [root, "authenticate"],
    }

    def refuse(*args, **kwargs):
        raise CommandRefused("memory budget refuses this root")

    monkeypatch.setattr(client, "call", refuse)
    monkeypatch.setattr(cli, "_load_index", lambda *a, **kw: pytest.fail("local index loaded"))
    import sys

    monkeypatch.setattr(sys, "argv", ["zemble", command, *args[command]])
    with pytest.raises(SystemExit) as raised:
        cli._cli_main()
    assert raised.value.code == 1
    assert "memory budget refuses this root" in capsys.readouterr().err


@pytest.mark.anyio
async def test_a_disabled_environment_answers_here(daemon_mode, tmp_project, monkeypatch):
    """ZEMBLE_DAEMON=0 leaves the request to this process instead of refusing it."""
    monkeypatch.setenv("ZEMBLE_DAEMON", "0")
    ran = _stub_local_paths(monkeypatch)
    with pytest.raises(Exception):  # the stub's exception, wrapped by FastMCP
        await mcp.create_server(IndexCache()).call_tool("search", {"query": "q", "repo": str(tmp_project)})
    assert ran == ["index"]


@pytest.mark.anyio
async def test_no_result_is_asked_again_then_answered_here(daemon_mode, tmp_project, monkeypatch):
    """A reply with no result is a failure: asked again, then answered here, never taken for an answer."""
    asked: list[str] = []
    monkeypatch.setattr(client, "call", lambda cmd, *a, **kw: asked.append(cmd))
    ran = _stub_local_paths(monkeypatch)
    with pytest.raises(Exception):  # the stub's exception, wrapped by FastMCP
        await mcp.create_server(IndexCache()).call_tool("search", {"query": "q", "repo": str(tmp_project)})
    assert asked == ["search"] * client.FAILED_TRIES and ran == ["index"]


@pytest.mark.anyio
async def test_mcp_startup_loads_no_embedder(daemon_mode, monkeypatch):
    """Stdio clients start without a model or background index initialization."""

    async def stdio(self):
        await asyncio.sleep(0)

    monkeypatch.setattr("mcp.server.fastmcp.FastMCP.run_stdio_async", stdio)
    monkeypatch.setattr("zemble.index_cache.load_embedder", lambda: pytest.fail("client preloaded a model"))
    await mcp.serve()


@pytest.mark.anyio
async def test_explicit_local_mode_can_load_an_index(tmp_project, monkeypatch):
    """The sole opt-out still permits local library work and lazy model initialization."""
    monkeypatch.setattr(client, "_disabled_reason", "--no-daemon")
    monkeypatch.setattr("zemble.index_cache.load_embedder", FakeEmbedder)
    monkeypatch.setattr(client, "call", lambda *a, **kw: pytest.fail("local mode contacted daemon"))
    cache = IndexCache()
    result = await mcp.create_server(cache).call_tool("search", {"query": "authenticate", "repo": str(tmp_project)})
    assert "auth.py" in result[0].text
    assert len(cache._tasks) == 1


@pytest.mark.anyio
async def test_busy_daemon_does_not_queue_another_request(daemon_mode, monkeypatch, tmp_path):
    """Capacity errors are cheap replies and status remains available."""
    instance = server.Daemon(watch=False)
    from zemble.daemon.admission import ReadAdmission

    instance.reads = ReadAdmission(1, 0)
    hold = asyncio.get_running_loop().create_future()
    running = asyncio.create_task(instance.reads.run(lambda: hold, asyncio.get_running_loop().time() + 5))
    await asyncio.sleep(0)
    try:
        response = await instance.handle(
            {"id": 1, "cmd": "search", "args": {"path": str(tmp_path), "query": "q"}, ACCEPTS_BUSY_FIELD: True}
        )
        assert response["kind"] == ErrorKind.BUSY.value
        assert (await instance.handle({"id": 2, "cmd": "status"}))["ok"]
    finally:
        hold.set_result(None)
        await running
    assert not instance.cache._tasks


@pytest.mark.anyio
async def test_busy_legacy_requests_use_the_known_refusal_lane(daemon_mode, tmp_path):
    """Old index clients must not fall back merely because they do not recognize a new busy vocabulary."""
    instance = server.Daemon(watch=False)
    from zemble.daemon.admission import ReadAdmission

    instance.reads = ReadAdmission(1, 0)
    hold = asyncio.get_running_loop().create_future()
    running = asyncio.create_task(instance.reads.run(lambda: hold, asyncio.get_running_loop().time() + 5))
    await asyncio.sleep(0)
    try:
        reply = await instance.handle({"id": 1, "cmd": "search", "args": {"path": str(tmp_path), "query": "q"}})
        assert reply["kind"] == ErrorKind.REFUSED.value
        assert "busy, retry" in reply["error"]
    finally:
        hold.set_result(None)
        await running


def connect_reply(monkeypatch, payload):
    """Supply one owned socket-pair response using the actual client codec."""
    local, peer = socket.socketpair()

    def reply():
        with peer, peer.makefile("rwb") as stream:
            request = decode(stream.readline())
            response = {"id": request["id"], **payload}
            stream.write(encode(response))
            stream.flush()

    thread = threading.Thread(target=reply, daemon=True)
    thread.start()
    monkeypatch.setattr(client, "_connect", lambda *args, **kwargs: local)
    return thread


@pytest.mark.anyio
async def test_long_running_mcp_follows_each_new_daemon_revision(daemon_mode, tmp_project, monkeypatch, caplog):
    """One MCP instance continues over real RPC envelopes from multiple daemon revisions."""
    instance = mcp.create_server(IndexCache())
    with caplog.at_level(logging.INFO):
        for revision in ("old-revision", "new-revision", "third-revision"):
            thread = connect_reply(
                monkeypatch,
                {
                    "ok": True,
                    "result": {"query": "q", "results": [], "revision": revision},
                    REVISION_FIELD: revision,
                },
            )
            result = await instance.call_tool("search", {"query": "q", "repo": str(tmp_project)})
            assert json.loads(result[0].text)["revision"] == revision
            assert client.daemon_revision() == revision
            thread.join(2)
    changes = [row.getMessage() for row in caplog.records if "revision changed" in row.getMessage()]
    assert len(changes) == 2
    assert all("continuing" in message for message in changes)


@pytest.mark.parametrize(
    "kind, exception",
    [("busy", CommandBusy), ("refused", CommandRefused), ("outgrown", CommandOutgrown), ("future", CommandFailed)],
)
def test_error_kinds_never_authorize_fallback(daemon_mode, monkeypatch, kind, exception):
    """Every daemon error kind is an exception on the wire; `client.answer` alone decides what to do with it."""
    thread = connect_reply(monkeypatch, {"ok": False, "kind": kind, "error": "not admitted"})
    with pytest.raises(exception, match="not admitted"):
        client.call("search", auto_start=False)
    thread.join(2)


def test_socket_timeout_is_an_availability_error(daemon_mode, monkeypatch):
    """A busy or stalled transport has a bounded lifetime and cannot start a local build."""
    local, peer = socket.socketpair()
    monkeypatch.setattr(client, "_connect", lambda *a, **kw: local)
    with peer, pytest.raises(DaemonUnavailable, match="connection lost"):
        client.call("search", auto_start=False, timeout=0.01)


@pytest.mark.parametrize("timeout, deadline_ms", [(None, 900000), (CONSTRUCTION_WAIT_SECONDS, 900000), (10.0, 10000)])
def test_every_request_names_a_deadline_that_waits_out_construction(daemon_mode, monkeypatch, timeout, deadline_ms):
    """A cold query blocks on the construction it joins; the daemon's legacy 25 s default never applies."""
    local, peer = socket.socketpair()
    seen = {}

    def reply():
        with peer, peer.makefile("rwb") as stream:
            request = decode(stream.readline())
            seen.update(request)
            stream.write(encode({"id": request["id"], "ok": True, "result": {}}))
            stream.flush()

    thread = threading.Thread(target=reply, daemon=True)
    thread.start()
    timeouts = []
    original = socket.socket.settimeout

    def record(self, value):
        timeouts.append(value)
        original(self, value)

    monkeypatch.setattr(client, "_connect", lambda *a, **kw: local)
    with monkeypatch.context() as patched:
        patched.setattr(socket.socket, "settimeout", record)
        assert client.call("graph", {"path": "/"}, auto_start=False, timeout=timeout) == {}
    thread.join(2)
    # 1. The deadline travels with the request, capped at the construction bound.
    assert seen[DEADLINE_FIELD] == deadline_ms, "step 1: the request names its deadline"
    # 2. The socket outlives that deadline, so even a late busy answer arrives.
    assert timeouts[-1] is None or timeouts[-1] > deadline_ms / 1000, "step 2: the socket outwaits the daemon"


@pytest.mark.parametrize("argv", [["zemble", "--no-daemon"], ["zemble", "mcp", "--no-daemon"]])
def test_mcp_cli_optout_is_explicit(daemon_mode, monkeypatch, argv):
    """Both supported stdio spellings select local execution only when explicitly requested."""
    import sys

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(asyncio, "run", lambda coro: coro.close())
    cli.main()
    assert client.in_process()


@pytest.mark.parametrize("option", ["--embedder", "--reranker", "--intent"])
def test_cli_overrides_do_not_authorize_local_indexes(daemon_mode, monkeypatch, capsys, option):
    """Unsupported per-query daemon overrides are refusals, not local build shortcuts."""
    monkeypatch.setattr(client, "call", lambda *args, **kwargs: pytest.fail("unsupported override sent"))
    with pytest.raises(SystemExit):
        cli._via_daemon(
            "search",
            {},
            False,
            "model2vec:other" if option == "--embedder" else None,
            local_option=None if option == "--embedder" else option,
        )
    assert "requires explicit --no-daemon" in capsys.readouterr().err


@pytest.mark.anyio
async def test_runtime_status_reports_the_current_daemon_revision(daemon_mode, monkeypatch):
    """The status tool distinguishes an older client identity from the daemon actually answering."""
    instance = mcp.create_server(IndexCache())
    thread = connect_reply(
        monkeypatch,
        {
            "ok": True,
            REVISION_FIELD: "new-daemon",
            "result": {"runtime": {"source_revision": "new-daemon"}},
        },
    )
    result = await instance.call_tool("status", {})
    payload = json.loads(result[0].text)
    assert payload["daemon_revision"] == "new-daemon"
    assert payload["daemon"]["runtime"]["source_revision"] == "new-daemon"
    assert "identity" in payload
    thread.join(2)

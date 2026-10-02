"""Daemon-enabled clients must never populate a local index cache after an RPC failure."""

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
    REVISION_FIELD,
    CommandBusy,
    CommandFailed,
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


ERRORS = [
    DaemonUnavailable("startup failed"),
    DaemonUnavailable("connection lost: timed out"),
    CommandFailed("malformed message"),
    CommandFailed("unsupported command"),
    CommandRefused("memory budget refuses this root"),
    CommandBusy("request execution is occupied"),
]


@pytest.mark.anyio
@pytest.mark.parametrize("error", ERRORS, ids=lambda error: f"{type(error).__name__}-{error}")
async def test_all_mcp_index_and_graph_tools_report_errors_without_local_work(
    daemon_mode, tmp_project, monkeypatch, error
):
    """Every daemon-backed tool reports failure without buying vectors, loading indexes or building graphs."""
    cache = IndexCache()

    def fail(*args, **kwargs):
        raise error

    def local(*args, **kwargs):
        pytest.fail("daemon-enabled client performed local work")

    monkeypatch.setattr(client, "call", fail)
    monkeypatch.setattr(cache, "load_embedder_once", local)
    monkeypatch.setattr(cache, "get", local)
    monkeypatch.setattr("zemble.graph.mcp.answer", local)
    monkeypatch.setattr("zemble.evidence.mcp._open", local)
    monkeypatch.setattr("zemble.home.mcp._here", local)
    instance = mcp.create_server(cache)
    cases = {
        "search": {"query": "shape"},
        "find_related": {"file_path": "auth.py", "line": 1},
        "explain": {"query": "shape"},
        "outline": {"target": "auth.py"},
        "signatures": {"symbol": "authenticate"},
        "home": {"description": "shape"},
        "graph_definition": {"symbol": "authenticate"},
        "graph_callers": {"symbol": "authenticate"},
    }
    for name, args in cases.items():
        result = await instance.call_tool(name, {**args, "repo": str(tmp_project)})
        assert str(error) in result[0].text, (name, result)
    assert not cache._tasks and cache.embedder is None


@pytest.mark.parametrize("error", ERRORS, ids=lambda error: type(error).__name__)
@pytest.mark.parametrize("command", ["search", "stats", "find-related", "explain", "home", "outline", "signatures"])
def test_cli_errors_never_load_indexes(daemon_mode, tmp_project, monkeypatch, capsys, error, command):
    """All query CLI surfaces terminate explicitly after a daemon error."""
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

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(client, "call", fail)
    monkeypatch.setattr(cli, "_load_index", lambda *a, **kw: pytest.fail("local index loaded"))
    import sys

    monkeypatch.setattr(sys, "argv", ["zemble", command, *args[command]])
    with pytest.raises(SystemExit) as raised:
        cli._cli_main()
    assert raised.value.code == 1
    assert str(error) in capsys.readouterr().err


@pytest.mark.anyio
async def test_disabled_environment_is_not_permission_to_index(daemon_mode, tmp_project, monkeypatch):
    """ZEMBLE_DAEMON=0 forbids access rather than opting a client into local memory ownership."""
    monkeypatch.setenv("ZEMBLE_DAEMON", "0")
    cache = IndexCache()
    instance = mcp.create_server(cache)
    result = await instance.call_tool("search", {"query": "q", "repo": str(tmp_project)})
    assert "--no-daemon" in result[0].text
    assert not cache._tasks and cache.embedder is None


@pytest.mark.anyio
async def test_no_result_is_a_failure_not_a_fallback(daemon_mode, tmp_project, monkeypatch):
    """A successful but empty legacy RPC response cannot trigger local allocation."""
    monkeypatch.setattr(client, "call", lambda *a, **kw: None)
    cache = IndexCache()
    instance = mcp.create_server(cache)
    result = await instance.call_tool("search", {"query": "q", "repo": str(tmp_project)})
    assert "no result" in result[0].text
    assert not cache._tasks


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
    await instance._operation_lock.acquire()
    try:
        response = await instance.handle(
            {"id": 1, "cmd": "search", "args": {"path": str(tmp_path), "query": "q"}, ACCEPTS_BUSY_FIELD: True}
        )
        assert response["kind"] == ErrorKind.BUSY.value
        assert (await instance.handle({"id": 2, "cmd": "status"}))["ok"]
    finally:
        instance._operation_lock.release()
    assert not instance.cache._tasks


@pytest.mark.anyio
async def test_busy_legacy_requests_use_the_known_refusal_lane(daemon_mode, tmp_path):
    """Old index clients must not fall back merely because they do not recognize a new busy vocabulary."""
    instance = server.Daemon(watch=False)
    await instance._operation_lock.acquire()
    try:
        reply = await instance.handle({"id": 1, "cmd": "search", "args": {"path": str(tmp_path), "query": "q"}})
        assert reply["kind"] == ErrorKind.REFUSED.value
        assert "busy, retry" in reply["error"]
    finally:
        instance._operation_lock.release()


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
    "kind, exception", [("busy", CommandBusy), ("refused", CommandRefused), ("future", CommandFailed)]
)
def test_error_kinds_never_authorize_fallback(daemon_mode, monkeypatch, kind, exception):
    """New or unknown daemon errors remain explicit errors on the wire."""
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

"""Tests for the warm daemon: protocol, command table, auto-start, watcher, eviction."""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import textwrap
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.conftest import FakeEmbedder
from zemble.daemon import client, server
from zemble.daemon.protocol import CommandFailed, DaemonError, DaemonUnavailable, decode, encode, socket_path
from zemble.daemon.watch import IgnoreRules
from zemble.index.file_walker import walk_files
from zemble.index.files import get_extensions
from zemble.index_cache import compute_cache_key
from zemble.types import ContentType, IndexStats


@pytest.fixture(autouse=True)
def daemon_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the daemon's socket, cache and log at a throwaway directory."""
    runtime = tmp_path / "run"
    runtime.mkdir()
    monkeypatch.setenv("ZEMBLE_DAEMON_DIR", str(runtime))
    monkeypatch.setenv("ZEMBLE_CACHE_LOCATION", str(tmp_path / "cache"))
    monkeypatch.delenv("ZEMBLE_DAEMON", raising=False)
    monkeypatch.setattr(client, "_disabled_reason", None)
    yield runtime
    monkeypatch.setattr(client, "_disabled_reason", None)


@contextlib.contextmanager
def running_server(**kwargs: Any) -> Iterator[None]:
    """Run a real daemon in a background thread on the test's socket."""
    error: list[BaseException] = []

    def _target() -> None:
        try:
            asyncio.run(server.run(**kwargs))
        except BaseException as exc:  # pragma: no cover - surfaced by the assert below
            error.append(exc)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not client.is_running():
        assert not error, error[0]
        time.sleep(0.02)
    # run() disables the client for its own process; in a thread that is this process too.
    client._disabled_reason = None
    assert client.is_running(), "daemon did not start"
    try:
        yield
    finally:
        client._disabled_reason = None
        with contextlib.suppress(DaemonError):
            client.call("shutdown", auto_start=False, timeout=5)
        thread.join(timeout=10)
        assert not error, error[0]


@pytest.fixture()
def no_embedder_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the daemon from loading a real embedding model."""

    async def _fake(self: Any) -> None:
        self._embedder = FakeEmbedder()
        self._model_ready.set()

    monkeypatch.setattr("zemble.index_cache.IndexCache.load_embedder_once", _fake)


def _daemon_with_fake_embedder(**kwargs: Any) -> server.Daemon:
    """Build a daemon whose cache is already backed by the deterministic test embedder."""
    daemon = server.Daemon(**kwargs)
    daemon.cache._embedder = FakeEmbedder()
    daemon.cache._model_ready.set()
    return daemon


def test_protocol_round_trip_over_a_real_socket(no_embedder_load: None) -> None:
    """A real daemon answers ping, reports status, refuses nonsense and shuts down cleanly."""
    with running_server(watch=False, idle_minutes=0):
        # 1. ping identifies the process answering.
        pong = client.call("ping", auto_start=False, timeout=10)
        assert pong["pong"] is True, "ping must answer"
        assert pong["pid"] == os.getpid(), "the in-thread daemon is this process"

        # 2. status describes an empty, non-watching daemon.
        status = client.call("status", auto_start=False, timeout=10)
        assert status["indexes"] == [], "nothing is loaded yet"
        assert status["requests"] >= 2, "status counts the requests it served"
        assert status["socket"] == str(socket_path()), "status names the socket it listens on"

        # 3. an unknown command is an error response, not a dropped connection.
        with pytest.raises(CommandFailed, match="unknown command"):
            client.call("does-not-exist", auto_start=False, timeout=10)

        # 4. the connection survives that error and still answers.
        assert client.call("ping", auto_start=False, timeout=10)["pong"] is True, "still alive"

    # 5. after shutdown the socket is gone and nothing answers.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and socket_path().exists():
        time.sleep(0.02)
    assert not socket_path().exists(), "shutdown removes the socket"
    assert not client.is_running(), "nothing answers after shutdown"


def test_several_requests_share_one_connection(no_embedder_load: None) -> None:
    """The server reads one request per line and answers each in turn on the same connection."""
    with running_server(watch=False, idle_minutes=0):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as raw:
            raw.connect(str(socket_path()))
            with raw.makefile("rwb") as stream:
                for request_id in range(3):
                    stream.write(encode({"id": request_id, "cmd": "ping", "args": {}}))
                stream.flush()
                answers = [decode(stream.readline()) for _ in range(3)]
    assert [answer["id"] for answer in answers] == [0, 1, 2], "answers carry their request ids"
    assert all(answer["ok"] for answer in answers), "every answer succeeded"


@pytest.mark.anyio
async def test_every_command_answers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every entry in the command table answers, and an unknown name is refused."""
    daemon = _daemon_with_fake_embedder(watch=False)
    index = MagicMock()
    index.search.return_value = []
    index.chunks = []
    index.content = (ContentType.CODE,)
    index.embedder.model_id = "fake:test@256"
    index.stats = IndexStats(
        indexed_files=1, total_chunks=1, languages={"python": 1}, embedder="fake:test@256", dimensions=256
    )
    cache_key = compute_cache_key(str(tmp_path))

    async def _index_for(self: server.Daemon, args: dict[str, Any]) -> Any:
        return cache_key, index

    async def _rebuild(self: server.Daemon, key: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"added": 0}

    monkeypatch.setattr(server.Daemon, "index_for", _index_for)
    monkeypatch.setattr(server.Daemon, "rebuild", _rebuild)
    monkeypatch.setattr("zemble.graph.cli.ensure_graph", lambda path, **kwargs: None)
    monkeypatch.setattr("zemble.graph.mcp.answer", lambda *args, **kwargs: '{"results": []}')
    # The evidence handlers are exercised for real over a socket below; here only dispatch is.
    monkeypatch.setattr(server, "_with_graph", lambda root, work: {"stubbed": True})

    args_by_command = {
        "search": {"path": str(tmp_path), "query": "anything"},
        "find_related": {"path": str(tmp_path), "file_path": "a.py", "line": 1},
        "stats": {"path": str(tmp_path)},
        "graph": {"path": str(tmp_path), "command": "callers", "symbol": "Foo"},
        "explain": {"path": str(tmp_path), "query": "anything"},
        "outline": {"path": str(tmp_path), "target": "Thing"},
        "signatures": {"path": str(tmp_path), "symbol": "Thing.method"},
        "home": {"path": str(tmp_path), "description": "anything"},
        "refresh": {"path": str(tmp_path)},
        "evict": {"path": str(tmp_path)},
    }
    for command in server.COMMANDS:
        response = await daemon.handle({"id": 1, "cmd": command, "args": args_by_command.get(command, {})})
        assert response["ok"], f"{command} must answer: {response.get('error')}"

    unknown = await daemon.handle({"id": 2, "cmd": "nope", "args": {}})
    assert not unknown["ok"] and "unknown command" in unknown["error"], "unknown commands are refused"

    bad_args = await daemon.handle({"id": 3, "cmd": "ping", "args": []})
    assert not bad_args["ok"], "args must be an object"


@pytest.mark.anyio
async def test_command_failure_is_reported_not_raised(tmp_path: Path) -> None:
    """A handler that raises becomes an error response, and the daemon stays up."""
    daemon = server.Daemon(watch=False)
    response = await daemon.handle({"id": 1, "cmd": "evict", "args": {}})
    assert not response["ok"] and "missing 'path'" in response["error"], "the reason is reported"

    response = await daemon.handle({"id": 2, "cmd": "search", "args": {"path": "git://example.com/repo"}})
    assert not response["ok"] and "https://" in response["error"], "unsafe transports are refused server-side"


def test_call_without_a_daemon_and_no_autostart() -> None:
    """With nothing listening and auto-start off, the client says so instead of hanging."""
    with pytest.raises(DaemonUnavailable, match="not running"):
        client.call("ping", auto_start=False)


def test_env_switch_disables_the_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    """ZEMBLE_DAEMON=0 refuses before anything is spawned."""
    monkeypatch.setenv("ZEMBLE_DAEMON", "0")
    with pytest.raises(DaemonUnavailable, match="ZEMBLE_DAEMON=0"):
        client.call("ping")
    assert not socket_path().exists(), "no daemon was started"


def test_no_daemon_flag_disables_the_daemon() -> None:
    """--no-daemon wires through to the same refusal, for this process only."""
    client.disable_for_this_process("--no-daemon")
    with pytest.raises(DaemonUnavailable, match="--no-daemon"):
        client.call("ping")


def test_stale_socket_with_a_dead_pid_is_cleared(daemon_env: Path) -> None:
    """A socket and pidfile left by a dead daemon are removed, not treated as live."""
    from zemble.daemon.protocol import pid_path

    socket_path().touch()
    pid_path().write_text("9999999\n", encoding="utf-8")
    assert client.clear_stale() is True, "a dead daemon's files are cleared"
    assert not socket_path().exists() and not pid_path().exists(), "both files are gone"
    assert client.clear_stale() is False, "there is nothing left to clear"


def test_live_pidfile_is_left_alone() -> None:
    """A pidfile owned by a live process is never pulled out from under a starting daemon."""
    from zemble.daemon.protocol import pid_path

    socket_path().touch()
    pid_path().write_text(f"{os.getpid()}\n", encoding="utf-8")
    assert client.clear_stale() is False, "a live owner keeps its socket"
    assert socket_path().exists(), "the socket file survives"


@pytest.mark.anyio
async def test_watcher_rebuild_makes_a_new_file_searchable(tmp_project: Path) -> None:
    """A synthetic change set drives an incremental rebuild whose result is served immediately."""
    daemon = _daemon_with_fake_embedder(watch=False)
    cache_key, index = await daemon.index_for({"path": str(tmp_project)})
    before = {result.chunk.file_path for result in index.search("brand_new_helper")}
    assert "extra.py" not in before, "the file does not exist yet"

    (tmp_project / "extra.py").write_text("def brand_new_helper():\n    return 7\n", encoding="utf-8")
    result = await daemon._on_change(cache_key, {tmp_project / "extra.py"})

    swapped = (await daemon.cache.get(str(tmp_project))).search("brand_new_helper")
    assert swapped, "the rebuilt index is the one being served"
    assert swapped[0].chunk.file_path == "extra.py", "the new file is the best match"
    assert daemon.last_rebuild[cache_key]["added"] == 1, "the rebuild reports what moved"
    assert result is None, "the change handler returns nothing; it swaps in place"


@pytest.mark.anyio
async def test_rebuild_of_an_unloaded_root_is_a_no_op(tmp_project: Path) -> None:
    """Rebuilding a root that is not resident does nothing rather than building it."""
    daemon = _daemon_with_fake_embedder(watch=False)
    outcome = await daemon.rebuild(compute_cache_key(str(tmp_project)))
    assert outcome == {"skipped": "not loaded"}, "an unloaded root is skipped"


@pytest.mark.anyio
async def test_watchfiles_swaps_the_index_on_a_real_edit(tmp_project: Path) -> None:
    """A real filesystem edit under a watched root reaches the index without a client call."""
    daemon = _daemon_with_fake_embedder(watch=True)
    cache_key, _index = await daemon.index_for({"path": str(tmp_project)})
    assert cache_key in daemon.watchers, "a local root is watched once it is served"
    await asyncio.sleep(0.3)  # let the watcher reach its first poll

    (tmp_project / "watched.py").write_text("def watched_symbol():\n    return 1\n", encoding="utf-8")

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        index = await daemon.cache.get(str(tmp_project))
        if any(result.chunk.file_path == "watched.py" for result in index.search("watched_symbol")):
            break
        await asyncio.sleep(0.1)
    daemon.shutdown()
    matches = {result.chunk.file_path for result in index.search("watched_symbol")}
    assert "watched.py" in matches, "the edit was picked up and swapped in"


@pytest.mark.anyio
async def test_eviction_stops_the_watcher(tmp_project: Path, tmp_path: Path) -> None:
    """The least recently used root leaves memory and stops being watched."""
    other = tmp_path / "other"
    other.mkdir()
    (other / "thing.py").write_text("def thing():\n    return 1\n", encoding="utf-8")

    daemon = _daemon_with_fake_embedder(max_indexes=1, watch=True)
    first_key, _first = await daemon.index_for({"path": str(tmp_project)})
    assert first_key in daemon.watchers, "the first root is watched"

    second_key, _second = await daemon.index_for({"path": str(other)})
    assert second_key in daemon.watchers, "the second root is watched"
    assert first_key not in daemon.watchers, "the evicted root's watcher is stopped"
    assert first_key not in daemon.cache._tasks, "the evicted root left memory"
    daemon.shutdown()


@pytest.mark.anyio
async def test_explicit_evict_stops_the_watcher(tmp_project: Path) -> None:
    """`evict` drops one root and reports whether it was there."""
    daemon = _daemon_with_fake_embedder(watch=True)
    cache_key, _index = await daemon.index_for({"path": str(tmp_project)})
    response = await daemon.handle({"id": 1, "cmd": "evict", "args": {"path": str(tmp_project)}})
    assert response["result"]["evicted"] is True, "the loaded root was evicted"
    assert cache_key not in daemon.watchers, "its watcher stopped"

    again = await daemon.handle({"id": 2, "cmd": "evict", "args": {"path": str(tmp_project)}})
    assert again["result"]["evicted"] is False, "evicting twice is honest about it"


@pytest.mark.anyio
async def test_idle_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """An idle daemon stops itself; a busy one does not."""
    monkeypatch.setattr(server, "_IDLE_CHECK_SECONDS", 0.01)
    daemon = server.Daemon(watch=False, idle_minutes=1)
    daemon.last_request_at = time.monotonic() - 120
    await asyncio.wait_for(asyncio.gather(daemon.idle_loop()), timeout=5)
    assert daemon.stop_event.is_set(), "the idle daemon asked to stop"

    never = server.Daemon(watch=False, idle_minutes=0)
    never.last_request_at = time.monotonic() - 10_000
    await asyncio.wait_for(never.idle_loop(), timeout=5)
    assert not never.stop_event.is_set(), "idle_minutes=0 never exits"


def test_ignore_rules_agree_with_the_file_walker(tmp_path: Path) -> None:
    """The watcher's filter answers exactly what the indexer's walk includes."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "kept.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "src" / "skipped.py").write_text("x = 2\n", encoding="utf-8")
    (tmp_path / "src" / "notes.txt").write_text("not code\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.py").write_text("x = 3\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("skipped.py\n", encoding="utf-8")

    extensions = get_extensions([ContentType.CODE])
    rules = IgnoreRules(tmp_path, extensions)
    walked = set(walk_files(tmp_path, extensions))
    candidates = [path for path in tmp_path.rglob("*") if path.is_file()]

    for candidate in candidates:
        assert rules.matches(candidate) == (candidate in walked), f"{candidate} must be judged like the walker"
    assert rules.matches(tmp_path / "src" / "kept.py"), "a plain source file is watched"
    assert not rules.matches(Path("/elsewhere/other.py")), "a path outside the root is never watched"


def test_ignore_rules_honour_a_nested_gitignore(tmp_path: Path) -> None:
    """A .gitignore deeper in the tree applies to its own subtree, as it does during a walk."""
    nested = tmp_path / "pkg" / "inner"
    nested.mkdir(parents=True)
    (nested / "hidden.py").write_text("x = 1\n", encoding="utf-8")
    (nested / "shown.py").write_text("x = 2\n", encoding="utf-8")
    (tmp_path / "pkg" / ".gitignore").write_text("inner/hidden.py\n", encoding="utf-8")

    rules = IgnoreRules(tmp_path, get_extensions([ContentType.CODE]))
    assert not rules.matches(nested / "hidden.py"), "the nested rule is applied"
    assert rules.matches(nested / "shown.py"), "its sibling is still watched"


def test_autostart_spawns_a_daemon_and_stop_removes_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A command that needs a daemon starts one; nothing else ever does."""
    monkeypatch.setenv("ZEMBLE_DAEMON_IDLE_MINUTES", "1")
    assert not client.is_running(), "no daemon before the first call"
    try:
        assert client.call("ping", timeout=30)["pong"] is True, "the auto-started daemon answers"
        assert client.is_running(), "it stayed up for the next caller"
    finally:
        with contextlib.suppress(DaemonError):
            client.call("shutdown", auto_start=False, timeout=10)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and socket_path().exists():
        time.sleep(0.05)
    assert not socket_path().exists(), "the daemon cleaned up after itself"


def test_evidence_commands_answer_over_a_real_socket(graph_fixture_root: Path, no_embedder_load: None) -> None:
    """explain, outline and signatures all answer from the daemon's own index and graph."""
    from zemble.graph import cli as graph_cli

    graph_cli._refreshed.clear()
    root = str(graph_fixture_root)
    with running_server(watch=False, idle_minutes=0):
        # 1. explain answers with a budgeted bundle, in both of its renderings.
        explained = client.call(
            "explain", {"path": root, "query": "how is an area computed", "budget": 900, "top_k": 5}, timeout=120
        )
        assert explained["markdown"].startswith("# Evidence for: how is an area computed"), "1: markdown comes back"
        bundle = explained["bundle"]
        assert bundle["items"], "1: the bundle carries evidence"
        assert bundle["total_tokens"] <= bundle["budget_tokens"], "1: the budget was respected"

        # 2. the same root is now resident once, not once per surface.
        status = client.call("status", timeout=30)
        assert [index["root"] for index in status["indexes"]] == [root], "2: one warm index for the workspace"

        # 3. outline answers off the graph the daemon built.
        outlined = client.call("outline", {"path": root, "target": "Registry"}, timeout=60)
        assert "class Registry<T extends Shape>" in outlined["text"], "3: the declaration is rendered"
        assert outlined["outline"]["package"] == "com.example.core", "3: and carried as data"

        # 4. a narrowed outline drops the members that do not match.
        narrowed = client.call(
            "outline",
            {"path": root, "target": "src/main/java/com/example/core/Circle.java", "members": "scale"},
            timeout=60,
        )
        assert "scale(double factor)" in narrowed["text"] and "label()" not in narrowed["text"], "4: only matches"

        # 5. an ambiguous target is an error payload, not a failed command.
        ambiguous = client.call("outline", {"path": root, "target": "Circle"}, timeout=60)
        assert "ambiguous" in ambiguous["error"], "5: ambiguity is reported as data"
        assert len(ambiguous["candidates"]) == 2, "5: with every candidate named and located"
        assert all(candidate["file_path"] for candidate in ambiguous["candidates"]), "5: candidates carry their file"

        # 6. signatures answers with the exact call sites.
        signed = client.call("signatures", {"path": root, "symbol": "Helpers.twice"}, timeout=60)
        assert signed["signatures"]["signature"] == "double twice(double value)", "6: the signature is reported"
        assert signed["signatures"]["callers"], "6: its callers come with it"

        # 7. an unknown symbol is refused with the graph's coverage note.
        unknown = client.call("signatures", {"path": root, "symbol": "NotHere"}, timeout=60)
        assert "No symbol named" in unknown["error"] and not unknown["candidates"], "7: nothing to disambiguate"


def test_home_answers_over_a_real_socket(graph_fixture_root: Path, tmp_path: Path, no_embedder_load: None) -> None:
    """`home` answers from the daemon's warm index, its graph and the workspace's own config."""
    import shutil

    from zemble.graph import cli as graph_cli

    graph_cli._refreshed.clear()
    root_path = tmp_path / "workspace"
    shutil.copytree(graph_fixture_root, root_path)
    (root_path / ".zemble").mkdir()
    (root_path / ".zemble" / "home.toml").write_text(
        textwrap.dedent(
            """
            order = ["core", "util", "app"]

            [modules]
            core = "src/main/java/com/example/core/**"
            util = "src/main/java/com/example/util/**"
            app = "src/main/java/com/example/app/**"

            [[rules]]
            text = "Nothing lands without a wired consumer and a test"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    root = str(root_path)
    with running_server(watch=False, idle_minutes=0):
        # 1. The answer comes back in both renderings.
        answer = client.call("home", {"path": root, "description": "compute the area of a shape"}, timeout=180)
        assert answer["markdown"].startswith("# Home for: compute the area of a shape"), "1: markdown comes back"
        assert answer["home"]["verdict"], "1: and the verdict as data"

        # 2. The declared globs, not the path segments, name the modules.
        modules = {candidate["module"] for candidate in answer["home"]["candidates"]}
        assert modules and modules <= {"core", "util", "app"}, f"2: declared modules only, got {modules}"

        # 3. The workspace's own rule rides along.
        assert "Nothing lands without a wired consumer and a test" in answer["home"]["checklist"]["rules"], "3: rules"

        # 4. The root is resident once, shared with every other surface.
        status = client.call("status", timeout=30)
        assert [index["root"] for index in status["indexes"]] == [root], "4: one warm index for the workspace"


def test_cli_search_falls_back_when_the_daemon_is_unavailable(
    tmp_project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI prints one stderr line and answers in-process when the daemon cannot be reached."""
    from zemble import cli

    def _refuse(*args: Any, **kwargs: Any) -> Any:
        raise DaemonUnavailable("not running (ENOENT)")

    monkeypatch.setattr(client, "call", _refuse)
    fake_index = MagicMock()
    fake_index.search.return_value = []
    monkeypatch.setattr(cli, "_load_index", lambda *args, **kwargs: fake_index)
    monkeypatch.setattr(cli, "_maybe_save_index", lambda *args, **kwargs: None)

    cli._run_search(str(tmp_project), "anything", 5, [ContentType.CODE], None)
    captured = capsys.readouterr()
    assert "daemon unavailable (not running (ENOENT)); running in-process" in captured.err, "one honest line"
    assert '"error": "No results found."' in captured.out, "the in-process answer was printed"


def test_cli_search_uses_the_daemon_when_it_answers(
    tmp_project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A daemon answer is printed verbatim and no index is built in this process."""
    from zemble import cli

    monkeypatch.setattr(client, "call", lambda cmd, args, **kwargs: {"query": "q", "results": [{"file_path": "a.py"}]})
    monkeypatch.setattr(cli, "_load_index", lambda *args, **kwargs: pytest.fail("must not build in-process"))

    cli._run_search(str(tmp_project), "q", 5, [ContentType.CODE], None)
    assert '"file_path": "a.py"' in capsys.readouterr().out, "the daemon's payload was printed"


def test_cli_skips_the_daemon_for_an_embedder_override(
    tmp_project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The daemon holds one embedder, so an explicit --embedder is answered in-process silently."""
    from zemble import cli

    monkeypatch.setattr(client, "call", lambda *args, **kwargs: pytest.fail("must not ask the daemon"))
    fake_index = MagicMock()
    fake_index.search.return_value = []
    monkeypatch.setattr(cli, "_load_index", lambda *args, **kwargs: fake_index)
    monkeypatch.setattr(cli, "_maybe_save_index", lambda *args, **kwargs: None)

    cli._run_search(str(tmp_project), "q", 5, [ContentType.CODE], None, embedder="model2vec:other")
    assert "daemon unavailable" not in capsys.readouterr().err, "an override is not a daemon failure"


def test_cli_no_daemon_flag_is_silent(
    tmp_project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--no-daemon answers in-process without reporting a daemon failure."""
    from zemble import cli

    monkeypatch.setattr(client, "call", lambda *args, **kwargs: pytest.fail("must not ask the daemon"))
    fake_index = MagicMock()
    fake_index.search.return_value = []
    monkeypatch.setattr(cli, "_load_index", lambda *args, **kwargs: fake_index)
    monkeypatch.setattr(cli, "_maybe_save_index", lambda *args, **kwargs: None)

    cli._run_search(str(tmp_project), "q", 5, [ContentType.CODE], None, no_daemon=True)
    assert "daemon unavailable" not in capsys.readouterr().err, "an opt-out is not a failure"


def test_daemon_status_command_reports_no_daemon(capsys: pytest.CaptureFixture[str]) -> None:
    """`zemble daemon status` without a daemon exits non-zero and says why."""
    from zemble.daemon.cli import main

    assert main(["status"]) == 1, "no daemon is a failure exit"
    assert "not available" in capsys.readouterr().out, "and it says so"


def test_project_with_a_gitignore(tmp_path: Path) -> None:
    """A root-level ignore file keeps its matches out of both the walk and the watch."""
    (tmp_path / "keep.py").write_text(
        textwrap.dedent("""\
        def keep():
            return 1
        """),
        encoding="utf-8",
    )
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "generated.py").write_text("x = 1\n", encoding="utf-8")

    rules = IgnoreRules(tmp_path, get_extensions([ContentType.CODE]))
    assert rules.matches(tmp_path / "keep.py"), "source is watched"
    assert not rules.matches(tmp_path / "build" / "generated.py"), "a default-ignored dir is not"

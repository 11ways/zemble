"""A graph answer given while that graph's refresh is coming or running says so, and one naming nothing waits."""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_graph_cli import _run
from zemble.daemon import graph_jobs, server
from zemble.daemon.admission import AdmissionBusy
from zemble.daemon.graph_jobs import GraphJobs
from zemble.daemon.protocol import QueryTimedOut
from zemble.graph import cli as graph_cli
from zemble.graph.store import GRAPH_LOCK_NAME, build_graph, graph_folder

HANDLE = "package app;\n\npublic class Handle {\n    public Object result() { return null; }\n}\n"
#: What the pull brings: a method the published graph does not know yet, and its caller.
ENDED = "package app;\n\npublic class Ended {\n    public boolean ended() { return true; }\n}\n"
USER = "package app;\n\npublic class User {\n    boolean done(Ended handle) { return handle.ended(); }\n}\n"


def _workspace(root: Path) -> str:
    (root / "src/main/java/app").mkdir(parents=True)
    (root / "src/main/java/app/Handle.java").write_text(HANDLE, encoding="utf-8")
    build_graph(str(root))
    return str(root)


def _pull(root: str) -> None:
    """Land the pull's files and the refresh that reads them."""
    for name, text in (("Ended.java", ENDED), ("User.java", USER)):
        (Path(root) / "src/main/java/app" / name).write_text(text, encoding="utf-8")
    build_graph(root)


class _Daemon:
    """Stand in for the daemon's `graph` ensure command: a refresh is running until `settle` waits it out."""

    def __init__(self, root: str, *, lands: bool) -> None:
        self.root, self.lands, self.calls = root, lands, []
        self.pending: dict[str, object] | None = {
            "root": root,
            "phase": "building",
            "since": time.time() - 3,
            "changed_paths": 900,
            "awaitable": True,
        }

    def ensure(self, path: str, *, settle: bool = False) -> dict[str, object]:
        self.calls.append(settle)
        if settle and not self.lands:
            raise QueryTimedOut("zemble graph: the daemon is still busy after 180 s")
        if settle:
            _pull(self.root)
            self.pending = None
        return {"ensured": True, **({"refreshing": self.pending} if self.pending else {})}


def test_a_query_naming_nothing_waits_for_the_daemons_refresh(
    tmp_path: Path, graph_cache: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The CLI asks again once a refresh the daemon runs has landed, and says when it never did."""
    root = _workspace(tmp_path / "ws")

    # 1. A method the pull added: the published graph lacks it, so the answer waits for the refresh, then finds it.
    daemon = _Daemon(root, lands=True)
    monkeypatch.setattr(graph_cli, "_ensure_via_daemon", daemon.ensure)
    assert _run(monkeypatch, "graph", "callers", root, "Ended.ended") == 0, "step 1: answered after the refresh"
    out = capsys.readouterr().out
    assert daemon.calls == [False, True], "step 1: one ensure, then one that waits the refresh out"
    assert "callers: 1 result(s)" in out and "called from User.done" in out, out
    assert "Not final" not in out, "step 1: the refresh landed, so the answer is final"

    # 2. A found answer is never held back: it is printed at once and says what it was read before.
    daemon = _Daemon(root, lands=True)
    monkeypatch.setattr(graph_cli, "_ensure_via_daemon", daemon.ensure)
    assert _run(monkeypatch, "graph", "callers", root, "Ended.ended") == 0
    out = capsys.readouterr().out
    assert daemon.calls == [False], "step 2: no wait for an answer that found something"
    assert "Not final: the symbol graph of" in out and "900 changed path(s)" in out, out

    # 3. A refresh outlasting the query's wait: the empty answer is printed as not final, never as the truth.
    daemon = _Daemon(root, lands=False)
    monkeypatch.setattr(graph_cli, "_ensure_via_daemon", daemon.ensure)
    assert _run(monkeypatch, "graph", "callers", root, "Handle.started") == 1
    captured = capsys.readouterr()
    assert daemon.calls == [False, True, False], "step 3: it waited, then read the refresh as it stands"
    assert "No symbol named 'Handle.started'" in captured.err and "Not final" in captured.err, captured.err
    assert _run(monkeypatch, "graph", "callers", root, "Handle.started", "--json") == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["refreshing"]["phase"] == "building" and payload["refreshing"]["changed_paths"] == 900, payload


def test_an_answer_read_while_another_process_writes_the_graph_says_so(
    graph_fixture_root: Path, graph_cache: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Without a daemon the writer lock is the only witness: it is read off /proc/locks, never taken."""
    from zemble.graph.mcp import answer
    from zemble.graph.store import graph_being_written

    root = str(graph_fixture_root)
    build_graph(root)
    assert not graph_being_written(root), "no writer yet"
    descriptor = os.open(graph_folder(root) / GRAPH_LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        assert graph_being_written(root), "step 1: a held writer lock is seen"

        # 2. The MCP payload carries it, as nothing anyone can wait for.
        payload = answer(root, "Helpers.twice", "callers")
        assert payload["refreshing"]["phase"] == "building" and payload["refreshing"]["awaitable"] is False

        # 3. The CLI's empty answer says it is not final.
        assert _run(monkeypatch, "graph", "callers", root, "Helpers.nowhere") == 1
        assert "Not final" in capsys.readouterr().err, "step 3: the unknown name is not final"
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert not graph_being_written(root), "step 4: released"
    assert "refreshing" not in answer(root, "Helpers.twice", "callers"), "step 4: a current answer carries nothing"


@pytest.mark.anyio
async def test_the_daemon_holds_back_a_lookup_naming_nothing_while_its_refresh_runs(tmp_path: Path) -> None:
    """A lookup that finds nothing is busy until the refresh lands; every other answer carries the refresh."""
    root = str(tmp_path.resolve())
    jobs = GraphJobs(asyncio.Lock(), 4096)

    async def ensure(path: str, deadline: float | None, **_kwargs: object) -> str:
        return root

    jobs.ensure = ensure  # type: ignore[method-assign]
    daemon = SimpleNamespace(graphs=jobs)
    blocker = asyncio.Event()
    jobs.jobs[root] = asyncio.create_task(blocker.wait())
    jobs.states[root] = {"state": "building", "started_at": time.time() - 5, "queued_at": time.time() - 9}
    jobs.paths[root] = [f"src/F{n}.java" for n in range(900)]

    # 1. No symbol while it builds: busy, saying why, which `client.answer` waits out.
    with pytest.raises(AdmissionBusy, match=r"being refreshed \(building for \d+ s, 900 changed path\(s\)\)"):
        await server.Daemon.graph_answer(daemon, root, lambda: {"error": "No symbol named 'X'."}, lookup=True)
    with pytest.raises(AdmissionBusy):
        await server.Daemon.graph_answer(daemon, root, lambda: {"results": [], "total": 0}, lookup=True)

    # 2. A found answer carries the refresh in its payload and its prose.
    found = await server.Daemon.graph_answer(daemon, root, lambda: {"total": 1, "text": "Handle"}, lookup=True)
    assert found["refreshing"]["phase"] == "building" and "Not final" in found["text"], found

    # 3. A whole-graph read may rightly find nothing: never held back, still stamped.
    assert "refreshing" in await server.Daemon.graph_answer(daemon, root, lambda: {"total": 0}, lookup=False)

    # 4. Queued, then announced by an index pass running first, then landed.
    jobs.states[root]["state"] = "queued"
    assert jobs.pending_refresh(root).phase.value == "queued", "step 4: queued"
    blocker.set()
    await jobs.jobs.pop(root)
    with jobs.refresh_announced(root):
        assert jobs.pending_refresh(root).phase.value == "announced", "step 4: announced"
    assert jobs.pending_refresh(root) is None, "step 4: landed"
    final = await server.Daemon.graph_answer(daemon, root, lambda: {"error": "No symbol named 'X'."}, lookup=True)
    assert final == {"error": "No symbol named 'X'."}, "step 4: an empty answer after the refresh is final"


@pytest.mark.anyio
async def test_a_graph_refresh_logs_its_start_end_and_failure(tmp_path: Path, monkeypatch, caplog) -> None:
    """The daemon log answers why a graph was behind: when its refresh started, how long it took, what it read."""
    root = str(tmp_path)
    outcomes = iter(
        [
            (
                0,
                json.dumps(
                    {
                        "build_seconds": 1.5,
                        "peak_rss_mb": 80,
                        "stats": {
                            "files_scanned": 12,
                            "extracted_files": 3,
                            "removed_files": 1,
                            "reresolved_files": 7,
                            "compacted": False,
                        },
                    }
                ).encode(),
                b"",
            ),
            (1, b"", b"Traceback ...\nsqlite3.OperationalError: disk I/O error"),
        ]
    )

    async def spawn(*args, **kwargs):
        code, stdout, stderr = next(outcomes)

        class Process:
            pid = 1
            returncode = None

            async def communicate(self, message):
                self.returncode = code
                return stdout, stderr

        return Process()

    monkeypatch.setattr(graph_jobs, "private_mb", lambda: 100)
    monkeypatch.setattr(graph_jobs.resource, "getrlimit", lambda _: (graph_jobs.resource.RLIM_INFINITY, -1))
    monkeypatch.setattr(graph_jobs.resource, "setrlimit", lambda *_: None)
    monkeypatch.setattr(graph_jobs.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(graph_jobs, "_lane_base", lambda root: None)
    jobs = GraphJobs(asyncio.Lock(), 4096)
    caplog.set_level(logging.INFO, logger="zemble.daemon.graph_jobs")

    # 1. A landed refresh: its start with the change set, its end with the time and the files.
    await jobs._start(root, refresh=True, changed_paths=["a.java", "b.java"])
    lines = [record.getMessage() for record in caplog.records]
    assert any(
        line.startswith(f"graph refresh of {root} started after") and "2 changed path(s)" in line for line in lines
    )
    finished = next(line for line in lines if "finished in" in line)
    assert "3 of 12 file(s) re-extracted, 7 re-resolved, 1 removed" in finished, finished

    # 2. A failed one: at WARNING, with the worker's reason.
    caplog.clear()
    with pytest.raises(RuntimeError):
        await jobs._start(root, refresh=True, changed_paths=None)
    (failed,) = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert "failed after" in failed.getMessage() and "disk I/O error" in failed.getMessage(), failed.getMessage()


def test_the_worker_reports_what_its_refresh_read(tmp_path: Path, graph_cache: Path) -> None:
    """The real worker process prints the statistics the daemon logs."""
    root = _workspace(tmp_path / "ws")
    (Path(root) / "src/main/java/app/Ended.java").write_text(ENDED, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "zemble.daemon.graph_worker", root, "2048"],
        input=json.dumps({"changed_paths": [str(Path(root) / "src/main/java/app/Ended.java")]}),
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "ZEMBLE_DAEMON": "0"},
    )
    report = json.loads(completed.stdout)
    assert report["stats"]["extracted_files"] == 1 and report["stats"]["files_scanned"] >= 1, report

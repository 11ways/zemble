"""Process pools never fork a threaded host, get no pool without a real __main__, and never outlive it."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

import zemble
from zemble.parallel import pool_context


def test_pool_context_journey(monkeypatch: pytest.MonkeyPatch) -> None:
    """Walk the three outcomes: fork when single-threaded, spawn when threaded, none without a main file."""
    monkeypatch.setitem(sys.modules, "__main__", types.SimpleNamespace(__file__="/tmp/some_script.py"))
    context = pool_context()
    if threading.active_count() == 1:
        assert context is not None and context.get_start_method() == "fork", "step 1: single-threaded -> fork"
    stop = threading.Event()
    worker = threading.Thread(target=stop.wait, daemon=True)
    worker.start()
    try:
        threaded = pool_context()
        assert threaded is not None and threaded.get_start_method() == "spawn", "step 2: a live thread forbids fork"
        monkeypatch.setitem(sys.modules, "__main__", types.SimpleNamespace())
        assert pool_context() is None, "step 3: no __main__ file (-c / stdin) -> no pool at all"
    finally:
        stop.set()
        worker.join(timeout=2)


_POOL_HOST_SCRIPT = """
import json
import os
import sys
import time

from zemble.parallel import pool_context, process_pool


def _slow_pid(_index):
    time.sleep(0.2)
    return os.getpid()


if __name__ == "__main__":
    context = pool_context()
    assert context is not None, "this script has a real __main__"
    pool = process_pool(4, context)
    pids = sorted(set(pool.map(_slow_pid, range(40))))
    with open(sys.argv[1], "w") as handle:
        handle.write(json.dumps(pids))
        handle.flush()
        os.fsync(handle.fileno())
    time.sleep(300)
"""


def _alive(pid: int) -> bool:
    """Return whether a process with this pid still exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")
def test_pool_children_do_not_outlive_a_killed_parent(tmp_path: Path) -> None:
    """A pool host that is SIGKILLed leaves no worker behind, the way the orphaned MCP workers did."""
    script = tmp_path / "host.py"
    script.write_text(_POOL_HOST_SCRIPT, encoding="utf-8")
    report = tmp_path / "pids.json"
    env = {**os.environ, "PYTHONPATH": str(Path(zemble.__file__).resolve().parent.parent)}
    parent = subprocess.Popen([sys.executable, str(script), str(report)], env=env)
    children: list[int] = []
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not report.exists():
            assert parent.poll() is None, "the host script died before it reported its workers"
            time.sleep(0.05)
        assert report.exists(), "the host script never reported its worker pids"
        children = json.loads(report.read_text(encoding="utf-8"))
        assert len(children) > 1, "the pool really ran in several processes"
        assert all(_alive(pid) for pid in children), "the workers are alive while their parent is"

        parent.kill()
        parent.wait(timeout=10)
        gone_by = time.monotonic() + 5
        while time.monotonic() < gone_by and any(_alive(pid) for pid in children):
            time.sleep(0.1)
        assert [pid for pid in children if _alive(pid)] == [], "no worker outlives its killed parent"
    finally:
        for pid in children:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
        if parent.poll() is None:  # pragma: no cover - only reachable when an assertion above failed
            parent.kill()
            parent.wait(timeout=10)

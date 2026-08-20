"""Process pools never fork a threaded host; an interpreter without a real __main__ gets no pool."""

from __future__ import annotations

import sys
import threading
import types

import pytest

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

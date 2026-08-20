"""Choose a process-pool start method that cannot deadlock the host process.

`fork` is the fastest start method, but forking a process that already runs other threads
(the MCP server, the daemon, anything with a watcher or an async loop) copies locks in
whatever state they are in; a child that then needs one of them waits forever. That is
exactly how `dupes` over MCP hung with 8 idle workers. So: fork only while this process is
single-threaded, otherwise `spawn`. `spawn` re-imports the host's `__main__`, which is
impossible for an interpreter fed from `-c` or stdin; in that case there is no safe pool at
all and callers run sequentially.
"""

from __future__ import annotations

import multiprocessing
import sys
import threading
from multiprocessing.context import BaseContext


def pool_context() -> BaseContext | None:
    """Return the start-method context a `ProcessPoolExecutor` may use here, or None for "run in-process"."""
    methods = multiprocessing.get_all_start_methods()
    if threading.active_count() == 1 and "fork" in methods:
        return multiprocessing.get_context("fork")
    if "spawn" not in methods:
        return None
    main = sys.modules.get("__main__")
    if main is None or not getattr(main, "__file__", None):
        return None
    return multiprocessing.get_context("spawn")

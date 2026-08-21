"""Choose a process-pool start method that cannot deadlock the host process, and outlive nothing.

`fork` is the fastest start method, but forking a process that already runs other threads
(the MCP server, the daemon, anything with a watcher or an async loop) copies locks in
whatever state they are in; a child that then needs one of them waits forever. That is
exactly how `dupes` over MCP hung with 8 idle workers. So: fork only while this process is
single-threaded, otherwise `spawn`. `spawn` re-imports the host's `__main__`, which is
impossible for an interpreter fed from `-c` or stdin; in that case there is no safe pool at
all and callers run sequentially.

Nothing reaps pool children when their parent dies without shutting the pool down (a killed
MCP server, a crashed daemon): they are re-parented to init and idle forever, holding open
whatever the parent had open. Every pool built here therefore installs a guard in each child
that makes it exit as soon as its parent is gone.
"""

from __future__ import annotations

import contextlib
import ctypes
import multiprocessing
import os
import signal
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.context import BaseContext

#: How often a child re-checks that its parent is still there.
PARENT_CHECK_SECONDS = 2.0

#: Linux `prctl` option number for "signal me when my parent dies".
PR_SET_PDEATHSIG = 1


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


def _set_parent_death_signal() -> None:
    """Ask the kernel to SIGTERM this process when its parent dies; a no-op off Linux."""
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except (OSError, AttributeError, ValueError):
        return


def _watch_parent(parent_pid: int) -> None:
    """Exit this process as soon as it is re-parented away from the pid that started it."""
    while True:
        time.sleep(PARENT_CHECK_SECONDS)
        if os.getppid() != parent_pid:
            os._exit(0)


def child_guard(parent_pid: int, use_pdeathsig: bool) -> None:
    """Pool initializer: make this worker exit when the process that created the pool is gone.

    AIDEV-NOTE: PR_SET_PDEATHSIG is per-thread of the parent -- it fires when the *thread*
    that created the child exits, not when the parent process does. A pool created from a
    worker thread (asyncio.to_thread, a daemon handler) would see its children killed the
    moment that thread returned, so the caller only asks for it from the main thread; the
    getppid watchdog is the fallback that covers every other case, at the cost of noticing
    a dead parent up to PARENT_CHECK_SECONDS late.
    """
    if use_pdeathsig:
        _set_parent_death_signal()
        if os.getppid() != parent_pid:
            # The parent died between fork and prctl; the signal will never come.
            os._exit(0)
    watchdog = threading.Thread(target=_watch_parent, args=(parent_pid,), daemon=True, name="zemble-parent-watch")
    watchdog.start()


def process_pool(workers: int, context: BaseContext) -> ProcessPoolExecutor:
    """Return a process pool whose workers cannot outlive this process.

    :param workers: Maximum worker processes.
    :param context: Start-method context, as returned by `pool_context`.
    """
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=child_guard,
        initargs=(os.getpid(), threading.current_thread() is threading.main_thread()),
    )


@contextlib.contextmanager
def pooled(workers: int, context: BaseContext) -> Iterator[ProcessPoolExecutor]:
    """Yield a guarded process pool, abandoning its workers instead of blocking when the body raises."""
    pool = process_pool(workers, context)
    try:
        yield pool
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)


__all__ = ["PARENT_CHECK_SECONDS", "child_guard", "pool_context", "pooled", "process_pool"]

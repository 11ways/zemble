"""Construct one graph without importing or retaining the daemon's serving index."""

import json
import resource
import signal
import sys
import time
from enum import StrEnum
from typing import Any

#: The build statistics a worker reports, which the daemon logs and shows in its status.
REPORTED_STATS = (
    "files_scanned",
    "extracted_files",
    "removed_files",
    "reresolved_files",
    "restored_files",
    "owed_names",
    "published_seconds",
    "compacted",
)
#: The key naming what a line of the worker's stdout reports.
REPORT_KEY = "report"


class WorkerReport(StrEnum):
    """What one line of a graph worker's stdout reports; the one home of these names."""

    #: A lane overlay's first pass is published: the lane answers with its own files, its dependents follow.
    PUBLISHED = "published"
    #: The build is done; the last line a worker prints.
    FINISHED = "finished"


def report_line(kind: WorkerReport, **fields: Any) -> bytes:
    """Encode one report as the newline-terminated JSON line the daemon reads as it arrives."""
    return (json.dumps({REPORT_KEY: kind.value, **fields}) + "\n").encode()


def main() -> None:
    """Set the worker's private-memory limit before importing graph machinery."""
    root, budget = sys.argv[1:]
    limit = int(budget) * 1024 * 1024
    _soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
    resource.setrlimit(resource.RLIMIT_DATA, (limit, hard))

    def stop(_signal: int, _frame: object) -> None:
        raise KeyboardInterrupt("graph worker shutdown")

    signal.signal(signal.SIGTERM, stop)
    from zemble.daemon import client
    from zemble.graph import store

    def write(line: bytes) -> None:
        sys.stdout.buffer.write(line)
        sys.stdout.buffer.flush()

    def published(stats: Any) -> None:
        write(report_line(WorkerReport.PUBLISHED, stats=_reported(stats)))

    client.disable_for_this_process("--no-daemon")
    store.DEFAULT_WORKERS = 1
    started = time.monotonic()
    changes = json.load(sys.stdin)["changed_paths"]
    if changes is None:
        # What `ensure_graph` does for a root the daemon already resolved, with the statistics kept.
        if store.graph_exists(root):
            stats = store.refresh_graph(root, on_published=published)
        else:
            stats = store.build_graph(root, on_published=published)
    else:
        from pathlib import Path

        stats = store.build_graph(
            root, workers=1, changed_paths=[Path(path) for path in changes], on_published=published
        )
    write(
        report_line(
            WorkerReport.FINISHED,
            build_seconds=time.monotonic() - started,
            peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            # None when another process held the writer lock: that writer is refreshing it.
            stats=None if stats is None else _reported(stats),
        )
    )


def _reported(stats: Any) -> dict[str, Any]:
    """The part of a build's statistics a worker reports."""
    view = stats.to_dict()
    return {key: view[key] for key in REPORTED_STATS}


if __name__ == "__main__":
    main()

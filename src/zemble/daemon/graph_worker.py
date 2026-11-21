"""Construct one graph without importing or retaining the daemon's serving index."""

import json
import resource
import signal
import sys
import time

#: The build statistics a worker reports, which the daemon logs and shows in its status.
REPORTED_STATS = ("files_scanned", "extracted_files", "removed_files", "reresolved_files", "compacted")


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

    client.disable_for_this_process("--no-daemon")
    store.DEFAULT_WORKERS = 1
    started = time.monotonic()
    changes = json.load(sys.stdin)["changed_paths"]
    if changes is None:
        # What `ensure_graph` does for a root the daemon already resolved, with the statistics kept.
        stats = store.refresh_graph(root) if store.graph_exists(root) else store.build_graph(root)
    else:
        from pathlib import Path

        stats = store.build_graph(root, workers=1, changed_paths=[Path(path) for path in changes])
    sys.stdout.write(
        json.dumps(
            {
                "build_seconds": time.monotonic() - started,
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                # None when another process held the writer lock: that writer is refreshing it.
                "stats": None if stats is None else {key: stats.to_dict()[key] for key in REPORTED_STATS},
            }
        )
    )


if __name__ == "__main__":
    main()

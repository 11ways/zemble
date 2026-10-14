"""Watch one root in a process of its own, writing each relevant change set to stdout as a JSON line.

Registering the watches of a large tree holds the GIL for seconds (5.5 s for the 108k directories of
the zenit workspace), which froze the daemon's event loop and every thread in it. Here it freezes nothing
else. The first line is the ready marker; the worker exits when its stdin closes.
"""

import json
import sys
import threading
from pathlib import Path


def main() -> None:
    """Read the spec from argv, watch until stdin closes, and report every relevant change set."""
    from watchfiles import watch

    from zemble.daemon.watch import DEBOUNCE_MS, READY_LINE, STEP_MS, WatchSpec

    spec = WatchSpec.from_json(sys.argv[1])
    rules = spec.rules()
    stop = threading.Event()

    def wait_for_parent() -> None:
        sys.stdin.read()
        stop.set()

    threading.Thread(target=wait_for_parent, daemon=True).start()
    ready = False
    # No watch_filter: watchfiles yields nothing while every change is filtered out, so a tree whose
    # ignored files churn (build output, the index's own cache) would never report ready. The rules
    # are applied here instead, after the yield that proves the watches exist.
    for changes in watch(
        spec.root,
        watch_filter=None,
        debounce=DEBOUNCE_MS,
        step=STEP_MS,
        stop_event=stop,
        rust_timeout=STEP_MS * 4,
        yield_on_timeout=True,
        raise_interrupt=False,
        recursive=True,
    ):
        if not ready:
            # watchfiles yields only once every watch is registered: edits from here on are seen.
            sys.stdout.write(READY_LINE + "\n")
            ready = True
        relevant = sorted({path for _change, path in changes if rules.matches(Path(path))})
        if relevant:
            sys.stdout.write(json.dumps(relevant) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

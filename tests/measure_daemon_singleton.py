"""Measure one index's ignored-event backlog while graph work is held, using only a private daemon."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import faulthandler
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
from pathlib import Path

from tests.measure_daemon_churn import client_environment, offline, stop_process, wait_for_owned_daemon
from tests.measure_daemon_memory import _environment, rss


def child():
    """Run the real CLI with a stale inherited env flag, offline vectors, and controllable graph delay."""
    import watchfiles.main as watch_main

    faulthandler.dump_traceback_later(60, repeat=True, file=sys.stderr)

    from zemble.daemon import server
    from zemble.daemon.cli import main
    from zemble.index_cache import IndexCache

    if not Path(server.__file__).resolve().is_relative_to(Path(os.environ["MEMORY_CODE_ROOT"])):
        raise RuntimeError("The measurement imported the wrong checkout")
    gate = asyncio.Event()
    entered = asyncio.Event()
    counts = {"raw_batches": 0, "raw_events": 0, "accepted_events": 0}
    prep = watch_main._prep_changes

    def counted(raw, watch_filter):
        counts["raw_batches"] += 1
        counts["raw_events"] += len(raw)
        result = prep(raw, watch_filter)
        counts["accepted_events"] += len(result)
        return result

    async def load(self):
        self._embedder = offline()
        self._model_ready.set()

    async def graph(self, root, changed_paths=None):
        entered.set()
        await gate.wait()
        return 0

    status = server.COMMANDS["status"]

    async def profile(daemon, args):
        action = args.get("action")
        if action == "trace":
            tracemalloc.start(1)
        elif action == "release":
            tracemalloc.stop()
            gate.set()
        result = await status(daemon, {})
        current, peak = tracemalloc.get_traced_memory()
        result["profile"] = {
            **counts,
            "graph_entered": entered.is_set(),
            "graph_held": entered.is_set() and not gate.is_set(),
            "python_current_mib": round(current / 1024**2, 1),
            "python_peak_mib": round(peak / 1024**2, 1),
            "tracer_mib": round(tracemalloc.get_tracemalloc_memory() / 1024**2, 1),
        }
        return result

    watch_main._prep_changes = counted
    IndexCache.load_embedder_once = load
    server.Daemon._refresh_graph = graph
    server.COMMANDS["memory_profile"] = profile
    raise SystemExit(main(["run", "--idle-minutes", "0"]))


def generate(root):
    """Generate approximately 26k chunks in 1756 files plus one Java change probe."""
    root.mkdir()
    for i in range(1756):
        source = f"class Service{i}:\n" + "".join(
            f"    def operation_{j}(self, value):\n        return value + {i + j}\n\n" for j in range(160)
        )
        (root / f"service_{i}.py").write_text(source)
    (root / "Probe.java").write_text("class Probe { int pulse() { return 0; } }\n")


def noise(root, events, call):
    """Create and delete unique ignored build artifacts, retaining no path list in the load generator."""
    build = root / "build" / "noise"
    build.mkdir(parents=True)
    time.sleep(0.5)
    points = []
    for i in range(events):
        path = build / (f"artifact-{i:07d}-" + "x" * 210 + ".class")
        descriptor = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(descriptor)
        path.unlink()
        if (i + 1) % 50000 == 0:
            points.append({"artifacts": i + 1, "status": call("memory_profile")})
            (root.parent / "progress.json").write_text(json.dumps(points))
    return points


def exercise(root, call, events):
    """Hold a completed index rebuild's graph phase while the native watcher receives ignored churn."""
    stats = call("stats", {"path": str(root), "content": ["code"]})
    startup = call("memory_profile")
    time.sleep(0.5)
    (root / "Probe.java").write_text("class Probe { int pulse() { return 1; } }\n")
    deadline = time.monotonic() + 120
    while not call("memory_profile")["profile"]["graph_entered"]:
        if time.monotonic() > deadline:
            raise RuntimeError("Rebuild did not reach the held graph phase")
        time.sleep(0.1)
    held = call("memory_profile", {"action": "trace"})
    points = noise(root, events, call)
    backlog = call("memory_profile")
    call("memory_profile", {"action": "release"})
    time.sleep(10)
    final = call("memory_profile")
    return {
        "stats": stats,
        "startup": startup,
        "held_before_noise": held,
        "checkpoints": points,
        "held_after_noise": backlog,
        "final": final,
    }


def measure(revision, events):
    """Check host headroom and run the singleton reproduction with a seven-GiB emergency stop."""
    subprocess.run(["free", "-g"], check=True)
    available = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available < 10 * 1024**2:
        raise RuntimeError("Need at least 10 GiB available")
    with tempfile.TemporaryDirectory(prefix="zemble-singleton-", dir="/tmp/opencode") as directory:
        base = Path(directory)
        root = base / "workspace"
        generate(root)
        env = _environment(base, root, revision)
        config = base / "env"
        config.write_text("ZEMBLE_DAEMON_MAX_INDEXES=1\nZEMBLE_DAEMON_MAX_RSS_MB=2048\n")
        env.update(ZEMBLE_ENV_FILE=str(config), _ZEMBLE_USER_ENV_LOADED="1", ZEMBLE_EMBED_BUDGET_TOKENS="100000000")
        client_environment(env)
        from zemble.daemon import client

        stop = threading.Event()
        samples = []
        with (base / "daemon.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "tests.measure_daemon_singleton", "--child"], env=env, stdout=log, stderr=log
            )

            def sample():
                while not stop.wait(0.02):
                    with contextlib.suppress(FileNotFoundError, ProcessLookupError):
                        value = rss(process.pid)
                        samples.append(value)
                        if value > 7 * 1024:
                            process.terminate()
                            return

            monitor = threading.Thread(target=sample, daemon=True)
            monitor.start()

            def call(command, args=None):
                return client.call(command, args, auto_start=False, timeout=300)

            try:
                wait_for_owned_daemon(process, call, client, base / "daemon.log")
                result = exercise(root, call, events)
                return {
                    "revision": revision or "working-tree",
                    "artifacts": events,
                    "peak_rss_mib": round(max(samples), 1),
                    **result,
                    "log": (base / "daemon.log").read_text(),
                }
            finally:
                stop_process(process, call)
                stop.set()
                monitor.join()


def main():
    """Select the singleton reproduction or its owned child."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision")
    parser.add_argument("--events", type=int, default=500000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        child()
    else:
        text = json.dumps(measure(args.revision, args.events), indent=2) + "\n"
        if args.output:
            args.output.write_text(text)
        else:
            sys.stdout.write(text)


if __name__ == "__main__":
    main()

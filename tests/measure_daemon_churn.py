"""Replay a generated multi-root load against an isolated daemon checkout."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tests.measure_daemon_memory import _environment, rss


def offline():
    """Create deterministic 1024-wide vectors without retaining call history."""
    from tests.conftest import FakeEmbedder

    class Offline(FakeEmbedder):
        def embed_documents(self, texts):
            return self._vectors(texts)

        def embed_queries(self, texts):
            return self._vectors(texts)

    return Offline(dimensions=1024)


def child():
    """Use the actual server with an offline embedder."""
    from zemble.daemon import server
    from zemble.index_cache import IndexCache

    if not Path(server.__file__).resolve().is_relative_to(Path(os.environ["MEMORY_CODE_ROOT"])):
        raise RuntimeError("The measurement imported the wrong checkout")

    async def load(self):
        self._embedder = offline()
        self._model_ready.set()

    IndexCache.load_embedder_once = load
    status = server.COMMANDS["status"]

    async def measured_status(daemon, args):
        result = await status(daemon, args)
        for row, (key, index) in zip(result["indexes"], daemon.cache.loaded()):
            row["vectors_mib"] = round(index._semantic_index.vectors.nbytes / 1024**2, 1)
            row["vector_storage"] = type(index._semantic_index.vectors).__name__
            row["chunk_storage"] = type(index.chunks).__name__
            row["bm25_delta_documents"] = index._bm25_index.delta_documents
            row["cached_views"] = len(index._views)
        return result

    server.COMMANDS["status"] = measured_status
    asyncio.run(server.run(watch=True, idle_minutes=0, max_indexes=4))


def memory_breakdown(pid):
    """Separate anonymous heap from file-backed resident pages."""
    values = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines()[1:])
    return {name: round(int(values[name].split()[0]) / 1024, 1) for name in ("Rss", "Anonymous", "Pss_File")}


def exercise(roots, selections, rounds, call):
    """Overlap six socket queries with two seconds of writes, then allow three seconds of quiet."""
    with ThreadPoolExecutor(max_workers=6) as pool:
        for iteration in range(rounds):
            end = time.monotonic() + 2
            queries = []
            while time.monotonic() < end:
                for root in roots:
                    probe = root / f"service_{iteration % 20}.py"
                    old, new = ("+", "-") if iteration % 2 else ("-", "+")
                    probe.write_text(probe.read_text().replace(f"return value {old}", f"return value {new}", 1))
                if len(queries) < 6:
                    root, content = selections[len(queries) % 3]
                    queries.append(
                        pool.submit(
                            call,
                            "search",
                            {"path": str(root), "content": content, "query": "operation value", "top_k": 3},
                        )
                    )
                time.sleep(0.05)
            for query in queries:
                query.result()
            time.sleep(3)


def generate_roots(base, files):
    """Generate the same source bytes independently of the revision being measured."""
    roots = [base / "workspace", base / "second"]
    for root, count in zip(roots, (files, files // 2)):
        root.mkdir()
        for i in range(count):
            source = f"class Service{i}:\n" + "".join(
                f"    def operation_{j}(self, value):\n        return value + {i + j}\n\n" for j in range(80)
            )
            (root / f"service_{i}.py").write_text(source)
            if i % 4 == 0:
                (root / f"guide_{i}.md").write_text(f"# Service {i}\n\n" + "Service operation guide.\n" * 40)
    return roots


def stop_process(process, call):
    """Stop only the daemon owned by this measurement."""
    if process.poll() is None:
        with contextlib.suppress(Exception):
            call("shutdown")
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def settle(call):
    """Allow pending watcher work to finish, or report its state after a bounded wait."""
    time.sleep(6)
    final = call("status")
    deadline = time.monotonic() + 60
    while final["pending_reindex"] or any(row["rebuilding"] for row in final["indexes"]):
        if time.monotonic() >= deadline:
            break
        time.sleep(0.5)
        final = call("status")
    return final


def wait_for_owned_daemon(process, call, client, log_path):
    """Verify the private socket belongs to the process this script started before indexing anything."""
    deadline = time.monotonic() + 60
    while not client.is_running():
        if process.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError(log_path.read_text())
        time.sleep(0.1)
    if call("ping")["pid"] != process.pid:
        raise RuntimeError("Refusing to measure a daemon this script did not start")


def client_environment(env):
    """Remove inherited socket overrides instead of allowing a client to contact the real daemon."""
    for key in tuple(os.environ):
        if key.startswith("ZEMBLE_"):
            os.environ.pop(key)
    os.environ.update({key: value for key, value in env.items() if key.startswith("ZEMBLE_")})


def measure(revision, files, rounds, budget):
    """Measure initial loads, continuous churn, and quiet-period rebuilds on private roots."""
    subprocess.run(["free", "-g"], check=True)
    available = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available < 10 * 1024**2:
        raise RuntimeError("Need at least 10 GiB available")
    with tempfile.TemporaryDirectory(prefix="zemble-churn-", dir="/tmp/opencode") as temporary:
        base = Path(temporary)
        roots = generate_roots(base, files)
        env = _environment(base, roots[0], revision)
        env["ZEMBLE_DAEMON_MAX_RSS_MB"] = str(budget)
        env["ZEMBLE_EMBED_BUDGET_TOKENS"] = "100000000"
        client_environment(env)
        from zemble.daemon import client

        samples = []
        phase = "startup"
        stop = threading.Event()
        with (base / "daemon.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "tests.measure_daemon_churn", "--child"], env=env, stdout=log, stderr=log
            )

            def sample():
                while not stop.wait(0.02):
                    with contextlib.suppress(FileNotFoundError, ProcessLookupError):
                        parent = rss(process.pid)
                        children = Path(f"/proc/{process.pid}/task/{process.pid}/children").read_text().split()
                        total = parent + sum(rss(int(pid)) for pid in children)
                        samples.append((phase, parent, total))
                        if total > 7 * 1024:
                            process.terminate()
                            return

            monitor = threading.Thread(target=sample, daemon=True)
            monitor.start()

            def call(command, args=None):
                return client.call(command, args, auto_start=False, timeout=300)

            try:
                wait_for_owned_daemon(process, call, client, base / "daemon.log")
                loads = []
                selections = [(roots[0], ["code"]), (roots[0], ["code", "docs"]), (roots[1], ["code", "docs"])]
                for number, (root, content) in enumerate(selections):
                    phase = f"load_{number + 1}"
                    stats = call("stats", {"path": str(root), "content": content})
                    loads.append(
                        {
                            "content": content,
                            "chunks": stats["total_chunks"],
                            "rss_mib": round(rss(process.pid), 1),
                            "memory_breakdown_mib": memory_breakdown(process.pid),
                            "status": call("status"),
                        }
                    )
                phase = "churn"
                exercise(roots, selections, rounds, call)
                phase = "settle"
                final = settle(call)
                peaks = {
                    name: {
                        "daemon_mib": round(max(p for s, p, t in samples if s == name), 1),
                        "daemon_and_children_mib": round(max(t for s, p, t in samples if s == name), 1),
                    }
                    for name in sorted({s for s, p, t in samples})
                }
                return {
                    "revision": revision or "working-tree",
                    "files": files,
                    "rounds": rounds,
                    "dimensions": 1024,
                    "budget_mib": budget,
                    "loads": loads,
                    "peaks": peaks,
                    "final": final,
                    "final_memory_breakdown_mib": memory_breakdown(process.pid),
                    "log": (base / "daemon.log").read_text(),
                }
            finally:
                stop_process(process, call)
                stop.set()
                monitor.join()


def main():
    """Select the measurement process or its isolated child."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision")
    parser.add_argument("--files", type=int, default=16000)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        child()
    else:
        result = measure(args.revision, args.files, args.rounds, args.budget)
        text = json.dumps(result, indent=2) + "\n"
        if args.output:
            args.output.write_text(text)
        else:
            sys.stdout.write(text)


if __name__ == "__main__":
    main()

"""Measure an isolated offline daemon under rebuild/search/graph contention.

Run with .venv/bin/python -m tests.measure_daemon_memory --repo /path/to/repo.
The source is copied into a size-limited temporary workspace; no real cache is opened.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def rss(pid: int) -> float:
    """Read current resident memory in MiB without contacting the daemon."""
    pages = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 1024**2


def child() -> None:
    """Run the actual socket server using the deterministic test embedder, without call history."""
    from tests.conftest import FakeEmbedder
    from zemble.daemon import server
    from zemble.graph.store import build_graph
    from zemble.index_cache import IndexCache

    if not Path(server.__file__).resolve().is_relative_to(Path(os.environ["MEMORY_CODE_ROOT"])):
        raise RuntimeError("The measurement imported the wrong checkout")

    class OfflineEmbedder(FakeEmbedder):
        def embed_documents(self, texts):
            return self._vectors(texts)

        def embed_queries(self, texts):
            return self._vectors(texts)

    async def load(self):
        self._embedder = OfflineEmbedder(dimensions=1024)
        self._model_ready.set()

    IndexCache.load_embedder_once = load
    # Serial graph extraction avoids multiplying worker interpreters on a low-RAM host.
    build_graph(os.environ["MEMORY_WORKSPACE"], workers=1)
    asyncio.run(server.run(watch=False, idle_minutes=0, max_indexes=1))


def snapshot(source: Path, target: Path) -> dict:
    """Copy a deterministic medium source subset, bounded to 3 MiB and 120 files."""
    extensions = {".zig", ".py", ".java", ".js", ".ts", ".tsx", ".rs"}
    ignored = {".git", ".claude", "vendor", "node_modules", "dist", "build", ".venv", "zig-out", "zig-pkg"}
    count = size = 0
    for directory, directories, files in os.walk(source):
        directories[:] = sorted(name for name in directories if name not in ignored and not name.startswith("."))
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix not in extensions or path.is_symlink():
                continue
            length = path.stat().st_size
            if length > 256 * 1024 or size + length > 3 * 1024**2 or count >= 120:
                continue
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            size += length
            count += 1
    if not count:
        raise ValueError("No supported source files in the requested repo")
    # A source file changed each round and a high-fanout graph query with deterministic identity.
    probe = target / "probe.py"
    probe.write_text(
        "def memory_probe():\n    return 0\n"
        + "\n".join(f"def memory_caller_{i}():\n    return memory_probe()\n" for i in range(2000))
    )
    digest = hashlib.sha256()
    for path in sorted(target.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(target).as_posix().encode() + b"\0" + path.read_bytes())
    return {"source_files": count, "source_bytes": size, "probe_callers": 2000, "snapshot_sha256": digest.hexdigest()}


def _exercise(workspace: Path, pid: int, call) -> list[float]:
    """Overlap each edit/rebuild with fifty socket queries and await every response."""
    args = {"path": str(workspace)}
    rounds = []
    with ThreadPoolExecutor(max_workers=52) as pool:
        for iteration in range(10):
            probe = workspace / "probe.py"
            probe.write_text(probe.read_text().replace(f"return {iteration}\n", f"return {iteration + 1}\n", 1))
            futures = [pool.submit(call, "refresh", args)]
            for query in range(50):
                if query % 2:
                    request = {**args, "command": "neighbors", "symbol": "memory_probe", "hops": 3, "limit": 10}
                    futures.append(pool.submit(call, "graph", request))
                else:
                    futures.append(
                        pool.submit(
                            call, "search", {**args, "query": "memory_probe", "exclude": ["absent/**"], "top_k": 3}
                        )
                    )
            for future in futures:
                result = future.result()
                if isinstance(result, dict) and "error" in result:
                    raise RuntimeError(result["error"])
            rounds.append(round(rss(pid), 1))
    return rounds


def _environment(base: Path, workspace: Path, revision: str | None) -> dict[str, str]:
    """Select an isolated cache/socket and optionally export baseline code without switching branches."""
    root = Path(__file__).resolve().parents[1]
    code_root = root / "src"
    if revision is not None:
        archive = subprocess.check_output(["git", "archive", revision, "src/zemble"], cwd=root)
        with tarfile.open(fileobj=io.BytesIO(archive)) as files:
            files.extractall(base / "baseline", filter="data")
        code_root = base / "baseline" / "src"
    env = {key: value for key, value in os.environ.items() if not key.startswith("ZEMBLE_")}
    env.update(
        ZEMBLE_CACHE_LOCATION=str(base / "cache"),
        ZEMBLE_DAEMON_DIR=str(base / "run"),
        ZEMBLE_ENV_FILE=str(base / "absent"),
        ZEMBLE_DAEMON="1",
        ZEMBLE_RERANKER="none",
        MEMORY_WORKSPACE=str(workspace),
        MEMORY_CODE_ROOT=str(code_root),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        PYTHONPATH=os.pathsep.join((str(code_root), str(root))),
    )
    return env


def _prepare(source: Path, workspace: Path, prepared: Path | None, info: dict | None) -> dict:
    """Snapshot a source tree or clone the frozen input of a paired measurement."""
    if prepared is None:
        return snapshot(source, workspace)
    assert info is not None, "a prepared snapshot needs its metadata"
    shutil.copytree(prepared, workspace, dirs_exist_ok=True)
    return info


def measure(
    source: Path, label: str, *, revision: str | None = None, prepared: Path | None = None, info: dict | None = None
) -> dict:
    """Run ten edit/rebuild rounds, each overlapping fifty socket queries."""
    available = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available < 6 * 1024**2:
        raise RuntimeError("Need at least 6 GiB available before measuring")
    with tempfile.TemporaryDirectory(prefix="zemble-memory-", dir="/tmp/opencode") as temporary:
        base = Path(temporary)
        workspace = base / "workspace"
        workspace.mkdir()
        copied = _prepare(source, workspace, prepared, info)
        env = _environment(base, workspace, revision)
        # The parent client is pointed at the same private socket; auto-start is always disabled.
        os.environ.update({key: value for key, value in env.items() if key.startswith("ZEMBLE_")})
        os.environ.pop("ZEMBLE_DAEMON_SOCKET", None)
        from zemble.daemon import client

        stop = threading.Event()
        samples: list[float] = []
        with (base / "daemon.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "tests.measure_daemon_memory", "--child"], env=env, stdout=log, stderr=log
            )

            def sample():
                while not stop.wait(0.02):
                    with contextlib.suppress(FileNotFoundError, ProcessLookupError):
                        value = rss(process.pid)
                        samples.append(value)
                        if value > 2500:
                            process.terminate()
                            return

            monitor = threading.Thread(target=sample, daemon=True)
            monitor.start()

            def call(command, args=None):
                return client.call(command, args, auto_start=False, timeout=180)

            try:
                deadline = time.monotonic() + 180
                while not client.is_running():
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError((base / "daemon.log").read_text()[-4000:])
                    time.sleep(0.1)
                args = {"path": str(workspace)}
                stats = call("stats", args)
                warm = rss(process.pid)
                rounds = _exercise(workspace, process.pid, call)
                time.sleep(1)
                final = rss(process.pid)
                return {
                    "label": label,
                    "revision": revision or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "working_tree": revision is None,
                    "repo": str(source),
                    **copied,
                    "dimensions": 1024,
                    "chunks": stats["total_chunks"],
                    "rebuilds": 10,
                    "concurrent_queries_per_round": 50,
                    "queries": 500,
                    "warm_rss_mib": round(warm, 1),
                    "peak_rss_mib": round(max(samples + [final]), 1),
                    "final_rss_mib": round(final, 1),
                    "round_rss_mib": rounds,
                }
            finally:
                if process.poll() is None:
                    with contextlib.suppress(Exception):
                        call("shutdown")
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                stop.set()
                monitor.join()


def compare(source: Path, revision: str) -> dict:
    """Run baseline and working-tree daemons sequentially against the exact same frozen source bytes."""
    with tempfile.TemporaryDirectory(prefix="zemble-memory-input-", dir="/tmp/opencode") as temporary:
        prepared = Path(temporary) / "input"
        prepared.mkdir()
        info = snapshot(source, prepared)
        baseline = measure(source, "before", revision=revision, prepared=prepared, info=info)
        after = measure(source, "after", prepared=prepared, info=info)
        assert baseline["chunks"] == after["chunks"], "paired runs must use identical input"
        return {"before": baseline, "after": after}


def main() -> None:
    """Run the isolated benchmark or its explicitly selected child process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--label", default="measurement")
    parser.add_argument("--compare-revision", help="Replay a git revision and the working tree on one frozen snapshot.")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        child()
    elif args.repo:
        source = args.repo.resolve()
        result = compare(source, args.compare_revision) if args.compare_revision else measure(source, args.label)
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
    else:
        parser.error("--repo is required")


if __name__ == "__main__":
    main()

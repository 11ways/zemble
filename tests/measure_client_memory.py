"""Replay MCP fallback failures and daemon restarts using private caches, sockets and offline vectors."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import replace
from pathlib import Path

from tests.measure_daemon_churn import client_environment, generate_roots, offline, wait_for_owned_daemon
from tests.measure_daemon_memory import _environment, rss


def offline_models():
    """Use deterministic embeddings without recording document or query call history."""
    from zemble.index_cache import IndexCache

    async def load(self):
        if not self._model_ready.is_set():
            self._embedder = offline()
            self._model_ready.set()

    IndexCache.load_embedder_once = load


def daemon_child():
    """Run an owned daemon whose fault replies and revision envelope are controlled by the load."""
    from zemble.daemon import server

    runtime = importlib.import_module("zemble.runtime.identity")

    offline_models()
    runtime._IDENTITY = replace(runtime.identity(), source_revision=os.environ["CLIENT_MEASURE_DAEMON_REV"])
    original = server.Daemon.handle
    fault_file = Path(os.environ["CLIENT_MEASURE_FAULT"])

    async def handle(self, request):
        fault = json.loads(fault_file.read_text()).get("kind")
        if fault and request.get("cmd") not in {"ping", "status", "shutdown"}:
            return {"id": request.get("id"), "ok": False, "kind": fault, "error": f"injected {fault}"}
        return await original(self, request)

    server.Daemon.handle = handle
    asyncio.run(server.run(watch=False, idle_minutes=0))


def mcp_child():
    """Use the real stdio server while exposing only its owned cache and allocation counters."""
    from zemble import mcp
    from zemble.daemon import client
    from zemble.daemon.protocol import DaemonUnavailable
    from zemble.index_cache import IndexCache

    if not Path(mcp.__file__).resolve().is_relative_to(Path(os.environ["MEMORY_CODE_ROOT"])):
        raise RuntimeError("The measurement imported the wrong client checkout")
    offline_models()
    create = mcp.create_server
    build = IndexCache._build_index
    remote = client.call
    local_builds = 0
    fault_file = Path(os.environ["CLIENT_MEASURE_FAULT"])

    def tracked_build(self, *args, **kwargs):
        nonlocal local_builds
        local_builds += 1
        return build(self, *args, **kwargs)

    def call(cmd, args=None, **kwargs):
        if json.loads(fault_file.read_text()).get("kind") == "unavailable":
            raise DaemonUnavailable("injected startup/connection failure")
        return remote(cmd, args, **kwargs)

    def configured(cache, **kwargs):
        instance = create(cache, **kwargs)

        @instance.tool(structured_output=False)
        def measurement_state() -> dict:
            """Report only the measurement process's memory and cache ownership."""
            revision = client.daemon_revision() if hasattr(client, "daemon_revision") else None
            return {
                "pid": os.getpid(),
                "rss_mib": round(rss(os.getpid()), 1),
                "local_builds": local_builds,
                "local_indexes": len(cache._tasks),
                "local_model_loaded": cache.embedder is not None,
                "daemon_revision": revision,
            }

        return instance

    IndexCache._build_index = tracked_build
    client.call = call
    mcp.create_server = configured
    asyncio.run(mcp.serve())


def check_headroom():
    """Require ten GiB available before starting the load."""
    subprocess.run(["free", "-g"], check=True)
    available = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available < 10 * 1024**2:
        raise RuntimeError("Need at least 10 GiB available")


def sample(processes, stop, samples, marker):
    """Sample owned client/daemon RSS and stop only those processes if the seven-GiB guard fires."""
    while not stop.wait(0.02):
        with contextlib.suppress(FileNotFoundError, ProcessLookupError):
            values = {name: rss(pid) for name, pid in list(processes.items())}
            samples.append(values)
            if sum(values.values()) > 7 * 1024:
                for pid in list(processes.values()):
                    environment = Path(f"/proc/{pid}/environ").read_bytes()
                    if f"CLIENT_MEASURE_FAULT={marker}".encode() in environment.split(b"\0"):
                        os.kill(pid, 15)
                return


def start_daemon(base, roots, label):
    """Start the daemon on a private socket without relying on client autostart."""
    env = _environment(base, roots[0], None)
    env.update(
        CLIENT_MEASURE_FAULT=str(base / "fault.json"),
        CLIENT_MEASURE_DAEMON_REV=label,
        ZEMBLE_EMBED_BUDGET_TOKENS="100000000",
    )
    client_environment(env)
    from zemble.daemon import client

    log = (base / f"daemon-{label}.log").open("w")
    process = subprocess.Popen(
        [sys.executable, "-m", "tests.measure_client_memory", "--daemon-child"], env=env, stdout=log, stderr=log
    )
    log.close()
    wait_for_owned_daemon(
        process, lambda cmd: client.call(cmd, auto_start=False, timeout=30), client, base / f"daemon-{label}.log"
    )
    return process


def stop_daemon(process):
    """Stop only the daemon created by this replay."""
    from zemble.daemon import client

    if process.poll() is None:
        with contextlib.suppress(Exception):
            client.call("shutdown", auto_start=False, timeout=10)
        try:
            process.wait(15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def text(result):
    """Extract the single unstructured MCP result lane."""
    return "".join(item.text for item in result.content if getattr(item, "type", None) == "text")


async def exercise(base, roots, revision, processes):
    """Keep one real MCP stdio session alive across error cases and a daemon restart."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = _environment(base / "client", roots[0], revision)
    env.update(
        ZEMBLE_CACHE_LOCATION=str(base / "cache"),
        ZEMBLE_DAEMON_DIR=str(base / "run"),
        CLIENT_MEASURE_FAULT=str(base / "fault.json"),
        ZEMBLE_EMBED_BUDGET_TOKENS="100000000",
    )
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "tests.measure_client_memory", "--mcp-child"], env=env
    )
    with (base / "mcp.log").open("w") as log:
        async with stdio_client(params, errlog=log) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()

                async def state():
                    return json.loads(text(await session.call_tool("measurement_state", {})))

                initial = await state()
                processes["client"] = initial["pid"]
                rows = []
                cases = [
                    (None, roots[0], "code"),
                    ("unavailable", roots[0], "code"),
                    ("busy", roots[0], "all"),
                    ("failed", roots[1], "all"),
                    ("refused", roots[1], "code"),
                ]
                for kind, root, content in cases:
                    (base / "fault.json").write_text(json.dumps({"kind": kind}))
                    result = await session.call_tool(
                        "search", {"repo": str(root), "query": "operation value", "content": content, "top_k": 3}
                    )
                    answer_text = text(result)
                    rows.append(
                        {
                            "case": kind or "healthy",
                            "answer": answer_text[:800],
                            "answer_sha256": hashlib.sha256(answer_text.encode()).hexdigest(),
                            "state": await state(),
                        }
                    )
                (base / "fault.json").write_text("{}")
                before_restart = await state()
                old = processes.pop("daemon")
                daemon = _DAEMONS.pop(old)
                await asyncio.to_thread(stop_daemon, daemon)
                replacement = await asyncio.to_thread(start_daemon, base, roots, "daemon-two")
                processes["daemon"] = replacement.pid
                _DAEMONS[replacement.pid] = replacement
                answer = await session.call_tool("search", {"repo": str(roots[0]), "query": "operation value"})
                after_restart = await state()
                assert initial["pid"] == after_restart["pid"], "MCP session must survive daemon restart"
                assert "results" in text(answer), "the replacement daemon must answer"
                return {
                    "initial": initial,
                    "cases": rows,
                    "before_restart": before_restart,
                    "after_restart": after_restart,
                    "mcp_log": (base / "mcp.log").read_text(),
                }


_DAEMONS = {}


def measure(revision):
    """Measure current or exported-baseline client behavior against the same isolated load."""
    check_headroom()
    with tempfile.TemporaryDirectory(prefix="zemble-client-memory-", dir="/tmp/opencode") as directory:
        base = Path(directory)
        roots = generate_roots(base, 4000)
        (base / "client").mkdir()
        (base / "fault.json").write_text("{}")
        daemon = start_daemon(base, roots, "daemon-one")
        _DAEMONS[daemon.pid] = daemon
        processes = {"daemon": daemon.pid}
        samples = []
        stop = threading.Event()
        monitor = threading.Thread(target=sample, args=(processes, stop, samples, base / "fault.json"), daemon=True)
        monitor.start()
        try:
            result = asyncio.run(exercise(base, roots, revision, processes))
            return {
                "client_revision": revision or "working-tree",
                "dimensions": 1024,
                "peak_client_rss_mib": round(max(row.get("client", 0) for row in samples), 1),
                "peak_daemon_rss_mib": round(max(row.get("daemon", 0) for row in samples), 1),
                **result,
            }
        finally:
            stop.set()
            monitor.join()
            for process in list(_DAEMONS.values()):
                stop_daemon(process)
            _DAEMONS.clear()


def main():
    """Run the owned daemon/client child or the paired memory replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mcp-child", action="store_true")
    parser.add_argument("--daemon-child", action="store_true")
    args = parser.parse_args()
    if args.mcp_child:
        mcp_child()
    elif args.daemon_child:
        daemon_child()
    else:
        output = json.dumps(measure(args.revision), indent=2) + "\n"
        if args.output:
            args.output.write_text(output)
        else:
            sys.stdout.write(output)


if __name__ == "__main__":
    main()

"""Command-line surface for the daemon.

Lives here rather than in `zemble.cli` so wiring it into the top-level parser is
three lines, matching how the graph and dupes subcommands are attached.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any

from zemble.daemon import client
from zemble.daemon.protocol import (
    DEFAULT_IDLE_MINUTES,
    DEFAULT_MAX_INDEXES,
    DaemonError,
    log_path,
    read_pid,
    socket_path,
)
from zemble.userenv import load_user_env

EXIT_ERROR = 1


def add_daemon_parser(sub: argparse._SubParsersAction) -> None:
    """Register the `daemon` subcommand tree on the main parser."""
    daemon_p = sub.add_parser("daemon", help="Manage the warm index daemon (started on demand, never at login).")
    daemon_sub = daemon_p.add_subparsers(dest="daemon_command", required=True)

    run_p = daemon_sub.add_parser("run", help="Run the daemon in the foreground.")
    run_p.add_argument(
        "--max-indexes",
        type=int,
        default=None,
        metavar="N",
        help=f"Resident index limit (default: $ZEMBLE_DAEMON_MAX_INDEXES, else {DEFAULT_MAX_INDEXES}).",
    )
    run_p.add_argument(
        "--idle-minutes",
        type=int,
        default=None,
        metavar="N",
        help=f"Exit after this long without a request; 0 never exits (default: {DEFAULT_IDLE_MINUTES}).",
    )
    run_p.add_argument("--no-watch", action="store_true", help="Do not watch loaded roots for changes.")

    daemon_sub.add_parser("start", help="Start a detached daemon if none is running.")
    daemon_sub.add_parser("stop", help="Ask a running daemon to exit.")
    daemon_sub.add_parser("restart", help="Stop a running daemon and start a fresh one.")
    status_p = daemon_sub.add_parser("status", help="Show what a running daemon holds.")
    status_p.add_argument("--json", action="store_true", help="Print machine-readable output.")


def run_daemon(args: argparse.Namespace) -> int:
    """Run a `zemble daemon ...` subcommand and return its exit code."""
    command = args.daemon_command
    if command == "run":
        return _run_foreground(args)
    if command == "start":
        return _start()
    if command == "stop":
        return _stop()
    if command == "restart":
        _stop()
        return _start()
    return _status(getattr(args, "json", False))


def _run_foreground(args: argparse.Namespace) -> int:
    """Run the daemon in this process until it stops."""
    import asyncio

    from zemble.daemon.server import SocketInUse, run

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        asyncio.run(run(max_indexes=args.max_indexes, idle_minutes=args.idle_minutes, watch=not args.no_watch))
    except SocketInUse as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover - interactive
        return 0
    return 0


def _start() -> int:
    """Start a detached daemon unless one is already listening."""
    if client.is_running():
        print(f"zemble daemon already running (pid {read_pid()}) on {socket_path()}")
        return 0
    try:
        client.spawn()
    except DaemonError as exc:
        print(f"Failed to start the daemon: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(f"zemble daemon started (pid {read_pid()}) on {socket_path()}; log: {log_path()}")
    return 0


def _stop() -> int:
    """Ask a running daemon to exit."""
    if not client.is_running():
        client.clear_stale()
        print("zemble daemon is not running")
        return 0
    try:
        client.call("shutdown", auto_start=False, timeout=10.0)
    except DaemonError as exc:
        print(f"Failed to stop the daemon: {exc}", file=sys.stderr)
        return EXIT_ERROR
    # Wait for the old process to actually go away, so `restart` (and a start right after a
    # stop) does not find the socket still owned by the exiting daemon.
    deadline = time.monotonic() + 10.0
    while client.is_running() and time.monotonic() < deadline:
        time.sleep(0.1)
    if client.is_running():
        print("zemble daemon did not exit within 10 s", file=sys.stderr)
        return EXIT_ERROR
    client.clear_stale()
    print("zemble daemon stopped")
    return 0


def _status(as_json: bool) -> int:
    """Print what the running daemon holds."""
    try:
        status: dict[str, Any] = client.call("status", auto_start=False, timeout=10.0)
    except DaemonError as exc:
        if as_json:
            print(json.dumps({"running": False, "reason": str(exc)}))
        else:
            print(f"zemble daemon not available: {exc}")
        return EXIT_ERROR
    if as_json:
        print(json.dumps({"running": True, **status}))
        return 0
    print(
        f"pid {status['pid']}  up {status['uptime_seconds']:.0f}s  rss {status['rss_mb']} MB  "
        f"{status['requests']} request(s)  idle {status['idle_seconds']:.0f}s"
    )
    print(f"socket {status['socket']}  max_indexes {status['max_indexes']}  idle_limit {status['idle_minutes_limit']}m")
    for entry in status["indexes"]:
        flags = []
        if entry["watching"]:
            flags.append("watching")
        if entry["rebuilding"]:
            flags.append("rebuilding")
        print(
            f"  {entry['root']} [{','.join(entry['content'])}] {entry['chunks']} chunks, "
            f"{entry['files']} files, {entry['embedder']}{'  (' + ', '.join(flags) + ')' if flags else ''}"
        )
    for entry in status["building"]:
        print(f"  {entry['root']} [{','.join(entry['content'])}] building...")
    for root in status["pending_reindex"]:
        print(f"  {root} pending reindex")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the daemon CLI standalone, as `python -m zemble.daemon`."""
    load_user_env()
    parser = argparse.ArgumentParser(prog="zemble daemon")
    sub = parser.add_subparsers(dest="command")
    add_daemon_parser(sub)
    args = parser.parse_args(["daemon", *(argv if argv is not None else sys.argv[1:])])
    return run_daemon(args)


__all__ = ["add_daemon_parser", "main", "run_daemon"]

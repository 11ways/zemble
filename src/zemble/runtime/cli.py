"""Command-line surface for `zemble status`.

Lives here rather than in `zemble.cli` so wiring it into the top-level parser is
three lines, the same shape the graph, home and daemon subcommands use.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from zemble.runtime.identity import identity, status_payload

STATUS_COMMANDS = ("status",)


def add_status_parser(sub: argparse._SubParsersAction) -> None:
    """Register the `status` subcommand."""
    parser = sub.add_parser("status", help="Show which zemble code this install runs, and the daemon's.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")


def _daemon_status() -> dict[str, Any] | None:
    """Return a running daemon's status, or None when no daemon answers."""
    from zemble.daemon import client
    from zemble.daemon.protocol import DaemonError

    try:
        return client.call("status", auto_start=False, timeout=10.0)
    except DaemonError:
        return None


def run_status(args: argparse.Namespace) -> int:
    """Print this process's runtime identity, plus the daemon's when one is reachable."""
    payload = status_payload(identity())
    daemon = _daemon_status()
    payload["daemon"] = {"running": False} if daemon is None else {"running": True, **daemon.get("runtime", {})}
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    info = payload["identity"]
    print(f"zemble {info['zemble_version']}  rev {info['source_revision'] or 'unknown'}")
    print(f"source {info['source_root']}{'  (editable)' if info['editable'] else ''}")
    print(f"pid {info['pid']}  started {info['process_started_at']}")
    if payload["note"]:
        print(payload["note"])
    if daemon is None:
        print("daemon: not running")
        return 0
    runtime = payload["daemon"]
    print(
        f"daemon: pid {daemon['pid']}  rev {runtime.get('source_revision') or 'unknown'}  "
        f"started {runtime.get('process_started_at', 'unknown')}  "
        f"up {daemon['uptime_seconds']:.0f}s  {'STALE' if runtime.get('stale') else 'current'}"
    )
    return 0


__all__ = ["STATUS_COMMANDS", "add_status_parser", "run_status"]

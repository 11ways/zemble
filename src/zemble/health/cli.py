"""Command-line surface for `zemble health`.

Lives here rather than in `zemble.cli` so wiring the feature into the top-level
parser is three lines, the same shape the home and evidence subcommands use.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from zemble.graph.cli import ensure_graph
from zemble.graph.provider import open_provider
from zemble.health.answers import health_payload, parse_signals
from zemble.health.model import Signal
from zemble.health.report import DEFAULT_LIMIT
from zemble.home.config import ConfigError, HomeConfig
from zemble.workspace import resolve_home_root

HEALTH_COMMANDS = ("health",)

#: What a run that could not answer exits with; a report with findings still exits 0.
EXIT_ERROR = 1


def add_health_parser(sub: argparse._SubParsersAction) -> None:
    """Register the `health` subcommand."""
    from zemble.cli import _add_daemon_arg, add_root_arg

    parser = sub.add_parser(
        "health",
        help="Ranked code-shape findings and a score (a report, never a gate).",
        epilog=f"Signals: {', '.join(signal.value for signal in Signal)}.",
    )
    add_root_arg(parser, nargs="?", default=".", help="Directory to report on (default: current directory).")
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, help=f"Findings to list (default: {DEFAULT_LIMIT})."
    )
    parser.add_argument("--signals", default=None, help="Comma-separated signals to measure (default: all).")
    parser.add_argument(
        "--paths",
        nargs="+",
        default=None,
        metavar="PATH",
        help="Restrict the report to these paths, relative to the directory (or absolute).",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    _add_daemon_arg(parser)


def run_health(args: argparse.Namespace) -> int:
    """Print one health report, from the warm daemon where there is one; 0 whenever it answered."""
    from zemble.cli import _via_daemon

    signals = args.signals.split(",") if args.signals else None
    try:
        parse_signals(signals)
        request = _daemon_args(args, signals)
        payload = _via_daemon(args.command, request, args.no_daemon, None)
        if payload is None:
            payload = in_process(request["path"], signals, request["paths"], args.limit)
    except (ConfigError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return EXIT_ERROR
    if args.json:
        print(json.dumps(payload["health"], indent=2))
    else:
        print(payload["markdown"], end="")
    return 0


def _daemon_args(args: argparse.Namespace, signals: list[str] | None) -> dict[str, Any]:
    """Shape the subcommand as daemon command arguments; the daemon refuses relative paths."""
    return {"path": os.path.abspath(args.path), "signals": signals, "paths": args.paths or [], "limit": args.limit}


def in_process(path: str, signals: list[str] | None, paths: list[str], limit: int) -> dict[str, Any]:
    """Answer in this process, building or refreshing the graph first.

    The graph is the requested directory's own, or an ancestor's view of it; the config is the
    nearest ancestor's `home.toml`, as `home` reads it.

    :raises ConfigError: If that `home.toml` is malformed.
    """
    config = HomeConfig.load(resolve_home_root(path))
    ensure_graph(path)
    provider = open_provider(path)
    try:
        return health_payload(provider, config, signals=signals, paths=paths, limit=limit)
    finally:
        provider.close()


__all__ = ["HEALTH_COMMANDS", "add_health_parser", "in_process", "run_health"]

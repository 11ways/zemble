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
from zemble.health.answers import HealthRequest, health_payload, parse_signals
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
    parser.add_argument(
        "--since",
        default=None,
        metavar="REF",
        help="List only the findings touching a line changed since this git ref (working tree and untracked "
        "files included); the score stays the whole scope's.",
    )
    parser.add_argument(
        "--copying",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Measure the duplication tier (clone_mass, cloned_body, pass_through, vocabulary_drift); "
        "default: on with --since, --paths or a sub-directory, off over a whole tree (it parses every file).",
    )
    parser.add_argument(
        "--baseline", default=None, metavar="FILE", help="Report resolved/remaining/new against a saved baseline."
    )
    parser.add_argument(
        "--save-baseline", default=None, metavar="FILE", help="Write this run's finding keys as a baseline."
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    _add_daemon_arg(parser)


def run_health(args: argparse.Namespace) -> int:
    """Print one health report, from the warm daemon where there is one; 0 whenever it answered."""
    from zemble.cli import _via_daemon

    path = os.path.abspath(args.path)
    request = HealthRequest(
        signals=tuple(args.signals.split(",")) if args.signals else None,
        paths=tuple(args.paths or ()),
        limit=args.limit,
        since=args.since,
        copying=args.copying,
        # The daemon refuses relative paths, and runs in another directory.
        baseline=os.path.abspath(args.baseline) if args.baseline else None,
        save_baseline=os.path.abspath(args.save_baseline) if args.save_baseline else None,
    )
    try:
        parse_signals(request.signals)
        payload = _via_daemon(args.command, request.to_args(path), args.no_daemon, None)
        if payload is None:
            payload = in_process(path, request)
    except (ConfigError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return EXIT_ERROR
    if args.json:
        print(json.dumps(payload["health"], indent=2))
    else:
        print(payload["markdown"], end="")
    return 0


def in_process(path: str, request: HealthRequest) -> dict[str, Any]:
    """Answer one request in this process, building or refreshing the graph first.

    The graph is the requested directory's own, or an ancestor's view of it; the config is the
    nearest ancestor's `home.toml`, as `home` reads it.

    :raises ConfigError: If that `home.toml` is malformed.
    :raises ValueError: If `health_payload` refuses the request.
    """
    config = HomeConfig.load(resolve_home_root(path))
    ensure_graph(path)
    provider = open_provider(path)
    try:
        return health_payload(provider, config, request)
    finally:
        provider.close()


__all__ = ["HEALTH_COMMANDS", "add_health_parser", "in_process", "run_health"]

"""Command-line surface for evidence bundles, outlines and signatures.

Lives here rather than in `zemble.cli` so wiring the feature into the top-level
parser is three lines, the same shape the graph subcommands use.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from zemble.evidence.answers import (
    DEFAULT_BUDGET,
    DEFAULT_TOP_K,
    explain_payload,
    outline_payload,
    signatures_payload,
)
from zemble.graph.cli import EXIT_AMBIGUOUS, EXIT_NOT_FOUND, ensure_graph
from zemble.graph.provider import SqliteGraphProvider
from zemble.index import ZembleIndex
from zemble.types import ContentType

EVIDENCE_COMMANDS = ("explain", "outline", "signatures")


def add_evidence_parser(sub: argparse._SubParsersAction) -> None:
    """Register the `explain`, `outline` and `signatures` subcommands."""
    explain_p = sub.add_parser("explain", help="Budgeted evidence bundle: search plus one graph hop.")
    explain_p.add_argument("path", help="Workspace directory to search and build the graph for.")
    explain_p.add_argument("query", help="Natural language or code query.")
    explain_p.add_argument(
        "--budget", type=int, default=DEFAULT_BUDGET, help=f"Token budget for the bundle (default: {DEFAULT_BUDGET})."
    )
    explain_p.add_argument(
        "-k", "--top-k", type=int, default=DEFAULT_TOP_K, help=f"Search results to expand (default: {DEFAULT_TOP_K})."
    )
    explain_p.add_argument("--json", action="store_true", help="Print machine-readable output.")
    # Only `explain` reads the index, so it is the only one an embedder override concerns.
    from zemble.cli import _add_daemon_arg, _add_embedder_arg

    _add_embedder_arg(explain_p)
    _add_daemon_arg(explain_p)

    outline_p = sub.add_parser("outline", help="Signature-only view of a file or a type.")
    outline_p.add_argument("path", help="Workspace directory the graph was built for.")
    outline_p.add_argument("target", help="Workspace-relative file path, or a simple or qualified type name.")
    outline_p.add_argument("--members", metavar="PATTERN", help="Only show members matching this name pattern.")
    outline_p.add_argument("--json", action="store_true", help="Print machine-readable output.")
    _add_daemon_arg(outline_p)

    signatures_p = sub.add_parser("signatures", help="A symbol's signature and the call sites resolved exactly.")
    signatures_p.add_argument("path", help="Workspace directory the graph was built for.")
    signatures_p.add_argument("symbol", help="Simple name, qualified name, or Type.member.")
    signatures_p.add_argument("--json", action="store_true", help="Print machine-readable output.")
    _add_daemon_arg(signatures_p)


def run_evidence(args: argparse.Namespace) -> int:
    """Run one evidence subcommand, from the warm daemon where there is one, and return its exit code."""
    from zemble.cli import _via_daemon

    payload = _via_daemon(args.command, _daemon_args(args), args.no_daemon, getattr(args, "embedder", None))
    if payload is None:
        payload = _in_process(args)
    if args.command == "explain":
        return _print_explain(args, payload)
    return _print_answer(args, payload, "outline" if args.command == "outline" else "signatures")


def _daemon_args(args: argparse.Namespace) -> dict[str, Any]:
    """Shape one evidence subcommand as daemon command arguments."""
    if args.command == "explain":
        return {
            "path": args.path,
            "query": args.query,
            "budget": args.budget,
            "top_k": args.top_k,
            "content": [ContentType.CODE.value],
        }
    if args.command == "outline":
        return {"path": args.path, "target": args.target, "members": args.members}
    return {"path": args.path, "symbol": args.symbol}


def _in_process(args: argparse.Namespace) -> dict[str, Any]:
    """Answer one evidence subcommand in this process, building whatever it needs."""
    ensure_graph(args.path)
    provider = SqliteGraphProvider(args.path)
    try:
        if args.command == "explain":
            return explain_payload(_load_index(args.path, args.embedder), provider, args.query, args.budget, args.top_k)
        if args.command == "outline":
            return outline_payload(provider, args.target, args.members)
        return signatures_payload(provider, args.symbol)
    finally:
        provider.close()


def _load_index(path: str, embedder: str | None = None) -> ZembleIndex:
    """Build or load the code index for a workspace, saving it back to the cache.

    Imported lazily: `zemble.cli` imports this module, so the reverse import can
    only happen once the parser is already built.
    """
    from zemble.cli import _load_index as load
    from zemble.cli import _maybe_save_index

    index = load(path, [ContentType.CODE], embedder)
    _maybe_save_index(index, path)
    return index


def _print_explain(args: argparse.Namespace, payload: dict[str, Any]) -> int:
    """Print an evidence bundle, as JSON or as markdown."""
    bundle = payload["bundle"]
    print(json.dumps(bundle, indent=2) if args.json else payload["markdown"])
    return 0 if bundle["items"] else EXIT_NOT_FOUND


def _print_answer(args: argparse.Namespace, payload: dict[str, Any], key: str) -> int:
    """Print an outline or a signatures answer, or the refusal that came instead."""
    if "error" in payload:
        candidates = payload["candidates"]
        if args.json:
            print(
                json.dumps(
                    {"error": payload["error"], "candidates": [c["qualified_name"] for c in candidates]}, indent=2
                )
            )
        else:
            print(payload["error"], file=sys.stderr)
            for candidate in candidates:
                print(f"  {candidate['qualified_name']}  {candidate['file_path']}", file=sys.stderr)
        return EXIT_AMBIGUOUS if candidates else EXIT_NOT_FOUND
    print(json.dumps(payload[key], indent=2) if args.json else payload["text"])
    return 0


__all__ = ["EVIDENCE_COMMANDS", "add_evidence_parser", "run_evidence"]

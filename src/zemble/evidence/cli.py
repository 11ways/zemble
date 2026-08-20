"""Command-line surface for evidence bundles, outlines and signatures.

Lives here rather than in `zemble.cli` so wiring the feature into the top-level
parser is three lines, the same shape the graph subcommands use.
"""

from __future__ import annotations

import argparse
import json
import sys

from zemble.evidence.bundle import build_bundle
from zemble.evidence.outline import OutlineError, outline, signatures
from zemble.graph.cli import EXIT_AMBIGUOUS, EXIT_NOT_FOUND, ensure_graph, select_symbol
from zemble.graph.provider import SqliteGraphProvider
from zemble.index import ZembleIndex
from zemble.types import ContentType

EVIDENCE_COMMANDS = ("explain", "outline", "signatures")

DEFAULT_BUDGET = 3000
DEFAULT_TOP_K = 20


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

    outline_p = sub.add_parser("outline", help="Signature-only view of a file or a type.")
    outline_p.add_argument("path", help="Workspace directory the graph was built for.")
    outline_p.add_argument("target", help="Workspace-relative file path, or a simple or qualified type name.")
    outline_p.add_argument("--members", metavar="PATTERN", help="Only show members matching this name pattern.")
    outline_p.add_argument("--json", action="store_true", help="Print machine-readable output.")

    signatures_p = sub.add_parser("signatures", help="A symbol's signature and the call sites resolved exactly.")
    signatures_p.add_argument("path", help="Workspace directory the graph was built for.")
    signatures_p.add_argument("symbol", help="Simple name, qualified name, or Type.member.")
    signatures_p.add_argument("--json", action="store_true", help="Print machine-readable output.")


def run_evidence(args: argparse.Namespace) -> int:
    """Run one evidence subcommand and return its exit code."""
    ensure_graph(args.path)
    provider = SqliteGraphProvider(args.path)
    try:
        if args.command == "explain":
            return _run_explain(args, provider)
        if args.command == "outline":
            return _run_outline(args, provider)
        return _run_signatures(args, provider)
    finally:
        provider.close()


def _load_index(path: str) -> ZembleIndex:
    """Build or load the code index for a workspace, saving it back to the cache.

    Imported lazily: `zemble.cli` imports this module, so the reverse import can
    only happen once the parser is already built.
    """
    from zemble.cli import _load_index as load
    from zemble.cli import _maybe_save_index

    index = load(path, [ContentType.CODE])
    _maybe_save_index(index, path)
    return index


def _run_explain(args: argparse.Namespace, provider: SqliteGraphProvider) -> int:
    """Build and print an evidence bundle."""
    index = _load_index(args.path)
    bundle = build_bundle(index, provider, args.query, args.budget, top_k=args.top_k)
    if args.json:
        print(json.dumps(bundle.to_dict(), indent=2))
    else:
        print(bundle.render())
    return 0 if bundle.items else EXIT_NOT_FOUND


def _run_outline(args: argparse.Namespace, provider: SqliteGraphProvider) -> int:
    """Print the outline of a file or a type."""
    try:
        rendered = outline(provider, args.target, args.members)
    except OutlineError as error:
        if args.json:
            print(
                json.dumps(
                    {
                        "error": error.message,
                        "candidates": [symbol.qualified_name for symbol in error.candidates],
                    },
                    indent=2,
                )
            )
        else:
            print(error.message, file=sys.stderr)
            for symbol in error.candidates:
                print(f"  {symbol.qualified_name}  {symbol.file_path}", file=sys.stderr)
        return EXIT_AMBIGUOUS if error.candidates else EXIT_NOT_FOUND
    print(json.dumps(rendered.to_dict(), indent=2) if args.json else rendered.render())
    return 0


def _run_signatures(args: argparse.Namespace, provider: SqliteGraphProvider) -> int:
    """Print a symbol's signature and its exact callers."""
    chosen, candidates = select_symbol(provider.definition(args.symbol), args.symbol)
    if chosen is None:
        message = (
            f"{args.symbol!r} is ambiguous; pass a qualified name."
            if candidates
            else f"No symbol named {args.symbol!r}. {provider.coverage_note()}"
        )
        if args.json:
            print(json.dumps({"error": message, "candidates": [s.qualified_name for s in candidates]}, indent=2))
        else:
            print(message, file=sys.stderr)
            for symbol in candidates:
                print(f"  {symbol.qualified_name}  {symbol.file_path}", file=sys.stderr)
        return EXIT_AMBIGUOUS if candidates else EXIT_NOT_FOUND
    answer = signatures(provider, chosen)
    print(json.dumps(answer.to_dict(), indent=2) if args.json else answer.render())
    return 0


__all__ = ["EVIDENCE_COMMANDS", "add_evidence_parser", "run_evidence"]

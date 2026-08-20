"""Command-line surface for the Java symbol graph.

Lives here rather than in `zemble.cli` so wiring the feature into the top-level
parser is three lines.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from zemble.graph.model import TYPE_KINDS, EdgeKind, Hit, Symbol, SymbolKind
from zemble.graph.provider import SqliteGraphProvider, display_name
from zemble.graph.store import build_graph, graph_exists

QUERY_COMMANDS = (
    "definition",
    "callers",
    "callees",
    "references",
    "implementations",
    "supertypes",
    "overrides-of",
    "overridden-by",
    "tests-of",
    "neighbors",
)

EXIT_AMBIGUOUS = 2
EXIT_NOT_FOUND = 1

_METHODS = {
    "callers": "callers",
    "callees": "callees",
    "references": "references",
    "implementations": "implementations",
    "supertypes": "supertypes",
    "overrides-of": "overrides_of",
    "overridden-by": "overridden_by",
    "tests-of": "tests_of",
}

_refreshed: set[str] = set()


def add_graph_parser(sub: argparse._SubParsersAction) -> None:
    """Register the `graph` subcommand tree on the main parser."""
    graph_p = sub.add_parser("graph", help="Java symbol graph: definitions, callers, implementations, tests.")
    graph_sub = graph_p.add_subparsers(dest="graph_command", required=True)

    build_p = graph_sub.add_parser("build", help="Build or incrementally refresh the symbol graph.")
    build_p.add_argument("path", nargs="?", default=".", help="Workspace directory (default: current directory).")
    build_p.add_argument("--stats", action="store_true", help="Print the full build statistics.")
    build_p.add_argument("--force", action="store_true", help="Re-extract every file instead of only changed ones.")
    build_p.add_argument("--json", action="store_true", help="Print machine-readable output.")
    build_p.add_argument("--no-daemon", action="store_true", help="Do not use the warm daemon.")

    for command in QUERY_COMMANDS:
        query_p = graph_sub.add_parser(command, help=f"Graph query: {command}.")
        query_p.add_argument("path", help="Workspace directory the graph was built for.")
        query_p.add_argument("symbol", help="Simple name, qualified name, or Type.member.")
        query_p.add_argument("--json", action="store_true", help="Print machine-readable output.")
        query_p.add_argument("--no-daemon", action="store_true", help="Do not use the warm daemon.")
        if command == "neighbors":
            query_p.add_argument("--hops", type=int, default=1, help="How far to walk (default: 1).")
            query_p.add_argument(
                "--kinds",
                nargs="+",
                choices=[kind.value for kind in EdgeKind],
                metavar="KIND",
                help="Only follow these edge kinds.",
            )


def run_graph(args: argparse.Namespace) -> int:
    """Run a `zemble graph ...` subcommand and return its exit code."""
    if getattr(args, "no_daemon", False):
        from zemble.daemon import client

        client.disable_for_this_process("--no-daemon")
    if args.graph_command == "build":
        return _run_build(args)
    return _run_query(args)


def _run_build(args: argparse.Namespace) -> int:
    """Build the graph and report what it did."""
    stats = build_graph(args.path, force=args.force)
    if args.json:
        print(json.dumps(stats.to_dict(), indent=2))
        return 0
    print(
        f"Graph built for {stats.root}: {stats.symbols} symbols, {stats.edges} edges "
        f"({stats.extracted_files} files extracted, {stats.unchanged_files} unchanged, "
        f"{stats.reresolved_files} re-resolved) in {stats.duration_seconds:.1f}s"
    )
    if stats.skipped_by_language:
        for language, count in sorted(stats.skipped_by_language.items(), key=lambda item: -item[1]):
            print(f"  skipped {count} {language} file(s): no graph extractor for {language}")
    if args.stats:
        print(json.dumps(stats.to_dict(), indent=2))
    return 0


def ensure_graph(path: str, *, refresh: bool = True, allow_daemon: bool = True) -> None:
    """Build the graph if it is missing, and refresh it once per process.

    A warm daemon does both instead where one is reachable: it keeps the graph fresh with
    its watcher, so the client skips the workspace scan a refresh costs.

    :param path: The workspace directory.
    :param refresh: Whether an existing graph is refreshed once per process.
    :param allow_daemon: Whether a daemon may be asked; a handler running INSIDE the daemon
        passes False, because asking a daemon to ensure the graph it is already ensuring
        recurses through its own socket.
    """
    if not graph_exists(path):
        _refreshed.add(path)
        if not (allow_daemon and _ensure_via_daemon(path)):
            build_graph(path)
        return
    if refresh and path not in _refreshed:
        _refreshed.add(path)
        if not (allow_daemon and _ensure_via_daemon(path)):
            build_graph(path)


def _ensure_via_daemon(path: str) -> bool:
    """Ask the daemon to guarantee a fresh graph for a path.

    :param path: The workspace directory.
    :return: Whether the daemon answered; False means do it in this process.
    """
    from zemble.daemon import client
    from zemble.daemon.protocol import DaemonError

    try:
        client.call("graph", {"path": path, "command": "ensure"})
    except DaemonError:
        return False
    return True


def select_symbol(symbols: Sequence[Symbol], name: str) -> tuple[Symbol | None, list[Symbol]]:
    """Pick the one symbol a written name means, or report the competing candidates.

    A constructor carries its type's name, so `Foo` would always look ambiguous; when
    a type of that name exists the type wins and its constructors drop out.
    """
    if not symbols:
        return None, []
    exact = [symbol for symbol in symbols if symbol.qualified_name == name]
    pool = list(exact or symbols)
    last = name.rsplit(".", 1)[-1]
    if any(symbol.kind in TYPE_KINDS and symbol.name == last for symbol in pool):
        pool = [symbol for symbol in pool if symbol.kind is not SymbolKind.CONSTRUCTOR]
    if len(pool) == 1:
        return pool[0], pool
    return None, pool


def _run_query(args: argparse.Namespace) -> int:
    """Run one graph query and print its answer."""
    ensure_graph(args.path)
    provider = SqliteGraphProvider(args.path)
    try:
        return _answer_query(args, provider)
    finally:
        provider.close()


def _answer_query(args: argparse.Namespace, provider: SqliteGraphProvider) -> int:
    """Resolve the written name and print the requested relationship."""
    symbols = provider.definition(args.symbol)
    if args.graph_command == "definition":
        return _print_definitions(args, provider, symbols)
    chosen, candidates = select_symbol(symbols, args.symbol)
    if chosen is None:
        return _report_selection_failure(args, provider, candidates)
    if args.graph_command == "neighbors":
        kinds = [EdgeKind(value) for value in args.kinds] if args.kinds else None
        hits = provider.neighbors(chosen.id, hops=args.hops, kinds=kinds)
    else:
        hits = getattr(provider, _METHODS[args.graph_command])(chosen.id)
    if args.json:
        print(json.dumps({"symbol": _symbol_json(chosen), "results": [_hit_json(hit) for hit in hits]}, indent=2))
        return 0
    print(f"{display_name(chosen)}  [{chosen.kind.value}]  {chosen.file_path}:{chosen.start_line}")
    print(f"{args.graph_command}: {len(hits)} result(s)")
    for hit in hits:
        print(f"  {hit.symbol.file_path}:{hit.line}  {hit.reason}")
    if not hits:
        print(f"  ({provider.coverage_note()})")
    return 0


def _print_definitions(args: argparse.Namespace, provider: SqliteGraphProvider, symbols: Sequence[Symbol]) -> int:
    """Print every declaration matching a name."""
    if args.json:
        print(json.dumps({"results": [_symbol_json(symbol) for symbol in symbols]}, indent=2))
        return 0 if symbols else EXIT_NOT_FOUND
    if not symbols:
        print(f"No symbol named {args.symbol!r}. {provider.coverage_note()}", file=sys.stderr)
        return EXIT_NOT_FOUND
    for symbol in symbols:
        print(f"{symbol.kind.value:14} {symbol.file_path}:{symbol.start_line}  {symbol.signature or symbol.name}")
    return 0


def _report_selection_failure(
    args: argparse.Namespace, provider: SqliteGraphProvider, candidates: Sequence[Symbol]
) -> int:
    """Report an unknown or ambiguous name with the exit code that says which."""
    if not candidates:
        message = f"No symbol named {args.symbol!r}. {provider.coverage_note()}"
        if args.json:
            print(json.dumps({"error": message, "candidates": []}, indent=2))
        else:
            print(message, file=sys.stderr)
        return EXIT_NOT_FOUND
    if args.json:
        print(
            json.dumps(
                {
                    "error": f"{args.symbol!r} is ambiguous",
                    "candidates": [_symbol_json(symbol) for symbol in candidates],
                },
                indent=2,
            )
        )
    else:
        print(f"{args.symbol!r} is ambiguous; {len(candidates)} candidates:", file=sys.stderr)
        for symbol in candidates:
            print(f"  {symbol.qualified_name}  [{symbol.kind.value}]  {symbol.file_path}", file=sys.stderr)
        print("Re-run with a qualified name.", file=sys.stderr)
    return EXIT_AMBIGUOUS


def _symbol_json(symbol: Symbol) -> dict[str, object]:
    """Render a symbol as JSON."""
    return {
        "id": symbol.id,
        "kind": symbol.kind.value,
        "name": symbol.name,
        "qualified_name": symbol.qualified_name,
        "file_path": symbol.file_path,
        "start_line": symbol.start_line,
        "end_line": symbol.end_line,
        "signature": symbol.signature,
        "modifiers": symbol.modifiers,
        "annotations": symbol.annotations,
        "is_test": symbol.is_test,
    }


def _hit_json(hit: Hit) -> dict[str, object]:
    """Render a hit as JSON."""
    return {
        **_symbol_json(hit.symbol),
        "edge_kind": hit.edge_kind.value,
        "line": hit.line,
        "resolution": hit.resolution.value,
        "reason": hit.reason,
        "depth": hit.depth,
    }


__all__ = ["EXIT_AMBIGUOUS", "QUERY_COMMANDS", "add_graph_parser", "ensure_graph", "run_graph"]

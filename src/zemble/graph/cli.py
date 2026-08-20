"""Command-line surface for the Java symbol graph.

Lives here rather than in `zemble.cli` so wiring the feature into the top-level
parser is three lines.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from zemble.graph.facts import TREE_SITTER_SOURCE, FactsFile, FactsOverlay, facts_source_globs, load_overlay
from zemble.graph.model import TYPE_KINDS, EdgeKind, Hit, Symbol, SymbolKind
from zemble.graph.provider import SqliteGraphProvider, display_name
from zemble.graph.store import build_graph, graph_exists, symbol_from_row

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

    facts_p = graph_sub.add_parser("facts", help="Inspect the graph facts overlay written by external tools.")
    facts_sub = facts_p.add_subparsers(dest="facts_command", required=True)
    status_p = facts_sub.add_parser("status", help="Show which facts files were found and what they cover.")
    status_p.add_argument("path", nargs="?", default=".", help="Workspace directory (default: current directory).")
    status_p.add_argument("--json", action="store_true", help="Print machine-readable output.")
    status_p.add_argument("--no-daemon", action="store_true", help="Do not use the warm daemon.")
    status_p.add_argument("--limit", type=int, default=20, help="How many unmapped refs to list (default: 20).")

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
    if args.graph_command == "facts":
        return _run_facts_status(args)
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


def _run_facts_status(args: argparse.Namespace) -> int:
    """Report every facts file found for a workspace and what the graph made of it."""
    ensure_graph(args.path)
    provider = SqliteGraphProvider(args.path)
    try:
        symbols = [symbol_from_row(row) for row in provider.connection.execute("SELECT * FROM symbols")]
        overlay = load_overlay(Path(args.path).expanduser().resolve(), symbols)
        calls = _call_grades(provider)
        by_source = {
            row["source"] or TREE_SITTER_SOURCE: row["n"]
            for row in provider.connection.execute("SELECT source, COUNT(*) AS n FROM edges GROUP BY source")
        }
    finally:
        provider.close()
    if args.json:
        print(json.dumps(_facts_json(overlay, by_source, calls, args.limit), indent=2))
        return 0
    _print_facts_status(args.path, overlay, by_source, calls, args.limit)
    return 0


def _call_grades(provider: SqliteGraphProvider) -> dict[str, dict[str, int]]:
    """Count CALLS edges by resolution, split into the ones a tool wrote and the rest."""
    grades: dict[str, dict[str, int]] = {"with_facts": {}, "without_facts": {}}
    rows = provider.connection.execute(
        "SELECT source, resolution, COUNT(*) AS n FROM edges WHERE kind = 'calls' GROUP BY source, resolution"
    )
    for row in rows:
        bucket = "without_facts" if (row["source"] or TREE_SITTER_SOURCE) == TREE_SITTER_SOURCE else "with_facts"
        grades[bucket][row["resolution"]] = grades[bucket].get(row["resolution"], 0) + row["n"]
    return grades


def _age(seconds: float | None) -> str:
    """Render an age in the largest unit that stays readable."""
    if seconds is None:
        return "unknown age"
    if seconds < 90:
        return f"{int(seconds)}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m ago"
    if seconds < 172800:
        return f"{seconds / 3600:.0f}h ago"
    return f"{seconds / 86400:.0f}d ago"


def _unmapped_summary(overlay: FactsOverlay, limit: int) -> list[dict[str, object]]:
    """Group the unmapped refs by ref and reason, most frequent first."""
    counted: dict[tuple[str, str, str], int] = {}
    for entry in overlay.unmapped:
        key = (entry.ref, entry.reason, entry.fact_kind)
        counted[key] = counted.get(key, 0) + 1
    ordered = sorted(counted.items(), key=lambda item: (-item[1], item[0]))
    return [
        {"ref": ref, "reason": reason, "fact_kind": kind, "count": count}
        for (ref, reason, kind), count in ordered[:limit]
    ]


def _facts_json(
    overlay: FactsOverlay, by_source: dict[str, int], calls: dict[str, dict[str, int]], limit: int
) -> dict[str, object]:
    """Render the facts status as JSON."""
    return {
        "root": str(overlay.root),
        "files": [
            {
                "path": loaded.relative_path,
                "tool": loaded.header.tool,
                "tool_version": loaded.header.tool_version,
                "generated_at": loaded.header.generated_at,
                "age_seconds": loaded.header.age_seconds,
                "language": loaded.header.language,
                "files_declared": len(loaded.sources),
                "files_fresh": len(loaded.fresh_files),
                "files_stale": len(loaded.stale_files),
                "unknown_facts": sum(loaded.unknown_kinds.values()),
                "orphan_facts": loaded.orphan_facts,
                "outside_root": loaded.outside_root,
            }
            for loaded in overlay.files
        ],
        "errors": [{"path": path, "error": message} for path, message in overlay.errors],
        "coverage": {**overlay.stats(), "edges_by_source": by_source, "calls": calls},
        "unmapped": _unmapped_summary(overlay, limit),
    }


def _print_facts_file(loaded: FactsFile) -> None:
    """Print one facts file's header line and what it declared."""
    header = loaded.header
    print(
        f"  {loaded.relative_path}  [{header.tool} {header.tool_version}, {header.language}]  "
        f"generated {_age(header.age_seconds)}"
    )
    print(
        f"    {len(loaded.sources)} file(s) declared, {len(loaded.fresh_files)} fresh, {len(loaded.stale_files)} stale"
    )
    for stale in sorted(loaded.stale_files)[:5]:
        print(f"      stale: {stale} ({loaded.sources[stale].reason})")
    if loaded.unknown_kinds:
        listed = ", ".join(f"{kind} ({count})" for kind, count in loaded.unknown_kinds.most_common(5))
        print(f"    skipped {sum(loaded.unknown_kinds.values())} unknown fact(s): {listed}")
    if loaded.orphan_facts:
        print(f"    skipped {loaded.orphan_facts} fact(s) with no preceding file line")
    if loaded.outside_root:
        print(f"    skipped {loaded.outside_root} file(s) outside the workspace root")


def _print_facts_status(
    path: str, overlay: FactsOverlay, by_source: dict[str, int], calls: dict[str, dict[str, int]], limit: int
) -> None:
    """Print the human-readable facts status."""
    print(f"Facts for {path}: {len(overlay.files)} file(s) found")
    for loaded in overlay.files:
        _print_facts_file(loaded)
    for error_path, message in overlay.errors:
        print(f"  {error_path}: REFUSED: {message}")
    if not overlay.files and not overlay.errors:
        print("  (none; looked for: " + ", ".join(_globs_of(overlay)) + ")")
    stats = overlay.stats()
    print(
        f"coverage: {stats['files_fresh']} of {stats['files_declared']} declared file(s) fresh, "
        f"{stats['edges']} fact edge(s), {stats['unmapped']} unmapped ref(s)"
    )
    print("  edges by source: " + (", ".join(f"{name} {count}" for name, count in sorted(by_source.items())) or "none"))
    for bucket, label in (("with_facts", "calls in covered files"), ("without_facts", "calls elsewhere")):
        grades = calls[bucket]
        listed = ", ".join(f"{grade} {count}" for grade, count in sorted(grades.items())) or "none"
        print(f"  {label}: {listed}")
    unmapped = _unmapped_summary(overlay, limit)
    if unmapped:
        print(f"unmapped refs (top {len(unmapped)}):")
        for entry in unmapped:
            print(f"  {entry['count']}x  {entry['ref']}  [{entry['fact_kind']}]  {entry['reason']}")


def _globs_of(overlay: FactsOverlay) -> tuple[str, ...]:
    """Return the discovery globs used for a workspace, for an empty report."""
    return facts_source_globs(overlay.root)


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
        "source": hit.source,
        "reason": hit.reason,
        "depth": hit.depth,
    }


__all__ = ["EXIT_AMBIGUOUS", "QUERY_COMMANDS", "add_graph_parser", "ensure_graph", "run_graph"]

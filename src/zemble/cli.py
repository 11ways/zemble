import argparse
import io
import json
import re
import sys
import warnings
from importlib.util import find_spec
from pathlib import Path
from shutil import rmtree
from typing import Literal

from zemble.cache import cache_key, resolve_cache_folder, save_index_to_cache
from zemble.dedup.cli import add_dupes_parser, run_dupes
from zemble.embedding.registry import EmbedderSpecError, resolve_embedder_spec
from zemble.evidence.cli import EVIDENCE_COMMANDS, add_evidence_parser, run_evidence
from zemble.graph.cli import add_graph_parser, run_graph
from zemble.index import ZembleIndex
from zemble.index.types import PersistencePath
from zemble.installer.agents import AGENTS, IntegrationType
from zemble.stats import format_savings_report
from zemble.types import ContentType
from zemble.utils import format_results, is_git_url, resolve_chunk
from zemble.version import __version__

_CLI_DISPATCH_ARGS = frozenset(
    {
        "search",
        "find-related",
        "install",
        "uninstall",
        "savings",
        "stats",
        "-h",
        "--help",
        "clear",
        "--version",
        "-V",
        "graph",
        "dupes",
        *EVIDENCE_COMMANDS,
    }
)
_CLEAR_CHOICE = Literal["all", "index", "savings", "orphans"]

#: Subcommands that own their own runner and return an exit code.
_SUBCOMMAND_RUNNERS = {"graph": run_graph, "dupes": run_dupes, **dict.fromkeys(EVIDENCE_COMMANDS, run_evidence)}

_SHA_256_REGEX = re.compile(r"^[a-f0-9]{64}$")


def _build_index(path: str, content: list[ContentType], embedder: str | None = None) -> ZembleIndex:
    """Build an index from a local path or git URL.

    :param path: Local path or git URL.
    :param content: Content types to index.
    :param embedder: Embedder spec, or None for the environment default.
    :return: The index.
    """
    return (
        ZembleIndex.from_git(path, content=content, embedder=embedder)
        if is_git_url(path)
        else ZembleIndex.from_path(path, content=content, embedder=embedder)
    )


def _maybe_save_index(index: ZembleIndex, path: str) -> None:
    """Save the index to the cache folder if it was not loaded from disk."""
    try:
        save_index_to_cache(index, path)
    except Exception as e:
        print(f"Error saving index: {e}", file=sys.stderr)


def _add_embedder_arg(p: argparse.ArgumentParser) -> None:
    """Add --embedder to a subparser."""
    p.add_argument(
        "--embedder",
        default=None,
        metavar="SPEC",
        help=(
            "Embedder spec: model2vec:<hf-model>, voyage:<model>[@<dims>], or openai:<base_url>#<model>[@<dims>]. "
            f"Default: $ZEMBLE_EMBEDDER, else {resolve_embedder_spec()}."
        ),
    )


def _run_stats(path: str, content: list[ContentType], embedder: str | None = None) -> None:
    """Handle the `stats` subcommand: report what an index holds and which embedder built it."""
    index = _load_index(path, content, embedder)
    stats = index.stats
    print(
        json.dumps(
            {
                "path": path,
                "embedder": stats.embedder,
                "dimensions": stats.dimensions,
                "indexed_files": stats.indexed_files,
                "total_chunks": stats.total_chunks,
                "content": [c.value for c in index.content],
                "languages": stats.languages,
            }
        )
    )
    _maybe_save_index(index, path)


def _add_content_args(p: argparse.ArgumentParser) -> None:
    """Add --content and deprecated --include-text-files to a subparser."""
    p.add_argument(
        "--content",
        nargs="+",
        default=["code"],
        choices=[ct.value for ct in ContentType] + ["all"],
        metavar="TYPE",
        help="Content types to search (space-separated, e.g. --content code docs). Choices: code, docs, config, all. Default: code.",
    )
    p.add_argument(
        "--include-text-files",
        action="store_true",
        help="Deprecated. Use --content all instead.",
    )


def main() -> None:
    """Entry point for the zemble command-line tool."""
    # Non-UTF-8 Windows consoles can't encode glyphs like "✓" and would otherwise crash.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="replace")
    if len(sys.argv) > 1 and sys.argv[1] in _CLI_DISPATCH_ARGS:
        _cli_main()
    else:
        _mcp_main()


def _mcp_main() -> None:
    parser = argparse.ArgumentParser(
        prog="zemble",
        description="Instant local code search for agents.",
    )
    _add_content_args(parser)
    args = parser.parse_args()
    from model2vec.utils import get_package_extras

    if any(find_spec(dep) is None for dep in get_package_extras("zemble", "mcp")):
        print("MCP dependencies are not installed. Run: pip install 'zemble[mcp]'", file=sys.stderr)
        raise SystemExit(1)
    import asyncio

    from zemble.mcp import serve

    content = _resolve_content(args.content, args.include_text_files)
    asyncio.run(serve(content))


def _resolve_content(content: list[str], include_text_files: bool) -> list[ContentType]:
    """Resolve --content and the deprecated --include-text-files into a list of ContentType values."""
    if include_text_files:
        warnings.warn(
            "--include-text-files is deprecated and will be removed in a future version. Use --content all instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if include_text_files or "all" in content:
        return [ContentType.CODE, ContentType.DOCS, ContentType.CONFIG]
    return [ContentType(c) for c in content]


def _load_index(path: str, content: list[ContentType], embedder: str | None = None) -> ZembleIndex:
    """Build an index from a local path or git URL, exiting on FileNotFoundError or a bad embedder spec.

    :param path: Local path or git URL.
    :param content: Content types to index.
    :param embedder: Embedder spec, or None for the environment default.
    :return: The index.
    """
    try:
        return _build_index(path, content, embedder)
    except (FileNotFoundError, EmbedderSpecError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


def _run_search(
    path: str,
    query: str,
    top_k: int,
    content: list[ContentType],
    max_snippet_lines: int | None,
    embedder: str | None = None,
) -> None:
    """Handle the `search` subcommand."""
    index = _load_index(path, content, embedder)
    results = index.search(query, top_k=top_k, max_snippet_lines=max_snippet_lines)
    out = format_results(query, results, max_snippet_lines) if results else {"error": "No results found."}
    print(json.dumps(out))
    _maybe_save_index(index, path)


def _run_find_related(
    path: str,
    file_path: str,
    line: int,
    top_k: int,
    content: list[ContentType],
    max_snippet_lines: int | None,
    embedder: str | None = None,
) -> None:
    """Handle the `find-related` subcommand."""
    index = _load_index(path, content, embedder)
    chunk = resolve_chunk(index.chunks, file_path, line)
    if chunk is None:
        print(f"No chunk found at {file_path}:{line}.", file=sys.stderr)
        sys.exit(1)
    results = index.find_related(chunk, top_k=top_k, max_snippet_lines=max_snippet_lines)
    label = f"Chunks related to {file_path}:{line}"
    out = (
        format_results(label, results, max_snippet_lines)
        if results
        else {"error": f"No related chunks found for {file_path}:{line}."}
    )
    print(json.dumps(out))
    _maybe_save_index(index, path)


def _clear_indexes(cache_folder: Path) -> None:
    """Remove all valid index entries from the cache folder."""
    indexes: set[Path] = set()
    for path in cache_folder.glob("*/index*"):
        if not _SHA_256_REGEX.match(path.parent.name):
            continue
        if PersistencePath.from_path(path).non_existing():
            continue
        indexes.add(path.parent)

    if not indexes:
        print(f"No indexes found to clear in `{cache_folder}`")
    else:
        for index_folder in indexes:
            rmtree(index_folder)
            print(f"Cleared index at `{index_folder}`")


def _clear_savings(cache_folder: Path) -> None:
    """Remove the savings file from the cache folder."""
    path = cache_folder / "savings.jsonl"
    if not path.exists():
        print(f"No savings file found at `{path}`")
    else:
        path.unlink()
        print(f"Cleared savings at `{path}`")


def _clear_orphans(cache_folder: Path) -> None:
    """Remove index entries whose local root_path no longer exists."""
    orphans: dict[Path, str] = {}
    for path in cache_folder.glob("*/index*"):
        if not _SHA_256_REGEX.match(path.parent.name):
            continue
        try:
            with open(path / "metadata.json", encoding="utf-8") as f:
                metadata = json.load(f)
                root_path = metadata.get("root_path") if isinstance(metadata, dict) else None
        except (OSError, json.JSONDecodeError):
            continue
        # Git-URL entries store their temp clone dir as root_path, so only trust entries whose key matches.
        if not isinstance(root_path, str) or not root_path or cache_key(root_path) != path.parent.name:
            continue
        if not Path(root_path).exists():
            orphans[path.parent] = root_path

    if not orphans:
        print("No orphaned indexes found")
    else:
        for index_folder, root_path in orphans.items():
            rmtree(index_folder)
            print(f"Cleared orphaned index for `{root_path}`")


def _run_clear(clear_type: _CLEAR_CHOICE) -> None:
    """Run the `clear` subcommand."""
    cache_folder = resolve_cache_folder()
    if clear_type == "index" or clear_type == "all":
        _clear_indexes(cache_folder)
    if clear_type == "savings" or clear_type == "all":
        _clear_savings(cache_folder)
    if clear_type == "orphans":
        _clear_orphans(cache_folder)


def _cli_main() -> None:
    parser = argparse.ArgumentParser(prog="zemble")
    parser.add_argument("-V", "--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    search_p = sub.add_parser("search", help="Search a codebase.")
    search_p.add_argument("query", help="Natural language or code query.")
    search_p.add_argument("path", nargs="?", default=".", help="Local path or git URL (default: current directory).")
    search_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results (default: 5).")
    search_p.add_argument(
        "--max-snippet-lines",
        type=int,
        default=None,
        metavar="N",
        help="Lines of source per result (default: full chunk). 10 = signature + body, 0 = no code.",
    )
    _add_content_args(search_p)
    _add_embedder_arg(search_p)

    stats_p = sub.add_parser("stats", help="Show what an index contains, including its embedder and dimensions.")
    stats_p.add_argument("path", nargs="?", default=".", help="Local path or git URL (default: current directory).")
    _add_content_args(stats_p)
    _add_embedder_arg(stats_p)

    clear_p = sub.add_parser("clear", help="Clear the index cache.")
    clear_p.add_argument(
        "type",
        choices=["all", "index", "savings", "orphans"],
        help="Type of cache to clear. `orphans` removes indexes whose source path no longer exists.",
    )

    related_p = sub.add_parser("find-related", help="Find code similar to a specific location.")
    related_p.add_argument("file_path", help="File path as shown in search results.")
    related_p.add_argument("line", type=int, help="Line number (1-indexed).")
    related_p.add_argument("path", nargs="?", default=".", help="Local path or git URL (default: current directory).")
    related_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results (default: 5).")
    related_p.add_argument(
        "--max-snippet-lines",
        type=int,
        default=None,
        metavar="N",
        help="Lines of source per result (default: full chunk). 10 = signature + body, 0 = no code.",
    )
    _add_content_args(related_p)
    _add_embedder_arg(related_p)

    sub.add_parser("savings", help="Show token savings and usage stats.")

    add_graph_parser(sub)
    add_dupes_parser(sub)
    add_evidence_parser(sub)

    install_p = sub.add_parser("install", help="Configure zemble across coding agents.")
    uninstall_p = sub.add_parser("uninstall", help="Remove zemble configuration from coding agents.")
    for p, verb in ((install_p, "configure"), (uninstall_p, "remove configuration from")):
        p.add_argument(
            "--agent",
            nargs="+",
            choices=[a.id for a in AGENTS],
            metavar="AGENT",
            help=f"Agent(s) to {verb} non-interactively, e.g. --agent claude pi. Skips prompts.",
        )
        p.add_argument(
            "--type",
            nargs="+",
            choices=[*(t.value for t in IntegrationType), "all"],
            metavar="TYPE",
            help="Integrations to include (mcp, instructions, subagent, or all). Default: all. Requires --agent.",
        )
        p.add_argument(
            "-y",
            "--yes",
            action="store_true",
            help="Skip the confirmation prompt. Combine with --agent for a fully non-interactive run.",
        )

    args = parser.parse_args()

    runner = _SUBCOMMAND_RUNNERS.get(args.command)
    if runner is not None:
        raise SystemExit(runner(args))
    if args.command == "savings":
        print(format_savings_report())
    elif args.command in ("install", "uninstall"):
        if args.type and not args.agent:
            parser.error("--type requires --agent")

        from zemble.installer import run

        integration_ids = None if not args.type or "all" in args.type else [IntegrationType(t) for t in args.type]
        run(args.command, agent_ids=args.agent, integration_ids=integration_ids, yes=args.yes)
    elif args.command == "clear":
        _run_clear(args.type)
    elif args.command == "search":
        _run_search(
            args.path,
            args.query,
            args.top_k,
            _resolve_content(args.content, args.include_text_files),
            args.max_snippet_lines,
            args.embedder,
        )
    elif args.command == "stats":
        _run_stats(args.path, _resolve_content(args.content, args.include_text_files), args.embedder)
    elif args.command == "find-related":
        _run_find_related(
            args.path,
            args.file_path,
            args.line,
            args.top_k,
            _resolve_content(args.content, args.include_text_files),
            args.max_snippet_lines,
            args.embedder,
        )

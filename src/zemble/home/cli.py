"""Command-line surface for `zemble home`.

Lives here rather than in `zemble.cli` so wiring the feature into the top-level
parser is three lines, the same shape the graph and evidence subcommands use.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from zemble.graph.cli import EXIT_NOT_FOUND, ensure_graph
from zemble.graph.provider import SqliteGraphProvider
from zemble.home.answers import DEFAULT_TOP_K, home_payload
from zemble.home.config import ConfigError, HomeConfig
from zemble.index import ZembleIndex
from zemble.types import ContentType

HOME_COMMANDS = ("home",)

#: `home` reads prose as well as code: a design note naming a module is evidence too.
HOME_CONTENT = (ContentType.CODE, ContentType.DOCS)


def add_home_parser(sub: argparse._SubParsersAction) -> None:
    """Register the `home` subcommand."""
    parser = sub.add_parser("home", help="Does this feature exist, and which module should it live in?")
    parser.add_argument("path", help="Workspace directory to search and build the graph for.")
    parser.add_argument("description", help="The feature you are about to build, in your own words.")
    parser.add_argument(
        "-k", "--top-k", type=int, default=DEFAULT_TOP_K, help=f"Code results to weigh (default: {DEFAULT_TOP_K})."
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    from zemble.cli import _add_daemon_arg, _add_embedder_arg

    _add_embedder_arg(parser)
    _add_daemon_arg(parser)


def run_home(args: argparse.Namespace) -> int:
    """Answer one `home` question, from the warm daemon where there is one."""
    from zemble.cli import _via_daemon

    try:
        payload = _via_daemon(args.command, _daemon_args(args), args.no_daemon, getattr(args, "embedder", None))
        if payload is None:
            payload = _in_process(args)
    except ConfigError as error:
        print(str(error), file=sys.stderr)
        return 1
    answer = payload["home"]
    print(json.dumps(answer, indent=2) if args.json else payload["markdown"])
    return 0 if (answer["mechanisms"] or answer["candidates"]) else EXIT_NOT_FOUND


def _daemon_args(args: argparse.Namespace) -> dict[str, Any]:
    """Shape the subcommand as daemon command arguments."""
    return {
        "path": args.path,
        "description": args.description,
        "top_k": args.top_k,
        "content": [item.value for item in HOME_CONTENT],
    }


def _in_process(args: argparse.Namespace) -> dict[str, Any]:
    """Answer in this process, building the index and the graph as needed.

    A sub-directory of an indexed tree searches that tree's index as a view speaking paths
    relative to the sub-directory, so its own config and graph are the ones that match.
    """
    index, _source_key = _load_index(args.path, args.embedder)
    root = args.path
    config = HomeConfig.load(root)
    ensure_graph(root)
    provider = SqliteGraphProvider(root)
    try:
        return home_payload(index, provider, config, args.description, args.top_k)
    finally:
        provider.close()


def _load_index(path: str, embedder: str | None = None) -> tuple[ZembleIndex, str]:
    """Build or load the code-and-docs index answering for a workspace, saving it back to the cache.

    Imported lazily: `zemble.cli` imports this module, so the reverse import can
    only happen once the parser is already built.

    :return: The index, and the root it was built from, which a sub-path request answers from.
    """
    from zemble.cli import _load_index as load
    from zemble.cli import _maybe_save_index

    index, source_key = load(path, list(HOME_CONTENT), embedder)
    _maybe_save_index(index, source_key)
    return index, source_key


__all__ = ["HOME_COMMANDS", "HOME_CONTENT", "add_home_parser", "run_home"]

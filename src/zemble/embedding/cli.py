"""Command-line surface for `zemble embed-status`.

Lives here rather than in `zemble.cli` so wiring it into the top-level parser is three
lines, the same shape the graph, home and dupes subcommands use.
"""

from __future__ import annotations

import argparse
import json
import sys

from zemble.embedding.preflight import embed_status
from zemble.embedding.registry import EmbedderSpecError

EMBED_STATUS_COMMANDS = ("embed-status",)

EXIT_ERROR = 1


def add_embed_status_parser(sub: argparse._SubParsersAction) -> None:
    """Register the `embed-status` subcommand."""
    from zemble.cli import _add_content_args, _add_embedder_arg, add_root_arg

    parser = sub.add_parser(
        "embed-status",
        help="Show what indexing a tree would embed and what it would cost. Embeds nothing.",
    )
    add_root_arg(parser, nargs="?", default=".", help="Local directory (default: current directory).")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        metavar="PATTERN",
        help="Gitignore-style patterns a build would skip, so the report answers for the build you would run.",
    )
    _add_content_args(parser)
    _add_embedder_arg(parser)


def run_embed_status(args: argparse.Namespace) -> int:
    """Run `zemble embed-status` and return its exit code."""
    from zemble.cli import _resolve_content

    content = _resolve_content(args.content, args.include_text_files)
    try:
        status = embed_status(args.path, content, args.embedder, exclude=tuple(args.exclude))
    except (FileNotFoundError, EmbedderSpecError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_ERROR
    print(json.dumps(status.to_dict()) if args.json else status.to_text())
    return 0


__all__ = ["EMBED_STATUS_COMMANDS", "add_embed_status_parser", "run_embed_status"]

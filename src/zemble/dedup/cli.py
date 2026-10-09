"""Command-line surface for duplication detection.

Lives here rather than in `zemble.cli` so wiring the feature into the top-level
parser is three lines, exactly like `zemble.graph.cli`.
"""

from __future__ import annotations

import argparse
import json

from zemble.dedup.baseline import load_baseline, save_baseline
from zemble.dedup.detect import DupeOptions, find_duplication
from zemble.dedup.languages import supported_extensions, supported_languages
from zemble.dedup.model import CloneKind, Lane
from zemble.dedup.report import baseline_diff_json, format_baseline_diff, format_report, report_json
from zemble.embedding.registry import EmbedderSpecError
from zemble.refusal import Refused

_KIND_CHOICES = [kind.value for kind in CloneKind] + ["all"]
_LANE_CHOICES = [lane.value for lane in Lane] + ["all"]

#: What a run that could not answer exits with, whichever shape it answered in.
EXIT_ERROR = 1


def add_dupes_parser(sub: argparse._SubParsersAction) -> None:
    """Register the `dupes` subcommand on the main parser."""
    from zemble.cli import _add_confirm_arg, add_root_arg

    languages = supported_languages()
    parser = sub.add_parser(
        "dupes",
        help="Report duplicated code: clone classes, idioms, re-implementations, vocabularies "
        f"({len(languages)} languages).",
        epilog=f"Languages: {', '.join(languages)}. File types: {', '.join(supported_extensions())}.",
    )
    add_root_arg(parser, nargs="?", default=".", help="Workspace directory (default: current directory).")
    parser.add_argument(
        "--kind",
        default="exact,renamed",
        help="Channels, comma-separated: exact, renamed, logic, holed, idiom, reimplements, vocabulary, all, "
        "or architectural candidates (default: exact,renamed).",
    )
    parser.add_argument("--limit", type=int, default=25, help="Clone classes printed per section (default: 25).")
    parser.add_argument(
        "--min-files", type=int, default=1, help="Only report classes spanning at least N files (default: 1)."
    )
    parser.add_argument(
        "--min-tokens", type=int, default=30, help="Smallest unit, in tokens, that may form a class (default: 30)."
    )
    parser.add_argument(
        "--min-statements",
        type=int,
        default=6,
        help="Smallest window of consecutive statements compared inside a body (default: 6).",
    )
    parser.add_argument("--no-windows", action="store_true", help="Compare whole bodies only, no statement windows.")
    parser.add_argument(
        "--logic-threshold", type=float, default=0.92, help="Cosine similarity a logic candidate needs (default: 0.92)."
    )
    parser.add_argument(
        "--logic-top-k", type=int, default=10, help="Embedding neighbours considered per unit (default: 10)."
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=None,
        metavar="PATH",
        help="Restrict the scan to these paths, relative to the workspace directory (or absolute).",
    )
    parser.add_argument(
        "--focus",
        nargs="+",
        default=None,
        metavar="PATH",
        help="Report only the classes with a member under these paths, compared against everything scanned; "
        "the other files are read from a per-root unit index, so the cost follows the focus, not the workspace.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        metavar="GLOB",
        help="Gitignore-style pattern, relative to the root, dropped before parsing (repeatable).",
    )
    parser.add_argument(
        "--lane",
        default="all",
        choices=_LANE_CHOICES,
        help="Report only one lane: production, mixed, test, or all (default: all).",
    )
    parser.add_argument("--brief", action="store_true", help="Header plus one line per class, nothing else.")
    parser.add_argument(
        "--show-suppressed", action="store_true", help="Also print the classes the ignore file took out."
    )
    parser.add_argument(
        "--baseline", default=None, metavar="FILE", help="Report resolved/remaining/new against a saved baseline."
    )
    parser.add_argument(
        "--save-baseline", default=None, metavar="FILE", help="Write this run's clone class keys as a baseline."
    )
    parser.add_argument("--embedder", default=None, metavar="SPEC", help="Embedder spec used by `--kind logic`.")
    parser.add_argument("--jobs", type=int, default=None, help="Extraction worker processes (default: up to 8).")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    # `--kind logic` buys one vector per candidate body, so this run can be refused by the bill
    # guard, whose refusal ends "--yes on the CLI". A flag a refusal advertises has to exist.
    _add_confirm_arg(parser)


def _kinds(raw: str) -> tuple[CloneKind, ...]:
    """Parse the --kind flag into clone kinds.

    :param raw: The raw flag value.
    :return: The selected kinds, in declaration order.
    :raises SystemExit: If a name is not a known kind.
    """
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if "all" in names:
        return tuple(CloneKind)
    unknown = [name for name in names if name not in _KIND_CHOICES]
    if unknown or not names:
        raise SystemExit(f"Unknown --kind {raw!r}; expected any of {', '.join(_KIND_CHOICES)}")
    selected = {CloneKind(name) for name in names}
    return tuple(kind for kind in CloneKind if kind in selected)


def _fail(message: str, as_json: bool) -> SystemExit:
    """Return the exit that answers a run nobody can complete, in the shape the caller asked for.

    A machine consumer of ``--json`` cannot parse a sentence: a refusal is the ANSWER to a
    ``--kind logic`` run, so it comes back as JSON when JSON was asked for.

    :param message: The refusal or error, already worded for a reader.
    :param as_json: Whether the caller asked for machine-readable output.
    :return: The exit to raise.
    """
    if not as_json:
        return SystemExit(message)
    print(json.dumps({"error": message}, indent=2))
    return SystemExit(EXIT_ERROR)


def _architectural_cli(args: argparse.Namespace) -> int:
    """Use the daemon's prepared graph and keep candidate output distinct from clones."""
    from zemble.daemon.client import call

    if args.baseline or args.save_baseline or args.lane != "all" or args.focus:
        raise _fail("architectural candidates do not use literal baselines, lane filtering or --focus", args.json)
    payload = call(
        "architectural",
        {
            "path": args.path,
            "paths": args.paths,
            "exclude": args.exclude,
            "limit": args.limit,
            "min_files": args.min_files,
        },
    )
    print(
        json.dumps(payload, indent=2)
        if args.json
        else "Architectural candidates (behavioral review required):\n"
        + "\n".join(f"{candidate['key']}: {candidate['reason']}" for candidate in payload.get("candidates", []))
    )
    return 0


def run_dupes(args: argparse.Namespace) -> int:
    """Run `zemble dupes` and return its exit code, which is 0 however much it finds."""
    if args.kind == "architectural":
        return _architectural_cli(args)
    if args.focus and (args.baseline or args.save_baseline):
        # A focused run leaves out every class without a focus member: a baseline would read them as resolved.
        raise _fail("--focus reports part of the classes, so it takes no --baseline or --save-baseline", args.json)
    options = DupeOptions(
        kinds=_kinds(args.kind),
        min_tokens=args.min_tokens,
        min_statements=args.min_statements,
        windows=not args.no_windows,
        min_files=args.min_files,
        logic_threshold=args.logic_threshold,
        logic_top_k=args.logic_top_k,
        embedder=args.embedder,
        paths=tuple(args.paths or ()),
        focus=tuple(args.focus or ()),
        exclude=tuple(args.exclude or ()),
        lane=None if args.lane == "all" else Lane(args.lane),
        jobs=args.jobs,
    )
    try:
        report = find_duplication(args.path, options)
    # A deliberate refusal - the spending budget of the logic lane - is the answer, printed the
    # way every other surface prints one, never a traceback out of `main`. The embedder spec is
    # here for the same reason: `zemble.cli` catches it for every other lane that resolves one.
    except (Refused, FileNotFoundError, EmbedderSpecError) as error:
        raise _fail(str(error), args.json) from None
    baseline = None
    if args.baseline:
        try:
            baseline = load_baseline(args.baseline)
        except ValueError as error:
            raise _fail(str(error), args.json) from None
    if args.save_baseline:
        written = save_baseline(args.save_baseline, report)
        print(f"Wrote {len(report.classes)} clone class key(s) to {written}")
    if args.json:
        payload = baseline_diff_json(report, baseline, args.limit) if baseline else report_json(report, args.limit)
        print(json.dumps(payload, indent=2))
    elif baseline:
        print(format_baseline_diff(report, baseline, limit=args.limit), end="")
    else:
        print(
            format_report(report, args.limit, brief=args.brief, show_suppressed=args.show_suppressed),
            end="",
        )
    return 0

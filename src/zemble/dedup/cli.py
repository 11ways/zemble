"""Command-line surface for duplication detection.

Lives here rather than in `zemble.cli` so wiring the feature into the top-level
parser is three lines, exactly like `zemble.graph.cli`.
"""

from __future__ import annotations

import argparse
import json

from zemble.dedup.baseline import load_baseline, save_baseline
from zemble.dedup.detect import DupeOptions, find_duplication
from zemble.dedup.model import CloneKind, Lane
from zemble.dedup.report import format_baseline_diff, format_report, report_json

_KIND_CHOICES = [kind.value for kind in CloneKind] + ["all"]
_LANE_CHOICES = [lane.value for lane in Lane] + ["all"]


def add_dupes_parser(sub: argparse._SubParsersAction) -> None:
    """Register the `dupes` subcommand on the main parser."""
    parser = sub.add_parser("dupes", help="Report duplicated Java code (exact, alpha-renamed, logic clone classes).")
    parser.add_argument("path", nargs="?", default=".", help="Workspace directory (default: current directory).")
    parser.add_argument(
        "--kind",
        default="exact,renamed",
        help="Which kinds to report: exact, renamed, logic, all, or a comma-separated list (default: exact,renamed).",
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
    parser.add_argument("--paths", nargs="+", default=None, metavar="PATH", help="Restrict the scan to these paths.")
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


def run_dupes(args: argparse.Namespace) -> int:
    """Run `zemble dupes` and return its exit code, which is 0 however much it finds."""
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
        exclude=tuple(args.exclude or ()),
        lane=None if args.lane == "all" else Lane(args.lane),
        jobs=args.jobs,
    )
    try:
        report = find_duplication(args.path, options)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None
    if args.save_baseline:
        written = save_baseline(args.save_baseline, report)
        print(f"Wrote {len(report.classes)} clone class key(s) to {written}")
    if args.json:
        print(json.dumps(report_json(report, args.limit), indent=2))
    elif args.baseline:
        try:
            baseline = load_baseline(args.baseline)
        except ValueError as error:
            raise SystemExit(str(error)) from None
        print(format_baseline_diff(report, baseline, limit=args.limit), end="")
    else:
        print(
            format_report(report, args.limit, brief=args.brief, show_suppressed=args.show_suppressed),
            end="",
        )
    return 0

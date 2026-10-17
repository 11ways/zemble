"""One `health` answer from an open graph and the workspace config, shared by the CLI, MCP and daemon."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from zemble.baseline import load_for_diff
from zemble.graph.provider import AnyProvider, SubtreeGraphProvider
from zemble.health.baseline import BASELINE_RELATIVE_PATH, diff_baseline, load_baseline, save_baseline
from zemble.health.collect import collect
from zemble.health.git import changed_lines
from zemble.health.history import DEFAULT_WINDOW_DAYS
from zemble.health.model import Signal, Tier
from zemble.health.report import DEFAULT_LIMIT, HealthReport, build_report
from zemble.health.signals import measure
from zemble.home.config import HomeConfig

#: Why a whole-tree run leaves the duplication tier out unless asked; docs/health.md has the measurement.
COPYING_OPT_IN = (
    "copying not measured: over a whole tree the duplication tier costs a parse of every file "
    "(64-88 s on the javaweb workspace), so it runs there only with --copying; --since, --paths "
    "or a sub-directory turn it on"
)


def parse_signals(names: Iterable[str] | None) -> tuple[Signal, ...]:
    """Every signal for None or nothing, else the named ones in declaration order.

    :raises ValueError: If a name is not a signal; the message lists the valid ones.
    """
    wanted = {name.strip().lower() for name in names or () if name.strip()}
    if not wanted:
        return tuple(Signal)
    known = {signal.value for signal in Signal}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(f"unknown signal(s): {', '.join(unknown)}; known: {', '.join(sorted(known))}")
    return tuple(signal for signal in Signal if signal.value in wanted)


@dataclass(frozen=True)
class HealthRequest:
    """One health question as every surface asks it; the field names are the daemon's argument names."""

    signals: tuple[str, ...] | None = None
    paths: tuple[str, ...] = ()
    limit: int = DEFAULT_LIMIT
    #: A git ref: list only the findings touching a line changed since it.
    since: str | None = None
    #: Whether to measure the duplication tier; None measures it only for a bounded scope (the cost rule).
    copying: bool | None = None
    #: A baseline file to diff the listed findings against; read before `save_baseline` writes.
    baseline: str | None = None
    #: Where to write every finding of the scope as a baseline.
    save_baseline: str | None = None
    #: Report a missing or unreadable `baseline` as a note instead of refusing.
    baseline_optional: bool = False
    #: Whether to measure the history axis from git.
    history: bool = True
    #: Days of git history the history axis reads.
    history_window: int = DEFAULT_WINDOW_DAYS

    @classmethod
    def from_args(cls, args: Mapping[str, Any]) -> HealthRequest:
        """Read a request from daemon arguments (JSON: lists, absent keys); `path` and unknown keys are ignored."""
        signals = args.get("signals")
        return cls(
            signals=tuple(signals) if signals is not None else None,
            paths=tuple(args.get("paths") or ()),
            limit=int(args.get("limit", DEFAULT_LIMIT)),
            since=args.get("since"),
            copying=args.get("copying"),
            baseline=args.get("baseline"),
            save_baseline=args.get("save_baseline"),
            baseline_optional=bool(args.get("baseline_optional", False)),
            history=bool(args.get("history", True)),
            history_window=int(args.get("history_window", DEFAULT_WINDOW_DAYS)),
        )

    def to_args(self, path: str) -> dict[str, Any]:
        """The daemon arguments asking this request about an absolute path."""
        return {"path": path, **asdict(self)}


def health_payload(provider: AnyProvider, config: HomeConfig, request: HealthRequest | None = None) -> dict[str, Any]:
    """Measure, score and render one report: `{"health": <json>, "markdown": <text>}`.

    :raises ValueError: If a signal name is unknown, `since` cannot be read, a required baseline cannot, or
        the history window is not a positive number of days.
    """
    request = request or HealthRequest()
    selected = parse_signals(request.signals)
    if request.since is not None and request.save_baseline is not None:
        # A --since run lists part of the findings and judges the duplication tier on part of the files.
        raise ValueError("--since reports a delta, so it takes no --save-baseline; save one without --since")
    if request.history_window < 1:
        raise ValueError(f"--history-window takes a positive number of days, got {request.history_window}")
    changes = changed_lines(os.path.realpath(provider.path), request.since) if request.since is not None else None
    selected, copying, notes = _copying_plan(provider, request, selected)
    history = any(signal.facts.tier is Tier.HISTORY for signal in selected)
    if history and not request.history:
        selected, history = _without(selected, Tier.HISTORY), False
        notes.append("history not measured: turned off")
    window = request.history_window if history else None
    facts = collect(provider, config, request.paths, copying=copying, changed=changes, history_window=window)
    if facts.copying is not None:
        notes.extend(facts.copying.notes)
    excluded = facts.excluded_note
    if excluded is not None:
        notes.append(excluded)
    if facts.history is not None and facts.history.unmeasured is not None:
        selected = _without(selected, Tier.HISTORY)
        notes.append(facts.history.unmeasured)
    report = build_report(
        facts.root,
        str(config.source) if config.source is not None else None,
        measure(facts, selected),
        facts.paths,
        since=changes,
        copying_focus=facts.copying.focus if facts.copying is not None else None,
        notes=tuple(notes),
    )
    report = _with_baseline(report, request)
    return {"health": report.to_dict(request.limit), "markdown": report.markdown(request.limit)}


def _copying_plan(
    provider: AnyProvider, request: HealthRequest, selected: tuple[Signal, ...]
) -> tuple[tuple[Signal, ...], bool, list[str]]:
    """Whether the duplication tier runs (the cost rule), the signals left to measure, and why any were dropped."""
    wanted = any(signal.facts.tier is Tier.DUPLICATION for signal in selected)
    bounded = request.since is not None or bool(request.paths) or isinstance(provider, SubtreeGraphProvider)
    if not wanted or (bounded if request.copying is None else request.copying):
        return selected, wanted, []
    kept = _without(selected, Tier.DUPLICATION)
    return kept, False, [COPYING_OPT_IN if request.copying is None else "copying not measured: turned off"]


def _without(selected: tuple[Signal, ...], tier: Tier) -> tuple[Signal, ...]:
    """The selected signals that do not rest on one tier, for a run that leaves that tier out."""
    return tuple(signal for signal in selected if signal.facts.tier is not tier)


def _with_baseline(report: HealthReport, request: HealthRequest) -> HealthReport:
    """Diff the report against the requested baseline, then save the new one; both are noted."""
    if request.baseline is not None:
        if request.baseline_optional:
            loaded, notes = load_for_diff(Path(request.baseline), BASELINE_RELATIVE_PATH, load_baseline)
        else:
            loaded, notes = load_baseline(request.baseline), []
        difference = diff_baseline(report, loaded) if loaded is not None else None
        report = replace(report, notes=(*report.notes, *notes), baseline=difference)
    if request.save_baseline is not None:
        written = save_baseline(request.save_baseline, report)
        wrote = f"baseline: wrote {len(report.findings)} finding key(s) to {written}"
        report = replace(report, notes=(*report.notes, wrote))
    return report


__all__ = ["COPYING_OPT_IN", "HealthRequest", "health_payload", "parse_signals"]

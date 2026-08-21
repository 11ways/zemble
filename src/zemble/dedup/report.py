"""Rendering a duplication report, in the shape `zenit-dev duplication` prints."""

from __future__ import annotations

from collections.abc import Sequence

from zemble.dedup.baseline import Baseline, BaselineDiff, diff_baseline
from zemble.dedup.homes import HomeVerdict
from zemble.dedup.ignore import IGNORE_RELATIVE_PATH
from zemble.dedup.model import CloneClass, CloneKind, DupeReport, Lane

_SECTION_TITLES = {
    CloneKind.EXACT: "EXACT",
    CloneKind.RENAMED: "RENAMED (alpha-renamed: only declared locals differ)",
    CloneKind.LOGIC: "LOGIC (embedding candidates that passed the structural check)",
}

_LANE_TITLES = {
    Lane.PRODUCTION: "PRODUCTION (every copy outside a test source set)",
    Lane.MIXED: "MIXED (production and test copies of the same code)",
    Lane.TEST: "TEST (test scaffolding only)",
}


def _header(report: DupeReport) -> str:
    """Build the one-line summary above the sections."""
    return (
        f"Analyzed {report.analyzed_files} file(s), {report.units} unit(s) "
        f"({report.body_units} bodies, min {report.min_tokens} tokens, "
        f"min {report.min_statements} statements per window), "
        f"{len(report.classes)} clone class(es) in {report.elapsed_seconds:.1f}s"
    )


def scan_notes(report: DupeReport) -> list[str]:
    """What the scan itself is worth saying: nothing walked, and files that would not parse.

    A run that walked nothing must never render as "No duplication found."; a missing grammar
    or an unreadable file must not read as a clean result either.

    :param report: The report to describe.
    :return: The notes, in the order both surfaces print them.
    """
    notes: list[str] = []
    if report.analyzed_files == 0:
        supported = ", ".join(report.supported_extensions) or "none"
        notes.append(
            f"Scanned 0 supported file(s) under {report.root} (supported: {supported}). "
            "Nothing analyzed: check --paths/--exclude/ignore files."
        )
    if report.failed_files:
        shown = ", ".join(report.failed_examples)
        more = "" if report.failed_files <= len(report.failed_examples) else ", ..."
        notes.append(f"extraction failed for {report.failed_files} file(s): {shown}{more}")
    return notes


def _head_line(index: int, clone: CloneClass, *, with_kind: bool) -> str:
    """Render a clone class's one-line summary."""
    head = clone.members[0]
    label = f"{clone.kind.value} {clone.lane.value}  " if with_kind else ""
    return (
        f"#{index}  {label}{len(clone.members)} copies x {clone.tokens} tokens (score {clone.score})  "
        f"root: {head.kind} {head.name}  files: {clone.files}  key: {clone.key}"
    )


def _class_lines(index: int, clone: CloneClass, verdict: HomeVerdict | None = None) -> list[str]:
    """Render one clone class the way the reference report does."""
    lines = [_head_line(index, clone, with_kind=False)]
    lines.extend(f"    {member.location}" for member in clone.members)
    if verdict is not None:
        head, *rest = verdict.describe_lines()
        lines.append(f"    home: {head}")
        # Continuation lines align under the text, not under the `home:` label.
        lines.extend(f"          {line}" for line in rest)
    lines.extend(f"    reason: {reason}" for reason in clone.wire_reasons)
    return lines


def _preamble(report: DupeReport) -> list[str]:
    """The header, the run's notes and any ignore-file violation."""
    lines = [_header(report)]
    lines.extend(f"  note: {note}" for note in scan_notes(report))
    lines.extend(f"  note: {note}" for note in report.notes)
    lines.extend(f"  ignore-file violation: {problem}" for problem in report.ignore_problems)
    return lines


def _trailer(report: DupeReport, show_suppressed: bool) -> list[str]:
    """The suppressed count, and the suppressed classes themselves when asked for."""
    if not report.suppressed:
        return []
    lines = ["", f"suppressed: {len(report.suppressed)} (justified in {IGNORE_RELATIVE_PATH})"]
    if show_suppressed:
        lines.append("")
        for index, clone in enumerate(report.suppressed, start=1):
            lines.append(_head_line(index, clone, with_kind=True))
    return lines


def _brief_lines(classes: Sequence[CloneClass]) -> list[str]:
    """Render every class as one line, lane-major, for a report that will be piped."""
    return [_head_line(index, clone, with_kind=True) for index, clone in enumerate(_ordered(classes), start=1)]


def _ordered(classes: Sequence[CloneClass]) -> list[CloneClass]:
    """Order classes the way the sections print them: lane first, then kind, then rank."""
    lanes = list(Lane)
    kinds = list(CloneKind)
    return sorted(classes, key=lambda clone: (lanes.index(clone.lane), kinds.index(clone.kind), -clone.score))


def format_report(report: DupeReport, limit: int = 25, *, brief: bool = False, show_suppressed: bool = False) -> str:
    """Render a full report as text.

    Sections are lane-major: production duplication first, then the mixed classes, then the
    test-only scaffolding, so copied fixtures can never outrank a production finding.

    :param report: The report to print.
    :param limit: Clone classes printed per section.
    :param brief: Print one line per class, with no members and no reasons.
    :param show_suppressed: Also print the classes the ignore file took out.
    :return: The report text.
    """
    lines = _preamble(report)
    if brief:
        lines.extend(_brief_lines(report.classes))
        lines.extend(_trailer(report, show_suppressed))
        return "\n".join(lines).rstrip() + "\n"
    for lane in Lane:
        if not any(clone.lane is lane for clone in report.classes):
            continue
        lines.extend(["", f"== {_LANE_TITLES[lane]} =="])
        for kind in CloneKind:
            classes = report.section(lane, kind)
            if not classes:
                continue
            lines.extend(["", f"-- {_SECTION_TITLES[kind]} --"])
            if len(classes) > limit:
                lines.append(f"  showing the top {limit} of {len(classes)}")
            lines.append("")
            for index, clone in enumerate(classes[:limit], start=1):
                lines.extend(_class_lines(index, clone, report.homes.get(clone.key)))
                lines.append("")
    if not report.classes and report.analyzed_files:
        lines.extend(["", "No duplication found."])
    lines.extend(_trailer(report, show_suppressed))
    return "\n".join(lines).rstrip() + "\n"


def format_baseline_diff(report: DupeReport, baseline: Baseline, *, limit: int = 25) -> str:
    """Render a run against a baseline: what is gone, what is left and what is new.

    :param report: The run in hand.
    :param baseline: The saved run to compare against.
    :param limit: Classes printed per section.
    :return: The diff text.
    """
    difference: BaselineDiff = diff_baseline(report, baseline)
    lines = _preamble(report)
    lines.append(
        f"  baseline: {len(baseline.entries)} class(es); {len(difference.resolved)} resolved, "
        f"{len(difference.changed)} changed, {len(difference.remaining)} remaining, {len(difference.new)} new"
    )
    lines.extend(["", "== RESOLVED (in the baseline, gone now) =="])
    if not difference.resolved:
        lines.append("  none")
    for index, entry in enumerate(difference.resolved[:limit], start=1):
        lines.append(
            f"#{index}  {entry.kind} {entry.lane}  {entry.copies} copies (score {entry.score})  "
            f"was: {entry.root_symbol}  key: {entry.key}"
        )
    lines.extend(["", "== CHANGED (same files, re-keyed by an edit) =="])
    if not difference.changed:
        lines.append("  none")
    for index, change in enumerate(difference.changed[:limit], start=1):
        head = change.now.members[0]
        delta = change.score_delta
        lines.append(
            f"#{index}  {change.now.kind.value} {change.now.lane.value}  "
            f"{change.was.copies} -> {len(change.now.members)} copies, "
            f"score {change.was.score} -> {change.now.score} ({'+' if delta >= 0 else ''}{delta})  "
            f"root: {head.kind} {head.name}  key: {change.was.key} -> {change.now.key}"
        )
    for title, classes in (("REMAINING (in the baseline, still here)", difference.remaining), ("NEW", difference.new)):
        lines.extend(["", f"== {title} =="])
        if not classes:
            lines.append("  none")
        for index, clone in enumerate(_ordered(classes)[:limit], start=1):
            lines.append(_head_line(index, clone, with_kind=True))
    lines.extend(_trailer(report, False))
    return "\n".join(lines).rstrip() + "\n"


def baseline_diff_json(report: DupeReport, baseline: Baseline, limit: int = 25) -> dict[str, object]:
    """Render a baseline diff as JSON, each bucket capped at the same limit as the text form.

    :param report: The run in hand.
    :param baseline: The saved run to compare against.
    :param limit: Entries kept per bucket.
    :return: A JSON-serializable dict.
    """
    difference = diff_baseline(report, baseline)
    return {
        "root": report.root,
        "baseline_classes": len(baseline.entries),
        "counts": {
            "resolved": len(difference.resolved),
            "changed": len(difference.changed),
            "remaining": len(difference.remaining),
            "new": len(difference.new),
        },
        "resolved": [entry.to_dict() for entry in difference.resolved[:limit]],
        "changed": [
            {"was": change.was.to_dict(), "now": report.clone_dict(change.now), "score_delta": change.score_delta}
            for change in difference.changed[:limit]
        ],
        "remaining": [report.clone_dict(clone) for clone in _ordered(difference.remaining)[:limit]],
        "new": [report.clone_dict(clone) for clone in _ordered(difference.new)[:limit]],
        "ignore_problems": list(report.ignore_problems),
        "suppressed": len(report.suppressed),
        "analyzed_files": report.analyzed_files,
        "failed_files": report.failed_files,
        "supported_extensions": list(report.supported_extensions),
        "notes": scan_notes(report) + list(report.notes),
    }


def report_json(report: DupeReport, limit: int = 25) -> dict[str, object]:
    """Render a report as JSON, each section capped at the same limit as the text form.

    :param report: The report to render.
    :param limit: Clone classes kept per section.
    :return: A JSON-serializable dict.
    """
    payload = report.to_dict()
    payload["notes"] = scan_notes(report) + list(report.notes)
    capped = []
    for lane in Lane:
        for kind in CloneKind:
            capped.extend(report.clone_dict(clone) for clone in report.section(lane, kind)[:limit])
    payload["classes"] = capped
    payload["class_counts"] = {kind.value: len(report.of_kind(kind)) for kind in CloneKind}
    payload["lane_counts"] = {
        lane.value: len([clone for clone in report.classes if clone.lane is lane]) for lane in Lane
    }
    return payload

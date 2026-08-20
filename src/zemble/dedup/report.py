"""Rendering a duplication report, in the shape `zenit-dev duplication` prints."""

from __future__ import annotations

from zemble.dedup.model import CloneClass, CloneKind, DupeReport

_SECTION_TITLES = {
    CloneKind.EXACT: "EXACT",
    CloneKind.RENAMED: "RENAMED (alpha-renamed: only declared locals differ)",
    CloneKind.LOGIC: "LOGIC (embedding candidates that passed the structural check)",
}


def _header(report: DupeReport) -> str:
    """Build the one-line summary above the sections."""
    return (
        f"Analyzed {report.analyzed_files} file(s), {report.units} unit(s) "
        f"({report.body_units} bodies, min {report.min_tokens} tokens, "
        f"min {report.min_statements} statements per window), "
        f"{len(report.classes)} clone class(es) in {report.elapsed_seconds:.1f}s"
    )


def _class_lines(index: int, clone: CloneClass) -> list[str]:
    """Render one clone class the way the reference report does."""
    head = clone.members[0]
    lines = [
        f"#{index}  {len(clone.members)} copies x {clone.tokens} tokens (score {clone.score})  "
        f"root: {head.kind} {head.name}  files: {clone.files}"
    ]
    lines.extend(f"    {member.location}" for member in clone.members)
    lines.extend(f"    reason: {reason}" for reason in clone.reasons)
    return lines


def format_report(report: DupeReport, limit: int = 25) -> str:
    """Render a full report as text.

    :param report: The report to print.
    :param limit: Clone classes printed per section.
    :return: The report text.
    """
    lines = [_header(report)]
    for note in report.notes:
        lines.append(f"  note: {note}")
    for kind in CloneKind:
        classes = report.of_kind(kind)
        if not classes:
            continue
        lines.extend(["", f"== {_SECTION_TITLES[kind]} =="])
        if len(classes) > limit:
            lines.append(f"  showing the top {limit} of {len(classes)}")
        lines.append("")
        for index, clone in enumerate(classes[:limit], start=1):
            lines.extend(_class_lines(index, clone))
            lines.append("")
    if len(lines) == 1:
        lines.append("")
        lines.append("No duplication found.")
    return "\n".join(lines).rstrip() + "\n"


def report_json(report: DupeReport, limit: int = 25) -> dict[str, object]:
    """Render a report as JSON, each section capped at the same limit as the text form.

    :param report: The report to render.
    :param limit: Clone classes kept per section.
    :return: A JSON-serializable dict.
    """
    payload = report.to_dict()
    capped = []
    for kind in CloneKind:
        capped.extend(clone.to_dict() for clone in report.of_kind(kind)[:limit])
    payload["classes"] = capped
    payload["class_counts"] = {kind.value: len(report.of_kind(kind)) for kind in CloneKind}
    return payload

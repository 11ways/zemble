"""One `HealthReport` and its two renderings, markdown and JSON, from the same data."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from zemble.health.collect import within
from zemble.health.git import ChangedLines
from zemble.health.model import Axis, Finding, Signal, Tier
from zemble.health.score import SCORE_DISCLAIMER, AxisScore, axis_scores, composite
from zemble.health.signals import Sample

if TYPE_CHECKING:  # pragma: no cover
    from zemble.health.baseline import BaselineDiff

#: Findings a report lists unless asked otherwise.
DEFAULT_LIMIT = 25


@dataclass(frozen=True)
class HealthReport:
    """The score, its axes and every finding, ranked; `limit` only cuts the rendering."""

    root: str
    #: The `home.toml` the layering axis read, or None in generic mode.
    config_source: str | None
    signals: tuple[Signal, ...]
    paths: tuple[str, ...]
    axes: tuple[AxisScore, ...]
    composite: float | None
    #: Every finding of the scope; `listed` is what the report shows.
    findings: tuple[Finding, ...]
    notes: tuple[str, ...]
    #: The lines a `--since` run lists findings for, or None.
    since: ChangedLines | None = None
    #: The files a focused duplication run judged, or None when it judged the whole scope.
    copying_focus: frozenset[str] | None = None
    #: Signals measured but left out of the score: their population is only part of the scope.
    unscored: tuple[Signal, ...] = ()
    baseline: BaselineDiff | None = None

    @property
    def listed(self) -> tuple[Finding, ...]:
        """The findings the report shows: those touching a changed line in a `--since` run, else all."""
        if self.since is None:
            return self.findings
        since = self.since
        return tuple(finding for finding in self.findings if finding.touched(since))

    def judges(self, signal_name: str, file_path: str) -> bool:
        """Whether this run measured a signal over a file, so the absence of a finding there means something.

        A name this version does not know is never judged (fails closed).
        """
        signal = next((member for member in self.signals if member.value == signal_name), None)
        if signal is None or not within(file_path, self.paths):
            return False
        focus = self.copying_focus
        return signal.facts.tier is not Tier.DUPLICATION or focus is None or file_path in focus

    def counts(self) -> dict[str, int]:
        """Listed findings per selected signal, zeros included."""
        counted = {signal.value: 0 for signal in self.signals}
        for finding in self.listed:
            counted[finding.signal.value] += 1
        return counted

    def to_dict(self, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
        """Render the report as JSON-ready data, the findings capped at `limit`."""
        listed = self.listed
        shown = folded(listed)[:limit]
        baseline = self.baseline
        findings = []
        for finding, members in shown:
            rendered = finding.to_dict()
            rendered["folded"] = [member.key for member in members]
            if baseline is not None:
                rendered["baseline"] = baseline.status(finding)
            findings.append(rendered)
        return {
            "root": self.root,
            "config": self.config_source,
            "signals": [signal.value for signal in self.signals],
            "paths": list(self.paths),
            "score": round(self.composite, 1) if self.composite is not None else None,
            "score_note": SCORE_DISCLAIMER,
            "axes": [axis.to_dict() for axis in self.axes],
            "unscored": [signal.value for signal in self.unscored],
            "since": None
            if self.since is None
            else {"ref": self.since.ref, "files": len(self.since.files), "scope_total": len(self.findings)},
            "counts": self.counts(),
            "total": len(listed),
            "truncated": len(shown) < len(folded(listed)),
            "findings": findings,
            "baseline": None if baseline is None else _baseline_dict(baseline, limit),
            "notes": list(self.notes),
        }

    def markdown(self, limit: int = DEFAULT_LIMIT) -> str:
        """Render the report as the text the CLI prints and the MCP tool returns."""
        score = "not measured" if self.composite is None else f"{self.composite:.1f} / 100"
        lines = [f"# Health: {self.root}", "", f"Score: {score} ({SCORE_DISCLAIMER})", ""]
        lines.extend(_axis_line(axis, self.unscored) for axis in self.axes)
        lines.append("")
        listed = self.listed
        if self.since is not None:
            lines.append(
                f"Since {self.since.ref}: {len(listed)} finding(s) touch {len(self.since.files)} changed file(s); "
                f"the score is the whole scope's ({len(self.findings)} finding(s))"
            )
        counts = self.counts()
        lines.append("Findings per signal: " + ", ".join(f"{name} {count}" for name, count in counts.items()))
        lines.extend(f"Note: {note}" for note in self.notes)
        lines.append("")
        if self.baseline is not None:
            lines.extend(_baseline_lines(self.baseline, limit))
            return "\n".join(lines).rstrip() + "\n"
        shown = folded(listed)[:limit]
        if not shown:
            lines.append("No findings.")
            return "\n".join(lines) + "\n"
        cut = f", truncated to {len(shown)}" if len(shown) < len(folded(listed)) else ""
        lines.append(f"## Findings ({len(listed)} total{cut}; ranked by percentile, then severity)")
        lines.append("")
        lines.extend(_finding_line(finding, len(members)) for finding, members in shown)
        return "\n".join(lines) + "\n"


def _baseline_lines(difference: BaselineDiff, limit: int) -> list[str]:
    """The baseline sections, in the shape `zemble dupes --baseline` prints them."""
    lines = [
        f"Baseline: {difference.entries} finding(s); {len(difference.resolved)} resolved, "
        f"{len(difference.remaining)} remaining, {len(difference.new)} new"
    ]
    lines.extend(["", "== RESOLVED (in the baseline, gone now) =="])
    if not difference.resolved:
        lines.append("  none")
    for index, entry in enumerate(difference.resolved[:limit], start=1):
        name = f"  {entry.name}" if entry.name else ""
        lines.append(f"#{index}  {entry.signal}  was: {entry.location}{name}  key: {entry.key}")
    for title, findings in (
        ("REMAINING (in the baseline, still here)", difference.remaining),
        ("NEW", difference.new),
    ):
        lines.extend(["", f"== {title} =="])
        if not findings:
            lines.append("  none")
        lines.extend(f"#{index}  {_finding_line(finding)}" for index, finding in enumerate(findings[:limit], start=1))
    return lines


def _baseline_dict(difference: BaselineDiff, limit: int) -> dict[str, Any]:
    """The baseline sections as JSON, each capped at the same limit as the text form."""
    return {
        "entries": difference.entries,
        "counts": {
            "resolved": len(difference.resolved),
            "remaining": len(difference.remaining),
            "new": len(difference.new),
        },
        "resolved": [entry.to_dict() for entry in difference.resolved[:limit]],
        "remaining": [finding.key for finding in difference.remaining[:limit]],
        "new": [finding.key for finding in difference.new[:limit]],
    }


def _axis_line(axis: AxisScore, unscored: tuple[Signal, ...]) -> str:
    """One axis: its score, weight and what it measured."""
    score = axis.score
    left_out = [signal.value for signal in unscored if signal.facts.axis is axis.axis]
    aside = f"; not scored, measured over part of the scope: {', '.join(left_out)}" if left_out else ""
    if score is None:
        return f"- {axis.axis.value}: not measured (weight {axis.weight:g}{aside})"
    measured = ", ".join(f"{item.signal.value} {item.past}/{item.measured}" for item in axis.signals if item.measured)
    return f"- {axis.axis.value}: {score:.1f} (weight {axis.weight:g}; past/measured: {measured}{aside})"


def folded(findings: Sequence[Finding]) -> list[tuple[Finding, tuple[Finding, ...]]]:
    """One row per finding, except that the members of a group follow their first-ranked member, folded under it."""
    rows: list[tuple[Finding, list[Finding]]] = []
    first: dict[tuple[Signal, str], list[Finding]] = {}
    for finding in findings:
        key = (finding.signal, finding.group)
        if finding.group and key in first:
            first[key].append(finding)
            continue
        members: list[Finding] = []
        if finding.group:
            first[key] = members
        rows.append((finding, members))
    return [(finding, tuple(members)) for finding, members in rows]


def _finding_line(finding: Finding, members: int = 0) -> str:
    """`signal  value/threshold  p<percentile>  [resolution]  file:start-end  name  reason`, then the folded count."""
    facts = finding.signal.facts
    parts = [finding.signal.value, f"{finding.value}/{facts.threshold}", f"p{finding.percentile:g}"]
    if facts.tier is Tier.GRAPH:
        parts.append(finding.resolution.value)
    parts.append(f"{finding.file_path}:{finding.start_line}-{finding.end_line}")
    if finding.name:
        parts.append(finding.name)
    parts.append(finding.reason)
    if members:
        parts.append(f"[+{members} more in the same {finding.group.split(' ', 1)[0]}]")
    return "  ".join(parts)


def _findings(signal: Signal, samples: list[Sample]) -> list[Finding]:
    """The samples past the signal's threshold, each with its percentile within its language."""
    facts = signal.facts
    by_language: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        by_language[sample.language].append(facts.comparison.badness(sample.value, facts.threshold))
    for values in by_language.values():
        values.sort()
    found: list[Finding] = []
    for sample in samples:
        if not facts.comparison.past(sample.value, facts.threshold):
            continue
        population = by_language[sample.language]
        rank = bisect_right(population, facts.comparison.badness(sample.value, facts.threshold))
        reason = f"{facts.advice} ({sample.detail})" if sample.detail else facts.advice
        found.append(
            Finding(
                signal=signal,
                language=sample.language,
                file_path=sample.file_path,
                start_line=sample.start_line,
                end_line=sample.end_line,
                name=sample.name,
                value=sample.value,
                percentile=round(100 * rank / len(population), 1),
                resolution=sample.resolution,
                reason=reason,
                key=f"{signal.value}:{sample.language}:{sample.identity}",
                elsewhere=sample.elsewhere,
                group=sample.group,
            )
        )
    return found


def _rank(finding: Finding) -> tuple[Any, ...]:
    """Percentile first, then how far past the threshold, then place: a total, stable order."""
    facts = finding.signal.facts
    badness = facts.comparison.badness(finding.value, facts.threshold)
    return (-finding.percentile, -badness, finding.signal.value, finding.file_path, finding.start_line, finding.key)


def build_report(
    root: str,
    config_source: str | None,
    samples: dict[Signal, list[Sample]],
    paths: tuple[str, ...] = (),
    *,
    since: ChangedLines | None = None,
    copying_focus: frozenset[str] | None = None,
    notes: tuple[str, ...] = (),
) -> HealthReport:
    """Score the samples and rank their findings into one report.

    A `--since` run scores the whole scope, so a duplication signal it measured over the changed files
    only is listed but left out of the score: a score over a subset says nothing about the scope.
    """
    unscored = tuple(
        signal for signal in samples if copying_focus is not None and signal.facts.tier is Tier.DUPLICATION
    )
    axes = tuple(axis_scores({signal: found for signal, found in samples.items() if signal not in unscored}))
    findings = sorted((finding for signal, found in samples.items() for finding in _findings(signal, found)), key=_rank)
    remarks = list(notes)
    layering = next(axis for axis in axes if axis.axis is Axis.LAYERING)
    if layering.score is None and Signal.FORBIDDEN_CROSSING in samples:
        remarks.append(
            "layering not measured: no .zemble/home.toml with [[forbidden]] rules declares what may not depend on what"
        )
    return HealthReport(
        root=root,
        config_source=config_source,
        signals=tuple(samples),
        paths=paths,
        axes=axes,
        composite=composite(axes),
        findings=tuple(findings),
        notes=tuple(remarks),
        since=since,
        copying_focus=copying_focus,
        unscored=unscored,
    )


__all__ = ["DEFAULT_LIMIT", "HealthReport", "build_report"]

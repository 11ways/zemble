"""One `HealthReport` and its two renderings, markdown and JSON, from the same data."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from zemble.health.model import Axis, Finding, Signal, Tier
from zemble.health.score import SCORE_DISCLAIMER, AxisScore, axis_scores, composite
from zemble.health.signals import Sample

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
    findings: tuple[Finding, ...]
    notes: tuple[str, ...]

    def counts(self) -> dict[str, int]:
        """Findings per selected signal, zeros included."""
        counted = {signal.value: 0 for signal in self.signals}
        for finding in self.findings:
            counted[finding.signal.value] += 1
        return counted

    def to_dict(self, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
        """Render the report as JSON-ready data, the findings capped at `limit`."""
        shown = self.findings[:limit]
        return {
            "root": self.root,
            "config": self.config_source,
            "signals": [signal.value for signal in self.signals],
            "paths": list(self.paths),
            "score": round(self.composite, 1) if self.composite is not None else None,
            "score_note": SCORE_DISCLAIMER,
            "axes": [axis.to_dict() for axis in self.axes],
            "counts": self.counts(),
            "total": len(self.findings),
            "truncated": len(shown) < len(self.findings),
            "findings": [finding.to_dict() for finding in shown],
            "notes": list(self.notes),
        }

    def markdown(self, limit: int = DEFAULT_LIMIT) -> str:
        """Render the report as the text the CLI prints and the MCP tool returns."""
        score = "not measured" if self.composite is None else f"{self.composite:.1f} / 100"
        lines = [f"# Health: {self.root}", "", f"Score: {score} ({SCORE_DISCLAIMER})", ""]
        lines.extend(_axis_line(axis) for axis in self.axes)
        lines.append("")
        counts = self.counts()
        lines.append("Findings per signal: " + ", ".join(f"{name} {count}" for name, count in counts.items()))
        lines.extend(f"Note: {note}" for note in self.notes)
        shown = self.findings[:limit]
        lines.append("")
        if not shown:
            lines.append("No findings.")
            return "\n".join(lines) + "\n"
        cut = f", truncated to {len(shown)}" if len(shown) < len(self.findings) else ""
        lines.append(f"## Findings ({len(self.findings)} total{cut}; ranked by percentile, then severity)")
        lines.append("")
        lines.extend(_finding_line(finding) for finding in shown)
        return "\n".join(lines) + "\n"


def _axis_line(axis: AxisScore) -> str:
    """One axis: its score, weight and what it measured."""
    score = axis.score
    if score is None:
        return f"- {axis.axis.value}: not measured (weight {axis.weight:g})"
    measured = ", ".join(f"{item.signal.value} {item.past}/{item.measured}" for item in axis.signals if item.measured)
    return f"- {axis.axis.value}: {score:.1f} (weight {axis.weight:g}; past/measured: {measured})"


def _finding_line(finding: Finding) -> str:
    """`signal  value/threshold  p<percentile>  [resolution]  file:start-end  name  reason`."""
    facts = finding.signal.facts
    parts = [finding.signal.value, f"{finding.value}/{facts.threshold}", f"p{finding.percentile:g}"]
    if facts.tier is Tier.GRAPH:
        parts.append(finding.resolution.value)
    parts.append(f"{finding.file_path}:{finding.start_line}-{finding.end_line}")
    if finding.name:
        parts.append(finding.name)
    parts.append(finding.reason)
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
) -> HealthReport:
    """Score the samples and rank their findings into one report."""
    axes = tuple(axis_scores(samples))
    findings = sorted((finding for signal, found in samples.items() for finding in _findings(signal, found)), key=_rank)
    notes: list[str] = []
    layering = next(axis for axis in axes if axis.axis is Axis.LAYERING)
    if layering.score is None and Signal.FORBIDDEN_CROSSING in samples:
        notes.append(
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
        notes=tuple(notes),
    )


__all__ = ["DEFAULT_LIMIT", "HealthReport", "build_report"]

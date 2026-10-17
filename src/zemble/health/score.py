"""The health score: one declared weight per axis over the thresholds already on each signal.

A signal's fraction is the share of its measured mass past the threshold, where a unit weighs its
own value for a lines signal (a 300-line callable is 300 lines of code past the bar) and 1 for
the rest. An axis scores 100 * (1 - the mean fraction of its measured signals); the composite is
the weighted mean of the measured axes. An axis with no measured signal has no score at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from zemble.health.model import Axis, Signal
from zemble.health.signals import Sample

#: The one weight table. Layering counts double: a forbidden edge breaks a rule the workspace wrote
#: down, while size, machinery, coupling, copying, vocabulary, consistency, history and documentation are
#: judged against general conventions or against the code's own peers.
AXIS_WEIGHTS: dict[Axis, float] = {
    Axis.SIZE: 1.0,
    Axis.MACHINERY: 1.0,
    Axis.LAYERING: 2.0,
    Axis.COUPLING: 1.0,
    Axis.COPYING: 1.0,
    Axis.VOCABULARY: 1.0,
    Axis.CONSISTENCY: 1.0,
    Axis.HISTORY: 1.0,
    Axis.DOCUMENTATION: 1.0,
}

_unweighted = set(Axis) - set(AXIS_WEIGHTS)
if _unweighted:  # pragma: no cover - a drift the import itself refuses
    raise RuntimeError(f"health axes without a weight: {sorted(axis.value for axis in _unweighted)}")

#: What every score line says next to the number, so nobody optimises the number itself.
SCORE_DISCLAIMER = "not a target: address or explain the findings"


def mass(signal: Signal, sample: Sample) -> int:
    """What one sample weighs in its signal's fraction."""
    return sample.value if signal.facts.unit.weighs_by_value else 1


@dataclass(frozen=True)
class SignalScore:
    """One signal's measured population and the part of it past the threshold."""

    signal: Signal
    measured: int
    past: int
    mass: int
    past_mass: int

    @property
    def fraction(self) -> float | None:
        """The share of the measured mass past the threshold, or None when nothing was measured."""
        return self.past_mass / self.mass if self.mass else None

    def to_dict(self) -> dict[str, Any]:
        """Render the signal's score as JSON-ready data."""
        fraction = self.fraction
        return {
            "signal": self.signal.value,
            "measured": self.measured,
            "past": self.past,
            "mass": self.mass,
            "past_mass": self.past_mass,
            "fraction": round(fraction, 4) if fraction is not None else None,
        }


@dataclass(frozen=True)
class AxisScore:
    """One axis: its weight, its measured signals and its score, or None when it measured nothing."""

    axis: Axis
    weight: float
    signals: tuple[SignalScore, ...]

    @property
    def score(self) -> float | None:
        """100 * (1 - the mean fraction of the measured signals), or None without any."""
        fractions = [fraction for item in self.signals if (fraction := item.fraction) is not None]
        return 100.0 * (1.0 - sum(fractions) / len(fractions)) if fractions else None

    def to_dict(self) -> dict[str, Any]:
        """Render the axis as JSON-ready data."""
        score = self.score
        return {
            "axis": self.axis.value,
            "weight": self.weight,
            "score": round(score, 1) if score is not None else None,
            "signals": [item.to_dict() for item in self.signals],
        }


def signal_score(signal: Signal, samples: Sequence[Sample]) -> SignalScore:
    """Score one signal's population against its threshold."""
    facts = signal.facts
    past = [sample for sample in samples if facts.comparison.past(sample.value, facts.threshold)]
    return SignalScore(
        signal,
        len(samples),
        len(past),
        sum(mass(signal, sample) for sample in samples),
        sum(mass(signal, sample) for sample in past),
    )


def axis_scores(samples: dict[Signal, list[Sample]]) -> list[AxisScore]:
    """Score every axis over the signals that ran, in declaration order."""
    return [
        AxisScore(
            axis,
            AXIS_WEIGHTS[axis],
            tuple(signal_score(signal, found) for signal, found in samples.items() if signal.facts.axis is axis),
        )
        for axis in Axis
    ]


def composite(axes: Sequence[AxisScore]) -> float | None:
    """The weighted mean of the measured axes, or None when none was measured."""
    scored = [(axis.weight, score) for axis in axes if (score := axis.score) is not None]
    total = sum(weight for weight, _ in scored)
    return sum(weight * score for weight, score in scored) / total if total else None


__all__ = ["AXIS_WEIGHTS", "SCORE_DISCLAIMER", "AxisScore", "SignalScore", "axis_scores", "composite", "mass"]

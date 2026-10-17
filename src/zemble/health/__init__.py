"""`zemble health`: a ranked, reference-free report of code-shape findings and a score that is never a gate."""

from zemble.health.answers import HealthRequest, health_payload, parse_signals
from zemble.health.model import Axis, Comparison, Finding, Signal, Tier, Unit

__all__ = [
    "Axis",
    "Comparison",
    "Finding",
    "HealthRequest",
    "Signal",
    "Tier",
    "Unit",
    "health_payload",
    "parse_signals",
]

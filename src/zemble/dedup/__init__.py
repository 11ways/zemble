"""Duplication detection over Java: exact, alpha-renamed and logic clone classes."""

from zemble.dedup.detect import DupeOptions, find_duplication
from zemble.dedup.model import CloneClass, CloneKind, DupeReport, Unit
from zemble.dedup.report import format_report, report_json
from zemble.dedup.units import extract_units

__all__ = [
    "CloneClass",
    "CloneKind",
    "DupeOptions",
    "DupeReport",
    "Unit",
    "extract_units",
    "find_duplication",
    "format_report",
    "report_json",
]

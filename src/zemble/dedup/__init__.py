"""Duplication detection over Java: exact, alpha-renamed and logic clone classes."""

from zemble.dedup.baseline import Baseline, BaselineDiff, diff_baseline, load_baseline, save_baseline
from zemble.dedup.detect import DupeOptions, find_duplication
from zemble.dedup.ignore import IGNORE_RELATIVE_PATH, IgnoreFile, apply_ignores
from zemble.dedup.model import CloneClass, CloneKind, DupeReport, Lane, PairReason, Unit
from zemble.dedup.report import format_baseline_diff, format_report, report_json
from zemble.dedup.units import extract_units

__all__ = [
    "IGNORE_RELATIVE_PATH",
    "Baseline",
    "BaselineDiff",
    "CloneClass",
    "CloneKind",
    "DupeOptions",
    "DupeReport",
    "IgnoreFile",
    "Lane",
    "PairReason",
    "Unit",
    "apply_ignores",
    "diff_baseline",
    "extract_units",
    "find_duplication",
    "format_baseline_diff",
    "format_report",
    "load_baseline",
    "report_json",
    "save_baseline",
]

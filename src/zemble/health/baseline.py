"""Health baselines: what a run found once, so a later run can say what is resolved, remaining and new.

The same document and diff vocabulary as `zemble dupes` (`zemble.dedup.baseline`), keyed by the
finding key. A finding key is the symbol's identity without its file or line, so an edit around a
finding never re-keys it: there is no CHANGED bucket, which dupes needs because its keys are content.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zemble.baseline import load_document, save_document
from zemble.health.model import Finding

if TYPE_CHECKING:  # pragma: no cover
    from zemble.health.report import HealthReport

BASELINE_VERSION = 1

#: Where the MCP tool keeps a workspace's baseline, relative to the reported root.
BASELINE_RELATIVE_PATH = ".zemble/health.baseline.json"

#: How a listed finding stands against a baseline.
NEW = "new"
REMAINING = "remaining"


@dataclass(frozen=True, slots=True)
class BaselineEntry:
    """One finding as a baseline remembers it."""

    key: str
    signal: str
    file_path: str
    start_line: int
    end_line: int
    name: str
    value: int

    @property
    def location(self) -> str:
        """File path and line range, the way the report prints it."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    def to_dict(self) -> dict[str, Any]:
        """Render the entry for the wire."""
        return {
            "key": self.key,
            "signal": self.signal,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "name": self.name,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class Baseline:
    """A saved run, keyed by finding key."""

    root: str
    entries: tuple[BaselineEntry, ...]

    @property
    def keys(self) -> set[str]:
        """Every key the baseline holds."""
        return {entry.key for entry in self.entries}


@dataclass(frozen=True, slots=True)
class BaselineDiff:
    """What changed between a baseline and the run in hand; remaining and new hold only listed findings."""

    entries: int
    resolved: tuple[BaselineEntry, ...]
    remaining: tuple[Finding, ...]
    new: tuple[Finding, ...]
    #: Every key of the baseline.
    known: frozenset[str]

    def status(self, finding: Finding) -> str:
        """Whether a listed finding was in the baseline."""
        return REMAINING if finding.key in self.known else NEW


def baseline_payload(report: HealthReport) -> dict[str, Any]:
    """Render every finding of a report's scope, not only the listed ones, as a baseline document."""
    return {
        "version": BASELINE_VERSION,
        "root": report.root,
        "findings": [
            {
                "key": finding.key,
                "signal": finding.signal.value,
                "file_path": finding.file_path,
                "start_line": finding.start_line,
                "end_line": finding.end_line,
                "name": finding.name,
                "value": finding.value,
            }
            for finding in report.findings
        ],
    }


def save_baseline(path: str | Path, report: HealthReport) -> Path:
    """Write a report's finding keys to a baseline file, creating parent directories."""
    return save_document(path, baseline_payload(report))


def load_baseline(path: str | Path) -> Baseline:
    """Read a baseline file.

    :raises ValueError: If the file is not a baseline document this version understands.
    """
    payload = load_document(path, BASELINE_VERSION)
    entries = tuple(
        BaselineEntry(
            key=str(item.get("key", "")),
            signal=str(item.get("signal", "")),
            file_path=str(item.get("file_path", "")),
            start_line=int(item.get("start_line", 0)),
            end_line=int(item.get("end_line", 0)),
            name=str(item.get("name", "")),
            value=int(item.get("value", 0)),
        )
        for item in payload.get("findings", [])
    )
    return Baseline(root=str(payload.get("root", "")), entries=entries)


def diff_baseline(report: HealthReport, baseline: Baseline) -> BaselineDiff:
    """Compare a run against a baseline.

    An entry is resolved only when the run measured its signal over its file and found nothing under
    its key: a run narrowed by `--signals`, `--paths` or a focused duplication run never calls what it
    did not look at resolved. The listed findings (the `--since` delta, or all) are split into
    remaining and new.
    """
    current = {finding.key for finding in report.findings}
    resolved = tuple(
        entry for entry in baseline.entries if entry.key not in current and report.judges(entry.signal, entry.file_path)
    )
    known = frozenset(baseline.keys)
    listed = report.listed
    return BaselineDiff(
        entries=len(baseline.entries),
        resolved=resolved,
        remaining=tuple(finding for finding in listed if finding.key in known),
        new=tuple(finding for finding in listed if finding.key not in known),
        known=known,
    )


__all__ = [
    "BASELINE_RELATIVE_PATH",
    "BASELINE_VERSION",
    "Baseline",
    "BaselineDiff",
    "BaselineEntry",
    "baseline_payload",
    "diff_baseline",
    "load_baseline",
    "save_baseline",
]

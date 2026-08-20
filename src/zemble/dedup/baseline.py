"""Baselines: what a run found once, so a later run can say what is resolved, remaining and new."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zemble.dedup.model import CloneClass, DupeReport

#: Bumped when the saved shape changes in a way a reader has to know about.
BASELINE_VERSION = 1


@dataclass(frozen=True, slots=True)
class BaselineEntry:
    """One clone class as a baseline remembers it."""

    key: str
    kind: str
    lane: str
    copies: int
    score: int
    members: tuple[str, ...]

    @property
    def root_symbol(self) -> str:
        """The first member's location, which is what the diff prints for a resolved class."""
        return self.members[0] if self.members else self.key


@dataclass(frozen=True, slots=True)
class Baseline:
    """A saved run, keyed by clone class key."""

    root: str
    entries: tuple[BaselineEntry, ...]

    @property
    def keys(self) -> set[str]:
        """Every key the baseline holds."""
        return {entry.key for entry in self.entries}


@dataclass(frozen=True, slots=True)
class BaselineDiff:
    """What changed between a baseline and the run in hand."""

    resolved: tuple[BaselineEntry, ...]
    remaining: tuple[CloneClass, ...]
    new: tuple[CloneClass, ...]


def baseline_payload(report: DupeReport) -> dict[str, Any]:
    """Render a report as a baseline document; suppressed classes are deliberately left out."""
    return {
        "version": BASELINE_VERSION,
        "root": report.root,
        "classes": [
            {
                "key": clone.key,
                "kind": clone.kind.value,
                "lane": clone.lane.value,
                "copies": len(clone.members),
                "score": clone.score,
                "members": [member.location for member in clone.members],
            }
            for clone in report.classes
        ],
    }


def save_baseline(path: str | Path, report: DupeReport) -> Path:
    """Write a report's class keys to a baseline file, creating parent directories."""
    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(baseline_payload(report), indent=2) + "\n", encoding="utf-8")
    return target


def load_baseline(path: str | Path) -> Baseline:
    """Read a baseline file.

    :param path: The file written by `--save-baseline`.
    :return: The baseline.
    :raises ValueError: If the file is not a baseline document this version understands.
    """
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"Cannot read baseline {source}: {error}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Baseline {source} is not valid JSON: {error}") from None
    if not isinstance(payload, dict) or payload.get("version") != BASELINE_VERSION:
        raise ValueError(f"Baseline {source} is not a version {BASELINE_VERSION} document")
    entries = []
    for item in payload.get("classes", []):
        entries.append(
            BaselineEntry(
                key=str(item.get("key", "")),
                kind=str(item.get("kind", "")),
                lane=str(item.get("lane", "")),
                copies=int(item.get("copies", 0)),
                score=int(item.get("score", 0)),
                members=tuple(str(member) for member in item.get("members", [])),
            )
        )
    return Baseline(root=str(payload.get("root", "")), entries=tuple(entries))


def _in_scope(entry: BaselineEntry, report: DupeReport) -> bool:
    """Whether a run that used a --kind or --lane filter is entitled to judge a baseline entry."""
    if entry.kind not in {kind.value for kind in report.kinds}:
        return False
    return report.lane is None or entry.lane == report.lane.value


def diff_baseline(report: DupeReport, baseline: Baseline) -> BaselineDiff:
    """Compare a run against a baseline.

    A class suppressed by the ignore file counts as neither remaining nor new, and a baseline
    entry that is now suppressed is not called resolved either: it is still there, on purpose.
    A run narrowed by --kind or --lane never calls the entries it did not look for resolved.

    :param report: The run in hand.
    :param baseline: The saved run.
    :return: The resolved, remaining and new classes.
    """
    current = {clone.key: clone for clone in report.classes}
    suppressed = {clone.key for clone in report.suppressed}
    resolved = tuple(
        entry
        for entry in baseline.entries
        if entry.key not in current and entry.key not in suppressed and _in_scope(entry, report)
    )
    known = baseline.keys
    remaining = tuple(clone for clone in report.classes if clone.key in known)
    new = tuple(clone for clone in report.classes if clone.key not in known)
    return BaselineDiff(resolved=resolved, remaining=remaining, new=new)


__all__ = [
    "BASELINE_VERSION",
    "Baseline",
    "BaselineDiff",
    "BaselineEntry",
    "baseline_payload",
    "diff_baseline",
    "load_baseline",
    "save_baseline",
]

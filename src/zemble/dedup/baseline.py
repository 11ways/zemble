"""Baselines: what a run found once, so a later run can say what is resolved, remaining and new."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zemble.dedup.model import CloneClass, DupeReport

#: Bumped when the saved shape changes in a way a reader has to know about. Version 2: class
#: keys are content-only (no file paths), so version 1 baselines would silently mis-diff.
BASELINE_VERSION = 2

#: Where the MCP tool keeps a workspace's baseline, relative to the scanned root.
BASELINE_RELATIVE_PATH = ".zemble/dupes.baseline.json"


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

    @property
    def files(self) -> set[str]:
        """The file paths of the members, with the line spans stripped off."""
        return {member.rsplit(":", 1)[0] for member in self.members}

    def to_dict(self) -> dict[str, Any]:
        """Render the entry for the wire."""
        return {
            "key": self.key,
            "kind": self.kind,
            "lane": self.lane,
            "copies": self.copies,
            "score": self.score,
            "members": list(self.members),
        }


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
class ChangedClass:
    """A baseline entry paired with the differently-keyed class its files still hold."""

    was: BaselineEntry
    now: CloneClass

    @property
    def score_delta(self) -> int:
        """How much the class's weight moved; negative means it shrank."""
        return self.now.score - self.was.score


@dataclass(frozen=True, slots=True)
class BaselineDiff:
    """What changed between a baseline and the run in hand."""

    resolved: tuple[BaselineEntry, ...]
    changed: tuple[ChangedClass, ...]
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


def _pair_changed(
    resolved: Sequence[BaselineEntry], candidates: Sequence[CloneClass]
) -> tuple[list[BaselineEntry], list[CloneClass], list[ChangedClass]]:
    """Pair each candidate with the gone entry of its kind sharing the most member files."""
    open_entries = list(resolved)
    unmatched: list[CloneClass] = []
    changed: list[ChangedClass] = []
    for clone in candidates:
        files = {member.file_path for member in clone.members}
        best: BaselineEntry | None = None
        best_overlap = 0
        for entry in open_entries:
            if entry.kind != clone.kind.value:
                continue
            overlap = len(files & entry.files)
            if overlap > best_overlap:
                best, best_overlap = entry, overlap
        if best is None:
            unmatched.append(clone)
        else:
            open_entries.remove(best)
            changed.append(ChangedClass(was=best, now=clone))
    return open_entries, unmatched, changed


def diff_baseline(report: DupeReport, baseline: Baseline) -> BaselineDiff:
    """Compare a run against a baseline.

    A class suppressed by the ignore file counts as neither remaining nor new, and a baseline
    entry that is now suppressed is not called resolved either: it is still there, on purpose
    (that also holds when the suppressed class re-keyed but still spans the entry's files).
    A gone entry whose files now hold a same-kind class under a new key is CHANGED, not
    resolved-plus-new: content-derived keys churn on edits, and the diff owes the reader the
    pairing. A run narrowed by --kind or --lane never calls what it did not look for resolved.

    :param report: The run in hand.
    :param baseline: The saved run.
    :return: The resolved, changed, remaining and new classes.
    """
    current = {clone.key: clone for clone in report.classes}
    suppressed = {clone.key for clone in report.suppressed}
    gone = [
        entry
        for entry in baseline.entries
        if entry.key not in current and entry.key not in suppressed and _in_scope(entry, report)
    ]
    known = baseline.keys
    remaining = tuple(clone for clone in report.classes if clone.key in known)
    fresh = [clone for clone in report.classes if clone.key not in known]
    # A re-keyed but suppressed class silently claims its old entry: still there, on purpose.
    gone, _, _ = _pair_changed(gone, [clone for clone in report.suppressed if clone.key not in known])
    resolved, new, changed = _pair_changed(gone, fresh)
    return BaselineDiff(resolved=tuple(resolved), changed=tuple(changed), remaining=remaining, new=tuple(new))


__all__ = [
    "BASELINE_RELATIVE_PATH",
    "BASELINE_VERSION",
    "Baseline",
    "BaselineDiff",
    "BaselineEntry",
    "ChangedClass",
    "baseline_payload",
    "diff_baseline",
    "load_baseline",
    "save_baseline",
]

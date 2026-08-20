"""The committed suppression file: deliberate duplication, each entry justified.

Source-guard convention: an entry without a justification is itself a violation, and an
entry that matches nothing is stale. Both are reported; neither suppresses anything.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from zemble.dedup.model import CloneClass

#: Where a workspace's suppression file lives, relative to the scanned root.
IGNORE_RELATIVE_PATH = ".zemble/dupes.ignore"


@dataclass(frozen=True, slots=True)
class IgnoreEntry:
    """One line of the ignore file: a clone class key and why it is allowed to stay."""

    key: str
    justification: str
    line: int


@dataclass(frozen=True, slots=True)
class IgnoreFile:
    """A parsed ignore file: the usable entries and the lines that are violations."""

    path: Path
    entries: tuple[IgnoreEntry, ...] = ()
    problems: tuple[str, ...] = ()

    @classmethod
    def load(cls, root: str | Path) -> IgnoreFile:
        """Read `<root>/.zemble/dupes.ignore`, returning an empty file when there is none."""
        path = Path(root) / IGNORE_RELATIVE_PATH
        if not path.is_file():
            return cls(path=path)
        entries: list[IgnoreEntry] = []
        problems: list[str] = []
        for number, raw in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 1:
                problems.append(f"{IGNORE_RELATIVE_PATH}:{number}: {parts[0]} has no justification")
                continue
            entries.append(IgnoreEntry(key=parts[0], justification=parts[1].strip(), line=number))
        return cls(path=path, entries=tuple(entries), problems=tuple(problems))


@dataclass(frozen=True, slots=True)
class Suppression:
    """The outcome of applying an ignore file to one run's classes."""

    kept: tuple[CloneClass, ...]
    suppressed: tuple[CloneClass, ...]
    problems: tuple[str, ...]


def apply_ignores(classes: Sequence[CloneClass], ignores: IgnoreFile, scanned_kinds: Sequence[str] = ()) -> Suppression:
    """Split classes into the reported ones and the suppressed ones.

    An entry is only called stale when its kind was actually scanned: `--kind exact` must not
    declare every `renamed:` entry dead.

    :param classes: The ranked classes of one run.
    :param ignores: The parsed ignore file.
    :param scanned_kinds: The kind names this run looked for; empty means all of them.
    :return: The kept classes, the suppressed classes and every ignore-file violation.
    """
    justified = {entry.key: entry for entry in ignores.entries}
    kept: list[CloneClass] = []
    suppressed: list[CloneClass] = []
    matched: set[str] = set()
    for clone in classes:
        if clone.key in justified:
            matched.add(clone.key)
            suppressed.append(clone)
        else:
            kept.append(clone)
    problems = list(ignores.problems)
    problems.extend(_stale(ignores.entries, matched, tuple(scanned_kinds)))
    return Suppression(kept=tuple(kept), suppressed=tuple(suppressed), problems=tuple(problems))


def _stale(entries: Iterable[IgnoreEntry], matched: set[str], scanned_kinds: tuple[str, ...]) -> list[str]:
    """Name every entry of a scanned kind that suppressed nothing this run."""
    return [
        f"{IGNORE_RELATIVE_PATH}:{entry.line}: {entry.key} is stale, it matches no clone class"
        for entry in entries
        if entry.key not in matched and (not scanned_kinds or entry.key.split(":", 1)[0] in scanned_kinds)
    ]


__all__ = ["IGNORE_RELATIVE_PATH", "IgnoreEntry", "IgnoreFile", "Suppression", "apply_ignores"]

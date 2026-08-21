"""The committed suppression files: deliberate duplication, each entry justified.

Source-guard convention: an entry without a justification is itself a violation, and an
entry that matches nothing is stale. Both are reported; neither suppresses anything.
A workspace scan honours the root's file plus every `.zemble/dupes.ignore` a directory
holding scanned files declares, so a repo's own entries hold from any ancestor root.
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
    """One line of an ignore file: a clone class key, why it may stay, and where it was written."""

    key: str
    justification: str
    line: int
    source: str = IGNORE_RELATIVE_PATH


@dataclass(frozen=True, slots=True)
class IgnoreFile:
    """A parsed ignore file: the usable entries and the lines that are violations."""

    path: Path
    entries: tuple[IgnoreEntry, ...] = ()
    problems: tuple[str, ...] = ()

    @classmethod
    def load(cls, root: str | Path) -> IgnoreFile:
        """Read `<root>/.zemble/dupes.ignore`, returning an empty file when there is none."""
        return cls.load_at(Path(root) / IGNORE_RELATIVE_PATH, IGNORE_RELATIVE_PATH)

    @classmethod
    def load_at(cls, path: Path, label: str) -> IgnoreFile:
        """Read one ignore file, naming it `label` in every entry and violation it yields."""
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
                problems.append(f"{label}:{number}: {parts[0]} has no justification")
                continue
            entries.append(IgnoreEntry(key=parts[0], justification=parts[1].strip(), line=number, source=label))
        return cls(path=path, entries=tuple(entries), problems=tuple(problems))


def find_ignore_files(root: Path, file_paths: Iterable[str]) -> tuple[IgnoreFile, ...]:
    """Load the root's ignore file and every one under a directory that holds scanned files.

    :param root: The scanned root.
    :param file_paths: Root-relative paths of every scanned file.
    :return: The non-empty ignore files, root first, then by path.
    """
    directories = {""}
    for file_path in file_paths:
        segments = file_path.replace("\\", "/").split("/")[:-1]
        for depth in range(1, len(segments) + 1):
            directories.add("/".join(segments[:depth]))
    files: list[IgnoreFile] = []
    for relative in sorted(directories):
        label = f"{relative}/{IGNORE_RELATIVE_PATH}" if relative else IGNORE_RELATIVE_PATH
        loaded = IgnoreFile.load_at(root / label, label)
        if loaded.entries or loaded.problems:
            files.append(loaded)
    return tuple(files)


@dataclass(frozen=True, slots=True)
class Suppression:
    """The outcome of applying the ignore files to one run's classes."""

    kept: tuple[CloneClass, ...]
    suppressed: tuple[CloneClass, ...]
    problems: tuple[str, ...]


def apply_ignores(
    classes: Sequence[CloneClass], ignores: Sequence[IgnoreFile], scanned_kinds: Sequence[str] = ()
) -> Suppression:
    """Split classes into the reported ones and the suppressed ones.

    An entry is only called stale when its kind was actually scanned: `--kind exact` must not
    declare every `renamed:` entry dead.

    :param classes: The ranked classes of one run.
    :param ignores: The parsed ignore files, in the order they were found.
    :param scanned_kinds: The kind names this run looked for; empty means all of them.
    :return: The kept classes, the suppressed classes and every ignore-file violation.
    """
    entries = [entry for ignore in ignores for entry in ignore.entries]
    justified = {entry.key for entry in entries}
    kept: list[CloneClass] = []
    suppressed: list[CloneClass] = []
    matched: set[str] = set()
    for clone in classes:
        if clone.key in justified:
            matched.add(clone.key)
            suppressed.append(clone)
        else:
            kept.append(clone)
    problems = [problem for ignore in ignores for problem in ignore.problems]
    problems.extend(_stale(entries, matched, tuple(scanned_kinds)))
    return Suppression(kept=tuple(kept), suppressed=tuple(suppressed), problems=tuple(problems))


def _stale(entries: Iterable[IgnoreEntry], matched: set[str], scanned_kinds: tuple[str, ...]) -> list[str]:
    """Name every entry of a scanned kind that suppressed nothing this run."""
    return [
        f"{entry.source}:{entry.line}: {entry.key} is stale, it matches no clone class"
        for entry in entries
        if entry.key not in matched and (not scanned_kinds or entry.key.split(":", 1)[0] in scanned_kinds)
    ]


__all__ = ["IGNORE_RELATIVE_PATH", "IgnoreEntry", "IgnoreFile", "Suppression", "apply_ignores", "find_ignore_files"]

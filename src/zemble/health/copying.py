"""The duplication facts `zemble health` judges: one `zemble dupes` run of the exact, renamed and vocabulary channels.

Nothing is detected here: the clone classes, shaped bodies and vocabulary sites are `find_duplication`'s
own, production lane only, `.zemble/dupes.ignore` honoured. The run is rooted at the graph's root, so a
sub-directory's copies are compared against the whole tree, and a bounded scope (`--paths`, `--since`, a
sub-directory) is a `--focus` run whose other files come from the per-root unit index.

AIDEV-NOTE: never the logic channel: it buys embeddings, and health never spends money.
"""

from __future__ import annotations

import os
import posixpath
from collections.abc import Sequence
from dataclasses import dataclass, replace

from zemble.dedup.detect import DupeOptions, find_duplication
from zemble.dedup.model import CloneClass, CloneKind, Lane, Unit
from zemble.dedup.report import scan_notes
from zemble.graph.provider import AnyProvider, SubtreeGraphProvider

#: The channels a health run reads; `CODE_KINDS` are copied code, the rest copied vocabulary.
COPYING_KINDS = (CloneKind.EXACT, CloneKind.RENAMED, CloneKind.VOCABULARY)
CODE_KINDS = frozenset({CloneKind.EXACT, CloneKind.RENAMED})


@dataclass(frozen=True)
class CopyingFacts:
    """One duplication run, every path relative to the report root (`../` for a member outside a sub-directory)."""

    classes: tuple[CloneClass, ...]
    #: Production shaped bodies and vocabulary sites of the whole tree.
    shaped: tuple[Unit, ...]
    sites: tuple[Unit, ...]
    #: The smallest body a clone class may hold, in tokens.
    min_tokens: int
    #: The files the run judged, or None for every file in the scope; a `--since` run judges the changed ones.
    focus: frozenset[str] | None
    notes: tuple[str, ...]

    def judged(self, file_path: str) -> bool:
        """Whether the run compared this file against the tree, so its units may be measured."""
        return self.focus is None or file_path in self.focus


def measure_copying(provider: AnyProvider, scope: Sequence[str], changed: Sequence[str] | None) -> CopyingFacts:
    """Run the duplication channels for a report.

    :param scope: The report's `--paths`, relative to the report root; empty for all of it.
    :param changed: The files a `--since` run reports on, relative to the report root, or None without one.
    """
    root, prefix = _dedup_root(provider)
    focus: tuple[str, ...]
    if changed is not None:
        focus = tuple(_join(prefix, path) for path in changed)
    elif scope:
        focus = tuple(_join(prefix, path) for path in scope)
    else:
        focus = (prefix,) if prefix else ()
    judged = frozenset(changed) if changed is not None else None
    if changed is not None and not focus:
        return CopyingFacts((), (), (), DupeOptions().min_tokens, judged, ("copying: no changed file to compare",))
    options = DupeOptions(kinds=COPYING_KINDS, lane=Lane.PRODUCTION, focus=focus, keep_units=True)
    report = find_duplication(root, options)
    clones = sum(clone.kind in CODE_KINDS for clone in report.classes)
    # No wall time here: the same tree must render the same report.
    notes = [
        f"copying: {clones} clone and {len(report.classes) - clones} vocabulary class(es) over "
        f"{report.analyzed_files} file(s) ({'focused' if focus else 'whole tree'})",
        *(f"copying: {note}" for note in scan_notes(report)),
        *(f"copying: ignore-file violation: {problem}" for problem in report.ignore_problems),
    ]
    return CopyingFacts(
        classes=tuple(
            replace(clone, members=tuple(_rebased(member, prefix) for member in clone.members))
            for clone in report.classes
        ),
        shaped=tuple(_rebased(unit, prefix) for unit in report.shaped if not unit.is_test),
        sites=tuple(_rebased(unit, prefix) for unit in report.sites if not unit.is_test),
        min_tokens=options.min_tokens,
        focus=judged,
        notes=tuple(notes),
    )


def _dedup_root(provider: AnyProvider) -> tuple[str, str]:
    """The tree the graph was built for and the report root's path inside it ("" for the same)."""
    if isinstance(provider, SubtreeGraphProvider):
        return os.path.realpath(provider.graph.path), provider.prefix
    return os.path.realpath(provider.path), ""


def _join(prefix: str, path: str) -> str:
    """A report-root-relative path relative to the graph's root."""
    return posixpath.join(prefix, path) if prefix else path


def _rebased(unit: Unit, prefix: str) -> Unit:
    """A unit whose path is relative to the report root; `is_test` was read before, on the tree's own path."""
    return replace(unit, file_path=posixpath.relpath(unit.file_path, prefix)) if prefix else unit


__all__ = ["CODE_KINDS", "COPYING_KINDS", "CopyingFacts", "measure_copying"]

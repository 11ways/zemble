"""The history axis's facts: per production file, the commits of a window that touched it, and the pairs that co-change.

AIDEV-NOTE: git is read in a subprocess only, through `zemble.health.git`. Commits are kept per
current production file of the view, so test files, deleted files and files under an old name never
take part. A changeset touching more than `MAX_CHANGESET_FILES` files still counts toward churn but
not toward co-change: a reformat or a rename sweep says nothing about which files belong together.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import combinations

from zemble.health.git import history_checkouts, window_commits

#: Days of history a run reads unless asked otherwise.
DEFAULT_WINDOW_DAYS = 365
#: Commits two files must share before their co-change is measured: code-maat's `min-shared-revs` default.
CO_CHANGE_MIN = 5
#: Commits a file needs in the window before its churn is ranked: code-maat's `min-revs` default. A short
#: history (a repository with 16 commits a year) otherwise puts a file changed 3 times in its top decile.
HOTSPOT_MIN_COMMITS = 5
#: Files a changeset may touch and still count toward co-change: code-maat's `max-changeset-size` default.
MAX_CHANGESET_FILES = 30


@dataclass(frozen=True)
class CoChange:
    """Two files that changed together in at least `CO_CHANGE_MIN` commits, paths sorted."""

    first: str
    second: str
    #: Commits that touched both.
    shared: int
    #: Commits that touched either.
    either: int


@dataclass(frozen=True)
class HistoryFacts:
    """What one window of history says about the view's production files."""

    window_days: int
    #: Per production file with at least one commit in the window, the ids of those commits.
    commits: dict[str, frozenset[str]] = field(default_factory=dict)
    co_changes: tuple[CoChange, ...] = ()
    #: The path prefix of every checkout read ("" for a root inside one), longest first.
    checkouts: tuple[str, ...] = ()
    #: Why the axis is not measured (no checkout, git failed or ran out of time), or None when it is.
    unmeasured: str | None = None

    def churn(self, file_path: str) -> int:
        """Commits in the window that touched a file."""
        return len(self.commits.get(file_path, ()))

    def checkout_of(self, file_path: str) -> str:
        """The prefix of the checkout a file's history was read from: each checkout is its own history."""
        return next((prefix for prefix in self.checkouts if file_path.startswith(prefix)), "")


def measure_history(root: str, files: Iterable[str], window_days: int) -> HistoryFacts:
    """Read the window's commits under `root` and keep those touching one of `files` (paths relative to it)."""
    checkouts = history_checkouts(root)
    if not checkouts:
        return HistoryFacts(window_days, unmeasured="history not measured: not in a git checkout and holds none")
    try:
        commits = window_commits(checkouts, window_days)
    except ValueError as error:
        return HistoryFacts(window_days, unmeasured=f"history not measured: {error}")
    wanted = set(files)
    touched: dict[str, set[str]] = defaultdict(set)
    small: dict[str, set[str]] = defaultdict(set)
    shared: Counter[tuple[str, str]] = Counter()
    for commit in commits:
        kept = sorted(set(commit.files) & wanted)
        for path in kept:
            touched[path].add(commit.id)
        if len(commit.files) <= MAX_CHANGESET_FILES:
            for path in kept:
                small[path].add(commit.id)
            shared.update(combinations(kept, 2))
    co_changes = tuple(
        CoChange(first, second, count, len(small[first] | small[second]))
        for (first, second), count in sorted(shared.items())
        if count >= CO_CHANGE_MIN
    )
    commits_by_path = {path: frozenset(ids) for path, ids in touched.items()}
    prefixes = tuple(sorted((prefix for _, prefix in checkouts), key=len, reverse=True))
    return HistoryFacts(window_days, commits_by_path, co_changes, prefixes)


__all__ = [
    "CO_CHANGE_MIN",
    "DEFAULT_WINDOW_DAYS",
    "HOTSPOT_MIN_COMMITS",
    "MAX_CHANGESET_FILES",
    "CoChange",
    "HistoryFacts",
    "measure_history",
]

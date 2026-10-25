"""Lanes: a folder whose git checkouts are linked worktrees of an already indexed workspace's.

A lane holds the same files as its base workspace except the handful its worktrees edited, so it
is served as an overlay of the base's index and graph instead of being indexed again. This
module is the one home of the identity rule and of the per-file "same as the base" decision;
`zemble.index.overlay` and `zemble.graph.store` read both from here.

Identity: a path P is in a lane when walking up from it (or, for a folder that is no checkout,
down into its checkouts) finds a linked worktree whose main worktree sits at the same relative
spot under another folder B that holds an index. The lane root L and the base B are the highest
such pair, so a sub-path of a lane is a sub-tree of the lane's overlay rather than a lane of its
own. Every checkout of L at relative path p whose main worktree is B/p is a matched repository;
anything else under L (an unmatched checkout, plain files at the lane root) is compared by bytes.
"""

from __future__ import annotations

import filecmp
import logging
import os
import subprocess
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from zemble.chunking.capsule import RepoRelativePaths

logger = logging.getLogger(__name__)

#: How deep below a folder that is no checkout its checkouts are looked for (`apps/<name>` is 2).
DISCOVERY_DEPTH = 2
#: Directory names never descended into while looking for checkouts.
_SKIPPED_DIRS = frozenset({"node_modules", "build", "dist", "target", "__pycache__", "venv"})
_GIT_TIMEOUT_SECONDS = 60
_GIT_WORKERS = 8


@dataclass(frozen=True)
class RepoPair:
    """One checkout of the lane and the base checkout it is a linked worktree of."""

    #: The checkout's path relative to the lane root, POSIX, "" for the lane root itself.
    prefix: str
    lane_dir: Path
    base_dir: Path


@dataclass(frozen=True)
class Lane:
    """A lane root, its base workspace, and the checkouts the two share."""

    root: Path
    base: Path
    repos: tuple[RepoPair, ...]


def _git_dir_entry(directory: Path) -> Path | None:
    """Return the `.git` entry of a checkout directory, or None when it is none."""
    entry = directory / ".git"
    return entry if entry.exists() else None


def main_worktree(directory: Path) -> Path | None:
    """Return the main worktree of the repository whose linked worktree *directory* is, or None.

    Read straight from the `.git` file and the `commondir` file it points at, without running git:
    a main checkout (a `.git` folder), a bare repository's worktree or an unreadable entry answer None.
    """
    entry = _git_dir_entry(directory)
    if entry is None or not entry.is_file():
        return None
    try:
        text = entry.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    git_dir = Path(text.removeprefix("gitdir:").strip())
    if not git_dir.is_absolute():
        git_dir = directory / git_dir
    try:
        common = (git_dir / (git_dir / "commondir").read_text(encoding="utf-8").strip()).resolve()
    except OSError:
        return None
    return common.parent if common.name == ".git" else None


def _checkouts_below(root: Path, depth: int = DISCOVERY_DEPTH) -> list[Path]:
    """List checkout directories under *root* (itself included), not descending into a checkout below the root."""
    found: list[Path] = []
    frontier = [(root, 0)]
    while frontier:
        directory, level = frontier.pop()
        is_checkout = _git_dir_entry(directory) is not None
        if is_checkout:
            found.append(directory)
            if directory != root:
                continue
        if level >= depth:
            continue
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith(".") or entry.name in _SKIPPED_DIRS:
                continue
            if entry.is_dir(follow_symlinks=False):
                frontier.append((Path(entry.path), level + 1))
    return sorted(found)


def _containing_checkout(path: Path) -> Path | None:
    """Return the nearest checkout directory at or above *path*."""
    for candidate in (path, *path.parents):
        if _git_dir_entry(candidate) is not None:
            return candidate
    return None


def _base_of(lane_dir: Path, prefix: str) -> Path | None:
    """Return the folder that would be the base if *lane_dir*, at *prefix* in its lane, matches."""
    main = main_worktree(lane_dir)
    if main is None:
        return None
    if not prefix:
        return main
    parts = Path(prefix).parts
    if main.parts[-len(parts) :] != parts:
        return None
    return Path(*main.parts[: -len(parts)])


def _candidate_pairs(resolved: Path) -> list[tuple[Path, Path]]:
    """List the (lane root, base) pairs a path could be in, from its checkout or the checkouts below it."""
    checkout = _containing_checkout(resolved)
    main = main_worktree(checkout) if checkout is not None else None
    candidates: list[tuple[Path, Path]] = []
    if checkout is not None and main is not None:
        # Walk up the lane checkout and its main worktree together: every level where they still
        # differ is a possible (lane, base) pair; the first level they share is above every lane.
        lane_side, base_side = checkout, main
        while lane_side != base_side and lane_side.name and base_side.name:
            candidates.append((lane_side, base_side))
            if lane_side.name != base_side.name and lane_side.parent == base_side.parent:
                break
            lane_side, base_side = lane_side.parent, base_side.parent
        return candidates
    votes: dict[Path, int] = {}
    for directory in _checkouts_below(resolved):
        prefix = directory.relative_to(resolved).as_posix()
        base = _base_of(directory, "" if prefix == "." else prefix)
        if base is not None and base != resolved:
            votes[base] = votes.get(base, 0) + 1
    return [(resolved, base) for base in sorted(votes, key=lambda base: -votes[base])]


def find_lane(path: str | Path, is_indexed: Callable[[str], bool]) -> tuple[Lane, str | None] | None:
    """Return the lane a path is in and its prefix inside the lane root, or None outside every lane.

    :param path: The requested path.
    :param is_indexed: Whether a folder holds an index a lane could overlay.
    :return: The lane and the path's lane-relative POSIX prefix (None for the lane root itself).
    """
    try:
        resolved = Path(path).expanduser().resolve()
    except OSError:
        return None
    if not resolved.is_dir():
        return None
    # The highest indexed pair wins: a sub-path of a lane is a sub-tree of the lane's overlay.
    for lane_root, base in sorted(_candidate_pairs(resolved), key=lambda pair: len(pair[0].parts)):
        if lane_root == base or not base.is_dir() or not is_indexed(str(base)):
            continue
        if not resolved.is_relative_to(lane_root):
            continue
        lane = describe_lane(lane_root, base)
        if not lane.repos:
            continue
        prefix = resolved.relative_to(lane_root).as_posix()
        return lane, (None if prefix in ("", ".") else prefix)
    return None


def describe_lane(root: Path, base: Path) -> Lane:
    """Pair every checkout under a lane root with the base checkout it is a linked worktree of."""
    pairs = []
    for directory in _checkouts_below(root):
        prefix = directory.relative_to(root).as_posix()
        prefix = "" if prefix == "." else prefix
        if _base_of(directory, prefix) == base:
            pairs.append(RepoPair(prefix, directory, base / prefix if prefix else base))
    return Lane(root=root, base=base, repos=tuple(pairs))


def _git(directory: Path, *arguments: str) -> list[str]:
    """Run one git command in *directory* and return its NUL-separated output records."""
    completed = subprocess.run(
        ["git", "-C", str(directory), *arguments],
        capture_output=True,
        timeout=_GIT_TIMEOUT_SECONDS,
        check=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    return [record for record in completed.stdout.decode("utf-8", "surrogateescape").split("\0") if record]


@dataclass
class _RepoState:
    """What git says about one matched checkout: which tracked files may differ from the base's."""

    tracked: frozenset[str]
    differing: frozenset[str]


def _repo_state(pair: RepoPair) -> _RepoState | None:
    """Ask git which tracked files of a lane checkout may differ from its base checkout's working tree.

    A file is the same in both when it is tracked in the lane and neither the lane's working tree
    differs from the base's HEAD for it, nor the base's working tree from its own HEAD.
    """
    try:
        base_head = _git(pair.base_dir, "rev-parse", "HEAD")[0].strip()
        tracked = _git(pair.lane_dir, "ls-files", "-z")
        lane_moved = _git(pair.lane_dir, "diff", "--name-only", "--no-renames", "-z", base_head, "--")
        base_moved = _git(pair.base_dir, "diff", "--name-only", "--no-renames", "-z", "HEAD", "--")
    except (OSError, subprocess.SubprocessError, IndexError) as exc:
        logger.info(
            "lane: git could not compare %s with %s (%s); comparing its files by content",
            pair.lane_dir,
            pair.base_dir,
            exc,
        )
        return None
    return _RepoState(frozenset(tracked), frozenset(lane_moved) | frozenset(base_moved))


@dataclass
class LaneComparer:
    """Decides, file by file, whether a lane's file is byte-for-byte its base's, asking git where it can."""

    lane: Lane
    _states: dict[str, _RepoState | None] = field(default_factory=dict)
    _prefixes: list[str] = field(default_factory=list)
    _lane_paths: RepoRelativePaths = field(default_factory=RepoRelativePaths)
    _base_paths: RepoRelativePaths = field(default_factory=RepoRelativePaths)

    def __post_init__(self) -> None:
        """Collect every matched checkout's git state once, in parallel."""
        with ThreadPoolExecutor(max_workers=_GIT_WORKERS) as pool:
            states = list(pool.map(_repo_state, self.lane.repos))
        self._states = {pair.prefix: state for pair, state in zip(self.lane.repos, states, strict=True)}
        # Longest first, so a file is judged by the innermost checkout holding it.
        self._prefixes = sorted(self._states, key=len, reverse=True)

    def _repo_of(self, relative: str) -> tuple[str, str] | None:
        """Return the matched checkout prefix holding a lane-relative path and the path inside it."""
        for prefix in self._prefixes:
            if not prefix:
                return prefix, relative
            if relative.startswith(f"{prefix}/"):
                return prefix, relative[len(prefix) + 1 :]
        return None

    def same(self, relative: str) -> bool:
        """Return whether the lane's file at a lane-relative path has exactly the base's bytes and capsule path."""
        lane_file, base_file = self.lane.root / relative, self.lane.base / relative
        if self._lane_paths.path_for(lane_file, relative) != self._base_paths.path_for(base_file, relative):
            return False
        held = self._repo_of(relative)
        if held is not None:
            state = self._states[held[0]]
            if state is not None and held[1] in state.tracked:
                return held[1] not in state.differing
        try:
            return filecmp.cmp(lane_file, base_file, shallow=False)
        except OSError:
            return False

    def partition(self, relatives: Iterable[str]) -> tuple[list[str], list[str]]:
        """Split lane-relative paths into those the base holds unchanged and those that differ."""
        same: list[str] = []
        differing: list[str] = []
        for relative in relatives:
            (same if self.same(relative) else differing).append(relative)
        return same, differing


__all__ = ["DISCOVERY_DEPTH", "Lane", "LaneComparer", "RepoPair", "describe_lane", "find_lane", "main_worktree"]

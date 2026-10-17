"""Git reading for `zemble health`, the one place it runs git: `--since` lines and the history window's commits.

AIDEV-NOTE: `git diff --unified=0 <ref>` compares the working tree with the ref, so staged and
unstaged edits both count; untracked files git would add (`ls-files --others --exclude-standard`)
count whole. Both run with `--relative` semantics from the report root, so every path is already
relative to it and nothing outside it is listed. A hunk that only deletes lines (`+c,0`) marks the
line it follows: the code around a removal changed too.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from bisect import bisect_right
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

#: Seconds one git command may take; a diff of a large branch is still well under it.
GIT_TIMEOUT_SECONDS = 60
#: Seconds a whole history read may take, every checkout together; past it the history axis is not
#: measured. It runs inside a daemon request, which holds a read slot that long at most.
HISTORY_TIMEOUT_SECONDS = 20
#: History reads run at once, one git process each.
_HISTORY_WORKERS = 8
#: Starts each commit of a history log: a byte no path or hash spells.
_RECORD = "\x1e"
#: The last line a whole new file is taken to span.
_WHOLE_FILE = 2**31 - 1
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_NEW_FILE = "+++ "


@dataclass(frozen=True)
class ChangedLines:
    """Per path relative to the report root, the sorted, disjoint line ranges changed since `ref`."""

    ref: str
    ranges: dict[str, tuple[tuple[int, int], ...]] = field(default_factory=dict)

    @property
    def files(self) -> tuple[str, ...]:
        """Every changed path, sorted."""
        return tuple(sorted(self.ranges))

    def touches(self, file_path: str, start_line: int, end_line: int) -> bool:
        """Whether a span of a file intersects a changed line."""
        ranges = self.ranges.get(file_path, ())
        # The last range starting at or before the span's end is the only one that can reach back into it.
        index = bisect_right(ranges, (end_line, _WHOLE_FILE)) - 1
        return index >= 0 and ranges[index][1] >= start_line


def changed_lines(root: str, ref: str) -> ChangedLines:
    """Read every line the working tree under `root` changed or added since `ref`.

    :raises ValueError: If `root` is not inside a git checkout, git is missing, or `ref` names no commit.
    """
    if not ref or ref.startswith("-"):
        raise ValueError(f"--since takes a git ref, got {ref!r}")
    _git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", failure=f"--since: {ref!r} names no commit")
    diff = _git(
        root,
        "-c",
        "core.quotePath=false",
        "diff",
        "--unified=0",
        "--no-color",
        "--no-ext-diff",
        "--relative",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        ref,
        "--",
        failure=f"--since: git diff {ref} failed",
    )
    spans = _hunk_spans(diff)
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z", failure="--since: git ls-files failed")
    # A nested checkout is listed as its directory (`dir/`): git would add it as a gitlink, never as files.
    for path in untracked.split("\0"):
        if path and not path.endswith("/"):
            spans[path].append((1, _WHOLE_FILE))
    return ChangedLines(ref, {path: merge_ranges(ranges) for path, ranges in spans.items()})


def _hunk_spans(diff: str) -> dict[str, list[tuple[int, int]]]:
    """The new-side line range of every hunk of a `--unified=0` diff, by the path it changes."""
    spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    current: str | None = None
    for line in diff.splitlines():
        if line.startswith(_NEW_FILE):
            # Git ends a path holding a space with a tab, for patch(1); a deleted file is `/dev/null`.
            target = line[len(_NEW_FILE) :].rstrip("\t")
            current = target[2:] if target.startswith("b/") else None
        elif current is not None and (hunk := _HUNK.match(line)):
            start, count = int(hunk.group(1)), int(hunk.group(2) or 1)
            spans[current].append((start, start + count - 1) if count else (max(start, 1), max(start, 1)))
    return spans


def merge_ranges(ranges: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Sorted ranges with the overlapping and adjacent ones joined."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


@dataclass(frozen=True)
class Commit:
    """One commit of a history window and the files it touched, relative to the report root."""

    id: str
    files: tuple[str, ...]


def history_checkouts(root: str) -> tuple[tuple[str, str], ...]:
    """The (checkout directory, path prefix) pairs a root's history is read from.

    A root inside a checkout is read from there (prefix ""), and so is each direct child directory
    holding a `.git` entry that is a checkout of its own (a workspace of sibling repositories, with or
    without a checkout at its top); a deeper checkout is not looked for. A child checkout's files never
    appear in the enclosing checkout's log (git records a gitlink or nothing for them), so no commit is
    counted twice. Empty when there is none.
    """
    found = [(root, "")] if _inside_checkout(root) else []
    try:
        children = sorted(entry.name for entry in os.scandir(root) if entry.is_dir(follow_symlinks=False))
    except OSError:
        return tuple(found)
    for name in children:
        directory = os.path.join(root, name)
        if os.path.exists(os.path.join(directory, ".git")) and _checkout_top(directory) == os.path.realpath(directory):
            found.append((directory, f"{name}/"))
    return tuple(found)


def _inside_checkout(directory: str) -> bool:
    """Whether git reads a directory as part of a work tree."""
    try:
        return _git(directory, "rev-parse", "--is-inside-work-tree", failure="").strip() == "true"
    except ValueError:
        return False


def _checkout_top(directory: str) -> str | None:
    """The resolved top directory of the work tree a directory is in, or None outside one."""
    try:
        top = _git(directory, "rev-parse", "--show-toplevel", failure="").strip()
    except ValueError:
        return None
    return os.path.realpath(top) if top else None


def window_commits(checkouts: tuple[tuple[str, str], ...], days: int) -> list[Commit]:
    """Every non-merge commit of the last `days` days of each checkout, with the files it touched under it.

    One `git log --name-only --no-renames` per checkout, run side by side within one shared deadline.
    Without rename detection a rename lists both paths, so the file under its current name still gets
    the commit, as it would with detection (which cost 5x on javaweb); history before a rename stays
    under the old name either way, since `--follow` is per file.

    :raises ValueError: When a checkout's log fails or the read passes `HISTORY_TIMEOUT_SECONDS`.
    """
    deadline = time.monotonic() + HISTORY_TIMEOUT_SECONDS
    with ThreadPoolExecutor(max_workers=max(1, min(_HISTORY_WORKERS, len(checkouts)))) as pool:
        logs = list(pool.map(lambda checkout: _window_log(*checkout, days, deadline), checkouts))
    return [commit for log in logs for commit in log]


def _window_log(directory: str, prefix: str, days: int, deadline: float) -> list[Commit]:
    """One checkout's commits of the window, paths relative to it and then prefixed."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ValueError(f"git log took longer than {HISTORY_TIMEOUT_SECONDS} s")
    try:
        log = _git(
            directory,
            "-c",
            "core.quotePath=false",
            "log",
            f"--since={days}.days.ago",
            "--no-merges",
            "--no-renames",
            "--name-only",
            "--relative",
            f"--format={_RECORD}%H",
            "--",
            ".",
            failure=f"git log in {directory} failed",
            timeout=remaining,
        )
    except ValueError:
        if not _has_commits(directory):
            return []  # a branch with no commit yet has no history to read
        raise
    commits = []
    for record in log.split(_RECORD)[1:]:
        lines = record.splitlines()
        files = tuple(f"{prefix}{line}" for line in lines[1:] if line)
        commits.append(Commit(lines[0].strip(), files))
    return commits


def _has_commits(directory: str) -> bool:
    """Whether a checkout's HEAD names a commit."""
    try:
        _git(directory, "rev-parse", "--verify", "--quiet", "HEAD", failure="")
    except ValueError:
        return False
    return True


def _git(root: str, *arguments: str, failure: str, timeout: float = GIT_TIMEOUT_SECONDS) -> str:
    """Run one git command in `root` and return its stdout.

    :raises ValueError: With `failure` and git's own message, when it cannot run, exits non-zero or times out.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", root, *arguments],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise ValueError(f"{failure}: took longer than {timeout:.0f} s") from None
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"{failure}: {error}") from None
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        raise ValueError(f"{failure}: {detail[-1]}" if detail else failure)
    return completed.stdout


__all__ = [
    "HISTORY_TIMEOUT_SECONDS",
    "ChangedLines",
    "Commit",
    "changed_lines",
    "history_checkouts",
    "merge_ranges",
    "window_commits",
]

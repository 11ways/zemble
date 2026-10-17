"""The lines a working tree changed since a git ref: what `zemble health --since` restricts its findings to.

AIDEV-NOTE: `git diff --unified=0 <ref>` compares the working tree with the ref, so staged and
unstaged edits both count; untracked files git would add (`ls-files --others --exclude-standard`)
count whole. Both run with `--relative` semantics from the report root, so every path is already
relative to it and nothing outside it is listed. A hunk that only deletes lines (`+c,0`) marks the
line it follows: the code around a removal changed too.
"""

from __future__ import annotations

import re
import subprocess
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field

#: Seconds one git command may take; a diff of a large branch is still well under it.
GIT_TIMEOUT_SECONDS = 60
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


def _git(root: str, *arguments: str, failure: str) -> str:
    """Run one git command in `root` and return its stdout.

    :raises ValueError: With `failure` and git's own message, when it cannot run or exits non-zero.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", root, *arguments],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"{failure}: {error}") from None
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        raise ValueError(f"{failure}: {detail[-1]}" if detail else failure)
    return completed.stdout


__all__ = ["ChangedLines", "changed_lines", "merge_ranges"]

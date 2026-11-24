"""Which processes hold a file open, read from /proc without opening the file itself."""

from __future__ import annotations

import os
from pathlib import Path

_PROC = Path("/proc")


def holders(path: Path, *, exclude_self: bool = False) -> list[int] | None:
    """Return the pids with *path* open, or None where the platform cannot tell.

    AIDEV-NOTE: this never opens *path*. A process that opens and then closes a sqlite file
    drops every POSIX lock sqlite holds on it in that process, so probing a live store by
    opening it would be the very corruption the probe exists to avoid. Each `/proc/<pid>/fd`
    link is stat'ed and compared by device and inode, which also finds a renamed file.
    Processes whose fds are not readable (another user's) are skipped.

    :param path: The file to look for.
    :param exclude_self: Leave this process out of the answer.
    :return: The holding pids, empty when nobody holds it, None without a readable /proc.
    """
    try:
        target = os.stat(path)
    except FileNotFoundError:
        return []
    key = (target.st_dev, target.st_ino)
    found = _open_by_pid(exclude_self=exclude_self)
    if found is None:
        return None
    return [pid for pid, files in found if key in files]


def open_files(*, exclude_self: bool = False) -> set[tuple[int, int]] | None:
    """Return the (device, inode) of every file any process holds open, or None where the platform cannot tell.

    One walk of /proc answers for any number of files, where `holders` walks it per file.
    """
    found = _open_by_pid(exclude_self=exclude_self)
    return None if found is None else set().union(*(files for _pid, files in found))


def _open_by_pid(*, exclude_self: bool) -> list[tuple[int, set[tuple[int, int]]]] | None:
    """Return each readable process's open files as (device, inode), None without a readable /proc."""
    if not (_PROC / "self" / "fd").is_dir():
        return None
    me = os.getpid()
    found: list[tuple[int, set[tuple[int, int]]]] = []
    for entry in os.listdir(_PROC):
        if not entry.isdigit() or (exclude_self and int(entry) == me):
            continue
        fd_dir = _PROC / entry / "fd"
        try:
            descriptors = os.listdir(fd_dir)
        except OSError:
            continue
        files: set[tuple[int, int]] = set()
        for descriptor in descriptors:
            try:
                linked = os.stat(fd_dir / descriptor)
            except OSError:
                continue
            files.add((linked.st_dev, linked.st_ino))
        found.append((int(entry), files))
    return found


def held_open(path: Path, *, exclude_self: bool = False) -> bool:
    """Return whether any process holds *path* open; True where that cannot be known, so callers fail closed."""
    found = holders(path, exclude_self=exclude_self)
    return found is None or bool(found)

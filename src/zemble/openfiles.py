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
    if not (_PROC / "self" / "fd").is_dir():
        return None
    key = (target.st_dev, target.st_ino)
    me = os.getpid()
    found: list[int] = []
    for entry in os.listdir(_PROC):
        if not entry.isdigit() or (exclude_self and int(entry) == me):
            continue
        fd_dir = _PROC / entry / "fd"
        try:
            descriptors = os.listdir(fd_dir)
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                linked = os.stat(fd_dir / descriptor)
            except OSError:
                continue
            if (linked.st_dev, linked.st_ino) == key:
                found.append(int(entry))
                break
    return found


def held_open(path: Path, *, exclude_self: bool = False) -> bool:
    """Return whether any process holds *path* open; True where that cannot be known, so callers fail closed."""
    found = holders(path, exclude_self=exclude_self)
    return found is None or bool(found)

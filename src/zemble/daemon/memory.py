"""Memory admission and the daemon process's allocation backstop."""

from __future__ import annotations

import os
import resource
from pathlib import Path

from zemble.refusal import Refused

MIB = 1024 * 1024
MEMORY_ENV = "ZEMBLE_DAEMON_MAX_RSS_MB"
VIRTUAL_ENV = "ZEMBLE_DAEMON_MAX_VIRTUAL_MB"


class MemoryRefused(Refused):
    """Daemon memory admission refused work without falling back to an unbounded client build."""

    DEFAULT_KNOB = MEMORY_ENV


def default_budget_mb() -> int:
    """Use at most fifteen percent of physical RAM and never more than four GiB."""
    try:
        total = next(
            int(line.split()[1])
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemTotal:")
        )
    except (OSError, StopIteration, ValueError):
        return 1024
    return min(4096, max(1, int(total / 1024 * 0.15)))


def virtual_mb() -> float:
    """Read address space, including mapped stores and allocator reservations."""
    return int(Path("/proc/self/statm").read_text().split()[0]) * os.sysconf("SC_PAGE_SIZE") / MIB


def allocation_backstop(budget_mb: int) -> int:
    """Cap virtual memory conservatively so even a missed RSS estimate cannot allocate past the budget."""
    limit = budget_mb * MIB
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    if soft != resource.RLIM_INFINITY:
        limit = min(limit, soft)
    if virtual_mb() * MIB >= limit:
        raise MemoryRefused(
            f"Daemon startup address space exceeds the {limit / MIB:.0f} MiB allocation ceiling "
            f"({VIRTUAL_ENV}; legacy {MEMORY_ENV}); "
            "use single-threaded BLAS or raise the budget/inherited address-space limit."
        )
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    return limit // MIB

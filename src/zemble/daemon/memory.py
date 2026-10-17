"""Memory admission and the daemon process's allocation backstop.

The budget counts PRIVATE memory only: heap and private writable mappings, which is what the
kernel's `VmData` and `RLIMIT_DATA` measure. Index stores are memory-mapped files, page cache the
kernel can drop and re-read at will, and counting them (as address space or as resident file
pages) made one large mapped index look like gigabytes of allocation.
"""

from __future__ import annotations

import resource
from pathlib import Path

from zemble.refusal import Refused

MIB = 1024 * 1024
MEMORY_ENV = "ZEMBLE_DAEMON_MAX_RSS_MB"


class MemoryRefused(Refused):
    """Daemon memory admission refused work without falling back to an unbounded client build."""

    DEFAULT_KNOB = MEMORY_ENV


class ReadOutgrown(RuntimeError):
    """A read this daemon cannot hold in its own process; it is answered from stored data, so another can.

    Not a `Refused`: a refusal is the same answer in every process, and this one is not.
    """

    def __init__(self, message: str) -> None:
        """Carry the memory knob, so a handler reading `knob` off any daemon error gets an answer."""
        super().__init__(message)
        self.knob = MEMORY_ENV


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


def private_mb() -> float:
    """Return this process's private memory (`VmData`) in MiB, or 0 where the platform reports none."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmData:"):
                return int(line.split()[1]) / 1024
    except (OSError, IndexError, ValueError):
        pass
    return 0.0


def data_limit_mb() -> float:
    """The process's current private-memory ceiling (the `RLIMIT_DATA` soft limit) in MiB, or infinity."""
    soft, _hard = resource.getrlimit(resource.RLIMIT_DATA)
    return float("inf") if soft == resource.RLIM_INFINITY else soft / MIB


def allocation_backstop(budget_mb: int) -> int:
    """Cap private memory so even a missed admission estimate cannot allocate past the budget.

    :return: The cap in MiB, which a stricter inherited limit lowers.
    :raises MemoryRefused: If the process already uses that much at startup.
    """
    limit = budget_mb * MIB
    soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    if soft != resource.RLIM_INFINITY:
        limit = min(limit, soft)
    if private_mb() * MIB >= limit:
        raise MemoryRefused(
            f"Daemon startup private memory exceeds the {limit / MIB:.0f} MiB allocation ceiling ({MEMORY_ENV}); "
            "raise the budget or the inherited data limit."
        )
    resource.setrlimit(resource.RLIMIT_DATA, (limit, hard))
    return limit // MIB

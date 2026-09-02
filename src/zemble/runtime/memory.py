"""Handing freed heap back to the operating system after a build.

Building or rebuilding an index churns through millions of short-lived Python objects
(chunk objects, posting dictionaries, token lists). Freeing them returns the memory to the
allocator, not to the kernel: glibc only shrinks the heap from its top, so one large build
leaves the daemon resident at its peak for the rest of its life even though nothing is
holding the memory. Measured on the javaweb workspace, a build's churn left 139 MB of the
143 MB it had freed still resident until the heap was trimmed.

There is no portable way to ask for this. `malloc_trim` is a glibc extension, so its absence
(musl, macOS) is the normal case rather than an error, and this module degrades to a no-op.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging

logger = logging.getLogger(__name__)


def _resolve_malloc_trim() -> "ctypes._FuncPointer | None":
    """Return glibc's malloc_trim, or None where the C library does not provide it."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=False)
        trim = libc.malloc_trim
    except (OSError, AttributeError):
        logger.debug("malloc_trim is unavailable; freed heap is returned at the allocator's discretion")
        return None
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return trim


_MALLOC_TRIM = _resolve_malloc_trim()


def release_free_heap() -> bool:
    """Return the allocator's free heap to the kernel, if the platform can.

    Safe to call at any point: it moves no live object and only ever releases memory nothing
    references. Callers use it after finishing a build, not inside one.

    :return: Whether memory was actually released.
    """
    if _MALLOC_TRIM is None:
        return False
    try:
        return bool(_MALLOC_TRIM(0))
    except OSError:  # pragma: no cover - a trim that fails is not a build failure
        logger.debug("malloc_trim failed", exc_info=True)
        return False

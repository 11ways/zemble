"""Shared building blocks for the memory-mapped index columns."""

from __future__ import annotations

import mmap
import os
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt


def _temporary_suffix() -> str:
    """Return a suffix no other writer of the same file can be using."""
    return f"{os.getpid()}.{threading.get_ident()}"


def atomic_bytes(path: Path, data: bytes) -> None:
    """Write a file by replacing it, never by truncating the one that is there.

    A warm process keeps the columns of the index it loaded memory-mapped, and those files
    live in the directory a save writes into. Truncating one under a live mapping is a
    SIGBUS on the next page touch, so every column is written beside its target and moved
    onto it: the mapping keeps the inode it already had.
    """
    temporary = path.with_name(f"{path.name}.{_temporary_suffix()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def atomic_save(path: Path, array: npt.NDArray) -> None:
    """Write one .npy column the way :func:`atomic_bytes` writes a blob."""
    temporary = path.with_name(f"{path.name}.{_temporary_suffix()}.tmp.npy")
    with open(temporary, "wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def map_blob(path: Path) -> bytes | mmap.mmap:
    """Map a byte blob read-only, falling back to empty bytes for an empty file."""
    with open(path, "rb") as handle:
        if path.stat().st_size == 0:
            return b""
        return mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)


def offsets_of(items: Sequence[bytes]) -> npt.NDArray[np.int64]:
    """Return the exclusive-end offsets of *items* concatenated back to back."""
    offsets = np.zeros(len(items) + 1, dtype=np.int64)
    np.cumsum(np.fromiter((len(item) for item in items), dtype=np.int64, count=len(items)), out=offsets[1:])
    return offsets


class StringTable:
    """A string table stored as one blob plus offsets, read without materializing its entries.

    ``index_of`` binary-searches the blob, so a lookup costs a handful of slices instead of the
    dict build a persisted vocabulary would otherwise need; it requires the table to have been
    saved in UTF-8 byte order (which is Python's own string order).
    """

    def __init__(self, blob: bytes | mmap.mmap, offsets: npt.NDArray[np.int64]) -> None:
        """Hold a mapped blob and its offsets."""
        self._blob = blob
        self._offsets = offsets

    def __len__(self) -> int:
        """The number of entries."""
        return len(self._offsets) - 1

    def at(self, index: int) -> str:
        """Decode the entry at *index*."""
        return bytes(self._blob[self._offsets[index] : self._offsets[index + 1]]).decode("utf-8")

    def raw(self, index: int) -> bytes:
        """Return the raw bytes of the entry at *index*."""
        return bytes(self._blob[self._offsets[index] : self._offsets[index + 1]])

    def index_of(self, value: str) -> int | None:
        """Binary-search a sorted table, returning the row of *value* or None."""
        needle = value.encode("utf-8")
        offsets = self._offsets
        blob = self._blob
        low, high = 0, len(self)
        while low < high:
            middle = (low + high) // 2
            candidate = bytes(blob[offsets[middle] : offsets[middle + 1]])
            if candidate < needle:
                low = middle + 1
            elif candidate > needle:
                high = middle
            else:
                return middle
        return None

    def to_list(self) -> list[str]:
        """Materialize every entry, in stored order."""
        offsets = np.asarray(self._offsets)
        blob = self._blob
        return [bytes(blob[start:end]).decode("utf-8") for start, end in zip(offsets[:-1], offsets[1:])]

    @staticmethod
    def file_names(name: str) -> tuple[str, str]:
        """Return the blob and offsets file names for a table called *name*."""
        return f"{name}.bin", f"{name}_offsets.npy"

    @classmethod
    def of(cls, values: Sequence[str]) -> "StringTable":
        """Build an in-memory table; *values* must already be in the order lookups expect."""
        encoded = [value.encode("utf-8") for value in values]
        return cls(b"".join(encoded), offsets_of(encoded))

    @classmethod
    def save(cls, path: Path, name: str, values: Sequence[str] | "StringTable") -> None:
        """Write a string table; *values* must already be in the order lookups expect."""
        table = values if isinstance(values, StringTable) else cls.of(values)
        blob_name, offsets_name = cls.file_names(name)
        atomic_bytes(path / blob_name, bytes(table._blob[:]))
        atomic_save(path / offsets_name, np.asarray(table._offsets))

    @classmethod
    def load(cls, path: Path, name: str) -> "StringTable":
        """Map a string table written by :meth:`save`."""
        blob_name, offsets_name = cls.file_names(name)
        return cls(map_blob(path / blob_name), np.load(path / offsets_name, mmap_mode="r"))

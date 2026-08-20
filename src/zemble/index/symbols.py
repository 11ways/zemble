"""Persisted symbol-definition lookup.

Reranking a symbol query used to regex-scan every chunk in the index for a definition of the
queried name. The same scan is done once at save time instead, and the result is stored as
name -> chunk indices, so a query costs two binary searches. The names come from
:func:`zemble.ranking.boosting.defined_symbol_names`, which mirrors the definition patterns the
scan used, so a lookup answers exactly what the scan answered.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import orjson

from zemble.index.columnar import StringTable
from zemble.ranking.boosting import NAMESPACE_CHAIN_RE, defined_symbol_names
from zemble.types import Chunk

#: Bumped when the columnar symbol layout changes shape.
_SYMBOLS_FORMAT = 1

_META_NAME = "symbols.json"
_GENERAL = "general"
_SQL = "sql"
_OFFSETS_SUFFIX = "_posting_offsets.npy"
_CHUNKS_SUFFIX = "_posting_chunks.npy"


def _is_chain_name(name: str) -> bool:
    """Return whether *name* is a plain identifier or namespace chain the stored names can hold."""
    return NAMESPACE_CHAIN_RE.fullmatch(name) is not None


class _NameTable:
    """One sorted name table plus its CSR postings into the chunk list."""

    def __init__(self, names: StringTable, offsets: npt.NDArray[np.int64], chunks: npt.NDArray[np.int32]) -> None:
        """Hold the mapped columns of one table."""
        self.names = names
        self.offsets = offsets
        self.chunks = chunks

    def postings(self, name: str) -> npt.NDArray[np.int32] | None:
        """Return the chunk indices defining *name*, or None if the name is unknown."""
        row = self.names.index_of(name)
        if row is None:
            return None
        return np.asarray(self.chunks[self.offsets[row] : self.offsets[row + 1]])

    @classmethod
    def save(cls, path: Path, prefix: str, postings: dict[str, list[int]]) -> None:
        """Write a name table and its postings."""
        names = sorted(postings)
        counts = np.fromiter((len(postings[name]) for name in names), dtype=np.int64, count=len(names))
        offsets = np.zeros(len(names) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        chunk_ids = np.fromiter(
            (chunk_index for name in names for chunk_index in postings[name]),
            dtype=np.int32,
            count=int(counts.sum()),
        )
        StringTable.save(path, prefix, names)
        np.save(path / f"{prefix}{_OFFSETS_SUFFIX}", offsets)
        np.save(path / f"{prefix}{_CHUNKS_SUFFIX}", chunk_ids)

    @classmethod
    def load(cls, path: Path, prefix: str) -> "_NameTable":
        """Map a name table written by :meth:`save`."""
        return cls(
            StringTable.load(path, prefix),
            np.load(path / f"{prefix}{_OFFSETS_SUFFIX}", mmap_mode="r"),
            np.load(path / f"{prefix}{_CHUNKS_SUFFIX}", mmap_mode="r"),
        )


class SymbolDefinitions:
    """Name -> chunk-index lookup replacing the per-query definition scan."""

    def __init__(self, general: _NameTable, sql: _NameTable) -> None:
        """Hold the general (case-sensitive) and SQL (lowercased) tables."""
        self._general = general
        self._sql = sql

    def chunks_defining(self, names: Iterable[str]) -> npt.NDArray[np.int32] | None:
        """Return the ascending chunk indices defining any of *names*.

        :param names: The queried symbol names.
        :return: Ascending chunk indices, or None when a name cannot be answered from the tables
            (it is not a plain identifier or namespace chain) and the caller must scan instead.
        """
        found: list[npt.NDArray[np.int32]] = []
        for name in names:
            if not _is_chain_name(name):
                return None
            for table, key in ((self._general, name), (self._sql, name.lower())):
                postings = table.postings(key)
                if postings is not None:
                    found.append(postings)
        if not found:
            return np.empty(0, dtype=np.int32)
        return np.unique(np.concatenate(found))

    @classmethod
    def load(cls, path: Path) -> "SymbolDefinitions":
        """Map the symbol tables in *path*.

        :param path: Directory the tables were saved to.
        :return: The mapped lookup.
        :raises ValueError: If the stored format is not the current one.
        """
        meta = orjson.loads((path / _META_NAME).read_bytes())
        if meta.get("format") != _SYMBOLS_FORMAT:
            raise ValueError(f"Unsupported symbol format {meta.get('format')!r}; expected {_SYMBOLS_FORMAT}")
        return cls(_NameTable.load(path, _GENERAL), _NameTable.load(path, _SQL))


def save_symbol_definitions(path: Path, chunks: Sequence[Chunk]) -> None:
    """Scan every chunk for definitions and write the lookup tables into *path*."""
    path.mkdir(parents=True, exist_ok=True)
    general: dict[str, list[int]] = {}
    sql: dict[str, list[int]] = {}
    for index, chunk in enumerate(chunks):
        general_names, sql_names = defined_symbol_names(chunk.content)
        for name in general_names:
            general.setdefault(name, []).append(index)
        for name in sql_names:
            sql.setdefault(name, []).append(index)

    _NameTable.save(path, _GENERAL, general)
    _NameTable.save(path, _SQL, sql)
    (path / _META_NAME).write_bytes(orjson.dumps({"format": _SYMBOLS_FORMAT, "n_chunks": len(chunks)}))


def symbol_files(path: Path) -> list[Path]:
    """Return every file a persisted symbol lookup is made of."""
    files = [path / _META_NAME]
    for prefix in (_GENERAL, _SQL):
        files.extend(path / name for name in StringTable.file_names(prefix))
        files.append(path / f"{prefix}{_OFFSETS_SUFFIX}")
        files.append(path / f"{prefix}{_CHUNKS_SUFFIX}")
    return files

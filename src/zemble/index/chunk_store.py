"""Columnar persistence for the chunk list.

A list of dicts costs a JSON parse plus one Python object per chunk at load time. The columns
here are memory-mapped instead, and ``Chunk`` objects are built only for the chunks a caller
actually touches; the path and language columns answer whole-index questions without
materializing anything.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, overload

import numpy as np
import numpy.typing as npt
import orjson

from zemble.index.columnar import StringTable, map_blob, offsets_of
from zemble.types import Chunk

#: Bumped when the columnar chunk layout changes shape.
_CHUNKS_FORMAT = 2

_META_NAME = "chunks.json"
_CONTENT_NAME = "content.bin"
_CONTENT_OFFSETS_NAME = "content_offsets.npy"
_CONTEXT_NAME = "context.bin"
_CONTEXT_OFFSETS_NAME = "context_offsets.npy"
_PATHS_TABLE = "paths"
_PATH_IDS_NAME = "path_ids.npy"
_LANGUAGES_TABLE = "languages"
_LANGUAGE_IDS_NAME = "language_ids.npy"
_LINES_NAME = "lines.npy"


class ChunkList(Sequence[Chunk]):
    """A read-only, lazily materializing view over the persisted chunk columns."""

    def __init__(
        self,
        content: bytes,
        content_offsets: npt.NDArray[np.int64],
        context: bytes,
        context_offsets: npt.NDArray[np.int64],
        path_table: list[str],
        path_ids: npt.NDArray[np.int32],
        language_table: list[str],
        language_ids: npt.NDArray[np.int32],
        lines: npt.NDArray[np.int32],
    ) -> None:
        """Hold the columns; nothing is decoded until a chunk is asked for."""
        self._content = content
        self._content_offsets = content_offsets
        self._context = context
        self._context_offsets = context_offsets
        self._path_table = path_table
        self._path_ids = path_ids
        self._language_table = language_table
        self._language_ids = language_ids
        self._lines = lines
        self._file_paths: list[str] | None = None
        self._languages: list[str | None] | None = None

    def __len__(self) -> int:
        """The number of chunks."""
        return len(self._path_ids)

    @overload
    def __getitem__(self, item: int) -> Chunk: ...

    @overload
    def __getitem__(self, item: slice) -> list[Chunk]: ...

    def __getitem__(self, item: int | slice) -> Chunk | list[Chunk]:
        """Materialize one chunk, or a list of chunks for a slice."""
        if isinstance(item, slice):
            return [self._chunk(i) for i in range(*item.indices(len(self)))]
        if item < 0:
            item += len(self)
        if not 0 <= item < len(self):
            raise IndexError(item)
        return self._chunk(item)

    def __iter__(self) -> Iterator[Chunk]:
        """Iterate over every chunk, materializing them one by one."""
        for index in range(len(self)):
            yield self._chunk(index)

    def __eq__(self, other: object) -> bool:
        """Compare element-wise against any other sequence of chunks."""
        if isinstance(other, Sequence):
            return len(self) == len(other) and all(mine == theirs for mine, theirs in zip(self, other))
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    def _chunk(self, index: int) -> Chunk:
        """Build the Chunk at *index* from the columns."""
        start, end = self._content_offsets[index], self._content_offsets[index + 1]
        context_start, context_end = self._context_offsets[index], self._context_offsets[index + 1]
        language_id = int(self._language_ids[index])
        return Chunk(
            content=bytes(self._content[start:end]).decode("utf-8"),
            file_path=self._path_table[int(self._path_ids[index])],
            start_line=int(self._lines[index, 0]),
            end_line=int(self._lines[index, 1]),
            language=self._language_table[language_id] if language_id >= 0 else None,
            context=bytes(self._context[context_start:context_end]).decode("utf-8"),
        )

    @property
    def file_paths(self) -> list[str]:
        """Every chunk's file path, in chunk order, without materializing chunks."""
        if self._file_paths is None:
            table = self._path_table
            self._file_paths = [table[index] for index in self._path_ids.tolist()]
        return self._file_paths

    @property
    def languages(self) -> list[str | None]:
        """Every chunk's language, in chunk order, without materializing chunks."""
        if self._languages is None:
            table = self._language_table
            self._languages = [table[index] if index >= 0 else None for index in self._language_ids.tolist()]
        return self._languages


def file_paths_of(chunks: Sequence[Chunk]) -> list[str]:
    """Return every chunk's file path, using the columns when the sequence has them."""
    columns = getattr(chunks, "file_paths", None)
    return columns if columns is not None else [chunk.file_path for chunk in chunks]


def languages_of(chunks: Sequence[Chunk]) -> list[str | None]:
    """Return every chunk's language, using the columns when the sequence has them."""
    columns = getattr(chunks, "languages", None)
    return columns if columns is not None else [chunk.language for chunk in chunks]


def save_chunks(path: Path, chunks: Sequence[Chunk]) -> None:
    """Write the chunk columns into *path*."""
    path.mkdir(parents=True, exist_ok=True)

    contents: list[bytes] = []
    contexts: list[bytes] = []
    path_ids: list[int] = []
    language_ids: list[int] = []
    lines = np.zeros((len(chunks), 2), dtype=np.int32)
    path_index: dict[str, int] = {}
    language_index: dict[str, int] = {}

    for row, chunk in enumerate(chunks):
        contents.append(chunk.content.encode("utf-8"))
        contexts.append(chunk.context.encode("utf-8"))
        path_ids.append(path_index.setdefault(chunk.file_path, len(path_index)))
        language = chunk.language
        language_ids.append(-1 if language is None else language_index.setdefault(language, len(language_index)))
        lines[row] = (chunk.start_line, chunk.end_line)

    (path / _CONTENT_NAME).write_bytes(b"".join(contents))
    np.save(path / _CONTENT_OFFSETS_NAME, offsets_of(contents))
    (path / _CONTEXT_NAME).write_bytes(b"".join(contexts))
    np.save(path / _CONTEXT_OFFSETS_NAME, offsets_of(contexts))
    StringTable.save(path, _PATHS_TABLE, list(path_index))
    StringTable.save(path, _LANGUAGES_TABLE, list(language_index))
    np.save(path / _PATH_IDS_NAME, np.array(path_ids, dtype=np.int32))
    np.save(path / _LANGUAGE_IDS_NAME, np.array(language_ids, dtype=np.int32))
    np.save(path / _LINES_NAME, lines)
    (path / _META_NAME).write_bytes(orjson.dumps({"format": _CHUNKS_FORMAT, "n_chunks": len(chunks)}))


def load_chunks(path: Path) -> ChunkList:
    """Map the chunk columns in *path*.

    :param path: Directory the chunks were saved to.
    :return: A lazily materializing chunk sequence.
    :raises ValueError: If the stored format is not the current one or the columns disagree.
    """
    meta: dict[str, Any] = orjson.loads((path / _META_NAME).read_bytes())
    if meta.get("format") != _CHUNKS_FORMAT:
        raise ValueError(f"Unsupported chunk format {meta.get('format')!r}; expected {_CHUNKS_FORMAT}")
    n_chunks = meta["n_chunks"]

    content = map_blob(path / _CONTENT_NAME)
    content_offsets = np.load(path / _CONTENT_OFFSETS_NAME, mmap_mode="r")
    context = map_blob(path / _CONTEXT_NAME)
    context_offsets = np.load(path / _CONTEXT_OFFSETS_NAME, mmap_mode="r")
    path_ids = np.load(path / _PATH_IDS_NAME, mmap_mode="r")
    language_ids = np.load(path / _LANGUAGE_IDS_NAME, mmap_mode="r")
    lines = np.load(path / _LINES_NAME, mmap_mode="r")
    path_table = StringTable.load(path, _PATHS_TABLE).to_list()
    language_table = StringTable.load(path, _LANGUAGES_TABLE).to_list()

    if not (
        len(content_offsets) == len(context_offsets) == n_chunks + 1
        and len(path_ids) == len(language_ids) == len(lines) == n_chunks
    ):
        raise ValueError("Persisted chunk columns have inconsistent lengths")

    return ChunkList(
        content, content_offsets, context, context_offsets, path_table, path_ids, language_table, language_ids, lines
    )


def chunk_files(path: Path) -> list[Path]:
    """Return every file a persisted chunk list is made of."""
    return [
        path / name
        for name in (
            _META_NAME,
            _CONTENT_NAME,
            _CONTENT_OFFSETS_NAME,
            _CONTEXT_NAME,
            _CONTEXT_OFFSETS_NAME,
            _PATH_IDS_NAME,
            _LANGUAGE_IDS_NAME,
            _LINES_NAME,
            *StringTable.file_names(_PATHS_TABLE),
            *StringTable.file_names(_LANGUAGES_TABLE),
        )
    ]

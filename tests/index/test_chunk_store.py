from pathlib import Path

import numpy as np
import pytest

from zemble.index.chunk_store import ChunkList, file_paths_of, languages_of, load_chunks, save_chunks
from zemble.types import Chunk


def _chunks() -> list[Chunk]:
    """A small chunk list covering repeated paths, a missing language and non-ASCII content."""
    return [
        Chunk(content="def a():\n    return 1\n", file_path="src/a.py", start_line=1, end_line=2, language="python"),
        Chunk(content="def b():\n    return 2\n", file_path="src/a.py", start_line=4, end_line=5, language="python"),
        Chunk(content="Ünïcode ✓ content", file_path="docs/readme.md", start_line=1, end_line=1, language=None),
        Chunk(content="", file_path="src/empty.java", start_line=1, end_line=1, language="java"),
    ]


def test_chunks_survive_the_columnar_roundtrip(tmp_path: Path) -> None:
    """Saving and mapping a chunk list gives back the same chunks, by value and by column."""
    chunks = _chunks()

    # 1. The mapped list equals the original, element by element and as a whole.
    save_chunks(tmp_path, chunks)
    loaded = load_chunks(tmp_path)
    assert isinstance(loaded, ChunkList)
    assert len(loaded) == len(chunks)
    assert loaded == chunks, "the mapped list compares equal to the list it was built from"
    assert list(loaded) == chunks, "iteration materializes the same chunks"
    assert loaded[0] == chunks[0] and loaded[-1] == chunks[-1], "indexing works from both ends"
    assert loaded[1:3] == chunks[1:3], "a slice materializes a plain list"

    # 2. The path and language columns answer without materializing chunks.
    assert file_paths_of(loaded) == [chunk.file_path for chunk in chunks]
    assert languages_of(loaded) == [chunk.language for chunk in chunks]
    assert file_paths_of(chunks) == file_paths_of(loaded), "a plain list answers the same question"
    assert languages_of(chunks) == languages_of(loaded)

    # 3. Chunks stay usable as dict keys, which the whole ranking pipeline relies on.
    scores = {chunk: index for index, chunk in enumerate(loaded)}
    assert scores[chunks[2]] == 2


def test_out_of_range_index_raises(tmp_path: Path) -> None:
    """An index past the end raises IndexError like a list does."""
    save_chunks(tmp_path, _chunks())
    loaded = load_chunks(tmp_path)

    with pytest.raises(IndexError):
        loaded[len(loaded)]


def test_load_rejects_a_foreign_format(tmp_path: Path) -> None:
    """A chunk directory written by another format version is refused."""
    save_chunks(tmp_path, _chunks())
    (tmp_path / "chunks.json").write_text('{"format": 999, "n_chunks": 4}')

    with pytest.raises(ValueError, match="Unsupported chunk format"):
        load_chunks(tmp_path)


def test_load_rejects_columns_of_disagreeing_length(tmp_path: Path) -> None:
    """Columns that describe different numbers of chunks are refused."""
    save_chunks(tmp_path, _chunks())
    np.save(tmp_path / "path_ids.npy", np.array([0], dtype=np.int32))

    with pytest.raises(ValueError, match="inconsistent lengths"):
        load_chunks(tmp_path)


def test_empty_chunk_list_roundtrips(tmp_path: Path) -> None:
    """An empty index saves and loads without special-casing."""
    save_chunks(tmp_path, [])
    loaded = load_chunks(tmp_path)

    assert len(loaded) == 0
    assert list(loaded) == []

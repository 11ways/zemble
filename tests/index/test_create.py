from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import orjson
import pytest

from zemble.cache import load_previous_for_incremental
from zemble.chunking.capsule import CapsuleOptions
from zemble.index import create as create_module
from zemble.index.bm25 import BM25, BM25Writer
from zemble.index.chunk_store import ChunkList, load_chunks, save_chunks
from zemble.index.index import ZembleIndex
from zemble.index.types import PreviousIndex, make_chunk_id
from zemble.types import ContentType


def _write_files(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def _build(
    root: Path,
    embedder: Any,
    target: Path,
    previous: ZembleIndex | None = None,
    changed: list[Path] | None = None,
) -> ZembleIndex:
    """Write one generation of *root* into *target*, carrying files over from *previous*."""
    index, _written = ZembleIndex.build(
        root,
        embedder,
        target,
        content=(ContentType.CODE,),
        capsules=CapsuleOptions.resolve(None),
        previous=_previous(previous) if previous is not None else None,
        changed_paths=changed,
    )
    return index


def _previous(index: ZembleIndex) -> PreviousIndex:
    """Describe a loaded index as the generation a build carries files over from."""
    assert isinstance(index.chunks, ChunkList)
    return PreviousIndex(
        chunks=index.chunks,
        vectors=index._semantic_index.vectors,
        manifest=index._manifest,
        bm25_index=index._bm25_index,
        definitions=index._definitions,
    )


def test_incremental_reindex_reuses_updates_and_prunes(mock_embedder: Any, tmp_path: Path) -> None:
    """One incremental pass reuses unchanged vectors, re-embeds changes, and keeps BM25 slots current."""
    root = tmp_path / "src"
    _write_files(
        root,
        {
            "a.py": "def stable_anchor():\n    return 1\n",
            "b.py": "def changed_value():\n    return 2\n",
            "c.py": "def unique_gone():\n    return 3\n",
            "emptying.py": "def becomes_empty():\n    return 4\n",
        },
    )
    first = _build(root, mock_embedder, tmp_path / "one")
    a_vectors = np.array(first._semantic_index.vectors[first._manifest["a.py"].start : first._manifest["a.py"].end])
    b_vectors = np.array(first._semantic_index.vectors[first._manifest["b.py"].start : first._manifest["b.py"].end])

    # 1. A build over an unchanged tree embeds nothing and reproduces every vector.
    mock_embedder.document_calls.clear()
    unchanged = _build(root, mock_embedder, tmp_path / "two", previous=first)
    assert mock_embedder.document_calls == [], "1: nothing is embedded again"
    np.testing.assert_array_equal(
        np.asarray(unchanged._semantic_index.vectors), np.asarray(first._semantic_index.vectors)
    )

    # 2. An edit re-embeds only its file and never writes into the generation it reads from.
    (root / "b.py").write_text("def changed_value():\n    return 999\n")
    previous_vectors = np.array(unchanged._semantic_index.vectors)
    second = _build(root, mock_embedder, tmp_path / "three", previous=unchanged)
    np.testing.assert_array_equal(np.asarray(unchanged._semantic_index.vectors), previous_vectors)

    # 3. A deleted file, an emptied one and a new one, in one pass.
    (root / "c.py").unlink()
    (root / "emptying.py").write_text(" " * 128)
    _write_files(root, {"d.py": "def brand_new_term():\n    return 4\n"})
    after = _build(root, mock_embedder, tmp_path / "four", previous=second)
    manifest = after._manifest
    vectors = after._semantic_index.vectors
    np.testing.assert_array_equal(vectors[manifest["a.py"].start : manifest["a.py"].end], a_vectors)
    assert not np.array_equal(b_vectors, vectors[manifest["b.py"].start : manifest["b.py"].end])
    assert "c.py" not in manifest and "d.py" in manifest
    assert manifest["emptying.py"].count == 0
    bm25 = after._bm25_index
    assert bm25.get_scores(["unique_gone"]).sum() == 0
    assert bm25.get_scores(["becomes_empty"]).sum() == 0
    assert bm25.get_scores(["brand", "new", "term"]).sum() > 0
    expected_ids = [
        make_chunk_id(indexed_path, slot) for indexed_path, entry in manifest.items() for slot in range(entry.count)
    ]
    assert bm25.doc_order == expected_ids


def _build_valid_cache(index_path: Path, mock_embedder: Any) -> dict:
    """Build a real, well-formed on-disk index at *index_path* and return its metadata dict for mutation."""
    src = index_path.parent / "src"
    _write_files(src, {"a.py": "def a():\n    return 1\n", "b.py": "def b():\n    return 2\n"})
    _build(src, mock_embedder, index_path)
    return orjson.loads((index_path / "metadata.json").read_bytes())


def _reverse_bm25(index_path: Path) -> None:
    """Rewrite an index's BM25 store with its documents in reverse order."""
    bm25_path = index_path / "bm25_index"
    stored = BM25.load(bm25_path)
    writer = BM25Writer(index_path / "reversed", stored)
    for row in reversed(range(stored.document_count)):
        writer.reuse(row, 1)
    writer.finish()
    for written in (index_path / "reversed").iterdir():
        written.replace(bm25_path / written.name)


@pytest.mark.parametrize(
    "corrupt",
    [
        "missing_cache",
        "missing_files_key",
        "metadata_mismatch",
        "component_length_mismatch",
        "length_mismatch",
        "overlapping_entries",
        "bm25_order_mismatch",
        "corrupt_json",
    ],
)
def test_load_previous_for_incremental_fails_closed(corrupt: str, tmp_path: Path, mock_embedder: Any) -> None:
    """Any structurally invalid or missing cache state yields None instead of raising."""
    index_path = tmp_path / "index"

    if corrupt != "missing_cache":
        metadata = _build_valid_cache(index_path, mock_embedder)
        if corrupt == "missing_files_key":
            del metadata["files"]
        elif corrupt == "metadata_mismatch":
            metadata["embedder"] = "model2vec:other/model"
        elif corrupt == "component_length_mismatch":
            chunks_path = index_path / "chunks"
            save_chunks(chunks_path, list(load_chunks(chunks_path))[:-1])
        elif corrupt == "length_mismatch":
            metadata["files"]["a.py"]["count"] += 5
        elif corrupt == "overlapping_entries":
            metadata["files"]["b.py"]["start"] = metadata["files"]["a.py"]["start"]
        elif corrupt == "bm25_order_mismatch":
            _reverse_bm25(index_path)
        elif corrupt == "corrupt_json":
            (index_path / "metadata.json").write_bytes(b"{not json")
            with patch("zemble.cache.find_index_from_cache_folder", return_value=index_path):
                assert load_previous_for_incremental("/some/path", mock_embedder.model_id, [ContentType.CODE]) is None
            return
        (index_path / "metadata.json").write_bytes(orjson.dumps(metadata))

    with patch("zemble.cache.find_index_from_cache_folder", return_value=index_path):
        result = load_previous_for_incremental("/some/path", mock_embedder.model_id, [ContentType.CODE])
    assert result is None


def test_load_previous_for_incremental_happy_path(mock_embedder: Any, tmp_path: Path) -> None:
    """A well-formed cache round-trips into a usable PreviousIndex."""
    index_path = tmp_path / "cache" / "index"
    _build_valid_cache(index_path, mock_embedder)

    with patch("zemble.cache.find_index_from_cache_folder", return_value=index_path):
        previous = load_previous_for_incremental(
            str(index_path.parent / "src"), mock_embedder.model_id, [ContentType.CODE]
        )

    assert previous is not None
    assert len(previous.chunks) == previous.vectors.shape[0] == len(previous.bm25_index.doc_order)
    assert "a.py" in previous.manifest


def test_change_set_build_matches_a_full_walk(mock_embedder: Any, tmp_path: Path) -> None:
    """A build driven by a change set indexes exactly what a re-walk would, without walking."""
    root = tmp_path / "src"
    _write_files(
        root,
        {
            "a.py": "def stable_anchor():\n    return 1\n",
            "b.py": "def changed_value():\n    return 2\n",
            "gone.py": "def disappearing_helper():\n    return 3\n",
        },
    )
    first = _build(root, mock_embedder, tmp_path / "one")

    # 1. One file is edited, one deleted and one added: the watcher names all three.
    (root / "b.py").write_text("def changed_value():\n    return 999\n")
    (root / "gone.py").unlink()
    _write_files(root, {"new.py": "def freshly_arrived_symbol():\n    return 4\n"})
    changed = [root / "b.py", root / "gone.py", root / "new.py"]

    with patch("zemble.index.create.walk_entries", side_effect=AssertionError("the tree must not be walked")):
        after = _build(root, mock_embedder, tmp_path / "two", previous=first, changed=changed)

    manifest_after = after._manifest
    assert "gone.py" not in manifest_after, "1: a deleted file leaves the manifest"
    assert "new.py" in manifest_after and "a.py" in manifest_after, "1: the new file arrived, the old one stayed"
    assert after._bm25_index.get_scores(["disappearing_helper"]).sum() == 0, "1: and its postings are gone"
    assert after._bm25_index.get_scores(["freshly", "arrived", "symbol"]).sum() > 0, "1: the new file is searchable"

    # 2. A full walk over the same tree produces the same index, chunk for chunk.
    walked = _build(root, mock_embedder, tmp_path / "three")
    assert {path: entry.count for path, entry in manifest_after.items()} == {
        path: entry.count for path, entry in walked._manifest.items()
    }, "2: the same files with the same chunk counts"
    assert sorted(chunk.content for chunk in after.chunks) == sorted(chunk.content for chunk in walked.chunks), (
        "2: and the same chunk content"
    )
    assert sorted(after._bm25_index.doc_order) == sorted(walked._bm25_index.doc_order), "2: over the same documents"
    for query in (["stable_anchor"], ["changed_value"], ["freshly", "arrived", "symbol"]):
        np.testing.assert_allclose(
            np.sort(after._bm25_index.get_scores(query))[-3:],
            np.sort(walked._bm25_index.get_scores(query))[-3:],
            atol=1e-6,
        )
    assert after._semantic_index.vectors.shape == walked._semantic_index.vectors.shape, "2: the same matrix shape"


def test_change_set_ignores_paths_the_walk_would_never_reach(mock_embedder: Any, tmp_path: Path) -> None:
    """A named path that is ignored, foreign or not a source file is refused, not indexed."""
    root = tmp_path / "src"
    _write_files(root, {"a.py": "def stable_anchor():\n    return 1\n", ".gitignore": "secret.py\n"})
    _write_files(
        root,
        {
            "secret.py": "def ignored_helper():\n    return 1\n",
            "build/generated.py": "def generated_helper():\n    return 1\n",
            "notes.txt": "not code\n",
        },
    )
    first = _build(root, mock_embedder, tmp_path / "one")
    after = _build(
        root,
        mock_embedder,
        tmp_path / "two",
        previous=first,
        changed=[root / "secret.py", root / "build" / "generated.py", root / "notes.txt", Path("/elsewhere/other.py")],
    )
    assert set(after._manifest) == set(first._manifest), "nothing the walk skips is let in through the change set"


def test_a_rebuild_leaves_the_previous_generation_untouched(mock_embedder: Any, tmp_path: Path) -> None:
    """The index a rebuild starts from keeps answering exactly as it did: nothing writes into it."""
    root = tmp_path / "src"
    _write_files(root, {"a.py": "def stable_anchor():\n    return 1\n"})
    served = _build(root, mock_embedder, tmp_path / "served")
    before = served._bm25_index.get_scores(["stable_anchor"]).copy()
    files_before = {path: path.read_bytes() for path in (tmp_path / "served").rglob("*") if path.is_file()}

    _write_files(root, {"b.py": "def brand_new_term():\n    return 2\n"})
    after = _build(root, mock_embedder, tmp_path / "next", previous=served, changed=[root / "b.py"])

    np.testing.assert_array_equal(served._bm25_index.get_scores(["stable_anchor"]), before)
    assert served._bm25_index.get_scores(["brand", "new", "term"]).sum() == 0, "the old index never saw the new file"
    assert after._bm25_index.get_scores(["brand", "new", "term"]).sum() > 0, "the new one did"
    assert {path: path.read_bytes() for path in (tmp_path / "served").rglob("*") if path.is_file()} == files_before


def test_publishing_replaces_files_a_reader_has_mapped(mock_embedder: Any, tmp_path: Path) -> None:
    """A generation published over the one a process has mapped never truncates what it reads."""
    root = tmp_path / "src"
    _write_files(root, {"a.py": "def stable_anchor():\n    return 1\n"})
    final = tmp_path / "index"
    served = _build(root, mock_embedder, final)
    inode = (final / "bm25_index" / "posting_docs.npy").stat().st_ino
    before = served._bm25_index.get_scores(["stable_anchor"]).copy()

    _write_files(root, {"b.py": "def stable_anchor_two():\n    return 2\n"})
    _build(root, mock_embedder, final, previous=served)

    assert (final / "bm25_index" / "posting_docs.npy").stat().st_ino != inode, "the column was replaced"
    np.testing.assert_array_equal(served._bm25_index.get_scores(["stable_anchor"]), before)
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".staging-")], "no staging folder left"


def test_vectors_are_written_in_bounded_batches_and_unit_length(
    mock_embedder: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh rows are embedded batch by batch and reused rows copied block by block, all unit length."""
    from zemble.index import create

    class Scaled:
        """The fake embedder, returning every vector five times too long."""

        model_id = mock_embedder.model_id
        dimensions = mock_embedder.dimensions

        def __init__(self) -> None:
            self.batches: list[int] = []

        def embed_documents(self, texts: list[str]) -> np.ndarray:
            self.batches.append(len(texts))
            return mock_embedder.embed_documents(texts) * 5

    monkeypatch.setattr(create, "_EMBED_ROWS", 3)
    monkeypatch.setattr(create, "_COPY_ROWS", 2)
    root = tmp_path / "src"
    _write_files(root, {f"m{n}.py": f"def function_{n}():\n    return {n}\n" for n in range(8)})
    scaled = Scaled()

    # 1. A cold build embeds every row, never more than one batch per provider round.
    first = _build(root, scaled, tmp_path / "one")
    vectors = np.asarray(first._semantic_index.vectors)
    assert len(vectors) == 8 and max(scaled.batches) == 3 and sum(scaled.batches) == 8, "1: batched"
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, rtol=1e-6, err_msg="1: unit rows")
    texts = [create.embedding_text(chunk) for chunk in first.chunks]
    np.testing.assert_allclose(vectors, mock_embedder.embed_documents(texts), rtol=1e-6, err_msg="1: the right rows")

    # 2. An edit embeds one row and copies the other seven, in blocks, into the same places.
    (root / "m3.py").write_text("def function_three():\n    return 33\n")
    scaled.batches.clear()
    second = _build(root, scaled, tmp_path / "two", previous=first)
    assert scaled.batches == [1], "2: only the edited file is embedded"
    texts = [create.embedding_text(chunk) for chunk in second.chunks]
    np.testing.assert_allclose(
        np.asarray(second._semantic_index.vectors), mock_embedder.embed_documents(texts), rtol=1e-6
    )


@pytest.mark.parametrize("kernel", [True, False], ids=["kernel-copy", "memory-copy"])
def test_reused_runs_broken_by_a_deletion_keep_every_row_in_place(
    mock_embedder: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kernel: bool
) -> None:
    """Unchanged files on either side of a deleted one keep their exact chunks, postings and vectors.

    Both ways of copying reused vector rows are held to it: file to file in the kernel, and through
    memory where the kernel cannot.
    """
    copies: list[bool] = []
    real = create_module._copy_in_kernel

    def copy(*args: Any) -> bool:
        copies.append(kernel and real(*args))
        return copies[-1]

    monkeypatch.setattr(create_module, "_copy_in_kernel", copy)
    root = tmp_path / "src"
    _write_files(root, {f"f{index}.py": f"def symbol_number_{index}():\n    return {index}\n" for index in range(6)})
    first = _build(root, mock_embedder, tmp_path / "one")

    # 1. Deleting files in the middle splits the previous rows into runs that must not be joined.
    for name in ("f2.py", "f4.py"):
        (root / name).unlink()
    after = _build(root, mock_embedder, tmp_path / "two", previous=first, changed=[root / "f2.py", root / "f4.py"])
    assert sorted(after._manifest) == ["f0.py", "f1.py", "f3.py", "f5.py"], "1: only the deleted files left"
    assert copies == [kernel], "1: the reused rows went through the copy under test, and it succeeded"
    assert after._semantic_index.vectors.offset == 4096, "1: rows start on a 4 KiB block"

    # 2. Every kept file's rows hold what its previous rows held, chunk, vector and postings.
    for path, entry in after._manifest.items():
        was = first._manifest[path]
        for offset in range(entry.count):
            row, old = entry.start + offset, was.start + offset
            assert after.chunks[row].content == first.chunks[old].content, f"2: {path} chunk {offset} moved intact"
            np.testing.assert_array_equal(
                after._semantic_index.vectors[row], first._semantic_index.vectors[old], err_msg=f"2: {path} vector"
            )
        number = path[1]
        assert after._bm25_index.get_scores([f"symbol_number_{number}"])[entry.start] > 0, f"2: {path} postings"

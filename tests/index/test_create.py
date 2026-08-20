from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import orjson
import pytest

from zemble.cache import load_previous_for_incremental
from zemble.index.bm25 import BM25
from zemble.index.chunk_store import load_chunks, save_chunks
from zemble.index.create import create_index_from_path
from zemble.index.index import ZembleIndex
from zemble.index.types import PreviousIndex, make_chunk_id
from zemble.types import ContentType


def _write_files(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def test_incremental_reindex_reuses_updates_and_prunes(mock_embedder: Any, tmp_path: Path) -> None:
    """One incremental pass reuses unchanged vectors, re-embeds changes, and keeps BM25 slots current."""
    _write_files(
        tmp_path,
        {
            "a.py": "def stable_anchor():\n    return 1\n",
            "b.py": "def changed_value():\n    return 2\n",
            "c.py": "def unique_gone():\n    return 3\n",
            "emptying.py": "def becomes_empty():\n    return 4\n",
        },
    )
    bm25_before, semantic_before, chunks_before, manifest_before = create_index_from_path(
        tmp_path, mock_embedder, display_root=tmp_path
    )
    a_entry = manifest_before["a.py"]
    b_entry = manifest_before["b.py"]
    a_vectors_before = semantic_before.vectors[a_entry.start : a_entry.end].copy()
    b_vectors_before = semantic_before.vectors[b_entry.start : b_entry.end].copy()
    previous = PreviousIndex(
        chunks=chunks_before,
        vectors=semantic_before.vectors,
        manifest=manifest_before,
        bm25_index=bm25_before,
    )
    _, semantic_unchanged, _, _ = create_index_from_path(
        tmp_path, mock_embedder, display_root=tmp_path, previous=previous
    )
    assert semantic_unchanged.vectors is semantic_before.vectors

    (tmp_path / "b.py").write_text("def changed_value():\n    return 999\n")
    bm25_before, semantic_before, chunks_before, manifest_before = create_index_from_path(
        tmp_path, mock_embedder, display_root=tmp_path, previous=previous
    )
    assert semantic_before.vectors is previous.vectors
    previous = PreviousIndex(
        chunks=chunks_before,
        vectors=semantic_before.vectors,
        manifest=manifest_before,
        bm25_index=bm25_before,
    )

    (tmp_path / "c.py").unlink()
    (tmp_path / "emptying.py").write_text(" " * 128)
    _write_files(tmp_path, {"d.py": "def brand_new_term():\n    return 4\n"})
    bm25_after, semantic_after, _, manifest_after = create_index_from_path(
        tmp_path, mock_embedder, display_root=tmp_path, previous=previous
    )

    a_entry_after = manifest_after["a.py"]
    b_entry_after = manifest_after["b.py"]
    np.testing.assert_array_equal(semantic_after.vectors[a_entry_after.start : a_entry_after.end], a_vectors_before)
    assert not np.array_equal(
        b_vectors_before,
        semantic_after.vectors[b_entry_after.start : b_entry_after.end],
    )
    assert "c.py" not in manifest_after
    assert "d.py" in manifest_after
    assert manifest_after["emptying.py"].count == 0
    assert bm25_after.get_scores(["unique_gone"]).sum() == 0
    assert bm25_after.get_scores(["becomes_empty"]).sum() == 0
    assert bm25_after.get_scores(["brand", "new", "term"]).sum() > 0
    expected_ids = {
        make_chunk_id(indexed_path, slot)
        for indexed_path, entry in manifest_after.items()
        for slot in range(entry.count)
    }
    assert set(bm25_after.doc_order) == expected_ids


def _build_valid_cache(index_path: Path, mock_embedder: Any) -> dict:
    """Build a real, well-formed on-disk index and return its metadata dict for mutation."""
    src = index_path.parent / "src"
    _write_files(src, {"a.py": "def a():\n    return 1\n", "b.py": "def b():\n    return 2\n"})

    with patch("zemble.index.index.load_embedder", return_value=mock_embedder):
        ZembleIndex.from_path(src).save(index_path)
    return orjson.loads((index_path / "metadata.json").read_bytes())


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
            bm25_path = index_path / "bm25_index"
            bm25 = BM25.load(bm25_path)
            bm25.set_doc_order(list(reversed(bm25.doc_order)))
            bm25.save(bm25_path)
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
    _write_files(
        tmp_path,
        {
            "a.py": "def stable_anchor():\n    return 1\n",
            "b.py": "def changed_value():\n    return 2\n",
            "gone.py": "def disappearing_helper():\n    return 3\n",
        },
    )
    bm25, semantic, chunks, manifest = create_index_from_path(tmp_path, mock_embedder, display_root=tmp_path)
    previous = PreviousIndex(chunks=chunks, vectors=semantic.vectors, manifest=manifest, bm25_index=bm25)

    # 1. One file is edited, one deleted and one added: the watcher names all three.
    (tmp_path / "b.py").write_text("def changed_value():\n    return 999\n")
    (tmp_path / "gone.py").unlink()
    _write_files(tmp_path, {"new.py": "def freshly_arrived_symbol():\n    return 4\n"})
    changed = [tmp_path / "b.py", tmp_path / "gone.py", tmp_path / "new.py"]

    with patch("zemble.index.create.walk_entries", side_effect=AssertionError("the tree must not be walked")):
        bm25_after, semantic_after, chunks_after, manifest_after = create_index_from_path(
            tmp_path, mock_embedder, display_root=tmp_path, previous=previous, changed_paths=changed
        )

    assert "gone.py" not in manifest_after, "1: a deleted file leaves the manifest"
    assert "new.py" in manifest_after and "a.py" in manifest_after, "1: the new file arrived, the old one stayed"
    assert bm25_after.get_scores(["disappearing_helper"]).sum() == 0, "1: and its postings are gone"
    assert bm25_after.get_scores(["freshly", "arrived", "symbol"]).sum() > 0, "1: the new file is searchable"

    # 2. A full walk over the same tree produces the same index, chunk for chunk.
    walked_bm25, walked_semantic, walked_chunks, walked_manifest = create_index_from_path(
        tmp_path, mock_embedder, display_root=tmp_path
    )
    assert {path: entry.count for path, entry in manifest_after.items()} == {
        path: entry.count for path, entry in walked_manifest.items()
    }, "2: the same files with the same chunk counts"
    assert sorted(chunk.content for chunk in chunks_after) == sorted(chunk.content for chunk in walked_chunks), (
        "2: and the same chunk content"
    )
    assert sorted(bm25_after.doc_order) == sorted(walked_bm25.doc_order), "2: over the same documents"
    for query in (["stable_anchor"], ["changed_value"], ["freshly", "arrived", "symbol"]):
        np.testing.assert_allclose(
            np.sort(bm25_after.get_scores(query))[-3:], np.sort(walked_bm25.get_scores(query))[-3:], atol=1e-6
        )
    assert semantic_after.vectors.shape == walked_semantic.vectors.shape, "2: and the same vector matrix shape"


def test_change_set_ignores_paths_the_walk_would_never_reach(mock_embedder: Any, tmp_path: Path) -> None:
    """A named path that is ignored, foreign or not a source file is refused, not indexed."""
    _write_files(tmp_path, {"a.py": "def stable_anchor():\n    return 1\n", ".gitignore": "secret.py\n"})
    _write_files(
        tmp_path,
        {
            "secret.py": "def ignored_helper():\n    return 1\n",
            "build/generated.py": "def generated_helper():\n    return 1\n",
            "notes.txt": "not code\n",
        },
    )
    bm25, semantic, chunks, manifest = create_index_from_path(tmp_path, mock_embedder, display_root=tmp_path)
    previous = PreviousIndex(chunks=chunks, vectors=semantic.vectors, manifest=manifest, bm25_index=bm25)

    _, _, _, manifest_after = create_index_from_path(
        tmp_path,
        mock_embedder,
        display_root=tmp_path,
        previous=previous,
        changed_paths=[
            tmp_path / "secret.py",
            tmp_path / "build" / "generated.py",
            tmp_path / "notes.txt",
            Path("/elsewhere/other.py"),
        ],
    )
    assert set(manifest_after) == set(manifest), "nothing the walk skips is let in through the change set"


def test_a_rebuild_leaves_the_previous_bm25_index_untouched(mock_embedder: Any, tmp_path: Path) -> None:
    """The index a rebuild starts from keeps answering exactly as it did: nothing mutates it."""
    _write_files(tmp_path, {"a.py": "def stable_anchor():\n    return 1\n"})
    bm25, semantic, chunks, manifest = create_index_from_path(tmp_path, mock_embedder, display_root=tmp_path)
    bm25.save(tmp_path / "postings")
    served = BM25.load(tmp_path / "postings")
    previous = PreviousIndex(chunks=chunks, vectors=semantic.vectors, manifest=manifest, bm25_index=served)
    before = served.get_scores(["stable_anchor"]).copy()

    _write_files(tmp_path, {"b.py": "def brand_new_term():\n    return 2\n"})
    bm25_after, _semantic, _chunks, _manifest = create_index_from_path(
        tmp_path, mock_embedder, display_root=tmp_path, previous=previous, changed_paths=[tmp_path / "b.py"]
    )

    assert bm25_after is not served, "the rebuild produced a new index"
    np.testing.assert_array_equal(served.get_scores(["stable_anchor"]), before)
    assert served.get_scores(["brand", "new", "term"]).sum() == 0, "the old index never saw the new file"
    assert bm25_after.get_scores(["brand", "new", "term"]).sum() > 0, "the new one did"
    assert served.document_count == len(chunks), "and the old one still holds exactly its own documents"

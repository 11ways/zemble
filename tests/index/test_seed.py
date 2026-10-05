"""A clone's first build borrows its sibling checkout's rows for every file whose bytes match."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import FakeEmbedder
from zemble.index.index import ZembleIndex

ORIGIN = "git@example.com:team/repo.git"
#: Files of a repository nested in the checkout, as the workspace's `zenit/` or `hawkeye/` are:
#: the capsule names them `lib/...` whatever folder the checkout itself lives in.
FILES = {f"lib/m{index}.py": f"def function_number_{index}():\n    return {index}\n" for index in range(5)}
#: A file of the top-level repository, which the capsule names after the checkout's own folder.
TOP = {"top.py": "def top_level_helper():\n    return 0\n"}


def _checkout(root: Path, origin: str, files: dict[str, str]) -> Path:
    """Write a git checkout with an `origin` remote holding a nested `lib` repository and the given files."""
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    for repository in (root, root / "lib"):
        subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(root), "remote", "add", "origin", origin], check=True)
    return root.resolve()


def _embedded(embedder: FakeEmbedder) -> list[str]:
    """Every document text the embedder was asked for since its calls were last cleared."""
    return [text for call in embedder.document_calls for text in call]


def _rows(index: ZembleIndex) -> dict[str, list[tuple[str, bytes]]]:
    """Each file's chunks with their vectors, which is what two equal indexes agree on."""
    vectors = index._semantic_index.vectors
    return {
        path: [
            (index.chunks[row].content, np.asarray(vectors[row]).tobytes())
            for row in range(entry.start, entry.start + entry.count)
        ]
        for path, entry in index._manifest.items()
    }


def test_a_clone_borrows_its_siblings_rows_for_every_unchanged_file(
    mock_embedder: FakeEmbedder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only what differs from the sibling is chunked and embedded, and the result is a normal index."""
    main = _checkout(tmp_path / "main", ORIGIN, {**FILES, **TOP})
    ZembleIndex.from_path(main, embedder=mock_embedder)

    # 1. A clone with one file edited and one added embeds those two, and the top-level file whose
    #    capsule names the clone's own folder, and nothing else.
    edited = {
        **FILES,
        **TOP,
        "lib/m1.py": "def function_number_1():\n    return 'edited'\n",
        "lib/extra.py": "def extra_one():\n    pass\n",
    }
    clone = _checkout(tmp_path / "clone", ORIGIN, edited)
    mock_embedder.document_calls.clear()
    seeded = ZembleIndex.from_path(clone, embedder=mock_embedder)
    embedded = _embedded(mock_embedder)
    assert any("'edited'" in text for text in embedded) and any("extra_one" in text for text in embedded), (
        "1: the edited and the new file are embedded"
    )
    assert any("top_level_helper" in text for text in embedded), "1: a top-level file chunks under the clone's name"
    assert not any("function_number_3" in text for text in embedded), "1: an unchanged file is borrowed"

    # 2. The seeded index holds exactly what the same clone built alone, borrowing nothing, holds.
    monkeypatch.setenv("ZEMBLE_CACHE_LOCATION", str(tmp_path / "alone"))
    mock_embedder.document_calls.clear()
    reference = ZembleIndex.from_path(clone, embedder=mock_embedder)
    assert any("function_number_3" in text for text in _embedded(mock_embedder)), "2: alone it borrows nothing"
    assert _rows(seeded) == _rows(reference), "2: the same chunks and vectors, file by file"

    # 3. The clone records its own modification times, so its next build reuses by them.
    for path, entry in seeded._manifest.items():
        assert entry.mtime_ns == (clone / path).stat().st_mtime_ns, f"3: {path} carries the clone's own mtime"


def test_a_sibling_file_edited_since_its_index_was_built_is_not_borrowed(
    mock_embedder: FakeEmbedder, tmp_path: Path
) -> None:
    """Matching bytes are not enough: the sibling's index must still describe the file it lends."""
    main = _checkout(tmp_path / "main", ORIGIN, FILES)
    ZembleIndex.from_path(main, embedder=mock_embedder)
    rewritten = "def function_number_3():\n    return 'rewritten'\n"
    (main / "lib/m3.py").write_text(rewritten, encoding="utf-8")
    later = (main / "lib/m3.py").stat().st_mtime_ns + 1_000_000_000
    os.utime(main / "lib/m3.py", ns=(later, later))

    clone = _checkout(tmp_path / "clone", ORIGIN, {**FILES, "lib/m3.py": rewritten})
    mock_embedder.document_calls.clear()
    ZembleIndex.from_path(clone, embedder=mock_embedder)
    embedded = _embedded(mock_embedder)
    assert any("'rewritten'" in text for text in embedded), "the stale sibling row was not lent"
    assert not any("function_number_2" in text for text in embedded), "the current ones were"


def test_a_checkout_of_another_origin_lends_nothing(mock_embedder: FakeEmbedder, tmp_path: Path) -> None:
    """Only a checkout of the same repository is a sibling, whatever files the two share."""
    ZembleIndex.from_path(_checkout(tmp_path / "main", ORIGIN, FILES), embedder=mock_embedder)
    other = _checkout(tmp_path / "other", "git@example.com:someone/else.git", FILES)
    mock_embedder.document_calls.clear()
    ZembleIndex.from_path(other, embedder=mock_embedder)
    assert len(_embedded(mock_embedder)) == len(FILES), "every file of the unrelated checkout is embedded"

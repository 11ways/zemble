from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.conftest import FakeEmbedder
from zemble import ZembleIndex
from zemble.cache import get_validated_cache
from zemble.types import ContentType


@pytest.fixture
def odd_embedder() -> FakeEmbedder:
    """An embedder at a deliberately non-256 width, so any hardcoded 256 fails loudly."""
    return FakeEmbedder(dimensions=313, model_id="fake:odd")


def test_index_and_search_at_a_non_standard_dimension(tmp_project: Path, odd_embedder: FakeEmbedder) -> None:
    """An index built at 313 dimensions searches, persists, reloads and reports its embedder."""
    # 1. Build: every vector must be the embedder's width, not the historical 256.
    with patch("zemble.index.index.load_embedder", return_value=odd_embedder):
        index = ZembleIndex.from_path(tmp_project)
    assert index._semantic_index.vectors.shape[1] == 313, "step 1: index width must follow the embedder"
    assert index.stats.embedder == "fake:odd@313", "step 1: stats must name the embedder"
    assert index.stats.dimensions == 313, "step 1: stats must report the width"

    # 2. Search: the query goes through embed_queries, not embed_documents.
    before = len(odd_embedder.query_calls)
    results = index.search("authenticate token", top_k=3)
    assert results, "step 2: a real query must return results"
    assert len(odd_embedder.query_calls) == before + 1, "step 2: the query must use embed_queries"

    # 3. Persist: the metadata records the embedder id and width instead of a model path.
    save_path = tmp_project / ".index"
    index.save(save_path)
    metadata = json.loads((save_path / "metadata.json").read_text())
    assert metadata["embedder"] == "fake:odd@313"
    assert metadata["dimensions"] == 313
    assert "model_path" not in metadata, "step 3: the legacy field must be gone"

    # 4. Reload: the same chunks and vectors come back.
    reloaded = ZembleIndex.load_from_disk(save_path, embedder=odd_embedder)
    assert reloaded.chunks == index.chunks
    assert reloaded._semantic_index.vectors.shape == index._semantic_index.vectors.shape

    # 5. Refuse: a cache built by another embedder is never reused.
    with patch("zemble.cache.find_index_from_cache_folder", return_value=save_path):
        assert get_validated_cache(str(tmp_project), "fake:odd@313", [ContentType.CODE]) is not None
        assert get_validated_cache(str(tmp_project), "voyage:voyage-code-4@256", [ContentType.CODE]) is None


def test_documents_are_embedded_once_per_full_build(tmp_project: Path, odd_embedder: FakeEmbedder) -> None:
    """A full build sends every chunk to the provider exactly once, in a single call."""
    with patch("zemble.index.index.load_embedder", return_value=odd_embedder):
        index = ZembleIndex.from_path(tmp_project)
    assert len(odd_embedder.document_calls) == 1, "a full build must be one batched call, not one per file"
    assert len(odd_embedder.document_calls[0]) == len(index.chunks)

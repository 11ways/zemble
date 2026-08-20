from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from zemble.embedding.cache import CachingEmbedder, EmbeddingCache, family_slug, text_hash


class CountingEmbedder:
    """An embedder that records every text it was actually asked to embed."""

    def __init__(self, dimensions: int = 8) -> None:
        """Initialise with a deterministic, dimension-dependent vector generator."""
        self._dimensions = dimensions
        self.document_batches: list[list[str]] = []
        self.query_batches: list[list[str]] = []

    @property
    def model_id(self) -> str:
        """The normalized spec string."""
        return f"fake:counting@{self._dimensions}"

    @property
    def dimensions(self) -> int:
        """The vector width."""
        return self._dimensions

    def _vectors(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Return one unit vector per text, derived from the text itself."""
        rows = []
        for text in texts:
            vector = np.arange(1, self._dimensions + 1, dtype=np.float32) * (len(text) + 1)
            rows.append(vector / np.linalg.norm(vector))
        return np.asarray(rows, dtype=np.float32).reshape(len(texts), self._dimensions)

    def embed_documents(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed documents, recording the batch."""
        self.document_batches.append(list(texts))
        return self._vectors(texts)

    def embed_queries(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed queries, recording the batch."""
        self.query_batches.append(list(texts))
        return self._vectors(texts)


def test_cache_journey(tmp_path: Path) -> None:
    """A caching embedder walks miss -> hit -> partial miss -> matryoshka slice -> query passthrough."""
    inner = CountingEmbedder(dimensions=8)
    embedder = CachingEmbedder(inner, "fake:counting", tmp_path)

    # 1. Cold: everything is a miss and reaches the provider exactly once.
    first = embedder.embed_documents(["alpha", "beta"])
    assert inner.document_batches == [["alpha", "beta"]], "step 1: both texts must go to the provider"
    assert first.shape == (2, 8), "step 1: shape must be (texts, dimensions)"

    # 2. Warm: the identical call costs nothing and returns the identical vectors.
    second = embedder.embed_documents(["alpha", "beta"])
    assert inner.document_batches == [["alpha", "beta"]], "step 2: a full hit must not call the provider"
    np.testing.assert_allclose(second, first, rtol=0, atol=1e-6)

    # 3. Partial: only the new text is sent, and order is preserved.
    third = embedder.embed_documents(["beta", "gamma", "alpha"])
    assert inner.document_batches[-1] == ["gamma"], "step 3: only the miss may be embedded"
    np.testing.assert_allclose(third[2], first[0], rtol=0, atol=1e-6)

    # 4. Duplicates inside one call share a single provider slot.
    inner.document_batches.clear()
    fourth = embedder.embed_documents(["delta", "delta"])
    assert inner.document_batches == [["delta"]], "step 4: a repeated text must be embedded once"
    np.testing.assert_allclose(fourth[0], fourth[1], rtol=0, atol=1e-6)

    # 5. Matryoshka: a narrower request slices the stored wider vector without a provider call.
    narrow_inner = CountingEmbedder(dimensions=4)
    narrow = CachingEmbedder(narrow_inner, "fake:counting", tmp_path)
    sliced = narrow.embed_documents(["alpha"])
    assert narrow_inner.document_batches == [], "step 5: a wider stored vector must satisfy a narrower request"
    expected = first[0][:4] / np.linalg.norm(first[0][:4])
    np.testing.assert_allclose(sliced[0], expected, rtol=0, atol=1e-6)

    # 6. The slice is derived, not stored: the narrow width stays absent from the file.
    assert narrow.cache.path == embedder.cache.path, "step 6: one family, one file"
    with narrow.cache._connection as connection:
        rows = connection.execute(
            "SELECT COUNT(*) FROM embeddings WHERE text_sha256 = ? AND dims = 4", (text_hash("alpha"),)
        ).fetchone()
    assert rows[0] == 0, "step 6: a derivable slice must not be stored"

    # 7. Queries never touch the cache, because an asymmetric provider answers them differently.
    embedder.embed_queries(["alpha"])
    assert inner.query_batches == [["alpha"]], "step 7: a query must always reach the provider"


def test_widening_still_calls_the_provider(tmp_path: Path) -> None:
    """A wider request cannot be derived from a narrower stored vector, so it is a real miss."""
    narrow = CountingEmbedder(dimensions=4)
    CachingEmbedder(narrow, "fake:counting", tmp_path).embed_documents(["alpha"])

    wide = CountingEmbedder(dimensions=16)
    CachingEmbedder(wide, "fake:counting", tmp_path).embed_documents(["alpha"])
    assert wide.document_batches == [["alpha"]]


def test_empty_input_returns_a_typed_empty_matrix(tmp_path: Path) -> None:
    """Embedding nothing returns an empty matrix of the right width without calling the provider."""
    inner = CountingEmbedder(dimensions=8)
    result = CachingEmbedder(inner, "fake:counting", tmp_path).embed_documents([])
    assert result.shape == (0, 8)
    assert result.dtype == np.float32
    assert inner.document_batches == []


def test_families_do_not_share_a_file(tmp_path: Path) -> None:
    """Two embedder families get two sqlite files, so a slice can never cross model boundaries."""
    voyage = EmbeddingCache("voyage:voyage-code-4", tmp_path)
    openai = EmbeddingCache("openai:http://localhost:11434/v1#nomic-embed-text", tmp_path)
    assert voyage.path != openai.path
    voyage.put_many([("abc", 4, np.ones(4, dtype=np.float32))])
    assert openai.get("abc", 4) is None
    assert voyage.get("abc", 4) is not None


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        ("voyage:voyage-code-4", "voyage-voyage-code-4"),
        ("openai:http://localhost:11434/v1#nomic-embed-text", "openai-http-localhost-11434-v1-nomic-embed-text"),
    ],
)
def test_family_slug(family: str, expected: str) -> None:
    """A family key becomes a readable, filesystem-safe filename."""
    assert family_slug(family) == expected


def test_family_slug_shortens_long_names() -> None:
    """An absurdly long family key is truncated and disambiguated by hash."""
    slug = family_slug("openai:" + "x" * 500 + "#model")
    assert len(slug) <= 80
    assert slug != family_slug("openai:" + "y" * 500 + "#model")


def test_a_failure_mid_call_keeps_the_vectors_already_paid_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Misses are flushed to sqlite per slice, so a provider failure loses only the slice in flight."""
    monkeypatch.setattr("zemble.embedding.cache.FLUSH_EVERY", 2)
    texts = [f"chunk {index}" for index in range(6)]

    class FailsOnTheThirdSlice(CountingEmbedder):
        """Serves two slices, then refuses."""

        def embed_documents(self, batch: list[str]) -> npt.NDArray[np.float32]:
            """Fail once two slices have been served."""
            if len(self.document_batches) == 2:
                raise RuntimeError("provider went away")
            return super().embed_documents(batch)

    # 1. The provider dies part-way through a single logical call.
    failing = FailsOnTheThirdSlice(dimensions=8)
    embedder = CachingEmbedder(failing, "fake:counting", tmp_path)
    with pytest.raises(RuntimeError):
        embedder.embed_documents(texts)
    assert failing.document_batches == [texts[0:2], texts[2:4]], "step 1: two slices must have been served"

    # 2. The four vectors that were paid for survive in sqlite.
    cache = EmbeddingCache("fake:counting", tmp_path)
    stored = [text for text in texts if cache.get(text_hash(text), 8) is not None]
    assert stored == texts[:4], "step 2: every flushed slice must be readable back"

    # 3. A retry only pays for what is still missing.
    healthy = CountingEmbedder(dimensions=8)
    CachingEmbedder(healthy, "fake:counting", tmp_path).embed_documents(texts)
    assert healthy.document_batches == [texts[4:6]], "step 3: the retry must ask only for the unflushed tail"

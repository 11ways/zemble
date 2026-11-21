from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from zemble.chunking.capsule import embedding_text
from zemble.embedding.cache import CachingEmbedder, EmbeddingCache, family_slug
from zemble.embedding.keys import cache_key, text_hash
from zemble.types import Chunk


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


def _moved(context: str, content: str = "def moved(): pass") -> str:
    """The embedding text of one chunk carrying the capsule of some path."""
    return embedding_text(Chunk(content=content, file_path="x.py", start_line=1, end_line=1, context=context))


def test_a_chunk_under_several_paths_is_one_purchase_and_an_older_purchase_is_reused(tmp_path: Path) -> None:
    """Copies of a chunk share one key and one slot; a vector stored under its whole-text key is never bought again."""
    inner = CountingEmbedder()
    embedder = CachingEmbedder(inner, "fake:counting", tmp_path)
    lane, clone = _moved("lane/basalt/a.py lane basalt a py | def moved"), _moved("clone/a.py clone a py | def moved")

    # 1. One batch holding the chunk under two paths buys it once, as the first text, path included.
    vectors = embedder.embed_documents([lane, "other", clone])
    assert inner.document_batches == [[lane, "other"]], "step 1"
    np.testing.assert_allclose(vectors[2], vectors[0])
    assert embedder.pending_documents([clone, lane]) == [], "step 1: the bill guard agrees"

    # 2. A vector an older zemble stored under the whole text serves the chunk, is copied to its key, buys nothing.
    old = _moved("old/a.py old a py | def older", "def older(): pass")
    stored = np.full(8, 1 / np.sqrt(8), dtype=np.float32)
    embedder.cache.put_many([(text_hash(old), 8, stored)])
    assert embedder.pending_documents([old]) == [] and embedder.stored_keys([old]) == {cache_key(old)}, "step 2"
    np.testing.assert_allclose(embedder.embed_documents([old, old])[1], stored)
    assert len(inner.document_batches) == 1, "step 2: nothing was bought"
    assert embedder.cache.covered([cache_key(old), text_hash(old)], 8) == {cache_key(old), text_hash(old)}, (
        "step 2: the copy is stored under the key; the old row stays for gc"
    )

    # 3. Rekeying again copies nothing.
    assert embedder.cache.rekey([(text_hash(old), cache_key(old)), (text_hash(lane), cache_key(lane))]) == 0


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


def test_coverage_probes_the_key_once_per_text_and_never_scans_the_store(tmp_path: Path) -> None:
    """A coverage lookup costs what it asks, not what the cache holds."""
    cache = EmbeddingCache("voyage:voyage-code-4", tmp_path)
    vector = np.ones(4, dtype=np.float32).tobytes()
    cache.put_missing([(text_hash(f"text {i}"), 4, vector) for i in range(50)])

    # 1. Stored texts are covered at their width and below it, never above it.
    asked = [text_hash("text 1"), text_hash("text 2"), text_hash("missing")]
    assert cache.covered(asked, 4) == set(asked[:2]), "step 1: stored texts are covered"
    assert cache.covered(asked, 2) == set(asked[:2]), "step 1: a wider vector serves a narrower width"
    assert cache.covered(asked, 8) == set(), "step 1: a narrower vector never serves a wider one"

    # 2. The lookup searches the key per wanted text; scanning the store made every call cost
    #    the whole file on a server holding millions of vectors.
    statements: list[str] = []
    cache._connection.set_trace_callback(statements.append)
    cache.covered(asked, 4)
    cache._connection.set_trace_callback(None)
    lookup = next(statement for statement in statements if statement.startswith("SELECT w.digest"))
    plan = [row[3] for row in cache._connection.execute(f"EXPLAIN QUERY PLAN {lookup}")]
    assert not any(step.startswith("SCAN e") for step in plan), f"step 2: the store is never scanned: {plan}"

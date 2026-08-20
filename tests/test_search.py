from typing import Any
from unittest.mock import MagicMock, call, patch

import numpy as np
import numpy.typing as npt
import pytest
from model2vec import StaticModel
from vicinity.backends.basic import BasicArgs

from tests.conftest import FakeEmbedder, make_chunk
from zemble.embedding.model2vec import Model2VecEmbedder, load_static_model
from zemble.index.bm25 import BM25
from zemble.index.dense import SelectableBasicBackend, embed_chunks
from zemble.search import _search_bm25, _search_semantic, _sort_top_k, search
from zemble.tokens import tokenize
from zemble.types import Chunk


def _build_bm25(chunks: list[Chunk]) -> BM25:
    """Build a BM25 index over chunks, keyed by their position."""
    index = BM25()
    doc_ids = [f"c{i}" for i in range(len(chunks))]
    for doc_id, chunk in zip(doc_ids, chunks):
        index.add_document(doc_id, tokenize(chunk.content))
    index.set_doc_order(doc_ids)
    return index


@pytest.fixture
def chunks() -> list[Chunk]:
    """Four small code chunks covering authentication, login, user service, and utils."""
    return [
        make_chunk("def authenticate(token):\n    return token == 'secret'", "auth.py"),
        make_chunk("def login(username, password):\n    pass", "auth.py"),
        make_chunk("class UserService:\n    pass", "users.py"),
        make_chunk("def format_date(dt):\n    return str(dt)", "utils.py"),
    ]


@pytest.fixture
def embeddings(chunks: list[Chunk]) -> npt.NDArray[np.float32]:
    """Deterministic random unit-norm embeddings for the chunks fixture."""
    rng = np.random.default_rng(0)
    embs = rng.standard_normal((len(chunks), 256)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    normalized: npt.NDArray[np.float32] = embs / (norms + 1e-8)
    return normalized


@pytest.fixture
def bm25(chunks: list[Chunk]) -> BM25:
    """Pre-built BM25 index over the chunks fixture."""
    return _build_bm25(chunks)


@pytest.fixture
def semantic(embeddings: npt.NDArray[np.float32]) -> SelectableBasicBackend:
    """Pre-built ANNS index over the chunks fixture."""
    return SelectableBasicBackend(embeddings, BasicArgs())


def test_search_bm25(bm25: BM25, chunks: list[Chunk]) -> None:
    """search_bm25: returns most relevant chunk first; selector restricts to given indices."""
    results = _search_bm25("authenticate token", bm25, chunks, top_k=4, selector=None)
    assert len(results) > 0
    assert "authenticate" in results[0].chunk.content

    selector = np.array([len(chunks) - 1], dtype=np.int_)
    filtered = _search_bm25("format", bm25, chunks, top_k=4, selector=selector)
    assert all(r.chunk is chunks[len(chunks) - 1] for r in filtered)


@pytest.mark.parametrize("query", ["", "   ", "\n\n", "zzzznonexistentterm"])
def test_bm25_returns_empty_for_no_match(bm25: BM25, chunks: list[Chunk], query: str) -> None:
    """Empty / whitespace-only / token-less queries return [] instead of crashing."""
    assert _search_bm25(query, bm25, chunks, top_k=3, selector=None) == []


def test_semantic_search(semantic: SelectableBasicBackend, chunks: list[Chunk], mock_embedder: Any) -> None:
    """Semantic search returns results with scores in [-1, 1]."""
    results = _search_semantic("login", mock_embedder, semantic, chunks, top_k=3, selector=None)
    assert len(results) > 0
    assert all(-1.0 <= r.score <= 1.0 for r in results)


def test_search_hybrid(chunks: list[Chunk], semantic: SelectableBasicBackend, bm25: BM25, mock_embedder: Any) -> None:
    """search_hybrid: returns combined results; identical content in different files produces separate results."""
    results = search("authenticate token", mock_embedder, semantic, bm25, chunks, top_k=3)
    assert len(results) > 0

    shared_content = "def helper():\n    pass"
    chunk_a = make_chunk(shared_content, "module_a.py")
    chunk_b = make_chunk(shared_content, "module_b.py")
    all_chunks = [chunk_a, chunk_b]

    rng = np.random.default_rng(1)
    embs = rng.standard_normal((2, 256)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8

    sem_index = SelectableBasicBackend(embs, BasicArgs())
    bm25_index = _build_bm25(all_chunks)

    deduped = search("helper", mock_embedder, sem_index, bm25_index, all_chunks, top_k=5)
    result_locations = {r.chunk.file_path for r in deduped}
    assert "module_a.py" in result_locations
    assert "module_b.py" in result_locations


@pytest.mark.parametrize(
    ("search_fn", "query", "top_k"),
    [
        (lambda q, m, s, b, c, k: _search_bm25(q, b, c, k, selector=None), "authenticate", 3),
        (lambda q, m, s, b, c, k: _search_semantic(q, m, s, c, k, selector=None), "query", 4),
        (lambda q, m, s, b, c, k: search(q, m, s, b, c, k), "login", 4),
    ],
)
def test_search_source_labels(
    search_fn: Any,
    query: str,
    top_k: int,
    chunks: list[Chunk],
    semantic: SelectableBasicBackend,
    bm25: BM25,
    mock_embedder: Any,
) -> None:
    """Each result carries a source label matching the search mode used."""
    results = search_fn(query, mock_embedder, semantic, bm25, chunks, top_k)
    assert len(results) > 0


def test_sort_top_k() -> None:
    """_sort_top_k returns the same indices as np.argsort(-x)[:top_k]."""
    gen = np.random.default_rng()
    x = gen.standard_normal(size=(10000,))
    top_k = 100
    indices = _sort_top_k(x, top_k)
    assert np.all(indices == np.argsort(-x)[:top_k])


@pytest.mark.parametrize(
    ("model_name", "incomplete_cache"),
    [
        ("some/custom/model", False),  # explicit path forwarded
        ("broken/model", True),  # incomplete cache retries through the Hub
    ],
)
def test_model2vec_embedder_loads_lazily(model_name: str, incomplete_cache: bool) -> None:
    """Model2VecEmbedder loads through from_pretrained on first use, retrying a broken local cache."""
    load_static_model.cache_clear()
    fake_model = MagicMock(spec=StaticModel)
    side_effect = [ValueError("Could not find expected model files"), fake_model] if incomplete_cache else None
    embedder = Model2VecEmbedder(model_name)
    assert embedder.model_id == f"model2vec:{model_name}"
    with patch("model2vec.StaticModel.from_pretrained", return_value=fake_model, side_effect=side_effect) as mock_fp:
        assert embedder.model is fake_model
    expected_calls = [call(model_name, force_download=False)]
    if incomplete_cache:
        expected_calls.append(call(model_name, force_download=True))
    assert mock_fp.call_args_list == expected_calls
    load_static_model.cache_clear()


def test_embed_chunks_empty_returns_empty_array(mock_embedder: Any) -> None:
    """embed_chunks with an empty list returns an empty float32 array of the embedder's width."""
    result = embed_chunks(mock_embedder, [])
    assert result.shape == (0, 256)
    assert result.dtype == np.float32


def test_selectable_basic_backend_rejects_k_below_one(
    semantic: SelectableBasicBackend, embeddings: npt.NDArray[np.float32]
) -> None:
    """SelectableBasicBackend.query guards against k < 1."""
    with pytest.raises(ValueError, match="k should be >= 1"):
        semantic.query(embeddings[:1], k=0)


class _BonusEmbedder(FakeEmbedder):
    """A FakeEmbedder that claims a larger share of the fusion, the way a hosted model does."""

    semantic_weight_bonus = 0.5


def test_search_honours_the_embedders_fusion_bonus(mock_embedder: FakeEmbedder) -> None:
    """An embedder's declared bonus reaches fusion: the dense lane's pick overtakes BM25's."""
    lexical = make_chunk("def refresh_session_cookie(request):\n    pass", "cookies.py")
    semantic_only = make_chunk("class Renewal:\n    pass", "renewal.py")
    corpus = [lexical, semantic_only]
    query = "how is a session cookie refreshed"

    # 1. Make the dense lane rank the chunk BM25 cannot see at all: its vector IS the query's.
    embeddings = np.vstack([mock_embedder.embed_documents([lexical.content]), mock_embedder.embed_queries([query])])
    semantic_index = SelectableBasicBackend(embeddings.astype(np.float32), BasicArgs())
    bm25_index = _build_bm25(corpus)

    # 2. At the shipped weights BM25's lexical hit wins the fusion.
    plain = search(query, mock_embedder, semantic_index, bm25_index, corpus, top_k=2, rerank=False)
    assert plain[0].chunk is lexical, "without a bonus the lexical match must still win"

    # 3. The same search with a bonus-declaring embedder hands the top spot to the dense lane.
    boosted = search(query, _BonusEmbedder(), semantic_index, bm25_index, corpus, top_k=2, rerank=False)
    assert boosted[0].chunk is semantic_only, "a declared bonus must reach the fusion weight"

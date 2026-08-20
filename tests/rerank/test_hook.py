from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import pytest
from vicinity.backends.basic import BasicArgs

from tests.conftest import make_chunk
from zemble.index.bm25 import BM25
from zemble.index.dense import SelectableBasicBackend
from zemble.rerank.apply import apply_reranker, passage_text
from zemble.rerank.registry import PassageMode, RerankSettings
from zemble.search import search
from zemble.tokens import tokenize
from zemble.types import Chunk


class FakeReranker:
    """Scores a passage by a substring table, and records every call."""

    def __init__(self, table: dict[str, float]) -> None:
        """Initialise the fake.

        :param table: Substring to score; a passage matching none of them scores 0.
        """
        self._table = table
        self.calls: list[tuple[str, list[str]]] = []

    @property
    def model_id(self) -> str:
        """The fake's spec string."""
        return "fake:table"

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Score every passage by the first table entry it contains.

        :param query: The search query.
        :param passages: The candidate passages.
        :return: One score per passage.
        """
        self.calls.append((query, passages))
        scores = []
        for passage in passages:
            scores.append(next((value for key, value in self._table.items() if key in passage), 0.0))
        return scores


def _ranked(names: list[str]) -> list[tuple[Chunk, float]]:
    """Build a ranked list whose fused scores descend by 1.0 in the given order."""
    return [
        (Chunk(content=f"body {name}", file_path=f"{name}.py", start_line=1, end_line=1, context=f"ctx {name}"), score)
        for name, score in zip(names, [10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
    ]


def test_rerank_pass_journey(caplog: pytest.LogCaptureFixture) -> None:
    """A reranked head is re-sorted, the tail is untouched, and the blend follows the formula."""
    ranked = _ranked(["a", "b", "c", "d", "e", "f"])
    reranker = FakeReranker({"body c": 3.0, "body b": 2.0, "body a": 1.0, "body f": 100.0})

    # 1. alpha=1.0: the head order is the reranker's order alone.
    with caplog.at_level(logging.INFO, logger="zemble.rerank.apply"):
        out = apply_reranker("q", ranked, reranker, RerankSettings(top_k=3, alpha=1.0))
    assert [c.file_path for c, _ in out] == ["c.py", "b.py", "a.py", "d.py", "e.py", "f.py"], "head re-sorted only"

    # 2. Only the window is scored: f scores highest but sits below the window and never moves.
    assert len(reranker.calls[0][1]) == 3, "only the top-k window is sent to the reranker"

    # 3. The score column stays the window's own descending scores, so head and tail compare.
    assert [score for _, score in out] == [10.0, 9.0, 8.0, 7.0, 6.0, 5.0], "scores handed out in the new order"

    # 4. The timing of the pass is logged for every query.
    assert "Reranked 3 candidates with fake:table" in caplog.text, "per-query timing logged"

    # 5. alpha=0.7: normalized fused = [1, 0.5, 0] and normalized rerank = [0, 0.5, 1],
    #    so blended = [0.3, 0.5, 0.7] and the middle candidate beats the one the fusion led with.
    blended = apply_reranker("q", ranked, reranker, RerankSettings(top_k=3, alpha=0.7))
    assert [c.file_path for c, _ in blended] == ["c.py", "b.py", "a.py", "d.py", "e.py", "f.py"]

    # 6. alpha=0.2 tips the same window back the other way: blended = [0.8, 0.5, 0.2].
    conservative = apply_reranker("q", ranked, reranker, RerankSettings(top_k=3, alpha=0.2))
    assert [c.file_path for c, _ in conservative] == ["a.py", "b.py", "c.py", "d.py", "e.py", "f.py"]

    # 7. A window of one has nothing to reorder and costs no reranker call.
    before = len(reranker.calls)
    assert apply_reranker("q", ranked, reranker, RerankSettings(top_k=1)) == ranked
    assert len(reranker.calls) == before


def test_passage_mode_decides_what_the_reranker_sees() -> None:
    """The context capsule is prefixed in context mode and absent in content mode."""
    chunk = Chunk(content="body", file_path="a.py", start_line=1, end_line=1, context="ctx a")
    assert passage_text(chunk, PassageMode.CONTEXT) == "ctx a\nbody"
    assert passage_text(chunk, PassageMode.CONTENT) == "body"

    without_capsule = Chunk(content="body", file_path="a.py", start_line=1, end_line=1)
    assert passage_text(without_capsule, PassageMode.CONTEXT) == "body", "no capsule, no leading newline"


def test_reranker_returning_the_wrong_count_is_refused() -> None:
    """A reranker that answers with the wrong number of scores fails loudly."""

    class Broken(FakeReranker):
        def score(self, query: str, passages: list[str]) -> list[float]:
            """Return one score too few."""
            return super().score(query, passages)[:-1]

    with pytest.raises(ValueError, match="returned 2 scores for 3 passages"):
        apply_reranker("q", _ranked(["a", "b", "c"]), Broken({}), RerankSettings(top_k=3))


@pytest.fixture
def hybrid_chunks() -> list[Chunk]:
    """Six chunks that all mention the query term, so the fused pool is deep."""
    return [make_chunk(f"def handler_{i}(token):\n    return token", f"mod_{i}.py") for i in range(6)]


@pytest.fixture
def hybrid_embeddings(hybrid_chunks: list[Chunk]) -> npt.NDArray[np.float32]:
    """Deterministic unit-norm embeddings for the hybrid chunks."""
    rng = np.random.default_rng(3)
    embeddings = rng.standard_normal((len(hybrid_chunks), 256)).astype(np.float32)
    normalized: npt.NDArray[np.float32] = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return normalized


def test_search_hook_pulls_a_deep_candidate_into_the_top(
    hybrid_chunks: list[Chunk], hybrid_embeddings: npt.NDArray[np.float32], mock_embedder: object
) -> None:
    """A reranker window wider than top_k can promote a candidate the fusion ranked below it."""
    bm25 = BM25()
    doc_ids = [f"c{i}" for i in range(len(hybrid_chunks))]
    for doc_id, chunk in zip(doc_ids, hybrid_chunks):
        bm25.add_document(doc_id, tokenize(chunk.content))
    bm25.set_doc_order(doc_ids)
    semantic = SelectableBasicBackend(hybrid_embeddings, BasicArgs())

    baseline = search("token handler", mock_embedder, semantic, bm25, hybrid_chunks, top_k=2)
    assert len(baseline) == 2

    last = search("token handler", mock_embedder, semantic, bm25, hybrid_chunks, top_k=6)[-1].chunk
    reranker = FakeReranker({last.content: 1.0})
    promoted = search(
        "token handler",
        mock_embedder,
        semantic,
        bm25,
        hybrid_chunks,
        top_k=2,
        reranker=reranker,
        rerank_settings=RerankSettings(top_k=6, alpha=1.0),
    )
    assert len(promoted) == 2, "top_k still bounds the returned list"
    assert promoted[0].chunk == last, "the reranker's pick leads, though the fusion ranked it last"

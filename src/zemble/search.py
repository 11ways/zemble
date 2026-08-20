from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from zemble.embedding.base import Embedder
from zemble.index.bm25 import BM25
from zemble.index.dense import SelectableBasicBackend
from zemble.index.sparse import selector_to_mask
from zemble.ranking import apply_query_boost, boost_multi_chunk_files, rerank_topk, resolve_alpha
from zemble.rerank.apply import apply_reranker
from zemble.rerank.base import Reranker
from zemble.rerank.registry import RerankSettings
from zemble.tokens import tokenize
from zemble.types import Chunk, SearchResult

if TYPE_CHECKING:
    from zemble.index.symbols import SymbolDefinitions

_RRF_K = 60


def _rrf_scores(scores: dict[Chunk, float]) -> dict[Chunk, float]:
    """Convert raw scores to RRF scores 1/(k + rank); higher raw score → rank 1."""
    if not scores:
        return scores
    ranked = sorted(scores, key=lambda c: -scores[c])
    return {chunk: 1.0 / (_RRF_K + rank) for rank, chunk in enumerate(ranked, 1)}


def _search_semantic(
    query: str,
    embedder: Embedder,
    semantic_index: SelectableBasicBackend,
    chunks: list[Chunk],
    top_k: int,
    selector: npt.NDArray[np.int_] | None,
) -> list[SearchResult]:
    """Run semantic search for a query."""
    query_embedding = embedder.embed_queries([query])
    indices, scores = semantic_index.query(query_embedding, k=top_k, selector=selector)[0]
    # Vicinity returns cosine distance; convert to similarity so higher = better.
    return [SearchResult(chunk=chunks[index], score=1.0 - float(distance)) for index, distance in zip(indices, scores)]


def _sort_top_k(arr: npt.NDArray, top_k: int) -> npt.NDArray[np.int_]:
    """Get the top k indices of an array in sort order."""
    neg_arr = -arr
    if top_k >= len(arr):
        return np.argsort(neg_arr)
    partitioned = np.argpartition(neg_arr, kth=top_k)[:top_k]
    return partitioned[np.argsort(neg_arr[partitioned])]


def _search_bm25(
    query: str,
    bm25_index: BM25,
    chunks: list[Chunk],
    top_k: int,
    selector: npt.NDArray[np.int_] | None,
) -> list[SearchResult]:
    """Return chunks ranked by BM25 score, excluding zero-score results."""
    tokens = tokenize(query)
    if not tokens:
        return []
    mask = selector_to_mask(selector, len(chunks))
    scores: npt.NDArray[np.float32] = bm25_index.get_scores(tokens, weight_mask=mask)
    indices = _sort_top_k(scores, top_k)

    # Exclude chunks with zero score, no query tokens matched.
    return [SearchResult(chunk=chunks[i], score=float(scores[i])) for i in indices if scores[i] > 0]


def search(
    query: str,
    embedder: Embedder,
    semantic_index: SelectableBasicBackend,
    bm25_index: BM25,
    chunks: list[Chunk],
    top_k: int,
    alpha: float | None = None,
    selector: npt.NDArray[np.int_] | None = None,
    rerank: bool = True,
    definitions: "SymbolDefinitions | None" = None,
    reranker: Reranker | None = None,
    rerank_settings: RerankSettings | None = None,
) -> list[SearchResult]:
    """Hybrid search: alpha-weighted combination of semantic and BM25 scores.

    Both score sets are converted to RRF scores before combining, so alpha has
    a consistent meaning regardless of raw score magnitude.

    :param query: Search query string.
    :param embedder: Embedder used for the query vector.
    :param semantic_index: Pre-built semantic (vector) index.
    :param bm25_index: Pre-built BM25 index.
    :param chunks: All indexed chunks (parallel to BM25 index).
    :param top_k: Number of results to return.
    :param alpha: Weight for semantic score (1-alpha goes to BM25). None = auto-detect based on query type.
    :param selector: Optional array of chunk indices to filter results by.
    :param rerank: Whether to perform code-tuned reranking. On by default for code search, off for docs search.
    :param definitions: Persisted symbol-definition lookup used by the rerank pass, when the index has one.
    :param reranker: Optional pairwise reranker applied to the head of the ranked list.
    :param rerank_settings: Window, blend weight and passage shape for that pass; defaults apply when omitted.
    :return: List of search results sorted by combined score descending.
    """
    alpha_weight = resolve_alpha(query, alpha)
    settings = rerank_settings or RerankSettings()
    # The reranker only sees candidates the heuristics already ranked, so the pool and the
    # ranked list both have to reach its window before it can reorder anything.
    ranked_count = max(top_k, settings.top_k) if reranker is not None else top_k

    # Over-fetch candidates so the merged pool is large enough after union and re-ranking.
    candidate_count = max(top_k * 5, ranked_count)

    semantic = _search_semantic(query, embedder, semantic_index, chunks, candidate_count, selector)
    semantic_scores: dict[Chunk, float] = {result.chunk: result.score for result in semantic}
    bm25_scores = {}
    for result in _search_bm25(query, bm25_index, chunks, candidate_count, selector):
        if result.score:
            bm25_scores[result.chunk] = result.score

    normalized_semantic = _rrf_scores(semantic_scores)
    normalized_bm25 = _rrf_scores(bm25_scores)

    # Sort by start line to counteract randomness introduced by hashing.
    all_candidates = sorted(
        {*normalized_semantic, *normalized_bm25},
        key=lambda c: c.start_line,
    )
    combined_scores: dict[Chunk, float] = {
        chunk: alpha_weight * normalized_semantic.get(chunk, 0.0)
        + (1.0 - alpha_weight) * normalized_bm25.get(chunk, 0.0)
        for chunk in all_candidates
    }

    # Remove chunks that have 0.0 score
    combined_scores = {chunk: score for chunk, score in combined_scores.items() if score}

    if rerank:
        # Boost files with multiple relevant chunks.
        boost_multi_chunk_files(combined_scores)
        # Boost queries with specific identifiers in them.
        combined_scores = apply_query_boost(combined_scores, query, chunks, definitions)
        # Rerank the top-k results by applying path-based penalties.
        ranked = rerank_topk(combined_scores, ranked_count, penalise_paths=alpha_weight < 1.0)
    else:
        sorted_by_score = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        ranked = sorted_by_score[:ranked_count]

    if reranker is not None:
        ranked = apply_reranker(query, ranked, reranker, settings)
    return [SearchResult(chunk=chunk, score=score) for chunk, score in ranked[:top_k]]

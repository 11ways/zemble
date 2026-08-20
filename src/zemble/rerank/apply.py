"""The post-fusion rerank pass: rescore the head of a ranked list and re-sort it."""

from __future__ import annotations

import logging
import time

from zemble.rerank.base import Reranker
from zemble.rerank.registry import PassageMode, RerankSettings
from zemble.types import Chunk

logger = logging.getLogger(__name__)


def passage_text(chunk: Chunk, mode: PassageMode) -> str:
    """Build the text the reranker is shown for a chunk.

    :param chunk: The candidate chunk.
    :param mode: Whether the context capsule is prefixed.
    :return: The passage text.
    """
    if mode is PassageMode.CONTEXT and chunk.context:
        return f"{chunk.context}\n{chunk.content}"
    return chunk.content


def _normalized(values: list[float]) -> list[float]:
    """Min-max scale values into ``[0, 1]``; an all-equal list becomes all zeros."""
    low = min(values)
    high = max(values)
    span = high - low
    if span <= 0.0:
        return [0.0] * len(values)
    return [(value - low) / span for value in values]


def apply_reranker(
    query: str,
    ranked: list[tuple[Chunk, float]],
    reranker: Reranker,
    settings: RerankSettings,
) -> list[tuple[Chunk, float]]:
    """Rescore the head of a ranked list with a pairwise reranker and re-sort it.

    The blend is ``alpha * normalized_rerank + (1 - alpha) * normalized_fused``, both sides
    min-max normalized over the rescored window only. Everything below the window keeps its
    position untouched.

    AIDEV-NOTE: the returned score column is the window's own original scores handed out in
    the new order, not the blend. The blend lives on a scale of its own, and emitting it
    would leave the reranked head incomparable with the untouched tail below it.

    :param query: The search query.
    :param ranked: The fused, ranked candidates, best first.
    :param reranker: The reranker to score with.
    :param settings: The window size, blend weight and passage shape.
    :return: The same candidates, with the head re-sorted.
    :raises ValueError: If the reranker answers with the wrong number of scores.
    """
    window = ranked[: settings.top_k]
    if len(window) < 2:
        return ranked
    tail = ranked[settings.top_k :]

    started = time.perf_counter()
    passages = [passage_text(chunk, settings.passage) for chunk, _ in window]
    rerank_scores = reranker.score(query, passages)
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info("Reranked %d candidates with %s in %.1f ms", len(window), reranker.model_id, elapsed_ms)

    if len(rerank_scores) != len(window):
        raise ValueError(f"{reranker.model_id} returned {len(rerank_scores)} scores for {len(window)} passages")

    normalized_rerank = _normalized(rerank_scores)
    normalized_fused = _normalized([score for _, score in window])
    blended = [
        settings.alpha * rerank + (1.0 - settings.alpha) * fused
        for rerank, fused in zip(normalized_rerank, normalized_fused)
    ]

    order = sorted(range(len(window)), key=lambda i: -blended[i])
    original_scores = [score for _, score in window]
    reordered = [(window[i][0], original_scores[position]) for position, i in enumerate(order)]
    return reordered + tail

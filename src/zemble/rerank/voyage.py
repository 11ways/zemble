"""The Voyage AI reranker, served by https://docs.voyageai.com/docs/reranker."""

from __future__ import annotations

import os

from zemble.embedding.http import CHARS_PER_TOKEN, EmbeddingRequestError, batched, post_json

VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank"
API_KEY_ENV = "VOYAGE_API_KEY"

#: Documented ceilings: 1,000 documents per request, and a per-request token budget that is
#: 300K for the small models. We stay under both; the character estimate is deliberately
#: pessimistic because code tokenizes worse than prose.
MAX_DOCUMENTS_PER_REQUEST = 100
MAX_TOKENS_PER_REQUEST = 100_000


class VoyageReranker:
    """Scores (query, passage) pairs through the Voyage rerank endpoint.

    Long candidate lists are split into several requests. ``relevance_score`` is calibrated
    per query rather than per request, so scores from two batches of the same query stay
    comparable and no cross-batch renormalization is needed.
    """

    def __init__(self, model: str) -> None:
        """Initialise the reranker.

        :param model: A Voyage rerank model name, e.g. ``rerank-2.5-lite``.
        """
        self._model = model
        self.total_tokens = 0
        self.request_count = 0

    @property
    def model_id(self) -> str:
        """The normalized spec string."""
        return f"voyage:{self._model}"

    def _api_key(self) -> str:
        """Return the API key, refusing loudly when the environment variable is unset.

        :return: The API key.
        :raises EmbeddingRequestError: If the variable is unset or empty.
        """
        key = os.environ.get(API_KEY_ENV, "").strip()
        if not key:
            raise EmbeddingRequestError(f"{API_KEY_ENV} is not set; it is required for {self.model_id}")
        return key

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Score every passage against the query.

        :param query: The search query.
        :param passages: The candidate passages, in candidate order.
        :return: One score per passage, in the same order; higher = more relevant.
        :raises EmbeddingRequestError: If a response holds the wrong number of scores or an out-of-range index.
        """
        if not passages:
            return []
        headers = {"Authorization": f"Bearer {self._api_key()}"}
        scores = [0.0] * len(passages)
        max_chars = int(MAX_TOKENS_PER_REQUEST * CHARS_PER_TOKEN)
        for offset, batch in batched(passages, MAX_DOCUMENTS_PER_REQUEST, max_chars):
            payload = {"query": query, "documents": batch, "model": self._model, "truncation": True}
            response = post_json(VOYAGE_RERANK_URL, payload, headers)
            self.request_count += 1
            usage = response.get("usage")
            if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int):
                self.total_tokens += int(usage["total_tokens"])
            data = response.get("data")
            if not isinstance(data, list) or len(data) != len(batch):
                raise EmbeddingRequestError(
                    f"{VOYAGE_RERANK_URL} returned {len(data or [])} scores for {len(batch)} documents"
                )
            for entry in data:
                index = int(entry.get("index", -1))
                if not 0 <= index < len(batch):
                    raise EmbeddingRequestError(f"{VOYAGE_RERANK_URL} returned out-of-range index {index}")
                scores[offset + index] = float(entry["relevance_score"])
        return scores

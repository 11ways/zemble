"""The reranker seam: everything that scores a (query, passage) pair implements this."""

from __future__ import annotations

from typing import Protocol


class Reranker(Protocol):
    """A pairwise relevance scorer applied to the head of an already-ranked candidate list.

    Scores are compared only against each other inside one call, so an implementation is
    free to return logits, probabilities or distances-turned-similarities as long as
    higher means more relevant.
    """

    @property
    def model_id(self) -> str:
        """The normalized spec string identifying this reranker."""
        ...

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Score every passage against the query.

        :param query: The search query.
        :param passages: The candidate passages, in candidate order.
        :return: One score per passage, in the same order; higher = more relevant.
        """
        ...

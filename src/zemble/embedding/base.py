"""The embedder seam: everything that turns text into vectors implements this."""

from __future__ import annotations

import logging
import os
from typing import Protocol

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

EmbeddingMatrix = npt.NDArray[np.float32]


class Embedder(Protocol):
    """A source of L2-normalized float32 embeddings.

    Documents and queries are embedded through separate methods on purpose:
    asymmetric models (Voyage's ``input_type``, instruction-prefixed models)
    produce a different vector for the same text depending on its role, and a
    single ``encode`` would silently pick one of the two.
    """

    @property
    def model_id(self) -> str:
        """The normalized spec string identifying this embedder; two indexes with the same value are comparable."""
        ...

    @property
    def dimensions(self) -> int:
        """The width of the vectors this embedder returns."""
        ...

    @property
    def is_remote(self) -> bool:
        """Whether embedding through this costs a round trip and money."""
        ...

    @property
    def semantic_weight_bonus(self) -> float:
        """How much of the RRF fusion this embedder's dense lane earns beyond the shipped weights.

        The shipped fusion weights were tuned around a static embedder; a contextual one
        deserves a larger share. Declared per embedder family so one number cannot be right
        for both, and measured in ``docs/voyage.md``.
        """
        ...

    def embed_documents(self, texts: list[str]) -> EmbeddingMatrix:
        """Embed indexable content.

        :param texts: The document texts.
        :return: An ``(len(texts), dimensions)`` float32 array with L2-normalized rows.
        """
        ...

    def embed_queries(self, texts: list[str]) -> EmbeddingMatrix:
        """Embed search queries.

        :param texts: The query texts.
        :return: An ``(len(texts), dimensions)`` float32 array with L2-normalized rows.
        """
        ...


def normalize_rows(vectors: EmbeddingMatrix) -> EmbeddingMatrix:
    """L2-normalize every row, leaving all-zero rows untouched instead of dividing by zero."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized: EmbeddingMatrix = vectors / np.where(norms == 0.0, 1.0, norms)
    return normalized.astype(np.float32, copy=False)


def is_remote(embedder: object) -> bool:
    """Return whether an embedder is a paid, remote one.

    An embedder that does not declare itself is treated as paid: the budget guard fails
    closed, and being asked to confirm a free embed is cheaper than an unasked bill.
    """
    return bool(getattr(embedder, "is_remote", True))


def declared_dimensions(embedder: object) -> int | None:
    """Return the vector width already known, or None when only a request or a model load could tell.

    Reading ``dimensions`` on a remote embedder whose model has no documented width sends a
    probe request, which a pre-flight report must never do.
    """
    return getattr(embedder, "declared_dimensions", None)


#: Env var overriding every embedder's declared fusion bonus, for sweeping the knob.
SEMANTIC_WEIGHT_BONUS_ENV = "ZEMBLE_SEMANTIC_WEIGHT_BONUS"

#: The bonus an embedder that declares nothing gets: today's shipped behaviour.
DEFAULT_SEMANTIC_WEIGHT_BONUS = 0.0


def semantic_weight_bonus(embedder: object) -> float:
    """Return the dense-lane fusion bonus for an embedder, honouring the env override.

    An embedder that declares nothing gets 0.0, which is exactly the shipped fusion
    weights: an unrecognised embedder never silently retunes ranking.

    :param embedder: The embedder, or None where no embedder is in play.
    :return: A bonus clamped to [0, 1].
    """
    override = os.environ.get(SEMANTIC_WEIGHT_BONUS_ENV, "").strip()
    if override:
        try:
            declared = float(override)
        except ValueError:
            logger.warning("Ignoring %s=%r: not a number", SEMANTIC_WEIGHT_BONUS_ENV, override)
            declared = DEFAULT_SEMANTIC_WEIGHT_BONUS
    else:
        raw = getattr(embedder, "semantic_weight_bonus", DEFAULT_SEMANTIC_WEIGHT_BONUS)
        try:
            declared = float(raw)
        except (TypeError, ValueError):
            declared = DEFAULT_SEMANTIC_WEIGHT_BONUS
    return min(1.0, max(0.0, declared))

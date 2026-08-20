"""The embedder seam: everything that turns text into vectors implements this."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

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

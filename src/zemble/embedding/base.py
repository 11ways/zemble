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

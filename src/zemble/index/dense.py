from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from vicinity.backends.basic import CosineBasicBackend
from vicinity.datatypes import QueryResult
from vicinity.utils import normalize

from zemble.embedding.base import Embedder
from zemble.types import Chunk


def embed_chunks(embedder: Embedder, chunks: list[Chunk]) -> npt.NDArray[np.float32]:
    """Embed chunk contents as documents.

    :param embedder: The embedder to use.
    :param chunks: The chunks to embed.
    :return: A float32 matrix, one row per chunk.
    """
    if not chunks:
        return np.empty((0, embedder.dimensions), dtype=np.float32)
    return embedder.embed_documents([chunk.content for chunk in chunks])


class SelectableBasicBackend(CosineBasicBackend):
    def _selector_dist(self, x: npt.NDArray, selector: npt.NDArray[np.int_]) -> npt.NDArray:
        """Compute cosine distance."""
        x_norm = normalize(x)
        sim = x_norm.dot(self._vectors[selector].T)
        return 1 - sim

    def query(self, vectors: npt.NDArray, k: int, selector: npt.NDArray[np.int_] | None = None) -> QueryResult:
        """Batched distance query.

        :param vectors: The vectors to query.
        :param k: The number of nearest neighbors to return.
        :param selector: Optional array of chunk indices to filter results by.
        :return: A list of tuples with the indices and distances.
        :raises ValueError: If k is less than 1.
        """
        if k < 1:
            raise ValueError(f"k should be >= 1, is now {k}")

        out: QueryResult = []
        num_vectors = len(self.vectors)
        effective_k = min(k, num_vectors)
        if selector is not None:
            effective_k = min(effective_k, len(selector))

        # Batch the queries
        for index in range(0, len(vectors), 1024):
            batch = vectors[index : index + 1024]
            if selector is not None:
                distances = self._selector_dist(batch, selector)
            else:
                distances = self._dist(batch)

            # Efficiently get the k smallest distances
            indices = np.argpartition(distances, kth=effective_k - 1, axis=1)[:, :effective_k]
            sorted_indices = np.take_along_axis(
                indices, np.argsort(np.take_along_axis(distances, indices, axis=1)), axis=1
            )
            sorted_distances = np.take_along_axis(distances, sorted_indices, axis=1)

            # Extend the output with tuples of (indices, distances)
            if selector is not None:
                sorted_indices = selector[sorted_indices]
            out.extend(zip(sorted_indices, sorted_distances))

        return out

    def save(self, path: Path) -> None:
        """Save the selectable basic backend."""
        path.mkdir(parents=True, exist_ok=True)
        super().save(path)

    @classmethod
    def load(cls, path: Path) -> "SelectableBasicBackend":
        """Load a selectable basic backend."""
        loaded = super().load(path)
        return SelectableBasicBackend(loaded.vectors, loaded.arguments)

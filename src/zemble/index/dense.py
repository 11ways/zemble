from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from vicinity.backends.basic import BasicArgs, BasicBackend, CosineBasicBackend
from vicinity.datatypes import QueryResult
from vicinity.utils import normalize

from zemble.chunking.capsule import embedding_text
from zemble.embedding.base import Embedder
from zemble.embedding.pricing import require_affordable_bill
from zemble.index.columnar import atomic_save
from zemble.types import Chunk


def embed_chunks(embedder: Embedder, chunks: list[Chunk]) -> npt.NDArray[np.float32]:
    """Embed chunk contents as documents, each prefixed by its context capsule when it has one.

    Every chunk a build embeds passes here in ONE call, incremental builds included, so this
    is where the bill for the whole build is judged - never per provider batch. An uncached set
    that is over budget raises ``EmbeddingBudgetExceeded`` out of the guard, having sent nothing.

    :param embedder: The embedder to use.
    :param chunks: The chunks to embed.
    :return: A float32 matrix, one row per chunk.
    """
    if not chunks:
        return np.empty((0, embedder.dimensions), dtype=np.float32)
    texts = [embedding_text(chunk) for chunk in chunks]
    require_affordable_bill(embedder, texts)
    return embedder.embed_documents(texts)


#: Maximum temporary embedding-row copy for a scattered selector, per query worker.
_SELECTOR_COPY_BYTES = 4 * 1024 * 1024


class SelectableBasicBackend(CosineBasicBackend):
    def _selector_dist(self, x: npt.NDArray, selector: npt.NDArray[np.int_]) -> npt.NDArray:
        """Score contiguous subtrees through a view, gathering scattered rows in bounded blocks."""
        x_norm = normalize(x)
        if not len(selector):
            return np.empty((len(x), 0), dtype=self._vectors.dtype)
        start, end = int(selector[0]), int(selector[-1]) + 1
        if 0 <= start < end <= len(self._vectors) and end - start == len(selector) and np.all(np.diff(selector) == 1):
            return 1 - x_norm.dot(self._vectors[start:end].T)
        # AIDEV-NOTE: advanced indexing allocates rows x dimensions; fifty requests must
        # never each gather the whole selected matrix. Contiguous subtrees need no copy.
        rows = max(1, _SELECTOR_COPY_BYTES // (self._vectors.shape[1] * self._vectors.dtype.itemsize))
        distances = np.empty((len(x), len(selector)), dtype=np.result_type(x_norm, self._vectors))
        for offset in range(0, len(selector), rows):
            block = selector[offset : offset + rows]
            distances[:, offset : offset + len(block)] = 1 - x_norm.dot(self._vectors[block].T)
        return distances

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
        """Save the backend; its vectors are already unit length.

        The matrix is replaced rather than truncated in place: another index generation may
        still have this very file mapped read-only while this one is written.
        """
        path.mkdir(parents=True, exist_ok=True)
        atomic_save(path / "vectors.npy", self.vectors)
        self.arguments.dump(path / "arguments.json")

    @classmethod
    def load(cls, path: Path, writable: bool = False) -> "SelectableBasicBackend":
        """Load a selectable basic backend, mapping the vectors instead of reading them.

        Vicinity's own loader reads the whole matrix and then re-normalizes it; the constructor
        normalizes before saving, so the stored rows are already unit length and both passes are
        pure cost. The mapped matrix is read-only, so any caller that writes into ``vectors``
        must ask for a writable copy; an incremental build does not, because it copies the
        matrix itself only once it has a row to write.

        :param path: Directory the backend was saved to.
        :param writable: Read the vectors into memory instead of mapping them read-only.
        :return: The loaded backend.
        """
        arguments = BasicArgs.load(path / "arguments.json")
        vectors = np.load(path / "vectors.npy", mmap_mode=None if writable else "r")
        backend = cls.__new__(cls)
        # Skips CosineBasicBackend.__init__, whose only extra work is the normalization pass.
        BasicBackend.__init__(backend, vectors, arguments)
        return backend

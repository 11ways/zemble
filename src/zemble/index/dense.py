from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from vicinity.backends.basic import BasicArgs, BasicBackend, CosineBasicBackend
from vicinity.datatypes import QueryResult
from vicinity.utils import normalize

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

    @classmethod
    def load(cls, path: Path) -> "SelectableBasicBackend":
        """Load a selectable basic backend, mapping the vectors instead of reading them.

        Vicinity's own loader reads the whole matrix and then re-normalizes it; a build writes
        unit-length rows, so both passes are pure cost. The mapped matrix is read-only; a build
        writes the next generation's matrix into a file of its own.

        :param path: Directory the backend was saved to.
        :return: The loaded backend.
        """
        arguments = BasicArgs.load(path / "arguments.json")
        vectors = np.load(path / "vectors.npy", mmap_mode="r")
        backend = cls.__new__(cls)
        # Skips CosineBasicBackend.__init__, whose only extra work is the normalization pass.
        BasicBackend.__init__(backend, vectors, arguments)
        return backend

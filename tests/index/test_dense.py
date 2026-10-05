from pathlib import Path

import numpy as np
from vicinity.backends.basic import BasicArgs

from zemble.index.dense import SelectableBasicBackend


def _store(path: Path, vectors: np.ndarray) -> None:
    """Write a matrix the way a build does: unit rows, plus the backend arguments."""
    path.mkdir(parents=True, exist_ok=True)
    BasicArgs().dump(path / "arguments.json")
    np.save(path / "vectors.npy", vectors / np.linalg.norm(vectors, axis=1, keepdims=True))


def test_load_maps_the_vectors_and_query_results_are_unchanged(tmp_path: Path) -> None:
    """Loading maps the matrix read-only and answers exactly what the in-memory backend answers."""
    rng = np.random.default_rng(20260820)
    vectors = rng.standard_normal((32, 8)).astype(np.float32)
    backend = SelectableBasicBackend(vectors, BasicArgs())
    _store(tmp_path, vectors)
    query = rng.standard_normal((1, 8)).astype(np.float32)
    expected = backend.query(query, k=5)[0]

    # 1. The mapped backend answers exactly what the in-memory one answered.
    loaded = SelectableBasicBackend.load(tmp_path)
    np.testing.assert_allclose(loaded.vectors, backend.vectors, rtol=1e-6)
    indices, distances = loaded.query(query, k=5)[0]
    np.testing.assert_array_equal(indices, expected[0])
    np.testing.assert_allclose(distances, expected[1], rtol=1e-5)

    # 2. The load is a read-only map, so nothing can write through it by accident.
    assert isinstance(loaded.vectors, np.memmap)
    assert not loaded.vectors.flags.writeable


def test_subtree_scoring_bounds_embedding_row_copies_and_preserves_rankings() -> None:
    """Contiguous subtrees copy no rows; scattered selectors copy at most 4 MiB per block."""
    from vicinity.utils import normalize

    copies = []

    class BoundedRowCopies(np.ndarray):
        def __getitem__(self, key):
            if isinstance(key, np.ndarray):
                size = len(key) * self.shape[1] * self.dtype.itemsize
                assert size <= 4 * 1024**2, "a scattered subtree must not copy its whole matrix"
                copies.append(size)
            return super().__getitem__(key)

    rng = np.random.default_rng(20261001)
    vectors = rng.standard_normal((4096, 1024)).astype(np.float32)
    backend = SelectableBasicBackend(vectors, BasicArgs())
    queries = rng.standard_normal((3, 1024)).astype(np.float32)
    backend._vectors = backend.vectors.view(BoundedRowCopies)
    contiguous = np.arange(10, 4000)
    expected = 1 - normalize(queries).dot(np.asarray(backend.vectors)[contiguous].T)
    np.testing.assert_array_equal(backend._selector_dist(queries, contiguous), expected)
    assert not copies, "a contiguous subtree must use a slice, not a gather"

    selector = np.arange(0, len(vectors), 3)
    expected = 1 - normalize(queries).dot(np.asarray(backend.vectors)[selector].T)
    actual = backend._selector_dist(queries, selector)
    # Blocking can change the BLAS tail kernel by a float32 rounding unit, not the ranking.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)
    assert len(copies) == 2
    expected_single = 1 - normalize(queries[:1]).dot(np.asarray(backend.vectors)[selector].T)
    indices, distances = backend.query(queries[:1], 10, selector)[0]
    positions = np.argsort(expected_single[0])[:10]
    np.testing.assert_array_equal(indices, selector[positions])
    np.testing.assert_allclose(distances, expected_single[0, positions], rtol=0, atol=2e-7)

    # A slice must not silently clip invalid row IDs or reinterpret negative NumPy IDs.
    negative = np.array([-3, -2, -1])
    expected = 1 - normalize(queries).dot(np.asarray(backend.vectors)[negative].T)
    np.testing.assert_array_equal(backend._selector_dist(queries, negative), expected)
    with np.testing.assert_raises(IndexError):
        backend._selector_dist(queries, np.array([len(vectors) - 1, len(vectors)]))

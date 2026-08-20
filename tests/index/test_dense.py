from pathlib import Path

import numpy as np
from vicinity.backends.basic import BasicArgs

from zemble.index.dense import SelectableBasicBackend


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """Test save and load roundtrip."""
    vecs = np.random.default_rng(seed=42).normal(size=(10, 32))
    args = BasicArgs()
    selectable = SelectableBasicBackend(vecs, args)
    selectable.save(tmp_path)

    selectable_2 = SelectableBasicBackend.load(tmp_path)
    assert np.allclose(selectable.vectors, selectable_2.vectors)


def test_load_maps_the_vectors_and_query_results_are_unchanged(tmp_path: Path) -> None:
    """Loading maps the matrix read-only, keeps every value, and only copies when asked to."""
    rng = np.random.default_rng(20260820)
    vectors = rng.standard_normal((32, 8)).astype(np.float32)
    backend = SelectableBasicBackend(vectors, BasicArgs())
    backend.save(tmp_path)
    query = rng.standard_normal((1, 8)).astype(np.float32)
    expected = backend.query(query, k=5)[0]

    # 1. The mapped backend answers exactly what the in-memory one answered.
    loaded = SelectableBasicBackend.load(tmp_path)
    np.testing.assert_array_equal(loaded.vectors, backend.vectors)
    indices, distances = loaded.query(query, k=5)[0]
    np.testing.assert_array_equal(indices, expected[0])
    np.testing.assert_array_equal(distances, expected[1])

    # 2. The default load is a read-only map, so nothing can write through it by accident.
    assert isinstance(loaded.vectors, np.memmap)
    assert not loaded.vectors.flags.writeable

    # 3. Incremental reindexing asks for a writable copy and gets one.
    writable = SelectableBasicBackend.load(tmp_path, writable=True)
    assert writable.vectors.flags.writeable
    writable.vectors[0] = 0.0
    np.testing.assert_array_equal(SelectableBasicBackend.load(tmp_path).vectors, backend.vectors)

"""
test_drc_umap_index.py
----------------------
UMAP's transform with the hnswlib search index (drc_umap_index): neighbours
match exact ones, placements match pynndescent's, small-data models are left
alone, and a model needs its index only once.

Usage:
    pytest tests/test_drc_umap_index.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

import drc_umap_index  # noqa: E402

umap = pytest.importorskip('umap')


_CENTRES = np.random.default_rng(0).normal(scale=4.0, size=(8, 5))


def _data(n, seed):
    """Events from the same eight populations; seed varies the draw."""
    rng = np.random.default_rng(seed)
    return (_CENTRES[rng.integers(0, 8, n)] + rng.normal(size=(n, 5))).astype(np.float32)


@pytest.fixture(scope='module')
def fitted():
    reducer = umap.UMAP(n_neighbors=15, n_epochs=200, random_state=42, n_jobs=1,
                        low_memory=False).fit(_data(5_000, 0))
    return reducer


def test_search_matches_exact_neighbours(fitted):
    from sklearn.neighbors import NearestNeighbors
    pool = np.asarray(fitted._raw_data)
    query = _data(500, 1)
    index = drc_umap_index.HnswSearchIndex(pool, 'l2', threads=1)
    idx, dist = index.query(query, 15)
    exact_d, exact_i = NearestNeighbors(n_neighbors=15).fit(pool).kneighbors(query)
    recall = np.mean([len(set(a) & set(b)) / 15 for a, b in zip(idx, exact_i)])
    assert recall > 0.99
    assert idx.dtype == np.int32 and dist.dtype == np.float32
    same = idx[:, 0] == exact_i[:, 0]
    assert same.mean() > 0.99
    assert np.allclose(dist[same, 0], exact_d[same, 0], rtol=1e-3, atol=1e-4)   # not squared


def test_transform_with_hnswlib_places_events_like_pynndescent(fitted):
    query = _data(1_000, 2)
    before = fitted.transform(query)
    pnd = fitted._knn_search_index
    assert drc_umap_index.needs_index(fitted)
    assert drc_umap_index.attach_search_index(fitted, threads=1)
    try:
        after = fitted.transform(query)
        assert not drc_umap_index.needs_index(fitted)
        assert not drc_umap_index.attach_search_index(fitted, threads=1)
    finally:
        fitted._knn_search_index = pnd
    span = np.ptp(fitted.embedding_, axis=0).max()
    shift = np.linalg.norm(after - before, axis=1) / span
    assert np.median(shift) < 0.01 and np.quantile(shift, 0.95) < 0.05


def test_small_data_model_keeps_umaps_exact_path():
    reducer = umap.UMAP(n_neighbors=15, n_epochs=30, random_state=42, n_jobs=1).fit(_data(1_000, 3))
    assert not drc_umap_index.needs_index(reducer)
    assert not drc_umap_index.attach_search_index(reducer, threads=1)

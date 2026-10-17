"""
drc_umap_index.py — hnswlib nearest-neighbour search for UMAP's transform
=========================================================================
Companion module to ``dr_clustering_tab.py`` (not a plugin: the filename
does not end in ``_tab.py``).

``UMAP.transform`` finds each new event's neighbours in the training pool
with pynndescent's search graph (``reducer._knn_search_index``), then builds
the membership graph and optimises the new points' positions.  On Tombi Lung
features the pynndescent search took 12.4 s for one 15,485-event sample
(0.8 ms per event; 32.5 s of a 41.9 s embed of six samples), where an hnswlib
index took 0.19 s.  Against exact neighbours, recall on a 16,450-event pool was
0.984 for pynndescent and 0.9998 for hnswlib.

``attach_search_index`` replaces that search object with an hnswlib index
built on the training pool.  Everything else in ``transform`` is UMAP's own
code.  The replacement provides the two members ``UMAP.transform`` uses,
``query(X, k, epsilon)`` and ``_angular_trees`` (umap-learn 0.5.12); a
check of both against a new umap-learn version belongs with the upgrade.

The pynndescent index is dropped, so its memory is freed.  It is used by
``transform`` only (and by ``UMAP.update``, which the plugin does not call).

Pure numpy and hnswlib; no Qt.
"""

from __future__ import annotations

import numpy as np

# UMAP metric -> hnswlib space.  hnswlib's 'l2' is the squared distance;
# its 'cosine' is 1 - cosine similarity, the same as UMAP's 'cosine'.
HNSW_SPACES = {'euclidean': 'l2', 'l2': 'l2', 'cosine': 'cosine'}
EF_CONSTRUCTION = 200
M = 16
SEED = 42
MIN_QUERY_EF = 50


class HnswSearchIndex:
    """hnswlib index standing in for pynndescent's search graph in
    UMAP.transform."""

    # Read by UMAP.transform to choose pynndescent's epsilon; not used here
    _angular_trees = False

    def __init__(self, data: np.ndarray, space: str, threads: int):
        import hnswlib
        self._space = space
        self._threads = max(int(threads), 1)
        self._index = hnswlib.Index(space=space, dim=data.shape[1])
        self._index.init_index(max_elements=len(data), ef_construction=EF_CONSTRUCTION,
                               M=M, random_seed=SEED)
        self._index.add_items(data, num_threads=self._threads)

    def query(self, X, k: int, epsilon=None):
        """(indices int32, distances float32) of each row's k nearest pool events."""
        self._index.set_ef(max(MIN_QUERY_EF, 2 * int(k)))
        labels, dist = self._index.knn_query(np.ascontiguousarray(X, dtype=np.float32),
                                             k=int(k), num_threads=self._threads)
        if self._space == 'l2':
            dist = np.sqrt(np.maximum(dist, 0.0))
        return labels.astype(np.int32), dist.astype(np.float32)


def needs_index(reducer) -> bool:
    """True for a fitted UMAP whose transform still searches a pynndescent
    index (dense data, a metric hnswlib supports, not the small-data exact
    path below 4,096 events)."""
    search = getattr(reducer, '_knn_search_index', None)
    return (getattr(reducer, 'metric', None) in HNSW_SPACES
            and search is not None and not isinstance(search, HnswSearchIndex)
            and not getattr(reducer, '_small_data', True)
            and not getattr(reducer, '_sparse_data', True))


def attach_search_index(reducer, threads: int) -> bool:
    """Give a fitted UMAP an hnswlib search index on its training pool.

    threads: 1 builds the index deterministically (a multi-threaded build
    inserts events in a scheduler-dependent order).  Returns False, leaving
    the reducer unchanged, when needs_index() is False."""
    if not needs_index(reducer):
        return False
    data = np.ascontiguousarray(reducer._raw_data, dtype=np.float32)
    reducer._knn_search_index = HnswSearchIndex(data, HNSW_SPACES[reducer.metric], threads)
    return True

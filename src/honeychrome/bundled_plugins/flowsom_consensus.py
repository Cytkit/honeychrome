"""
flowsom_consensus.py — SOM training + consensus metaclustering for FlowSOM
===========================================================================
Companion module to ``drc_clustering.py`` (filename intentionally NOT
``*_tab.py``, matching the ``drc_pipeline.py``/``drc_logging.py`` convention
so ``plugin_loaders`` won't try to load it as a separate tab).

Replaces ``saeyslab/flowsom`` + ``anndata``/``mudata`` for the DR/Clustering
plugin's FlowSOM path. ``saeyslab/flowsom`` internally mutates AnnData/MuData
arrays in place (e.g. writing NaN into a cluster-level MuData's ``.X``),
which is incompatible with pandas 3.0's mandatory Copy-on-Write — no
released ``anndata``/``flowsom`` pair currently supports it (confirmed via
anndata's own dev-branch release notes: AnnData.X only becomes properly
copy-on-write in the *upcoming* 0.14).

The SOM step itself no longer uses ``angelolab/pyFlowSOM``: its C
extension (``flowsom.c`` / ``cyFlowSOM.pyx``) is a single-threaded port of
the classic online/per-event Kohonen trainer with no confirmed arm64 wheel
(flagged, never resolved, in DR_CLUSTERING_REVAMP_PLAN.md). SOM training and
assignment now go through ``som_kernel_wrapper.py`` — an in-house
OpenMP-accelerated *batch* SOM kernel, the same algorithm AutoSpectral's
``AutoSpectralRcpp::som_train_batch_cpp()`` already uses (see that
project's ``som.R``/``som.cpp`` for the reference this was ported from).
Batch training reads the codebook as it stood at the end of the previous
epoch and updates it once per epoch (Gaussian neighbourhood kernel)
instead of once per event (bubble kernel) — a different schedule from
classic FlowSOM, not a bit-for-bit reproduction, but consensus
metaclustering below is specifically designed to be robust to this kind of
training-run codebook variation.

The R FlowSOM algorithm's consensus hierarchical metaclustering step
(``ConsensusClusterPlus``, Monti et al. 2003) is implemented from scratch
here, independent of the SOM training method.

Consensus metaclustering operates on the SOM's node codebook (xdim*ydim
vectors, typically 100-400), not raw events, so it stays cheap regardless
of how many cells are in the experiment.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.spatial import distance_matrix

import som_kernel_wrapper
from drc_logging import get_logger, log_stage

log = get_logger(__name__)

_DISTF_EUCLIDEAN = 2


def _grid_neighbor_distance(xdim: int, ydim: int) -> np.ndarray:
    """Chebyshev distance between SOM grid nodes -- same convention
    pyFlowSOM's neighborhood_distance() used, and the R FlowSOM/kohonen
    default. Grid topology is unchanged by the kernel swap, only the
    training algorithm that fills the codebook is."""
    grid = np.meshgrid(np.arange(1, xdim + 1), np.arange(1, ydim + 1))
    grid = np.column_stack((grid[0].flat, grid[1].flat))
    return distance_matrix(grid, grid, p=np.inf)


def train_som(data: np.ndarray, xdim: int, ydim: int, n_iter: int,
              seed: int = 42) -> np.ndarray:
    """Train a batch SOM via the in-house OpenMP kernel (som_kernel_wrapper).
    Returns node codebook, shape (xdim*ydim, n_channels).

    Radius/epoch schedule matches AutoSpectral's get.som.codes() (som.R):
    radius anneals from the grid's 67th-percentile neighbour distance down
    to 10% of that (never to 0 -- the batch trainer's Gaussian kernel needs
    a strictly positive radius throughout), one radius per epoch, n_iter
    epochs total.
    """
    log_stage(log, "SOM TRAINING")
    data_f = np.asarray(data, dtype=np.float64)
    ncodes = xdim * ydim

    nhbrdist = _grid_neighbor_distance(xdim, ydim)
    radius_start = float(np.percentile(nhbrdist, 67))
    radius_end = radius_start * 0.1
    radii = np.linspace(radius_start, radius_end, n_iter)

    rng = np.random.default_rng(seed)
    init_idx = rng.choice(data_f.shape[0], ncodes, replace=False)
    init_codes = data_f[init_idx]

    node_weights = som_kernel_wrapper.train_som_batch(
        data_f, init_codes, nhbrdist, radii,
        dist=_DISTF_EUCLIDEAN, n_threads=0,
    )
    log.info("SOM trained: %d nodes (%dx%d), %d channels, %d epochs",
             node_weights.shape[0], xdim, ydim, node_weights.shape[1], n_iter)
    return node_weights


def assign_to_nodes(node_weights: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Map each row of data to its nearest SOM node. Returns 0-based node
    index per row. The in-house kernel returns clean 0-based indices by
    construction -- no defensive index-convention detection needed (that
    was only required for pyFlowSOM's undocumented 1-based return)."""
    node_ids, _dists = som_kernel_wrapper.map_to_codes(
        data, node_weights, dist=_DISTF_EUCLIDEAN, n_threads=0,
    )
    return node_ids.astype(np.int64)


def _consensus_matrix(node_weights: np.ndarray, k: int, H: int,
                      resample_frac: float, linkage_method: str,
                      metric: str, seed) -> np.ndarray:
    """Build the H-round consensus (co-clustering) matrix for a given k.
    Shared by consensus_metacluster (final cut) and consensus_stability (PAC)."""
    n_nodes = node_weights.shape[0]
    sample_size = max(k, int(round(resample_frac * n_nodes)))
    co_occurrence = np.zeros((n_nodes, n_nodes))
    co_cluster = np.zeros((n_nodes, n_nodes))
    rng = np.random.default_rng(seed)

    for _h in range(H):
        sample_idx = rng.choice(n_nodes, size=sample_size, replace=False)
        sub = node_weights[sample_idx]
        Z = linkage(sub, method=linkage_method, metric=metric)
        labels = fcluster(Z, t=k, criterion='maxclust')
        same = np.equal.outer(labels, labels)
        co_cluster[np.ix_(sample_idx, sample_idx)] += same
        co_occurrence[np.ix_(sample_idx, sample_idx)] += 1

    with np.errstate(invalid='ignore', divide='ignore'):
        M = np.where(co_occurrence > 0, co_cluster / co_occurrence, 0.0)
    np.fill_diagonal(M, 1.0)
    return M


def consensus_metacluster(node_weights: np.ndarray, k: int, H: int = 100,
                          resample_frac: float = 0.8,
                          linkage_method: str = 'average',
                          metric: str = 'euclidean',
                          seed=42) -> np.ndarray:
    """ConsensusClusterPlus-style metaclustering of the SOM node codebook.
    Returns 0-based metacluster label per node, shape (n_nodes,)."""
    log_stage(log, "CONSENSUS METACLUSTERING")
    M = _consensus_matrix(node_weights, k, H, resample_frac,
                          linkage_method, metric, seed)
    condensed = squareform(1.0 - M, checks=False)
    Z_final = linkage(condensed, method=linkage_method)
    labels = fcluster(Z_final, t=k, criterion='maxclust') - 1
    log.info("consensus metaclustering: k=%d, H=%d rounds, %d nodes -> %d metaclusters",
             k, H, node_weights.shape[0], len(np.unique(labels)))
    return labels.astype(np.int32)


def _pac(M: np.ndarray, lower: float = 0.1, upper: float = 0.9) -> float:
    """Proportion of Ambiguous Clustering — fraction of pairwise consensus
    values falling in the ambiguous middle band. Lower is more stable."""
    n = M.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = M[iu]
    return float(np.mean((vals > lower) & (vals < upper)))


def consensus_stability(node_weights: np.ndarray, k_range: range,
                        H: int = 100, resample_frac: float = 0.8,
                        linkage_method: str = 'average',
                        metric: str = 'euclidean',
                        seed=42):
    """Run consensus metaclustering across a range of k and score stability
    via PAC. Returns (best_k, {k: pac_score})."""
    log_stage(log, "CONSENSUS STABILITY (auto k)")
    scores = {}
    for k in k_range:
        M = _consensus_matrix(node_weights, k, H, resample_frac,
                              linkage_method, metric, seed)
        scores[k] = _pac(M)
        log.debug("  k=%d: PAC=%.4f", k, scores[k])
    best_k = min(scores, key=scores.get)
    log.info("auto k selection: best_k=%d (PAC=%.4f) over range %s",
             best_k, scores[best_k], list(k_range))
    return best_k, scores
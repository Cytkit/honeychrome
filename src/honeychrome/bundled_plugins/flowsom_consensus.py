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

``angelolab/pyFlowSOM`` has no anndata/mudata dependency at all — plain
numpy in, numpy out — so it's immune to that whole class of issue. It only
implements the SOM step, though; the R FlowSOM algorithm's consensus
hierarchical metaclustering step (``ConsensusClusterPlus``, Monti et al.
2003) is implemented from scratch here.

Consensus metaclustering operates on the SOM's node codebook (xdim*ydim
vectors, typically 100-400), not raw events, so it stays cheap regardless
of how many cells are in the experiment.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from drc_logging import get_logger, log_stage

log = get_logger(__name__)

_DISTF_EUCLIDEAN = 2


def train_som(data: np.ndarray, xdim: int, ydim: int, n_iter: int,
              seed: int = 42) -> np.ndarray:
    """Train a SOM via pyFlowSOM. Returns node codebook, shape (xdim*ydim, n_channels)."""
    import pyFlowSOM
    data_f = np.asarray(data, dtype=np.float64, order='F')
    log_stage(log, "SOM TRAINING")
    node_weights = pyFlowSOM.som(data_f, xdim=xdim, ydim=ydim, rlen=n_iter,
                                  distf=_DISTF_EUCLIDEAN, seed=seed)
    node_weights = np.asarray(node_weights, dtype=np.float64)
    log.info("SOM trained: %d nodes (%dx%d), %d channels",
             node_weights.shape[0], xdim, ydim, node_weights.shape[1])
    return node_weights


def assign_to_nodes(node_weights: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Map each row of data to its nearest SOM node. Returns 0-based node index per row.

    pyFlowSOM's documented return isn't explicit about 0- vs 1-based
    indexing, so this detects it defensively from the observed range rather
    than assuming: [0, n_nodes-1] is left as-is, [1, n_nodes] is shifted
    down by one. Any other range raises, since that would mean the
    indexing convention isn't what either assumption expects.
    """
    import pyFlowSOM
    node_weights_f = np.asarray(node_weights, dtype=np.float64, order='F')
    data_f = np.asarray(data, dtype=np.float64, order='F')
    node_ids, _dists = pyFlowSOM.map_data_to_nodes(node_weights_f, data_f,
                                                    distf=_DISTF_EUCLIDEAN)
    node_ids = np.asarray(node_ids, dtype=np.int64)
    n_nodes = node_weights.shape[0]
    lo, hi = int(node_ids.min()), int(node_ids.max())
    if lo == 1 and hi == n_nodes:
        node_ids = node_ids - 1
    elif lo == 0 and hi == n_nodes - 1:
        pass
    else:
        raise ValueError(
            f"Unexpected node index range [{lo}, {hi}] for {n_nodes} nodes — "
            f"pyFlowSOM.map_data_to_nodes indexing convention may have changed."
        )
    return node_ids


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
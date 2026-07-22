"""
drc_clustering.py — Clustering for the DR/Clustering plugin
===========================================================
Companion to ``dr_clustering_tab.py`` (filename intentionally NOT ``*_tab.py``).

Split out of ``PluginWidget`` so the FlowSOM / Leiden / DBSCAN paths and the
shared label-assignment logic can be inspected and tested in isolation. Uses the
corrected ``drc_pipeline`` for all data loading and ``drc_logging`` for stage
instrumentation.

FlowSOM uses ``flowsom_consensus`` (pyFlowSOM + our own consensus
hierarchical metaclustering) rather than ``saeyslab/flowsom`` — the latter's
internal AnnData/MuData mutation is incompatible with pandas 3.0's
Copy-on-Write. All cluster labels (FlowSOM/Leiden/DBSCAN) are 0-based with
−1 reserved for noise, matching Leiden/DBSCAN and the stats ``range(n_clusters)``
loop by construction — no post-hoc index shift needed for FlowSOM anymore.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import drc_pipeline
import flowsom_consensus
from drc_logging import get_logger, log_stage, log_array

log = get_logger(__name__)


def _noop(_msg: str) -> None:
    pass


def _progress(progress, msg: str) -> None:
    log.info("%s", msg)
    (progress or _noop)(msg)


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

def assign_cluster_colors(state, labels: np.ndarray) -> None:
    """Populate ``state.cluster_colors`` from the colorcet glasbey palette.
    Noise (−1) is grey; metaclusters get distinct hues."""
    import colorcet as cc
    palette = cc.glasbey
    unique = sorted(int(l) for l in np.unique(labels))
    state.cluster_colors = {}
    color_idx = 0
    for lbl in unique:
        if lbl == -1:
            state.cluster_colors[-1] = '#7f7f7f'
        else:
            state.cluster_colors[lbl] = palette[color_idx % len(palette)]
            color_idx += 1
    log.info("assigned %d cluster colours (labels %s)",
             len(state.cluster_colors), unique[:20])


# ---------------------------------------------------------------------------
# FlowSOM
# ---------------------------------------------------------------------------

def run_flowsom(controller, state, params: dict, progress=None):
    """Train a SOM (pyFlowSOM) and consensus-metacluster its nodes, then
    assign 0-based metacluster labels to every training sample.

    Replaces saeyslab/flowsom, which internally mutates AnnData/MuData
    arrays in place — incompatible with pandas 3.0's mandatory
    Copy-on-Write. pyFlowSOM has no anndata/mudata dependency at all;
    metaclustering is our own consensus-hierarchical implementation in
    flowsom_consensus.py, matching the R algorithm's ConsensusClusterPlus
    step rather than a single hierarchical cut.
    """
    log_stage(log, "FLOWSOM")
    data = drc_pipeline.load_training_pool(controller, state)
    if data is None:
        _progress(progress, "FlowSOM: no training data.")
        return
    log_array(log, "flowsom_input", data,
              [c for c in state.selected_channels if c not in drc_pipeline.META_CHANNELS])

    xdim, ydim = params['xdim'], params['ydim']
    n_meta = params['n_metaclusters']  # 0 = auto
    n_iter = params['n_iter']

    _progress(progress, f"Training SOM ({xdim}×{ydim} grid, rlen={n_iter}) …")
    node_weights = flowsom_consensus.train_som(data, xdim, ydim, n_iter, seed=42)

    if n_meta == 0:
        _progress(progress, "Selecting metacluster count via consensus stability …")
        k, _scores = flowsom_consensus.consensus_stability(node_weights, range(2, 31))
        _progress(progress, f"Auto-selected {k} metaclusters.")
    else:
        k = n_meta

    _progress(progress, f"Consensus metaclustering ({k} metaclusters) …")
    node_to_meta = flowsom_consensus.consensus_metacluster(node_weights, k)

    flowsom_assign_all(controller, state, node_weights, node_to_meta, progress)
    state.n_clusters = k
    state.active_clustering_algorithm = 'FlowSOM'
    state.trained_reducers['FlowSOM'] = {
        'node_weights': node_weights,
        'node_to_meta': node_to_meta,
        'xdim': xdim, 'ydim': ydim,
    }
    _progress(progress, f"FlowSOM done: {k} metaclusters.")


def flowsom_assign_all(controller, state, node_weights, node_to_meta, progress=None) -> None:
    """Map each training sample's events to SOM nodes, then to metaclusters."""
    raw_subdir = controller.experiment.settings['raw']['raw_samples_subdirectory']
    all_samples = list(controller.experiment.samples.get('all_samples', {}).keys())

    state.cluster_labels = {}
    for sample_key in all_samples:
        try:
            rel_path = str(Path(sample_key).relative_to(raw_subdir))
        except ValueError:
            rel_path = sample_key
        if rel_path not in state.training_sample_ids:
            continue
        sample_data = drc_pipeline.load_sample_features(controller, state, rel_path)
        if sample_data is None:
            continue
        try:
            node_ids = flowsom_consensus.assign_to_nodes(node_weights, sample_data)
            labels = node_to_meta[node_ids].astype(np.int32)
            state.cluster_labels[rel_path] = labels
            _progress(progress, f"  FlowSOM assigned {rel_path}: {len(labels):,} events "
                                f"({len(np.unique(labels))} metaclusters)")
        except Exception as e:
            _progress(progress, f"  FlowSOM assignment failed for {rel_path}: {e}")

    if state.cluster_labels:
        all_labels = np.concatenate(list(state.cluster_labels.values()))
        assign_cluster_colors(state, all_labels)


# ---------------------------------------------------------------------------
# Leiden
# ---------------------------------------------------------------------------

def get_training_embeddings(controller, state, algo: str | None) -> np.ndarray | None:
    """Concatenated training-sample embeddings for *algo*, or the transformed
    feature matrix if algo is None/unavailable."""
    if algo and algo in state.embeddings:
        emb_dict = state.embeddings[algo]
        chunks = [emb_dict[r] for r in state.training_sample_ids if r in emb_dict]
        if chunks:
            return np.concatenate(chunks, axis=0).astype(np.float32)
    return drc_pipeline.load_training_pool(controller, state)


def run_leiden(controller, state, params: dict, progress=None) -> None:
    """Build a kNN graph and run Leiden community detection."""
    import igraph
    import leidenalg
    import hnswlib

    log_stage(log, "LEIDEN")
    space = params.get('_space', 'raw')
    dr_algo = params.get('_dr_algo', None)

    if space == 'dr' and dr_algo:
        data = get_training_embeddings(controller, state, dr_algo)
    else:
        data = drc_pipeline.load_training_pool(controller, state)
        dr_algo = None
    if data is None:
        _progress(progress, "Leiden: no training data.")
        return
    log_array(log, "leiden_input", data)

    resolution = params['resolution']
    n_neighbors = params['n_neighbors']

    if space != 'dr' and state.umap_knn_index is not None:
        # Only reuse the UMAP index when clustering the raw feature space it was
        # built on. (Building on a DR embedding uses a fresh index below.)
        _progress(progress, "Reusing UMAP hnswlib kNN index for Leiden …")
        index = state.umap_knn_index
    else:
        _progress(progress, f"Building hnswlib kNN index (k={n_neighbors}) for Leiden …")
        index = hnswlib.Index(space='l2', dim=data.shape[1])
        index.init_index(max_elements=len(data), ef_construction=200, M=16)
        index.add_items(data)
        index.set_ef(50)

    _progress(progress, "Querying kNN …")
    neigh, _ = index.knn_query(data, k=n_neighbors)

    _progress(progress, "Building igraph and running Leiden …")
    n = len(data)
    edges = [(i, int(j)) for i, row in enumerate(neigh) for j in row if int(j) != i]
    g = igraph.Graph(n=n, edges=edges, directed=False)
    g.simplify()

    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution, seed=42)
    train_labels = np.array(partition.membership, dtype=np.int32)

    nearest_centroid_assign_all(controller, state, data, train_labels,
                                dr_space=dr_algo if space == 'dr' else None,
                                progress=progress)
    n_cl = int(train_labels.max()) + 1
    state.n_clusters = n_cl
    state.active_clustering_algorithm = 'Leiden'
    _progress(progress, f"Leiden done: {n_cl} communities.")


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------

def run_dbscan(controller, state, params: dict, progress=None) -> None:
    """Run DBSCAN on the chosen feature space (default: transformed features)."""
    from sklearn.cluster import DBSCAN

    log_stage(log, "DBSCAN")
    space = params.get('_space', 'raw')
    dr_algo = params.get('_dr_algo', None)

    if space == 'dr' and dr_algo:
        data = get_training_embeddings(controller, state, dr_algo)
        space_label = f'{dr_algo} embedding'
    else:
        data = drc_pipeline.load_training_pool(controller, state)
        space_label = 'transformed feature space'
    if data is None:
        _progress(progress, "DBSCAN: no training data.")
        return
    log_array(log, "dbscan_input", data)

    _progress(progress, f"Running DBSCAN on {space_label} "
                        f"(eps={params['eps']}, min_samples={params['min_samples']}) …")
    db = DBSCAN(eps=params['eps'], min_samples=params['min_samples'], n_jobs=-1).fit(data)
    train_labels = db.labels_.astype(np.int32)
    n_noise = int(np.sum(train_labels == -1))
    n_cl = int(train_labels.max()) + 1
    _progress(progress, f"DBSCAN: {n_cl} cluster(s), {n_noise} noise events (label −1)")

    nearest_centroid_assign_all(controller, state, data, train_labels,
                                dr_space=dr_algo if space == 'dr' else None,
                                progress=progress)
    state.n_clusters = n_cl
    state.active_clustering_algorithm = 'DBSCAN'


# ---------------------------------------------------------------------------
# Shared assignment
# ---------------------------------------------------------------------------

def nearest_centroid_assign_all(controller, state, train_data, train_labels,
                                dr_space: str | None = None, progress=None) -> None:
    """Assign cluster labels to all samples by nearest-centroid in the SAME
    space the clustering was computed in (DR embedding or transformed features)."""
    from sklearn.neighbors import NearestCentroid

    mask = train_labels != -1
    if mask.sum() == 0:
        _progress(progress, "  All training points are noise — nothing to assign.")
        return
    clf = NearestCentroid().fit(train_data[mask], train_labels[mask])

    all_samples = list(controller.experiment.samples.get('all_samples', {}).keys())
    raw_subdir = controller.experiment.settings['raw']['raw_samples_subdirectory']
    state.cluster_labels = {}

    for sample_key in all_samples:
        try:
            rel_path = str(Path(sample_key).relative_to(raw_subdir))
        except ValueError:
            rel_path = sample_key

        # Only assign training samples — matches flowsom_assign_all's behavior.
        # Excludes single-stain controls etc. that were never in the training pool.
        if rel_path not in state.training_sample_ids:
            continue

        if dr_space:
            sample_data = state.embeddings.get(dr_space, {}).get(rel_path)
            if sample_data is None:
                _progress(progress, f"  No {dr_space} embedding for {rel_path} — skipping")
                continue
        else:
            sample_data = drc_pipeline.load_sample_features(controller, state, rel_path)
        if sample_data is None:
            continue
        try:
            labels = clf.predict(sample_data).astype(np.int32)
            state.cluster_labels[rel_path] = labels
        except Exception as e:
            _progress(progress, f"  Assignment failed for {rel_path}: {e}")

    all_labels = (np.concatenate(list(state.cluster_labels.values()))
                  if state.cluster_labels else train_labels)
    assign_cluster_colors(state, all_labels)
    log.info("assigned labels to %d samples", len(state.cluster_labels))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_clustering(controller, state, algo: str, params: dict,
                   progress=None) -> None:
    """Run the selected clustering algorithm (called from the worker thread)."""
    if algo == 'FlowSOM':
        run_flowsom(controller, state, params, progress)
    elif algo == 'Leiden':
        run_leiden(controller, state, params, progress)
    elif algo == 'DBSCAN':
        run_dbscan(controller, state, params, progress)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algo}")

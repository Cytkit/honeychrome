"""
drc_clustering.py — Clustering for the DR/Clustering plugin
===========================================================
Companion to ``dr_clustering_tab.py`` (filename intentionally NOT ``*_tab.py``).

Split out of ``PluginWidget`` so the FlowSOM / Leiden / HDBSCAN paths and the
shared label-assignment logic can be inspected and tested in isolation. Uses the
corrected ``drc_pipeline`` for all data loading and ``drc_logging`` for stage
instrumentation.

FlowSOM uses ``flowsom_consensus`` (pyFlowSOM + our own consensus
hierarchical metaclustering) rather than ``saeyslab/flowsom`` — the latter's
internal AnnData/MuData mutation is incompatible with pandas 3.0's
Copy-on-Write. All cluster labels (FlowSOM/Leiden/HDBSCAN) are 0-based with
−1 reserved for noise, matching Leiden/HDBSCAN and the stats ``range(n_clusters)``
loop by construction — no post-hoc index shift needed for FlowSOM anymore.

Assignment architecture (Items 8/9/11)
---------------------------------------
Training Samples only ever decide what fits the model (SOM / kNN graph /
HDBSCAN density estimate) — who gets labelled afterward is a separate
question, controlled by two independent UI toggles collected into
``params`` by ``ConfigTab._on_run_clustering_clicked``:

- ``params['_event_cap']`` (Item 11): ``None`` (default) trains on every
  gated event from every training sample; an int caps each training
  sample's contribution ("Downsample training data" checked). Whenever a
  training sample's events were NOT capped out of the pool, Leiden and
  HDBSCAN's own fit output already contains that sample's exact label for
  every one of its events — no approximate re-assignment needed for it at
  all (see ``_assign_by_slicing_or_predict``). FlowSOM has no such
  shortcut: a SOM never outputs per-event labels directly, so mapping
  every event onto the trained grid (``assign_to_nodes``) is always
  required regardless of ``_event_cap`` — Item 11 only changes how much
  data trains the grid, not how assignment works for it.
- ``params['_assign_all_samples']`` (Item 9): ``False`` (default) restricts
  labelling to ``state.training_sample_ids``; ``True`` additionally labels
  every other non-control sample in the experiment, via the same
  approximate method used for any downsampled/out-of-pool events.
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


def _snapshot_marker_values(controller, state, rel_path: str, labels: np.ndarray,
                            af_state=None) -> None:
    """
    Load UNTRANSFORMED marker values for *rel_path* right now -- the same
    instant *labels* was computed -- and cache them in
    state.cluster_marker_values, guaranteed row-for-row aligned to *labels*
    since nothing about gates/channels can have changed in between. Cached
    for archiving into the clustering run (see archive_clustering_run's
    marker_values payload); skipped (with a log warning) if lengths still
    don't match, which should only happen if two loads of the exact same
    sample somehow disagree.
    """
    mv = drc_pipeline.load_sample_marker_values(controller, state, rel_path, af_state=af_state)
    if mv is None:
        return
    values, names = mv
    if len(values) != len(labels):
        log.warning(
            "_snapshot_marker_values: %s -- marker values (%d) vs labels (%d) "
            "length mismatch even at classification time -- skipping snapshot.",
            rel_path, len(values), len(labels),
        )
        return
    state.cluster_marker_values[rel_path] = (values, names)


# ---------------------------------------------------------------------------
# Shared assignment scope + training pool (Items 8/9/11)
# ---------------------------------------------------------------------------

def _assignable_sample_paths(controller, state, assign_all: bool) -> set[str]:
    """
    Which samples should receive cluster labels this run (Item 9).

    assign_all=True: every experiment sample except single-stain/
    unstained controls -- those are spectral reference samples, never
    biological samples to cluster (same exclusion
    dr_clustering_tab._non_control_sample_paths already applies to the
    Training Samples picker).

    assign_all=False (default): only samples in state.training_sample_ids.
    """
    samples = controller.experiment.samples
    raw_subdir = controller.experiment.settings['raw']['raw_samples_subdirectory']

    if not assign_all:
        return set(state.training_sample_ids)

    all_samples_abs = list(samples.get('all_samples', {}).keys())
    excluded = set(samples.get('single_stain_controls', []) or []) \
             | set(samples.get('unstained_samples', []) or [])

    def _to_rel(abs_key):
        try:
            return str(Path(abs_key).relative_to(raw_subdir))
        except ValueError:
            return abs_key

    return {_to_rel(sp) for sp in all_samples_abs if sp not in excluded}


def _pool_training_data_with_boundaries(controller, state, event_cap: int | None,
                                        seed: int = 42, af_state=None):
    """
    Clustering-specific training pool (Item 11) -- separate from
    drc_pipeline.load_training_pool (still used by DR, which always
    applies state.n_training_events): event_cap=None (the new default)
    means every gated event from every training sample, no downsampling;
    an int caps each training sample's contribution the same way DR's
    pool always has.

    Returns (data, boundaries); boundaries is
    [(rel_path, start_row, end_row, was_downsampled), ...] in the same
    row order as data, so a fit's own label output can be sliced
    directly for any sample that wasn't downsampled -- see
    _assign_by_slicing_or_predict. was_downsampled is False whenever
    event_cap is None or the sample had <= event_cap gated events to
    begin with.
    """
    rng = np.random.default_rng(seed)
    chunks = []
    boundaries = []
    row = 0
    for rel_path in state.training_sample_ids:
        feats = drc_pipeline.load_sample_features(controller, state, rel_path, af_state=af_state)
        if feats is None or feats.shape[0] == 0:
            log.warning("  %s contributed no events", rel_path)
            continue
        was_downsampled = bool(event_cap) and feats.shape[0] > event_cap
        chunk = drc_pipeline._downsample(feats, event_cap, rng) if was_downsampled else feats
        log.info("  %s -> %d events%s", rel_path, chunk.shape[0],
                 " after downsample" if was_downsampled else " (all gated events)")
        boundaries.append((rel_path, row, row + chunk.shape[0], was_downsampled))
        row += chunk.shape[0]
        chunks.append(chunk)
    if not chunks:
        log.error("no training data could be loaded")
        return None, []
    data = np.concatenate(chunks, axis=0).astype(np.float32)
    log_array(log, "clustering_training_pool", data,
              [c for c in state.selected_channels if c not in drc_pipeline.META_CHANNELS])
    return data, boundaries


def _assign_one_sample_approximate(controller, state, rel_path, predict_fn,
                                   dr_space, progress, af_state) -> None:
    """Load one sample's full event set and label it via *predict_fn* --
    used for a training sample that WAS downsampled (its beyond-the-cap
    events aren't covered by the fit -- predict_fn re-runs over the whole
    sample for simplicity, since that's cheap relative to the fit itself),
    and for a non-training sample when assign_all is set."""
    if dr_space:
        sample_data = state.embeddings.get(dr_space, {}).get(rel_path)
        if sample_data is None:
            _progress(progress, f"  No {dr_space} embedding for {rel_path} — skipping")
            return
    else:
        sample_data = drc_pipeline.load_sample_features(controller, state, rel_path, af_state=af_state)
    if sample_data is None:
        return
    try:
        labels = predict_fn(sample_data)
        state.cluster_labels[rel_path] = labels
        _snapshot_marker_values(controller, state, rel_path, labels, af_state=af_state)
        if dr_space:
            state.cluster_dr_positions[rel_path] = np.asarray(sample_data)
    except Exception as e:
        _progress(progress, f"  Assignment failed for {rel_path}: {e}")


def _assign_by_slicing_or_predict(controller, state, pooled_data, boundaries, fit_labels,
                                  predict_fn, assign_all: bool,
                                  dr_space: str | None = None, progress=None,
                                  af_state=None) -> None:
    """
    Shared assignment for Leiden/HDBSCAN (Items 8/9/11): whenever a
    training sample's events were NOT downsampled out of the pool
    (was_downsampled is False in *boundaries* -- always true when
    event_cap is None, the default), its EXACT labels come straight out
    of the fit's own output (pooled_data[start:end] <-> fit_labels[start:end])
    -- no k-NN-vote / approximate_predict needed for it at all. Only a
    sample that WAS downsampled, or (if assign_all) a sample outside the
    training list entirely, goes through predict_fn.

    predict_fn: callable(sample_data: np.ndarray) -> np.ndarray[int32] --
        the approximate method specific to the calling algorithm (k-NN
        majority vote for Leiden, hdbscan.approximate_predict for
        HDBSCAN).
    """
    state.cluster_labels = {}
    state.cluster_marker_values = {}
    state.cluster_dr_positions = {}

    trained_rel_paths = set()
    for rel_path, start, end, was_downsampled in boundaries:
        trained_rel_paths.add(rel_path)
        if not was_downsampled:
            labels = fit_labels[start:end].astype(np.int32)
            state.cluster_labels[rel_path] = labels
            _snapshot_marker_values(controller, state, rel_path, labels, af_state=af_state)
            if dr_space:
                state.cluster_dr_positions[rel_path] = np.asarray(pooled_data[start:end])
            _progress(progress, f"  {rel_path}: {len(labels):,} events labelled directly "
                                f"from the fit (no prediction needed)")
        else:
            _assign_one_sample_approximate(controller, state, rel_path, predict_fn,
                                           dr_space, progress, af_state)

    if assign_all:
        assignable = _assignable_sample_paths(controller, state, assign_all=True)
        for rel_path in sorted(assignable - trained_rel_paths):
            _assign_one_sample_approximate(controller, state, rel_path, predict_fn,
                                           dr_space, progress, af_state)

    all_labels = (np.concatenate(list(state.cluster_labels.values()))
                  if state.cluster_labels else np.asarray(fit_labels, dtype=np.int32))
    final_ids, final_counts = np.unique(all_labels, return_counts=True)
    log.info("assign: FULL assigned label distribution: %s",
             dict(zip(final_ids.tolist(), final_counts.tolist())))
    assign_cluster_colors(state, all_labels)
    log.info("assigned labels to %d samples", len(state.cluster_labels))


# ---------------------------------------------------------------------------
# FlowSOM
# ---------------------------------------------------------------------------

def run_flowsom(controller, state, params: dict, progress=None, af_state=None):
    """Train a SOM (pyFlowSOM) and consensus-metacluster its nodes, then
    assign 0-based metacluster labels to samples.

    Replaces saeyslab/flowsom, which internally mutates AnnData/MuData
    arrays in place — incompatible with pandas 3.0's mandatory
    Copy-on-Write. pyFlowSOM has no anndata/mudata dependency at all;
    metaclustering is our own consensus-hierarchical implementation in
    flowsom_consensus.py, matching the R algorithm's ConsensusClusterPlus
    step rather than a single hierarchical cut.

    Item 11: trains on every gated event from every training sample by
    default (params['_event_cap'] is None unless "Downsample training
    data" is checked). Unlike Leiden/HDBSCAN, a SOM never outputs
    per-event labels directly -- mapping every event onto the trained
    grid (assign_to_nodes, in flowsom_assign_all) is its native,
    always-required mechanism, so there's no exact-slice shortcut to
    take here even when nothing was downsampled; only the "train on more
    data by default" half of Item 11 applies to FlowSOM.

    af_state: optional AF snapshot — see drc_pipeline.apply_unmixing_af_aware()
        docstring. Must be passed when called from a background worker thread.
    """
    log_stage(log, "FLOWSOM")
    event_cap = params.get('_event_cap')
    data, _boundaries = _pool_training_data_with_boundaries(
        controller, state, event_cap, af_state=af_state)
    if data is None:
        _progress(progress, "FlowSOM: no training data.")
        return
    log_array(log, "flowsom_input", data,
              [c for c in state.selected_channels if c not in drc_pipeline.META_CHANNELS])

    xdim, ydim = params['xdim'], params['ydim']
    n_meta = params['n_metaclusters']  # 0 = auto
    n_iter = params['n_iter']

    _progress(progress, f"Training SOM ({xdim}×{ydim} grid, rlen={n_iter}, "
                        f"{len(data):,} events) …")
    node_weights = flowsom_consensus.train_som(data, xdim, ydim, n_iter, seed=42)

    if n_meta == 0:
        _progress(progress, "Selecting metacluster count via consensus stability …")
        k, _scores = flowsom_consensus.consensus_stability(node_weights, range(2, 31))
        _progress(progress, f"Auto-selected {k} metaclusters.")
    else:
        k = n_meta

    _progress(progress, f"Consensus metaclustering ({k} metaclusters) …")
    node_to_meta = flowsom_consensus.consensus_metacluster(node_weights, k)
    meta_ids, node_counts = np.unique(node_to_meta, return_counts=True)
    log.info("run_flowsom: SOM nodes per metacluster: %s",
             dict(zip(meta_ids.tolist(), node_counts.tolist())))

    node_event_counts = flowsom_assign_all(
        controller, state, node_weights, node_to_meta,
        assign_all=params.get('_assign_all_samples', False),
        progress=progress, af_state=af_state)
    state.n_clusters = k
    state.active_clustering_algorithm = 'FlowSOM'
    state.trained_reducers['FlowSOM'] = {
        'node_weights': node_weights,
        'node_to_meta': node_to_meta,
        'node_counts': node_event_counts,
        'xdim': xdim, 'ydim': ydim,
    }
    _progress(progress, f"FlowSOM done: {k} metaclusters.")


def flowsom_assign_all(controller, state, node_weights, node_to_meta,
                       assign_all: bool = False, progress=None,
                       af_state=None) -> np.ndarray:
    """Map each sample's events to SOM nodes, then to metaclusters.

    Returns node_counts: np.ndarray shape (n_nodes,) — total event count
    assigned to each SOM node across all assigned samples, used by the
    FlowSOM MST tree view to size node bubbles.

    assign_all: see _assignable_sample_paths (Item 9) -- default (False)
        restricts to state.training_sample_ids; True additionally labels
        every other non-control sample.
    af_state: optional AF snapshot — see drc_pipeline.apply_unmixing_af_aware()
        docstring. Must be passed when called from a background worker thread.
    """
    assignable = _assignable_sample_paths(controller, state, assign_all)
    raw_subdir = controller.experiment.settings['raw']['raw_samples_subdirectory']
    all_samples = list(controller.experiment.samples.get('all_samples', {}).keys())

    state.cluster_labels = {}
    state.cluster_marker_values = {}
    node_counts = np.zeros(len(node_weights), dtype=np.int64)
    for sample_key in all_samples:
        try:
            rel_path = str(Path(sample_key).relative_to(raw_subdir))
        except ValueError:
            rel_path = sample_key
        if rel_path not in assignable:
            continue
        sample_data = drc_pipeline.load_sample_features(controller, state, rel_path, af_state=af_state)
        if sample_data is None:
            continue
        try:
            node_ids = flowsom_consensus.assign_to_nodes(node_weights, sample_data)
            node_counts += np.bincount(node_ids, minlength=len(node_weights))
            labels = node_to_meta[node_ids].astype(np.int32)
            state.cluster_labels[rel_path] = labels
            _snapshot_marker_values(controller, state, rel_path, labels, af_state=af_state)
            _progress(progress, f"  FlowSOM assigned {rel_path}: {len(labels):,} events "
                                f"({len(np.unique(labels))} metaclusters)")
        except Exception as e:
            _progress(progress, f"  FlowSOM assignment failed for {rel_path}: {e}")

    if state.cluster_labels:
        all_labels = np.concatenate(list(state.cluster_labels.values()))
        assign_cluster_colors(state, all_labels)

    return node_counts


def build_flowsom_tree(node_weights: np.ndarray, node_to_meta: np.ndarray,
                       node_counts: np.ndarray) -> dict:
    """
    Classic FlowSOM tree: minimum spanning tree over SOM codebook vectors
    (Euclidean distance), laid out with igraph's Kamada-Kawai algorithm —
    the same approach R FlowSOM's BuildMST/PlotStars uses.

    Returns {'positions': (n_nodes, 2) ndarray, 'edges': list[(i, j)]}.
    Pure function — no state access, safe to call from the Workspace tab's
    render path directly.
    """
    import igraph
    from scipy.spatial.distance import pdist, squareform
    from scipy.sparse.csgraph import minimum_spanning_tree

    dist = squareform(pdist(node_weights, metric='euclidean'))
    mst_sparse = minimum_spanning_tree(dist)
    mst_coo = mst_sparse.tocoo()
    edges = list(zip(mst_coo.row.tolist(), mst_coo.col.tolist()))

    n_nodes = node_weights.shape[0]
    g = igraph.Graph(n=n_nodes, edges=edges)
    layout = g.layout_kamada_kawai()
    positions = np.array(layout.coords)
    return {'positions': positions, 'edges': edges}


# ---------------------------------------------------------------------------
# Leiden
# ---------------------------------------------------------------------------

def get_training_embeddings(controller, state, algo: str | None, event_cap: int | None,
                            af_state=None, seed: int = 42):
    """Concatenated training-sample embeddings for *algo*, with per-sample
    row boundaries (Item 11) -- respects event_cap the same way the raw
    feature-space pool does (Item 14 fix: this branch previously ignored
    event_cap entirely and always used each sample's FULL embedding,
    regardless of the "Downsample training data" checkbox -- confirmed
    from a real run where "Downsample training data" was checked but the
    HDBSCAN log still showed every event). Falls back to the pooled raw
    feature space (respecting event_cap) if *algo*'s embeddings aren't
    available.

    Returns (data, boundaries); af_state only used on the fallback path.
    boundaries: [(rel_path, start_row, end_row, was_downsampled), ...].
    """
    if algo and algo in state.embeddings:
        emb_dict = state.embeddings[algo]
        rng = np.random.default_rng(seed)
        chunks = []
        boundaries = []
        row = 0
        for rel in state.training_sample_ids:
            if rel not in emb_dict:
                continue
            chunk = emb_dict[rel]
            was_downsampled = bool(event_cap) and len(chunk) > event_cap
            if was_downsampled:
                chunk = drc_pipeline._downsample(chunk, event_cap, rng)
            boundaries.append((rel, row, row + len(chunk), was_downsampled))
            row += len(chunk)
            chunks.append(chunk)
        if chunks:
            return np.concatenate(chunks, axis=0).astype(np.float32), boundaries
    return _pool_training_data_with_boundaries(controller, state, event_cap, af_state=af_state)


def run_leiden(controller, state, params: dict, progress=None, af_state=None) -> None:
    """Build a kNN graph and run Leiden community detection.

    Item 11: trains on every gated event from every training sample by
    default (params['_event_cap'] is None unless "Downsample training
    data" is checked) -- every training sample's OWN events then get
    their EXACT label straight out of the partition itself (see
    _assign_by_slicing_or_predict), with no k-NN-vote approximation for
    them at all. k-NN-vote only runs for a sample that WAS downsampled,
    or (Item 9) a non-training sample if "Assign clusters to all
    samples" is checked.

    af_state: optional AF snapshot — see drc_pipeline.apply_unmixing_af_aware()
        docstring. Must be passed when called from a background worker thread.
    """
    import igraph
    import leidenalg
    import hnswlib
    from scipy.stats import mode as _scipy_mode

    log_stage(log, "LEIDEN")
    space = params.get('_space', 'raw')
    dr_algo = params.get('_dr_algo', None)
    event_cap = params.get('_event_cap')

    if space == 'dr' and dr_algo:
        data, boundaries = get_training_embeddings(controller, state, dr_algo, event_cap,
                                                    af_state=af_state)
    else:
        data, boundaries = _pool_training_data_with_boundaries(controller, state, event_cap,
                                                               af_state=af_state)
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
        _progress(progress, f"Building hnswlib kNN index (k={n_neighbors}) for Leiden "
                            f"({len(data):,} events) …")
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

    def _predict(sample_data):
        neigh_ids, _dists = index.knn_query(sample_data, k=n_neighbors)
        neigh_labels = train_labels[neigh_ids]
        return _scipy_mode(neigh_labels, axis=1, keepdims=False).mode.astype(np.int32)

    _assign_by_slicing_or_predict(
        controller, state, data, boundaries, train_labels, _predict,
        assign_all=params.get('_assign_all_samples', False),
        dr_space=dr_algo if space == 'dr' else None,
        progress=progress, af_state=af_state,
    )
    n_cl = int(train_labels.max()) + 1
    state.n_clusters = n_cl
    state.active_clustering_algorithm = 'Leiden'
    _progress(progress, f"Leiden done: {n_cl} communities.")


# ---------------------------------------------------------------------------
# HDBSCAN
# ---------------------------------------------------------------------------

def run_hdbscan(controller, state, params: dict, progress=None, af_state=None) -> None:
    """
    Run HDBSCAN on a DR embedding only (Item 13).

    Confirmed on a real dataset: HDBSCAN degenerates badly directly on
    the full multichannel feature space (hundreds of near-meaningless
    clusters out of a few hundred thousand events) -- high-dimensional
    density estimates just don't behave the way they do on a 2-D DR
    embedding. The UI already prevents selecting raw-feature-space
    HDBSCAN (ConfigTab._on_cl_algo_changed force-selects DR space and
    disables the raw-space radio button whenever HDBSCAN is chosen); the
    early return below is a second, defensive guard in case params ever
    arrive some other way.

    Uses the standalone `hdbscan` package (scikit-learn-contrib/hdbscan),
    not sklearn.cluster.HDBSCAN (Item 8/10): sklearn's built-in has no
    out-of-sample prediction at all -- only fit/fit_predict/labels_ -- so
    assigning anything beyond the fit previously had to fall back to
    NearestCentroid, which assumes convex clusters and can never output
    noise, defeating the point of using HDBSCAN. The standalone package's
    approximate_predict() uses the trained condensed tree directly,
    respecting density-based, non-convex cluster shapes and legitimately
    labelling new points as noise (-1).

    Item 11: trains on every gated event from every training sample by
    default (params['_event_cap'] is None unless "Downsample training
    data" is checked) -- every training sample's OWN events then get
    their EXACT label straight out of clusterer.labels_ (see
    _assign_by_slicing_or_predict), with no approximate_predict call for
    them at all. approximate_predict only runs for a sample that WAS
    downsampled, or (Item 9) a non-training sample if "Assign clusters
    to all samples" is checked.

    af_state: optional AF snapshot — see drc_pipeline.apply_unmixing_af_aware()
        docstring. Must be passed when called from a background worker thread.
    """
    import hdbscan as hdbscan_lib

    log_stage(log, "HDBSCAN")
    space = params.get('_space', 'raw')
    dr_algo = params.get('_dr_algo', None)
    event_cap = params.get('_event_cap')

    if space != 'dr' or not dr_algo:
        _progress(progress, "HDBSCAN requires a DR embedding -- it doesn't behave "
                            "well directly on the full feature space. Train a DR "
                            "algorithm (UMAP/PaCMAP) and select 'DR embedding' above.")
        return

    data, boundaries = get_training_embeddings(controller, state, dr_algo, event_cap,
                                                af_state=af_state)
    space_label = f'{dr_algo} embedding'
    if data is None:
        _progress(progress, "HDBSCAN: no training data.")
        return
    log_array(log, "hdbscan_input", data)

    min_cluster_size = params['min_cluster_size']
    min_samples = params['min_samples'] or None       # 0 = package default (None)
    cluster_selection_epsilon = params['cluster_selection_epsilon']

    _progress(progress, f"Running HDBSCAN on {space_label} "
                        f"({len(data):,} events; min_cluster_size={min_cluster_size}, "
                        f"min_samples={min_samples}, "
                        f"cluster_selection_epsilon={cluster_selection_epsilon}) …")
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        core_dist_n_jobs=-1,   # NOT 'n_jobs' -- different name than sklearn's HDBSCAN
        prediction_data=True,  # required for approximate_predict() below
    ).fit(data)
    train_labels = clusterer.labels_.astype(np.int32)
    n_noise = int(np.sum(train_labels == -1))
    n_cl = int(train_labels.max()) + 1
    _progress(progress, f"HDBSCAN: {n_cl} cluster(s), {n_noise} noise events (label −1)")

    def _predict(sample_data):
        labels, _strengths = hdbscan_lib.approximate_predict(
            clusterer, np.ascontiguousarray(sample_data, dtype=np.float64))
        return labels.astype(np.int32)

    _assign_by_slicing_or_predict(
        controller, state, data, boundaries, train_labels, _predict,
        assign_all=params.get('_assign_all_samples', False),
        dr_space=dr_algo,
        progress=progress, af_state=af_state,
    )
    state.n_clusters = n_cl
    state.active_clustering_algorithm = 'HDBSCAN'


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_clustering(controller, state, algo: str, params: dict,
                   progress=None, af_state=None) -> None:
    """Run the selected clustering algorithm (called from the worker thread).

    af_state: optional (transfer_matrix, af_precomputed, af_spectra) snapshot,
        captured on the main thread before the worker started — see
        drc_pipeline.apply_unmixing_af_aware() docstring.
    """
    if algo == 'FlowSOM':
        run_flowsom(controller, state, params, progress, af_state=af_state)
    elif algo == 'Leiden':
        run_leiden(controller, state, params, progress, af_state=af_state)
    elif algo == 'HDBSCAN':
        run_hdbscan(controller, state, params, progress, af_state=af_state)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algo}")
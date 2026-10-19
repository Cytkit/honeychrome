"""
drc_clustering.py — Clustering for the Honeycluster plugin
===========================================================
Companion to ``dr_clustering_tab.py`` (filename intentionally NOT ``*_tab.py``).

Split out of ``PluginWidget`` so the FlowSOM / Leiden / HDBSCAN paths and the
shared label-assignment logic can be inspected and tested in isolation. Uses the
corrected ``drc_pipeline`` for all data loading and ``drc_logging`` for stage
instrumentation.

FlowSOM uses ``flowsom_consensus`` (an in-house OpenMP batch SOM kernel +
our own consensus hierarchical metaclustering) rather than
``saeyslab/flowsom`` — the latter's internal AnnData/MuData mutation is
incompatible with pandas 3.0's Copy-on-Write. All cluster labels
(FlowSOM/Leiden/HDBSCAN) are 0-based with
−1 reserved for noise, matching Leiden/HDBSCAN and the stats ``range(n_clusters)``
loop by construction — no post-hoc index shift needed for FlowSOM anymore.

Assignment architecture
---------------------------------------
Training Samples only ever decide what fits the model (SOM / kNN graph /
HDBSCAN density estimate) — who gets labelled afterward is a separate
question, controlled by two independent UI toggles collected into
``params`` by ``ConfigTab._on_run_clustering_clicked``:

- ``params['_event_cap']``: ``None`` (default) trains on every
  gated event from every training sample; an int caps each training
  sample's contribution ("Downsample training data" checked). Whenever a
  training sample's events were NOT capped out of the pool, Leiden and
  HDBSCAN's own fit output already contains that sample's exact label for
  every one of its events — no approximate re-assignment needed for it at
  all (see ``_assign_by_slicing_or_predict``). FlowSOM has no such
  shortcut: a SOM never outputs per-event labels directly, so mapping
  every event onto the trained grid (``assign_to_nodes``) is always
  required regardless of ``_event_cap``.
- ``params['_assign_all_samples']`` : ``False`` (default) restricts
  labelling to ``state.training_sample_ids``; ``True`` additionally labels
  every other non-control sample in the experiment, via the same
  approximate method used for any downsampled/out-of-pool events.
"""

from __future__ import annotations

import time
import zlib
from pathlib import Path

import numpy as np

import drc_pipeline
import flowsom_consensus
from drc_logging import get_logger, log_stage, log_array

log = get_logger(__name__)

# Marker values kept per cluster (all samples) for violins and Cluster ID
MARKER_SAMPLE_PER_CLUSTER = 5000
MARKER_SAMPLE_SEED = 42


def clusters_in_dr_space(algo: str, params: dict) -> bool:
    """True when the run clusters a DR embedding.  FlowSOM always trains its
    SOM on the transformed marker values, whatever space is selected."""
    return params.get('_space') == 'dr' and bool(params.get('_dr_algo')) \
        and algo != 'FlowSOM'


def _sample_rows(controller, state, rel_path, dr_space, af_state):
    """Event keys of the rows a sample's labels follow: the gated events, or
    the rows of the DR embedding the run clustered."""
    if dr_space:
        rows = (getattr(state, 'embedding_rows', {}) or {}).get(dr_space, {}).get(rel_path)
        return None if rows is None else np.asarray(rows, dtype=np.int32)
    return drc_pipeline.load_sample_rows(controller, state, rel_path, af_state=af_state)


def _add_time(state, stage: str, seconds: float, events: int) -> None:
    """Accumulate a measured stage on a StagedResult (no-op on the plain state)."""
    times = getattr(state, 'stage_seconds', None)
    if isinstance(times, dict):
        times[stage] = times.get(stage, 0.0) + seconds
        state.stage_events[stage] = state.stage_events.get(stage, 0) + int(events)


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


def _in_order(state, order: list[str]) -> None:
    """Put the per-sample results in *order* (samples are processed with the
    cached ones first, drc_pipeline.cached_first)."""
    for name in ('cluster_labels', 'cluster_rows', 'cluster_marker_values',
                 'cluster_marker_keys'):
        results = getattr(state, name)
        setattr(state, name, {**{rel: results[rel] for rel in order if rel in results},
                              **results})


def _bottom_k(labels: np.ndarray, keys: np.ndarray, k: int) -> np.ndarray:
    """Rows holding the k smallest *keys* of each label >= 0, ascending (int32)."""
    labels = np.asarray(labels)
    order = np.lexsort((keys, labels))
    sorted_labels = labels[order]
    starts = np.flatnonzero(np.r_[True, sorted_labels[1:] != sorted_labels[:-1]])
    group_start = np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    keep = (np.arange(len(order)) - group_start < k) & (sorted_labels >= 0)
    return np.sort(order[keep]).astype(np.int32)


def _prune_marker_sample(state, final: bool = False) -> None:
    """
    Keep each cluster's MARKER_SAMPLE_PER_CLUSTER smallest keys over every
    sample snapshotted so far. Runs once the excess reaches the number
    kept, so memory and cost stay proportional to clusters x cap; *final*
    prunes whatever the excess.
    """
    rels = list(state.cluster_marker_values)
    if not rels:
        return
    sampled_labels = [np.asarray(state.cluster_labels[rel])[state.cluster_marker_values[rel][2]]
                      for rel in rels]
    all_labels = np.concatenate(sampled_labels)
    counts = np.bincount(all_labels) if len(all_labels) else np.zeros(0, dtype=np.int64)
    excess = int(np.clip(counts - MARKER_SAMPLE_PER_CLUSTER, 0, None).sum())
    if excess == 0 or (not final and excess < len(all_labels) - excess):
        return
    keys = np.concatenate([state.cluster_marker_keys[rel] for rel in rels])
    keep = np.zeros(len(all_labels), dtype=bool)
    keep[_bottom_k(all_labels, keys, MARKER_SAMPLE_PER_CLUSTER)] = True
    start = 0
    for rel, labels in zip(rels, sampled_labels):
        mask = keep[start:start + len(labels)]
        start += len(labels)
        if mask.all():
            continue
        values, names, rows = state.cluster_marker_values[rel]
        state.cluster_marker_values[rel] = (values[mask], names, rows[mask])
        state.cluster_marker_keys[rel] = state.cluster_marker_keys[rel][mask]


def _snapshot_marker_values(controller, state, rel_path: str, labels: np.ndarray,
                            af_state=None) -> None:
    """
    Sample one sample's untransformed selected-channel values for the
    run's violins and Cluster ID, at the instant *labels* is computed, from
    the same gated events (drc_pipeline's cache, so nothing is unmixed
    again): state.cluster_marker_values[rel_path] = (values, channel_names,
    rows), rows indexing *labels*.

    Each event gets a seeded random key and each cluster keeps its
    MARKER_SAMPLE_PER_CLUSTER smallest keys over all samples (bottom-k,
    pruned as samples arrive), a uniform random sample without replacement
    that is reproducible for the same samples. Noise (label < 0) is not
    sampled. Skipped, with a log warning, if the gated events are not the
    labelled ones.
    """
    try:
        gated = drc_pipeline.load_unmixed_gated(
            controller, state, drc_pipeline.sample_abs_path(controller, rel_path),
            af_state=af_state)
    except Exception as exc:
        log.exception("_snapshot_marker_values: could not load %s: %s", rel_path, exc)
        return
    if len(gated) != len(labels):
        log.warning(
            "_snapshot_marker_values: %s -- marker values (%d) vs labels (%d) "
            "length mismatch even at classification time -- skipping snapshot.",
            rel_path, len(gated), len(labels),
        )
        return
    label_rows = (getattr(state, 'cluster_rows', {}) or {}).get(rel_path)
    if label_rows is not None:
        current_rows = drc_pipeline.load_sample_rows(controller, state, rel_path,
                                                     af_state=af_state)
        if current_rows is None or not np.array_equal(label_rows, current_rows):
            log.warning("_snapshot_marker_values: %s -- the labelled events are not the "
                        "events gated now -- skipping snapshot.", rel_path)
            return
    rng = np.random.default_rng([MARKER_SAMPLE_SEED, zlib.crc32(rel_path.encode('utf-8'))])
    keys = rng.random(len(labels))
    rows = _bottom_k(labels, keys, MARKER_SAMPLE_PER_CLUSTER)
    values, names = drc_pipeline.select_untransformed_channels(controller, state, gated[rows])
    state.cluster_marker_values[rel_path] = (values, names, rows)
    state.cluster_marker_keys[rel_path] = keys[rows]
    _prune_marker_sample(state)


# ---------------------------------------------------------------------------
# Shared assignment scope + training pool
# ---------------------------------------------------------------------------

def _assignable_sample_paths(controller, state, assign_all: bool) -> set[str]:
    """
    Which samples should receive cluster labels this run.

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
    Clustering-specific training pool -- separate from
    drc_pipeline.load_training_pool_with_rows (used by DR, which always
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
        state.cluster_rows[rel_path] = _sample_rows(controller, state, rel_path, dr_space,
                                                    af_state)
        _snapshot_marker_values(controller, state, rel_path, labels, af_state=af_state)
    except Exception as e:
        _progress(progress, f"  Assignment failed for {rel_path}: {e}")


def _assign_by_slicing_or_predict(controller, state, pooled_data, boundaries, fit_labels,
                                  predict_fn, assign_all: bool,
                                  dr_space: str | None = None, progress=None,
                                  af_state=None) -> None:
    """
    Shared assignment for Leiden/HDBSCAN: whenever a
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
    state.cluster_marker_keys = {}
    state.cluster_rows = {}

    order = [b[0] for b in boundaries]
    by_rel = {b[0]: b for b in boundaries}
    trained_rel_paths = set(order)
    for rel in drc_pipeline.cached_first(controller, state, order, af_state):
        rel_path, start, end, was_downsampled = by_rel[rel]
        if not was_downsampled:
            labels = fit_labels[start:end].astype(np.int32)
            state.cluster_labels[rel_path] = labels
            state.cluster_rows[rel_path] = _sample_rows(controller, state, rel_path,
                                                        dr_space, af_state)
            _snapshot_marker_values(controller, state, rel_path, labels, af_state=af_state)
            _progress(progress, f"  {rel_path}: {len(labels):,} events labelled directly "
                                f"from the fit (no prediction needed)")
        else:
            _assign_one_sample_approximate(controller, state, rel_path, predict_fn,
                                           dr_space, progress, af_state)

    if assign_all:
        assignable = _assignable_sample_paths(controller, state, assign_all=True)
        extra = sorted(assignable - trained_rel_paths)
        order += extra
        for rel_path in drc_pipeline.cached_first(controller, state, extra, af_state):
            _assign_one_sample_approximate(controller, state, rel_path, predict_fn,
                                           dr_space, progress, af_state)

    _in_order(state, order)
    _prune_marker_sample(state, final=True)
    state.cluster_marker_keys = {}

    all_labels = (np.concatenate(list(state.cluster_labels.values()))
                  if state.cluster_labels else np.asarray(fit_labels, dtype=np.int32))
    final_ids, final_counts = np.unique(all_labels, return_counts=True)
    log.info("assign: FULL assigned label distribution: %s",
             dict(zip(final_ids.tolist(), final_counts.tolist())))
    assign_cluster_colors(state, all_labels)
    log.info("assigned labels to %d samples", len(state.cluster_labels))


def build_centroid_tree_data(fit_data: np.ndarray, fit_labels: np.ndarray,
                             all_labels: np.ndarray) -> dict:
    """
    Cluster-centroid tree data for Leiden/HDBSCAN -- same shape as
    FlowSOM's state.trained_reducers['FlowSOM'] entry ('node_weights',
    'node_to_meta', 'node_counts'), so build_flowsom_tree() renders an MST
    for these algorithms too. Leiden/HDBSCAN have no SOM-style sub-node
    structure -- each 'node' here IS a final cluster, so node_to_meta is
    the identity mapping (each node's own label) rather than a
    many-nodes-to-few-metaclusters rollup.

    fit_data / fit_labels: the training pool and its fitted labels --
        node_weights is each cluster's centroid over this pool, the same
        "training data only" scope FlowSOM's SOM codebook is trained on.
    all_labels: every assigned label across every sample this run
        touched (state.cluster_labels, concatenated) -- node_counts uses
        this full set, matching FlowSOM's node_counts semantics (total
        event count per node across ALL assigned samples, not just the
        training pool).
    """
    unique_labels = np.unique(fit_labels)
    node_weights = np.array(
        [fit_data[fit_labels == lbl].mean(axis=0) for lbl in unique_labels])
    node_to_meta = unique_labels.astype(np.int32)

    all_ids, all_counts = np.unique(all_labels, return_counts=True)
    count_map = dict(zip(all_ids.tolist(), all_counts.tolist()))
    node_counts = np.array([count_map.get(int(lbl), 0) for lbl in unique_labels],
                           dtype=np.int64)

    return {'node_weights': node_weights, 'node_to_meta': node_to_meta,
            'node_counts': node_counts}


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

    Trains on every gated event from every training sample by
    default (params['_event_cap'] is None unless "Downsample training
    data" is checked). Unlike Leiden/HDBSCAN, a SOM never outputs
    per-event labels directly -- every event is mapped onto the trained
    grid (assign_to_nodes, in flowsom_assign_all); a training sample that
    was not downsampled is mapped from its rows of the training pool, so
    it is not loaded and transformed a second time.

    af_state: optional AF snapshot — see drc_pipeline.apply_unmixing_af_aware()
        docstring. Must be passed when called from a background worker thread.
    """
    log_stage(log, "FLOWSOM")
    event_cap = params.get('_event_cap')
    data, boundaries = _pool_training_data_with_boundaries(
        controller, state, event_cap, af_state=af_state)
    if data is None:
        raise RuntimeError(
            "FlowSOM: no training data. Check the gate and training sample "
            "selection on the Configuration tab."
        )
    log_array(log, "flowsom_input", data,
              [c for c in state.selected_channels if c not in drc_pipeline.META_CHANNELS])

    xdim, ydim = params['xdim'], params['ydim']
    n_meta = params['n_metaclusters']  # 0 = auto
    n_iter = params['n_iter']

    _progress(progress, f"Training SOM ({xdim}×{ydim} grid, rlen={n_iter}, "
                        f"{len(data):,} events) …")
    start = time.perf_counter()
    node_weights = flowsom_consensus.train_som(data, xdim, ydim, n_iter, seed=42)
    _add_time(state, 'som', time.perf_counter() - start, len(data))

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
        progress=progress, af_state=af_state, pool=(data, boundaries))
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
                       af_state=None, pool=None) -> np.ndarray:
    """Map each sample's events to SOM nodes, then to metaclusters.

    Returns node_counts: np.ndarray shape (n_nodes,) — total event count
    assigned to each SOM node across all assigned samples, used by the
    FlowSOM MST tree view to size node bubbles.

    assign_all: see _assignable_sample_paths -- default (False)
        restricts to state.training_sample_ids; True additionally labels
        every other non-control sample.
    af_state: optional AF snapshot — see drc_pipeline.apply_unmixing_af_aware()
        docstring. Must be passed when called from a background worker thread.
    pool: optional (data, boundaries) from _pool_training_data_with_boundaries;
        a sample in it that was not downsampled is mapped from its pool rows
        (the same transformed float32 values load_sample_features returns).
    """
    assignable = _assignable_sample_paths(controller, state, assign_all)
    pool_data, pool_bounds = pool if pool is not None else (None, [])
    from_pool = {rel: (start, end) for rel, start, end, down in pool_bounds if not down}
    raw_subdir = controller.experiment.settings['raw']['raw_samples_subdirectory']
    all_samples = list(controller.experiment.samples.get('all_samples', {}).keys())

    state.cluster_labels = {}
    state.cluster_marker_values = {}
    state.cluster_marker_keys = {}
    state.cluster_rows = {}
    node_counts = np.zeros(len(node_weights), dtype=np.int64)
    order = []
    for sample_key in all_samples:
        try:
            rel_path = str(Path(sample_key).relative_to(raw_subdir))
        except ValueError:
            rel_path = sample_key
        if rel_path in assignable:
            order.append(rel_path)
    for rel_path in drc_pipeline.cached_first(controller, state, order, af_state):
        if rel_path in from_pool:
            start, end = from_pool[rel_path]
            sample_data = pool_data[start:end]
        else:
            sample_data = drc_pipeline.load_sample_features(controller, state, rel_path,
                                                            af_state=af_state)
        if sample_data is None:
            continue
        try:
            start = time.perf_counter()
            node_ids = flowsom_consensus.assign_to_nodes(node_weights, sample_data)
            _add_time(state, 'map', time.perf_counter() - start, len(sample_data))
            node_counts += np.bincount(node_ids, minlength=len(node_weights))
            labels = node_to_meta[node_ids].astype(np.int32)
            state.cluster_labels[rel_path] = labels
            state.cluster_rows[rel_path] = drc_pipeline.load_sample_rows(
                controller, state, rel_path, af_state=af_state)
            _snapshot_marker_values(controller, state, rel_path, labels, af_state=af_state)
            _progress(progress, f"  FlowSOM assigned {rel_path}: {len(labels):,} events "
                                f"({len(np.unique(labels))} metaclusters)")
        except Exception as e:
            _progress(progress, f"  FlowSOM assignment failed for {rel_path}: {e}")

    _in_order(state, order)
    _prune_marker_sample(state, final=True)
    state.cluster_marker_keys = {}

    if state.cluster_labels:
        all_labels = np.concatenate(list(state.cluster_labels.values()))
        assign_cluster_colors(state, all_labels)

    return node_counts


def build_flowsom_tree(node_weights: np.ndarray, node_to_meta: np.ndarray,
                       node_counts: np.ndarray) -> dict:
    """
    Classic FlowSOM tree: minimum spanning tree over SOM codebook vectors
    (Euclidean distance), laid out with the Kamada-Kawai spring layout
    (Kamada and Kawai 1989, via networkx) on the unweighted tree — the same
    approach R FlowSOM's BuildMST/PlotStars uses.

    Returns {'positions': (n_nodes, 2) ndarray, 'edges': list[(i, j)]}.
    Positions are scaled to [-1, 1]. Pure function — no state access, safe
    to call from the Workspace tab's render path directly.
    """
    import networkx as nx
    from scipy.spatial.distance import pdist, squareform
    from scipy.sparse.csgraph import minimum_spanning_tree

    dist = squareform(pdist(node_weights, metric='euclidean'))
    mst_sparse = minimum_spanning_tree(dist)
    mst_coo = mst_sparse.tocoo()
    edges = list(zip(mst_coo.row.tolist(), mst_coo.col.tolist()))

    n_nodes = node_weights.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(edges)
    layout = nx.kamada_kawai_layout(graph)
    positions = np.array([layout[i] for i in range(n_nodes)], dtype=float).reshape(n_nodes, 2)
    return {'positions': positions, 'edges': edges}


# ---------------------------------------------------------------------------
# Leiden
# ---------------------------------------------------------------------------

def knn_adjacency(neigh: np.ndarray):
    """
    Symmetric, unweighted kNN graph as a scipy CSR matrix.

    neigh: (n, k) neighbour ids from a kNN query of the n points against
        an index of those same n points. Each point's own id, if present,
        is dropped; an edge found from both ends is stored once with
        weight 1.
    """
    from scipy import sparse

    n, k = neigh.shape
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = neigh.reshape(-1).astype(np.int64)
    keep = rows != cols
    adj = sparse.csr_matrix(
        (np.ones(int(keep.sum())), (rows[keep], cols[keep])), shape=(n, n))
    adj = (adj + adj.T).tocsr()
    adj.data[:] = 1.0
    adj.sort_indices()
    return adj


def leiden_labels(adj, resolution: float, seed: int = 42,
                  iterations: int = 2) -> np.ndarray:
    """
    Leiden community detection (Traag, Waltman and van Eck 2019) on a
    symmetric CSR adjacency matrix, maximising modularity with resolution
    `resolution`, via graspologic-native.

    Returns int32 labels 0..k-1 ordered by descending community size
    (ties: community of the lowest node id first).
    """
    from graspologic_native import leiden_csr

    n = adj.shape[0]
    _quality, membership = leiden_csr(
        indptr=adj.indptr.astype(np.int64),
        indices=adj.indices.astype(np.int32),
        data=adj.data.astype(np.float64),
        n_nodes=n,
        resolution=float(resolution),
        iterations=iterations,
        use_modularity=True,
        seed=seed,
    )
    raw = np.fromiter((membership[i] for i in range(n)), dtype=np.int64, count=n)
    _ids, first, inverse, counts = np.unique(
        raw, return_index=True, return_inverse=True, return_counts=True)
    order = np.lexsort((first, -counts))
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[inverse].astype(np.int32)


def get_training_embeddings(controller, state, algo: str | None, event_cap: int | None,
                            af_state=None, seed: int = 42):
    """Concatenated training-sample embeddings for *algo*, with per-sample
    row boundaries -- respects event_cap the same way the raw
    feature-space pool does. Falls back to the pooled raw
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

    Trains on every gated event from every training sample by
    default (params['_event_cap'] is None unless "Downsample training
    data" is checked) -- every training sample's OWN events then get
    their EXACT label straight out of the partition itself (see
    _assign_by_slicing_or_predict), with no k-NN-vote approximation for
    them at all. k-NN-vote only runs for a sample that WAS downsampled,
    or a non-training sample if "Assign clusters to all
    samples" is checked.

    af_state: optional AF snapshot — see drc_pipeline.apply_unmixing_af_aware()
        docstring. Must be passed when called from a background worker thread.
    """
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
        raise RuntimeError(
            "Leiden: no training data. Check the gate and training sample "
            "selection on the Configuration tab."
        )
    log_array(log, "leiden_input", data)

    resolution = params['resolution']
    n_neighbors = params['n_neighbors']

    _progress(progress, f"Building hnswlib kNN index (k={n_neighbors}) for Leiden "
                        f"({len(data):,} events) …")
    start = time.perf_counter()
    index = hnswlib.Index(space='l2', dim=data.shape[1])
    index.init_index(max_elements=len(data), ef_construction=200, M=16)
    index.add_items(data)
    index.set_ef(50)

    _progress(progress, "Querying kNN …")
    neigh, _ = index.knn_query(data, k=n_neighbors)
    _add_time(state, 'knn', time.perf_counter() - start, len(data))

    _progress(progress, "Running Leiden …")
    start = time.perf_counter()
    train_labels = leiden_labels(knn_adjacency(neigh), resolution)
    _add_time(state, 'leiden', time.perf_counter() - start, len(data))

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
    all_labels_final = (np.concatenate(list(state.cluster_labels.values()))
                        if state.cluster_labels else train_labels)
    state.trained_reducers['Leiden'] = build_centroid_tree_data(
        data, train_labels, all_labels_final)
    _progress(progress, f"Leiden done: {n_cl} communities.")


# ---------------------------------------------------------------------------
# HDBSCAN
# ---------------------------------------------------------------------------

def run_hdbscan(controller, state, params: dict, progress=None, af_state=None) -> None:
    """
    Run HDBSCAN on a DR embedding only.

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
    not sklearn.cluster.HDBSCAN: sklearn's built-in has no
    out-of-sample prediction at all -- only fit/fit_predict/labels_ -- so
    assigning anything beyond the fit previously had to fall back to
    NearestCentroid, which assumes convex clusters and can never output
    noise, defeating the point of using HDBSCAN. The standalone package's
    approximate_predict() uses the trained condensed tree directly,
    respecting density-based, non-convex cluster shapes and legitimately
    labelling new points as noise (-1).

    Trains on every gated event from every training sample by
    default (params['_event_cap'] is None unless "Downsample training
    data" is checked) -- every training sample's OWN events then get
    their EXACT label straight out of clusterer.labels_ (see
    _assign_by_slicing_or_predict), with no approximate_predict call for
    them at all. approximate_predict only runs for a sample that WAS
    downsampled, or a non-training sample if "Assign clusters
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
        raise RuntimeError(
            "HDBSCAN requires a DR embedding -- it doesn't behave well "
            "directly on the full feature space. Train a DR algorithm "
            "(UMAP/PaCMAP) and select 'DR embedding' above."
        )

    data, boundaries = get_training_embeddings(controller, state, dr_algo, event_cap,
                                                af_state=af_state)
    space_label = f'{dr_algo} embedding'
    if data is None:
        raise RuntimeError(
            f"HDBSCAN: no training data in the {space_label}. Re-run the DR "
            "step, or check the gate and training sample selection."
        )
    log_array(log, "hdbscan_input", data)

    min_cluster_size = params['min_cluster_size']
    min_samples = params['min_samples'] or None       # 0 = package default (None)
    cluster_selection_epsilon = params['cluster_selection_epsilon']

    _progress(progress, f"Running HDBSCAN on {space_label} "
                        f"({len(data):,} events; min_cluster_size={min_cluster_size}, "
                        f"min_samples={min_samples}, "
                        f"cluster_selection_epsilon={cluster_selection_epsilon}) …")
    start = time.perf_counter()
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        core_dist_n_jobs=-1,   # NOT 'n_jobs' -- different name than sklearn's HDBSCAN
        prediction_data=True,  # required for approximate_predict() below
    ).fit(data)
    _add_time(state, 'fit', time.perf_counter() - start, len(data))
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
    all_labels_final = (np.concatenate(list(state.cluster_labels.values()))
                        if state.cluster_labels else train_labels)
    state.trained_reducers['HDBSCAN'] = build_centroid_tree_data(
        data, train_labels, all_labels_final)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class StagedResult:
    """
    The result fields of one clustering run, held apart from the plugin state
    until the run is committed (commit_clustering_result).

    A clustering function given a StagedResult instead of the state reads
    everything else (embeddings, selected channels, caches) through to the
    real state, but writes its labels, colours, cluster count and tree data
    only here.  A run that is cancelled, or fails, therefore leaves the
    plugin state exactly as it was, even if its worker thread is still
    finishing an uninterruptible step.
    """

    def __init__(self, state):
        self._state = state
        self.cluster_labels = dict(state.cluster_labels)
        self.cluster_marker_values = dict(state.cluster_marker_values)
        self.cluster_marker_keys = {}
        self.cluster_rows = dict(getattr(state, 'cluster_rows', {}) or {})
        self.cluster_colors = dict(state.cluster_colors)
        self.n_clusters = state.n_clusters
        self.active_clustering_algorithm = state.active_clustering_algorithm
        self.trained_reducers = {}
        # Measured stage times and event counts (drc_time_estimate)
        self.stage_seconds = {}
        self.stage_events = {}

    def __getattr__(self, name):
        return getattr(self._state, name)


def run_clustering_staged(controller, state, algo: str, params: dict,
                          progress=None, af_state=None) -> StagedResult:
    """Run the selected clustering algorithm (called from the worker thread)
    and return its results without touching *state*.

    progress may raise to abort the run; it is called at every stage and
    every sample.

    af_state: optional unmixing snapshot, captured on the main thread
        before the worker started — see
        drc_pipeline.apply_unmixing_af_aware() docstring.
    """
    staged = StagedResult(state)
    labelled = (sorted(_assignable_sample_paths(controller, state, True))
                if params.get('_assign_all_samples') else list(state.training_sample_ids))
    _progress(progress, "Loading samples …")
    seconds, events = drc_pipeline.timed_preload(controller, state, labelled,
                                                 af_state=af_state)
    if events:
        _add_time(staged, 'io', seconds, events)
    if algo == 'FlowSOM':
        run_flowsom(controller, staged, params, progress, af_state=af_state)
    elif algo == 'Leiden':
        run_leiden(controller, staged, params, progress, af_state=af_state)
    elif algo == 'HDBSCAN':
        run_hdbscan(controller, staged, params, progress, af_state=af_state)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algo}")
    return staged


def commit_clustering_result(state, staged: StagedResult, algo: str) -> None:
    """Publish a finished run's results to the plugin state (main thread)."""
    state.cluster_labels = staged.cluster_labels
    state.cluster_marker_values = staged.cluster_marker_values
    state.cluster_rows = staged.cluster_rows
    state.cluster_colors = staged.cluster_colors
    state.n_clusters = staged.n_clusters
    state.active_clustering_algorithm = staged.active_clustering_algorithm
    if algo in staged.trained_reducers:
        state.trained_reducers[algo] = staged.trained_reducers[algo]

"""
drc_time_estimate.py — Run-time estimate for large DR / clustering runs
=======================================================================
Companion module to ``dr_clustering_tab.py``.  Not a plugin itself (the
filename deliberately does **not** end in ``_tab.py`` so ``plugin_loaders``
will not try to load it as a separate tab).

A DR training or clustering run on more than ``event_threshold()`` events is
first timed on probes of ``TIMING_PROBE_SMALL_EVENTS`` and
``TIMING_PROBE_EVENTS`` events drawn from the training samples in proportion
to their size.  The probes run the chosen algorithm with the run's own
settings.  Each timed stage is modelled as

    seconds = fixed + per_event * events

with both terms solved from the two probe times, so a start-up cost that does
not grow with the event count is not multiplied up to the full run.  If the
two times do not resolve a positive per-event cost (timer noise on a fast
stage), the larger probe is scaled in proportion instead, which is an upper
bound.

What is timed
-------------
DR       'fit': fitting the reducer, and 'embed': embedding the probe with
         it.  A training run fits on the capped training pool but then embeds
         every gated event of the training samples, one sample at a time, so
         the stages are scaled to their own event counts (and the embed fixed
         cost to the sample count).
tSNE     uses the gradient method the full run will choose (openTSNE picks
         Barnes-Hut below 10,000 events and FFT above).
FlowSOM  SOM training and mapping events to nodes.  Metaclustering works on
         the SOM nodes only, so it does not grow with the event count and is
         left out.
Leiden   kNN index, kNN query and Leiden partition; the estimate is scaled by
         ``RUN_TIME_FACTORS`` because the probe underestimates it.
HDBSCAN  the HDBSCAN fit.

Reading and unmixing files, and assigning labels to samples, are not timed.

Algorithms that JIT-compile on first use (UMAP, PaCMAP) are probed with
repeats so the compile time is left out (fastest run used).

Pure numpy; no Qt.  The probe function for DR is supplied by the plugin
(it needs the plugin's reducer methods); clustering probes are built here.
Every probe function has the signature ``probe(data, run_events)``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import numpy as np

import drc_clustering
import drc_pipeline
import flowsom_consensus
from drc_logging import get_logger

log = get_logger(__name__)

TIMING_EVENT_THRESHOLD = 50_000    # runs above this are timed and confirmed
TIMING_PROBE_EVENTS = 2_000        # events in the larger timed probe
TIMING_PROBE_SMALL_EVENTS = 1_000  # events in the smaller timed probe

# FlowSOM is fast enough that timing and confirming would take longer than
# the run itself below this size.
ALGO_EVENT_THRESHOLDS = {('cl', 'FlowSOM'): 2_000_000}

# Multipliers for algorithms whose probe is known to underestimate the run,
# either because the algorithm costs more than its probe suggests or because
# the probe leaves out work such as loading and transforming the data.
RUN_TIME_FACTORS = {
    ('cl', 'Leiden'): 2.0,
    ('cl', 'HDBSCAN'): 2.0,
    ('cl', 'FlowSOM'): 5.0,
    ('dr', 'UMAP'): 2.0,
    ('dr', 'PaCMAP'): 5.0,
}

KIND_LABELS = {'dr': 'DR training', 'cl': 'Clustering'}
KIND_NOUNS = {'dr': 'DR training', 'cl': 'clustering'}   # for use mid-sentence


def event_threshold(kind: str, algo: str) -> int:
    """Event count above which a run of this kind and algorithm is timed."""
    return ALGO_EVENT_THRESHOLDS.get((kind, algo), TIMING_EVENT_THRESHOLD)


class StageFit(NamedTuple):
    """Cost model of one timed stage: seconds = fixed + per_event * events."""
    fixed: float
    per_event: float
    resolved: bool      # False: per-event cost not separable from the fixed cost


@dataclass
class RunTimeEstimate:
    """Event count of a run and, when it is large enough to time, the estimate."""
    kind: str                      # 'dr' or 'cl'
    algo: str
    n_samples: int = 0
    total_events: int = 0          # events the model is fitted on
    embed_events: int = 0          # events embedded afterwards (DR only)
    threshold: int = TIMING_EVENT_THRESHOLD
    probe_events: int = 0          # events in the larger probe
    probe_small_events: int = 0
    stage_fits: dict = field(default_factory=dict)      # stage -> StageFit
    factor: float = 1.0            # multiplier applied to the seconds
    seconds: float = 0.0
    error: str = ''                # set when the probe failed

    @property
    def needs_confirmation(self) -> bool:
        return max(self.total_events, self.embed_events) > self.threshold


class PoolEntry(NamedTuple):
    """One training sample's share of the training pool."""
    rel_path: str
    n_available: int               # events the sample holds
    n_used: int                    # events after the per-sample cap


def format_duration(seconds) -> str:
    """'45s', '12.5min' or '1.2hr'; 'unknown' for a missing or non-finite value."""
    if seconds is None or not math.isfinite(seconds):
        return 'unknown'
    if seconds < 60:
        return f'{seconds:.0f}s'
    if seconds < 3600:
        return f'{seconds / 60:.1f}min'
    return f'{seconds / 3600:.1f}hr'


def allocate_rows(counts, n: int) -> list[int]:
    """Split ``min(n, sum(counts))`` rows across samples in proportion to
    their counts (largest remainder), never more than a sample holds."""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total <= 0:
        return [0] * len(counts)
    n = min(int(n), total)
    exact = counts * n / total
    alloc = np.floor(exact).astype(np.int64)
    remainder = n - int(alloc.sum())
    order = np.argsort(-(exact - alloc), kind='stable')
    alloc[order[:remainder]] += 1
    return np.minimum(alloc, counts).astype(int).tolist()


def pool_spec(kind: str, state, params: dict) -> tuple[int | None, str | None]:
    """(per-sample event cap, DR algorithm whose embeddings form the pool or None)
    for the run, matching how the run itself builds its training pool."""
    if kind == 'dr':
        return state.n_training_events, None
    dr_algo = params.get('_dr_algo') if params.get('_space') == 'dr' else None
    return params.get('_event_cap'), dr_algo


def _pool_embeddings(state, embedding_algo: str | None):
    """The embeddings dict the pool is built from, or None for the feature space."""
    emb = state.embeddings.get(embedding_algo) if embedding_algo else None
    if emb and any(rel in emb for rel in state.training_sample_ids):
        return emb
    return None


def training_pool_entries(controller, state, event_cap: int | None,
                          embedding_algo: str | None = None,
                          af_state=None) -> list[PoolEntry]:
    """Per-sample event counts of the training pool.

    Gated data is read through drc_pipeline.load_unmixed_gated, which caches
    it on the state, so the run itself reuses what is loaded here."""
    def capped(n: int) -> int:
        return min(n, event_cap) if event_cap else n

    emb = _pool_embeddings(state, embedding_algo)
    if emb is not None:
        return [PoolEntry(rel, len(emb[rel]), capped(len(emb[rel])))
                for rel in state.training_sample_ids if rel in emb]

    entries = []
    for rel in state.training_sample_ids:
        try:
            gated = drc_pipeline.load_unmixed_gated(
                controller, state, drc_pipeline.sample_abs_path(controller, rel),
                af_state=af_state)
        except Exception:
            log.exception("could not count events for %s", rel)
            continue
        n = len(gated)
        if n == 0:
            continue
        entries.append(PoolEntry(rel, n, capped(n)))
    return entries


def draw_probe(controller, state, entries: list[PoolEntry],
               embedding_algo: str | None = None,
               probe_n: int = TIMING_PROBE_EVENTS, af_state=None,
               seed: int = 42) -> np.ndarray:
    """Probe events drawn from every sample in proportion to its share of the
    pool, in the same space (transformed features or embedding) as the run."""
    rng = np.random.default_rng(seed)
    emb = _pool_embeddings(state, embedding_algo)
    blocks = []
    for entry, k in zip(entries, allocate_rows([e.n_used for e in entries], probe_n)):
        if k == 0:
            continue
        rows = np.sort(rng.choice(entry.n_available, size=k, replace=False))
        if emb is not None:
            block = np.asarray(emb[entry.rel_path])[rows]
        else:
            gated = drc_pipeline.load_unmixed_gated(
                controller, state, drc_pipeline.sample_abs_path(controller, entry.rel_path),
                af_state=af_state)
            block = drc_pipeline.transform_selected_channels(controller, state, gated[rows])
        blocks.append(block)
    pooled = np.vstack(blocks)
    # Shuffled so that any leading slice is itself proportional across samples
    return np.ascontiguousarray(pooled[rng.permutation(len(pooled))], dtype=np.float32)


def time_probe(probe_fn: Callable[[np.ndarray, int], object], probe: np.ndarray,
               run_events: int, repeats: int = 1) -> dict[str, float]:
    """Seconds per stage taken by ``probe_fn(probe, run_events)``: the fastest
    of ``repeats`` runs, so that one-off start-up costs (JIT compilation) are
    left out when repeats > 1.

    probe_fn may return a dict of stage name -> seconds ('fit', 'embed');
    otherwise the whole call is timed as one 'fit' stage."""
    best: dict[str, float] = {}
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        stages = probe_fn(probe, run_events)
        if not isinstance(stages, dict):
            stages = {'fit': time.perf_counter() - start}
        for name, seconds in stages.items():
            best[name] = min(best.get(name, math.inf), seconds)
    return {name: max(seconds, 1e-9) for name, seconds in best.items()}


def fit_stage(n_small: int, t_small: float, n_large: int, t_large: float) -> StageFit:
    """Solve seconds = fixed + per_event * events from two probe timings."""
    per_event = (t_large - t_small) / (n_large - n_small)
    if per_event > 0:
        return StageFit(t_large - per_event * n_large, per_event, True)
    return StageFit(0.0, t_large / n_large, False)


def estimate_run(controller, state, kind: str, algo: str, params: dict,
                 probe_fn: Callable[[np.ndarray, int], object],
                 af_state=None, probe_repeats: int = 1) -> RunTimeEstimate:
    """Count the events of a run and, when above the threshold, time two probes
    and extrapolate to the whole run.

    Called from a worker thread.  Counting errors propagate; a failed probe is
    reported in ``error`` so the caller can still offer to start the run."""
    event_cap, embedding_algo = pool_spec(kind, state, params)
    entries = training_pool_entries(controller, state, event_cap, embedding_algo,
                                    af_state=af_state)
    total = sum(e.n_used for e in entries)
    # A DR training run embeds every gated event of the training samples
    embed = sum(e.n_available for e in entries) if kind == 'dr' else 0
    estimate = RunTimeEstimate(kind=kind, algo=algo, n_samples=len(entries),
                               total_events=total, embed_events=embed,
                               threshold=event_threshold(kind, algo),
                               factor=RUN_TIME_FACTORS.get((kind, algo), 1.0))
    if not estimate.needs_confirmation:
        return estimate

    try:
        probe = draw_probe(controller, state, entries, embedding_algo, af_state=af_state)
        small = probe[:TIMING_PROBE_SMALL_EVENTS]
        t_small = time_probe(probe_fn, small, total, probe_repeats)
        t_large = time_probe(probe_fn, probe, total, probe_repeats)
    except Exception as exc:
        log.exception("%s probe failed", KIND_LABELS.get(kind, kind))
        estimate.error = str(exc)
        return estimate

    estimate.probe_events = int(len(probe))
    estimate.probe_small_events = int(len(small))
    estimate.stage_fits = {
        name: fit_stage(estimate.probe_small_events, t_small[name],
                        estimate.probe_events, t_large[name])
        for name in t_large}
    stage_events = {'fit': (1, total), 'embed': (len(entries), embed)}
    estimate.seconds = estimate.factor * sum(
        fit.fixed * stage_events[name][0] + fit.per_event * stage_events[name][1]
        for name, fit in estimate.stage_fits.items())
    log.info("%s %s: %d events (%d embedded), probes %d/%d events %s x%.1f -> about %s",
             KIND_LABELS.get(kind, kind), algo, total, embed,
             estimate.probe_small_events, estimate.probe_events,
             {k: tuple(round(v, 4) if isinstance(v, float) else v for v in f)
              for k, f in estimate.stage_fits.items()},
             estimate.factor, format_duration(estimate.seconds))
    return estimate


def estimate_message(estimate: RunTimeEstimate) -> str:
    """Text for the confirm/cancel prompt."""
    label = KIND_NOUNS.get(estimate.kind, estimate.kind)
    costs = []
    for name, fit in estimate.stage_fits.items():
        costs.append(f'{name}: {fit.fixed:.1f}s fixed + '
                     f'{fit.per_event * 1000:.2f}s per 1,000 events')
    if estimate.kind == 'dr':
        size = (f'fitting on {estimate.total_events:,} events and embedding '
                f'{estimate.embed_events:,} events')
    else:
        size = f'{estimate.total_events:,} events'
    lines = [
        f'Estimated {label} time: about {format_duration(estimate.seconds)} for {size} '
        f'from {estimate.n_samples} sample(s).',
        '',
        f'Based on probes of {estimate.probe_small_events:,} and {estimate.probe_events:,} '
        f'events drawn from the training samples, run with the same {estimate.algo} settings '
        f'({"; ".join(costs)}), extrapolated to the full event counts.',
    ]
    unresolved = [name for name, fit in estimate.stage_fits.items() if not fit.resolved]
    if unresolved:
        lines.append(f'The per-event cost of the {" and ".join(unresolved)} stage could not be '
                     'separated from its fixed cost, so that part is scaled in proportion and '
                     'is likely an overestimate.')
    if estimate.factor != 1.0:
        lines.append(f'{estimate.algo} estimates are multiplied by {estimate.factor:g}, as the '
                     'probes do not capture all of its run time (loading and transforming the '
                     'data, for example).')
    lines += [
        'The cost per event usually rises with the size of the run (FlowSOM is the exception '
        'and scales close to linearly), so the real time may be longer.',
        '',
        f'Start {label}?',
    ]
    return '\n'.join(lines)


def clustering_probe_fn(algo: str, params: dict) -> Callable[[np.ndarray, int], object]:
    """A function timing the event-count-dependent part of a clustering fit on
    a data array, using the run's own settings and no plugin state."""
    if algo == 'FlowSOM':
        def probe(data, run_events):
            nodes = flowsom_consensus.train_som(
                data, params['xdim'], params['ydim'], params['n_iter'], seed=42)
            flowsom_consensus.assign_to_nodes(nodes, data)
    elif algo == 'Leiden':
        def probe(data, run_events):
            import hnswlib
            index = hnswlib.Index(space='l2', dim=data.shape[1])
            index.init_index(max_elements=len(data), ef_construction=200, M=16)
            index.add_items(data)
            index.set_ef(50)
            neigh, _ = index.knn_query(data, k=params['n_neighbors'])
            drc_clustering.leiden_labels(drc_clustering.knn_adjacency(neigh),
                                         params['resolution'])
    elif algo == 'HDBSCAN':
        def probe(data, run_events):
            import hdbscan as hdbscan_lib
            limit = len(data) - 1   # sizes above the probe size would fail
            hdbscan_lib.HDBSCAN(
                min_cluster_size=min(params['min_cluster_size'], limit),
                min_samples=min(params['min_samples'], limit) or None,
                cluster_selection_epsilon=params['cluster_selection_epsilon'],
                core_dist_n_jobs=-1,
                prediction_data=True,
            ).fit(data)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algo}")
    return probe

"""
drc_time_estimate.py — Run-time estimate for DR / clustering runs
=================================================================
Companion module to ``dr_clustering_tab.py``.  Not a plugin itself (the
filename deliberately does **not** end in ``_tab.py`` so ``plugin_loaders``
will not try to load it as a separate tab).

Model
-----
A run is split into stages, each with a reference cost curve measured once
on a reference machine (``REFERENCE``, seconds = a + b * n**p for n events;
``EMBED_REFERENCE`` for embedding new events with a fitted DR model), and a
per-machine factor that places this machine on the curve:

    seconds(stage) = factor(algorithm, stage, threads) * reference(stage, n)

Stages: 'io' (reading, unmixing and gating files not yet loaded this
session, per file event), 'jit' (UMAP's numba compilation, once per
session), the fit stages of each algorithm, and for DR 'index' (UMAP's
hnswlib search index, built once per model before its first embed) and
'embed'.

The factors are kept per machine (QSettings, written by the plugin) and
learned from every completed run (``observe_run``).  When a stage has no
factor yet, a conservative ``PRIOR_FACTOR`` is used first; only if that
predicts at least ``CONFIRM_SECONDS`` is the stage timed on a small probe of
the run's own events (``PROBE_EVENTS``) to calibrate it.  A run predicted to
take less than ``CONFIRM_SECONDS`` starts without asking.

Event counts come from capped reads (drc_pipeline.estimate_gated_count), so
estimating a run does not load every sample first.

Reference curves: Linux VM on an Apple M-series MacBook Air, 4 cores, all
cores used, Tombi Lung (7 channels), numba 0.68, umap-learn 0.5.12, pacmap
0.8.0, openTSNE 1.0.4 (claude/honeycluster_perf/bench_vm.csv; UMAP index
and embed with hnswlib search: umap_embed_vm.csv).  The curve shapes agreed
to within +/-0.05 in exponent between that machine and an x86 cloud machine,
and held on the same Mac natively (bench_mac.csv); the factor absorbs the
machine's speed.

Pure numpy; no Qt.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import numpy as np

import drc_clustering
import drc_jit
import drc_pipeline
import drc_umap_index
import flowsom_consensus
from drc_logging import get_logger

log = get_logger(__name__)

CONFIRM_SECONDS = 30.0      # runs predicted to take longer ask first
PRIOR_FACTOR = 4.0          # an uncalibrated machine is assumed this slow
CALIBRATION_VERSION = 2     # bump when REFERENCE changes; older factors are ignored
RUN_WEIGHT = 0.5            # weight of a new run's factor against the stored one
RUN_WEIGHT_AFTER_PROBE = 0.75

KIND_LABELS = {'dr': 'DR training', 'cl': 'Clustering'}
KIND_NOUNS = {'dr': 'DR training', 'cl': 'clustering'}   # for use mid-sentence

STAGE_LABELS = {
    'io': 'reading and unmixing files', 'jit': 'compiling UMAP (once per session)',
    'fit': 'fit', 'fit_bh': 'fit (Barnes-Hut)', 'fit_fft': 'fit (FFT)',
    'embed': 'embedding events not in the fit', 'som': 'SOM training',
    'map': 'mapping events to SOM nodes', 'knn': 'nearest-neighbour graph',
    'leiden': 'Leiden partition', 'index': 'search index for embedding',
}


class Curve(NamedTuple):
    """seconds = a + b * n**p on the reference machine."""
    a: float
    b: float
    p: float

    def seconds(self, n: float) -> float:
        return self.a + self.b * float(n) ** self.p if n > 0 else 0.0


class EmbedCost(NamedTuple):
    """seconds = per_model + per_call * calls
                 + per_event * events * (pool / 1e5)**pool_exp * scale

    per_model: one-off preparation on the first transform.  scale: the run's
    settings relative to the reference settings (embed_scale)."""
    per_model: float
    per_call: float
    per_event: float
    pool_exp: float

    def seconds(self, events: int, calls: int, pool: int, scale: float = 1.0) -> float:
        if events <= 0:
            return 0.0
        return (self.per_model + self.per_call * calls
                + self.per_event * events * (max(pool, 1) / 1e5) ** self.pool_exp * scale)


REFERENCE = {
    ('FlowSOM', 'som'): Curve(0.0007, 2.586e-06, 0.997),
    ('FlowSOM', 'map'): Curve(0.0, 2.522e-07, 1.001),
    ('Leiden', 'knn'): Curve(0.0122, 1.927e-06, 1.197),
    ('Leiden', 'leiden'): Curve(0.0026, 1.482e-06, 1.172),
    ('HDBSCAN', 'fit'): Curve(0.0, 4.997e-06, 1.109),
    ('UMAP', 'fit'): Curve(0.692, 2.401e-05, 1.208),
    ('UMAP', 'index'): Curve(0.0061, 1.53e-06, 1.205),
    ('PaCMAP', 'fit'): Curve(0.0, 1.239e-04, 1.055),
    ('tSNE', 'fit_bh'): Curve(0.0, 2.314e-04, 1.206),
    ('tSNE', 'fit_fft'): Curve(45.5, 1.05e-03, 1.0),
}
EMBED_REFERENCE = {
    'UMAP': EmbedCost(0.0, 0.065, 7.13e-05, 0.087),
    'PaCMAP': EmbedCost(0.0, 0.10, 1.4e-04, 0.10),
    'tSNE': EmbedCost(0.0, 0.05, 1.3e-04, 0.15),
}
IO_SECONDS_PER_EVENT = 5e-6
JIT_SECONDS = {'UMAP': 9.0}

# openTSNE's FFT fit is barely multi-threaded, and its Barnes-Hut fit is
# multi-threaded only where openTSNE was built with OpenMP (not the macOS
# wheels).  Barnes-Hut on one thread takes this many times the reference
# (4-thread) curve at the probe's size on both machines measured, and a
# machine's FFT factor matches its one-thread Barnes-Hut factor within
# 0.9-1.3x (Apple M-series native and Linux VM), so the FFT factor is probed
# that way.
TSNE_BH_ONE_THREAD = 2.14

# Settings the reference curves were measured with: UMAP 500 epochs (layout
# optimisation, 76% of the fit at 150,000 events, scales with epochs) and
# tSNE 1,000 iterations after 250 of early exaggeration.
REFERENCE_UMAP_EPOCHS = 500
UMAP_LAYOUT_SHARE = 0.76
REFERENCE_TSNE_ITERATIONS = 1_250
TSNE_EXAGGERATION_ITERATIONS = 250
# UMAP's embed reference: transforms searching an hnswlib index
# (drc_umap_index) of a model fitted with 200 epochs; the transform
# optimises for n_epochs // 3, which is 32% of its time at 200 epochs
# (claude/honeycluster_perf/umap_embed_vm.csv, fitted within +/-13%)
REFERENCE_UMAP_EMBED_EPOCHS = 200
UMAP_EMBED_LAYOUT_SHARE = 0.32


def work_scale(algo: str, stage: str, params: dict) -> float:
    """Cost of a fit with the run's settings relative to the reference settings."""
    if algo == 'UMAP' and stage == 'fit':
        epochs = params.get('n_epochs') or REFERENCE_UMAP_EPOCHS
        return (1 - UMAP_LAYOUT_SHARE) + UMAP_LAYOUT_SHARE * epochs / REFERENCE_UMAP_EPOCHS
    if algo == 'tSNE' and stage in ('fit_bh', 'fit_fft'):
        iterations = (params.get('n_iter') or 500) + TSNE_EXAGGERATION_ITERATIONS
        return iterations / REFERENCE_TSNE_ITERATIONS
    return 1.0


def embed_scale(algo: str, params: dict) -> float:
    """Per-event embed cost with the run's settings relative to the reference."""
    if algo == 'UMAP':
        epochs = params.get('n_epochs') or REFERENCE_UMAP_EMBED_EPOCHS
        return ((1 - UMAP_EMBED_LAYOUT_SHARE)
                + UMAP_EMBED_LAYOUT_SHARE * epochs / REFERENCE_UMAP_EMBED_EPOCHS)
    return 1.0


def reference_seconds(algo: str, stage: str, events: int, params: dict,
                      pool: int = 0, calls: int = 0) -> float:
    """Seconds the reference machine takes for one stage of a run."""
    if stage == 'io':
        return IO_SECONDS_PER_EVENT * events
    if stage == 'jit':
        return JIT_SECONDS.get(algo, 0.0)
    if stage == 'embed':
        return EMBED_REFERENCE[algo].seconds(events, calls, pool, embed_scale(algo, params))
    return REFERENCE[(algo, stage)].seconds(events) * work_scale(algo, stage, params)

# openTSNE's 'auto' gradient method switches to FFT at this many events,
# which the plugin keeps where openTSNE has no OpenMP
TSNE_FFT_MIN_EVENTS = 10_000
# With OpenMP, Barnes-Hut uses every core and FFT barely any: on the
# reference VM (4 threads) Barnes-Hut took 15-16.5 s at 10k, 33 s at 20k and
# 99 s at 50k events, FFT 52 s, 70 s and 102 s (tsne_bh_vm.csv, bench_vm.csv)
TSNE_FFT_MIN_EVENTS_OPENMP = 50_000
# UMAP uses exact nearest neighbours below this many events (other code)
UMAP_SMALL_DATA_EVENTS = 4_096

# Probe sizes (events) when a stage has no factor for this machine yet:
# large enough to run the same code path, in the multi-threaded regime,
# as the run itself.
PROBE_EVENTS = {
    'FlowSOM': 20_000, 'Leiden': 10_000, 'HDBSCAN': 10_000,
    'UMAP': 10_000, 'PaCMAP': 2_000, 'tSNE': 2_000,
}
PROBE_EMBED_EVENTS = 2_000
# A probe with fewer events than this share of its size is not run (the
# stage keeps the prior); UMAP below UMAP_SMALL_DATA_EVENTS would also time
# different code from the run's
MIN_PROBE_FRACTION = 0.6
# Probe factors outside this range (machine speed relative to the
# reference) are taken to be measurement artefacts and not stored
PROBE_FACTOR_RANGE = (0.05, 20.0)


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


# ---------------------------------------------------------------------------
# Calibration keys and factors
# ---------------------------------------------------------------------------

def threads_key(algo: str, params: dict) -> str:
    """Thread setting a stage's speed depends on: UMAP and tSNE follow
    n_jobs, PaCMAP's optimisation is single-threaded whatever the setting,
    FlowSOM, Leiden and HDBSCAN use every core."""
    if algo in ('UMAP', 'tSNE'):
        n_jobs = params.get('n_jobs', -1)
        if n_jobs in (-1, None):
            n_jobs = os.cpu_count() or 1
        return f't{int(n_jobs)}'
    if algo == 'PaCMAP':
        return 'any'
    return f't{os.cpu_count() or 1}'


def calibration_key(algo: str, stage: str, threads: str) -> str:
    """'io' is per machine, not per algorithm or thread setting."""
    if stage == 'io':
        return 'io'
    if stage == 'jit':
        return f'{algo}/fit/{threads}'          # compilation scales with the fit
    return f'{algo}/{stage}/{threads}'


def _stored_factor(calibration: dict, key: str):
    entry = calibration.get(key)
    if not isinstance(entry, dict) or entry.get('v') != CALIBRATION_VERSION:
        return None
    try:
        factor = float(entry['f'])
    except (KeyError, TypeError, ValueError):
        return None
    return entry if math.isfinite(factor) and factor > 0 else None


def updated_entry(old: dict | None, factor: float, source: str) -> dict:
    """Calibration entry after one more observation of *factor*."""
    if old is None or old.get('v') != CALIBRATION_VERSION:
        return {'f': factor, 'n': 1, 'src': source, 'v': CALIBRATION_VERSION}
    if source == 'probe':
        if old.get('src') == 'run':
            return old                       # a probe never overrides runs
        weight = RUN_WEIGHT
    else:
        weight = RUN_WEIGHT_AFTER_PROBE if old.get('src') == 'probe' else RUN_WEIGHT
    log_f = (1 - weight) * math.log(float(old['f'])) + weight * math.log(factor)
    return {'f': math.exp(log_f), 'n': int(old.get('n', 0)) + 1,
            'src': 'run' if source == 'run' or old.get('src') == 'run' else source,
            'v': CALIBRATION_VERSION}


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

class SampleCount(NamedTuple):
    rel_path: str
    events: int          # gated events under the current gates
    exact: bool
    file_events: int     # events in the FCS file (read when not cached)
    cached: bool         # already read, unmixed and gated this session


def count_samples(controller, state, rel_paths, af_state) -> list[SampleCount]:
    """Gated event counts without loading whole files (capped reads)."""
    out = []
    for rel in rel_paths:
        cached = drc_pipeline.is_gated_cached(controller, state, rel, af_state=af_state)
        n, exact = drc_pipeline.estimate_gated_count(controller, state, rel, af_state)
        out.append(SampleCount(rel, int(n), bool(exact),
                               drc_pipeline.sample_file_events(controller, rel), cached))
    return out


# ---------------------------------------------------------------------------
# Run plan
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    name: str                 # 'io', 'jit', 'fit', 'embed', 'som', ...
    events: int               # events the stage processes
    pool: int = 0             # training-pool size (embed cost depends on it)
    calls: int = 0            # transform calls (embed)
    factor: float = PRIOR_FACTOR
    source: str = 'prior'     # 'prior', 'probe' or 'N run(s)'
    seconds: float = 0.0


@dataclass
class RunTimeEstimate:
    """Event counts of a run and the predicted seconds of each stage."""
    kind: str                      # 'dr' or 'cl'
    algo: str
    threads: str = ''
    n_samples: int = 0
    total_events: int = 0          # events the model is fitted on
    embed_events: int = 0          # events embedded by transform (DR only)
    counts_exact: bool = True
    stages: list = field(default_factory=list)
    seconds: float = 0.0
    calibration_updates: dict = field(default_factory=dict)   # key -> entry, from probes
    error: str = ''                # set when a probe failed
    probe_seconds: float = 0.0

    @property
    def needs_confirmation(self) -> bool:
        return self.seconds >= CONFIRM_SECONDS


def fit_stage_name(algo: str, n_fit: int) -> list[str]:
    """Fit stages of *algo* for a run fitted on n_fit events."""
    if algo == 'FlowSOM':
        return ['som', 'map']
    if algo == 'Leiden':
        return ['knn', 'leiden']
    if algo == 'tSNE':
        return [f'fit_{tsne_gradient_method(n_fit)}']
    return ['fit']


def tsne_fft_min_events() -> int:
    """Fewest fit events for which tSNE uses the FFT gradient."""
    return TSNE_FFT_MIN_EVENTS_OPENMP if drc_jit.opentsne_has_openmp() else TSNE_FFT_MIN_EVENTS


def tsne_gradient_method(n_fit: int) -> str:
    """openTSNE negative_gradient_method for a fit on n_fit events."""
    return 'fft' if n_fit >= tsne_fft_min_events() else 'bh'


def plan_stages(kind: str, algo: str, params: dict, state, counts: list[SampleCount],
                embedding_counts: dict | None = None, jit_warm: bool = True) -> tuple:
    """(stages, n_fit, n_embed) for a run, from per-sample counts.

    embedding_counts: {rel_path: rows} of the DR embedding a DR-space
    clustering run is fitted on."""
    by_rel = {c.rel_path: c for c in counts}
    training = [rel for rel in state.training_sample_ids if rel in by_rel]
    io_events = sum(c.file_events for c in counts if not c.cached)
    stages = [Stage('io', io_events)] if io_events else []

    if kind == 'dr':
        cap = state.n_training_events
        n_fit = sum(min(by_rel[r].events, cap) for r in training)
        n_embed = sum(max(by_rel[r].events - cap, 0) for r in training)
        calls = sum(1 for r in training if by_rel[r].events > cap)
        if algo in JIT_SECONDS and not jit_warm:
            stages.append(Stage('jit', 0))
        stages += [Stage(name, n_fit) for name in fit_stage_name(algo, n_fit)]
        if n_embed:
            if (algo == 'UMAP' and n_fit >= UMAP_SMALL_DATA_EVENTS
                    and params.get('metric', 'euclidean') in drc_umap_index.HNSW_SPACES):
                stages.append(Stage('index', n_fit))
            stages.append(Stage('embed', n_embed, pool=n_fit, calls=calls))
        return stages, n_fit, n_embed

    cap = params.get('_event_cap')
    if embedding_counts is not None:
        sizes = [embedding_counts.get(r, 0) for r in state.training_sample_ids]
    else:
        sizes = [by_rel[r].events for r in training]
    n_fit = sum(min(n, cap) if cap else n for n in sizes)
    labelled = sum(c.events for c in counts)
    for name in fit_stage_name(algo, n_fit):
        stages.append(Stage(name, labelled if name == 'map' else n_fit))
    return stages, n_fit, 0


def price_stages(algo: str, threads: str, stages: list, calibration: dict,
                 params: dict) -> float:
    """Fill each stage's factor (stored, or the prior) and seconds; return the total."""
    total = 0.0
    for stage in stages:
        key = calibration_key(algo, stage.name, threads)
        entry = _stored_factor(calibration, key)
        if entry is None and stage.name == 'fit_fft':
            # FFT factor unknown: the Barnes-Hut factor of this machine, until
            # a probe or an FFT run measures it
            entry = _stored_factor(calibration, calibration_key(algo, 'fit_bh', threads))
        if entry is not None:
            stage.factor = float(entry['f'])
            stage.source = ('probe' if entry.get('src') == 'probe'
                            else f"{entry.get('n', 1)} run(s)")
        stage.seconds = stage.factor * reference_seconds(
            algo, stage.name, stage.events, params, stage.pool, stage.calls)
        total += stage.seconds
    return total


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def draw_probe(previews: dict, rel_paths: list[str], counts: dict, n: int,
               seed: int = 42) -> np.ndarray | None:
    """Up to n events drawn from the samples' previews in proportion to their
    gated counts, shuffled, float32."""
    rng = np.random.default_rng(seed)
    usable = [r for r in rel_paths if previews.get(r) is not None and len(previews[r])]
    if not usable:
        return None
    blocks = []
    for rel, k in zip(usable, allocate_rows([counts.get(r, 0) for r in usable], n)):
        data = previews[rel]
        k = min(k, len(data))
        if k:
            blocks.append(data[rng.choice(len(data), size=k, replace=False)])
    if not blocks:
        return None
    pooled = np.vstack(blocks)
    return np.ascontiguousarray(pooled[rng.permutation(len(pooled))], dtype=np.float32)


def clustering_probe_fn(algo: str, params: dict) -> Callable[[np.ndarray], dict]:
    """Function timing *algo*'s fit stages on a data array with the run's own
    settings; returns {stage: seconds}."""
    def timed(stages):
        out = {}
        for name, fn in stages:
            start = time.perf_counter()
            fn()
            out[name] = time.perf_counter() - start
        return out

    if algo == 'FlowSOM':
        def probe(data):
            nodes = {}
            return timed([
                ('som', lambda: nodes.setdefault('w', flowsom_consensus.train_som(
                    data, params['xdim'], params['ydim'], params['n_iter'], seed=42))),
                ('map', lambda: flowsom_consensus.assign_to_nodes(nodes['w'], data)),
            ])
    elif algo == 'Leiden':
        def probe(data):
            import hnswlib
            graph = {}

            def knn():
                index = hnswlib.Index(space='l2', dim=data.shape[1])
                index.init_index(max_elements=len(data), ef_construction=200, M=16)
                index.add_items(data)
                index.set_ef(50)
                graph['neigh'], _ = index.knn_query(data, k=params['n_neighbors'])
            return timed([
                ('knn', knn),
                ('leiden', lambda: drc_clustering.leiden_labels(
                    drc_clustering.knn_adjacency(graph['neigh']), params['resolution'])),
            ])
    elif algo == 'HDBSCAN':
        def probe(data):
            import hdbscan as hdbscan_lib
            limit = len(data) - 1
            return timed([('fit', lambda: hdbscan_lib.HDBSCAN(
                min_cluster_size=min(params['min_cluster_size'], limit),
                min_samples=min(params['min_samples'], limit) or None,
                cluster_selection_epsilon=params['cluster_selection_epsilon'],
                core_dist_n_jobs=-1, prediction_data=True).fit(data))])
    else:
        raise ValueError(f"Unknown clustering algorithm: {algo}")
    return probe


def _probe_factors(algo: str, threads: str, probe_fn, data: np.ndarray,
                   calibration: dict, params: dict, tsne_fft: bool = False) -> dict:
    """Run the probe and return calibration entries for its stages.

    probe_fn(data) returns {stage: seconds}; for DR it is
    (fit_data, embed_data, single_thread=False) -> {'fit': s, 'embed': s}.
    tsne_fft: calibrate tSNE's FFT fit, from a one-thread Barnes-Hut probe
    (see TSNE_BH_ONE_THREAD).  The probe's own embedding uses Barnes-Hut on
    a small pool (4-5x the reference per event), unlike an FFT model's, so
    the embed stage takes the FFT fit factor (within 0.9-1.3x of measured
    embeds from 10,000-event pools up)."""
    n = len(data)
    if algo in EMBED_REFERENCE:
        n_embed = min(PROBE_EMBED_EVENTS, max(n // 5, 1))
        if tsne_fft:
            timings = probe_fn(data[n_embed:], data[:n_embed], single_thread=True)
        else:
            timings = probe_fn(data[n_embed:], data[:n_embed])
        n_fit = n - n_embed
    else:
        timings = probe_fn(data)
        n_fit = n
    updates = {}
    for name, seconds in timings.items():
        if name == 'fit' and algo == 'tSNE':
            # The probe itself is below the FFT threshold (Barnes-Hut)
            name = 'fit_fft' if tsne_fft else 'fit_bh'
        if name == 'embed':
            if tsne_fft:
                continue
            reference = reference_seconds(algo, name, n_embed, params, pool=n_fit, calls=1)
        elif name == 'fit_fft':
            reference = reference_seconds(algo, 'fit_bh', n_fit, params) * TSNE_BH_ONE_THREAD
        else:
            reference = reference_seconds(algo, name, n_fit, params)
        if reference <= 0 or seconds <= 0:
            continue
        factor = seconds / reference
        if not PROBE_FACTOR_RANGE[0] <= factor <= PROBE_FACTOR_RANGE[1]:
            log.warning("%s %s probe: factor %.2f outside %s, not used", algo, name, factor,
                        PROBE_FACTOR_RANGE)
            continue
        key = calibration_key(algo, name, threads)
        updates[key] = updated_entry(calibration.get(key), factor, 'probe')
        if name == 'fit_fft':
            key = calibration_key(algo, 'embed', threads)
            updates[key] = updated_entry(calibration.get(key), factor, 'probe')
    return updates


def needs_probe(algo: str, threads: str, stages: list, calibration: dict) -> bool:
    """True when a fit stage of the run has no factor for this machine yet."""
    for stage in stages:
        if stage.name in ('io', 'jit'):
            continue
        if _stored_factor(calibration, calibration_key(algo, stage.name, threads)) is None:
            return True
    return False


# ---------------------------------------------------------------------------
# Estimate
# ---------------------------------------------------------------------------

def estimate_run(controller, state, kind: str, algo: str, params: dict,
                 calibration: dict, af_state=None, dr_probe_fn=None,
                 jit_warm: bool = True) -> RunTimeEstimate:
    """Count a run's events and predict its run time (called from a worker).

    calibration: {key: entry} snapshot of this machine's factors.  Stages
    without a factor are priced with PRIOR_FACTOR; if that predicts
    CONFIRM_SECONDS or more, they are timed on a probe and the new factors
    are returned in ``calibration_updates`` for the caller to store.

    dr_probe_fn(fit_data, embed_data) -> {'fit': s, 'embed': s} times a DR
    fit and embedding (supplied by the plugin, which owns the reducers).
    jit_warm: UMAP's numba code is already compiled in this session."""
    threads = threads_key(algo, params)
    estimate = RunTimeEstimate(kind=kind, algo=algo, threads=threads)

    dr_space = kind == 'cl' and params.get('_space') == 'dr' and params.get('_dr_algo')
    embedding_counts = None
    if dr_space:
        emb = state.embeddings.get(params['_dr_algo'], {}) or {}
        embedding_counts = {rel: len(emb[rel]) for rel in state.training_sample_ids
                            if rel in emb}
    labelled = list(state.training_sample_ids)
    if kind == 'cl' and params.get('_assign_all_samples'):
        labelled = sorted(drc_clustering._assignable_sample_paths(controller, state, True))
    counts = count_samples(controller, state, labelled, af_state)
    by_rel = {c.rel_path: c for c in counts}
    estimate.counts_exact = all(c.exact for c in counts)

    stages, n_fit, n_embed = plan_stages(kind, algo, params, state, counts,
                                         embedding_counts, jit_warm)
    estimate.n_samples = len([r for r in state.training_sample_ids if r in by_rel])
    estimate.total_events = n_fit
    estimate.embed_events = n_embed
    estimate.seconds = price_stages(algo, threads, stages, calibration, params)
    estimate.stages = stages

    if estimate.seconds < CONFIRM_SECONDS or not needs_probe(algo, threads, stages, calibration):
        _log_estimate(estimate)
        return estimate

    # Calibrate on a probe of the run's own events
    start = time.perf_counter()
    try:
        n_probe = PROBE_EVENTS[algo] + (PROBE_EMBED_EVENTS if kind == 'dr' else 0)
        if dr_space:
            emb = state.embeddings[params['_dr_algo']]
            previews = {rel: np.asarray(emb[rel], dtype=np.float32)
                        for rel in state.training_sample_ids if rel in emb}
            sizes = embedding_counts
        else:
            previews = {rel: drc_pipeline.preview_features(controller, state, rel, af_state)
                        for rel in state.training_sample_ids if rel in by_rel}
            if sum(len(v) for v in previews.values()) < n_probe:
                # Rare gate: the capped reads hold too few gated events, so
                # load whole samples (cached for the run) until there are enough
                previews, total = {}, 0
                for rel in sorted(by_rel, key=lambda r: -by_rel[r].events):
                    if rel not in state.training_sample_ids:
                        continue
                    feats = drc_pipeline.load_sample_features(controller, state, rel,
                                                              af_state=af_state)
                    if feats is not None and len(feats):
                        previews[rel] = feats
                        total += len(feats)
                    if total >= n_probe:
                        break
            sizes = {rel: by_rel[rel].events for rel in previews}
        probe = draw_probe(previews, list(previews), sizes, n_probe)
        if probe is None or len(probe) < MIN_PROBE_FRACTION * n_probe:
            raise RuntimeError(f'only {0 if probe is None else len(probe):,} events to time '
                               f'(at least {int(MIN_PROBE_FRACTION * n_probe):,} needed)')
        if kind == 'dr':
            if dr_probe_fn is None:
                raise RuntimeError('no DR probe function')
            probe_fn = dr_probe_fn
        else:
            probe_fn = clustering_probe_fn(algo, params)
        tsne_fft = any(s.name == 'fit_fft' for s in stages)
        updates = _probe_factors(algo, threads, probe_fn, probe, calibration, params,
                                 tsne_fft=tsne_fft)
    except Exception as exc:
        log.exception("%s probe failed", KIND_LABELS.get(kind, kind))
        estimate.error = str(exc)
        _log_estimate(estimate)
        return estimate
    estimate.probe_seconds = time.perf_counter() - start
    estimate.calibration_updates = updates
    merged = dict(calibration)
    merged.update(updates)
    if algo in JIT_SECONDS and kind == 'dr':
        # The probe compiled UMAP's numba code
        stages = [s for s in stages if s.name != 'jit']
        estimate.stages = stages
    estimate.seconds = price_stages(algo, threads, stages, merged, params)
    _log_estimate(estimate)
    return estimate


def _log_estimate(estimate: RunTimeEstimate) -> None:
    log.info("%s %s (%s): %d fit events, %d embedded, %s -> about %s",
             KIND_LABELS.get(estimate.kind, estimate.kind), estimate.algo, estimate.threads,
             estimate.total_events, estimate.embed_events,
             ', '.join(f'{s.name} {s.events} x{s.factor:.2f} [{s.source}] {s.seconds:.1f}s'
                       for s in estimate.stages),
             format_duration(estimate.seconds))


def estimate_message(estimate: RunTimeEstimate) -> str:
    """Text for the confirm/cancel prompt."""
    label = KIND_NOUNS.get(estimate.kind, estimate.kind)
    if estimate.kind == 'dr':
        size = (f'fitting on {estimate.total_events:,} events and embedding '
                f'{estimate.embed_events:,} more')
    else:
        size = f'{estimate.total_events:,} events'
    approx = '' if estimate.counts_exact else ' (event counts estimated from part of each file)'
    lines = [f'Estimated {label} time: about {format_duration(estimate.seconds)} for {size} '
             f'from {estimate.n_samples} sample(s){approx}.', '']
    for stage in estimate.stages:
        if stage.seconds < 0.5:
            continue
        what = STAGE_LABELS.get(stage.name, stage.name)
        events = f' ({stage.events:,} events)' if stage.events and stage.name != 'io' else ''
        lines.append(f'  • {what}{events}: {format_duration(stage.seconds)}'
                     f' [{_source_text(stage.source)}]')
    if any(s.source == 'prior' for s in estimate.stages if s.seconds >= 0.5):
        lines.append('')
        lines.append('Stages marked "not yet measured" assume a slow machine; estimates '
                     'improve after each run on this computer.')
    lines += ['', f'Start {label}?']
    return '\n'.join(lines)


def _source_text(source: str) -> str:
    if source == 'prior':
        return 'not yet measured on this computer'
    if source == 'probe':
        return 'from a short timing probe'
    return f'calibrated from {source} on this computer'


# ---------------------------------------------------------------------------
# Learning from completed runs
# ---------------------------------------------------------------------------

def observe_run(algo: str, params: dict, calibration: dict, seconds: dict, events: dict,
                pool: int = 0, calls: int = 0, jit_included: bool = False) -> dict:
    """Calibration entries updated from a completed run's measured stages.

    seconds / events: {stage: value} as measured ('io', 'fit', 'embed', or
    the clustering stages).  For tSNE 'fit' is recorded as fit_bh or fit_fft
    by the run's size.  Fit and embed times that included numba compilation
    are not used.
    Stages under 0.2 s are too noisy to use."""
    threads = threads_key(algo, params)
    updates = {}
    for name, secs in seconds.items():
        n = int(events.get(name, 0) or 0)
        if secs is None or secs < 0.2 or n <= 0:
            continue
        if name in ('fit', 'embed') and jit_included:
            continue
        if algo == 'tSNE' and name == 'fit':
            name = fit_stage_name('tSNE', n)[0]
        if name == 'embed' and algo not in EMBED_REFERENCE:
            continue
        if name not in ('io', 'embed') and (algo, name) not in REFERENCE:
            continue
        reference = reference_seconds(algo, name, n, params, pool=pool, calls=max(calls, 1))
        if reference <= 0:
            continue
        key = calibration_key(algo, name, threads)
        updates[key] = updated_entry(calibration.get(key), secs / reference, 'run')
    return updates

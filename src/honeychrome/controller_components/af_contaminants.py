"""
af_contaminants.py
------------------
Screening an unstained sample for events contaminated by single-stain
controls before its AF profile is extracted.

In plain terms
--------------
When plates are stained, a little of a neighbouring well (stained cells or
antibody) can end up in the unstained well. Those events carry a dye. If
they reach AF extraction they join the SOM nodes, and some AF spectra then
contain a little of the dye; every cell later unmixed with such a spectrum
loses that much dye signal. The refinement stage makes this worse: it looks
for cells the library fits poorly, and contaminated cells are exactly that.

The screen works in four steps:

1. Suspects. The plate layout (plate_layout.py) names the dyes in the wells
   around the unstained well and in the well acquired just before it.
2. A score per event and dye. Each event is unmixed against the panel and
   the sample's mean AF spectrum, and each dye's apparent amount is compared
   with its median among the event's 25 nearest neighbours in scatter (the
   event itself left out), so a bright granulocyte is compared with other
   granulocytes. The difference is expressed as a robust z score among
   events of similar AF brightness.
3. A threshold per dye, from clean samples. The experiment's other
   unstained cell samples (or, with fewer than two of them, the cell
   single-stain controls on the events negative for their own dye) show how
   often each dye scores high by chance. A dye's threshold is the score that
   clean events exceed only 20 times in 100,000. Dyes that resemble AF get
   high thresholds this way, so genuine AF is not removed.
4. Removal and guard. Events above threshold for a suspect dye from a
   single-stain neighbour are removed. During refinement, a new candidate
   AF spectrum is rejected when one suspect dye accounts for more of it than
   that dye ever does in libraries from the clean samples.

A dye that is unusually frequent in this sample but has no suspect
neighbour, and a neighbour holding the whole panel (a fully stained sample),
are reported as warnings only.

Statistical details
-------------------
* Score: OLS of each event on [panel spectra; sample mean AF spectrum]; a
  dye's amount minus its median among the event's k = 25 nearest neighbours
  in robustly standardised log scatter (all scatter channels except -T
  trigger channels), the event itself excluded. AF directions are never
  learnt from the screened sample itself: where contamination is common it
  is one of the main directions of variation and would be absorbed.
* Only the suspect dyes are scored on every event; the check for
  unexpected dyes scores all dyes on a 50,000-event subsample, as the
  clean samples are scored.
* Scale: per dye, median and 1.4826 x MAD within 20 quantile bins of AF
  brightness; a second pass leaves events above z 6 out of the scale.
* Rates: events per 100,000 above z 6. A source is left out of the null
  when any dye's rate exceeds 5x the second-highest rate of the other
  sources and 100 per 100,000 (a contaminated sample must not set the null).
* Thresholds: the pooled null z of each dye at rank ceil(2e-4 x events)
  from the top (each source keeps its top 200 scores per dye), floored at 3.
* Unexpected dye warning: rate above 5x the second-highest null rate and
  above 300 per 100,000.
* Guard limit: the largest share a dye takes in any spectrum of base
  libraries (SOM stage) extracted from the null sources, floored at 0.02.
  Share = |c_d s_d| / |a| from NNLS of spectrum a on the rest of the library
  plus dye spectrum s_d.

Each clean sample is summarised once from a seeded 50,000-event subsample
(rates, top scores, base-library shares) and cached in the experiment's
AutoSpectral/contaminant_null folder, keyed by the file, the spectra and the
settings, so later extractions only score the sample being extracted.

Deliberate v1 simplifications
-----------------------------
* Guard limits come from base (SOM-stage) libraries of the null sources,
  not refined ones, to keep extraction time bounded.
* Flagged events are removed, not down-weighted.
* A whole-panel neighbour and unexpected dyes are warnings; nothing is
  removed for them.
* With fewer than two usable null sources nothing is screened (warning).
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
from scipy.optimize import nnls
from sklearn.neighbors import NearestNeighbors

from honeychrome.controller_components import plate_layout

logger = logging.getLogger(__name__)

K_NEIGHBOURS = 25
N_BINS = 20
CHUNK = 4000
Z_RATE = 6.0                      # rates count events above this z
THRESH_RATE = 2e-4                # clean events above a dye's threshold
Z_FLOOR = 3.0
CONFIRM_RATIO = 5.0
CONFIRM_MIN_PER_100K = 100.0
UNEXPECTED_MIN_PER_100K = 300.0
OWN_NEGATIVE_Z = 3.0              # control events counted as negative for their own dye
GUARD_FLOOR = 0.02
HEAVY_REMOVAL_FRACTION = 0.03
NULL_EVENTS = 50_000              # events scored per clean sample
MIN_SOURCE_EVENTS = 1_000
N_TAIL = 200                      # highest z per dye kept per clean sample
NULL_VERSION = 2                  # bump when the null summaries change meaning
NULL_CACHE_SUBDIR = 'contaminant_null'
NULL_WORKERS = 6                  # clean samples summarised in parallel (at most one per core)
SCREEN_KEY = 'af_contaminant_screen'
MIN_NULL_SOURCES = 2
SEED = 11
EXCLUDED_SUFFIX = ' excluded.npy'


def screen_on(process: dict | None) -> bool:
    """The experiment's contaminant screen setting (off when unset, so older experiments are unchanged)."""
    return bool((process or {}).get(SCREEN_KEY, False))


def set_screen(process: dict, on: bool) -> None:
    process[SCREEN_KEY] = bool(on)


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def scatter_features(sc: np.ndarray) -> np.ndarray:
    """Log scatter, robustly standardised per channel."""
    x = np.log10(np.clip(np.asarray(sc, dtype=np.float64), 1.0, None))
    med = np.median(x, axis=0)
    mad = 1.4826 * np.median(np.abs(x - med), axis=0)
    return (x - med) / np.where(mad > 1e-9, mad, 1.0)


def loo_median(values: np.ndarray, feats: np.ndarray, k: int = K_NEIGHBOURS, n_jobs: int = -1) -> np.ndarray:
    """Per-column median of ``values`` over each event's k scatter neighbours, the event itself excluded."""
    k = min(k, len(values) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=n_jobs).fit(feats)
    v32 = np.asarray(values, dtype=np.float32)
    out = np.empty(values.shape, dtype=np.float64)
    for s in range(0, len(values), CHUNK):
        e = min(s + CHUNK, len(values))
        idx = nn.kneighbors(feats[s:e], return_distance=False)
        not_self = idx != np.arange(s, e)[:, None]
        order = np.argsort(~not_self, axis=1, kind='stable')[:, :k]
        out[s:e] = np.median(v32[np.take_along_axis(idx, order, axis=1)], axis=1)
    return out


def _binned_z(a: np.ndarray, k: np.ndarray, use: np.ndarray) -> np.ndarray:
    edges = np.quantile(k[use], np.linspace(0, 1, N_BINS + 1))
    b = np.clip(np.searchsorted(edges, k, side='right') - 1, 0, N_BINS - 1)
    bu = b[use]
    au = a[use]
    z = np.zeros_like(a)
    for i in range(N_BINS):
        ref = au[bu == i]
        sel = b == i
        if len(ref) == 0 or not sel.any():
            continue
        med = np.median(ref, axis=0)
        sd = 1.4826 * np.median(np.abs(ref - med), axis=0)
        z[sel] = (a[sel] - med) / np.maximum(sd, 1e-9)
    return z


def dye_z_scores(fl: np.ndarray, sc: np.ndarray, fluor_spectra: np.ndarray, dyes=None,
                 n_jobs: int = -1) -> np.ndarray:
    """
    (n_events, n_dyes) robust z score of each dye's amount (only the rows
    ``dyes`` of fluor_spectra when given) in each event, against the median
    amount among its scatter neighbours.
    """
    fl = np.asarray(fl, dtype=np.float64)
    mean_af = fl.mean(axis=0)
    mean_af = mean_af / max(np.abs(mean_af).max(), 1e-12)
    coef = fl @ np.linalg.pinv(np.vstack([fluor_spectra, mean_af]))
    k = coef[:, -1]
    cols = list(range(len(fluor_spectra))) if dyes is None else list(dyes)
    values = coef[:, cols]
    a = values - loo_median(values, scatter_features(sc), n_jobs=n_jobs)
    scale = 1.4826 * np.median(np.abs(a - np.median(a, axis=0)), axis=0)
    norms = np.linalg.norm(a / np.maximum(scale, 1e-9), axis=1)
    use = norms <= np.quantile(norms, 0.98)
    z = _binned_z(a, k, use)
    use &= z.max(axis=1) <= Z_RATE
    return _binned_z(a, k, use)


def tail_rates(z: np.ndarray) -> np.ndarray:
    """Events per 100,000 above Z_RATE, per dye."""
    return 1e5 * (z > Z_RATE).mean(axis=0)


def second_highest(values: np.ndarray) -> float:
    v = np.sort(values[np.isfinite(values)])[::-1]
    return float(v[1]) if len(v) > 1 else (float(v[0]) if len(v) else np.nan)


# ---------------------------------------------------------------------------
# Null
# ---------------------------------------------------------------------------

@dataclass
class NullSource:
    """What one clean sample tells about each dye: chance rates, the top of its z scores, its library's shares."""
    key: str
    rates: np.ndarray                    # (n_dyes,) per 100,000 above Z_RATE; NaN where not counted
    tail: np.ndarray                     # (N_TAIL, n_dyes) highest z per dye, descending; NaN-padded
    n_events: int
    shares: np.ndarray | None = None     # (n_dyes,) largest share of each dye in its base library
    own_dye: int | None = None           # a control's own dye (never counted)


@dataclass
class NullModel:
    kind: str                            # 'unstained', 'controls' or 'none'
    sources: list
    thresholds: np.ndarray               # (n_dyes,)
    reference: np.ndarray                # (n_dyes,) second-highest rate
    excluded: list = field(default_factory=list)   # (source key, dye index) that left a source out


def summarise_source(key: str, z: np.ndarray, own_dye: int | None = None) -> NullSource:
    """Rates and the top N_TAIL z scores of each dye of one clean sample (z of its counted events)."""
    rates = tail_rates(z)
    top = -np.sort(-z, axis=0)[:N_TAIL]
    tail = np.full((N_TAIL, z.shape[1]), np.nan)
    tail[:len(top)] = top
    if own_dye is not None:
        rates[own_dye] = np.nan
        tail[:, own_dye] = np.nan
    return NullSource(key=key, rates=rates, tail=tail, n_events=int(len(z)), own_dye=own_dye)


def build_null(kind: str, sources: list, n_dyes: int) -> NullModel:
    """
    Per-dye thresholds and reference rates. A source with any dye whose rate
    exceeds CONFIRM_RATIO x the second-highest rate of the other sources and
    CONFIRM_MIN_PER_100K is left out entirely: a contaminated sample is not a
    clean reference for any dye (a contaminant also raises the scores of dyes
    spectrally close to it). A dye's threshold is the z of rank
    ceil(THRESH_RATE x events) among the pooled top scores of the sources
    that count for it, floored at Z_FLOOR.
    """
    excluded = []
    if len(sources) > 1:
        rates = np.array([src.rates for src in sources])
        for i in range(len(sources)):
            others = np.delete(rates, i, axis=0)
            ref = np.array([second_highest(others[:, j]) for j in range(n_dyes)])
            with np.errstate(invalid='ignore'):
                hit = (rates[i] > CONFIRM_RATIO * ref) & (rates[i] > CONFIRM_MIN_PER_100K)
            excluded += [(sources[i].key, int(j)) for j in np.flatnonzero(hit)]
    dropped = {key for key, _ in excluded}
    sources = [src for src in sources if src.key not in dropped]
    reference = np.full(n_dyes, np.nan)
    thresholds = np.full(n_dyes, np.inf)
    for j in range(n_dyes):
        counted = [src for src in sources if src.own_dye != j and np.isfinite(src.rates[j])]
        if not counted:
            continue
        reference[j] = second_highest(np.array([src.rates[j] for src in counted]))
        values = np.concatenate([src.tail[:, j] for src in counted])
        values = -np.sort(-values[np.isfinite(values)])
        rank = max(1, int(np.ceil(THRESH_RATE * sum(src.n_events for src in counted))))
        if len(values):
            thresholds[j] = max(Z_FLOOR, float(values[min(rank, len(values)) - 1]))
    return NullModel(kind=kind, sources=sources, thresholds=thresholds, reference=reference, excluded=excluded)


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

def dye_shares(spectrum: np.ndarray, library: np.ndarray, dye_spectra: np.ndarray) -> np.ndarray:
    """Share of ``spectrum`` taken by each dye in turn: NNLS on ``library`` (mean row 0 excluded) plus that dye."""
    others = library[1:] if len(library) > 1 else library
    norm = max(np.linalg.norm(spectrum), 1e-12)
    out = np.zeros(len(dye_spectra))
    for i, s_d in enumerate(dye_spectra):
        coef, _ = nnls(np.vstack([others, s_d]).T, spectrum)
        out[i] = np.linalg.norm(coef[-1] * s_d) / norm
    return out


def library_max_shares(library: np.ndarray, dye_spectra: np.ndarray) -> np.ndarray:
    """Largest share each dye takes in any spectrum of ``library`` (each spectrum against the rest)."""
    best = np.zeros(len(dye_spectra))
    for j in range(len(library)):
        rest = np.delete(library, j, axis=0)
        best = np.maximum(best, dye_shares(library[j], rest, dye_spectra))
    return best


def guard_rejects(candidate: np.ndarray, library: np.ndarray, guard: dict | None,
                  fluor_spectra: np.ndarray) -> bool:
    """True when a refinement candidate gives one guarded dye more than its limit."""
    if not guard:
        return False
    idx = list(guard)
    shares = dye_shares(candidate, library, fluor_spectra[idx])
    return bool((shares > np.array([guard[i] for i in idx])).any())


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------

@dataclass
class ScreenResult:
    keep: np.ndarray                      # (n_events,) bool
    guard: dict                           # {dye index: limit} for refinement
    summary: dict                         # JSON-ready, stored with the profile
    warnings: list                        # plain-language warnings for the user


def remove_events(z: np.ndarray, acted: list, null: NullModel, fluor_names: list) -> tuple:
    """(keep mask, removed per dye) from the z scores of the acted-on dyes (columns in ``acted`` order)."""
    keep = np.ones(len(z), dtype=bool)
    removed_by = {}
    for col, d in enumerate(acted):
        hit = z[:, col] > null.thresholds[fluor_names.index(d)]
        removed_by[d] = int(hit.sum())
        keep &= ~hit
    return keep, removed_by


def unexpected_dyes(rates: np.ndarray, acted: list, null: NullModel, fluor_names: list) -> list:
    """Dyes no neighbour explains whose rate is far above the null's second-highest rate."""
    out = []
    for i, d in enumerate(fluor_names):
        ref = null.reference[i]
        if d not in acted and np.isfinite(ref) and rates[i] > CONFIRM_RATIO * ref \
                and rates[i] > UNEXPECTED_MIN_PER_100K:
            out.append(d)
    return out


def screen_job(controller, sample_key: str, fluor_names: list, fluor_spectra: np.ndarray) -> dict:
    """Everything screen_unstained needs, read on the main thread."""
    from honeychrome.controller_components.particle_type import is_bead_sample
    from honeychrome.controller_components.spread_model import control_records, raw_layout

    layout = raw_layout(controller)
    raw = controller.experiment.settings['raw']
    pnn = raw.get('whitelisted_pnn') or raw['event_channels_pnn']
    layout = dict(layout)
    layout['sc_ids'] = [i for i in layout['sc_ids'] if not str(pnn[i]).upper().endswith('-T')]
    samples = controller.experiment.samples
    all_samples = dict(samples.get('all_samples', {}))
    unstained = [str(k) for k in samples.get('unstained_samples', [])]
    records = control_records(controller, list(fluor_names), layout['detectors'])
    return {
        'experiment_dir': str(controller.experiment_dir), 'layout': layout,
        'pnn_raw': list(pnn), 'whitelisted': raw.get('whitelisted_pnn') is not None,
        'sample_key': str(sample_key), 'fluor_names': list(fluor_names),
        'fluor_spectra': np.asarray(fluor_spectra, dtype=np.float64),
        'all_samples': all_samples, 'unstained': unstained,
        'unstained_cells': [k for k in unstained if not is_bead_sample(samples, k)],
        'controls': [{'path': r['path'], 'label': r['label']} for r in records if r['is_cell']],
        'control_labels': {r['path']: r['label'] for r in records},
    }


def _source_fingerprint(job: dict, key: str, own_dye: int | None) -> str | None:
    try:
        st = (Path(job['experiment_dir']) / key).stat()
    except OSError:
        return None
    h = hashlib.sha256()
    for part in (NULL_VERSION, NULL_EVENTS, N_TAIL, SEED, Z_RATE, K_NEIGHBOURS, N_BINS, OWN_NEGATIVE_Z, key,
                 st.st_size, st.st_mtime_ns, own_dye, tuple(job['layout']['detectors']),
                 tuple(job['layout']['sc_ids'])):
        h.update(repr(part).encode())
    h.update(np.ascontiguousarray(job['fluor_spectra'], dtype=np.float64).tobytes())
    return h.hexdigest()[:24]


def _cache_path(job: dict, fingerprint: str) -> Path:
    from honeychrome.controller_components.autospectral_functions import AF_SUBDIR
    return Path(job['experiment_dir']) / AF_SUBDIR / NULL_CACHE_SUBDIR / f'{fingerprint}.npz'


def _load_cached(job: dict, key: str, fingerprint: str) -> NullSource | None:
    try:
        with np.load(_cache_path(job, fingerprint)) as data:
            own = int(data['own_dye'])
            return NullSource(key=key, rates=data['rates'].copy(), tail=data['tail'].copy(),
                              n_events=int(data['n_events']), shares=data['shares'].copy(),
                              own_dye=None if own < 0 else own)
    except (OSError, KeyError, ValueError):
        return None


def _store_cached(job: dict, fingerprint: str, src: NullSource) -> None:
    path = _cache_path(job, fingerprint)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, rates=src.rates, tail=src.tail, n_events=src.n_events, shares=src.shares,
                 own_dye=-1 if src.own_dye is None else src.own_dye)
    except OSError as exc:
        logger.warning('Contaminant screen: null summary not cached (%s)', exc)


def _read_events(job: dict, key: str) -> tuple:
    """(fluorescence, scatter) of a seeded NULL_EVENTS-event subsample of one sample."""
    from honeychrome.controller_components.sample_loader import read_raw_subsample
    layout = job['layout']
    snap = SimpleNamespace(pnn_raw=tuple(job['pnn_raw']), whitelisted=job['whitelisted'])
    raw, _idx, _n, _kept = read_raw_subsample(Path(job['experiment_dir']) / key, snap, key, NULL_EVENTS, seed=SEED)
    return (np.ascontiguousarray(raw[:, layout['fl_ids']], dtype=np.float64),
            np.ascontiguousarray(raw[:, layout['sc_ids']], dtype=np.float64))


def _base_library(fl: np.ndarray, fluor_spectra: np.ndarray) -> np.ndarray:
    from honeychrome.controller_components.autospectral_functions import get_af_spectra
    return get_af_spectra(fl, fluor_spectra, refine=False)


def _null_source(job: dict, key: str, own_label: str | None, n_jobs: int = -1) -> NullSource | None:
    """One clean sample's summary, from the cache or computed now (its z scores and base library)."""
    names, S = job['fluor_names'], job['fluor_spectra']
    own = names.index(own_label) if own_label in names else None
    fingerprint = _source_fingerprint(job, key, own)
    if fingerprint is None:
        return None
    cached = _load_cached(job, key, fingerprint)
    if cached is not None:
        return cached
    try:
        fl, sc = _read_events(job, key)
    except Exception as exc:
        logger.warning('Contaminant screen: %s not read (%s)', key, exc)
        return None
    if len(fl) < MIN_SOURCE_EVENTS:
        return None
    z = dye_z_scores(fl, sc, S, n_jobs=n_jobs)
    if own is not None:
        negative = z[:, own] < OWN_NEGATIVE_Z
        z, fl = z[negative], fl[negative]
    src = summarise_source(key, z, own)
    try:
        src.shares = library_max_shares(_base_library(fl, S), S)
    except ValueError as exc:
        logger.warning('Contaminant screen: no library for %s (%s)', key, exc)
        src.shares = np.full(len(names), np.nan)
    if own is not None:
        src.shares[own] = np.nan
    _store_cached(job, fingerprint, src)
    return src


def build_experiment_null(job: dict, progress) -> NullModel:
    """The null from the experiment's other unstained cell samples, else from its cell single-stain controls."""
    n_dyes = len(job['fluor_names'])
    others = [k for k in job['unstained_cells'] if k != job['sample_key']]
    plans = (('unstained', [(k, None) for k in others]),
             ('controls', [(c['path'], c['label']) for c in job['controls']]))
    for kind, items in plans:
        if len(items) < MIN_NULL_SOURCES:
            continue
        done = []
        workers = max(1, min(NULL_WORKERS, os.cpu_count() or 1, len(items)))

        def one(item):
            src = _null_source(job, *item, n_jobs=1 if workers > 1 else -1)
            done.append(item)
            progress(f'Contaminant screen: clean reference {len(done)}/{len(items)}')
            return src

        with ThreadPoolExecutor(max_workers=workers) as pool:
            sources = [src for src in pool.map(one, items) if src is not None]
        if len(sources) >= MIN_NULL_SOURCES:
            return build_null(kind, sources, n_dyes)
    return NullModel(kind='none', sources=[], thresholds=np.full(n_dyes, np.inf),
                     reference=np.full(n_dyes, np.nan))


def guard_limits(null: NullModel, acted_idx: list) -> dict:
    """{dye index: limit}: the largest share each acted-on dye takes in the null sources' base libraries."""
    limits = {}
    for j in acted_idx:
        shares = [src.shares[j] for src in null.sources if src.shares is not None and np.isfinite(src.shares[j])]
        limits[j] = float(max([GUARD_FLOOR] + shares))
    return limits


def screen_unstained(job: dict, fl: np.ndarray, sc: np.ndarray, progress=None) -> ScreenResult:
    """
    Screen the events ``fl`` / ``sc`` (all events of ``job['sample_key']``,
    in file order) and return the keep mask, the refinement guard, a summary
    to store with the profile and warnings for the user.
    """
    progress = progress or (lambda _msg: None)
    names, S, key = job['fluor_names'], job['fluor_spectra'], job['sample_key']
    layout = plate_layout.read_layout(job['experiment_dir'], list(job['all_samples']))
    candidates = plate_layout.neighbour_candidates(
        key, layout, names, job['control_labels'], set(job['unstained']), job['all_samples'])
    null = build_experiment_null(job, progress)
    warnings = []
    if null.kind == 'none':
        keep = np.ones(len(fl), dtype=bool)
        acted, unexpected, removed_by, guard = [], [], {}, {}
        warnings.append('Contaminant screen skipped: fewer than two other unstained cell samples '
                        'and fewer than two cell single-stain controls to compare with.')
    else:
        acted = [d for d in candidates.single if d in names]
        acted_idx = [names.index(d) for d in acted]
        keep, removed_by = np.ones(len(fl), dtype=bool), {}
        if acted:
            progress('Contaminant screen: scoring the unstained sample')
            keep, removed_by = remove_events(dye_z_scores(fl, sc, S, acted_idx), acted, null, names)
        progress('Contaminant screen: checking the other dyes')
        sel = np.sort(np.random.default_rng(SEED).choice(len(fl), min(NULL_EVENTS, len(fl)), replace=False))
        unexpected = unexpected_dyes(tail_rates(dye_z_scores(fl[sel], sc[sel], S)), acted, null, names)
        guard = guard_limits(null, acted_idx)
    removed = int((~keep).sum())
    sample_names = job['all_samples']
    if candidates.whole_panel:
        wells = ', '.join(Path(sample_names.get(k, k)).stem for k in candidates.sources.get(plate_layout.WHOLE_PANEL, []))
        warnings.append(f'A neighbouring well holds a fully stained sample ({wells}); '
                        'contamination from it is not screened.')
    if unexpected:
        warnings.append('Unusually many events carry ' + ', '.join(unexpected)
                        + ', which no neighbouring well explains; nothing was removed for them.')
    if removed > HEAVY_REMOVAL_FRACTION * len(fl):
        warnings.append(f'{100 * removed / len(fl):.1f}% of events were removed as contaminated; '
                        'consider extracting AF from another unstained sample.')
    summary = {
        'null': null.kind, 'null_sources': [s.key for s in null.sources],
        'null_left_out': [[k, names[j]] for k, j in null.excluded],
        'candidates': {d: list(v) for d, v in candidates.sources.items()},
        'acted_on': acted,
        'thresholds': {d: round(float(null.thresholds[names.index(d)]), 3) for d in acted},
        'removed_by_dye': removed_by, 'removed': removed, 'events': int(len(fl)),
        'guard_limits': {names[j]: round(v, 4) for j, v in guard.items()},
        'unexpected': unexpected, 'warnings': list(warnings),
    }
    logger.info('Contaminant screen of %s: %s', key, summary)
    return ScreenResult(keep=keep, guard=guard, summary=summary, warnings=warnings)


# ---------------------------------------------------------------------------
# Storage and use by other consumers of the unstained sample
# ---------------------------------------------------------------------------

def excluded_file_name(profile_name: str) -> str:
    return f'{profile_name}{EXCLUDED_SUFFIX}'


def save_excluded(keep: np.ndarray, experiment_dir, file_name: str) -> None:
    """Indices (in file order) of the events left out of a profile's extraction."""
    from honeychrome.controller_components.autospectral_functions import AF_SUBDIR
    folder = Path(experiment_dir) / AF_SUBDIR
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / file_name, np.flatnonzero(~np.asarray(keep, dtype=bool)).astype(np.int64))


def load_excluded(experiment_dir, file_name: str) -> np.ndarray | None:
    from honeychrome.controller_components.autospectral_functions import AF_SUBDIR
    path = Path(experiment_dir) / AF_SUBDIR / file_name
    try:
        return np.load(path)
    except (OSError, ValueError):
        return None


def excluded_for_sample(experiment_dir, af_profiles: dict, sample_key: str) -> np.ndarray | None:
    """
    Event indices left out of every AF profile extracted from ``sample_key``
    (their union); None when no profile of that sample was screened.
    """
    found = []
    for entry in (af_profiles or {}).values():
        screen = entry.get('contaminant_screen') or {}
        if str(entry.get('source_fcs')) != str(sample_key) or not screen.get('excluded_file'):
            continue
        idx = load_excluded(experiment_dir, screen['excluded_file'])
        if idx is not None:
            found.append(idx)
    if not found:
        return None
    return np.unique(np.concatenate(found))


def keep_mask(excluded: np.ndarray | None, n_events: int) -> np.ndarray | None:
    """Boolean keep mask over a file's events from excluded indices; None when nothing is excluded."""
    if excluded is None or len(excluded) == 0:
        return None
    keep = np.ones(n_events, dtype=bool)
    keep[excluded[excluded < n_events]] = False
    return keep


def excluded_digest(excluded: np.ndarray | None) -> str | None:
    if excluded is None or len(excluded) == 0:
        return None
    return hashlib.sha256(np.ascontiguousarray(excluded, dtype=np.int64).tobytes()).hexdigest()[:16]

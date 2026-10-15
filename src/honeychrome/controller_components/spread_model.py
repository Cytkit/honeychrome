"""
spread_model.py
---------------
Spillover spread of the spectral model's single-stain controls, for the
negative-envelope AF scoring.

For each control, the Residual Model (AutoSpectral's get.fluor.variants())
fits how much variance the control's dye adds to every other fluorophore's
unmixed channel per unit of its own abundance. The variance line is fitted
through per-bin variances (squared normalised MAD) over quantile bins of the
dye's abundance. The result is the source x target spread matrix stored as
experiment.process['af_spread'].

Public API
----------
spread_snapshot(controller, fluor_names, fluor_spectra, unstained_path)
    Main-thread snapshot of everything compute_spread_entry needs.

compute_spread_entry(snapshot, progress, loader)
    Spread for every control; safe to run on a worker thread. Returns the
    dict stored as experiment.process['af_spread'], or None when no control
    gave an estimate.

spread_is_current(entry, fluor_names, fluor_spectra)
    Whether a stored spread still matches the current spectra.

residual_model_spread_one(pos_raw, pos_scatter, neg_raw, neg_scatter, spectra,
                          fluor_idx, peak_ch_idx, raw_pos_threshold, af_pcs,
                          is_cell_control, saturation_ceiling)
    Spread of one control (NumPy only).
"""

import logging
from pathlib import Path

import numpy as np

from honeychrome.controller_components.autospectral_functions import spectra_unchanged
from honeychrome.controller_components.spectral_cleaning import knn_scatter_match
from honeychrome.controller_components.spectral_refinement import huber_fit

logger = logging.getLogger(__name__)

SPREAD_ESTIMATOR = 'binned_mad'
NEGATIVE_GATE = 'Neg Unstained'
N_BINS = 20
MIN_BIN_EVENTS = 20
HUBER_K = 1.345
_MAX_NEGATIVE_EVENTS = 100_000


# ---------------------------------------------------------------------------
# Residual Model for one control
# ---------------------------------------------------------------------------

def _binned_variance_line(b: np.ndarray, proj: np.ndarray, n_bins: int, min_bin_events: int,
                          huber_k: float) -> tuple:
    """
    Huber line of per-bin variance (squared normalised MAD) on the bin's mean
    abundance, over quantile bins of b. Returns (coef, converged) as
    huber_fit does; nothing converges with fewer than three bins.
    """
    n_bins = int(max(1, min(n_bins, len(b) // max(min_bin_events, 1))))
    n_targets = proj.shape[1]
    if n_bins < 3:
        return np.zeros((2, n_targets)), np.zeros(n_targets, dtype=bool)
    edges = np.quantile(b, np.linspace(0.0, 1.0, n_bins + 1))
    which = np.clip(np.searchsorted(edges, b, side='right') - 1, 0, n_bins - 1)
    centres, variances = [], []
    for i in range(n_bins):
        sel = which == i
        if sel.sum() < min_bin_events:
            continue
        p = proj[sel]
        centres.append(b[sel].mean())
        variances.append((1.4826 * np.median(np.abs(p - np.median(p, axis=0)), axis=0)) ** 2)
    if len(centres) < 3:
        return np.zeros((2, n_targets)), np.zeros(n_targets, dtype=bool)
    return huber_fit(np.array(centres), np.array(variances), k=huber_k)


def residual_model_spread_one(
    pos_raw: np.ndarray,
    pos_scatter: np.ndarray | None,
    neg_raw: np.ndarray | None,
    neg_scatter: np.ndarray | None,
    spectra: np.ndarray,
    fluor_idx: int,
    peak_ch_idx: int,
    raw_pos_threshold: float,
    af_pcs: np.ndarray | None,
    is_cell_control: bool,
    saturation_ceiling: float,
    k_neighbors: int = 3,
    af_collinear_threshold: float = 0.95,
    max_events: int = 20_000,
    min_events: int = 20,
) -> dict | None:
    """
    Spillover spread of one single-stain control by the Residual Model.

    Positive events (peak channel above raw_pos_threshold, unsaturated, the
    brightest max_events) are background corrected by kNN scatter matching to
    the negative, then fitted with a low-rank design: the AF principal
    components plus this fluorophore's own row (cells, when the row is not
    collinear with AF), or the row alone. b_h is the row's abundance and the
    residual is projected through the full-panel OLS unmixing matrix;
    Var(unmixed_k) = intercept_k + slope_k * b_h is fitted through binned
    variances, and slope_k is SS(fluor, k), the variance added per unit of
    on-channel abundance.

    Returns {'slope', 'intercept'} ((F,), NaN on the diagonal and where the
    fit did not converge) and 'n', or None with fewer than min_events
    positive events.
    """
    pos_raw = np.asarray(pos_raw, dtype=np.float64)
    unsaturated = ~np.any(pos_raw >= 0.999 * saturation_ceiling, axis=1)
    pos = pos_raw[unsaturated]
    scatter = None if pos_scatter is None else np.asarray(pos_scatter, dtype=np.float64)[unsaturated]

    idx = np.where(pos[:, peak_ch_idx] > raw_pos_threshold)[0]
    if len(idx) > max_events:
        idx = idx[np.argsort(pos[idx, peak_ch_idx])[::-1][:max_events]]
    if len(idx) < min_events:
        return None

    if (neg_raw is not None and len(neg_raw) > 0 and scatter is not None
            and neg_scatter is not None and len(neg_scatter) > 0):
        corrected, _ = knn_scatter_match(pos[idx], scatter[idx], np.asarray(neg_raw, dtype=np.float64),
                                         np.asarray(neg_scatter, dtype=np.float64), k=k_neighbors)
    else:
        rest = np.setdiff1d(np.arange(len(pos)), idx)
        background = pos[rest].mean(axis=0) if len(rest) > 50 else np.zeros(pos.shape[1])
        corrected = pos[idx] - background

    reference = spectra[fluor_idx]
    design = reference[np.newaxis, :]
    if is_cell_control and af_pcs is not None and len(af_pcs) > 0:
        pc_norms = np.linalg.norm(af_pcs, axis=1) + 1e-9
        cosine = np.abs(af_pcs @ reference) / (pc_norms * (np.linalg.norm(reference) + 1e-9))
        if float(cosine.max()) < af_collinear_threshold:
            design = np.vstack([af_pcs, reference])

    low_rank = corrected @ np.linalg.pinv(design)
    b_h = low_rank[:, -1]
    resid = corrected - low_rank @ design

    unmix_full = np.linalg.solve(spectra @ spectra.T, spectra)
    proj = resid @ unmix_full.T

    coef, converged = _binned_variance_line(b_h, proj, N_BINS, MIN_BIN_EVENTS, HUBER_K)
    intercept = np.where(converged, coef[0], np.nan)
    slope = np.where(converged, coef[1], np.nan)
    intercept[fluor_idx] = np.nan
    slope[fluor_idx] = np.nan
    return {'slope': slope, 'intercept': intercept, 'n': int(len(b_h))}


def _align_spread(slope: np.ndarray) -> np.ndarray:
    """Finite, non-negative, zero diagonal."""
    spread = np.maximum(np.where(np.isfinite(slope), slope, 0.0), 0.0)
    np.fill_diagonal(spread, 0.0)
    return spread


def _af_pcs(unstained_raw: np.ndarray, n_pcs: int = 4) -> np.ndarray:
    """Top AF principal components of the negative, unit length."""
    centred = unstained_raw - unstained_raw.mean(axis=0)
    n_eff = max(1, min(n_pcs, centred.shape[0], centred.shape[1]))
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    pcs = vt[:n_eff]
    norms = np.linalg.norm(pcs, axis=1, keepdims=True)
    return pcs / np.where(norms < 1e-12, 1.0, norms)


def spread_is_current(entry: dict | None, fluor_names, fluor_spectra: np.ndarray) -> bool:
    """Whether a stored spread matches the current fluorophores within the spectral tolerance."""
    return bool(entry) and spectra_unchanged(entry['fluor_names'], entry['fluor_spectra'],
                                             fluor_names, fluor_spectra)


# ---------------------------------------------------------------------------
# Controls and events
# ---------------------------------------------------------------------------

def raw_layout(controller) -> dict:
    """Raw channel selectors for get_raw_events, remapped to the whitelisted order."""
    raw = controller.experiment.settings['raw']
    full = raw['event_channels_pnn']
    pnn = raw.get('whitelisted_pnn') or full
    fl_ids = [pnn.index(full[i]) for i in controller.filtered_raw_fluorescence_channel_ids]
    sc_ids = [pnn.index(full[i]) for i in raw['scatter_channel_ids']]
    return {
        'fl_ids': fl_ids, 'sc_ids': sc_ids, 'col_order': raw.get('whitelisted_pnn'),
        'detectors': [pnn[i] for i in fl_ids],
        'ceiling': float(raw.get('magnitude_ceiling') or 2 ** 18),
        'n_scatter': len(sc_ids),
    }


def control_records(controller, fluor_names: list, detectors: list) -> list:
    """Single-stain controls of the spectral model, one dict per fluorophore."""
    process = controller.experiment.process
    samples = controller.experiment.samples.get('all_samples', {})
    by_name = {v: k for k, v in samples.items()}
    profiles = process.get('profiles') or {}
    records = []
    for c in process.get('spectral_model', []):
        label = c.get('label')
        if not label or label == 'AF' or 'unstained' in label.lower() or label not in fluor_names:
            continue
        path = by_name.get(c.get('sample_name') or '')
        if path is None:
            continue
        try:
            peak = detectors.index(c.get('gate_channel'))
        except ValueError:
            peak = int(np.argmax(profiles[label]))
        records.append({
            'label': label, 'fluor_idx': fluor_names.index(label), 'path': path,
            'neg_path': by_name.get(c.get('universal_negative_name') or ''),
            'peak': peak, 'is_cell': str(c.get('particle_type', 'cells')).lower() == 'cells',
        })
    return records


def spread_snapshot(controller, fluor_names: list, fluor_spectra: np.ndarray, unstained_path: str) -> dict:
    """
    Everything compute_spread_entry needs, read on the main thread. The
    positives are the controls' events in the base gate; the negative is the
    control's universal negative ("Neg Unstained" gate) when it has one, else
    the unstained sample the AF profile is extracted from. Each sample is
    gated with its own gates (its per-sample overrides and Time range).
    """
    from honeychrome.controller_components.functions import resolve_base_gate
    from honeychrome.controller_components.per_sample_gating_functions import SampleGating

    base_gate, _ = resolve_base_gate(controller.raw_gating, controller.experiment.process.get('base_gate_priority_order', []))
    layout = raw_layout(controller)
    return {
        'experiment_dir': str(controller.experiment_dir),
        'gating': SampleGating.from_controller(controller, scope='raw'), 'layout': layout,
        'fluor_names': list(fluor_names), 'spectra': np.asarray(fluor_spectra, dtype=np.float64),
        'controls': control_records(controller, list(fluor_names), layout['detectors']),
        'unstained_path': unstained_path, 'base_gate': base_gate,
    }


def load_events(experiment_dir, rel_path: str, gate_label: str | None, layout: dict, gating) -> tuple:
    """(fluorescence, scatter) raw events of one sample inside gate_label (all events when None).
    rel_path is the sample's experiment key; gating a SampleGating or GatingStrategy."""
    from honeychrome.controller_components.functions import sample_from_fcs
    from honeychrome.controller_components.spectral_functions import get_raw_events

    sample = sample_from_fcs(str(Path(experiment_dir) / rel_path))
    fl, sc = get_raw_events(sample, layout['fl_ids'], gate_label=gate_label, gating_strategy=gating,
                            extra_channel_ids=layout['sc_ids'], col_order=layout['col_order'],
                            sample_key=rel_path)
    return np.asarray(fl, dtype=np.float64), np.asarray(sc, dtype=np.float64)


def _load_negative(loader, snapshot: dict, path: str | None):
    """Negative events of a control: its own sample first, then the unstained sample."""
    for source in (path, snapshot['unstained_path']):
        if source is None:
            continue
        for gate in (NEGATIVE_GATE, snapshot['base_gate']):
            try:
                fl, sc = loader(snapshot['experiment_dir'], source, gate, snapshot['layout'], snapshot['gating'])
            except Exception as exc:
                logger.warning(f'Spread: {source} not loaded in gate "{gate}" ({exc})')
                continue
            if len(fl) > 0:
                return fl, sc
    return None


def compute_spread_entry(snapshot: dict, progress, loader=None) -> dict | None:
    """
    Spillover spread for every control in the snapshot (see the module
    docstring). progress(text) is called before each control. Returns the
    dict stored as experiment.process['af_spread'], or None when no control
    gave an estimate.
    """
    loader = loader or load_events
    names, spectra, layout = snapshot['fluor_names'], snapshot['spectra'], snapshot['layout']
    n_fluors = len(names)
    slope = np.full((n_fluors, n_fluors), np.nan)
    negatives = {}

    def negative(path):
        if path not in negatives:
            loaded = _load_negative(loader, snapshot, path)
            if loaded is None:
                negatives[path] = None
            else:
                fl, sc = loaded
                rng = np.random.default_rng(0)
                sub = fl if len(fl) <= _MAX_NEGATIVE_EVENTS else fl[
                    np.sort(rng.choice(len(fl), _MAX_NEGATIVE_EVENTS, replace=False))]
                negatives[path] = (fl, sc, _af_pcs(sub), np.percentile(sub, 99.5, axis=0))
        return negatives[path]

    controls = snapshot['controls']
    for n, rec in enumerate(controls):
        progress(f'Spread: {rec["label"]} ({n + 1}/{len(controls)})')
        try:
            pos_fl, pos_sc = loader(snapshot['experiment_dir'], rec['path'], snapshot['base_gate'],
                                    layout, snapshot['gating'])
        except Exception as exc:
            logger.warning(f'Spread: control {rec["label"]} not loaded ({exc})')
            continue
        neg = negative(rec['neg_path'])
        if neg is None:
            logger.warning(f'Spread: no negative events for {rec["label"]}')
            continue
        neg_fl, neg_sc, af_pcs, raw_thr = neg
        n_sc = layout['n_scatter']
        row = residual_model_spread_one(
            pos_fl, pos_sc[:, :n_sc], neg_fl, neg_sc[:, :n_sc], spectra, rec['fluor_idx'], rec['peak'],
            raw_thr[rec['peak']], af_pcs if rec['is_cell'] else None, rec['is_cell'], layout['ceiling'])
        if row is not None:
            slope[rec['fluor_idx']] = row['slope']
    n_estimated = int(np.sum(np.isfinite(slope).any(axis=1)))
    if n_estimated == 0:
        return None
    return {
        'fluor_names': list(names),
        'fluor_spectra': spectra.tolist(),
        'spread': _align_spread(slope).tolist(),
        'n_estimated': n_estimated,
        'estimator': SPREAD_ESTIMATOR,
    }

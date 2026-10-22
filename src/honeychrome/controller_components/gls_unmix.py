"""
gls_unmix.py
------------
Noise-weighted unmixing for the AutoSpectral tab: the detector noise model
fitted from the experiment's controls (Noise page) and M2pg, penalised GLS
with the positivity prior. No Qt.

Noise model fit (``fit_experiment_noise_model``, worker thread)
    Event selection (``NoiseSetupSettings.selection``): ``'auto'`` loads
    every sample at root and keeps its scatter core and singlets
    (``noise_selection.select_events``), ignoring the user's gates; the
    counts per sample are stored. ``'gates'`` (A/B comparison only, not in
    the UI) uses the gates as below in brackets.

    1. The unstained cell sample is loaded (in its gate) and split into
       training and held-out events (``unstained_train_frac``); the AF
       projection basis comes from the training events. "Auto" picks, among
       the unstained cell samples with a stored AF profile, the one with the
       lowest mean raw signal (from the profile's stored unstained
       subsample when it has one).
    2. Bead negatives anchor sigma0: unstained bead samples (or the one the
       user chose), else the bead controls' universal negatives, else
       internal negatives found inside the bead controls (``bead_mode``).
    3. Each single-stain control is loaded (in the base gate) and split
       (``control_train_frac``, default 70% training). Its positives
       (unsaturated, above the 99.5th percentile of its negative in the peak
       detector) give its variants, discovered on the training split with
       the Spectral Variants page's default settings and cached in
       ``AutoSpectral/noise_variants.npz``, and its variant basis (PCA of
       the centred deltas). Only the events the fit and the calibration use
       are kept, one control at a time.
    4. The noise model (``noise_model.fit_noise_model``) is fitted on the
       training positives, unstained and bead events, and calibrated on the
       held-out events.
    5. The OLS variance of the held-out unstained cells and of the bead
       negatives per fluorophore: the prior's negative SD for samples
       unmixed without AF.

    Design and estimator options (default off; ``NoiseSetupSettings`` and
    its ``noise``, a ``noise_model.NoiseSettings``):

    * ``bead_af_rows`` (3 when on): a bead negative is projected onto that
      many AF rows of its own training events instead of its mean, and each
      bead control's design takes its negative's rows instead of the offset.
    * ``laser_split``: each control's design takes one row per laser, its
      spectrum restricted to that laser's detectors, when that part and the
      rest each reach ``laser_split_min`` of the peak. Lasers come from the
      cytometer database (``detector_lasers``), else $PnL, else the name.
    * ``noise.per_control_quadratic``, ``noise.sigma0_by_class`` (classes
      from the particle type: bead controls and bead negatives 'beads',
      the rest 'cells') and ``noise.sigma0_fallback``: see noise_model.py.

    Clamped data (``clamp_mode = 'auto'``): when the unstained cells and
    bead negatives have no detector with 1% of values at or below zero
    (``noise_model.negatives_clamped``), the fit is stored with
    ``clamped`` and Model Noise sets ``process['unmixing']['weights'] =
    'clamped'`` instead of the noise model: penalised GLS with the
    positivity prior and the weights of ``clamp_weights``,

    * ``'poisson'`` (M1p): ``V_d = s g_d (F + mu)``, the fitted ``g = 1 /
      kappa``, the floor ``F`` = ``clamp_floor_pe`` photoelectrons in raw
      units (times the median ``g``);
    * ``'flat'`` (M0p): one variance for every detector and signal,

    each scaled (``s``, or the flat variance ``flat_phi``) so that it
    predicts the residual variance of the held-out unstained cells about
    their AF basis (median over detectors).

    The order of the random draws matches the GLS Test plugin's Setup, so
    the same inputs give the same model.

M2pg (``unmix_m2pg``)
    The noise model of the sample's particle class (``NoiseModel.for_class``)
    is passed in by the caller. The per-cell AF start (the main app's
    positivity-weighted assignment) chosen on events, spectra and AF rows
    whitened by one weight per detector for the sample, ``1 / V(median raw signal)``; then two
    plug-in iterations of penalised GLS: weights ``1 / V(mu)`` at the
    current signal (saturated detectors almost excluded) and the ridge
    ``lambda w_g / tau_g^2`` per fluorophore, ``w_g`` the positivity weight
    of the current estimate against the negative's SD ``tau_g``; each
    channel is reported with its own ridge removed (Sherman-Morrison). The
    compiled kernel (bundled_plugins/gls_m2p_kernel.cpp) runs it when built;
    otherwise the NumPy path here, with the same result.

Storage
    ``process['noise_model']`` holds the fit's description (unstained
    sample, gate, bead sources, calibration summary, the no-AF negative
    variances, the fingerprint of the spectra it was fitted with) and names
    ``AutoSpectral/noise_model.npz`` (``noise_model.save_noise_model``) and
    ``AutoSpectral/noise_negatives.npz`` (up to 5,000 held-out unstained
    cells and bead negatives, for the no-AF variances under other spectra,
    ``null_variances``).
    ``process['unmixing']`` holds ``noise_model`` (in use) and
    ``prior_lambda``.

v1 simplifications
    * The Auto unstained choice compares the profiles' stored unstained
      subsamples where they exist; those have no scatter, so are not
      re-selected.
    * Laser split is skipped when any detector's laser is unknown.
    * With several AF profiles assigned, the prior's negative SD uses the
      null variance of the first profile's mean AF row (as the kernel and
      the GLS Test plugin do).
    * A bead control with no negative of its own, when the only bead
      negatives are internal ones, is processed after the others.
    * A model fitted with earlier spectra stays usable (it describes the
      detectors); ``spectra_changed`` reports it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from honeychrome.controller_components import noise_model as nm
from honeychrome.controller_components import noise_selection as ns
from honeychrome.controller_components.autospectral_functions import AF_SUBDIR

logger = logging.getLogger(__name__)

PROCESS_KEY = 'noise_model'
UNMIXING_KEY = 'unmixing'
NOISE_FILE = 'noise_model.npz'
NEGATIVES_FILE = 'noise_negatives.npz'
NEGATIVE_SAMPLE_EVENTS = 5000
VARIANT_FILE = 'noise_variants.npz'
VARIANT_CACHE_VERSION = 1
AUTO = '__auto__'
CLAMPED_WEIGHTS = 'clamped'
DEFAULT_PRIOR_LAMBDA = 1.0
N_ITER = 2
_EPS = 1e-12


def _bundled():
    """The bundled-plugins folder on sys.path, for the AutoSpectral modules and kernels kept there."""
    from honeychrome.controller_components.plugin_loaders import bundled_plugins_path
    if str(bundled_plugins_path) not in sys.path:
        sys.path.insert(0, str(bundled_plugins_path))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class VariantSettings:
    """The Spectral Variants page's defaults."""
    n_cells: int = 10_000
    som_dim: int = 5
    k_neighbors: int = 3
    sim_threshold: float = 0.985
    sim_threshold_floor: float = 0.90
    af_collinear_threshold: float = 0.95
    dedup_threshold: float = 0.01
    max_variants: int = 10


@dataclass
class NoiseSetupSettings:
    unstained_train_frac: float = 0.5
    control_train_frac: float = 0.7
    max_unstained: int = 200_000
    max_test_unstained: int = 20_000
    max_control_events: int = 100_000
    max_test_control: int = 5_000
    max_fit_events: int = 6_000
    max_cal_events: int = 2_000
    seed: int = 42
    n_af_rows: int = 4
    basis_var_explained: float = 0.95
    basis_max_rank: int = 5
    bead_mode: str = 'auto'
    bead_factor: float = 10.0
    selection: str = 'auto'
    bead_af_rows: int = 0
    laser_split: bool = False
    laser_split_min: float = 0.01
    clamp_mode: str = 'auto'
    clamp_weights: str = 'poisson'
    clamp_floor_pe: float = 10.0
    variant: VariantSettings = field(default_factory=VariantSettings)
    noise: nm.NoiseSettings = field(default_factory=nm.NoiseSettings)


# ---------------------------------------------------------------------------
# Splits, bases, bead negatives
# ---------------------------------------------------------------------------

def robust_sd(x: np.ndarray, axis=0) -> np.ndarray:
    med = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=axis)


def holdout_split(n: int, train_frac: float, seed: int, max_test: int | None = None) -> tuple:
    """Sorted training and held-out indices; the held-out set capped at ``max_test``."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError('The training fraction must be between 0 and 1.')
    order = np.random.default_rng(seed).permutation(n)
    n_train = int(round(n * train_frac))
    train, test = order[:n_train], order[n_train:]
    if max_test is not None and len(test) > max_test:
        test = test[:max_test]
    return np.sort(train), np.sort(test)


def subsample(n: int, cap: int, seed: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))


def positive_rows(raw: np.ndarray, peak: int, threshold: float, ceiling: float) -> np.ndarray:
    """Unsaturated events above ``threshold`` in the peak detector."""
    unsaturated = ~np.any(raw >= 0.999 * ceiling, axis=1)
    return np.flatnonzero(unsaturated & (raw[:, peak] > threshold))


def af_basis(unstained_raw: np.ndarray, n_rows: int = 4, max_events: int = 20_000, seed: int = 0) -> np.ndarray:
    """
    Leading right singular vectors of the uncentred unstained events, unit
    length: the first follows the mean AF spectrum, the rest its main
    variation.
    """
    raw = np.asarray(unstained_raw, dtype=np.float64)
    raw = raw[subsample(len(raw), max_events, seed)]
    _, _, vt = np.linalg.svd(raw, full_matrices=False)
    rows = vt[:max(1, min(n_rows, vt.shape[0]))]
    sign = np.sign(rows.sum(axis=1, keepdims=True))
    return rows * np.where(sign == 0, 1.0, sign)


@dataclass
class VariantBasis:
    """Centred variant covariance of one fluorophore: ``basis`` (r x D) and ``lam`` (r,)."""
    label: str
    basis: np.ndarray
    lam: np.ndarray
    mean_delta: np.ndarray


def variant_basis(label: str, delta: np.ndarray, var_explained: float = 0.95, max_rank: int = 5,
                  min_lambda: float = 1e-4) -> VariantBasis | None:
    """
    PCA of the variant deltas (rows 1.., row 0 being the reference itself)
    minus their mean. The rank is the smallest reaching ``var_explained``,
    at most ``max_rank``, dropping eigenvalues below ``min_lambda`` of the
    largest. None with fewer than two variants.
    """
    delta = np.asarray(delta, dtype=np.float64)[1:]
    if len(delta) < 2:
        return None
    mean_delta = delta.mean(axis=0)
    _, sv, vt = np.linalg.svd(delta - mean_delta, full_matrices=False)
    lam = sv ** 2 / (len(delta) - 1)
    if lam[0] <= 0:
        return None
    keep = np.flatnonzero(lam >= min_lambda * lam[0])[:max_rank]
    cum = np.cumsum(lam[keep]) / lam.sum()
    if np.any(cum >= var_explained):
        keep = keep[:int(np.argmax(cum >= var_explained)) + 1]
    return VariantBasis(label, vt[keep], lam[keep], mean_delta)


def laser_rows(spectrum: np.ndarray, lasers, split_min: float = 0.01) -> list:
    """
    The spectrum restricted to each laser's detectors, (1, D) rows, for the
    lasers where that part and the rest each reach ``split_min`` of the
    spectrum's peak.
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    las = np.asarray([str(x) for x in lasers])
    peak = float(np.max(np.abs(spectrum)))
    rows = []
    for laser in dict.fromkeys(las.tolist()):
        block = np.where(las == laser, spectrum, 0.0)
        if np.max(np.abs(block)) >= split_min * peak and np.max(np.abs(spectrum - block)) >= split_min * peak:
            rows.append(block[np.newaxis, :])
    return rows


def control_design(spectrum: np.ndarray, vb: VariantBasis | None, af_rows: np.ndarray | None,
                   mean_delta_min: float = 1e-3, lasers=None, split_min: float = 0.01) -> np.ndarray:
    """
    Projection design of one control: its reference spectrum (row 0, the
    binning variable), its per-laser rows when ``lasers`` is given
    (``laser_rows``), the mean variant delta when it exceeds
    ``mean_delta_min`` of the peak, the variant basis and the AF rows.
    """
    rows = [np.asarray(spectrum, dtype=np.float64)[np.newaxis, :]]
    if lasers is not None:
        rows.extend(laser_rows(spectrum, lasers, split_min))
    if vb is not None:
        if np.max(np.abs(vb.mean_delta)) > mean_delta_min:
            rows.append(vb.mean_delta[np.newaxis, :])
        rows.append(vb.basis)
    if af_rows is not None:
        rows.append(af_rows)
    return np.vstack(rows)


def otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """Otsu split of ``values``."""
    hist, edges = np.histogram(values, bins=n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres)
    with np.errstate(invalid='ignore', divide='ignore'):
        between = w0 * w1 * (m0 / w0 - (m0[-1] - m0) / w1) ** 2
    between = np.where(np.isfinite(between), between, -1.0)
    return float(centres[int(np.argmax(between))])


def bead_reference_levels(medians: list, spectra: np.ndarray, hard_frac: float = 0.005) -> np.ndarray:
    """
    Dark level of bead events per detector, (D,): the median, over bead
    controls whose dye emits at most ``hard_frac`` of its peak there, of
    that control's median signal in the detector. ``medians`` is a list of
    ``(median (D,), fluor_idx)``. NaN where no control is that dark.
    """
    d = spectra.shape[1]
    levels = [[] for _ in range(d)]
    for med, f in medians:
        for j in np.flatnonzero(spectra[f] <= hard_frac):
            levels[j].append(med[j])
    return np.array([np.median(v) if v else np.nan for v in levels])


def internal_bead_negative(events: np.ndarray, peak: int, reference_level: float, factor: float = 10.0,
                           tolerance: float = 3.0, min_events: int = 200) -> tuple:
    """
    Unstained beads inside a bead single-stain control: an Otsu split of
    log10 of the peak detector, accepted when the lower population's median
    is at least ``factor`` times below the upper one's, it has ``min_events``
    events, and its median is within ``tolerance`` times the dark level
    other bead controls show in this detector (``reference_level``, ignored
    when NaN). Returns ``(mask or None, reason)``.
    """
    p = np.log10(np.maximum(events[:, peak], 1.0))
    if len(p) < 2 * min_events or np.ptp(p) < np.log10(factor):
        return None, 'no bimodal peak channel'
    cut = otsu_threshold(p)
    low = p < cut
    if low.sum() < min_events or (~low).sum() < min_events:
        return None, 'too few events on one side of the split'
    lo_med = float(np.median(events[low, peak]))
    hi_med = float(np.median(events[~low, peak]))
    if hi_med < factor * max(lo_med, 1.0):
        return None, f'populations only {hi_med / max(lo_med, 1.0):.1f}x apart'
    if np.isfinite(reference_level) and lo_med > tolerance * max(reference_level, 1.0):
        return None, f'lower population ({lo_med:,.0f}) above the bead dark level ({reference_level:,.0f})'
    low &= events[:, peak] < hi_med / factor
    if low.sum() < min_events:
        return None, 'too few events below the positives / factor'
    return low, f'{int(low.sum()):,} events, {hi_med / max(lo_med, 1.0):.0f}x below the positives'


# ---------------------------------------------------------------------------
# Variant cache
# ---------------------------------------------------------------------------

def fingerprint(spectra: np.ndarray, payload: dict) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(np.asarray(spectra, dtype=np.float64)).tobytes())
    h.update(json.dumps(payload, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def save_variants(variants: dict, path, fp: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays, meta = {}, {}
    for i, (label, entry) in enumerate(variants.items()):
        for key in ('v_mats', 'delta', 'delta_norms'):
            arrays[f'v{i}__{key}'] = np.asarray(entry[key])
        meta[label] = {'index': i, **{k: v for k, v in entry.items()
                                      if k not in ('v_mats', 'delta', 'delta_norms')}}
    np.savez(path, meta=np.array(json.dumps(meta, default=float)), fingerprint=np.array(fp),
             version=np.array(VARIANT_CACHE_VERSION), **arrays)


def load_variants(path, fp: str) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data['fingerprint']) != fp or int(data['version']) != VARIANT_CACHE_VERSION:
                return None
            meta = json.loads(str(data['meta']))
            out = {}
            for label, fields in meta.items():
                i = fields.pop('index')
                out[label] = {**fields, **{k: np.array(data[f'v{i}__{k}'])
                                           for k in ('v_mats', 'delta', 'delta_norms')}}
            return out
    except Exception as exc:
        logger.warning('Noise model: variant cache %s not read (%s)', path, exc)
        return None


# ---------------------------------------------------------------------------
# Snapshot (main thread)
# ---------------------------------------------------------------------------

def fluor_names_and_spectra(controller) -> tuple:
    """Fluorophores with a reference spectrum, in spectral-model order."""
    process = controller.experiment.process
    profiles = process.get('profiles') or {}
    names = [c['label'] for c in process.get('spectral_model', []) if c['label'] in profiles]
    if not names:
        return [], None
    return names, np.array([profiles[n] for n in names], dtype=np.float64)


def _is_bead(samples: dict, path: str) -> bool:
    from honeychrome.controller_components.particle_type import is_bead_sample
    return bool(is_bead_sample(samples, path))


def _is_unstained(samples: dict, path: str, name: str) -> bool:
    tagged = set(samples.get('unstained_samples', []))
    return path in tagged or 'unstained' in name.lower() or 'unstained' in path.lower()


def unstained_cell_samples(samples: dict) -> list:
    """(path, name) of unstained cell samples."""
    return [(p, n) for p, n in samples.get('all_samples', {}).items()
            if not _is_bead(samples, p) and _is_unstained(samples, p, n)]


def unstained_bead_samples(samples: dict) -> list:
    """(path, name) of unstained bead samples."""
    return [(p, n) for p, n in samples.get('all_samples', {}).items()
            if _is_bead(samples, p) and _is_unstained(samples, p, n)]


def stored_af_for(controller, path: str, detectors: list) -> tuple:
    """
    ``(spectra, sidecar)`` of the AF profile extracted from the unstained
    sample at ``path``: its spectra (or None) and the path of its envelope's
    5,000-event unstained subsample (or None).
    """
    _bundled()
    from autospectral_optimization_functions import _stored_af_profile
    found = _stored_af_profile(controller, path, detectors)
    if found is None:
        return None, None
    _name, entry, spectra = found
    sidecar = (entry.get('envelope') or {}).get('unstained_file')
    if sidecar:
        sidecar = Path(controller.experiment_dir) / AF_SUBDIR / sidecar
        if not sidecar.exists():
            sidecar = None
    return spectra, None if not sidecar else str(sidecar)


def _name_laser(detector: str) -> str | None:
    """The laser in a detector name that carries one (355CH1-A, UV3-A, YG10-A), else None."""
    m = re.match(r'^(\d{3})CH\d+', detector)
    if m:
        return m.group(1)
    m = re.match(r'^(UV|V|B|YG|Y|G|R|IR|DUV)\d+(?: \(\d+\))?-[AH]$', detector)
    return m.group(1) if m else None


def detector_lasers(raw_settings: dict, detectors: list, fcs_path: str | None = None) -> list:
    """
    The laser of each detector (None when unknown): the cytometer
    database's map, else the file's $PnL, else the name (``_name_laser``).
    Main thread (the database module imports Qt).
    """
    from honeychrome.controller_components.cytometer_whitelist import get_detector_laser_map

    known = get_detector_laser_map(raw_settings.get('cytometer_db_col') or '')
    out = [known.get(d) for d in detectors]
    if fcs_path and any(x is None for x in out):
        try:
            import flowio
            text = flowio.FlowData(str(fcs_path), only_text=True).text
            pnl = {text.get(f'p{i}n'): (text.get(f'p{i}l') or '').strip()
                   for i in range(1, int(text['par']) + 1)}
            out = [x if x is not None else (pnl.get(d) or None) for x, d in zip(out, detectors)]
        except Exception as exc:
            logger.info('Noise model: $PnL not read from %s (%s)', fcs_path, exc)
    return [x if x is not None else _name_laser(d) for x, d in zip(out, detectors)]


def scatter_columns_for(controller, layout: dict) -> tuple | None:
    """Positions of FSC-A, SSC-A and FSC-H in the scatter columns ``load_events`` returns (``layout['sc_ids']``)."""
    from honeychrome.controller_components.particle_type import scatter_channel_pair

    raw = controller.experiment.settings['raw']
    pnn = raw.get('whitelisted_pnn') or raw['event_channels_pnn']
    names = [pnn[i] for i in layout['sc_ids']]
    pair = scatter_channel_pair(raw, controller.experiment.cytometry.get('raw_plots'))
    if pair is None:
        return None
    return ns.scatter_columns(names, pair[0], pair[1])


def noise_setup_snapshot(controller, settings: NoiseSetupSettings, unstained_path: str, gate: str) -> dict:
    """
    Everything ``fit_experiment_noise_model`` needs, read on the main
    thread. ``unstained_path`` is a sample key or ``AUTO``; ``settings.
    bead_mode`` is ``'auto'``, ``'none'`` or a bead sample key; ``gate``
    is used with ``settings.selection = 'gates'`` only.
    """
    from honeychrome.controller_components.functions import resolve_base_gate
    from honeychrome.controller_components.per_sample_gating_functions import SampleGating
    from honeychrome.controller_components.spread_model import control_records, raw_layout

    names, spectra = fluor_names_and_spectra(controller)
    if spectra is None:
        raise ValueError('No fluorophore spectra: complete the spectral model first.')
    process = controller.experiment.process
    samples = controller.experiment.samples
    layout = raw_layout(controller)
    base_gate, _ = resolve_base_gate(controller.raw_gating, process.get('base_gate_priority_order', []))
    controls = control_records(controller, names, layout['detectors'])
    if not controls:
        raise ValueError('No single-stain control in the spectral model has a sample.')
    if unstained_path == AUTO:
        candidates = [p for p, _n in unstained_cell_samples(samples)]
    else:
        candidates = [unstained_path]
    stored, sidecars = {}, {}
    for path in candidates + [r['neg_path'] for r in controls if r['is_cell'] and r['neg_path']]:
        if path not in stored:
            stored[path], sidecars[path] = stored_af_for(controller, path, layout['detectors'])
    if unstained_path == AUTO:
        candidates = [p for p in candidates if stored.get(p) is not None]
        if not candidates:
            raise ValueError('No unstained cell sample has a stored AF profile: extract one on the '
                             'Autofluorescence page, or choose the unstained sample.')
    folder = Path(controller.experiment_dir) / AF_SUBDIR
    raw_settings = controller.experiment.settings['raw']
    lasers = detector_lasers(raw_settings, layout['detectors'], Path(controller.experiment_dir) / controls[0]['path'])
    return {
        'experiment_dir': str(controller.experiment_dir), 'unstained_candidates': candidates,
        'gate': gate, 'base_gate': base_gate, 'layout': layout,
        'gating': SampleGating.from_controller(controller, scope='raw'),
        'fluor_names': names, 'spectra': spectra, 'settings': settings, 'controls': controls,
        'bead_unstained': [p for p, _n in unstained_bead_samples(samples)],
        'stored_af': stored, 'stored_sidecar': sidecars, 'variant_cache': str(folder / VARIANT_FILE),
        'scatter_columns': scatter_columns_for(controller, layout), 'lasers': lasers,
    }


def _variant_fingerprint(snap: dict, unstained_path: str) -> str:
    s = snap['settings']
    return fingerprint(snap['spectra'], {
        'controls': [[r['label'], r['path'], r['neg_path'], r['peak'], r['is_cell']] for r in snap['controls']],
        'unstained': unstained_path, 'gate': snap['gate'], 'base_gate': snap['base_gate'],
        'detectors': snap['layout']['detectors'],
        'split': [s.control_train_frac, s.unstained_train_frac, s.seed, s.max_control_events, s.max_unstained],
        'beads': [s.bead_mode, s.bead_factor], 'variant': asdict(s.variant), 'selection': s.selection})


# ---------------------------------------------------------------------------
# Noise model fit (worker)
# ---------------------------------------------------------------------------

@dataclass
class NoiseFit:
    """A fitted noise model with what it was fitted from."""
    model: nm.NoiseModel
    calibration: dict
    unstained_path: str
    gate: str
    af_source: str
    bead_sources: list        # [{'name', 'source'}]: a sample key or an internal negative's name
    cell_null_var: np.ndarray
    bead_null_var: np.ndarray | None
    n_controls: int
    notes: list
    spectra_fingerprint: str
    cell_negatives: np.ndarray | None = None
    bead_negatives: np.ndarray | None = None
    clamped: bool = False
    flat_phi: float | None = None
    clamped_weights: dict = field(default_factory=dict)  # {'kind', 'floor', 'scale'}: the weights for clamped data
    selection: dict = field(default_factory=dict)     # {'mode', 'counts': {path: counts}, 'sources': {group: path}}
    options: dict = field(default_factory=dict)       # the design and estimator settings used


def _load(snap: dict, loader, path: str, gate: str) -> tuple:
    return loader(snap['experiment_dir'], path, gate, snap['layout'], snap['gating'])


def _selecting_loader(loader, columns: tuple | None, counts: dict):
    """``loader`` at root with the automatic selection applied (gates ignored); counts per path into ``counts``."""
    def load(experiment_dir, path, _gate, layout, gating):
        fl, sc = loader(experiment_dir, path, None, layout, gating)
        if columns is None or len(fl) == 0:
            counts[path] = {'n_root': int(len(fl)), 'n_scatter': int(len(fl)), 'n_singlet': int(len(fl))}
            return fl, sc
        mask, counts[path] = ns.select_events(sc, columns)
        return fl[mask], sc[mask]
    return load


def _where(snap: dict) -> str:
    return 'the automatic selection' if snap['settings'].selection == 'auto' else f'gate "{snap["gate"]}"'


def _load_first(snap: dict, loader, path: str, gates) -> tuple | None:
    for gate in gates:
        try:
            fl, sc = _load(snap, loader, path, gate)
        except Exception as exc:
            logger.warning('Noise model: %s not loaded in gate "%s" (%s)', path, gate, exc)
            continue
        if len(fl) > 0:
            return fl, sc
    return None


def _threshold_entry(neg_raw: np.ndarray, snap: dict, is_cell: bool, af_spectra: np.ndarray | None) -> dict:
    _bundled()
    from autospectral_optimization_functions import _unstained_threshold_entry
    return _unstained_threshold_entry(neg_raw, snap['spectra'], snap['fluor_names'], is_cell, 4,
                                      random_state=snap['settings'].seed, af_spectra=af_spectra)


def _ols(raw: np.ndarray, spectra: np.ndarray) -> np.ndarray:
    return raw @ np.linalg.solve(spectra @ spectra.T, spectra).T


def _choose_unstained(snap: dict, loader, progress) -> tuple:
    """
    ``(path, fl, sc)`` of the unstained cell sample: the only candidate, or
    the one with the lowest mean raw signal (summed over the detectors), read
    from its AF profile's stored unstained subsample when it has one.
    """
    candidates = snap['unstained_candidates']
    if len(candidates) > 1:
        levels = {}
        for path in candidates:
            sidecar = (snap.get('stored_sidecar') or {}).get(path)
            if sidecar:
                levels[path] = float(np.mean(np.load(sidecar).sum(axis=1)))
                continue
            progress(f'Mean signal of {path}')
            fl, _sc = _load(snap, loader, path, snap['gate'])
            if len(fl) >= 500:
                levels[path] = float(np.mean(fl.sum(axis=1)))
        if not levels:
            raise ValueError(f'No unstained cell sample has 500 events in {_where(snap)}.')
        candidates = [min(levels, key=levels.get)]
    path = candidates[0]
    progress(f'Loading the unstained sample {path}')
    fl, sc = _load(snap, loader, path, snap['gate'])
    if len(fl) < 500:
        raise ValueError(f'Only {len(fl)} unstained events in {_where(snap)}.')
    return path, fl, sc


def fit_experiment_noise_model(snap: dict, progress, loader=None) -> NoiseFit:
    """Worker body of Model Noise (see the module docstring). ``loader`` is ``spread_model.load_events`` unless given."""
    _bundled()
    from autospectral_optimization_functions import discover_fluor_variants
    from honeychrome.controller_components.spread_model import load_events

    loader = loader or load_events
    s: NoiseSetupSettings = snap['settings']
    spectra, names = snap['spectra'], snap['fluor_names']
    layout = snap['layout']
    ceiling = layout['ceiling']
    n_sc = layout['n_scatter']
    notes = []
    counts, group_paths = {}, {}
    if s.selection == 'auto':
        if snap.get('scatter_columns') is None:
            notes.append('No FSC-A and SSC-A: every event of each sample is used.')
        loader = _selecting_loader(loader, snap.get('scatter_columns'), counts)
    lasers = snap.get('lasers')
    if (s.laser_split or s.noise.sigma0_fallback == 'laser') and (lasers is None or any(x is None for x in lasers)):
        notes.append('The laser of some detectors is unknown: no per-laser design rows or same-laser fallback.')
        lasers = None

    def bead_rows(neg_fl):
        """AF rows of a bead negative's training split (``bead_af_rows``), or None."""
        if s.bead_af_rows <= 0 or neg_fl is None or len(neg_fl) < 10:
            return None
        tr, _te = holdout_split(len(neg_fl), s.control_train_frac, s.seed + 7)
        return af_basis(neg_fl[tr], s.bead_af_rows, seed=s.seed)

    unstained_path, fl, sc = _choose_unstained(snap, loader, progress)
    keep = subsample(len(fl), s.max_unstained, s.seed)
    fl, sc = fl[keep], sc[keep]
    train, test = holdout_split(len(fl), s.unstained_train_frac, s.seed, s.max_test_unstained)
    u_train, u_test = fl[train], fl[test]
    setup_af = snap['stored_af'].get(unstained_path)
    af_source = 'stored AF profile' if setup_af is not None else 'AF extracted for the thresholds'
    setup_basis = af_basis(u_train, s.n_af_rows, seed=s.seed)
    bases_af = {unstained_path: setup_basis}
    neg_cache = {unstained_path: (u_train, sc[train])}
    thr_cache = {}

    def threshold(key, is_cell, af_spectra):
        tk = (key, is_cell)
        if tk not in thr_cache:
            thr_cache[tk] = _threshold_entry(neg_cache[key][0], snap, is_cell, af_spectra if is_cell else None)
        return thr_cache[tk]

    records = list(enumerate(snap['controls']))
    bead_recs = [(n, r) for n, r in records if not r['is_cell']]

    # Bead negatives: anchors (sigma0) and each bead control's own negative.
    gates = ('Neg Unstained', snap['base_gate'], 'root')
    by_path = {}

    def bead_neg(path, source):
        if path not in by_path:
            progress(f'Bead negative: {path}')
            got = _load_first(snap, loader, path, gates)
            by_path[path] = None if got is None else (path, source, got[0], got[1])
        return by_path[path]

    anchors, bead_own = [], {}
    if s.bead_mode == 'auto':
        for path in snap.get('bead_unstained', []):
            got = bead_neg(path, 'unstained bead sample')
            if got is not None:
                anchors.append(got)
    elif s.bead_mode != 'none':
        got = bead_neg(s.bead_mode, 'chosen bead sample')
        if got is not None:
            anchors.append(got)
    for _n, rec in bead_recs:
        if not rec['neg_path']:
            continue
        got = bead_neg(rec['neg_path'], f'universal negative of {rec["label"]}')
        if got is None:
            continue
        bead_own[rec['label']] = (got[2], got[3], got[1])
        if s.bead_mode == 'auto' and all(a[0] != got[0] for a in anchors):
            anchors.append(got)

    def load_control(n, rec):
        got = _load_first(snap, loader, rec['path'], (snap['base_gate'],))
        if got is None:
            return None
        cfl, csc = got
        keep = subsample(len(cfl), s.max_control_events, s.seed + n)
        return cfl[keep], csc[keep]

    # Dark levels of every bead control, for the internal negatives of those without their own.
    bead_medians = []
    if any(rec['label'] not in bead_own for _n, rec in bead_recs):
        for n, rec in bead_recs:
            progress(f'Bead levels: {rec["label"]}')
            got = load_control(n, rec)
            if got is not None:
                bead_medians.append((rec['label'], np.median(got[0], axis=0), rec['fluor_idx']))

    variant_fp = _variant_fingerprint(snap, unstained_path)
    cached = load_variants(snap['variant_cache'], variant_fp) if snap.get('variant_cache') else None
    variants = dict(cached or {})
    discovered = cached is None
    vs = s.variant
    kept = []          # (label, fluor_idx, is_cell, neg_key, background, bead AF rows, fit, cal) in control order
    internal = []
    deferred = []

    def process(n, rec, cfl, csc, neg_fl, neg_sc, neg_key, background, raw_thr, entry, b_rows=None):
        tr, te = holdout_split(len(cfl), s.control_train_frac, s.seed + 1000 + n)
        pos_tr = positive_rows(cfl[tr], rec['peak'], raw_thr, ceiling)
        pos_te = positive_rows(cfl[te], rec['peak'], raw_thr, ceiling)
        if len(pos_tr) < 100 or len(pos_te) < 50:
            notes.append(f'{rec["label"]}: too few positives ({len(pos_tr)} training, {len(pos_te)} held out)')
            return
        test_rows = te[pos_te][subsample(len(pos_te), s.max_test_control, s.seed + n)]
        if discovered:
            progress(f'Variants (training split): {rec["label"]}')
            nfl, nsc = (neg_fl, neg_sc) if neg_sc is not None else (None, None)
            out = discover_fluor_variants(
                label=rec['label'], pos_events_raw=cfl[tr], pos_scatter=csc[tr][:, :n_sc], neg_events_raw=nfl,
                neg_scatter=None if nsc is None else nsc[:, :n_sc], is_cell_control=rec['is_cell'],
                reference_spectra=spectra, fluor_names=names, fluor_idx=rec['fluor_idx'], peak_ch_idx=rec['peak'],
                raw_pos_threshold=raw_thr, unmixed_pos_threshold=float(entry['unmixed'][rec['fluor_idx']]),
                af_pcs=entry['af_pcs'] if rec['is_cell'] else None, saturation_ceiling=ceiling,
                n_cells=vs.n_cells, som_dim=vs.som_dim, k_neighbors=vs.k_neighbors,
                sim_threshold=vs.sim_threshold, sim_threshold_floor=vs.sim_threshold_floor,
                af_collinear_threshold=vs.af_collinear_threshold, dedup_threshold=vs.dedup_threshold,
                max_variants=vs.max_variants, random_state=s.seed)
            if out is not None:
                variants[rec['label']] = out
        k = len(kept)
        train_pos = cfl[tr][pos_tr]
        group_paths[rec['label']] = rec['path']
        kept.append((rec['label'], rec['fluor_idx'], rec['is_cell'], neg_key, background, b_rows,
                     train_pos[subsample(len(train_pos), s.max_fit_events, s.seed + k)],
                     cfl[test_rows][subsample(len(test_rows), s.max_cal_events, s.seed + k)]))

    for n, rec in records:
        progress(f'Controls: {rec["label"]} ({n + 1}/{len(records)})')
        got = load_control(n, rec)
        if got is None:
            notes.append(f'{rec["label"]}: no events in "{snap["base_gate"]}"')
            continue
        cfl, csc = got
        neg_fl = neg_sc = background = None
        neg_key = None
        if rec['is_cell']:
            neg_key = unstained_path
            if rec['neg_path'] and rec['neg_path'] != unstained_path:
                if rec['neg_path'] not in neg_cache:
                    neg_cache[rec['neg_path']] = _load_first(snap, loader, rec['neg_path'],
                                                             ('Neg Unstained', snap['base_gate']))
                    if neg_cache[rec['neg_path']] is not None:
                        bases_af[rec['neg_path']] = af_basis(neg_cache[rec['neg_path']][0], s.n_af_rows,
                                                             seed=s.seed)
                if neg_cache[rec['neg_path']] is not None:
                    neg_key = rec['neg_path']
            neg_fl, neg_sc = neg_cache[neg_key]
            raw_thr = float(np.percentile(neg_fl[subsample(len(neg_fl), 100_000, 0)][:, rec['peak']], 99.5))
            af_for = setup_af if neg_key == unstained_path else snap['stored_af'].get(neg_key)
            entry = threshold(neg_key, True, af_for)
            if not rec['neg_path']:
                neg_fl = neg_sc = None
        else:
            own = bead_own.get(rec['label'])
            if own is None:
                others = [(med, f) for _lbl, med, f in bead_medians if f != rec['fluor_idx']]
                level = bead_reference_levels(others, spectra)[rec['peak']] if others else np.nan
                mask, reason = internal_bead_negative(cfl, rec['peak'], level, factor=s.bead_factor)
                if mask is not None:
                    source = f'internal negative of {rec["label"]} ({reason})'
                    own = (cfl[mask], csc[mask], source)
                    internal.append((f'{rec["label"]} negative', source, cfl[mask], csc[mask]))
                    group_paths[f'{rec["label"]} negative'] = rec['path']
                else:
                    logger.info('Noise model: no internal bead negative in %s (%s)', rec['label'], reason)
            if own is None:
                deferred.append((n, rec, cfl, csc))
                continue
            neg_fl, neg_sc, _source = own
            background = neg_fl.mean(axis=0)
            raw_thr = float(np.percentile(neg_fl[:, rec['peak']], 99.5))
            key = f'beads::{rec["label"]}'
            neg_cache[key] = (neg_fl, neg_sc)
            entry = threshold(key, False, None)
        process(n, rec, cfl, csc, neg_fl, neg_sc, neg_key, background, raw_thr, entry,
                None if rec['is_cell'] else bead_rows(neg_fl))

    if s.bead_mode == 'auto' and not anchors:
        anchors = internal
    for n, rec, cfl, csc in deferred:
        if anchors:
            neg_fl = np.vstack([a[2] for a in anchors])
            neg_sc = np.vstack([a[3] for a in anchors])
            background = neg_fl.mean(axis=0)
            raw_thr = float(np.percentile(neg_fl[:, rec['peak']], 99.5))
            key = f'beads::{rec["label"]}'
            neg_cache[key] = (neg_fl, neg_sc)
            entry = threshold(key, False, None)
        else:
            neg_fl = neg_sc = background = None
            p = np.log10(np.maximum(cfl[:, rec['peak']], 1.0))
            raw_thr = float(10 ** otsu_threshold(p))
            entry = threshold(unstained_path, False, None)
            notes.append(f'{rec["label"]}: no bead negative; positives above the Otsu split')
        process(n, rec, cfl, csc, neg_fl, neg_sc, None, background, raw_thr, entry, bead_rows(neg_fl))
    deferred.clear()

    if not kept:
        raise ValueError('No single-stain control has enough positive events.')
    if discovered and snap.get('variant_cache'):
        try:
            save_variants(variants, snap['variant_cache'], variant_fp)
        except OSError as exc:
            logger.warning('Noise model: variant cache not saved (%s)', exc)

    bases = {}
    for label, entry in variants.items():
        vb = variant_basis(label, entry['delta'], s.basis_var_explained, s.basis_max_rank)
        if vb is not None:
            bases[label] = vb

    split_lasers = lasers if s.laser_split else None
    fit_groups, cal_groups = [], []
    for label, f, is_cell, neg_key, background, b_rows, fit_ev, cal_ev in kept:
        af_rows = bases_af.get(neg_key) if is_cell else b_rows
        design = control_design(spectra[f], bases.get(label), af_rows, lasers=split_lasers,
                                split_min=s.laser_split_min)
        offset = None if b_rows is not None else background
        cls = 'cells' if is_cell else 'beads'
        fit_groups.append(nm.Group(label, 'control', fit_ev, design, offset, cls=cls))
        cal_groups.append(nm.Group(label, 'control', cal_ev, design, offset, cls=cls))
    fit_groups.append(nm.Group('Unstained', 'unstained',
                               u_train[subsample(len(u_train), 4 * s.max_fit_events, s.seed)], setup_basis))
    cal_groups.append(nm.Group('Unstained', 'unstained',
                               u_test[subsample(len(u_test), 2 * s.max_cal_events, s.seed)], setup_basis))
    group_paths['Unstained'] = unstained_path
    bead_test = []
    for name, _source, bfl, _bsc in anchors:
        tr, te = holdout_split(len(bfl), s.control_train_frac, s.seed + 7)
        design = bead_rows(bfl)
        fit_groups.append(nm.Group(name, 'beads', bfl[tr][subsample(len(tr), s.max_fit_events, s.seed)], design,
                                   cls='beads'))
        cal_groups.append(nm.Group(name, 'beads', bfl[te][subsample(len(te), s.max_cal_events, s.seed)], design,
                                   cls='beads'))
        group_paths.setdefault(name, name)
        bead_test.append(bfl[te])

    clamped = False
    if s.clamp_mode == 'auto':
        clamped, _share = nm.negatives_clamped([fl] + [a[2] for a in anchors])
        if clamped:
            notes.append('Clamped data: the negatives have no values at or below zero on any detector, so their '
                         'noise is truncated at zero. The experiment is unmixed with '
                         f'{clamped_weights_text(s.clamp_weights)} and the positivity prior unless the noise model '
                         'is ticked.')
    u_proj = nm.project(nm.Group('Unstained', 'unstained', u_test, setup_basis))
    flat_phi = float(np.median(nm.robust_var(u_proj.resid)))

    progress('Fitting the noise model')
    model = nm.fit_noise_model(fit_groups, layout['detectors'], s.noise, lasers=lasers)
    floor = float(s.clamp_floor_pe * np.median(model.g))
    scale = nm.dispersion_scale(nm.poisson_model(layout['detectors'], model.g, floor), u_proj)
    clamped_weights = {'kind': s.clamp_weights, 'floor': floor, 'scale': scale}
    del u_proj
    progress('Calibrating the noise model on held-out events')
    calibration = nm.calibrate(model, nm.calibration_sets(cal_groups))
    options = {'selection': s.selection, 'bead_af_rows': int(s.bead_af_rows), 'laser_split': bool(s.laser_split),
               'laser_split_min': float(s.laser_split_min), 'clamp_mode': s.clamp_mode,
               'clamp_weights': s.clamp_weights, 'clamp_floor_pe': float(s.clamp_floor_pe),
               'per_control_quadratic': bool(s.noise.per_control_quadratic),
               'sigma0_by_class': bool(s.noise.sigma0_by_class), 'sigma0_fallback': s.noise.sigma0_fallback}
    model.fingerprint = nm.noise_fingerprint(spectra, layout['detectors'], [variant_fp],
                                             {'basis': [s.basis_var_explained, s.basis_max_rank, s.n_af_rows],
                                              'fit_events': s.max_fit_events, 'noise': asdict(s.noise),
                                              'options': options})
    bead_var = robust_sd(_ols(np.vstack(bead_test), spectra)) ** 2 if bead_test else None
    bead_all = np.vstack(bead_test) if bead_test else None
    return NoiseFit(
        model=model, calibration=calibration, unstained_path=unstained_path,
        gate=snap['gate'] if s.selection == 'gates' else None,
        af_source=af_source, bead_sources=[{'name': a[0], 'source': a[1]} for a in anchors],
        cell_null_var=robust_sd(_ols(u_test, spectra)) ** 2, bead_null_var=bead_var,
        n_controls=len(kept), notes=notes, spectra_fingerprint=spectra_fingerprint(names, spectra),
        cell_negatives=u_test[subsample(len(u_test), NEGATIVE_SAMPLE_EVENTS, s.seed)].astype(np.float32),
        bead_negatives=None if bead_all is None else
        bead_all[subsample(len(bead_all), NEGATIVE_SAMPLE_EVENTS, s.seed)].astype(np.float32),
        clamped=bool(clamped), flat_phi=flat_phi, clamped_weights=clamped_weights,
        selection={'mode': s.selection, 'counts': counts, 'sources': group_paths}, options=options)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def spectra_fingerprint(names: list, spectra: np.ndarray) -> str:
    return fingerprint(np.round(np.asarray(spectra, dtype=np.float64), 8), {'names': list(names)})


def calibration_summary(calibration: dict) -> dict:
    """The JSON-safe part of ``noise_model.calibrate``'s result."""
    def num(v):
        return None if v is None or not np.isfinite(v) else float(v)
    return {
        'band': [float(b) for b in calibration.get('band', (0.8, 1.25))],
        'in_band': num(calibration.get('in_band')),
        'in_band_controls': num(calibration.get('in_band_controls')),
        'per_control_c': bool(calibration.get('per_control_c')),
        'groups': [{k: (float(v) if isinstance(v, (float, np.floating)) else v)
                    for k, v in g.items() if k != 'per_detector'} for g in calibration.get('groups', [])],
    }


def store_noise_fit(process: dict, experiment_dir, fit: NoiseFit, prior_lambda: float | None = None) -> None:
    """
    Save the model's file and describe the fit in ``process``. The noise
    model becomes the one in use, or, for clamped data, the weights for
    clamped data (``clamped_model``).
    """
    nm.save_noise_model(fit.model, Path(experiment_dir) / AF_SUBDIR / NOISE_FILE)
    negatives_file = None
    if fit.cell_negatives is not None:
        arrays = {'cell': fit.cell_negatives, 'fingerprint': np.array(fit.model.fingerprint)}
        if fit.bead_negatives is not None:
            arrays['bead'] = fit.bead_negatives
        np.savez(Path(experiment_dir) / AF_SUBDIR / NEGATIVES_FILE, **arrays)
        negatives_file = NEGATIVES_FILE
    process[PROCESS_KEY] = {
        'file': NOISE_FILE, 'fingerprint': fit.model.fingerprint, 'detectors': list(fit.model.detectors),
        'spectra_fingerprint': fit.spectra_fingerprint, 'unstained': fit.unstained_path, 'gate': fit.gate,
        'af_source': fit.af_source, 'beads': list(fit.bead_sources), 'n_controls': int(fit.n_controls),
        'cell_null_var': [float(v) for v in fit.cell_null_var],
        'bead_null_var': None if fit.bead_null_var is None else [float(v) for v in fit.bead_null_var],
        'calibration': calibration_summary(fit.calibration), 'notes': list(fit.notes),
        'flagged_detectors': int(np.count_nonzero(fit.model.flags)),
        'fitted': time.strftime('%Y-%m-%d %H:%M'), 'negatives_file': negatives_file,
        'clamped': bool(fit.clamped), 'flat_phi': None if fit.flat_phi is None else float(fit.flat_phi),
        'clamped_weights': dict(fit.clamped_weights),
        'selection': fit.selection, 'options': dict(fit.options), 'version': nm.NOISE_MODEL_VERSION,
    }
    unmixing = process.setdefault(UNMIXING_KEY, {})
    unmixing['noise_model'] = not fit.clamped
    if fit.clamped:
        unmixing['weights'] = CLAMPED_WEIGHTS
    else:
        unmixing.pop('weights', None)
    unmixing['prior_lambda'] = float(prior_lambda if prior_lambda is not None
                                     else unmixing.get('prior_lambda', DEFAULT_PRIOR_LAMBDA))


@dataclass
class NoiseState:
    """A stored noise model ready for unmixing."""
    model: nm.NoiseModel
    entry: dict
    cell_null_var: np.ndarray
    bead_null_var: np.ndarray | None
    cell_negatives: np.ndarray | None = None
    bead_negatives: np.ndarray | None = None
    clamped_model: nm.NoiseModel | None = None

    def model_for(self, is_bead: bool) -> nm.NoiseModel:
        """The noise model of a sample's particle class."""
        return self.model.for_class('beads' if is_bead else 'cells')


def _load_negatives(experiment_dir, entry: dict) -> tuple:
    """The stored held-out negatives (cells, beads) of a fit, or Nones."""
    name = entry.get('negatives_file')
    if not name:
        return None, None
    try:
        with np.load(Path(experiment_dir) / AF_SUBDIR / name, allow_pickle=False) as data:
            if str(data['fingerprint']) != entry.get('fingerprint'):
                return None, None
            cell = np.asarray(data['cell'], dtype=np.float64)
            bead = np.asarray(data['bead'], dtype=np.float64) if 'bead' in data.files else None
        return cell, bead
    except Exception as exc:
        logger.warning('Noise model: held-out negatives not read (%s)', exc)
        return None, None


def null_variances(state: 'NoiseState', base_spectra: np.ndarray, spectra: np.ndarray) -> tuple:
    """
    The no-AF negative variances (cells, beads) under ``spectra``: the
    stored ones moved by the change the held-out negatives show between the
    base spectra and ``spectra``; the stored ones when the fit kept no
    negatives.
    """
    out = []
    for stored, events in ((state.cell_null_var, state.cell_negatives), (state.bead_null_var, state.bead_negatives)):
        if stored is None or events is None or np.array_equal(spectra, base_spectra):
            out.append(stored)
            continue
        change = robust_sd(_ols(events, spectra)) ** 2 - robust_sd(_ols(events, base_spectra)) ** 2
        out.append(np.maximum(stored + change, 0.0))
    return tuple(out)


def load_noise_state(process: dict, experiment_dir, detectors: list) -> NoiseState | None:
    """The stored model when its file reads and it covers exactly ``detectors``, else None."""
    entry = process.get(PROCESS_KEY)
    if not entry:
        return None
    model = nm.load_noise_model(Path(experiment_dir) / AF_SUBDIR / entry.get('file', NOISE_FILE),
                                entry.get('fingerprint', ''))
    if model is None:
        return None
    if list(model.detectors) != list(detectors):
        logger.warning('Noise model: fitted on different detectors from the current ones; fit it again.')
        return None
    bead = entry.get('bead_null_var')
    cell_neg, bead_neg = _load_negatives(experiment_dir, entry)
    return NoiseState(model, entry, np.asarray(entry['cell_null_var'], dtype=np.float64),
                      None if bead is None else np.asarray(bead, dtype=np.float64), cell_neg, bead_neg,
                      clamped_model(entry, model))


def clamped_model(entry: dict, model: nm.NoiseModel) -> nm.NoiseModel | None:
    """The weights for clamped data stored with a fit (see the module docstring), or None."""
    weights = entry.get('clamped_weights') or {}
    if weights.get('kind') == 'poisson' and weights.get('scale'):
        return nm.poisson_model(model.detectors, model.g, weights['floor'], weights['scale'])
    if weights.get('kind') == 'flat' and entry.get('flat_phi'):
        return nm.flat_model(model.detectors, entry['flat_phi'])
    return None


def clamped_weights_text(kind: str | None) -> str:
    return 'flat weights (M0p)' if kind == 'flat' else 'Poisson weights with a floor (M1p)'


def spectra_changed(entry: dict, names: list, spectra: np.ndarray) -> bool:
    """Whether the spectra have changed since the model was fitted."""
    return entry.get('spectra_fingerprint') != spectra_fingerprint(names, spectra)


def prior_lambda(process: dict) -> float:
    return float((process.get(UNMIXING_KEY) or {}).get('prior_lambda', DEFAULT_PRIOR_LAMBDA))


def clamped_weights_requested(process: dict) -> bool:
    """The weights for clamped data (with the positivity prior) asked for."""
    return (process.get(UNMIXING_KEY) or {}).get('weights') == CLAMPED_WEIGHTS


def noise_in_use(process: dict) -> bool:
    """The Noise page's unmixing (the noise model, or the weights for clamped data) is in use."""
    unmixing = process.get(UNMIXING_KEY) or {}
    return bool(process.get(PROCESS_KEY)) and (bool(unmixing.get('noise_model'))
                                               or clamped_weights_requested(process))


# ---------------------------------------------------------------------------
# M2pg
# ---------------------------------------------------------------------------

def sample_af_weights(raw: np.ndarray, noise: nm.NoiseModel) -> np.ndarray:
    """One weight per detector for a sample's AF choice: ``1 / V(median raw signal)``."""
    return 1.0 / noise.variance(np.maximum(np.median(raw, axis=0), 0.0))


def _saturation_weights(w: np.ndarray, raw: np.ndarray, ceiling: float | None) -> np.ndarray:
    """Weights of detectors at the ceiling cut to 1e-6 of the event's median weight."""
    if not ceiling:
        return w
    sat = raw >= 0.999 * ceiling
    if not sat.any():
        return w
    small = 1e-6 * np.median(w, axis=1, keepdims=True)
    return np.where(sat, small, w)


def per_cell_af_whitened(raw: np.ndarray, spectra: np.ndarray, af_spectra: np.ndarray, envelope: dict | None,
                         w0: np.ndarray) -> dict:
    """
    The main app's per-cell AF (``apply_af_unmixing``) on events, spectra and
    AF rows scaled by ``sqrt(w0)`` per detector, so the AF row is chosen in
    the geometry of a weighted fit; the abundances keep their units. Events
    with detectors at the envelope's saturation level are re-fitted on their
    unsaturated detectors, grouped by pattern. ``af_idx`` is 0-based.
    """
    from honeychrome.controller_components.autospectral_functions import (
        apply_af_unmixing, build_af_precomputed, fluorophores_identifiable,
    )

    sw = np.sqrt(np.maximum(w0, 0.0))
    s_w, a_w = spectra * sw, af_spectra * sw
    y = np.ascontiguousarray(raw * sw, dtype=np.float64)
    out = apply_af_unmixing(y, build_af_precomputed(s_w, a_w, envelope), a_w, handle_saturation=False)
    x, alpha = out['unmixed'], out['af_scale']
    idx = np.asarray(out['af_idx'], dtype=np.int64) - 1
    if envelope is not None:
        sat = raw >= envelope['saturated_frac'] * envelope['ceiling']
        rows_any = np.flatnonzero(sat.any(axis=1))
        if rows_any.size:
            patterns, inverse = np.unique(sat[rows_any], axis=0, return_inverse=True)
            inverse = np.asarray(inverse).ravel()
            for k, pattern in enumerate(patterns):
                keep = ~pattern
                if not fluorophores_identifiable(s_w, keep):
                    continue
                rows = rows_any[inverse == k]
                s_k, a_k = np.ascontiguousarray(s_w[:, keep]), np.ascontiguousarray(a_w[:, keep])
                sub = apply_af_unmixing(np.ascontiguousarray(y[rows][:, keep]),
                                        build_af_precomputed(s_k, a_k, envelope), a_k, handle_saturation=False)
                x[rows], alpha[rows] = sub['unmixed'], sub['af_scale']
                idx[rows] = np.asarray(sub['af_idx'], dtype=np.int64) - 1
    return {'unmixed': x, 'af_idx': idx, 'af_scale': alpha}


def negative_sd(raw: np.ndarray, spectra: np.ndarray, af_spectra: np.ndarray | None, envelope: dict | None,
                null_var_no_af: np.ndarray | None, spread: np.ndarray | None) -> np.ndarray | None:
    """
    SD of the negative per event and channel: with an AF library and its
    envelope, the null variance at the mean AF row's abundance plus the
    spillover spread at the mean-AF position; without AF, ``null_var_no_af``
    plus the spread at the OLS position. None without either.
    """
    if af_spectra is not None and envelope is not None:
        from honeychrome.controller_components.autospectral_functions import mean_af_position

        position, k0 = mean_af_position(raw, spectra, af_spectra)
        c0, c1, c2 = np.asarray(envelope['null_var'], dtype=np.float64)[0]
        k = k0[:, np.newaxis]
        var = c0 + k * (c1 + k * c2)
    elif af_spectra is None and null_var_no_af is not None:
        position = _ols(raw, spectra)
        var = np.broadcast_to(null_var_no_af, position.shape).copy()
    else:
        return None
    if spread is not None and np.any(spread):
        var = var + np.maximum(position, 0.0) @ spread
    return np.sqrt(var)


def pgls_solve(y: np.ndarray, spectra: np.ndarray, af_rows: np.ndarray | None, w: np.ndarray,
               fl_pen: np.ndarray | None = None) -> dict:
    """
    One penalised GLS solve per event against ``[spectra; af_rows[i]]`` with
    diagonal weights ``w`` (B, D) and the ridge ``fl_pen`` (B, F) on the
    fluorophores. Events whose AF abundance comes out negative are refitted
    without the AF row. Returns ``x`` (B, F), ``alpha`` (B,) and
    ``inv_diag`` (B, F), the fluorophore diagonal of the inverse normal
    matrix.
    """
    b, d = y.shape
    f = spectra.shape[0]
    blocks = [np.broadcast_to(spectra, (b, f, d))]
    has_af = af_rows is not None
    if has_af:
        blocks.append(af_rows[:, np.newaxis, :])
    design = np.concatenate(blocks, axis=1)
    dw = design * w[:, np.newaxis, :]
    normal = dw @ design.transpose(0, 2, 1)
    if fl_pen is not None:
        normal[:, np.arange(f), np.arange(f)] += fl_pen
    diag_f = normal[:, np.arange(f), np.arange(f)]
    normal[:, np.arange(f), np.arange(f)] += 1e-12 * diag_f.sum(axis=1, keepdims=True) / f
    rhs = np.einsum('bkd,bd->bk', dw, y)
    inv = np.linalg.inv(normal)
    theta = np.einsum('bkl,bl->bk', inv, rhs)
    out = {'x': theta[:, :f].copy(), 'alpha': theta[:, f].copy() if has_af else np.zeros(b),
           'inv_diag': np.diagonal(inv, axis1=1, axis2=2)[:, :f].copy()}
    if has_af:
        neg = out['alpha'] < 0
        if neg.any():
            sub = pgls_solve(y[neg], spectra, None, w[neg], None if fl_pen is None else fl_pen[neg])
            for key in ('x', 'alpha', 'inv_diag'):
                out[key][neg] = sub[key]
    return out


def _m2p_refit(raw: np.ndarray, spectra: np.ndarray, start: dict, af_spectra: np.ndarray | None,
               noise: nm.NoiseModel, tau: np.ndarray | None, lam: float, n_iter: int,
               ceiling: float | None) -> dict:
    """The plug-in penalised GLS iterations of M2pg from ``start`` (see the module docstring)."""
    from honeychrome.controller_components.autospectral_functions import positivity_weight

    n, f = len(raw), spectra.shape[0]
    x_out = np.empty((n, f))
    a_out = np.zeros(n)
    idx_out = np.asarray(start['af_idx']).copy()
    has_af = af_spectra is not None and np.any(idx_out >= 0)
    lam = lam if tau is not None else 0.0
    chunk = int(max(64, min(2048, 4_000_000 // ((f + 1) * (f + 1)))))
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        y = np.ascontiguousarray(raw[lo:hi], dtype=np.float64)
        x = start['unmixed'][lo:hi].copy()
        alpha = start['af_scale'][lo:hi].copy() if has_af else np.zeros(hi - lo)
        idx = idx_out[lo:hi]
        rows = af_spectra[np.clip(idx, 0, None)] if has_af else None
        t2 = None if tau is None else np.maximum(tau[lo:hi] ** 2, _EPS)
        for _ in range(max(1, int(n_iter))):
            mu = np.maximum(x, 0.0) @ spectra
            if rows is not None:
                mu += np.maximum(alpha, 0.0)[:, np.newaxis] * rows
            w = _saturation_weights(1.0 / noise.variance(mu), y, ceiling)
            pen = None if lam <= 0 else lam * positivity_weight(x, t2) / t2
            sol = pgls_solve(y, spectra, rows, w, pen)
            x, alpha = sol['x'], sol['alpha']
            if pen is not None:
                x = x / np.maximum(1.0 - pen * sol['inv_diag'], 1e-3)
        x_out[lo:hi] = x
        a_out[lo:hi] = alpha
    return {'unmixed': x_out, 'af_idx': idx_out if has_af else np.full(n, -1), 'af_scale': a_out}


def kernel_available() -> bool:
    _bundled()
    try:
        from gls_m2p_kernel_wrapper import GLS_M2P_KERNEL_AVAILABLE
    except ImportError:
        return False
    return bool(GLS_M2P_KERNEL_AVAILABLE)


def unmix_m2pg(raw_fl: np.ndarray, spectra: np.ndarray, noise: nm.NoiseModel, af_spectra: np.ndarray | None = None,
               envelope: dict | None = None, null_var_no_af: np.ndarray | None = None,
               spread: np.ndarray | None = None, prior_lambda: float = DEFAULT_PRIOR_LAMBDA,
               ceiling: float | None = None, n_threads: int = 0, use_kernel: bool = True) -> dict:
    """
    M2pg on raw fluorescence events (see the module docstring). With
    ``af_spectra`` (two or more rows) each event takes one AF row, scored
    with ``envelope`` (the runtime envelope; joint L2 without one, and then
    no prior); without, ``null_var_no_af`` sets the prior's negative
    variance and ``spread`` the spillover spread at the OLS position.
    Returns ``unmixed`` (n, F, uncompensated), ``af_idx`` (0-based, -1
    without AF) and ``af_scale``.
    """
    raw_fl = np.ascontiguousarray(raw_fl, dtype=np.float64)
    spectra = np.ascontiguousarray(spectra, dtype=np.float64)
    has_af = af_spectra is not None and len(af_spectra) > 0
    if has_af:
        af_spectra = np.ascontiguousarray(af_spectra, dtype=np.float64)
        tau_spread = None if envelope is None else envelope.get('spread')
    else:
        af_spectra, envelope, tau_spread = None, None, spread
    if use_kernel and kernel_available():
        from gls_m2p_kernel_wrapper import unmix_m2p_kernel
        out = unmix_m2p_kernel(
            raw_fl, spectra, af_spectra, envelope, noise.sigma0_sq, noise.g, noise.c,
            bead_var=None if has_af else null_var_no_af, tau_spread=tau_spread, prior_lambda=prior_lambda,
            n_iter=N_ITER, ceiling=ceiling or 0.0, n_threads=n_threads,
            af_choice_weights=sample_af_weights(raw_fl, noise) if has_af else None)
        return {'unmixed': out['unmixed'], 'af_idx': out['af_idx'] if has_af else np.full(len(raw_fl), -1),
                'af_scale': out['af_scale']}
    if has_af:
        start = per_cell_af_whitened(raw_fl, spectra, af_spectra, envelope, sample_af_weights(raw_fl, noise))
    else:
        start = {'unmixed': _ols(raw_fl, spectra), 'af_idx': np.full(len(raw_fl), -1),
                 'af_scale': np.zeros(len(raw_fl))}
    spread_m = None if tau_spread is None else np.asarray(tau_spread, dtype=np.float64)
    tau = negative_sd(raw_fl, spectra, af_spectra, envelope, None if has_af else null_var_no_af, spread_m)
    return _m2p_refit(raw_fl, spectra, start, af_spectra, noise, tau, float(prior_lambda), N_ITER, ceiling)


def assemble_unmixed(raw_event_data: np.ndarray, transfer_matrix: np.ndarray, settings: dict, result: dict,
                     spillover=None, af_index_map=None) -> np.ndarray:
    """
    Full unmixed event array with the fluorescence columns from ``result``:
    scatter, time and event id from the transfer matrix, compensation when
    ``spillover`` is set, and the AF Abundance / AF Index channels when
    present (AF Index 0 where no AF row was fitted).
    """
    from honeychrome.settings import af_abundance_channel, af_index_channel

    unmixed_settings = settings['unmixed']
    fl_ids_unmixed = np.array(unmixed_settings['fluorescence_channel_ids'])
    unmixed = raw_event_data @ transfer_matrix
    fl = result['unmixed']
    if spillover is not None:
        fl = (np.linalg.inv(np.array(spillover)).T @ fl.T).T
    unmixed[:, fl_ids_unmixed] = fl
    pnn = unmixed_settings.get('event_channels_pnn') or []
    af_idx = np.asarray(result['af_idx'])
    index = np.where(af_idx >= 0, af_idx + 1, 0)
    if af_index_map is not None:
        lookup = np.asarray(af_index_map)
        index = np.where(af_idx >= 0, lookup[np.clip(af_idx, 0, len(lookup) - 1)], 0)
    for label, values in ((af_abundance_channel, result['af_scale']), (af_index_channel, index)):
        if label in pnn and pnn.index(label) < unmixed.shape[1]:
            unmixed[:, pnn.index(label)] = values
    return unmixed

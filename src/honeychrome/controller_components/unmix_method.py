"""
unmix_method.py
---------------
The experiment's unmixing method, resolved once and applied everywhere: the
main window's display, the unmixed export, Honeycluster, the gating plugins
and the Statistics tab all unmix through ``unmix``. No Qt.

Methods (``process['unmixing']``)
    R1    Per-cell AF (``apply_af_transfer``), or the transfer matrix (OLS)
          for a sample without AF. The default, and always the method when
          nothing else is ticked.
    M2pg  The noise model with the positivity prior (``gls_unmix.unmix_m2pg``),
          when a noise model is stored and ``noise_model`` is ticked; the
          noise model of the sample's particle class (beads or cells).
    Mc    Data clamped at zero: M2pg's path with ``NoiseState.clamped_model``
          (Poisson weights with a floor, M1p, or flat weights, M0p), when
          ``weights`` is ``'clamped'`` (set by Model Noise for clamped data)
          and the noise model is not ticked.
    R2    The AutoSpectral C++ kernel with per-cell spectral variants
          (``unmix_autospectral_optimization``), when ``variants`` is ticked
          and the noise model is not. Samples without an AF library of two or
          more spectra, or with the kernel not built, stay on R1.

    ``method_state(controller)`` reads everything the method needs on the
    main thread and returns None when every sample would be unmixed by R1
    with the base spectra, so the R1 code path is unchanged.

Spectral corrections (``process['spectral_corrections']``, written by the
Spectral Correction page)
    ::

        {'base_fingerprint': gls_unmix.spectra_fingerprint(names, base spectra),
         'groups': {group: {'samples': [sample keys], 'unstained': sample key,
                            'spread': af_spread-style entry or None,
                            'fluorophores': {label: {'spectrum': [D floats],
                                                     'active': bool}}}}}

    With ``process['unmixing']['corrections']`` ticked, each sample of a
    group is unmixed with the base spectra whose Active rows are replaced by
    its group's corrected spectra. What depends on the spectra follows each
    set (``set_extras``): the AF envelopes' null variance is rebuilt from
    each profile's stored unstained subsample; the spillover spread is the
    group's (recomputed from the controls by Fix Unmix) or the base one,
    whichever was built on spectra closer to the set's; the no-AF negative
    variances move by the change the noise fit's held-out negatives show;
    each corrected fluorophore's spectral variants move with its correction,
    and the positivity thresholds by the change the group's unstained
    subsample shows. The noise model itself describes the detectors and is
    kept. The overlay is ignored, with a warning, when the base spectra
    have changed since it was made.

Display cache
    M2pg and R2 results of the displayed events, and of whole samples
    unmixed in the background with a ``cache_file`` (``unmix``), are cached
    under ``<experiment>/cache/unmixed/`` (fluorophore abundances before
    compensation, AF abundance and row), keyed by everything that decides
    them (``display_cache_key``), least recently used files removed beyond
    the App Settings' unmixing cache size (``unmixing_cache_gb``).

v1 simplifications
    * Live acquisition stays on the transfer matrix (the controller does not
      call this module for live data).
    * A set whose Active rows differ from the ones Fix Unmix computed the
      group's spread for takes the nearer stored spread; beyond the
      envelope tolerance the envelopes carry no spread term (as for stale
      base spectra).
    * Samples without AF under M2pg take the prior's negative variance of
      the noise fit's unstained cells, or of its bead negatives for bead
      samples; the spillover spread is added while it matches the base
      spectra.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

UNMIXING_KEY = 'unmixing'
CORRECTIONS_KEY = 'spectral_corrections'
VARIANT_KERNEL_KEY = 'variant_kernel'

R1 = 'R1'
M2PG = 'M2pg'
CLAMPED = 'Mc'
R2 = 'R2'
NOISE_METHODS = (M2PG, CLAMPED)

UNMIXING_DEFAULTS = {'noise_model': False, 'variants': False, 'prior_lambda': 1.0, 'corrections': False,
                     'weights': None}

# The Spectral Variants page's Unmixing defaults (threads are set at run time).
VARIANT_KERNEL_DEFAULTS = {
    'n_passes': 1, 'n_af_passes': 1, 'refine_af_quantile': 0.5, 'cell_weight': False,
    'noise_floor': 125.0, 'alpha': 0.5, 'collinear_thresh': 0.5, 'joint_pair_resolution': True,
}

DISPLAY_CACHE_SUBDIR = Path('cache') / 'unmixed'
DISPLAY_CACHE_VERSION = 1

_noise_cache: dict = {}


def _bundled():
    from honeychrome.controller_components.plugin_loaders import bundled_plugins_path
    if str(bundled_plugins_path) not in sys.path:
        sys.path.insert(0, str(bundled_plugins_path))


def n_threads_default() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def unmixing_settings(process: dict) -> dict:
    """``process['unmixing']`` with its defaults filled in."""
    out = dict(UNMIXING_DEFAULTS)
    out.update({k: v for k, v in (process.get(UNMIXING_KEY) or {}).items() if k in UNMIXING_DEFAULTS})
    return out


def set_unmixing(process: dict, **values) -> None:
    """Update ``process['unmixing']``."""
    entry = process.setdefault(UNMIXING_KEY, {})
    entry.update(values)


def variant_kernel_settings(process: dict, cytometer_db_col: str | None = None) -> dict:
    """R2 kernel settings stored with the experiment; ``cell_weight`` defaults on for the ID7000."""
    out = dict(VARIANT_KERNEL_DEFAULTS)
    out['cell_weight'] = cytometer_db_col == 'ID7000'
    stored = (process.get(UNMIXING_KEY) or {}).get(VARIANT_KERNEL_KEY) or {}
    out.update({k: v for k, v in stored.items() if k in VARIANT_KERNEL_DEFAULTS})
    return out


def requested_method(process: dict) -> str:
    """The method the ticks ask for (D1: the noise model wins over variants)."""
    from honeychrome.controller_components import gls_unmix as gu

    s = unmixing_settings(process)
    if s['noise_model'] and process.get(gu.PROCESS_KEY):
        return M2PG
    if s['weights'] == gu.CLAMPED_WEIGHTS and process.get(gu.PROCESS_KEY):
        return CLAMPED
    if s['variants']:
        return R2
    return R1


def method_label(method: str, prior_lambda: float | None = None) -> str:
    """Name of a method for the status bar and the FCS UNMIXINGMETHOD keyword."""
    if method == M2PG:
        return f'AutoSpectral noise model (M2pg, lambda {prior_lambda:g})'
    if method == CLAMPED:
        return f'AutoSpectral weights for clamped data with the positivity prior (lambda {prior_lambda:g})'
    if method == R2:
        return 'AutoSpectral spectral variants (R2)'
    return 'AutoSpectral per-cell AF'


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------

def _update(h, obj) -> None:
    if obj is None:
        h.update(b'\x00none')
    elif isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())
    elif isinstance(obj, dict):
        h.update(b'{')
        for k in sorted(obj, key=str):
            h.update(str(k).encode())
            _update(h, obj[k])
        h.update(b'}')
    elif isinstance(obj, (list, tuple)):
        h.update(b'[')
        for v in obj:
            _update(h, v)
        h.update(b']')
    else:
        h.update(json.dumps(obj, default=str).encode())


def digest(*parts) -> str:
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        _update(h, p)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Spectral corrections
# ---------------------------------------------------------------------------

def corrections_in_effect(process: dict, names: list, base: np.ndarray) -> tuple:
    """
    ``(sets, membership)``: corrected spectra per group (only groups with an
    Active change) and each member sample's group, from the overlay when it
    is ticked and was made on the current base spectra.
    """
    if not unmixing_settings(process)['corrections']:
        return {}, {}
    entry = process.get(CORRECTIONS_KEY) or {}
    if not entry.get('groups'):
        return {}, {}
    from honeychrome.controller_components.gls_unmix import spectra_fingerprint
    if entry.get('base_fingerprint') != spectra_fingerprint(names, base):
        logger.warning('Spectral corrections were made on earlier spectra and are not applied; '
                       'run Fix Unmix again.')
        return {}, {}
    sets, membership = {}, {}
    for group, g in entry['groups'].items():
        spectra = np.array(base, dtype=np.float64, copy=True)
        changed = False
        for label, fc in (g.get('fluorophores') or {}).items():
            if label in names and fc.get('active', True) and fc.get('spectrum') is not None:
                row = np.asarray(fc['spectrum'], dtype=np.float64)
                if row.shape != (spectra.shape[1],):
                    logger.warning('Spectral correction for %s in %s has the wrong length; ignored.', label, group)
                    continue
                spectra[names.index(label)] = row
                changed = True
        if changed:
            sets[str(group)] = spectra
            for key in g.get('samples') or []:
                membership[str(key)] = str(group)
    return sets, membership


def unmixing_matrix_for(spectra: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """``(S W S')^-1 S W`` with ``W = diag(1 / weights)``, as calculate_spectral_process."""
    s = np.asarray(spectra, dtype=np.float64)
    w = np.ones(s.shape[1]) if weights is None else 1.0 / np.asarray(weights, dtype=np.float64)
    sw = s * w
    return np.linalg.solve(sw @ s.T, sw)


def transfer_with_spectra(transfer_matrix: np.ndarray, fl_ids_raw, fl_ids_unmixed, unmixing_matrix: np.ndarray,
                          spillover=None) -> np.ndarray:
    """``transfer_matrix`` (raw x unmixed) with its fluorescence block from ``unmixing_matrix``,
    compensated as the original when ``spillover`` is given."""
    t = np.array(transfer_matrix, dtype=np.float64, copy=True)
    w = np.asarray(unmixing_matrix, dtype=np.float64)
    if spillover is not None:
        w = np.linalg.inv(np.asarray(spillover, dtype=np.float64)).T @ w
    t[np.ix_(np.asarray(fl_ids_raw), np.asarray(fl_ids_unmixed))] = w.T
    return t


# ---------------------------------------------------------------------------
# Method state (main thread)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class MethodState:
    """Everything the experiment's unmixing method needs, read on the main thread."""
    method: str
    fluor_names: tuple
    spectra: np.ndarray
    ceiling: float
    prior_lambda: float = 1.0
    noise: object = None                     # gls_unmix.NoiseState
    spread_no_af: np.ndarray | None = None
    variants: dict | None = None             # {'meta', 'active', 'thresholds', 'kernel'}
    corrected_sets: dict = field(default_factory=dict)
    sample_sets: dict = field(default_factory=dict)
    unmixing_weights: np.ndarray | None = None
    bead_samples: frozenset = frozenset()
    m2p_kernel: bool = False
    r2_kernel: bool = False
    n_threads: int = 1
    experiment_dir: str | None = None
    af_profiles: dict = field(default_factory=dict)     # {name: {'spectra', 'envelope'}}
    set_extras: dict = field(default_factory=dict)      # {set: {'spread', 'null_var', 'variants'}}
    fingerprint: str = ''
    _derived: dict = field(default_factory=dict, repr=False)

    def set_for(self, sample_key) -> str:
        """The spectra set of a sample: '' for the base spectra, else its correction group."""
        return self.sample_sets.get(str(sample_key), '') if sample_key is not None else ''

    def spectra_for(self, set_id: str) -> np.ndarray:
        return self.corrected_sets[set_id] if set_id else self.spectra

    def effective_method(self, af, sample_key=None) -> str:
        """The method a sample is unmixed with."""
        if self.method == M2PG and self.noise is not None:
            return M2PG
        if self.method == CLAMPED and self.noise is not None and self.noise.clamped_model is not None:
            return CLAMPED
        if (self.method == R2 and self.r2_kernel and self.variants is not None
                and af is not None and af.n_spectra >= 2):
            return R2
        return R1

    def is_bead(self, sample_key) -> bool:
        return sample_key is not None and str(sample_key) in self.bead_samples

    def sample_fingerprint(self, sample_key, af) -> str:
        """Digest of everything deciding a sample's unmixed values under this state."""
        method = self.effective_method(af, sample_key)
        set_id = self.set_for(sample_key)
        af_part = None
        if af is not None:
            env = af.precomputed.get('envelope') if af.precomputed else None
            af_part = [list(af.profile_names), af.spectra,
                       None if env is None else [env.get('null_var'), env.get('spread'),
                                                 env.get('mean_rows'), env.get('ceiling')]]
        return digest(self.fingerprint, method, set_id, self.spectra_for(set_id) if set_id else None,
                      self.is_bead(sample_key) if af is None else None, af_part)


def _noise_state(controller):
    from honeychrome.controller_components import gls_unmix as gu
    from honeychrome.controller_components.spread_model import raw_layout

    process = controller.experiment.process
    entry = process.get(gu.PROCESS_KEY)
    if not entry:
        return None
    detectors = raw_layout(controller)['detectors']
    key = (str(controller.experiment_dir), entry.get('file'), entry.get('fingerprint'), entry.get('fitted'),
           tuple(detectors))
    if key not in _noise_cache:
        _noise_cache.clear()
        _noise_cache[key] = gu.load_noise_state(process, controller.experiment_dir, detectors)
    return _noise_cache[key]


def _variants_payload(controller, names: list, spectra: np.ndarray) -> dict | None:
    _bundled()
    from autospectral_optimization_functions import active_variant_labels, cached_unmixed_thresholds

    meta = dict(getattr(controller, 'autospectral_variants', None) or {})
    if not meta:
        return None
    process = controller.experiment.process
    active = sorted(active_variant_labels(process))
    db_col = (controller.experiment.settings.get('raw') or {}).get('cytometer_db_col')
    return {
        'meta': meta, 'active': active,
        'thresholds': cached_unmixed_thresholds(controller, spectra),
        'kernel': variant_kernel_settings(process, db_col),
    }


def _r2_kernel_available() -> bool:
    _bundled()
    try:
        from autospectral_opt_kernel_wrapper import AUTOSPECTRAL_OPT_KERNEL_AVAILABLE
    except ImportError:
        return False
    return bool(AUTOSPECTRAL_OPT_KERNEL_AVAILABLE)


def method_state(controller, corrections: tuple | None = None) -> MethodState | None:
    """
    The experiment's unmixing method for every sample, or None when all of
    them are unmixed by R1 with the base spectra. ``corrections``
    (``({set: spectra}, {sample: set})``, as ``corrections_in_effect``
    returns) replaces the stored overlay, e.g. ``({}, {})`` for none. Main
    thread only.
    """
    experiment = getattr(controller, 'experiment', None)
    process = getattr(experiment, 'process', None)
    if not process or (corrections is None and UNMIXING_KEY not in process and CORRECTIONS_KEY not in process):
        return None
    from honeychrome.controller_components import gls_unmix as gu
    from honeychrome.controller_components.particle_type import is_bead_sample
    from honeychrome.controller_components.spread_model import raw_layout, spread_is_current

    names, spectra = gu.fluor_names_and_spectra(controller)
    if spectra is None:
        return None
    method = requested_method(process)
    corrected, membership = (corrections_in_effect(process, names, spectra) if corrections is None
                             else corrections)
    if method == R1 and not corrected:
        return None

    lam = gu.prior_lambda(process)
    noise = spread = variants = None
    if method in NOISE_METHODS:
        noise = _noise_state(controller)
        if noise is None:
            logger.warning('Unmixing: the noise model could not be read for the current detectors; '
                           'per-cell AF is used. Model the noise again.')
            method = R1
        elif method == CLAMPED and noise.clamped_model is None:
            logger.warning('Unmixing: weights for clamped data are asked for but the noise fit stored none; '
                           'per-cell AF is used. Model the noise again.')
            method = R1
        else:
            spread_entry = process.get('af_spread')
            if spread_is_current(spread_entry, names, spectra):
                spread = np.asarray(spread_entry['spread'], dtype=np.float64)
    r2_kernel = False
    if method == R2:
        r2_kernel = _r2_kernel_available()
        variants = _variants_payload(controller, names, spectra)
        if variants is None:
            logger.warning('Unmixing: spectral variants are ticked but none are computed; '
                           'per-cell AF is used.')
            method = R1
        elif not r2_kernel:
            logger.warning('Unmixing: the AutoSpectral kernel is not built; per-cell AF is used.')
            method = R1
    if method == R1 and not corrected:
        return None

    samples = controller.experiment.samples
    beads = frozenset(str(k) for k in (samples.get('all_samples') or {}) if is_bead_sample(samples, k))
    weights = process.get('unmixing_weights')
    weights = None if weights is None else np.asarray(weights, dtype=np.float64)
    ceiling = raw_layout(controller)['ceiling']
    m2p_kernel = gu.kernel_available() if method in NOISE_METHODS else False

    parts = {'method': method, 'names': list(names), 'spectra': spectra, 'ceiling': ceiling,
             'weights': weights, 'corrected': corrected}
    if method in NOISE_METHODS:
        parts.update(noise=[noise.entry.get('fingerprint'), noise.entry.get('fitted'), noise.cell_null_var,
                            noise.bead_null_var, noise.entry.get('flat_phi'), noise.entry.get('clamped_weights')],
                     prior_lambda=lam, spread=spread,
                     kernel=m2p_kernel)
    if method == R2:
        meta = variants['meta']
        parts.update(active=variants['active'], thresholds=variants['thresholds'], kernel=variants['kernel'],
                     variants={k: [meta[k].get('v_mats'), meta[k].get('delta')]
                               for k in variants['active'] if k in meta})
    extras = set_extras(controller, process, list(names), spectra, corrected, noise, variants)
    if extras:
        parts['extras'] = {k: [None if e['spread'] is None else e['spread']['spread'], e.get('null_var'),
                               None if e.get('variants') is None else e['variants']['thresholds']]
                           for k, e in extras.items()}
    af_profiles = {n: {'spectra': np.asarray(e['spectra'], dtype=np.float64), 'envelope': e.get('envelope')}
                   for n, e in (process.get('af_profiles') or {}).items()} if corrected else {}
    return MethodState(
        method=method, fluor_names=tuple(names), spectra=spectra, ceiling=ceiling, prior_lambda=lam,
        noise=noise, spread_no_af=spread, variants=variants, corrected_sets=corrected,
        sample_sets=membership, unmixing_weights=weights, bead_samples=beads, m2p_kernel=m2p_kernel,
        r2_kernel=r2_kernel, n_threads=n_threads_default(), experiment_dir=str(controller.experiment_dir),
        af_profiles=af_profiles, set_extras=extras, fingerprint=digest(parts),
    )


# ---------------------------------------------------------------------------
# What follows the spectra of a corrected set (main thread)
# ---------------------------------------------------------------------------

_threshold_cache: dict = {}


def _max_angle(a, b) -> float:
    from honeychrome.controller_components.autospectral_functions import spectral_angles_deg
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(spectral_angles_deg(a, b))) if a.shape == b.shape else np.inf


def nearest_spread(entries, names: list, spectra: np.ndarray) -> dict | None:
    """Of stored spread entries over ``names``, the one built on spectra closest to ``spectra``."""
    usable = [e for e in entries if e and list(e.get('fluor_names', [])) == list(names)]
    if not usable:
        return None
    return min(usable, key=lambda e: _max_angle(e['fluor_spectra'], spectra))


def shifted_variants(meta: dict, names: list, base: np.ndarray, spectra: np.ndarray) -> dict:
    """Variant libraries with each changed fluorophore's variant spectra moved by its change, peak-normalised."""
    out = dict(meta)
    for i, name in enumerate(names):
        if name not in meta or meta[name].get('v_mats') is None or np.array_equal(base[i], spectra[i]):
            continue
        v = np.maximum(np.asarray(meta[name]['v_mats'], dtype=np.float64) + (spectra[i] - base[i]), 0.0)
        v = v / np.maximum(v.max(axis=1, keepdims=True), 1e-12)
        entry = dict(meta[name])
        entry['v_mats'] = v
        entry['delta'] = v - v[:1]
        out[name] = entry
    return out


def threshold_shift(controller, unstained_key, names: list, base: np.ndarray, spectra: np.ndarray) -> np.ndarray:
    """
    Change of the 99.5th-percentile unmixed positivity thresholds from the
    base spectra to ``spectra``, on the stored 5,000-event subsample of the
    unstained sample's AF profile (per-cell AF); zeros when none is stored.
    """
    from honeychrome.controller_components.autospectral_functions import (
        AF_SUBDIR, apply_af_unmixing, build_af_precomputed,
    )
    zeros = np.zeros(len(names))
    profiles = controller.experiment.process.get('af_profiles') or {}
    found = [(n, e) for n, e in profiles.items()
             if unstained_key is not None and str(e.get('source_fcs')) == str(unstained_key)
             and (e.get('envelope') or {}).get('unstained_file')]
    if not found:
        return zeros
    name, entry = found[0]
    path = Path(controller.experiment_dir) / AF_SUBDIR / entry['envelope']['unstained_file']
    key = digest(str(path), base, spectra)
    if key in _threshold_cache:
        return _threshold_cache[key]
    try:
        sample = np.load(path).astype(np.float64)
    except (OSError, ValueError) as exc:
        logger.warning('Unmixing: unstained subsample %s not read (%s); thresholds kept.', path.name, exc)
        return zeros
    lib = np.asarray(entry['spectra'], dtype=np.float64)
    if sample.shape[1] != base.shape[1]:
        return zeros
    pre = controller.precomputed_for_profile(name) if hasattr(controller, 'precomputed_for_profile') else None
    envelope = None if pre is None else pre.get('envelope')
    q = []
    for s in (base, spectra):
        out = apply_af_unmixing(sample, build_af_precomputed(s, lib, envelope), lib)['unmixed']
        q.append(np.percentile(out, 99.5, axis=0))
    _threshold_cache.clear()
    _threshold_cache[key] = q[1] - q[0]
    return _threshold_cache[key]


def set_extras(controller, process: dict, names: list, base: np.ndarray, sets: dict, noise=None,
               variants: dict | None = None) -> dict:
    """{set: {'spread', 'null_var', 'variants'}} for corrected spectra sets (see the module docstring)."""
    if not sets:
        return {}
    from honeychrome.controller_components import gls_unmix as gu

    groups = (process.get(CORRECTIONS_KEY) or {}).get('groups') or {}
    out = {}
    for set_id, spectra in sets.items():
        g = groups.get(set_id) or {}
        extra = {'spread': nearest_spread([g.get('spread'), process.get('af_spread')], names, spectra)}
        if noise is not None:
            extra['null_var'] = gu.null_variances(noise, base, spectra)
        if variants is not None:
            shift = threshold_shift(controller, g.get('unstained'), names, base, spectra)
            extra['variants'] = {**variants, 'meta': shifted_variants(variants['meta'], names, base, spectra),
                                 'thresholds': np.asarray(variants['thresholds'], dtype=np.float64) + shift}
        out[set_id] = extra
    return out


def method_token(state: MethodState | None) -> str:
    """Hashable description of the method for worker caches ('' for R1 with the base spectra)."""
    return '' if state is None else state.fingerprint


# ---------------------------------------------------------------------------
# Unmixing (worker-safe)
# ---------------------------------------------------------------------------

def _set_matrices(state: MethodState, set_id: str, transfer_matrix, fl_ids_raw, settings, spillover):
    """Transfer matrix for a corrected spectra set (cached per transfer matrix)."""
    if not set_id:
        return transfer_matrix
    key = ('transfer', set_id, spillover is None)
    cached = state._derived.get(key)
    if cached is None or cached[0] is not transfer_matrix:
        w = unmixing_matrix_for(state.spectra_for(set_id), state.unmixing_weights)
        cached = (transfer_matrix, transfer_with_spectra(
            transfer_matrix, fl_ids_raw, settings['unmixed']['fluorescence_channel_ids'], w, spillover))
        state._derived[key] = cached
    return cached[1]


def rebuilt_envelope(profile_names, af_profiles: dict, fluor_names, spectra: np.ndarray, experiment_dir,
                     ceiling: float, spread_entry: dict | None = None):
    """
    An AF library's runtime envelope under other spectra: each profile's
    null variance rebuilt from its stored unstained subsample, with
    ``spread_entry``'s spillover spread when it matches ``spectra``.
    ``af_profiles`` is {name: {'spectra', 'envelope'}}. None when a profile
    has no envelope or cannot be rebuilt.
    """
    from honeychrome.controller_components.autospectral_functions import (
        build_runtime_envelope, combine_envelopes, refresh_envelope_entry,
    )
    envelopes, sizes = [], []
    for name in profile_names:
        profile = af_profiles.get(name)
        if profile is None or not profile.get('envelope'):
            return None
        refreshed, _changed = refresh_envelope_entry(profile['envelope'], list(fluor_names), spectra,
                                                     profile['spectra'], experiment_dir, tolerance_deg=0.0)
        runtime = None if refreshed is None else build_runtime_envelope(
            refreshed, spread_entry, list(fluor_names), spectra, ceiling)
        if runtime is None:
            return None
        envelopes.append(runtime)
        sizes.append(len(profile['spectra']))
    return combine_envelopes(envelopes, sizes)


def _set_envelope(state: MethodState, set_id: str, af, spectra: np.ndarray):
    """
    The AF library's runtime envelope under a corrected spectra set
    (``rebuilt_envelope`` with the set's spillover spread); the base envelope
    when a profile cannot be rebuilt, None (joint L2) when the base library
    has none.
    """
    base_env = af.precomputed.get('envelope') if af.precomputed else None
    if base_env is None:
        return None
    spread_entry = (state.set_extras.get(set_id) or {}).get('spread')
    env = rebuilt_envelope(af.profile_names, state.af_profiles, state.fluor_names, spectra, state.experiment_dir,
                           state.ceiling, spread_entry)
    return base_env if env is None else env


def _set_af(state: MethodState, set_id: str, af):
    """The sample's AF library precomputed with a corrected spectra set and its rebuilt envelope."""
    if not set_id or af is None:
        return af
    key = ('af', set_id, tuple(af.profile_names))
    if key not in state._derived:
        from honeychrome.controller_components.autospectral_functions import build_af_precomputed
        spectra = state.spectra_for(set_id)
        pre = build_af_precomputed(spectra, np.asarray(af.spectra, dtype=np.float64),
                                   _set_envelope(state, set_id, af, spectra))
        state._derived[key] = replace(af, precomputed=pre)
    return state._derived[key]


def unmix_fluorescence(raw_fl: np.ndarray, state: MethodState, af, sample_key=None, method: str | None = None) -> dict:
    """
    M2pg, the clamped-data weights or R2 on raw fluorescence events. Returns ``unmixed`` (n, F,
    uncompensated), ``af_scale`` and ``af_idx`` (0-based, -1 without AF).
    """
    from honeychrome.controller_components import gls_unmix as gu

    method = method or state.effective_method(af, sample_key)
    set_id = state.set_for(sample_key)
    spectra = state.spectra_for(set_id)
    af = _set_af(state, set_id, af)
    extra = state.set_extras.get(set_id) or {} if set_id else {}
    raw_fl = np.ascontiguousarray(raw_fl, dtype=np.float64)
    envelope = None if af is None or not af.precomputed else af.precomputed.get('envelope')
    if method in NOISE_METHODS:
        noise = state.noise
        model = noise.clamped_model if method == CLAMPED else noise.model_for(state.is_bead(sample_key))
        null_var = spread_no_af = None
        if af is None:
            cell_var, bead_var = extra.get('null_var') or (noise.cell_null_var, noise.bead_null_var)
            null_var = bead_var if state.is_bead(sample_key) else cell_var
            spread_no_af = state.spread_no_af
            if extra.get('spread') is not None:
                from honeychrome.controller_components.spread_model import spread_is_current
                if spread_is_current(extra['spread'], list(state.fluor_names), spectra):
                    spread_no_af = np.asarray(extra['spread']['spread'], dtype=np.float64)
        return gu.unmix_m2pg(raw_fl, spectra, model, None if af is None else af.spectra, envelope,
                             null_var_no_af=null_var, spread=spread_no_af,
                             prior_lambda=state.prior_lambda, ceiling=state.ceiling, n_threads=state.n_threads)
    if method == R2:
        _bundled()
        from autospectral_optimization_functions import unmix_autospectral_optimization
        v = extra.get('variants') or state.variants
        res = unmix_autospectral_optimization(
            raw_fl, spectra, list(state.fluor_names), np.asarray(af.spectra, dtype=np.float64), v['meta'],
            v['active'], v['thresholds'], n_threads=state.n_threads, envelope=envelope, **v['kernel'])
        idx = np.asarray(res['af_idx'], dtype=np.int64)
        return {'unmixed': res['unmixed'], 'af_scale': res['af_scale'], 'af_idx': np.where(idx > 0, idx - 1, -1)}
    raise ValueError(f'unmix_fluorescence: {method} is not a background method')


def assemble(raw_event_data, transfer_matrix, settings, fl_result: dict, spillover=None, af=None) -> dict:
    """Full unmixed array from a fluorescence result; ``af_idx`` returned 1-based (0 = no AF)."""
    from honeychrome.controller_components import gls_unmix as gu

    unmixed = gu.assemble_unmixed(raw_event_data, transfer_matrix, settings, fl_result, spillover=spillover,
                                  af_index_map=None if af is None else af.index_map)
    idx = np.asarray(fl_result['af_idx'], dtype=np.int64)
    return {'unmixed': unmixed, 'af_scale': np.asarray(fl_result['af_scale'], dtype=np.float64),
            'af_idx': np.where(idx >= 0, idx + 1, 0)}


def unmix(raw_event_data, transfer_matrix, settings, fl_ids_raw, spillover, af, state: MethodState | None,
          sample_key=None, force_r1: bool = False, cache_file=None) -> dict:
    """
    Unmix raw events with the experiment's method for ``sample_key``.

    ``transfer_matrix`` (raw x unmixed) and ``spillover`` follow the caller's
    convention: a compensated transfer matrix with ``spillover``, or an
    uncompensated one with ``spillover=None``. Returns ``unmixed``,
    ``af_scale``, ``af_idx`` (1-based into the sample's AF library, 0 = no
    AF) and ``method``. ``force_r1`` gives R1 with the sample's spectra (the
    progressive display's first pass). ``cache_file`` (the sample's FCS file)
    reads an M2pg or R2 result from the display cache, or computes and
    stores it; the result is then always the cache's float32 values, so a
    first and a later load agree exactly.
    """
    from honeychrome.controller_components.autospectral_functions import apply_af_transfer
    from honeychrome.controller_components.functions import apply_transfer_matrix

    n = len(raw_event_data)
    method = R1 if state is None or force_r1 else state.effective_method(af, sample_key)
    set_id = '' if state is None else state.set_for(sample_key)
    if set_id:
        transfer_matrix = _set_matrices(state, set_id, transfer_matrix, fl_ids_raw, settings, spillover)
        af = _set_af(state, set_id, af)
    if method == R1:
        if af is not None:
            result = apply_af_transfer(raw_event_data, transfer_matrix, af.precomputed, af.spectra, settings,
                                       filtered_fl_ids_raw=list(fl_ids_raw), spillover=spillover,
                                       af_index_map=af.index_map)
            return {'unmixed': result['unmixed'], 'af_scale': np.asarray(result['af_scale'], dtype=np.float64),
                    'af_idx': np.asarray(result['af_idx'], dtype=np.int64), 'method': R1}
        return {'unmixed': apply_transfer_matrix(transfer_matrix, raw_event_data),
                'af_scale': np.zeros(n), 'af_idx': np.zeros(n, dtype=np.int64), 'method': R1}
    cache_key = fl = None
    if cache_file is not None and state.experiment_dir:
        cache_key = display_cache_key(state, sample_key, af, raw_event_data, fl_ids_raw, cache_file)
        fl = load_display_cache(state.experiment_dir, cache_key, n, len(state.fluor_names))
        if fl is not None:
            logger.debug('unmix: %s of %s from the unmixing cache', method, sample_key)
    if fl is None:
        fl = unmix_fluorescence(np.asarray(raw_event_data)[:, list(fl_ids_raw)], state, af, sample_key, method)
        if cache_key is not None:
            store_display_cache(state.experiment_dir, cache_key, fl)
            fl = {'unmixed': np.asarray(fl['unmixed'], dtype=np.float32).astype(np.float64),
                  'af_scale': np.asarray(fl['af_scale'], dtype=np.float32).astype(np.float64),
                  'af_idx': np.asarray(fl['af_idx'], dtype=np.int32).astype(np.int64)}
    out = assemble(raw_event_data, transfer_matrix, settings, fl, spillover, af)
    out['method'] = method
    return out


# ---------------------------------------------------------------------------
# Display cache
# ---------------------------------------------------------------------------

def display_cache_key(state: MethodState, sample_key, af, raw_event_data: np.ndarray, fl_ids_raw,
                      sample_file=None) -> str:
    """Digest of the method, the sample's file and a strided sample of its displayed raw events."""
    file_part = None
    if sample_file is not None:
        try:
            st = Path(sample_file).stat()
            file_part = [st.st_size, st.st_mtime_ns]
        except OSError:
            file_part = None
    raw = np.asarray(raw_event_data)
    step = max(1, len(raw) // 4096)
    probe = np.ascontiguousarray(raw[::step][:, list(fl_ids_raw)], dtype=np.float64)
    return digest(DISPLAY_CACHE_VERSION, state.sample_fingerprint(sample_key, af), str(sample_key), file_part,
                  raw.shape, probe)


def _cache_file(experiment_dir, key: str) -> Path:
    return Path(experiment_dir) / DISPLAY_CACHE_SUBDIR / f'{key}.npz'


def load_display_cache(experiment_dir, key: str, n_events: int, n_fluors: int) -> dict | None:
    """A cached fluorescence result for ``key``, or None."""
    path = _cache_file(experiment_dir, key)
    if not path.exists():
        return None
    try:
        with np.load(path) as z:
            out = {'unmixed': z['unmixed'].astype(np.float64), 'af_scale': z['af_scale'].astype(np.float64),
                   'af_idx': z['af_idx'].astype(np.int64)}
        os.utime(path)
    except Exception as exc:
        logger.warning('Unmixing cache: %s unreadable (%s)', path.name, exc)
        return None
    if out['unmixed'].shape != (n_events, n_fluors):
        return None
    return out


def store_display_cache(experiment_dir, key: str, fl_result: dict, max_bytes: int | None = None) -> None:
    """Write a fluorescence result (float32) and drop the oldest files beyond ``max_bytes``."""
    path = _cache_file(experiment_dir, key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp.npz')
        np.savez(tmp, unmixed=np.asarray(fl_result['unmixed'], dtype=np.float32),
                 af_scale=np.asarray(fl_result['af_scale'], dtype=np.float32),
                 af_idx=np.asarray(fl_result['af_idx'], dtype=np.int32))
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning('Unmixing cache: could not write %s (%s)', path.name, exc)
        return
    prune_display_cache(path.parent, max_bytes)


def prune_display_cache(folder, max_bytes: int | None = None) -> None:
    """Remove the least recently used files beyond ``max_bytes`` (default:
    the App Settings' unmixing cache size)."""
    if max_bytes is None:
        from honeychrome import settings as app_settings
        max_bytes = int(app_settings.unmixing_cache_gb_retrieved) * 1024 ** 3
    files = []
    for p in Path(folder).glob('*.npz'):
        try:
            st = p.stat()
        except OSError:
            continue
        files.append((st.st_mtime, st.st_size, p))
    total = sum(f[1] for f in files)
    for _mtime, size, p in sorted(files):
        if total <= max_bytes:
            break
        try:
            p.unlink()
            total -= size
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Export time
# ---------------------------------------------------------------------------

def needs_time_probe(state: MethodState | None) -> bool:
    """R2, and M2pg or the clamped-data weights without the compiled kernel, are slow enough to time first."""
    if state is None:
        return False
    return state.method == R2 or (state.method in NOISE_METHODS and not state.m2p_kernel)


def time_probe(raw_fl: np.ndarray, state: MethodState, af, sample_key=None) -> float:
    """Events per second of the method on ``raw_fl``."""
    start = time.perf_counter()
    unmix_fluorescence(raw_fl, state, af, sample_key)
    return len(raw_fl) / max(time.perf_counter() - start, 1e-9)

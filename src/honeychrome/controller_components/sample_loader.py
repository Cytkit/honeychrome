"""
sample_loader.py — thread-safe loading and unmixing of experiment samples
=========================================================================

Plugins and background workers that load many samples (training pools,
batch gating, per-sample statistics) need the same unmixing the main window
applies to the displayed sample, including each sample's own autofluorescence
(AF) profile assignment, but without touching the controller's live state.

The work is split in two:

``snapshot_unmix_state(controller, sample_keys)``
    Main thread only. Captures everything unmixing needs: the transfer
    matrix, raw channel order, fluorescence channel positions, spillover,
    a deep copy of the experiment settings, and the AF library assigned to
    each requested sample (resolved from ``samples['sample_af_profiles']``
    and the controller's precomputed AF cache). Also collects per-sample
    time-QC keep-masks when the ``time_qc`` module is present.

``load_unmixed(experiment_dir, sample_key, snapshot)``
    Worker-safe. Reads one FCS file, applies an optional keep-mask and a
    seeded event cap, and unmixes it with that sample's own AF library.
    Optionally returns the per-cell AF abundance and AF index.

Sample keys are the keys of ``experiment.samples['all_samples']``: paths
relative to the experiment directory, e.g. ``'Raw/Spleen/S1.fcs'``.

AF index convention: ``af_idx`` is 1-based into the sample's combined AF
library; 0 means the event was unmixed without AF correction. The AF Index
channel of the unmixed array holds the experiment-wide index instead (see
``autospectral_functions.af_index_lookup``).
"""

from __future__ import annotations

import hashlib
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from honeychrome.controller_components.autospectral_functions import (
    af_index_lookup,
    apply_af_transfer,
    combine_af_precomputed,
    precompute_joint_cov_extras,
)

logger = logging.getLogger(__name__)

__all__ = [
    'AFState', 'UnmixSnapshot', 'sample_key_for_path', 'af_profile_names',
    'resolve_af_for_profiles', 'snapshot_unmix_state', 'read_raw_events',
    'unmix_events', 'load_unmixed', 'seeded_subsample_indices',
    'control_sample_keys', 'analysis_sample_keys',
]


# ---------------------------------------------------------------------------
# Snapshot types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AFState:
    """Combined AF library for one profile assignment.

    ``index_map`` holds the experiment-wide AF Index of each library row.
    """
    profile_names: tuple
    precomputed: dict
    spectra: np.ndarray
    index_map: np.ndarray | None = None

    @property
    def n_spectra(self) -> int:
        return int(self.spectra.shape[0])


@dataclass(frozen=True)
class UnmixSnapshot:
    """Everything needed to unmix samples off the main thread.

    ``transfer_matrix`` is a reference to the controller's array at snapshot
    time. The controller replaces that array rather than mutating it, so a
    worker holding this snapshot is unaffected by later reassignments.
    """
    transfer_matrix: np.ndarray
    pnn_raw: tuple
    whitelisted: bool
    fl_ids_raw: tuple
    spillover: np.ndarray | None
    settings: dict
    af_by_sample: dict = field(default_factory=dict)
    qc_by_sample: dict = field(default_factory=dict)

    def af_for(self, sample_key: str | None) -> AFState | None:
        """AF library assigned to *sample_key*, or None for plain unmixing."""
        if sample_key is None:
            return None
        return self.af_by_sample.get(str(sample_key))

    def af_profile_key(self, sample_key: str | None) -> tuple:
        """Hashable description of the sample's AF assignment (for caches)."""
        af = self.af_for(sample_key)
        return af.profile_names if af is not None else ()

    def legacy_af_state(self, sample_key: str | None) -> tuple:
        """``(transfer_matrix, af_precomputed, af_spectra)`` for *sample_key*."""
        af = self.af_for(sample_key)
        if af is None:
            return (self.transfer_matrix, None, None)
        return (self.transfer_matrix, af.precomputed, af.spectra)

    def max_af_spectra(self) -> int:
        """Largest AF library size across the snapshot's samples (0 if none)."""
        sizes = [af.n_spectra for af in self.af_by_sample.values() if af is not None]
        return max(sizes) if sizes else 0


# ---------------------------------------------------------------------------
# Sample keys
# ---------------------------------------------------------------------------

def sample_key_for_path(experiment_dir, path) -> str:
    """Return the ``all_samples`` key for *path*.

    *path* may be absolute or already experiment-relative.
    """
    p = Path(path)
    if p.is_absolute():
        try:
            return str(p.relative_to(Path(experiment_dir)))
        except ValueError:
            return str(p)
    return str(p)


def control_sample_keys(controller) -> set:
    """Keys of single-stain controls and unstained samples."""
    samples = controller.experiment.samples
    return set(samples.get('single_stain_controls') or []) | set(
        samples.get('unstained_samples') or []
    )


def analysis_sample_keys(controller, include_controls: bool = False) -> list:
    """All sample keys, in experiment order, optionally without controls."""
    keys = list((controller.experiment.samples.get('all_samples') or {}).keys())
    if include_controls:
        return keys
    controls = control_sample_keys(controller)
    return [k for k in keys if k not in controls]


# ---------------------------------------------------------------------------
# AF resolution (main thread)
# ---------------------------------------------------------------------------

def af_profile_names(controller, sample_key: str) -> list:
    """AF profile names assigned to *sample_key* (may be empty)."""
    return list(
        controller.experiment.samples
        .get('sample_af_profiles', {})
        .get(str(sample_key), []) or []
    )


def resolve_af_for_profiles(controller, profile_names) -> AFState | None:
    """Combine cached AF matrices for an explicit list of profile names.

    Main thread only: reads ``controller.af_precomputed_cache`` and
    ``experiment.process['af_profiles']``, both of which the main thread can
    replace. Mirrors ``Controller.initialise_af_matrices``. Profiles that are
    not cached or have no stored spectra are skipped. Returns None when no
    profile can be used.
    """
    names = [str(n) for n in (profile_names or [])]
    if not names:
        return None
    cache = getattr(controller, 'af_precomputed_cache', None) or {}
    af_profiles = controller.experiment.process.get('af_profiles', {}) or {}
    usable = [n for n in names if n in cache and n in af_profiles]
    if len(usable) != len(names):
        missing = [n for n in names if n not in usable]
        logger.warning(
            'resolve_af_for_profiles: profile(s) %s not cached or not stored; '
            'using %s', missing, usable,
        )
    if not usable:
        return None

    spectra = np.vstack([np.array(af_profiles[n]['spectra']) for n in usable])
    cached = [cache[n] for n in usable]
    if len(cached) == 1:
        combined = cached[0]
    else:
        combined = combine_af_precomputed(cached)
        combined.update(precompute_joint_cov_extras(combined, spectra))
    return AFState(profile_names=tuple(usable), precomputed=combined, spectra=spectra,
                   index_map=af_index_lookup(af_profiles, usable))


def _time_qc_masks(controller, sample_keys) -> dict:
    """Per-sample keep-masks from the time-QC module, when it is installed."""
    try:
        from honeychrome.controller_components import time_qc
    except ImportError:
        return {}
    getter = getattr(time_qc, 'get_keep_mask', None)
    if getter is None:
        return {}
    masks = {}
    for key in sample_keys:
        try:
            mask = getter(controller, key)
        except Exception as exc:
            logger.warning('time QC mask unavailable for %s: %s', key, exc)
            continue
        if mask is not None:
            masks[str(key)] = np.asarray(mask, dtype=bool).copy()
    return masks


def snapshot_unmix_state(controller, sample_keys=None,
                         include_qc: bool = True) -> UnmixSnapshot:
    """Capture unmixing state for *sample_keys* (default: every sample).

    Main thread only. AF libraries are combined once per distinct profile
    assignment and shared between samples with the same assignment.
    """
    if controller.transfer_matrix is None:
        raise RuntimeError('No transfer matrix: the spectral model is not set up.')

    settings = deepcopy(controller.experiment.settings)
    raw = settings['raw']
    whitelist = raw.get('whitelisted_pnn') or None
    pnn_raw = list(whitelist or raw['event_channels_pnn'])
    full_pnn_raw = raw['event_channels_pnn']

    fl_ids_full = getattr(controller, 'filtered_raw_fluorescence_channel_ids', None)
    if fl_ids_full is None:
        fl_ids_full = raw.get('fluorescence_channel_ids') or []
    fl_ids_raw = tuple(
        pnn_raw.index(full_pnn_raw[i]) for i in fl_ids_full
        if full_pnn_raw[i] in pnn_raw
    )

    spillover = controller.experiment.process.get('spillover')
    spillover = None if spillover is None else np.array(spillover, dtype=np.float64)

    if sample_keys is None:
        sample_keys = list((controller.experiment.samples.get('all_samples') or {}).keys())
    sample_keys = [str(k) for k in sample_keys]

    by_assignment: dict[tuple, AFState | None] = {}
    af_by_sample: dict[str, AFState | None] = {}
    for key in sample_keys:
        names = tuple(af_profile_names(controller, key))
        if names not in by_assignment:
            by_assignment[names] = resolve_af_for_profiles(controller, names) if names else None
        af_by_sample[key] = by_assignment[names]

    qc = _time_qc_masks(controller, sample_keys) if include_qc else {}

    return UnmixSnapshot(
        transfer_matrix=controller.transfer_matrix,
        pnn_raw=tuple(pnn_raw),
        whitelisted=whitelist is not None,
        fl_ids_raw=fl_ids_raw,
        spillover=spillover,
        settings=settings,
        af_by_sample=af_by_sample,
        qc_by_sample=qc,
    )


# ---------------------------------------------------------------------------
# Loading and unmixing (worker-safe)
# ---------------------------------------------------------------------------

def read_raw_events(abs_path, snap: UnmixSnapshot) -> np.ndarray:
    """Read raw events in the experiment's raw channel order.

    Uses the channel whitelist when one is set (as ``Controller.load_sample``
    does) and replaces NaN values with 0.
    """
    from honeychrome.controller_components.functions import sample_from_fcs

    sample = sample_from_fcs(abs_path)
    if snap.whitelisted:
        try:
            raw = sample.get_events(source='raw', col_order=list(snap.pnn_raw))
        except (KeyError, ValueError) as exc:
            logger.warning('read_raw_events: col_order failed for %s (%s); '
                           'reading all channels', abs_path, exc)
            raw = sample.get_events(source='raw')
    else:
        raw = sample.get_events(source='raw')
    raw = np.asarray(raw, dtype=np.float64)
    if np.isnan(raw).any():
        raw = np.where(np.isnan(raw), 0.0, raw)
    return raw


def unmix_events(raw_event_data: np.ndarray, snap: UnmixSnapshot,
                 af: AFState | None, return_af: bool = False) -> dict:
    """Unmix raw events, with AF correction when *af* is given.

    Returns ``{'unmixed', 'af_scale', 'af_idx'}``. The AF arrays are None
    unless *return_af* is set; without AF correction they are zeros.
    """
    n = len(raw_event_data)
    if af is not None:
        result = apply_af_transfer(
            raw_event_data,
            snap.transfer_matrix,
            af.precomputed,
            af.spectra,
            snap.settings,
            filtered_fl_ids_raw=list(snap.fl_ids_raw),
            spillover=snap.spillover,
            af_index_map=af.index_map,
        )
        out = {'unmixed': result['unmixed'], 'af_scale': None, 'af_idx': None}
        if return_af:
            out['af_scale'] = np.asarray(result['af_scale'], dtype=np.float64)
            out['af_idx'] = np.asarray(result['af_idx'], dtype=np.int32)
        return out

    unmixed = raw_event_data @ snap.transfer_matrix
    out = {'unmixed': unmixed, 'af_scale': None, 'af_idx': None}
    if return_af:
        out['af_scale'] = np.zeros(n, dtype=np.float64)
        out['af_idx'] = np.zeros(n, dtype=np.int32)
    return out


def _key_seed(sample_key: str, seed: int) -> int:
    digest = hashlib.blake2b(str(sample_key).encode('utf-8'), digest_size=8).digest()
    return (int.from_bytes(digest, 'little') ^ int(seed)) & 0xFFFFFFFF


def seeded_subsample_indices(n: int, max_events: int | None, sample_key: str,
                             seed: int = 42) -> np.ndarray:
    """Sorted row indices of a reproducible subsample of ``n`` rows.

    The generator is seeded from *seed* and *sample_key*, so each sample's
    subsample is the same whichever order samples are loaded in.
    """
    if max_events is None or max_events <= 0 or n <= max_events:
        return np.arange(n)
    rng = np.random.default_rng(_key_seed(sample_key, seed))
    return np.sort(rng.choice(n, size=int(max_events), replace=False))


def load_unmixed(experiment_dir, sample_key: str, snap: UnmixSnapshot,
                 keep_mask: np.ndarray | None = None,
                 max_events: int | None = None, seed: int = 42,
                 return_af: bool = False) -> dict:
    """Load and unmix one sample with its own AF assignment.

    Parameters
    ----------
    experiment_dir : experiment folder; ``sample_key`` is relative to it.
    sample_key     : ``all_samples`` key.
    snap           : snapshot from ``snapshot_unmix_state``.
    keep_mask      : boolean mask over the file's events; events outside it
                     are dropped before capping. Defaults to the snapshot's
                     time-QC mask for this sample, if any.
    max_events     : optional cap, applied as a seeded random subsample.
    return_af      : also return per-cell AF abundance and index.

    Returns
    -------
    dict with keys ``unmixed`` (n, n_unmixed_channels), ``event_index``
    (row positions in the original file), ``n_events_file``,
    ``n_events_kept`` (after the keep-mask, before the cap), ``af_scale``,
    ``af_idx`` and ``af_profiles`` (tuple of profile names used).

    Raises ValueError if the keep-mask length does not match the file.
    """
    key = str(sample_key)
    raw = read_raw_events(Path(experiment_dir) / key, snap)
    n_file = len(raw)

    mask = keep_mask if keep_mask is not None else snap.qc_by_sample.get(key)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (n_file,):
            raise ValueError(
                f'keep-mask for {key} has {mask.shape[0]} entries but the file '
                f'has {n_file} events'
            )
        kept_index = np.flatnonzero(mask)
    else:
        kept_index = np.arange(n_file)

    sub = seeded_subsample_indices(len(kept_index), max_events, key, seed)
    event_index = kept_index[sub]
    raw = raw[event_index]

    af = snap.af_for(key)
    result = unmix_events(raw, snap, af, return_af=return_af)
    result.update({
        'event_index': event_index,
        'n_events_file': n_file,
        'n_events_kept': int(len(kept_index)),
        'af_profiles': af.profile_names if af is not None else (),
    })
    return result

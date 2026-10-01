"""
autospectral_optimization_functions.py
-----------------------------------------
AutoSpectral Optimization (per-cell fluorophore + AF joint unmixing) for
Honeychrome. Lives entirely inside bundled_plugins — this plugin is self-contained, unlike the
built-in AutoSpectral AF tab which is split across controller_components/
and view_components/.

Public API
----------
discover_all_variants(controller, ...)
    Loops experiment.process['spectral_model']
    and runs discover_fluor_variants() for each fluorophore. Returns
    {'variants': ..., 'raw_pos_thresholds': ..., 'unmixed_pos_thresholds': ...}
    — see the function's own docstring for details.

discover_fluor_variants(...)
    Per-fluorophore port of get.fluor.variants().

calculate_optimize_necessity(spectra, fluor_names, delta_dict, ...)
    Port of calculate.optimize.necessity(). Table section scoring.

unmix_autospectral_optimization(...)
    Assembles the active-variant list from Table state and calls the
    compiled joint kernel. Used by both the Compare and Unmix
    sections.

active_variant_labels / cached_unmixed_thresholds / make_optimization_unmix_fn
    Inputs and per-chunk unmixing function for batch export through
    joint_unmix_export.JointUnmixExporter.

"""

from __future__ import annotations

import logging
import os
import re
import sys

import numpy as np

from honeychrome.controller_components.autospectral_functions import (
    apply_af_unmixing,
    get_af_spectra,
    get_som_codes,
    precompute_af_matrices,
    precompute_joint_cov_extras,
)
from honeychrome.controller_components.functions import sample_from_fcs
from honeychrome.controller_components.spectral_functions import get_raw_events
from honeychrome.controller_components.spectral_cleaning import (
    exclude_saturated,
    knn_scatter_match,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled kernel — same-directory import, since this module and the
# wrapper both live directly in bundled_plugins/.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from autospectral_opt_kernel_wrapper import (   # noqa: E402
    unmix_autospectral_joint,
    AUTOSPECTRAL_OPT_KERNEL_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Small standalone helpers
# ---------------------------------------------------------------------------

def compute_af_pcs_from_unstained(unstained_raw: np.ndarray, n_pcs: int = 4) -> np.ndarray:
    """
    Top-n_pcs AF principal components via SVD directly on a representative
    unstained control file. Entirely self-contained — does not
    depend on Honeychrome's own af_profiles AF spectra.
    """
    mean_vec = unstained_raw.mean(axis=0)
    centered = unstained_raw - mean_vec
    n_pcs_eff = max(1, min(n_pcs, centered.shape[0], centered.shape[1]))
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pcs = vt[:n_pcs_eff]
    norms = np.linalg.norm(pcs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pcs / norms


def project_out_af_pcs(events: np.ndarray, af_pcs: np.ndarray, reference_vec: np.ndarray) -> np.ndarray:
    """
    Unmix `events` against [af_pcs; reference_vec], back-project the AF
    component into raw space, subtract it.
    """
    combined = np.vstack([af_pcs, reference_vec[np.newaxis, :]])       # (n_pcs+1, D)
    P_combined = np.linalg.solve(combined @ combined.T, combined)      # (n_pcs+1, D)
    unmixed_combined = events @ P_combined.T                           # (n_events, n_pcs+1)
    n_pcs = af_pcs.shape[0]
    af_projection = unmixed_combined[:, :n_pcs] @ af_pcs
    return events - af_projection


def cosine_qc_select(
    events: np.ndarray,
    reference_vec: np.ndarray,
    sim_threshold: float = 0.985,
    sim_threshold_floor: float = 0.90,
    min_events: int = 20,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Keep events with cosine similarity to `reference_vec` >= sim_threshold.

    When fewer than `min_events` pass, the threshold is relaxed in 0.01
    steps down to `sim_threshold_floor`, stopping at the first value that
    retains `min_events`. A fluorophore collinear with AF can have its whole
    population sit just under the nominal threshold, where a fixed cutoff
    would discard a real, if noisier, population rather than screen out
    contaminants. Relaxation is logged as a warning.

    Note: this is deliberately not spectral_cleaning.py::cosine_filter(),
    which selects events least similar to a reference (for base-spectrum
    construction) — the opposite selection direction from what's needed here.

    Returns (selected_idx, cosine_values, threshold_used) — cosine_values
    covers every input event, selected_idx indexes into `events`.
    """
    ref_norm = np.linalg.norm(reference_vec) + 1e-9
    row_norm = np.linalg.norm(events, axis=1) + 1e-9
    cosine = (events @ reference_vec) / (row_norm * ref_norm)

    threshold_used = sim_threshold
    selected_idx = np.where(cosine >= sim_threshold)[0]

    if len(selected_idx) < min_events and sim_threshold > sim_threshold_floor:
        n_steps = int(np.floor(round((sim_threshold - sim_threshold_floor) / 0.01, 9)))
        for step in range(1, n_steps + 1):
            relaxed = sim_threshold - 0.01 * step
            candidate_idx = np.where(cosine >= relaxed)[0]
            if len(candidate_idx) >= min_events:
                selected_idx = candidate_idx
                threshold_used = relaxed
                logger.warning(
                    f'cosine_qc_select: fewer than {min_events} events at '
                    f'sim_threshold = {sim_threshold}; relaxed to {relaxed:.3f} to '
                    f'retain {len(selected_idx)} event(s). This fluorophore is likely '
                    f'collinear with autofluorescence, very dim, or otherwise unreliable.'
                )
                break

    return selected_idx, cosine, threshold_used


def compute_positivity_thresholds(raw_events: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """
    Per-channel raw-space positivity threshold from a representative
    unstained sample (get_spectral_variants.R's `raw.thresholds`).
    """
    return np.percentile(raw_events, percentile, axis=0)


def compute_unmixed_positivity_thresholds(
    raw_events: np.ndarray,
    reference_spectra: np.ndarray,
    percentile: float = 99.5,
) -> np.ndarray:
    """
    Per-fluorophore unmixed-space positivity threshold, plain OLS (no AF
    correction). Used for bead controls, where AF extraction is not
    meaningful; cell controls use compute_af_extracted_unmixed_thresholds().
    """
    P_full = np.linalg.solve(reference_spectra @ reference_spectra.T, reference_spectra)
    unmixed = raw_events @ P_full.T
    return np.percentile(unmixed, percentile, axis=0)


def compute_af_extracted_unmixed_thresholds(
    unstained_raw: np.ndarray,
    unstained_subsample: np.ndarray,
    reference_spectra: np.ndarray,
    fluor_names: list,
    percentile: float = 99.5,
) -> np.ndarray:
    """
    Per-fluorophore unmixed-space positivity threshold, AF-extracted.

    Port of get_spectral_variants.R's `unmixed.thresholds`: AF spectra are
    extracted from the unstained sample (get_af_spectra, base stage only),
    the unstained sample is unmixed with per-cell AF extraction, and the
    `percentile` of each fluorophore column is taken. Plain OLS would leave
    autofluorescence smeared across every fluorophore column and inflate the
    threshold.

    unstained_raw       : (N, D) all unstained events, for AF extraction
    unstained_subsample : (n, D) events to unmix for the percentile
    """
    af_spectra = get_af_spectra(unstained_raw, reference_spectra, refine=False)

    if AUTOSPECTRAL_OPT_KERNEL_AVAILABLE and af_spectra.shape[0] >= 2:
        result = unmix_autospectral_joint(
            raw_data_in=unstained_subsample,
            spectra=reference_spectra,
            af_spectra=af_spectra,
            fluor_names=fluor_names,
            pos_thresholds=np.zeros(reference_spectra.shape[0]),
            variants=[],
            n_threads=max(1, (os.cpu_count() or 2) - 1),
        )
        fluor_unmixed = result[:, :reference_spectra.shape[0]]
    else:
        precomputed = precompute_af_matrices(reference_spectra, af_spectra)
        precomputed.update(precompute_joint_cov_extras(precomputed, af_spectra))
        fluor_unmixed = apply_af_unmixing(unstained_subsample, precomputed, af_spectra)['unmixed']

    return np.percentile(fluor_unmixed, percentile, axis=0)


# ---------------------------------------------------------------------------
# Per-fluorophore variant discovery — port of get.fluor.variants()
# ---------------------------------------------------------------------------

def discover_fluor_variants(
    label: str,
    pos_events_raw: np.ndarray,
    pos_scatter: np.ndarray,
    neg_events_raw: np.ndarray | None,
    neg_scatter: np.ndarray | None,
    is_cell_control: bool,
    reference_spectra: np.ndarray,
    fluor_names: list,
    fluor_idx: int,
    peak_ch_idx: int,
    raw_pos_threshold: float,
    unmixed_pos_threshold: float,
    af_pcs: np.ndarray | None,
    saturation_ceiling: float,
    n_cells: int = 10_000,
    som_dim: int = 5,
    k_neighbors: int = 3,
    sim_threshold: float = 0.985,
    sim_threshold_floor: float = 0.90,
    af_collinear_threshold: float = 0.95,
    random_state: int = 0,
) -> dict | None:
    """
    Port of get.fluor.variants().

    Positive events are background-corrected (scatter-matched unstained
    neighbours, or the control's own negatives), AF principal components are
    projected out for cell controls, events still positive after an OLS unmix
    are screened by cosine similarity to the reference spectrum (with
    adaptive relaxation, see cosine_qc_select), and the survivors are
    clustered with a som_dim x som_dim cosine-distance SOM (KMeans fallback,
    see get_som_codes). Each node's spectral code, L-inf normalised and
    shrunk toward the reference in off-peak channels, is a candidate variant.
    Row 0 of the result is always the reference spectrum itself.

    AF projection is skipped when the reference spectrum's cosine
    similarity to any AF principal component reaches af_collinear_threshold:
    a joint fit against near-collinear AF and fluorophore directions can push
    real fluorophore signal into the AF term.

    Returns None when there isn't enough clean signal to characterise
    variation for `label` — caller should fall back to the single reference
    spectrum. Otherwise:
        {'v_mats': (n_variants, D), 'delta': (n_variants, D),
         'delta_norms': (n_variants,), 'n_events_used': int,
         'cosine_threshold_used': float, 'af_collinear': bool}
    """
    reference_vec = np.asarray(reference_spectra[fluor_idx], dtype=np.float64)

    # 1. Saturation exclusion.
    pos_kept, _n_removed, sat_mask = exclude_saturated(pos_events_raw, saturation_ceiling)
    scatter_kept = pos_scatter[~sat_mask]

    # 2. Positivity selection in the (empirical) peak channel, capped at 2*n_cells.
    pos_idx = np.where(pos_kept[:, peak_ch_idx] > raw_pos_threshold)[0]
    if len(pos_idx) > n_cells * 2:
        order = np.argsort(pos_kept[pos_idx, peak_ch_idx])[::-1][: n_cells * 2]
        pos_idx = pos_idx[order]
    if len(pos_idx) < 20:
        logger.info(
            f'discover_fluor_variants: "{label}" — only {len(pos_idx)} positive '
            f'events, falling back to reference spectrum.'
        )
        return None

    # 3. Per-event background subtraction.
    if neg_events_raw is not None and len(neg_events_raw) > 0 and neg_scatter is not None and len(neg_scatter) > 0:
        spectral_sub, _matched_neg_scatter = knn_scatter_match(
            pos_kept[pos_idx], scatter_kept[pos_idx],
            neg_events_raw, neg_scatter, k=k_neighbors,
        )
    else:
        neg_idx = np.setdiff1d(np.arange(len(pos_kept)), pos_idx)
        if len(neg_idx) > 50:
            background = pos_kept[neg_idx].mean(axis=0)
            spectral_sub = pos_kept[pos_idx] - background
        else:
            spectral_sub = pos_kept[pos_idx].copy()

    # 4. AF PC projection — cell controls only, and only when the reference
    #    is not collinear with an AF direction.
    af_collinear = False
    if is_cell_control and af_pcs is not None and len(af_pcs) > 0:
        pc_norms = np.linalg.norm(af_pcs, axis=1) + 1e-9
        ref_norm = np.linalg.norm(reference_vec) + 1e-9
        af_pc_cosine = float(np.max(np.abs(af_pcs @ reference_vec) / (pc_norms * ref_norm)))
        af_collinear = af_pc_cosine >= af_collinear_threshold
        if af_collinear:
            logger.info(
                f'discover_fluor_variants: "{label}" is highly collinear with '
                f'autofluorescence (cosine = {af_pc_cosine:.3f} >= {af_collinear_threshold}); '
                f'skipping AF-component projection.'
            )
        else:
            spectral_sub = project_out_af_pcs(spectral_sub, af_pcs, reference_vec)

    # 5. OLS unmix in full reference-spectra space.
    P_full = np.linalg.solve(reference_spectra @ reference_spectra.T, reference_spectra)
    unmixed_full = spectral_sub @ P_full.T   # (n_events, F)

    # Re-select events still positive after background correction.
    keep_idx = np.where(unmixed_full[:, fluor_idx] > unmixed_pos_threshold * 2)[0]
    if len(keep_idx) < 20:
        logger.info(
            f'discover_fluor_variants: "{label}" — only {len(keep_idx)} events '
            f'positive after correction, falling back to reference spectrum.'
        )
        return None

    # 6. Cosine-similarity QC against the reference spectrum.
    cosine_idx, _cos_vals, threshold_used = cosine_qc_select(
        spectral_sub[keep_idx], reference_vec, sim_threshold, sim_threshold_floor,
    )
    if len(cosine_idx) < 20:
        logger.info(
            f'discover_fluor_variants: "{label}" — only {len(cosine_idx)} events '
            f'passed cosine QC even at {sim_threshold_floor}, falling back to '
            f'reference spectrum.'
        )
        return None

    final_idx = keep_idx[cosine_idx]
    event_n = len(final_idx)

    # 7. SOM clustering (cosine distance), grid shrunk for small event counts.
    if event_n < 500:
        som_dim = max(2, int(np.floor(np.sqrt(event_n / 3))))

    cluster_input = np.concatenate(
        [unmixed_full[final_idx], spectral_sub[final_idx]], axis=1
    )
    codes, engine = get_som_codes(cluster_input, som_dim, dist=4, random_state=random_state)

    n_channels = spectral_sub.shape[1]
    centres_spectral = codes[:, -n_channels:]

    # L-inf normalise (plain max, matching get_fluor_variants.R's `x / max(x)`).
    peak_vals = centres_spectral.max(axis=1, keepdims=True)
    peak_vals = np.where(peak_vals <= 0, 1.0, peak_vals)
    v_mats = centres_spectral / peak_vals
    valid = ~np.isnan(v_mats).any(axis=1)
    v_mats = v_mats[valid]

    if len(v_mats) == 0:
        logger.info(f'discover_fluor_variants: "{label}" — no valid SOM codes, falling back to reference spectrum.')
        return None

    # 8. Off-peak shrinkage: blend 50/50 toward the reference in channels
    #    where the reference contributes < 5% of its own peak.
    peak_mask = reference_vec > 0.05
    v_mats_shrunk = v_mats.copy()
    v_mats_shrunk[:, ~peak_mask] = 0.5 * v_mats[:, ~peak_mask] + 0.5 * reference_vec[~peak_mask]

    # Row 0 is always the reference spectrum itself.
    v_mats_shrunk = np.vstack([reference_vec[np.newaxis, :], v_mats_shrunk])

    # 9. Delta matrix + norms.
    delta = v_mats_shrunk - reference_vec[np.newaxis, :]
    delta_norms = np.linalg.norm(delta, axis=1)

    logger.info(
        f'discover_fluor_variants: "{label}" — {len(v_mats_shrunk) - 1} variant(s) '
        f'from {event_n} qualifying events ({som_dim}x{som_dim} {engine}).'
    )
    return {
        'v_mats': v_mats_shrunk,
        'delta': delta,
        'delta_norms': delta_norms,
        'n_events_used': event_n,
        'cosine_threshold_used': threshold_used,
        'af_collinear': af_collinear,
    }


def fluorescence_channel_names(controller) -> list:
    """
    Detector names in the same order as reference_spectra's columns (and
    every array returned by discover_fluor_variants/discover_all_variants).
    Used by the Plot section for x-axis tick labels.
    """
    raw_settings = controller.experiment.settings['raw']
    full_pnn_raw = raw_settings['event_channels_pnn']
    pnn_raw = raw_settings.get('whitelisted_pnn') or full_pnn_raw
    fl_ids = [pnn_raw.index(full_pnn_raw[i]) for i in controller.filtered_raw_fluorescence_channel_ids]
    return [pnn_raw[i] for i in fl_ids]


def _is_bead_sample(path: str, name: str) -> bool:
    """Mirrors spectral_controller.py::get_unstained_negative's _is_bead()."""
    return bool(re.search(r'bead', name, re.IGNORECASE) or re.search(r'bead', path, re.IGNORECASE))


def _resolve_unstained_cell_sample_names(samples: dict) -> list:
    """
    Returns the names of every unstained cell sample (all matches, not just
    the first), so discover_all_variants() can take a per-channel median
    across them rather than depending on a single, possibly unrepresentative,
    sample. Mirrors spectral_controller.py::get_unstained_negative's own
    manually-tagged-first / regex-fallback / Cells-vs-Beads split, but
    collects every Cells match instead of returning only the first.
    """
    all_samples = samples.get('all_samples', {})
    manually_unstained = set(samples.get('unstained_samples', []))

    def _is_unstained(path: str, name: str) -> bool:
        return (path in manually_unstained
                or 'unstained' in path.lower()
                or 'unstained' in name.lower())

    names = []
    for path, name in all_samples.items():
        if _is_unstained(path, name) and not _is_bead_sample(path, name):
            names.append(name)
    return names


def _unstained_threshold_entry(
    unstained_raw: np.ndarray,
    reference_spectra: np.ndarray,
    fluor_names: list,
    is_cell_control: bool,
    n_af_pcs: int,
    subsample_n: int = 100_000,
    random_state: int = 0,
) -> dict:
    """
    Positivity thresholds and AF-PC basis from one unstained sample.

    Raw thresholds, AF PCs and the unmixed percentile use at most
    `subsample_n` events (get_spectral_variants.R's gate.downsample.n.cells);
    AF spectra for the unmixed thresholds are extracted from every event.
    Cell samples get AF-extracted unmixed thresholds; bead samples plain OLS,
    since AF extraction on beads isn't meaningful and discover_fluor_variants()
    never projects AF out of bead controls.
    """
    subsample = unstained_raw
    if len(subsample) > subsample_n:
        rng = np.random.default_rng(random_state)
        subsample = subsample[np.sort(rng.choice(len(subsample), subsample_n, replace=False))]

    if is_cell_control:
        af_pcs = compute_af_pcs_from_unstained(subsample, n_pcs=n_af_pcs)
        unmixed = compute_af_extracted_unmixed_thresholds(
            unstained_raw, subsample, reference_spectra, fluor_names,
        )
    else:
        af_pcs = None
        unmixed = compute_unmixed_positivity_thresholds(subsample, reference_spectra)

    return {
        'raw': compute_positivity_thresholds(subsample),
        'unmixed': unmixed,
        'af_pcs': af_pcs,
    }


def discover_all_variants(
    controller,
    n_cells: int = 10_000,
    som_dim: int = 5,
    k_neighbors: int = 3,
    sim_threshold: float = 0.985,
    sim_threshold_floor: float = 0.90,
    af_collinear_threshold: float = 0.95,
    n_af_pcs: int = 4,
    progress_callback=None,
) -> dict:
    """
    Setup section orchestration. Loops
    experiment.process['spectral_model'] (excluding AF and any
    "Unstained"-labelled entries) and runs discover_fluor_variants() for
    each fluorophore.

    progress_callback(n, total, label), if supplied, is called once per
    fluorophore — safe to connect to bus.progress.emit from a QThread run()
    method (do not call Qt widgets directly from here).

    Returns {'variants': dict, 'raw_pos_thresholds': (D,) ndarray,
    'unmixed_pos_thresholds': (F,) ndarray}. 'variants' is keyed by
    fluorophore label; labels absent from it fell back to the single
    reference spectrum (insufficient clean signal — not an error, just
    nothing to optimise for that fluorophore). The threshold arrays are the
    per-channel/per-fluorophore median across all unstained cell samples —
    callers should cache these (see autospectral_optimization_tab.py's Setup
    completion handler) and reuse them for Compare/Unmix instead of
    recomputing from whatever stained sample is currently being processed.
    """
    raw_settings = controller.experiment.settings['raw']
    full_pnn_raw = raw_settings['event_channels_pnn']
    pnn_raw = raw_settings.get('whitelisted_pnn') or full_pnn_raw
    col_order = raw_settings.get('whitelisted_pnn')
    # filtered_raw_fluorescence_channel_ids (like scatter_channel_ids) indexes
    # the FULL event_channels_pnn list; remap to positions within pnn_raw
    # before using as a column selector against whitelisted-PNN-ordered
    # arrays — same pattern as controller.py::initialise_transfer_matrix().
    fl_ids = [pnn_raw.index(full_pnn_raw[i]) for i in controller.filtered_raw_fluorescence_channel_ids]
    sc_ids = [pnn_raw.index(full_pnn_raw[i]) for i in raw_settings['scatter_channel_ids']]
    saturation_ceiling = raw_settings.get('magnitude_ceiling') or 2 ** 18

    spectral_model = controller.experiment.process.get('spectral_model', [])
    profiles = controller.experiment.process.get('profiles', {})
    fluor_names_all = [c['label'] for c in spectral_model if c['label'] in profiles]
    reference_spectra = np.array([profiles[name] for name in fluor_names_all])

    all_samples = controller.experiment.samples.get('all_samples', {})
    all_samples_rev = {v: k for k, v in all_samples.items()}
    raw_gating = controller.raw_gating
    experiment_dir = controller.experiment_dir

    def _load_gated(sample_name, gate_label):
        rel_path = all_samples_rev.get(sample_name)
        if rel_path is None:
            return None, None
        full_path = str(experiment_dir / rel_path)
        sample = sample_from_fcs(full_path)
        fluor, scatter = get_raw_events(
            sample, fl_ids, gate_label=gate_label, gating_strategy=raw_gating,
            extra_channel_ids=sc_ids, col_order=col_order,
        )
        return fluor, scatter

    # All unstained cell samples -> per-channel/per-fluorophore positivity
    # thresholds (median across samples, used as the fallback below), plus
    # one AF-PC basis per sample. Cached in threshold_cache (keyed by sample
    # name) so per-control pairing below can prefer a specific sample's own
    # threshold/AF-PC entry instead of the median.
    unstained_names = _resolve_unstained_cell_sample_names(controller.experiment.samples)
    if not unstained_names:
        raise ValueError(
            'discover_all_variants: no unstained cell sample found. Mark a cell '
            'control as Unstained (right-click in the sample panel) before running Setup.'
        )

    threshold_cache = {}   # sample name -> {'raw': (D,), 'unmixed': (F,), 'af_pcs': (n_pcs, D)}
    for name in unstained_names:
        unstained_raw, _unstained_scatter = _load_gated(name, 'root')
        if unstained_raw is None or len(unstained_raw) == 0:
            logger.warning(f'discover_all_variants: could not load events for unstained sample "{name}" — skipping.')
            continue
        threshold_cache[name] = _unstained_threshold_entry(
            unstained_raw, reference_spectra, fluor_names_all, True, n_af_pcs,
        )

    if not threshold_cache:
        raise ValueError(
            'discover_all_variants: none of the unstained cell samples '
            f'({", ".join(unstained_names)}) could be loaded.'
        )

    raw_pos_thresholds = np.median(np.vstack([e['raw'] for e in threshold_cache.values()]), axis=0)
    unmixed_pos_thresholds = np.median(np.vstack([e['unmixed'] for e in threshold_cache.values()]), axis=0)
    af_pcs_by_unstained = {name: e['af_pcs'] for name, e in threshold_cache.items()}
    logger.info(
        f'discover_all_variants: positivity thresholds computed from '
        f'{len(threshold_cache)} unstained cell sample(s): {", ".join(threshold_cache.keys())}.'
    )

    def _resolve_control_threshold_entry(sample_name, is_cell_control):
        """
        Per-control paired-unstained threshold resolution: prefer the
        control's own paired unstained sample (universal_negative_name),
        loading and caching it on demand if it wasn't already picked up as
        a generic unstained-cell sample above (e.g. its name doesn't match
        the "unstained" pattern even though it's assigned as a pairing).
        Returns None — caller falls back to the median-across-all entry —
        when there's no pairing or it can't be loaded.

        `is_cell_control` selects which unmixed-threshold calculation is
        used: AF-extracted for cell controls, matching
        `get_spectral_variants.R`'s `unmixed.thresholds`; plain OLS for
        bead controls (see _unstained_threshold_entry()).
        Cell and bead entries for the same underlying sample name are
        cached separately so a sample used as both a cell and bead pairing
        (unusual, but not impossible) can't return the wrong entry type.
        """
        if not sample_name:
            return None
        cache_key = sample_name if is_cell_control else f'{sample_name}::beads'
        if cache_key in threshold_cache:
            return threshold_cache[cache_key]
        paired_raw, _paired_scatter = _load_gated(sample_name, 'root')
        if paired_raw is None or len(paired_raw) == 0:
            return None
        entry = _unstained_threshold_entry(
            paired_raw, reference_spectra, fluor_names_all, is_cell_control, n_af_pcs,
        )
        threshold_cache[cache_key] = entry
        return entry

    results: dict = {}
    controls = [
        c for c in spectral_model
        if c.get('label') and c['label'] != 'AF'
        and 'unstained' not in c['label'].lower()
        and c['label'] in fluor_names_all
    ]

    for n, control in enumerate(controls):
        label = control['label']
        if progress_callback:
            progress_callback(n, len(controls), label)

        sample_name = control.get('sample_name')
        if not sample_name:
            logger.info(f'discover_all_variants: "{label}" has no sample assigned — skipping.')
            continue

        pos_raw, pos_scatter = _load_gated(sample_name, 'Singlets')
        if pos_raw is None or len(pos_raw) == 0:
            logger.warning(f'discover_all_variants: no events for "{label}" ({sample_name}) — skipping.')
            continue

        universal_neg_name = control.get('universal_negative_name') or ''
        neg_raw, neg_scatter = (None, None)
        if universal_neg_name:
            neg_raw, neg_scatter = _load_gated(universal_neg_name, 'Neg Unstained')
            if neg_raw is None or len(neg_raw) == 0:
                neg_raw, neg_scatter = _load_gated(universal_neg_name, 'root')

        is_cell_control = str(control.get('particle_type', 'cells')).lower() == 'cells'

        fluor_idx = fluor_names_all.index(label)
        peak_channel_name = control.get('gate_channel')
        try:
            peak_ch_idx = [pnn_raw[i] for i in fl_ids].index(peak_channel_name)
        except (ValueError, TypeError):
            peak_ch_idx = int(np.argmax(reference_spectra[fluor_idx]))

        # Prefer this control's own paired unstained sample for both
        # thresholds and the AF-PC basis (Major-Channel raw threshold +
        # AF-extracted unmixed threshold, cells only — beads get plain-OLS
        # thresholds, see _resolve_control_threshold_entry()); fall back to
        # the median-across-all-unstained-cell-samples entry when there's
        # no pairing or it can't be loaded.
        paired_entry = _resolve_control_threshold_entry(universal_neg_name, is_cell_control)
        if paired_entry is not None:
            control_raw_pos_threshold = paired_entry['raw'][peak_ch_idx]
            control_unmixed_pos_threshold = paired_entry['unmixed'][fluor_idx]
            af_pcs = paired_entry['af_pcs']
        else:
            control_raw_pos_threshold = raw_pos_thresholds[peak_ch_idx]
            control_unmixed_pos_threshold = unmixed_pos_thresholds[fluor_idx]
            af_pcs = next(iter(af_pcs_by_unstained.values()), None) if is_cell_control else None

        variant_result = discover_fluor_variants(
            label=label,
            pos_events_raw=pos_raw,
            pos_scatter=pos_scatter,
            neg_events_raw=neg_raw,
            neg_scatter=neg_scatter,
            is_cell_control=is_cell_control,
            reference_spectra=reference_spectra,
            fluor_names=fluor_names_all,
            fluor_idx=fluor_idx,
            peak_ch_idx=peak_ch_idx,
            raw_pos_threshold=control_raw_pos_threshold,
            unmixed_pos_threshold=control_unmixed_pos_threshold,
            af_pcs=af_pcs,
            saturation_ceiling=saturation_ceiling,
            n_cells=n_cells, som_dim=som_dim, k_neighbors=k_neighbors,
            sim_threshold=sim_threshold, sim_threshold_floor=sim_threshold_floor,
            af_collinear_threshold=af_collinear_threshold,
        )
        if variant_result is not None:
            results[label] = variant_result

    if progress_callback:
        progress_callback(len(controls), len(controls), 'Done')

    return {
        'variants': results,
        'raw_pos_thresholds': raw_pos_thresholds,
        'unmixed_pos_thresholds': unmixed_pos_thresholds,
    }


# ---------------------------------------------------------------------------
# Optimisation-necessity scoring — port of calculate.optimize.necessity()
# ---------------------------------------------------------------------------

def calculate_optimize_necessity(
    spectra: np.ndarray,
    fluor_names: list,
    delta_dict: dict,
    mu: dict | None = None,
    threshold: float = 0.01,
    ridge: float = 1e-4,
) -> dict:
    """
    Port of calculate.optimize.necessity(). `delta_dict` maps
    fluorophore label -> delta matrix (n_variants x D) — the 'delta' key
    from discover_fluor_variants()'s output for that label.

    `mu`, if supplied, maps label -> per-fluorophore MFI from a
    representative stained sample; scores are multiplied by this (clamped to
    >= 0) *after* the geometric score is computed but *before* normalisation,
    matching the R implementation. Optional — pass None to use the geometric
    score alone (this is the current Table section's default; wiring up an
    automatic `mu` from a representative sample is a reasonable follow-up but
    is not done here).

    Returns {'scores_raw': {...}, 'scores_norm': {...}, 'optimize_recommended': {...}},
    each keyed by fluorophore label, restricted to labels present in `delta_dict`.
    """
    score_fluors = [f for f in fluor_names if f in delta_dict]
    if not score_fluors:
        return {'scores_raw': {}, 'scores_norm': {}, 'optimize_recommended': {}}

    scores_raw = {f: 0.0 for f in score_fluors}

    for fl in score_fluors:
        delta = delta_dict[fl]
        if delta is None or delta.size == 0:
            continue
        delta_norms = np.linalg.norm(delta, axis=1)
        if np.all(delta_norms < 1e-12):
            continue

        other_idx = [i for i, name in enumerate(fluor_names) if name != fl]
        if not other_idx:
            continue
        S_nof = spectra[other_idx]   # (F-1, D)
        try:
            U_nof = np.linalg.solve(S_nof @ S_nof.T, S_nof)
        except np.linalg.LinAlgError:
            # Moore-Penrose fallback via SVD, matching the R tryCatch branch.
            U_nof = np.linalg.pinv(S_nof.T).T

        if delta.shape[0] > 1:
            delta_cov = np.cov(delta, rowvar=False)
        else:
            delta_cov = (delta.T @ delta) / max(delta.shape[0] - 1, 1)
        delta_cov = delta_cov + ridge * np.eye(delta_cov.shape[0])

        leakage_cov = U_nof @ delta_cov @ U_nof.T
        scores_raw[fl] = float(np.sum(np.sqrt(np.abs(np.diag(leakage_cov)))))

    if mu:
        for fl in score_fluors:
            if fl in mu:
                scores_raw[fl] *= max(float(mu[fl]), 0.0)

    max_score = max(scores_raw.values()) if scores_raw else 0.0
    if max_score > 0:
        scores_norm = {f: v / max_score for f, v in scores_raw.items()}
    else:
        # All scores zero — every fluorophore is geometrically independent;
        # default to "not recommended" for all (matches R's fallback branch).
        scores_norm = dict(scores_raw)

    optimize_recommended = {f: (scores_norm[f] >= threshold) for f in score_fluors}

    return {
        'scores_raw': scores_raw,
        'scores_norm': scores_norm,
        'optimize_recommended': optimize_recommended,
    }


# ---------------------------------------------------------------------------
# Joint kernel orchestration — used by both the Compare and Unmix sections
# ---------------------------------------------------------------------------

def unmix_autospectral_optimization(
    raw_fl_events: np.ndarray,
    reference_spectra: np.ndarray,
    fluor_names: list,
    af_spectra: np.ndarray,
    variants_meta: dict,
    active_labels,
    unmixed_pos_thresholds: np.ndarray,
    n_passes: int = 1,
    n_threads: int | None = None,
    cell_weight: bool = False,
    noise_floor: float = 125.0,
    alpha: float = 0.5,
    collinear_thresh: float = 0.5,
    joint_pair_resolution: bool = True,
    n_af_passes: int = 1,
    refine_af_quantile: float = 0.5,
    exact_variant_scan: bool = False,
) -> dict:
    """
    Assembles the `variants` list from the Table section's active
    fluorophores and calls the compiled joint kernel.

    Parameters
    ----------
    raw_fl_events : (N, D) float64 raw fluorescence events.
    reference_spectra : (F, D) reference fluorophore spectra (no AF row).
    fluor_names   : length F, row order matching reference_spectra.
    af_spectra    : (nAF, D) AF spectra for the current sample (nAF >= 2 —
                    the joint core cannot fall back to plain OLS; the Table/
                    Compare/Unmix sections must gate on an assigned AF
                    profile before calling this.
    variants_meta : controller.autospectral_variants — dict keyed by
                    fluorophore label, each value from discover_fluor_variants().
    active_labels : iterable of fluorophore labels currently checked 'Active'
                    in the Table (defaults to optimize_recommended).
                    Labels with no entry in variants_meta are silently
                    skipped (AF-only contribution from that fluorophore).
    unmixed_pos_thresholds : (F,) float64, unmixed-space positivity
                    thresholds — the cached, AF-extracted thresholds from
                    discover_all_variants() (see
                    compute_af_extracted_unmixed_thresholds()). Callers with
                    no cached value available (Setup hasn't been run this
                    session) should pass all-zeros rather than recomputing
                    anything from the sample being processed — see
                    autospectral_optimization_tab.py's Compare/Unmix
                    fallback paths.

    Returns
    -------
    {'unmixed': (N, F), 'af_scale': (N,), 'af_idx': (N,) 1-based}
    """
    if not AUTOSPECTRAL_OPT_KERNEL_AVAILABLE:
        raise ImportError(
            'Compiled AutoSpectral Optimization kernel not available. '
            'Run build_autospectral_opt_kernel.py first.'
        )

    variants = []
    for label in active_labels:
        entry = variants_meta.get(label)
        if entry is None:
            continue
        variants.append({
            'name': label,
            'v_mats': entry['v_mats'],
            'delta_obs': entry['delta'],
        })

    cpu_count = os.cpu_count() or 2
    if n_threads is None:
        n_threads = max(1, cpu_count - 1)
    else:
        # Programmatic-use safety net — the UI also caps this, but this
        # function has its own callers (e.g. scripted batch export).
        n_threads = max(1, min(int(n_threads), cpu_count))

    result = unmix_autospectral_joint(
        raw_data_in=raw_fl_events,
        spectra=reference_spectra,
        af_spectra=af_spectra,
        fluor_names=fluor_names,
        pos_thresholds=unmixed_pos_thresholds,
        variants=variants,
        n_passes=n_passes,
        n_threads=n_threads,
        cell_weight=cell_weight,
        noise_floor=noise_floor,
        alpha=alpha,
        collinear_thresh=collinear_thresh,
        joint_pair_resolution=joint_pair_resolution,
        n_af_passes=n_af_passes,
        refine_af_quantile=refine_af_quantile,
        exact_variant_scan=exact_variant_scan,
    )
    F = reference_spectra.shape[0]
    return {
        'unmixed':  result[:, :F],
        'af_scale': result[:, F],
        'af_idx':   result[:, F + 1].astype(np.int64),
    }


# ---------------------------------------------------------------------------
# Batch export inputs — used with joint_unmix_export.JointUnmixExporter
# ---------------------------------------------------------------------------

def active_variant_labels(process: dict) -> set:
    """Fluorophores marked Active in the Table section."""
    meta = process.get('autospectral_variants_meta', {}) or {}
    return {label for label, m in meta.items() if m.get('active')}


def has_active_variants(variants_meta: dict, active_labels) -> bool:
    """True when at least one Active fluorophore has computed variants, i.e.
    the kernel does per-cell variant optimisation rather than AF only."""
    return any(label in variants_meta for label in active_labels)


def cached_unmixed_thresholds(controller, reference_spectra: np.ndarray) -> np.ndarray:
    """
    The unstained-derived unmixed-space positivity thresholds cached by
    Setup, or all-zero thresholds (with a warning) when Setup has not been
    run or the cache no longer matches the spectral model. Thresholds are
    never recomputed from the stained samples being exported.
    """
    cached = getattr(controller, 'autospectral_unmixed_pos_thresholds', None)
    if (cached is not None and reference_spectra is not None
            and len(cached) == reference_spectra.shape[0]):
        return np.asarray(cached, dtype=np.float64)
    logger.warning(
        'AutoSpectral Optimization export: no cached unstained-derived '
        'thresholds (run Setup first) — falling back to all-zero '
        'thresholds rather than recomputing from each exported (stained) sample.'
    )
    n_fluors = 0 if reference_spectra is None else reference_spectra.shape[0]
    return np.zeros(n_fluors)


def make_optimization_unmix_fn(
    reference_spectra: np.ndarray,
    fluor_names: list,
    variants_meta: dict,
    active_labels,
    unmixed_pos_thresholds: np.ndarray,
    kernel_kwargs: dict,
):
    """
    Per-chunk unmixing function for JointUnmixExporter:
    ``unmix_fn(raw_fl_chunk, raw_chunk, af)`` runs
    unmix_autospectral_optimization() with the sample's AF library
    (``af.spectra``). ``raw_chunk`` is not used.
    """
    fluor_names = list(fluor_names)
    active_labels = set(active_labels)

    def unmix_fn(raw_fl_chunk, raw_chunk, af):
        return unmix_autospectral_optimization(
            raw_fl_events=raw_fl_chunk,
            reference_spectra=reference_spectra,
            fluor_names=fluor_names,
            af_spectra=af.spectra,
            variants_meta=variants_meta,
            active_labels=active_labels,
            unmixed_pos_thresholds=unmixed_pos_thresholds,
            **kernel_kwargs,
        )

    return unmix_fn

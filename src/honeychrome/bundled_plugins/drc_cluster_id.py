"""
drc_cluster_id.py — Cluster ID suggestions (MEM + cell-type scoring) for the
DR/Clustering plugin
================================================================================
Companion module to ``dr_clustering_tab.py`` (filename intentionally does NOT
end in ``_tab.py``, so it is not picked up as a separate plugin tab).

Two independent scoring mechanisms feed the Cluster Annotation tab's Item 15
"Cluster ID Suggestions" controls:

  1. MEM (Marker Enrichment Modeling) -- ported from cluster_id.md.
     calculate_mem_scores() -> generate_mem_labels(). A descriptive
     statistic of the cluster's own data; safe to auto-adopt.
  2. Cell-type scoring -- ported from flow_cluster_id_score.R (your own
     scType-derived adaptation; original scType by Aleksandr Ianevski,
     GNU GPL-3.0, https://github.com/IanevskiAleksandr/sc-type). Looks MEM
     scores up against a marker-signature database
     (drc_cell_type_database.csv); a biological CLAIM rather than a computed
     statistic, so the tab only ever displays this as a suggestion.

Values are TRANSFORMED using each channel's configured transform
(state.channel_transform_params -- the same logicle/log/linear transform
the Transforms tab and clustering itself use), via
drc_pipeline.load_sample_transformed_values(). This is a deliberate
REVERSAL of this module's original choice to mirror
drc_stats.compute_mfis's log1p(max(raw, 0)) convention -- that worked for
MFI significance testing (a robust summary statistic across sample
groups), but for MEM's positive/negative separation, the scale needs to
match what the Transforms tab (and clustering itself) actually considers
positive/negative for a channel, not a generic log1p over the full raw
dynamic range. A channel with no transform configured yet in the
Transforms tab falls back to whatever transform_channels()'s default
resolves to (id=1 / logicle, matching _build_flowkit_transform's default)
rather than failing.

Marker naming is enforced in two tiers, both keyed off the Spectral
Process tab's Antigen field (never overwritten -- only read and, for
scoring, canonicalised at lookup time):
  - channels_missing_antigen() -- HARD gate. Any checked channel with no
    Antigen text at all blocks the whole computation (checked by the tab
    before calling compute_cluster_id_suggestions). Without this, a blank
    Antigen silently fell back to the channel's fluorophore Label in MEM
    Label output, mixing marker names and fluorophore names in the same
    run with no indication why.
  - build_channel_marker_map()'s `unmatched` return -- SOFT warning. An
    Antigen that IS filled in but doesn't match anything in
    marker_database.csv contributes nothing to cell-type scoring (no
    cell_type_database.csv row references arbitrary free text); the tab
    surfaces this as a non-blocking warning after computing, rather than
    letting it fail silently.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

import drc_pipeline
from drc_logging import get_logger, log_stage

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pooling -- same pattern as ClusterAnnotationTab._plot_violins /
# drc_stats.compute_mfis, but returns full concatenated per-cluster arrays
# (not just means) since MEM needs medians and IQRs, not just a mean.
# ---------------------------------------------------------------------------

def pool_cluster_marker_values(controller, state, cl_run: dict,
                                channels: list[str],
                                progress_callback=None,
                                af_state=None,
                                max_events_per_cluster: int | None = 1000,
                                seed: int = 42) -> dict[str, dict[int, np.ndarray]]:
    """
    Pool TRANSFORMED marker values (state.channel_transform_params -- the
    SAME per-channel transform the Transforms tab and clustering itself
    use) per (channel, cluster) across cl_run's OWN training samples, split
    by cl_run's OWN per-sample label arrays -- never the "currently active"
    globals, matching _plot_violins's per-run isolation. Noise (cluster id
    < 0) is excluded, same as the violin plots.

    progress_callback: optional callable(n_samples_done: int), invoked
    after EACH training sample finishes loading (whether or not it
    contributed data) -- lets the tab drive a progress bar without a
    background thread, since loading + unmixing each sample from disk is
    the dominant cost here, not the in-memory MEM math afterward.

    af_state: optional AF snapshot (transfer_matrix, af_precomputed,
    af_spectra) -- see drc_pipeline.apply_unmixing_af_aware()'s docstring.
    Pass this when calling from a background worker thread; leave as None
    only for main-thread callers.

    max_events_per_cluster: caps each (channel, cluster) pooled array at
    this many events via random downsampling (default 1000). MEM's
    median/IQR and the Otsu threshold are order statistics that don't
    need every event, and calculate_mem_scores in particular rebuilds
    and np.percentile-sorts a REF array of every OTHER cluster's events
    for EVERY (channel, cluster) pair -- left uncapped, that cost scales
    with the full pooled event count, not a fixed small number.
    Downsampled INDEPENDENTLY per (channel, cluster) with a fixed seed --
    safe because every downstream stat here (MEM, cluster medians, Otsu
    thresholds) is a per-channel MARGINAL statistic; nothing compares
    events across channels, so there's no requirement that the same
    events survive downsampling in every channel. Pass None to disable
    (use every pooled event -- the previous behaviour).

    seed: numpy default_rng seed for the downsampling above. Fixed at 42
    (matching drc_pipeline.load_training_pool's own downsampling
    convention) so repeated runs against the same run/channels are
    reproducible.

    Returns {channel: {cluster_id: concatenated (and possibly
    downsampled) np.ndarray}}. A channel/cluster combination with no
    data is simply absent from the inner dict (callers must use .get()).
    """
    labels_dict = cl_run.get('labels', {}) or {}
    training_samples = cl_run.get('training_sample_ids', [])
    pooled: dict[str, dict[int, list]] = {ch: {} for ch in channels}

    log.info(
        "pool_cluster_marker_values: pooling %d channel(s) across %d training "
        "sample(s), transformed scale",
        len(channels), len(training_samples),
    )
    for i, rel in enumerate(training_samples):
        mv = drc_pipeline.load_sample_transformed_values(controller, state, rel, channels, af_state=af_state)
        if mv is None:
            log.warning("pool_cluster_marker_values: %s -- could not load transformed values, skipped", rel)
            if progress_callback is not None:
                progress_callback(i + 1)
            continue
        values, names = mv
        labels = labels_dict.get(rel)
        if labels is None:
            log.warning("pool_cluster_marker_values: %s -- no cluster labels recorded for this sample, skipped", rel)
            if progress_callback is not None:
                progress_callback(i + 1)
            continue
        labels = np.asarray(labels)
        m = min(len(values), len(labels))
        if m != len(values) or m != len(labels):
            log.warning(
                "pool_cluster_marker_values: %s -- values (%d) vs labels (%d) "
                "length mismatch, truncating to %d.",
                rel, len(values), len(labels), m,
            )
        values, labels = values[:m], labels[:m]
        for ch in channels:
            if ch not in names:
                continue
            col = values[:, names.index(ch)]
            for cl_id in np.unique(labels):
                if cl_id < 0:
                    continue
                pooled[ch].setdefault(int(cl_id), []).append(col[labels == cl_id])
        if progress_callback is not None:
            progress_callback(i + 1)

    rng = np.random.default_rng(seed)
    result: dict[str, dict[int, np.ndarray]] = {}
    n_capped = 0
    n_total = 0
    for ch, by_cl in pooled.items():
        result[ch] = {}
        for cl, vals in by_cl.items():
            if not vals:
                continue
            arr = np.concatenate(vals)
            n_total += 1
            if max_events_per_cluster is not None and len(arr) > max_events_per_cluster:
                idx = rng.choice(len(arr), size=max_events_per_cluster, replace=False)
                arr = arr[idx]
                n_capped += 1
            result[ch][cl] = arr

    for ch, by_cl in result.items():
        if not by_cl:
            log.warning(
                "pool_cluster_marker_values: channel %r has NO pooled data for "
                "ANY cluster -- missing from every training sample's "
                "transformed columns?", ch,
            )
        else:
            log.debug(
                "  %-22s pooled %d cluster(s), n_events: %s",
                ch, len(by_cl), {cl: len(v) for cl, v in by_cl.items()},
            )
    log.info(
        "pool_cluster_marker_values: capped %d/%d (channel, cluster) cell(s) "
        "to <= %s events (seed=%d)",
        n_capped, n_total, max_events_per_cluster, seed,
    )
    return result


# ---------------------------------------------------------------------------
# MEM scoring (ported from cluster_id.md)
# ---------------------------------------------------------------------------

def calculate_mem_scores(pooled: dict[str, dict[int, np.ndarray]],
                          iqr_floor: float = 0.5) -> pd.DataFrame:
    """
    Marker Enrichment Modeling score per (cluster, channel), ported from
    cluster_id.md's calculate_mem_scores(). POP = this cluster's pooled
    values for a channel; REF = every OTHER cluster's pooled values for
    that channel (auto-reference), same background definition as the
    original. This is a PER-CHANNEL comparison only -- POP and REF never
    mix data from other channels; the only place channels interact at all
    is the final rescale below.

        MAG       = median(POP) - median(REF)
        IQR_Ratio = IQR(REF) / max(IQR(POP), iqr_floor)
        MEM_raw   = MAG * IQR_Ratio

    Scaled to a discrete [-10, +10] scale by dividing by the matrix-wide
    max absolute value, exactly as the original. This IS a cross-channel,
    cross-cluster coupling: a single outlier (cluster, channel) cell here
    compresses every OTHER cell's scaled score once divided through, which
    can produce widespread "Uncharacterized" MEM Labels that have nothing
    to do with the threshold setting. See the logging below, which
    identifies exactly which cell produced that max.

    Returns a (cluster x channel) DataFrame; a cell is NaN if that cluster
    had no pooled data for that channel (e.g. the channel wasn't in this
    sample's PnN list). NaN cells are dropped downstream by
    generate_mem_labels() and score_cell_types(), never treated as 0.
    """
    channels = list(pooled.keys())
    clusters = sorted({cl for by_cl in pooled.values() for cl in by_cl})
    if not clusters or not channels:
        return pd.DataFrame(index=clusters, columns=channels, dtype=float)

    mem_raw = pd.DataFrame(index=clusters, columns=channels, dtype=float)
    for ch in channels:
        by_cl = pooled[ch]
        for cl in clusters:
            pop = by_cl.get(cl)
            if pop is None or len(pop) == 0:
                continue
            ref = np.concatenate([v for other, v in by_cl.items()
                                  if other != cl and len(v)])
            if len(ref) == 0:
                continue
            mag = float(np.median(pop) - np.median(ref))
            iqr_pop = float(np.percentile(pop, 75) - np.percentile(pop, 25))
            iqr_ref = float(np.percentile(ref, 75) - np.percentile(ref, 25))
            iqr_pop_clipped = max(iqr_pop, iqr_floor)
            mem_raw.loc[cl, ch] = mag * (iqr_ref / iqr_pop_clipped)
            log.debug(
                "  MEM raw  cluster=%s channel=%-22s mag=% .4g iqr_pop=% .4g "
                "iqr_ref=% .4g (floor=%.2g) -> raw=% .4g",
                cl, ch, mag, iqr_pop, iqr_ref, iqr_floor, mem_raw.loc[cl, ch],
            )

    arr = mem_raw.to_numpy(dtype=float)
    if np.all(np.isnan(arr)):
        log.warning("calculate_mem_scores: no non-NaN cells -- returning unscaled (all-NaN) matrix")
        return mem_raw
    n_nan = int(np.count_nonzero(np.isnan(arr)))
    global_max_abs = float(np.nanmax(np.abs(arr)))
    flat_idx = int(np.nanargmax(np.abs(arr)))
    outlier_cl, outlier_ch = np.unravel_index(flat_idx, arr.shape)
    log.info(
        "calculate_mem_scores: %d/%d cell(s) NaN, global_max_abs=%.4g "
        "(cluster=%s channel=%s) -- EVERY score is divided by this before "
        "scaling to [-10, 10], so an unusually large value here compresses "
        "every other cell's scaled score toward zero",
        n_nan, arr.size, global_max_abs,
        mem_raw.index[outlier_cl], mem_raw.columns[outlier_ch],
    )
    if global_max_abs > 0:
        mem_scaled = (mem_raw / global_max_abs * 10).round()
    else:
        mem_scaled = mem_raw
    log.info("MEM matrix: %s", mem_scaled.shape)
    return mem_scaled


def calculate_cluster_medians(pooled: dict[str, dict[int, np.ndarray]]) -> pd.DataFrame:
    """
    Per-cluster ABSOLUTE median of the TRANSFORMED marker values in
    `pooled` -- no reference cluster, no rescale, no comparison of any
    kind between clusters. This is what score_cell_types takes as input,
    matching flow_cluster_id_score.R's own "thresholded flow expression
    matrix" convention: the biexponential/logicle transform already puts
    the negative/positive boundary at (approximately) zero, so a
    cluster's median transformed value on its own IS the signal the R
    function expects -- unlike calculate_mem_scores (auto-reference +
    global rescale) or the z-score approach this replaces (auto-reference
    + population SD), both of which are RELATIVE: "how different is this
    cluster from the rest of this run."

    That relativity is exactly what broke scoring against a marker with
    NO cross-cluster variation -- e.g. CD3 in an all-CD3+, pre-gated
    T-cell run, where every cluster is already CD3+CD45+ and there is no
    cross-cluster difference for MEM or a z-score to detect, so both
    assign it ~0 regardless of how the result is rescaled or weighted.
    An absolute median doesn't have this problem: CD3's median transformed
    value sits well above the transform's zero point for every cluster in
    that run, because the cells really are CD3+, and now that correctly
    registers as positive evidence for every one of them.

    Returns a (cluster x channel) DataFrame; NaN cells follow the same
    convention as calculate_mem_scores (no pooled data for that
    cluster/channel).
    """
    channels = list(pooled.keys())
    clusters = sorted({cl for by_cl in pooled.values() for cl in by_cl})
    if not clusters or not channels:
        return pd.DataFrame(index=clusters, columns=channels, dtype=float)

    med = pd.DataFrame(index=clusters, columns=channels, dtype=float)
    for ch in channels:
        by_cl = pooled[ch]
        for cl in clusters:
            pop = by_cl.get(cl)
            if pop is None or len(pop) == 0:
                continue
            med.loc[cl, ch] = float(np.median(pop))
    arr = med.to_numpy(dtype=float)
    if np.any(~np.isnan(arr)):
        log.info(
            "cluster-median matrix: %s, value range [%.3g, %.3g], mean %.3g "
            "-- this is the ABSOLUTE scale score_cell_types compares "
            "against min_score",
            med.shape, float(np.nanmin(arr)), float(np.nanmax(arr)), float(np.nanmean(arr)),
        )
    else:
        log.warning("cluster-median matrix: %s, ALL NaN -- no pooled data at all", med.shape)
    return med


def _otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """
    Standard 1-D Otsu threshold: the cut point that maximises between-class
    variance, splitting `values` into a "low" and "high" group. Used by
    calculate_channel_thresholds() to find the negative/positive split for
    a channel -- does NOT require true bimodality to behave sensibly; a
    unimodal distribution (e.g. a marker that's uniformly positive across
    an entire pre-gated run) still gets a cutoff at the edge of that one
    population, not a nonsensical mid-population split.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    vmin, vmax = float(finite.min()), float(finite.max())
    if vmax <= vmin:
        return vmin

    hist, edges = np.histogram(finite, bins=n_bins, range=(vmin, vmax))
    hist = hist.astype(float)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    weight_low = np.cumsum(hist)
    weight_high = hist.sum() - weight_low
    cum_val = np.cumsum(hist * bin_centers)
    total_val = cum_val[-1]

    mean_low = np.divide(cum_val, weight_low, out=np.zeros_like(cum_val), where=weight_low > 0)
    mean_high = np.divide(total_val - cum_val, weight_high,
                           out=np.zeros_like(cum_val), where=weight_high > 0)
    between_class_var = weight_low * weight_high * (mean_low - mean_high) ** 2

    return float(bin_centers[int(np.argmax(between_class_var))])


def _is_bead_sample(path: str, name: str) -> bool:
    """Mirrors spectral_controller.py::get_unstained_negative's _is_bead()."""
    return bool(re.search(r'bead', name, re.IGNORECASE) or re.search(r'bead', path, re.IGNORECASE))


def _resolve_unstained_cell_sample_paths(controller) -> list[str]:
    """
    Returns the rel_path of every unstained CELL sample (manually tagged via
    the sample panel, OR name/path matching "unstained", excluding Beads) --
    same resolution logic as
    autospectral_optimization_functions.py::_resolve_unstained_cell_sample_names,
    but returns rel_path keys (what training_sample_ids and
    _load_unstained_gated_transformed below expect) rather than display
    names.
    """
    samples = getattr(controller.experiment, 'samples', {}) or {}
    all_samples = samples.get('all_samples', {})
    manually_unstained = set(samples.get('unstained_samples', []))

    def _is_unstained(path: str, name: str) -> bool:
        return (path in manually_unstained
                or 'unstained' in path.lower()
                or 'unstained' in name.lower())

    return [
        path for path, name in all_samples.items()
        if _is_unstained(path, name) and not _is_bead_sample(path, name)
    ]


def resolve_unstained_af_states(controller, cl_run: dict) -> dict[str, tuple | None]:
    """
    MAIN-THREAD ONLY -- call this BEFORE starting a background worker,
    exactly like the existing af_state snapshot passed into
    compute_cluster_id_suggestions, and forward the result through as
    unstained_af_states. See drc_pipeline.resolve_af_state_for_profiles's
    docstring for why: this reads controller.af_precomputed_cache and
    controller.experiment.process['af_profiles'], both of which the main
    thread can reassign while a background worker is reading them.

    The unstained sample(s) used by calculate_unstained_channel_thresholds
    are never assigned an AF profile themselves in the AutoSpectral AF tab
    -- there's no per-cell classification to run AF discovery against on
    an unstained control -- so each one needs a stand-in AF assignment to
    be unmixed consistently with the STAINED training samples it's meant
    to set a threshold for:

      1. An AF profile whose name matches the unstained sample's OWN
         display name exactly (experiment.process['af_profiles'] keys) --
         lets you deliberately name a profile after the unstained sample
         it represents.
      2. Otherwise, the MOST-FREQUENTLY-ASSIGNED profile combination
         across cl_run['training_sample_ids'] -- exact assigned lists
         compared as tuples, so two samples combining two profiles the
         same way count as the same combination rather than splitting
         across their individual profile names.
      3. If none of the training samples have ANY AF profile assigned,
         that unstained sample gets af_state=None -- unmixed WITHOUT
         per-cell AF correction, matching the run itself having none.

    Returns {unstained_rel_path: af_state_or_None} for every unstained
    cell sample _resolve_unstained_cell_sample_paths finds --
    calculate_unstained_channel_thresholds looks its own rel_path up in
    this dict (via .get(), so a path this somehow omitted just resolves to
    None -- AF-unaware -- rather than raising).
    """
    unstained_paths = _resolve_unstained_cell_sample_paths(controller)
    if not unstained_paths:
        return {}

    samples = getattr(controller.experiment, 'samples', {}) or {}
    all_samples_names = samples.get('all_samples', {})
    sample_af_profiles = samples.get('sample_af_profiles', {})
    af_profiles = controller.experiment.process.get('af_profiles', {})
    raw_subdir = controller.experiment.settings['raw']['raw_samples_subdirectory']

    most_common_combo: tuple = ()
    most_common_resolved = False

    def _resolve_most_common_combo() -> tuple:
        # sample_af_profiles is keyed the same way as all_samples
        # (experiment_dir-relative) -- training_sample_ids is picker-
        # relative (relative to raw_samples_subdirectory), same conversion
        # _load_unstained_gated_transformed's rel_path convention note
        # documents elsewhere in this module.
        assignments = []
        for rel in cl_run.get('training_sample_ids', []):
            key = str(Path(raw_subdir) / rel)
            assigned = tuple(sample_af_profiles.get(key, []))
            if assigned:
                assignments.append(assigned)
        if not assignments:
            log.info(
                "resolve_unstained_af_states: no training sample in this run "
                "has an AF profile assigned -- unstained sample(s) will be "
                "unmixed without AF correction"
            )
            return ()
        combo, count = Counter(assignments).most_common(1)[0]
        log.info(
            "resolve_unstained_af_states: most common AF assignment across "
            "%d training sample(s) is %s (%d/%d sample(s) assigned)",
            len(cl_run.get('training_sample_ids', [])), combo, count, len(assignments),
        )
        return combo

    result: dict[str, tuple | None] = {}
    af_state_cache: dict[tuple, tuple | None] = {}
    for rel in unstained_paths:
        display_name = all_samples_names.get(rel)
        if display_name and display_name in af_profiles:
            profile_names: tuple = (display_name,)
            log.info(
                "resolve_unstained_af_states: %s -- AF profile matching its "
                "own name (%r) found, using it", rel, display_name,
            )
        else:
            if not most_common_resolved:
                most_common_combo = _resolve_most_common_combo()
                most_common_resolved = True
            profile_names = most_common_combo

        if profile_names not in af_state_cache:
            af_state_cache[profile_names] = (
                drc_pipeline.resolve_af_state_for_profiles(controller, list(profile_names))
                if profile_names else None
            )
        result[rel] = af_state_cache[profile_names]

    return result


def _load_unstained_gated_transformed(
    controller, state, rel_path, channels: list[str], af_state=None,
    min_singlets_events: int = 500,
) -> tuple[np.ndarray, list[str]] | None:
    """
    Load ONE unstained sample -> TRANSFORMED values for `channels`.

    Gated on 'Singlets', NOT state.selected_gates -- an unstained sample
    carries no staining, so it will not fall inside whatever stained gate
    the training samples use; forcing it through state.selected_gates (the
    way drc_pipeline.load_sample_transformed_values does for training
    samples) would return zero events. 'Singlets' should always exist on
    the Raw Data gating hierarchy (settings.py's base_gate_priority_order
    puts it first), so it's used whenever present AND it has at least
    min_singlets_events events; otherwise falls back to 'root' (every event
    in the file, ungated) -- a near-empty or corrupt Singlets gate would
    otherwise hand back a percentile computed from a handful of events.

    Otherwise mirrors drc_pipeline.load_unmixed_gated + transform_channels:
    same AF-aware unmixing (apply_unmixing_af_aware), same per-channel
    transform (transform_channels). Deliberately NOT cached in
    state.gated_data_cache -- that cache is keyed on state.selected_gates,
    which this intentionally bypasses, and this is only called once per
    unstained sample per Cluster ID run, not a hot path.

    af_state: optional AF snapshot -- see apply_unmixing_af_aware()'s
    docstring. Pass this when calling from a background worker thread.

    Returns (values, channel_names) or None on error.
    """
    try:
        # rel_path comes from all_samples (experiment_dir-relative, already
        # includes raw_samples_subdirectory) -- NOT training_sample_ids'
        # picker-relative convention, so sample_abs_path() must NOT be used
        # here (it would prepend raw_samples_subdirectory a second time).
        abs_path = controller.experiment_dir / rel_path
        raw_settings = controller.experiment.settings['raw']
        whitelisted_pnn = raw_settings.get('whitelisted_pnn') or None

        sample = drc_pipeline.sample_from_fcs(abs_path)
        if whitelisted_pnn is not None:
            try:
                raw = sample.get_events(source='raw', col_order=whitelisted_pnn)
            except (KeyError, ValueError) as e:
                log.warning(
                    "_load_unstained_gated_transformed: col_order get_events "
                    "failed (%s) -- reading all channels", e,
                )
                raw = sample.get_events(source='raw')
        else:
            raw = sample.get_events(source='raw')

        unmixed = drc_pipeline.apply_unmixing_af_aware(controller, raw, af_state=af_state)

        cdd = deepcopy(controller.data_for_cytometry_plots_unmixed)
        cdd['event_data'] = unmixed
        gate_names = [g[0] for g in cdd['gating'].get_gate_ids()]

        gated = None
        if 'Singlets' in gate_names:
            singlets = drc_pipeline.apply_gate_by_lookup_table(deepcopy(cdd), 'Singlets')
            if len(singlets) >= min_singlets_events:
                gated = singlets
            else:
                log.info(
                    "_load_unstained_gated_transformed: %s -- 'Singlets' has only "
                    "%d event(s) (< %d) -- falling back to 'root'",
                    rel_path, len(singlets), min_singlets_events,
                )
        else:
            log.info(
                "_load_unstained_gated_transformed: %s -- no 'Singlets' gate found "
                "-- falling back to 'root'", rel_path,
            )

        if gated is None:
            gated = unmixed  # 'root' gate membership is all-True by construction

        return drc_pipeline.transform_channels(controller, state, gated, channels)
    except Exception as exc:
        log.exception("_load_unstained_gated_transformed: could not load %s: %s", rel_path, exc)
        return None


def calculate_unstained_channel_thresholds(
    controller, state, channels: list[str], unstained_af_states: dict | None = None,
    percentile: float = 98.0, safety_factor: float = 1.2,
    min_singlets_events: int = 500,
) -> dict[str, float]:
    """
    Per-channel positivity threshold derived from the experiment's own
    unstained CELL sample(s), gated on 'Singlets' (or 'root' -- see
    _load_unstained_gated_transformed) and TRANSFORMED the same way as the
    clustering channels themselves.

    Threshold = safety_factor * the `percentile`-th percentile of the
    unstained sample's transformed values for that channel (default: 2x
    the 99th percentile). This is deliberately NOT the Otsu cut
    (_otsu_threshold / calculate_channel_thresholds) -- Otsu assumes SOME
    negative population is present to split against, which breaks down
    for a marker that's uniformly positive across the whole run (e.g. CD45)
    or has been gated to be uniformly positive (e.g. CD3 in pre-gated T
    cells): Otsu then finds a nonsensical mid-population split instead of
    "positive," and the marker is lost to Cluster ID scoring. An unstained
    sample has no such marker on it at all, so its own background spread
    gives a channel threshold that works regardless of what the STAINED
    training samples look like.

    unstained_af_states: {rel_path: af_state_or_None}, as returned by
    resolve_unstained_af_states -- the unstained sample never has an AF
    profile of its own, so it must be unmixed with a STAND-IN AF state
    resolved per sample, never the training samples' own af_state (they
    have no bearing on an unstained control -- see
    resolve_unstained_af_states for how the stand-in is chosen). A rel_path
    missing from this dict (or a None value) is unmixed WITHOUT per-cell
    AF correction, same as any other AF-unaware call.

    If more than one unstained cell sample is found, the per-channel
    threshold is the MEDIAN across samples (each sample's own
    percentile*safety_factor computed independently first) -- one
    atypical unstained run doesn't skew the threshold for every channel.

    Returns {channel: threshold} covering ONLY channels present in at
    least one loadable unstained sample -- callers must treat a channel
    missing from this dict as "no unstained-derived threshold available"
    and fall back to Otsu (see calculate_channel_thresholds), not assume
    0.0 here. Returns {} if no unstained cell sample is found/loadable at
    all.
    """
    unstained_paths = _resolve_unstained_cell_sample_paths(controller)
    if not unstained_paths:
        log.info("calculate_unstained_channel_thresholds: no unstained cell sample found")
        return {}

    per_sample: list[dict[str, float]] = []
    for rel in unstained_paths:
        af_state = (unstained_af_states or {}).get(rel)
        mv = _load_unstained_gated_transformed(
            controller, state, rel, channels, af_state=af_state,
            min_singlets_events=min_singlets_events,
        )
        if mv is None:
            log.warning("calculate_unstained_channel_thresholds: %s -- could not load, skipped", rel)
            continue
        values, names = mv
        if values.shape[0] == 0:
            log.warning("calculate_unstained_channel_thresholds: %s -- no events, skipped", rel)
            continue
        per_sample.append({
            ch: float(safety_factor * np.percentile(values[:, names.index(ch)], percentile))
            for ch in channels if ch in names
        })

    if not per_sample:
        log.warning(
            "calculate_unstained_channel_thresholds: found %d unstained sample(s) "
            "but none could be loaded", len(unstained_paths),
        )
        return {}

    thresholds = {
        ch: float(np.median([d[ch] for d in per_sample if ch in d]))
        for ch in channels if any(ch in d for d in per_sample)
    }
    log.info(
        "calculate_unstained_channel_thresholds: %d unstained sample(s) -> %s",
        len(per_sample), {k: round(v, 3) for k, v in thresholds.items()},
    )
    return thresholds


def calculate_channel_thresholds(controller, state, pooled: dict[str, dict[int, np.ndarray]],
                                  channels: list[str], unstained_af_states: dict | None = None) -> dict[str, float]:
    """
    Per-channel positivity threshold, PREFERRING the unstained-sample-derived
    threshold (calculate_unstained_channel_thresholds) and falling back to
    the Otsu cut (_otsu_threshold) per-channel wherever the former has
    nothing for that channel -- either because no unstained cell sample
    exists in the experiment at all, or because that particular channel
    wasn't present in any loadable unstained sample.

    The Otsu path (computed from EVERY pooled event across EVERY cluster
    combined -- a property of the channel's overall distribution in THIS
    run, not of any one cluster relative to another) is unchanged from
    before; it's still needed as the fallback since it doesn't require an
    unstained sample to produce a usable cut -- see _otsu_threshold's
    docstring.

    Subtracting this from calculate_cluster_medians's output (see
    score_cell_types's caller) is the missing "thresholding" step
    flow_cluster_id_score.R's docstring assumes: without it, "off" reads as
    a small positive number rather than ~0/negative, which lets cell-type
    entries with MORE listed positive markers win purely by accumulating
    more small positive contributions, regardless of whether those markers
    are actually on (see this module's top-level notes on the 2024
    "Memory CD4 Treg wins everything" diagnosis for the concrete numbers).

    unstained_af_states: forwarded to calculate_unstained_channel_thresholds
    -- see its docstring and resolve_unstained_af_states.

    Returns {channel: threshold}. A channel with no unstained-derived value
    AND no pooled data anywhere gets threshold 0.0 (_otsu_threshold's own
    empty-input fallback).
    """
    unstained_thresholds = calculate_unstained_channel_thresholds(
        controller, state, channels, unstained_af_states=unstained_af_states,
    )

    thresholds: dict[str, float] = dict(unstained_thresholds)
    otsu_channels = [ch for ch in channels if ch not in unstained_thresholds]
    for ch in otsu_channels:
        by_cl = pooled.get(ch, {})
        all_vals = np.concatenate([v for v in by_cl.values() if len(v)]) if by_cl else np.array([])
        thresholds[ch] = _otsu_threshold(all_vals) if all_vals.size else 0.0

    if otsu_channels:
        log.info(
            "calculate_channel_thresholds: Otsu fallback for channel(s) without an "
            "unstained-derived threshold: %s", otsu_channels,
        )
    log.info("calculate_channel_thresholds: %s", {k: round(v, 3) for k, v in thresholds.items()})
    return thresholds


def generate_mem_labels(mem_scores: pd.DataFrame, display_labels: dict[str, str],
                         threshold: float = 2.0) -> dict[int, str]:
    """
    Human-readable label per cluster from a MEM score matrix, e.g.
    'CD4+6 CD8-5', ported from cluster_id.md's generate_mem_labels().
    Markers with |score| < threshold are dropped as noise; a cluster with
    nothing above threshold gets 'Uncharacterized' (never blank, so it's
    still a meaningful one-click MEM Label adoption target).

    display_labels: channel -> display name (antigen if assigned, else the
    channel name itself) -- see _channel_to_antigen_map().
    """
    labels: dict[int, str] = {}
    for cl in mem_scores.index:
        row = mem_scores.loc[cl].astype(float).dropna()
        if row.empty:
            log.info("generate_mem_labels: cluster %s -- no non-NaN MEM scores at all -- Uncharacterized", cl)
            labels[int(cl)] = 'Uncharacterized'
            continue
        max_abs = float(row.abs().max())
        significant = row[row.abs() >= threshold].sort_values(
            key=lambda s: s.abs(), ascending=False
        )
        if significant.empty:
            log.info(
                "generate_mem_labels: cluster %s -- max |MEM score| = %.2g across "
                "%d channel(s), below threshold %.2g -- Uncharacterized",
                cl, max_abs, len(row), threshold,
            )
        parts = []
        for ch, score in significant.items():
            name = display_labels.get(ch, ch)
            sign = '+' if score > 0 else ''
            parts.append(f"{name}{sign}{int(score)}")
        labels[int(cl)] = ' '.join(parts) if parts else 'Uncharacterized'
    return labels


# ---------------------------------------------------------------------------
# Channel <-> canonical marker name bridging
# ---------------------------------------------------------------------------

def _channel_to_antigen_map(controller) -> dict[str, str]:
    """
    channel -> raw antigen text from the spectral model, mirroring
    dr_clustering_tab._antigen_dash_labels's own lookup but returning the
    PLAIN antigen (not the 'Antigen - Channel' display string), since this
    is fed into marker-name canonicalisation, not shown directly.
    """
    try:
        unmixed_pnn = controller.experiment.settings.get('unmixed', {}).get(
            'event_channels_pnn') or []
        spectral_model = controller.experiment.process.get('spectral_model') or []
    except (AttributeError, KeyError):
        return {}
    label_to_antigen = {c.get('label'): (c.get('antigen') or '') for c in spectral_model}
    return {ch: label_to_antigen.get(ch, '') for ch in unmixed_pnn}


def channels_missing_antigen(controller, channels: list[str]) -> list[str]:
    """
    Hard pre-condition check for the tab: which of `channels` have NO
    Antigen assigned in the Spectral Process tab's spectral model.

    Cluster ID's MEM Label is built from `_channel_to_antigen_map`'s raw
    antigen text, falling back to the channel's fluorophore Label only when
    antigen is blank (see the fallback in `compute_cluster_id_suggestions`'s
    `display_labels`). Left unchecked, that fallback silently produces MEM
    Label output that mixes marker names (e.g. "CD4") and fluorophore names
    (e.g. "BUV395") in the same run, which reads as one consistent kind of
    label but isn't. Rather than let that fallback fire, the tab calls this
    FIRST and refuses to proceed (see Change 10 below) if anything comes
    back non-empty, forcing every checked channel to have an Antigen typed
    in before Cluster ID will run at all.

    Deliberately does NOT check the WHOLE spectral model, only the channels
    actually feeding this computation (the checked-channel set already
    reused from Panel 1) -- an unrelated dye you haven't checked and aren't
    using for Cluster ID shouldn't block the feature.
    """
    antigen_map = _channel_to_antigen_map(controller)
    return [ch for ch in channels if not (antigen_map.get(ch) or '').strip()]


def build_channel_marker_map(controller, channels: list[str]) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """
    channel -> canonical marker name, RE-canonicalised via label_matching at
    score time (not just a read of the stored antigen text) since antigen
    labels can be freely retyped after import and may no longer match
    marker_database.csv's canonical spelling exactly.

    We NEVER overwrite what the user typed into Antigen -- this canonical
    name is used ONLY to look the marker up in cell_type_database.csv
    (score_cell_types), never written back to the spectral model and never
    shown as the MEM Label (that stays the raw antigen text, unchanged).

    Returns (channel_marker_map, unmatched):
      channel_marker_map -- channel -> canonical marker if matched, else
                            the raw antigen text unchanged (so every
                            channel still gets *some* label to carry
                            around, even though an unmatched one won't
                            actually be referenced by any
                            cell_type_database.csv row).
      unmatched          -- [(channel, antigen_text), ...] for every
                            channel whose antigen text did NOT match
                            anything in marker_database.csv. The tab
                            surfaces this as a warning (Change 10) rather
                            than letting it fail silently -- these
                            channels are skipped entirely by
                            score_cell_types(), same as the R original's
                            "subset to markers found in the data."

    Callers are expected to have already run channels_missing_antigen() and
    refused to proceed if it returned anything -- this function assumes
    every channel here HAS some antigen text (it doesn't re-check for
    blank antigen itself).
    """
    try:
        from honeychrome.controller_components.label_matching import match_marker, get_marker_db
    except ImportError:
        log.warning("label_matching unavailable -- cell-type scoring will use raw antigen text only")
        antigen_map = _channel_to_antigen_map(controller)
        result = {ch: (antigen_map.get(ch) or ch) for ch in channels}
        return result, []

    antigen_map = _channel_to_antigen_map(controller)
    marker_db = get_marker_db()
    result = {}
    unmatched: list[tuple[str, str]] = []
    for ch in channels:
        antigen = antigen_map.get(ch) or ch
        canonical = match_marker(antigen, marker_db)
        result[ch] = canonical or antigen
        if not canonical:
            unmatched.append((ch, antigen))
    log.info(
        "build_channel_marker_map: %d/%d channel(s) matched a canonical "
        "marker; unmatched (raw antigen text, won't match any "
        "cell_type_database.csv row): %s",
        len(channels) - len(unmatched), len(channels),
        [antigen for _ch, antigen in unmatched],
    )
    return result, unmatched


# ---------------------------------------------------------------------------
# Cell-type database (bundled CSV -- see drc_cell_type_database.csv)
# ---------------------------------------------------------------------------

_cell_type_db_cache: list[dict] | None = None


def load_cell_type_database(path: Path | str | None = None) -> list[dict]:
    """
    Bundled cell-type marker-signature database, same convention as
    honeychrome's fluorophore_database.csv / marker_database.csv. Each row
    is a cell type plus semicolon-separated CANONICAL marker names (must
    match marker_database.csv's 'marker' column -- canonicalisation happens
    on the CHANNEL side in build_channel_marker_map(), not here).

    Ships with a starter set of common human/mouse immune populations --
    extend by editing the CSV, no code change needed to add a cell type.
    Cached at module level after first load (explicit path bypasses cache).
    """
    global _cell_type_db_cache
    if path is None and _cell_type_db_cache is not None:
        return _cell_type_db_cache

    if path is None:
        path = Path(__file__).parent / 'drc_cell_type_database.csv'
    path = Path(path)
    if not path.exists():
        log.warning("cell-type database not found at %s -- cell-type scoring will return no matches", path)
        return []

    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            pos = [m.strip() for m in (r.get('positive_markers') or '').split(';') if m.strip()]
            neg = [m.strip() for m in (r.get('negative_markers') or '').split(';') if m.strip()]
            if not pos and not neg:
                continue
            rows.append({
                'cell_type': (r.get('cell_type') or '').strip(),
                'species': (r.get('species') or '').strip(),
                'positive': pos,
                'negative': neg,
            })
    log.info("loaded cell-type database: %d entries from %s", len(rows), path)
    if path == Path(__file__).parent / 'drc_cell_type_database.csv':
        _cell_type_db_cache = rows
    return rows


def filter_cell_type_db_by_species(cell_type_db: list[dict],
                                    species: str | None) -> list[dict]:
    """
    Restrict cell_type_db to entries whose species field includes
    `species`. Rows store species as a ';'-joined list (e.g. 'human;mouse'
    for the handful of entries -- Astrocyte, LSEC, Oligodendrocyte -- whose
    canonicalised marker set happens to be identical across species; see
    drc_cell_type_database.csv), so a mouse-panel run still sees those.

    species=None (or '') returns cell_type_db unchanged -- callers that
    haven't wired up a species selector (or want the old unrestricted
    behaviour) aren't forced to pass anything.
    """
    if not species:
        log.info("filter_cell_type_db_by_species: no species set -- scoring against all %d entries", len(cell_type_db))
        return cell_type_db
    species = species.strip().lower()
    filtered = [e for e in cell_type_db if species in (e['species'] or '').lower().split(';')]
    log.info(
        "filter_cell_type_db_by_species: %d/%d entries kept for species=%r",
        len(filtered), len(cell_type_db), species,
    )
    return filtered


# ---------------------------------------------------------------------------
# Cell-type scoring (ported from flow_cluster_id_score.R)
# ---------------------------------------------------------------------------

def score_cell_types(scores: pd.DataFrame, channel_marker_map: dict[str, str],
                      cell_type_db: list[dict],
                      min_score: float = 0.3,
                      pos_evidence_floor: float = 0.0) -> pd.DataFrame:
    """
    Python port of flow_cluster_id_score.R (scType-derived), realigned
    with the R original's actual shape:

        score(type) = sum(scores[pos markers]) / sqrt(n_pos)
                      - sum(scores[neg markers]) / sqrt(n_neg)

    `scores` is calculate_cluster_medians's output with each channel's
    calculate_channel_thresholds() threshold subtracted (done by the
    caller, compute_cluster_id_suggestions) -- each cluster's OWN median
    transformed value, centred so "off" reads near/below zero and "on"
    reads clearly positive, matching the R function's "thresholded flow
    expression matrix" input. This is NOT MEM and NOT a z-score, and
    critically, NOT compared to other clusters at all -- a marker with
    zero cross-cluster variation (e.g. CD3 in an all-CD3+ pre-gated run)
    still reads as strongly positive here, since the threshold is a
    property of the channel's overall distribution, not of what's
    different between clusters -- something neither MEM nor a z-score
    can do. Without the threshold subtraction, "off" markers still read
    as a small positive number rather than ~0, which biases scoring
    toward cell types with longer positive-marker lists regardless of
    whether those markers are actually on -- see calculate_channel_thresholds's
    docstring for the concrete numbers that exposed this.

    Deliberate deviation from the R original (unchanged from previous
    ports): a missing marker contributes 0 to its side rather than
    resolving to NaN, and a cell type is skipped ENTIRELY only if NEITHER
    its positive nor negative markers have any representation in
    channel_marker_map.

    ABSENCE-ONLY GUARD (pos_evidence_floor) -- RE-INTRODUCED: the
    negative-marker term is a REWARD when those markers read absent (as
    expected, i.e. below their threshold) and a PENALTY when they
    unexpectedly read present -- both are real signal on their own. The
    problem is the reward half: since `scores` is already
    threshold-subtracted, a cluster with a low value on EVERY channel in
    the panel (debris, dying cells, unmixing noise near the threshold
    everywhere) reads as "absent" for every negative marker of every cell
    type, and that absence reward alone can outscore a type with genuine
    positive evidence. So the reward component only counts when this cell
    type ALSO has net-positive support from its OWN positive markers in
    this cluster (sum_pos > pos_evidence_floor); the penalty component
    (neg markers unexpectedly PRESENT) always counts regardless, since
    that's real contradicting evidence, not an artifact of "everything
    reads low here." This was implemented once already against the old
    MEM-then-z-score scoring inputs and dropped when scoring switched to
    the (unrelated) threshold-subtracted absolute values below -- same
    guard, reapplied to the current input.

    MINIMUM SCORE (min_score, default 0.3): a cluster with
    nothing scoring at least this gets 'Uncharacterized' rather than
    whatever happened to be least-negative -- several unrelated types
    tying at or near 0 (none of their positive markers are even in this
    panel) is an absence of a suggestion, not one.

    HIERARCHY TIE-BREAK (unchanged): an exact tie resolves to whichever
    entry was seen FIRST in cell_type_db -- drc_cell_type_database.csv
    lists parent/less-resolved populations before their children, so keep
    new entries appended in that order, not alphabetised, or this
    silently starts preferring whatever sorts first instead.

    scores columns are CHANNEL names; cell_type_db positive/negative
    lists are CANONICAL MARKER names -- channel_marker_map bridges the
    two. Returns a DataFrame indexed by cluster id with columns
    ['suggested_type', 'score', 'low_confidence']. A cluster with no cell
    type scoreable at all, or nothing clearing min_score, gets
    suggested_type='', score=None, low_confidence=False.
    """
    marker_to_channel: dict[str, str] = {}
    for ch, marker in channel_marker_map.items():
        marker_to_channel.setdefault(marker, ch)
    log.info(
        "score_cell_types: %d cell-type entries to check, %d canonical marker(s) "
        "resolved from this panel's channels: %s",
        len(cell_type_db), len(marker_to_channel), sorted(marker_to_channel),
    )

    n_evaluable = 0  # entries with at least one marker present in this panel

    records = []
    for cl in scores.index:
        row = scores.loc[cl]
        best_score = -np.inf
        best_type = ''
        best_has_pos_evidence = False

        for entry in cell_type_db:
            pos_channels = [marker_to_channel[m] for m in entry['positive'] if m in marker_to_channel]
            neg_channels = [marker_to_channel[m] for m in entry['negative'] if m in marker_to_channel]
            if not pos_channels and not neg_channels:
                continue  # no evidence either way in this panel -- skip entirely
            n_evaluable += 1

            pos_vals = [row[ch] for ch in pos_channels if ch in row.index and pd.notna(row[ch])]
            neg_vals = [row[ch] for ch in neg_channels if ch in row.index and pd.notna(row[ch])]
            sum_pos = (sum(pos_vals) / np.sqrt(len(pos_vals))) if pos_vals else 0.0
            sum_neg = (sum(neg_vals) / np.sqrt(len(neg_vals))) if neg_vals else 0.0

            has_pos_evidence = bool(pos_vals) and sum_pos > pos_evidence_floor
            neg_contribution = -sum_neg
            neg_reward = max(neg_contribution, 0.0)   # markers absent as expected
            neg_penalty = min(neg_contribution, 0.0)  # markers unexpectedly present -- always counts
            score = float(sum_pos + neg_penalty + (neg_reward if has_pos_evidence else 0.0))

            if score > best_score:
                best_score = score
                best_type = entry['cell_type']
                best_has_pos_evidence = has_pos_evidence
            # exact ties: keep whichever was seen FIRST (higher in the
            # hierarchy, per cell_type_db's own row order) -- do not
            # overwrite on score == best_score.

        cleared = best_type and best_score >= min_score
        log.info(
            "  cluster=%-3s best=%-20s score=%s%s%s",
            cl, best_type or '(none evaluable)',
            f"{best_score:.3f}" if best_score > -np.inf else 'n/a',
            '' if cleared else f"  <- below min_score={min_score}",
            '  [low_confidence]' if cleared and not best_has_pos_evidence else '',
        )

        if not best_type or best_score < min_score:
            records.append({
                'cluster': int(cl), 'suggested_type': '', 'score': None,
                'low_confidence': False,
            })
        else:
            records.append({
                'cluster': int(cl),
                'suggested_type': best_type,
                'score': round(float(best_score), 2),
                'low_confidence': not best_has_pos_evidence,
            })

    log.info(
        "score_cell_types: %d/%d (cluster, cell-type entry) pair(s) had at least "
        "one marker present in this panel",
        n_evaluable, len(scores.index) * len(cell_type_db),
    )
    return pd.DataFrame.from_records(records).set_index('cluster') if records \
        else pd.DataFrame(columns=['suggested_type', 'score', 'low_confidence'])


# ---------------------------------------------------------------------------
# Single entry point for the tab
# ---------------------------------------------------------------------------

def total_progress_steps(cl_run: dict) -> int:
    """
    Total progress_callback ticks compute_cluster_id_suggestions will
    emit for this run: one per training sample loaded (pooling), plus
    one per remaining pipeline stage (MEM scores, cluster medians,
    channel thresholds, MEM labels + marker mapping, cell-type scoring
    -- 5 stages). Call this BEFORE compute_cluster_id_suggestions to
    size a progress bar's range, so the range and the actual callback
    count can't drift apart if a stage is ever added or removed below.
    """
    n_samples = len(cl_run.get('training_sample_ids', [])) or 1
    return n_samples + 5


def compute_cluster_id_suggestions(controller, state, cl_run: dict, channels: list[str],
                                    mem_threshold: float = 2.0, iqr_floor: float = 0.5,
                                    min_score: float = 0.3,
                                    pos_evidence_floor: float = 0.0,
                                    cell_type_db_path: Path | str | None = None,
                                    species: str | None = None,
                                    progress_callback=None,
                                    af_state=None,
                                    unstained_af_states: dict | None = None,
                                    max_events_per_cluster: int | None = 1000,
                                    downsample_seed: int = 42):
    """
    One-call pipeline for the Cluster Annotation tab's Item 15 controls.

    Does NOT itself check for missing Antigen -- the tab calls
    channels_missing_antigen() BEFORE this and refuses to proceed if
    anything comes back, so by the time this runs every channel is assumed
    to have Antigen text (marker MATCHING against marker_database.csv is a
    separate, non-blocking concern -- see unmatched_markers below).

    progress_callback: optional callable(n_samples_done: int), forwarded
    straight through to pool_cluster_marker_values -- see its docstring.

    species: 'human', 'mouse', or None. Restricts cell-type scoring to
    drc_cell_type_database.csv entries defined for that species (see
    filter_cell_type_db_by_species) -- has NO effect on MEM Label, which
    is purely descriptive of the data and species-agnostic. None scores
    against the whole database, human and mouse entries both.

    min_score, pos_evidence_floor: forwarded to score_cell_types -- see
    its docstring.

    af_state: optional AF snapshot (transfer_matrix, af_precomputed,
    af_spectra), forwarded straight through to pool_cluster_marker_values --
    see its docstring. Pass this when calling from a background worker
    thread.

    unstained_af_states: {rel_path: af_state_or_None}, as returned by
    resolve_unstained_af_states -- forwarded to calculate_channel_thresholds
    (and from there to calculate_unstained_channel_thresholds). NOT the
    same thing as af_state above: the unstained sample used to derive
    positivity thresholds has no AF profile of its own, so it needs its
    OWN resolved stand-in, never the training samples' af_state. Must be
    resolved on the MAIN THREAD before this function is called from a
    background worker -- see resolve_unstained_af_states's docstring.

    max_events_per_cluster, downsample_seed: forwarded straight through
    to pool_cluster_marker_values -- see its docstring.

    Returns (mem_labels, mem_scores, cell_type_df, unmatched_markers):
      mem_labels        -- dict[cluster_id -> str], ready to display/adopt
      mem_scores        -- (cluster x channel) DataFrame, in case a future
                           caller wants the raw matrix (e.g. a heatmap)
      cell_type_df      -- DataFrame indexed by cluster, columns
                           ['suggested_type', 'score', 'low_confidence']
                           -- see score_cell_types for how ties are
                           resolved (hierarchy order, not reported) and
                           what low_confidence means (absence-only guard)
      unmatched_markers -- [(channel, antigen_text), ...] from
                           build_channel_marker_map -- Antigen entries that
                           didn't match marker_database.csv and so will
                           NOT contribute to cell-type scoring. The tab
                           surfaces this as a warning after computing
                           rather than failing or silently dropping them.
    """
    log_stage(log, "CLUSTER ID SUGGESTIONS")
    n_samples = len(cl_run.get('training_sample_ids', [])) or 1

    pooled = pool_cluster_marker_values(
        controller, state, cl_run, channels,
        progress_callback=progress_callback,
        af_state=af_state,
        max_events_per_cluster=max_events_per_cluster,
        seed=downsample_seed,
    )

    mem_scores = calculate_mem_scores(pooled, iqr_floor=iqr_floor)
    if progress_callback is not None:
        progress_callback(n_samples + 1)

    cluster_medians = calculate_cluster_medians(pooled)
    if progress_callback is not None:
        progress_callback(n_samples + 2)

    channel_thresholds = calculate_channel_thresholds(
        controller, state, pooled, channels, unstained_af_states=unstained_af_states,
    )
    thresholded = cluster_medians.subtract(pd.Series(channel_thresholds), axis=1)
    if progress_callback is not None:
        progress_callback(n_samples + 3)

    antigen_map = _channel_to_antigen_map(controller)
    display_labels = {ch: (antigen_map.get(ch) or ch) for ch in channels}
    mem_labels = generate_mem_labels(mem_scores, display_labels, threshold=mem_threshold)
    n_uncharacterized = sum(1 for v in mem_labels.values() if v == 'Uncharacterized')
    log.info(
        "compute_cluster_id_suggestions: %d/%d cluster(s) Uncharacterized (threshold=%.2g)",
        n_uncharacterized, len(mem_labels), mem_threshold,
    )
    channel_marker_map, unmatched_markers = build_channel_marker_map(controller, channels)
    cell_type_db = load_cell_type_database(cell_type_db_path)
    cell_type_db = filter_cell_type_db_by_species(cell_type_db, species)
    if progress_callback is not None:
        progress_callback(n_samples + 4)

    cell_type_df = score_cell_types(thresholded, channel_marker_map, cell_type_db,
                                     min_score=min_score,
                                     pos_evidence_floor=pos_evidence_floor)
    if progress_callback is not None:
        progress_callback(n_samples + 5)

    return mem_labels, mem_scores, cell_type_df, unmatched_markers
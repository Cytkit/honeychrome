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
import math
import time
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


def calculate_channel_thresholds(pooled: dict[str, dict[int, np.ndarray]]) -> dict[str, float]:
    """
    Per-channel Otsu threshold (see _otsu_threshold), computed from EVERY
    pooled event across EVERY cluster combined -- a property of the
    channel's overall distribution in this run, not of any one cluster
    relative to another. Subtracting this from calculate_cluster_medians's
    output (see score_cell_types's caller) is the missing "thresholding"
    step flow_cluster_id_score.R's docstring assumes: without it, "off"
    reads as a small positive number rather than ~0/negative, which lets
    cell-type entries with MORE listed positive markers win purely by
    accumulating more small positive contributions, regardless of whether
    those markers are actually on (see this module's top-level notes on
    the 2024 "Memory CD4 Treg wins everything" diagnosis for the concrete
    numbers).

    Deliberately NOT per-cluster or per-run-relative like calculate_mem_scores
    or the z-score approach both replaced by calculate_cluster_medians: a
    channel that's uniformly positive across every cluster in a pre-gated
    run (e.g. CD3 in an all-CD3+ dataset) still gets a sensible threshold
    at the edge of that single population, so every cluster stays above
    it and CD3 still counts as positive evidence for all of them.

    Returns {channel: threshold}. A channel with no pooled data anywhere
    gets threshold 0.0.
    """
    thresholds: dict[str, float] = {}
    for ch, by_cl in pooled.items():
        all_vals = np.concatenate([v for v in by_cl.values() if len(v)]) if by_cl else np.array([])
        thresholds[ch] = _otsu_threshold(all_vals) if all_vals.size else 0.0
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
                      min_score: float = 1.0) -> pd.DataFrame:
    """
    Python port of flow_cluster_id_score.R (scType-derived), realigned
    with the R original's actual shape:

        score(type) = sum(scores[pos markers]) / sqrt(n_pos)
                      - sum(scores[neg markers]) / sqrt(n_neg)

    `scores` is calculate_cluster_medians's output with each channel's
    calculate_channel_thresholds() Otsu threshold subtracted (done by the
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

    MINIMUM SCORE (min_score, default 1.0, unchanged from the previous
    revision): a cluster with nothing scoring at least this gets
    'Uncharacterized' rather than whatever happened to be least-negative
    -- several unrelated types tying at or near 0 (none of their positive
    markers are even in this panel) is an absence of a suggestion, not
    one.

    HIERARCHY TIE-BREAK (unchanged): an exact tie resolves to whichever
    entry was seen FIRST in cell_type_db -- drc_cell_type_database.csv
    lists parent/less-resolved populations before their children, so keep
    new entries appended in that order, not alphabetised, or this
    silently starts preferring whatever sorts first instead.

    NOT currently implemented (dropped from the last two revisions,
    which built them to fight problems specific to a RELATIVE scoring
    input): the absence-only reward/penalty split and signed-square
    marker weighting. Re-evaluate whether either is still needed once
    this plain version has been checked against real data -- don't
    assume it is.

    scores columns are CHANNEL names; cell_type_db positive/negative
    lists are CANONICAL MARKER names -- channel_marker_map bridges the
    two. Returns a DataFrame indexed by cluster id with columns
    ['suggested_type', 'score']. A cluster with no cell type scoreable at
    all, or nothing clearing min_score, gets suggested_type='',
    score=None.
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
            score = float(sum_pos - sum_neg)

            if score > best_score:
                best_score = score
                best_type = entry['cell_type']
            # exact ties: keep whichever was seen FIRST (higher in the
            # hierarchy, per cell_type_db's own row order) -- do not
            # overwrite on score == best_score.

        cleared = best_type and best_score >= min_score
        log.info(
            "  cluster=%-3s best=%-20s score=%s%s",
            cl, best_type or '(none evaluable)',
            f"{best_score:.3f}" if best_score > -np.inf else 'n/a',
            '' if cleared else f"  <- below min_score={min_score}",
        )

        if not best_type or best_score < min_score:
            records.append({'cluster': int(cl), 'suggested_type': '', 'score': None})
        else:
            records.append({
                'cluster': int(cl),
                'suggested_type': best_type,
                'score': round(float(best_score), 2),
            })

    log.info(
        "score_cell_types: %d/%d (cluster, cell-type entry) pair(s) had at least "
        "one marker present in this panel",
        n_evaluable, len(scores.index) * len(cell_type_db),
    )
    return pd.DataFrame.from_records(records).set_index('cluster') if records \
        else pd.DataFrame(columns=['suggested_type', 'score'])


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
                                    min_score: float = 0.2,
                                    cell_type_db_path: Path | str | None = None,
                                    species: str | None = None,
                                    progress_callback=None,
                                    af_state=None,
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

    min_score: forwarded to score_cell_types -- see its docstring.

    af_state: optional AF snapshot (transfer_matrix, af_precomputed,
    af_spectra), forwarded straight through to pool_cluster_marker_values --
    see its docstring. Pass this when calling from a background worker
    thread.

    max_events_per_cluster, downsample_seed: forwarded straight through
    to pool_cluster_marker_values -- see its docstring.

    Returns (mem_labels, mem_scores, cell_type_df, unmatched_markers):
      mem_labels        -- dict[cluster_id -> str], ready to display/adopt
      mem_scores        -- (cluster x channel) DataFrame, in case a future
                           caller wants the raw matrix (e.g. a heatmap)
      cell_type_df      -- DataFrame indexed by cluster, columns
                           ['suggested_type', 'score'] -- see
                           score_cell_types for how ties are resolved
                           (hierarchy order, not reported)
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

    channel_thresholds = calculate_channel_thresholds(pooled)
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
                                     min_score=min_score)
    if progress_callback is not None:
        progress_callback(n_samples + 5)

    return mem_labels, mem_scores, cell_type_df, unmatched_markers
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
                                progress_callback=None) -> dict[str, dict[int, np.ndarray]]:
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

    Returns {channel: {cluster_id: concatenated np.ndarray}}. A channel/
    cluster combination with no data is simply absent from the inner dict
    (callers must use .get()).
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
        mv = drc_pipeline.load_sample_transformed_values(controller, state, rel, channels)
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

    result = {
        ch: {cl: np.concatenate(vals) for cl, vals in by_cl.items() if len(vals)}
        for ch, by_cl in pooled.items()
    }
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


# ---------------------------------------------------------------------------
# Cell-type scoring (ported from flow_cluster_id_score.R)
# ---------------------------------------------------------------------------

def score_cell_types(mem_scores: pd.DataFrame, channel_marker_map: dict[str, str],
                      cell_type_db: list[dict]) -> pd.DataFrame:
    """
    Python port of flow_cluster_id_score.R (scType-derived): per cluster,

        score(type) = sum(MEM[pos markers]) / sqrt(n_pos)
                      - sum(MEM[neg markers]) / sqrt(n_neg)

    The highest-scoring type wins; an EXACT tie is flagged (is_tie=True)
    rather than arbitrarily broken -- same behaviour as the R original,
    which returns all tied names joined by ' | '. cluster_id.md's "Advanced
    Tie-Breaking Strategies" (marker weighting, margin-based near-ties,
    hierarchical fallback) are deliberately NOT implemented here -- natural
    follow-ups, not part of this integration.

    Deliberate deviation from the R original: flow_cluster_id_score.R
    doesn't guard the POSITIVE-marker sum against an empty marker set (only
    the negative one), so a cell type with zero of its positive markers
    present in the data resolves to NaN in R, and NaN comparisons there
    would raise rather than resolve sensibly. Here, a missing marker simply
    contributes 0 to its side (matching cluster_id.md's own
    assign_cell_types_robust), and a cell type is skipped ENTIRELY only if
    NEITHER its positive nor negative markers have any representation in
    channel_marker_map -- i.e. this specific panel has no evidence for or
    against it at all, so scoring it as "0, no evidence either way" would
    let it win by default in a database of otherwise-irrelevant entries
    (see the human vs. mouse Neutrophil rows in drc_cell_type_database.csv
    for why this matters in practice).

    mem_scores columns are CHANNEL names; cell_type_db positive/negative
    lists are CANONICAL MARKER names -- channel_marker_map bridges the two.
    Returns a DataFrame indexed by cluster id with columns
    ['suggested_type', 'score', 'is_tie']. A cluster with no cell type
    scoreable at all gets suggested_type='' , score=None, is_tie=False.
    """
    marker_to_channel: dict[str, str] = {}
    for ch, marker in channel_marker_map.items():
        marker_to_channel.setdefault(marker, ch)

    records = []
    for cl in mem_scores.index:
        row = mem_scores.loc[cl]
        best_score = -np.inf
        best_types: list[str] = []

        for entry in cell_type_db:
            pos_channels = [marker_to_channel[m] for m in entry['positive'] if m in marker_to_channel]
            neg_channels = [marker_to_channel[m] for m in entry['negative'] if m in marker_to_channel]
            if not pos_channels and not neg_channels:
                continue  # no evidence either way in this panel -- skip entirely

            pos_vals = [row[ch] for ch in pos_channels if ch in row.index and pd.notna(row[ch])]
            neg_vals = [row[ch] for ch in neg_channels if ch in row.index and pd.notna(row[ch])]
            sum_pos = (sum(pos_vals) / np.sqrt(len(pos_vals))) if pos_vals else 0.0
            sum_neg = (-sum(neg_vals) / np.sqrt(len(neg_vals))) if neg_vals else 0.0
            score = float(sum_pos + sum_neg)

            if score > best_score:
                best_score = score
                best_types = [entry['cell_type']]
            elif score == best_score:
                best_types.append(entry['cell_type'])

        if not best_types:
            records.append({'cluster': int(cl), 'suggested_type': '', 'score': None, 'is_tie': False})
        else:
            records.append({
                'cluster': int(cl),
                'suggested_type': ' | '.join(dict.fromkeys(best_types)),  # de-dupe, keep order
                'score': round(float(best_score), 2),
                'is_tie': len(set(best_types)) > 1,
            })

    return pd.DataFrame.from_records(records).set_index('cluster') if records \
        else pd.DataFrame(columns=['suggested_type', 'score', 'is_tie'])


# ---------------------------------------------------------------------------
# Single entry point for the tab
# ---------------------------------------------------------------------------

def compute_cluster_id_suggestions(controller, state, cl_run: dict, channels: list[str],
                                    mem_threshold: float = 2.0, iqr_floor: float = 0.5,
                                    cell_type_db_path: Path | str | None = None,
                                    progress_callback=None):
    """
    One-call pipeline for the Cluster Annotation tab's Item 15 controls.

    Does NOT itself check for missing Antigen -- the tab calls
    channels_missing_antigen() BEFORE this and refuses to proceed if
    anything comes back, so by the time this runs every channel is assumed
    to have Antigen text (marker MATCHING against marker_database.csv is a
    separate, non-blocking concern -- see unmatched_markers below).

    progress_callback: optional callable(n_samples_done: int), forwarded
    straight through to pool_cluster_marker_values -- see its docstring.

    Returns (mem_labels, mem_scores, cell_type_df, unmatched_markers):
      mem_labels        -- dict[cluster_id -> str], ready to display/adopt
      mem_scores        -- (cluster x channel) DataFrame, in case a future
                           caller wants the raw matrix (e.g. a heatmap)
      cell_type_df      -- DataFrame indexed by cluster, columns
                           ['suggested_type', 'score', 'is_tie']
      unmatched_markers -- [(channel, antigen_text), ...] from
                           build_channel_marker_map -- Antigen entries that
                           didn't match marker_database.csv and so will
                           NOT contribute to cell-type scoring. The tab
                           surfaces this as a warning after computing
                           (Change 10) rather than failing or silently
                           dropping them.
    """
    log_stage(log, "CLUSTER ID SUGGESTIONS")
    pooled = pool_cluster_marker_values(controller, state, cl_run, channels,
                                         progress_callback=progress_callback)
    mem_scores = calculate_mem_scores(pooled, iqr_floor=iqr_floor)

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
    cell_type_df = score_cell_types(mem_scores, channel_marker_map, cell_type_db)

    return mem_labels, mem_scores, cell_type_df, unmatched_markers
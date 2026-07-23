"""
drc_stats.py — Differential statistics (inmoose/limma) for the DR/Clustering plugin
===================================================================================
Companion to ``dr_clustering_tab.py`` (filename intentionally NOT ``*_tab.py``).

Computes cluster-frequency and cluster-MFI differential statistics between two
sample groups via ``inmoose.limma`` (lmFit → eBayes → topTable), verified against
CONTEXT_InMoose.md (v0.9.1).

Item 14 adds a parallel negative-binomial GLM (statsmodels) path for cluster
differential ABUNDANCE on raw counts, run alongside (not instead of) the
Frequency/limma test — both remain independently selectable so the two can be
compared on the same data. MFI/differential-expression testing is unaffected;
it stays on the limma/InMoose path either way.

Fixes baked in (see the diagnosis docs):
  • S0 — inmoose ``topTable`` returns a DEResults object whose columns are
         ``log2FoldChange / AveExpr / stat / pvalue / adj_pvalue / B``. These are
         normalised to the R/limma names (``logFC / P.Value / adj.P.Val / t``)
         the rest of the plugin (volcano/heatmap/significant) already expects.
  • S1 — ``sort_by='none'`` keeps rows in input order so feature labels line up
         (no relabelling of a p-sorted table).
  • S2 — MFI channel values come from drc_pipeline (correct channel→column map),
         not a filtered-index lookup against full-width data.
  • S6 — each sample's FCS is loaded ONCE, not once per channel.

Group bookkeeping note: ``state.sample_groups`` values are slot keys
('A'/'B'/'Unassigned'). The rename handler in the tab must not rewrite them
(fix S3 in DR_CLUSTERING_STATS_GROUPS_FIXES.md); this module relies on that.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import drc_pipeline
from drc_logging import get_logger, log_stage, log_files

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Group resolution
# ---------------------------------------------------------------------------

def resolve_group_samples(controller, state, cluster_labels_override=None):
    """
    Resolve the slot-A / slot-B samples to rel-path keys present in
    ``cluster_labels_override`` (or ``state.cluster_labels`` if not supplied).

    Returns ``(a_rel, b_rel, all_rel, group_vec)``.
    Raises RuntimeError if either group has < 3 labelled samples.
    """
    log_stage(log, "RESOLVE GROUPS")
    cluster_labels = cluster_labels_override if cluster_labels_override is not None \
        else state.cluster_labels
    a_samples = [sp for sp, g in state.sample_groups.items() if g == 'A']
    b_samples = [sp for sp, g in state.sample_groups.items() if g == 'B']
    raw_subdir = controller.experiment.settings['raw']['raw_samples_subdirectory']

    def _to_rel(sp):
        try:
            return str(Path(sp).relative_to(raw_subdir))
        except ValueError:
            return sp

    log.info("cluster_labels keys: %s", list(cluster_labels.keys()))
    log.info("raw_subdir: %r", raw_subdir)

    a_rel = []
    for sp in a_samples:
        rel = _to_rel(sp)
        if rel in cluster_labels:
            a_rel.append(rel)
        else:
            log.info("group A miss: sp=%r  rel=%r", sp, rel)

    b_rel = []
    for sp in b_samples:
        rel = _to_rel(sp)
        if rel in cluster_labels:
            b_rel.append(rel)
        else:
            log.info("group B miss: sp=%r  rel=%r", sp, rel)

    log.info("group A: %d assigned, %d with cluster labels", len(a_samples), len(a_rel))
    log.info("group B: %d assigned, %d with cluster labels", len(b_samples), len(b_rel))

    if len(a_rel) < 3 or len(b_rel) < 3:
        raise RuntimeError(
            f"Not enough samples with cluster labels: A={len(a_rel)}, B={len(b_rel)}. "
            "Assign ≥3 per group and run 'Apply to All Samples' first."
        )

    all_rel = a_rel + b_rel
    group_vec = (['A'] * len(a_rel)) + (['B'] * len(b_rel))
    log_files(log, "samples entering limma", all_rel)
    return a_rel, b_rel, all_rel, group_vec


def n_clusters_from_labels(state, all_rel, cluster_labels_override=None) -> int:
    """Number of (non-noise) clusters across the given samples. 0-based labels,
    so result = max_label + 1. Guards the all-noise case."""
    cluster_labels = cluster_labels_override if cluster_labels_override is not None \
        else state.cluster_labels
    positive = [int(lbl)
                for rel in all_rel
                for lbl in np.unique(cluster_labels[rel]) if lbl >= 0]
    if not positive:
        raise RuntimeError("No non-noise clusters to test.")
    n = int(max(positive) + 1)
    log.info("n_clusters across %d samples: %d", len(all_rel), n)
    return n


# ---------------------------------------------------------------------------
# Feature matrices
# ---------------------------------------------------------------------------

def _label_for(state, cl: int, names_override: dict | None) -> str:
    """
    Resolve a cluster's display name. When names_override is given (the
    SELECTED run's own 'names' dict — see drc_scatter.py / Item 8), it is
    authoritative and state.cluster_names is never consulted, so labels
    can't bleed in from an unrelated run. Falls back to the legacy global
    dict only for the "Active (unsaved)" pseudo-run, which has no run
    entry of its own yet.
    """
    if names_override is not None:
        if cl in names_override:
            return names_override[cl]
        return 'Noise' if cl < 0 else str(cl)
    return state.cluster_label(cl)


def compute_frequencies(state, all_rel, n_clusters, cluster_labels_override=None,
                        names_override: dict | None = None) -> pd.DataFrame:
    """Per-sample % of events in each cluster → (n_samples × n_clusters)."""
    cluster_labels = cluster_labels_override if cluster_labels_override is not None \
        else state.cluster_labels
    freq_mat = np.zeros((len(all_rel), n_clusters), dtype=float)
    for i, rel in enumerate(all_rel):
        labels = np.asarray(cluster_labels[rel])
        total = max(len(labels), 1)
        for cl in range(n_clusters):
            freq_mat[i, cl] = np.sum(labels == cl) / total * 100.0
    df = pd.DataFrame(freq_mat, index=all_rel,
                      columns=[_label_for(state, cl, names_override) for cl in range(n_clusters)])
    log.info("frequency matrix: %s", df.shape)
    return df


def compute_counts(state, all_rel, n_clusters, cluster_labels_override=None,
                   names_override: dict | None = None) -> pd.DataFrame:
    """
    Per-sample RAW event count in each cluster → (n_samples × n_clusters).

    Same shape/index/columns as compute_frequencies() — just unnormalized.
    This is the counterpart Item 14's GLM-on-counts path needs alongside the
    existing percentage matrix; run_statistics() computes both from the same
    already-resolved all_rel/n_clusters rather than each re-deriving them.
    """
    cluster_labels = cluster_labels_override if cluster_labels_override is not None \
        else state.cluster_labels
    count_mat = np.zeros((len(all_rel), n_clusters), dtype=float)
    for i, rel in enumerate(all_rel):
        labels = np.asarray(cluster_labels[rel])
        for cl in range(n_clusters):
            count_mat[i, cl] = np.sum(labels == cl)
    df = pd.DataFrame(count_mat, index=all_rel,
                      columns=[_label_for(state, cl, names_override) for cl in range(n_clusters)])
    log.info("counts matrix: %s", df.shape)
    return df


def resolve_mfi_channels(state, include_type_markers: bool = False) -> list[str]:
    """
    Return the channel list MFI significance testing should use, filtered
    by marker role (Item 11 — diffcyt's type/state split, §5). 'type'
    channels (those that drove the clustering assignment) are excluded by
    default, to avoid the same channel driving both the cluster call and
    its own significance test. include_type_markers=True restores the
    previous all-selected-channels behaviour.
    """
    channels = [c for c in state.selected_channels if c not in drc_pipeline.META_CHANNELS]
    if include_type_markers:
        return channels
    return [ch for ch in channels if state.marker_roles.get(ch, 'state') != 'type']


def compute_mfis(controller, state, all_rel, n_clusters,
                 cluster_labels_override=None, channels=None,
                 names_override: dict | None = None,
                 af_state=None) -> pd.DataFrame | None:
    """
    Per-sample mean intensity of each selected channel within each cluster.

    Returns (n_samples × (n_clusters · n_channels)). Loads each sample's
    untransformed selected-channel values ONCE via drc_pipeline (correct
    channel→column mapping), then iterates channels in memory.

    channels: explicit channel list to test (Item 11). Defaults to every
              selected channel (pre-Item-11 behaviour) when not supplied,
              so any other caller keeps working unchanged.

    af_state: optional (transfer_matrix, af_precomputed, af_spectra) snapshot,
        captured on the main thread before this (background-thread) call —
        see drc_pipeline.apply_unmixing_af_aware() docstring for why this
        must not be read live off ``controller`` from a worker thread.
    """
    log_stage(log, "MFI MATRIX")
    cluster_labels = cluster_labels_override if cluster_labels_override is not None \
        else state.cluster_labels
    if channels is None:
        channels = [c for c in state.selected_channels if c not in drc_pipeline.META_CHANNELS]

    sample_vals = {}
    for rel in all_rel:
        mv = drc_pipeline.load_sample_marker_values(controller, state, rel, af_state=af_state)
        if mv is not None:
            sample_vals[rel] = mv          # (values (n,n_sel), names)

    frames = []
    for ch in channels:
        mfi_mat = np.zeros((len(all_rel), n_clusters), dtype=float)
        for i, rel in enumerate(all_rel):
            mv = sample_vals.get(rel)
            if mv is None:
                continue
            values, names = mv
            if ch not in names:
                continue
            col = values[:, names.index(ch)]
            labels = np.asarray(cluster_labels[rel])
            m = min(len(col), len(labels))      # defensive; should be equal
            col, lab = col[:m], labels[:m]
            for cl in range(n_clusters):
                sel = lab == cl
                if sel.any():
                    vals = np.maximum(col[sel], 0.0)
                    mfi_mat[i, cl] = float(np.mean(np.log1p(vals)))
        frames.append(pd.DataFrame(
            mfi_mat, index=all_rel,
            columns=[f'{_label_for(state, cl, names_override)}_{ch}' for cl in range(n_clusters)]))

    if not frames:
        log.warning("no MFI features could be built")
        return None
    full = pd.concat(frames, axis=1)
    log.info("MFI matrix: %s", full.shape)
    return full


# ---------------------------------------------------------------------------
# Composition views (Items 10 & 12 — CyCONDOR comparison, §5)
# ---------------------------------------------------------------------------

def compute_confusion_matrix(controller, state, cluster_labels_override=None,
                             normalize_to: int = 1000,
                             names_override: dict | None = None) -> pd.DataFrame:
    """
    Per-group-normalized cluster composition heatmap (CyCONDOR's
    plot_confusion_HM).

    Pools each group's events across its samples, normalizes each group's
    total event count to ``normalize_to``, then for every cluster returns
    the normalized share contributed by each group. Independent of any
    limma results — usable as soon as groups are assigned and a clustering
    run is selected (same availability gate as run_statistics(), via
    resolve_group_samples()'s ≥3-per-group check).

    Returns (n_clusters × 2) DataFrame, columns ['A', 'B'] (slot keys, not
    display names — the caller maps these to the user-facing group names),
    index = cluster display labels.
    """
    log_stage(log, "CONFUSION MATRIX")
    a_rel, b_rel, all_rel, _group_vec = resolve_group_samples(
        controller, state, cluster_labels_override=cluster_labels_override
    )
    cluster_labels = cluster_labels_override if cluster_labels_override is not None \
        else state.cluster_labels
    n_clusters = n_clusters_from_labels(
        state, all_rel, cluster_labels_override=cluster_labels_override
    )

    group_rel = {'A': a_rel, 'B': b_rel}
    conf = np.zeros((n_clusters, 2), dtype=float)
    for gi, grp in enumerate(('A', 'B')):
        rels = group_rel[grp]
        if not rels:
            continue
        pooled = np.concatenate([np.asarray(cluster_labels[rel]) for rel in rels])
        total = len(pooled)
        if total == 0:
            continue
        scale = normalize_to / total
        for cl in range(n_clusters):
            conf[cl, gi] = float(np.sum(pooled == cl)) * scale

    df = pd.DataFrame(conf, columns=['A', 'B'],
                      index=[_label_for(state, cl, names_override) for cl in range(n_clusters)])
    log.info("confusion matrix: %s", df.shape)
    return df


def get_counts_table(controller, state, group_var: str = 'sample',
                     cluster_labels_override=None,
                     names_override: dict | None = None) -> pd.DataFrame:
    """
    Raw event counts per cluster (CyCONDOR's getTable(), counts variant).

    group_var: 'sample' → one row per sample (rel-path index).
               'group'  → one row per assigned group (display name),
                          summed across that group's samples.
    Returns (n_rows × n_clusters) DataFrame, columns = cluster display
    labels. Also the natural building block for CSV export alongside the
    existing "Save Statistics CSVs" pattern.
    """
    log_stage(log, "COUNTS TABLE")
    _a_rel, _b_rel, all_rel, group_vec = resolve_group_samples(
        controller, state, cluster_labels_override=cluster_labels_override
    )
    n_clusters = n_clusters_from_labels(
        state, all_rel, cluster_labels_override=cluster_labels_override
    )
    cluster_labels = cluster_labels_override if cluster_labels_override is not None \
        else state.cluster_labels
    cols = [_label_for(state, cl, names_override) for cl in range(n_clusters)]

    per_sample = np.zeros((len(all_rel), n_clusters), dtype=int)
    for i, rel in enumerate(all_rel):
        labels = np.asarray(cluster_labels[rel])
        for cl in range(n_clusters):
            per_sample[i, cl] = int(np.sum(labels == cl))

    if group_var == 'sample':
        df = pd.DataFrame(per_sample, index=all_rel, columns=cols)
    elif group_var == 'group':
        group_names = getattr(state, '_group_names', ['A', 'B'])
        name_a = group_names[0] if len(group_names) > 0 else 'A'
        name_b = group_names[1] if len(group_names) > 1 else 'B'
        a_mask = np.array([g == 'A' for g in group_vec])
        b_mask = ~a_mask
        summed = np.stack([
            per_sample[a_mask].sum(axis=0) if a_mask.any() else np.zeros(n_clusters),
            per_sample[b_mask].sum(axis=0) if b_mask.any() else np.zeros(n_clusters),
        ])
        df = pd.DataFrame(summed, index=[name_a, name_b], columns=cols)
    else:
        raise ValueError(f"group_var must be 'sample' or 'group', got {group_var!r}")

    log.info("counts table (%s): %s", group_var, df.shape)
    return df


def get_frequency_table(controller, state, group_var: str = 'sample',
                        cluster_labels_override=None,
                        names_override: dict | None = None) -> pd.DataFrame:
    """
    Per-row % of events per cluster (CyCONDOR's getTable(), frequency
    variant). Same shape/grouping as get_counts_table, row-normalized to
    100%.
    """
    counts_df = get_counts_table(
        controller, state, group_var=group_var,
        cluster_labels_override=cluster_labels_override,
        names_override=names_override,
    )
    totals = counts_df.sum(axis=1).replace(0, 1)   # guard empty rows
    freq_df = counts_df.div(totals, axis=0) * 100.0
    log.info("frequency table (%s): %s", group_var, freq_df.shape)
    return freq_df


# ---------------------------------------------------------------------------
# limma
# ---------------------------------------------------------------------------

def run_limma(data_df: pd.DataFrame, group_vec: list[str],
              pval_threshold: float, fc_threshold: float) -> pd.DataFrame:
    """
    lmFit → eBayes → topTable for an A-vs-B contrast.

    data_df  : rows = samples, columns = features (clusters / channel-clusters)
    group_vec: 'A'/'B' per row, aligned to data_df.index

    Returns a DataFrame with R/limma-style columns:
        feature, logFC, AveExpr, t, P.Value, adj.P.Val, B, significant
    """
    from inmoose.limma import lmFit, eBayes, topTable

    n = len(group_vec)
    # col 0 = intercept (group A baseline), col 1 = group-B effect.
    # lmFit renames columns to 'column0', 'column1' regardless of input names.
    design = np.zeros((n, 2), dtype=float)
    design[:, 0] = 1.0
    design[:, 1] = [1.0 if g == 'B' else 0.0 for g in group_vec]

    expr = data_df.values.T.astype(float)        # (features, samples) — lmFit orientation
    n_features = expr.shape[0]

    fit = eBayes(lmFit(expr, design=design))
    tt = topTable(fit, coef='column1', number=np.inf, adjust_method='fdr_bh')

    # S0: normalise inmoose DEResults columns → R/limma names used downstream.
    tt = pd.DataFrame(tt).rename(columns={
        'log2FoldChange': 'logFC',
        'pvalue':         'P.Value',
        'adj_pvalue':     'adj.P.Val',
        'stat':           't',
    })
    feature_names = np.asarray(data_df.columns)
    tt.insert(0, 'feature', feature_names[tt.index.to_numpy()])
    tt = tt.reset_index(drop=True)        
    tt['significant'] = (
        (tt['adj.P.Val'] <= pval_threshold) &
        (tt['logFC'].abs() >= fc_threshold)
    )
    log.info("limma: %d features tested, %d significant",
             len(tt), int(tt['significant'].sum()))
    return tt


def run_glm_counts(counts_df: pd.DataFrame, group_vec: list[str],
                   pval_threshold: float, fc_threshold: float) -> pd.DataFrame:
    """
    Per-cluster negative-binomial GLM differential abundance test on raw
    event counts (Item 14) — diffcyt::testDA_edgeR's approach, offered
    alongside (not replacing) the Frequency/limma path in run_limma().
    Counts avoid the compositional (bounded, non-independent) nature of
    per-sample percentages, which is the more defensible model for
    differential ABUNDANCE specifically; this has no bearing on MFI/
    differential-expression testing, which stays on the limma/InMoose path
    regardless of this item.

    counts_df : rows = samples, columns = clusters, raw event counts
    group_vec : 'A'/'B' per row, aligned to counts_df.index

    statsmodels' NegativeBinomial family needs its dispersion (alpha)
    supplied rather than estimating it itself, so alpha is estimated per
    cluster via the standard auxiliary-OLS method (Cameron & Trivedi): fit
    a Poisson GLM, then regress the auxiliary variable
    ((y - mu)**2 - y) / mu on mu (no intercept) — the slope is alpha. Falls
    back to the plain Poisson fit if the alpha estimate is non-positive or
    the NB fit doesn't converge, and to an untestable (NaN) row if even the
    Poisson fit fails (e.g. an all-zero cluster in one group), rather than
    losing every other cluster's result to one bad fit.

    Returns the same feature/logFC/P.Value/adj.P.Val/t/significant schema
    run_limma() produces, so both methods feed the same heatmap/volcano/CSV
    export code unchanged, and the two result tables are directly
    comparable side-by-side.
    """
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests

    n = len(group_vec)
    # col 0 = intercept (group A baseline), col 1 = group-B effect —
    # same design construction as run_limma(), so both tests share one
    # contrast definition.
    design = np.zeros((n, 2), dtype=float)
    design[:, 0] = 1.0
    design[:, 1] = [1.0 if g == 'B' else 0.0 for g in group_vec]

    LN2 = np.log(2.0)   # NB's log link is natural log; convert to log2 for logFC
    clusters = list(counts_df.columns)
    logfc = np.full(len(clusters), np.nan)
    tvals = np.full(len(clusters), np.nan)
    pvals = np.full(len(clusters), np.nan)

    for i, cl in enumerate(clusters):
        y = counts_df[cl].values.astype(float)

        poisson_fit = None
        try:
            poisson_fit = sm.GLM(y, design, family=sm.families.Poisson()).fit()
        except Exception as e:
            log.warning("cluster %r: Poisson GLM failed (%s) — marking untestable", cl, e)

        fit = None
        if poisson_fit is not None:
            try:
                mu = poisson_fit.mu
                aux_y = ((y - mu) ** 2 - y) / mu
                alpha = float(sm.OLS(aux_y, mu).fit().params[0])
                if not np.isfinite(alpha) or alpha <= 0:
                    raise ValueError("non-positive alpha estimate")
                fit = sm.GLM(y, design, family=sm.families.NegativeBinomial(alpha=alpha)).fit()
            except Exception as e:
                log.warning("cluster %r: NB GLM failed (%s), falling back to Poisson", cl, e)
                fit = poisson_fit

        if fit is None:
            continue
        logfc[i] = fit.params[1] / LN2
        tvals[i] = fit.tvalues[1]
        pvals[i] = fit.pvalues[1]

    adj_pvals = np.full(len(clusters), np.nan)
    valid = np.isfinite(pvals)
    if valid.any():
        adj_pvals[valid] = multipletests(pvals[valid], method='fdr_bh')[1]

    tt = pd.DataFrame({
        'feature':   clusters,
        'logFC':     logfc,
        't':         tvals,
        'P.Value':   pvals,
        'adj.P.Val': adj_pvals,
    })
    tt['significant'] = (
        (tt['adj.P.Val'] <= pval_threshold) &
        (tt['logFC'].abs() >= fc_threshold)
    )
    log.info("GLM counts: %d clusters tested, %d significant (%d untestable)",
             len(tt), int(tt['significant'].sum()), int((~valid).sum()))
    return tt


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_statistics(controller, state, run_freq: bool, run_mfi: bool,
                   pval_threshold: float, fc_threshold: float,
                   cluster_labels_override=None,
                   include_type_markers: bool = False,
                   names_override: dict | None = None,
                   run_counts: bool = False,
                   af_state=None):
    """
    Run the requested differential tests. Writes results onto ``state`` and
    returns ``(freq_results, mfi_results, counts_results)`` (any may be
    None).

    include_type_markers: Item 11 — when False (default), the MFI branch
        tests 'state'-role channels only, excluding whichever channels
        drove the clustering assignment. When True, restores the
        pre-Item-11 behaviour of testing every selected channel.

    run_counts: Item 14 — when True, additionally runs the negative-
        binomial GLM abundance test on raw cluster counts (run_glm_counts),
        alongside (not instead of) the Frequency/limma test controlled by
        run_freq. Independent of run_freq/run_mfi — any combination of the
        three may be requested in the same call.
    """
    log_stage(log, "DIFFERENTIAL STATISTICS")
    _a, _b, all_rel, group_vec = resolve_group_samples(
        controller, state, cluster_labels_override=cluster_labels_override
    )
    n_clusters = n_clusters_from_labels(
        state, all_rel, cluster_labels_override=cluster_labels_override
    )

    freq_results = mfi_results = counts_results = None

    # Store group metadata so the heatmap can reconstruct per-sample columns
    state.stats_all_rel  = all_rel
    state.stats_group_vec = group_vec

    if run_freq:
        freq_df = compute_frequencies(
            state, all_rel, n_clusters,
            cluster_labels_override=cluster_labels_override,
            names_override=names_override,
        )
        freq_results = run_limma(freq_df, group_vec, pval_threshold, fc_threshold)
        state.freq_results = freq_results
        state.freq_df = freq_df          # raw (samples × features) matrix

    if run_counts:
        counts_df = compute_counts(
            state, all_rel, n_clusters,
            cluster_labels_override=cluster_labels_override,
            names_override=names_override,
        )
        counts_results = run_glm_counts(counts_df, group_vec, pval_threshold, fc_threshold)
        state.counts_results = counts_results
        state.counts_df = counts_df      # raw (samples × features) count matrix

    if run_mfi:
        mfi_channels = resolve_mfi_channels(state, include_type_markers=include_type_markers)
        if not mfi_channels:
            log.warning(
                "MFI testing: no 'state'-role channels selected — check "
                "'Include clustering (type) markers' or assign roles."
            )
        mfi_df = compute_mfis(
            controller, state, all_rel, n_clusters,
            cluster_labels_override=cluster_labels_override,
            channels=mfi_channels,
            names_override=names_override,
            af_state=af_state,
        )
        if mfi_df is not None:
            mfi_results = run_limma(mfi_df, group_vec, pval_threshold, fc_threshold)
            state.mfi_results = mfi_results
            state.mfi_df = mfi_df        # raw (samples × features) matrix

    return freq_results, mfi_results, counts_results

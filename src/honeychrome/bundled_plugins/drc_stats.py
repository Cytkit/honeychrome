"""
drc_stats.py — Differential statistics for the DR/Clustering plugin
===================================================================
Companion to ``dr_clustering_tab.py`` (filename intentionally NOT ``*_tab.py``).

The test engine (design matrices, moderated fits, GLMs, significance) is
shared with other plugins in ``controller_components/differential_stats.py``;
this module assembles the DR/Clustering feature tables and runs it.

Tests cluster abundance and per-cluster marker expression between sample
groups:

* Frequencies — log2 of each cluster's share of a sample's events, tested
  with a moderated linear model (``moderated_stats``: least squares per
  cluster, empirical Bayes variance moderation after Smyth 2004).
* Counts — raw event counts per cluster, tested with a negative-binomial
  GLM (statsmodels) with a log(total events) offset.
* MFIs — mean transformed intensity of each marker within each cluster,
  tested with the same moderated linear model. A cluster with no events in
  a sample is missing (NaN) for that sample, not 0.

Every test builds one design matrix: an intercept, one indicator per
non-baseline group, optional pairing (blocking) levels, and optional
adjustment covariates — numeric columns enter as centred continuous
terms, anything else as treatment-coded factors.

Significance is set by ``apply_significance``:

* TREAT (McCarthy & Smyth 2009): with ``use_treat`` the p-value tests
  whether |effect| exceeds the threshold, instead of filtering on the
  estimate after testing for a non-zero effect.
* FDR: Benjamini–Hochberg, pooled over every comparison ('global') or
  within each comparison ('per_comparison').
* Cluster-first MFI testing (``hierarchical=True``): clusters are screened
  first (Simes-combined p-value of their markers, BH across clusters);
  markers are then tested only inside selected clusters, at the level
  adjusted for the number of clusters selected (Benjamini & Bogomolov
  2014).

Results carry the estimate's standard error and degrees of freedom, so a
threshold change re-derives p-values and significance without refitting.

Contrasts: ``state.contrast_mode`` 'reference' fits once across every
tested group and reads one coefficient per non-reference group;
'pairwise' fits each pair of groups separately. Results carry a
``comparison`` column.

``state.compare_group_a``/``state.compare_group_b`` are used only by
T-REX, not by this module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import drc_pipeline
from drc_logging import get_logger, log_stage
from honeychrome.controller_components import differential_stats as _ds
from honeychrome.controller_components.differential_stats import (
    apply_significance,
    build_contrasts,
    build_design,
    covariate_kind,
    log2_frequencies,
    run_glm_counts,
    run_moderated,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Group resolution
# ---------------------------------------------------------------------------

def suggest_covariates_from_names(
    sample_rel_paths: list[str],
    display_names: dict[str, str] | None = None,
) -> dict:
    """
    Scan sample names for repeated, delimiter-separated tokens that could
    serve as a Group or pairing/covariate column in the Groups & Stats tab
    -- e.g. 'Mouse1_Spleen.fcs' / 'Mouse1_LN.fcs' / 'Mouse2_Spleen.fcs'
    suggests a 2-level Spleen/LN grouping field and a per-mouse pairing
    field, since 'Mouse1' recurs across two different tissue values.

    Tokenizes both the raw filename (Path(rel_path).name) and, when
    given, the experiment's display name for that sample (already
    FCS-keyword-derived when Settings -> Sample name source is
    'tubename'/'fil' -- see experiment_model.py) -- so no separate FCS
    keyword read is needed here.

    Only samples sharing the most common ("modal") token count are
    compared position-by-position; samples with a different token count
    are reported separately under 'irregular_samples' and excluded from
    every suggestion.

    Returns:
        {
            'suggestions': [
                {
                    'field_name':  str,   -- e.g. 'Name field 2 of 3 (filename)'
                    'source':      'filename' | 'display_name',
                    'position':    int,
                    'values':      dict[sample_rel_path, str],
                    'n_distinct':  int,
                    'examples':    list[str],   -- up to 5 distinct values
                    'role_guess':  'group' | 'pairing' | 'covariate',
                },
                ...
            ],
            'irregular_samples': list[str],  -- rel paths excluded from all suggestions
        }
    Returns {'suggestions': [], 'irregular_samples': [...]} if no usable
    pattern is found (e.g. fewer than 2 samples, or every position is
    either constant or fully unique).
    """
    import re
    from pathlib import Path
    from collections import Counter

    def _tokenize(name: str) -> list[str]:
        stem = Path(name).stem
        return [t for t in re.split(r'[^A-Za-z0-9]+', stem) if t]

    sources: list[tuple[str, dict[str, list[str]]]] = []

    filename_tokens = {sp: _tokenize(Path(sp).name) for sp in sample_rel_paths}
    sources.append(('filename', filename_tokens))

    if display_names:
        display_tokens = {
            sp: _tokenize(display_names[sp])
            for sp in sample_rel_paths if sp in display_names
        }
        # Only worth treating as a separate source if it actually differs
        # from the filename tokenization for at least one sample --
        # otherwise it's the same information twice (plain-filename mode).
        if any(display_tokens.get(sp) != filename_tokens.get(sp) for sp in display_tokens):
            sources.append(('display_name', display_tokens))

    all_suggestions = []
    irregular_all: set[str] = set()

    for source_label, tokens_by_sample in sources:
        if len(tokens_by_sample) < 2:
            continue
        lengths = Counter(len(v) for v in tokens_by_sample.values())
        modal_length, _ = lengths.most_common(1)[0]
        regular = {sp: toks for sp, toks in tokens_by_sample.items() if len(toks) == modal_length}
        irregular = set(tokens_by_sample) - set(regular)
        irregular_all |= irregular
        n_samples = len(regular)
        if n_samples < 2 or modal_length == 0:
            continue

        candidates = []
        for pos in range(modal_length):
            values = {sp: toks[pos] for sp, toks in regular.items()}
            n_distinct = len(set(values.values()))
            if n_distinct <= 1 or n_distinct >= n_samples:
                continue  # constant, or no repeats at all -- not usable
            candidates.append((pos, values, n_distinct))

        if not candidates:
            continue

        # Primary "group" candidate: smallest distinct count within a
        # sane range for a comparison group (2-8 levels).
        group_candidates = [c for c in candidates if 2 <= c[2] <= 8]
        primary = min(group_candidates, key=lambda c: c[2]) if group_candidates else None

        for pos, values, n_distinct in sorted(candidates, key=lambda c: c[2]):
            if primary is not None and pos == primary[0]:
                role = 'group'
            elif primary is not None:
                # Cross-tab against the primary group candidate: does this
                # field's value recur across >=2 different group values
                # (pairing candidate), or is it nested inside a single
                # group value (just a correlated covariate)?
                _, group_values, _ = primary
                co_occurrence: dict[str, set[str]] = {}
                for sp, v in values.items():
                    co_occurrence.setdefault(v, set()).add(group_values[sp])
                crosses = sum(1 for s in co_occurrence.values() if len(s) >= 2)
                role = 'pairing' if crosses > len(co_occurrence) / 2 else 'covariate'
            else:
                role = 'covariate'

            distinct_vals = sorted(set(values.values()))
            all_suggestions.append({
                'field_name': f"Name field {pos + 1} of {modal_length} ({source_label})",
                'source': source_label,
                'position': pos,
                'values': values,
                'n_distinct': n_distinct,
                'examples': distinct_vals[:5],
                'role_guess': role,
            })

    # Group suggestions first, then pairing, then covariate; smallest
    # n_distinct first within each role.
    role_order = {'group': 0, 'pairing': 1, 'covariate': 2}
    all_suggestions.sort(key=lambda s: (role_order[s['role_guess']], s['n_distinct']))

    return {'suggestions': all_suggestions, 'irregular_samples': sorted(irregular_all)}

def resolve_test_groups(controller, state, cluster_labels_override=None):
    """
    Resolve every group in ``state.testing_group_selection`` 
    (falls back to every defined group if the selection is empty) to
    rel-path keys present in ``cluster_labels_override`` (or
    ``state.cluster_labels`` if not supplied).

    Groups with < 3 labelled samples are silently DROPPED from the
    returned dict rather than raising — a user testing 5 groups where one
    only has 2 samples should still get results for the other 4;
    ``build_contrasts``/the caller decide whether what's left is enough.

    Returns dict[group_name, list[rel_path]] for qualifying groups only.
    Raises RuntimeError if fewer than 2 groups qualify.
    """
    log_stage(log, "RESOLVE TEST GROUPS")
    cluster_labels = cluster_labels_override if cluster_labels_override is not None \
        else state.cluster_labels
    raw_subdir = controller.experiment.settings['raw']['raw_samples_subdirectory']

    def _to_rel(sp):
        try:
            return str(Path(sp).relative_to(raw_subdir))
        except ValueError:
            return sp

    selection = state.testing_group_selection or list(state.group_names)
    result = {}
    for name in selection:
        rels = []
        for sp, g in state.sample_groups.items():
            if g != name:
                continue
            rel = _to_rel(sp)
            if rel in cluster_labels:
                rels.append(rel)
        log.info("group %r: %d with cluster labels", name, len(rels))
        if len(rels) >= 3:
            result[name] = rels
        else:
            log.info("group %r dropped from testing (only %d qualifying samples)",
                     name, len(rels))

    if len(result) < 2:
        raise RuntimeError(
            "Need at least 2 groups with ≥3 labelled samples each in "
            "'Groups to Test'. Qualifying: "
            + (", ".join(f"{k}={len(v)}" for k, v in result.items()) or "none")
        )
    return result


def missing_covariate_values(state, all_rel: list[str], names: list[str]) -> dict[str, list[str]]:
    """
    {covariate name: [rel paths with no value]} for each of ``names``
    that is missing (or blank) for at least one sample in ``all_rel``.
    """
    return _ds.missing_covariate_values(state.covariates, all_rel, names)


def covariate_frame(state, all_rel: list[str], names: list[str]) -> pd.DataFrame | None:
    """
    Adjustment covariates for ``all_rel`` (rows in that order), as strings.
    Returns None when ``names`` is empty. Raises RuntimeError if any sample
    lacks a value.
    """
    return _ds.covariate_frame(state.covariates, all_rel, names)


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
    SELECTED run's own 'names' dict — see drc_scatter.py), it is
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
    Per-sample RAW event count in each cluster → (n_samples by n_clusters).

    Same shape/index/columns as compute_frequencies() — just unnormalized.
    This is the counterpart the GLM-on-counts path needs alongside the
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


def sample_event_totals(state, all_rel, cluster_labels_override=None) -> np.ndarray:
    """Events per sample, including noise (-1) labels — the frequency denominator."""
    cluster_labels = cluster_labels_override if cluster_labels_override is not None \
        else state.cluster_labels
    return np.array([len(np.asarray(cluster_labels[rel])) for rel in all_rel], dtype=float)


def mfi_feature_table(state, n_clusters: int, channels: list[str],
                      names_override: dict | None = None) -> pd.DataFrame:
    """
    One row per compute_mfis() column, in the same order: feature name,
    cluster display label, cluster index and channel.
    """
    rows = []
    for ch in channels:
        for cl in range(n_clusters):
            label = _label_for(state, cl, names_override)
            rows.append({'feature': f'{label}_{ch}', 'cluster': label,
                         'cluster_id': cl, 'channel': ch})
    return pd.DataFrame(rows)


def resolve_mfi_channels(state, include_type_markers: bool = False) -> list[str]:
    """
    Return the channel list MFI significance testing should use, filtered
    by marker role (diffcyt's type/state split). 'type'
    channels (those that drove the clustering assignment) can be excluded by
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

    Returns (n_samples by (n_clusters · n_channels)), columns ordered
    channel-major (every cluster for the first channel, then the next).
    Loads each sample's selected-channel values ONCE via drc_pipeline, each
    channel on its own configured Transforms-tab scale (Logicle/
    biexponential/linear — the same scale the main cytometry plots and
    Transforms tab use), then iterates channels in memory. A cluster with
    no events in a sample is NaN there: it has no intensity to compare.

    channels: explicit channel list to test. Defaults to every
              selected channel when not supplied,
              so any other caller keeps working unchanged.

    af_state: optional unmixing snapshot, captured on the main thread
        before this (background-thread) call —
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
        mv = drc_pipeline.load_sample_transformed_values(controller, state, rel, channels, af_state=af_state)
        if mv is not None:
            sample_vals[rel] = mv          # (values (n,n_sel), names)

    frames = []
    for ch in channels:
        mfi_mat = np.full((len(all_rel), n_clusters), np.nan, dtype=float)
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
                    mfi_mat[i, cl] = float(np.mean(col[sel]))
        frames.append(pd.DataFrame(
            mfi_mat, index=all_rel,
            columns=[f'{_label_for(state, cl, names_override)}_{ch}' for cl in range(n_clusters)]))

    if not frames:
        log.warning("no MFI features could be built")
        return None
    full = pd.concat(frames, axis=1)
    log.info("MFI matrix: %s", full.shape)
    return full


def compute_sample_mfis(controller, state, all_rel, channels=None,
                        af_state=None) -> pd.DataFrame | None:
    """
    Per-sample mean intensity of each selected channel across the WHOLE
    sample (every gated event), independent of cluster assignment.

    Companion to compute_mfis() (cluster x channel, for the per-cluster
    differential MFI test / Volcano) — that granularity isn't right for a
    heatmap: with dozens of clusters it either needs one heatmap per
    cluster or collapses to whichever single cluster happens to pass
    significance, and a cluster with zero cells in one group produces a
    fabricated near-zero MFI there rather than a real biological zero.
    This is the plain sample-level view for the MFI Heatmap instead:
    no cluster breakdown, no significance filtering.

    Each channel is on its own configured Transforms-tab scale (Logicle/
    biexponential/linear), same as compute_mfis().

    Returns (n_samples x n_channels), NaN where a sample has no events for
    a channel (rather than a fabricated 0).
    """
    log_stage(log, "SAMPLE MFI MATRIX")
    if channels is None:
        channels = [c for c in state.selected_channels if c not in drc_pipeline.META_CHANNELS]
    if not channels:
        log.warning("no sample-level MFI features could be built")
        return None

    sample_vals = {}
    for rel in all_rel:
        mv = drc_pipeline.load_sample_transformed_values(controller, state, rel, channels, af_state=af_state)
        if mv is not None:
            sample_vals[rel] = mv

    mfi_mat = np.full((len(all_rel), len(channels)), np.nan, dtype=float)
    for j, ch in enumerate(channels):
        for i, rel in enumerate(all_rel):
            mv = sample_vals.get(rel)
            if mv is None:
                continue
            values, names = mv
            if ch not in names:
                continue
            col = values[:, names.index(ch)]
            if len(col):
                mfi_mat[i, j] = float(np.mean(col))

    df = pd.DataFrame(mfi_mat, index=all_rel, columns=list(channels))
    log.info("sample MFI matrix: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Sample PCA
# ---------------------------------------------------------------------------

def compute_sample_pca(state, use_freq: bool, use_counts: bool, use_mfi: bool,
                       n_loadings: int = 10) -> dict | None:
    """
    Sample-level PCA over any combination of the per-sample feature
    matrices Run Statistics already computed (``state.freq_df`` /
    ``state.counts_df`` / ``state.mfi_df`` — samples x cluster-features,
    raw scale). Independent of "Viewing comparison": uses every sample
    across every group in ``state.stats_group_vec``, not one pairwise
    comparison at a time.

    Each requested matrix's columns are z-scored independently before
    concatenation, so frequency (%), count (raw events), and MFI
    (log1p-intensity) don't dominate one another on scale alone. A single
    joint 2-component PCA is then fit on the combined, standardized
    matrix.

    Loadings are scaled by sqrt(explained_variance) per axis (a
    correlation-biplot convention) and reduced to the top ``n_loadings``
    by 2-D vector length, so the arrows shown are the most differentiating
    variables rather than every feature in the (potentially huge)
    cluster x channel matrix.

    Returns None if no requested source is available/populated, or if
    fewer than 2 samples or 2 non-degenerate (non-zero-variance) features
    remain. Otherwise a dict:
      'scores':   DataFrame (n_samples x ['PC1','PC2']), index=rel-path,
                  in state.stats_all_rel order.
      'loadings': DataFrame (<=n_loadings x ['PC1','PC2']), index=feature
                  label, already reduced to the top-N.
      'explained_variance_ratio': (float, float) — PC1, PC2.
      'groups':   list[str] aligned 1:1 to scores.index.
      'sources':  list[str] subset of ['Freq','Counts','MFI'] — which
                  matrices actually contributed (for the plot title).
    """
    log_stage(log, "SAMPLE PCA")

    sources, frames = [], []
    if use_freq and state.freq_df is not None and not state.freq_df.empty:
        sources.append('Freq')
        frames.append(state.freq_df.add_prefix('Freq: '))
    if use_counts and state.counts_df is not None and not state.counts_df.empty:
        sources.append('Counts')
        frames.append(state.counts_df.add_prefix('Counts: '))
    if use_mfi and state.mfi_df is not None and not state.mfi_df.empty:
        sources.append('MFI')
        frames.append(state.mfi_df.add_prefix('MFI: '))

    if not frames:
        log.warning("Sample PCA: no requested source is available -- "
                   "run Statistics with Frequencies/Counts/MFIs checked first.")
        return None

    combined = pd.concat(frames, axis=1, join='inner')
    if combined.shape[0] < 2:
        log.warning("Sample PCA: fewer than 2 samples after aligning sources.")
        return None

    # Drop zero-variance columns -- guards the z-score division and keeps
    # a cluster with an identical value in every sample from contributing
    # a meaningless (but numerically NaN-producing) loading.
    stds = combined.std(axis=0, ddof=0)
    combined = combined[stds[stds > 1e-12].index]
    if combined.shape[1] < 2:
        log.warning("Sample PCA: fewer than 2 non-degenerate features to run PCA on.")
        return None

    means = combined.mean(axis=0)
    stds = combined.std(axis=0, ddof=0)
    # Missing MFIs (cluster absent from a sample) sit at the column mean.
    z = ((combined - means) / stds).fillna(0.0)

    n_comp = min(2, z.shape[0], z.shape[1])
    pca = PCA(n_components=n_comp)
    pcs = pca.fit_transform(z.values)

    if n_comp < 2:
        pcs = np.hstack([pcs, np.zeros((pcs.shape[0], 2 - n_comp))])
        explained = list(pca.explained_variance_ratio_) + [0.0] * (2 - n_comp)
    else:
        explained = list(pca.explained_variance_ratio_[:2])

    scores = pd.DataFrame(pcs[:, :2], index=combined.index, columns=['PC1', 'PC2'])

    # Correlation-biplot scaling: component loadings * sqrt(eigenvalue),
    # so arrow length reflects how much variance that feature explains on
    # each axis (not just the raw eigenvector direction).
    load_mat = np.zeros((z.shape[1], 2))
    for c in range(n_comp):
        load_mat[:, c] = pca.components_[c] * np.sqrt(max(pca.explained_variance_[c], 0.0))
    loadings = pd.DataFrame(load_mat, index=combined.columns, columns=['PC1', 'PC2'])

    magnitude = np.sqrt(loadings['PC1'] ** 2 + loadings['PC2'] ** 2)
    top_idx = magnitude.sort_values(ascending=False).head(max(1, int(n_loadings))).index
    loadings_top = loadings.loc[top_idx]

    group_by_rel = dict(zip(state.stats_all_rel, state.stats_group_vec))
    groups = [group_by_rel.get(rel, 'Unassigned') for rel in scores.index]

    log.info("Sample PCA: %s, %d samples, %d features (%d shown as loadings), "
             "PC1=%.1f%% PC2=%.1f%%",
             '+'.join(sources), scores.shape[0], combined.shape[1], len(loadings_top),
             explained[0] * 100, explained[1] * 100)

    return {
        'scores': scores,
        'loadings': loadings_top,
        'explained_variance_ratio': tuple(explained[:2]),
        'groups': groups,
        'sources': sources,
    }


# ---------------------------------------------------------------------------
# Composition views
# ---------------------------------------------------------------------------

def get_counts_table(controller, state, group_var: str = 'sample',
                     cluster_labels_override=None,
                     names_override: dict | None = None) -> pd.DataFrame:
    """
    Raw event counts per cluster (CyCONDOR's getTable(), counts variant).

    group_var: 'sample' → one row per sample (rel-path index).
               'group'  → one row per qualifying tested group (display
                          name), summed across that group's samples.
    Returns (n_rows × n_clusters) DataFrame, columns = cluster display
    labels. Also the natural building block for CSV export alongside the
    existing "Save Statistics CSVs" pattern.
    """
    log_stage(log, "COUNTS TABLE")
    group_rel = resolve_test_groups(controller, state, cluster_labels_override=cluster_labels_override)
    qualifying = [g for g in (state.testing_group_selection or state.group_names) if g in group_rel]
    all_rel = [rel for g in qualifying for rel in group_rel[g]]
    group_vec = [g for g in qualifying for _rel in group_rel[g]]
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
        masks = [np.array([g == name for g in group_vec]) for name in qualifying]
        summed = np.stack([
            per_sample[m].sum(axis=0) if m.any() else np.zeros(n_clusters)
            for m in masks
        ])
        df = pd.DataFrame(summed, index=qualifying, columns=cols)
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


_TOO_FEW_CLUSTERS_HINT = (
    "This usually means the current clustering run produced too few clusters "
    "for differential testing. Increase clustering granularity (e.g. lower "
    "HDBSCAN's min_cluster_size) and re-run, or test a space/run with more "
    "clusters."
)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_statistics(controller, state, run_freq: bool, run_mfi: bool,
                   pval_threshold: float, fc_threshold: float,
                   cluster_labels_override=None,
                   include_type_markers: bool = False,
                   names_override: dict | None = None,
                   run_counts: bool = False,
                   af_state=None,
                   fdr_scope: str = 'global',
                   covariate_names: list[str] | None = None,
                   use_treat: bool = True,
                   mfi_threshold: float | None = None,
                   mfi_hierarchical: bool = True):
    """
    Run the requested differential tests. Writes results onto ``state`` and
    returns ``(freq_results, mfi_results, counts_results)`` (any may be
    None).

    Resolves state.testing_group_selection (falling back to every defined
    group), builds the contrast list from state.contrast_mode/
    reference_group, the pairing vector (if state.paired and
    state.pairing_variable names a covariates column) and the adjustment
    covariates, all shared across freq/counts/mfi so the three tests
    report the same comparisons from the same design.

    fc_threshold  : |log2 fold change| threshold for Frequencies and Counts.
    mfi_threshold : |difference| threshold for MFIs, in transformed units
                    (defaults to fc_threshold).
    covariate_names: state.covariates columns to adjust for.
    use_treat     : test against the thresholds (TREAT) rather than zero.
    mfi_hierarchical: cluster-first testing of MFIs.
    include_type_markers: when False, the MFI branch tests 'state'-role
        channels only. When True, tests every selected channel.
    run_counts: also run the negative-binomial GLM on raw counts.
    """
    log_stage(log, "DIFFERENTIAL STATISTICS")
    if mfi_threshold is None:
        mfi_threshold = fc_threshold

    group_rel = resolve_test_groups(controller, state, cluster_labels_override=cluster_labels_override)
    qualifying = [g for g in (state.testing_group_selection or state.group_names) if g in group_rel]
    contrasts = build_contrasts(qualifying, state.contrast_mode, state.reference_group)

    all_rel = [rel for g in qualifying for rel in group_rel[g]]
    group_vec = [g for g in qualifying for _rel in group_rel[g]]

    pairing_vec = None
    if state.paired and state.pairing_variable and state.covariates is not None \
            and state.pairing_variable in state.covariates.columns:
        cov = state.covariates[state.pairing_variable]
        # Any rel missing a covariate value gets its own singleton blocking
        # level rather than failing the whole run.
        pairing_vec = [str(cov[rel]) if rel in cov.index else f"__unpaired_{rel}"
                       for rel in all_rel]

    adjust = [c for c in (covariate_names or [])
              if not (pairing_vec is not None and c == state.pairing_variable)]
    covariates = covariate_frame(state, all_rel, adjust)
    if covariates is not None:
        log.info("adjusting for covariates: %s", list(covariates.columns))

    n_clusters = n_clusters_from_labels(
        state, all_rel, cluster_labels_override=cluster_labels_override
    )

    freq_results = mfi_results = counts_results = None

    # Group metadata for the heatmap and the 'Viewing comparison' selector.
    state.stats_all_rel = all_rel
    state.stats_group_vec = group_vec
    state.stats_comparisons = contrasts

    if run_freq or run_counts:
        counts_df = compute_counts(
            state, all_rel, n_clusters,
            cluster_labels_override=cluster_labels_override,
            names_override=names_override,
        )

    if run_freq:
        freq_df = compute_frequencies(
            state, all_rel, n_clusters,
            cluster_labels_override=cluster_labels_override,
            names_override=names_override,
        )
        totals = sample_event_totals(state, all_rel, cluster_labels_override)
        freq_results = run_moderated(
            log2_frequencies(counts_df, totals), group_vec, contrasts, state.contrast_mode,
            pval_threshold, fc_threshold, pairing_vec=pairing_vec, fdr_scope=fdr_scope,
            covariates=covariates, use_treat=use_treat,
            too_few_hint=_TOO_FEW_CLUSTERS_HINT,
        )
        state.freq_results = freq_results
        state.freq_df = freq_df          # % per cluster (display)

    if run_counts:
        counts_results = run_glm_counts(counts_df, group_vec, contrasts, state.contrast_mode,
                                        pval_threshold, fc_threshold, pairing_vec=pairing_vec,
                                        fdr_scope=fdr_scope, covariates=covariates,
                                        use_treat=use_treat)
        state.counts_results = counts_results
        state.counts_df = counts_df      # raw (samples × clusters) counts

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
            meta = mfi_feature_table(state, n_clusters, mfi_channels, names_override)
            meta = meta[meta['feature'].isin(mfi_df.columns)].drop_duplicates('feature')
            mfi_results = run_moderated(
                mfi_df, group_vec, contrasts, state.contrast_mode,
                pval_threshold, mfi_threshold, pairing_vec=pairing_vec, fdr_scope=fdr_scope,
                covariates=covariates, use_treat=use_treat,
                feature_meta=meta, hierarchical=mfi_hierarchical,
                too_few_hint=_TOO_FEW_CLUSTERS_HINT,
            )
            state.mfi_results = mfi_results
            state.mfi_df = mfi_df        # raw (samples × cluster·channel) MFIs

        state.mfi_sample_df = compute_sample_mfis(
            controller, state, all_rel, channels=mfi_channels, af_state=af_state,
        )

    return freq_results, mfi_results, counts_results

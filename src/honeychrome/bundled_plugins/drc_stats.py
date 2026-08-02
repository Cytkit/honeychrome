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
  • S1 — feature labels are recovered via ``tt.index`` (not row position), so
         ``topTable``'s sort order never causes a relabelling mismatch.
  • S2 — MFI channel values come from drc_pipeline (correct channel→column map),
         not a filtered-index lookup against full-width data.
  • S6 — each sample's FCS is loaded ONCE, not once per channel.

Group bookkeeping note (Item 13): ``state.sample_groups`` values are the
group's own name (a free-form user-defined string) or the reserved
'Unassigned' sentinel — there is no longer a slot/display-name split (that
indirection, and the S3 bug it existed to guard against, are both gone: a
rename now rewrites every affected sample_groups entry directly).

Phase 2 adds true N-group testing: ``state.testing_group_selection`` picks
which groups participate in Frequency/Counts/MFI/Confusion-Matrix/
Composition-by-group, ``state.contrast_mode`` ('reference' | 'pairwise')
plus ``state.reference_group`` decide the contrast(s), and
``state.paired``/``state.pairing_variable`` add an optional fixed-effect
blocking term (patsy formula ``+ C(pair_id)`` — InMoose documents no
``duplicateCorrelation``-style random-effect blocking, so this is a
fixed-effect-only approach; see the Item 13 change doc §2.2). Results carry
a ``comparison`` column when more than one contrast was run.

``state.compare_group_a``/``state.compare_group_b`` remain — but only for
T-REX now (its neighbour-fraction score is only defined for two
conditions), not for anything in this module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import drc_pipeline
from drc_logging import get_logger, log_stage

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
    Resolve every group in ``state.testing_group_selection`` (Item 13 phase
    2 — falls back to every defined group if the selection is empty) to
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


def build_contrasts(group_names: list[str], mode: str,
                    reference: str | None) -> list[tuple[str, str]]:
    """
    Item 13 phase 2. Returns the list of (baseline, other) pairs to test.

    mode='reference': (reference, other) for every OTHER qualifying group —
        one joint fit, multiple coefficients (§2.1).
    mode='pairwise':  every unique pair among group_names, in group_names
        order — each pair gets its own independent fit (§2.1).
    """
    if mode == 'reference':
        if reference not in group_names:
            raise RuntimeError(
                f"Reference group {reference!r} is not one of the groups "
                f"qualifying for this run ({group_names!r})."
            )
        return [(reference, g) for g in group_names if g != reference]
    elif mode == 'pairwise':
        pairs = []
        for i, g1 in enumerate(group_names):
            for g2 in group_names[i + 1:]:
                pairs.append((g1, g2))
        return pairs
    else:
        raise ValueError(f"contrast_mode must be 'reference' or 'pairwise', got {mode!r}")


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
        mv = drc_pipeline.load_sample_marker_values(controller, state, rel, af_state=af_state)
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
                mfi_mat[i, j] = float(np.mean(np.log1p(np.maximum(col, 0.0))))

    df = pd.DataFrame(mfi_mat, index=all_rel, columns=list(channels))
    log.info("sample MFI matrix: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Sample PCA (Item 16)
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
    z = (combined - means) / stds

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
    resolve_test_groups()'s ≥3-per-group check).

    Returns (n_clusters × n_groups) DataFrame — one column per qualifying
    group in state.testing_group_selection (Item 13 phase 2: no longer
    limited to exactly two).
    """
    log_stage(log, "CONFUSION MATRIX")
    group_rel = resolve_test_groups(controller, state, cluster_labels_override=cluster_labels_override)
    qualifying = [g for g in (state.testing_group_selection or state.group_names) if g in group_rel]
    all_rel = [rel for g in qualifying for rel in group_rel[g]]
    cluster_labels = cluster_labels_override if cluster_labels_override is not None \
        else state.cluster_labels
    n_clusters = n_clusters_from_labels(
        state, all_rel, cluster_labels_override=cluster_labels_override
    )

    conf = np.zeros((n_clusters, len(qualifying)), dtype=float)
    for gi, grp in enumerate(qualifying):
        rels = group_rel[grp]
        pooled = np.concatenate([np.asarray(cluster_labels[rel]) for rel in rels])
        total = len(pooled)
        if total == 0:
            continue
        scale = normalize_to / total
        for cl in range(n_clusters):
            conf[cl, gi] = float(np.sum(pooled == cl)) * scale

    df = pd.DataFrame(conf, columns=qualifying,
                      index=[_label_for(state, cl, names_override) for cl in range(n_clusters)])
    log.info("confusion matrix: %s", df.shape)
    return df


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


# ---------------------------------------------------------------------------
# limma
# ---------------------------------------------------------------------------

def _limma_fit_one(data_df: pd.DataFrame, group_vec: list[str], baseline: str,
                   other: str, pairing_vec: list[str] | None) -> pd.DataFrame:
    """
    One lmFit → eBayes → topTable call for a single baseline-vs-other
    coefficient, against whatever samples/groups are already in data_df/
    group_vec (the caller decides whether that's the full N-group set —
    'reference' mode, calling this once per contrast against ONE shared
    fit — or a 2-group subset — 'pairwise' mode, calling this once per
    pair with its own fit). Shared by both modes in run_limma() below.
    """
    from inmoose.limma import lmFit, eBayes, topTable
    import patsy

    sample_info = pd.DataFrame({'group': group_vec})
    formula = f"~ C(group, Treatment({baseline!r}))"
    if pairing_vec is not None:
        sample_info['pair_id'] = pairing_vec
        formula += " + C(pair_id)"
    design = patsy.dmatrix(formula, sample_info, return_type='dataframe')

    if np.linalg.matrix_rank(design.values) < design.shape[1]:
        raise RuntimeError(
            f"Design matrix for {other!r} vs {baseline!r} is rank-deficient — "
            "check the pairing variable for missing or group-confounded values."
        )

    expr = data_df.values.T.astype(float)        # (features, samples)
    n_features = expr.shape[0]
    if n_features < 3:
        # See eBayes' Infdf/squeezeVar note (unchanged from Phase 1) — fails
        # clearly here instead of a bare KeyError deep inside inmoose.
        raise RuntimeError(
            f"Only {n_features} feature(s) to test (need >= 3) for "
            f"{other!r} vs {baseline!r} — this usually means the current "
            "clustering run produced too few clusters for differential "
            "testing. Increase clustering granularity (e.g. lower HDBSCAN's "
            "min_cluster_size) and re-run, or test a space/run with more "
            "clusters."
        )

    design_col_name = f"C(group, Treatment({baseline!r}))[T.{other}]"
    try:
        coef_idx = list(design.columns).index(design_col_name)
    except ValueError:
        raise RuntimeError(
            f"Could not find expected coefficient {design_col_name!r} among "
            f"design columns {list(design.columns)!r} — patsy named this "
            "contrast differently than expected; the fit cannot proceed."
        )

    fit = eBayes(lmFit(expr, design=design))
    # inmoose's fit.coefficients doesn't carry patsy's column names — it
    # labels columns generically as 'column0', 'column1', ... in the same
    # order as the design matrix, so translate position -> that name.
    coef_col = f"column{coef_idx}"
    tt = topTable(fit, coef=coef_col, number=np.inf, adjust_method='fdr_bh', sort_by='p')

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
    tt['comparison'] = f"{other} vs {baseline}"
    return tt


def run_limma(data_df: pd.DataFrame, group_vec: list[str],
              contrasts: list[tuple[str, str]], mode: str,
              pval_threshold: float, fc_threshold: float,
              pairing_vec: list[str] | None = None) -> pd.DataFrame:
    """
    lmFit → eBayes → topTable, generalised to N groups and multiple
    contrasts (Item 13 phase 2).

    data_df   : rows = samples (index aligned to group_vec/pairing_vec),
                columns = features (clusters / channel-clusters)
    group_vec : group name per row, aligned to data_df.index
    contrasts : list of (baseline, other) pairs from build_contrasts()
    mode      : 'reference' — ONE joint fit across every sample in
                  data_df/group_vec (borrows variance-shrinkage strength
                  across all selected groups at once); one topTable() call
                  per contrast against that single fit.
                'pairwise'  — each contrast gets its OWN independent fit,
                  with data_df/group_vec/pairing_vec subset to just that
                  pair's samples first. No documented InMoose equivalent to
                  limma's makeContrasts for extracting custom linear
                  combinations from one N-level fit, so each pairwise
                  comparison is a fresh, self-contained 2-group test — see
                  §2.1 of the Item 13 change doc.
    pairing_vec: optional per-row blocking id (e.g. donor), same order as
                group_vec; added as a fixed-effect term in the formula.

    Returns one combined DataFrame — feature, logFC, AveExpr, t, P.Value,
    adj.P.Val, B, comparison, significant — concatenated across every
    requested contrast. A single 2-group call (Phase 1's only case) returns
    the exact same rows as before, with one added 'comparison' column.
    """
    frames = []

    if mode == 'reference':
        baseline = contrasts[0][0]
        for base, other in contrasts:
            assert base == baseline, "reference mode expects a shared baseline"
            frames.append(_limma_fit_one(data_df, group_vec, base, other, pairing_vec))
        # NOTE: this still calls lmFit/eBayes fresh per contrast rather than
        # sharing ONE fit object across coefficients — see the caveat below.

    elif mode == 'pairwise':
        for base, other in contrasts:
            mask = [g in (base, other) for g in group_vec]
            sub_rel = [rel for rel, m in zip(data_df.index, mask) if m]
            sub_df = data_df.loc[sub_rel]
            sub_group_vec = [g for g, m in zip(group_vec, mask) if m]
            sub_pairing_vec = (
                [p for p, m in zip(pairing_vec, mask) if m] if pairing_vec is not None else None
            )
            n_a, n_b = sub_group_vec.count(base), sub_group_vec.count(other)
            if n_a < 3 or n_b < 3:
                raise RuntimeError(
                    f"Not enough samples for {other} vs {base}: "
                    f"{base}={n_a}, {other}={n_b}."
                )
            frames.append(_limma_fit_one(sub_df, sub_group_vec, base, other, sub_pairing_vec))
    else:
        raise ValueError(f"mode must be 'reference' or 'pairwise', got {mode!r}")

    combined = pd.concat(frames, ignore_index=True)

    # Item 13 phase 2 addendum (§2.7): adj.P.Val above is corrected
    # SEPARATELY per contrast (topTable's own per-call default). Add a
    # GLOBAL correction -- one BH-FDR pass over every finite P.Value in
    # the whole combined table -- and use that for 'significant' instead.
    # Identical to adj.P.Val when there's only one contrast (Phase 1's
    # 2-group case), so this changes nothing for that path.
    from statsmodels.stats.multitest import multipletests
    combined['adj.P.Val.global'] = np.nan
    valid = np.isfinite(combined['P.Value'].values.astype(float))
    if valid.any():
        combined.loc[valid, 'adj.P.Val.global'] = multipletests(
            combined.loc[valid, 'P.Value'].values.astype(float), method='fdr_bh'
        )[1]

    combined['significant'] = (
        (combined['adj.P.Val.global'] <= pval_threshold) &
        (combined['logFC'].abs() >= fc_threshold)
    )
    log.info("limma: %d rows across %d comparison(s), %d significant (global FDR)",
             len(combined), combined['comparison'].nunique(),
             int(combined['significant'].sum()))
    return combined


def _glm_fit_one_cluster(y: np.ndarray, design: np.ndarray):
    """Poisson-then-NB fit for one cluster's raw counts (unchanged
    Cameron & Trivedi auxiliary-OLS alpha estimate from Phase 1/Item 14).
    Returns the fitted GLM result, or None if even the Poisson fit fails."""
    import statsmodels.api as sm

    try:
        poisson_fit = sm.GLM(y, design, family=sm.families.Poisson()).fit()
    except Exception as e:
        log.warning("Poisson GLM failed (%s) — marking untestable", e)
        return None
    try:
        mu = poisson_fit.mu
        aux_y = ((y - mu) ** 2 - y) / mu
        alpha = float(sm.OLS(aux_y, mu).fit().params[0])
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("non-positive alpha estimate")
        return sm.GLM(y, design, family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    except Exception as e:
        log.warning("NB GLM failed (%s), falling back to Poisson", e)
        return poisson_fit


def _glm_counts_one_contrast(counts_df: pd.DataFrame, group_vec: list[str],
                             baseline: str, other: str,
                             pairing_vec: list[str] | None) -> pd.DataFrame:
    """One design + per-cluster NB/Poisson fit + FDR correction for a single
    baseline-vs-other coefficient. Shared by both contrast modes in
    run_glm_counts() below, mirroring _limma_fit_one()."""
    import patsy
    from statsmodels.stats.multitest import multipletests

    sample_info = pd.DataFrame({'group': group_vec})
    formula = f"~ C(group, Treatment({baseline!r}))"
    if pairing_vec is not None:
        sample_info['pair_id'] = pairing_vec
        formula += " + C(pair_id)"
    design = patsy.dmatrix(formula, sample_info, return_type='dataframe')

    if np.linalg.matrix_rank(design.values) < design.shape[1]:
        raise RuntimeError(
            f"Design matrix for {other!r} vs {baseline!r} is rank-deficient — "
            "check the pairing variable for missing or group-confounded values."
        )
    coef_idx = list(design.columns).index(f"C(group, Treatment({baseline!r}))[T.{other}]")
    design_mat = design.values

    LN2 = np.log(2.0)
    clusters = list(counts_df.columns)
    logfc = np.full(len(clusters), np.nan)
    tvals = np.full(len(clusters), np.nan)
    pvals = np.full(len(clusters), np.nan)

    for i, cl in enumerate(clusters):
        y = counts_df[cl].values.astype(float)
        fit = _glm_fit_one_cluster(y, design_mat)
        if fit is None:
            continue
        logfc[i] = fit.params[coef_idx] / LN2
        tvals[i] = fit.tvalues[coef_idx]
        pvals[i] = fit.pvalues[coef_idx]

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
    tt['comparison'] = f"{other} vs {baseline}"
    return tt


def run_glm_counts(counts_df: pd.DataFrame, group_vec: list[str],
                   contrasts: list[tuple[str, str]], mode: str,
                   pval_threshold: float, fc_threshold: float,
                   pairing_vec: list[str] | None = None) -> pd.DataFrame:
    """
    Per-cluster negative-binomial GLM differential abundance test on raw
    event counts (Item 14), generalised to N groups/multiple contrasts —
    same contrasts/mode/pairing_vec semantics as run_limma() (Item 13
    phase 2), kept in lockstep per the docstring's own promise.

    Unlike run_limma()'s single shared eBayes fit in 'reference' mode, each
    (cluster, contrast) pair here is its own independent per-cluster GLM —
    there's no cross-contrast fit to share, since statsmodels' GLM doesn't
    have an eBayes-style moderation step that would benefit from it. In
    'reference' mode this still means ONE design matrix built once (all
    selected groups' samples, one column per non-reference group), reused
    across every cluster; 'pairwise' mode subsets samples/design per pair,
    same as run_limma().

    Returns the same feature/logFC/P.Value/adj.P.Val/t/comparison/
    significant schema run_limma() produces.
    """
    frames = []

    if mode == 'reference':
        for base, other in contrasts:
            frames.append(_glm_counts_one_contrast(counts_df, group_vec, base, other, pairing_vec))

    elif mode == 'pairwise':
        for base, other in contrasts:
            mask = [g in (base, other) for g in group_vec]
            sub_rel = [rel for rel, m in zip(counts_df.index, mask) if m]
            sub_df = counts_df.loc[sub_rel]
            sub_group_vec = [g for g, m in zip(group_vec, mask) if m]
            sub_pairing_vec = (
                [p for p, m in zip(pairing_vec, mask) if m] if pairing_vec is not None else None
            )
            n_a, n_b = sub_group_vec.count(base), sub_group_vec.count(other)
            if n_a < 3 or n_b < 3:
                raise RuntimeError(
                    f"Not enough samples for {other} vs {base}: "
                    f"{base}={n_a}, {other}={n_b}."
                )
            frames.append(_glm_counts_one_contrast(sub_df, sub_group_vec, base, other, sub_pairing_vec))
    else:
        raise ValueError(f"mode must be 'reference' or 'pairwise', got {mode!r}")

    combined = pd.concat(frames, ignore_index=True)

    # Item 13 phase 2 addendum (§2.7) -- same global-vs-separate fix as
    # run_limma(): adj.P.Val above is corrected separately per contrast
    # (each _glm_counts_one_contrast() call runs its own multipletests()
    # over just that contrast's clusters). Add a global BH-FDR pass over
    # every finite P.Value in the whole combined table for 'significant'.
    from statsmodels.stats.multitest import multipletests
    combined['adj.P.Val.global'] = np.nan
    valid = np.isfinite(combined['P.Value'].values.astype(float))
    if valid.any():
        combined.loc[valid, 'adj.P.Val.global'] = multipletests(
            combined.loc[valid, 'P.Value'].values.astype(float), method='fdr_bh'
        )[1]

    combined['significant'] = (
        (combined['adj.P.Val.global'] <= pval_threshold) &
        (combined['logFC'].abs() >= fc_threshold)
    )
    log.info("GLM counts: %d rows across %d comparison(s), %d significant (global FDR), %d untestable",
             len(combined), combined['comparison'].nunique(),
             int(combined['significant'].sum()), int(combined['logFC'].isna().sum()))
    return combined


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

    Item 13 phase 2: resolves state.testing_group_selection (falling back
    to every defined group), builds the contrast list from
    state.contrast_mode/reference_group, and — if state.paired is set and
    state.pairing_variable names a real state.covariates column — builds a
    per-sample blocking vector, all shared across freq/counts/mfi so the
    three tests always report the exact same set of comparisons.

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
        # level (via the f-string fallback) rather than crashing the whole
        # run — it simply gets no pairing benefit, instead of no results.
        pairing_vec = [str(cov[rel]) if rel in cov.index else f"__unpaired_{rel}"
                       for rel in all_rel]

    n_clusters = n_clusters_from_labels(
        state, all_rel, cluster_labels_override=cluster_labels_override
    )

    freq_results = mfi_results = counts_results = None

    # Store group metadata so the heatmap can reconstruct per-sample columns
    # and so the tab can populate its 'Viewing comparison:' selector.
    state.stats_all_rel = all_rel
    state.stats_group_vec = group_vec
    state.stats_comparisons = contrasts

    if run_freq:
        freq_df = compute_frequencies(
            state, all_rel, n_clusters,
            cluster_labels_override=cluster_labels_override,
            names_override=names_override,
        )
        freq_results = run_limma(freq_df, group_vec, contrasts, state.contrast_mode,
                                 pval_threshold, fc_threshold, pairing_vec=pairing_vec)
        state.freq_results = freq_results
        state.freq_df = freq_df          # raw (samples × features) matrix

    if run_counts:
        counts_df = compute_counts(
            state, all_rel, n_clusters,
            cluster_labels_override=cluster_labels_override,
            names_override=names_override,
        )
        counts_results = run_glm_counts(counts_df, group_vec, contrasts, state.contrast_mode,
                                        pval_threshold, fc_threshold, pairing_vec=pairing_vec)
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
            mfi_results = run_limma(mfi_df, group_vec, contrasts, state.contrast_mode,
                                    pval_threshold, fc_threshold, pairing_vec=pairing_vec)
            state.mfi_results = mfi_results
            state.mfi_df = mfi_df        # raw (samples × features) matrix

        state.mfi_sample_df = compute_sample_mfis(
            controller, state, all_rel, channels=mfi_channels, af_state=af_state,
        )

    return freq_results, mfi_results, counts_results

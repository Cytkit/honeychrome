"""
differential_stats.py — differential testing between sample groups
===================================================================

Shared engine for group comparisons of per-sample features: cluster or
gated-population frequencies, raw counts, and per-population marker
intensities. Used by the DR/Clustering and Automated Gating plugins.

* ``run_moderated`` — moderated linear model per feature (``moderated_stats``:
  least squares, empirical Bayes variance moderation after Smyth 2004).
* ``run_glm_counts`` — negative-binomial GLM per feature on raw counts, with
  a log library-size offset. The offset is the sample's total by default, or
  a per-feature denominator (e.g. each population's parent count).
* ``apply_significance`` — TREAT p-values (McCarthy & Smyth 2009),
  Benjamini–Hochberg FDR pooled over every comparison or within each, and
  optional two-stage testing of markers within clusters (Benjamini &
  Bogomolov 2014).

Every test builds one design matrix: an intercept, one indicator per
non-baseline group, optional pairing (blocking) levels, and optional
adjustment covariates — numeric columns enter as centred continuous terms,
anything else as treatment-coded factors.

Frequency scales:

* ``log2_frequencies`` — log2 percentage of a total; differences are log2
  fold changes. Suited to rare clusters.
* ``logit_proportions`` — log2 odds of a population within its parent, with
  half an event added to both the population and its complement. Suited to
  gated populations, whose fractions often sit near 0 or 1.

Results carry the estimate's standard error and degrees of freedom, so a
threshold change re-derives p-values and significance without refitting.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from honeychrome.controller_components import moderated_stats as ms

log = logging.getLogger(__name__)

__all__ = [
    'build_contrasts', 'covariate_kind', 'missing_covariate_values',
    'covariate_frame', 'build_design', 'log2_frequencies',
    'logit_proportions', 'run_moderated', 'run_glm_counts',
    'apply_significance',
]


# ---------------------------------------------------------------------------
# Contrasts
# ---------------------------------------------------------------------------

def build_contrasts(group_names: list[str], mode: str,
                    reference: str | None) -> list[tuple[str, str]]:
    """
    Returns the list of (baseline, other) pairs to test.

    mode='reference': (reference, other) for every OTHER qualifying group —
        one joint fit, multiple coefficients.
    mode='pairwise':  every unique pair among group_names, in group_names
        order — each pair gets its own independent fit.
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


# ---------------------------------------------------------------------------
# Design matrix and covariates
# ---------------------------------------------------------------------------

def _is_numeric_column(values) -> bool:
    """True when every non-blank value parses as a finite float."""
    seen = False
    for v in values:
        text = str(v).strip()
        if not text:
            continue
        try:
            if not np.isfinite(float(text)):
                return False
        except ValueError:
            return False
        seen = True
    return seen


def covariate_kind(values) -> str:
    """'numeric' when every filled value is a number, else 'categorical'."""
    return 'numeric' if _is_numeric_column(values) else 'categorical'


def missing_covariate_values(covariates: pd.DataFrame | None, sample_ids: list[str],
                             names: list[str]) -> dict[str, list[str]]:
    """
    {covariate name: [sample ids with no value]} for each of ``names`` that
    is missing (or blank) for at least one sample in ``sample_ids``.
    ``covariates`` is indexed by sample id, one column per covariate.
    """
    missing: dict[str, list[str]] = {}
    cov = covariates
    for name in names:
        bad = [
            sid for sid in sample_ids
            if cov is None or name not in cov.columns or sid not in cov.index
            or not str(cov.loc[sid, name]).strip()
        ]
        if bad:
            missing[name] = bad
    return missing


def covariate_frame(covariates: pd.DataFrame | None, sample_ids: list[str],
                    names: list[str]) -> pd.DataFrame | None:
    """
    Adjustment covariates for ``sample_ids`` (rows in that order), as
    strings. Returns None when ``names`` is empty. Raises RuntimeError if
    any sample lacks a value.
    """
    names = [n for n in names if n]
    if not names:
        return None
    missing = missing_covariate_values(covariates, sample_ids, names)
    if missing:
        detail = "; ".join(
            f"'{name}': " + ", ".join(Path(r).stem for r in rels[:5])
            + (" …" if len(rels) > 5 else "")
            for name, rels in missing.items()
        )
        raise RuntimeError(f"Adjustment covariates have missing values — {detail}")
    return pd.DataFrame(
        {name: [str(covariates.loc[sid, name]).strip() for sid in sample_ids]
         for name in names},
        index=list(sample_ids),
    )


def build_design(group_vec: list[str], baseline: str,
                 pairing_vec: list[str] | None = None,
                 covariates: pd.DataFrame | None = None):
    """
    Treatment-coded design matrix.

    Columns, in order: intercept; one indicator per non-baseline group (in
    order of first appearance); one indicator per pairing level except the
    first in sorted order; then each covariate — numeric columns centred,
    categorical columns as indicators for every level except the first in
    sorted order. Covariates that are constant over these samples are
    dropped.

    Returns (X, column_names, group_columns) where group_columns maps each
    non-baseline group to its column index. Raises RuntimeError naming the
    term that makes the design rank-deficient.
    """
    n = len(group_vec)
    cols = [np.ones(n)]
    names = ['(Intercept)']
    group_cols: dict[str, int] = {}
    for g in dict.fromkeys(group_vec):
        if g == baseline:
            continue
        group_cols[g] = len(cols)
        cols.append(np.array([1.0 if x == g else 0.0 for x in group_vec]))
        names.append(f"group[{g}]")
    if baseline not in group_vec:
        raise RuntimeError(f"Baseline group {baseline!r} has no samples in this comparison.")

    def _rank_ok() -> bool:
        return np.linalg.matrix_rank(np.column_stack(cols)) == len(cols)

    if pairing_vec is not None:
        levels = sorted(set(pairing_vec))
        for lvl in levels[1:]:
            cols.append(np.array([1.0 if x == lvl else 0.0 for x in pairing_vec]))
            names.append(f"pair[{lvl}]")
        if not _rank_ok():
            raise RuntimeError(
                "The design is rank-deficient with the pairing variable — check "
                "for missing values, or pairing levels confounded with group."
            )

    if covariates is not None:
        for name in covariates.columns:
            values = covariates[name].tolist()
            if len(set(values)) < 2:
                log.info("covariate %r is constant over these samples — dropped", name)
                continue
            added = []
            if covariate_kind(values) == 'numeric':
                x = np.array([float(v) for v in values])
                added.append((x - x.mean(), f"{name}"))
            else:
                for lvl in sorted(set(values))[1:]:
                    added.append((np.array([1.0 if v == lvl else 0.0 for v in values]),
                                  f"{name}[{lvl}]"))
            for col, label in added:
                cols.append(col)
                names.append(label)
            if not _rank_ok():
                raise RuntimeError(
                    f"Covariate '{name}' is confounded with group or pairing in this "
                    "comparison (e.g. every sample of one group shares one level). "
                    "Remove it from the adjustment list."
                )

    X = np.column_stack(cols)
    if not _rank_ok():
        raise RuntimeError("The design matrix is rank-deficient.")
    return X, names, group_cols


def _subset_rows(values, mask):
    return None if values is None else [v for v, m in zip(values, mask) if m]


# ---------------------------------------------------------------------------
# Frequency scales
# ---------------------------------------------------------------------------

def log2_frequencies(counts_df: pd.DataFrame, totals) -> pd.DataFrame:
    """
    log2 of each feature's percentage of the sample's events, with half an
    event added to every count (and one to every total) so empty features
    stay finite: log2((count + 0.5) / (total + 1) · 100). Differences on
    this scale are log2 fold changes of frequency.
    """
    totals = np.asarray(totals, dtype=float)[:, None]
    vals = np.log2((counts_df.values.astype(float) + 0.5) / (totals + 1.0) * 100.0)
    return pd.DataFrame(vals, index=counts_df.index, columns=counts_df.columns)


def logit_proportions(counts_df: pd.DataFrame, denominators_df: pd.DataFrame) -> pd.DataFrame:
    """
    Empirical log2 odds of each feature within its own denominator:
    log2((count + 0.5) / (denominator − count + 0.5)).

    ``denominators_df`` has the same shape, index and columns as
    ``counts_df`` (e.g. each population's parent count per sample). Adding
    half an event to both the population and its complement keeps 0% and
    100% finite. Differences on this scale are log2 odds ratios.
    """
    counts = counts_df.values.astype(float)
    denom = denominators_df.reindex(index=counts_df.index,
                                    columns=counts_df.columns).values.astype(float)
    if np.any(counts > denom):
        raise ValueError("A population count exceeds its denominator.")
    vals = np.log2((counts + 0.5) / (denom - counts + 0.5))
    return pd.DataFrame(vals, index=counts_df.index, columns=counts_df.columns)


# ---------------------------------------------------------------------------
# Moderated linear model
# ---------------------------------------------------------------------------

_TOO_FEW_FEATURES = (
    "Only {n} feature(s) to test (need >= 3) for {other!r} vs {base!r}."
)


def _moderated_fit_contrasts(data_df: pd.DataFrame, group_vec: list[str],
                             contrasts: list[tuple[str, str]],
                             pairing_vec: list[str] | None,
                             covariates: pd.DataFrame | None,
                             too_few_hint: str = '') -> pd.DataFrame:
    """
    One moderated fit over data_df's samples, read out for every contrast
    in ``contrasts`` (all sharing one baseline). Returns the stacked
    per-contrast tables with 'feature' and 'comparison' columns.
    """
    base = contrasts[0][0]
    X, _names, group_cols = build_design(group_vec, base, pairing_vec, covariates)
    expr = data_df.values.T.astype(float)            # (features, samples)
    n_features = expr.shape[0]
    if n_features < 3:
        msg = _TOO_FEW_FEATURES.format(n=n_features, other=contrasts[0][1], base=base)
        raise RuntimeError(f"{msg} {too_few_hint}".strip())
    try:
        fit = ms.e_bayes(ms.lm_fit(expr, X))
    except ValueError as exc:
        raise RuntimeError(
            f"{exc} ({contrasts[0][1]!r} vs {base!r}). Too few features "
            "have enough observed samples to fit."
        ) from exc
    log.info("moderated fit: %d features, %d samples, d0=%.3g, s0²=%.3g",
             n_features, X.shape[0], fit.df_prior, fit.s2_prior)

    frames = []
    for b, other in contrasts:
        tt = ms.top_table(fit, group_cols[other])
        tt.insert(0, 'feature', list(data_df.columns))
        tt['comparison'] = f"{other} vs {b}"
        frames.append(tt)
    return pd.concat(frames, ignore_index=True)


def _pair_subsets(data_df, group_vec, contrasts, pairing_vec, covariates):
    """Yield (sub_df, sub_groups, sub_pairing, sub_covariates, base, other) per pair."""
    for base, other in contrasts:
        mask = [g in (base, other) for g in group_vec]
        sub_df = data_df.loc[[sid for sid, m in zip(data_df.index, mask) if m]]
        sub_groups = _subset_rows(group_vec, mask)
        n_a, n_b = sub_groups.count(base), sub_groups.count(other)
        if n_a < 3 or n_b < 3:
            raise RuntimeError(
                f"Not enough samples for {other} vs {base}: {base}={n_a}, {other}={n_b}."
            )
        sub_cov = covariates.loc[sub_df.index] if covariates is not None else None
        yield sub_df, sub_groups, _subset_rows(pairing_vec, mask), sub_cov, base, other


def run_moderated(data_df: pd.DataFrame, group_vec: list[str],
                  contrasts: list[tuple[str, str]], mode: str,
                  pval_threshold: float, fc_threshold: float,
                  pairing_vec: list[str] | None = None,
                  fdr_scope: str = 'global',
                  covariates: pd.DataFrame | None = None,
                  use_treat: bool = True,
                  feature_meta: pd.DataFrame | None = None,
                  hierarchical: bool = False,
                  too_few_hint: str = '') -> pd.DataFrame:
    """
    Moderated linear model test of every feature, for N groups and
    multiple contrasts.

    data_df   : rows = samples (aligned to group_vec/pairing_vec/covariates),
                columns = features
    contrasts : (baseline, other) pairs from build_contrasts()
    mode      : 'reference' — one fit across all samples, one coefficient
                  read per contrast (the variance prior is shared).
                'pairwise'  — each contrast fitted on its two groups only.
    covariates: optional adjustment covariates (covariate_frame()).
    feature_meta: optional columns merged onto the results by 'feature'
                (e.g. cluster, cluster_id, channel for marker intensities).
    use_treat, hierarchical, fdr_scope: see apply_significance().
    too_few_hint: appended to the error raised when fewer than three
                features can be tested.

    Returns one row per (feature, comparison): feature, logFC, CI.L, CI.R,
    AveExpr, t, P.Value, adj.P.Val, SE, df.total, comparison,
    adj.P.Val.global, threshold, significant (plus the cluster-level
    columns when hierarchical).
    """
    if mode == 'reference':
        baseline = contrasts[0][0]
        if any(b != baseline for b, _o in contrasts):
            raise ValueError("reference mode expects a shared baseline")
        combined = _moderated_fit_contrasts(data_df, group_vec, contrasts, pairing_vec,
                                            covariates, too_few_hint)
    elif mode == 'pairwise':
        frames = [
            _moderated_fit_contrasts(sub_df, sub_groups, [(base, other)], sub_pair, sub_cov,
                                     too_few_hint)
            for sub_df, sub_groups, sub_pair, sub_cov, base, other
            in _pair_subsets(data_df, group_vec, contrasts, pairing_vec, covariates)
        ]
        combined = pd.concat(frames, ignore_index=True)
    else:
        raise ValueError(f"mode must be 'reference' or 'pairwise', got {mode!r}")

    if feature_meta is not None:
        combined = combined.merge(feature_meta, on='feature', how='left', sort=False)

    combined = apply_significance(combined, pval_threshold, fc_threshold,
                                  fdr_scope=fdr_scope, use_treat=use_treat,
                                  hierarchical=hierarchical)
    log.info("moderated test: %d rows across %d comparison(s), %d significant (%s FDR%s%s)",
             len(combined), combined['comparison'].nunique(),
             int(combined['significant'].sum()), fdr_scope,
             ", TREAT" if use_treat else "", ", cluster-first" if hierarchical else "")
    return combined


# ---------------------------------------------------------------------------
# Negative-binomial GLM on counts
# ---------------------------------------------------------------------------

def _glm_fit_one_feature(y: np.ndarray, design: np.ndarray, offset: np.ndarray):
    """Poisson-then-NB fit for one feature's raw counts, with the NB
    dispersion from the Cameron & Trivedi auxiliary-OLS estimate.
    ``offset`` is the log of the per-sample denominator (a library-size
    term), so the fit compares rates, not raw totals, across samples.
    Returns the fitted GLM result, or None if even the Poisson fit fails."""
    import statsmodels.api as sm

    try:
        poisson_fit = sm.GLM(y, design, family=sm.families.Poisson(), offset=offset).fit()
    except Exception as e:
        log.warning("Poisson GLM failed (%s) — marking untestable", e)
        return None
    try:
        mu = poisson_fit.mu
        aux_y = ((y - mu) ** 2 - y) / mu
        alpha = float(sm.OLS(aux_y, mu).fit().params[0])
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("non-positive alpha estimate")
        return sm.GLM(y, design, family=sm.families.NegativeBinomial(alpha=alpha), offset=offset).fit()
    except Exception as e:
        log.warning("NB GLM failed (%s), falling back to Poisson", e)
        return poisson_fit


def _glm_counts_contrasts(counts_df: pd.DataFrame, group_vec: list[str],
                          contrasts: list[tuple[str, str]],
                          pairing_vec: list[str] | None,
                          covariates: pd.DataFrame | None,
                          offset_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    One design + per-feature NB/Poisson fit, read out for every contrast in
    ``contrasts`` (shared baseline). The offset is log(row total of
    counts_df) unless ``offset_df`` gives a per-feature denominator. The
    coefficient is a rate difference between groups, reported on the log2
    scale with Wald standard errors (df.total = inf).
    """
    base = contrasts[0][0]
    design_mat, _names, group_cols = build_design(group_vec, base, pairing_vec, covariates)

    features = list(counts_df.columns)
    if offset_df is None:
        totals = counts_df.sum(axis=1).values.astype(float)
        row_offset = np.log(np.maximum(totals, 1.0))
        offsets = {f: row_offset for f in features}
    else:
        denom = offset_df.reindex(index=counts_df.index, columns=counts_df.columns)
        offsets = {f: np.log(np.maximum(denom[f].values.astype(float), 1.0)) for f in features}

    LN2 = np.log(2.0)
    Z95 = 1.959963984540054
    fits = [_glm_fit_one_feature(counts_df[f].values.astype(float), design_mat, offsets[f])
            for f in features]

    frames = []
    for b, other in contrasts:
        j = group_cols[other]
        logfc = np.full(len(features), np.nan)
        se = np.full(len(features), np.nan)
        tvals = np.full(len(features), np.nan)
        for i, fit in enumerate(fits):
            if fit is None:
                continue
            logfc[i] = fit.params[j] / LN2
            se[i] = fit.bse[j] / LN2
            tvals[i] = fit.tvalues[j]
        p = ms.treat_pvalues(logfc, se, np.inf, 0.0)
        frames.append(pd.DataFrame({
            'feature':   features,
            'logFC':     logfc,
            'CI.L':      logfc - Z95 * se,
            'CI.R':      logfc + Z95 * se,
            't':         tvals,
            'P.Value':   p,
            'adj.P.Val': ms.bh_adjust(p),
            'SE':        se,
            'df.total':  np.inf,
            'comparison': f"{other} vs {b}",
        }))
    return pd.concat(frames, ignore_index=True)


def run_glm_counts(counts_df: pd.DataFrame, group_vec: list[str],
                   contrasts: list[tuple[str, str]], mode: str,
                   pval_threshold: float, fc_threshold: float,
                   pairing_vec: list[str] | None = None,
                   fdr_scope: str = 'global',
                   covariates: pd.DataFrame | None = None,
                   use_treat: bool = True,
                   offset_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Per-feature negative-binomial GLM differential abundance test on raw
    event counts, with the same contrasts/mode/pairing/covariate semantics
    as run_moderated(). In 'reference' mode each feature is fitted once and
    every contrast is read from that fit; 'pairwise' fits each pair
    separately.

    offset_df: optional per-sample, per-feature denominators (same index and
        columns as counts_df), e.g. each gated population's parent count.
        Default: the sample's total over all features.

    Returns the same schema as run_moderated() minus AveExpr.
    """
    if mode == 'reference':
        combined = _glm_counts_contrasts(counts_df, group_vec, contrasts, pairing_vec,
                                         covariates, offset_df)
    elif mode == 'pairwise':
        frames = [
            _glm_counts_contrasts(sub_df, sub_groups, [(base, other)], sub_pair, sub_cov,
                                  None if offset_df is None else offset_df.loc[sub_df.index])
            for sub_df, sub_groups, sub_pair, sub_cov, base, other
            in _pair_subsets(counts_df, group_vec, contrasts, pairing_vec, covariates)
        ]
        combined = pd.concat(frames, ignore_index=True)
    else:
        raise ValueError(f"mode must be 'reference' or 'pairwise', got {mode!r}")

    combined = apply_significance(combined, pval_threshold, fc_threshold,
                                  fdr_scope=fdr_scope, use_treat=use_treat)
    log.info("GLM counts: %d rows across %d comparison(s), %d significant (%s FDR), %d untestable",
             len(combined), combined['comparison'].nunique(),
             int(combined['significant'].sum()), fdr_scope, int(combined['logFC'].isna().sum()))
    return combined


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------

def apply_significance(results: pd.DataFrame, pval_threshold: float,
                       fc_threshold: float, fdr_scope: str = 'global',
                       use_treat: bool = True,
                       hierarchical: bool = False) -> pd.DataFrame:
    """
    Recompute p-values, FDR and the 'significant' flag of a results table
    for the given thresholds, without refitting. Returns a new DataFrame.

    use_treat   : True — P.Value is the TREAT p-value for |logFC| >
                  fc_threshold (McCarthy & Smyth 2009). False — P.Value
                  tests logFC ≠ 0 and fc_threshold only filters estimates.
    fdr_scope   : 'global' — BH pooled over every comparison in the table
                  decides significance; 'per_comparison' — each
                  comparison's own BH. Both adj.P.Val (per comparison) and
                  adj.P.Val.global are always written.
    hierarchical: family-first testing for tables with a 'cluster_id'
                  column (marker intensities). Adds cluster.P.Value (Simes
                  over the family's markers), cluster.adj.P.Val (BH over
                  families) and stagewise.adj.P.Val; a marker is
                  significant when its stage-wise value ≤ pval_threshold.

    Tables without SE/df.total columns (results saved before these columns
    existed) keep their stored p-values and use fc_threshold as a filter.
    """
    df = results.copy()
    tau = float(fc_threshold) if use_treat else 0.0
    can_retest = {'SE', 'df.total', 'logFC'}.issubset(df.columns)
    if can_retest:
        df['P.Value'] = ms.treat_pvalues(df['logFC'].values, df['SE'].values,
                                         df['df.total'].values.astype(float), tau)
        df['threshold'] = tau
        adj = np.full(len(df), np.nan)
        for _comp, idx in df.groupby('comparison', sort=False).indices.items():
            adj[idx] = ms.bh_adjust(df['P.Value'].values[idx])
        df['adj.P.Val'] = adj
    df['adj.P.Val.global'] = ms.bh_adjust(df['P.Value'].values.astype(float))

    q_col = 'adj.P.Val.global' if fdr_scope == 'global' else 'adj.P.Val'
    passes_fc = df['logFC'].abs() >= fc_threshold

    if hierarchical and 'cluster_id' in df.columns:
        fam_p = np.full(len(df), np.nan)
        fam_adj = np.full(len(df), np.nan)
        stage = np.full(len(df), np.nan)
        groups = ({'all': np.arange(len(df))} if fdr_scope == 'global'
                  else df.groupby('comparison', sort=False).indices)
        for _key, idx in groups.items():
            family = [f"{c}\x1f{k}" for c, k in
                      zip(df['comparison'].values[idx], df['cluster_id'].values[idx])]
            res = ms.stagewise_adjust(df['P.Value'].values[idx].astype(float), family,
                                      pval_threshold)
            fam_p[idx] = res['family_p']
            fam_adj[idx] = res['family_adj']
            stage[idx] = res['adjusted']
        df['cluster.P.Value'] = fam_p
        df['cluster.adj.P.Val'] = fam_adj
        df['stagewise.adj.P.Val'] = stage
        df['significant'] = (df['stagewise.adj.P.Val'] <= pval_threshold) & passes_fc
    else:
        for col in ('cluster.P.Value', 'cluster.adj.P.Val', 'stagewise.adj.P.Val'):
            if col in df.columns:
                df = df.drop(columns=[col])
        df['significant'] = (df[q_col] <= pval_threshold) & passes_fc
    return df

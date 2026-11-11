"""
moderated_stats.py — moderated linear models for differential testing
=====================================================================

Per-feature linear models with empirical Bayes variance moderation, for
testing many features (clusters, cluster × marker MFIs, gated populations)
across a small number of samples.

Methods and the papers they are written from:

* ``lm_fit`` — ordinary least squares per feature, allowing missing values.
* ``weighted_lm_fit`` — the same with a weight per observation (weighted
  least squares; limma's ``lmFit`` with weights).
* ``e_bayes`` — the hierarchical model of Smyth (2004): residual variances
  are shrunk towards a common prior whose degrees of freedom ``d0`` and
  scale ``s0²`` are estimated by the method of moments on log variances.
  Gives moderated t-statistics and p-values (the p-values are worked out
  when first read).
* ``f_test`` — moderated F test of several coefficients at once (limma's
  ``topTableF``), e.g. the columns of a spline in age, with each
  feature's partial R² (the share of variation those columns explain).
* ``block_correlation`` and ``gls_lm_fit`` — several samples from one
  individual (e.g. four tissues of each mouse) as a random effect, after
  limma's ``duplicateCorrelation`` (Smyth, Michaud & Scott 2005): the
  correlation between samples of one individual is estimated per feature
  by REML, the features' values are pooled into one consensus correlation,
  and every feature is then fitted by generalised least squares with that
  correlation.
* ``variance_components`` — REML estimates of several variance components
  per feature (all categorical variables as random effects), the model
  variancePartition (Hoffman & Schadt 2016) fits with lme4, giving the
  share of each feature's variation explained by each variable.
* ``treat_pvalues`` — TREAT (McCarthy & Smyth 2009): tests whether a
  coefficient's magnitude exceeds a threshold, rather than whether it
  differs from zero.
* ``bh_adjust`` — Benjamini & Hochberg (1995) false discovery rate.
* ``simes`` and ``stagewise_adjust`` — two-stage testing of hypotheses
  grouped into families (e.g. markers within clusters): families are
  screened with Simes' (1986) combined p-value and BH, then hypotheses in
  selected families are tested by BH at level q·R/C, following Benjamini &
  Bogomolov (2014).

References
----------
Smyth GK (2004). Linear models and empirical Bayes methods for assessing
    differential expression in microarray experiments. Stat Appl Genet Mol
    Biol 3(1), Article 3. doi:10.2202/1544-6115.1027
McCarthy DJ, Smyth GK (2009). Testing significance relative to a
    fold-change threshold is a TREAT. Bioinformatics 25(6):765-771.
    doi:10.1093/bioinformatics/btp053
Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate.
    J R Stat Soc B 57(1):289-300.
Simes RJ (1986). An improved Bonferroni procedure for multiple tests of
    significance. Biometrika 73(3):751-754.
Smyth GK, Michaud J, Scott HS (2005). Use of within-array replicate spots
    for assessing differential expression in microarray experiments.
    Bioinformatics 21(9):2067-2075. doi:10.1093/bioinformatics/bti270
Hoffman GE, Schadt EE (2016). variancePartition: interpreting drivers of
    variation in complex gene expression studies. BMC Bioinformatics
    17:483. doi:10.1186/s12859-016-1323-z
Stram DO, Lee JW (1994). Variance components testing in the longitudinal
    mixed effects model. Biometrics 50(4):1171-1177.
Benjamini Y, Bogomolov M (2014). Selective inference on multiple families
    of hypotheses. J R Stat Soc B 76(1):297-318.

Only numpy, scipy and pandas are used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import special, stats

__all__ = [
    'LinearFit', 'ModeratedFit', 'FTest', 'lm_fit', 'weighted_lm_fit', 'e_bayes', 'top_table',
    'f_test', 'f_pvalues', 'BlockCorrelation', 'block_correlation', 'gls_lm_fit',
    'VarianceComponents', 'variance_components', 'trimmed_mean',
    'treat_pvalues', 'bh_adjust', 'simes', 'stagewise_adjust',
    'fit_prior_variance', 'trigamma_inverse', 'estimable_columns',
]

_MIN_FEATURES = 3

# Smallest ratio of the diagonal of R (X = QR) to its largest entry for a
# design to be taken as full rank without computing its singular values.
FULL_RANK_QR_RATIO = 1e-6


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class LinearFit:
    """Per-feature least-squares fit. G features, p coefficients."""
    coefficients: np.ndarray      # (G, p); NaN where not estimable
    stdev_unscaled: np.ndarray    # (G, p); sqrt(diag((XᵀX)⁻¹)) per feature
    sigma2: np.ndarray            # (G,) residual variance s²; NaN if df = 0
    df_residual: np.ndarray       # (G,) n_obs − rank; 0 if not estimable
    amean: np.ndarray             # (G,) mean of observed values
    n_obs: np.ndarray             # (G,) observed samples per feature
    design: np.ndarray            # (n, p)


@dataclass
class ModeratedFit(LinearFit):
    """LinearFit plus the empirical Bayes quantities of Smyth (2004)."""
    df_prior: float = np.nan      # d0 (may be inf)
    s2_prior: float = np.nan      # s0²
    s2_post: np.ndarray = None    # (G,) posterior variance s̃²
    df_total: np.ndarray = None   # (G,) d0 + d, capped at the pooled df
    t: np.ndarray = None          # (G, p) moderated t
    p_value: np.ndarray = None    # (G, p) two-sided; see _moderated_p_value


def _moderated_p_value(self) -> np.ndarray | None:
    """
    Two-sided p-values of the moderated t, computed the first time they are
    read when the fit was made without them (e_bayes). Permutation tests
    refit the model many times and read only one coefficient's t, so the
    t-distribution tail of every coefficient (e.g. dozens of pairing
    columns) is not worked out unless something asks for it.
    """
    p = self.__dict__.get('_p_value')
    if p is None and self.t is not None and self.df_total is not None:
        p = 2.0 * _t_sf(np.abs(self.t), np.asarray(self.df_total, dtype=float)[:, None])
        self.__dict__['_p_value'] = p
    return p


def _set_moderated_p_value(self, value) -> None:
    self.__dict__['_p_value'] = value


ModeratedFit.p_value = property(_moderated_p_value, _set_moderated_p_value)


# ---------------------------------------------------------------------------
# Linear model
# ---------------------------------------------------------------------------

def estimable_columns(X) -> tuple[np.ndarray, np.ndarray]:
    """
    A full-rank basis of the columns of X that are not all zero, and which
    of its coefficients are estimable.

    Columns are taken left to right and kept while each adds to the rank,
    so a column that is a combination of earlier ones is left out without
    changing the space the fit spans. With treatment coding this drops one
    blocking indicator when the reference blocking level has no observed
    sample (e.g. a feature measured only in mice whose names sort after
    the first mouse). A kept column's coefficient is estimable when
    removing that column lowers the rank; otherwise its value depends on
    which columns were left out and it must not be read.

    Returns (kept column indices, estimable flag per kept column). For a
    full-rank X these are its non-zero columns, all estimable.
    """
    X = np.asarray(X, dtype=float)
    nonzero = np.flatnonzero(np.any(X != 0.0, axis=0))
    if nonzero.size == 0:
        return nonzero, np.zeros(0, dtype=bool)
    # Most designs are full rank: a QR decomposition (cheaper than the
    # singular values) settles that when no column comes close to the span
    # of the ones before it; anything else takes the exact route below.
    if X.shape[0] >= nonzero.size:
        diag = np.abs(np.diag(np.linalg.qr(X[:, nonzero], mode='r')))
        if diag.min() > FULL_RANK_QR_RATIO * diag.max():
            return nonzero, np.ones(nonzero.size, dtype=bool)
    rank = np.linalg.matrix_rank(X[:, nonzero])
    if rank == nonzero.size:
        return nonzero, np.ones(nonzero.size, dtype=bool)
    kept: list[int] = []
    for j in nonzero:
        if np.linalg.matrix_rank(X[:, kept + [int(j)]]) > len(kept):
            kept.append(int(j))
    estimable = np.array([np.linalg.matrix_rank(X[:, nonzero[nonzero != j]]) < rank for j in kept])
    return np.asarray(kept, dtype=int), estimable


def lm_fit(expr, design) -> LinearFit:
    """
    Fit ``expr[g] ~ design`` by least squares for every feature g.

    expr   : (features × samples) array; NaN marks a missing value.
    design : (samples × coefficients) array.

    Each feature uses only its observed samples. Design columns that are
    all zero over those samples (e.g. a blocking level whose samples are
    all missing) are dropped for that feature and their coefficients are
    NaN. If the remaining columns are rank-deficient, the fit uses the
    full-rank basis of ``estimable_columns``; coefficients that are not
    estimable (e.g. a group compared with a baseline that has no observed
    sample) are NaN, and the residual df is the observed samples less the
    rank. A feature with no estimable coefficient is not estimable: all its
    coefficients are NaN and its residual df is 0.
    """
    Y = np.asarray(expr, dtype=float)
    X = np.asarray(design, dtype=float)
    if Y.ndim == 1:
        Y = Y[None, :]
    if X.ndim != 2 or X.shape[0] != Y.shape[1]:
        raise ValueError(
            f"design has {X.shape[0] if X.ndim == 2 else '?'} rows but expr has "
            f"{Y.shape[1]} samples")
    G, n = Y.shape
    p = X.shape[1]

    coef = np.full((G, p), np.nan)
    stdu = np.full((G, p), np.nan)
    sigma2 = np.full(G, np.nan)
    df_res = np.zeros(G)
    obs_mask = np.isfinite(Y)
    n_obs = obs_mask.sum(axis=1)
    with np.errstate(invalid='ignore'):
        amean = np.where(n_obs > 0, np.nansum(np.where(obs_mask, Y, 0.0), axis=1)
                         / np.maximum(n_obs, 1), np.nan)

    # Features sharing a missing-value pattern share one decomposition.
    patterns: dict[bytes, list[int]] = {}
    for g in range(G):
        patterns.setdefault(obs_mask[g].tobytes(), []).append(g)

    for key, rows in patterns.items():
        obs = np.frombuffer(key, dtype=bool)
        n_o = int(obs.sum())
        if n_o == 0:
            continue
        Xo = X[obs]
        cols, estimable = estimable_columns(Xo)
        if not estimable.any():
            continue
        Xc = Xo[:, cols]
        rank = cols.size
        Q, R = np.linalg.qr(Xc)
        Rinv = np.linalg.inv(R)
        unscaled = np.sqrt(np.sum(Rinv ** 2, axis=1))       # diag((XᵀX)⁻¹)
        Yo = Y[np.ix_(rows, np.flatnonzero(obs))].T          # (n_o, k)
        B = Rinv @ (Q.T @ Yo)                                # (rank, k)
        resid = Yo - Xc @ B
        d = n_o - rank
        idx = np.asarray(rows)
        coef[np.ix_(idx, cols)] = B.T
        stdu[np.ix_(idx, cols)] = unscaled
        hidden = cols[~estimable]
        coef[np.ix_(idx, hidden)] = np.nan
        stdu[np.ix_(idx, hidden)] = np.nan
        df_res[idx] = d
        if d > 0:
            sigma2[idx] = np.sum(resid ** 2, axis=0) / d

    return LinearFit(coefficients=coef, stdev_unscaled=stdu, sigma2=sigma2,
                     df_residual=df_res, amean=amean, n_obs=n_obs, design=X)


def weighted_lm_fit(expr, design, weights) -> LinearFit:
    """
    Weighted least squares of ``expr[g] ~ design`` with per-observation
    weights, for every feature g: ``moderated_stats.lm_fit`` with weights.

    expr, weights : (features x samples); NaN in expr, or a weight that is
                    not finite and positive, marks a missing observation.
    design        : (samples x coefficients).

    Missing observations, dropped design columns and coefficients that are
    not estimable are handled as ``lm_fit`` handles them. Features sharing a
    pattern of observed samples share the column choice and are solved
    together from their normal equations. The residual variance is
    sum(w r^2) / (observed - rank); the unscaled standard errors are the
    square roots of the diagonal of (X^T W X)^-1.
    """
    Y = np.asarray(expr, dtype=float)
    X = np.asarray(design, dtype=float)
    W = np.asarray(weights, dtype=float)
    if Y.ndim == 1:
        Y = Y[None, :]
        W = W.reshape(1, -1)
    if X.ndim != 2 or X.shape[0] != Y.shape[1]:
        raise ValueError(f'design has {X.shape[0] if X.ndim == 2 else "?"} rows but expr has '
                         f'{Y.shape[1]} samples')
    if W.shape != Y.shape:
        raise ValueError('weights must have the shape of expr')
    G, _n = Y.shape
    p = X.shape[1]

    coef = np.full((G, p), np.nan)
    stdu = np.full((G, p), np.nan)
    sigma2 = np.full(G, np.nan)
    df_res = np.zeros(G)
    obs_mask = np.isfinite(Y) & np.isfinite(W) & (W > 0)
    n_obs = obs_mask.sum(axis=1)
    with np.errstate(invalid='ignore'):
        amean = np.where(n_obs > 0, np.where(obs_mask, Y, 0.0).sum(axis=1) / np.maximum(n_obs, 1),
                         np.nan)

    patterns: dict[bytes, list[int]] = {}
    for g in range(G):
        patterns.setdefault(obs_mask[g].tobytes(), []).append(g)

    for key, rows in patterns.items():
        obs = np.frombuffer(key, dtype=bool)
        n_o = int(obs.sum())
        if n_o == 0:
            continue
        cols, estimable = estimable_columns(X[obs])
        if not estimable.any():
            continue
        Xc = X[obs][:, cols]
        idx = np.asarray(rows)
        sel = np.flatnonzero(obs)
        Yo = Y[np.ix_(idx, sel)]
        Wo = W[np.ix_(idx, sel)]
        A = np.einsum('kn,nr,ns->krs', Wo, Xc, Xc)
        rhs = np.einsum('kn,nr,kn->kr', Wo, Xc, Yo)
        Ainv = np.linalg.inv(A)
        B = np.einsum('krs,ks->kr', Ainv, rhs)
        resid = Yo - B @ Xc.T
        d = n_o - cols.size
        coef[np.ix_(idx, cols)] = B
        stdu[np.ix_(idx, cols)] = np.sqrt(np.diagonal(Ainv, axis1=1, axis2=2))
        hidden = cols[~estimable]
        coef[np.ix_(idx, hidden)] = np.nan
        stdu[np.ix_(idx, hidden)] = np.nan
        df_res[idx] = d
        if d > 0:
            sigma2[idx] = np.sum(Wo * resid ** 2, axis=1) / d

    return LinearFit(coefficients=coef, stdev_unscaled=stdu, sigma2=sigma2,
                     df_residual=df_res, amean=amean, n_obs=n_obs, design=X)

# ---------------------------------------------------------------------------
# Empirical Bayes
# ---------------------------------------------------------------------------

def trigamma_inverse(x, tol: float = 1e-8, max_iter: int = 50):
    """
    Solve ψ'(y) = x for y > 0 by Newton's method on 1/ψ'(y), started at
    y = 0.5 + 1/x (Smyth 2004, Appendix). Works element-wise.
    """
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x) & (x > 0)
    big = ok & (x > 1e7)
    small = ok & (x < 1e-6)
    out[big] = 1.0 / np.sqrt(x[big])
    out[small] = 1.0 / x[small]
    mid = ok & ~big & ~small
    if mid.any():
        xm = x[mid]
        y = 0.5 + 1.0 / xm
        for _ in range(max_iter):
            tri = special.polygamma(1, y)
            delta = tri * (1.0 - tri / xm) / special.polygamma(2, y)
            y = y + delta
            if np.all(np.abs(delta / y) < tol):
                break
        out[mid] = y
    return out if out.ndim else float(out)


def fit_prior_variance(s2, df) -> tuple[float, float]:
    """
    Method-of-moments estimate of the prior (d0, s0²) from residual
    variances ``s2`` with residual degrees of freedom ``df``.

    Under the model s² | σ² ~ σ²χ²_d / d and 1/σ² ~ χ²_{d0} / (d0 s0²),
    e = log s² − ψ(d/2) + log(d/2) has mean log s0² − ψ(d0/2) + log(d0/2)
    and variance ψ'(d/2) + ψ'(d0/2). Matching the sample moments gives d0
    and s0². When the sample variance of e is no larger than its expected
    value under d0 = ∞, every feature shares one variance; its estimate is
    the mean of the s², and (inf, mean s²) is returned.

    Only finite, strictly positive variances with df > 0 are used.
    """
    s2 = np.asarray(s2, dtype=float)
    df = np.asarray(df, dtype=float)
    use = np.isfinite(s2) & (s2 > 0) & np.isfinite(df) & (df > 0)
    n = int(use.sum())
    if n == 0:
        return np.nan, np.nan
    e = np.log(s2[use]) - special.digamma(df[use] / 2.0) + np.log(df[use] / 2.0)
    e_mean = float(np.mean(e))
    if n < 2:
        return np.inf, float(np.mean(s2[use]))
    e_var = float(np.sum((e - e_mean) ** 2) / (n - 1))
    excess = e_var - float(np.mean(special.polygamma(1, df[use] / 2.0)))
    if excess > 0:
        d0 = 2.0 * float(trigamma_inverse(excess))
        s0_2 = float(np.exp(e_mean + special.digamma(d0 / 2.0) - np.log(d0 / 2.0)))
        return d0, s0_2
    return np.inf, float(np.mean(s2[use]))


def _t_sf(x, df):
    """Upper-tail probability of t_df, normal where df is infinite."""
    x = np.asarray(x, dtype=float)
    df = np.broadcast_to(np.asarray(df, dtype=float), x.shape)
    out = np.full(x.shape, np.nan)
    inf = np.isinf(df)
    fin = np.isfinite(df) & (df > 0)
    out[inf] = stats.norm.sf(x[inf])
    out[fin] = stats.t.sf(x[fin], df[fin])
    return out


def _t_isf(q, df):
    """Inverse upper-tail of t_df (normal where df is infinite)."""
    q = np.asarray(q, dtype=float)
    df = np.broadcast_to(np.asarray(df, dtype=float), q.shape)
    out = np.full(q.shape, np.nan)
    inf = np.isinf(df)
    fin = np.isfinite(df) & (df > 0)
    out[inf] = stats.norm.isf(q[inf])
    out[fin] = stats.t.isf(q[fin], df[fin])
    return out


def e_bayes(fit: LinearFit) -> ModeratedFit:
    """
    Empirical Bayes moderation of a LinearFit (Smyth 2004).

    Posterior variance s̃² = (d0·s0² + d·s²)/(d0 + d); moderated
    t = β/(u·s̃) with u the unscaled standard error; total df d0 + d,
    capped at the residual df pooled over all features. Features with no
    residual df (d = 0) take the prior variance and df d0.
    """
    s2 = fit.sigma2
    d = fit.df_residual
    estimable = np.any(np.isfinite(fit.coefficients), axis=1)
    if int(estimable.sum()) < _MIN_FEATURES:
        raise ValueError(
            f"Only {int(estimable.sum())} estimable feature(s); empirical Bayes "
            f"moderation needs at least {_MIN_FEATURES}.")

    d0, s0_2 = fit_prior_variance(s2, d)
    if not np.isfinite(s0_2):
        raise ValueError("Could not estimate a prior variance: no feature has "
                         "a positive residual variance.")

    s2_obs = np.where(np.isfinite(s2), s2, 0.0)
    if np.isinf(d0):
        s2_post = np.full(s2.shape, s0_2)
    else:
        s2_post = (d0 * s0_2 + d * s2_obs) / (d0 + d)
    s2_post = np.where(estimable, s2_post, np.nan)

    df_pooled = float(np.sum(d[estimable]))
    df_total = np.minimum(d0 + d, df_pooled) if df_pooled > 0 else np.full(d.shape, d0)
    df_total = np.where(estimable, df_total, np.nan)

    se = fit.stdev_unscaled * np.sqrt(s2_post)[:, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = fit.coefficients / se

    return ModeratedFit(
        coefficients=fit.coefficients, stdev_unscaled=fit.stdev_unscaled,
        sigma2=fit.sigma2, df_residual=fit.df_residual, amean=fit.amean,
        n_obs=fit.n_obs, design=fit.design,
        df_prior=float(d0), s2_prior=float(s0_2), s2_post=s2_post,
        df_total=df_total, t=t, p_value=None,
    )


# ---------------------------------------------------------------------------
# TREAT and multiple testing
# ---------------------------------------------------------------------------

def treat_pvalues(estimate, se, df, threshold: float = 0.0) -> np.ndarray:
    """
    TREAT p-value for H0: |β| ≤ τ against |β| > τ (McCarthy & Smyth 2009):

        p = S((|b| − τ)/se) + S((|b| + τ)/se)

    with S the upper tail of t_df (normal where df is infinite). With
    τ = 0 this is the ordinary two-sided p-value.
    """
    b = np.abs(np.asarray(estimate, dtype=float))
    se = np.asarray(se, dtype=float)
    tau = abs(float(threshold))
    with np.errstate(divide='ignore', invalid='ignore'):
        p = _t_sf((b - tau) / se, df) + _t_sf((b + tau) / se, df)
    return np.minimum(p, 1.0)


def bh_adjust(p) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values. NaNs are kept and not counted."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv, kind='mergesort')
    ranked = pv[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def simes(p) -> float:
    """Simes' combined p-value: min_i m·p_(i)/i over the finite p-values."""
    p = np.asarray(p, dtype=float)
    p = np.sort(p[np.isfinite(p)])
    m = p.size
    if m == 0:
        return np.nan
    return float(min(1.0, np.min(p * m / np.arange(1, m + 1))))


def stagewise_adjust(p, family, alpha: float) -> dict:
    """
    Two-stage FDR control over hypotheses grouped into families.

    Stage 1 screens each family with its Simes p-value and applies BH
    across families; R of the C testable families are selected at level
    ``alpha``. Stage 2 applies BH within each selected family; a
    hypothesis is rejected when its within-family BH value is ≤ alpha·R/C
    (Benjamini & Bogomolov 2014). The returned ``adjusted`` value is the
    within-family BH value × C/R, capped at 1, so it is compared with
    ``alpha`` directly; hypotheses in unselected families get 1.

    Returns dict of arrays aligned to ``p``: 'family_p' (Simes),
    'family_adj' (BH over families), 'adjusted', and scalars 'n_selected',
    'n_families'.
    """
    p = np.asarray(p, dtype=float)
    # Families in order of first appearance; members kept in input order
    codes, labels = pd.factorize(np.asarray(family, dtype=object), sort=False,
                                 use_na_sentinel=False)
    order = np.argsort(codes, kind='stable')
    members = np.split(order, np.cumsum(np.bincount(codes, minlength=len(labels)))[:-1])
    fam_p = np.array([simes(p[m]) for m in members], dtype=float)
    fam_adj = bh_adjust(fam_p)
    n_fam = int(np.isfinite(fam_p).sum())
    selected = np.isfinite(fam_adj) & (fam_adj <= alpha)
    n_sel = int(selected.sum())

    family_p = fam_p[codes]
    family_adj = fam_adj[codes]
    adjusted = np.where(np.isfinite(p), 1.0, np.nan)
    if n_sel > 0:
        for k in np.flatnonzero(selected):
            m = members[k]
            adjusted[m] = np.minimum(bh_adjust(p[m]) * n_fam / n_sel, 1.0)
    return {'family_p': family_p, 'family_adj': family_adj, 'adjusted': adjusted,
            'n_selected': n_sel, 'n_families': n_fam}


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def top_table(fit: ModeratedFit, coef: int, threshold: float = 0.0,
              confint: float = 0.95) -> pd.DataFrame:
    """
    Results for one coefficient, in input feature order.

    Columns: logFC (the coefficient), CI.L, CI.R, AveExpr, t, P.Value,
    adj.P.Val (BH over this table), SE (moderated standard error) and
    df.total. With ``threshold`` > 0, P.Value is the TREAT p-value for
    |logFC| > threshold; t is always the ordinary moderated value.
    """
    b = fit.coefficients[:, coef]
    se = fit.stdev_unscaled[:, coef] * np.sqrt(fit.s2_post)
    df = fit.df_total
    q = _t_isf(np.full(b.shape, (1.0 - confint) / 2.0), df)
    p = treat_pvalues(b, se, df, threshold) if threshold else fit.p_value[:, coef]
    return pd.DataFrame({
        'logFC': b,
        'CI.L': b - q * se,
        'CI.R': b + q * se,
        'AveExpr': fit.amean,
        't': fit.t[:, coef],
        'P.Value': p,
        'adj.P.Val': bh_adjust(p),
        'SE': se,
        'df.total': df,
    })


# ---------------------------------------------------------------------------
# Several samples per individual: correlation within individuals
# ---------------------------------------------------------------------------

# Margin kept from the edges of the correlation's range, where the
# covariance matrix stops being positive definite.
_RHO_MARGIN = 1e-4
_RHO_GRID = 121
_GOLDEN_STEPS = 30


def _factor_codes(values) -> tuple[np.ndarray, int]:
    """Integer codes (order of first appearance) and the number of levels."""
    codes, uniques = pd.factorize(np.asarray(values, dtype=object), sort=False,
                                  use_na_sentinel=False)
    return codes.astype(int), len(uniques)


def _rho_range(sizes: np.ndarray) -> tuple[float, float]:
    """Open interval of correlations giving a positive definite
    V = (1 − ρ) I + ρ ZZᵀ, less a small margin."""
    m_max = int(sizes.max())
    lo = -1.0 / (m_max - 1) if m_max > 1 else -1.0
    return lo + _RHO_MARGIN, 1.0 - _RHO_MARGIN


def _whiten(values: np.ndarray, codes: np.ndarray, sizes: np.ndarray, rho) -> np.ndarray:
    """
    V^(-1/2) times ``values`` for V = (1 − ρ) I + ρ ZZᵀ, Z the individual
    indicators. Within an individual with m samples, V^(-1/2) =
    (1 − ρ)^(-1/2) (I − c J/m) with c = 1 − sqrt((1 − ρ)/(1 + (m − 1)ρ)),
    so each value has c times its individual's mean taken off and is then
    scaled.

    values : (L, n, k), samples along the middle axis.
    rho    : scalar, or one value per L.
    """
    L = values.shape[0]
    rho_l = np.broadcast_to(np.asarray(rho, dtype=float), (L,))
    onehot = np.zeros((codes.size, sizes.size))
    onehot[np.arange(codes.size), codes] = 1.0
    means = np.einsum('nb,lnk->lbk', onehot, values) / sizes[None, :, None]
    r = rho_l[:, None]
    c = 1.0 - np.sqrt((1.0 - r) / (1.0 + (sizes[None, :] - 1.0) * r))       # (L, B)
    a = 1.0 / np.sqrt(1.0 - rho_l)
    shared = (c[:, :, None] * means)[:, codes, :]
    return a[:, None, None] * (values - shared)


def _log_det_v(sizes: np.ndarray, rho) -> np.ndarray:
    """log |V| for V = (1 − ρ) I + ρ ZZᵀ, one value per ρ."""
    r = np.atleast_1d(np.asarray(rho, dtype=float))[:, None]
    return np.sum((sizes[None, :] - 1.0) * np.log(1.0 - r)
                  + np.log(1.0 + (sizes[None, :] - 1.0) * r), axis=1)


def _block_reml(Y: np.ndarray, X: np.ndarray, codes: np.ndarray, sizes: np.ndarray,
                rho: np.ndarray) -> np.ndarray:
    """
    Restricted log-likelihood (constants dropped, residual variance
    profiled out) of each row of Y (features × samples, all observed) under
    y = Xβ + e, Var(e) = σ² V(ρ), at that row's ρ:

        −½ [(n − p) log RSS(ρ) + log |V(ρ)| + log |Xᵀ V⁻¹ X|]

    with RSS(ρ) the generalised residual sum of squares. X has full column
    rank.
    """
    G, n = Y.shape
    p = X.shape[1]
    Yw = _whiten(Y[:, :, None], codes, sizes, rho)[:, :, 0]
    Xw = _whiten(np.broadcast_to(X, (G, n, p)), codes, sizes, rho)
    A = np.einsum('gnp,gnq->gpq', Xw, Xw)
    b = np.einsum('gnp,gn->gp', Xw, Yw)
    beta = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    rss = np.sum(Yw ** 2, axis=1) - np.sum(b * beta, axis=1)
    _sign, logdet_a = np.linalg.slogdet(A)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = -0.5 * ((n - p) * np.log(np.maximum(rss, 0.0)) + _log_det_v(sizes, rho) + logdet_a)
    return np.where(np.isfinite(out), out, -np.inf)


def _block_reml_shared(Y: np.ndarray, X: np.ndarray, codes: np.ndarray, sizes: np.ndarray,
                       rho: float) -> np.ndarray:
    """_block_reml with one ρ for every row, so the whitened design and its
    decomposition are shared."""
    n, p = X.shape
    Xw = _whiten(X[None], codes, sizes, rho)[0]
    Yw = _whiten(Y.T[None], codes, sizes, rho)[0]                 # (n, G)
    Q, R = np.linalg.qr(Xw)
    rss = np.sum(Yw ** 2, axis=0) - np.sum((Q.T @ Yw) ** 2, axis=0)
    logdet_a = 2.0 * np.sum(np.log(np.abs(np.diag(R))))
    with np.errstate(divide='ignore', invalid='ignore'):
        out = -0.5 * ((n - p) * np.log(np.maximum(rss, 0.0)) + _log_det_v(sizes, rho)[0]
                      + logdet_a)
    return np.where(np.isfinite(out), out, -np.inf)


def _maximise_block_reml(Y, X, codes, sizes, lo, hi):
    """Per-row maximiser of _block_reml over [lo, hi]: a grid, then golden
    section search between the grid points either side of the best."""
    grid = np.linspace(lo, hi, _RHO_GRID)
    ll = np.stack([_block_reml_shared(Y, X, codes, sizes, r) for r in grid])  # (K, G)
    best = np.argmax(ll, axis=0)
    left = grid[np.maximum(best - 1, 0)]
    right = grid[np.minimum(best + 1, grid.size - 1)]
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1 = _block_reml(Y, X, codes, sizes, x1)
    f2 = _block_reml(Y, X, codes, sizes, x2)
    for _ in range(_GOLDEN_STEPS):
        move_right = f1 < f2
        left = np.where(move_right, x1, left)
        right = np.where(move_right, right, x2)
        new1 = right - ratio * (right - left)
        new2 = left + ratio * (right - left)
        x1, x2 = np.where(move_right, x2, new1), np.where(move_right, new2, x1)
        f_new = _block_reml(Y, X, codes, sizes, np.where(move_right, x2, x1))
        f1, f2 = np.where(move_right, f2, f_new), np.where(move_right, f_new, f1)
    rho = np.where(f1 > f2, x1, x2)
    return rho, np.maximum(f1, f2), grid, ll


def trimmed_mean(x, trim: float) -> float:
    """Mean after dropping floor(n·trim) values from each end (R's
    mean(x, trim = trim)); NaNs ignored."""
    x = np.sort(np.asarray(x, dtype=float)[np.isfinite(x)])
    n = x.size
    if n == 0:
        return np.nan
    if trim >= 0.5:
        return float(np.median(x))
    cut = int(np.floor(n * trim))
    return float(np.mean(x[cut:n - cut]))


@dataclass
class BlockCorrelation:
    """Correlation between samples of one individual, per feature and
    pooled."""
    consensus: float              # tanh of the trimmed mean of atanh(ρ_g)
    rho: np.ndarray               # (G,) per-feature REML estimate; NaN if not estimable
    lr: np.ndarray                # (G,) REML likelihood ratio for ρ > 0 against ρ = 0
    p_value: np.ndarray           # (G,) from the 50:50 mixture of χ²₀ and χ²₁
    n_estimated: int              # features the consensus was taken over


def block_correlation(expr, design, block, trim: float = 0.15) -> BlockCorrelation:
    """
    Correlation between samples from the same individual, for every
    feature, after limma's ``duplicateCorrelation`` with ``block``.

    In plain terms: when each mouse gives several samples (tissues), the
    samples of one mouse tend to sit above or below the others together.
    That shared mouse effect is a random effect, and ρ is the share of the
    remaining variation it accounts for, σ²_mouse / (σ²_mouse + σ²).

    Model per feature: y = Xβ + b_individual + e, Var(y) = σ² V(ρ) with
    V = (1 − ρ) I + ρ ZZᵀ. ρ is estimated by REML (σ² profiled out, so a
    one-dimensional search) over the feature's observed samples, between
    −1/(m − 1) and 1 for m the largest number of samples of one individual.
    As in limma, features need more observed samples than coefficients + 2.
    The consensus is tanh of the trimmed mean (``trim`` from each end) of
    atanh(ρ_g). Negative values are kept, as limma keeps them.

    Also returns each feature's test of ρ > 0: the REML likelihood ratio
    between the best ρ ≥ 0 and ρ = 0, whose null distribution is the 50:50
    mixture of χ²₀ and χ²₁ because ρ = 0 is on the boundary (Stram & Lee
    1994).
    """
    Y = np.asarray(expr, dtype=float)
    X = np.asarray(design, dtype=float)
    if Y.ndim == 1:
        Y = Y[None, :]
    codes_all, _n = _factor_codes(block)
    G = Y.shape[0]
    rho = np.full(G, np.nan)
    lr = np.full(G, np.nan)
    obs_mask = np.isfinite(Y)
    patterns: dict[bytes, list[int]] = {}
    for g in range(G):
        patterns.setdefault(obs_mask[g].tobytes(), []).append(g)
    for key, rows in patterns.items():
        obs = np.frombuffer(key, dtype=bool)
        cols, estimable = estimable_columns(X[obs])
        if not estimable.any():
            continue
        Xc = X[obs][:, cols]
        n_o, p = Xc.shape
        codes, n_blocks = _factor_codes(codes_all[obs])
        sizes = np.bincount(codes, minlength=n_blocks).astype(float)
        if n_o <= p + 2 or sizes.max() < 2:
            continue
        idx = np.asarray(rows)
        Yo = Y[np.ix_(idx, np.flatnonzero(obs))]
        lo, hi = _rho_range(sizes)
        r_hat, ll_hat, grid, ll_grid = _maximise_block_reml(Yo, Xc, codes, sizes, lo, hi)
        ll_zero = _block_reml_shared(Yo, Xc, codes, sizes, 0.0)
        ok = np.isfinite(ll_hat)
        rho[idx] = np.where(ok, r_hat, np.nan)
        # Best ρ ≥ 0: the unrestricted maximum when it is positive, otherwise
        # the best grid value from 0 upwards (0 itself when the likelihood
        # falls away from 0).
        nonneg = grid >= 0.0
        ll_pos = np.maximum(np.max(ll_grid[nonneg], axis=0), ll_zero)
        ll_restricted = np.where(r_hat >= 0.0, np.maximum(ll_hat, ll_pos), ll_pos)
        lr[idx] = np.where(ok, np.maximum(2.0 * (ll_restricted - ll_zero), 0.0), np.nan)
    p_value = np.where(np.isfinite(lr), np.where(lr > 0, 0.5 * stats.chi2.sf(lr, 1), 1.0), np.nan)
    finite = rho[np.isfinite(rho)]
    consensus = (float(np.tanh(trimmed_mean(np.arctanh(np.clip(finite, -1.0 + 1e-12, 1.0 - 1e-12)),
                                            trim))) if finite.size else np.nan)
    return BlockCorrelation(consensus=consensus, rho=rho, lr=lr, p_value=p_value,
                            n_estimated=int(finite.size))


def gls_lm_fit(expr, design, block, correlation: float) -> LinearFit:
    """
    Generalised least squares of ``expr[g] ~ design`` with samples of one
    individual correlated by ``correlation`` (limma's ``lmFit`` with
    ``block`` and ``correlation``).

    Each feature's observed values and design rows are whitened by
    V^(-1/2), V = (1 − ρ) I + ρ ZZᵀ over its observed samples, and fitted
    by ``lm_fit``; coefficients, unscaled standard errors, residual
    variance and residual df (observed samples less the rank) are those of
    the whitened fit. ρ is kept inside the range where V is positive
    definite for that feature's samples. amean and n_obs are of the
    unwhitened values.
    """
    Y = np.asarray(expr, dtype=float)
    X = np.asarray(design, dtype=float)
    if Y.ndim == 1:
        Y = Y[None, :]
    if X.ndim != 2 or X.shape[0] != Y.shape[1]:
        raise ValueError(f"design has {X.shape[0] if X.ndim == 2 else '?'} rows but expr has "
                         f"{Y.shape[1]} samples")
    G, _n = Y.shape
    p = X.shape[1]
    codes_all, _nb = _factor_codes(block)
    coef = np.full((G, p), np.nan)
    stdu = np.full((G, p), np.nan)
    sigma2 = np.full(G, np.nan)
    df_res = np.zeros(G)
    obs_mask = np.isfinite(Y)
    n_obs = obs_mask.sum(axis=1)
    with np.errstate(invalid='ignore'):
        amean = np.where(n_obs > 0, np.where(obs_mask, Y, 0.0).sum(axis=1) / np.maximum(n_obs, 1),
                         np.nan)
    patterns: dict[bytes, list[int]] = {}
    for g in range(G):
        patterns.setdefault(obs_mask[g].tobytes(), []).append(g)
    for key, rows in patterns.items():
        obs = np.frombuffer(key, dtype=bool)
        if not obs.any():
            continue
        idx = np.asarray(rows)
        codes, n_blocks = _factor_codes(codes_all[obs])
        sizes = np.bincount(codes, minlength=n_blocks).astype(float)
        lo, hi = _rho_range(sizes)
        r = float(np.clip(correlation, lo, hi)) if sizes.max() > 1 else 0.0
        Yo = Y[np.ix_(idx, np.flatnonzero(obs))]
        Yw = _whiten(Yo[None, :, :].transpose(0, 2, 1), codes, sizes, r)[0].T
        Xw = _whiten(X[obs][None], codes, sizes, r)[0]
        sub = lm_fit(Yw, Xw)
        coef[idx] = sub.coefficients
        stdu[idx] = sub.stdev_unscaled
        sigma2[idx] = sub.sigma2
        df_res[idx] = sub.df_residual
    return LinearFit(coefficients=coef, stdev_unscaled=stdu, sigma2=sigma2,
                     df_residual=df_res, amean=amean, n_obs=n_obs, design=X)


# ---------------------------------------------------------------------------
# Variance components (all categorical variables as random effects)
# ---------------------------------------------------------------------------

_VC_MAX_ITER = 200
_VC_TOL = 1e-7


@dataclass
class VarianceComponents:
    """Per-feature variance of each component and its share of the total."""
    names: list                   # random factors, then fixed terms, then 'Residuals'
    kinds: list                   # 'random', 'fixed' or 'residual' per name
    variance: np.ndarray          # (G, K)
    fraction: np.ndarray          # (G, K); rows sum to 1 where estimated
    converged: np.ndarray         # (G,) bool


def _one_hot(codes: np.ndarray, n_levels: int) -> np.ndarray:
    z = np.zeros((codes.size, n_levels))
    z[np.arange(codes.size), codes] = 1.0
    return z


def _reml_components(Y: np.ndarray, X: np.ndarray, Zs: list) -> tuple[np.ndarray, np.ndarray,
                                                                       np.ndarray]:
    """
    REML variance components for every row of Y (features × samples, all
    observed) under y = Xβ + Σ_j Z_j u_j + e, u_j ~ N(0, σ²_j I),
    e ~ N(0, σ²_0 I), by Fisher scoring on the restricted likelihood
    (Harville 1977) with every component kept ≥ 0: a component at 0 stays
    out of a step while its score is negative.

    Returns (variances (G, 1 + len(Zs)) with the residual first, β (G, p),
    converged (G,)).
    """
    G, n = Y.shape
    p = X.shape[1]
    comps = [None] + list(Zs)             # None: the residual (Z = I)
    K = len(comps)
    resid = Y - (Y @ np.linalg.pinv(X).T) @ X.T
    s2 = np.sum(resid ** 2, axis=1) / max(n - p, 1)
    sig = np.repeat((s2 / K)[:, None], K, axis=1)
    sig[:, 0] = np.maximum(sig[:, 0], 1e-12)
    floor = 1e-10 * np.maximum(s2, 1e-300)
    converged = np.zeros(G, dtype=bool)
    eye = np.eye(n)
    zzt = [eye] + [Z @ Z.T for Z in Zs]
    beta = np.zeros((G, p))
    for _ in range(_VC_MAX_ITER):
        V = np.einsum('gk,kij->gij', sig, np.stack(zzt))
        Vinv = np.linalg.inv(V)
        XtVi = np.einsum('ip,gij->gpj', X, Vinv)
        M = XtVi @ X
        Minv = np.linalg.inv(M)
        P = Vinv - np.einsum('gpi,gpq,gqj->gij', XtVi, Minv, XtVi)
        Py = np.einsum('gij,gj->gi', P, Y)
        PZ = [P if Z is None else P @ Z for Z in comps]
        score = np.empty((G, K))
        info = np.empty((G, K, K))
        for j, Z in enumerate(comps):
            if Z is None:
                tr = np.einsum('gii->g', P)
                quad = np.sum(Py ** 2, axis=1)
            else:
                tr = np.einsum('il,gil->g', Z, PZ[j])
                quad = np.sum((Py @ Z) ** 2, axis=1)
            score[:, j] = 0.5 * (quad - tr)
            for k in range(j, K):
                if Z is None:
                    cross = PZ[k]
                else:
                    cross = np.einsum('il,gim->glm', Z, PZ[k])
                val = 0.5 * np.sum(cross ** 2, axis=(1, 2))
                info[:, j, k] = val
                info[:, k, j] = val
        active = (sig > floor[:, None]) | (score > 0)
        active[:, 0] = True
        info_a = np.where(active[:, :, None] & active[:, None, :], info, 0.0)
        info_a = info_a + np.eye(K)[None] * (~active)[:, :, None]
        step = np.linalg.solve(info_a, np.where(active, score, 0.0)[:, :, None])[:, :, 0]
        new = np.maximum(sig + step, 0.0)
        new[:, 0] = np.maximum(new[:, 0], floor)
        change = np.max(np.abs(new - sig), axis=1) / np.maximum(np.sum(new, axis=1), 1e-300)
        sig = new
        converged = change < _VC_TOL
        if converged.all():
            break
    V = np.einsum('gk,kij->gij', sig, np.stack(zzt))
    Vinv = np.linalg.inv(V)
    XtVi = np.einsum('ip,gij->gpj', X, Vinv)
    beta = np.linalg.solve(XtVi @ X, np.einsum('gpj,gj->gp', XtVi, Y)[:, :, None])[:, :, 0]
    return sig, beta, converged


def variance_components(expr, random: dict, fixed: dict | None = None) -> VarianceComponents:
    """
    Share of each feature's variation explained by each variable, after
    variancePartition (Hoffman & Schadt 2016).

    In plain terms: for each feature, how much of the spread of its values
    across samples comes from the group, from the tissue, from the
    individual animal or donor, and how much is left over. The shares add
    up to 100%. This describes the data; it is not a test.

    Model per feature: y = intercept + Σ fixed terms + Σ_j u_j + e, every
    categorical variable in ``random`` a random effect u_j (one value per
    level, variance σ²_j), fitted by REML over the feature's observed
    samples. A fixed (numeric) term's variance is the sample variance of
    its fitted contribution x·β. Fraction = component / (sum of random
    components + fixed variances + residual variance).

    random : {name: one value per sample}; a factor with one level, or one
             level per sample, over a feature's observed samples cannot be
             separated from the intercept or the residual and gets 0.
    fixed  : {name: (samples,) or (samples × q) numeric columns}.
    """
    Y = np.asarray(expr, dtype=float)
    if Y.ndim == 1:
        Y = Y[None, :]
    G, n = Y.shape
    fixed = dict(fixed or {})
    rnames = list(random)
    fnames = list(fixed)
    names = rnames + fnames + ['Residuals']
    kinds = ['random'] * len(rnames) + ['fixed'] * len(fnames) + ['residual']
    rcodes = [_factor_codes(random[r])[0] for r in rnames]
    fcols, fterm = [np.ones((n, 1))], [-1]
    for t, name in enumerate(fnames):
        col = np.asarray(fixed[name], dtype=float).reshape(n, -1)
        fcols.append(col)
        fterm.extend([t] * col.shape[1])
    Xall = np.column_stack(fcols)
    fterm = np.asarray(fterm)

    variance = np.full((G, len(names)), np.nan)
    converged = np.zeros(G, dtype=bool)
    obs_mask = np.isfinite(Y)
    patterns: dict[bytes, list[int]] = {}
    for g in range(G):
        patterns.setdefault(obs_mask[g].tobytes(), []).append(g)
    for key, rows in patterns.items():
        obs = np.frombuffer(key, dtype=bool)
        n_o = int(obs.sum())
        cols, estimable = estimable_columns(Xall[obs])
        if not estimable.any() or n_o - cols.size < 2:
            continue
        Xo = Xall[obs][:, cols]
        kept, Zs = [], []
        for j, codes in enumerate(rcodes):
            c, n_levels = _factor_codes(codes[obs])
            if 1 < n_levels < n_o:
                kept.append(j)
                Zs.append(_one_hot(c, n_levels))
        idx = np.asarray(rows)
        Yo = Y[np.ix_(idx, np.flatnonzero(obs))]
        sig, beta, conv = _reml_components(Yo, Xo, Zs)
        var = np.zeros((idx.size, len(names)))
        var[:, -1] = sig[:, 0]
        for pos, j in enumerate(kept):
            var[:, j] = sig[:, pos + 1]
        for t in range(len(fnames)):
            sel = np.flatnonzero(fterm[cols] == t)
            if sel.size:
                contrib = beta[:, sel] @ Xo[:, sel].T
                var[:, len(rnames) + t] = np.var(contrib, axis=1, ddof=1)
        variance[idx] = var
        converged[idx] = conv
    with np.errstate(invalid='ignore', divide='ignore'):
        fraction = variance / np.sum(variance, axis=1, keepdims=True)
    return VarianceComponents(names=names, kinds=kinds, variance=variance, fraction=fraction,
                              converged=converged)


# ---------------------------------------------------------------------------
# Moderated F test of several coefficients
# ---------------------------------------------------------------------------

@dataclass
class FTest:
    """Joint test of some design columns for every feature."""
    F: np.ndarray                 # (G,) moderated F
    df1: np.ndarray               # (G,) columns tested (estimable for that feature)
    df2: np.ndarray               # (G,) d0 + d, as for the moderated t
    p_value: np.ndarray           # (G,)
    partial_r2: np.ndarray        # (G,) share of the residual variation explained
    fit: ModeratedFit             # the full model


def f_pvalues(F, df1, df2) -> np.ndarray:
    """Upper tail of F(df1, df2); the chi-square limit (F·df1 on df1) where
    df2 is infinite."""
    F, df1, df2 = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (F, df1, df2)))
    finite = np.where(np.isinf(df2), 1.0, df2)
    with np.errstate(invalid='ignore'):
        return np.where(np.isinf(df2), stats.chi2.sf(F * df1, df1), stats.f.sf(F, df1, finite))


def f_test(expr, design, coefs, block=None, correlation: float = 0.0) -> FTest:
    """
    Moderated F test that the design columns ``coefs`` are all zero.

    F = (RSS_reduced − RSS_full) / df1 / s̃², where the reduced model drops
    ``coefs``, df1 is the drop in residual df and s̃² the moderated residual
    variance of the full model (e_bayes). For least squares the RSS drop
    equals β̂ᵀ V⁻¹ β̂ with V the unscaled covariance of the tested
    coefficients, so this is limma's moderated F (topTableF); with one
    column it is the square of the moderated t. The p-value is from
    F(df1, d0 + d). partial_r2 = (RSS_reduced − RSS_full) / RSS_reduced on
    the ordinary residuals: the share of the variation left after the other
    columns that the tested columns explain (the ANOVA partial η²).

    With ``block`` (one individual per sample) the fits are generalised
    least squares with samples of one individual correlated by
    ``correlation`` (gls_lm_fit), and the sums of squares and partial R²
    are those of the whitened values.

    Features with no residual df, or with none of ``coefs`` estimable over
    their observed samples, get NaN.
    """
    X = np.asarray(design, dtype=float)
    coefs = sorted(int(c) for c in coefs)
    if block is None:
        fitter = lm_fit
    else:
        def fitter(y, x):
            return gls_lm_fit(y, x, block, correlation)
    full = e_bayes(fitter(expr, X))
    reduced = fitter(expr, np.delete(X, coefs, axis=1))
    rss_full = np.where(full.df_residual > 0, full.sigma2 * full.df_residual, np.nan)
    rss_reduced = np.where(reduced.df_residual > 0, reduced.sigma2 * reduced.df_residual, np.nan)
    df1 = (reduced.df_residual - full.df_residual).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        drop = np.maximum(rss_reduced - rss_full, 0.0)
        F = np.where(df1 > 0, drop / df1 / full.s2_post, np.nan)
        r2 = np.where(rss_reduced > 0, drop / rss_reduced, np.nan)
    F = np.where(full.df_residual > 0, F, np.nan)
    df1 = np.where(np.isfinite(F), df1, np.nan)
    return FTest(F=F, df1=df1, df2=full.df_total, p_value=f_pvalues(F, df1, full.df_total),
                 partial_r2=np.where(np.isfinite(F), r2, np.nan), fit=full)

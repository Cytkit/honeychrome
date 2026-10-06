"""
moderated_stats.py — moderated linear models for differential testing
=====================================================================

Per-feature linear models with empirical Bayes variance moderation, for
testing many features (clusters, cluster × marker MFIs, gated populations)
across a small number of samples.

Methods and the papers they are written from:

* ``lm_fit`` — ordinary least squares per feature, allowing missing values.
* ``e_bayes`` — the hierarchical model of Smyth (2004): residual variances
  are shrunk towards a common prior whose degrees of freedom ``d0`` and
  scale ``s0²`` are estimated by the method of moments on log variances.
  Gives moderated t-statistics and p-values.
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
    'LinearFit', 'ModeratedFit', 'lm_fit', 'e_bayes', 'top_table',
    'treat_pvalues', 'bh_adjust', 'simes', 'stagewise_adjust',
    'fit_prior_variance', 'trigamma_inverse',
]

_MIN_FEATURES = 3


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
    p_value: np.ndarray = None    # (G, p) two-sided


# ---------------------------------------------------------------------------
# Linear model
# ---------------------------------------------------------------------------

def lm_fit(expr, design) -> LinearFit:
    """
    Fit ``expr[g] ~ design`` by least squares for every feature g.

    expr   : (features × samples) array; NaN marks a missing value.
    design : (samples × coefficients) array.

    Each feature uses only its observed samples. Design columns that are
    all zero over those samples (e.g. a blocking level whose samples are
    all missing) are dropped for that feature and their coefficients are
    NaN. If the remaining columns are still rank-deficient the feature is
    not estimable: all its coefficients are NaN and its residual df is 0.
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
        cols = np.flatnonzero(np.any(Xo != 0.0, axis=0))
        if cols.size == 0:
            continue
        Xc = Xo[:, cols]
        rank = np.linalg.matrix_rank(Xc)
        if rank < cols.size:
            continue
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
        df_res[idx] = d
        if d > 0:
            sigma2[idx] = np.sum(resid ** 2, axis=0) / d

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
    p = 2.0 * _t_sf(np.abs(t), df_total[:, None])

    return ModeratedFit(
        coefficients=fit.coefficients, stdev_unscaled=fit.stdev_unscaled,
        sigma2=fit.sigma2, df_residual=fit.df_residual, amean=fit.amean,
        n_obs=fit.n_obs, design=fit.design,
        df_prior=float(d0), s2_prior=float(s0_2), s2_post=s2_post,
        df_total=df_total, t=t, p_value=p,
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
    fam = np.asarray(family, dtype=object)
    labels = list(dict.fromkeys(fam.tolist()))
    fam_p = np.array([simes(p[fam == f]) for f in labels], dtype=float)
    fam_adj = bh_adjust(fam_p)
    n_fam = int(np.isfinite(fam_p).sum())
    selected = np.isfinite(fam_adj) & (fam_adj <= alpha)
    n_sel = int(selected.sum())

    family_p = np.full(p.shape, np.nan)
    family_adj = np.full(p.shape, np.nan)
    adjusted = np.where(np.isfinite(p), 1.0, np.nan)
    for f, fp, fa, sel in zip(labels, fam_p, fam_adj, selected):
        members = fam == f
        family_p[members] = fp
        family_adj[members] = fa
        if sel and n_sel > 0:
            within = bh_adjust(p[members])
            adjusted[members] = np.minimum(within * n_fam / n_sel, 1.0)
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

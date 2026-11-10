"""
trend_stats.py — trends of per-sample features with a numeric variable
=====================================================================

Shared engine for asking whether per-sample features (population or
cluster frequencies, counts as log odds, marker intensities) change with a
numeric variable: age, days after vaccination or infection, any time
course. Used by the Honeycluster and Automated Gating
plugins.

In plain terms: each feature's values are plotted against the variable
and two things are fitted. A smooth curve answers "does this change at
all, whatever the shape?" (one p-value for the variable as a whole, as in
an ANOVA). A straight line gives the direction and the rate (log2 units
per week, per day, or per doubling of time). Partial R² says how much of
the sample-to-sample variation the variable explains once groups and
adjustment covariates are accounted for.

Methods:

* Shape of the variable's effect (``SHAPES``):
  - ``'curve'`` — natural cubic spline with k degrees of freedom (default
    2: one bend), boundary knots at the 5th and 95th percentiles and
    interior knots at quantiles. Beyond the boundary knots the curve is a
    straight line, so a single sample at the youngest or oldest end cannot
    bend it. k is capped so there are at least 10 samples per curve df.
  - ``'line'`` — a straight line.
  - ``'points'`` — each distinct value of the variable is its own level
    (designed time courses such as day 0, 3, 7, 14): a one-way ANOVA.
* Tests, on the moderated linear model of Smyth (2004)
  (``moderated_stats``):
  - the moderated F test of the shape's columns (limma's ``topTableF``);
  - the moderated t test of a straight-line term (the slope).
  Counts are tested as log odds with the moderated model; the
  quasi-likelihood counts test is not used here, because with a curve it
  gave far too many false trends on real data (a human cohort with age
  shuffled among the samples of one group: 18% of p-values below 0.05).
* Scale (``SCALES``): ``'linear'``, or ``'log2'`` (log2(t + 1)) for time
  courses sampled on a doubling schedule; the slope is then per doubling.
* Groups: with several groups the trend is common to them and the groups
  are adjustment terms. With ``interaction=True`` (exactly two groups) the
  question is whether the trend differs between the groups: the tested
  columns are group × shape, the variable's own effect and the group
  are in the model, and the slope is the difference in slope (second
  group minus the first). Without pairing, each group's own curve and
  band are kept as well (``TrendResult.group_curves``), so the two groups
  can be drawn on the same axes.
* Pairing: kept when the variable changes within pairing levels (a donor
  sampled at several times). When every level holds a single value of the
  variable (two tissues of one donor taken at the same age) the pairing
  would absorb a common trend, so the common trend is fitted without
  pairing, while the interaction keeps it (it then tests the within-pair
  difference against the variable).
* Effect sizes: partial R² of the curve (and of the line), the slope, and
  from the fitted curve on a grid over the observed range: range (largest
  minus smallest fitted value), net change (last minus first) and the
  values of the variable at the curve's maximum and minimum.
* Confidence band: the pointwise standard error of each fitted curve on
  the grid (``TrendResult.curve_se``), gᵀ V g · s̃² with V the unscaled
  covariance of the curve's coefficients and s̃² the moderated variance;
  a 95% band is the curve ± t(0.975, df2) · SE.
* Calling (``trend_significance``, re-applied without refitting):
  ``'either'`` (default) — curve or slope significant, each at FDR α/2 so
  the combined error stays at α; ``'curve'``; ``'slope'``; each with a
  minimum partial R² (default 0.1) for the test that made the call.

* Shape patterns (``add_patterns``, ``assign_patterns``): the called
  features' fitted curves, each scaled to mean 0 and SD 1, are grouped by
  Ward's hierarchical clustering, so features group by the shape of their
  change whatever its size; each group is named from its mean curve
  ('rises', 'falls then levels off', 'peaks at 27 weeks'). The number of
  patterns can be chosen from stability: the test is rerun on subsamples
  of 80% of the samples (whole pairs when paired) and the largest number
  of patterns whose grouping agrees with the full data's (adjusted Rand
  index) at least PATTERN_STABILITY on average is used.

Deliberate v1 simplifications: the interaction takes exactly two groups;
pairing is a fixed blocking term only (no random effect such as limma's
duplicateCorrelation); quasi-likelihood counts are not offered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from honeychrome.controller_components import differential_stats as ds
from honeychrome.controller_components import moderated_stats as ms

log = logging.getLogger(__name__)

__all__ = [
    'SHAPES', 'SCALES', 'CALLS', 'DEFAULT_CURVE_DF', 'DEFAULT_BOUNDARY', 'DEFAULT_MIN_R2',
    'SplineBasis', 'curve_df', 'suggest_shape', 'TrendResult', 'run_trend',
    'trend_significance', 'design_check', 'result_to_jsonable', 'result_from_jsonable',
    'DESIGN_CHECK_LIMIT', 'TrendDesign', 'trend_design',
    'zscore_curves', 'cluster_curves', 'pattern_name', 'assign_patterns', 'recall',
    'pattern_stability', 'choose_pattern_k', 'add_patterns', 'PATTERN_K_RANGE',
    'PATTERN_STABILITY', 'fit_composition', 'merge_small_parts', 'COMPOSITION_PSEUDOCOUNT',
    'composition_panels',
]

SHAPES = ('curve', 'line', 'points')
SCALES = ('linear', 'log2')
CALLS = ('either', 'curve', 'slope')
DEFAULT_CURVE_DF = 2
DEFAULT_BOUNDARY = 0.05
DEFAULT_MIN_R2 = 0.1
SAMPLES_PER_CURVE_DF = 10
MAX_POINT_LEVELS = 6
MIN_SAMPLES_PER_POINT = 3
MIN_TREND_SAMPLES = 6
DESIGN_CHECK_LIMIT = 0.5
COMPOSITION_PSEUDOCOUNT = 0.5


class SplineBasis:
    """
    Natural cubic spline basis (Hastie, Tibshirani & Friedman, The Elements
    of Statistical Learning, eq. 5.4–5.5) of ``k`` degrees of freedom, with
    boundary knots at the ``boundary`` and 1 − ``boundary`` quantiles of x
    and k − 1 interior knots at quantiles between them. The basis is linear
    outside the boundary knots. Its columns are centred and orthonormalised
    on the fitting values, and the same transform is applied to any other
    values (e.g. a plotting grid), so fitted coefficients can be read on
    the grid. Tied quantiles lower k (``self.k`` is the df used).
    """

    def __init__(self, x, k: int, boundary: float = 0.0):
        x = np.asarray(x, dtype=float)
        k = max(int(k), 1)
        knots = np.unique(np.quantile(x, np.linspace(boundary, 1.0 - boundary, k + 1)))
        if knots.size < 2:
            knots = np.unique([x.min(), x.max()])
        self.knots = knots
        self.k = knots.size - 1
        raw = self._raw(x)
        self.mean = raw.mean(axis=0)
        _q, r = np.linalg.qr(raw - self.mean)
        self.rinv = np.linalg.inv(r) * np.sqrt(x.size)

    def _cube(self, x, v):
        lo, hi = self.knots[0], self.knots[-1]
        u = np.clip(x, lo, hi)
        inside = np.clip(u - v, 0.0, None) ** 3
        return inside + 3.0 * np.clip(hi - v, 0.0, None) ** 2 * np.clip(x - hi, 0.0, None)

    def _raw(self, x):
        x = np.asarray(x, dtype=float)
        if self.k == 1:
            return x[:, None]
        xi = self.knots
        last = xi.size - 1

        def d(j):
            return (self._cube(x, xi[j]) - self._cube(x, xi[last])) / (xi[last] - xi[j])
        return np.column_stack([x] + [d(j) - d(last - 1) for j in range(last - 1)])

    def __call__(self, x) -> np.ndarray:
        return (self._raw(x) - self.mean) @ self.rinv


def curve_df(n_samples: int, k: int = DEFAULT_CURVE_DF) -> int:
    """The curve df used: ``k``, capped so each curve df has at least
    SAMPLES_PER_CURVE_DF samples (never below 1, a straight line)."""
    return int(max(1, min(int(k), n_samples // SAMPLES_PER_CURVE_DF)))


def suggest_shape(values) -> str:
    """'points' for a designed time course (at most MAX_POINT_LEVELS distinct
    values, each with at least MIN_SAMPLES_PER_POINT samples), else
    'curve'."""
    v = pd.Series(np.asarray(values, dtype=float)).dropna()
    counts = v.value_counts()
    if 2 <= counts.size <= MAX_POINT_LEVELS and counts.min() >= MIN_SAMPLES_PER_POINT:
        return 'points'
    return 'curve'


@dataclass
class TrendResult:
    """Per-feature trend results.

    table   : one row per feature (see run_trend).
    grid    : values of the variable (original units) the curves are on.
    curves  : (features × grid) fitted curve of each feature, in the data's
              units; with an interaction, the difference between the groups.
    notes   : plain-language lines on samples, terms or pairing left out.
    settings: the options used (variable, unit, scale, shape, df, groups).
    curve_se: (features × grid) pointwise standard error of each curve.
    group_curves, group_curve_se: (2 × features × grid) each group's own
              fitted curve and its standard error, in the order of
              settings['groups']; only for a group × trend test without
              pairing (with pairing only the difference is estimable).
    """
    table: pd.DataFrame
    grid: np.ndarray
    curves: np.ndarray
    notes: list[str] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    curve_se: np.ndarray | None = None
    group_curves: np.ndarray | None = None
    group_curve_se: np.ndarray | None = None


class _Shape:
    """Columns of the tested shape at the fitting values and at any others."""

    def __init__(self, x, shape: str, k: int, boundary: float):
        self.shape = shape
        if shape == 'points':
            self.levels = np.unique(x)
            if self.levels.size < 2:
                raise RuntimeError("The variable has a single value over these samples.")
            self.k = self.levels.size - 1
        else:
            self.spline = SplineBasis(x, 1 if shape == 'line' else k,
                                      boundary if shape == 'curve' else 0.0)
            self.k = self.spline.k

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.shape == 'points':
            return np.column_stack([np.isclose(x, lv).astype(float) for lv in self.levels[1:]])
        return self.spline(x)


def _transform(t: np.ndarray, scale: str) -> np.ndarray:
    return np.log2(t + 1.0) if scale == 'log2' else t


def _single_value_per_level(values: np.ndarray, pairing: list[str]) -> bool:
    frame = pd.DataFrame({'v': values, 'p': [str(p) for p in pairing]})
    per_level = frame.groupby('p')['v'].nunique()
    return bool((per_level <= 1).all())


def _curve_se(Y: np.ndarray, X: np.ndarray, cols: list[int], G: np.ndarray,
              s2_post: np.ndarray, offset_col: int | None = None) -> np.ndarray:
    """
    Pointwise standard error of each feature's fitted curve on a grid.

    G holds the tested columns' values at the grid points (grid × k). The
    variance of the curve at a grid point is gᵀ V g · s̃², with V the
    unscaled covariance (XᵀX)⁻¹ of the tested coefficients over the samples
    the feature was measured in and s̃² its moderated variance. With
    ``offset_col`` (the second group's column of an interaction) the curve
    is that coefficient plus the tested part, and their covariance is
    included. Otherwise the curve's level is a mean over the n samples,
    whose variance is taken as s̃² / n (exact when the variable is the only
    term besides the intercept).
    """
    observed = np.isfinite(Y)
    se = np.full((Y.shape[0], G.shape[0]), np.nan)
    if offset_col is None:
        idx, C = list(cols), G
    else:
        idx, C = [offset_col] + list(cols), np.column_stack([np.ones(G.shape[0]), G])
    patterns, inverse = np.unique(observed, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    for p, mask in enumerate(patterns):
        rows = np.flatnonzero(inverse == p)
        Xm = X[mask]
        V = np.linalg.pinv(Xm.T @ Xm)[np.ix_(idx, idx)]
        var = np.einsum('gi,ij,gj->g', C, V, C)
        if offset_col is None:
            var = var + 1.0 / max(int(mask.sum()), 1)
        se[rows] = np.sqrt(np.maximum(var, 0.0))[None, :] * np.sqrt(s2_post[rows])[:, None]
    return se


def _group_curves(Y: np.ndarray, X: np.ndarray, main_cols: list[int], int_cols: list[int],
                  S: np.ndarray, G: np.ndarray, second: np.ndarray, beta: np.ndarray,
                  s2_post: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Each group's own fitted curve and its pointwise standard error in an
    unpaired group × trend model (shape columns *main_cols* for the first
    group, plus the group × shape columns *int_cols* for the second).

    The first group's shape coefficients are b = β_shape, the second
    group's b = β_shape + β_interaction. Each curve is the group's mean
    value plus the shape measured from the group's mean shape value,
    ȳ_g + (G − S̄_g) b: without adjustment covariates this is exactly the
    group's own least-squares curve. The group mean and the shape
    coefficients are uncorrelated (each group has its own level in the
    model), so the curve's variance is (1 / n_g + cᵀ V c) · s̃², with
    c = G − S̄_g and V the unscaled covariance of the coefficients over the
    samples the feature was measured in.

    second : True for the samples of the second group (rows of Y's columns).
    Returns (curves, se), each 2 × features × grid.
    """
    n_feat, n_grid = Y.shape[0], G.shape[0]
    b_main = np.nan_to_num(beta[:, main_cols])
    b_int = np.nan_to_num(beta[:, int_cols])
    curves = np.full((2, n_feat, n_grid), np.nan)
    se = np.full((2, n_feat, n_grid), np.nan)
    observed = np.isfinite(Y)
    patterns, inverse = np.unique(observed, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    for gi, in_group in enumerate((~second, second)):
        b = b_main + b_int if gi else b_main
        with np.errstate(invalid='ignore'):
            level = np.nanmean(Y[:, in_group] - b @ S[in_group].T, axis=1)
        curves[gi] = level[:, None] + b @ G.T
        idx = list(main_cols) + (list(int_cols) if gi else [])
        for p, mask in enumerate(patterns):
            rows = np.flatnonzero(inverse == p)
            members = mask & in_group
            if not members.any():
                continue
            centred = G - S[members].mean(axis=0)
            C = np.column_stack([centred, centred]) if gi else centred
            Xm = X[mask]
            V = np.linalg.pinv(Xm.T @ Xm)[np.ix_(idx, idx)]
            var = np.einsum('gi,ij,gj->g', C, V, C) + 1.0 / int(members.sum())
            se[gi, rows] = (np.sqrt(np.maximum(var, 0.0))[None, :]
                            * np.sqrt(s2_post[rows])[:, None])
    return curves, se


@dataclass
class TrendDesign:
    """The model of a trend test over its samples (see trend_design).

    rows      : positions (in the samples given) of the samples used.
    t, x      : the variable over those samples, in its units and on the
                tested scale.
    groups, order, base : group of each used sample, the groups in order,
                the baseline group.
    pairing   : pairing level of each used sample, or None when not kept.
    absorbed_time : the variable is constant within every pairing level.
    X0, group_cols : groups, pairing and adjustment covariates (no
                variable); column of each non-baseline group.
    shape, S  : the tested shape and its columns at x.
    X_s, cols_s : design and tested columns of the shape (curve) test.
    X_l, cols_l : the same for the straight line (slope).
    notes     : plain-language lines on samples, terms or pairing left out.
    """
    rows: np.ndarray
    t: np.ndarray
    x: np.ndarray
    groups: list
    order: list
    base: str
    pairing: list | None
    absorbed_time: bool
    X0: np.ndarray
    group_cols: dict
    shape: '_Shape'
    S: np.ndarray
    X_s: np.ndarray
    cols_s: list
    X_l: np.ndarray
    cols_l: list
    interaction: bool
    notes: list = field(default_factory=list)

    @property
    def X_null(self) -> np.ndarray:
        """The model without the tested columns (the null of both tests)."""
        return np.delete(self.X_s, self.cols_s, axis=1)

    def permutation(self, rng) -> np.ndarray:
        """
        A permutation of the used samples that is exchangeable when the
        variable has no effect, for residual permutation (Freedman & Lane
        1983):

        * when pairing is kept and every pairing level holds one value of
          the variable (two tissues of one donor taken at the same age),
          whole pairing levels exchange places among levels of the same
          make-up, each sample taking the place of the same group's sample
          in the other level: each pair keeps its shared value;
        * when pairing is kept and the variable changes within levels (a
          donor sampled at several times), samples move within their level;
        * otherwise samples move within their group.
        """
        n = self.rows.size
        out = np.arange(n)
        if self.pairing is not None and self.absorbed_time:
            levels: dict = {}
            for i, (p, g) in enumerate(zip(self.pairing, self.groups)):
                levels.setdefault(str(p), []).append((str(g), i))
            by_makeup: dict = {}
            for members in levels.values():
                members.sort()
                by_makeup.setdefault(tuple(g for g, _i in members), []).append(members)
            for same in by_makeup.values():
                for src_members, dst in zip(same, [same[i] for i in rng.permutation(len(same))]):
                    for (_g, i_from), (_h, i_to) in zip(src_members, dst):
                        out[i_from] = i_to
            return out
        blocks = self.pairing if self.pairing is not None else self.groups
        blocks = np.array([str(b) for b in blocks])
        for b in np.unique(blocks):
            members = np.flatnonzero(blocks == b)
            out[members] = rng.permutation(members)
        return out


def trend_design(samples, time, *, variable: str = 'time', scale: str = 'linear',
                 shape: str = 'curve', k: int = DEFAULT_CURVE_DF,
                 boundary: float = DEFAULT_BOUNDARY, group_vec: list[str] | None = None,
                 interaction: bool = False, baseline: str | None = None,
                 pairing_vec: list[str] | None = None,
                 covariates: pd.DataFrame | None = None) -> TrendDesign:
    """
    The model of a trend test (arguments as for run_trend, with *samples*
    the sample ids in the order of *time*): which samples are used, whether
    pairing is kept, and the designs of the curve and slope tests. Shared
    by run_trend and by permutation tests built on it.
    """
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}, got {shape!r}")
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}, got {scale!r}")
    notes: list[str] = []
    samples = list(samples)
    t_all = pd.to_numeric(pd.Series(list(time), index=samples), errors='coerce').to_numpy(float)
    keep = np.isfinite(t_all)
    if scale == 'log2':
        keep &= t_all > -1.0
    if (~keep).any():
        notes.append(f"{int((~keep).sum())} sample(s) without a usable value for '{variable}' "
                     "were left out.")
    groups_all = list(group_vec) if group_vec is not None else ['all'] * len(samples)

    cov = None
    if covariates is not None and len(covariates.columns):
        cov = covariates.reindex(samples)
        cov = cov[[c for c in cov.columns if c != variable]]
        if len(cov.columns):
            gone = keep & cov.isna().any(axis=1).to_numpy()
            if gone.any():
                notes.append(f"{int(gone.sum())} sample(s) without a value for an adjustment "
                             "covariate were left out.")
                keep &= ~gone
        else:
            cov = None

    rows = np.flatnonzero(keep)
    n = rows.size
    if n < MIN_TREND_SAMPLES:
        raise RuntimeError(f"Only {n} sample(s) have a value for '{variable}'; a trend needs "
                           f"at least {MIN_TREND_SAMPLES}.")
    t = t_all[rows]
    x = _transform(t, scale)
    groups = [groups_all[i] for i in rows]
    order = list(dict.fromkeys(groups))
    base = baseline if baseline in order else order[0]
    order = [base] + [g for g in order if g != base]
    if interaction and len(order) != 2:
        raise RuntimeError("A group × trend test needs exactly two groups; "
                           f"these samples have {len(order)}.")
    pairing = [pairing_vec[i] for i in rows] if pairing_vec is not None else None
    if pairing is not None:
        multi = pd.Series(pairing).value_counts().max() > 1
        if not multi:
            pairing = None
        elif not interaction and _single_value_per_level(t, pairing):
            notes.append(f"Each pairing level has a single value of '{variable}', so pairing "
                         "would absorb the trend; it was fitted without pairing.")
            pairing = None
    sub_cov = cov.iloc[rows] if cov is not None else None
    absorbed_time = pairing is not None and _single_value_per_level(t, pairing)
    X0, _names, group_cols = ds.build_design(groups, base, pairing, sub_cov)
    for name in ds.pairing_absorbed(sub_cov, pairing):
        notes.append(f"'{name}' is the same within every pairing level, so the pairing already "
                     "accounts for it; it was left out of the model.")

    k_used = curve_df(n, k) if shape == 'curve' else k
    if shape == 'curve' and k_used < k:
        notes.append(f"The curve was given {k_used} df instead of {k}: {n} samples allow "
                     f"{SAMPLES_PER_CURVE_DF} samples per df.")
    sh = _Shape(x, shape, k_used, boundary)
    S = sh(x)
    line = (x - x.mean())[:, None]
    other = X0[:, group_cols[order[1]]][:, None] if interaction else None

    def with_term(columns: np.ndarray):
        if not interaction:
            return np.column_stack([X0, columns]), list(range(X0.shape[1], X0.shape[1] + columns.shape[1]))
        main = [] if absorbed_time else [columns]
        X = np.column_stack([X0] + main + [other * columns])
        return X, list(range(X.shape[1] - columns.shape[1], X.shape[1]))

    X_s, cols_s = with_term(S)
    X_l, cols_l = with_term(line)
    return TrendDesign(rows=rows, t=t, x=x, groups=groups, order=order, base=base,
                       pairing=pairing, absorbed_time=bool(absorbed_time), X0=X0,
                       group_cols=group_cols, shape=sh, S=S, X_s=X_s, cols_s=cols_s, X_l=X_l,
                       cols_l=cols_l, interaction=bool(interaction), notes=notes)


def run_trend(data_df: pd.DataFrame, time, *, variable: str = 'time', unit: str = '',
              scale: str = 'linear', shape: str = 'curve', k: int = DEFAULT_CURVE_DF,
              boundary: float = DEFAULT_BOUNDARY, group_vec: list[str] | None = None,
              interaction: bool = False, baseline: str | None = None,
              pairing_vec: list[str] | None = None, covariates: pd.DataFrame | None = None,
              pval_threshold: float = 0.05, min_r2: float = DEFAULT_MIN_R2,
              call: str = 'either', grid_points: int = 50,
              feature_meta: pd.DataFrame | None = None) -> TrendResult:
    """
    Trend of every feature with a numeric variable.

    data_df    : rows = samples, columns = features, values on the scale
                 tested (log2 frequencies, log2 odds, transformed MFIs);
                 NaN = not measured in that sample.
    time       : the variable for each row of data_df (NaN = no value; the
                 sample is left out with a note).
    variable, unit: names for notes and settings (unit e.g. 'weeks').
    scale      : 'linear' or 'log2' (log2(t + 1)).
    shape, k, boundary: see the module docstring; k is capped by curve_df.
    group_vec  : group of each row; None = one group.
    interaction: test whether the trend differs between the two groups.
    baseline   : reference group (default: the first group in row order).
    pairing_vec: pairing level of each row (e.g. donor); see module
                 docstring for when it is kept.
    covariates : adjustment covariates indexed by sample id (covariate
                 frame). A column named like ``variable`` is left out;
                 samples missing a covariate value are left out with a note.
    pval_threshold, min_r2, call: see trend_significance().
    feature_meta: optional columns merged onto the table by 'feature'.

    Table columns: feature, n_obs, AveExpr, slope, slope.SE, slope.t,
    slope.P.Value, slope.df, R2.line, F, df1, df2, P.Value, R2, range,
    net.change, at.max, at.min, then trend_significance()'s columns.
    """
    d = trend_design(list(data_df.index), time, variable=variable, scale=scale, shape=shape,
                     k=k, boundary=boundary, group_vec=group_vec, interaction=interaction,
                     baseline=baseline, pairing_vec=pairing_vec, covariates=covariates)
    notes = list(d.notes)
    samples, rows, n, t, S, sh = list(data_df.index), d.rows, d.rows.size, d.t, d.S, d.shape
    order, base, pairing, group_cols = d.order, d.base, d.pairing, d.group_cols
    X_s, cols_s, X_l, cols_l = d.X_s, d.cols_s, d.X_l, d.cols_l
    Y = data_df.iloc[rows].to_numpy(dtype=float).T
    if Y.shape[0] < 3:
        raise RuntimeError("A trend test needs at least three features.")
    try:
        f_shape = ms.f_test(Y, X_s, cols_s)
        f_line = ms.f_test(Y, X_l, cols_l)
    except ValueError as exc:
        raise RuntimeError(f"{exc} Too few features have enough observed samples to fit.") from exc
    slope_fit = f_line.fit
    j = cols_l[0]
    slope = slope_fit.coefficients[:, j]
    slope_se = slope_fit.stdev_unscaled[:, j] * np.sqrt(slope_fit.s2_post)

    grid = (np.unique(t) if shape == 'points'
            else np.linspace(t.min(), t.max(), int(grid_points)))
    beta = f_shape.fit.coefficients[:, cols_s]
    G = sh(_transform(grid, scale))
    curve_part = np.nan_to_num(beta) @ G.T
    if interaction:
        offset = f_shape.fit.coefficients[:, group_cols[order[1]]]
    else:
        with np.errstate(invalid='ignore'):
            offset = np.nanmean(Y - np.nan_to_num(beta) @ S.T, axis=1)
    curves = offset[:, None] + curve_part
    curves[~np.isfinite(f_shape.F)] = np.nan
    curve_se = _curve_se(Y, X_s, cols_s, G, f_shape.fit.s2_post,
                         group_cols[order[1]] if interaction else None)
    curve_se[~np.isfinite(curves)] = np.nan
    group_curves = group_curve_se = None
    if interaction and pairing is None:
        k_cols = len(cols_s)
        second = np.array([g == order[1] for g in d.groups])
        group_curves, group_curve_se = _group_curves(
            Y, X_s, [c - k_cols for c in cols_s], cols_s, S, G, second,
            f_shape.fit.coefficients, f_shape.fit.s2_post)
        unfitted = ~np.isfinite(f_shape.F)
        group_curves[:, unfitted] = np.nan
        group_curve_se[:, unfitted] = np.nan
    finite = np.isfinite(curves).all(axis=1)
    at_max = np.full(curves.shape[0], np.nan)
    at_min = np.full(curves.shape[0], np.nan)
    at_max[finite] = grid[np.argmax(curves[finite], axis=1)]
    at_min[finite] = grid[np.argmin(curves[finite], axis=1)]

    table = pd.DataFrame({
        'feature': list(data_df.columns),
        'n_obs': np.isfinite(Y).sum(axis=1),
        'AveExpr': f_shape.fit.amean,
        'slope': slope,
        'slope.SE': slope_se,
        'slope.t': slope_fit.t[:, j],
        'slope.P.Value': slope_fit.p_value[:, j],
        'slope.df': slope_fit.df_total,
        'R2.line': f_line.partial_r2,
        'F': f_shape.F,
        'df1': f_shape.df1,
        'df2': f_shape.df2,
        'P.Value': f_shape.p_value,
        'R2': f_shape.partial_r2,
        'range': curves.max(axis=1) - curves.min(axis=1),
        'net.change': curves[:, -1] - curves[:, 0],
        'at.max': at_max,
        'at.min': at_min,
    })
    if feature_meta is not None:
        table = table.merge(feature_meta, on='feature', how='left', sort=False)
    table = trend_significance(table, pval_threshold, min_r2, call)
    settings = {'variable': variable, 'unit': unit, 'scale': scale, 'shape': shape,
                'df': int(sh.k), 'boundary': boundary if shape == 'curve' else 0.0,
                'knots': [float(v) for v in getattr(getattr(sh, 'spline', None), 'knots', [])],
                'groups': order, 'baseline': base, 'interaction': bool(interaction),
                'paired': pairing is not None, 'n_samples': int(n),
                'samples': [samples[i] for i in rows],
                'pval_threshold': pval_threshold, 'min_r2': min_r2, 'call': call}
    log.info("trend test (%s, %s df%s): %d features, %d samples, %d called",
             shape, sh.k, ", interaction" if interaction else "", len(table), n,
             int(table['significant'].sum()))
    return TrendResult(table=table, grid=grid, curves=curves, notes=notes, settings=settings,
                       curve_se=curve_se, group_curves=group_curves,
                       group_curve_se=group_curve_se)


def trend_significance(table: pd.DataFrame, pval_threshold: float = 0.05,
                       min_r2: float = DEFAULT_MIN_R2, call: str = 'either') -> pd.DataFrame:
    """
    FDR and the call of a trend table, without refitting. Returns a copy.

    Adds adj.P.Val (BH of the curve test), slope.adj.P.Val (BH of the slope
    test), called.by ('curve', 'slope', 'both' or ''), significant,
    curve.only (the curve but not the slope is significant: a rise and
    fall, or a change that levels off) and direction ('up'/'down', the sign
    of the net change; the slope's sign when the net change is zero).

    call 'either': curve or slope, each at FDR pval_threshold / 2;
    'curve' or 'slope': that test at pval_threshold. A test calls a
    feature only when its partial R² (R2 for the curve, R2.line for the
    slope) is at least ``min_r2``.
    """
    if call not in CALLS:
        raise ValueError(f"call must be one of {CALLS}, got {call!r}")
    df = table.copy()
    df['adj.P.Val'] = ms.bh_adjust(df['P.Value'].to_numpy(float))
    df['slope.adj.P.Val'] = ms.bh_adjust(df['slope.P.Value'].to_numpy(float))
    alpha = pval_threshold / 2.0 if call == 'either' else pval_threshold
    curve_hit = (df['adj.P.Val'] <= alpha) & (df['R2'] >= min_r2)
    slope_hit = (df['slope.adj.P.Val'] <= alpha) & (df['R2.line'] >= min_r2)
    if call == 'curve':
        slope_hit = slope_hit & False
    elif call == 'slope':
        curve_hit = curve_hit & False
    df['called.by'] = np.select([curve_hit & slope_hit, curve_hit, slope_hit],
                                ['both', 'curve', 'slope'], default='')
    df['significant'] = curve_hit | slope_hit
    df['curve.only'] = ((df['adj.P.Val'] <= alpha) & (df['R2'] >= min_r2)
                        & ~(df['slope.adj.P.Val'] <= alpha))
    sign = np.sign(df['net.change'].to_numpy(float))
    sign = np.where(sign == 0, np.sign(df['slope'].to_numpy(float)), sign)
    df['direction'] = np.where(sign > 0, 'up', np.where(sign < 0, 'down', ''))
    return df


def design_check(time, covariates: pd.DataFrame | None = None,
                 extra: dict | None = None, limit: float = DESIGN_CHECK_LIMIT) -> list[dict]:
    """
    How closely the trend variable follows each other variable over the
    same samples, so a trend that is really batch, tissue or sample quality
    can be spotted before testing.

    time       : the variable for each sample (NaN = no value).
    covariates : other covariates, one row per sample in the same order;
                 numeric columns are compared by Pearson's r, the others
                 (categories) by the correlation ratio η, the square root
                 of the share of the variable's variance that lies between
                 the categories. Both run from 0 (unrelated) to 1 (the
                 variable is fixed by the other); r keeps its sign.
    extra      : further per-sample quantities {name: values} (e.g. events
                 in the sample), compared by r.

    Returns one record per variable with at least three shared values:
    {'name', 'measure' ('r' or 'η'), 'value', 'n', 'warn' (|value| ≥ limit)}.
    """
    t = np.asarray(pd.to_numeric(pd.Series(list(time)), errors='coerce'), dtype=float)
    columns: list[tuple[str, object]] = []
    if covariates is not None:
        columns += [(name, covariates[name].tolist()) for name in covariates.columns]
    columns += list((extra or {}).items())
    out = []
    for name, values in columns:
        values = list(values)
        numeric = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(float)
        texts = [None if ds.is_missing_value(v) else str(v).strip() for v in values]
        filled = sum(v is not None for v in texts)
        is_number = filled > 0 and int(np.isfinite(numeric).sum()) == filled
        if is_number:
            ok = np.isfinite(t) & np.isfinite(numeric)
            if ok.sum() < 3 or np.std(t[ok]) == 0 or np.std(numeric[ok]) == 0:
                continue
            value, measure = float(np.corrcoef(t[ok], numeric[ok])[0, 1]), 'r'
        else:
            ok = np.isfinite(t) & np.array([v is not None for v in texts])
            if ok.sum() < 3:
                continue
            tt = t[ok]
            labels = np.array([v for v, keep in zip(texts, ok) if keep], dtype=object)
            total = float(np.sum((tt - tt.mean()) ** 2))
            if total == 0 or len(set(labels)) < 2:
                continue
            between = sum(float((labels == lv).sum()) * (tt[labels == lv].mean() - tt.mean()) ** 2
                          for lv in set(labels))
            value, measure = float(np.sqrt(between / total)), 'η'
        out.append({'name': name, 'measure': measure, 'value': value, 'n': int(ok.sum()),
                    'warn': abs(value) >= limit})
    return out


_TABLE_NUMERIC = ('n_obs', 'AveExpr', 'slope', 'slope.SE', 'slope.t', 'slope.P.Value',
                  'slope.df', 'R2.line', 'F', 'df1', 'df2', 'P.Value', 'R2', 'range',
                  'net.change', 'at.max', 'at.min', 'adj.P.Val', 'slope.adj.P.Val')


def _plain(value):
    """JSON-safe copy: numpy scalars as Python numbers, non-finite as None."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def result_to_jsonable(result: TrendResult) -> dict:
    """A TrendResult as JSON-safe values, for saving."""
    split = result.table.to_dict(orient='split')
    return _plain({'table': {'columns': split['columns'], 'data': split['data']},
                   'grid': result.grid, 'curves': result.curves,
                   'curve_se': result.curve_se, 'group_curves': result.group_curves,
                   'group_curve_se': result.group_curve_se,
                   'notes': list(result.notes), 'settings': result.settings})


def result_from_jsonable(payload: dict) -> TrendResult:
    """The TrendResult that result_to_jsonable saved."""
    table = pd.DataFrame(payload['table']['data'], columns=payload['table']['columns'])
    for col in table.columns:
        if col in _TABLE_NUMERIC:
            table[col] = pd.to_numeric(table[col], errors='coerce').astype(float)
    def matrix(key):
        values = np.array([[np.nan if v is None else v for v in row]
                           for row in payload.get(key) or []], dtype=float)
        return values.reshape(len(table), -1) if values.size else None

    def per_group(key):
        values = np.array([[[np.nan if v is None else v for v in row] for row in block]
                           for block in payload.get(key) or []], dtype=float)
        return values.reshape(-1, len(table), values.shape[-1]) if values.size else None

    curves = matrix('curves')
    return TrendResult(table=table, grid=np.asarray(payload.get('grid') or [], dtype=float),
                       curves=curves if curves is not None else np.full((len(table), 0), np.nan),
                       notes=list(payload.get('notes') or []),
                       settings=dict(payload.get('settings') or {}),
                       curve_se=matrix('curve_se'), group_curves=per_group('group_curves'),
                       group_curve_se=per_group('group_curve_se'))


# ---------------------------------------------------------------------------
# Composition over the variable
# ---------------------------------------------------------------------------

def merge_small_parts(counts: pd.DataFrame, keep: int, other: str = 'Other') -> pd.DataFrame:
    """
    *counts* (samples × parts) with all but the *keep* parts of largest
    mean share added together into one column named *other* (added to an
    existing column of that name). Parts keep their order.
    """
    if counts.shape[1] <= keep + 1:
        return counts
    totals = counts.sum(axis=1).replace(0, np.nan)
    share = counts.div(totals, axis=0).mean(axis=0)
    big = set(share.drop(labels=[other], errors='ignore').sort_values(ascending=False)
              .index[:keep])
    small = [c for c in counts.columns if c not in big]
    out = counts[[c for c in counts.columns if c in big]].copy()
    out[other] = counts[small].sum(axis=1)
    return out


def fit_composition(counts: pd.DataFrame, time, *, variable: str = 'time', unit: str = '',
                    scale: str = 'linear', shape: str = 'curve', k: int = DEFAULT_CURVE_DF,
                    boundary: float = DEFAULT_BOUNDARY, group_vec: list[str] | None = None,
                    covariates: pd.DataFrame | None = None, grid_points: int = 50) -> dict:
    """
    How the make-up of a set of populations changes with the variable,
    with fitted shares that add up to 100% at every value.

    In plain terms: the populations that together make up a parent (one
    gate's populations plus the rest of the parent, or a run's clusters)
    are fitted together, so the curves can be stacked. Each population is
    compared with the average of all of them on a log scale, the same
    smooth curve as the trend test is fitted to each, and the fitted values
    are turned back into shares. The result is the make-up of a typical
    sample at each value of the variable; it never goes below 0% or above
    100%.

    Maths: centred log-ratio transform (Aitchison 1986) of the counts with
    a pseudocount of 0.5, clr_j = ln(c_j + 0.5) − mean over parts; each
    part's clr fitted by least squares on the trend design (``trend_design``:
    the same shape, scale, df and boundary knots; groups and covariates as
    adjustment; no pairing); the fitted clr on a grid turned into shares by
    softmax, exp(f_j) / Σ exp(f). Because the clr of a sample sums to 0
    over parts and least squares is linear, the fitted curves also sum to
    0, and the shares are the closed (normalised) geometric-mean
    composition. The trend tests themselves stay per feature.

    counts : samples × parts, event counts (rows with a missing or zero
             total are left out with a note).
    time   : the variable for each row of *counts*.
    Returns {'parts', 'grid', 'shares' (parts × grid, summing to 1),
    'samples', 'time', 'observed' (samples × parts shares), 'groups',
    'notes', 'settings'}.
    """
    counts = counts.astype(float)
    parts = [str(c) for c in counts.columns]
    if len(parts) < 2:
        raise RuntimeError("A composition needs at least two populations.")
    notes: list[str] = []
    total = counts.sum(axis=1, min_count=len(parts))
    usable = (total > 0).to_numpy() & counts.notna().all(axis=1).to_numpy()
    if (~usable).any():
        notes.append(f"{int((~usable).sum())} sample(s) without counts for every population "
                     "were left out.")
    counts = counts[usable]
    t_all = np.asarray(list(time), dtype=float)[usable]
    groups_all = list(np.asarray(group_vec, dtype=object)[usable]) if group_vec is not None else None
    log_c = np.log(counts.to_numpy(float) + COMPOSITION_PSEUDOCOUNT)
    clr = log_c - log_c.mean(axis=1, keepdims=True)
    d = trend_design(list(counts.index), t_all, variable=variable, scale=scale, shape=shape, k=k,
                     boundary=boundary, group_vec=groups_all, covariates=covariates)
    notes.extend(d.notes)
    Y = clr[d.rows]
    beta = np.linalg.lstsq(d.X_s, Y, rcond=None)[0]
    b = beta[d.cols_s]
    t = d.t
    grid = (np.unique(t) if shape == 'points'
            else np.linspace(t.min(), t.max(), int(grid_points)))
    G = d.shape(_transform(grid, scale))
    level = (Y - d.S @ b).mean(axis=0)
    fitted = level[None, :] + G @ b
    e = np.exp(fitted - fitted.max(axis=1, keepdims=True))
    shares = (e / e.sum(axis=1, keepdims=True)).T
    observed = counts.to_numpy(float)[d.rows]
    observed = observed / observed.sum(axis=1, keepdims=True)
    used = [counts.index[i] for i in d.rows]
    return {'parts': parts, 'grid': grid, 'shares': shares, 'samples': used, 'time': t,
            'observed': observed, 'groups': list(d.groups), 'notes': notes,
            'settings': {'variable': variable, 'unit': unit, 'scale': scale, 'shape': shape,
                         'df': int(d.shape.k), 'n_samples': len(used)}}


def composition_panels(counts: pd.DataFrame, time: dict, groups: dict, settings: dict, *,
                       variable: str = '', unit: str = '') -> list[dict]:
    """
    fit_composition with the options of a trend run (*settings*: a
    TrendResult's settings): its scale, shape, df and boundary knots, one
    panel per group for a group × trend run, otherwise one panel with the
    groups as adjustment when there are several.

    counts : samples × parts (the trend run's samples; others are ignored).
    time, groups : {sample: value}, {sample: group}.
    Returns [{'label', **fit_composition result}].
    """
    used = [s for s in (settings.get('samples') or list(counts.index)) if s in counts.index]
    counts = counts.loc[used]
    order = list(settings.get('groups') or dict.fromkeys(groups.get(s, '') for s in used))
    kw = dict(variable=variable or settings.get('variable', 'time'),
              unit=unit or settings.get('unit', ''), scale=settings.get('scale', 'linear'),
              shape=settings.get('shape', 'curve'), k=int(settings.get('df') or DEFAULT_CURVE_DF),
              boundary=float(settings.get('boundary') or DEFAULT_BOUNDARY))
    times = pd.to_numeric(pd.Series([time.get(s) for s in used]), errors='coerce').to_numpy(float)
    if settings.get('interaction'):
        panels = []
        for g in order:
            rows = [i for i, s in enumerate(used) if groups.get(s) == g]
            panels.append(dict(fit_composition(counts.iloc[rows], times[rows], **kw), label=g))
        return panels
    labels = [groups.get(s, '') for s in used]
    several = len(set(labels)) > 1
    comp = fit_composition(counts, times, group_vec=labels if several else None, **kw)
    return [dict(comp, label=', '.join(str(g) for g in order))]


# ---------------------------------------------------------------------------
# Shape patterns
# ---------------------------------------------------------------------------

PATTERN_K_RANGE = (2, 8)
PATTERN_STABILITY = 0.75
PATTERN_SUBSAMPLES = 20
PATTERN_SUBSAMPLE_FRACTION = 0.8
PATTERN_PEAK_SHARE = 0.35        # interior peak or dip above both ends by this share of the range
PATTERN_EDGE = 0.15              # a peak within this share of either end counts as an end
PATTERN_LEVEL_SHARE = 0.75       # share of the net change in one half: "then levels off" / "late"


def zscore_curves(curves) -> np.ndarray:
    """Each curve (row) scaled to mean 0 and SD 1 across the grid; a flat
    curve becomes zeros."""
    c = np.asarray(curves, dtype=float)
    sd = c.std(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(sd > 0, (c - c.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1.0),
                        0.0)


def cluster_curves(curves, k: int) -> np.ndarray:
    """
    Pattern of each curve (0 to k − 1): Ward's hierarchical clustering of
    the z-scored curves. On z-scored curves the squared Euclidean distance
    is 2·m·(1 − r) for m grid points, so curves group by correlation, that
    is by shape, whatever the size of the change.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    z = zscore_curves(curves)
    n = z.shape[0]
    k = int(max(1, min(int(k), n)))
    if n < 2 or k == 1:
        return np.zeros(n, dtype=int)
    labels = fcluster(linkage(z, method='ward'), t=k, criterion='maxclust')
    return labels.astype(int) - 1


def pattern_name(mean_curve, grid, unit: str = '') -> str:
    """
    Plain name of a pattern from its mean z-scored curve: 'peaks at 27
    weeks' or 'lowest at 24 weeks' for an interior maximum or minimum well
    above or below both ends; otherwise 'rises' or 'falls', with 'then
    levels off' when most of the change is in the first half of the range
    and 'late' when most is in the second half.
    """
    m = np.asarray(mean_curve, dtype=float)
    grid = np.asarray(grid, dtype=float)
    span = float(m.max() - m.min())
    if not np.isfinite(span) or span <= 0:
        return 'flat'
    suffix = f" {unit}" if unit else ''
    n = m.size
    edge = max(1, int(round(PATTERN_EDGE * (n - 1))))
    i_max, i_min = int(np.argmax(m)), int(np.argmin(m))
    ends_hi, ends_lo = max(m[0], m[-1]), min(m[0], m[-1])
    if edge <= i_max <= n - 1 - edge and m[i_max] - ends_hi >= PATTERN_PEAK_SHARE * span:
        return f"peaks at {grid[i_max]:.3g}{suffix}"
    if edge <= i_min <= n - 1 - edge and ends_lo - m[i_min] >= PATTERN_PEAK_SHARE * span:
        return f"lowest at {grid[i_min]:.3g}{suffix}"
    net = float(m[-1] - m[0])
    word = 'rises' if net >= 0 else 'falls'
    if net == 0:
        return word
    early = float(m[n // 2] - m[0]) / net
    if early >= PATTERN_LEVEL_SHARE:
        return f"{word} then levels off"
    if early <= 1.0 - PATTERN_LEVEL_SHARE:
        return f"{word} late"
    return word


def assign_patterns(result: 'TrendResult', k: int | None = None) -> 'TrendResult':
    """
    *result* with the called features grouped into shape patterns: the
    table gains 'pattern' (the pattern's name, '' when not called) and
    settings['patterns'] lists each pattern ({'name', 'n', 'mean'}: the
    mean z-scored curve on the grid), ordered by where the mean curve
    peaks. k defaults to settings['pattern_k'] (or 4); it is capped by the
    number of called features. Returns a new TrendResult.
    """
    from dataclasses import replace

    settings = dict(result.settings)
    k = int(k if k is not None else settings.get('pattern_k') or 4)
    settings['pattern_k'] = k
    table = result.table.copy()
    table['pattern'] = ''
    called = np.flatnonzero(table['significant'].to_numpy(bool)
                            & np.isfinite(result.curves).all(axis=1)) if result.curves.size \
        else np.zeros(0, int)
    patterns = []
    if called.size >= 2:
        z = zscore_curves(result.curves[called])
        labels = cluster_curves(result.curves[called], k)
        means = [z[labels == c].mean(axis=0) for c in range(labels.max() + 1)]
        order = sorted(range(len(means)), key=lambda c: (int(np.argmax(means[c])),
                                                          int(np.argmin(means[c]))))
        names: dict = {}
        unit = settings.get('unit', '')
        for c in order:
            name = pattern_name(means[c], result.grid, unit)
            names[name] = names.get(name, 0) + 1
            if names[name] > 1:
                name = f"{name} ({names[name]})"
            members = called[labels == c]
            table.iloc[members, table.columns.get_loc('pattern')] = name
            patterns.append({'name': name, 'n': int(members.size),
                             'mean': [float(v) for v in means[c]]})
    settings['patterns'] = patterns
    return replace(result, table=table, settings=settings)


def recall(result: 'TrendResult', pval_threshold: float, min_r2: float, call: str,
           n_patterns: int = 0) -> 'TrendResult':
    """*result* with its calls redone for new thresholds (trend_significance)
    and its features regrouped into *n_patterns* patterns (0: the number
    chosen from stability when the run was made, see add_patterns)."""
    from dataclasses import replace

    settings = dict(result.settings, pval_threshold=pval_threshold, min_r2=min_r2, call=call)
    table = trend_significance(result.table.drop(columns=['pattern'], errors='ignore'),
                               pval_threshold, min_r2, call)
    auto = int(n_patterns) == 0
    k = int(n_patterns) or int(settings.get('pattern_k_auto') or settings.get('pattern_k') or 4)
    settings['pattern_auto'] = auto and bool(settings.get('pattern_k_auto'))
    return assign_patterns(replace(result, table=table, settings=settings), k)


def add_patterns(result: 'TrendResult', data_df: pd.DataFrame, time, n_patterns: int = 0,
                 **run_kwargs) -> 'TrendResult':
    """
    *result* (the run_trend of *data_df* and *time* with *run_kwargs*) with
    its called features grouped into *n_patterns* shape patterns. 0 chooses the
    number from pattern_stability (choose_pattern_k), which reruns the test
    on subsamples; the chosen k and the stability table are kept in
    settings ('pattern_k_auto', 'pattern_stability').
    """
    from dataclasses import replace

    settings = dict(result.settings)
    n_called = int(result.table['significant'].sum())
    k = int(n_patterns)
    if k == 0:
        auto = PATTERN_K_RANGE[0]
        if n_called > PATTERN_K_RANGE[0] + 1:
            ks = range(PATTERN_K_RANGE[0], min(PATTERN_K_RANGE[1], n_called - 1) + 1)
            run_kwargs = {key: v for key, v in run_kwargs.items() if key != 'feature_meta'}
            stability = pattern_stability(data_df, time, ks=ks, result=result, **run_kwargs)
            auto = choose_pattern_k(stability)
            settings['pattern_stability'] = stability.to_dict(orient='records')
        settings['pattern_k_auto'] = int(auto)
        settings['pattern_auto'] = True
        k = auto
    else:
        settings['pattern_auto'] = False
    return assign_patterns(replace(result, settings=settings), int(k))


def _adjusted_rand(a, b) -> float:
    """Adjusted Rand index of two labellings (1 = identical groupings,
    about 0 = no more alike than chance; Hubert & Arabie 1985)."""
    from scipy.special import comb
    a = np.asarray(a)
    b = np.asarray(b)
    _ua, ia = np.unique(a, return_inverse=True)
    _ub, ib = np.unique(b, return_inverse=True)
    table = np.zeros((ia.max() + 1, ib.max() + 1))
    np.add.at(table, (ia, ib), 1)
    sum_cells = comb(table, 2).sum()
    sum_a = comb(table.sum(axis=1), 2).sum()
    sum_b = comb(table.sum(axis=0), 2).sum()
    expected = sum_a * sum_b / comb(a.size, 2) if a.size > 1 else 0.0
    top = 0.5 * (sum_a + sum_b) - expected
    return float((sum_cells - expected) / top) if top > 0 else 1.0


def pattern_stability(data_df: pd.DataFrame, time, *, ks=None,
                      n_sub: int = PATTERN_SUBSAMPLES,
                      fraction: float = PATTERN_SUBSAMPLE_FRACTION, seed: int = 0,
                      result: 'TrendResult | None' = None, **run_kwargs) -> pd.DataFrame:
    """
    How reproducibly the called features fall into the same patterns when
    the study is repeated on part of its samples, for each number of
    patterns k.

    Each of *n_sub* subsamples keeps *fraction* of the samples (whole
    pairing levels when samples are paired), reruns run_trend with the same
    settings, and groups the features called in both the subsample and the
    full data into k patterns; the agreement with the full data's grouping
    of those features is the adjusted Rand index. Returns one row per k:
    k, ari (mean), ari_sd, n (subsamples that could be compared).

    run_kwargs are those of run_trend (group_vec and pairing_vec per row of
    data_df); *result* is the full-data run (made here when not given).
    """
    ks = list(ks if ks is not None else range(PATTERN_K_RANGE[0], PATTERN_K_RANGE[1] + 1))
    full = result if result is not None else run_trend(data_df, time, **run_kwargs)
    samples = list(data_df.index)
    time = list(time)
    units = run_kwargs.get('pairing_vec') or samples
    unit_ids = list(dict.fromkeys(str(u) for u in units))
    rng = np.random.default_rng(seed)
    called_full = full.table['feature'][full.table['significant']].astype(str).tolist()
    curve_of = dict(zip(full.table['feature'].astype(str), full.curves))
    scores = {k: [] for k in ks}
    for _i in range(int(n_sub)):
        chosen = set(rng.choice(unit_ids, size=max(2, int(round(fraction * len(unit_ids)))),
                                replace=False))
        keep = [i for i, u in enumerate(units) if str(u) in chosen]
        kwargs = dict(run_kwargs)
        for key in ('group_vec', 'pairing_vec'):
            if kwargs.get(key) is not None:
                kwargs[key] = [kwargs[key][i] for i in keep]
        try:
            part = run_trend(data_df.iloc[keep], [time[i] for i in keep], **kwargs)
        except RuntimeError:
            continue
        sub_called = part.table['significant'].to_numpy(bool)
        sub_curve = dict(zip(part.table['feature'].astype(str)[sub_called],
                             part.curves[sub_called]))
        both = [f for f in called_full if f in sub_curve and np.isfinite(curve_of[f]).all()
                and np.isfinite(sub_curve[f]).all()]
        for k in ks:
            if len(both) <= k:
                continue
            a = cluster_curves(np.array([curve_of[f] for f in both]), k)
            b = cluster_curves(np.array([sub_curve[f] for f in both]), k)
            scores[k].append(_adjusted_rand(a, b))
    return pd.DataFrame([{'k': k, 'ari': float(np.mean(v)) if v else np.nan,
                          'ari_sd': float(np.std(v)) if v else np.nan, 'n': len(v)}
                         for k, v in scores.items()])


def choose_pattern_k(stability: pd.DataFrame, threshold: float = PATTERN_STABILITY) -> int:
    """The largest number of patterns whose mean agreement (adjusted Rand
    index) is at least *threshold*; the smallest k tried when none is."""
    ok = stability[stability['ari'] >= threshold]
    if len(ok):
        return int(ok['k'].max())
    return int(stability['k'].min()) if len(stability) else PATTERN_K_RANGE[0]

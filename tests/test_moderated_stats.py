"""
test_moderated_stats.py
-----------------------
Tests for controller_components/moderated_stats.py: which design columns
are estimable, and p-values worked out when first read.

Usage:
    pytest tests/test_moderated_stats.py -m numpy_only
"""

import numpy as np
import pytest
from scipy import stats

from honeychrome.controller_components import moderated_stats as ms


def _greedy(X):
    """estimable_columns by singular values only (the reference route)."""
    nonzero = np.flatnonzero(np.any(X != 0.0, axis=0))
    rank = np.linalg.matrix_rank(X[:, nonzero])
    kept = []
    for j in nonzero:
        if np.linalg.matrix_rank(X[:, kept + [int(j)]]) > len(kept):
            kept.append(int(j))
    estimable = np.array([np.linalg.matrix_rank(X[:, nonzero[nonzero != j]]) < rank
                          for j in kept])
    return np.asarray(kept, dtype=int), estimable


def _paired_design(n_pairs, rng):
    """Intercept, group, pair blocks (treatment coded) and a spline-like term."""
    n = 2 * n_pairs
    group = np.tile([0.0, 1.0], n_pairs)
    blocks = np.zeros((n, n_pairs - 1))
    for i in range(1, n_pairs):
        blocks[2 * i:2 * i + 2, i - 1] = 1.0
    age = np.repeat(rng.uniform(18, 37, n_pairs), 2)
    return np.column_stack([np.ones(n), group, blocks, group * (age - age.mean())])


@pytest.mark.numpy_only
def test_estimable_columns_match_the_singular_value_route():
    rng = np.random.default_rng(1)
    X = _paired_design(12, rng)
    for drop in ([], [5], [0, 1], [3, 4, 9], list(range(0, 24, 3))):
        keep = np.setdiff1d(np.arange(X.shape[0]), drop)
        Xo = X[keep]
        kept, est = ms.estimable_columns(Xo)
        ref_kept, ref_est = _greedy(Xo)
        np.testing.assert_array_equal(kept, ref_kept)
        np.testing.assert_array_equal(est, ref_est)
    collinear = np.column_stack([X, X[:, 1] + X[:, 2]])
    kept, est = ms.estimable_columns(collinear)
    ref_kept, ref_est = _greedy(collinear)
    np.testing.assert_array_equal(kept, ref_kept)
    np.testing.assert_array_equal(est, ref_est)
    assert collinear.shape[1] - 1 not in kept


@pytest.mark.numpy_only
def test_p_values_are_worked_out_when_read_and_can_be_given():
    rng = np.random.default_rng(2)
    X = np.column_stack([np.ones(20), np.repeat([0.0, 1.0], 10)])
    Y = rng.normal(size=(30, 20))
    Y[:5, 10:] += 2.0
    fit = ms.e_bayes(ms.lm_fit(Y, X))
    expected = 2.0 * stats.t.sf(np.abs(fit.t), fit.df_total[:, None])
    np.testing.assert_allclose(fit.p_value, expected, rtol=1e-12)
    given = ms.ModeratedFit(coefficients=fit.coefficients, stdev_unscaled=fit.stdev_unscaled,
                            sigma2=fit.sigma2, df_residual=fit.df_residual, amean=fit.amean,
                            n_obs=fit.n_obs, design=fit.design, df_prior=fit.df_prior,
                            s2_prior=fit.s2_prior, s2_post=fit.s2_post, df_total=fit.df_total,
                            t=fit.t, p_value=np.zeros_like(fit.t))
    assert not given.p_value.any()

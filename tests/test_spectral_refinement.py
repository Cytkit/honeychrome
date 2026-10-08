"""
test_spectral_refinement.py
---------------------------
Numerics of controller_components/spectral_refinement.py on synthetic
single-stained controls built from Aurora reference spectra, with per-cell
autofluorescence of varying shape and Poisson-like detector noise. The
first-pass ("cleaned") row of one fluorophore is contaminated with
autofluorescence, as an AF-rich cosine selection would leave it, and the
refinement must recover the true row and remove the residual spillover it
causes.

Usage:
    pytest tests/test_spectral_refinement.py -m numpy_only
    pytest tests/test_spectral_refinement.py -k benchmark -s
"""

import time

import numpy as np
import pytest

from honeychrome.controller_components import spectral_cleaning as sc
from honeychrome.controller_components import spectral_refinement as sr

from spectral_synthetic import (  # noqa: E402
    PANEL, CONTAMINATED, panel_spectra, simulate, contaminate, angle,
)

pytestmark = pytest.mark.numpy_only


@pytest.fixture(scope='module')
def experiment():
    rng = np.random.default_rng(7)
    spectra, _ = panel_spectra()
    k = PANEL.index(CONTAMINATED)
    control = simulate(spectra[k], 60_000, rng)
    unstained = simulate(None, 60_000, rng)
    return {'spectra': spectra, 'k': k, 'control': control, 'unstained': unstained,
            'af_basis': unstained.mean(axis=0)}


# ---------------------------------------------------------------------------
# huber_fit
# ---------------------------------------------------------------------------

def _huber_reference(x, y, k=1.345, max_iter=100, tol=1e-4):
    """Scalar transcription of fix_huber_slope_rcpp()."""
    w = np.ones_like(x)
    coef0 = coef1 = 0.0
    for _ in range(max_iter):
        sw = w.sum()
        xm, ym = (w * x).sum() / sw, (w * y).sum() / sw
        sxx = (w * (x - xm) ** 2).sum()
        slope = (w * (x - xm) * (y - ym)).sum() / sxx
        intercept = ym - slope * xm
        r = y - (intercept + slope * x)
        scale = max(np.median(np.abs(r - np.median(r))) / 0.6745, np.finfo(float).eps)
        w = np.minimum(1.0, k / np.maximum(np.abs(r / scale), np.finfo(float).eps))
        moved = max(abs(intercept - coef0), abs(slope - coef1)) < tol * max(abs(intercept), abs(slope), 1.0)
        coef0, coef1 = intercept, slope
        if moved:
            return np.array([coef0, coef1])
    return np.zeros(2)


def test_huber_matches_scalar_reference():
    rng = np.random.default_rng(1)
    x = rng.gamma(2.0, 1000.0, 3000)
    y = np.column_stack([0.05 * x + rng.normal(0, 50, x.size),
                         -0.02 * x + 10 + rng.standard_t(2, x.size) * 40,
                         rng.normal(0, 30, x.size)])
    coef, converged = sr.huber_fit(x, y)
    assert converged.all()
    for c in range(y.shape[1]):
        np.testing.assert_allclose(coef[:, c], _huber_reference(x, y[:, c]), rtol=1e-9, atol=1e-9)


def test_huber_resists_outliers_where_ols_does_not():
    rng = np.random.default_rng(2)
    x = rng.gamma(2.0, 1000.0, 5000)
    y = 0.03 * x + rng.normal(0, 20, x.size)
    bad = rng.random(x.size) < 0.1
    y[bad] += 0.5 * x[bad]
    slope_ols = np.polyfit(x, y, 1)[0]
    coef, converged = sr.huber_fit(x, y)
    assert converged
    assert abs(coef[1] - 0.03) < 0.25 * abs(slope_ols - 0.03)


def test_huber_multiple_predictors():
    rng = np.random.default_rng(3)
    x = rng.gamma(2.0, 1000.0, (4000, 3))
    true = np.array([[0.02, -0.01], [0.0, 0.05], [0.01, 0.0]])
    y = x @ true + 5.0 + rng.normal(0, 15, (4000, 2))
    coef, converged = sr.huber_fit(x, y)
    assert converged.all()
    np.testing.assert_allclose(coef[1:], true, atol=2e-3)


def test_huber_singular_design_returns_zero():
    coef, converged = sr.huber_fit(np.full(100, 3.0), np.arange(100.0))
    assert not converged
    np.testing.assert_array_equal(coef, 0.0)


# ---------------------------------------------------------------------------
# residual_spillover
# ---------------------------------------------------------------------------

def test_residual_spillover_keeps_bright_and_caps_bulk():
    rng = np.random.default_rng(4)
    n = 100_000
    src = np.where(rng.random(n) < 0.05, rng.gamma(3.0, 5000.0, n), rng.normal(0, 100, n))
    tgt = np.column_stack([0.08 * src + rng.normal(0, 100, n), rng.normal(0, 100, n)])
    fit = sr.residual_spillover(src, tgt, threshold_source=500.0, max_events=20_000)
    n_bright = int((src > 500.0).sum())
    assert fit['n'] == 20_000 and n_bright < 20_000
    np.testing.assert_allclose(fit['slope'], [0.08, 0.0], atol=3e-3)
    assert sr.residual_spillover(src[:100], tgt[:100]) is None


# ---------------------------------------------------------------------------
# extract_raw_signature and refine_row
# ---------------------------------------------------------------------------

def test_raw_signature_recovers_row_without_noise():
    spectra, _ = panel_spectra()
    row = spectra[PANEL.index('PE')]
    rng = np.random.default_rng(5)
    abundance = rng.gamma(2.0, 5000.0, 5000)
    raw = np.outer(abundance, row) + 200.0
    est = sr.extract_raw_signature(raw, raw @ row / (row @ row), row)
    assert est['stats']['deg_change'] < 1e-6
    assert est['stats']['intercept_rel'] < 0.05


def test_refine_recovers_contaminated_row(experiment):
    true = experiment['spectra'][experiment['k']]
    start = contaminate(true)
    result = sr.refine_row(experiment['control'], start)
    assert result.accepted
    assert angle(start, true) > 1.0
    assert angle(result.row, true) < 0.25 * angle(start, true)


def test_refine_gates_dim_dye_on_bright_af():
    """With dye variance comparable to AF variance the candidate is biased
    and must be rejected; once the dye dominates it is accepted."""
    spectra, _ = panel_spectra()
    true = spectra[PANEL.index(CONTAMINATED)]
    start = contaminate(true)
    dim = sr.refine_row(simulate(true, 60_000, np.random.default_rng(8), brightness=3e3), start)
    assert not dim.accepted
    np.testing.assert_allclose(dim.row, start / start.max())
    moderate = sr.refine_row(simulate(true, 60_000, np.random.default_rng(8), brightness=1e4), start)
    assert moderate.accepted
    assert angle(moderate.row, true) < 0.5 * angle(start, true)


def test_af_projection_regressor_would_bias_row(experiment):
    """Why refine_row has no AF regressor: the projection of each event
    onto the unstained mean b carries the dye's own loading c = row.b/|b|^2,
    and a joint fit on it returns row - c b."""
    true = experiment['spectra'][experiment['k']]
    raw, b = experiment['control'], experiment['af_basis']
    x = np.column_stack([np.ones(len(raw)), raw @ true / (true @ true), raw @ b / (b @ b)])
    slope = np.linalg.lstsq(x, raw, rcond=None)[0][1]
    c = (true @ b) / (b @ b)
    assert angle(slope, true - c * b) < 6.0
    assert angle(slope, true) > 20.0


def test_refine_leaves_correct_row_alone(experiment):
    true = experiment['spectra'][experiment['k']]
    result = sr.refine_row(experiment['control'], true)
    assert angle(result.row, true) < 0.5


def test_refine_rejects_large_change(experiment):
    true = experiment['spectra'][experiment['k']]
    start = contaminate(true, amount=0.6)
    result = sr.refine_row(experiment['control'], start)
    assert not result.accepted
    assert result.log[0]['reject'] == 'angle'
    np.testing.assert_allclose(result.row, start / start.max())


# ---------------------------------------------------------------------------
# crosstalk_check
# ---------------------------------------------------------------------------

def test_crosstalk_removed_by_refinement(experiment):
    spectra = experiment['spectra']
    k = experiment['k']
    start = contaminate(spectra[k])
    refined = sr.refine_row(experiment['control'], start).row

    before_panel = spectra.copy()
    before_panel[k] = start
    after_panel = spectra.copy()
    after_panel[k] = refined
    pool = sr.make_check_pool(experiment['control'], start, experiment['unstained'],
                              experiment['af_basis'])
    s = sr.RefineSettings()
    assert len(pool.raw) <= s.max_check_events + s.min_events
    before = sr.crosstalk_check(pool, before_panel, PANEL, CONTAMINATED)
    after = sr.crosstalk_check(pool, after_panel, PANEL, CONTAMINATED)

    worst_before = max(abs(r['slope']) for r in before)
    worst_after = max(abs(r['slope']) for r in after)
    assert len(before) == len(PANEL)
    assert before[-1]['target'] == sr.AF_TARGET
    assert worst_after < 0.25 * worst_before
    assert not any(r['flagged'] for r in after)


# ---------------------------------------------------------------------------
# Timing against the cleaning compute
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('n_events', [100_000, 400_000])
def test_benchmark_refine_vs_cleaning(n_events):
    rng = np.random.default_rng(11)
    spectra, _ = panel_spectra()
    k = PANEL.index(CONTAMINATED)
    control = simulate(spectra[k], n_events, rng)
    unstained = simulate(None, n_events, rng)
    scatter_pos = rng.normal(8e4, 1e4, (n_events, 2))
    scatter_neg = rng.normal(8e4, 1e4, (n_events, 2))
    af_basis = unstained.mean(axis=0)
    peak = int(np.argmax(spectra[k]))

    t0 = time.perf_counter()
    pos, _, _ = sc.exclude_saturated(control, 4e6)
    neg, _, _ = sc.exclude_saturated(unstained, 4e6)
    sel, _ = sc.cosine_filter(pos, np.median(neg, axis=0), peak_ch_idx=peak)
    sub, _ = sc.knn_scatter_match(pos[sel], scatter_pos[sel], neg, scatter_neg)
    first = np.clip(np.median(sub, axis=0), 0, None)
    first /= first.max()
    t_clean = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = sr.refine_row(pos, first)
    pool = sr.make_check_pool(pos, result.row, neg, af_basis)
    t_refine = time.perf_counter() - t0

    t0 = time.perf_counter()
    panel = spectra.copy()
    panel[k] = first
    sr.crosstalk_check(pool, panel, PANEL, CONTAMINATED)
    panel[k] = result.row
    sr.crosstalk_check(pool, panel, PANEL, CONTAMINATED)
    t_check = time.perf_counter() - t0

    print(f'\n{n_events} events/control: cleaning {t_clean * 1e3:.0f} ms, '
          f'row refinement {t_refine * 1e3:.0f} ms ({len(result.log)} passes), '
          f'panel check x2 {t_check * 1e3:.0f} ms')

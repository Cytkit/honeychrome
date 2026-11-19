"""
test_gls_unmix.py
-----------------
The NumPy path of M2pg (gls_unmix): pgls_solve against a plain per-event
solve, and unmix_m2pg recovering known abundances.

Usage:
    pytest tests/test_gls_unmix.py
"""

import numpy as np
import pytest

from honeychrome.controller_components import gls_unmix
from honeychrome.controller_components.noise_model import NoiseModel

F, D, B = 6, 16, 300


def _spectra(rng):
    s = np.abs(rng.normal(0, 1, (F, D))) ** 3
    return s / s.max(axis=1, keepdims=True)


def _reference_pgls(y, spectra, af_rows, w, fl_pen):
    """pgls_solve's documented result, one event at a time with an explicit design."""
    f = spectra.shape[0]
    x = np.empty((len(y), f))
    alpha = np.zeros(len(y))
    inv_diag = np.empty((len(y), f))

    def solve(i, use_af):
        design = np.vstack([spectra, af_rows[i]]) if use_af else spectra
        normal = (design * w[i]) @ design.T
        if fl_pen is not None:
            normal[np.arange(f), np.arange(f)] += fl_pen[i]
        normal[np.arange(f), np.arange(f)] += 1e-12 * np.trace(normal[:f, :f]) / f
        inv = np.linalg.inv(normal)
        return inv @ ((design * w[i]) @ y[i]), np.diag(inv)[:f]

    for i in range(len(y)):
        theta, diag = solve(i, af_rows is not None)
        if af_rows is not None and theta[f] < 0:
            theta, diag = solve(i, False)
            theta = np.append(theta, 0.0)
        x[i] = theta[:f]
        alpha[i] = theta[f] if af_rows is not None else 0.0
        inv_diag[i] = diag
    return x, alpha, inv_diag


@pytest.mark.parametrize('with_af', [False, True])
@pytest.mark.parametrize('with_pen', [False, True])
def test_pgls_solve_matches_per_event_solve(with_af, with_pen):
    rng = np.random.default_rng(1)
    spectra = _spectra(rng)
    af_rows = np.abs(rng.normal(1, 0.3, (B, D))) if with_af else None
    y = np.abs(rng.lognormal(5, 1.5, (B, F))) @ spectra + rng.normal(0, 30, (B, D))
    if with_af:
        # half the events carry AF, the rest should come out negative and be refitted without it
        y[: B // 2] += rng.uniform(200, 2000, (B // 2, 1)) * af_rows[: B // 2]
    w = 1.0 / rng.uniform(1e2, 1e4, (B, D))
    pen = rng.uniform(0, 1e-3, (B, F)) if with_pen else None

    out = gls_unmix.pgls_solve(y, spectra, af_rows, w, pen)
    x, alpha, inv_diag = _reference_pgls(y, spectra, af_rows, w, pen)

    np.testing.assert_allclose(out['x'], x, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(out['alpha'], alpha, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(out['inv_diag'], inv_diag, rtol=1e-8)
    if with_af:
        assert (alpha == 0).any() and (alpha > 0).any()


def test_unmix_m2pg_numpy_path_recovers_noise_free_abundances():
    rng = np.random.default_rng(2)
    spectra = _spectra(rng)
    abundances = rng.uniform(0, 5000, (2500, F))
    raw = abundances @ spectra
    noise = NoiseModel(detectors=list(range(D)), sigma0_sq=np.full(D, 400.0), g=np.full(D, 0.5),
                       c=np.full(D, 1e-4), flags=np.zeros(D, dtype=int))

    out = gls_unmix.unmix_m2pg(raw, spectra, noise, prior_lambda=0.0, use_kernel=False)

    np.testing.assert_allclose(out['unmixed'], abundances, rtol=1e-6, atol=1e-6)
    assert (out['af_idx'] == -1).all()

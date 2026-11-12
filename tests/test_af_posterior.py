"""
test_af_posterior.py
--------------------
Noise-aware AF abundance (af_posterior): the posterior's limits, the fast
and full noise variance, apply_af_unmixing with a posterior (chunk
invariance, unchanged variant choice), the stored priors and their
fingerprint, and M2pg with the AF fixed.

Usage:
    pytest tests/test_af_posterior.py
"""

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from honeychrome.controller_components import af_posterior as ap
from honeychrome.controller_components import gls_unmix as gu
from honeychrome.controller_components import noise_model as nm
from honeychrome.controller_components.autospectral_functions import (
    _joint_l2_library_terms,
    apply_af_unmixing,
    build_af_precomputed,
)
from honeychrome.controller_components.sample_loader import AFState

import af_synthetic as syn

SSC_COL = syn.RAW_PNN.index('SSC-A')
FL_COLS = list(range(3, 3 + syn.N_DET))


def _noise(d=syn.N_DET, c=0.0):
    return np.full(d, 400.0), np.full(d, 0.8), np.full(d, c)


def _prior(**kw):
    values = dict(intercept=600.0, slope=0.0, tau2=200.0 ** 2, delta=900.0, tau2_broad=1500.0 ** 2,
                  pi_broad=0.1, choice_shrinkage=1.0, k_typical=600.0, log_ssc_median=11.0, n_events=10_000,
                  fingerprint='f', source='s')
    values.update(kw)
    return ap.AFPrior(**values)


@pytest.fixture(scope='module')
def library():
    fl, af = syn.fluor_spectra(), syn.af_spectra()
    return fl, af, build_af_precomputed(fl, af)


@pytest.fixture(scope='module')
def events():
    raw = syn.raw_events(6000, seed=3)
    return raw, np.ascontiguousarray(raw[:, FL_COLS]), np.log(np.maximum(raw[:, SSC_COL], 1.0))


@pytest.mark.numpy_only
def test_posterior_follows_estimate_when_noise_is_small():
    k_hat = np.array([50.0, 400.0, 900.0, 3000.0])
    out = ap.posterior_mixture(k_hat, np.full(4, 1e-6), np.full(4, 11.0), _prior())
    assert np.allclose(out, k_hat, rtol=1e-6, atol=1e-3)


@pytest.mark.numpy_only
def test_posterior_is_prior_mean_when_noise_is_large():
    p = _prior()
    out = ap.posterior_mixture(np.array([-5000.0, 0.0, 8000.0]), np.full(3, 1e14), np.full(3, 11.0), p)
    means, sds = [p.intercept, p.intercept + p.delta], [np.sqrt(p.tau2), np.sqrt(p.tau2_broad)]
    weights = [1.0 - p.pi_broad, p.pi_broad]
    mass = [w * norm.cdf(m / s) for w, m, s in zip(weights, means, sds)]
    cond = [m + s * norm.pdf(-m / s) / norm.cdf(m / s) for m, s in zip(means, sds)]
    expected = sum(a * b for a, b in zip(mass, cond)) / sum(mass)
    assert np.allclose(out, expected, rtol=1e-3)


@pytest.mark.numpy_only
def test_posterior_is_positive_and_smooth_through_zero():
    k_hat = np.linspace(-3000.0, 3000.0, 601)
    out = ap.posterior_mixture(k_hat, np.full(k_hat.size, 300.0 ** 2), np.full(k_hat.size, 11.0), _prior())
    assert np.all(out > 0.0)
    assert np.all(np.diff(out) > 0.0)


@pytest.mark.numpy_only
def test_posterior_finite_far_below_zero():
    k_hat = np.array([-1e6, -1e4, -50.0, 0.0])
    p = _prior(tau2=1.0, tau2_broad=4.0)
    out = ap.posterior_mixture(k_hat, np.full(4, 1e-8), np.full(4, 11.0), p)
    assert np.all(np.isfinite(out)) and np.all(out >= 0.0)
    assert out[0] < 1e-3


@pytest.mark.numpy_only
def test_fast_variance_equals_full_form(library, events):
    fl, af, pre = library
    _raw, raw_fl, _ssc = events
    _w, k_denom, _c = _joint_l2_library_terms(pre, af)
    u_pos = np.maximum(raw_fl @ pre['P'].T, 0.0)
    rows = np.arange(len(raw_fl)) % len(af)
    s0, g, c = _noise()
    fast = ap.af_variance(u_pos, rows, pre, af, k_denom, s0, g, c, 500.0)
    full = ap.af_variance(u_pos, rows, pre, af, k_denom, s0, g, np.full_like(c, 1e-300), 500.0)
    assert np.allclose(fast, full, rtol=1e-10)


@pytest.mark.numpy_only
def test_unmixing_with_posterior(library, events):
    fl, af, pre = library
    _raw, raw_fl, log_ssc = events
    prior = ap.fit_af_prior(raw_fl, log_ssc, pre, af, *_noise(), fingerprint='f', source='s')
    assert ap.SHRINKAGE_RANGE[0] <= prior.choice_shrinkage <= ap.SHRINKAGE_RANGE[1]
    assert prior.tau2 > 0.0 and 0.0 < prior.pi_broad <= 0.5
    posterior = ap.AFPosterior(prior, *_noise(), ssc_name='SSC-A')
    legacy = apply_af_unmixing(raw_fl, pre, af)
    whole = apply_af_unmixing(raw_fl, pre, af, af_posterior=posterior, log_ssc=log_ssc)
    chunked = apply_af_unmixing(raw_fl, pre, af, chunk_size=37, af_posterior=posterior, log_ssc=log_ssc)
    assert np.array_equal(whole['af_idx'], legacy['af_idx'])
    assert np.all(whole['af_scale'] > 0.0)
    assert np.allclose(whole['af_scale'], chunked['af_scale'], rtol=1e-10, atol=1e-9)
    assert np.allclose(whole['unmixed'], chunked['unmixed'], rtol=1e-10, atol=1e-9)
    u = raw_fl @ pre['P'].T
    v = pre['v_library'].T[whole['af_idx'] - 1]
    assert np.allclose(whole['unmixed'], u - whole['af_scale'][:, None] * v)
    assert np.mean(legacy['af_scale'] == 0.0) > np.mean(whole['af_scale'] == 0.0)


def _context(tmp_path, raw):
    syn.write_raw_fcs(tmp_path / syn.AF_SAMPLE, raw)
    s0, g, c = _noise()
    return ap.AFNoiseContext(experiment_dir=str(tmp_path), cell_noise=(s0, g, c), bead_noise=(s0, g, c),
                             noise_fingerprint='n', bead_samples=frozenset(), references={},
                             pnn_raw=tuple(syn.RAW_PNN), whitelisted=True, fl_ids_raw=tuple(FL_COLS),
                             ssc_name='SSC-A')


def test_stored_prior_and_fingerprint(tmp_path, library, events, monkeypatch):
    fl, af, pre = library
    raw, _raw_fl, _ssc = events
    ctx = _context(tmp_path, raw)
    state = AFState(profile_names=('P',), precomputed=pre, spectra=af)
    first = ctx.posterior_for(state, syn.AF_SAMPLE)
    assert first is not None
    assert (tmp_path / 'AutoSpectral' / ap.PRIORS_FILE).exists()
    ap._memory.clear()

    def no_fit(*_a, **_k):
        raise AssertionError('refitted although stored')

    monkeypatch.setattr(ap.AFNoiseContext, '_fit', no_fit)
    again = ctx.posterior_for(state, syn.AF_SAMPLE)
    assert again.prior == first.prior
    monkeypatch.undo()

    path = tmp_path / syn.AF_SAMPLE
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    changed = ctx.posterior_for(state, syn.AF_SAMPLE)
    assert changed.fingerprint != first.fingerprint
    attached = ctx.attach(state, syn.AF_SAMPLE)
    assert attached.posterior is not None and state.posterior is None


def test_live_sample_uses_reference_prior(tmp_path, library, events):
    fl, af, pre = library
    raw, _raw_fl, _ssc = events
    ctx = _context(tmp_path, raw)
    ctx = ap.AFNoiseContext(**{**ctx.__dict__, 'references': {'P': syn.AF_SAMPLE}})
    syn.write_raw_fcs(tmp_path / syn.PLAIN_SAMPLE, raw[:100])
    state = AFState(profile_names=('P',), precomputed=pre, spectra=af)
    live = ctx.posterior_for(state, syn.PLAIN_SAMPLE, live=True)
    few = ctx.posterior_for(state, syn.PLAIN_SAMPLE)
    assert live is not None and live.prior.source == syn.AF_SAMPLE
    assert few is not None and few.prior.source == syn.AF_SAMPLE


@pytest.mark.numpy_only
@pytest.mark.parametrize('use_kernel', [False, True])
def test_m2pg_with_fixed_af(library, events, use_kernel):
    if use_kernel and not gu.kernel_available():
        pytest.skip('M2pg kernel not built')
    fl, af, pre = library
    _raw, raw_fl, log_ssc = events
    s0, g, c = _noise()
    model = nm.NoiseModel(detectors=list(syn.DET_PNN), sigma0_sq=s0, g=g, c=c,
                          flags=np.zeros(syn.N_DET, dtype=np.int64))
    prior = ap.fit_af_prior(raw_fl, log_ssc, pre, af, s0, g, c)
    r1 = apply_af_unmixing(raw_fl, pre, af, af_posterior=ap.AFPosterior(prior, s0, g, c), log_ssc=log_ssc)
    fixed = (r1['af_idx'] - 1, r1['af_scale'])
    out = gu.unmix_m2pg(raw_fl, fl, model, af, None, prior_lambda=0.0, use_kernel=use_kernel, fixed_af=fixed)
    assert np.array_equal(np.asarray(out['af_idx']), fixed[0])
    assert np.allclose(out['af_scale'], fixed[1])
    y = raw_fl - fixed[1][:, None] * af[fixed[0]]
    ols = y @ np.linalg.solve(fl @ fl.T, fl).T
    assert np.median(np.abs(out['unmixed'] - ols)) < 0.05 * np.median(np.abs(ols)) + 1.0


def test_m2pg_fixed_af_kernel_matches_numpy(library, events):
    if not gu.kernel_available():
        pytest.skip('M2pg kernel not built')
    fl, af, pre = library
    _raw, raw_fl, log_ssc = events
    s0, g, c = _noise()
    model = nm.NoiseModel(detectors=list(syn.DET_PNN), sigma0_sq=s0, g=g, c=c,
                          flags=np.zeros(syn.N_DET, dtype=np.int64))
    prior = ap.fit_af_prior(raw_fl, log_ssc, pre, af, s0, g, c)
    r1 = apply_af_unmixing(raw_fl, pre, af, af_posterior=ap.AFPosterior(prior, s0, g, c), log_ssc=log_ssc)
    fixed = (r1['af_idx'] - 1, r1['af_scale'])
    a = gu.unmix_m2pg(raw_fl, fl, model, af, None, use_kernel=False, fixed_af=fixed)
    b = gu.unmix_m2pg(raw_fl, fl, model, af, None, use_kernel=True, fixed_af=fixed)
    assert np.allclose(a['unmixed'], b['unmixed'], rtol=1e-6, atol=1e-6)

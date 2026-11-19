"""
test_af_shape_bend.py
---------------------
AF shape bend (af_shape_bend): learning on synthetic events whose AF varies
in shape (24 detectors, 5 fluorophores, an AF library of close variants as
a SOM-derived library is), the bend's effect on per-cell AF unmixing (only high-AF events
change, AF abundance and index unchanged, chunk invariance, lower error in
fluorophores that are truly 0), no change without shape variability, the
stored bend on the noise-aware context, and M2pg with the bend integrated out
of its noise covariance (GLS under V + E' T E, only high-AF unsaturated events
change, the compiled kernel against NumPy).

Usage:
    pytest tests/test_af_shape_bend.py
"""

import functools

import numpy as np
import pytest

from honeychrome.controller_components import af_posterior as ap
from honeychrome.controller_components import af_shape_bend as asb
from honeychrome.controller_components import functions as fn
from honeychrome.controller_components import gls_unmix as gu
from honeychrome.controller_components import noise_model as nm
from honeychrome.controller_components.autospectral_functions import (
    apply_af_unmixing, build_af_precomputed, build_runtime_envelope, make_envelope_entry,
)
from honeychrome.controller_components.sample_loader import AFState

import af_synthetic as syn

N_DET = 24
RAW_PNN = ['Time', 'FSC-A', 'SSC-A'] + [f'D{i + 1}-A' for i in range(N_DET)]
SSC_COL = RAW_PNN.index('SSC-A')
FL_COLS = list(range(3, 3 + N_DET))
NOISE = (np.full(N_DET, 400.0), np.full(N_DET, 0.8), np.zeros(N_DET))
BEND_SD = 0.06
CH = np.arange(N_DET)


def _bump(centre, width):
    return np.exp(-0.5 * ((CH - centre) / width) ** 2)


def fluor_spectra():
    s = np.array([_bump(c, 1.5) for c in (3, 8, 12, 16, 21)])
    return s / s.max(axis=1, keepdims=True)


def af_library():
    """Ten close variants of one broad AF spectrum."""
    rng = np.random.default_rng(0)
    base = _bump(6.0, 6.0) + 0.4 * _bump(14.0, 4.0)
    rows = [base * (1.0 + 0.08 * rng.normal(0.0, 1.0) * (CH - 12.0) / 12.0) for _ in range(10)]
    s = np.array(rows)
    return s / s.max(axis=1, keepdims=True)


def _true_shapes():
    s = np.array([_bump(18.0, 3.0), (CH - 10.0) / 10.0 * _bump(10.0, 5.0)])
    return s / np.linalg.norm(s, axis=1, keepdims=True)


def bent_events(n, seed, bend_sd=BEND_SD, stained=True):
    """
    (raw events, true fluorophore abundances): per-cell AF whose shape bends
    along two broad shapes by bend_sd x the AF amount (relative to the AF
    spectrum's norm), noise as NOISE; unstained (every abundance 0) when
    stained is False, as the sample a bend is learnt on.
    """
    rng = np.random.default_rng(seed)
    fl, af = fluor_spectra(), af_library()
    abund = rng.gamma(0.8, 2000.0, (n, len(fl))) * (rng.random((n, len(fl))) < 0.4) * float(stained)
    k = rng.gamma(2.0, 1500.0, n)
    rows = af[rng.integers(0, len(af), n)]
    c = rng.normal(0.0, bend_sd, (n, 2)) * (k * np.linalg.norm(rows, axis=1))[:, None]
    signal = abund @ fl + k[:, None] * rows + c @ _true_shapes()
    sd = np.sqrt(NOISE[0] + NOISE[1] * np.maximum(signal, 0.0))
    det = signal + rng.normal(0.0, 1.0, signal.shape) * sd
    time = np.sort(rng.uniform(0, 100, n))
    scatter = rng.normal(8e4, 1e4, (n, 2))
    return np.column_stack([time, scatter, det]), abund


@pytest.fixture(scope='module')
def library():
    fl, af = fluor_spectra(), af_library()
    return fl, af, build_af_precomputed(fl, af)


def _split(raw):
    return np.ascontiguousarray(raw[:, FL_COLS]), np.log(np.maximum(raw[:, SSC_COL], 1.0))


@pytest.fixture(scope='module')
def learnt(library):
    _fl, af, pre = library
    raw, _abund = bent_events(20_000, seed=1, stained=False)
    raw_fl, log_ssc = _split(raw)
    prior = ap.fit_af_prior(raw_fl, log_ssc, pre, af, *NOISE, fingerprint='f', source='s')
    posterior = ap.AFPosterior(prior, *NOISE, ssc_name='SSC-A')
    bend = asb.learn_bend(raw_fl, log_ssc, pre, af, posterior, fingerprint='b', source='s')
    return posterior, bend


@pytest.mark.numpy_only
def test_learnt_bend(learnt):
    _posterior, bend = learnt
    assert bend.active
    assert np.allclose(np.linalg.norm(bend.shapes, axis=1), 1.0)
    assert all(asb._participation(v) >= asb.MIN_PARTICIPATION for v in bend.shapes)
    assert np.all(bend.sigma2 >= 0.0) and np.any(bend.sigma2[-1] > 0.0)
    assert np.all(np.diff(bend.log_k) > 0.0)
    again = asb.AFBend.from_dict(bend.to_dict())
    assert np.array_equal(again.shapes, bend.shapes) and np.array_equal(again.sigma2, bend.sigma2)
    assert again.from_k == bend.from_k and again.full_k == bend.full_k
    assert bend.full_k > bend.from_k


@pytest.mark.numpy_only
def test_bend_changes_only_high_af_fluorophores(library, learnt):
    _fl, af, pre = library
    posterior, bend = learnt
    raw, abund = bent_events(8000, seed=2)
    raw_fl, log_ssc = _split(raw)
    bent_post = ap.AFPosterior(posterior.prior, *NOISE, ssc_name='SSC-A', bend=bend)
    plain = apply_af_unmixing(raw_fl, pre, af, af_posterior=posterior, log_ssc=log_ssc)
    bent = apply_af_unmixing(raw_fl, pre, af, af_posterior=bent_post, log_ssc=log_ssc)
    off = apply_af_unmixing(raw_fl, pre, af, af_posterior=bent_post, log_ssc=log_ssc, shape_bend=False)
    chunked = apply_af_unmixing(raw_fl, pre, af, chunk_size=777, af_posterior=bent_post, log_ssc=log_ssc)
    assert np.array_equal(bent['af_idx'], plain['af_idx'])
    assert np.array_equal(bent['af_scale'], plain['af_scale'])
    assert np.array_equal(off['unmixed'], plain['unmixed'])
    low = plain['af_scale'] < bend.from_k
    assert low.any() and (~low).any()
    assert np.array_equal(bent['unmixed'][low], plain['unmixed'][low])
    assert not np.allclose(bent['unmixed'][~low], plain['unmixed'][~low])
    assert np.allclose(chunked['unmixed'], bent['unmixed'], rtol=1e-9, atol=1e-6)

    zero = (abund == 0.0) & (plain['af_scale'] >= bend.full_k)[:, None]
    rms_plain = np.sqrt(np.mean(plain['unmixed'][zero] ** 2))
    rms_bent = np.sqrt(np.mean(bent['unmixed'][zero] ** 2))
    assert rms_bent < 0.9 * rms_plain


@pytest.mark.numpy_only
def test_no_shape_variability_no_change(library, learnt):
    _fl, af, pre = library
    posterior, bend = learnt
    still = asb.AFBend(shapes=bend.shapes, log_k=bend.log_k, sigma2=np.zeros_like(bend.sigma2),
                       from_k=bend.from_k, n_events=bend.n_events, fingerprint='z', source='s')
    raw, _abund = bent_events(3000, seed=3)
    raw_fl, log_ssc = _split(raw)
    plain = apply_af_unmixing(raw_fl, pre, af, af_posterior=posterior, log_ssc=log_ssc)
    bent = apply_af_unmixing(raw_fl, pre, af, af_posterior=ap.AFPosterior(posterior.prior, *NOISE,
                                                                          ssc_name='SSC-A', bend=still),
                             log_ssc=log_ssc)
    assert np.allclose(bent['unmixed'], plain['unmixed'], rtol=0.0, atol=1e-6)


@pytest.mark.numpy_only
def test_subset_drops_bend(learnt):
    posterior, bend = learnt
    bent_post = ap.AFPosterior(posterior.prior, *NOISE, ssc_name='SSC-A', bend=bend)
    keep = np.ones(N_DET, dtype=bool)
    keep[0] = False
    assert bent_post.subset(keep).bend is None


def _write_raw_fcs(path, events):
    keywords = {'$CYT': 'Synthetic'}
    for i, name in enumerate(RAW_PNN, start=1):
        keywords.update({f'$P{i}N': name, f'$P{i}B': '32', f'$P{i}E': '0,0',
                         f'$P{i}R': '100' if name == 'Time' else str(int(syn.MAGNITUDE_CEILING))})
    path.parent.mkdir(parents=True, exist_ok=True)
    fn.write_fcs(np.asarray(events, dtype=np.float64), keywords, path)


def _context(tmp_path, raw, shape_bend):
    _write_raw_fcs(tmp_path / syn.AF_SAMPLE, raw)
    return ap.AFNoiseContext(experiment_dir=str(tmp_path), cell_noise=NOISE, bead_noise=NOISE,
                             noise_fingerprint='n', bead_samples=frozenset(), references={'P': syn.AF_SAMPLE},
                             pnn_raw=tuple(RAW_PNN), whitelisted=True, fl_ids_raw=tuple(FL_COLS),
                             ssc_name='SSC-A', shape_bend=shape_bend)


def test_stored_bend(tmp_path, library, monkeypatch):
    _fl, af, pre = library
    raw, _abund = bent_events(20_000, seed=4, stained=False)
    ctx = _context(tmp_path, raw, shape_bend=True)
    state = AFState(profile_names=('P',), precomputed=pre, spectra=af)
    first = ctx.posterior_for(state, syn.AF_SAMPLE)
    assert first is not None and first.bend is not None and first.bend.active
    assert first.bend.source == syn.AF_SAMPLE
    assert (tmp_path / 'AutoSpectral' / asb.BENDS_FILE).exists()
    ap._memory.clear()

    def no_learning(*_a, **_k):
        raise AssertionError('learnt again although stored')

    monkeypatch.setattr(asb, 'learn_bend', no_learning)
    again = ctx.posterior_for(state, syn.AF_SAMPLE)
    assert again.bend.fingerprint == first.bend.fingerprint
    assert np.array_equal(again.bend.shapes, first.bend.shapes)

    off = _context(tmp_path, raw, shape_bend=False)
    assert off.posterior_for(state, syn.AF_SAMPLE).bend is None
    assert ap.signature(off) != ap.signature(ctx)


@pytest.mark.numpy_only
def test_setting_defaults_on():
    assert asb.shape_bend_on({}) and asb.shape_bend_on(None)
    process = {}
    asb.set_shape_bend(process, False)
    assert not asb.shape_bend_on(process)


# ---------------------------------------------------------------------------
# M2pg with the AF shape bend
# ---------------------------------------------------------------------------

def _noise_model():
    return nm.NoiseModel(detectors=RAW_PNN[3:], sigma0_sq=NOISE[0], g=NOISE[1], c=NOISE[2],
                         flags=np.zeros(N_DET, dtype=np.int64))


@pytest.fixture(scope='module')
def m2pg_inputs(library, learnt):
    """Stained events, their true abundances, the noise-aware AF fixed for M2pg and an envelope for the prior."""
    fl, af, pre = library
    posterior, _bend = learnt
    raw, abund = bent_events(4000, seed=6)
    raw_fl, log_ssc = _split(raw)
    r1 = apply_af_unmixing(raw_fl, pre, af, af_posterior=posterior, log_ssc=log_ssc)
    fixed = (np.asarray(r1['af_idx'], dtype=np.int64) - 1, np.asarray(r1['af_scale'], dtype=np.float64))
    unstained, _abund = bent_events(4000, seed=7, stained=False)
    names = [f'F{i + 1}' for i in range(len(fl))]
    entry, _sample = make_envelope_entry(_split(unstained)[0], names, fl, af)
    env = build_runtime_envelope(entry, None, names, fl, syn.MAGNITUDE_CEILING)
    return raw_fl, abund, fixed, env


@pytest.mark.numpy_only
def test_m2pg_bend_is_gls_under_the_bent_covariance(library, learnt):
    fl, _af, _pre = library
    _posterior, bend = learnt
    rng = np.random.default_rng(8)
    n, f, n_l = 40, len(fl), len(bend.shapes)
    y = 2000.0 + rng.normal(0.0, 300.0, (n, N_DET))
    w = rng.uniform(1e-4, 1e-3, (n, N_DET))
    pen = rng.uniform(0.0, 1e-5, (n, f))
    t2 = rng.uniform(1e3, 1e6, (n, n_l))
    t2[:10] = 0.0
    out = gu.pgls_solve(y, fl, None, w, pen, bend.shapes, gu.bend_precision(t2))
    for i in range(n):
        w_full = np.linalg.inv(np.diag(1.0 / w[i]) + bend.shapes.T @ np.diag(t2[i]) @ bend.shapes)
        normal = fl @ w_full @ fl.T + np.diag(pen[i])
        normal += 1e-12 * np.trace(normal) / f * np.eye(f)
        x = np.linalg.solve(normal, fl @ w_full @ y[i])
        np.testing.assert_allclose(out['x'][i], x, rtol=1e-7, atol=1e-7 * np.max(np.abs(x)))
        np.testing.assert_allclose(out['inv_diag'][i], np.diag(np.linalg.inv(normal)), rtol=1e-7)


@pytest.mark.numpy_only
def test_m2pg_bend_changes_only_high_af_events(library, learnt, m2pg_inputs):
    fl, af, _pre = library
    _posterior, bend = learnt
    raw_fl, abund, fixed, env = m2pg_inputs
    raw_fl = raw_fl.copy()
    ceiling = 10.0 * float(raw_fl.max())
    saturated = np.flatnonzero(fixed[1] >= bend.full_k)[:20]
    raw_fl[saturated, 0] = ceiling
    model = _noise_model()
    args = dict(ceiling=ceiling, use_kernel=False, fixed_af=fixed)
    plain = gu.unmix_m2pg(raw_fl, fl, model, af, env, **args)
    bent = gu.unmix_m2pg(raw_fl, fl, model, af, env, af_bend=bend, **args)
    still = asb.AFBend(shapes=bend.shapes, log_k=bend.log_k, sigma2=np.zeros_like(bend.sigma2),
                       from_k=bend.from_k, n_events=bend.n_events, fingerprint='z', source='s', full_k=bend.full_k)
    unbent = gu.unmix_m2pg(raw_fl, fl, model, af, env, af_bend=still, **args)
    assert np.array_equal(unbent['unmixed'], plain['unmixed'])
    assert np.array_equal(bent['af_idx'], plain['af_idx'])
    assert np.array_equal(bent['af_scale'], plain['af_scale'])
    low = fixed[1] < bend.from_k
    assert low.any() and (~low).any()
    same = low.copy()
    same[saturated] = True
    assert np.array_equal(bent['unmixed'][same], plain['unmixed'][same])
    assert not np.allclose(bent['unmixed'][~same], plain['unmixed'][~same])

    zero = (abund == 0.0) & ((fixed[1] >= bend.full_k) & ~same)[:, None]
    rms_plain = np.sqrt(np.mean(plain['unmixed'][zero] ** 2))
    rms_bent = np.sqrt(np.mean(bent['unmixed'][zero] ** 2))
    assert rms_bent < 0.9 * rms_plain


@pytest.mark.parametrize('lanes', [0, 8])
def test_m2pg_bend_kernel_matches_numpy(library, learnt, m2pg_inputs, monkeypatch, lanes):
    if not gu.kernel_available():
        pytest.skip('M2pg kernel not built')
    import gls_m2p_kernel_wrapper as wrapper

    monkeypatch.setattr(wrapper, 'unmix_m2p_kernel', functools.partial(wrapper.unmix_m2p_kernel, batch_lanes=lanes))
    fl, af, _pre = library
    _posterior, bend = learnt
    raw_fl, _abund, fixed, env = m2pg_inputs
    model = _noise_model()
    py = gu.unmix_m2pg(raw_fl, fl, model, af, env, use_kernel=False, fixed_af=fixed, af_bend=bend)
    k = gu.unmix_m2pg(raw_fl, fl, model, af, env, use_kernel=True, n_threads=2, fixed_af=fixed, af_bend=bend)
    plain = gu.unmix_m2pg(raw_fl, fl, model, af, env, use_kernel=True, n_threads=2, fixed_af=fixed)
    scale = gu.robust_sd(py['unmixed'])
    assert np.max(np.abs(k['unmixed'] - py['unmixed']) / scale) < 1e-6
    assert not np.allclose(k['unmixed'], plain['unmixed'])


@pytest.mark.parametrize('spread', [False, True])
def test_r1_bend_kernel_matches_numpy(library, learnt, monkeypatch, spread):
    if not gu.kernel_available():
        pytest.skip('M2pg kernel not built')
    import gls_m2p_kernel_wrapper as wrapper
    if not wrapper.BEND_R1_AVAILABLE:
        pytest.skip('kernel built without the per-cell AF shape bend')
    _fl, af, pre = library
    posterior, bend = learnt
    raw, _abund = bent_events(6000, seed=9)
    raw_fl, log_ssc = _split(raw)
    plain = apply_af_unmixing(raw_fl, pre, af, af_posterior=posterior, log_ssc=log_ssc)
    k, j = plain['af_scale'], np.asarray(plain['af_idx'], dtype=np.int64) - 1
    if spread:
        n_fluors = plain['unmixed'].shape[1]
        pre = dict(pre, envelope={'spread': np.full((n_fluors, n_fluors), 0.02) + 0.1 * np.eye(n_fluors)})
    monkeypatch.setattr(asb, 'USE_KERNEL', False)
    reference = asb.apply_bend(raw_fl, plain['unmixed'], k, j, pre, af, NOISE, bend)
    monkeypatch.setattr(asb, 'USE_KERNEL', True)
    compiled = asb.apply_bend(raw_fl, plain['unmixed'], k, j, pre, af, NOISE, bend)
    bent = k >= bend.from_k
    assert bent.any() and not np.allclose(reference[bent], plain['unmixed'][bent])
    assert np.max(np.abs(compiled - reference) / gu.robust_sd(reference)) < 1e-7
    assert np.array_equal(compiled[~bent], reference[~bent])

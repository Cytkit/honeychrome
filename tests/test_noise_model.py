"""
Tests for controller_components/noise_model.py, noise_selection.py and
gls_unmix.py: the noise model on simulated groups with a known model, the
experiment fit on a simulated experiment (bead negatives, variant cache,
storage), M2pg (kernel against NumPy, the prior, the whitened AF start,
assembly), the estimator options, clamped data and flat weights, the
automatic event selection and version 1 files.
"""

import json

import numpy as np
import pytest

import noise_synthetic as syn
from honeychrome.controller_components import gls_unmix as gu
from honeychrome.controller_components import noise_model as nm
from honeychrome.controller_components import noise_selection as ns
from honeychrome.controller_components.autospectral_functions import (
    apply_af_unmixing, build_af_precomputed, build_runtime_envelope, make_envelope_entry,
)


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------

def test_noise_model_recovers_kappa_and_sigma0():
    panel = syn.simulated_panel(12, 48, seed=6)
    model = nm.fit_noise_model(syn.simulated_groups(panel, 4_000), panel['detectors'])
    rk = model.kappa / panel['kappa']
    rs = np.sqrt(model.sigma0_sq) / panel['sigma0']
    assert np.median(rk) == pytest.approx(1.0, abs=0.05)
    assert np.median(rs) == pytest.approx(1.0, abs=0.05)
    cal = nm.calibrate(model, nm.calibration_sets(syn.simulated_groups(panel, 2_000)))
    assert cal['in_band_controls'] >= 0.9


def test_noise_model_save_load_and_fingerprint(tmp_path):
    panel = syn.simulated_panel(6, 40, seed=9)
    model = nm.fit_noise_model(syn.simulated_groups(panel, 2_000), panel['detectors'])
    model.fingerprint = 'abc'
    path = tmp_path / 'noise_model.npz'
    nm.save_noise_model(model, path)
    back = nm.load_noise_model(path, 'abc')
    assert back is not None and back.detectors == model.detectors
    assert np.allclose(back.g, model.g) and np.allclose(back.sigma0_sq, model.sigma0_sq)
    assert nm.load_noise_model(path, 'other') is None


# ---------------------------------------------------------------------------
# Experiment fit
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def experiment():
    panel, samples, snap = syn.simulated_experiment()
    fit = gu.fit_experiment_noise_model(snap, lambda _t: None, syn.loader_for(samples))
    return panel, samples, snap, fit


def test_fit_on_simulated_experiment(experiment):
    panel, _samples, snap, fit = experiment
    assert fit.n_controls == 12
    assert len(fit.bead_sources) == 4 and all('internal negative' in b['source'] for b in fit.bead_sources)
    assert np.median(fit.model.kappa / panel['kappa']) == pytest.approx(1.0, abs=0.15)
    assert fit.calibration['in_band_controls'] >= 0.8
    assert fit.cell_null_var.shape == (12,) and fit.bead_null_var.shape == (12,)
    assert fit.unstained_path == 'Unstained.fcs'


def test_variant_cache_gives_the_same_model(experiment, tmp_path):
    _panel, samples, snap, fit = experiment
    cached = dict(snap, variant_cache=str(tmp_path / gu.VARIANT_FILE))
    first = gu.fit_experiment_noise_model(cached, lambda _t: None, syn.loader_for(samples))
    assert (tmp_path / gu.VARIANT_FILE).exists()
    second = gu.fit_experiment_noise_model(cached, lambda _t: None, syn.loader_for(samples))
    np.testing.assert_array_equal(first.model.sigma0_sq, fit.model.sigma0_sq)
    np.testing.assert_array_equal(second.model.sigma0_sq, fit.model.sigma0_sq)
    np.testing.assert_array_equal(second.model.g, fit.model.g)


def test_store_and_load(experiment, tmp_path):
    panel, _samples, snap, fit = experiment
    process = {}
    gu.store_noise_fit(process, tmp_path, fit, prior_lambda=0.5)
    assert gu.noise_in_use(process) and gu.prior_lambda(process) == 0.5
    entry = process[gu.PROCESS_KEY]
    assert entry['calibration']['in_band_controls'] == pytest.approx(fit.calibration['in_band_controls'])
    state = gu.load_noise_state(process, tmp_path, panel['detectors'])
    assert state is not None
    np.testing.assert_array_equal(state.model.g, fit.model.g)
    np.testing.assert_allclose(state.cell_null_var, fit.cell_null_var)
    assert entry['negatives_file'] == gu.NEGATIVES_FILE
    np.testing.assert_array_equal(state.cell_negatives, fit.cell_negatives.astype(np.float64))
    assert 0 < len(state.cell_negatives) <= gu.NEGATIVE_SAMPLE_EVENTS
    assert (state.bead_negatives is None) == (fit.bead_negatives is None)
    assert gu.load_noise_state(process, tmp_path, panel['detectors'][:-1]) is None
    assert not gu.spectra_changed(entry, snap['fluor_names'], snap['spectra'])
    assert gu.spectra_changed(entry, snap['fluor_names'], snap['spectra'] * 1.01)


# ---------------------------------------------------------------------------
# M2pg
# ---------------------------------------------------------------------------

def _library(panel):
    af = np.vstack([panel['af0'], panel['af0'] + panel['af_modes'][0], panel['af0'] - panel['af_modes'][1]])
    return af / af.max(axis=1, keepdims=True)


@pytest.fixture(scope='module')
def unstained(experiment):
    panel, samples, snap, fit = experiment
    raw = samples['Unstained.fcs'][0]
    af = _library(panel)
    ceiling = snap['layout']['ceiling']
    entry, _sample = make_envelope_entry(raw[:20_000], snap['fluor_names'], snap['spectra'], af)
    env = build_runtime_envelope(entry, None, snap['fluor_names'], snap['spectra'], ceiling)
    return raw[20_000:24_000].copy(), af, env, ceiling


def test_whitened_af_start_with_unit_weights_is_the_main_app(experiment, unstained):
    _panel, _samples, snap, _fit = experiment
    raw, af, env, _ceiling = unstained
    ref = apply_af_unmixing(raw, build_af_precomputed(snap['spectra'], af, env), af)
    start = gu.per_cell_af_whitened(raw, snap['spectra'], af, env, np.ones(raw.shape[1]))
    np.testing.assert_array_equal(start['af_idx'], np.asarray(ref['af_idx']) - 1)
    np.testing.assert_allclose(start['unmixed'], ref['unmixed'], rtol=1e-9, atol=1e-6)


def test_prior_narrows_unstained_negatives(experiment, unstained):
    _panel, _samples, snap, fit = experiment
    raw, af, env, ceiling = unstained
    r1 = apply_af_unmixing(raw, build_af_precomputed(snap['spectra'], af, env), af)['unmixed']
    m2pg = gu.unmix_m2pg(raw, snap['spectra'], fit.model, af, env, ceiling=ceiling, use_kernel=False)
    gls = gu.unmix_m2pg(raw, snap['spectra'], fit.model, af, env, prior_lambda=0.0, ceiling=ceiling,
                        use_kernel=False)
    # Weighting alone narrows them (sigma0 varies across detectors); the prior narrows them further.
    assert np.median(gu.robust_sd(gls['unmixed']) / gu.robust_sd(r1)) < 1.0
    assert np.median(gu.robust_sd(m2pg['unmixed']) / gu.robust_sd(gls['unmixed'])) < 0.95


@pytest.mark.skipif(not gu.kernel_available(), reason='the _gls_m2p_kernel extension is not built')
def test_kernel_matches_numpy(experiment, unstained):
    _panel, samples, snap, fit = experiment
    raw, af, env, ceiling = unstained
    raw = raw.copy()
    raw[::9, 7] = ceiling
    py = gu.unmix_m2pg(raw, snap['spectra'], fit.model, af, env, ceiling=ceiling, use_kernel=False)
    k = gu.unmix_m2pg(raw, snap['spectra'], fit.model, af, env, ceiling=ceiling, n_threads=2)
    scale = gu.robust_sd(py['unmixed'])
    assert np.max(np.abs(k['unmixed'] - py['unmixed']) / scale) < 1e-6
    np.testing.assert_array_equal(k['af_idx'], py['af_idx'])
    beads = samples['F9 Beads.fcs'][0][:2000]
    py = gu.unmix_m2pg(beads, snap['spectra'], fit.model, null_var_no_af=fit.bead_null_var, use_kernel=False)
    k = gu.unmix_m2pg(beads, snap['spectra'], fit.model, null_var_no_af=fit.bead_null_var)
    assert np.max(np.abs(k['unmixed'] - py['unmixed']) / gu.robust_sd(py['unmixed'])) < 1e-6
    assert np.all(k['af_idx'] == -1)
    flat = nm.flat_model(snap['layout']['detectors'], float(fit.flat_phi))
    py = gu.unmix_m2pg(raw, snap['spectra'], flat, af, env, ceiling=ceiling, use_kernel=False)
    k = gu.unmix_m2pg(raw, snap['spectra'], flat, af, env, ceiling=ceiling, n_threads=2)
    assert np.max(np.abs(k['unmixed'] - py['unmixed']) / gu.robust_sd(py['unmixed'])) < 1e-6


def test_assemble_unmixed_fills_fluorophores_and_af_channels():
    from honeychrome.settings import af_abundance_channel, af_index_channel

    pnn = ['Time', 'F0', 'F1', af_abundance_channel, af_index_channel]
    settings = {'unmixed': {'fluorescence_channel_ids': [1, 2], 'event_channels_pnn': pnn}}
    raw = np.arange(12.0).reshape(3, 4)
    transfer = np.zeros((4, 5))
    transfer[0, 0] = 1.0
    result = {'unmixed': np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), 'af_idx': np.array([0, -1, 1]),
              'af_scale': np.array([10.0, 0.0, 20.0])}
    out = gu.assemble_unmixed(raw, transfer, settings, result, af_index_map=np.array([7, 9]))
    np.testing.assert_array_equal(out[:, 0], raw[:, 0])
    np.testing.assert_array_equal(out[:, 1:3], result['unmixed'])
    np.testing.assert_array_equal(out[:, 3], result['af_scale'])
    np.testing.assert_array_equal(out[:, 4], [7, 0, 9])


# ---------------------------------------------------------------------------
# Estimator options, clamped data, automatic selection, version 1 files
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def option_panel():
    return syn.simulated_panel(12, 48, seed=6)


def test_bead_af_rows_recover_kappa_and_sigma0(option_panel):
    """Bright bead AF with a 20% bead-to-bead CV: its own AF rows recover kappa; the mean-only negative does not."""
    panel = option_panel
    af_det = slice(0, 12)
    fits = {}
    for rows in (0, 3):
        fit_g, _cal_g, _lasers = syn.option_groups(panel, 3000, bead_af_cv=0.2, bead_af_rows=rows)
        fits[rows] = nm.fit_noise_model(fit_g, panel['detectors'])
    with_rows, mean_only = fits[3], fits[0]
    assert np.median(with_rows.kappa[af_det] / panel['kappa'][af_det]) == pytest.approx(1.0, abs=0.1)
    assert np.median(np.sqrt(with_rows.sigma0_sq) / panel['sigma0']) == pytest.approx(1.0, abs=0.1)
    assert not np.any(with_rows.flags & nm.FLAG_SIGMA0_LOW)
    assert np.median(mean_only.kappa[af_det] / panel['kappa'][af_det]) < 0.9
    assert np.count_nonzero(mean_only.flags & nm.FLAG_SIGMA0_LOW) >= 5


def test_laser_rows_recover_kappa_with_inter_laser_jitter(option_panel):
    panel = option_panel
    fit_g, _cal_g, lasers = syn.option_groups(panel, 3000, laser_jitter=0.03, laser_split=True)
    assert any(len(g.design) > 1 + 3 for g in fit_g if g.kind == 'control' and g.cls == 'cells')
    model = nm.fit_noise_model(fit_g, panel['detectors'])
    peaks = np.argmax(panel['spectra'], axis=1)
    assert np.median(model.kappa[peaks] / panel['kappa'][peaks]) == pytest.approx(1.0, abs=0.1)
    assert np.median(model.kappa / panel['kappa']) == pytest.approx(1.0, abs=0.1)


def test_laser_rows_follow_the_one_percent_rule():
    spectrum = np.array([0.005, 0.2, 1.0, 0.3, 0.004, 0.0])
    rows = gu.laser_rows(spectrum, ['A', 'A', 'B', 'B', 'C', 'C'])
    assert len(rows) == 2          # laser C below 1% of the peak
    np.testing.assert_array_equal(rows[0][0], [0.005, 0.2, 0, 0, 0, 0])
    assert gu.laser_rows(spectrum, ['A'] * 6) == []


def test_per_control_quadratic_recovers_kappa(option_panel):
    panel = option_panel
    spread = np.random.default_rng(5).uniform(0.01, 0.02, 12)
    fit_g, cal_g, _lasers = syn.option_groups(panel, 3000, control_spread=spread)
    model = nm.fit_noise_model(fit_g, panel['detectors'], nm.NoiseSettings(per_control_quadratic=True))
    assert np.median(model.kappa / panel['kappa']) == pytest.approx(1.0, abs=0.1)
    assert set(model.group_c) == {f'F{f}' for f in range(12)}
    rel = np.array([np.sqrt(model.group_c[f'F{f}']) / spread[f] for f in range(12)])
    assert np.median(rel) == pytest.approx(1.0, abs=0.25)
    np.testing.assert_array_equal(model.c, 0.0)           # not part of the unmixing variance
    cal = nm.calibrate(model, nm.calibration_sets(cal_g))
    assert cal['per_control_c'] and cal['in_band_controls'] >= 0.9


def test_sigma0_per_class_recovers_both(option_panel):
    """Cells' sigma0 0.9 of the beads'; judged on detectors where cell AF is dark (sigma0 identifiable)."""
    panel = option_panel
    fit_g, cal_g, _lasers = syn.option_groups(panel, 4000, sigma0_cells_ratio=0.9)
    model = nm.fit_noise_model(fit_g, panel['detectors'], nm.NoiseSettings(sigma0_by_class=True))
    assert model.classes == ['cells', 'beads'] and model.sigma0_sq_by_class.shape == (2, 48)
    dark = panel['af0'] < 0.1
    cells = np.sqrt(model.sigma0_sq_by_class[0]) / (0.9 * panel['sigma0'])
    beads = np.sqrt(model.sigma0_sq_by_class[1]) / panel['sigma0']
    assert np.median(cells[dark]) == pytest.approx(1.0, abs=0.05)
    assert np.median(beads[dark]) == pytest.approx(1.0, abs=0.05)
    np.testing.assert_array_equal(model.for_class('beads').sigma0_sq, model.sigma0_sq_by_class[1])
    np.testing.assert_array_equal(model.variance(np.zeros(48), cls='beads'), model.sigma0_sq_by_class[1])
    cal = nm.calibrate(model, nm.calibration_sets(cal_g))
    for row in cal['groups']:
        if row['kind'] != 'control':
            assert row['median_ratio'] == pytest.approx(1.0, abs=0.05)


def test_laser_fallback_replaces_the_floor(option_panel):
    """A detector with no dark events, whose sigma0 the fit floors, takes the median of its laser's others."""
    panel = option_panel
    j = 40
    background = np.zeros(48)
    background[j] = 2e5
    fit_g, _cal_g, lasers = syn.option_groups(panel, 3000, background=background)
    floor = nm.fit_noise_model(fit_g, panel['detectors'])
    assert floor.flags[j] & nm.FLAG_SIGMA0_LOW
    laser = nm.fit_noise_model(fit_g, panel['detectors'], nm.NoiseSettings(sigma0_fallback='laser'), lasers=lasers)
    assert laser.flags[j] & nm.FLAG_SIGMA0_FALLBACK and not floor.flags[j] & nm.FLAG_SIGMA0_FALLBACK
    peers = [i for i, x in enumerate(lasers) if x == lasers[j] and i != j and not laser.flags[i] & 3]
    assert len(peers) >= 3
    assert np.sqrt(laser.sigma0_sq[j]) == pytest.approx(float(np.median(np.sqrt(laser.sigma0_sq[peers]))))
    assert np.sqrt(laser.sigma0_sq[j]) > 5 * np.sqrt(floor.sigma0_sq[j])
    assert 'sigma0 from same laser' in nm.detector_table(laser)[1][j][-1]


def test_clamped_data_detected_and_unmixed_with_clamped_weights(tmp_path):
    rng = np.random.default_rng(0)
    events = rng.normal(100.0, 150.0, (5000, 20))
    assert not nm.negatives_clamped([events])[0]
    assert nm.negatives_clamped([np.maximum(events, 0.08)])[0]
    assert not nm.negatives_clamped([np.maximum(events, 0.08)[:50]])[0]       # too few events to tell

    panel, samples, snap = syn.simulated_experiment(n_cell_controls=4, n_bead_controls=2, n_events=6000, seed=2)
    clamped = {k: (np.maximum(fl, 0.08), sc) for k, (fl, sc) in samples.items()}
    fit = gu.fit_experiment_noise_model(snap, lambda _t: None, syn.loader_for(clamped))
    assert fit.clamped and fit.flat_phi > 0 and any('Clamped data' in n for n in fit.notes)
    weights = fit.clamped_weights
    assert weights['kind'] == 'poisson' and weights['scale'] > 0
    assert weights['floor'] == pytest.approx(10.0 * np.median(fit.model.g))
    process = {}
    gu.store_noise_fit(process, tmp_path, fit)
    assert process[gu.UNMIXING_KEY]['noise_model'] is False and gu.clamped_weights_requested(process)
    assert gu.noise_in_use(process)
    from honeychrome.controller_components import unmix_method
    assert unmix_method.requested_method(process) == unmix_method.CLAMPED
    state = gu.load_noise_state(process, tmp_path, panel['detectors'])
    mu = np.full(len(panel['detectors']), 1e4)
    np.testing.assert_allclose(state.clamped_model.variance(mu),
                               weights['scale'] * fit.model.g * (weights['floor'] + mu))
    # scaled to the held-out unstained cells' residual variance about their AF basis
    neg = np.asarray(fit.cell_negatives, dtype=np.float64)
    proj = nm.project(nm.Group('u', 'unstained', neg, gu.af_basis(neg, 4, seed=42)))
    assert nm.dispersion_scale(state.clamped_model, proj) == pytest.approx(1.0, abs=0.25)
    process[gu.UNMIXING_KEY]['noise_model'] = True                           # the user can still tick it
    assert unmix_method.requested_method(process) == unmix_method.M2PG

    flat_snap = dict(snap, settings=gu.NoiseSetupSettings(**{**snap['settings'].__dict__, 'clamp_weights': 'flat'}))
    flat_fit = gu.fit_experiment_noise_model(flat_snap, lambda _t: None, syn.loader_for(clamped))
    gu.store_noise_fit(process, tmp_path, flat_fit)
    state = gu.load_noise_state(process, tmp_path, panel['detectors'])
    np.testing.assert_allclose(state.clamped_model.variance(mu), flat_fit.flat_phi)

    fit_plain = gu.fit_experiment_noise_model(snap, lambda _t: None, syn.loader_for(samples))
    assert not fit_plain.clamped
    gu.store_noise_fit(process, tmp_path, fit_plain)
    assert process[gu.UNMIXING_KEY]['noise_model'] is True and not gu.clamped_weights_requested(process)


def test_flat_weights_without_prior_are_per_cell_af(experiment, unstained):
    """M0p reuses the M2pg path: with lambda 0, flat weights give the main app's per-cell AF."""
    _panel, _samples, snap, _fit = experiment
    raw, af, env, ceiling = unstained
    flat = nm.flat_model(snap['layout']['detectors'], 1234.0)
    ref = apply_af_unmixing(raw, build_af_precomputed(snap['spectra'], af, env), af)
    out = gu.unmix_m2pg(raw, snap['spectra'], flat, af, env, prior_lambda=0.0, ceiling=ceiling, use_kernel=False)
    same = out['af_idx'] == np.asarray(ref['af_idx']) - 1
    assert same.mean() > 0.99
    scale = gu.robust_sd(ref['unmixed'])
    assert np.max(np.abs(out['unmixed'][same] - ref['unmixed'][same]) / scale) < 1e-6


def test_automatic_selection_keeps_singlets_and_drops_doublets():
    """
    Without saturation the gate keeps >= 90% of singlets. With 30% of them
    clipped at the SSC ceiling the density mode lies on the clipped line:
    every clipped singlet is kept but only part of the unclipped tail.
    """
    for saturated, min_kept in ((0.0, 0.9), (0.3, 0.7)):
        fsc_a, ssc_a, fsc_h, kind = syn.scatter_population(saturated_frac=saturated)
        sc = np.column_stack([fsc_a, ssc_a, fsc_h])
        mask, counts = ns.select_events(sc, (0, 1, 2))
        assert mask[kind == 0].mean() >= min_kept
        assert mask[kind == 1].mean() < 0.05
        assert mask[kind == 2].mean() < 0.05
        assert counts['n_root'] == len(kind) and counts['n_singlet'] == int(mask.sum()) <= counts['n_scatter']
        if saturated:
            assert mask[(kind == 0) & (ssc_a >= ssc_a.max())].mean() > 0.99
    assert ns.scatter_columns(['FSC-A', 'FSC-H', 'SSC-A'], 'FSC-A', 'SSC-A') == (0, 2, 1)
    assert ns.scatter_columns(['FSC-A', 'SSC-A'], 'FSC-A', 'SSC-A') == (0, 1, None)


def test_experiment_fit_with_every_option(experiment):
    _panel, samples, snap, _fit = experiment
    settings = gu.NoiseSetupSettings(max_test_unstained=8_000, max_test_control=3_000, max_fit_events=4_000,
                                     max_cal_events=1_500, bead_af_rows=3, laser_split=True)
    settings.noise = nm.NoiseSettings(per_control_quadratic=True, sigma0_by_class=True, sigma0_fallback='laser')
    fit = gu.fit_experiment_noise_model(dict(snap, settings=settings), lambda _t: None, syn.loader_for(samples))
    assert fit.model.classes == ['cells', 'beads'] and fit.model.group_c
    assert fit.options['bead_af_rows'] == 3 and fit.options['sigma0_by_class']
    assert fit.selection['mode'] == 'auto' and set(fit.selection['counts']) >= {'Unstained.fcs', 'F0 Cells.fcs'}
    n = fit.selection['counts']['F0 Cells.fcs']
    assert n['n_singlet'] < n['n_root'] == len(samples['F0 Cells.fcs'][0])
    assert fit.selection['sources']['F0'] == 'F0 Cells.fcs'
    assert fit.calibration['in_band_controls'] >= 0.8


def test_version_1_file_loads(tmp_path):
    panel = syn.simulated_panel(6, 40, seed=9)
    model = nm.fit_noise_model(syn.simulated_groups(panel, 2_000), panel['detectors'])
    path = tmp_path / 'noise_model.npz'
    table = {f'fit_{k}': np.asarray(v) for k, v in model.fit_table.items() if k != 'group_c'}
    np.savez(path, sigma0_sq=model.sigma0_sq, g=model.g, c=model.c, flags=model.flags,
             detectors=np.array(json.dumps(model.detectors)), groups=np.array(json.dumps([])),
             settings=np.array(json.dumps({})), source=np.array('fitted'), fingerprint=np.array('v1'),
             version=np.array(1), **table)
    back = nm.load_noise_model(path, 'v1')
    assert back is not None and back.sigma0_sq_by_class is None and back.group_c == {}
    np.testing.assert_array_equal(back.sigma0_sq, model.sigma0_sq)
    np.testing.assert_array_equal(back.variance(np.full(40, 1e4)), model.variance(np.full(40, 1e4)))
    np.testing.assert_array_equal(back.for_class('beads').sigma0_sq, model.sigma0_sq)

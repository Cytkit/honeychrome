"""
Tests for controller_components/unmix_method.py and its use through
sample_loader: per-cell AF unchanged when nothing is ticked, M2pg through
the snapshot equal to the Noise page's computation (with and without AF,
cells and beads), the clamped-data weights and the bead sigma0 of a per-class
model, per-sample corrected spectra, R2 dispatch, the display
cache key and files.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import noise_synthetic as syn
from honeychrome.controller_components import gls_unmix as gu
from honeychrome.controller_components import sample_loader as sl
from honeychrome.controller_components import unmix_method as um
from honeychrome.controller_components.autospectral_functions import (
    af_index_lookup, apply_af_transfer, build_af_precomputed, build_runtime_envelope, make_envelope_entry,
)
from honeychrome.settings import af_abundance_channel, af_index_channel

A, B, BEADS = 'Raw/A.fcs', 'Raw/B.fcs', 'Raw/Beads.fcs'


@pytest.fixture(scope='module')
def fitted():
    panel, samples, snap = syn.simulated_experiment(n_cell_controls=6, n_bead_controls=2, n_events=8_000)
    fit = gu.fit_experiment_noise_model(snap, lambda _t: None, syn.loader_for(samples))
    raw_un = samples['Unstained.fcs'][0]
    af = np.vstack([panel['af0'], panel['af0'] + panel['af_modes'][0], panel['af0'] - panel['af_modes'][1]])
    af = af / af.max(axis=1, keepdims=True)
    entry, _ = make_envelope_entry(raw_un[:15_000], snap['fluor_names'], snap['spectra'], af)
    events = {A: samples['F1 Cells.fcs'], B: samples['F2 Cells.fcs'], BEADS: samples['F7 Beads.fcs']}
    return panel, snap, fit, af, entry, events


def _transfer(spectra, d, f, spillover=None):
    w = um.unmixing_matrix_for(spectra, None)
    t = np.zeros((d + 2, 4 + f + 2))
    t = um.transfer_with_spectra(t, list(range(d)), list(range(4, 4 + f)), w, spillover)
    t[d, 2] = 1.0
    t[d + 1, 3] = 1.0
    return t


def make_controller(fitted, tmp_path, unmixing=None):
    panel, snap, fit, af, entry, _events = fitted
    names, spectra = snap['fluor_names'], snap['spectra']
    f, d = spectra.shape
    ceiling = snap['layout']['ceiling']
    env = build_runtime_envelope(entry, None, names, spectra, ceiling)
    pnn_raw = list(panel['detectors']) + ['FSC-A', 'SSC-A']
    pnn_unmixed = ['Time', 'event_id', 'FSC-A', 'SSC-A'] + list(names) + [af_abundance_channel, af_index_channel]
    settings = {
        'raw': {'event_channels_pnn': pnn_raw, 'whitelisted_pnn': None, 'magnitude_ceiling': ceiling,
                'scatter_channel_ids': [d, d + 1], 'fluorescence_channel_ids': list(range(d))},
        'unmixed': {'event_channels_pnn': pnn_unmixed, 'fluorescence_channel_ids': list(range(4, 4 + f))},
    }
    process = {
        'profiles': {n: spectra[i].tolist() for i, n in enumerate(names)},
        'spectral_model': [{'label': n} for n in names],
        'spillover': np.eye(f).tolist(), 'unmixing_weights': [1.0] * d,
        'af_profiles': {'P': {'spectra': af.tolist(), 'envelope': entry}},
    }
    gu.store_noise_fit(process, tmp_path, fit, prior_lambda=1.0)
    process['unmixing'] = dict(unmixing or {'noise_model': False})
    samples = {'all_samples': {A: 'A', B: 'B', BEADS: 'Beads'}, 'sample_af_profiles': {A: ['P']},
               'bead_samples': [BEADS]}
    experiment = SimpleNamespace(settings=settings, process=process, samples=samples)
    return SimpleNamespace(
        experiment=experiment, experiment_dir=tmp_path, transfer_matrix=_transfer(spectra, d, f),
        filtered_raw_fluorescence_channel_ids=list(range(d)),
        af_precomputed_cache={'P': build_af_precomputed(spectra, af, env)},
        autospectral_variants={}, autospectral_unmixed_pos_thresholds=np.zeros(f),
    ), env


def _raw(fitted, key, n=3_000):
    fl, sc = fitted[5][key]
    return np.ascontiguousarray(np.column_stack([fl[:n], sc[:n]]))


def _r1(ctrl, raw, af):
    return apply_af_transfer(raw, ctrl.transfer_matrix, af.precomputed, af.spectra, ctrl.experiment.settings,
                             filtered_fl_ids_raw=ctrl.filtered_raw_fluorescence_channel_ids,
                             spillover=ctrl.experiment.process['spillover'], af_index_map=af.index_map)['unmixed']


def test_nothing_ticked_is_per_cell_af(fitted, tmp_path):
    ctrl, _env = make_controller(fitted, tmp_path)
    assert um.method_state(ctrl) is None
    del ctrl.experiment.process['unmixing']
    assert um.method_state(ctrl) is None
    snap = sl.snapshot_unmix_state(ctrl, include_qc=False)
    assert snap.method is None and snap.method_key(A) == ''
    raw = _raw(fitted, A)
    out = sl.unmix_events(raw, snap, snap.af_for(A), sample_key=A)['unmixed']
    np.testing.assert_array_equal(out, _r1(ctrl, raw, snap.af_for(A)))


def test_noise_model_wins_over_variants(fitted, tmp_path):
    ctrl, _env = make_controller(fitted, tmp_path, {'noise_model': True, 'variants': True})
    assert um.requested_method(ctrl.experiment.process) == um.M2PG
    ctrl.experiment.process['unmixing']['noise_model'] = False
    assert um.requested_method(ctrl.experiment.process) == um.R2
    del ctrl.experiment.process[gu.PROCESS_KEY]
    ctrl.experiment.process['unmixing'] = {'noise_model': True}
    assert um.requested_method(ctrl.experiment.process) == um.R1


def test_m2pg_through_snapshot_is_the_noise_page(fitted, tmp_path):
    ctrl, env = make_controller(fitted, tmp_path, {'noise_model': True, 'prior_lambda': 1.0})
    _panel, snap_n, fit, _af, _entry, _events = fitted
    spectra, ceiling = snap_n['spectra'], snap_n['layout']['ceiling']
    snap = sl.snapshot_unmix_state(ctrl, include_qc=False)
    state = snap.method
    assert state.method == um.M2PG and state.noise is not None
    settings, tm, spill = ctrl.experiment.settings, ctrl.transfer_matrix, ctrl.experiment.process['spillover']
    d = spectra.shape[1]
    nthreads = um.n_threads_default()

    raw = _raw(fitted, A)
    af = snap.af_for(A)
    page = gu.assemble_unmixed(raw, tm, settings, gu.unmix_m2pg(
        raw[:, :d], spectra, state.noise.model, af.spectra, env, ceiling=ceiling, n_threads=nthreads),
        spillover=spill, af_index_map=af.index_map)
    out = sl.unmix_events(raw, snap, af, return_af=True, sample_key=A)
    np.testing.assert_array_equal(out['unmixed'], page)
    assert out['af_idx'].min() >= 1
    assert not np.allclose(out['unmixed'], _r1(ctrl, raw, af))

    for key, null_var in ((B, state.noise.cell_null_var), (BEADS, state.noise.bead_null_var)):
        raw = _raw(fitted, key)
        page = gu.assemble_unmixed(raw, tm, settings, gu.unmix_m2pg(
            raw[:, :d], spectra, state.noise.model, None, None, null_var_no_af=null_var, ceiling=ceiling,
            n_threads=nthreads), spillover=spill)
        out = sl.unmix_events(raw, snap, None, return_af=True, sample_key=key)
        np.testing.assert_array_equal(out['unmixed'], page)
        assert np.all(out['af_idx'] == 0)
    assert snap.method_key(A) != snap.method_key(B) != snap.method_key(BEADS)

    # load_unmixed passes the sample key: beads keep their own prior
    raw_b = _raw(fitted, BEADS)
    as_cells = sl.unmix_events(raw_b, snap, None, sample_key=B)['unmixed']
    as_beads = sl.unmix_events(raw_b, snap, None, sample_key=BEADS)['unmixed']
    assert not np.allclose(as_cells, as_beads)


def test_m0p_and_particle_class_through_snapshot(fitted, tmp_path):
    """Clamped-data weights run M2pg's path with their own model; M2pg uses the bead sigma0 for bead samples."""
    _panel, snap_n, _fit, _af, _entry, _events = fitted
    spectra, ceiling = snap_n['spectra'], snap_n['layout']['ceiling']
    d = spectra.shape[1]
    nthreads = um.n_threads_default()
    ctrl, env = make_controller(fitted, tmp_path, {'noise_model': False, 'weights': gu.CLAMPED_WEIGHTS})
    assert um.requested_method(ctrl.experiment.process) == um.CLAMPED
    snap = sl.snapshot_unmix_state(ctrl, include_qc=False)
    state = snap.method
    assert state.method == um.CLAMPED and state.noise.clamped_model is not None
    raw = _raw(fitted, A)
    af = snap.af_for(A)
    expected = gu.unmix_m2pg(raw[:, :d], spectra, state.noise.clamped_model, af.spectra, env, ceiling=ceiling,
                             n_threads=nthreads)
    page = gu.assemble_unmixed(raw, ctrl.transfer_matrix, ctrl.experiment.settings, expected,
                               spillover=ctrl.experiment.process['spillover'], af_index_map=af.index_map)
    np.testing.assert_array_equal(sl.unmix_events(raw, snap, af, sample_key=A)['unmixed'], page)
    assert 'clamped data' in um.method_label(um.CLAMPED, 1.0)

    panel, samples, snap_e = syn.simulated_experiment(n_cell_controls=6, n_bead_controls=2, n_events=8_000)
    snap_e['settings'].noise = gu.nm.NoiseSettings(sigma0_by_class=True)
    fit = gu.fit_experiment_noise_model(snap_e, lambda _t: None, syn.loader_for(samples))
    assert fit.model.classes == ['cells', 'beads']
    gu.store_noise_fit(ctrl.experiment.process, tmp_path, fit)
    ctrl.experiment.process['unmixing'] = {'noise_model': True, 'prior_lambda': 1.0}
    um._noise_cache.clear()
    snap = sl.snapshot_unmix_state(ctrl, include_qc=False)
    noise = snap.method.noise
    raw_b = _raw(fitted, BEADS)
    out = sl.unmix_events(raw_b, snap, None, sample_key=BEADS)['unmixed']
    beads = gu.unmix_m2pg(raw_b[:, :d], spectra, noise.model.for_class('beads'), None, None,
                          null_var_no_af=noise.bead_null_var, ceiling=ceiling, n_threads=nthreads)
    page = gu.assemble_unmixed(raw_b, ctrl.transfer_matrix, ctrl.experiment.settings, beads,
                               spillover=ctrl.experiment.process['spillover'])
    np.testing.assert_array_equal(out, page)
    np.testing.assert_array_equal(noise.model_for(True).sigma0_sq, fit.model.sigma0_sq_by_class[1])
    assert not np.array_equal(noise.model_for(True).sigma0_sq, noise.model_for(False).sigma0_sq)


def _corrections(fitted, members, active=True, base_fp=None):
    _panel, snap, _fit, _af, _entry, _events = fitted
    names, spectra = snap['fluor_names'], snap['spectra']
    row = spectra[1].copy()
    row[np.argmax(row) + 2] += 0.08
    return {'base_fingerprint': base_fp or gu.spectra_fingerprint(names, spectra),
            'groups': {'Day1': {'samples': members, 'fluorophores': {names[1]: {'spectrum': row.tolist(),
                                                                                'active': active}}}}}, row


def test_corrected_spectra_per_sample(fitted, tmp_path):
    ctrl, env = make_controller(fitted, tmp_path, {'corrections': True})
    _panel, snap_n, _fit, af_spectra, _entry, _events = fitted
    names, spectra = snap_n['fluor_names'], snap_n['spectra']
    corr, row = _corrections(fitted, [A, B])
    ctrl.experiment.process[um.CORRECTIONS_KEY] = corr
    state = um.method_state(ctrl)
    assert state is not None and state.method == um.R1 and state.set_for(A) == 'Day1' and state.set_for(BEADS) == ''
    s_corr = spectra.copy()
    s_corr[1] = row
    snap = sl.snapshot_unmix_state(ctrl, include_qc=False)
    settings, spill = ctrl.experiment.settings, ctrl.experiment.process['spillover']
    d, f = spectra.shape[1], spectra.shape[0]
    tm_corr = _transfer(s_corr, d, f, spill)

    raw = _raw(fitted, A)
    expected = apply_af_transfer(raw, tm_corr, build_af_precomputed(s_corr, af_spectra, env), af_spectra, settings,
                                 filtered_fl_ids_raw=list(range(d)), spillover=spill,
                                 af_index_map=af_index_lookup(ctrl.experiment.process['af_profiles'], ['P']))
    out = sl.unmix_events(raw, snap, snap.af_for(A), sample_key=A)['unmixed']
    np.testing.assert_allclose(out, expected['unmixed'], rtol=1e-10, atol=1e-6)
    raw = _raw(fitted, B)
    np.testing.assert_allclose(sl.unmix_events(raw, snap, None, sample_key=B)['unmixed'], raw @ tm_corr,
                               rtol=1e-10, atol=1e-6)
    raw = _raw(fitted, BEADS)
    np.testing.assert_array_equal(sl.unmix_events(raw, snap, None, sample_key=BEADS)['unmixed'],
                                  raw @ ctrl.transfer_matrix)

    # inactive rows, a stale overlay, or the tick off: base spectra everywhere
    ctrl.experiment.process[um.CORRECTIONS_KEY] = _corrections(fitted, [A], active=False)[0]
    assert um.method_state(ctrl) is None
    ctrl.experiment.process[um.CORRECTIONS_KEY] = _corrections(fitted, [A], base_fp='stale')[0]
    assert um.method_state(ctrl) is None
    ctrl.experiment.process[um.CORRECTIONS_KEY] = corr
    ctrl.experiment.process['unmixing']['corrections'] = False
    assert um.method_state(ctrl) is None

    # with the noise model, the corrected spectra are the ones M2pg fits
    ctrl.experiment.process['unmixing'].update(corrections=True, noise_model=True)
    snap = sl.snapshot_unmix_state(ctrl, include_qc=False)
    state = snap.method
    raw = _raw(fitted, A)
    af = snap.af_for(A)
    expected = gu.unmix_m2pg(raw[:, :d], s_corr, state.noise.model, af.spectra, env,
                             ceiling=snap_n['layout']['ceiling'], n_threads=state.n_threads)
    out = sl.unmix_events(raw, snap, af, sample_key=A)['unmixed']
    np.testing.assert_allclose(out[:, 4:4 + f], expected['unmixed'], rtol=1e-9, atol=1e-6)
    assert names[1] in state.fluor_names


@pytest.mark.skipif(not um._r2_kernel_available(), reason='the AutoSpectral kernel is not built')
def test_r2_dispatch(fitted, tmp_path):
    ctrl, env = make_controller(fitted, tmp_path, {'variants': True, 'noise_model': False})
    _panel, snap_n, _fit, _af, _entry, _events = fitted
    names, spectra = snap_n['fluor_names'], snap_n['spectra']
    ref = spectra[2]
    mode = np.roll(ref, 1) - ref
    v = np.vstack([ref, ref + 0.05 * mode, ref - 0.05 * mode])
    v = v / v.max(axis=1, keepdims=True)
    ctrl.autospectral_variants = {names[2]: {'v_mats': v, 'delta': v - ref}}
    ctrl.experiment.process['autospectral_variants_meta'] = {names[2]: {'active': True}}
    ctrl.experiment.process['unmixing']['variant_kernel'] = {'alpha': 0.4}
    state = um.method_state(ctrl)
    assert state.method == um.R2 and state.variants['kernel']['alpha'] == 0.4
    assert um.needs_time_probe(state)
    snap = sl.snapshot_unmix_state(ctrl, include_qc=False)
    raw = _raw(fitted, A)
    af = snap.af_for(A)
    import sys
    from honeychrome.controller_components.plugin_loaders import bundled_plugins_path
    sys.path.insert(0, str(bundled_plugins_path))
    from autospectral_optimization_functions import unmix_autospectral_optimization
    d = spectra.shape[1]
    kernel = dict(um.VARIANT_KERNEL_DEFAULTS, alpha=0.4)
    res = unmix_autospectral_optimization(raw[:, :d], spectra, list(names), af.spectra,
                                          ctrl.autospectral_variants, [names[2]], np.zeros(len(names)),
                                          n_threads=state.n_threads, envelope=env, **kernel)
    out = sl.unmix_events(raw, snap, af, return_af=True, sample_key=A)
    np.testing.assert_allclose(out['unmixed'][:, 4:4 + len(names)], res['unmixed'], rtol=1e-12, atol=1e-9)
    np.testing.assert_array_equal(out['af_idx'], res['af_idx'])
    # without an AF library R2 leaves the sample on OLS
    raw = _raw(fitted, B)
    np.testing.assert_array_equal(sl.unmix_events(raw, snap, None, sample_key=B)['unmixed'],
                                  raw @ ctrl.transfer_matrix)


def test_display_cache(fitted, tmp_path):
    ctrl, _env = make_controller(fitted, tmp_path, {'noise_model': True, 'prior_lambda': 1.0})
    state = um.method_state(ctrl)
    snap = sl.snapshot_unmix_state(ctrl, include_qc=False)
    raw = _raw(fitted, A)
    fl_ids = ctrl.filtered_raw_fluorescence_channel_ids
    af = snap.af_for(A)
    key = um.display_cache_key(state, A, af, raw, fl_ids)
    assert key == um.display_cache_key(um.method_state(ctrl), A, af, raw.copy(), fl_ids)
    assert key != um.display_cache_key(state, A, None, raw, fl_ids)
    assert key != um.display_cache_key(state, A, af, raw[:-1], fl_ids)
    raw2 = raw.copy()
    raw2[0, 0] += 1.0
    assert key != um.display_cache_key(state, A, af, raw2, fl_ids)
    ctrl.experiment.process['unmixing']['prior_lambda'] = 0.5
    assert key != um.display_cache_key(um.method_state(ctrl), A, af, raw, fl_ids)

    fl = um.unmix_fluorescence(raw[:, fl_ids], state, af, A)
    um.store_display_cache(tmp_path, key, fl)
    back = um.load_display_cache(tmp_path, key, len(raw), len(state.fluor_names))
    np.testing.assert_allclose(back['unmixed'], fl['unmixed'], rtol=1e-6)
    np.testing.assert_array_equal(back['af_idx'], fl['af_idx'])
    assert um.load_display_cache(tmp_path, key, len(raw) - 1, len(state.fluor_names)) is None
    assert um.load_display_cache(tmp_path, 'missing', len(raw), len(state.fluor_names)) is None
    um.store_display_cache(tmp_path, 'second', fl, max_bytes=1)
    assert not list((tmp_path / um.DISPLAY_CACHE_SUBDIR).glob('*.npz'))


def test_whole_sample_load_uses_the_cache(fitted, tmp_path, monkeypatch):
    ctrl, _env = make_controller(fitted, tmp_path, {'noise_model': True, 'prior_lambda': 1.0})
    snap = sl.snapshot_unmix_state(ctrl, include_qc=False)
    raw, af = _raw(fitted, A), snap.af_for(A)
    fcs = tmp_path / 'A.fcs'
    fcs.write_bytes(b'fcs')
    first = sl.unmix_events(raw, snap, af, return_af=True, sample_key=A, cache_file=fcs)
    assert len(list((tmp_path / um.DISPLAY_CACHE_SUBDIR).glob('*.npz'))) == 1
    fresh = sl.unmix_events(raw, snap, af, return_af=True, sample_key=A)['unmixed']
    np.testing.assert_allclose(first['unmixed'], fresh, rtol=0, atol=1e-6 * np.abs(fresh).max())
    monkeypatch.setattr(um, 'unmix_fluorescence', lambda *a, **k: pytest.fail('unmixed again'))
    again = sl.unmix_events(raw, snap, af, return_af=True, sample_key=A, cache_file=fcs)
    np.testing.assert_array_equal(again['unmixed'], first['unmixed'])
    np.testing.assert_array_equal(again['af_idx'], first['af_idx'])


def test_unmix_fluorescence_r1_is_the_fluorescence_of_unmix(fitted, tmp_path):
    """R1 through unmix_fluorescence (used by plugins that unmix raw
    fluorescence only) gives unmix()'s fluorescence columns, with each
    sample's corrected spectra, with and without AF."""
    ctrl, _env = make_controller(fitted, tmp_path, {'corrections': True})
    ctrl.experiment.process[um.CORRECTIONS_KEY] = _corrections(fitted, [A, B])[0]
    _panel, snap_n, _fit, _af, _entry, _events = fitted
    f, d = snap_n['spectra'].shape
    snap = sl.snapshot_unmix_state(ctrl, include_qc=False)
    state = snap.method
    tm = _transfer(snap_n['spectra'], d, f)
    for key, af in ((A, snap.af_for(A)), (B, None), (BEADS, None)):
        raw = _raw(fitted, key)
        fl = um.unmix_fluorescence(raw[:, :d], state, af, key)
        full = um.unmix(raw, tm, ctrl.experiment.settings, list(range(d)), None, af, state, key)
        np.testing.assert_allclose(fl['unmixed'], full['unmixed'][:, 4:4 + f], rtol=1e-10, atol=1e-6)
        np.testing.assert_array_equal(np.where(fl['af_idx'] >= 0, fl['af_idx'] + 1, 0), full['af_idx'])
        if af is None:
            assert np.all(fl['af_idx'] == -1) and np.all(fl['af_scale'] == 0)


def test_probe_allocation_and_duration_text():
    assert um.allocate_probe_rows([100, 300, 0], 40) == [10, 30, 0]
    assert um.allocate_probe_rows([3, 5], 100) == [3, 5]
    assert sum(um.allocate_probe_rows([7, 11, 13], 10)) == 10
    assert um.allocate_probe_rows([0, 0], 10) == [0, 0]
    assert um.format_duration(45) == '45s' and um.format_duration(750) == '12.5min'
    assert um.format_duration(4320) == '1.2hr' and um.format_duration(float('nan')) == 'unknown'


def test_load_unmixed_caches_only_uncapped_reads(fitted, tmp_path, monkeypatch):
    ctrl, _env = make_controller(fitted, tmp_path, {'noise_model': True})
    snap = sl.snapshot_unmix_state(ctrl, include_qc=False)
    raw = _raw(fitted, A)
    seen = []
    monkeypatch.setattr(sl, 'read_raw_subsample', lambda *a, **k: (raw, np.arange(len(raw)), len(raw), len(raw)))
    monkeypatch.setattr(sl, 'unmix_events', lambda *a, cache_file=None, **k: seen.append(cache_file) or
                        {'unmixed': raw, 'af_scale': None, 'af_idx': None})
    sl.load_unmixed(tmp_path, A, snap, use_cache=True)
    sl.load_unmixed(tmp_path, A, snap, use_cache=True, max_events=100)
    sl.load_unmixed(tmp_path, A, snap)
    assert seen == [tmp_path / A, None, None]


def test_unmixing_signature_follows_the_method_and_each_samples_set(fitted, tmp_path):
    ctrl, _env = make_controller(fitted, tmp_path)
    base = sl.unmixing_signature(ctrl, [B, A])
    assert list(base['af_assignment']) == [A, B] and 'method' not in base
    ctrl.experiment.process['unmixing'] = {'noise_model': True, 'corrections': True}
    ctrl.experiment.process[um.CORRECTIONS_KEY] = _corrections(fitted, [B])[0]
    with_method = sl.unmixing_signature(ctrl, [A, B])
    state = um.method_state(ctrl)
    assert with_method['method'] == {A: [state.fingerprint, ''], B: [state.fingerprint, 'Day1']}
    assert {k: v for k, v in with_method.items() if k != 'method'} == base


def test_automated_gating_unmixing_digest_changes_with_the_method(fitted, tmp_path):
    pytest.importorskip('PySide6')
    import automated_gating_tab as agt

    ctrl, _env = make_controller(fitted, tmp_path)
    plain = agt.unmixing_digest(ctrl)
    assert plain == agt.unmixing_digest(ctrl)
    ctrl.experiment.process['unmixing'] = {'noise_model': True}
    assert agt.unmixing_digest(ctrl) != plain

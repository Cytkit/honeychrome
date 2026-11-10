"""
Tests for controller_components/spectral_correction.py: default samples and
groups, the scatter match vote, a per-group run on synthetic samples through
correction_snapshot and run_corrections, the stored overlay reaching each
group's samples through unmix_method, Active ticks, and clearing on a change
of the base spectra.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import fmu_synthetic as syn
from honeychrome.controller_components import af_auto_assign
from honeychrome.controller_components import spectral_correction as sc
from honeychrome.controller_components import unmix_method as um
from honeychrome.controller_components import envelope_builder as eb
from honeychrome.controller_components import gls_unmix as gu
from honeychrome.controller_components import spread_model
from honeychrome.controller_components.autospectral_functions import (
    build_af_precomputed, build_runtime_envelope, envelope_needs_source, make_envelope_entry,
    positivity_null_variance, save_envelope_sample,
)
from honeychrome.controller_components.sample_loader import resolve_af_for_profiles
from honeychrome.settings import af_abundance_channel, af_index_channel

G1A, G1B, G2A = 'Raw/G1/a.fcs', 'Raw/G1/b.fcs', 'Raw/G2/c.fcs'
UNST, CTRL, BEADS = 'Raw/AF/Unstained.fcs', 'Raw/Controls/Dye 0.fcs', 'Raw/Controls/Beads.fcs'
# group G1 carries Dye 4 <- Dye 3 in the base spectra, group G2 Dye 6 <- Dye 5
ERRORS = {'Raw/G1': ('Dye 4', 'Dye 3'), 'Raw/G2': ('Dye 6', 'Dye 5')}


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(5)
    truth, names = syn.panel()
    events = {UNST: syn.af_cells(30_000, rng)}
    for key in (G1A, G1B, G2A):
        target, source = ERRORS[sc.group_of(key)]
        # the cells follow spectra in which the target carries less of the source than the base says
        cells = syn.inject(truth, names, [(source, target, 0.0)])
        t = names.index(target)
        cells[t] = truth[t] - 0.09 * truth[names.index(source)]
        cells[t] = np.maximum(cells[t], 0.0) / cells[t].max()
        events[key], _ab = syn.stained_events(15_000, cells, rng)
    return truth, names, events


def make_controller(data, tmp_path):
    truth, names, events = data
    f, d = truth.shape
    lib = syn.af_library()
    entry, sample = make_envelope_entry(events[UNST], names, truth, lib, 'P unstained.npy')
    save_envelope_sample(sample, tmp_path, entry['unstained_file'])
    env = build_runtime_envelope(entry, None, names, truth, syn.CEILING)
    pnn_raw = [f'D{i}-A' for i in range(d)] + ['FSC-A', 'SSC-A']
    pnn_unmixed = ['Time', 'event_id', 'FSC-A', 'SSC-A'] + list(names) + [af_abundance_channel, af_index_channel]
    w = um.unmixing_matrix_for(truth, None)
    transfer = um.transfer_with_spectra(np.zeros((d + 2, 4 + f + 2)), list(range(d)), list(range(4, 4 + f)), w)
    settings = {
        'raw': {'event_channels_pnn': pnn_raw, 'whitelisted_pnn': None, 'magnitude_ceiling': syn.CEILING,
                'scatter_channel_ids': [d, d + 1], 'fluorescence_channel_ids': list(range(d))},
        'unmixed': {'event_channels_pnn': pnn_unmixed, 'fluorescence_channel_ids': list(range(4, 4 + f))},
    }
    process = {
        'profiles': {n: truth[i].tolist() for i, n in enumerate(names)},
        'spectral_model': [{'label': n} for n in names], 'unmixing_weights': [1.0] * d,
        'af_profiles': {'P': {'spectra': lib.tolist(), 'envelope': entry, 'source_fcs': UNST}},
    }
    samples = {
        'all_samples': {G1A: 'a', G1B: 'b', G2A: 'c', UNST: 'Unstained', CTRL: 'Dye 0', BEADS: 'Beads'},
        'single_stain_controls': [CTRL], 'unstained_samples': [UNST], 'bead_samples': [BEADS],
        'sample_af_profiles': {G1A: ['P'], G1B: ['P'], G2A: ['P']},
    }
    cache = {'P': build_af_precomputed(truth, lib, env)}
    return SimpleNamespace(
        experiment=SimpleNamespace(settings=settings, process=process, samples=samples, cytometry={}),
        experiment_dir=tmp_path, transfer_matrix=transfer, filtered_raw_fluorescence_channel_ids=list(range(d)),
        af_precomputed_cache=cache, precomputed_for_profile=lambda name: cache.get(name),
        autospectral_variants={}, raw_gating=None, current_sample_path=None,
    )


def loader_for(events):
    def load(_experiment_dir, path, _gate, _layout, _gating):
        fl = events[path]
        return fl, np.ones((len(fl), 2))
    return load


@pytest.fixture(scope='module')
def run(data, tmp_path_factory):
    ctrl = make_controller(data, tmp_path_factory.mktemp('exp'))
    setup = sc.CorrectionSetup(per_group=True, gate='root', n_events=30_000)
    groups = sc.sample_groups(sc.default_samples(ctrl.experiment.samples), True)
    snap = sc.correction_snapshot(ctrl, setup, {g: UNST for g in groups})
    out = sc.run_corrections(snap, lambda _t: None, loader_for(data[2]))
    return ctrl, snap, out


# ---------------------------------------------------------------------------
# Samples, groups, unstained match
# ---------------------------------------------------------------------------

def test_default_samples_leave_out_controls_unstained_and_beads(data, tmp_path):
    ctrl = make_controller(data, tmp_path)
    assert sc.default_samples(ctrl.experiment.samples) == [G1A, G1B, G2A]


def test_include_ticks_offer_control_types(data, tmp_path):
    samples = make_controller(data, tmp_path).experiment.samples
    assert sc.sample_types(samples, CTRL) == {'single_stain'}
    assert sc.sample_types(samples, UNST) == {'unstained'} and sc.sample_types(samples, G1A) == set()
    assert sc.offered_samples(samples) == [G1A, G1B, G2A]
    assert sc.offered_samples(samples, {'beads': True}) == [G1A, G1B, G2A, BEADS]
    assert sc.offered_samples(samples, {'unstained': True, 'single_stain': True}) == [G1A, G1B, G2A, UNST, CTRL]
    # a sample of two types is offered when either is included
    samples = dict(samples, bead_samples=[BEADS, CTRL])
    assert sc.sample_types(samples, CTRL) == {'single_stain', 'beads'}
    assert CTRL in sc.offered_samples(samples, {'beads': True})
    assert CTRL not in sc.offered_samples(samples, {'unstained': True})


def test_groups_are_raw_subfolders():
    assert sc.sample_groups([G1A, G2A, G1B], False) == {sc.ALL_GROUP: [G1A, G2A, G1B]}
    assert sc.sample_groups([G1A, G2A, G1B], True) == {'Raw/G1': [G1A, G1B], 'Raw/G2': [G2A]}
    assert sc.group_of('top.fcs') == sc.TOP_FOLDER
    assert sc.sample_groups([], False) == {}


def test_match_unstained_takes_the_reference_with_most_events(monkeypatch):
    def fake(_dir, references, targets, _pair, _ceilings, progress=None, max_references=3):
        assert max_references == 1 and set(references) == {'U1', 'U2'}
        details = {'a': {'references': ['U1'], 'n_events': 100}, 'b': {'references': ['U2'], 'n_events': 300},
                   'c': {'references': ['U1'], 'n_events': 50}}
        return {'details': {t: details[t] for t in targets if t in details}, 'skipped': {'d': '5 usable events'}}
    monkeypatch.setattr(af_auto_assign, 'suggest_af_profiles', fake)
    out = sc.match_unstained('/x', {'G': ['a', 'b', 'c', 'd'], 'H': ['c']}, ['U1', 'U2'], ('FSC-A', 'SSC-A'),
                             [1.0, 1.0])
    assert out['G']['unstained'] == 'U2' and out['G']['votes'] == {'U1': 150, 'U2': 300}
    assert out['G']['skipped'] == {'d': '5 usable events'}
    assert out['H']['unstained'] == 'U1'
    none = sc.match_unstained('/x', {'G': ['a']}, [], ('FSC-A', 'SSC-A'), [1.0, 1.0])
    assert none['G']['unstained'] is None


def test_setup_round_trip_and_fix_settings():
    process = {}
    setup = sc.CorrectionSetup(samples=[G1A], per_group=True, gate='Cells', n_events=1000, seed=3,
                               unstained={'Raw/G1': UNST}, include={'beads': True},
                               fix={'spread_kappa': 1.0, 'unstained_margin': 1.5, 'max_iter': 4, 'nonsense': 1,
                                    'smooth_mode': 'consensus'})
    sc.save_setup(process, setup)
    back = sc.load_setup(process)
    assert back == setup
    s = sc.fix_settings(back)
    # a value stored before the range was narrowed is held to it; a setting no longer offered keeps its default
    assert s.spread_kappa == 2.0 and s.unstained_margin == 1.5 and s.max_iter == sc.fm.FixSettings().max_iter
    assert s.convergence_quantile == 1.0 and s.n_threads >= 1
    assert s.smooth_mode == 'consensus' and sc.fm.FixSettings().smooth_mode == 'damped'
    assert s.fixed_fluorophores == ()
    assert sc.load_setup({}) == sc.CorrectionSetup() and sc.CorrectionSetup().n_events == 400_000
    setup.fixed = ['F1', 'F2']
    sc.save_setup(process, setup)
    assert sc.load_setup(process).fixed == ['F1', 'F2']
    assert sc.fix_settings(sc.load_setup(process)).fixed_fluorophores == ('F1', 'F2')


# ---------------------------------------------------------------------------
# A run per group
# ---------------------------------------------------------------------------

def test_snapshot_reads_each_sample_af(run):
    _ctrl, snap, _out = run
    assert snap['gate'] == 'root' and snap['spread'] is None and snap['n_variants'] == 0
    assert set(snap['groups']) == {'Raw/G1', 'Raw/G2'}
    assert snap['af_of'][G1A] == ('P',) and snap['af_of'][UNST] == ('P',)
    assert list(snap['libraries']) == [('P',)]


def test_each_group_corrects_its_own_error(run, data):
    truth, names, _events = data
    _ctrl, _snap, out = run
    for group, (target, _source) in ERRORS.items():
        res = out['groups'][group]['result']
        accepted = {res.fluor_names[i] for i in np.flatnonzero(res.accepted)}
        assert target in accepted, (group, accepted)
        # every row is re-measured: another group's target may be adopted, but barely moves
        moved = dict(zip(res.fluor_names, sc.fm.row_angle(res.spectra_start, res.spectra)))
        others = [t for g, (t, _s) in ERRORS.items() if g != group]
        assert all(moved[t] < 0.1 * moved[target] for t in others), (group, moved)
    g1 = out['groups']['Raw/G1']
    assert g1['events'] == {G1A: 15_000, G1B: 15_000} and g1['n_unstained'] == 30_000


def test_stored_overlay_reaches_each_group_sample(run, data):
    ctrl, _snap, out = run
    truth, names, _events = data
    process = ctrl.experiment.process
    sc.store_corrections(process, out)
    entry = process[sc.STORE_KEY]
    assert um.unmixing_settings(process)['corrections'] is True
    assert set(entry['groups']['Raw/G1']['samples']) == {G1A, G1B}
    sets, membership = um.corrections_in_effect(process, names, truth)
    assert membership == {G1A: 'Raw/G1', G1B: 'Raw/G1', G2A: 'Raw/G2'}
    for group, (target, _s) in ERRORS.items():
        res = out['groups'][group]['result']
        t = names.index(target)
        assert np.allclose(sets[group][t], res.spectra[res.fluor_names.index(target)])
        unchanged = [i for i, n in enumerate(names) if n not in entry['groups'][group]['fluorophores']]
        assert np.array_equal(sets[group][unchanged], truth[unchanged])
        assert np.array_equal(sc.group_spectra(entry, group, names, truth), sets[group])
    state = um.method_state(ctrl)
    assert state is not None and state.method == um.R1
    assert state.set_for(G1A) == 'Raw/G1' and state.set_for(G2A) == 'Raw/G2' and state.set_for(UNST) == ''
    assert sc.group_for_sample(entry, G2A) == 'Raw/G2' and sc.group_for_sample(entry, UNST) is None
    rows = entry['groups']['Raw/G1']['log']
    assert rows[0]['accepted'] and rows[0]['angle'] > 0 and 'phase_one_angle' in rows[0]
    token = um.method_token(state)
    um.set_unmixing(process, corrections_apply_to=['Raw/G1'])
    _sets, membership = um.corrections_in_effect(process, names, truth, ctrl.experiment.samples)
    assert membership == {G1A: 'Raw/G1', G1B: 'Raw/G1'}
    state = um.method_state(ctrl)
    assert state.set_for(G2A) == '' and um.method_token(state) != token


def test_corrections_reach_every_fully_stained_sample_but_controls():
    g1d, g3e = 'Raw/G1/d.fcs', 'Raw/G3/e.fcs'
    samples = {'all_samples': {G1A: 'a', G1B: 'b', G2A: 'c', g1d: 'd', g3e: 'e', UNST: 'Unstained', CTRL: 'Dye 0',
                               BEADS: 'Beads'},
               'single_stain_controls': [CTRL], 'unstained_samples': [UNST], 'bead_samples': [BEADS]}
    assert sc.sample_folders(samples) == ['Raw/G1', 'Raw/G2', 'Raw/G3']
    per_group = {'per_group': True, 'groups': {'Raw/G1': {'samples': [G1A]}, 'Raw/G2': {'samples': [G2A]}}}
    assert sc.correction_membership(per_group, samples) == {G1A: 'Raw/G1', G1B: 'Raw/G1', g1d: 'Raw/G1',
                                                            G2A: 'Raw/G2'}
    assert sc.correction_group(per_group, samples, g3e) is None
    assert sc.correction_group(per_group, samples, UNST) is None
    pooled = {'per_group': False, 'groups': {sc.ALL_GROUP: {'samples': [G1A, CTRL]}}}
    every = sc.ALL_GROUP
    assert sc.correction_membership(pooled, samples) == {G1A: every, CTRL: every, G1B: every, G2A: every,
                                                         g1d: every, g3e: every}
    assert sc.correction_group(pooled, samples, BEADS) == every          # a control is previewed, not corrected
    assert sc.correction_membership(pooled, samples, apply_to=['Raw/G1', 'Raw/G3']) == {
        G1A: every, G1B: every, g1d: every, g3e: every}
    assert sc.correction_membership(pooled, None) == {G1A: every, CTRL: every}


def test_active_tick_and_comparison_override(run, data):
    ctrl, _snap, out = run
    truth, names, _events = data
    process = ctrl.experiment.process
    sc.store_corrections(process, out)
    target = ERRORS['Raw/G2'][0]
    stored = list(process[sc.STORE_KEY]['groups']['Raw/G2']['fluorophores'])
    assert target in stored
    assert all(sc.set_active(process, 'Raw/G2', label, False) for label in stored)
    assert not sc.set_active(process, 'Raw/G2', 'Not a dye', False)
    sets, membership = um.corrections_in_effect(process, names, truth)
    assert 'Raw/G2' not in sets and G2A not in membership
    corrected = sc.group_spectra(process[sc.STORE_KEY], 'Raw/G2', names, truth)
    assert np.array_equal(sc.group_spectra(process[sc.STORE_KEY], 'Raw/G2', names, truth, active_only=True), truth)
    base = um.method_state(ctrl, corrections=({}, {}))
    assert base is None
    preview = um.method_state(ctrl, corrections=({'Raw/G2': corrected}, {G2A: 'Raw/G2'}))
    assert preview.set_for(G2A) == 'Raw/G2' and preview.set_for(G1A) == ''
    um.set_unmixing(process, corrections=False)
    assert um.method_state(ctrl) is None


def test_cleared_when_the_base_spectra_change(run, data):
    ctrl, _snap, out = run
    truth, names, _events = data
    process = ctrl.experiment.process
    sc.store_corrections(process, out)
    assert not sc.clear_if_stale(process, names, truth)
    moved = truth.copy()
    moved[0] = np.roll(moved[0], 1)
    assert sc.corrections_stale(process, names, moved)
    assert sc.clear_if_stale(process, names, moved)
    assert sc.STORE_KEY not in process and um.unmixing_settings(process)['corrections'] is False
    assert not sc.clear_if_stale(process, names, moved)


def test_all_samples_together_and_missing_unstained(data, tmp_path):
    ctrl = make_controller(data, tmp_path)
    setup = sc.CorrectionSetup(per_group=False, gate='root', n_events=12_000,
                               fix={'spread_kappa': 2.5, 'smooth_updates': False})
    with pytest.raises(ValueError):
        sc.correction_snapshot(ctrl, setup, {})
    snap = sc.correction_snapshot(ctrl, setup, {sc.ALL_GROUP: UNST})
    out = sc.run_corrections(snap, lambda _t: None, loader_for(data[2]))
    g = out['groups'][sc.ALL_GROUP]
    assert g['events'] == {G1A: 4000, G1B: 4000, G2A: 4000}
    assert g['result'].settings['spread_kappa'] == 2.5


def test_af_row_in_the_spectral_model_is_left_out(data, tmp_path):
    truth, names, events = data
    ctrl = make_controller(data, tmp_path)
    process = ctrl.experiment.process
    af_row = syn.af_library()[0]
    process['profiles']['AF'] = af_row.tolist()
    process['spectral_model'].append({'label': 'AF'})
    spectra = np.vstack([truth, af_row])
    lib = syn.af_library()
    env = build_runtime_envelope(make_envelope_entry(events[UNST], names + ['AF'], spectra, lib)[0], None,
                                 names + ['AF'], spectra, syn.CEILING)
    ctrl.af_precomputed_cache['P'] = build_af_precomputed(spectra, lib, env)
    setup = sc.CorrectionSetup(samples=[G1A], gate='root', n_events=10_000, fix={'smooth_updates': False})
    snap = sc.correction_snapshot(ctrl, setup, {sc.ALL_GROUP: UNST})
    assert snap['fl'] == list(range(len(names)))
    assert snap['tandems'] == sc.fm.tandem_pairs(names)
    pre = snap['libraries'][('P',)]['precomputed']
    assert pre['P'].shape[0] == len(names) and np.asarray(pre['envelope']['null_var']).shape[-1] == len(names)
    out = sc.run_corrections(snap, lambda _t: None, loader_for(events))
    assert out['groups'][sc.ALL_GROUP]['result'].fluor_names == names


def test_exposed_settings_are_fix_settings_fields():
    defaults = sc.fm.FixSettings()
    for entry in sc.EXPOSED_SETTINGS:
        assert len(entry) == 7
        key, _label, kind, lo, hi, step, _decimals = entry
        assert hasattr(defaults, key)
        if isinstance(kind, tuple):
            assert getattr(defaults, key) in [value for value, _text in kind]
        elif kind in ('int', 'float'):
            assert lo <= getattr(defaults, key) <= hi and 0 < step <= hi - lo


def test_smooth_updates_off_runs_one_pass(data, tmp_path):
    ctrl = make_controller(data, tmp_path)
    setup = sc.CorrectionSetup(samples=[G1A, G1B], gate='root', n_events=30_000, fix={'smooth_updates': False})
    snap = sc.correction_snapshot(ctrl, setup, {sc.ALL_GROUP: UNST})
    res = sc.run_corrections(snap, lambda _t: None, loader_for(data[2]))['groups'][sc.ALL_GROUP]['result']
    assert res.smooth_log is None
    target = ERRORS['Raw/G1'][0]
    moved = dict(zip(res.fluor_names, sc.fm.row_angle(res.spectra_start, res.spectra)))
    assert res.accepted[res.fluor_names.index(target)]
    assert max(m for n, m in moved.items() if n != target) < 0.1 * moved[target]


def test_smooth_rounds_unmix_with_libraries_rebuilt_for_their_spectra(run, data):
    _ctrl, snap, out = run
    events = data[2]
    assert out['groups']['Raw/G1']['result'].smooth_log                # the default runs smooth updates
    fl_names = [snap['names'][i] for i in snap['fl']]
    lib = snap['libraries'][('P',)]
    paths = [G1A, G1B]
    raws = [events[p].astype(np.float32) for p in paths]
    background = sc.fm.background_for([lib['spectra']], snap['spectra'].shape[1])
    project = sc._smooth_projector(snap, UNST, events[UNST], paths, raws, {('P',): 0}, background, fl_names, None)
    keep = np.arange(0, 30_000, 7)
    base = snap['spectra'][snap['fl']]
    _u, st, _bg = project(base, keep)
    first = sc._project(snap, G1A, raws[0][keep[keep < 15_000]].astype(np.float64))
    assert np.array_equal(st['abundance'][:len(first['abundance'])], first['abundance'])
    moved = out['groups']['Raw/G1']['result'].spectra
    u, _st, _bg = project(moved, keep)
    env = um.rebuilt_envelope(('P',), snap['af_profiles'], fl_names, moved, snap['experiment_dir'],
                              snap['layout']['ceiling'])
    expected = sc.fm.project_events(events[UNST], moved, build_af_precomputed(moved, lib['spectra'], env),
                                    lib['spectra'])
    assert np.array_equal(u['abundance'], expected['abundance'])
    assert not np.allclose(env['null_var'], lib['precomputed']['envelope']['null_var'])


# ---------------------------------------------------------------------------
# What follows the corrected spectra
# ---------------------------------------------------------------------------

def test_envelope_rebuilt_for_a_corrected_set(run, data):
    ctrl, _snap, out = run
    truth, names, events = data
    process = ctrl.experiment.process
    sc.store_corrections(process, out)
    state = um.method_state(ctrl)
    af = resolve_af_for_profiles(ctrl, ['P'])
    set_af = um._set_af(state, 'Raw/G1', af)
    corrected = state.spectra_for('Raw/G1')
    sample = np.load(ctrl.experiment_dir / 'AutoSpectral' / 'P unstained.npy').astype(np.float64)
    expected = positivity_null_variance(sample, corrected, syn.af_library())
    np.testing.assert_allclose(set_af.precomputed['envelope']['null_var'][0], expected)
    assert not np.allclose(set_af.precomputed['envelope']['null_var'], af.precomputed['envelope']['null_var'])
    assert um._set_af(state, '', af) is af and um._set_af(state, 'Raw/G1', af) is set_af
    assert set(state.set_extras) == {'Raw/G1', 'Raw/G2'}
    assert state.set_extras['Raw/G1']['spread'] is None       # no controls, no base spread here


def test_spread_projection_sets_reuse_each_control_residual():
    rng = np.random.default_rng(9)
    truth, _names = syn.panel()
    abundance = np.zeros((3000, 8))
    abundance[:, 2] = rng.lognormal(np.log(30_000), 0.6, 3000)
    signal = abundance @ truth
    pos = signal + rng.normal(size=signal.shape) * np.sqrt(signal) + rng.normal(0, 50, signal.shape)
    moved = truth.copy()
    moved[3] = syn.inject(truth, _names, [('Dye 3', 'Dye 2', 0.1)])[3]
    row = spread_model.residual_model_spread_one(pos, None, None, None, truth, 2, int(np.argmax(truth[2])), 1000.0,
                                                 None, False, syn.CEILING, projection_spectra={'b': truth, 'm': moved})
    np.testing.assert_array_equal(row['sets']['b'], row['slope'])
    assert not np.allclose(np.nan_to_num(row['sets']['m']), np.nan_to_num(row['slope']))


def test_variants_move_with_the_correction():
    truth, names = syn.panel()
    v = np.vstack([truth[4], truth[4] * 0.9 + 0.1 * truth[5]])
    meta = {'Dye 4': {'v_mats': v, 'delta': v - v[:1]}, 'Dye 1': {'v_mats': truth[1:2], 'delta': np.zeros((1, 40))}}
    moved = truth.copy()
    moved[4] = syn.inject(truth, names, [('Dye 4', 'Dye 3', 0.1)])[4]
    out = um.shifted_variants(meta, names, truth, moved)
    np.testing.assert_allclose(out['Dye 4']['v_mats'][0], moved[4])
    np.testing.assert_allclose(out['Dye 4']['delta'][0], 0.0)
    assert out['Dye 1'] is meta['Dye 1'] and meta['Dye 4']['v_mats'] is v


def test_no_af_negative_variances_follow_the_spectra():
    rng = np.random.default_rng(3)
    truth, names = syn.panel()
    neg = syn.af_cells(5000, rng)
    stored = gu.robust_sd(neg @ np.linalg.pinv(truth)) ** 2
    state = gu.NoiseState(model=None, entry={}, cell_null_var=stored, bead_null_var=None, cell_negatives=neg)
    cell, bead = gu.null_variances(state, truth, truth)
    assert cell is stored and bead is None
    moved = truth.copy()
    moved[4] = syn.inject(truth, names, [('Dye 4', 'Dye 3', 0.2)])[4]
    cell, _bead = gu.null_variances(state, truth, moved)
    np.testing.assert_allclose(cell, gu.robust_sd(neg @ np.linalg.pinv(moved)) ** 2, rtol=1e-9, atol=1e-6)
    old = gu.NoiseState(model=None, entry={}, cell_null_var=stored, bead_null_var=None)
    assert gu.null_variances(old, truth, moved)[0] is stored


def test_threshold_shift_from_the_unstained_subsample(data, tmp_path):
    truth, names, _events = data
    ctrl = make_controller(data, tmp_path)
    assert np.array_equal(um.threshold_shift(ctrl, UNST, names, truth, truth), np.zeros(8))
    moved = truth.copy()
    moved[4] = syn.inject(truth, names, [('Dye 4', 'Dye 3', 0.2)])[4]
    shift = um.threshold_shift(ctrl, UNST, names, truth, moved)
    assert shift.shape == (8,) and np.any(shift != 0)
    assert np.array_equal(um.threshold_shift(ctrl, 'Raw/none.fcs', names, truth, moved), np.zeros(8))


# ---------------------------------------------------------------------------
# Envelopes built from a profile's source sample
# ---------------------------------------------------------------------------

def test_envelope_needs_source(data, tmp_path):
    truth, names, _events = data
    ctrl = make_controller(data, tmp_path)
    profile = ctrl.experiment.process['af_profiles']['P']
    assert not envelope_needs_source(profile, names, truth, tmp_path)
    assert envelope_needs_source({'spectra': profile['spectra']}, names, truth, tmp_path)
    moved = truth.copy()
    moved[0] = np.roll(moved[0], 2)
    assert not envelope_needs_source(profile, names, moved, tmp_path)        # the subsample rebuilds it
    lost = dict(profile, envelope=dict(profile['envelope'], unstained_file='gone.npy'))
    assert not envelope_needs_source(lost, names, truth, tmp_path)           # still current
    assert envelope_needs_source(lost, names, moved, tmp_path)


def test_envelope_built_from_the_source_sample(data, tmp_path):
    truth, names, events = data
    ctrl = make_controller(data, tmp_path)
    (tmp_path / UNST).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / UNST).write_bytes(b'')
    profile = {'spectra': ctrl.experiment.process['af_profiles']['P']['spectra'], 'source_fcs': UNST}
    job = {'experiment_dir': tmp_path, 'profiles': {'P': profile, 'Q': {'spectra': profile['spectra'],
                                                                      'source_fcs': 'Raw/missing.fcs'}},
           'layout': None, 'fluor_names': names, 'fluor_spectra': truth}
    built, failed = eb.build_envelopes(job, lambda _t: None, loader_for(events))
    entry, sample = built['P']
    ref, ref_sample = make_envelope_entry(events[UNST], names, truth, syn.af_library(), 'P unstained.npy')
    assert entry == ref and np.array_equal(sample, ref_sample)
    assert set(failed) == {'Q'}

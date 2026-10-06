"""
test_sample_loader.py
---------------------
Pure-numpy tests for controller_components/sample_loader.py: per-sample AF
resolution in the snapshot, keep-mask indexing, and reproducible event caps.

No FCS files, Qt or controller are needed: a minimal controller stand-in
provides the attributes the snapshot reads, and ``read_raw_events`` is
replaced with an in-memory reader.

Usage:
    pytest tests/test_sample_loader.py -m numpy_only
"""

from types import SimpleNamespace

import numpy as np
import pytest

from honeychrome.controller_components import sample_loader as sl
from honeychrome.controller_components.autospectral_functions import (
    precompute_af_matrices,
    precompute_joint_cov_extras,
)

N_CH = 6
FL_IDS = [0, 1, 2, 3]


def _spectra():
    rng = np.random.default_rng(0)
    fluor = np.abs(rng.normal(size=(2, len(FL_IDS)))) + 0.1
    af_a = np.abs(rng.normal(size=(1, len(FL_IDS)))) + 0.1
    af_b = np.abs(rng.normal(size=(2, len(FL_IDS)))) + 0.1
    return fluor, af_a, af_b


def _cache_entry(fluor, af):
    pre = precompute_af_matrices(fluor, af)
    pre.update(precompute_joint_cov_extras(pre, af))
    return pre


def make_controller():
    fluor, af_a, af_b = _spectra()
    pnn = [f'C{i}' for i in range(N_CH)]
    settings = {
        'raw': {
            'event_channels_pnn': pnn,
            'whitelisted_pnn': None,
            'fluorescence_channel_ids': FL_IDS,
        },
        'unmixed': {
            'event_channels_pnn': ['F1', 'F2', 'S1', 'S2'],
            'fluorescence_channel_ids': [0, 1],
        },
    }
    tm = np.zeros((N_CH, 4))
    tm[:4, :2] = np.linalg.pinv(fluor)
    tm[4, 2] = 1.0
    tm[5, 3] = 1.0
    experiment = SimpleNamespace(
        settings=settings,
        process={
            'spillover': np.eye(2),
            'af_profiles': {'A': {'spectra': af_a.tolist()}, 'B': {'spectra': af_b.tolist()}},
        },
        samples={
            'all_samples': {'Raw/s1.fcs': 's1', 'Raw/s2.fcs': 's2', 'Raw/s3.fcs': 's3',
                            'Raw/ctrl.fcs': 'ctrl', 'Raw/unst.fcs': 'unst'},
            'sample_af_profiles': {'Raw/s1.fcs': ['A'], 'Raw/s2.fcs': ['A', 'B']},
            'single_stain_controls': ['Raw/ctrl.fcs'],
            'unstained_samples': ['Raw/unst.fcs'],
        },
    )
    return SimpleNamespace(
        experiment=experiment,
        transfer_matrix=tm,
        filtered_raw_fluorescence_channel_ids=FL_IDS,
        af_precomputed_cache={'A': _cache_entry(fluor, af_a), 'B': _cache_entry(fluor, af_b)},
    )


def _fake_reader(n_events):
    rng = np.random.default_rng(1)
    data = np.abs(rng.normal(size=(n_events, N_CH))) * 100
    data[:, -1] = np.arange(n_events)

    def reader(abs_path, snap):
        return data.copy()
    return reader, data


@pytest.mark.numpy_only
def test_snapshot_resolves_each_samples_own_af():
    ctrl = make_controller()
    snap = sl.snapshot_unmix_state(ctrl)
    assert snap.af_for('Raw/s1.fcs').profile_names == ('A',)
    assert snap.af_for('Raw/s2.fcs').profile_names == ('A', 'B')
    assert snap.af_for('Raw/s2.fcs').n_spectra == 3
    assert snap.af_for('Raw/s3.fcs') is None
    assert snap.max_af_spectra() == 3
    tm, pre, spec = snap.legacy_af_state('Raw/s3.fcs')
    assert tm is ctrl.transfer_matrix and pre is None and spec is None


@pytest.mark.numpy_only
def test_snapshot_shares_combined_library_between_same_assignments():
    ctrl = make_controller()
    ctrl.experiment.samples['sample_af_profiles']['Raw/s3.fcs'] = ['A', 'B']
    snap = sl.snapshot_unmix_state(ctrl)
    assert snap.af_for('Raw/s2.fcs') is snap.af_for('Raw/s3.fcs')
    assert 'af_error_weights' in snap.af_for('Raw/s2.fcs').precomputed


@pytest.mark.numpy_only
def test_snapshot_is_isolated_from_later_controller_changes():
    ctrl = make_controller()
    snap = sl.snapshot_unmix_state(ctrl)
    old_tm = ctrl.transfer_matrix
    ctrl.transfer_matrix = np.zeros_like(old_tm)
    ctrl.experiment.samples['sample_af_profiles']['Raw/s1.fcs'] = ['B']
    ctrl.experiment.settings['raw']['fluorescence_channel_ids'] = []
    assert snap.transfer_matrix is old_tm
    assert snap.af_for('Raw/s1.fcs').profile_names == ('A',)
    assert snap.settings['raw']['fluorescence_channel_ids'] == FL_IDS


@pytest.mark.numpy_only
def test_uncached_profile_falls_back_to_plain_unmixing():
    ctrl = make_controller()
    ctrl.experiment.samples['sample_af_profiles']['Raw/s3.fcs'] = ['missing']
    snap = sl.snapshot_unmix_state(ctrl)
    assert snap.af_for('Raw/s3.fcs') is None


@pytest.mark.numpy_only
def test_load_unmixed_uses_per_sample_af(monkeypatch):
    ctrl = make_controller()
    reader, raw = _fake_reader(200)
    monkeypatch.setattr(sl, 'read_raw_events', reader)
    snap = sl.snapshot_unmix_state(ctrl)

    with_af = sl.load_unmixed('/exp', 'Raw/s1.fcs', snap, return_af=True)
    plain = sl.load_unmixed('/exp', 'Raw/s3.fcs', snap, return_af=True)

    assert with_af['af_profiles'] == ('A',)
    assert np.all(with_af['af_idx'] == 1)
    assert plain['af_profiles'] == ()
    assert np.all(plain['af_idx'] == 0) and np.all(plain['af_scale'] == 0)
    np.testing.assert_allclose(plain['unmixed'], raw @ ctrl.transfer_matrix)
    assert not np.allclose(with_af['unmixed'][:, :2], plain['unmixed'][:, :2])
    np.testing.assert_allclose(with_af['unmixed'][:, 2:], plain['unmixed'][:, 2:])


@pytest.mark.numpy_only
def test_keep_mask_indexes_original_rows(monkeypatch):
    ctrl = make_controller()
    reader, raw = _fake_reader(100)
    monkeypatch.setattr(sl, 'read_raw_events', reader)
    snap = sl.snapshot_unmix_state(ctrl)
    mask = np.zeros(100, dtype=bool)
    mask[10:30] = True
    out = sl.load_unmixed('/exp', 'Raw/s3.fcs', snap, keep_mask=mask)
    np.testing.assert_array_equal(out['event_index'], np.arange(10, 30))
    assert out['n_events_file'] == 100 and out['n_events_kept'] == 20
    np.testing.assert_allclose(out['unmixed'][:, 3], np.arange(10, 30))

    with pytest.raises(ValueError, match='keep-mask'):
        sl.load_unmixed('/exp', 'Raw/s3.fcs', snap, keep_mask=np.ones(5, dtype=bool))


@pytest.mark.numpy_only
def test_seeded_cap_is_reproducible_and_order_independent(monkeypatch):
    ctrl = make_controller()
    reader, _ = _fake_reader(1000)
    monkeypatch.setattr(sl, 'read_raw_events', reader)
    snap = sl.snapshot_unmix_state(ctrl)
    a1 = sl.load_unmixed('/exp', 'Raw/s1.fcs', snap, max_events=100)['event_index']
    b = sl.load_unmixed('/exp', 'Raw/s2.fcs', snap, max_events=100)['event_index']
    a2 = sl.load_unmixed('/exp', 'Raw/s1.fcs', snap, max_events=100)['event_index']
    np.testing.assert_array_equal(a1, a2)
    assert len(a1) == 100 and np.all(np.diff(a1) > 0)
    assert not np.array_equal(a1, b)
    other_seed = sl.load_unmixed('/exp', 'Raw/s1.fcs', snap, max_events=100, seed=7)
    assert not np.array_equal(a1, other_seed['event_index'])


@pytest.mark.numpy_only
def test_cap_applies_after_keep_mask(monkeypatch):
    ctrl = make_controller()
    reader, _ = _fake_reader(500)
    monkeypatch.setattr(sl, 'read_raw_events', reader)
    snap = sl.snapshot_unmix_state(ctrl)
    mask = np.zeros(500, dtype=bool)
    mask[::2] = True
    out = sl.load_unmixed('/exp', 'Raw/s3.fcs', snap, keep_mask=mask, max_events=50)
    assert len(out['event_index']) == 50
    assert np.all(out['event_index'] % 2 == 0)


@pytest.mark.numpy_only
def test_analysis_keys_exclude_controls_by_default():
    ctrl = make_controller()
    assert sl.analysis_sample_keys(ctrl) == ['Raw/s1.fcs', 'Raw/s2.fcs', 'Raw/s3.fcs']
    assert len(sl.analysis_sample_keys(ctrl, include_controls=True)) == 5


@pytest.mark.numpy_only
def test_sample_key_for_path():
    assert sl.sample_key_for_path('/exp', '/exp/Raw/a.fcs') == 'Raw/a.fcs'
    assert sl.sample_key_for_path('/exp', 'Raw/a.fcs') == 'Raw/a.fcs'

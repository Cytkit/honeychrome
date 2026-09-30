"""
test_af_channels.py
-------------------
Pure-numpy tests for the per-cell AF channels (AF Abundance, AF Index):
channel-list migration and default transforms, filling by AF-corrected
unmixing (zero under plain unmixing), the spectral-model antigen map, and
plain-number ticks for small linear ranges.

Usage:
    pytest tests/test_af_channels.py -m numpy_only
"""

import numpy as np
import pytest

import honeychrome.settings as hc_settings
from honeychrome.controller_components import functions as fn
from honeychrome.controller_components.autospectral_functions import (
    apply_af_transfer,
    precompute_af_matrices,
    precompute_joint_cov_extras,
)
from honeychrome.controller_components.transform import Transform

AF_A = hc_settings.af_abundance_channel
AF_I = hc_settings.af_index_channel


def _unmixed_settings(with_af=False):
    pnn = ['Time', 'event_id', 'FSC-A', 'SSC-A', 'F1', 'F2']
    if with_af:
        pnn += [AF_A, AF_I]
    return {
        'event_channels_pnn': pnn,
        'scatter_channel_ids': [2, 3],
        'fluorescence_channel_ids': [4, 5],
        'width_channels': [],
        'magnitude_ceiling': 262144,
        'width_ceiling': 50000,
        'default_ceiling': 60,
    }


@pytest.mark.numpy_only
def test_ensure_af_channels_appends_once_without_moving_existing_channels():
    st = _unmixed_settings()
    before = list(st['event_channels_pnn'])
    transforms = {'F1': {'id': 1}}
    assert fn.ensure_af_channels(st, transforms, n_af_spectra=37)
    assert st['event_channels_pnn'][:len(before)] == before
    assert st['event_channels_pnn'][-2:] == [AF_A, AF_I]
    assert st['af_channel_ids'] == [6, 7]
    assert st['fluorescence_channel_ids'] == [4, 5]
    assert transforms['F1'] == {'id': 1}
    assert transforms[AF_A]['id'] == 1 and transforms[AF_A]['scale_t'] == 262144
    assert transforms[AF_I]['id'] == 0 and transforms[AF_I]['scale_t'] == 37
    assert transforms[AF_I]['linear_a'] == 0 and transforms[AF_I]['limits'] == [0, 1]
    snapshot = (list(st['event_channels_pnn']), dict(transforms))
    assert not fn.ensure_af_channels(st, transforms, n_af_spectra=99)
    assert (st['event_channels_pnn'], transforms) == snapshot


@pytest.mark.numpy_only
def test_ensure_af_channels_ignores_unset_channel_list_and_keeps_user_transforms():
    empty = {'event_channels_pnn': None}
    assert not fn.ensure_af_channels(empty, {})
    assert empty == {'event_channels_pnn': None}
    st = _unmixed_settings(with_af=True)
    custom = {AF_I: {'id': 0, 'scale_t': 12, 'linear_a': 0, 'limits': [0, 1]}}
    fn.ensure_af_channels(st, custom)
    assert custom[AF_I]['scale_t'] == 12
    assert st['af_channel_ids'] == [6, 7]


@pytest.mark.numpy_only
def test_default_transforms_for_af_channels():
    st = _unmixed_settings(with_af=True)
    tr = fn.assign_default_transforms(st, n_af_spectra=None)
    assert tr[AF_I]['scale_t'] == hc_settings.default_af_index_range
    assert tr[AF_A]['id'] == 1
    assert tr['F1']['id'] == 1 and tr['FSC-A']['id'] == 0
    assert fn.assign_default_transforms(st, n_af_spectra=5)[AF_I]['scale_t'] == 5


def _af_setup(with_af_channels=True):
    rng = np.random.default_rng(0)
    n_raw = 6
    fl_raw = [0, 1, 2, 3]
    fluor = np.abs(rng.normal(size=(2, 4))) + 0.1
    af = np.abs(rng.normal(size=(3, 4))) + 0.1
    pre = precompute_af_matrices(fluor, af)
    pre.update(precompute_joint_cov_extras(pre, af))
    unmixed = _unmixed_settings(with_af=with_af_channels)
    n_out = len(unmixed['event_channels_pnn'])
    tm = np.zeros((n_raw, n_out))
    tm[:4, 4:6] = np.linalg.pinv(fluor)
    tm[4, 2] = 1.0
    tm[5, 3] = 1.0
    settings = {'raw': {'fluorescence_channel_ids': fl_raw}, 'unmixed': unmixed}
    raw = np.abs(rng.normal(size=(500, n_raw))) * 100
    raw[:, :4] += rng.random((500, 1)) * af[rng.integers(0, 3, 500)] * 50
    return raw, tm, pre, af, settings


@pytest.mark.numpy_only
def test_af_transfer_fills_af_channels_and_plain_unmixing_leaves_them_zero():
    raw, tm, pre, af, settings = _af_setup()
    result = apply_af_transfer(raw, tm, pre, af, settings, filtered_fl_ids_raw=[0, 1, 2, 3])
    out = result['unmixed']
    pnn = settings['unmixed']['event_channels_pnn']
    np.testing.assert_array_equal(out[:, pnn.index(AF_A)], result['af_scale'])
    np.testing.assert_array_equal(out[:, pnn.index(AF_I)], result['af_idx'])
    assert out[:, pnn.index(AF_I)].min() >= 1 and out[:, pnn.index(AF_I)].max() <= len(af)
    plain = fn.apply_transfer_matrix(tm, raw)
    assert not plain[:, [pnn.index(AF_A), pnn.index(AF_I)]].any()
    np.testing.assert_array_equal(out[:, 2:4], plain[:, 2:4])


@pytest.mark.numpy_only
def test_af_transfer_without_af_channels_is_unchanged():
    raw, tm, pre, af, settings = _af_setup(with_af_channels=False)
    result = apply_af_transfer(raw, tm, pre, af, settings, filtered_fl_ids_raw=[0, 1, 2, 3])
    assert result['unmixed'].shape == (len(raw), 6)


@pytest.mark.numpy_only
def test_build_antigen_map_uses_spectral_model():
    model = [{'label': 'F1', 'antigen': ' CD4 '}, {'label': 'F2', 'antigen': ''},
             {'label': 'Other', 'antigen': 'CD8'}]
    got = fn.build_antigen_map(['FSC-A', 'F1', 'F2', AF_A], model)
    assert got == {'FSC-A': '', 'F1': 'CD4', 'F2': '', AF_A: ''}
    assert fn.build_antigen_map(None, model) == {}


class _Linear:
    """x / t, as FlowKit's LinearTransform with param_a = 0."""

    def __init__(self, t):
        self.t = float(t)

    def apply(self, v):
        return np.asarray(v, dtype=float) / self.t

    def inverse(self, v):
        return np.asarray(v, dtype=float) * self.t


def _ticks(t):
    tr = Transform(scale_t=t, linear_a=0)
    tr.xform = _Linear(t)
    tr.limits = [0, 1]
    minor, major = tr.linear_ticks()
    return minor, major


@pytest.mark.numpy_only
def test_small_linear_range_gets_plain_number_ticks():
    _minor, major = _ticks(100)
    assert [lbl for _p, lbl in major] == ['0', '20', '40', '60', '80', '100']
    assert major[-1][0] == pytest.approx(1.0)
    _minor, major7 = _ticks(7)
    assert [lbl for _p, lbl in major7] == [str(i) for i in range(8)]
    minor, _major = _ticks(3)
    assert all(0 <= p <= 1 + 1e-9 for p, _l in minor)


@pytest.mark.numpy_only
def test_wide_linear_range_keeps_power_of_ten_labels():
    _minor, major = _ticks(262144)
    labels = [lbl for _p, lbl in major if lbl]
    assert labels and all('10' in lbl for lbl in labels if lbl != '0')

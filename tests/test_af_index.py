"""
test_af_index.py
----------------
Experiment-wide AF Index: ranking of each profile's spectra by spectral
angle to the profile mean, numbering across all stored profiles, and its
use in the AF Index channel (apply_af_transfer, sample_loader), the AF
Index axis, AF abundance clamping, and the import-time exclusion of
Honeychrome's derived channels from the detector list.

Usage:
    pytest tests/test_af_index.py -m numpy_only
"""

from types import SimpleNamespace

import numpy as np
import pytest

import honeychrome.settings as hc_settings
from honeychrome.controller_components import functions as fn
from honeychrome.controller_components import sample_loader as sl
from honeychrome.controller_components.autospectral_functions import (
    af_index_lookup,
    apply_af_transfer,
    apply_af_unmixing,
    precompute_af_matrices,
    precompute_joint_cov_extras,
    rank_af_spectra,
)
from honeychrome.controller_components.import_fcs_controller import unrecognised_fluorescence_ids

import af_synthetic as syn

AF_A = hc_settings.af_abundance_channel
AF_I = hc_settings.af_index_channel


def _cosine_to_mean(spectra):
    mean = spectra.mean(axis=0)
    return (spectra @ mean) / (np.linalg.norm(spectra, axis=1) * np.linalg.norm(mean))


def _precomputed(fl, af):
    pre = precompute_af_matrices(fl, af)
    pre.update(precompute_joint_cov_extras(pre, af))
    return pre


def _transfer_matrix():
    """(n_raw, n_unmixed) plain OLS transfer matrix for the synthetic panel."""
    fl = syn.fluor_spectra()
    tm = np.zeros((len(syn.UNMIXED_PNN), len(syn.RAW_PNN)))
    tm[np.ix_(range(4, 4 + syn.N_FLUOR), range(3, 3 + syn.N_DET))] = np.linalg.solve(fl @ fl.T, fl)
    tm[0, 0] = tm[2, 1] = tm[3, 2] = 1.0
    return tm.T


# ---------------------------------------------------------------------------
# Ranking and numbering
# ---------------------------------------------------------------------------

@pytest.mark.numpy_only
def test_rank_orders_spectra_by_spectral_angle_to_the_profile_mean():
    af = syn.af_spectra()
    order = rank_af_spectra(af)
    assert sorted(order.tolist()) == list(range(len(af)))
    cos = _cosine_to_mean(af)
    assert np.all(np.diff(cos[order]) <= 1e-12), 'closest to the mean must come first'


@pytest.mark.numpy_only
def test_rank_puts_the_mean_spectrum_first():
    """get_af_spectra() stores the population mean as row 0; ranked against
    the profile mean, a row equal to the mean of the others comes first."""
    others = syn.af_spectra()[[3, 1, 2]]
    spectra = np.vstack([others[0], others[1], others.mean(axis=0), others[2]])
    assert rank_af_spectra(spectra)[0] == 2


@pytest.mark.numpy_only
def test_lookup_numbers_every_profile_across_the_experiment():
    other, af = syn.other_af_spectra(), syn.af_spectra()
    profiles = {'Other profile': {'spectra': other.tolist()}, 'AF profile': {'spectra': af.tolist()}}

    first = af_index_lookup(profiles, ['Other profile'])
    second = af_index_lookup(profiles, ['AF profile'])
    both = af_index_lookup(profiles, ['AF profile', 'Other profile'])

    assert sorted(first.tolist()) == list(range(1, syn.N_OTHER_AF + 1))
    assert sorted(second.tolist()) == list(range(syn.N_OTHER_AF + 1, syn.N_AF_TOTAL + 1))
    np.testing.assert_array_equal(both, np.concatenate([second, first]))
    # Within a profile, index order is spectral-angle order.
    cos = _cosine_to_mean(af)
    assert np.all(np.diff(cos[np.argsort(second)]) <= 1e-12)


@pytest.mark.numpy_only
def test_lookup_skips_unknown_names_and_handles_empty_assignments():
    profiles = {'AF profile': {'spectra': syn.af_spectra().tolist()}}
    assert af_index_lookup(profiles, []) is None
    assert af_index_lookup(profiles, ['missing']) is None
    assert af_index_lookup({}, ['AF profile']) is None
    assert len(af_index_lookup(profiles, ['missing', 'AF profile'])) == syn.N_AF


@pytest.mark.numpy_only
def test_adding_a_profile_leaves_existing_indices_unchanged():
    profiles = {'AF profile': {'spectra': syn.af_spectra().tolist()}}
    before = af_index_lookup(profiles, ['AF profile'])
    profiles['Later profile'] = {'spectra': syn.other_af_spectra().tolist()}
    np.testing.assert_array_equal(af_index_lookup(profiles, ['AF profile']), before)


# ---------------------------------------------------------------------------
# AF Index channel
# ---------------------------------------------------------------------------

@pytest.mark.numpy_only
def test_af_transfer_writes_experiment_wide_index_and_returns_local_index():
    fl, af = syn.fluor_spectra(), syn.af_spectra()
    raw = syn.raw_events(2000, seed=4)
    settings = {'raw': syn.raw_settings(), 'unmixed': syn.unmixed_settings()}
    index_map = np.array([7, 5, 4, 6])
    result = apply_af_transfer(raw, _transfer_matrix(), _precomputed(fl, af), af, settings,
                               af_index_map=index_map)
    col = syn.UNMIXED_PNN.index(AF_I)
    assert result['af_idx'].min() >= 1 and result['af_idx'].max() <= syn.N_AF
    np.testing.assert_array_equal(result['unmixed'][:, col], index_map[result['af_idx'] - 1])


@pytest.mark.numpy_only
def test_sample_loader_writes_experiment_wide_index():
    fl, af = syn.fluor_spectra(), syn.af_spectra()
    index_map = np.array([7, 5, 4, 6])
    af_state = sl.AFState(profile_names=('AF profile',), precomputed=_precomputed(fl, af),
                          spectra=af, index_map=index_map)
    snap = sl.UnmixSnapshot(
        transfer_matrix=_transfer_matrix(), pnn_raw=tuple(syn.RAW_PNN), whitelisted=False,
        fl_ids_raw=tuple(range(3, 3 + syn.N_DET)), spillover=None,
        settings={'raw': syn.raw_settings(), 'unmixed': syn.unmixed_settings()},
    )
    out = sl.unmix_events(syn.raw_events(1500, seed=6), snap, af_state, return_af=True)
    col = syn.UNMIXED_PNN.index(AF_I)
    np.testing.assert_array_equal(out['unmixed'][:, col], index_map[out['af_idx'] - 1])


@pytest.mark.numpy_only
def test_resolve_af_for_profiles_carries_the_index_map():
    fl = syn.fluor_spectra()
    profiles = {'Other profile': {'spectra': syn.other_af_spectra().tolist()},
                'AF profile': {'spectra': syn.af_spectra().tolist()}}
    controller = SimpleNamespace(
        af_precomputed_cache={n: _precomputed(fl, np.array(p['spectra'])) for n, p in profiles.items()},
        experiment=SimpleNamespace(process={'af_profiles': profiles}),
    )
    state = sl.resolve_af_for_profiles(controller, ['AF profile'])
    np.testing.assert_array_equal(state.index_map, af_index_lookup(profiles, ['AF profile']))


# ---------------------------------------------------------------------------
# AF Index axis
# ---------------------------------------------------------------------------

@pytest.mark.numpy_only
def test_sync_af_index_transform_resizes_only_the_af_index_axis():
    transforms = fn.assign_default_transforms(syn.unmixed_settings(), n_af_spectra=None)
    abundance_before = dict(transforms[AF_A])
    assert transforms[AF_I]['scale_t'] == hc_settings.default_af_index_range
    assert fn.sync_af_index_transform(transforms, syn.N_AF_TOTAL)
    assert transforms[AF_I]['scale_t'] == syn.N_AF_TOTAL
    assert transforms[AF_A] == abundance_before
    assert not fn.sync_af_index_transform(transforms, syn.N_AF_TOTAL)
    assert not fn.sync_af_index_transform({}, 5)


# ---------------------------------------------------------------------------
# AF abundance
# ---------------------------------------------------------------------------

@pytest.mark.numpy_only
def test_af_abundance_is_clamped_at_zero():
    fl, af = syn.fluor_spectra(), syn.af_spectra()
    raw = syn.raw_events(4000, seed=8, with_af=False)[:, 3:]
    pre = _precomputed(fl, af)
    out = apply_af_unmixing(raw, pre, af)
    assert np.all(out['af_scale'] >= 0)
    ols = raw @ pre['P'].T
    zero = out['af_scale'] == 0
    assert zero.any()
    np.testing.assert_allclose(out['unmixed'][zero], ols[zero])


@pytest.mark.numpy_only
def test_near_in_span_af_variant_abundance_uses_the_floored_denominator():
    """A variant almost inside the fluorophore span has a tiny out-of-span
    residual; its abundance is divided by that residual's self-dot floored at
    1% of the library's largest, so it cannot explode."""
    fl, lib = syn.fluor_spectra(), syn.near_in_span_af_library()
    pre = _precomputed(fl, lib)
    j = len(lib) - 1
    floor = 0.01 * pre['r_dots'].max()
    assert pre['r_dots'][j] < floor

    raw = syn.raw_events(20000, seed=9)[:, 3:]
    out = apply_af_unmixing(raw, pre, lib)
    cells = out['af_idx'] == j + 1
    assert cells.any()
    numerator = raw[cells] @ pre['r_library'][:, j]
    np.testing.assert_allclose(out['af_scale'][cells], np.maximum(numerator, 0.0) / floor)
    np.testing.assert_allclose(
        out['unmixed'][cells],
        raw[cells] @ pre['P'].T - out['af_scale'][cells, None] * pre['v_library'][:, j],
    )


# ---------------------------------------------------------------------------
# FCS import
# ---------------------------------------------------------------------------

@pytest.mark.numpy_only
def test_import_excludes_honeychrome_derived_channels_from_detectors():
    """Re-importing a Honeychrome unmixed export on an unrecognised
    cytometer must not treat event_id or the AF channels as detectors."""
    pnn = ['Time', 'event_id', 'FSC-A', 'SSC-A', 'SSC-B-A', 'FSC-Width',
           'BV421-A', 'PE-A', AF_A, AF_I]
    ids = unrecognised_fluorescence_ids(pnn, set(pnn))
    assert [pnn[i] for i in ids] == ['BV421-A', 'PE-A']
    assert unrecognised_fluorescence_ids(pnn, {'PE-A', AF_A}) == [pnn.index('PE-A')]

"""
test_af_negative_envelope_opt.py
--------------------------------
Positivity-weighted (negative-envelope) AF scoring in the AutoSpectral
Optimization plugin: the envelope reaches the compiled kernel, events with
saturated detectors are re-fitted on the unsaturated detectors, and the
kernel's AF choice matches a NumPy replica of its scoring.

Kernel-dependent tests skip when the extension is not built or predates the
positivity-weighted envelope.

Usage:
    pytest tests/test_af_negative_envelope_opt.py -m numpy_only
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from honeychrome.controller_components import autospectral_functions as asf

import af_synthetic as syn

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

import autospectral_optimization_functions as aof  # noqa: E402
from autospectral_opt_kernel_wrapper import (  # noqa: E402
    ENVELOPE_KERNEL_AVAILABLE,
    unmix_autospectral_joint,
)

requires_envelope_kernel = pytest.mark.skipif(
    not ENVELOPE_KERNEL_AVAILABLE,
    reason='compiled kernel with positivity-weighted envelope support not built '
           '(run bundled_plugins/build_autospectral_opt_kernel.py)',
)


def _envelope(n_fluor=syn.N_FLUOR, spread=None):
    null_var = np.zeros((1, 3, n_fluor))
    null_var[0, 0] = 15.0 ** 2
    null_var[0, 1] = 0.5
    null_var[0, 2] = 1e-4
    return {
        'null_var': null_var,
        'spread': spread,
        'mean_rows': np.array([0]),
        'ceiling': syn.MAGNITUDE_CEILING,
        'saturated_frac': 0.999,
    }


class _RecordingKernel:
    """Stands in for unmix_autospectral_joint: records each call and returns
    a result whose AF index is the number of detectors it was given."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        n = kwargs['raw_data_in'].shape[0]
        n_fluor = kwargs['spectra'].shape[0]
        out = np.zeros((n, n_fluor + 2))
        out[:, n_fluor] = 1.0
        out[:, n_fluor + 1] = kwargs['raw_data_in'].shape[1]
        return out


def _run(raw, envelope, variants_meta=None, active=()):
    kernel = _RecordingKernel()
    original_kernel = aof.unmix_autospectral_joint
    original_flag = aof.AUTOSPECTRAL_OPT_KERNEL_AVAILABLE
    aof.unmix_autospectral_joint = kernel
    aof.AUTOSPECTRAL_OPT_KERNEL_AVAILABLE = True
    try:
        result = aof.unmix_autospectral_optimization(
            raw_fl_events=raw,
            reference_spectra=syn.fluor_spectra(),
            fluor_names=syn.FLUOR_LABELS,
            af_spectra=syn.af_spectra(),
            variants_meta=variants_meta or {},
            active_labels=active,
            unmixed_pos_thresholds=np.zeros(syn.N_FLUOR),
            n_threads=1,
            envelope=envelope,
        )
    finally:
        aof.unmix_autospectral_joint = original_kernel
        aof.AUTOSPECTRAL_OPT_KERNEL_AVAILABLE = original_flag
    return result, kernel


def _raw(n=20, seed=0):
    return np.random.default_rng(seed).uniform(10.0, 1000.0, (n, syn.N_DET))


@pytest.mark.numpy_only
def test_envelope_reaches_the_kernel_and_unsaturated_events_run_once():
    envelope = _envelope()
    result, kernel = _run(_raw(), envelope)
    assert len(kernel.calls) == 1
    assert kernel.calls[0]['envelope'] is envelope
    assert result['unmixed'].shape == (20, syn.N_FLUOR)
    assert np.all(result['af_idx'] == syn.N_DET)


@pytest.mark.numpy_only
def test_without_an_envelope_there_is_no_saturation_refit():
    raw = _raw()
    raw[3, 5] = syn.MAGNITUDE_CEILING
    result, kernel = _run(raw, None)
    assert len(kernel.calls) == 1
    assert kernel.calls[0]['envelope'] is None
    assert np.all(result['af_idx'] == syn.N_DET)


@pytest.mark.numpy_only
def test_saturated_events_are_refitted_on_the_unsaturated_detectors():
    raw = _raw()
    raw[3, 5] = syn.MAGNITUDE_CEILING
    raw[4, 5] = syn.MAGNITUDE_CEILING
    raw[7, [1, 2]] = syn.MAGNITUDE_CEILING
    variants_meta = {'F1': {
        'v_mats': np.tile(syn.fluor_spectra()[0], (3, 1)),
        'delta': np.zeros((3, syn.N_DET)),
    }}
    result, kernel = _run(raw, _envelope(), variants_meta, active=['F1'])

    # One full call, then one per saturation pattern.
    assert len(kernel.calls) == 3
    by_width = {c['raw_data_in'].shape[1]: c for c in kernel.calls[1:]}
    one_detector = by_width[syn.N_DET - 1]
    assert one_detector['raw_data_in'].shape[0] == 2
    assert one_detector['spectra'].shape == (syn.N_FLUOR, syn.N_DET - 1)
    assert one_detector['af_spectra'].shape[1] == syn.N_DET - 1
    assert one_detector['variants'][0]['v_mats'].shape == (3, syn.N_DET - 1)
    assert one_detector['variants'][0]['delta_obs'].shape == (3, syn.N_DET - 1)
    assert not np.any(one_detector['raw_data_in'] >= syn.MAGNITUDE_CEILING)
    assert by_width[syn.N_DET - 2]['raw_data_in'].shape[0] == 1

    expected = np.full(20, syn.N_DET)
    expected[[3, 4]] = syn.N_DET - 1
    expected[7] = syn.N_DET - 2
    np.testing.assert_array_equal(result['af_idx'], expected)


@pytest.mark.numpy_only
def test_pattern_leaving_a_fluorophore_unidentifiable_keeps_the_first_result():
    raw = _raw()
    raw[2, : syn.N_DET - 2] = syn.MAGNITUDE_CEILING    # leaves 2 detectors for 3 fluorophores
    result, kernel = _run(raw, _envelope())
    assert len(kernel.calls) == 1
    assert result['af_idx'][2] == syn.N_DET


@pytest.mark.numpy_only
def test_combine_envelopes_stacks_each_profiles_null_variance_and_needs_every_profile():
    a = _envelope()
    b = _envelope()
    b['null_var'] = b['null_var'] * 2.0
    merged = asf.combine_envelopes([a, b], [4, 6])
    np.testing.assert_array_equal(merged['null_var'], np.concatenate([a['null_var'], b['null_var']]))
    np.testing.assert_array_equal(merged['mean_rows'], [0, 4])
    assert asf.combine_envelopes([a], [4]) is a
    assert asf.combine_envelopes([a, None], [4, 6]) is None
    assert asf.combine_envelopes([], []) is None


@pytest.mark.numpy_only
def test_af_envelope_for_profiles_combines_the_assigned_profiles(tmp_path):
    controller = syn.make_experiment(tmp_path, n_events=50)
    a, b = _envelope(), _envelope()
    b['null_var'] = b['null_var'] * 2.0
    v_library = np.zeros((syn.N_FLUOR, len(syn.af_spectra())))
    cached = {'AF profile': {'envelope': a, 'v_library': v_library},
              'Other profile': {'envelope': b, 'v_library': v_library},
              'No envelope': {'v_library': v_library}}
    controller.precomputed_for_profile = lambda name: cached[name]
    controller.experiment.process['af_profiles']['No envelope'] = {'spectra': syn.af_spectra().tolist()}

    both = aof.af_envelope_for_profiles(controller, ['AF profile', 'Other profile'])
    np.testing.assert_array_equal(both['null_var'], np.concatenate([a['null_var'], b['null_var']]))
    np.testing.assert_array_equal(both['mean_rows'], [0, len(syn.af_spectra())])
    assert aof.af_envelope_for_profiles(controller, ['AF profile']) is a
    assert aof.af_envelope_for_profiles(controller, ['AF profile', 'No envelope']) is None
    assert aof.af_envelope_for_profiles(controller, ['Not a profile']) is None
    assert aof.af_envelope_for_profiles(controller, []) is None


@requires_envelope_kernel
def test_kernel_scores_af_with_an_envelope():
    rng = np.random.default_rng(5)
    fl, af = syn.fluor_spectra(), syn.af_spectra()
    n = 2000
    true_j = rng.integers(0, len(af), n)
    raw = (rng.gamma(2.0, 400.0, n)[:, None] * af[true_j]
           + rng.gamma(0.6, 300.0, (n, syn.N_FLUOR)) @ fl
           + rng.normal(0.0, 20.0, (n, syn.N_DET)))
    spread = np.abs(rng.normal(0.0, 0.01, (syn.N_FLUOR, syn.N_FLUOR)))
    kwargs = dict(raw_data_in=raw, spectra=fl, af_spectra=af, fluor_names=syn.FLUOR_LABELS,
                  pos_thresholds=np.zeros(syn.N_FLUOR), variants=[], n_threads=1)
    flat = unmix_autospectral_joint(envelope=_envelope(), **kwargs)
    widened = unmix_autospectral_joint(envelope=_envelope(spread=spread), **kwargs)
    plain = unmix_autospectral_joint(**kwargs)
    for out in (flat, widened, plain):
        assert out.shape == (n, syn.N_FLUOR + 2)
        assert np.all(np.isfinite(out))
        assert out[:, -1].min() >= 1 and out[:, -1].max() <= len(af)


def _kernel_replica_af_index(raw, spectra, af, envelope=None):
    """
    AF index (0-based) that core_joint_unmix.cpp's AF scoring should choose
    with cell weighting off: joint L2, or with an envelope the
    positivity-weighted term on the kernel's own abundances and weights.
    """
    P = np.linalg.solve(spectra @ spectra.T, spectra)
    v = P @ af.T
    r = af.T - spectra.T @ v
    r_dots_w = np.sum(r * r, axis=0)
    r_dots = np.maximum(r_dots_w, 0.01 * max(r_dots_w.max(), 1e-10))
    w = np.sqrt(np.abs(np.diag(P @ np.cov(af, rowvar=False) @ P.T))) + 1e-8
    v_w = v * w[:, None]
    init = raw @ P.T
    base = raw - init @ spectra
    cross = raw @ r
    k = np.maximum(cross, 0.0) / r_dots
    base_resid = np.maximum(np.einsum('ij,ij->i', base, base), 1e-16)
    w_init = init * w
    base_fluor = np.maximum(np.einsum('ij,ij->i', w_init, w_init), 1e-16)
    presid = np.sqrt(np.maximum(base_resid[:, None] - 2 * k * cross + k * k * r_dots_w, 0.0)) / np.sqrt(base_resid)[:, None]
    pfluor = np.sqrt(np.maximum(base_fluor[:, None] - 2 * k * (w_init @ v_w) + k * k * np.sum(v_w * v_w, axis=0), 0.0))
    pfluor /= np.sqrt(base_fluor)[:, None]
    best = np.argmin(presid * pfluor, axis=1)
    if envelope is None:
        return best
    rows = np.arange(len(raw))
    mean_rows = np.asarray(envelope.get('mean_rows', [0]))
    pick = np.argmin(presid[:, mean_rows], axis=1)
    mean_j = mean_rows[pick]
    k_mean = k[rows, mean_j][:, None]
    position = init - k_mean * v.T[mean_j]
    coef = np.asarray(envelope['null_var'])[pick]
    var = coef[:, 0] + k_mean * (coef[:, 1] + k_mean * coef[:, 2])
    if envelope.get('spread') is not None:
        var = var + np.maximum(position, 0.0) @ envelope['spread']
    weight = w * asf.positivity_weight(position, var)
    base_p = np.einsum('ij,ij->i', weight, init * init)[:, None] + 1e-6
    term = np.maximum(base_p - 2 * k * ((weight * init) @ v) + k * k * (weight @ (v * v)), 0.0) / base_p
    score = (term + 1e-12) * (presid * presid + 1e-12)
    return np.argmax(score <= score.min(axis=1, keepdims=True) * (1.0 + 1e-9), axis=1)


@requires_envelope_kernel
def test_kernel_af_choice_matches_the_numpy_replica():
    rng = np.random.default_rng(8)
    fl, af = syn.fluor_spectra(), syn.af_spectra()
    n = 3000
    true_j = rng.integers(0, len(af), n)
    raw = (rng.gamma(2.0, 400.0, n)[:, None] * af[true_j]
           + rng.gamma(0.6, 300.0, (n, syn.N_FLUOR)) @ fl
           + rng.normal(0.0, 20.0, (n, syn.N_DET)))
    spread = np.abs(rng.normal(0.0, 0.05, (syn.N_FLUOR, syn.N_FLUOR)))
    np.fill_diagonal(spread, 0.0)
    kwargs = dict(raw_data_in=raw, spectra=fl, af_spectra=af, fluor_names=syn.FLUOR_LABELS,
                  pos_thresholds=np.zeros(syn.N_FLUOR), variants=[], n_threads=2)
    plain = unmix_autospectral_joint(**kwargs)[:, -1].astype(int) - 1
    np.testing.assert_array_equal(plain, _kernel_replica_af_index(raw, fl, af))
    two_means = dict(_envelope(spread=spread), mean_rows=np.array([0, len(af) - 1]))
    two_means['null_var'] = np.concatenate([two_means['null_var'], 4.0 * two_means['null_var']])
    for envelope in (_envelope(spread=spread), two_means):
        weighted = unmix_autospectral_joint(envelope=envelope, **kwargs)[:, -1].astype(int) - 1
        np.testing.assert_array_equal(weighted, _kernel_replica_af_index(raw, fl, af, envelope))
    assert np.any(weighted != plain)


@requires_envelope_kernel
def test_kernel_rejects_mean_rows_outside_the_af_library():
    fl, af = syn.fluor_spectra(), syn.af_spectra()
    raw = np.random.default_rng(1).uniform(10.0, 1000.0, (5, syn.N_DET))
    with pytest.raises(ValueError):
        unmix_autospectral_joint(raw_data_in=raw, spectra=fl, af_spectra=af, fluor_names=syn.FLUOR_LABELS,
                                 pos_thresholds=np.zeros(syn.N_FLUOR), variants=[], n_threads=1,
                                 envelope=dict(_envelope(), mean_rows=np.array([len(af)])))
    with pytest.raises(ValueError):
        unmix_autospectral_joint(raw_data_in=raw, spectra=fl, af_spectra=af, fluor_names=syn.FLUOR_LABELS,
                                 pos_thresholds=np.zeros(syn.N_FLUOR), variants=[], n_threads=1,
                                 envelope=dict(_envelope(), mean_rows=np.array([0, 1])))

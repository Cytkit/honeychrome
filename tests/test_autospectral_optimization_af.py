"""
test_autospectral_optimization_af.py
------------------------------------
AutoSpectral Optimization plugin against the current AF and SOM code:
the compiled joint kernel's AF-only mode versus the Python AF unmixing,
variant discovery through get_som_codes() (SOM kernel and KMeans
fallback), the saturation ceiling used during variant discovery, and the
AF channels in the Compare section's event arrays.

Kernel-dependent tests skip when _autospectral_opt_kernel is not built.

Usage:
    pytest tests/test_autospectral_optimization_af.py -m numpy_only
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import honeychrome.settings as hc_settings
from honeychrome.controller_components import autospectral_functions as asf
from honeychrome.controller_components.autospectral_functions import (
    apply_af_unmixing,
    precompute_af_matrices,
    precompute_joint_cov_extras,
)

import af_synthetic as syn

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

import autospectral_optimization_functions as aof  # noqa: E402
from autospectral_opt_kernel_wrapper import (  # noqa: E402
    AUTOSPECTRAL_OPT_KERNEL_AVAILABLE,
    unmix_autospectral_joint,
)

requires_kernel = pytest.mark.skipif(
    not AUTOSPECTRAL_OPT_KERNEL_AVAILABLE,
    reason='compiled AutoSpectral Optimization kernel not built '
           '(run bundled_plugins/build_autospectral_opt_kernel.py)',
)

AF_A = hc_settings.af_abundance_channel
AF_I = hc_settings.af_index_channel


def _af_mixture(n=20000, seed=1):
    """Raw fluorescence with known AF variant and abundance per cell."""
    rng = np.random.default_rng(seed)
    fl = syn.fluor_spectra()
    af = syn.af_spectra()
    true_j = rng.integers(0, len(af), n)
    true_k = rng.gamma(2.0, 400.0, n)
    abund = rng.gamma(0.6, 300.0, (n, len(fl))) * (rng.random((n, len(fl))) < 0.4)
    raw = abund @ fl + true_k[:, None] * af[true_j] + rng.normal(0.0, 20.0, (n, fl.shape[1]))
    return raw, fl, af, true_j


def _python_af(raw, fl, af):
    pre = precompute_af_matrices(fl, af)
    pre.update(precompute_joint_cov_extras(pre, af))
    return apply_af_unmixing(raw, pre, af)


def _positive_control(n=3000, seed=3):
    """Fluorophore F1 with three spectral variants on AF-bearing cells, plus
    a matched unstained population."""
    rng = np.random.default_rng(seed)
    ch = np.arange(syn.N_DET)
    af_base = np.exp(-0.5 * ((ch - 3) / 5.0) ** 2)
    af_base /= af_base.max()

    def cells(m):
        k = rng.gamma(2.0, 300.0, m)
        return k[:, None] * af_base * (1 + 0.1 * rng.normal(size=(m, 1))) + rng.normal(0, 15, (m, syn.N_DET))

    variants = np.array([np.exp(-0.5 * ((ch - 2 - 0.4 * s) / (1.5 + 0.2 * s)) ** 2) for s in range(3)])
    variants /= variants.max(axis=1, keepdims=True)
    neg = cells(4000)
    pos = cells(n) + rng.gamma(3.0, 3000.0, n)[:, None] * variants[rng.integers(0, 3, n)]
    return pos, rng.normal(1e5, 1e4, (n, 2)), neg, rng.normal(1e5, 1e4, (4000, 2))


def _discover_f1(**overrides):
    pos, pos_sc, neg, neg_sc = _positive_control()
    fl = syn.fluor_spectra()
    kwargs = dict(
        label='F1', pos_events_raw=pos, pos_scatter=pos_sc,
        neg_events_raw=neg, neg_scatter=neg_sc, is_cell_control=True,
        reference_spectra=fl, fluor_names=syn.FLUOR_LABELS, fluor_idx=0,
        peak_ch_idx=int(np.argmax(fl[0])),
        raw_pos_threshold=float(np.percentile(neg[:, int(np.argmax(fl[0]))], 99.5)),
        unmixed_pos_threshold=50.0,
        af_pcs=aof.compute_af_pcs_from_unstained(neg, n_pcs=4),
        saturation_ceiling=syn.MAGNITUDE_CEILING,
    )
    kwargs.update(overrides)
    return aof.discover_fluor_variants(**kwargs)


# ---------------------------------------------------------------------------
# Kernel versus Python AF unmixing
# ---------------------------------------------------------------------------

@requires_kernel
@pytest.mark.numpy_only
def test_kernel_af_only_mode_assigns_the_same_af_variant_as_python():
    """No active variants: the joint kernel reduces to per-cell AF
    extraction, which the Compare section shows next to the Python
    AF-corrected panel and which the two exporters write. Nearly every
    cell should get the same AF variant from both."""
    raw, fl, af, true_j = _af_mixture()
    py = _python_af(raw, fl, af)
    cpp = unmix_autospectral_joint(raw, fl, af, syn.FLUOR_LABELS, np.zeros(len(fl)), [], n_threads=2)

    cpp_idx = cpp[:, len(fl) + 1].astype(int)
    agree = cpp_idx == py['af_idx']
    assert agree.mean() >= 0.98, f'AF index agreement {agree.mean():.3f}'
    assert cpp_idx.min() >= 1 and cpp_idx.max() <= len(af)
    assert np.mean(cpp_idx - 1 == true_j) >= 0.9


@requires_kernel
@pytest.mark.numpy_only
def test_kernel_af_only_mode_matches_python_values_for_the_same_variant():
    """Where both pick the same AF variant, AF abundance and fluorophore
    values should be identical; both clamp the AF abundance at 0."""
    raw, fl, af, _true_j = _af_mixture()
    py = _python_af(raw, fl, af)
    cpp = unmix_autospectral_joint(raw, fl, af, syn.FLUOR_LABELS, np.zeros(len(fl)), [], n_threads=2)

    n_f = len(fl)
    agree = cpp[:, n_f + 1].astype(int) == py['af_idx']
    np.testing.assert_allclose(cpp[agree, n_f], py['af_scale'][agree], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cpp[agree, :n_f], py['unmixed'][agree], rtol=1e-6, atol=1e-6)


@requires_kernel
@pytest.mark.numpy_only
def test_kernel_and_python_agree_with_a_near_in_span_af_variant():
    """Both floor the abundance denominator for a variant almost inside the
    fluorophore span, so cells assigned it get the same bounded abundance."""
    fl, lib = syn.fluor_spectra(), syn.near_in_span_af_library()
    raw = syn.raw_events(20000, seed=9)[:, 3:]
    py = _python_af(raw, fl, lib)
    cpp = unmix_autospectral_joint(raw, fl, lib, syn.FLUOR_LABELS, np.zeros(len(fl)), [], n_threads=2)

    n_f = len(fl)
    agree = cpp[:, n_f + 1].astype(int) == py['af_idx']
    assert agree.mean() >= 0.98
    assert np.any(py['af_idx'][agree] == len(lib))
    np.testing.assert_allclose(cpp[agree, n_f], py['af_scale'][agree], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cpp[agree, :n_f], py['unmixed'][agree], rtol=1e-6, atol=1e-6)


@requires_kernel
@pytest.mark.numpy_only
def test_kernel_runs_with_discovered_variants():
    variants = _discover_f1()
    assert variants is not None
    raw, fl, af, _true_j = _af_mixture(n=4000)
    out = aof.unmix_autospectral_optimization(
        raw_fl_events=raw, reference_spectra=fl, fluor_names=syn.FLUOR_LABELS,
        af_spectra=af, variants_meta={'F1': variants}, active_labels={'F1'},
        unmixed_pos_thresholds=np.zeros(len(fl)), n_threads=1,
    )
    assert out['unmixed'].shape == (len(raw), len(fl))
    assert np.isfinite(out['unmixed']).all()
    assert out['af_idx'].min() >= 1 and out['af_idx'].max() <= len(af)
    assert np.all(out['af_scale'] >= 0)


# ---------------------------------------------------------------------------
# Variant discovery through get_som_codes()
# ---------------------------------------------------------------------------

@pytest.mark.numpy_only
@pytest.mark.parametrize('engine', ['kmeans', 'som'])
def test_discover_fluor_variants_with_each_clustering_engine(engine, monkeypatch):
    if engine == 'kmeans':
        monkeypatch.setattr(asf, '_load_som_kernel', lambda: None)
    elif asf._load_som_kernel() is None:
        pytest.skip('SOM kernel not built (run bundled_plugins/build_som_kernel.py)')

    result = _discover_f1()
    assert result is not None
    v, delta = result['v_mats'], result['delta']
    ref = syn.fluor_spectra()[0]
    assert v.shape[0] >= 2 and v.shape[1] == syn.N_DET
    np.testing.assert_allclose(v[0], ref)
    np.testing.assert_allclose(delta, v - ref)
    np.testing.assert_allclose(v.max(axis=1), 1.0, atol=1e-9)
    assert np.isfinite(v).all()


@pytest.mark.numpy_only
def test_discover_all_variants_uses_the_cytometer_saturation_ceiling(monkeypatch, tmp_path):
    """Saturation exclusion during variant discovery must use the
    instrument's ceiling (raw magnitude_ceiling, e.g. 4194304 on an Aurora),
    as the spectral controller does, not a fixed 2**18."""
    captured = {}
    rng = np.random.default_rng(0)
    controller = syn.FakeController(tmp_path)
    controller.experiment.settings['raw']['magnitude_ceiling'] = 4194304.0
    controller.experiment.samples['all_samples'] = {
        'Raw/Unstained cells.fcs': 'Unstained cells', 'Raw/F1.fcs': 'F1 control',
    }
    controller.experiment.process['spectral_model'] = [
        {'label': 'F1', 'sample_name': 'F1 control', 'particle_type': 'Cells'},
    ]

    monkeypatch.setattr(aof, 'sample_from_fcs', lambda path: None)
    monkeypatch.setattr(
        aof, 'get_raw_events',
        lambda sample, fl_ids, **kw: (rng.random((500, len(fl_ids))), rng.random((500, 2))),
    )
    monkeypatch.setattr(
        aof, '_unstained_threshold_entry',
        lambda raw, ref, names, is_cell, n_pcs, **kw: {
            'raw': np.zeros(raw.shape[1]), 'unmixed': np.zeros(len(ref)), 'af_pcs': None,
        },
    )

    def fake_discover(**kwargs):
        captured['saturation_ceiling'] = kwargs['saturation_ceiling']
        return None

    monkeypatch.setattr(aof, 'discover_fluor_variants', fake_discover)
    aof.discover_all_variants(controller)
    assert captured['saturation_ceiling'] == 4194304.0


# ---------------------------------------------------------------------------
# Compare section event arrays
# ---------------------------------------------------------------------------

@requires_kernel
@pytest.mark.numpy_only
def test_comparison_arrays_fill_af_channels(tmp_path):
    """The AF-corrected and Optimization comparison arrays carry per-cell
    AF Abundance and AF Index (as the main unmixed data does), so gates
    drawn on the AF channels select the same cells in the Compare plots."""
    from autospectral_optimization_tab import JointComparisonWorker

    controller = syn.FakeController(tmp_path)
    raw = syn.raw_events(3000, seed=5)
    pnn = syn.UNMIXED_PNN
    tm = np.zeros((len(pnn), len(syn.RAW_PNN)))
    unmixing = np.array(controller.experiment.process['unmixing_matrix'])
    fl_raw = list(range(3, 3 + syn.N_DET))
    tm[np.ix_(range(4, 4 + syn.N_FLUOR), fl_raw)] = unmixing
    tm[2, 1] = tm[3, 2] = tm[0, 0] = 1.0
    tm = tm.T

    index_map = controller.get_af_index_map_for_sample(syn.AF_SAMPLE)
    results, errors = [], []
    worker = JointComparisonWorker(
        raw, tm, fl_raw, np.arange(4, 4 + syn.N_FLUOR),
        syn.fluor_spectra(), syn.FLUOR_LABELS, syn.af_spectra(), {}, set(),
        np.zeros(syn.N_FLUOR), None, 10_000, {'n_threads': 1},
        unmixed_pnn=list(pnn), af_index_map=index_map,
    )
    worker.finished.connect(lambda *a: results.append(a))
    worker.error.connect(errors.append)
    worker.run()
    assert not errors, errors[0]

    ols, af_data, opt = results[0]
    a, i = pnn.index(AF_A), pnn.index(AF_I)
    assert not ols[:, [a, i]].any()
    for data in (af_data, opt):
        assert data[:, i].min() >= syn.N_OTHER_AF + 1 and data[:, i].max() <= syn.N_AF_TOTAL
        assert data[:, a].any()
    expected = _python_af(raw[:, fl_raw], syn.fluor_spectra(), syn.af_spectra())
    np.testing.assert_array_equal(af_data[:, i], index_map[expected['af_idx'] - 1])

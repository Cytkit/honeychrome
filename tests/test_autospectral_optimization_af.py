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


def _discover_f1(n=3000, **overrides):
    pos, pos_sc, neg, neg_sc = _positive_control(n=n)
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
# Variant deduplication
# ---------------------------------------------------------------------------

def _shifted(ref, shift, channel):
    v = ref.copy()
    v[channel] += shift
    return v


@pytest.mark.numpy_only
def test_deduplicate_variants_drops_reference_matches_and_keeps_the_more_populated():
    ref = syn.fluor_spectra()[0]
    v = np.vstack([
        ref,
        _shifted(ref, 0.005, 0),    # within 0.02 of the reference
        _shifted(ref, 0.050, 1),    # distinct
        _shifted(ref, 0.052, 1),    # within 0.02 of the row above
    ])
    kept, threshold = aof.deduplicate_variants(v, np.array([0, 10, 40, 90]), 0.02, 10, min_variants=0)
    assert kept.tolist() == [0, 3]
    assert threshold == 0.02


@pytest.mark.numpy_only
def test_deduplicate_variants_adds_back_the_furthest_to_reach_the_floor():
    ref = syn.fluor_spectra()[0]
    v = np.vstack([ref, _shifted(ref, 0.001, 0), _shifted(ref, 0.004, 1), _shifted(ref, 0.003, 2), _shifted(ref, 0.002, 3)])
    kept, threshold = aof.deduplicate_variants(v, np.ones(len(v)), 0.01, 10)
    assert kept.tolist() == [0, 2, 3]
    assert threshold == 0.01
    one, _ = aof.deduplicate_variants(v, np.ones(len(v)), 0.01, 10, min_variants=1)
    assert one.tolist() == [0, 2]


@pytest.mark.numpy_only
@pytest.mark.parametrize('min_distance', [0.0, 0.02])
def test_deduplicate_variants_raises_the_threshold_to_meet_the_cap(min_distance):
    ref = syn.fluor_spectra()[0]
    v = np.vstack([ref] + [_shifted(ref, 0.03 * (i + 1), 1) for i in range(30)])
    kept, threshold = aof.deduplicate_variants(v, np.ones(len(v)), min_distance, 5)
    assert kept[0] == 0
    assert 1 < len(kept) <= 6
    assert threshold > min_distance
    sub = v[kept]
    pair = np.abs(sub[:, None, :] - sub[None, :, :]).max(axis=2)
    assert pair[np.triu_indices(len(kept), 1)].min() >= threshold - 1e-12


@pytest.mark.numpy_only
def test_discover_fluor_variants_caps_and_deduplicates():
    uncapped = _discover_f1(dedup_threshold=0.0, max_variants=100)
    capped = _discover_f1(dedup_threshold=0.0, max_variants=3)
    assert uncapped is not None and capped is not None
    assert len(capped['v_mats']) - 1 <= 3 < len(uncapped['v_mats']) - 1
    ref = syn.fluor_spectra()[0]
    np.testing.assert_allclose(capped['v_mats'][0], ref)
    np.testing.assert_allclose(capped['delta'], capped['v_mats'] - ref)
    assert capped['dedup_threshold_used'] > 0.0


@pytest.mark.numpy_only
def test_discover_fluor_variants_keeps_the_two_most_distinct_when_all_match_the_reference():
    result = _discover_f1(dedup_threshold=2.0)
    assert result is not None
    assert len(result['v_mats']) == 3
    everything = _discover_f1(dedup_threshold=0.0, max_variants=100)
    ref = syn.fluor_spectra()[0]
    reach = np.abs(everything['v_mats'] - ref).max(axis=1)
    kept_reach = np.abs(result['v_mats'][1:] - ref).max(axis=1)
    assert kept_reach.min() >= np.sort(reach)[-2] - 1e-9


@pytest.mark.numpy_only
def test_discover_fluor_variants_small_sample_never_enlarges_the_som_grid(monkeypatch):
    """With under 500 qualifying events the grid is shrunk toward three
    events per node, but never beyond the requested som_dim."""
    grids = []
    real = aof.get_som_codes

    def spy(data, som_dim, **kw):
        grids.append(som_dim)
        return real(data, som_dim, **kw)

    monkeypatch.setattr(aof, 'get_som_codes', spy)
    result = _discover_f1(n=420, som_dim=3)
    assert result is not None and result['n_events_used'] < 500
    assert grids == [3]


# ---------------------------------------------------------------------------
# Stored AF profiles reused for the unstained thresholds
# ---------------------------------------------------------------------------

def _stored_af_controller(tmp_path, monkeypatch, profile, name='Unstained cells AutoSpectral AF'):
    rng = np.random.default_rng(0)
    controller = syn.FakeController(tmp_path)
    controller.experiment.samples['all_samples'] = {
        'Raw/Unstained cells.fcs': 'Unstained cells', 'Raw/F1.fcs': 'F1 control',
    }
    controller.experiment.process['spectral_model'] = [
        {'label': 'F1', 'sample_name': 'F1 control', 'particle_type': 'Cells'},
        {'label': 'F2', 'particle_type': 'Cells'},
        {'label': 'F3', 'particle_type': 'Cells'},
    ]
    if profile is not None:
        controller.experiment.process['af_profiles'][name] = profile

    extractions = []

    def fake_get_af_spectra(raw, fluor_spectra, **kw):
        extractions.append(raw.shape)
        return syn.af_spectra()

    monkeypatch.setattr(aof, 'sample_from_fcs', lambda path: None)
    monkeypatch.setattr(
        aof, 'get_raw_events',
        lambda sample, fl_ids, **kw: (rng.random((500, len(fl_ids))) * 100.0, rng.random((500, 2))),
    )
    monkeypatch.setattr(aof, 'get_af_spectra', fake_get_af_spectra)
    monkeypatch.setattr(aof, 'discover_fluor_variants', lambda **kw: None)
    return controller, extractions


def _stored_profile(**overrides):
    entry = {
        'spectra': syn.af_spectra().tolist(),
        'source_fcs': 'Raw/Unstained cells.fcs',
        'channel_names': list(syn.DET_PNN),
    }
    entry.update(overrides)
    return entry


@pytest.mark.numpy_only
def test_discover_all_variants_reuses_the_stored_af_profile(tmp_path, monkeypatch):
    controller, extractions = _stored_af_controller(tmp_path, monkeypatch, _stored_profile())
    out = aof.discover_all_variants(controller)
    assert extractions == []
    assert np.isfinite(out['unmixed_pos_thresholds']).all()


@pytest.mark.numpy_only
def test_discover_all_variants_matches_a_stored_profile_on_its_source_file(tmp_path, monkeypatch):
    controller, extractions = _stored_af_controller(
        tmp_path, monkeypatch, _stored_profile(), name='Renamed profile')
    aof.discover_all_variants(controller)
    assert extractions == []


@pytest.mark.numpy_only
@pytest.mark.parametrize('profile, name', [
    (None, None),
    (_stored_profile(channel_names=list(reversed(syn.DET_PNN))), 'Unstained cells AutoSpectral AF'),
    (_stored_profile(spectra=syn.af_spectra()[:, :-1].tolist()), 'Unstained cells AutoSpectral AF'),
    (_stored_profile(source_fcs='Raw/Other.fcs'), 'Other AutoSpectral AF'),
], ids=['absent', 'other-detectors', 'other-width', 'other-source'])
def test_discover_all_variants_extracts_af_when_no_usable_profile_exists(tmp_path, monkeypatch, profile, name):
    controller, extractions = _stored_af_controller(tmp_path, monkeypatch, profile, name=name)
    aof.discover_all_variants(controller)
    assert len(extractions) == 1


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

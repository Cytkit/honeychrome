"""
test_drc_trend.py
-----------------
Trend tests of Honeycluster clusters with a numeric covariate
(drc_stats.run_trend_statistics): a cluster whose share rises with age is
called, the results survive the model sidecar's storage, calls are redone
without refitting and cluster renames reach the trend tables.

Usage:
    pytest tests/test_drc_trend.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

drc_stats = pytest.importorskip('drc_stats')
drc_persist = pytest.importorskip('drc_persist')


class _State:
    """Twenty-four samples aged 18 to 37 weeks; cluster 0's share rises
    from 10% to 40% with age, the other three share the rest."""
    testing_group_selection = []
    group_names = ['Blood']
    reference_group = 'Blood'
    paired = False
    pairing_variable = None
    population_split = False
    stats_normalisation = 'total'
    numeric_covariates = ['age']
    trend_variable = 'age'
    trend_unit = 'weeks'
    trend_interaction = False
    trend_shape = 'curve'
    trend_df = 2
    trend_scale = 'linear'
    trend_call = 'either'
    trend_min_r2 = 0.1
    trend_results = {}

    def __init__(self):
        rng = np.random.default_rng(7)
        ages = np.linspace(18, 37, 24)
        self.cluster_labels, self.sample_groups = {}, {}
        for i, age in enumerate(ages):
            p0 = 0.1 + 0.3 * (age - 18) / 19
            p = np.array([p0] + [(1 - p0) / 3] * 3)
            self.cluster_labels[f's{i}'] = rng.choice(4, size=4000, p=p)
            self.sample_groups[f's{i}'] = 'Blood'
        self.covariates = pd.DataFrame({'age': [f'{a:.2f}' for a in ages]},
                                       index=list(self.cluster_labels))

    def cluster_label(self, cl):
        return f'c{cl}'


def _controller():
    settings = {'raw': {'raw_samples_subdirectory': 'Raw'}}
    return SimpleNamespace(experiment=SimpleNamespace(settings=settings))


def _run(state):
    return drc_stats.run_trend_statistics(_controller(), state, run_freq=True, run_mfi=False,
                                          pval_threshold=0.05, run_counts=True)


def test_cluster_rising_with_age_is_called():
    out = _run(_State())
    freq = out['freq'].table.set_index('feature')
    assert freq.loc['c0', 'significant'] and freq.loc['c0', 'direction'] == 'up'
    assert freq.loc['c0', 'slope'] > 0
    assert out['counts'] is not None and out['counts'].table.set_index('feature').loc['c0', 'significant']
    assert out['settings']['df'] == 2 and len(out['samples']) == 24
    assert list(out['freq_data'].columns) == ['c0', 'c1', 'c2', 'c3']


def test_trend_results_survive_the_sidecar_and_follow_renames(tmp_path):
    state = _State()
    out = _run(state)
    drc_persist.save_bundle(tmp_path / 'bundle', {'trend_results': drc_stats.trend_to_bundle(out)})
    back = drc_stats.trend_from_bundle(drc_persist.load_bundle(tmp_path / 'bundle')['trend_results'])
    pd.testing.assert_frame_equal(back['freq'].table, out['freq'].table, check_dtype=False)
    np.testing.assert_allclose(back['freq'].curves, out['freq'].curves)
    np.testing.assert_allclose(back['freq'].curve_se, out['freq'].curve_se)
    assert back['cluster_names'] == {0: 'c0', 1: 'c1', 2: 'c2', 3: 'c3'}

    strict = drc_stats.reapply_trend_thresholds(back, 0.05, 1.0, 'either')
    assert not strict['freq'].table['significant'].any()

    state.trend_results = back
    renames = drc_stats.relabel_trend_results(state, {0: 'Treg', 1: 'c1', 2: 'c2', 3: 'c3'})
    assert renames == {'c0': 'Treg'}
    assert 'Treg' in set(state.trend_results['freq'].table['feature'])
    assert 'Treg' in state.trend_results['freq_data'].columns
    assert 'Treg' in state.trend_results['cluster_counts'].columns
    pd.testing.assert_frame_equal(back['cluster_counts'], out['cluster_counts'], check_dtype=False)


def test_cluster_composition_over_the_variable():
    out = _run(_State())
    panels = drc_stats.trend_composition(out)
    assert len(panels) == 1
    comp = panels[0]
    assert comp['parts'] == ['c0', 'c1', 'c2', 'c3']
    np.testing.assert_allclose(comp['shares'].sum(axis=0), 1.0)
    c0 = comp['parts'].index('c0')
    assert comp['shares'][c0, 0] == pytest.approx(0.1, abs=0.03)
    assert comp['shares'][c0, -1] == pytest.approx(0.4, abs=0.03)
    merged = drc_stats.trend_composition(out, keep=2)[0]
    assert merged['parts'][-1] == 'Other clusters' and len(merged['parts']) == 3


def test_samples_of_groups_outside_the_trend_get_frequencies_for_prediction():
    state = _State()
    state.group_names = ['Blood', 'Cord']
    state.testing_group_selection = ['Blood']
    rng = np.random.default_rng(3)
    for i in range(3):
        state.cluster_labels[f'o{i}'] = rng.choice(4, size=4000)
        state.sample_groups[f'o{i}'] = 'Cord'
    out = _run(state)
    assert len(out['samples']) == 24
    other = out['other_freq_data']
    assert list(other.index) == ['o0', 'o1', 'o2']
    assert list(other.columns) == list(out['freq_data'].columns)
    assert out['other_groups'] == {'o0': 'Cord', 'o1': 'Cord', 'o2': 'Cord'}
    assert all(np.isnan(v) for v in out['other_time'].values())
    np.testing.assert_allclose(other.loc['o0'], np.log2((
        np.bincount(state.cluster_labels['o0'], minlength=4) + 0.5) / 4001 * 100), rtol=1e-9)


def test_group_by_trend_needs_two_groups():
    state = _State()
    state.trend_interaction = True
    with pytest.raises(RuntimeError, match='exactly two groups'):
        _run(state)


def test_sample_pca_of_the_trend_run_uses_its_tables_and_groups():
    state = _State()
    state.freq_df = state.counts_df = state.mfi_df = None
    state.stats_all_rel, state.stats_group_vec = [], []
    result = _run(state)
    pca = drc_stats.compute_sample_pca(
        state, True, True, False, tables={'freq': result['freq_data'],
                                          'counts': result['counts_data'], 'mfi': None},
        group_of=dict(zip(result['samples'], result['group_vec'])))
    assert pca['sources'] == ['Freq', 'Counts']
    assert list(pca['scores'].index) == result['samples']
    assert set(pca['groups']) == {'Blood'}

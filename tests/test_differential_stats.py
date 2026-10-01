"""
test_differential_stats.py
--------------------------
Tests for controller_components/differential_stats.py and the figure
builders in view_components/differential_plots.py, on synthetic data.

Usage:
    pytest tests/test_differential_stats.py -m numpy_only
"""

import numpy as np
import pandas as pd
import pytest

from honeychrome.controller_components import differential_stats as ds


def _samples(n_per_group=4, n_features=8, effect=3.0, seed=3):
    rng = np.random.default_rng(seed)
    groups = ['A'] * n_per_group + ['B'] * n_per_group
    idx = [f's{i}' for i in range(len(groups))]
    data = pd.DataFrame(rng.normal(size=(len(groups), n_features)), index=idx,
                        columns=[f'f{j}' for j in range(n_features)])
    data.loc[idx[n_per_group:], 'f0'] += effect
    return data, groups


@pytest.mark.numpy_only
def test_logit_proportions_is_finite_at_bounds():
    counts = pd.DataFrame({'p': [0, 50, 100]}, index=['a', 'b', 'c'])
    parents = pd.DataFrame({'p': [100, 100, 100]}, index=['a', 'b', 'c'])
    out = ds.logit_proportions(counts, parents)
    assert np.all(np.isfinite(out.values))
    assert out.loc['b', 'p'] == pytest.approx(0.0)
    assert out.loc['a', 'p'] == pytest.approx(-out.loc['c', 'p'])
    assert out.loc['a', 'p'] == pytest.approx(np.log2(0.5 / 100.5))


@pytest.mark.numpy_only
def test_logit_proportions_rejects_count_above_parent():
    counts = pd.DataFrame({'p': [5]}, index=['a'])
    parents = pd.DataFrame({'p': [4]}, index=['a'])
    with pytest.raises(ValueError):
        ds.logit_proportions(counts, parents)


@pytest.mark.numpy_only
def test_run_moderated_detects_shifted_feature():
    data, groups = _samples()
    contrasts = ds.build_contrasts(['A', 'B'], 'reference', 'A')
    res = ds.run_moderated(data, groups, contrasts, 'reference', 0.05, 0.5)
    top = res.sort_values('P.Value').iloc[0]
    assert top['feature'] == 'f0'
    assert bool(top['significant'])
    assert set(res['comparison']) == {'B vs A'}


@pytest.mark.numpy_only
def test_too_few_features_message_carries_hint():
    data, groups = _samples(n_features=2)
    contrasts = ds.build_contrasts(['A', 'B'], 'reference', 'A')
    with pytest.raises(RuntimeError, match='add more gates'):
        ds.run_moderated(data, groups, contrasts, 'reference', 0.05, 0.5,
                         too_few_hint='add more gates')


@pytest.mark.numpy_only
def test_covariate_frame_reports_missing_values():
    cov = pd.DataFrame({'age': ['30', '', '40']}, index=['a', 'b', 'c'])
    assert ds.missing_covariate_values(cov, ['a', 'b', 'c'], ['age']) == {'age': ['b']}
    with pytest.raises(RuntimeError, match='missing values'):
        ds.covariate_frame(cov, ['a', 'b', 'c'], ['age'])
    frame = ds.covariate_frame(cov, ['a', 'c'], ['age'])
    assert list(frame['age']) == ['30', '40']


def test_glm_offset_uses_per_feature_denominator():
    pytest.importorskip('statsmodels')
    rng = np.random.default_rng(0)
    groups = ['A'] * 5 + ['B'] * 5
    idx = [f's{i}' for i in range(10)]
    parents = pd.DataFrame({'p1': [1000] * 5 + [4000] * 5,
                            'p2': [2000] * 10, 'p3': [3000] * 10}, index=idx)
    # Same 10% rate in both groups for p1; only the parent count differs.
    counts = pd.DataFrame({
        'p1': rng.poisson(parents['p1'] * 0.1),
        'p2': rng.poisson(parents['p2'] * 0.2),
        'p3': rng.poisson(parents['p3'] * 0.3),
    }, index=idx)
    contrasts = ds.build_contrasts(['A', 'B'], 'reference', 'A')
    with_offset = ds.run_glm_counts(counts, groups, contrasts, 'reference', 0.05, 0.5,
                                    offset_df=parents)
    row = with_offset.set_index('feature').loc['p1']
    assert abs(row['logFC']) < 0.3
    assert not bool(row['significant'])


@pytest.mark.numpy_only
def test_figures_build_without_canvas():
    pytest.importorskip('matplotlib')
    from honeychrome.view_components import differential_plots as dp

    data, groups = _samples()
    contrasts = ds.build_contrasts(['A', 'B'], 'reference', 'A')
    res = ds.run_moderated(data, groups, contrasts, 'reference', 0.05, 0.5)

    heat = dp.make_heatmap_figure(res, data, 'Frequencies', dict(zip(data.index, groups)),
                                  group_a='A', group_b='B', value_label='log2 odds')
    assert len(heat.axes) >= 4

    volcano = dp.make_volcano_figure(res, 'Frequencies', 0.05, 0.5, run_label='run 1')
    assert callable(getattr(volcano, '_hover_handler', None))

    meta = pd.DataFrame({'feature': data.columns, 'cluster': ['c1'] * 4 + ['c2'] * 4,
                         'cluster_id': [1] * 4 + [2] * 4, 'channel': list('wxyz') * 2})
    mfi = ds.run_moderated(data, groups, contrasts, 'reference', 0.05, 0.2,
                           feature_meta=meta, hierarchical=True)
    summary = dp.make_marker_summary_figure(mfi, 'Medians', {}, family_label='population')
    assert summary.axes

    empty = dp.make_heatmap_figure(res.assign(significant=False), data, 'Frequencies', {})
    assert empty.axes[0].texts

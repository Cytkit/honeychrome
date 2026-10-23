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
    frame = ds.covariate_frame(cov, ['a', 'c'], ['age'], numeric=[])
    assert list(frame['age']) == ['30', '40']


@pytest.mark.numpy_only
def test_numeric_covariate_is_continuous_only_when_requested():
    cov = pd.DataFrame({'batch': ['1', '2', '3', '1', '2', '3']},
                       index=[f's{i}' for i in range(6)])
    ids = list(cov.index)
    groups = ['A', 'A', 'A', 'B', 'B', 'B']

    as_numeric = ds.covariate_frame(cov, ids, ['batch'], numeric=['batch'])
    as_factor = ds.covariate_frame(cov, ids, ['batch'], numeric=[])
    assert pd.api.types.is_numeric_dtype(as_numeric['batch'])
    assert not pd.api.types.is_numeric_dtype(as_factor['batch'])

    x_num, names_num, _ = ds.build_design(groups, 'A', covariates=as_numeric)
    _x_fac, names_fac, _ = ds.build_design(groups, 'A', covariates=as_factor)
    assert names_num == ['(Intercept)', 'group[B]', 'batch']
    assert names_fac == ['(Intercept)', 'group[B]', 'batch[2]', 'batch[3]']
    assert abs(x_num[:, 2].mean()) < 1e-12


@pytest.mark.numpy_only
def test_numeric_covariate_rejects_text_values():
    cov = pd.DataFrame({'crp': ['1.5', 'high', '2.0']}, index=['a', 'b', 'c'])
    with pytest.raises(RuntimeError, match="'crp' is set to numeric"):
        ds.covariate_frame(cov, ['a', 'b', 'c'], ['crp'], numeric=['crp'])


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


def test_glm_fits_are_statsmodels_fits():
    """The batched GLM gives statsmodels' numbers exactly, with unmeasured
    samples, an empty feature, an empty group, per-feature offsets and a
    covariate."""
    pytest.importorskip('statsmodels')
    rng = np.random.default_rng(1)
    groups = ['A'] * 6 + ['B'] * 6 + ['C'] * 6
    cov = pd.DataFrame({'age': rng.normal(50, 10, 18)}, index=[f's{i}' for i in range(18)])
    design, _names, _cols = ds.build_design(groups, 'A', None, cov)
    mu = rng.lognormal(5, 1.5, (40, 1)) * np.exp(rng.normal(0, 0.5, (40, 18)))
    counts = rng.negative_binomial(5, 5 / (5 + mu)).astype(float)
    offsets = np.log(rng.lognormal(11, 0.3, (40, 18)))
    counts[0] = 0.0                              # no events
    counts[1, 12:] = 0.0                         # none in group C
    counts[2, rng.random(18) < 0.3] = np.nan     # unmeasured samples
    counts[3, 6:12] = np.nan                     # group B unmeasured
    offsets[4, 5] = np.nan
    fits = ds._glm_fit_features(counts, design, offsets)
    n_batched = 0
    for i, (fit, columns) in enumerate(fits):
        ref, ref_columns = ds._glm_fit_observed(counts[i], design, offsets[i])
        assert (fit is None) == (ref is None)
        if ref is None:
            continue
        n_batched += isinstance(fit, ds._GLMFit)
        assert np.array_equal(columns, ref_columns)
        for k in ('params', 'bse', 'tvalues'):
            assert np.array_equal(np.asarray(getattr(fit, k)), np.asarray(getattr(ref, k)))
    assert fits[0] == (None, None) and n_batched >= 35


@pytest.mark.numpy_only
def test_stagewise_adjust_matches_per_family_definition():
    from honeychrome.controller_components import moderated_stats as ms
    rng = np.random.default_rng(2)
    p = rng.beta(0.3, 1.0, 600)
    p[rng.random(600) < 0.05] = np.nan
    family = [f'c{k % 7}\x1f{k % 13}' for k in rng.permutation(600)]
    got = ms.stagewise_adjust(p, family, 0.05)
    fam = np.asarray(family, dtype=object)
    labels = list(dict.fromkeys(family))
    fam_p = np.array([ms.simes(p[fam == f]) for f in labels])
    fam_adj = ms.bh_adjust(fam_p)
    n_fam = int(np.isfinite(fam_p).sum())
    selected = np.isfinite(fam_adj) & (fam_adj <= 0.05)
    expected = np.where(np.isfinite(p), 1.0, np.nan)
    for f, sel in zip(labels, selected):
        if sel:
            expected[fam == f] = np.minimum(ms.bh_adjust(p[fam == f]) * n_fam / selected.sum(), 1.0)
    assert got['n_selected'] == int(selected.sum()) > 0 and got['n_families'] == n_fam
    assert np.array_equal(got['adjusted'], expected, equal_nan=True)
    assert np.array_equal(got['family_p'], fam_p[[labels.index(f) for f in family]], equal_nan=True)


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


def test_comparison_heatmap_colours_only_significant_cells():
    pytest.importorskip('matplotlib')
    from honeychrome.view_components import differential_plots as dp

    rng = np.random.default_rng(5)
    groups = ['Ref'] * 4 + ['B'] * 4 + ['C'] * 4
    data = pd.DataFrame(rng.normal(scale=0.3, size=(12, 9)), index=[f's{i}' for i in range(12)],
                        columns=[f'g{j}' for j in range(9)])
    data.iloc[4:8, 0] += 3.0
    data.iloc[8:12, 1] -= 2.5
    contrasts = ds.build_contrasts(['Ref', 'B', 'C'], 'reference', 'Ref')
    res = ds.run_moderated(data, groups, contrasts, 'reference', 0.05, 1.0, use_treat=False)

    fig = dp.make_comparison_heatmap_figure(res, 'All comparisons', n_rows=2)
    panels = [ax for ax in fig.axes if ax.images]
    assert len(panels) == 2
    shown = np.hstack([np.asarray(ax.images[0].get_array()) for ax in panels])
    names = [t.get_text() for ax in panels for t in ax.get_xticklabels()]
    rows = [t.get_text() for t in panels[0].get_yticklabels()]
    assert rows == ['B vs Ref', 'C vs Ref'] and sorted(names) == sorted(data.columns)
    expected = res.assign(v=np.where(res['significant'], res['logFC'], 0.0)).pivot(
        index='comparison', columns='feature', values='v').reindex(index=rows, columns=names)
    np.testing.assert_allclose(shown, expected.values)
    assert shown[0, names.index('g0')] > 2 and shown[1, names.index('g1')] < -2
    # effect order: most negative mean first
    assert names[0] == 'g1' and names[-1] == 'g0'

    alpha = dp.make_comparison_heatmap_figure(res, 'All', order='alphabetical')
    assert [t.get_text() for t in alpha.axes[0].get_xticklabels()] == sorted(data.columns)

    none = dp.make_comparison_heatmap_figure(res.assign(significant=False), 'All')
    assert none.axes[0].texts


@pytest.mark.numpy_only
def test_sparse_features_counts_samples_above_zero():
    counts = pd.DataFrame({'none': [0.0, 0.0, 0.0, 0.0],
                           'two': [0.0, 3.0, 0.0, 1.0],
                           'three': [1.0, 2.0, 0.0, 5.0],
                           'unmeasured': [np.nan, np.nan, 4.0, 2.0]})
    assert ds.sparse_features(counts, 3) == ['none', 'two', 'unmeasured']
    assert ds.sparse_features(counts, 1) == ['none']


def _paired_null(rng, n_mice=6, n_features=40, total=400, alpha=0.5):
    """Spleen and tissue per mouse, NB counts around mouse-specific shares,
    no group effect."""
    base = rng.lognormal(0, 0.5, n_features)
    rows, groups, pairs = [], [], []
    for m in range(n_mice):
        share = base * rng.lognormal(0, 0.7, n_features)
        mu = total * share / share.sum()
        for g in ('A', 'B'):
            rows.append(rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu)))
            groups.append(g)
            pairs.append(f'm{m}')
    idx = [f's{i}' for i in range(len(rows))]
    return pd.DataFrame(rows, index=idx, columns=[f'f{j}' for j in range(n_features)]), groups, pairs


def test_ql_counts_test_holds_its_false_positive_rate():
    """Paired null data: the quasi-likelihood F test stays near the nominal
    5%; the Wald test is far above it."""
    pytest.importorskip('statsmodels')
    rng = np.random.default_rng(11)
    rates = {'ql': [], 'wald': []}
    for _rep in range(25):
        counts, groups, pairs = _paired_null(rng)
        contrasts = ds.build_contrasts(['A', 'B'], 'reference', 'A')
        for method in rates:
            res = ds.run_glm_counts(counts, groups, contrasts, 'reference', 0.05, 0.0, pairing_vec=pairs,
                                    use_treat=False, method=method)
            p = res['P.Value'].values
            rates[method].append(np.mean(p[np.isfinite(p)] < 0.05))
    assert np.mean(rates['ql']) < 0.075
    assert np.mean(rates['wald']) > 0.12


def test_ql_results_rethreshold_to_the_f_test():
    """SE and df.total stored by the QL test give back its F-test p-value,
    so threshold changes need no refit; unmeasured groups are untestable."""
    pytest.importorskip('statsmodels')
    from scipy import stats
    rng = np.random.default_rng(4)
    counts, groups, pairs = _paired_null(rng, total=2000)
    counts.loc[counts.index[::2], 'f0'] = np.nan            # never measured in group A
    contrasts = ds.build_contrasts(['A', 'B'], 'reference', 'A')
    res = ds.run_glm_counts(counts.astype(float), groups, contrasts, 'reference', 0.05, 0.0,
                            pairing_vec=pairs, use_treat=False, method='ql')
    res = res.set_index('feature')
    assert np.isnan(res.loc['f0', 'logFC'])
    tested = res.drop(index='f0')
    f_stat = tested['t'].values ** 2
    df_total = tested['df.total'].values
    expected = (stats.chi2.sf(f_stat, 1) if np.all(np.isinf(df_total))
                else stats.f.sf(f_stat, 1, df_total))
    np.testing.assert_allclose(tested['P.Value'], expected, rtol=1e-8)
    again = ds.apply_significance(res.reset_index(), 0.05, 0.0, use_treat=False).set_index('feature')
    np.testing.assert_allclose(again['P.Value'], res['P.Value'])


def test_feature_missing_in_the_first_pairing_level_is_tested():
    """A feature not measured in any sample of the first-sorted pairing
    level is tested, with the same result as when that level sorts last;
    one with no sample in the baseline group stays untestable."""
    pytest.importorskip('statsmodels')
    rng = np.random.default_rng(5)
    counts, groups, pairs = _paired_null(rng, total=2000)
    counts = counts.astype(float)
    counts.loc[[s for s, m in zip(counts.index, pairs) if m == 'm0'], 'f1'] = np.nan
    counts.loc[counts.index[::2], 'f2'] = np.nan            # never measured in group A
    renamed = ['m9' if m == 'm0' else m for m in pairs]
    contrasts = ds.build_contrasts(['A', 'B'], 'reference', 'A')
    freqs = np.log2(counts + 0.5)
    runs = {}
    for name, pairing in (('first', pairs), ('last', renamed)):
        runs[name] = {
            'moderated': ds.run_moderated(freqs, groups, contrasts, 'reference', 0.05, 0.0,
                                          pairing_vec=pairing, use_treat=False),
            'ql': ds.run_glm_counts(counts, groups, contrasts, 'reference', 0.05, 0.0,
                                    pairing_vec=pairing, use_treat=False, method='ql'),
            'wald': ds.run_glm_counts(counts, groups, contrasts, 'reference', 0.05, 0.0,
                                      pairing_vec=pairing, use_treat=False, method='wald'),
        }
    for test in ('moderated', 'ql', 'wald'):
        first = runs['first'][test].set_index('feature')
        last = runs['last'][test].set_index('feature')
        assert np.isfinite(first.loc['f1', 'logFC'])
        assert np.isnan(first.loc['f2', 'logFC'])
        np.testing.assert_allclose(first['logFC'], last['logFC'], rtol=1e-6, atol=1e-9, equal_nan=True)
        np.testing.assert_allclose(first['P.Value'], last['P.Value'], rtol=1e-6, atol=1e-12, equal_nan=True)


@pytest.mark.numpy_only
def test_library_sizes():
    rng = np.random.default_rng(2)
    share = rng.lognormal(0, 0.5, 30)
    share /= share.sum()
    clone = share.copy()
    clone[1:] *= 0.4 / clone[1:].sum()
    clone[0] = 0.6
    counts = pd.DataFrame([rng.poisson(5000 * share), rng.poisson(5000 * clone)],
                          columns=[f'c{i}' for i in range(30)])
    totals = counts.sum(axis=1).values
    np.testing.assert_allclose(ds.library_sizes(counts, totals, 'total'), totals)
    robust = ds.library_sizes(counts, totals, 'robust')
    # the clone sample's other features sit at 0.4 of its total
    ratio = (robust[1] / totals[1]) / (robust[0] / totals[0])
    assert 0.33 < ratio < 0.48
    control = ds.library_sizes(counts, totals, 'control', control=['c5', 'c6', 'c7'])
    ctrl = counts[['c5', 'c6', 'c7']].sum(axis=1).values
    np.testing.assert_allclose(control / control.sum(), ctrl / ctrl.sum(), rtol=1e-3)
    with pytest.raises(ValueError):
        ds.library_sizes(counts, totals, 'control', control=[])

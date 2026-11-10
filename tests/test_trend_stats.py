"""
test_trend_stats.py
-------------------
Tests for controller_components/trend_stats.py and moderated_stats.f_test
on synthetic data.

Usage:
    pytest tests/test_trend_stats.py -m numpy_only
"""

import numpy as np
import pandas as pd
import pytest

from honeychrome.controller_components import moderated_stats as ms
from honeychrome.controller_components import trend_stats as ts


def _data(t, signals, n_noise=40, sd=0.3, seed=5):
    """Samples × features: the given signal columns plus noise features."""
    rng = np.random.default_rng(seed)
    cols = {name: f(t) + rng.normal(scale=sd, size=t.size) for name, f in signals.items()}
    for j in range(n_noise):
        cols[f'noise{j}'] = rng.normal(scale=sd, size=t.size)
    return pd.DataFrame(cols, index=[f's{i}' for i in range(t.size)])


@pytest.mark.numpy_only
def test_spline_basis_is_orthonormal_and_linear_beyond_boundary_knots():
    x = np.linspace(18, 37, 120)
    sp = ts.SplineBasis(x, 3, 0.05)
    B = sp(x)
    np.testing.assert_allclose(B.T @ B / x.size, np.eye(3), atol=1e-10)
    for side in (x < sp.knots[0], x > sp.knots[-1]):
        np.testing.assert_allclose(np.diff(B[side], 2, axis=0), 0.0, atol=1e-10)


@pytest.mark.numpy_only
def test_f_test_of_one_column_is_the_moderated_t_squared():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(20), rng.normal(size=20)])
    Y = rng.normal(size=(30, 20))
    ft = ms.f_test(Y, X, [1])
    t = ms.e_bayes(ms.lm_fit(Y, X)).t[:, 1]
    np.testing.assert_allclose(ft.F, t ** 2, rtol=1e-8, atol=1e-12)
    assert np.all((ft.partial_r2 >= 0) & (ft.partial_r2 <= 1))


@pytest.mark.numpy_only
def test_curve_finds_a_peak_the_slope_misses():
    t = np.linspace(18, 37, 80)
    data = _data(t, {'steady': lambda v: 0.15 * (v - 27),
                     'peak': lambda v: 1.5 * np.exp(-((v - 27.5) / 3.0) ** 2)})
    res = ts.run_trend(data, t, variable='age', unit='weeks')
    tb = res.table.set_index('feature')
    assert res.settings['df'] == ts.DEFAULT_CURVE_DF
    assert tb.loc['steady', 'significant'] and tb.loc['steady', 'direction'] == 'up'
    assert tb.loc['steady', 'slope'] == pytest.approx(0.15, abs=0.03)
    assert tb.loc['peak', 'significant'] and tb.loc['peak', 'curve.only']
    assert tb.loc['peak', 'called.by'] == 'curve'
    assert tb.loc['peak', 'at.max'] == pytest.approx(27.5, abs=2.5)
    assert tb.filter(like='noise', axis=0)['significant'].sum() <= 1
    assert res.curves.shape == (data.shape[1], 50)


@pytest.mark.numpy_only
def test_line_shape_tests_the_slope_and_calls_can_be_changed_without_refitting():
    t = np.linspace(0, 10, 40)
    data = _data(t, {'up': lambda v: 0.2 * v})
    tb = ts.run_trend(data, t, shape='line').table
    np.testing.assert_allclose(tb['F'], tb['slope.t'] ** 2, rtol=1e-8, atol=1e-12)
    slope_only = ts.trend_significance(tb, 0.05, 0.0, 'slope')
    assert set(slope_only.loc[slope_only.significant, 'called.by']) <= {'slope'}
    strict = ts.trend_significance(tb, 0.05, 0.99, 'either')
    assert not strict.significant.any()


@pytest.mark.numpy_only
def test_curve_df_is_capped_by_the_number_of_samples():
    t = np.linspace(1, 25, 25)
    res = ts.run_trend(_data(t, {'up': lambda v: 0.1 * v}), t, k=4)
    assert res.settings['df'] == 2
    assert any('instead of 4' in n for n in res.notes)
    assert ts.curve_df(200, 4) == 4 and ts.curve_df(5, 3) == 1


@pytest.mark.numpy_only
def test_points_shape_for_a_designed_time_course():
    t = np.repeat([0.0, 3.0, 7.0, 14.0], 5)
    assert ts.suggest_shape(t) == 'points'
    assert ts.suggest_shape(np.linspace(18, 37, 40)) == 'curve'
    data = _data(t, {'day7': lambda v: 1.5 * (v == 7.0)})
    res = ts.run_trend(data, t, shape='points', scale='log2')
    tb = res.table.set_index('feature')
    assert res.settings['df'] == 3 and list(res.grid) == [0.0, 3.0, 7.0, 14.0]
    assert tb.loc['day7', 'significant'] and tb.loc['day7', 'at.max'] == 7.0


@pytest.mark.numpy_only
def test_log2_scale_reports_the_slope_per_doubling():
    t = np.repeat([0.0, 1.0, 3.0, 7.0, 15.0, 31.0], 6)
    data = _data(t, {'doubling': lambda v: 0.8 * np.log2(v + 1.0)}, sd=0.1)
    tb = ts.run_trend(data, t, scale='log2', shape='line').table.set_index('feature')
    assert tb.loc['doubling', 'slope'] == pytest.approx(0.8, abs=0.05)


@pytest.mark.numpy_only
def test_missing_values_and_the_variable_itself_are_left_out_of_the_model():
    t = np.linspace(18, 37, 40)
    data = _data(t, {'up': lambda v: 0.1 * v})
    with_gap = t.copy()
    with_gap[:3] = np.nan
    cov = pd.DataFrame({'age': t, 'batch': ['a', 'b'] * 20}, index=data.index)
    res = ts.run_trend(data, with_gap, variable='age', covariates=cov)
    assert res.settings['n_samples'] == 37
    assert any('3 sample(s)' in n for n in res.notes)


@pytest.mark.numpy_only
def test_pairing_that_would_absorb_a_common_trend_is_dropped():
    t = np.repeat(np.linspace(18, 37, 20), 2)
    groups = ['Blood', 'Spleen'] * 20
    pairs = list(np.repeat([f'd{i}' for i in range(20)], 2))
    data = _data(t, {'up': lambda v: 0.1 * v})
    res = ts.run_trend(data, t, group_vec=groups, pairing_vec=pairs)
    assert not res.settings['paired']
    assert any('absorb the trend' in n for n in res.notes)


@pytest.mark.numpy_only
def test_pairing_is_kept_for_donors_sampled_at_several_times():
    rng = np.random.default_rng(2)
    donors = list(np.repeat([f'd{i}' for i in range(10)], 4))
    t = np.tile([0.0, 7.0, 14.0, 28.0], 10)
    level = np.repeat(rng.normal(scale=2.0, size=10), 4)
    data = _data(t, {'rises': lambda v: 0.05 * v})
    data['rises'] += level
    paired = ts.run_trend(data, t, pairing_vec=donors, shape='line')
    unpaired = ts.run_trend(data, t, shape='line')
    assert paired.settings['paired']
    p_paired = paired.table.set_index('feature').loc['rises', 'slope.P.Value']
    p_unpaired = unpaired.table.set_index('feature').loc['rises', 'slope.P.Value']
    assert p_paired < p_unpaired


@pytest.mark.numpy_only
def test_interaction_tests_the_difference_between_two_groups_within_pairs():
    t = np.repeat(np.linspace(18, 37, 30), 2)
    groups = ['Blood', 'Spleen'] * 30
    pairs = list(np.repeat([f'd{i}' for i in range(30)], 2))
    is_m = np.array(groups) == 'Spleen'
    rng = np.random.default_rng(4)
    data = _data(t, {'shared': lambda v: 0.2 * v})
    data['diverges'] = np.where(is_m, 0.2 * t, 0.0) + rng.normal(scale=0.3, size=t.size)
    res = ts.run_trend(data, t, group_vec=groups, baseline='Blood', interaction=True,
                       pairing_vec=pairs)
    tb = res.table.set_index('feature')
    assert res.settings['paired'] and res.settings['interaction']
    assert tb.loc['diverges', 'significant'] and tb.loc['diverges', 'slope'] == pytest.approx(0.2, abs=0.03)
    assert not tb.loc['shared', 'significant']
    with pytest.raises(RuntimeError, match='exactly two groups'):
        ts.run_trend(data, t, group_vec=['a', 'b', 'c'] * 20, interaction=True)


@pytest.mark.numpy_only
def test_unpaired_interaction_keeps_each_groups_own_curve_and_band():
    t = np.concatenate([np.linspace(18, 37, 30), np.linspace(20, 36, 25)])
    groups = ['A'] * 30 + ['B'] * 25
    rng = np.random.default_rng(8)
    data = _data(t, {'shared': lambda v: 0.1 * v}, n_noise=15, seed=9)
    data['diverges'] = np.where(np.array(groups) == 'B', 0.2 * t, 1.0) + rng.normal(scale=0.3, size=t.size)
    res = ts.run_trend(data, t, group_vec=groups, baseline='A', interaction=True, k=2)
    assert res.group_curves.shape == (2,) + res.curves.shape
    assert res.group_curve_se.shape == res.group_curves.shape
    basis = ts.SplineBasis(t, res.settings['df'], ts.DEFAULT_BOUNDARY)
    G = np.column_stack([np.ones(res.grid.size), basis(res.grid)])
    s2 = ms.f_test(data.to_numpy(float).T,
                   np.column_stack([np.ones(t.size), np.array(groups) == 'B', basis(t),
                                    (np.array(groups) == 'B')[:, None] * basis(t)]),
                   [4, 5]).fit.s2_post
    j = list(data.columns).index('diverges')
    for gi, g in enumerate(['A', 'B']):
        m = np.array(groups) == g
        Xg = np.column_stack([np.ones(m.sum()), basis(t[m])])
        coef = np.linalg.lstsq(Xg, data['diverges'].to_numpy(float)[m], rcond=None)[0]
        np.testing.assert_allclose(res.group_curves[gi, j], G @ coef, rtol=1e-8)
        V = np.linalg.inv(Xg.T @ Xg)
        expected = np.sqrt(np.einsum('gi,ij,gj->g', G, V, G) * s2[j])
        np.testing.assert_allclose(res.group_curve_se[gi, j], expected, rtol=1e-8)
    np.testing.assert_allclose(res.group_curves[1, j] - res.group_curves[0, j], res.curves[j],
                               atol=1e-8)
    import json
    back = ts.result_from_jsonable(json.loads(json.dumps(ts.result_to_jsonable(res))))
    np.testing.assert_allclose(back.group_curves, res.group_curves)
    np.testing.assert_allclose(back.group_curve_se, res.group_curve_se)
    tp_ = np.repeat(np.linspace(18, 37, 30), 2)
    paired = ts.run_trend(_data(tp_, {'up': lambda v: 0.1 * v}, n_noise=10), tp_,
                          group_vec=['A', 'B'] * 30, baseline='A', interaction=True,
                          pairing_vec=list(np.repeat([f'd{i}' for i in range(30)], 2)))
    assert paired.settings['paired'] and paired.group_curves is None


@pytest.mark.numpy_only
def test_composition_shares_add_up_and_follow_the_counts():
    t = np.linspace(18, 37, 60)
    rng = np.random.default_rng(3)
    share_a = 0.2 + 0.5 * (t - 18) / 19          # rises from 20% to 70%
    probs = np.column_stack([share_a, 0.9 - share_a, np.full(t.size, 0.1)])
    counts = pd.DataFrame([rng.multinomial(5000, p) for p in probs],
                          columns=['A', 'B', 'Other'], index=[f's{i}' for i in range(t.size)])
    counts.iloc[3] = np.nan
    comp = ts.fit_composition(counts, t, variable='age', k=2)
    shares = comp['shares']
    assert shares.shape == (3, comp['grid'].size)
    np.testing.assert_allclose(shares.sum(axis=0), 1.0, atol=1e-12)
    assert (shares > 0).all()
    assert len(comp['samples']) == t.size - 1 and comp['notes']
    assert shares[0, 0] == pytest.approx(0.2, abs=0.04)
    assert shares[0, -1] == pytest.approx(0.7, abs=0.04)
    np.testing.assert_allclose(comp['observed'].sum(axis=1), 1.0)
    groups = ['x', 'y'] * 30
    by_group = ts.fit_composition(counts, t, group_vec=groups)
    assert by_group['groups'][:2] == ['x', 'y']
    with pytest.raises(RuntimeError, match='two populations'):
        ts.fit_composition(counts[['A']], t)


@pytest.mark.numpy_only
def test_merge_small_parts_keeps_the_largest_and_adds_up_the_rest():
    counts = pd.DataFrame({'a': [50, 60], 'b': [30, 20], 'c': [5, 6], 'd': [4, 3],
                           'Other': [1, 1]})
    merged = ts.merge_small_parts(counts, keep=2)
    assert list(merged.columns) == ['a', 'b', 'Other']
    np.testing.assert_array_equal(merged['Other'], [10, 10])
    assert ts.merge_small_parts(counts, keep=4) is counts


@pytest.mark.numpy_only
def test_design_check_flags_a_covariate_that_follows_the_variable():
    t = np.linspace(18, 37, 30)
    cov = pd.DataFrame({'batch': ['early'] * 15 + ['late'] * 15,
                        'sex': ['F', 'M'] * 15})
    checks = {c['name']: c for c in ts.design_check(t, cov, {'events': t * 1000 + 5})}
    assert checks['batch']['warn'] and checks['batch']['measure'] == 'η'
    assert not checks['sex']['warn']
    assert checks['events']['measure'] == 'r' and checks['events']['value'] == pytest.approx(1.0)


@pytest.mark.numpy_only
def test_trend_result_round_trips_through_json():
    import json
    t = np.linspace(18, 37, 40)
    res = ts.run_trend(_data(t, {'up': lambda v: 0.1 * v}), t, variable='age')
    back = ts.result_from_jsonable(json.loads(json.dumps(ts.result_to_jsonable(res))))
    pd.testing.assert_frame_equal(back.table, res.table, check_dtype=False)
    np.testing.assert_allclose(back.curves, res.curves)
    assert back.settings == json.loads(json.dumps(res.settings))
    np.testing.assert_allclose(back.curve_se, res.curve_se)


@pytest.mark.numpy_only
def test_curve_standard_error_matches_least_squares():
    t = np.linspace(18, 37, 40)
    data = _data(t, {'up': lambda v: 0.1 * v}, n_noise=10)
    res = ts.run_trend(data, t, k=2)
    basis = ts.SplineBasis(t, res.settings['df'], ts.DEFAULT_BOUNDARY)
    X = np.column_stack([np.ones(t.size), basis(t)])
    G = np.column_stack([np.ones(res.grid.size), basis(res.grid)])
    V = np.linalg.inv(X.T @ X)
    fit = ms.e_bayes(ms.lm_fit(data.to_numpy(float).T, X))
    j = list(data.columns).index('up')
    expected = np.sqrt(np.einsum('gi,ij,gj->g', G, V, G) * fit.s2_post[j])
    np.testing.assert_allclose(res.curve_se[j], expected, rtol=1e-8)
    assert res.curve_se.shape == res.curves.shape


@pytest.mark.numpy_only
def test_permutation_keeps_pairs_together_when_they_share_the_value():
    t = np.repeat(np.linspace(18, 37, 10), 2)
    groups = ['F', 'M'] * 10
    pairs = list(np.repeat([f'd{i}' for i in range(10)], 2))
    d = ts.trend_design([f's{i}' for i in range(20)], t, group_vec=groups, interaction=True,
                        pairing_vec=pairs)
    assert d.absorbed_time and d.pairing is not None
    perm = d.permutation(np.random.default_rng(0))
    assert sorted(perm) == list(range(20))
    for i in range(0, 20, 2):
        a, b = perm[i], perm[i + 1]
        assert pairs[a] == pairs[b] and groups[a] == 'F' and groups[b] == 'M'
    common = ts.trend_design([f's{i}' for i in range(20)], t, group_vec=groups)
    perm = common.permutation(np.random.default_rng(0))
    assert all(groups[p] == g for p, g in zip(perm, groups))


@pytest.mark.numpy_only
def test_patterns_group_rising_and_falling_features_and_name_them():
    t = np.linspace(18, 37, 50)
    signals = {f'up{j}': (lambda v, j=j: 0.1 * (1 + j % 3) * v) for j in range(6)}
    signals.update({f'down{j}': (lambda v, j=j: -0.1 * (1 + j % 3) * v) for j in range(6)})
    signals.update({f'peak{j}': (lambda v: -0.08 * (v - 27.5) ** 2) for j in range(6)})
    data = _data(t, signals, n_noise=20, sd=0.2)
    res = ts.run_trend(data, t, variable='age', unit='weeks')
    out = ts.assign_patterns(res, 3)
    by = out.table.set_index('feature')['pattern']
    assert len({by[f'up{j}'] for j in range(6)}) == 1 and by['up0'].startswith('rises')
    assert by['down0'].startswith('falls') and by['peak0'].startswith('peaks at')
    assert {p['name'] for p in out.settings['patterns']} == {by['up0'], by['down0'], by['peak0']}
    again = ts.recall(out, 0.05, 1.0, 'either', n_patterns=3)
    assert set(again.table['pattern']) == {''} and again.settings['patterns'] == []
    stability = ts.pattern_stability(data, t, ks=[2, 3], n_sub=5, result=res, variable='age')
    assert stability.set_index('k').loc[3, 'ari'] > 0.75
    auto = ts.add_patterns(res, data, t, n_patterns=0, variable='age', unit='weeks')
    assert auto.settings['pattern_auto']
    assert auto.settings['pattern_k'] == auto.settings['pattern_k_auto'] >= 3


@pytest.mark.numpy_only
def test_pattern_names():
    grid = np.linspace(0, 10, 50)
    assert ts.pattern_name(grid, grid, 'days') == 'rises'
    assert ts.pattern_name(np.exp(-grid / 2.0), grid) == 'falls then levels off'
    assert ts.pattern_name(grid ** 3, grid) == 'rises late'
    assert ts.pattern_name(-(grid - 5) ** 2, grid, 'days') == 'peaks at 4.9 days'


@pytest.mark.numpy_only
def test_interaction_with_the_baseline_second_in_sample_order():
    t = np.repeat(np.linspace(18, 37, 30), 2)
    groups = ['Spleen', 'Blood'] * 30
    pairs = list(np.repeat([f'd{i}' for i in range(30)], 2))
    rng = np.random.default_rng(4)
    data = _data(t, {}, n_noise=10)
    data['diverges'] = np.where(np.array(groups) == 'Spleen', 0.2 * t, 0.0) \
        + rng.normal(scale=0.3, size=t.size)
    res = ts.run_trend(data, t, group_vec=groups, baseline='Blood', interaction=True,
                       pairing_vec=pairs)
    assert res.settings['groups'] == ['Blood', 'Spleen']
    assert res.table.set_index('feature').loc['diverges', 'slope'] == pytest.approx(0.2, abs=0.03)

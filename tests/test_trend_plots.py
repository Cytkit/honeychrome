"""
test_trend_plots.py
-------------------
Figures of trend results (view_components/trend_plots.py): the trend
volcano, the per-feature trend plot, the trajectory heatmap and samples
coloured by a numeric variable. Built off screen with matplotlib only.

Usage:
    pytest tests/test_trend_plots.py -m numpy_only
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('matplotlib')
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402

from honeychrome.controller_components import trend_stats as ts  # noqa: E402
from honeychrome.view_components import trend_plots as tp  # noqa: E402


def _result(n=40, seed=3):
    t = np.linspace(18, 37, n)
    rng = np.random.default_rng(seed)
    cols = {'rises': 0.15 * t, 'falls': -0.15 * t,
            'peaks': -0.08 * (t - 27.5) ** 2}
    for j in range(20):
        cols[f'noise{j}'] = np.zeros(n)
    data = pd.DataFrame({k: v + rng.normal(scale=0.3, size=n) for k, v in cols.items()},
                        index=[f's{i}' for i in range(n)])
    return t, data, ts.run_trend(data, t, variable='Age', unit='weeks')


def _event_at(fig, ax, x, y, button=1, dblclick=False):
    from matplotlib.backend_bases import MouseEvent
    FigureCanvasAgg(fig).draw()
    px, py = ax.transData.transform((x, y))
    return MouseEvent('button_press_event', fig.canvas, px, py, button=button,
                      dblclick=dblclick)


@pytest.mark.numpy_only
def test_volcano_marks_curve_only_features_and_finds_clicked_points():
    _t, _data, res = _result()
    tb = res.table.set_index('feature')
    assert tb.loc['peaks', 'significant'] and tb.loc['peaks', 'curve.only']
    fig = tp.make_trend_volcano_figure(res.table, 'Frequencies', x_label='Slope')
    assert set(fig._point_keys) == set(res.table['feature'])
    i = fig._point_keys.index('rises')
    x, y = fig._point_xy[i]
    assert tp.point_under(fig, _event_at(fig, fig._point_axes, x, y)) == 'rises'
    assert tp.point_under(fig, _event_at(fig, fig._point_axes, 0.0, 1e-3)) != 'rises'


@pytest.mark.numpy_only
def test_trend_plot_draws_samples_curve_and_band():
    t, data, res = _result()
    inputs = tp.feature_plot_inputs(res, data, list(data.index), t, ['Blood'] * t.size, 'peaks')
    assert inputs['samples'] == list(data.index) and inputs['curve_se'] is not None
    fig = tp.make_trend_plot_figure(**inputs, title='peaks', value_label='log2 odds',
                                    show_line=True)
    FigureCanvasAgg(fig).draw()
    ax = fig.axes[0]
    assert len(ax.collections) >= 3          # band, samples, hover overlay
    assert 'partial R²' in ax.get_title()
    table = tp.trend_plot_table(inputs['time'], inputs['values'], inputs['groups'],
                                inputs['samples'], grid=inputs['grid'], curve=inputs['curve'],
                                curve_se=inputs['curve_se'], df2=inputs['row']['df2'],
                                variable='Age')
    fit = table[table['kind'] == 'fit']
    assert len(fit) == res.grid.size and (fit['lower_95'] < fit['fitted']).all()
    with pytest.raises(KeyError):
        tp.feature_plot_inputs(res, data, list(data.index), t, ['F'] * t.size, 'absent')


@pytest.mark.numpy_only
def test_interaction_plot_shows_the_within_pair_differences():
    t = np.repeat(np.linspace(18, 37, 30), 2)
    groups = ['Blood', 'Spleen'] * 30
    pairs = list(np.repeat([f'd{i}' for i in range(30)], 2))
    rng = np.random.default_rng(4)
    is_m = np.array(groups) == 'Spleen'
    data = pd.DataFrame({'diverges': np.where(is_m, 0.2 * t, 0.0),
                         **{f'n{j}': np.zeros(t.size) for j in range(10)}},
                        index=[f's{i}' for i in range(t.size)]) + rng.normal(scale=0.3, size=(t.size, 11))
    res = ts.run_trend(data, t, group_vec=groups, baseline='Blood', interaction=True,
                       pairing_vec=pairs)
    inputs = tp.feature_plot_inputs(res, data, list(data.index), t, groups, 'diverges',
                                    dict(zip(data.index, pairs)))
    assert inputs['interaction'] and len(inputs['pairs']) == t.size
    fig = tp.make_trend_plot_figure(**inputs, title='diverges')
    FigureCanvasAgg(fig).draw()
    assert len(fig.axes) == 2
    assert 'Spleen − Blood' in fig.axes[1].get_ylabel()
    assert inputs['group_curves'] is None


@pytest.mark.numpy_only
def test_unpaired_interaction_plot_draws_each_groups_curve_on_one_panel():
    t = np.concatenate([np.linspace(18, 37, 30), np.linspace(18, 37, 30)])
    groups = ['A'] * 30 + ['B'] * 30
    rng = np.random.default_rng(4)
    is_b = np.array(groups) == 'B'
    data = pd.DataFrame({'diverges': np.where(is_b, 0.2 * t, 0.0),
                         **{f'n{j}': np.zeros(t.size) for j in range(10)}},
                        index=[f's{i}' for i in range(t.size)]) + rng.normal(scale=0.3, size=(t.size, 11))
    res = ts.run_trend(data, t, group_vec=groups, baseline='A', interaction=True)
    inputs = tp.feature_plot_inputs(res, data, list(data.index), t, groups, 'diverges')
    assert set(inputs['group_curves']) == {'A', 'B'}
    colours = {'A': '#1f77b4', 'B': '#d62728'}
    fig = tp.make_trend_plot_figure(**inputs, title='diverges', group_colours=colours)
    FigureCanvasAgg(fig).draw()
    assert len(fig.axes) == 1
    lines = [ln for ln in fig.axes[0].get_lines() if ln.get_linewidth() == 1.8]
    assert {ln.get_color() for ln in lines} == set(colours.values())
    assert len(fig.axes[0].collections) >= 4
    table = tp.trend_plot_table(inputs['time'], inputs['values'], inputs['groups'],
                                inputs['samples'], grid=inputs['grid'], curve=inputs['curve'],
                                curve_se=inputs['curve_se'], df2=float(inputs['row']['df2']),
                                group_curves=inputs['group_curves'])
    fits = table[table['kind'] == 'fit']
    assert set(fits['group']) == {'A', 'B'} and fits['lower_95'].notna().all()
    assert (table['kind'] == 'difference').sum() == res.grid.size


@pytest.mark.numpy_only
def test_composition_figure_stacks_the_fit_over_the_samples():
    t = np.linspace(18, 37, 30)
    rng = np.random.default_rng(1)
    counts = pd.DataFrame(rng.poisson([[400, 300, 50]] * t.size), columns=['Naive', 'TCM', 'Other'],
                          index=[f's{i}' for i in range(t.size)])
    comp = ts.fit_composition(counts, t, variable='Age', unit='weeks')
    panels = [dict(comp, label='Blood'), dict(comp, label='Spleen')]
    fig = tp.make_composition_figure(panels, title='CD4 Tconv', colours={'Naive': '#123456'})
    FigureCanvasAgg(fig).draw()
    assert len(fig.axes) == 4
    assert fig.axes[0].collections[0].get_facecolor()[0][:3] == pytest.approx((0x12 / 255, 0x34 / 255, 0x56 / 255))
    table = tp.composition_table(panels)
    fit = table[(table['kind'] == 'fit') & (table['panel'] == 'Blood')]
    np.testing.assert_allclose(fit.groupby('Age')['percent'].sum(), 100.0)
    assert (table['kind'] == 'sample').sum() == 2 * 3 * t.size
    assert tp.part_colours(['a', 'Other'])['Other'] == tp.OTHER_COLOUR


@pytest.mark.numpy_only
def test_embedding_by_variable_weights_samples_equally_and_maps_the_shift():
    rng = np.random.default_rng(5)
    embeddings, time = {}, {}
    for i, age in enumerate(np.linspace(18, 37, 16)):
        late = (age - 18) / 19
        n = 300 if i % 2 else 3000          # sample size must not matter
        k = int(n * late)
        xy = np.vstack([rng.normal([5, 5], 0.5, size=(k, 2)),
                        rng.normal([-5, -5], 0.5, size=(n - k, 2))])
        embeddings[f's{i}'] = xy
        time[f's{i}'] = age
    time['no embedding'] = 20.0
    embeddings['no value'] = rng.normal(size=(50, 2))
    dens = tp.embedding_densities(embeddings, time, n_bins=4)
    assert len(dens['bins']) == 4 and sum(len(b['samples']) for b in dens['bins']) == 16
    assert dens['n_left_out'] == 2
    for b in dens['bins']:
        assert b['density'].sum() == pytest.approx(1.0, rel=1e-6)
    x0, x1, y0, y1 = dens['extent']
    size = dens['difference'].shape[0]
    to_px = lambda v, a, b: int((v - a) / (b - a) * size)
    assert dens['difference'][to_px(5, y0, y1), to_px(5, x0, x1)] > 0
    assert dens['difference'][to_px(-5, y0, y1), to_px(-5, x0, x1)] < 0
    fig = tp.make_embedding_by_variable_figure(dens, title='UMAP by age', variable='age',
                                               unit='weeks', axis_labels=('UMAP1', 'UMAP2'))
    FigureCanvasAgg(fig).draw()
    table = tp.embedding_bins_table(dens, time, 'age')
    assert len(table) == 16 and set(table['bin']) == {1, 2, 3, 4}
    points = tp.embedding_densities(embeddings, time, levels=[18.0, 37.0])
    assert [b['low'] for b in points['bins']] == [18.0, 37.0]


def _age_result():
    rng = np.random.default_rng(2)
    n = 30
    samples = [f's{i}' for i in range(n)]
    actual = np.linspace(18, 37, n)
    predicted = actual + rng.normal(scale=1.5, size=n)
    preds = pd.DataFrame({'sample': samples + ['other'], 'actual': list(actual) + [25.0],
                          'predicted': list(predicted) + [27.0], 'predicted_sd': 0.3,
                          'gap': list(predicted - actual) + [2.0],
                          'gap_corrected': list(predicted - actual) + [1.5],
                          'cross_validated': [True] * n + [False]})
    return {'performance': pd.DataFrame([{'mae_mean': 1.2, 'mae_sd': 0.1, 'r2_mean': 0.8,
                                          'r2_sd': 0.02, 'spearman_mean': 0.9,
                                          'spearman_sd': 0.01, 'mae_baseline': 4.0,
                                          'p_chance': 0.01}]),
            'predictions': preds,
            'importance': pd.DataFrame({'feature': ['Treg', 'B cells', 'NK'],
                                        'selection': [1.0, 0.6, 0.0], 'coef': [0.5, -0.2, 0.0],
                                        'direction': [1, -1, 0]}),
            'chance': pd.DataFrame({'mae': rng.normal(4.0, 0.2, 50)}),
            'summary': {'n': n, 'n_features': 3, 'n_permutations': 50}}


@pytest.mark.numpy_only
def test_age_model_figures():
    res = _age_result()
    groups = {f's{i}': 'Blood' for i in range(30)}
    groups['other'] = 'Cord'
    fig = tp.make_age_model_figure(res, variable='Age', unit='years', groups=groups,
                                   group_colours={'Blood': '#112233'})
    FigureCanvasAgg(fig).draw()
    assert len(fig.axes) == 4
    assert 'not in the model' in ' '.join(t.get_text() for t in fig.axes[0].get_legend().get_texts())
    assert 'Cord (predicted)' in [t.get_text() for t in fig.axes[3].get_xticklabels()]
    one = tp.make_age_model_figure(res, variable='Age', unit='years')
    assert len(one.axes) == 3
    feats = tp.make_age_features_figure(res, variable='Age', keys={'Treg': 'k1', 'B cells': 'k2'},
                                        kinds={'Treg': 'freq', 'B cells': 'mfi'})
    FigureCanvasAgg(feats).draw()
    assert [t.get_text() for t in feats.axes[0].get_yticklabels()] == ['Treg', 'B cells']
    assert feats._row_keys == ['k2', 'k1'] and feats._row_kinds == ['mfi', 'freq']
    assert not feats.axes[0].xaxis.get_gridlines()[0].get_visible()


@pytest.mark.numpy_only
def test_trajectory_heatmap_orders_called_features_by_the_peak():
    _t, _data, res = _result()
    fig = tp.make_trajectory_heatmap_figure(res.table, res.grid, res.curves,
                                            title='Frequencies', variable='Age',
                                            unit='weeks')
    called = set(res.table.loc[res.table['significant'], 'feature'])
    assert set(fig._row_keys) == called
    at_max = res.table.set_index('feature').loc[fig._row_keys, 'at.max'].to_numpy()
    assert (np.diff(at_max) >= 0).all()
    none = res.table.assign(significant=False)
    msg = tp.make_trajectory_heatmap_figure(none, res.grid, res.curves, title='x',
                                            variable='age')
    assert not hasattr(msg, '_row_keys')


@pytest.mark.numpy_only
def test_scatter_by_value_adds_a_colour_bar_and_marks_samples_without_a_value():
    from matplotlib.figure import Figure
    fig = Figure()
    ax = fig.add_subplot(111)
    sc = tp.scatter_by_value(fig, ax, [0, 1, 2, 3], [1, 0, 1, 0], [20, 25, None, 30],
                             ['A', 'A', 'B', 'B'], label='Age (weeks)')
    assert sc is not None and len(fig.axes) == 2
    labels = [t.get_text() for t in ax.legend().get_texts()]
    assert 'no value' in labels and 'A' in labels and 'B' in labels


@pytest.mark.numpy_only
def test_trend_plot_draws_into_the_axes_of_a_larger_figure():
    from matplotlib.figure import Figure
    t, data, res = _result()
    inputs = tp.feature_plot_inputs(res, data, list(data.index), t, ['Blood'] * t.size, 'rises')
    fig = Figure(figsize=(9, 4), constrained_layout=True)
    axes = fig.subplots(1, 2)
    out = tp.make_trend_plot_figure(**inputs, title='rises', axes=[axes[1]])
    assert out is fig and not hasattr(fig, '_hover_handler')
    assert axes[1].get_xlabel() == 'Age (weeks)' and not axes[0].lines
    FigureCanvasAgg(fig).draw()
    assert tp.trend_p_text(inputs['row']).endswith('called')
    assert tp.trend_p_text(None) == 'not tested'


@pytest.mark.numpy_only
def test_percent_view_draws_the_fit_back_on_the_percent_scale():
    t, data, res = _result()
    inputs = tp.feature_plot_inputs(res, data, list(data.index), t, ['F'] * t.size, 'rises')
    fig = tp.make_trend_plot_figure(**inputs, title='', back_transform='percent',
                                    figsize=(2.2, 2.2), compact=True, transparent=True)
    curve = fig.axes[0].lines[0].get_ydata()
    expected = 100.0 / (1.0 + 2.0 ** -np.asarray(inputs['curve']))
    np.testing.assert_allclose(curve, expected)
    assert fig.patch.get_alpha() == 0.0 and fig.axes[0].get_legend() is None
    table = tp.trend_plot_table(inputs['time'], inputs['values'], inputs['groups'],
                                inputs['samples'], grid=inputs['grid'], curve=inputs['curve'],
                                curve_se=inputs['curve_se'], back_transform='percent')
    fit = table[table['kind'] == 'fit']
    np.testing.assert_allclose(fit['fitted'], expected)

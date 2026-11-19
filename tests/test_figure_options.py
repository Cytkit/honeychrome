"""
test_figure_options.py
----------------------
Appearance options of the plugins' figures
(view_components/figure_options.py): applying and undoing them, and
keeping them per experiment, scope and figure.

Usage:
    pytest tests/test_figure_options.py -m numpy_only
"""

import json

import pytest

pytest.importorskip('matplotlib')
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from honeychrome.view_components import differential_plots as dp  # noqa: E402
from honeychrome.view_components import figure_options as fo  # noqa: E402


def _figure(is_dark=False):
    fig = Figure(figsize=(5, 4), layout='constrained')
    ax1, ax2 = fig.subplots(1, 2)
    for ax, name in ((ax1, 'left panel'), (ax2, 'right panel')):
        ax.plot([0, 1], [0, 1], label='line')
        ax.grid(True)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('x', fontsize=9)
        ax.text(0.5, 0.5, 'note', fontsize=7)
    ax1.legend(fontsize=7)
    fig.suptitle('Main title', fontsize=10)
    dp.style_figure_theme(fig, is_dark)
    dp.stamp_run_label(fig, 'Trend with age')
    return fig


class _Backing:
    def __init__(self):
        self.values = {}

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value

    def sync(self):
        pass


@pytest.mark.numpy_only
def test_options_change_the_figure_and_can_be_undone():
    fig = _figure()
    ax1, ax2 = fig.axes
    fo.apply_figure_options(fig, fo.FigureOptions())
    FigureCanvasAgg(fig).draw()
    assert not any(line.get_visible() for line in ax1.get_xgridlines())
    assert fig._run_label.get_visible()

    opts = fo.FigureOptions(grid=True, run_label=False, title='custom', title_text='Figure 2',
                            panel_titles=False, title_size=14, label_size=12, tick_size=11,
                            legend_size=10, text_size=6, background='black')
    fo.apply_figure_options(fig, opts)
    FigureCanvasAgg(fig).draw()
    assert any(line.get_visible() for line in ax1.get_xgridlines())
    assert not fig._run_label.get_visible()
    assert fig._suptitle.get_text() == 'Figure 2' and fig._suptitle.get_fontsize() == 14
    assert ax1.title.get_text() == '' and ax2.title.get_text() == ''
    assert ax1.xaxis.label.get_fontsize() == 12
    assert ax1.get_xticklabels()[0].get_fontsize() == 11
    assert ax1.get_legend().get_texts()[0].get_fontsize() == 10
    assert ax1.texts[0].get_fontsize() == 6
    assert to_rgba(fig.get_facecolor()) == to_rgba('black')
    assert to_rgba(ax1.xaxis.label.get_color()) == to_rgba('white')

    fo.apply_figure_options(fig, fo.FigureOptions(title='hidden'))
    assert fig._suptitle.get_text() == ''
    fo.apply_figure_options(fig, fo.FigureOptions())
    FigureCanvasAgg(fig).draw()
    assert fig._suptitle.get_text() == 'Main title' and fig._suptitle.get_fontsize() == 10
    assert ax1.title.get_text() == 'left panel' and ax1.xaxis.label.get_fontsize() == 9
    assert to_rgba(fig.get_facecolor()) == to_rgba('white')
    assert to_rgba(ax1.xaxis.label.get_color()) == to_rgba('black')
    assert fig._run_label.get_visible()


@pytest.mark.numpy_only
def test_transparent_background_and_polar_grid_kept():
    fig = Figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='polar')
    ax.plot([0, 1], [1, 1])
    fo.apply_figure_options(fig, fo.FigureOptions(background='transparent'))
    assert fig.get_facecolor()[3] == 0
    assert any(line.get_visible() for line in ax.get_xgridlines())


@pytest.mark.numpy_only
def test_store_keeps_defaults_and_each_figures_own_options():
    backing = _Backing()
    store = fo.FigureOptionsStore(backing)
    seen = []
    store.subscribe(seen.append)
    store.set('ag', 'volcano', fo.FigureOptions(grid=True, title='custom', title_text='A'))
    assert store.options('ag', 'volcano').grid and not store.options('ag', 'heatmap').grid
    assert seen == ['ag']
    store.set_defaults('ag', fo.FigureOptions(tick_size=12, title='custom', title_text='ignored'))
    volcano = store.options('ag', 'volcano')
    assert volcano.tick_size == 12 and not volcano.grid and volcano.title_text == 'A'
    assert store.options('ag', 'heatmap').title == 'auto'
    again = fo.FigureOptionsStore(backing)
    assert again.options('ag', 'volcano').title_text == 'A'
    assert json.loads(backing.values['options'])['ag']['defaults']['tick_size'] == 12
    store.reset('ag', 'volcano')
    assert not store.has_own('ag', 'volcano') and store.options('ag', 'volcano').title == 'auto'
    assert store.options('hc', 'volcano').tick_size == 0


@pytest.mark.numpy_only
def test_with_options_applies_to_every_new_figure():
    store = fo.FigureOptionsStore()
    store.set('ag', 'f', fo.FigureOptions(title='hidden'))
    ref = fo.FigureOptionsRef(store, 'ag', 'f')
    make = fo.with_options(_figure, ref)
    assert make()._suptitle.get_text() == ''
    assert fo.with_options(_figure, None) is _figure

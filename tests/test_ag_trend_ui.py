"""
test_ag_trend_ui.py
-------------------
The Automated Gating Results page in trend mode, on synthetic gated samples
whose T-cell fraction rises with age (test_ag_results._aged_summaries): the
trend settings and design check, a trend run with its tables and figures,
a feature's trend plot, the Populations page and report items in trend
mode, the sample PCA coloured by the variable, calls redone
without refitting, and switching back to group comparisons.

Needs pytest-qt (pip install pytest-qt); without it the module is skipped.

Usage:
    pytest tests/test_ag_trend_ui.py -m ui
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip('pytestqt', reason='pytest-qt is not installed: pip install pytest-qt')

_REPO = Path(__file__).resolve().parents[1]
for folder in (_REPO / 'src' / 'honeychrome' / 'bundled_plugins', Path(__file__).resolve().parent):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
if os.environ.get('HONEYCHROME_SHOW_UI', '').lower() not in ('1', 'true', 'yes', 'on'):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from test_ag_results import _aged_summaries  # noqa: E402

pytestmark = pytest.mark.ui


def _controller():
    settings = {'unmixed': {'event_channels_pnn': ['A', 'B', 'C']}}
    return SimpleNamespace(experiment=SimpleNamespace(settings=settings, process={}, samples={}),
                           experiment_dir='', unmixed_transformations={})


def test_trend_mode_runs_and_shows_its_tables(qtbot):
    import automated_gating_tab as agt
    from honeychrome.view_components.event_bus import EventBus

    agt._ensure_qt_imports()
    gates, summaries, groups, time = _aged_summaries()
    state = agt.GatingRunState(
        applied_model={'gate_definitions': gates}, summaries=summaries,
        run_samples=list(summaries), group_names=['Blood'], sample_groups=groups,
        covariates={'age': {k: str(v) for k, v in time.items()},
                    'batch': {k: ('early' if v < 27 else 'late') for k, v in time.items()}},
        numeric_covariates=['age'], stats_test='trend', trend_unit='weeks', trend_min_r2=0.0)
    tab = agt.ResultsTab(state, EventBus(), _controller())
    qtbot.addWidget(tab)
    tab.refresh()

    assert not tab.trend_box.isHidden() and tab.btn_stats.text() == "Run Trend Test"
    assert state.trend_variable == 'age'
    check = tab.lbl_trend_check.text()
    assert '24 sample(s) with a value; the curve gets 2 df' in check
    assert '⚠ batch follows age' in check

    with qtbot.waitSignal(tab.trend_finished, timeout=60000):
        tab._on_run_stats()
    assert state.trend and state.trend['freq'] is not None
    assert tab.results_tabs.tabText(0) == 'Frequencies'
    called = int(state.trend['freq'].table['significant'].sum())
    assert called >= 1 and 'Trend with age: 24 samples' in tab.lbl_stats.text()

    qtbot.waitUntil(lambda: tab._figure_worker is None and tab.figures_tabs.count() > 0,
                    timeout=30000)
    titles = [tab.figures_tabs.tabText(i) for i in range(tab.figures_tabs.count())]
    assert 'Frequency trend volcano' in titles and 'Frequency trajectories' in titles
    assert 'Frequency patterns' in titles
    inputs = tab.trend_enrichment_inputs()
    assert inputs['variable'] == 'age' and len(inputs['samples']) == 24
    assert inputs['patterns'] is not None and inputs['design']['groups']
    assert 'pattern' in state.trend['freq'].table.columns
    tab.sb_trend_patterns.setValue(2)
    assert state.trend_patterns == 2 and state.trend['freq'].settings['pattern_k'] == 2
    tab.sb_trend_patterns.setValue(0)
    qtbot.waitUntil(lambda: tab._figure_worker is None, timeout=30000)
    feature = str(state.trend['freq'].table['feature'].iloc[0])
    tab._trend_plot_window('freq', feature)
    qtbot.waitUntil(lambda: not tab._popout_workers, timeout=30000)
    from PySide6.QtWidgets import QDialog
    assert any(d.windowTitle().startswith('Trend ') for d in tab.findChildren(QDialog))

    data = tab._population_data(feature)
    assert data['trend'] is not None and 'curve FDR' in data['p_text']
    assert data['trend_percent']['back_transform'] == 'percent'
    assert 'trend with age' in data['title']
    report_fig = tab._population_report_figure(feature)
    assert report_fig is not None and len(report_fig.axes) >= 1
    items = {it.key: it for it in tab._report_items()}
    assert 'trend_freq_volcano:age' in items and 'trend_freq_heatmap:age' in items
    assert items['trend_freq_volcano:age'].get_figure() is not None
    assert 'trend_frequencies' in items['trend_freq_volcano:age'].get_tables()
    assert 'Trend with a numeric variable:' in tab._settings_text(list(summaries))
    assert 'changing with age' in tab.lbl_pop_count.text()

    tab.chk_pca_freq.setChecked(True)
    tab._on_show_pca()
    assert len(state.pca['colour_values']) == len(state.pca['scores'])
    assert state.pca['colour_label'] == 'age (weeks)'

    tab.sb_trend_r2.setValue(1.0)
    assert int(state.trend['freq'].table['significant'].sum()) == 0
    tab.sb_trend_r2.setValue(0.0)
    assert int(state.trend['freq'].table['significant'].sum()) == called

    tab.cb_test.setCurrentIndex(tab.cb_test.findData('groups'))
    assert state.stats_test == 'groups'
    assert tab.trend_box.isHidden() and tab.btn_stats.text() == "Run Statistics"
    assert tab.results_tabs.count() == 0
    assert all(not tab.figures_tabs.tabText(i).startswith('Frequency trend')
               for i in range(tab.figures_tabs.count()))


def _fake_enrichment(label: str, kind: str) -> dict:
    states = pd.DataFrame([{'comparison': label, 'immune_state': 'State', 'direction': 'gained',
                            'NES': 1.5, 'p': 0.01, 'FDR': 0.02, 'FWER': 0.05, 'n_tested': 6,
                            'leading_edge': '', 'n_leading': 3, 'n_library': 6, 'note': ''}])
    out = {'settings': {'kind': kind, 'engine': 'test', 'n_perm': 100}, 'seconds': 1.0,
           'samples': [], 'library': {}, 'states': states, 'members': pd.DataFrame(),
           'ranks': pd.DataFrame(), 'notes': [], 'blocked': []}
    if kind == 'trend':
        out.update(labels=[label], comparisons=[])
    else:
        base, other = label.split(' vs ')[1], label.split(' vs ')[0]
        out['comparisons'] = [[base, other]]
    return out


def test_enrichment_page_keeps_group_and_trend_results(qtbot):
    import ag_fsea_ui
    import automated_gating_tab as agt
    from honeychrome.view_components.event_bus import EventBus

    agt._ensure_qt_imports()
    gates, summaries, _groups, time = _aged_summaries()
    keys = list(summaries)
    groups = {k: ('Blood' if i % 2 else 'Spleen') for i, k in enumerate(keys)}
    state = agt.GatingRunState(applied_model={'gate_definitions': gates}, summaries=summaries,
                               run_samples=keys, group_names=['Blood', 'Spleen'],
                               sample_groups=groups, reference_group='Blood')
    trend = {'samples': keys, 'time': time, 'variable': 'age', 'unit': 'weeks',
             'scale': 'linear', 'shape': 'curve', 'k': 2, 'interaction': False,
             'baseline': 'Blood', 'groups': ['Blood', 'Spleen'],
             'design': agt.statistics_design(state, keys), 'patterns': None}
    tab = ag_fsea_ui.EnrichmentTab(state, EventBus(), _controller(), samples_fn=lambda: keys,
                                   design_fn=lambda s: agt.statistics_design(state, s),
                                   antigens_fn=dict, experiment_dir_fn=lambda: '',
                                   trend_fn=lambda: trend)
    qtbot.addWidget(tab)
    tab.refresh()
    texts = [tab.cb_comparison.itemText(i) for i in range(tab.cb_comparison.count())]
    assert 'Spleen vs Blood' in texts and texts[-1] == 'Trend with age'
    assert tab.btn_run.text() == "Run enrichment: group comparisons"

    tab._on_run_done(True, '', _fake_enrichment('Spleen vs Blood', 'groups'))
    tab.cb_comparison.setCurrentIndex(texts.index('Trend with age'))
    assert tab.btn_run.text() == "Run enrichment: Trend with age"
    assert 'No enrichment run yet' in tab.lbl_status.text()
    tab._on_run_done(True, '', _fake_enrichment('Trend with age', 'trend'))
    assert state.fsea and state.fsea_trend
    assert tab._selected() == ('trend', 'Trend with age') and tab.tbl_results.rowCount() == 1
    tab.cb_comparison.setCurrentIndex(texts.index('Spleen vs Blood'))
    assert tab._selected() == ('groups', 'Spleen vs Blood')
    assert tab.tbl_results.rowCount() == 1 and state.fsea['comparisons'] == [['Blood', 'Spleen']]

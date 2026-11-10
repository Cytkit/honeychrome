"""
test_drc_trend_ui.py
--------------------
The Honeycluster Stats tab in trend mode: the Analysis choice shows the
trend settings and the design check, the Run button follows the trend's own
sample rule, a trend run's results open in the Trend tables tab, calls are
redone without refitting, and switching back hides the trend settings.

Needs pytest-qt (pip install pytest-qt); without it the module is skipped.

Usage:
    pytest tests/test_drc_trend_ui.py -m ui
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('pytestqt', reason='pytest-qt is not installed: pip install pytest-qt')

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))
if os.environ.get('HONEYCHROME_SHOW_UI', '').lower() not in ('1', 'true', 'yes', 'on'):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytestmark = pytest.mark.ui


def _state(drt):
    """Twenty-four blood samples aged 18 to 37 weeks; cluster 0's share
    rises with age and a batch covariate splits young from old."""
    rng = np.random.default_rng(7)
    ages = np.linspace(18, 37, 24)
    state = drt.PipelineState()
    state.group_names = ['Blood']
    keys = [f's{i}' for i in range(24)]
    state.sample_groups = {k: 'Blood' for k in keys}
    labels = {}
    for k, age in zip(keys, ages):
        p0 = 0.1 + 0.3 * (age - 18) / 19
        labels[k] = rng.choice(4, size=4000, p=[p0] + [(1 - p0) / 3] * 3)
    state.cluster_labels = labels
    state.covariates = pd.DataFrame({'age': [f'{a:.2f}' for a in ages],
                                     'batch': ['early' if a < 27 else 'late' for a in ages]},
                                    index=keys)
    state.numeric_covariates = ['age']
    return state


def test_trend_mode_settings_check_and_tables(qtbot):
    import dr_clustering_tab as drt
    import drc_stats
    from honeychrome.view_components.event_bus import EventBus

    drt._ensure_qt_imports()
    state = _state(drt)
    controller = SimpleNamespace(
        experiment=SimpleNamespace(settings={'raw': {'raw_samples_subdirectory': 'Raw'}}))
    tab = drt.GroupsStatsTab(state, EventBus(), controller)
    qtbot.addWidget(tab)
    tab._load_trend_controls()
    assert tab.trend_box.isHidden()

    tab.stats_test_combo.setCurrentIndex(tab.stats_test_combo.findData('trend'))
    assert state.stats_test == 'trend' and state.trend_variable == 'age'
    assert not tab.trend_box.isHidden() and tab.run_stats_btn.text() == "▶  Run Trend Test"
    assert state.trend_stats_runnable() and not state.n_group_stats_runnable()
    check = tab.trend_check_label.text()
    assert '24 sample(s) with a value; the curve gets 2 df' in check
    assert '⚠ batch follows age' in check

    result = drc_stats.run_trend_statistics(controller, state, run_freq=True, run_mfi=False,
                                            pval_threshold=0.05)
    state.trend_results = result
    tab._show_trend_tables()
    tab._show_trend_figures()
    assert tab._has_results_tab('trend_tables')
    assert tab._has_results_tab('trend_freq_volcano') and tab._has_results_tab('trend_freq_heatmap')
    assert tab._has_results_tab('trend_freq_patterns')
    tab.trend_patterns_spin.setValue(2)
    assert state.trend_results['freq'].settings['pattern_k'] == 2
    tab.trend_patterns_spin.setValue(0)
    current = tab._results_tabs.widget(tab._results_tabs.currentIndex())
    assert current.property('_tab_key') == 'trend_tables'
    feature = str(state.trend_results['freq'].table['feature'].iloc[0])
    fig = tab._make_trend_plot('freq', feature)
    assert hasattr(fig, '_hover_handler')
    tab._trend_plot_tab('freq', feature)
    assert tab._has_results_tab(f'trendplot:freq:{feature}')
    values = tab._trend_plot_values('freq', feature)
    assert set(values['kind']) == {'sample', 'fit'}
    called = int(state.trend_results['freq'].table['significant'].sum())
    assert called >= 1
    tab.trend_r2_spin.setValue(1.0)
    assert int(state.trend_results['freq'].table['significant'].sum()) == 0
    tab.trend_r2_spin.setValue(0.1)
    assert int(state.trend_results['freq'].table['significant'].sum()) == called

    tab.stats_test_combo.setCurrentIndex(tab.stats_test_combo.findData('groups'))
    assert state.stats_test == 'groups'
    assert tab.trend_box.isHidden() and tab.run_stats_btn.text() == "▶  Run Statistics"

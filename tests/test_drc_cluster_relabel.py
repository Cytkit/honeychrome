"""
test_drc_cluster_relabel.py
---------------------------
Renaming a run's clusters after its statistics were computed gives the
stored tables the new names without refitting (drc_stats.relabel_stats_results,
drc_stats.relabel_composition).

Usage:
    pytest tests/test_drc_cluster_relabel.py
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

NAMES = {0: 'T', 1: 'B', 2: 'NK'}
RENAMED = {0: 'B', 1: 'T', 2: 'NK cells'}     # T and B swapped, NK renamed
CHANNELS = ['CD4', 'CD8']


class _State:
    """Twelve samples in two groups, three clusters."""
    testing_group_selection = ['A', 'B']
    group_names = ['A', 'B']
    contrast_mode = 'reference'
    reference_group = 'A'
    paired = False
    pairing_variable = None
    covariates = None
    numeric_covariates = []

    def __init__(self):
        rng = np.random.default_rng(5)
        self.cluster_labels, self.sample_groups = {}, {}
        for i in range(12):
            group = 'A' if i < 6 else 'B'
            p = [0.5, 0.3, 0.2] if group == 'A' else [0.3, 0.5, 0.2]
            self.cluster_labels[f's{i}'] = rng.choice(3, size=2000, p=p)
            self.sample_groups[f's{i}'] = group
        for attr in ('counts_df', 'counts_results', 'mfi_df', 'mfi_results', 'mfi_sample_df',
                     'pca_loadings_df', 'predictive_result', 'composition_df'):
            setattr(self, attr, None)
        self.stats_cluster_names = {}
        self.composition_names = {}

    def cluster_label(self, cl):
        return str(cl)


def _state_with_results():
    controller = SimpleNamespace(experiment=SimpleNamespace(
        settings={'raw': {'raw_samples_subdirectory': 'Raw'}}))
    state = _State()
    drc_stats.run_statistics(controller, state, run_freq=True, run_mfi=False,
                             pval_threshold=0.05, fc_threshold=0.0, names_override=NAMES)
    features = [f'{NAMES[cl]}_{ch}' for ch in CHANNELS for cl in NAMES]
    state.mfi_df = pd.DataFrame(np.arange(12 * 6, dtype=float).reshape(12, 6),
                                index=state.freq_df.index, columns=features)
    state.mfi_sample_df = pd.DataFrame(np.ones((12, 2)), index=state.freq_df.index,
                                       columns=CHANNELS)
    state.mfi_results = pd.DataFrame({
        'feature': features,
        'cluster': [NAMES[cl] for ch in CHANNELS for cl in NAMES],
        'cluster_id': [cl for ch in CHANNELS for cl in NAMES],
        'channel': [ch for ch in CHANNELS for cl in NAMES]})
    state.pca_loadings_df = pd.DataFrame(np.ones((2, 2)), index=['Freq: T', 'MFI: NK_CD8'],
                                         columns=['PC1', 'PC2'])
    state.predictive_result = {'importance': pd.DataFrame({'feature': ['MFI: B_CD4', 'Freq: NK']}),
                               'summary': {}}
    return state


def test_results_take_the_new_names_without_refitting():
    state = _state_with_results()
    freq_before = state.freq_df.copy()
    logfc_before = state.freq_results.set_index('feature')['logFC']

    renames = drc_stats.relabel_stats_results(state, RENAMED)

    assert renames == {'T': 'B', 'B': 'T', 'NK': 'NK cells'}
    assert list(state.freq_df.columns) == ['B', 'T', 'NK cells']
    np.testing.assert_array_equal(state.freq_df.values, freq_before.values)
    logfc_after = state.freq_results.set_index('feature')['logFC']
    assert logfc_after['B'] == logfc_before['T']
    assert logfc_after['NK cells'] == logfc_before['NK']
    assert list(state.mfi_df.columns) == ['B_CD4', 'T_CD4', 'NK cells_CD4',
                                          'B_CD8', 'T_CD8', 'NK cells_CD8']
    assert list(state.mfi_results['feature']) == list(state.mfi_df.columns)
    assert list(state.mfi_results['cluster']) == ['B', 'T', 'NK cells'] * 2
    assert list(state.pca_loadings_df.index) == ['Freq: B', 'MFI: NK cells_CD8']
    assert list(state.predictive_result['importance']['feature']) == ['MFI: T_CD4',
                                                                      'Freq: NK cells']
    assert state.stats_cluster_names == RENAMED
    assert drc_stats.relabel_stats_results(state, RENAMED) == {}


def test_duplicate_names_change_nothing():
    state = _state_with_results()
    with pytest.raises(ValueError):
        drc_stats.relabel_stats_results(state, {0: 'T', 1: 'T', 2: 'NK'})
    assert list(state.freq_df.columns) == ['T', 'B', 'NK']
    assert state.stats_cluster_names == NAMES


def test_results_saved_without_recorded_names_are_read_from_the_tables():
    state = _state_with_results()
    state.stats_cluster_names = {}
    assert drc_stats.stored_cluster_names(state) == NAMES
    drc_stats.relabel_stats_results(state, RENAMED)
    assert list(state.freq_df.columns) == ['B', 'T', 'NK cells']


def test_composition_takes_the_new_names():
    state = _State()
    state.composition_df = pd.DataFrame(np.ones((2, 3)), columns=['T', 'B', 'NK'])
    state.composition_names = dict(NAMES)
    assert drc_stats.relabel_composition(state, RENAMED) == {'T': 'B', 'B': 'T',
                                                             'NK': 'NK cells'}
    assert list(state.composition_df.columns) == ['B', 'T', 'NK cells']
    assert state.composition_names == RENAMED

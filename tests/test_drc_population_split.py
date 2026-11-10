"""
test_drc_population_split.py
----------------------------
Comparing gated populations within samples (drc_stats.split_by_population):
each sample is split into the chosen populations, the populations are the
groups and the design is paired on the sample.

Usage:
    pytest tests/test_drc_population_split.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

drc_stats = pytest.importorskip('drc_stats')
drc_pipeline = pytest.importorskip('drc_pipeline')

GATES = ['CD45.1+', 'CD45.1-']


class _State:
    """Eight chimeras, half of each one's cells CD45.1+. CD45.1+ cells are
    enriched in cluster 0 and brighter on CH1 there; mice differ strongly in
    cluster 1, the same way in both populations."""
    testing_group_selection = ['Chimera']
    group_names = ['Chimera']
    contrast_mode = 'reference'
    reference_group = 'Chimera'
    paired = False
    pairing_variable = ''
    covariates = None
    numeric_covariates = []
    population_split = True
    population_gates = list(GATES)
    population_reference = 'CD45.1-'
    selected_channels = ['CH1']
    marker_roles = {}

    def __init__(self, n_mice=8, n_events=6000, overlap=0.0):
        rng = np.random.default_rng(11)
        self.cluster_labels, self.sample_groups = {}, {}
        self.membership, self.values = {}, {}
        half = n_events // 2
        for i in range(n_mice):
            rel = f'mouse{i}.fcs'
            mouse_c1 = rng.uniform(0.1, 0.5)
            p_neg = np.array([0.15, mouse_c1, 0.3, 0.2])
            p_pos = np.array([0.35, mouse_c1, 0.3, 0.2])
            labels = np.concatenate([rng.choice(4, half, p=p_pos / p_pos.sum()),
                                     rng.choice(4, n_events - half, p=p_neg / p_neg.sum())])
            membership = np.zeros((n_events, 2), dtype=bool)
            membership[:half, 0] = True
            membership[half:, 1] = True
            n_shared = int(round(overlap * n_events))
            membership[half:half + n_shared, 0] = True
            ch1 = rng.normal(1.0, 0.1, n_events)
            ch1[:half][labels[:half] == 0] += 0.5
            self.cluster_labels[rel] = labels
            self.sample_groups[rel] = 'Chimera'
            self.membership[rel] = membership
            self.values[rel] = ch1[:, None].astype(np.float32)

    def cluster_label(self, cl):
        return f'c{cl}'


def _controller():
    settings = {'raw': {'raw_samples_subdirectory': 'Raw'}}
    return SimpleNamespace(experiment=SimpleNamespace(settings=settings))


@pytest.fixture
def patched(monkeypatch):
    def _patch(state):
        def _populations(controller, st, rel, gates, af_state=None):
            cols = [GATES.index(g) for g in gates]
            return state.membership[rel][:, cols], np.arange(len(state.membership[rel]))

        def _values(controller, st, rel, channels, af_state=None):
            return state.values[rel], ['CH1']

        monkeypatch.setattr(drc_pipeline, 'load_sample_populations', _populations)
        monkeypatch.setattr(drc_pipeline, 'load_sample_transformed_values', _values)
        return state
    return _patch


def test_population_key_names_the_sample_and_population():
    key = drc_stats.population_key('Day 1/mouse3.fcs', 'CD45.1+')
    assert Path(key).stem == 'mouse3 · CD45.1+'
    assert Path(key).parent == Path('Day 1')
    assert '/' not in Path(drc_stats.population_key('m.fcs', 'Live/CD45.1+')).name


def test_populations_are_compared_within_each_sample(patched):
    state = patched(_State())
    freq, mfi, _counts = drc_stats.run_statistics(
        _controller(), state, run_freq=True, run_mfi=True,
        pval_threshold=0.05, fc_threshold=0.5, include_type_markers=True,
        mfi_threshold=0.2)
    assert state.stats_comparisons == [('CD45.1-', 'CD45.1+')]
    assert sorted(set(state.stats_group_vec)) == sorted(GATES)
    assert len(state.stats_all_rel) == 16
    assert {tuple(v) for v in state.stats_population_of.values()} == {
        (f'mouse{i}.fcs', g) for i in range(8) for g in GATES}
    significant = set(freq.loc[freq['significant'], 'feature'])
    assert significant == {'c0'}
    # True effect: log2(0.35/1.15 over 0.15/0.95), about 0.95 at the mean mouse.
    assert freq.loc[freq['feature'] == 'c0', 'logFC'].iloc[0] > 0.7
    assert set(mfi.loc[mfi['significant'], 'feature']) == {'c0_CH1'}
    # Frequencies are shares of each population's own events.
    assert np.allclose(state.freq_df.sum(axis=1), 100.0)


def test_small_overlap_is_left_out_and_large_overlap_stops(patched):
    state = patched(_State(overlap=0.004))
    drc_stats.run_statistics(_controller(), state, run_freq=True, run_mfi=False,
                             pval_threshold=0.05, fc_threshold=0.5)
    assert 'left out' in state.stats_alignment_note
    state = patched(_State(overlap=0.05))
    with pytest.raises(RuntimeError, match='share cells'):
        drc_stats.run_statistics(_controller(), state, run_freq=True, run_mfi=False,
                                 pval_threshold=0.05, fc_threshold=0.5)


def test_fewer_than_two_populations_stops(patched):
    state = patched(_State())
    state.population_gates = ['CD45.1+']
    with pytest.raises(RuntimeError, match='at least two populations'):
        drc_stats.run_statistics(_controller(), state, run_freq=True, run_mfi=False,
                                 pval_threshold=0.05, fc_threshold=0.5)


def test_export_tables_name_the_sample_and_population(patched):
    state = patched(_State())
    drc_stats.run_statistics(_controller(), state, run_freq=True, run_mfi=False,
                             pval_threshold=0.05, fc_threshold=0.5)
    state.counts_df = state.mfi_df = state.counts_results = state.mfi_results = None
    table = drc_stats.export_tables(state)['cluster_frequencies']
    assert set(table['sample_path']) == {f'mouse{i}.fcs' for i in range(8)}
    assert set(table['population']) == set(GATES)
    assert list(table['group']) == list(table['population'])

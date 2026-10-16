"""
test_drc_stats_sparse.py
------------------------
Clusters with events in too few samples are left out of the abundance tests
of drc_stats.run_statistics.

Usage:
    pytest tests/test_drc_stats_sparse.py
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


class _State:
    """Twelve samples in two groups whose event totals differ about
    twenty-five-fold. Cluster 3 is empty everywhere and cluster 4 has events
    in two samples only."""
    testing_group_selection = ['A', 'B']
    group_names = ['A', 'B']
    contrast_mode = 'reference'
    reference_group = 'A'
    paired = False
    pairing_variable = None
    covariates = None

    def __init__(self):
        rng = np.random.default_rng(3)
        probabilities = np.array([0.4, 0.3, 0.2, 0.0, 0.0, 0.1])
        self.cluster_labels, self.sample_groups = {}, {}
        for i in range(12):
            group = 'A' if i < 6 else 'B'
            labels = rng.choice(6, size=20000 if group == 'A' else 800, p=probabilities)
            if i in (2, 9):
                labels[:2] = 4
            self.cluster_labels[f's{i}'] = labels
            self.sample_groups[f's{i}'] = group

    def cluster_label(self, cl):
        return f'c{cl}'


def _controller():
    settings = {'raw': {'raw_samples_subdirectory': 'Raw'}}
    return SimpleNamespace(experiment=SimpleNamespace(settings=settings))


def test_frequency_test_skips_clusters_with_events_in_too_few_samples():
    freq, _mfi, _counts = drc_stats.run_statistics(
        _controller(), _State(), run_freq=True, run_mfi=False,
        pval_threshold=0.05, fc_threshold=0.0)
    assert list(freq['feature']) == ['c0', 'c1', 'c2', 'c5']
    assert not freq['significant'].any()

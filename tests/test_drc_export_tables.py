"""
test_drc_export_tables.py
-------------------------
The tables Honeycluster writes for third-party software
(drc_stats.export_tables): sample tables with group, covariates and the
frequency denominator, the cluster list, and the differential results.

Usage:
    pytest tests/test_drc_export_tables.py
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

SAMPLES = ['ctrl/s1.fcs', 'ctrl/s2.fcs', 'trt/s3.fcs']
CLUSTERS = ['T cells', 'B cells', 'Cluster 2']
COUNTS = np.array([[30.0, 60.0, 0.0], [20.0, 70.0, 10.0], [50.0, 40.0, 5.0]])
LABELLED = np.array([100.0, 110.0, 100.0])


def _state(**overrides):
    """Three samples whose labelled events (100, 110, 100) exceed the sum of
    their cluster counts (90, 100, 95): the rest is noise."""
    fields = dict(
        freq_df=pd.DataFrame(COUNTS / LABELLED[:, None] * 100.0, index=SAMPLES, columns=CLUSTERS),
        counts_df=pd.DataFrame(COUNTS, index=SAMPLES, columns=CLUSTERS),
        mfi_df=None,
        freq_results=pd.DataFrame({'feature': CLUSTERS, 'logFC': [0.5, -0.2, 1.1],
                                   'comparison': ['trt vs ctrl'] * 3}),
        counts_results=None,
        mfi_results=None,
        stats_all_rel=SAMPLES,
        stats_group_vec=['ctrl', 'ctrl', 'trt'],
        covariates=pd.DataFrame({'donor': ['d1', 'd2', 'd1']}, index=SAMPLES),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_tables_follow_the_tests_that_were_run():
    tables = drc_stats.export_tables(_state())
    assert list(tables) == ['clusters', 'cluster_frequencies', 'cluster_counts',
                            'differential_frequencies']


def test_sample_tables_carry_group_covariates_and_denominator():
    table = drc_stats.export_tables(_state())['cluster_frequencies']
    assert list(table.columns) == ['sample', 'sample_path', 'group', 'donor', 'n_events'] + CLUSTERS
    assert list(table['sample']) == ['s1.fcs', 's2.fcs', 's3.fcs']
    assert list(table['group']) == ['ctrl', 'ctrl', 'trt']
    assert list(table['donor']) == ['d1', 'd2', 'd1']
    assert list(table['n_events']) == list(LABELLED)


def test_frequencies_are_recomputable_from_counts_and_n_events():
    tables = drc_stats.export_tables(_state())
    counts = tables['cluster_counts'][CLUSTERS].to_numpy()
    n_events = tables['cluster_counts']['n_events'].to_numpy()
    assert np.allclose(counts / n_events[:, None] * 100.0,
                       tables['cluster_frequencies'][CLUSTERS].to_numpy())


def test_cluster_table_lists_events_labels_and_suggestions():
    cell_types = pd.DataFrame({'suggested_type': ['CD4 T', '', 'B']}, index=[0, 1, 2])
    table = drc_stats.export_tables(_state(), {0: 'CD3+', 2: 'CD19+'}, cell_types)['clusters']
    assert list(table.columns) == ['cluster_id', 'cluster_name', 'n_events',
                                   'mem_label', 'suggested_type']
    assert list(table['cluster_id']) == [0, 1, 2]
    assert list(table['cluster_name']) == CLUSTERS
    assert list(table['n_events']) == [100, 170, 15]
    assert list(table['mem_label']) == ['CD3+', '', 'CD19+']
    assert list(table['suggested_type']) == ['CD4 T', '', 'B']


def test_mfi_table_has_no_event_total_and_nothing_is_written_without_results():
    mfi = pd.DataFrame(np.ones((3, 2)), index=SAMPLES, columns=['T cells_CD3', 'B cells_CD19'])
    tables = drc_stats.export_tables(_state(freq_df=None, counts_df=None, freq_results=None,
                                            mfi_df=mfi))
    assert list(tables) == ['cluster_mfis']
    assert list(tables['cluster_mfis'].columns) == ['sample', 'sample_path', 'group', 'donor',
                                                    'T cells_CD3', 'B cells_CD19']
    assert drc_stats.export_tables(_state(freq_df=None, counts_df=None, freq_results=None)) == {}

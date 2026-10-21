"""
test_drc_run_align.py
---------------------
Matching archived runs to events by event key (drc_run_align), the label
overlay in drc_scatter, and the frequency and MFI tables' handling of
unlabelled events.

Usage:
    pytest tests/test_drc_run_align.py
"""

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

import drc_pipeline  # noqa: E402
import drc_run_align as ra  # noqa: E402
import drc_scatter  # noqa: E402
import drc_stats  # noqa: E402

pytestmark = pytest.mark.numpy_only


def test_join_rows_and_align_values():
    ia, ib = ra.join_rows(np.array([2, 5, 9, 11]), np.array([1, 5, 9, 12]))
    assert ia.tolist() == [1, 2] and ib.tolist() == [1, 2]
    out, matched = ra.align_values(np.array([10, 20, 30, 40], np.int32),
                                   np.array([2, 5, 9, 11]), np.array([1, 5, 9, 12]), -2)
    assert out.tolist() == [-2, 20, 30, -2] and out.dtype == np.int32
    assert matched.tolist() == [False, True, True, False]
    vals, _ = ra.align_values(np.array([[1.0], [2.0]], np.float32), np.array([3, 4]),
                              np.array([4, 5]), np.nan)
    assert vals[0, 0] == 2.0 and np.isnan(vals[1, 0])


def test_join_modes():
    keyed = {'event_keys': True}
    legacy_a = {'sync_fingerprint': 'x'}
    legacy_b = {'sync_fingerprint': 'y'}
    legacy_none = {}
    assert ra.runs_join_mode(keyed, dict(keyed)) == ra.JOIN_KEYS
    assert ra.runs_join_mode(legacy_a, dict(legacy_a)) == ra.JOIN_POSITIONAL
    assert ra.runs_join_mode(legacy_a, legacy_b) == ra.JOIN_NONE
    assert ra.runs_join_mode(legacy_none, legacy_none) == ra.JOIN_POSITIONAL
    assert ra.runs_join_mode(keyed, legacy_none) == ra.JOIN_NONE
    assert ra.live_join_mode(keyed, 'anything') == ra.JOIN_KEYS
    assert ra.live_join_mode(legacy_a, 'x') == ra.JOIN_POSITIONAL
    assert ra.live_join_mode(legacy_a, 'z') == ra.JOIN_NONE
    assert ra.live_join_mode(legacy_none, 'z') == ra.JOIN_NONE


def test_pair_arrays_refuses_unverifiable_pairs():
    assert ra.pair_arrays(np.arange(3), 3, ra.JOIN_NONE) == (None, None)
    assert ra.pair_arrays(np.arange(3), 4, ra.JOIN_POSITIONAL) == (None, None)
    vals, matched = ra.pair_arrays(np.arange(3), 3, ra.JOIN_POSITIONAL)
    assert vals.tolist() == [0, 1, 2] and matched.all()


def test_overlay_by_keys_greys_unmatched_points():
    labels = np.array([1, 2, 3], np.int32)
    aligned, ok = drc_scatter.align_labels_to_embedding(
        labels, 4, label_rows=np.array([10, 20, 30]), emb_rows=np.array([5, 10, 20, 30]),
        mode=ra.JOIN_KEYS)
    assert aligned.tolist() == [-1, 1, 2, 3] and not ok
    same_len_other_cells, ok = drc_scatter.align_labels_to_embedding(
        labels, 3, mode=ra.JOIN_NONE)
    assert (same_len_other_cells == -1).all() and not ok


def test_coverage_text():
    cov = ra.coverage_total([ra.Coverage(90, 10, 5, 3), ra.Coverage(10, 0, 0, 0)])
    assert cov == ra.Coverage(100, 10, 5, 3) and cov.unlabelled == 2
    text = ra.coverage_text(cov)
    assert '90.9%' in text and '4.8% new' in text and '1.9% unlabelled' in text


@pytest.fixture
def current_events(monkeypatch):
    """One sample whose current gate holds file rows 0..9, values = row."""
    rows = np.arange(10, dtype=np.int32)
    gated = np.column_stack([rows.astype(float), rows.astype(float) * 2])
    monkeypatch.setattr(drc_pipeline, 'load_sample_rows', lambda c, s, rel, af_state=None: rows)
    monkeypatch.setattr(drc_pipeline, 'sample_abs_path', lambda c, rel: rel)
    monkeypatch.setattr(drc_pipeline, 'load_unmixed_gated',
                        lambda c, s, p, af_state=None, channels=None: gated)
    monkeypatch.setattr(ra, 'transform_with_params',
                        lambda c, g, channels, params: g.astype(np.float32))
    return rows


def _run(**extra):
    run = {'kind': 'clustering', 'event_keys': True, 'channels': ['x', 'y'],
           'params': {'_space': 'raw'}, 'transform_params': {'x': {}, 'y': {}},
           # the run saw rows 3..12 (three since left the gate) with labels 0/1
           'labels': {'s': np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0], np.int32)},
           'event_rows': {'s': np.arange(3, 13)}}
    run.update(extra)
    return run


def test_new_events_labelled_from_codebook(current_events):
    tree = {'node_weights': np.array([[0.0, 0.0], [100.0, 200.0]]),
            'node_to_meta': np.array([5, 6])}
    labels, cov = ra.labels_for_current_events(None, None, _run(tree_data=tree), 's',
                                               ra.JOIN_KEYS)
    assert labels[:3].tolist() == [5, 5, 5]          # rows 0..2 are new: nearest node 0
    assert labels[3:].tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert cov == ra.Coverage(kept=7, dropped=3, new=3, labelled_new=3)


def test_flowsom_with_dr_space_ticked_still_labels_new_events(current_events):
    tree = {'node_weights': np.array([[0.0, 0.0], [100.0, 200.0]]),
            'node_to_meta': np.array([5, 6])}
    run = _run(algorithm='FlowSOM', tree_data=tree, params={'_space': 'dr', '_dr_algo': 'UMAP'})
    labels, cov = ra.labels_for_current_events(None, None, run, 's', ra.JOIN_KEYS)
    assert labels[:3].tolist() == [5, 5, 5] and cov.labelled_new == 3


def test_new_events_unlabelled_without_a_model(current_events):
    labels, cov = ra.labels_for_current_events(
        None, None, _run(params={'_space': 'dr', '_dr_algo': 'UMAP'}), 's', ra.JOIN_KEYS)
    assert (labels[:3] == ra.UNLABELLED).all()
    assert cov.unlabelled == 3


def test_legacy_run_out_of_sync_is_not_matched(current_events):
    run = _run(event_keys=False)
    run.pop('event_rows')
    assert ra.labels_for_current_events(None, None, run, 's', ra.JOIN_NONE) == (None, None)
    labels, _ = ra.labels_for_current_events(None, None, run, 's', ra.JOIN_POSITIONAL)
    assert labels is not None                          # same length and declared in sync


def test_frequencies_leave_unlabelled_events_out():
    state = types.SimpleNamespace(cluster_names={}, cluster_label=lambda cl: str(cl))
    labels = {'s': np.array([0, 0, 1, ra.UNLABELLED, ra.UNLABELLED, -1], np.int32)}
    freq = drc_stats.compute_frequencies(state, ['s'], 2, cluster_labels_override=labels,
                                         names_override={})
    assert freq.loc['s'].tolist() == pytest.approx([50.0, 25.0])
    assert drc_stats.sample_event_totals(state, ['s'], labels).tolist() == [4.0]


def test_mfis_never_truncate_mismatched_samples():
    state = types.SimpleNamespace(cluster_names={}, selected_channels=['x'])
    labels = {'a': np.array([0, 0, 1]), 'b': np.array([0, 1])}
    values = {'a': (np.array([[1.0], [3.0], [10.0]]), ['x']),
              'b': (np.array([[5.0], [7.0], [9.0]]), ['x'])}        # 3 values, 2 labels
    mfi = drc_stats.compute_mfis(None, state, ['a', 'b'], 2, cluster_labels_override=labels,
                                 channels=['x'], names_override={}, values_override=values)
    assert mfi.loc['a'].tolist() == [2.0, 10.0]
    assert mfi.loc['b'].isna().all()


def test_sample_mfis_reuse_the_samples_compute_mfis_loaded(monkeypatch):
    state = types.SimpleNamespace(cluster_names={}, selected_channels=['x'])
    labels = {'a': np.array([0, 1]), 'b': np.array([1, 1])}
    calls = []

    def load(_controller, _state, rel, _channels, af_state=None):
        calls.append(rel)
        return np.array([[1.0], [3.0]]), ['x']

    monkeypatch.setattr(drc_stats.drc_pipeline, 'load_sample_transformed_values', load)
    loaded = {}
    mfi = drc_stats.compute_mfis(None, state, ['a', 'b'], 2, cluster_labels_override=labels,
                                 channels=['x'], names_override={}, loaded=loaded)
    sample = drc_stats.compute_sample_mfis(None, state, ['a', 'b'], channels=['x'], loaded=loaded)
    assert calls == ['a', 'b']
    assert mfi.loc['a'].tolist() == [1.0, 3.0] and np.isnan(mfi.loc['b'].tolist()[0])
    assert sample['x'].tolist() == [2.0, 2.0]


def test_statistics_inputs_for_legacy_run_use_stored_values(monkeypatch):
    run = {'labels': {'s': np.array([0, 1])},
           'marker_values': {'s': (np.array([[100.0, 1.0], [200.0, 2.0]]), ['x', 'y'])}}
    state = types.SimpleNamespace(channel_transform_params={'x': {'id': 0}, 'y': {'id': 0}})
    labels, values, note = ra.statistics_inputs(None, state, run, ra.JOIN_NONE)
    assert labels['s'].tolist() == [0, 1]
    vals, names = values['s']
    assert names == ['x', 'y'] and vals[:, 0].tolist() == [100.0, 200.0]
    assert 'predates event keys' in note
    assert isinstance(pd.DataFrame(vals), pd.DataFrame)

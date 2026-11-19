"""
test_drc_cluster_id_anchor.py
-----------------------------
Cluster ID's reference (anchor) population: choosing it on the gating
hierarchy, MEM scored against it, the Otsu fallback cut on it, and the
channels scored beyond the run's clustering channels.

Usage:
    pytest tests/test_drc_cluster_id_anchor.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

import drc_cluster_id as cid  # noqa: E402

VIABILITY = {'LD'}


def _gating():
    """root > Cells (FSC/SSC) > Singlets (FSC-A/FSC-H) > Live (LD) >
    T cells (CD3) > CD4, CD8."""
    from flowkit import Dimension, GatingStrategy, gates

    def rect(name, x, y):
        return gates.RectangleGate(name, [Dimension(x, range_min=0.0, range_max=1.0),
                                          Dimension(y, range_min=0.0, range_max=1.0)])

    g = GatingStrategy()
    g.add_gate(rect('Cells', 'FSC-A', 'SSC-A'), ('root',))
    g.add_gate(rect('Singlets', 'FSC-A', 'FSC-H'), ('root', 'Cells'))
    g.add_gate(rect('Live', 'LD', 'SSC-A'), ('root', 'Cells', 'Singlets'))
    g.add_gate(rect('T cells', 'BUV395', 'SSC-A'), ('root', 'Cells', 'Singlets', 'Live'))
    g.add_gate(rect('CD4', 'BV421', 'BV605'), ('root', 'Cells', 'Singlets', 'Live', 'T cells'))
    g.add_gate(rect('CD8', 'BV605', 'BV421'), ('root', 'Cells', 'Singlets', 'Live', 'T cells'))
    return g


def test_auto_anchor_is_the_deepest_viability_gate_above_the_clustered_cells():
    g = _gating()
    assert cid.resolve_anchor(g, ['T cells'], '', VIABILITY) == ('Live', '')
    assert cid.resolve_anchor(g, ['CD4', 'CD8'], '', VIABILITY) == ('Live', '')


def test_auto_anchor_falls_back_to_singlets_then_cells_then_root():
    g = _gating()
    assert cid.resolve_anchor(g, ['T cells'], '', set()) == ('Singlets', '')
    assert cid.resolve_anchor(g, ['Singlets'], '', VIABILITY) == ('Singlets', '')
    assert cid.resolve_anchor(g, ['Cells'], '', VIABILITY) == ('Cells', '')
    assert cid.resolve_anchor(g, [], '', VIABILITY) == ('root', '')
    assert cid.resolve_anchor(g, ['No such gate'], '', VIABILITY) == ('root', '')


def test_a_chosen_anchor_must_contain_the_clustered_cells():
    g = _gating()
    assert cid.resolve_anchor(g, ['T cells'], 'Cells', VIABILITY) == ('Cells', '')
    assert cid.resolve_anchor(g, ['T cells'], 'T cells', VIABILITY) == ('T cells', '')
    used, note = cid.resolve_anchor(g, ['T cells'], 'CD4', VIABILITY)
    assert used == 'T cells' and '"CD4"' in note
    used, note = cid.resolve_anchor(g, ['CD4', 'CD8'], 'CD4', VIABILITY)
    assert used == 'T cells' and note


def _two_cd3_positive_clusters(rng):
    """Both clusters CD3+ (BUV395 ~ 3); cluster 0 CD4+ (BV421 ~ 3), cluster
    1 CD4- (~ 0). The reference adds CD3- CD4- cells (label -1)."""
    n = 400
    pooled = {
        'BUV395': {0: rng.normal(3.0, 0.2, n), 1: rng.normal(3.0, 0.2, n)},
        'BV421': {0: rng.normal(3.0, 0.2, n), 1: rng.normal(0.0, 0.2, n)},
    }
    pooled = {ch: {cl: v.astype(np.float32) for cl, v in by_cl.items()} for ch, by_cl in pooled.items()}
    values = np.vstack([
        np.column_stack([pooled['BUV395'][0], pooled['BV421'][0]]),
        np.column_stack([pooled['BUV395'][1], pooled['BV421'][1]]),
        np.column_stack([rng.normal(0.0, 0.2, 2 * n), rng.normal(0.0, 0.2, 2 * n)]),
    ]).astype(np.float32)
    labels = np.concatenate([np.zeros(n, int), np.ones(n, int), np.full(2 * n, -1)])
    reference = {'values': values, 'names': ['BUV395', 'BV421'], 'labels': labels}
    return pooled, reference


def test_mem_against_the_anchor_shows_a_marker_every_cluster_carries():
    pooled, reference = _two_cd3_positive_clusters(np.random.default_rng(0))
    among_clusters = cid.calculate_mem_scores(pooled)
    against_anchor = cid.calculate_mem_scores(pooled, reference=reference)
    assert abs(among_clusters.loc[0, 'BUV395']) <= 1 and abs(among_clusters.loc[1, 'BUV395']) <= 1
    assert against_anchor.loc[0, 'BUV395'] >= 5 and against_anchor.loc[1, 'BUV395'] >= 5
    assert against_anchor.loc[0, 'BV421'] > 0 and among_clusters.loc[0, 'BV421'] > 0


def test_mem_leaves_a_cluster_out_of_its_own_reference():
    """A reference holding only the clustered cells reduces to MEM against
    the other clusters."""
    pooled, reference = _two_cd3_positive_clusters(np.random.default_rng(1))
    clustered = reference['labels'] >= 0
    only_clusters = {'values': reference['values'][clustered], 'names': reference['names'],
                     'labels': reference['labels'][clustered]}
    scores = cid.calculate_mem_scores(pooled, reference=only_clusters)
    assert scores.equals(cid.calculate_mem_scores(pooled))


def test_mem_without_the_channel_in_the_reference_is_nan():
    pooled, reference = _two_cd3_positive_clusters(np.random.default_rng(2))
    partial = {'values': reference['values'][:, :1], 'names': ['BUV395'], 'labels': reference['labels']}
    scores = cid.calculate_mem_scores(pooled, reference=partial)
    assert scores['BV421'].isna().all() and scores['BUV395'].notna().all()


def test_otsu_fallback_is_cut_on_the_reference_population():
    pooled, reference = _two_cd3_positive_clusters(np.random.default_rng(3))
    on_clusters = cid.calculate_channel_thresholds(None, None, pooled, ['BUV395'], unstained_thresholds={})
    on_reference = cid.calculate_channel_thresholds(None, None, pooled, ['BUV395'], unstained_thresholds={},
                                                    reference=reference)
    assert 2.4 < on_clusters['BUV395'] < 3.6
    assert 0.6 < on_reference['BUV395'] < 2.4
    kept = cid.calculate_channel_thresholds(None, None, pooled, ['BUV395'], unstained_thresholds={'BUV395': 1.5},
                                            reference=reference)
    assert kept == {'BUV395': 1.5}


def test_scoring_channels_add_every_stained_channel_except_viability():
    pnn = ['FSC-A', 'SSC-A', 'BUV395', 'BV421', 'BV605', 'LD', 'AF', 'Time']
    antigens = {'BUV395': 'CD3', 'BV421': 'CD4', 'BV605': 'CD8', 'LD': 'Viability'}
    controller = SimpleNamespace(experiment=SimpleNamespace(
        settings={'unmixed': {'event_channels_pnn': pnn}},
        process={'spectral_model': [{'label': k, 'antigen': v} for k, v in antigens.items()]}))
    channels, extra = cid.scoring_channels(controller, ['BV605', 'BV421'])
    assert channels == ['BV605', 'BV421', 'BUV395'] and extra == ['BUV395']


def test_shared_markers_are_those_every_cluster_shares_against_the_anchor():
    pooled, reference = _two_cd3_positive_clusters(np.random.default_rng(4))
    against_anchor = cid.calculate_mem_scores(pooled, reference=reference)
    assert cid.shared_markers(against_anchor) == {'BUV395': 1}
    assert cid.shared_markers(cid.calculate_mem_scores(pooled)) == {}


def test_stored_event_picks_cap_each_cluster_and_return_file_rows():
    labels = {'a.fcs': np.array([0, 0, 1, 1, -1, 0]), 'b.fcs': np.array([1, 0, 0])}
    cl_run = {
        'training_sample_ids': ['a.fcs', 'b.fcs'],
        'labels': labels,
        'event_rows': {'a.fcs': np.array([10, 11, 12, 13, 14, 15]), 'b.fcs': np.array([5, 6, 7])},
    }
    stored = {'a.fcs': (None, [], np.array([0, 1, 2, 4, 5])), 'b.fcs': (None, [], np.array([0, 1, 2]))}
    picks = cid._stored_event_picks(cl_run, stored, max_events_per_cluster=2, seed=0)
    all_labels = np.concatenate([picks[r][1] for r in picks])
    assert (all_labels == 0).sum() == 2 and (all_labels == 1).sum() == 2 and (all_labels >= 0).all()
    for rel, (rows, labs) in picks.items():
        assert np.all(np.diff(rows) > 0)
        lookup = dict(zip(cl_run['event_rows'][rel].tolist(), labels[rel].tolist()))
        assert [lookup[r] for r in rows.tolist()] == labs.tolist()
    assert cid._stored_event_picks(cl_run, {'a.fcs': stored['a.fcs']}, 2, 0) is None

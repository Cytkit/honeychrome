"""
test_ag_core.py
---------------
Pure-numpy tests for bundled_plugins/ag_core.py (automated gating engine):
threshold algorithms, open-ended gate masks, nested hierarchies, replicate
gates, fixed/recalculate application, population statistics, the
.agmodel format and transform remapping.

Usage:
    pytest tests/test_ag_core.py -m numpy_only
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

import ag_core as ag  # noqa: E402


def _bimodal(n=4000, seed=0, lo=0.2, hi=0.7, sd=0.05, frac_hi=0.4):
    rng = np.random.default_rng(seed)
    k = int(n * frac_hi)
    return np.concatenate([rng.normal(lo, sd, n - k), rng.normal(hi, sd, k)])


def _unimodal(n=4000, seed=1, mu=0.3, sd=0.05):
    return np.random.default_rng(seed).normal(mu, sd, n)


def _gate(name, gtype, x, y=None, parent=None, parent_pop='root', algorithm=None,
          number=1, **extra):
    g = {
        'gate_number': number, 'gate_name': name, 'gate_type': gtype,
        'gate_marker_x': x, 'gate_marker_y': y,
        'parent_gate': parent, 'parent_popul': parent_pop if parent else 'root',
        'populations': ag.default_populations_for_type(gtype),
        'algorithm': algorithm or ag.default_algorithm_for_type(gtype),
        'gate_param': {}, 'stats_parent': {}, 'origin_gate': None,
    }
    g.update(extra)
    return g


def _three_channel_data(n=6000, seed=2):
    """A: bimodal, B: bimodal (independent), C: bimodal only within A+."""
    rng = np.random.default_rng(seed)
    a = _bimodal(n, seed=seed)
    b = _bimodal(n, seed=seed + 1, frac_hi=0.5)
    c = np.where(a > 0.45, _bimodal(n, seed=seed + 2, frac_hi=0.3), rng.normal(0.2, 0.05, n))
    return np.column_stack([a, b, c]), {'A': 0, 'B': 1, 'C': 2}


# ---------------------------------------------------------------------------
# Threshold algorithms
# ---------------------------------------------------------------------------

@pytest.mark.numpy_only
def test_valley_methods_split_bimodal_data():
    x = _bimodal()
    for algo in ('kde_min', 'mixture', 'otsu', 'bimodal'):
        t = ag.compute_threshold(x, algo)
        assert 0.3 < t < 0.6, (algo, t)


@pytest.mark.numpy_only
def test_tail_sits_above_negative_population():
    x = _unimodal(mu=0.3, sd=0.05)
    t = ag.compute_threshold(x, 'tail', {'tail_fraction': 0.01})
    assert 0.3 + 2.0 * 0.05 < t < 0.3 + 3.0 * 0.05
    assert np.mean(x >= t) < 0.03
    # A positive population does not move the tail threshold much.
    t_mixed = ag.threshold_tail(_bimodal(lo=0.3, hi=0.8, frac_hi=0.3), 0.01)
    assert abs(t_mixed - t) < 0.05


@pytest.mark.numpy_only
def test_kde_min_falls_back_on_single_peak():
    x = _unimodal()
    assert ag.threshold_kde_min(x) == pytest.approx(ag.threshold_tail(x), abs=1e-9)


@pytest.mark.numpy_only
def test_otsu_on_two_point_masses():
    x = np.concatenate([np.full(500, 1.0), np.full(500, 3.0)])
    assert 1.0 <= ag.threshold_otsu(x) < 3.0


@pytest.mark.numpy_only
def test_otsu_threshold_puts_whole_lower_class_below():
    # The split bin belongs to the lower class, so the threshold is that
    # bin's upper edge; its centre would leave part of the bin above it.
    rng = np.random.default_rng(3)
    lower = rng.uniform(0.0, 1.0, 3000)
    upper = rng.uniform(9.0, 10.0, 3000)
    t = ag.threshold_otsu(np.concatenate([lower, upper]))
    assert np.all(lower < t)
    assert np.all(upper >= t)


@pytest.mark.numpy_only
def test_bimodality_check():
    assert ag.bimodality_delta_bic(_bimodal()) > 10
    assert ag.bimodality_delta_bic(_unimodal()) < 10
    g = _gate('A', '1dsep', 'A')
    data = _bimodal().reshape(-1, 1)
    assert ag.recommend_algorithm(g, data, {'A': 0}) == 'mixture'
    assert ag.recommend_algorithm(g, _unimodal().reshape(-1, 1), {'A': 0}) == 'tail'
    assert ag.recommend_algorithm(_gate('F', 'free', 'A', 'B'), data, {'A': 0}) == 'all'


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

@pytest.mark.numpy_only
def test_1d_masks_are_open_ended_and_classify_every_event():
    # Events far outside the display range and exactly at y=0 must still
    # be classified.
    g = _gate('A', '1dsep', 'A')
    calc = ag.GateCalculator({'A': (0.0, 1.0)})
    boundary = calc.threshold_boundaries(g, 0.4, None)
    data = np.array([[-5.0], [0.0], [0.39], [0.4], [0.9], [7.0]])
    masks = ag.population_masks(g, boundary, data, {'A': 0})
    np.testing.assert_array_equal(masks['neg'], [True, True, True, False, False, False])
    np.testing.assert_array_equal(masks['pos'], ~masks['neg'])


@pytest.mark.numpy_only
def test_2d_quadrants_are_open_ended():
    g = _gate('Q', '2dsep', 'A', 'B')
    boundary = ag.GateCalculator().threshold_boundaries(g, 0.5, 0.5)
    data = np.array([[-9, -9], [9, -9], [-9, 9], [9, 9], [0.5, 0.5]], dtype=float)
    m = ag.population_masks(g, boundary, data, {'A': 0, 'B': 1})
    assert list(np.flatnonzero(m['DN'])) == [0]
    assert list(np.flatnonzero(m['X+'])) == [1]
    assert list(np.flatnonzero(m['Y+'])) == [2]
    assert list(np.flatnonzero(m['DP'])) == [3, 4]
    assert sum(int(v.sum()) for v in m.values()) == len(data)


@pytest.mark.numpy_only
def test_regions_inferred_for_legacy_population_order():
    g = _gate('A', '1dsep', 'A')
    g['populations'] = {'lo': {}, 'hi': {}}
    boundary = {'lo': {'threshold_x': 0.5}, 'hi': {'threshold_x': 0.5}}
    m = ag.population_masks(g, boundary, np.array([[0.1], [0.9]]), {'A': 0})
    assert m['lo'].tolist() == [True, False] and m['hi'].tolist() == [False, True]


# ---------------------------------------------------------------------------
# Hierarchies
# ---------------------------------------------------------------------------

@pytest.mark.numpy_only
def test_three_level_nesting_restricts_children_to_parent():
    data, idx = _three_channel_data()
    gates = [
        _gate('gA', '1dsep', 'A', number=1, algorithm='kde_min'),
        _gate('gB', '1dsep', 'B', parent='gA', parent_pop='pos', number=2, algorithm='kde_min'),
        _gate('gC', '1dsep', 'C', parent='gB', parent_pop='pos', number=3, algorithm='kde_min'),
    ]
    res = ag.train_boundaries(gates, data, idx)
    assert set(res.boundaries) == {'gA', 'gB', 'gC'}
    a_pos = res.masks['gA']['pos']
    b_pos = res.masks['gB']['pos']
    c_pos = res.masks['gC']['pos']
    assert not np.any(b_pos & ~a_pos)
    assert not np.any(c_pos & ~b_pos)
    assert res.masks['gB']['neg'].sum() + b_pos.sum() == a_pos.sum()
    # Counts are taken on untrimmed parent data and never exceed the parent.
    for name in ('gA', 'gB', 'gC'):
        n_parent = int(res.parent_masks[name].sum())
        total = sum(e['n_events'] for e in res.boundaries[name].values())
        assert total == n_parent
        assert all(0.0 <= e['fraction'] <= 1.0 for e in res.boundaries[name].values())


@pytest.mark.numpy_only
def test_order_follows_parents_not_numbers():
    gates = [
        _gate('child', '1dsep', 'B', parent='parent', parent_pop='pos', number=1),
        _gate('parent', '1dsep', 'A', number=2),
    ]
    assert [g['gate_name'] for g in ag.ordered_gate_defs(gates)] == ['parent', 'child']
    gates[1]['parent_gate'] = 'child'
    gates[1]['parent_popul'] = 'pos'
    with pytest.raises(ValueError, match='cycle'):
        ag.ordered_gate_defs(gates)


@pytest.mark.numpy_only
def test_missing_parent_population_skips_child():
    data, idx = _three_channel_data()
    gates = [
        _gate('gA', '1dsep', 'A', number=1),
        _gate('gB', '1dsep', 'B', parent='gA', parent_pop='nonexistent', number=2),
    ]
    res = ag.train_boundaries(gates, data, idx)
    assert 'gB' not in res.masks
    assert res.status['gB']['flag'] == 'skipped'


@pytest.mark.numpy_only
def test_replicate_uses_origin_from_same_run():
    data, idx = _three_channel_data()
    gates = [
        _gate('gA', '1dsep', 'A', number=1, algorithm='kde_min'),
        _gate('gB', '1dsep', 'B', parent='gA', parent_pop='neg', number=2, algorithm='kde_min'),
        {'gate_number': 3, 'gate_name': 'gB_rep', 'gate_type': 'replicate',
         'gate_marker_x': '', 'gate_marker_y': None, 'parent_gate': 'gA',
         'parent_popul': 'pos', 'populations': {}, 'algorithm': 'replicate',
         'gate_param': {}, 'stats_parent': {}, 'origin_gate': 'gB'},
    ]
    res = ag.train_boundaries(gates, data, idx)
    assert 'gB_rep' in res.boundaries
    t_origin = res.boundaries['gB']['pos']['threshold_x']
    assert res.boundaries['gB_rep']['pos']['threshold_x'] == pytest.approx(t_origin)
    # Applied to its own parent population, not the origin's.
    assert not np.any(res.masks['gB_rep']['pos'] & ~res.masks['gA']['pos'])
    assert res.masks['gB_rep']['pos'].sum() == \
        np.sum(res.masks['gA']['pos'] & (data[:, 1] >= t_origin))


@pytest.mark.numpy_only
def test_polygon_gates_count_events_inside():
    rng = np.random.default_rng(5)
    x = rng.normal(0.5, 0.05, 3000)
    data = np.column_stack([x, x + rng.normal(0, 0.01, 3000)])
    data[:50, 1] += 0.4        # doublets off the diagonal
    gates = [_gate('S', 'singlets', 'A', 'B'), _gate('F', 'free', 'A', 'B', number=2,
                                                        parent='S', parent_pop='in')]
    res = ag.train_boundaries(gates, data, {'A': 0, 'B': 1})
    singlets = res.masks['S']['in']
    assert singlets[:50].sum() == 0
    assert singlets[50:].mean() > 0.95
    assert res.boundaries['S']['in']['n_events'] == int(singlets.sum())
    assert 'F' in res.masks


# ---------------------------------------------------------------------------
# Application modes
# ---------------------------------------------------------------------------

@pytest.mark.numpy_only
def test_fixed_applies_reference_and_recalculate_flags_drift():
    data, idx = _three_channel_data()
    gates = [_gate('gA', '1dsep', 'A', algorithm='kde_min')]
    ref = ag.train_boundaries(gates, data, idx).boundaries

    shifted = data.copy()
    shifted[:, 0] += 0.2
    fixed = ag.run_gating(gates, shifted, idx, reference=ref, default_mode='fixed')
    assert fixed.boundaries['gA']['pos']['threshold_x'] == ref['gA']['pos']['threshold_x']
    assert fixed.status['gA']['source'] == 'fixed'

    recalc = ag.run_gating(gates, shifted, idx, reference=ref,
                           modes={'gA': 'recalculate'}, drift_limit=0.05)
    assert recalc.status['gA']['flag'] == 'drift'
    assert recalc.status['gA']['drift'] == pytest.approx(0.2, abs=0.05)
    assert recalc.boundaries['gA']['pos']['threshold_x'] > ref['gA']['pos']['threshold_x'] + 0.1

    held = ag.run_gating(gates, shifted, idx, reference=ref, modes={'gA': 'recalculate'},
                         drift_limit=0.05, drift_action='reference')
    assert held.boundaries['gA']['pos']['threshold_x'] == ref['gA']['pos']['threshold_x']
    assert held.status['gA']['source'] == 'reference'


@pytest.mark.numpy_only
def test_recalculate_falls_back_to_reference_on_too_few_events():
    data, idx = _three_channel_data()
    gates = [_gate('gA', '1dsep', 'A')]
    ref = ag.train_boundaries(gates, data, idx).boundaries
    res = ag.run_gating(gates, data[:5], idx, reference=ref, default_mode='recalculate')
    assert res.status['gA']['flag'] == 'failed'
    assert res.status['gA']['source'] == 'reference'
    assert 'gA' in res.masks


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@pytest.mark.numpy_only
def test_population_table_uses_parent_and_stats_parent():
    data, idx = _three_channel_data()
    gates = [
        _gate('gA', '1dsep', 'A', number=1, algorithm='kde_min'),
        _gate('gB', '1dsep', 'B', parent='gA', parent_pop='pos', number=2, algorithm='kde_min',
              stats_parent={'pos': 'root'}),
    ]
    res = ag.train_boundaries(gates, data, idx)
    rows = {r['key']: r for r in ag.population_table(gates, res, n_total=len(data))}
    b_pos = rows['gB/pos']
    assert b_pos['n_parent'] == int(res.masks['gA']['pos'].sum())
    assert b_pos['parent_key'] == 'gA/pos'
    assert b_pos['fraction_of_parent'] == pytest.approx(b_pos['n_events'] / b_pos['n_parent'])
    assert b_pos['n_stats_parent'] == len(data)
    assert 'stats_parent' not in rows['gB/neg']

    med = ag.population_medians(gates, res, data, idx, ['C'])
    assert med[('gA/pos', 'C')] == pytest.approx(np.median(data[res.masks['gA']['pos'], 2]))
    assert ag.ancestor_channels(gates, 'gB') == ['B', 'A']


# ---------------------------------------------------------------------------
# Model format
# ---------------------------------------------------------------------------

@pytest.mark.numpy_only
def test_v1_model_upgrades_to_v2(tmp_path):
    v1 = {
        'format_version': 1,
        'metadata': {'experiment_name': 'x'},
        'gate_definitions': [_gate('gA', '1dsep', 'A'), _gate('gB', '1dsep', 'B', parent='gA',
                                                             parent_pop='pos', number=2)],
        'trained_boundaries': {'gA': {'neg': {'threshold_x': 0.4, 'boundary': [[0, 0]]},
                                      'pos': {'threshold_x': 0.4, 'boundary': [[0, 0]]}}},
        'channel_map': {'A': 'CD3'},
        'training_summary': {'training_sample_names': ['a.fcs']},
    }
    path = tmp_path / 'm.agmodel'
    path.write_text(json.dumps(v1))
    model, warnings = ag.read_model(path)
    assert model['format_version'] == 2
    assert model['transforms'] == {}
    assert model['application']['mode_per_gate'] == {'gA': 'fixed', 'gB': 'fixed'}
    assert any('transform' in w for w in warnings)
    assert any('gB' in w for w in warnings)


@pytest.mark.numpy_only
def test_v2_round_trip_with_numpy_values(tmp_path):
    data, idx = _three_channel_data()
    gates = [_gate('gA', '1dsep', 'A')]
    res = ag.train_boundaries(gates, data, idx)
    model = ag.new_model(
        gates, res.boundaries, channel_map={'A': 'CD3'},
        transforms={'A': {'id': 1, 'scale_t': 262144.0}},
        training_summary={'n_events_per_sample': {'s1': np.int64(5)}},
        metadata={'experiment_name': 'e'},
        application={'mode_per_gate': {'gA': 'recalculate'}},
    )
    path = tmp_path / 'm.agmodel'
    ag.write_model(path, model)
    loaded, warnings = ag.read_model(path)
    assert warnings == []
    assert loaded['application']['mode_per_gate']['gA'] == 'recalculate'
    assert loaded['trained_boundaries']['gA']['pos']['threshold_x'] == \
        pytest.approx(res.boundaries['gA']['pos']['threshold_x'])

    with pytest.raises(ValueError):
        ag.normalize_model({'format_version': 9, 'gate_definitions': []})


# ---------------------------------------------------------------------------
# Transform remapping
# ---------------------------------------------------------------------------

class _ArcsinhTransform:
    """Stand-in for a logicle transform: asinh(x / w) / top."""

    def __init__(self, params):
        self.w = float(params['logicle_w'])
        self.top = float(params['scale_t'])

    def apply(self, v):
        return np.arcsinh(np.asarray(v, dtype=float) / self.w) / np.arcsinh(self.top / self.w)

    def inverse(self, v):
        return np.sinh(np.asarray(v, dtype=float) * np.arcsinh(self.top / self.w)) * self.w


@pytest.mark.numpy_only
def test_threshold_remap_round_trip():
    src = {'A': {'id': 1, 'scale_t': 262144.0, 'logicle_w': 0.5},
           'B': {'id': 1, 'scale_t': 262144.0, 'logicle_w': 0.5}}
    dst = {'A': {'id': 1, 'scale_t': 262144.0, 'logicle_w': 2.0},
           'B': {'id': 1, 'scale_t': 262144.0, 'logicle_w': 0.5}}
    gates = [_gate('Q', '2dsep', 'A', 'B'),
             _gate('F', 'free', 'A', 'B', number=2)]
    boundaries = {
        'Q': ag.GateCalculator({'A': (0, 1), 'B': (0, 1)}).threshold_boundaries(gates[0], 0.6, 0.3),
        'F': {'in': {'boundary': [[0.5, 0.2], [0.7, 0.2], [0.7, 0.4], [0.5, 0.2]]}},
    }
    moved, report = ag.remap_boundaries(gates, boundaries, src, dst,
                                        transform_factory=_ArcsinhTransform)
    assert report == {'A': 'remapped', 'B': 'same'}
    tx = moved['Q']['DP']['threshold_x']
    raw = _ArcsinhTransform(src['A']).inverse([0.6])[0]
    assert tx == pytest.approx(_ArcsinhTransform(dst['A']).apply([raw])[0])
    assert moved['Q']['DP']['threshold_y'] == pytest.approx(0.3)
    assert len(moved['F']['in']['boundary']) > len(boundaries['F']['in']['boundary'])

    back, _ = ag.remap_boundaries(gates, moved, dst, src, transform_factory=_ArcsinhTransform)
    assert back['Q']['DP']['threshold_x'] == pytest.approx(0.6)
    pts = np.asarray(back['F']['in']['boundary'])
    np.testing.assert_allclose(pts[0], [0.5, 0.2], atol=1e-9)

    unknown, rep = ag.remap_boundaries(gates, boundaries, {}, dst,
                                       transform_factory=_ArcsinhTransform)
    assert rep['A'] == 'unknown'
    assert unknown['Q']['DP']['threshold_x'] == pytest.approx(0.6)


@pytest.mark.numpy_only
def test_remap_model_converts_and_reports():
    gates = [_gate('Q', '2dsep', 'A', 'B'), _gate('H', '1dsep', 'C', number=2)]
    src = {'A': {'id': 1, 'scale_t': 262144.0, 'logicle_w': 0.5},
           'B': {'id': 1, 'scale_t': 262144.0, 'logicle_w': 0.5}}
    dst = {'A': {'id': 1, 'scale_t': 262144.0, 'logicle_w': 2.0},
           'B': {'id': 1, 'scale_t': 262144.0, 'logicle_w': 0.5},
           'C': {'id': 1, 'scale_t': 262144.0, 'logicle_w': 0.5}}
    calc = ag.GateCalculator({'A': (0, 1), 'B': (0, 1), 'C': (0, 1)})
    model = {
        'gate_definitions': gates,
        'trained_boundaries': {'Q': calc.threshold_boundaries(gates[0], 0.6, 0.3),
                               'H': calc.threshold_boundaries(gates[1], 0.4, None)},
        'transforms': src,
    }
    out, messages = ag.remap_model(model, dst, transform_factory=_ArcsinhTransform)

    raw = _ArcsinhTransform(src['A']).inverse([0.6])[0]
    expected = _ArcsinhTransform(dst['A']).apply([raw])[0]
    assert out['trained_boundaries']['Q']['DP']['threshold_x'] == pytest.approx(expected)
    assert out['trained_boundaries']['H']['pos']['threshold_x'] == pytest.approx(0.4)
    assert out['transforms']['A'] == dst['A']
    assert model['transforms']['A'] == src['A']
    assert any('converted' in m and 'A' in m for m in messages)
    assert any('no stored transform' in m and 'C' in m for m in messages)

    unchanged, none = ag.remap_model(dict(model, transforms={}), dst,
                                     transform_factory=_ArcsinhTransform)
    assert none == []
    assert unchanged['trained_boundaries']['Q']['DP']['threshold_x'] == pytest.approx(0.6)


@pytest.mark.numpy_only
def test_write_model_is_atomic(tmp_path):
    path = tmp_path / 'm.agmodel'
    ag.write_model(path, {'format_version': 2, 'x': np.float32(1.5)})
    assert json.loads(path.read_text())['x'] == 1.5
    assert not (tmp_path / 'm.agmodel.tmp').exists()


@pytest.mark.numpy_only
def test_rename_channels_for_alignment():
    gates = [_gate('Q', '2dsep', 'BV421-A', 'PE-A')]
    out = ag.rename_channels(gates, {'BV421-A': 'V1-A'})
    assert out[0]['gate_marker_x'] == 'V1-A' and out[0]['gate_marker_y'] == 'PE-A'
    assert gates[0]['gate_marker_x'] == 'BV421-A'

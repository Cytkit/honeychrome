"""
test_ag_report.py
-----------------
Tests for bundled_plugins/ag_report.py (Automated Gating report items):
pooled display events, fractions of a drawn boundary, population label
placement, gate figures and the item list.

ag_report builds on drc_report, which imports PySide6, so these tests are
skipped where PySide6 or matplotlib is unavailable.

Usage:
    pytest tests/test_ag_report.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('PySide6')
pytest.importorskip('matplotlib')

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

import ag_core  # noqa: E402
import ag_report  # noqa: E402


class _Scale:
    """Display transform stand-in: x / t on [0, 1]."""

    def __init__(self, t=1.0):
        self.xform = self
        self.t = float(t)
        self.limits = [0, 1]

    def apply(self, v):
        return np.asarray(v, dtype=float) / self.t

    def ticks(self):
        return [[], [(0.0, '0'), (1.0, str(self.t))]]


def _threshold_gate():
    gate = {'gate_name': 'T', 'gate_type': '1dsep', 'gate_marker_x': 'B',
            'populations': {'neg': {'label': 'T-'}, 'pos': {'label': 'T+'}}}
    boundary = {
        'neg': {'boundary': [[0.0, 0.0], [0.4, 0.0], [0.4, 1.0], [0.0, 1.0]],
                'threshold_x': 0.4, 'region': 'below'},
        'pos': {'boundary': [[0.4, 0.0], [1.0, 0.0], [1.0, 1.0], [0.4, 1.0]],
                'threshold_x': 0.4, 'region': 'above'},
    }
    return gate, boundary


def _quadrant_gate(tx=0.5, ty=0.5):
    gate = {'gate_name': 'Q', 'gate_type': '2dsep', 'gate_marker_x': 'A', 'gate_marker_y': 'B',
            'populations': {'DN': {}, 'XP': {}, 'YP': {}, 'DP': {}}}
    boundary = {}
    for pop, region in zip(gate['populations'], ag_core.QUADRANT_REGIONS):
        x0, x1 = (0.0, tx) if region.startswith('x-') else (tx, 1.0)
        y0, y1 = (0.0, ty) if region.endswith('y-') else (ty, 1.0)
        boundary[pop] = {'boundary': [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                         'threshold_x': tx, 'threshold_y': ty, 'region': region}
    return gate, boundary


def _display(n=400, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for key in ('s1', 's2'):
        events = rng.random((n, 2)) * [1.0, 200.0]
        out[key] = {'channels': ['A', 'B'], 'events': events,
                    'parent_masks': {'T': events[:, 0] > 0.5}}
    return out


def test_pooled_display_concatenates_samples_and_parent_masks():
    display = _display()
    events, parent, channels = ag_report.pooled_display(display, ['s1', 's2', 'missing'], 'T')
    assert channels == ['A', 'B']
    assert events.shape == (800, 2) and parent.shape == (800,)
    np.testing.assert_array_equal(parent, events[:, 0] > 0.5)
    none_events, none_parent, none_channels = ag_report.pooled_display(display, ['missing'], 'T')
    assert none_events is None and none_parent is None and none_channels == []
    _e, other_parent, _c = ag_report.pooled_display(display, ['s1'], 'no such gate')
    assert not other_parent.any()


def test_boundary_fractions_apply_the_drawn_boundary_within_the_parent():
    events, parent, channels = ag_report.pooled_display(_display(), ['s1', 's2'], 'T')
    gate, boundary = _threshold_gate()
    transforms = {'B': _Scale(200.0)}
    got = ag_report.boundary_fractions(gate, boundary, events, channels, parent, transforms)
    expected = float((events[parent, 1] / 200.0 >= 0.4).mean())
    assert got['pos'] == pytest.approx(expected)
    assert got['neg'] + got['pos'] == pytest.approx(1.0)
    assert ag_report.boundary_fractions(gate, boundary, None, channels, parent, transforms) == {}
    assert ag_report.boundary_fractions(gate, {}, events, channels, parent, transforms) == {}
    assert ag_report.boundary_fractions(gate, boundary, events, ['A'], parent, transforms) == {}


def test_quadrant_labels_sit_in_their_outer_corners():
    gate, boundary = _quadrant_gate()
    placed = {region: ag_report.label_position('2dsep', region, boundary[pop], (0, 1), (0, 1))
              for pop, region in zip(boundary, ag_core.QUADRANT_REGIONS)}
    x, y, ha, va = placed['x+y+']
    assert x > 0.9 and y > 0.9 and (ha, va) == ('right', 'top')
    x, y, ha, va = placed['x-y-']
    assert x < 0.1 and y < 0.1 and (ha, va) == ('left', 'bottom')
    assert placed['x+y-'][2:] == ('right', 'bottom')
    assert placed['x-y+'][2:] == ('left', 'top')


def test_threshold_and_polygon_labels_are_centred_in_their_region():
    _gate, boundary = _threshold_gate()
    x, y, ha, va = ag_report.label_position('1dsep', 'above', boundary['pos'], (0, 1), (0, 1))
    assert 0.4 < x < 1.0 and y == pytest.approx(0.5) and (ha, va) == ('center', 'center')
    entry = {'boundary': [[0.2, 0.2], [0.4, 0.2], [0.4, 0.6], [0.2, 0.6]]}
    x, y, _ha, _va = ag_report.label_position('free', 'inside', entry, (0, 1), (0, 1))
    assert (x, y) == pytest.approx((0.3, 0.4))


def test_gate_figure_labels_fractions_inside_the_axes():
    rng = np.random.default_rng(3)
    events = rng.random((1000, 2))
    gate, boundary = _quadrant_gate(0.3, 0.6)
    transforms = {'A': _Scale(), 'B': _Scale()}
    fractions = ag_report.boundary_fractions(gate, boundary, events, ['A', 'B'], None, transforms)
    fig = ag_report.make_gate_figure(gate, boundary, events, ['A', 'B'], None, transforms,
                                     {'A': 'CD3'}, 'Q', fractions=fractions)
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert len(texts) == 4
    assert f"DP\n{fractions['DP'] * 100:.1f}%" in texts
    for t in ax.texts:
        tx, ty = t.get_position()
        assert 0 <= tx <= 1 and 0 <= ty <= 1
    assert ax.get_xlabel() == 'CD3'


def test_report_items_list_gates_only_with_display_events():
    gate, boundary = _threshold_gate()
    gate = dict(gate, gate_number=1, parent_gate='root', parent_pop='root')
    summaries = {'s1': {'populations': []}, 's2': {'populations': []}}
    common = dict(gate_defs=[gate], reference_boundaries={'T': boundary}, modes={},
                  summaries=summaries, samples=['s1', 's2'], load_info={}, groups={},
                  flags={}, antigens={}, transforms={'B': _Scale(200.0)}, axis_labels={},
                  stats=None, pval_threshold=0.05, fc_threshold=0.0, marker_threshold=0.0,
                  fdr_scope='global', build_figures=lambda *a: {})
    with_display = ag_report.build_report_items(display=_display(), **common)
    keys = [item.key for item in with_display]
    assert keys == ['samples', 'frequencies', 'populations', 'markers', 'gate:T']
    fig = with_display[-1].get_figure()
    assert any('pooled display events' in t.get_text() for t in [fig.axes[0].title])
    without = ag_report.build_report_items(display={}, **common)
    assert 'gate:T' not in [item.key for item in without]

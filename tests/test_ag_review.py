"""
test_ag_review.py
-----------------
Tests for bundled_plugins/ag_review.py: which populations can be reviewed,
how daughter gates are grouped into plots, the comparison with the fractions
recorded at import, and rewriting an imported boundary after an edit.

Usage:
    pytest tests/test_ag_review.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

import ag_core  # noqa: E402
import ag_review  # noqa: E402


def _poly_gate(name, number, x, y, parent=None, parent_pop='root', source=None):
    box = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8], [0.2, 0.2]]
    g = {'gate_name': name, 'gate_number': number, 'gate_type': 'free',
         'gate_marker_x': x, 'gate_marker_y': y, 'parent_gate': parent,
         'parent_popul': parent_pop, 'algorithm': 'imported', 'gate_param': {},
         'stats_parent': {}, 'origin_gate': None,
         'populations': {name: {'label': name, 'label_pos': 1, 'region': 'inside'}},
         'template': {name: {'boundary': box, 'threshold_x': None, 'threshold_y': None,
                             'region': 'inside', 'label': name}}}
    g['template_original'] = {k: dict(v) for k, v in g['template'].items()}
    if source:
        g['source'] = source
    return g


def _thr_gate(name, number, x, y, parent, parent_pop, regions=('x+y-', 'x-y+')):
    pops = {r: {'label': r, 'label_pos': 1, 'region': r} for r in regions}
    return {'gate_name': name, 'gate_number': number, 'gate_type': '2dsep',
            'gate_marker_x': x, 'gate_marker_y': y, 'parent_gate': parent,
            'parent_popul': parent_pop, 'algorithm': 'imported', 'gate_param': {},
            'stats_parent': {}, 'origin_gate': None, 'populations': pops}


@pytest.fixture
def gates():
    src = {'origin': 'FlowJo', 'sample': 'C4.fcs', 'parent_count': 1000,
           'counts': {'Cells': 800}, 'populations': {'Cells': 'Cells'}}
    cells = _poly_gate('Cells', 1, 'FSC-A', 'SSC-A', source=src)
    t = _poly_gate('T', 2, 'CD3', 'CD4', 'Cells', 'Cells', source={'origin': 'FlowJo'})
    b = _poly_gate('B', 3, 'CD3', 'CD4', 'Cells', 'Cells', source={'origin': 'FlowJo'})
    nk = _poly_gate('NK', 4, 'CD56', 'CD16', 'Cells', 'Cells', source={'origin': 'FlowJo'})
    return [cells, t, b, nk]


def test_population_choices_lists_parents_only(gates):
    labels = [c['label'] for c in ag_review.population_choices(gates)]
    assert labels == ['All events (root)', 'Cells']
    assert [c['n_gates'] for c in ag_review.population_choices(gates)] == [1, 3]


def test_daughter_groups_one_plot_per_parameter_set(gates):
    groups = ag_review.daughter_groups(gates, 'Cells', 'Cells')
    assert [g['channels'] for g in groups] == [('CD3', 'CD4'), ('CD56', 'CD16')]
    assert [x['gate_name'] for x in groups[0]['gates']] == ['T', 'B']
    root = ag_review.daughter_groups(gates, None, 'root')
    assert [x['gate_name'] for x in root[0]['gates']] == ['Cells']


def test_daughters_only_of_the_chosen_population():
    parent = _thr_gate('P', 1, 'A', 'B', None, 'root')
    c1 = _poly_gate('c1', 2, 'X', 'Y', 'P', 'x+y-')
    c2 = _poly_gate('c2', 3, 'X', 'Y', 'P', 'x-y+')
    groups = ag_review.daughter_groups([parent, c1, c2], 'P', 'x+y-')
    assert [g['gate_name'] for g in groups[0]['gates']] == ['c1']
    labels = [c['label'] for c in ag_review.population_choices([parent, c1, c2])]
    assert 'P / x+y-' in labels and 'P / x-y+' in labels


def test_reference_prefers_trained_over_template(gates):
    trained = {'Cells': {'Cells': {'boundary': [[0, 0], [1, 0], [1, 1], [0, 0]]}}}
    ref = ag_review.reference_boundaries(gates, trained)
    assert ref['Cells'] is trained['Cells'] and ref['T'] is gates[1]['template']


def test_imported_fraction_and_discrepancy(gates):
    assert ag_review.imported_fraction(gates[0], 'Cells') == pytest.approx(0.8)
    assert ag_review.imported_fraction(gates[1], 'T') is None
    assert not ag_review.fraction_discrepancy(0.801, 0.8)
    assert ag_review.fraction_discrepancy(0.77, 0.80)          # 3 points
    assert ag_review.fraction_discrepancy(0.0356, 0.0381)       # 6.6% relative
    assert not ag_review.fraction_discrepancy(None, 0.5)
    assert not ag_review.fraction_discrepancy(0.0001, 0.0002)   # tiny, ignored


def test_matches_import_sample(gates):
    assert ag_review.matches_import_sample(gates[0], 'Unmixed/c4.FCS')
    assert not ag_review.matches_import_sample(gates[0], 'Unmixed/C5.fcs')
    assert not ag_review.matches_import_sample(gates[1], 'Unmixed/C4.fcs')


def test_with_thresholds_moves_every_region():
    g = _thr_gate('P', 1, 'A', 'B', None, 'root')
    entries = ag_review.with_thresholds(g, 0.4, 0.6, {'A': (0.0, 1.0), 'B': (0.0, 1.0)})
    assert set(entries) == {'x+y-', 'x-y+'}
    assert entries['x+y-']['threshold_x'] == pytest.approx(0.4)
    assert entries['x+y-']['threshold_y'] == pytest.approx(0.6)
    box = np.asarray(entries['x+y-']['boundary'])
    assert box[:, 0].min() == pytest.approx(0.4) and box[:, 1].max() == pytest.approx(0.6)
    assert entries['x+y-']['label'] == 'x+y-'


def test_with_polygon_closes_and_labels(gates):
    entries = ag_review.with_polygon(gates[0], 'Cells', [(0, 0), (1, 0), (1, 1)])
    assert entries['Cells']['boundary'][0] == entries['Cells']['boundary'][-1]
    assert entries['Cells']['label'] == 'Cells'
    assert gates[0]['template']['Cells']['boundary'][2] == [0.8, 0.8]   # input untouched


def test_with_range_orders_limits(gates):
    entries = ag_review.with_range(gates[0], 'Cells', 0.9, 0.1)
    assert entries['Cells']['range'] == [0.1, 0.9]


def test_reset_template_returns_a_copy(gates):
    gates[0]['template']['Cells']['boundary'] = [[0, 0]]
    original = ag_review.reset_template(gates[0])
    assert original['Cells']['boundary'][2] == [0.8, 0.8]
    original['Cells']['boundary'].clear()
    assert gates[0]['template_original']['Cells']['boundary']


def test_polygon_vertices_drop_closing_point_and_thin():
    ring = [[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 97)]
    ring[-1] = list(ring[0])
    pts = ag_review.polygon_vertices(ring, max_vertices=32)
    assert len(pts) == 32 and pts[0] == ring[0]
    assert len(ag_review.polygon_vertices([[0, 0], [1, 0], [1, 1], [0, 0]])) == 3


def test_edited_template_changes_gating(gates):
    g = _thr_gate('P', 1, 'A', 'B', None, 'root', regions=('x+y+',))
    g['template'] = ag_review.with_thresholds(g, 0.5, 0.5, {'A': (0, 1), 'B': (0, 1)})
    data = np.random.default_rng(0).uniform(0, 1, (4000, 2))
    ci = {'A': 0, 'B': 1}
    ref = lambda gd: {gd['gate_name']: gd['template']}   # noqa: E731
    before = ag_core.run_gating([g], data, ci, reference=ref(g), default_mode=ag_core.MODE_FIXED)
    g['template'] = ag_review.with_thresholds(g, 0.8, 0.8, {'A': (0, 1), 'B': (0, 1)})
    after = ag_core.run_gating([g], data, ci, reference=ref(g), default_mode=ag_core.MODE_FIXED)
    assert after.population_mask('P', 'x+y+').sum() < before.population_mask('P', 'x+y+').sum()


def test_convert_target_by_gate_type():
    thr = _thr_gate('P', 1, 'A', 'B', None, 'root')
    assert ag_review.convert_target(thr) == ('tail', '')
    assert ag_review.convert_target(thr, 'mixture') == ('mixture', '')
    assert ag_review.convert_target(thr, 'not-an-algorithm')[0] == 'tail'
    sing = dict(_thr_gate('S', 2, 'FSC-A', 'FSC-H', None, 'root'), gate_type='singlets')
    assert ag_review.convert_target(sing) == ('singlets', '')
    free = _poly_gate('F', 3, 'X', 'Y')
    algorithm, note = ag_review.convert_target(free)
    assert algorithm == 'cells' and note == ''
    assert ag_review.is_free_shape(free) and not ag_review.is_free_shape(thr)


def test_convert_target_ignores_gates_that_are_not_imported():
    thr = dict(_thr_gate('P', 1, 'A', 'B', None, 'root'), algorithm='mixture')
    assert ag_review.convert_target(thr) == (None, '')


def test_population_tree_nests_populations_and_views(gates):
    nodes = ag_review.population_tree(gates)
    by_key = {(n['gate'], n['pop']): n for n in nodes}
    assert [n['label'] for n in nodes] == ['All events (root)', 'Cells  [anchor]', 'T', 'B', 'NK']
    assert by_key[('', 'root')]['parent'] is None
    assert by_key[('Cells', 'Cells')]['parent'] == ('', 'root')
    assert by_key[('T', 'T')]['parent'] == ('Cells', 'Cells')
    # populations with gates drawn on them show themselves; leaves show their parent
    assert by_key[('Cells', 'Cells')]['view'] == ('Cells', 'Cells')
    assert by_key[('Cells', 'Cells')]['n_gates'] == 3
    assert by_key[('T', 'T')]['n_gates'] == 0
    assert by_key[('T', 'T')]['view'] == ('Cells', 'Cells')


def test_population_tree_parents_precede_children_and_orphans_go_to_root(gates):
    orphan = _poly_gate('Orphan', 5, 'CD8', 'CD4', 'Missing', 'Missing')
    nodes = ag_review.population_tree(gates + [orphan])
    seen = set()
    for n in nodes:
        assert n['parent'] is None or n['parent'] in seen
        seen.add((n['gate'], n['pop']))
    assert [n for n in nodes if n['gate'] == 'Orphan'][0]['parent'] == ('', 'root')


def test_population_tree_empty_model():
    assert ag_review.population_tree([]) == []


# ---------------------------------------------------------------------------
# Shapes as drawn, trainable forms, 1dsep orientation and splitting
# ---------------------------------------------------------------------------

def _shape_gate(name, gate_type='2dsep', x='A', y='B', regions=None, parent=None,
                parent_pop='root', proposed='mixture'):
    regions = regions or {name: 'x+y-'}
    template = {p: {'shape_channels': [x, y],
                    'boundary': [[0.5, 0.0], [1.0, 0.0], [1.0, 0.4], [0.5, 0.4], [0.5, 0.0]],
                    'threshold_x': None, 'threshold_y': None, 'region': r, 'label': p}
                for p, r in regions.items()}
    return {'gate_name': name, 'gate_number': 1, 'gate_type': gate_type,
            'gate_marker_x': x, 'gate_marker_y': y, 'sep_axis': 'x',
            'parent_gate': parent, 'parent_popul': parent_pop,
            'algorithm': ag_core.ALGORITHM_IMPORTED, 'proposed_algorithm': proposed,
            'gate_param': {}, 'stats_parent': {}, 'origin_gate': None,
            'populations': {p: {'label': p, 'label_pos': 1, 'region': r} for p, r in regions.items()},
            'template': template}


def test_convert_target_prefers_the_proposed_algorithm():
    g = _shape_gate('G')
    assert ag_review.convert_target(g, 'tail') == ('mixture', '')
    g['proposed_algorithm'] = None
    assert ag_review.convert_target(g, 'tail') == ('tail', '')


def test_shapes_as_drawn_are_applied_exactly():
    g = _shape_gate('G')
    data = np.array([[0.6, 0.1], [0.6, 0.5], [0.4, 0.1]])
    boundary = ag_core.GateCalculator().calculate(data, g, {'A': 0, 'B': 1})
    assert ag_core.is_shape_boundary(boundary)
    masks = ag_core.population_masks(g, boundary, data, {'A': 0, 'B': 1})
    assert masks['G'].tolist() == [True, False, False]


def test_trainable_definition_uses_the_conversion_target():
    g = _shape_gate('G')
    t = ag_review.trainable_definition(g)
    assert t['algorithm'] == 'mixture' and g['algorithm'] == ag_core.ALGORITHM_IMPORTED
    assert ag_review.trainable_definition(g, 'otsu')['algorithm'] == 'otsu'


def test_drawn_channels_follow_the_shape_then_the_plot():
    g = _shape_gate('G', gate_type='1dsep', x='V', y='FSC-A')
    g['template']['G']['shape_channels'] = ['FSC-A', 'V']
    assert ag_review.drawn_channels(g) == ('FSC-A', 'V')
    del g['template']
    g['sep_axis'] = 'y'
    assert ag_review.drawn_channels(g) == ('FSC-A', 'V')
    g['gate_marker_y'] = None
    assert ag_review.drawn_channels(g) == ('V',)


def test_display_entries_transpose_a_1dsep_separating_on_y():
    g = {'gate_name': 'G', 'gate_type': '1dsep', 'gate_marker_x': 'V', 'gate_marker_y': 'FSC-A',
         'sep_axis': 'y', 'populations': {'neg': {'region': 'below'}}}
    entries = ag_core.GateCalculator({'V': (0.0, 1.0)}).threshold_boundaries(g, 0.3, None)
    shown = ag_core.display_entries(g, entries)
    assert shown['neg']['threshold_y'] == 0.3 and shown['neg']['threshold_x'] is None
    assert np.asarray(shown['neg']['boundary'])[:, 1].max() == pytest.approx(0.3)
    assert entries['neg']['threshold_x'] == 0.3


def test_rename_channels_renames_shapes():
    g = _shape_gate('G')
    out = ag_core.rename_channels([g], {'A': 'A2'})
    assert out[0]['template']['G']['shape_channels'] == ['A2', 'B']
    b = ag_core.rename_boundary_channels({'G': g['template']}, {'B': 'B2'})
    assert b['G']['G']['shape_channels'] == ['A', 'B2']


def test_split_2dsep_chains_gates_and_keeps_population_names():
    g = _shape_gate('CD4 & CD8', x='BUV395', y='SUV387', regions={'CD4': 'x+y-', 'CD8': 'x-y+'})
    g['gate_param'] = {'tail_fraction_y': 0.05}
    child = dict(_poly_gate('Treg', 2, 'CD25', 'FoxP3', 'CD4 & CD8', 'CD4'))
    steps = [{'channel': 'SUV387', 'algorithm': 'tail', 'view': 'BUV395', 'sep_axis': 'y'},
             {'channel': 'BUV395', 'algorithm': 'mixture', 'view': None, 'sep_axis': 'x'}]
    out, names = ag_review.split_2dsep([g, child], 'CD4 & CD8', steps)
    by = {d['gate_name']: d for d in out}
    first = by['CD4 & CD8 · SUV387']
    assert first['gate_marker_x'] == 'SUV387' and first['sep_axis'] == 'y'
    assert first['gate_param']['tail_fraction'] == 0.05
    neg = by['CD4 & CD8 · BUV395 (SUV387-)']
    pos = by['CD4 & CD8 · BUV395 (SUV387+)']
    assert (neg['parent_gate'], neg['parent_popul']) == (first['gate_name'], 'SUV387-')
    assert neg['populations']['CD4']['region'] == 'above'
    assert pos['populations']['CD8']['region'] == 'below'
    assert (by['Treg']['parent_gate'], by['Treg']['parent_popul']) == (neg['gate_name'], 'CD4')
    assert 'CD4 & CD8' not in by and len(names) == 3
    ag_core.ordered_gate_defs(out)


def test_split_2dsep_rejects_taken_names():
    g = _shape_gate('G', regions={'P': 'x+y+'})
    taken = dict(_poly_gate('G · A', 2, 'X', 'Y'))
    steps = [{'channel': 'A', 'algorithm': 'tail', 'view': 'B', 'sep_axis': 'x'},
             {'channel': 'B', 'algorithm': 'tail', 'view': 'A', 'sep_axis': 'y'}]
    with pytest.raises(ValueError):
        ag_review.split_2dsep([g, taken], 'G', steps)


# ---------------------------------------------------------------------------
# Same-named gates, replicate sources, deleting, copying and pasting
# ---------------------------------------------------------------------------

def _hist_gate(name, number, parent, parent_pop, group=None, algorithm='imported'):
    return {'gate_name': name, 'gate_number': number, 'gate_type': '1dsep',
            'gate_marker_x': 'CD80', 'gate_marker_y': None, 'parent_gate': parent,
            'parent_popul': parent_pop, 'algorithm': algorithm, 'proposed_algorithm': 'tail',
            'gate_param': {}, 'stats_parent': {}, 'origin_gate': None,
            'name_group': group or name,
            'populations': {'CD80+': {'label': 'CD80+', 'label_pos': 2, 'region': 'above'}}}


@pytest.fixture
def cd80():
    root = _poly_gate('Cells', 1, 'FSC-A', 'SSC-A')
    a = _hist_gate('CD80+ (Mono)', 2, 'Cells', 'Cells', 'CD80+')
    b = _hist_gate('CD80+ (B)', 3, 'Cells', 'Cells', 'CD80+')
    c = _hist_gate('CD80+ (DC)', 4, 'Cells', 'Cells', 'CD80+')
    return [root, a, b, c]


def test_converting_one_converts_its_same_named_gates(cd80):
    by = ag_core.gates_by_name(cd80)
    by['CD80+ (Mono)'].update(algorithm='mixture', sync_origin=ag_review.SYNC_USER)
    by['CD80+ (Mono)']['gate_param']['tail_fraction'] = 0.02
    changed, conflict = ag_review.apply_name_sync(cd80, 'CD80+ (Mono)')
    assert conflict is None and sorted(changed) == ['CD80+ (B)', 'CD80+ (DC)']
    assert by['CD80+ (B)']['algorithm'] == 'mixture'
    assert by['CD80+ (B)']['gate_param']['tail_fraction'] == 0.02
    assert by['CD80+ (B)']['sync_origin'] == ag_review.SYNC_AUTO


def test_conflicting_choices_stop_syncing_until_reconciled(cd80):
    by = ag_core.gates_by_name(cd80)
    by['CD80+ (Mono)'].update(algorithm='mixture', sync_origin=ag_review.SYNC_USER)
    by['CD80+ (B)'].update(algorithm='tail', sync_origin=ag_review.SYNC_USER)
    changed, conflict = ag_review.apply_name_sync(cd80, 'CD80+ (B)')
    assert changed == [] and conflict == 'CD80+'
    assert by['CD80+ (DC)']['algorithm'] == 'imported'
    assert ag_review.sync_conflicts(cd80) == ['CD80+']
    by['CD80+ (B)']['algorithm'] = 'mixture'
    changed, conflict = ag_review.apply_name_sync(cd80, 'CD80+ (B)')
    assert conflict is None and changed == ['CD80+ (DC)']
    assert ag_review.sync_conflicts(cd80) == []


def test_gates_returned_to_their_shapes_are_left_alone(cd80):
    by = ag_core.gates_by_name(cd80)
    by['CD80+ (DC)']['sync_origin'] = ag_review.SYNC_DRAWN
    by['CD80+ (Mono)'].update(algorithm='mixture', sync_origin=ag_review.SYNC_USER)
    changed, _ = ag_review.apply_name_sync(cd80, 'CD80+ (Mono)')
    assert changed == ['CD80+ (B)'] and by['CD80+ (DC)']['algorithm'] == 'imported'


def test_sync_never_makes_a_gate_replicate_itself(cd80):
    by = ag_core.gates_by_name(cd80)
    by['CD80+ (Mono)'].update(algorithm='replicate', sync_origin=ag_review.SYNC_USER)
    by['CD80+ (Mono)']['gate_param']['replicate_x'] = {'gate': 'CD80+ (B)'}
    changed, _ = ag_review.apply_name_sync(cd80, 'CD80+ (Mono)')
    assert changed == ['CD80+ (DC)']
    assert by['CD80+ (B)']['algorithm'] == 'imported'


def test_replicate_choices_exclude_the_gate_and_its_branch(cd80):
    by = ag_core.gates_by_name(cd80)
    by['Cells']['populations']['Cells']['reference'] = True
    for g in cd80[1:]:
        g['algorithm'] = 'tail'
    labels = [lbl for lbl, _src in ag_review.replicate_choices(cd80, 'CD80+ (B)', 'x')]
    assert labels == ['Copy from gate: CD80+ (Mono)', 'Copy from gate: CD80+ (DC)',
                      'Reference population: Cells (Cells)']
    assert ag_review.replicate_choices(cd80, 'Cells', 'x') == []


def test_convert_target_scatter_shape_and_histogram_rule():
    g = _poly_gate('Cells', 1, 'FSC-A', 'SSC-A')
    assert ag_review.convert_target(g)[0] == 'cells'
    h = _hist_gate('CD80+', 2, None, 'root')
    h['proposal_rule'] = 'histogram'
    assert ag_review.convert_target(h)[0] == 'tail'
    assert ag_review.convert_target(h, positive_fraction=0.4)[0] == 'mixture'
    assert ag_review.convert_target(h, positive_fraction=0.1)[0] == 'tail'


def test_delete_gate_removes_branch_and_clears_sources(cd80):
    child = _hist_gate('child', 5, 'CD80+ (B)', 'CD80+')
    user = _hist_gate('user', 6, 'Cells', 'Cells', algorithm='replicate')
    user['gate_param'] = {'replicate_x': {'gate': 'child'}}
    defs, removed, orphaned = ag_review.delete_gates(cd80 + [child, user], ['CD80+ (B)'])
    assert sorted(removed) == ['CD80+ (B)', 'child'] and orphaned == ['user']
    assert ag_core.gates_by_name(defs)['user']['gate_param']['replicate_x'] is None


def test_delete_population_keeps_the_gates_other_populations():
    q = _thr_gate('Q', 1, 'A', 'B', None, 'root')
    below = _poly_gate('below', 2, 'X', 'Y', 'Q', 'x+y-')
    other = _poly_gate('other', 3, 'X', 'Y', 'Q', 'x-y+')
    defs, removed, _ = ag_review.delete_population([q, below, other], 'Q', 'x+y-')
    by = ag_core.gates_by_name(defs)
    assert removed == ['below'] and list(by['Q']['populations']) == ['x-y+'] and 'other' in by
    defs, removed, _ = ag_review.delete_population(defs, 'Q', 'x-y+')
    assert sorted(removed) == ['Q', 'other']


def test_paste_branch_renames_and_keeps_internal_links(cd80):
    child = _hist_gate('child', 5, 'CD80+ (B)', 'CD80+')
    child['gate_param'] = {'replicate_x': {'gate': 'CD80+ (B)'}}
    clip = ag_review.copy_gates(cd80 + [child], ['CD80+ (B)'], branch=True)
    defs, names = ag_review.paste_gates(cd80 + [child], clip, 'CD80+ (Mono)', 'CD80+')
    by = ag_core.gates_by_name(defs)
    assert names == ['CD80+ (CD80+)', 'child (CD80+)']
    assert by['CD80+ (CD80+)']['parent_gate'] == 'CD80+ (Mono)'
    assert by['child (CD80+)']['parent_gate'] == 'CD80+ (CD80+)'
    assert by['child (CD80+)']['gate_param']['replicate_x'] == {'gate': 'CD80+ (CD80+)'}
    assert by['CD80+ (CD80+)']['name_group'] == 'CD80+'
    assert len(ag_review.sync_members(defs, 'CD80+ (CD80+)')) == 4
    defs2, names2 = ag_review.paste_gates(defs, clip, 'CD80+ (Mono)', 'CD80+')
    assert names2[0] == 'CD80+ (CD80+) 2'


def test_rename_references_follows_sources_and_stats_parents(cd80):
    by = ag_core.gates_by_name(cd80)
    by['CD80+ (B)']['gate_param']['replicate_x'] = {'population': 'Cells/Cells'}
    by['CD80+ (B)']['stats_parent'] = {'CD80+': 'Cells/Cells'}
    ag_review.rename_references(cd80, 'Cells', 'Live')
    assert by['CD80+ (B)']['parent_gate'] == 'Live'
    assert by['CD80+ (B)']['gate_param']['replicate_x'] == {'population': 'Live/Cells'}
    assert by['CD80+ (B)']['stats_parent'] == {'CD80+': 'Live/Cells'}


# ---------------------------------------------------------------------------
# Filters, feeding a gate's settings forward and conversion flags
# ---------------------------------------------------------------------------

@pytest.fixture
def cd25_tree():
    root = _poly_gate('Cells', 1, 'FSC-A', 'SSC-A')
    tconv = _poly_gate('Tconv', 2, 'CD3', 'CD4', 'Cells', 'Cells')
    tem = _poly_gate('Tem', 3, 'CCR7', 'CD45RA', 'Tconv', 'Tconv')
    b = _poly_gate('B', 4, 'CD3', 'CD19', 'Cells', 'Cells')
    on_tconv = _hist_gate('CD80+ (Tconv)', 5, 'Tconv', 'Tconv', 'CD80+')
    on_tem = _hist_gate('CD80+ (Tem)', 6, 'Tem', 'Tem', 'CD80+')
    on_b = _hist_gate('CD80+ (B)', 7, 'B', 'B', 'CD80+')
    return [root, tconv, tem, b, on_tconv, on_tem, on_b]


def test_feed_forward_reaches_same_named_gates_below_only(cd25_tree):
    by = ag_core.gates_by_name(cd25_tree)
    by['CD80+ (Tconv)'].update(algorithm='mixture', sync_origin=ag_review.SYNC_USER)
    by['CD80+ (Tconv)']['gate_param']['tail_fraction'] = 0.03
    # Tem is below Tconv, so the gate drawn on Tem is "below" the Tconv gate's population.
    changed, skipped = ag_review.feed_forward(cd25_tree, 'CD80+ (Tconv)')
    assert changed == ['CD80+ (Tem)'] and skipped == []
    assert by['CD80+ (Tem)']['algorithm'] == 'mixture'
    assert by['CD80+ (Tem)']['gate_param']['tail_fraction'] == 0.03
    assert by['CD80+ (Tem)']['sync_origin'] == ag_review.SYNC_USER
    assert by['CD80+ (B)']['algorithm'] == 'imported'


def test_feed_forward_everywhere_reaches_every_same_named_gate(cd25_tree):
    by = ag_core.gates_by_name(cd25_tree)
    by['CD80+ (Tem)'].update(algorithm='mixture', sync_origin=ag_review.SYNC_USER)
    by['CD80+ (Tem)']['gate_param']['tail_fraction'] = 0.03
    assert ag_review.feed_forward(cd25_tree, 'CD80+ (Tem)') == ([], [])
    changed, skipped = ag_review.feed_forward(cd25_tree, 'CD80+ (Tem)', everywhere=True)
    assert sorted(changed) == ['CD80+ (B)', 'CD80+ (Tconv)'] and skipped == []
    assert all(by[n]['algorithm'] == 'mixture' and by[n]['gate_param']['tail_fraction'] == 0.03
               for n in changed)


def test_feed_forward_ignores_a_gate_still_as_drawn(cd25_tree):
    assert ag_review.feed_forward(cd25_tree, 'CD80+ (Tconv)') == ([], [])


def test_feed_forward_never_makes_a_gate_replicate_itself(cd25_tree):
    by = ag_core.gates_by_name(cd25_tree)
    by['CD80+ (Tconv)'].update(algorithm='replicate', sync_origin=ag_review.SYNC_USER)
    by['CD80+ (Tconv)']['gate_param']['replicate_x'] = {'gate': 'CD80+ (Tem)'}
    changed, skipped = ag_review.feed_forward(cd25_tree, 'CD80+ (Tconv)')
    assert changed == [] and skipped == ['CD80+ (Tem)']


def test_conflicting_gate_names_lists_the_whole_group(cd80):
    by = ag_core.gates_by_name(cd80)
    assert ag_review.conflicting_gate_names(cd80) == set()
    by['CD80+ (Mono)'].update(algorithm='mixture', sync_origin=ag_review.SYNC_USER)
    by['CD80+ (B)'].update(algorithm='tail', sync_origin=ag_review.SYNC_USER)
    assert ag_review.conflicting_gate_names(cd80) == {'CD80+ (Mono)', 'CD80+ (B)', 'CD80+ (DC)'}


def test_gates_below_population_follows_the_branch(cd25_tree):
    assert ag_review.gates_below_population(cd25_tree, 'Tconv', 'Tconv') == {
        'Tem', 'CD80+ (Tconv)', 'CD80+ (Tem)'}
    assert len(ag_review.gates_below_population(cd25_tree, None, 'root')) == len(cd25_tree)


def test_conversion_check_flags_large_moves_only():
    g = _hist_gate('CD80+', 1, None, 'root')
    drawn = {'CD80+': {'threshold_x': 0.40, 'threshold_y': None}}
    near = {'CD80+': {'threshold_x': 0.43, 'threshold_y': None}}
    far = {'CD80+': {'threshold_x': 0.60, 'threshold_y': None}}
    assert ag_review.conversion_check(g, drawn, near, {'CD80+': 0.20}, {'CD80+': 0.22}) is None
    flag = ag_review.conversion_check(g, drawn, far, {'CD80+': 0.20}, {'CD80+': 0.22})
    assert flag['shift'] == pytest.approx(0.2) and 'threshold' in flag['message']
    flag = ag_review.conversion_check(g, drawn, near, {'CD80+': 0.20}, {'CD80+': 0.45})
    assert flag['fraction'] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Changing a gate between 1dsep and 2dsep, renaming its populations
# ---------------------------------------------------------------------------

def test_2dsep_to_1dsep_keeps_population_names_and_sides():
    g = _shape_gate('CD14', x='SBUV540', y='SUV387',
                    regions={'CD14+CD11b+': 'x+y-', 'CD14-': 'x-y-'})
    pops, problem = ag_review.convert_threshold_regions(
        g['populations'], '2dsep', 'SBUV540', 'SUV387', '1dsep', 'SBUV540', 'SUV387')
    assert problem == ''
    assert {p: m['region'] for p, m in pops.items()} == {'CD14+CD11b+': 'above', 'CD14-': 'below'}
    pops, problem = ag_review.convert_threshold_regions(
        g['populations'], '2dsep', 'SBUV540', 'SUV387', '1dsep', 'SUV387', 'SBUV540')
    assert pops is None and 'same side' in problem
    joined = {'A': {'region': 'x-y-|x+y-'}, 'B': {'region': 'x-y+'}}
    pops, problem = ag_review.convert_threshold_regions(joined, '2dsep', 'X', 'Y', '1dsep', 'X', 'Y')
    assert pops is None and 'both sides' in problem


def test_1dsep_to_2dsep_turns_sides_into_halves():
    pops = {'neg': {'region': 'below'}, 'pos': {'region': 'above'}}
    out, problem = ag_review.convert_threshold_regions(pops, '1dsep', 'A', 'B', '2dsep', 'B', 'A')
    assert problem == ''
    assert out['pos']['region'] == 'x-y+|x+y+' and out['neg']['region'] == 'x-y-|x+y-'


def test_rename_populations_moves_every_reference():
    g = _shape_gate('Gate', regions={'neg': 'x-y-', 'pos': 'x+y+'})
    child = dict(_poly_gate('Child', 2, 'C', 'D', 'Gate', 'pos'))
    other = dict(_poly_gate('Other', 3, 'C', 'D'))
    other['gate_param'] = {'replicate_x': {'population': 'Gate/pos', 'algorithm': 'tail'}}
    other['stats_parent'] = {'gate': 'Gate/pos'}
    trained = {'Gate': {'neg': {}, 'pos': {}}}
    assert ag_review.orphaned_by_populations([g, child, other], 'Gate', {'pos'}) == ['Child']
    ag_review.rename_populations([g, child, other], 'Gate', {'pos': 'CD14+'}, trained)
    assert child['parent_popul'] == 'CD14+'
    assert other['gate_param']['replicate_x']['population'] == 'Gate/CD14+'
    assert other['stats_parent']['gate'] == 'Gate/CD14+'
    assert set(g['template']) == {'neg', 'CD14+'} and set(trained['Gate']) == {'neg', 'CD14+'}


def test_split_2dsep_keeps_a_population_covering_a_whole_side():
    g = _shape_gate('G', regions={'L': 'x-y-|x-y+', 'P': 'x+y-', 'Q': 'x+y+'})
    child = dict(_poly_gate('Kid', 2, 'C', 'D', 'G', 'L'))
    steps = [{'channel': 'A', 'algorithm': 'tail', 'view': 'B', 'sep_axis': 'x'},
             {'channel': 'B', 'algorithm': 'tail', 'view': 'A', 'sep_axis': 'y'}]
    out, _names = ag_review.split_2dsep([g, child], 'G', steps)
    by = {d['gate_name']: d for d in out}
    first = by['G · A']
    assert set(first['populations']) == {'L', 'A+'}
    assert first['populations']['L']['region'] == 'below'
    assert (by['Kid']['parent_gate'], by['Kid']['parent_popul']) == ('G · A', 'L')
    with pytest.raises(ValueError):
        ag_review.split_2dsep([g], 'G', list(reversed(steps)))


# ---------------------------------------------------------------------------
# Overlapping populations
# ---------------------------------------------------------------------------

def test_overlap_flags_a_population_defined_on_two_branches():
    root_a = dict(_poly_gate('Mono', 1, 'A', 'B'))
    root_b = dict(_poly_gate('Lin-', 2, 'C', 'D'))
    deep = dict(_poly_gate('CD14+', 3, 'A', 'B', 'Lin-', 'Lin-'))
    below = dict(_poly_gate('Act', 4, 'E', 'F', 'Mono', 'Mono'))
    below_deep = dict(_poly_gate('Act2', 5, 'E', 'F', 'CD14+', 'CD14+'))
    marker = dict(_poly_gate('HLADR+', 6, 'G', 'H', 'Lin-', 'Lin-'))
    defs = [root_a, root_b, deep, below, below_deep, marker]
    n = 1000
    mono = np.zeros(n, bool); mono[:100] = True
    lin = np.zeros(n, bool); lin[:500] = True
    cd14 = np.zeros(n, bool); cd14[5:100] = True
    masks = {('Mono', 'Mono'): mono, ('Lin-', 'Lin-'): lin, ('CD14+', 'CD14+'): cd14,
             ('Act', 'Act'): mono.copy(), ('Act2', 'Act2'): cd14.copy(),
             ('HLADR+', 'HLADR+'): mono.copy()}
    pairs = ag_review.overlapping_populations(defs, masks)
    assert [(a, b) for a, b, _s in pairs] == [(('Mono', 'Mono'), ('CD14+', 'CD14+'))]
    assert pairs[0][2] == pytest.approx(0.95)
    flags = ag_review.overlap_flags(pairs)
    assert set(flags) == {'Mono', 'CD14+'} and 'CD14+' in flags['Mono']['message']
    assert ag_review.overlapping_populations(defs, masks, limit=0.96) == []


# ---------------------------------------------------------------------------
# Fraction rule for imported threshold gates
# ---------------------------------------------------------------------------

def test_fraction_rule_sets_each_axis_from_the_drawn_fractions():
    g = _shape_gate('G', regions={'DP': 'x+y+', 'L': 'x-y-|x+y-|x-y+'})
    g['proposal_rule'] = ag_review.RULE_FRACTION
    fr = ag_review.positive_fractions(g, {'DP': 0.02, 'L': 0.98})
    assert fr == {'x': pytest.approx(0.02), 'y': pytest.approx(0.02)}
    assert ag_review.proposed_axis_algorithms(g, None, fr) == {'x': 'tail', 'y': 'tail'}
    fr = ag_review.positive_fractions(g, {'DP': 0.4, 'L': 0.6})
    assert ag_review.proposed_axis_algorithms(g, None, fr) == {'x': 'mixture', 'y': 'mixture'}
    g['gate_param'] = {'algorithm_y': 'otsu'}
    assert ag_review.proposed_axis_algorithms(g, None, fr)['y'] == 'otsu'
    g['gate_param'] = {}
    assert ag_review.proposed_axis_algorithms(g, None, None) == {'x': 'mixture', 'y': 'mixture'}
    d = ag_review.trainable_definition(dict(g, gate_param={}), 'tail', {'x': 'tail', 'y': 'mixture'})
    assert ag_core.axis_algorithm(d, 'y') == 'mixture'


def test_fraction_rule_on_a_1dsep_gate_with_a_view():
    g = _shape_gate('G', gate_type='1dsep', regions={'pos': 'above'})
    g['proposal_rule'] = ag_review.RULE_FRACTION
    assert ag_review.convert_target(g, None, {'x': 0.9}) == ('mixture', '')
    assert ag_review.convert_target(g, None, {'x': 0.1}) == ('tail', '')
    g.pop('proposal_rule')
    assert ag_review.convert_target(g, None, {'x': 0.1}) == ('mixture', '')


# ---------------------------------------------------------------------------
# Proposals from the review events
# ---------------------------------------------------------------------------

def _rect(x0, x1, y0, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def _drawn(name, gate_type, x, y, shapes, parent=None, parent_pop='root', rule='fraction',
           regions=None):
    """Imported gate with drawn shapes {pop: polygon or ('range', lo, hi)}."""
    chans = [x] if y is None else [x, y]
    template, pops = {}, {}
    for k, (p, shape) in enumerate(shapes.items()):
        region = (regions or {}).get(p) or ('inside' if gate_type == 'free' else 'above')
        if isinstance(shape, tuple):
            lo, hi = shape[1], shape[2]
            entry = {'shape_channels': [x], 'range': [lo, hi], 'boundary': _rect(lo, hi, 0, 1)}
        else:
            entry = {'shape_channels': chans, 'boundary': shape}
        template[p] = dict(entry, region=region, label=p)
        pops[p] = {'label': p, 'label_pos': k + 1, 'region': region}
    return {'gate_name': name, 'gate_number': 1, 'gate_type': gate_type, 'gate_marker_x': x,
            'gate_marker_y': y, 'sep_axis': 'x', 'parent_gate': parent,
            'parent_popul': parent_pop, 'algorithm': ag_core.ALGORITHM_IMPORTED,
            'proposed_algorithm': 'mixture', 'proposal_rule': rule, 'gate_param': {},
            'stats_parent': {}, 'origin_gate': None, 'populations': pops, 'template': template}


def _evidence(defs, data, index, name, zeros=None):
    reference = {g['gate_name']: g['template'] for g in defs}
    res = ag_core.run_gating(defs, data, index, reference=reference, default_mode=ag_core.MODE_FIXED,
                             keep_partial=True)
    g = ag_core.gates_by_name(defs)[name]
    return ag_review.conversion_evidence(g, defs, data, index, res, None, zeros or {}), res


def test_single_shape_axis_that_cuts_nothing_converts_to_1dsep():
    rng = np.random.default_rng(1)
    n = 6000
    data = np.column_stack([rng.normal(0.5, 0.08, n),
                            np.where(rng.random(n) < 0.5, rng.normal(0.25, 0.04, n),
                                     rng.normal(0.7, 0.05, n))])
    index = {'A': 0, 'B': 1}
    wide = _drawn('CD3', '2dsep', 'A', 'B', {'CD3': _rect(0.0, 1.0, 0.45, 0.95)},
                  regions={'CD3': 'x-y+'})
    assert ag_review.separating_axis_dropped(wide, data, index) == 'x'
    one = ag_review.without_axis(wide, 'x')
    assert (one['gate_type'], one['gate_marker_x'], one['gate_marker_y'], one['sep_axis']) == \
        ('1dsep', 'B', 'A', 'y')
    assert one['populations']['CD3']['region'] == 'above'
    assert ag_core.plot_channels(one) == ('A', 'B')
    ev, _res = _evidence([wide], data, index, 'CD3')
    assert ev['drop_axis'] == 'x'
    proposed = ag_review.proposed_definition(wide, evidence=ev)
    assert proposed['gate_type'] == '1dsep' and proposed['algorithm'] in ag_review.FIT_ALGORITHMS
    narrow = _drawn('CD3', '2dsep', 'A', 'B', {'CD3': _rect(0.55, 1.0, 0.45, 0.95)},
                    regions={'CD3': 'x+y+'})
    assert ag_review.separating_axis_dropped(narrow, data, index) is None


def _marker_tree(values):
    """Cells (all events) → Sub (all events) → M+ histogram on M drawn from 0.3."""
    n = len(values)
    data = np.column_stack([np.full(n, 0.5), np.full(n, 0.5), values])
    index = {'FSC': 0, 'SSC': 1, 'M': 2}
    cells = _drawn('Cells', 'free', 'FSC', 'SSC', {'Cells': _rect(0, 1, 0, 1)}, rule=None)
    sub = _drawn('Sub', 'free', 'FSC', 'SSC', {'Sub': _rect(0, 1, 0, 1)}, 'Cells', 'Cells',
                 rule=None)
    marker = _drawn('M+', '1dsep', 'M', None, {'M+': ('range', 0.3, 1.0)}, 'Sub', 'Sub',
                    rule='histogram')
    return [cells, sub, marker], data, index


def test_1dsep_centred_on_zero_proposes_tail():
    rng = np.random.default_rng(2)
    defs, data, index = _marker_tree(rng.normal(0.2, 0.05, 5000))
    ev, _res = _evidence(defs, data, index, 'M+', zeros={'M': 0.2})
    assert ev['below_zero']['x'] > ag_review.CENTRED_BELOW_ZERO
    assert ag_review.evidence_axes(ev)['x'][0] == 'tail'


def test_1dsep_spanning_zero_takes_the_better_fit():
    rng = np.random.default_rng(3)
    values = np.concatenate([rng.normal(0.2, 0.03, 3000), rng.normal(0.6, 0.05, 3000)])
    defs, data, index = _marker_tree(values)
    ev, _res = _evidence(defs, data, index, 'M+', zeros={'M': 0.18})
    assert ag_review.SPANS_ZERO_BELOW <= ev['below_zero']['x'] < ag_review.CENTRED_BELOW_ZERO
    best = max(ev['fit'], key=ev['fit'].get)[0]
    assert ag_review.evidence_axes(ev)['x'][0] == best
    assert ag_review.convert_target(defs[2], evidence=ev)[0] == best


def test_1dsep_mostly_positive_above_zero_replicates_from_the_population_above():
    rng = np.random.default_rng(4)
    values = np.concatenate([rng.normal(0.35, 0.03, 1500), rng.normal(0.6, 0.05, 4500)])
    defs, data, index = _marker_tree(values)
    ev, _res = _evidence(defs, data, index, 'M+', zeros={'M': 0.1})
    assert ev['below_zero']['x'] < ag_review.SPANS_ZERO_BELOW
    assert ev['positive']['x'] > ag_review.REPLICATE_POSITIVE
    assert ev['reference'] == 'Cells/Cells'
    assert ag_review.evidence_axes(ev)['x'][0] == ag_core.ALGORITHM_REPLICATE
    proposed = ag_review.proposed_definition(defs[2], evidence=ev)
    src = proposed['gate_param']['replicate_x']
    assert src['population'] == 'Cells/Cells' and src['default'] is True
    assert src['algorithm'] == 'mixture'
    assert ag_review.default_source_axes(proposed) == ['x']
    labels = [label for label, _src in ag_review.replicate_choices(defs, 'M+', 'x')]
    assert "Anchor population: Cells (Cells)" in labels
    # A hand-chosen source is kept.
    chosen = dict(defs[2], gate_param={'replicate_x': {'population': 'Sub/Sub', 'algorithm': 'tail'}})
    assert ag_review.proposed_definition(chosen, evidence=ev)['gate_param']['replicate_x'] == \
        {'population': 'Sub/Sub', 'algorithm': 'tail'}


def test_default_reference_sources_sync_as_each_gates_own():
    root = _drawn('Root', 'free', 'F', 'S', {'Cells': _rect(0, 1, 0, 1)})
    top = _drawn('Top', 'free', 'F', 'S', {'TopA': _rect(0, 1, 0, 1), 'TopB': _rect(0, 1, 0, 1)},
                 'Root', 'Cells')
    sub_a = _drawn('SubA', 'free', 'F', 'S', {'A': _rect(0, 1, 0, 1)}, 'Top', 'TopA')
    a = _drawn('M+ (A)', '1dsep', 'M', None, {'M+': ('range', 0.3, 1.0)}, 'SubA', 'A')
    b = _drawn('M+ (B)', '1dsep', 'M', None, {'M+': ('range', 0.3, 1.0)}, 'Top', 'TopB')
    c = _drawn('M+ (C)', '1dsep', 'M', None, {'M+': ('range', 0.3, 1.0)}, 'Root', 'Cells')
    for g in (a, b, c):
        g['name_group'] = 'M+'
    a.update(algorithm=ag_core.ALGORITHM_REPLICATE,
             gate_param={'replicate_x': {'population': 'Root/Cells', 'algorithm': 'tail',
                                         'default': True}})
    defs = [root, top, sub_a, a, b, c]
    changed, conflict = ag_review.apply_name_sync(defs, 'M+ (A)')
    assert sorted(changed) == ['M+ (B)', 'M+ (C)'] and conflict is None
    assert b['gate_param']['replicate_x'] == {'population': 'Root/Cells', 'algorithm': 'tail',
                                              'default': True}
    # Drawn on the anchor itself: no default reference.
    assert c['gate_param']['replicate_x'] is None
    assert ag_review.sync_fingerprint(a) == ag_review.sync_fingerprint(b)


def test_anchor_is_the_viable_population_then_singlets_then_cells():
    cells = _drawn('Cells', 'free', 'FSC-A', 'SSC-A', {'Cells': _rect(0, 1, 0, 1)})
    singlets = dict(_drawn('Singlets', 'singlets', 'FSC-A', 'FSC-H', {'Single': _rect(0, 1, 0, 1)},
                           'Cells', 'Cells'), gate_type='singlets')
    live = _drawn('Live', '1dsep', 'ViaKrome 808', 'FSC-A',
                  {'Dead': _rect(0.5, 1, 0, 1), 'Live': _rect(0, 0.5, 0, 1)}, 'Singlets', 'Single',
                  regions={'Dead': 'above', 'Live': 'below'})
    t = _drawn('T', 'free', 'CD3', 'CD4', {'T': _rect(0, 1, 0, 1)}, 'Live', 'Live')
    assert ag_core.viability_channels(['ViaKrome 808', 'CD3', 'FSC-A']) == {'ViaKrome 808'}
    assert ag_core.anchor_population([cells, singlets, live, t]) == 'Live/Live'
    assert ag_core.anchor_population([cells, singlets, t]) == 'Singlets/Single'
    assert ag_core.anchor_population([cells, t]) == 'Cells/Cells'
    assert ag_core.anchor_population([t]) is None
    t['populations']['T'][ag_core.ANCHOR_FLAG] = True
    assert ag_core.anchor_population([cells, singlets, live, t]) == 'T/T'
    del t['populations']['T'][ag_core.ANCHOR_FLAG]
    below = _drawn('M+', '1dsep', 'M', None, {'M+': ('range', 0.3, 1.0)}, 'T', 'T')
    on_anchor = _drawn('N+', '1dsep', 'N', None, {'N+': ('range', 0.3, 1.0)}, 'Live', 'Live')
    defs = [cells, singlets, live, t, below, on_anchor]
    assert ag_review.default_reference(defs, below) == 'Live/Live'
    assert ag_review.default_reference(defs, on_anchor) is None


def test_trimodal_axis_fits_a_three_component_cut():
    rng = np.random.default_rng(7)
    values = np.concatenate([rng.normal(0.2, 0.03, 4000), rng.normal(0.45, 0.04, 2500),
                             rng.normal(0.75, 0.04, 1500)])
    assert 0.55 < ag_core.threshold_mixture(values, 3, cut='upper') < 0.65
    assert 0.27 < ag_core.threshold_mixture(values, 3, cut='lower') < 0.37
    defs, data, index = _marker_tree(values)
    defs[2]['template']['M+']['range'] = [0.6, 1.0]
    defs[2]['template']['M+']['boundary'] = _rect(0.6, 1.0, 0, 1)
    ev, _res = _evidence(defs, data, index, 'M+', zeros={'M': 0.18})
    algo, reason, params = ag_review.evidence_axes(ev)['x']
    assert algo == 'mixture' and params == {'mixture_components': 3, 'mixture_cut': 'upper'}
    proposed = ag_review.proposed_definition(defs[2], evidence=ev)
    assert proposed['gate_param']['mixture_components'] == 3
    assert proposed['gate_param']['mixture_cut'] == 'upper'


def test_split_steps_and_split_keeping_the_shapes_as_drawn():
    g = _drawn('B', '2dsep', 'A', 'C',
               {'nTnP': _rect(0.0, 0.5, 0.0, 1.0), 'PB': _rect(0.5, 1.0, 0.0, 0.3),
                'Tr': _rect(0.5, 1.0, 0.3, 1.0)},
               regions={'nTnP': 'x-y-|x-y+', 'PB': 'x+y-', 'Tr': 'x+y+'})
    child = _drawn('Kid', 'free', 'F', 'S', {'K': _rect(0, 1, 0, 1)}, 'B', 'Tr')
    steps = ag_review.split_steps(g)
    assert [st['channel'] for st in steps] == ['A', 'C']
    assert [st['sep_axis'] for st in steps] == ['x', 'y']
    out, made = ag_review.split_2dsep([g, child], 'B', steps, names={'A': 'CD38', 'C': 'CD24'},
                                      keep_drawn=True)
    by = {d['gate_name']: d for d in out}
    first, second = by[made[0]], by[made[1]]
    assert made == ['B · CD38', 'B · CD24 (CD38+)']
    assert set(first['populations']) == {'nTnP', 'CD38+'}
    assert first['algorithm'] == ag_core.ALGORITHM_IMPORTED and first['proposal_rule'] == 'fraction'
    hull = np.asarray(first['template']['CD38+']['boundary'])
    assert hull[:, 0].min() == pytest.approx(0.5) and hull[:, 1].max() == pytest.approx(1.0)
    assert set(second['template']) == {'PB', 'Tr'}
    assert ag_core.plot_channels(second) == ('A', 'C')
    assert by['Kid']['parent_gate'] == made[1]
    assert ag_review.split_steps(_drawn('Q', '2dsep', 'A', 'C', {'P': _rect(0, 1, 0, 1)},
                                        regions={'P': 'x+y+'})) is None


def test_scatter_shape_on_the_debris_keeps_the_debris_side():
    rng = np.random.default_rng(5)
    data = np.vstack([np.column_stack([rng.normal(0.1, 0.03, 3000), rng.normal(0.1, 0.04, 3000)]),
                      np.column_stack([rng.normal(0.5, 0.06, 7000), rng.normal(0.4, 0.1, 7000)])])
    index = {'FSC': 0, 'SSC': 1}
    low = _drawn('low', 'free', 'FSC', 'SSC', {'low': _rect(0.0, 0.2, 0.0, 0.25)}, rule=None)
    ev, _res = _evidence([low], data, index, 'low')
    assert ev['cell_side'] == ag_core.CELL_SIDE_DEBRIS
    assert ag_review.proposed_definition(low, evidence=ev)['gate_param']['cell_side'] == \
        ag_core.CELL_SIDE_DEBRIS
    cells = _drawn('cells', 'free', 'FSC', 'SSC', {'cells': _rect(0.3, 0.8, 0.1, 0.8)}, rule=None)
    assert _evidence([cells], data, index, 'cells')[0]['cell_side'] == ag_core.CELL_SIDE_CELLS


def test_threshold_sides_of_a_joined_region():
    g = {'gate_type': '2dsep', 'populations': {
        'L': {'region': 'x-y-|x-y+'}, 'P': {'region': 'x+y-'}, 'Q': {'region': 'x+y+'}}}
    assert ag_core.threshold_sides(g, 'x') == {'-', '+'}
    assert ag_core.threshold_sides(g, 'y') == {'+'}

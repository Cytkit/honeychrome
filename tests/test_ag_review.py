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
    assert [n['label'] for n in nodes] == ['All events (root)', 'Cells', 'T', 'B', 'NK']
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

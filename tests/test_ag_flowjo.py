"""
test_ag_flowjo.py
-----------------
Tests for bundled_plugins/ag_flowjo.py (FlowJo workspace import for the
Gating Model Builder): parsing, parameter alignment, drop planning, and
conversion to gate definitions whose imported boundaries reproduce the
FlowJo gates on Honeychrome's display scale.

A small synthetic workspace in FlowJo 10 format is built in this file. The
optional test at the end also reads a real workspace named by the
HC_FLOWJO_TEST_WSP environment variable.

Usage:
    pytest tests/test_ag_flowjo.py
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

import ag_core  # noqa: E402
import ag_flowjo as fj  # noqa: E402

pytest.importorskip('flowkit')

NS = ('xmlns:gating="http://www.isac-net.org/std/Gating-ML/v2.0/gating" '
      'xmlns:transforms="http://www.isac-net.org/std/Gating-ML/v2.0/transformations" '
      'xmlns:data-type="http://www.isac-net.org/std/Gating-ML/v2.0/datatypes"')

T_MAX = 262144.0


def _dim(name, lo=None, hi=None):
    attrs = ''
    if lo is not None:
        attrs += f' gating:min="{lo}"'
    if hi is not None:
        attrs += f' gating:max="{hi}"'
    return (f'<gating:dimension{attrs}><data-type:fcs-dimension data-type:name="{name}"/>'
            '</gating:dimension>')


def _vertices(points):
    return ''.join(
        '<gating:vertex>' + ''.join(f'<gating:coordinate data-type:value="{v}"/>' for v in p)
        + '</gating:vertex>' for p in points)


def _rect(pop, gid, dims, children='', inside=1, quad=None):
    q = f' quadId="{quad}"' if quad is not None else ''
    body = ''.join(_dim(*d) for d in dims)
    return (f'<Population name="{pop}" count="0"><Gate gating:id="{gid}">'
            f'<gating:RectangleGate eventsInside="{inside}"{q}>{body}</gating:RectangleGate></Gate>'
            f'<Subpopulations>{children}</Subpopulations></Population>')


def _poly(pop, gid, x, y, points, children=''):
    return (f'<Population name="{pop}" count="0"><Gate gating:id="{gid}">'
            f'<gating:PolygonGate eventsInside="1">{_dim(x)}{_dim(y)}{_vertices(points)}'
            f'</gating:PolygonGate></Gate><Subpopulations>{children}</Subpopulations></Population>')


def _ellipse(pop, gid, x, y, foci, edge):
    return (f'<Population name="{pop}" count="0"><Gate gating:id="{gid}">'
            f'<gating:EllipsoidGate eventsInside="1">{_dim(x)}{_dim(y)}'
            f'<gating:foci>{_vertices(foci)}</gating:foci><gating:edge>{_vertices(edge)}</gating:edge>'
            f'</gating:EllipsoidGate></Gate></Population>')


SINGLET_POLY = [(20000, 15000), (250000, 230000), (250000, 260000), (20000, 30000)]
QUAD_T = (1000.0, 800.0)          # PE, APC quadrant cross in raw units


def _tree():
    pe, apc = 'Comp-PE-A', 'Comp-APC-A'
    tx, ty = QUAD_T
    quads = (
        _rect('Q1: CD4- , CD8+', 'q1', [(pe, None, tx), (apc, ty, None)], quad=0)
        + _rect('Q2: CD4+ , CD8+', 'q2', [(pe, tx, None), (apc, ty, None)], quad=1)
        + _rect('Q3: CD4+ , CD8-', 'q3', [(pe, tx, None), (apc, None, ty)], quad=2,
                children=_rect('CD25+/hi', 'r1', [('Comp-FITC-A', 500, None)]))
        + _rect('Q4: CD4- , CD8-', 'q4', [(pe, None, tx), (apc, None, ty)], quad=3)
    )
    staggered = (
        _rect('T cells', 't1', [('Comp-BV421-A', 2000, 200000), ('Comp-APC-A', -500, 800)])
        + _rect('Other', 't2', [('Comp-BV421-A', -800, 1200), ('Comp-APC-A', 1500, 100000)])
    )
    dropped = (
        _rect('AF bright', 'a1', [('Comp-AF-A', 3000, None)],
              children=_rect('below AF', 'a2', [('Comp-PE-A', 100, None)]))
        + '<NotNode name="not T" count="0"><Subpopulations></Subpopulations></NotNode>'
        + _rect('Excluded', 'x1', [('Comp-PE-A', 100, 900), ('Comp-APC-A', 100, 900)], inside=0)
        + _rect('CD4 height', 'h1', [('Comp-PE-H', 100, None)])
    )
    blob = _ellipse('Blob', 'e1', 'Comp-PE-A', 'Comp-APC-A',
                    foci=[(500, 500), (2000, 2000)],
                    edge=[(3000, 3000), (100, 100), (300, 1500), (1500, 300)])
    singlets = _poly('Single Cells', 'p1', 'FSC-A', 'FSC-H', SINGLET_POLY,
                     children=quads + staggered + dropped + blob)
    return _rect('Lymph', 'g1', [('FSC-A', 50000, 220000), ('SSC-A', 10000, 150000)],
                 children=singlets)


def _transforms():
    lin = '<transforms:linear transforms:minRange="0" transforms:maxRange="262144" transforms:gain="1">' \
          '<data-type:parameter data-type:name="{p}"/></transforms:linear>'
    bx = '<transforms:biex transforms:length="256" transforms:maxRange="262144" transforms:neg="0" ' \
         'transforms:width="-100" transforms:pos="4.42"><data-type:parameter data-type:name="{p}"/>' \
         '</transforms:biex>'
    out = ''.join(lin.format(p=p) for p in ('FSC-A', 'FSC-H', 'SSC-A'))
    out += ''.join(bx.format(p=p) for p in ('Comp-PE-A', 'Comp-APC-A', 'Comp-BV421-A',
                                            'Comp-FITC-A', 'Comp-AF-A', 'Comp-PE-H'))
    return out


LABELS = {'FSC-A': '', 'FSC-H': '', 'SSC-A': '', 'PE-A': 'CD4_IgM', 'APC-A': 'CD8',
          'BV421-A': 'CD3', 'FITC-A': 'CD25', 'AF-A': '', 'PE-H': 'CD4_IgM',
          'Comp-PE-A': 'CD4_IgM', 'Comp-APC-A': 'CD8', 'Comp-BV421-A': 'CD3',
          'Comp-FITC-A': 'CD25', 'Comp-AF-A': '', 'Comp-PE-H': 'CD4_IgM'}


def _workspace_xml():
    kws = ''.join(
        f'<Keyword name="$P{i}N" value="{n}"/><Keyword name="$P{i}S" value="{s}"/>'
        for i, (n, s) in enumerate(LABELS.items(), start=1))
    tree = _tree()
    return (
        f'<?xml version="1.0" encoding="UTF-8"?><Workspace version="20.0" flowJoVersion="10.10.2" {NS}>'
        '<Matrices><transforms:spilloverMatrix prefix="Comp-" suffix="" name="Acquisition-defined"/>'
        '</Matrices>'
        '<Groups><GroupNode name="All Samples"><Group name="All Samples"><SampleRefs>'
        '<SampleRef sampleID="1"/></SampleRefs></Group>'
        f'<Subpopulations>{tree}</Subpopulations></GroupNode></Groups>'
        '<SampleList><Sample><DataSet uri="file:/x/s1.fcs" sampleID="1"/>'
        f'<Transformations>{_transforms()}</Transformations><Keywords>{kws}</Keywords>'
        '<SampleNode name="s1.fcs" sampleID="1" count="1000"><Subpopulations>'
        + _rect('Lymph', 'g9', [('FSC-A', 50000, 220000), ('SSC-A', 10000, 150000)])
        + '</Subpopulations></SampleNode></Sample></SampleList></Workspace>'
    )


def _exp():
    lin = {'id': 0, 'scale_t': T_MAX, 'linear_a': 0.0, 'logicle_w': 0.5, 'logicle_m': 4.5,
           'logicle_a': 0.0, 'log_m': 6, 'limits': [0.0, 1.0]}
    lgc = dict(lin, id=1)
    channels = ['Time', 'FSC-A', 'FSC-H', 'SSC-A', 'BV421', 'PE', 'APC', 'FITC', 'AF Abundance']
    fl = ['BV421', 'PE', 'APC', 'FITC']
    return fj.ExperimentChannels(
        channels=channels, fluorescence=fl, scatter=['FSC-A', 'FSC-H', 'SSC-A'],
        antigens={'BV421': 'CD3', 'PE': 'CD4', 'APC': 'CD8', 'FITC': 'CD25'},
        transforms={**{c: dict(lin) for c in ('FSC-A', 'FSC-H', 'SSC-A')},
                    **{c: dict(lgc) for c in fl}},
        singlet_pair=('FSC-A', 'FSC-H'),
    )


def _fluor_match(name):
    for f in ('BV421', 'PE', 'APC', 'FITC'):
        if fj._norm(name) == fj._norm(f):
            return f
    return None


def _marker_match(name):
    return name.upper() if name.upper() in ('CD3', 'CD4', 'CD8', 'CD25', 'IGM') else None


@pytest.fixture
def ws():
    return fj.parse_workspace(ET.fromstring(_workspace_xml()), 'synthetic.wsp')


@pytest.fixture
def source(ws):
    return ws.default_source()


@pytest.fixture
def mapping(ws, source):
    al = fj.align_parameters(fj.gated_parameters(source), source.labels, _exp(), ws.affixes,
                             fluor_match=_fluor_match, marker_match=_marker_match)
    return {p: v['channel'] for p, v in al.items()}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_sources_and_default(ws):
    kinds = [(s.kind, s.name, s.n_populations) for s in ws.sources]
    assert ('group', 'All Samples', 15) in kinds and ('sample', 's1.fcs', 1) in kinds
    assert ws.default_source().kind == 'group'
    assert ('Comp-', '') in ws.affixes


def test_group_inherits_member_labels_and_transforms(source):
    assert source.labels['Comp-APC-A'] == 'CD8'
    assert source.transforms['Comp-PE-A'].kind == 'biex'
    assert source.transforms['FSC-A'].kind == 'linear'
    assert source.transforms['FSC-A'].display() is None


def test_raw_coordinates_are_kept(source):
    pops = {p.key: p for p in source.iter_populations()}
    lymph = pops['Lymph'].gate
    assert lymph.kind == 'rectangle' and lymph.bounds == [(50000.0, 220000.0), (10000.0, 150000.0)]
    sing = pops['Lymph/Single Cells'].gate
    assert sing.kind == 'polygon' and sing.vertices == [tuple(map(float, v)) for v in SINGLET_POLY]
    q1 = pops['Lymph/Single Cells/Q1: CD4- , CD8+'].gate
    assert q1.quad_id == 0 and q1.bounds == [(None, 1000.0), (800.0, None)]
    assert pops['Lymph/Single Cells/not T'].node_type == 'NotNode'
    assert pops['Lymph/Single Cells/Excluded'].gate.inside is False
    blob = pops['Lymph/Single Cells/Blob'].gate
    assert blob.kind == 'ellipse' and len(blob.edge) == 4


def test_not_a_workspace():
    with pytest.raises(ValueError):
        fj.parse_workspace(ET.fromstring('<Gating-ML/>'))


# ---------------------------------------------------------------------------
# Alignment and drop planning
# ---------------------------------------------------------------------------

def test_alignment_methods(ws, source):
    al = fj.align_parameters(fj.gated_parameters(source), source.labels, _exp(), ws.affixes,
                             fluor_match=_fluor_match, marker_match=_marker_match)
    assert al['FSC-A']['method'] == fj.MATCH_NAME
    assert al['Comp-PE-A'] == {'channel': 'PE', 'method': fj.MATCH_FLUOROPHORE, 'note': '',
                               'label': 'CD4_IgM'}
    assert al['Comp-AF-A']['channel'] is None
    assert al['Comp-PE-H']['channel'] is None and 'Height' in al['Comp-PE-H']['note']


def test_antigen_fallback_and_label_disagreement():
    exp = _exp()
    exp.fluorescence = exp.fluorescence + ['KLRG1 PerCP-Fire 810_controls']
    exp.channels = exp.channels + ['KLRG1 PerCP-Fire 810_controls']
    exp.antigens['KLRG1 PerCP-Fire 810_controls'] = 'KLRG1'
    labels = {'Comp-PerCP Fire 806-A': 'KLRG1_IgD', 'Comp-APC-A': 'CD19'}

    def marker(name):
        return name.upper() if name.upper() in ('KLRG1', 'IGD', 'CD19', 'CD8') else None
    al = fj.align_parameters(list(labels), labels, exp, None, _fluor_match, marker)
    assert al['Comp-PerCP Fire 806-A']['channel'] == 'KLRG1 PerCP-Fire 810_controls'
    assert al['Comp-PerCP Fire 806-A']['method'] == fj.MATCH_ANTIGEN
    assert al['Comp-APC-A']['method'] == fj.MATCH_FLUOROPHORE_LABEL_DIFFERS


def test_mapping_problems():
    assert fj.mapping_problems({'Comp-PE-A': 'PE', 'PE-A': 'PE'}) == []
    assert fj.mapping_problems({'Comp-PE-A': 'PE', 'Comp-APC-A': 'PE'})


def test_plan_drops_cascade(source, mapping):
    plan = fj.plan_import(source, mapping)
    base = 'Lymph/Single Cells/'
    assert base + 'AF bright' in plan.dropped and base + 'AF bright/below AF' in plan.dropped
    assert 'parent' in plan.dropped[base + 'AF bright/below AF']
    assert 'Boolean' in plan.dropped[base + 'not T']
    assert 'outside' in plan.dropped[base + 'Excluded']
    assert base + 'CD4 height' in plan.dropped
    msgs = plan.messages()
    assert any("'Lymph/Single Cells/AF bright' and 1 population(s) below it" in m for m in msgs)
    assert len(plan.keep) == 10


def test_user_drop_removes_branch(source, mapping):
    m = dict(mapping, **{'Comp-PE-A': None})
    plan = fj.plan_import(source, m)
    q3 = 'Lymph/Single Cells/Q3: CD4+ , CD8-'
    assert q3 in plan.dropped and q3 + '/CD25+/hi' in plan.dropped


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def _defs(source, mapping, **kw):
    gdefs, report = fj.build_gate_definitions(source, mapping, _exp(), **kw)
    return {g['gate_name']: g for g in gdefs}, report


def test_gate_types_and_hierarchy(source, mapping):
    defs, report = _defs(source, mapping)
    assert defs['Lymph']['gate_type'] == 'free'
    assert defs['Single Cells']['gate_type'] == 'singlets'
    assert all(g['algorithm'] == ag_core.ALGORITHM_IMPORTED for g in defs.values())
    ag_core.ordered_gate_defs(list(defs.values()))


def test_singlet_gate_becomes_the_top_of_the_hierarchy(source, mapping):
    defs, report = _defs(source, mapping)
    single, lymph = defs['Single Cells'], defs['Lymph']
    assert (single['parent_gate'], single['parent_popul']) == (None, 'root')
    assert (lymph['parent_gate'], lymph['parent_popul']) == ('Single Cells', 'Single Cells')
    assert single['gate_number'] == 1 and lymph['gate_number'] == 2
    quad = next(g for g in defs.values() if 'Q1: CD4- , CD8+' in g['populations'])
    assert (quad['parent_gate'], quad['parent_popul']) == ('Lymph', 'Lymph')
    assert single['source']['parent_count'] is None and lymph['source']['parent_count'] is None
    assert any('top of the hierarchy' in n for n in report['notes']['Single Cells'])
    assert report['populations']['Lymph/Single Cells'] == 'Lymph/Lymph'
    assert 'Lymph' not in report['populations']


def test_quadrants_merge_into_one_gate_with_shapes_as_drawn(source, mapping):
    defs, report = _defs(source, mapping)
    quad = [g for g in defs.values() if 'Q1: CD4- , CD8+' in g['populations']]
    assert len(quad) == 1
    g = quad[0]
    assert g['gate_type'] == '2dsep' and g['proposed_algorithm'] == 'mixture'
    regions = {p: v['region'] for p, v in g['populations'].items()}
    assert regions == {'Q1: CD4- , CD8+': 'x-y+', 'Q2: CD4+ , CD8+': 'x+y+',
                       'Q3: CD4+ , CD8-': 'x+y-', 'Q4: CD4- , CD8-': 'x-y-'}
    exp = _exp()
    tx = float(ag_core.make_transform(exp.transforms['PE']).apply(np.array([QUAD_T[0]]))[0])
    q3 = g['template']['Q3: CD4+ , CD8-']
    assert q3['shape_channels'] == ['PE', 'APC'] and q3['threshold_x'] is None
    box = np.asarray(q3['boundary'])
    assert box[:, 0].min() == pytest.approx(tx)
    assert box[:, 0].max() > exp.transforms['PE']['limits'][1]
    child = defs['CD25+∕hi']
    assert child['parent_gate'] == g['gate_name'] and child['parent_popul'] == 'Q3: CD4+ , CD8-'
    assert child['gate_type'] == '1dsep' and child['gate_marker_y'] is None
    assert child['proposed_algorithm'] == 'tail'
    assert child['populations']['CD25+∕hi']['region'] == ag_core.REGION_ABOVE


def test_siblings_on_one_pair_become_one_gate_without_filler_populations(source, mapping):
    defs, report = _defs(source, mapping)
    g = defs['T cells & Other']
    assert g['gate_type'] == '2dsep' and g['proposed_algorithm'] == 'mixture'
    assert {p: v['region'] for p, v in g['populations'].items()} == {
        'T cells': 'x+y-', 'Other': 'x-y+'}
    assert set(g['template']) == {'T cells', 'Other'}
    assert not any('bound' in n for notes in report['notes'].values() for n in notes)


def test_siblings_stay_apart_when_merging_is_off(source, mapping):
    defs, _ = _defs(source, mapping, merge_siblings=False)
    assert defs['T cells']['gate_type'] == '2dsep' and defs['Other']['gate_type'] == '2dsep'
    assert list(defs['T cells']['populations']) == ['T cells']


def test_ellipse_becomes_ten_vertex_polygon_of_equal_area(source, mapping):
    defs, report = _defs(source, mapping)
    singlets = defs['Single Cells']['template']['Single Cells']['boundary']
    assert len(singlets) == 5
    blob = defs['Blob']
    pts = np.asarray(blob['template']['Blob']['boundary'])
    assert len(pts) == fj.ELLIPSE_VERTICES + 1 and np.allclose(pts[0], pts[-1])
    assert any('10-vertex' in n for n in report['notes']['Blob'])
    # The traced ellipse reaches raw zero on both axes: double negative.
    assert blob['gate_type'] == '2dsep'
    assert blob['populations']['Blob']['region'] == 'x-y-'


def test_ellipse_polygon_area_matches_ellipse():
    k = fj._ellipse_scale(fj.ELLIPSE_VERTICES)
    n = fj.ELLIPSE_VERTICES
    area = 0.5 * n * k * k * np.sin(2 * np.pi / n)
    assert area == pytest.approx(np.pi)


def test_names_are_unique_and_slash_free(source, mapping):
    defs, _ = fj.build_gate_definitions(source, mapping, _exp(), existing_names={'Lymph'})
    names = [g['gate_name'] for g in defs]
    assert len(set(names)) == len(names)
    assert 'Lymph (All events)' in names and all('/' not in n for n in names)
    assert {g['name_group'] for g in defs if g['gate_name'].startswith('Lymph')} == {'Lymph'}
    for g in defs:
        assert all('/' not in p for p in g['populations'])


def _piece_for(dims, kind='rectangle', name='g'):
    exp = _exp()
    gate = fj.FJGate(kind=kind, channels=[d[0] for d in dims], bounds=[d[1:] for d in dims])
    pop = fj.FJPopulation(name=name, path=(name,), gate=gate)
    src = fj.FJSource(kind='hierarchy', name='t', roots=[pop])
    mapping = {d[0]: d[0] for d in dims}
    return fj.build_gate_definitions(src, mapping, exp, native_display=True)[0][0]


def test_single_gate_fluorescence_against_scatter_is_1dsep_on_fluorescence():
    g = _piece_for([('FSC-A', 0.2, 0.9), ('APC', 0.02, 0.3)])
    assert g['gate_type'] == '1dsep' and g['proposed_algorithm'] == 'mixture'
    assert (g['gate_marker_x'], g['gate_marker_y'], g['sep_axis']) == ('APC', 'FSC-A', 'y')
    assert g['populations']['g']['region'] == ag_core.REGION_BELOW
    assert ag_core.plot_channels(g) == ('FSC-A', 'APC')
    g = _piece_for([('APC', 0.4, 0.9), ('SSC-A', 0.0, 1.0)])
    assert (g['gate_marker_x'], g['gate_marker_y'], g['sep_axis']) == ('APC', 'SSC-A', 'x')
    assert g['populations']['g']['region'] == ag_core.REGION_ABOVE


def test_single_gate_on_two_fluorescence_channels_is_2dsep_by_sign():
    assert _piece_for([('PE', 0.3, 0.9), ('APC', 0.4, 0.9)])['populations']['g']['region'] == 'x+y+'
    assert _piece_for([('PE', 0.0, 0.1), ('APC', 0.0, 0.1)])['populations']['g']['region'] == 'x-y-'
    g = _piece_for([('PE', 0.3, 0.9), ('APC', 0.0, 0.3)])
    assert g['gate_type'] == '2dsep' and g['populations']['g']['region'] == 'x+y-'
    assert list(g['populations']) == ['g']


def test_histogram_gate_is_1dsep_tail_with_histogram_view():
    g = _piece_for([('FITC', 0.5, 0.9)])
    assert g['gate_type'] == '1dsep' and g['gate_marker_y'] is None
    assert g['proposed_algorithm'] == 'tail' and g['sep_axis'] == 'x'
    assert g['template']['g']['range'] == [pytest.approx(0.5), pytest.approx(0.9)]


# ---------------------------------------------------------------------------
# Imported boundaries reproduce the FlowJo gates on transformed data
# ---------------------------------------------------------------------------

def _events(n=20000, seed=3):
    rng = np.random.default_rng(seed)
    fsc = rng.uniform(1000, 260000, n)
    fsch = fsc * rng.uniform(0.7, 1.2, n)
    ssc = rng.uniform(1000, 200000, n)
    pe = np.where(rng.random(n) < 0.5, rng.normal(0, 300, n), rng.lognormal(8.5, 0.6, n))
    apc = np.where(rng.random(n) < 0.4, rng.normal(0, 250, n), rng.lognormal(8.2, 0.6, n))
    fitc = np.where(rng.random(n) < 0.7, rng.normal(0, 200, n), rng.lognormal(7.5, 0.5, n))
    bv = rng.normal(0, 300, n)
    return {'FSC-A': fsc, 'FSC-H': fsch, 'SSC-A': ssc, 'PE': pe, 'APC': apc, 'FITC': fitc,
            'BV421': bv}


def _raw_mask(gate, ev, mapping):
    from matplotlib.path import Path as MplPath
    if gate.kind == 'rectangle':
        m = np.ones(len(ev['FSC-A']), dtype=bool)
        for ch, (lo, hi) in zip(gate.channels, gate.bounds):
            x = ev[mapping[ch]]
            if lo is not None:
                m &= x >= lo
            if hi is not None:
                m &= x < hi
        return m
    pts = np.column_stack([ev[mapping[c]] for c in gate.channels])
    return MplPath(np.asarray(gate.vertices)).contains_points(pts)


def test_imported_boundaries_reproduce_raw_gating(source, mapping):
    gdefs, report = fj.build_gate_definitions(source, mapping, _exp())
    ev = _events()
    exp = _exp()
    chans = list(ev)
    data = np.column_stack([
        ag_core.make_transform(exp.transforms[c]).apply(ev[c]) for c in chans])
    ci = {c: i for i, c in enumerate(chans)}
    # Without display limits the shapes are pure geometry, as FlowKit applies them.
    result = ag_core.train_boundaries(gdefs, data, ci)
    counts = {r['key']: r['n_events'] for r in ag_core.population_table(gdefs, result)}

    pops = {p.key: p for p in source.iter_populations()}
    checked = ['Lymph/Single Cells'] + [
        f'Lymph/Single Cells/{q}' for q in ('Q1: CD4- , CD8+', 'Q2: CD4+ , CD8+',
                                           'Q3: CD4+ , CD8-', 'Q4: CD4- , CD8-',
                                           'T cells', 'Other')]
    checked.append('Lymph/Single Cells/Q3: CD4+ , CD8-/CD25+/hi')
    for key in checked:
        path_masks = np.ones(len(ev['FSC-A']), dtype=bool)
        parts = key.split('/')
        for i in range(1, len(parts) + 1):
            k = '/'.join(parts[:i])
            if k in pops:
                path_masks &= _raw_mask(pops[k].gate, ev, mapping)
        expected = int(path_masks.sum())
        got = counts[report['populations'][key]]
        assert abs(got - expected) <= max(2, 0.001 * expected), (key, got, expected)


def test_imported_algorithm_without_template_is_empty():
    g = {'gate_name': 'g', 'gate_type': '1dsep', 'gate_marker_x': 'A', 'gate_marker_y': None,
         'algorithm': ag_core.ALGORITHM_IMPORTED,
         'populations': ag_core.default_populations_for_type('1dsep')}
    assert ag_core.GateCalculator().calculate(np.zeros((50, 1)), g, {'A': 0}) == {}


def test_imported_shape_needs_its_view_channel():
    g = _piece_for([('FSC-A', 0.2, 0.9), ('APC', 0.02, 0.3)])
    assert ag_core.GateCalculator().calculate(np.zeros((50, 1)), g, {'APC': 0}) == {}
    assert ag_core.referenced_channels(g) == ['APC', 'FSC-A']


def test_remap_model_converts_shape_templates(source, mapping):
    gdefs, _ = fj.build_gate_definitions(source, mapping, _exp())
    exp = _exp()
    model = ag_core.new_model(gdefs, {}, channel_map={}, transforms=exp.transforms,
                              training_summary={}, metadata={})
    dst = dict(exp.transforms)
    dst['PE'] = dict(dst['PE'], logicle_w=1.0)
    out, messages = ag_core.remap_model(model, dst)
    before = {g['gate_name']: g for g in model['gate_definitions']}
    after = {g['gate_name']: g for g in out['gate_definitions']}
    quad = next(n for n, g in before.items() if 'Q3: CD4+ , CD8-' in g['populations'])
    x0 = np.asarray(before[quad]['template']['Q3: CD4+ , CD8-']['boundary'])[:, 0].min()
    x1 = np.asarray(after[quad]['template']['Q3: CD4+ , CD8-']['boundary'])[:, 0].min()
    raw0 = ag_core.make_transform(exp.transforms['PE']).inverse(np.array([x0]))[0]
    raw1 = ag_core.make_transform(dst['PE']).inverse(np.array([x1]))[0]
    assert x0 != pytest.approx(x1) and raw0 == pytest.approx(raw1, rel=1e-6)
    assert any('PE' in m for m in messages)


# ---------------------------------------------------------------------------
# The experiment's own hierarchy as an import source
# ---------------------------------------------------------------------------

class _Dim:
    def __init__(self, ident, lo=None, hi=None, ref=None):
        self.id, self.min, self.max = ident, lo, hi
        self.dimension_ref = ref or ident
        self.values = []


class _Quad:
    def __init__(self, ranges):
        self._divider_ranges = ranges

    def get_divider_range(self, div_id):
        return self._divider_ranges[div_id]


class _KitGate:
    def __init__(self, name, gate_type, dims, vertices=None, quadrants=None):
        self.gate_name, self.gate_type = name, gate_type
        self.dimensions, self.vertices, self.quadrants = dims, vertices or [], quadrants or {}

    def get_dimension_ids(self):
        return [d.id for d in self.dimensions]


class _KitStrategy:
    """Duck-typed FlowKit GatingStrategy: [(gate, path)] listed top-down."""

    def __init__(self, entries, quadrant_names=()):
        self._entries = entries
        self._quadrant_names = set(quadrant_names)
        self._gates = {g.gate_name: g for g, _ in entries}

    def get_gate_ids(self):
        return [(g.gate_name, path) for g, path in self._entries]

    def get_gate(self, name):
        return self._gates[name]

    def _get_gate_node(self, name, path):
        class Node:
            gate_type = 'Quadrant' if name in self._quadrant_names else self._gates[name].gate_type
        return Node()


def _kit_strategy():
    cells = _KitGate('Cells', 'RectangleGate', [_Dim('FSC-A', 0.2, 0.8), _Dim('SSC-A', 0.1, 0.7)])
    singlets = _KitGate('Singlets', 'PolygonGate', [_Dim('FSC-A'), _Dim('FSC-H')],
                        vertices=[(0.2, 0.1), (0.8, 0.6), (0.8, 0.9), (0.2, 0.3)])
    viable = _KitGate('viable', 'RectangleGate', [_Dim('FSC-A', 0.2, 0.9), _Dim('APC', 0.04, 0.3)])
    # CD3: BV421 positive, PE negative; CD19: reverse.
    cd3 = _KitGate('CD3', 'RectangleGate', [_Dim('BV421', 0.45, 0.9), _Dim('PE', 0.0, 0.2)])
    quad = _KitGate('CD4 CD8', 'QuadrantGate', [_Dim('dx', ref='PE'), _Dim('dy', ref='APC')], quadrants={
        'DN': _Quad({'dx': (None, 0.4), 'dy': (None, 0.35)}),
        'CD4': _Quad({'dx': (0.4, None), 'dy': (None, 0.35)}),
        'CD8': _Quad({'dx': (None, 0.4), 'dy': (0.35, None)}),
        'DP': _Quad({'dx': (0.4, None), 'dy': (0.35, None)}),
    })
    treg = _KitGate('Treg', 'RectangleGate', [_Dim('FITC', 0.5, 0.9), _Dim('BV421', 0.45, 0.9)])
    entries = [
        (cells, ('root',)),
        (singlets, ('root', 'Cells')),
        (viable, ('root', 'Cells', 'Singlets')),
        (cd3, ('root', 'Cells', 'Singlets', 'viable')),
        (quad, ('root', 'Cells', 'Singlets', 'viable', 'CD3')),
        (treg, ('root', 'Cells', 'Singlets', 'viable', 'CD3', 'CD4 CD8', 'CD4')),
    ]
    return _KitStrategy(entries)


def test_hierarchy_source_tree_and_names():
    src = fj.source_from_gating(_kit_strategy())
    keys = [p.key for p in src.iter_populations()]
    assert keys == ['Cells', 'Cells/Singlets', 'Cells/Singlets/viable',
                    'Cells/Singlets/viable/CD3', 'Cells/Singlets/viable/CD3/DN',
                    'Cells/Singlets/viable/CD3/CD4', 'Cells/Singlets/viable/CD3/CD4/Treg',
                    'Cells/Singlets/viable/CD3/CD8', 'Cells/Singlets/viable/CD3/DP']


def _kit_defs():
    exp = _exp()
    src = fj.source_from_gating(_kit_strategy())
    mapping = {p: p for p in fj.gated_parameters(src)}
    gdefs, report = fj.build_gate_definitions(src, mapping, exp, native_display=True,
                                              origin='Experiment hierarchy')
    return {g['gate_name']: g for g in gdefs}, gdefs, report


def test_hierarchy_scatter_rectangles_keep_all_sides_and_names():
    by, _, _ = _kit_defs()
    cells = by['Cells']
    assert cells['gate_type'] == 'free' and list(cells['populations']) == ['Cells']
    box = np.asarray(cells['template']['Cells']['boundary'])
    assert box[:, 0].min() == pytest.approx(0.2) and box[:, 0].max() == pytest.approx(0.8)
    assert box[:, 1].min() == pytest.approx(0.1) and box[:, 1].max() == pytest.approx(0.7)
    assert by['Singlets']['gate_type'] == 'singlets'


def test_hierarchy_children_sit_in_the_drawn_population():
    by, _, _ = _kit_defs()
    # viable is drawn on scatter x fluorescence: a 1dsep on APC, drawn on y.
    viable = by['viable']
    assert viable['gate_type'] == '1dsep' and viable['sep_axis'] == 'y'
    assert viable['populations']['viable']['region'] == ag_core.REGION_BELOW
    assert (by['CD3']['parent_gate'], by['CD3']['parent_popul']) == ('viable', 'viable')
    assert (by['Singlets']['parent_gate'], by['Cells']['parent_gate']) == (None, 'Singlets')
    assert (by['viable']['parent_gate'], by['viable']['parent_popul']) == ('Cells', 'Cells')


def test_hierarchy_threshold_side_follows_raw_zero():
    by, _, _ = _kit_defs()
    cd3 = by['CD3']
    assert cd3['gate_type'] == '2dsep'
    # BV421 above raw zero (+), PE interval containing it (-)
    assert cd3['populations']['CD3']['region'] == 'x+y-'
    box = np.asarray(cd3['template']['CD3']['boundary'])
    assert box[:, 0].min() == pytest.approx(0.45) and box[:, 1].max() == pytest.approx(0.2)


def test_hierarchy_quadrants_become_one_gate_and_children_attach_to_quadrant():
    by, gdefs, _ = _kit_defs()
    quad = next(g for g in gdefs if set(g['populations']) >= {'DN', 'CD4', 'CD8', 'DP'})
    assert quad['gate_type'] == '2dsep'
    assert (by['Treg']['parent_gate'], by['Treg']['parent_popul']) == (quad['gate_name'], 'CD4')


def test_hierarchy_gates_are_imported_and_reset_copy_kept():
    _, gdefs, _ = _kit_defs()
    for g in gdefs:
        assert g['algorithm'] == fj.ALGORITHM_IMPORTED
        assert g['template'] == g['template_original']
        assert g['source']['origin'] == 'Experiment hierarchy'


def test_hierarchy_unknown_channel_drops_branch():
    exp = _exp()
    src = fj.source_from_gating(_kit_strategy())
    mapping = {p: (None if p == 'PE' else p) for p in fj.gated_parameters(src)}
    gdefs, report = fj.build_gate_definitions(src, mapping, exp, native_display=True)
    names = {g['gate_name'] for g in gdefs}
    assert 'CD3' not in names and 'Treg' not in names
    assert any('CD3' in m for m in report['plan'].messages())


def test_hierarchy_gating_chain_runs_on_templates():
    from matplotlib.path import Path as MplPath
    by, gdefs, _ = _kit_defs()
    ref = {g['gate_name']: g['template'] for g in gdefs}
    channels = _exp().channels
    rng = np.random.default_rng(3)
    data = rng.uniform(0, 1, (5000, len(channels)))
    res = ag_core.run_gating(gdefs, data, {c: i for i, c in enumerate(channels)},
                             reference=ref, default_mode=ag_core.MODE_FIXED)
    single = MplPath([(0.2, 0.1), (0.8, 0.6), (0.8, 0.9), (0.2, 0.3), (0.2, 0.1)]).contains_points(
        data[:, [1, 2]])
    box = ((data[:, 1] >= 0.2) & (data[:, 1] <= 0.8) & (data[:, 3] >= 0.1) & (data[:, 3] <= 0.7))
    assert (res.population_mask('Singlets', 'Singlets') == single).all()
    assert (res.population_mask('Cells', 'Cells') == (single & box)).all()
    viable = res.population_mask('viable', 'viable')
    expect = single & box & (data[:, 1] >= 0.2) & (data[:, 1] <= 0.9) \
        & (data[:, 6] >= 0.04) & (data[:, 6] <= 0.3)
    assert (viable == expect).all()
    assert res.population_mask('CD3', 'CD3').sum() <= viable.sum()


def test_workspace_gates_record_counts_for_review():
    ws = fj.parse_workspace(ET.fromstring(_workspace_xml()))
    src = next(s for s in ws.sources if s.kind == 'sample')
    mapping = {p: {'FSC-A': 'FSC-A', 'SSC-A': 'SSC-A'}.get(p) for p in fj.gated_parameters(src)}
    gdefs, _ = fj.build_gate_definitions(src, mapping, _exp())
    assert gdefs and gdefs[0]['source']['sample'] == 's1.fcs'
    assert gdefs[0]['source']['parent_count'] == 1000


def test_hierarchy_ellipsoid_is_traced_as_polygon():
    class Ellipsoid(_KitGate):
        coordinates = np.array([0.5, 0.5])
        covariance_matrix = np.array([[0.01, 0.0], [0.0, 0.04]])
        distance_square = 1.0

    gate = Ellipsoid('E', 'EllipsoidGate', [_Dim('BV421'), _Dim('PE')])
    src = fj.source_from_gating(_KitStrategy([(gate, ('root',))]))
    verts = np.asarray(src.roots[0].gate.vertices)
    assert src.roots[0].gate.kind == 'polygon' and len(verts) == fj.ELLIPSE_VERTICES
    assert np.ptp(verts[:, 0]) == pytest.approx(0.2, abs=0.01)
    assert np.ptp(verts[:, 1]) == pytest.approx(0.4, abs=0.01)


# ---------------------------------------------------------------------------
# Optional: a real workspace
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.environ.get('HC_FLOWJO_TEST_WSP'),
                    reason='set HC_FLOWJO_TEST_WSP to a .wsp or .wspt file')
def test_real_workspace_parses():
    ws = fj.read_workspace(os.environ['HC_FLOWJO_TEST_WSP'])
    src = ws.default_source()
    assert src is not None and src.n_populations > 0
    assert fj.gated_parameters(src)


# ---------------------------------------------------------------------------
# Names of same-named gates, the histogram rule, scatter and singlet gates
# ---------------------------------------------------------------------------

def _pop(name, path, gate, count, children=()):
    p = fj.FJPopulation(name=name, path=path + (name,), gate=gate, count=count)
    p.children = [c for c in children]
    return p


def _hist(channel, lo):
    return fj.FJGate(kind='rectangle', channels=[channel], bounds=[(lo, None)])


def _named_source():
    lymph = ('Lymph',)
    cd4 = _pop('CD4', lymph, _hist('Comp-PE-A', 1000), 1000,
               [_pop('TIGIT+', lymph + ('CD4',), _hist('Comp-FITC-A', 500), 300)])
    cd8 = _pop('CD8', lymph, _hist('Comp-APC-A', 800), 1000,
               [_pop('TIGIT+', lymph + ('CD8',), _hist('Comp-FITC-A', 500), 50)])
    ssc = _pop('SSC singlets', lymph, fj.FJGate(kind='polygon', channels=['SSC-A', 'SSC-H'],
                                                vertices=SINGLET_POLY), 900)
    root = _pop('Lymph', (), fj.FJGate(kind='rectangle', channels=['FSC-A', 'SSC-A'],
                                       bounds=[(50000, 220000), (10000, 150000)]), 5000,
                [cd4, cd8, ssc])
    return fj.FJSource(kind='group', name='g', roots=[root], count=10000)


def _named_exp():
    exp = _exp()
    exp.channels = exp.channels + ['SSC-H']
    exp.scatter = exp.scatter + ['SSC-H']
    exp.transforms['SSC-H'] = dict(exp.transforms['SSC-A'])
    return exp


def test_same_named_gates_are_named_by_their_parent_and_grouped():
    mapping = {'FSC-A': 'FSC-A', 'SSC-A': 'SSC-A', 'SSC-H': 'SSC-H', 'Comp-PE-A': 'PE',
               'Comp-APC-A': 'APC', 'Comp-FITC-A': 'FITC'}
    defs, report = fj.build_gate_definitions(_named_source(), mapping, _named_exp())
    by = {g['gate_name']: g for g in defs}
    assert 'TIGIT+ (CD4)' in by and 'TIGIT+ (CD8)' in by and 'TIGIT+' not in by
    assert by['TIGIT+ (CD4)']['name_group'] == by['TIGIT+ (CD8)']['name_group'] == 'TIGIT+'
    assert by['TIGIT+ (CD4)']['parent_gate'] == 'CD4' and by['CD4']['name_group'] == 'CD4'
    assert set(report['notes']) == set(by)


def test_histogram_rule_uses_the_recorded_frequency():
    mapping = {'FSC-A': 'FSC-A', 'SSC-A': 'SSC-A', 'SSC-H': 'SSC-H', 'Comp-PE-A': 'PE',
               'Comp-APC-A': 'APC', 'Comp-FITC-A': 'FITC'}
    by = {g['gate_name']: g for g in fj.build_gate_definitions(_named_source(), mapping,
                                                              _named_exp())[0]}
    assert by['TIGIT+ (CD4)']['proposed_algorithm'] == 'mixture'     # 30% of parent
    assert by['TIGIT+ (CD8)']['proposed_algorithm'] == 'tail'        # 5% of parent
    assert by['TIGIT+ (CD8)']['proposal_rule'] == 'histogram'
    assert fj.histogram_algorithm(None) == 'tail' and fj.histogram_algorithm(0.26) == 'mixture'


def test_scatter_shapes_become_cell_gates_and_side_scatter_pairs_singlets():
    mapping = {'FSC-A': 'FSC-A', 'SSC-A': 'SSC-A', 'SSC-H': 'SSC-H', 'Comp-PE-A': 'PE',
               'Comp-APC-A': 'APC', 'Comp-FITC-A': 'FITC'}
    by = {g['gate_name']: g for g in fj.build_gate_definitions(_named_source(), mapping,
                                                              _named_exp())[0]}
    assert by['Lymph']['gate_type'] == 'free' and by['Lymph']['proposed_algorithm'] == 'cells'
    assert by['Lymph']['gate_param']['debris_axis'] == 'x'
    assert by['SSC singlets']['gate_type'] == 'singlets'
    assert by['SSC singlets']['parent_gate'] == 'Lymph'


def test_final_names_fall_back_to_more_ancestors_then_numbers():
    bases = {'a': 'X', 'b': 'X', 'c': 'X', 'd': 'Y'}
    paths = {'a': ('P', 'T'), 'b': ('Q', 'T'), 'c': ('Q', 'T'), 'd': ()}
    out = fj._final_names(bases, paths, taken={'Y'})
    assert out['d'] == 'Y (All events)'
    assert out['a'] == 'X (P › T)' and out['b'] == 'X (Q › T)' and out['c'] == 'X (Q › T) 2'


def test_upgrade_gives_older_imports_the_current_rules():
    gates = [
        {'gate_name': 'TIGIT+', 'gate_type': '1dsep', 'gate_marker_x': 'FITC', 'gate_marker_y': None,
         'algorithm': 'imported', 'proposed_algorithm': 'tail', 'gate_param': {},
         'populations': {'TIGIT+': {'region': 'above'}},
         'source': {'parent_count': 100, 'counts': {'TIGIT+': 40}}},
        {'gate_name': 'TIGIT+ (2)', 'gate_type': '1dsep', 'gate_marker_x': 'FITC',
         'gate_marker_y': None, 'algorithm': 'tail', 'proposed_algorithm': 'tail',
         'gate_param': {}, 'populations': {'TIGIT+': {'region': 'above'}},
         'source': {'origin': 'FlowJo'}},
        {'gate_name': 'Cells', 'gate_type': 'free', 'gate_marker_x': 'FSC-A',
         'gate_marker_y': 'SSC-A', 'algorithm': 'imported', 'proposed_algorithm': None,
         'gate_param': {}, 'populations': {'Cells': {'region': 'inside'}},
         'source': {'origin': 'FlowJo'}},
    ]
    assert fj.upgrade_gate_definitions(gates) == 3
    assert gates[0]['name_group'] == 'TIGIT+' and gates[0]['proposed_algorithm'] == 'mixture'
    assert gates[1]['name_group'] == 'TIGIT+' and 'proposal_rule' not in gates[1]
    assert gates[2]['proposed_algorithm'] == 'cells' and gates[2]['gate_param']['debris_axis'] == 'x'
    assert fj.upgrade_gate_definitions(gates) == 0


# ---------------------------------------------------------------------------
# Quadrant layout of sibling shapes on two fluorescence parameters
# ---------------------------------------------------------------------------

def _siblings(shapes, x='PE', y='APC'):
    """One gate from sibling polygons {name: [(x, y), ...]} in display units."""
    pops = [fj.FJPopulation(name=n, path=(n,), gate=fj.FJGate(kind='polygon', channels=[x, y],
                                                             vertices=list(v)))
            for n, v in shapes.items()]
    src = fj.FJSource(kind='hierarchy', name='t', roots=pops)
    defs = fj.build_gate_definitions(src, {x: x, y: y}, _exp(), native_display=True)[0]
    assert len(defs) == 1
    return defs[0]


def _box(x0, x1, y0, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def test_layout_joins_the_quadrants_a_shape_spans():
    g = _siblings({'Main': _box(0.05, 0.4, 0.1, 0.9), 'Low': _box(0.5, 0.9, 0.1, 0.4),
                   'High': _box(0.5, 0.9, 0.5, 0.9)})
    assert g['gate_type'] == '2dsep'
    assert {p: v['region'] for p, v in g['populations'].items()} == {
        'Main': 'x-y-|x-y+', 'Low': 'x+y-', 'High': 'x+y+'}


def test_layout_gives_an_l_shape_three_quadrants():
    l_shape = [(0.05, 0.3), (0.3, 0.3), (0.3, 0.05), (0.9, 0.05), (0.9, 0.9), (0.05, 0.9)]
    g = _siblings({'NK': l_shape, 'non-NK': _box(0.05, 0.28, 0.05, 0.28)})
    assert g['gate_type'] == '2dsep'
    assert {p: v['region'] for p, v in g['populations'].items()} == {
        'NK': 'x+y-|x-y+|x+y+', 'non-NK': 'x-y-'}


def test_layout_drops_a_parameter_that_separates_nothing():
    g = _siblings({'Pos': _box(0.35, 0.8, 0.25, 0.7), 'Neg': _box(0.05, 0.3, 0.05, 0.7)})
    assert g['gate_type'] == '1dsep'
    assert (g['gate_marker_x'], g['gate_marker_y'], g['sep_axis']) == ('PE', 'APC', 'x')
    assert {p: v['region'] for p, v in g['populations'].items()} == {
        'Pos': ag_core.REGION_ABOVE, 'Neg': ag_core.REGION_BELOW}


def test_layout_keeps_opposite_corners_two_dimensional():
    g = _siblings({'CD4': _box(0.05, 0.3, 0.5, 0.9), 'CD8': _box(0.5, 0.9, 0.05, 0.3)})
    assert g['gate_type'] == '2dsep'
    assert {p: v['region'] for p, v in g['populations'].items()} == {'CD4': 'x-y+', 'CD8': 'x+y-'}


def test_threshold_gates_follow_the_fraction_rule():
    g = _siblings({'Main': _box(0.05, 0.4, 0.1, 0.9), 'Low': _box(0.5, 0.9, 0.1, 0.4),
                   'High': _box(0.5, 0.9, 0.5, 0.9)})
    assert g['proposal_rule'] == 'fraction' and g['proposed_algorithm'] == 'mixture'
    hist = _piece_for([('FITC', 0.5, 0.9)])
    assert hist['proposal_rule'] == 'histogram'
    old = dict(g)
    old.pop('proposal_rule')
    assert fj.upgrade_gate_definitions([old]) == 1 and old['proposal_rule'] == 'fraction'

"""Which defined cell types a panel can tell apart (cell_type_panel), and
Cluster ID suggesting a group when the panel cannot separate its members."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from honeychrome.controller_components import cell_type_panel as ctp

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

DEFS = [
    ('T cell', ['CD3', 'CD90.2'], ['CD19']),
    ('ab T cell', ['CD3', 'CD90.2', 'TCRbeta'], ['CD19', 'TCRgd']),
    ('gd T cell', ['CD3', 'TCRgd'], ['CD19', 'TCRbeta']),
    ('CD4 T cell', ['CD3', 'CD4'], ['CD8', 'CD19']),
    ('CD8 T cell', ['CD3', 'CD8'], ['CD4', 'CD19']),
    ('CD4 Treg', ['CD3', 'CD4', 'FoxP3'], ['CD8', 'CD19']),
    ('Th1', ['CD3', 'CD4', 'CD44', 'T-bet'], ['CD8', 'FoxP3']),
    ('Th2', ['CD3', 'CD4', 'CD44', 'GATA-3'], ['CD8', 'FoxP3']),
    ('B cell', ['CD19'], ['CD3']),
    ('Astrocyte', ['GFAP'], []),
]
PANEL = ['CD3', 'CD4', 'CD8', 'CD19', 'FoxP3', 'CD44']


def by_name(groups, name):
    return groups.groups[groups.group_of(name)]


def test_definitions_that_read_the_same_on_the_panel_form_one_group():
    groups = ctp.panel_groups(DEFS, PANEL)
    t = by_name(groups, 'T cell')
    assert t.names == ('T cell', 'ab T cell', 'gd T cell')
    assert (t.positive, t.negative) == (('CD3',), ('CD19',))
    assert t.separated_by == ('CD90.2', 'TCRbeta', 'TCRgd')
    assert t.label == 'CD3+ CD19-: T cell / ab T cell / gd T cell'
    th = by_name(groups, 'Th1')
    assert th.names == ('Th1', 'Th2') and th.separated_by == ('GATA-3', 'T-bet')
    assert by_name(groups, 'CD4 Treg').label == 'CD4 Treg'
    assert groups.no_marker == ('Astrocyte',)
    assert [g.names[0] for g in groups.groups] == ['T cell', 'CD4 T cell', 'CD8 T cell', 'CD4 Treg', 'Th1', 'B cell']


def test_nesting_from_shared_and_conflicting_signs():
    groups = ctp.panel_groups(DEFS, PANEL)
    g = {name: groups.group_of(name) for name in ('T cell', 'CD4 T cell', 'CD8 T cell', 'CD4 Treg', 'B cell')}
    assert groups.contains[g['T cell'], g['CD4 T cell']]
    assert groups.contains[g['CD4 T cell'], g['CD4 Treg']]
    assert not groups.contains[g['CD4 T cell'], g['CD8 T cell']]
    assert not groups.contains[g['T cell'], g['B cell']]
    assert groups.parent(g['CD4 Treg']) == g['CD4 T cell']
    assert groups.parent(g['T cell']) is None


def test_a_root_rules_out_only_definitions_that_contradict_it():
    groups = ctp.panel_groups(DEFS, PANEL, root={'CD3': 1})
    assert groups.outside_root == ('B cell',)
    t = by_name(groups, 'T cell')
    assert (t.positive, t.negative) == (('CD3',), ('CD19',))
    assert by_name(groups, 'CD4 T cell').positive == ('CD3', 'CD4')
    assert groups.no_marker == ('Astrocyte',)


def test_definitions_silent_on_a_root_marker_are_kept():
    groups = ctp.panel_groups(DEFS + [('Naive CD4 T cell', ['CD3', 'CD4'], ['CD44', 'CD8'])],
                              ['CD3', 'CD4', 'CD8', 'CD19', 'FoxP3'], root={'CD44': 1})
    assert groups.outside_root == ('Naive CD4 T cell',)
    cd4 = by_name(groups, 'CD4 T cell')
    assert cd4.label == 'CD4 T cell' and 'CD44' not in cd4.positive
    assert by_name(groups, 'Th1').positive == ('CD3', 'CD4', 'CD44')


def test_a_negative_root_sign_rules_out_definitions_that_need_the_marker():
    groups = ctp.panel_groups(DEFS, PANEL, root={'CD19': -1})
    assert groups.outside_root == ('B cell',)
    assert groups.group_of('Th1') is not None and groups.group_of('Astrocyte') is None
    t = by_name(groups, 'T cell')
    assert (t.positive, t.negative) == (('CD3',), ('CD19',))


def test_an_empty_database_or_panel_gives_no_groups():
    assert ctp.panel_groups([], PANEL).groups == []
    groups = ctp.panel_groups(DEFS, [])
    assert groups.groups == [] and len(groups.no_marker) == len(DEFS)
    assert groups.contains.shape == (0, 0)


def test_the_database_reads_with_unique_names_per_species():
    import csv
    with open(ctp.DATABASE_PATH, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    keys = [(r['cell_type'], r['species']) for r in rows]
    assert len(keys) == len(set(keys))
    assert all(r['positive_markers'] or r['negative_markers'] for r in rows)


def test_cluster_id_suggests_the_group_when_the_panel_cannot_separate_its_members():
    drc_cluster_id = pytest.importorskip('drc_cluster_id')
    db = [{'cell_type': n, 'species': 'mouse', 'positive': p, 'negative': q} for n, p, q in DEFS]
    channels = {'BUV395': 'CD4', 'BUV805': 'CD8', 'SBY720': 'CD3', 'APC': 'CD19', 'PE': 'FoxP3', 'APC-Fire 750': 'CD44'}
    scores = pd.DataFrame(
        [[2.0, -2.0, 2.0, -2.0, -2.0, 2.0],     # CD4+ CD44+ FoxP3-: Th1 or Th2
         [2.0, -2.0, 2.0, -2.0, 2.0, -1.0],     # CD4+ FoxP3+: Treg
         [np.nan] * 6],
        index=[0, 1, 2], columns=list(channels))
    out = drc_cluster_id.score_cell_types(scores, channels, db)
    assert list(out.columns) == ['suggested_type', 'score', 'low_confidence', 'separated_by']
    assert out.loc[0, 'suggested_type'] == 'CD3+ CD4+ CD44+ CD8- FoxP3-: Th1 / Th2'
    assert out.loc[0, 'separated_by'] == 'GATA-3, T-bet'
    assert out.loc[1, 'suggested_type'] == 'CD4 Treg' and out.loc[1, 'separated_by'] == ''
    assert out.loc[2, 'suggested_type'] == '' and np.isnan(out.loc[2, 'score'])

"""Groupings suggested from sample names (differential_stats) and the shared
automatic choice (view_components.suggest_groupings_dialog)."""

import pytest

from honeychrome.controller_components.differential_stats import suggest_groupings_from_names

TISSUES = ('Spleen', 'LN', 'Pancreas', 'Blood')
MICE = ('M1', 'M2', 'M3', 'M4', 'M5', 'M6')


def _flowcode_style_names():
    # The lab's convention: tissue is the 2nd field, the mouse the 7th, split on spaces and underscores.
    return [f'Raw/CD8_{t}_x1 x2_x3_x4_{m}_s9_Unmixed.fcs' for t in TISSUES for m in MICE]


def test_tissue_is_the_group_and_mouse_the_pairing():
    keys = _flowcode_style_names()
    result = suggest_groupings_from_names(keys)
    assert result['irregular_samples'] == []
    roles = {s['position']: s['role_guess'] for s in result['suggestions']}
    assert roles == {1: 'group', 6: 'pairing'}
    group = result['suggestions'][0]
    assert group['role_guess'] == 'group' and group['n_distinct'] == 4
    assert group['values'][keys[0]] == 'Spleen' and group['values'][keys[-1]] == 'Blood'
    pairing = result['suggestions'][1]
    assert pairing['role_guess'] == 'pairing' and pairing['n_distinct'] == 6


def test_names_with_a_different_field_count_are_left_out():
    keys = _flowcode_style_names() + ['Raw/odd name.fcs']
    result = suggest_groupings_from_names(keys)
    assert result['irregular_samples'] == ['Raw/odd name.fcs']
    assert all('Raw/odd name.fcs' not in s['values'] for s in result['suggestions'])


def test_mouse_nested_in_group_is_a_covariate_not_a_pairing():
    # each mouse appears in one group only: e.g. COPD vs Healthy, one tissue per mouse
    keys = [f'{status}_{m}.fcs' for status, mice in (('COPD', 'ABCDEF'), ('Healthy', 'GHIJKL'))
            for m in mice for _rep in (0,)]
    keys = [k.replace('.fcs', f'_{t}.fcs') for k in keys for t in ('Lung', 'Blood')]
    result = suggest_groupings_from_names(keys)
    by_pos = {s['position']: s for s in result['suggestions']}
    assert by_pos[0]['role_guess'] == 'group'
    assert by_pos[1]['role_guess'] == 'covariate'


def test_automatic_choice_takes_the_group_and_the_pairing():
    sgd = pytest.importorskip('honeychrome.view_components.suggest_groupings_dialog')
    keys = _flowcode_style_names()
    chosen = sgd.automatic_groupings(suggest_groupings_from_names(keys))
    assert [c['role'] for c in chosen] == ['group', 'pairing']
    assert chosen[0]['name'] == 'Group' and chosen[1]['name'] == 'field_7'
    assert set(chosen[0]['values'].values()) == set(TISSUES)
    text = sgd.describe_groupings(chosen)
    assert text.startswith('Groups suggested from sample names: Group = name field 2 of 9')
    assert "paired by 'field_7'" in text and '6 levels' in text


def test_automatic_choice_is_empty_without_a_group():
    sgd = pytest.importorskip('honeychrome.view_components.suggest_groupings_dialog')
    assert sgd.automatic_groupings(suggest_groupings_from_names(['a_1.fcs', 'b_2.fcs'])) == []

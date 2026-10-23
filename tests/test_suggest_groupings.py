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


def test_cohort_is_not_a_pairing_and_dotted_mouse_tags_survive():
    """Tissue x mouse names with a cohort tag, a dotted mouse tag and a folder
    suffix whose field count varies: every name keeps its leading fields, the
    per-mouse field survives the dot, and the cohort, which repeats within a
    tissue, is never a pairing variable."""
    folders = {'Spleen': 'Spleen, axLN, mLN', 'Liver': 'PP, Liver, SI IEL', 'PP': 'PP, Liver, SI IEL'}
    mice = [('LSRB31.3a', '001'), ('LSRB31.3b', '002'), ('LSRB36.2d', '003'), ('LSRB36.2e', '004')]
    keys = [f'Raw/{folders[t]}/A{i} {t}_Library1_{tag}_{n}_{folders[t]}.fcs'
            for t in folders for i, (tag, n) in enumerate(mice)]
    result = suggest_groupings_from_names(keys, {k: k.split('/')[-1][:-4] for k in keys})
    assert result['irregular_samples'] == []
    by_examples = {tuple(s['examples'][:2]): s for s in result['suggestions']}
    assert len(by_examples[('001', '002')]['values']) == len(keys)
    assert len(by_examples[('Liver', 'PP')]['values']) == len(keys)
    assert all(s['role_guess'] != 'pairing' for s in result['suggestions']
               if s['examples'][:2] == ['LSRB31', 'LSRB36'])
    # with the tissue as the smallest field, the mouse field pairs it
    keys = [f'Raw/A{i} {t}_Library1_{n}.fcs' for t in folders for i, (_tag, n) in enumerate(mice)]
    roles = {tuple(s['examples'][:2]): s['role_guess'] for s in suggest_groupings_from_names(keys)['suggestions']}
    assert roles[('Liver', 'PP')] == 'group' and roles[('001', '002')] == 'pairing'

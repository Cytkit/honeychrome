import numpy as np
import pandas as pd
import pytest

from honeychrome.controller_components import covariate_check as cc
from honeychrome.controller_components import differential_stats as ds


def test_parse_number_reads_the_usual_spellings():
    assert cc.parse_number('98%') == 98
    assert cc.parse_number('768,500') == 768500
    assert cc.parse_number('1.116x10^6 ') == pytest.approx(1.116e6)
    assert cc.parse_number('5.4 X 10^6') == pytest.approx(5.4e6)
    assert cc.parse_number('2.95x10^6 (note: stored after RBC depletion)') == pytest.approx(2.95e6)
    assert cc.parse_number('33+1') == pytest.approx(33.1429)
    assert cc.parse_number('7.958,833') is None
    assert cc.parse_number('Y') is None
    assert cc.format_number(1116000.0) == '1116000'


SAMPLES = [f"s{i}" for i in range(8)]
GROUPS = {s: ('A' if i < 4 else 'B') for i, s in enumerate(SAMPLES)}


def _issues(covs, numeric=(), comparisons=(('A', 'B'),)):
    return cc.check_covariates(covs, SAMPLES, GROUPS, list(comparisons), list(numeric))


def test_missing_unknown_and_blank_are_separate_items_with_fixes():
    covs = {'vacc': dict(zip(SAMPLES, ['Y', 'N', 'Unknown', 'Uknown', 'na', '-', '', 'Y']))}
    kinds = {i.kind: i for i in _issues(covs) if i.column == 'vacc'}
    assert set(kinds) >= {'blank', 'missing', 'unknown'}
    fixes = [(kinds['blank'], cc.FILL_NA), (kinds['missing'], cc.TO_NA),
             (kinds['unknown'], cc.TO_NA)]
    out, _numeric, removed = cc.apply_fixes(covs, [], SAMPLES, fixes)
    assert removed == []
    assert [out['vacc'][s] for s in SAMPLES] == ['Y', 'N', 'NA', 'NA', 'NA', 'NA', 'NA', 'Y']


def test_similar_categories_merge_into_the_common_spelling():
    covs = {'sex': dict(zip(SAMPLES, ['Female', 'female', 'Female', 'Male', 'Male', 'Male',
                                      'Female', 'Male']))}
    dup = [i for i in _issues(covs) if i.kind == 'duplicate']
    assert len(dup) == 1 and dup[0].payload['into'] == 'Female'
    out, _n, _r = cc.apply_fixes(covs, [], SAMPLES, [(dup[0], cc.MERGE)])
    assert set(out['sex'].values()) == {'Female', 'Male'}


def test_numbers_as_text_convert_and_tick_numeric_and_unreadable_cells_stay():
    covs = {'via': dict(zip(SAMPLES, ['98%', '95%', '100%', '87%', '90%', 'na', '93%', 'lots']))}
    conv = [i for i in _issues(covs) if i.kind == 'convert']
    assert len(conv) == 1 and conv[0].actions[0] == cc.CONVERT and "'lots'" in conv[0].summary
    out, numeric, _r = cc.apply_fixes(covs, [], SAMPLES, [(conv[0], cc.CONVERT)])
    assert numeric == ['via'] and out['via']['s0'] == '98' and out['via']['s7'] == 'lots'
    unreadable = [i for i in _issues(out, numeric) if i.kind == 'unreadable']
    assert len(unreadable) == 1


def test_confounded_and_left_out_are_reported_per_comparison():
    covs = {'site': {s: ('x' if GROUPS[s] == 'A' else 'y') for s in SAMPLES},
            'age': {s: ('NA' if GROUPS[s] == 'B' else str(20 + i)) for i, s in enumerate(SAMPLES)},
            'donor': {s: f"d{i}" for i, s in enumerate(SAMPLES)}}
    issues = _issues(covs)
    conf = [i for i in issues if i.kind == 'confounded']
    assert any("'site'" in i.summary for i in conf)
    assert any("'donor'" in i.summary and 'its own value' in i.summary for i in conf)
    left = [i for i in issues if i.kind == 'left_out']
    assert len(left) == 1 and "'age'" in left[0].summary


def test_remove_column():
    covs = {'x': dict(zip(SAMPLES, [''] * 8))}
    blank = [i for i in _issues(covs) if i.kind == 'blank'][0]
    out, _n, removed = cc.apply_fixes(covs, [], SAMPLES, [(blank, cc.REMOVE)])
    assert removed == ['x'] and 'x' not in out


def _frame(values, index):
    return pd.DataFrame(values, index=index)


def test_comparison_parts_drop_a_covariate_empty_in_a_group_and_leave_out_samples():
    index = [f"s{i}" for i in range(9)]
    groups = ['A'] * 3 + ['B'] * 3 + ['C'] * 3
    cov = _frame({'vacc': ['Y', 'N', 'Y', 'N', None, 'Y', None, None, None],
                  'sex': ['F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F']}, index)
    parts = ds.comparison_parts(groups, [('A', 'B'), ('A', 'C')], cov, min_per_group=2)
    ab, ac = parts
    assert list(ab['covariates'].columns) == ['vacc', 'sex'] and ab['mask'].sum() == 5
    assert any('left out' in n for n in ab['notes'])
    assert list(ac['covariates'].columns) == ['sex'] and ac['mask'].sum() == 6
    assert any("'vacc' was left out" in n for n in ac['notes'])


def test_run_moderated_with_missing_covariates_fits_each_comparison_and_notes_it():
    rng = np.random.default_rng(0)
    index = [f"s{i}" for i in range(12)]
    groups = ['A'] * 4 + ['B'] * 4 + ['C'] * 4
    data = pd.DataFrame(rng.normal(size=(12, 6)), index=index,
                        columns=[f"f{j}" for j in range(6)])
    data.loc[index[4:8], 'f0'] += 3
    raw = pd.DataFrame({'vacc': ['Y', 'N', 'Y', 'N', 'N', 'Y', 'NA', 'Y',
                                 'na', '', 'NA', '-']}, index=index)
    cov = ds.covariate_frame(raw, index, ['vacc'], numeric=[], allow_missing=True)
    assert cov['vacc'].isna().sum() == 5
    out = ds.run_moderated(data, groups, [('A', 'B'), ('A', 'C')], 'reference', 0.05, 0.0,
                           covariates=cov)
    assert set(out['comparison']) == {'B vs A', 'C vs A'}
    notes = out.attrs['notes']
    assert any("C vs A: 'vacc' was left out" in n for n in notes)
    assert any('B vs A: 1 sample(s)' in n for n in notes)
    with pytest.raises(RuntimeError):
        ds.covariate_frame(raw, index, ['vacc'], numeric=[])

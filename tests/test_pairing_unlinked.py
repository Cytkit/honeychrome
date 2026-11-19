import numpy as np
import pandas as pd

from honeychrome.controller_components import differential_stats as ds


def _names():
    keys, groups = [], {}
    for i in range(1, 9):
        for kind, group in (('FB', 'Foetal'), ('MB', 'Maternal')):
            k = f"Raw/A{i} POMS_{i:03d}_{kind}_B_Plate {i % 3 + 1}.fcs"
            keys.append(k)
            groups[k] = group
    for i in range(1, 5):
        k = f"Raw/C{i} CB_{i:03d}_Plate 4.fcs"
        keys.append(k)
        groups[k] = 'Cord Blood'
    for name in ('C9 IC_BuffyCoat_Donor_E_Plate 4', 'C10 IC_PBMC_Donor_1_Plate 4'):
        keys.append(f"Raw/{name}.fcs")
    return keys, groups


def test_suggestions_judge_pairing_against_the_current_groups():
    keys, groups = _names()
    result = ds.suggest_groupings_from_names(keys, current_groups=groups)
    assert result['judged_against_groups']
    roles = {s['position']: s['role_guess'] for s in result['suggestions']}
    assert 'group' not in roles.values()
    assert roles[2] == 'pairing'
    donor = next(s for s in result['suggestions'] if s['position'] == 2)
    assert donor['values']['Raw/A3 POMS_003_FB_B_Plate 1.fcs'] == '003'


def test_unlinked_pairing_is_dropped_for_that_comparison_only():
    rng = np.random.default_rng(1)
    groups = ['Foetal'] * 6 + ['Maternal'] * 6 + ['Cord'] * 4
    index = [f"s{i}" for i in range(16)]
    pairing = [f"d{i}" for i in range(6)] * 2 + [f"__unpaired_s{i}" for i in range(12, 16)]
    data = pd.DataFrame(rng.normal(size=(16, 5)), index=index, columns=list('abcde'))
    out = ds.run_moderated(data, groups, [('Foetal', 'Maternal'), ('Foetal', 'Cord')],
                           'reference', 0.05, 0.0, pairing_vec=pairing)
    assert set(out['comparison']) == {'Maternal vs Foetal', 'Cord vs Foetal'}
    notes = out.attrs['notes']
    assert any('Cord vs Foetal was fitted without pairing' in n for n in notes)
    assert not any('Maternal vs Foetal was fitted without pairing' in n for n in notes)
    paired = ds.run_moderated(data.iloc[:12], groups[:12], [('Foetal', 'Maternal')],
                              'reference', 0.05, 0.0, pairing_vec=pairing[:12])
    mine = out[out['comparison'] == 'Maternal vs Foetal'].reset_index(drop=True)
    np.testing.assert_allclose(mine['logFC'], paired['logFC'])
    counts = pd.DataFrame(rng.poisson(50, size=(16, 5)), index=index, columns=list('abcde'))
    glm = ds.run_glm_counts(counts, groups, [('Foetal', 'Maternal'), ('Foetal', 'Cord')],
                            'reference', 0.05, 0.0, pairing_vec=pairing, method='ql')
    assert set(glm['comparison']) == {'Maternal vs Foetal', 'Cord vs Foetal'}


def test_covariate_shared_within_pairs_keeps_the_pairing():
    rng = np.random.default_rng(3)
    groups = ['Foetal'] * 6 + ['Maternal'] * 6
    index = [f"s{i}" for i in range(12)]
    pairing = [f"d{i}" for i in range(6)] * 2
    age = list(rng.uniform(18, 37, size=6)) * 2
    data = pd.DataFrame(rng.normal(size=(12, 5)), index=index, columns=list('abcde'))
    cov = pd.DataFrame({'age': age}, index=index)
    assert ds.pairing_absorbed(cov, pairing) == ['age']
    out = ds.run_moderated(data, groups, [('Foetal', 'Maternal')], 'reference', 0.05, 0.0,
                           pairing_vec=pairing, covariates=cov)
    notes = out.attrs['notes']
    assert not any('without pairing' in n for n in notes)
    assert any("'age' is the same within every pairing level" in n for n in notes)
    paired = ds.run_moderated(data, groups, [('Foetal', 'Maternal')], 'reference', 0.05, 0.0,
                              pairing_vec=pairing)
    np.testing.assert_allclose(out['logFC'], paired['logFC'])
    X, names, _cols = ds.build_design(groups, 'Foetal', pairing, cov)
    assert 'age' not in names

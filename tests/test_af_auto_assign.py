"""Tests for controller_components/af_auto_assign.py (scatter-based AF profile suggestions)."""

import numpy as np
import pytest

from honeychrome.controller_components import af_auto_assign as aa
from af_synthetic import RAW_PNN, MAGNITUDE_CEILING, write_raw_fcs

# log10 (FSC, SSC) centre and SD of each synthetic population
POPULATIONS = {
    'lymph': ((4.70, 4.30), (0.05, 0.08)),
    'mono': ((5.00, 4.75), (0.06, 0.10)),
    'dead': ((4.45, 4.60), (0.15, 0.20)),
}


def scatter(parts, n=20000, seed=0):
    """Raw (n, 2) scatter drawn from {population: share}."""
    rng = np.random.default_rng(seed)
    names = list(parts)
    which = rng.choice(len(names), n, p=[parts[k] for k in names])
    centre = np.array([POPULATIONS[names[i]][0] for i in which])
    sd = np.array([POPULATIONS[names[i]][1] for i in which])
    return 10.0 ** (centre + sd * rng.standard_normal((n, 2)))


def write_sample(path, sc, seed=0):
    rng = np.random.default_rng(seed)
    events = rng.normal(100.0, 10.0, (len(sc), len(RAW_PNN)))
    events[:, 0] = np.sort(rng.uniform(0, 100, len(sc)))
    events[:, 1:3] = sc
    write_raw_fcs(path, events)


def test_folder_groups_nested():
    groups = aa.folder_groups(['Raw/Day1/A.fcs', 'Raw/Day1/B.fcs', 'Raw/Day2/C.fcs', 'D.fcs'])
    assert groups == {'Raw': ['Raw/Day1/A.fcs', 'Raw/Day1/B.fcs', 'Raw/Day2/C.fcs'],
                      'Raw/Day1': ['Raw/Day1/A.fcs', 'Raw/Day1/B.fcs'],
                      'Raw/Day2': ['Raw/Day2/C.fcs']}


def test_reference_sources_excludes_absent_and_beads():
    profiles = {'P1': {'source_fcs': 'Raw/U1.fcs'}, 'P1b': {'source_fcs': 'Raw/U1.fcs'},
                'P2': {'source_fcs': 'Raw/Beads.fcs'}, 'P3': {'source_fcs': 'Other/X.fcs'}, 'P4': {}}
    refs = aa.reference_sources(profiles, ['Raw/U1.fcs', 'Raw/Beads.fcs'], exclude=['Raw/Beads.fcs'])
    assert refs == {'Raw/U1.fcs': ['P1', 'P1b']}


@pytest.fixture(scope='module')
def models():
    return {k: aa.fit_reference(np.log10(scatter({k: 1.0}, seed=i)))
            for i, k in enumerate(POPULATIONS)}


@pytest.mark.parametrize('parts, min_weight, expected', [
    ({'lymph': 1.0}, aa.MIN_WEIGHT, ['lymph']),
    ({'mono': 1.0}, aa.MIN_WEIGHT, ['mono']),
    ({'lymph': 0.5, 'mono': 0.5}, aa.MIN_WEIGHT, ['lymph', 'mono']),
    ({'lymph': 0.95, 'mono': 0.05}, aa.MIN_WEIGHT, ['lymph']),
    ({'lymph': 0.6, 'mono': 0.25, 'dead': 0.15}, 0.10, ['lymph', 'mono', 'dead']),
])
def test_choose_references(models, parts, min_weight, expected):
    names = list(models)
    xy = np.log10(scatter(parts, seed=11))
    log_dens = np.column_stack([models[k].score_samples(xy) for k in names])
    choice = aa.choose_references(log_dens, min_weight=min_weight)
    assert [names[i] for i in choice['indices']] == expected
    shares = np.array([parts[k] for k in expected])
    assert choice['weights'] == pytest.approx(shares / shares.sum(), abs=0.03)
    assert choice['bic'] <= choice['bic_single']


def test_choose_references_default_drops_small_shares(models):
    names = list(models)
    xy = np.log10(scatter({'lymph': 0.6, 'mono': 0.25, 'dead': 0.15}, seed=11))
    log_dens = np.column_stack([models[k].score_samples(xy) for k in names])
    choice = aa.choose_references(log_dens)
    assert len(choice['indices']) < 3
    assert min(choice['weights']) >= aa.MIN_WEIGHT


def test_choose_references_single(models):
    names = list(models)
    xy = np.log10(scatter({'lymph': 0.5, 'mono': 0.5}, seed=11))
    log_dens = np.column_stack([models[k].score_samples(xy) for k in names])
    choice = aa.choose_references(log_dens, max_references=1)
    assert choice['indices'] == [int(np.argmax(log_dens.sum(axis=0)))]
    assert choice['weights'] == [1.0] and choice['bic'] == choice['bic_single']


def test_mixture_weights_recovers_shares():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 1, 30000), rng.normal(4, 1, 10000)
    x = np.concatenate([a, b])
    log_dens = np.column_stack([-0.5 * x ** 2, -0.5 * (x - 4) ** 2]) - 0.5 * np.log(2 * np.pi)
    w, _ll = aa.mixture_weights(log_dens)
    assert w == pytest.approx([0.75, 0.25], abs=0.01)


def test_suggest_af_profiles_end_to_end(tmp_path):
    for i, k in enumerate(POPULATIONS):
        write_sample(tmp_path / 'Raw' / f'Unstained {k}.fcs', scatter({k: 1.0}, seed=i), seed=i)
    write_sample(tmp_path / 'Raw/S/Blood.fcs', scatter({'lymph': 0.5, 'mono': 0.5}, seed=20), seed=20)
    write_sample(tmp_path / 'Raw/S/Node.fcs', scatter({'lymph': 1.0}, seed=21), seed=21)
    write_sample(tmp_path / 'Raw/S/Tiny.fcs', scatter({'lymph': 1.0}, n=50, seed=22), seed=22)
    refs = {f'Raw/Unstained {k}.fcs': [f'{k} AF'] for k in POPULATIONS}
    refs['Raw/Missing.fcs'] = ['missing AF']
    pair, ceilings = ('FSC-A', 'SSC-A'), [MAGNITUDE_CEILING] * 2
    texts = []
    out = aa.suggest_af_profiles(tmp_path, refs, ['Raw/S/Blood.fcs', 'Raw/S/Node.fcs', 'Raw/S/Tiny.fcs'],
                                 pair, ceilings, progress=texts.append)
    assert out['assignments'] == {'Raw/S/Blood.fcs': ['lymph AF', 'mono AF'], 'Raw/S/Node.fcs': ['lymph AF']}
    assert set(out['skipped']) == {'Raw/S/Tiny.fcs'}
    assert set(out['unused_references']) == {'Raw/Missing.fcs'}
    assert out['details']['Raw/S/Blood.fcs']['delta_bic'] > 0
    for target in ('Raw/S/Blood.fcs', 'Raw/S/Node.fcs'):
        assert out['details'][target]['outside'] < 2 * aa.OUTSIDE_QUANTILE
    assert texts and texts[-1] == ''


def test_outside_share_flags_shifted_scatter(tmp_path):
    write_sample(tmp_path / 'U.fcs', scatter({'lymph': 1.0}, seed=1), seed=1)
    write_sample(tmp_path / 'S.fcs', scatter({'lymph': 1.0}, seed=2) * [10 ** 0.2, 1.0], seed=2)
    out = aa.suggest_af_profiles(tmp_path, {'U.fcs': ['U AF']}, ['S.fcs'], ('FSC-A', 'SSC-A'),
                                 [MAGNITUDE_CEILING] * 2)
    assert out['assignments'] == {'S.fcs': ['U AF']}
    assert out['details']['S.fcs']['outside'] > aa.POOR_FIT_SHARE


def test_saturated_and_nonpositive_dropped():
    sc = np.array([[1e4, 1e4], [0.0, 1e4], [MAGNITUDE_CEILING, 1e4], [np.nan, 1.0]])
    assert aa.log_scatter(sc, MAGNITUDE_CEILING).tolist() == [[4.0, 4.0]]

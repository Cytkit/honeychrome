"""
test_ag_results.py
------------------
Pure-numpy tests for bundled_plugins/ag_results.py (Automated Gating):
antigen-based channel alignment, per-sample summaries, feature tables for
the differential tests, QC flags and exports.

Usage:
    pytest tests/test_ag_results.py -m numpy_only
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

import ag_core as ag  # noqa: E402
import ag_results as ar  # noqa: E402


def _gate(name, gtype, x, y=None, parent=None, parent_pop='root', number=1, **extra):
    g = {
        'gate_number': number, 'gate_name': name, 'gate_type': gtype,
        'gate_marker_x': x, 'gate_marker_y': y,
        'parent_gate': parent, 'parent_popul': parent_pop if parent else 'root',
        'populations': ag.default_populations_for_type(gtype),
        'algorithm': 'tail', 'gate_param': {}, 'stats_parent': {}, 'origin_gate': None,
    }
    g.update(extra)
    return g


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

@pytest.mark.numpy_only
def test_alignment_prefers_antigen_over_channel_name():
    model_map = {'B1-A': 'CD4', 'B2-A': 'CD8', 'FSC-A': 'FSC-A', 'R1-A': 'Ki67'}
    exp_pnn = ['FSC-A', 'B1-A', 'B2-A', 'V1-A', 'R1-A']
    exp_pns = ['', 'CD8', 'CD19', 'CD4', 'Ki-67']
    out = ar.align_channels(list(model_map), model_map, exp_pnn, exp_pns)
    assert out['B1-A']['channel'] == 'V1-A' and out['B1-A']['method'] == ar.ALIGN_ANTIGEN
    assert out['B2-A']['channel'] == 'B1-A' and out['B2-A']['method'] == ar.ALIGN_ANTIGEN
    assert out['FSC-A']['channel'] == 'FSC-A' and out['FSC-A']['method'] == ar.ALIGN_EXACT
    assert out['R1-A']['channel'] == 'R1-A' and out['R1-A']['method'] == ar.ALIGN_EXACT


@pytest.mark.numpy_only
def test_alignment_flags_channel_only_and_ambiguous_matches():
    model_map = {'B1-A': 'CD3', 'B2-A': 'CD25'}
    exp_pnn = ['B1-A', 'B2-A', 'V1-A']
    exp_pns = ['TCRb', 'CD25', 'CD25']
    out = ar.align_channels(list(model_map), model_map, exp_pnn, exp_pns)
    assert out['B1-A'] == {'channel': 'B1-A', 'method': ar.ALIGN_CHANNEL,
                           'antigen': 'CD3', 'exp_antigen': 'TCRb'}
    assert out['B2-A']['channel'] == 'B2-A' and out['B2-A']['method'] == ar.ALIGN_EXACT
    out2 = ar.align_channels(['X-A'], {'X-A': 'CD25'}, exp_pnn, exp_pns)
    assert out2['X-A']['method'] == ar.ALIGN_AMBIGUOUS


@pytest.mark.numpy_only
def test_alignment_uses_canonical_synonyms_and_fuzzy_fallback():
    synonyms = {'CD197': 'CCR7', 'CCR7': 'CCR7'}
    out = ar.align_channels(['B1-A', 'B3-A'], {'B1-A': 'CD197', 'B3-A': 'CD45RA'},
                            ['V5-A', 'V6-A'], ['CCR7', 'CD45-RA '],
                            canonical=synonyms.get)
    assert out['B1-A']['channel'] == 'V5-A' and out['B1-A']['method'] == ar.ALIGN_ANTIGEN
    assert out['B3-A']['channel'] == 'V6-A'
    none = ar.align_channels(['Q-A'], {'Q-A': 'Zzz'}, ['V5-A'], ['CCR7'])
    assert none['Q-A']['channel'] is None and none['Q-A']['method'] is None


@pytest.mark.numpy_only
def test_alignment_problems_and_apply():
    assert ar.alignment_problems({'a': 'X', 'b': None}) == ['Not assigned: b']
    probs = ar.alignment_problems({'a': 'X', 'b': 'X'})
    assert len(probs) == 1 and 'X' in probs[0]
    model = {'gate_definitions': [_gate('G', '2dsep', 'B1-A', 'B2-A')],
             'channel_map': {'B1-A': 'CD4', 'B2-A': 'CD8'},
             'transforms': {'B1-A': {'id': 1}, 'B2-A': {'id': 1}}}
    out = ar.apply_alignment(model, {'B1-A': 'V1-A', 'B2-A': 'B2-A'})
    assert out['gate_definitions'][0]['gate_marker_x'] == 'V1-A'
    assert set(out['transforms']) == {'V1-A', 'B2-A'}
    assert model['gate_definitions'][0]['gate_marker_x'] == 'B1-A'


# ---------------------------------------------------------------------------
# Summaries and feature tables
# ---------------------------------------------------------------------------

def _hierarchy():
    return [
        _gate('Lymph', '1dsep', 'A', number=1),
        _gate('T', '1dsep', 'B', parent='Lymph', parent_pop='pos', number=2,
              stats_parent={'pos': 'root'}),
    ]


def _sample(seed, frac_t=0.5, marker_shift=0.0, n=3000):
    rng = np.random.default_rng(seed)
    a = np.where(rng.random(n) < 0.7, rng.normal(0.7, 0.03, n), rng.normal(0.2, 0.03, n))
    b = np.where(rng.random(n) < frac_t, rng.normal(0.7, 0.03, n), rng.normal(0.2, 0.03, n))
    c = rng.normal(0.4 + marker_shift, 0.05, n)
    return np.column_stack([a, b, c])


def _summarise(data, gates, ref):
    idx = {'A': 0, 'B': 1, 'C': 2}
    res = ag.run_gating(gates, data, idx, reference=ref)
    return ar.summarise_sample(gates, res, data, idx, ['A', 'B', 'C'], len(data))


def _reference(gates):
    calc = ag.GateCalculator({'A': (0, 1), 'B': (0, 1)})
    return {'Lymph': calc.threshold_boundaries(gates[0], 0.45, None),
            'T': calc.threshold_boundaries(gates[1], 0.45, None)}


@pytest.mark.numpy_only
def test_summary_counts_parents_and_quartiles():
    gates = _hierarchy()
    data = _sample(1)
    summ = _summarise(data, gates, _reference(gates))
    rows = {r['key']: r for r in summ['populations']}
    lymph = int((data[:, 0] >= 0.45).sum())
    t_pos = int(((data[:, 0] >= 0.45) & (data[:, 1] >= 0.45)).sum())
    assert rows['Lymph/pos']['n_events'] == lymph
    assert rows['T/pos']['n_events'] == t_pos
    assert rows['T/pos']['n_parent'] == lymph
    assert rows['T/pos']['n_stats_parent'] == len(data)
    med, q25, q75 = summ['markers']['T/pos']['C']
    sel = data[(data[:, 0] >= 0.45) & (data[:, 1] >= 0.45), 2]
    assert med == pytest.approx(np.median(sel))
    assert q25 <= med <= q75
    import json
    json.dumps(summ)


@pytest.mark.numpy_only
def test_population_tables_parent_and_stats_parent_denominators():
    gates = _hierarchy()
    ref = _reference(gates)
    summaries = {f's{i}': _summarise(_sample(i), gates, ref) for i in range(3)}
    samples = list(summaries)
    counts, denom, dropped = ar.population_tables(summaries, samples, use_stats_parent=False)
    assert dropped == []
    for s in samples:
        rows = {r['key']: r for r in summaries[s]['populations']}
        assert denom.loc[s, 'T/pos'] == rows['T/pos']['n_parent']
    _c, denom_sp, _d = ar.population_tables(summaries, samples, use_stats_parent=True)
    assert all(denom_sp.loc[s, 'T/pos'] == summaries[s]['n_total'] for s in samples)


@pytest.mark.numpy_only
def test_population_tables_drop_missing_and_zero_denominators():
    summaries = {
        's1': {'populations': [{'key': 'G/pos', 'n_events': 5, 'n_parent': 10},
                               {'key': 'H/pos', 'n_events': 0, 'n_parent': 0}]},
        's2': {'populations': [{'key': 'G/pos', 'n_events': 6, 'n_parent': 12},
                               {'key': 'H/pos', 'n_events': 1, 'n_parent': 3},
                               {'key': 'K/pos', 'n_events': 1, 'n_parent': 3}]},
    }
    counts, denom, dropped = ar.population_tables(summaries, ['s1', 's2'])
    assert list(counts.columns) == ['G/pos']
    assert set(dropped) == {'H/pos', 'K/pos'}


@pytest.mark.numpy_only
def test_marker_table_excludes_gating_markers_and_small_populations():
    gates = _hierarchy()
    ref = _reference(gates)
    summaries = {f's{i}': _summarise(_sample(i), gates, ref) for i in range(3)}
    samples = list(summaries)
    df, meta, dropped = ar.marker_table(summaries, samples, gates)
    assert ar.marker_feature('T/pos', 'C') in df.columns
    assert ar.marker_feature('T/pos', 'A') not in df.columns
    assert ar.marker_feature('T/pos', 'B') not in df.columns
    assert ar.marker_feature('Lymph/pos', 'B') in df.columns
    assert set(meta.columns) == {'feature', 'cluster', 'cluster_id', 'channel'}
    assert set(meta['feature']) == set(df.columns)
    df_all, _m, _d = ar.marker_table(summaries, samples, gates, exclude_gating_markers=False)
    assert ar.marker_feature('T/pos', 'A') in df_all.columns
    _df, _m2, dropped_big = ar.marker_table(summaries, samples, gates, min_events=10 ** 6)
    assert set(dropped_big) == {'Lymph/pos', 'Lymph/neg', 'T/pos', 'T/neg'}


@pytest.mark.numpy_only
def test_end_to_end_logit_test_detects_group_shift():
    from honeychrome.controller_components import differential_stats as ds

    gates = _hierarchy()
    ref = _reference(gates)
    summaries = {}
    groups = []
    for i in range(10):
        grp = 'ctrl' if i < 5 else 'treated'
        summaries[f's{i}'] = _summarise(_sample(100 + i, frac_t=0.3 if grp == 'ctrl' else 0.6),
                                        gates, ref)
        groups.append(grp)
    samples = list(summaries)
    counts, denom, _ = ar.population_tables(summaries, samples)
    data = ds.logit_proportions(counts, denom)
    res = ds.run_moderated(data, groups, [('ctrl', 'treated')], 'reference',
                           pval_threshold=0.05, fc_threshold=0.0)
    sig = set(res.loc[res['significant'], 'feature'])
    assert {'T/pos', 'T/neg'} <= sig
    assert 'Lymph/pos' not in sig


@pytest.mark.numpy_only
def test_flags_and_exports():
    summ = {'status': {'G': {'flag': 'drift', 'message': 'moved 0.1'}, 'H': {'flag': None}},
            'populations': [{'key': 'G/pos', 'n_events': 5, 'n_parent': 10}],
            'markers': {'G/pos': {'C': [0.5, 0.4, 0.6]}}}
    flags = ar.sample_flags(summ, {'n_events_file': 1000, 'n_events_kept': 900})
    assert any('moved 0.1' in f for f in flags)
    assert any('10.0%' in f for f in flags)
    assert ar.sample_flags({'status': {}}, {'n_events_file': 5, 'n_events_kept': 5}) == []
    pe = ar.population_export({'s': summ}, ['s'], {'s': 'ctrl'})
    assert list(pe['group']) == ['ctrl'] and list(pe['key']) == ['G/pos']
    me = ar.marker_export({'s': summ}, ['s'], {'C': 'CD4'})
    assert me.loc[0, 'antigen'] == 'CD4' and me.loc[0, 'iqr'] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Statistics orchestration
# ---------------------------------------------------------------------------

def _grouped_summaries(n_per_group=4, shift=True):
    gates = _hierarchy()
    ref = _reference(gates)
    summaries, groups = {}, {}
    plan = [('ctrl', 0.3, 0.0), ('low', 0.3, 0.0), ('high', 0.6 if shift else 0.3, 0.15)]
    seed = 200
    for grp, frac, mshift in plan:
        for _ in range(n_per_group):
            key = f'{grp}{seed}'
            summaries[key] = _summarise(_sample(seed, frac_t=frac, marker_shift=mshift), gates, ref)
            groups[key] = grp
            seed += 1
    return gates, summaries, groups


@pytest.mark.numpy_only
def test_run_statistics_reference_contrasts_and_marker_families():
    gates, summaries, groups = _grouped_summaries()
    samples = list(summaries)
    out = ar.run_statistics(summaries, samples, groups, gates,
                            group_order=['ctrl', 'low', 'high'], reference='ctrl',
                            fc_threshold=0.0, marker_threshold=0.0)
    assert out['comparisons'] == [('ctrl', 'low'), ('ctrl', 'high')]
    freq = out['freq']
    hi = freq[freq['comparison'] == 'high vs ctrl']
    lo = freq[freq['comparison'] == 'low vs ctrl']
    assert 'T/pos' in set(hi.loc[hi['significant'], 'feature'])
    assert not lo['significant'].any()
    assert out['freq_values'].loc[samples[0]].between(0, 100).all()
    mk = out['markers']
    assert 'stagewise.adj.P.Val' in mk.columns and 'cluster_id' in mk.columns
    hit = mk[(mk['comparison'] == 'high vs ctrl') & mk['significant']]
    # C is shifted in every event, so it rises in every population.
    assert set(hit.loc[hit['channel'] == 'C', 'cluster_id']) == {
        'Lymph/pos', 'Lymph/neg', 'T/pos', 'T/neg'}
    # T populations are never tested on their own gating markers.
    t_rows = mk[mk['cluster_id'].str.startswith('T/')]
    assert set(t_rows['channel']) == {'C'}
    assert not mk.loc[mk['comparison'] == 'low vs ctrl', 'significant'].any()
    assert out['counts'] is None


@pytest.mark.numpy_only
def test_run_statistics_pairwise_pairing_and_covariates():
    import pandas as pd
    gates, summaries, groups = _grouped_summaries()
    samples = list(summaries)
    pairing = {s: f'donor{i % 4}' for i, s in enumerate(samples)}
    ages = np.random.default_rng(5).integers(20, 70, len(samples))
    cov = pd.DataFrame({'age': [str(a) for a in ages]}, index=samples)
    out = ar.run_statistics(summaries, samples, groups, gates, group_order=['ctrl', 'low', 'high'],
                            contrast_mode='pairwise', pairing=pairing, covariates=cov,
                            run_markers=False, fc_threshold=0.0)
    assert out['comparisons'] == [('ctrl', 'low'), ('ctrl', 'high'), ('low', 'high')]
    assert set(out['freq']['comparison']) == {'low vs ctrl', 'high vs ctrl', 'high vs low'}
    assert out['markers'] is None


@pytest.mark.numpy_only
def test_run_statistics_needs_two_qualifying_groups():
    gates, summaries, groups = _grouped_summaries()
    only = {s: (g if g == 'ctrl' else '') for s, g in groups.items()}
    with pytest.raises(RuntimeError, match='two groups'):
        ar.run_statistics(summaries, list(summaries), only, gates)
    assert ar.qualifying_groups(list(summaries), groups, ['high']) == ['high', 'ctrl', 'low']


@pytest.mark.numpy_only
def test_reapply_thresholds_without_refit():
    gates, summaries, groups = _grouped_summaries()
    out = ar.run_statistics(summaries, list(summaries), groups, gates, reference='ctrl',
                            fc_threshold=0.0, marker_threshold=0.0)
    strict = ar.reapply_thresholds(out, 1e-12, 50.0, 50.0, 'global', True)
    assert not strict['freq']['significant'].any()
    assert not strict['markers']['significant'].any()
    assert out['freq']['significant'].any()
    np.testing.assert_allclose(strict['freq']['logFC'], out['freq']['logFC'])


@pytest.mark.numpy_only
def test_run_statistics_counts_glm_uses_parent_offset():
    pytest.importorskip('statsmodels')
    gates, summaries, groups = _grouped_summaries()
    out = ar.run_statistics(summaries, list(summaries), groups, gates, reference='ctrl',
                            run_freq=False, run_markers=False, run_counts=True, fc_threshold=0.0)
    hi = out['counts'][out['counts']['comparison'] == 'high vs ctrl']
    assert 'T/pos' in set(hi.loc[hi['significant'], 'feature'])

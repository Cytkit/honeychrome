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
import pandas as pd
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
def test_population_tables_keep_populations_measured_in_some_samples():
    summaries = {}
    for i in range(5):
        pops = [{'key': 'G/pos', 'n_events': 5 + i, 'n_parent': 10},
                {'key': 'H/pos', 'n_events': 0 if i == 1 else 2, 'n_parent': 0 if i == 0 else 8}]
        if i < 2:
            pops.append({'key': 'K/pos', 'n_events': 1, 'n_parent': 3})
        summaries[f's{i}'] = {'populations': pops}
    counts, denom, dropped = ar.population_tables(summaries, list(summaries))
    assert dropped == ['K/pos']
    assert list(counts.columns) == ['G/pos', 'H/pos']
    # Parent empty: not measurable, so NaN rather than zero.
    assert np.isnan(counts.loc['s0', 'H/pos']) and np.isnan(denom.loc['s0', 'H/pos'])
    # No events in a sample whose parent has events: a true zero.
    assert counts.loc['s1', 'H/pos'] == 0 and denom.loc['s1', 'H/pos'] == 8
    assert counts['G/pos'].notna().all()


def _summaries_with_unmeasured_population():
    rng = np.random.default_rng(1)
    summaries, groups = {}, {}
    for i in range(12):
        grp = 'ctrl' if i < 6 else 'treated'
        pops = []
        for key, base in (('A/pos', 0.2), ('B/pos', 0.3), ('C/pos', 0.1), ('D/pos', 0.4)):
            if key == 'D/pos' and i in (0, 7):
                pops.append({'key': key, 'n_events': 0, 'n_parent': 0})
                continue
            frac = base * 3 if (key == 'A/pos' and grp == 'treated') else base
            pops.append({'key': key, 'n_events': int(rng.binomial(1000, frac)), 'n_parent': 1000})
        summaries[f's{i}'] = {'populations': pops}
        groups[f's{i}'] = grp
    return summaries, groups


@pytest.mark.numpy_only
def test_run_statistics_tests_populations_unmeasured_in_some_samples():
    summaries, groups = _summaries_with_unmeasured_population()
    out = ar.run_statistics(summaries, list(summaries), groups, [], run_freq=True,
                            run_counts=True, run_markers=False, fc_threshold=0.0)
    assert 'D/pos' in set(out['freq']['feature'])
    assert out['freq_values']['D/pos'].isna().sum() == 2
    assert any('D/pos' in note for note in out['notes'])
    d_counts = out['counts'][out['counts']['feature'] == 'D/pos'].iloc[0]
    assert np.isfinite(d_counts['logFC']) and np.isfinite(d_counts['P.Value'])
    assert 'A/pos' in set(out['freq'].loc[out['freq']['significant'], 'feature'])


@pytest.mark.numpy_only
def test_counts_untestable_when_a_group_has_no_measured_sample():
    summaries, groups = _summaries_with_unmeasured_population()
    for key, grp in groups.items():
        if grp == 'treated':
            for row in summaries[key]['populations']:
                if row['key'] == 'D/pos':
                    row.update(n_events=0, n_parent=0)
    out = ar.run_statistics(summaries, list(summaries), groups, [], run_freq=True,
                            run_counts=True, run_markers=False, fc_threshold=0.0)
    d_counts = out['counts'][out['counts']['feature'] == 'D/pos'].iloc[0]
    assert np.isnan(d_counts['logFC'])
    assert not bool(d_counts['significant'])


def _summaries_with_empty_populations():
    """Twelve samples; the groups' parent counts differ about fifty-fold.
    'Z/pos' has no events anywhere, 'Y/pos' has events in two samples only."""
    rng = np.random.default_rng(2)
    summaries, groups = {}, {}
    for i in range(12):
        grp = 'ctrl' if i < 6 else 'treated'
        n_parent = 5000 if grp == 'ctrl' else 100
        pops = [{'key': 'A/pos', 'n_events': int(rng.binomial(n_parent, 0.3)),
                 'n_parent': n_parent},
                {'key': 'B/pos', 'n_events': int(rng.binomial(n_parent, 0.2)),
                 'n_parent': n_parent},
                {'key': 'C/pos', 'n_events': int(rng.binomial(n_parent, 0.1)),
                 'n_parent': n_parent},
                {'key': 'Y/pos', 'n_events': 1 if i in (3, 9) else 0, 'n_parent': n_parent},
                {'key': 'Z/pos', 'n_events': 0, 'n_parent': n_parent}]
        summaries[f's{i}'] = {'populations': pops}
        groups[f's{i}'] = grp
    return summaries, groups


@pytest.mark.numpy_only
def test_population_tables_drop_populations_with_events_in_too_few_samples():
    summaries, _groups = _summaries_with_empty_populations()
    counts, denom, dropped = ar.population_tables(summaries, list(summaries))
    assert dropped == ['Y/pos', 'Z/pos']
    assert list(counts.columns) == ['A/pos', 'B/pos', 'C/pos']
    assert list(denom.columns) == ['A/pos', 'B/pos', 'C/pos']


@pytest.mark.numpy_only
def test_empty_populations_are_not_tested_despite_different_parent_counts():
    from honeychrome.controller_components import differential_stats as ds
    summaries, groups = _summaries_with_empty_populations()
    samples = list(summaries)
    # Untreated by the filter, an empty population looks different between
    # the groups only because its parent counts differ.
    counts = pd.DataFrame({'Z/pos': [0.0] * 12, 'A/pos': [r['n_events'] for s in samples
                                                          for r in summaries[s]['populations']
                                                          if r['key'] == 'A/pos'],
                           'B/pos': [r['n_events'] for s in samples
                                     for r in summaries[s]['populations'] if r['key'] == 'B/pos'],
                           'C/pos': [r['n_events'] for s in samples
                                     for r in summaries[s]['populations'] if r['key'] == 'C/pos']},
                          index=samples, dtype=float)
    denom = pd.DataFrame({c: [summaries[s]['populations'][0]['n_parent'] for s in samples]
                          for c in counts.columns}, index=samples, dtype=float)
    group_vec = [groups[s] for s in samples]
    contrasts = ds.build_contrasts(['ctrl', 'treated'], 'reference', 'ctrl')
    raw = ds.run_moderated(ds.logit_proportions(counts, denom), group_vec, contrasts,
                           'reference', 0.05, 0.0)
    assert bool(raw.loc[raw['feature'] == 'Z/pos', 'significant'].iloc[0])

    out = ar.run_statistics(summaries, samples, groups, [], run_freq=True, run_counts=True,
                            run_markers=False, fc_threshold=0.0)
    for table in ('freq', 'counts'):
        assert not {'Y/pos', 'Z/pos'} & set(out[table]['feature'])
    assert list(out['freq_values'].columns) == ['A/pos', 'B/pos', 'C/pos']
    assert any('Y/pos' in note and 'Z/pos' in note for note in out['notes'])


@pytest.mark.numpy_only
def test_stats_round_trip_through_json():
    import json
    summaries, groups = _summaries_with_unmeasured_population()
    out = ar.run_statistics(summaries, list(summaries), groups, [], run_freq=True,
                            run_counts=True, run_markers=False, fc_threshold=0.0)
    back = ar.stats_from_jsonable(json.loads(json.dumps(ar.stats_to_jsonable(out))))
    assert back['comparisons'] == out['comparisons']
    assert back['samples'] == out['samples'] and back['notes'] == out['notes']
    assert back['markers'] is None
    for key in ('freq', 'counts', 'freq_values', 'counts_values'):
        pd.testing.assert_frame_equal(back[key], out[key], check_dtype=False)
    again = ar.reapply_thresholds(back, 0.05, 0.0, 0.03, 'global', True)
    assert list(again['counts']['significant']) == list(out['counts']['significant'])


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
                            numeric_covariates=['age'], run_markers=False, fc_threshold=0.0)
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


# ---------------------------------------------------------------------------
# Sample tables, sample PCA and predictive features
# ---------------------------------------------------------------------------

def test_sample_table_matches_names_and_reads_covariates():
    keys = ['Raw/Plate 1/S1.fcs', 'Raw/Plate 1/S2.fcs', 'Raw/S3.fcs']
    df = pd.DataFrame({'Sample': ['S1', 's2.FCS', 'Raw/S3.fcs', 'missing'],
                       'Group': ['A', 'B', '', 'A'], 'donor': ['d1', 'd2', 'd3', 'd4'],
                       'age': ['30', None, '41', '50']})
    out = ar.read_sample_table(df, keys)
    assert out['groups'] == {keys[0]: 'A', keys[1]: 'B'}
    assert out['covariates']['donor'] == {keys[0]: 'd1', keys[1]: 'd2', keys[2]: 'd3'}
    assert out['covariates']['age'] == {keys[0]: '30', keys[2]: '41'}
    assert out['unmatched'] == ['missing']
    back = ar.read_sample_table(ar.sample_table_frame(keys, out['groups'], out['covariates']), keys)
    assert back['groups'] == out['groups'] and back['covariates']['donor'] == out['covariates']['donor']
    with pytest.raises(ValueError):
        ar.read_sample_table(pd.DataFrame({'x': [1]}), keys)


def _stats_for_pca(n=24, seed=0):
    rng = np.random.default_rng(seed)
    samples = [f'S{i}' for i in range(n)]
    groups = ['A'] * (n // 2) + ['B'] * (n - n // 2)
    freq = pd.DataFrame(rng.normal(10, 1, (n, 5)), index=samples,
                        columns=[f'g{i}/p' for i in range(5)])
    freq.iloc[n // 2:, :3] += 6       # three populations move together between groups
    med = pd.DataFrame(rng.normal(0.5, 0.05, (n, 3)), index=samples,
                       columns=[ar.marker_feature('g0/p', c) for c in ('PE', 'APC', 'FITC')])
    return {'samples': samples, 'group_vec': groups, 'freq_values': freq,
            'counts_values': None, 'marker_values': med}


def test_sample_pca_separates_groups_on_the_varying_feature():
    pytest.importorskip('sklearn')
    stats = _stats_for_pca()
    pca = ar.sample_pca(stats, use_freq=True, use_markers=True, n_loadings=2,
                        labels={'g0/p': 'CD4 T', 'PE': 'CD25'})
    assert pca['sources'] == ['Freq', 'Median']
    assert 'Freq: CD4 T' in list(pca['loadings'].index)
    pc1 = pca['scores']['PC1'].values
    assert abs(pc1[:12].mean() - pc1[12:].mean()) > 2
    assert pca['groups'] == stats['group_vec']
    assert ar.sample_pca(stats, use_freq=False, use_counts=True) is None


def test_predictive_matrix_names_features_for_display():
    stats = _stats_for_pca()
    X = ar.predictive_matrix(stats, use_freq=True, use_markers=True, labels={'g0/p': 'CD4 T'})
    assert X.shape == (24, 8)
    assert 'Freq: CD4 T' in X.columns and 'Median: CD4 T PE' in X.columns
    assert np.allclose(X['Freq: CD4 T'], np.log2(stats['freq_values']['g0/p'] + 0.01))


# ---------------------------------------------------------------------------
# Saved tables, display records and the population comparison
# ---------------------------------------------------------------------------

def test_frame_from_jsonable_keeps_column_types():
    df = pd.DataFrame({'feature': ['a', 'b', 'c'], 'n': [1, 2, 3], 'x': [0.5, None, 1.5],
                       'significant': [True, False, True]}, index=['r1', 'r2', 'r3'])
    back = ar._frame_from_jsonable(ar._frame_to_jsonable(df))
    assert list(back.columns) == list(df.columns) and list(back.index) == list(df.index)
    assert back['n'].dtype == np.int64 and back['significant'].dtype == bool
    assert np.isnan(back['x'].iloc[1]) and back['x'].iloc[2] == 1.5


@pytest.mark.numpy_only
def test_display_record_packs_population_membership(tmp_path):
    gates = _hierarchy()
    data = _sample(3)
    idx = {'A': 0, 'B': 1, 'C': 2}
    res = ag.run_gating(gates, data, idx, reference=_reference(gates))
    rows = np.arange(0, len(data), 3)
    disp = ar.pack_display(gates, res, rows, data[rows], ['A', 'B', 'C'])
    t_pos = ar.display_population_mask(disp, 'T/pos')
    assert np.array_equal(t_pos, res.population_mask('T', 'pos')[rows])
    assert ar.display_population_mask(disp, 'root').all()
    assert ar.display_population_mask(disp, 'nothing/here') is None
    parent = ar.display_parent_mask(disp, 'T', gates[1])
    assert np.array_equal(parent, res.population_mask('Lymph', 'pos')[rows])
    path = tmp_path / ar.display_file_name('S1.fcs')
    ar.save_display(path, disp, 'S1.fcs', 'abc')
    loaded, meta = ar.load_display(path)
    assert meta == {'sample': 'S1.fcs', 'fingerprint': 'abc'}
    assert np.array_equal(ar.display_population_mask(loaded, 'T/pos'), t_pos)
    assert np.allclose(loaded['events'], data[rows])


@pytest.mark.numpy_only
def test_group_values_and_population_result_match_the_statistics():
    gates, summaries, groups = _grouped_summaries()
    samples = list(summaries)
    out = ar.run_statistics(summaries, samples, groups, gates,
                            group_order=['ctrl', 'low', 'high'], reference='ctrl',
                            fc_threshold=0.0, marker_threshold=0.0)
    values = ar.group_population_values(summaries, samples, groups, 'T/pos')
    assert set(values) == {'ctrl', 'low', 'high'}
    assert all(len(v) == 4 for v in values.values())
    hi = np.mean([v for _s, v in values['high']])
    ctrl = np.mean([v for _s, v in values['ctrl']])
    assert hi > ctrl
    res = ar.population_result(out, 'T/pos', 'high vs ctrl')
    assert res['significant'] and res['P'] < 0.05 and res['logFC'] > 0
    assert ar.population_result(out, 'T/pos', 'no such comparison') is None
    tables = ar.statistics_tables(out, labels={'T/pos': 'T cells'}, antigens={'C': 'CD25'})
    assert 'T cells' in set(tables['frequencies']['population'])
    mk = tables['markers']
    assert set(mk.loc[mk['channel'] == 'C', 'antigen']) == {'CD25'}
    assert set(mk.loc[mk['channel'] == 'B', 'antigen']) == {'B'}
    assert list(tables['values_frequencies']['group']) == out['group_vec']

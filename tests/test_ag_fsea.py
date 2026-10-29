"""Tests for Flow Set Enrichment Analysis (ag_fsea, ag_fsea_library)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_PLUGINS = os.path.join(os.path.dirname(__file__), '..', 'src', 'honeychrome', 'bundled_plugins')
sys.path.insert(0, os.path.abspath(_PLUGINS))

import ag_fsea as fs  # noqa: E402
import ag_fsea_library as fl  # noqa: E402
from honeychrome.controller_components import moderated_stats as ms  # noqa: E402

CD8_MARKERS = ('PD1', 'TIGIT', 'LAG3', 'CTLA4', 'CD39', 'ICOS')
CD4_MARKERS = ('Ki67', 'HLADR', 'CD38', 'CD69', 'CD25', 'CD27')


# ---------------------------------------------------------------------------
# Synthetic experiment
# ---------------------------------------------------------------------------

def _gate(name, gtype, parent_gate, parent_pop, pops, x='FSC-A', y=None):
    return {'gate_name': name, 'gate_type': gtype, 'gate_marker_x': x, 'gate_marker_y': y,
            'parent_gate': parent_gate, 'parent_popul': parent_pop,
            'populations': {p: {'label': p, 'region': 'above'} for p in pops},
            'algorithm': 'tail', 'gate_param': {}, 'stats_parent': {}}


def synthetic_gates():
    gates = [_gate('Live', 'free', None, 'root', ['Live cells']),
             _gate('T cells', '2dsep', 'Live', 'Live cells', ['T cells'], 'ch_CD3', 'ch_CD19'),
             _gate('CD4 & CD8', '2dsep', 'T cells', 'T cells', ['CD4', 'CD8'], 'ch_CD8', 'ch_CD4')]
    for parent, markers in (('CD8', CD8_MARKERS), ('CD4', CD4_MARKERS)):
        for m in markers + CD8_MARKERS[:2]:
            name = f"{m}+ ({parent})"
            if not any(g['gate_name'] == name for g in gates):
                gates.append(_gate(name, '1dsep', 'CD4 & CD8', parent, [f"{m}+"], f"ch_{m}"))
    return gates


def synthetic_summaries(n_per_group=20, shift=0.0, seed=0):
    """Counts with a logit shift of the CD8 positivity of CD8_MARKERS in group B."""
    rng = np.random.default_rng(seed)
    gates = synthetic_gates()
    samples, groups, summaries = [], {}, {}
    for gi, grp in enumerate(('A', 'B')):
        for i in range(n_per_group):
            s = f"{grp}{i:02d}.fcs"
            samples.append(s)
            groups[s] = grp
            live = int(rng.integers(40000, 60000))
            t = int(live * 0.4)
            n = {'Live/Live cells': live, 'T cells/T cells': t,
                 'CD4 & CD8/CD4': int(t * 0.6), 'CD4 & CD8/CD8': int(t * 0.3)}
            parent_n = {'Live/Live cells': 100000, 'T cells/T cells': live,
                        'CD4 & CD8/CD4': t, 'CD4 & CD8/CD8': t}
            donor = rng.normal(0, 0.4)
            for g in gates[3:]:
                pop = next(iter(g['populations']))
                par = f"CD4 & CD8/{g['parent_popul']}"
                marker = pop[:-1]
                eta = -1.5 + donor + rng.normal(0, 0.5)
                if gi == 1 and g['parent_popul'] == 'CD8' and marker in CD8_MARKERS:
                    eta += shift
                p = 1.0 / (1.0 + np.exp(-eta))
                key = f"{g['gate_name']}/{pop}"
                n[key] = int(rng.binomial(n[par], p))
                parent_n[key] = n[par]
            rows = [{'key': k, 'n_events': v, 'n_parent': parent_n[k]} for k, v in n.items()]
            markers = {k: {'ch_X': [float(rng.normal(1, 0.1)), 0.0, 2.0]}
                       for k in ('CD4 & CD8/CD4', 'CD4 & CD8/CD8', 'T cells/T cells')}
            summaries[s] = {'populations': rows, 'markers': markers, 'n_total': 100000}
    return summaries, samples, groups, gates


def library_csv(tmp_path):
    rows = ["immune_state,description,cell_type,feature_kind,marker,direction,weight,species"]
    for m in ('PD-1', 'TIGIT', 'LAG3', 'CTLA-4', 'CD39', 'ICOS'):
        rows.append(f"Planted,planted,CD8 T cell,Marker,{m},UP,High,human")
    for m in ('Ki67', 'HLA-DR', 'CD38', 'CD69', 'CD25', 'CD27'):
        rows.append(f"Null,null,CD4 T cell,Marker,{m},UP,High,human")
    rows.append("Small,small,CD4 T cell,Marker,PD-1,UP,High,human")
    path = tmp_path / 'lib.csv'
    path.write_text("\n".join(rows) + "\n")
    return path


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_group_rank_stats_match_moderated_fit():
    rng = np.random.default_rng(3)
    Y = rng.normal(size=(60, 24)) + 5.0
    Y[rng.random(Y.shape) < 0.1] = np.nan
    Y[0, :8] = np.nan                       # no observations in the baseline group
    codes = np.repeat([0, 1, 2], 8)
    out = fs.group_rank_stats(Y, codes[None, :])
    X = np.column_stack([np.ones(24), codes == 1, codes == 2]).astype(float)
    fit = ms.e_bayes(ms.lm_fit(Y, X))
    for k in (1, 2):
        np.testing.assert_allclose(out[k]['t'][0], fit.t[:, k], atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(out[k]['df'][0], fit.df_total, atol=1e-10, equal_nan=True)
        p0 = ms.treat_pvalues(fit.coefficients[:, k],
                              fit.stdev_unscaled[:, k] * np.sqrt(fit.s2_post), fit.df_total, 0)
        expected = np.sign(fit.coefficients[:, k]) * -np.log10(p0)
        np.testing.assert_allclose(out[k]['r'][0], expected, atol=1e-8, equal_nan=True)


def _brute_es(r, idx, signs):
    x = r.copy()
    x[idx] = r[idx] * signs
    member = np.zeros(r.size, bool)
    member[idx] = True
    order = sorted(range(r.size), key=lambda i: (-x[i], not member[i]))
    nr = np.abs(x[member]).sum()
    run, best = 0.0, 0.0
    for i in order:
        run += abs(x[i]) / nr if member[i] else -1.0 / (r.size - len(idx))
        if abs(run) > abs(best):
            best = run
    return best


def test_enrichment_scores_match_brute_force():
    rng = np.random.default_rng(5)
    R = rng.normal(size=(6, 200))
    sets = [(np.array([3, 17, 40, 41, 99]), np.array([1., 1., -1., 1., -1.])),
            (np.arange(0, 200, 25), np.ones(8))]
    es = fs.enrichment_scores(R, sets)
    for i in range(R.shape[0]):
        for j, (idx, sg) in enumerate(sets):
            assert es[i, j] == pytest.approx(_brute_es(R[i], idx, sg), abs=1e-12)
    rs = fs.running_sum(R[0], *sets[0])
    assert rs['es'] == pytest.approx(es[0, 0])
    assert rs['curve'][0] == 0.0 and rs['curve'][-1] == pytest.approx(0.0, abs=1e-12)


def test_collapse_duplicates_mirrors_constants():
    v = pd.DataFrame({'a': [1.0, 2.0, np.nan], 'b': [1.0, 2.0, np.nan], 'c': [-1.0, -2.0, np.nan],
                      'd': [3.0, 3.0, 3.0], 'e': [0.5, 0.1, 0.2]})
    meta = pd.DataFrame({'feature': list(v.columns)})
    kept, meta2, hyg = fs.collapse_redundant(v, meta)
    assert list(kept.columns) == ['a', 'e']
    m = meta2.set_index('feature')
    assert (m.loc['b', 'kept'], m.loc['b', 'orientation']) == ('a', 1)
    assert (m.loc['c', 'kept'], m.loc['c', 'orientation']) == ('a', -1)
    assert m.loc['d', 'kept'] == ''
    assert set(hyg['kind']) == {'duplicate', 'mirror', 'constant'}


def test_minimum_sample_gate():
    def plans(n_a, n_b):
        samples = [f"a{i}" for i in range(n_a)] + [f"b{i}" for i in range(n_b)]
        groups = {s: s[0].upper() for s in samples}
        return fs.comparison_plans(samples, groups, group_order=['A', 'B'], reference='A')
    p, blocked, _u = plans(10, 5)
    assert not p and '5 + 10 samples' in blocked[0]['reason']
    p, blocked, _u = plans(25, 5)
    assert not p and blocked
    p, blocked, _u = plans(15, 15)
    assert len(p) == 1 and not blocked and p[0]['fast']


# ---------------------------------------------------------------------------
# Library and matching
# ---------------------------------------------------------------------------

def test_bundled_library_loads_cleanly():
    lib, sha, messages = fl.load_library()
    assert len(sha) == 64 and not messages
    assert set(lib['direction']) <= {'UP', 'DOWN'}
    assert lib.groupby('immune_state').size().min() >= 8


def test_bundled_mouse_library_loads_cleanly():
    lib, sha, messages = fl.load_library(species='mouse')
    assert len(sha) == 64 and not messages
    assert set(lib['species']) == {'mouse'}
    assert lib.groupby('immune_state').size().min() >= 5


def test_library_validation(tmp_path):
    path = tmp_path / 'bad.csv'
    path.write_text(
        "immune_state,cell_type,feature_kind,marker,direction,surplus\n"
        "S,CD8 T cell,Marker,PD-1,UP,x\n"
        "S,CD8 T cell,Marker,PD1,UP,x\n"
        "S,CD8 T cell,Marker,PD-1,SIDEWAYS,x\n"
        "S,Martian cell,Marker,PD-1,UP,x\n"
        "S,CD8 T cell,Marker,NotAMarker99,UP,x\n"
        "S,CD8 T cell,Abund,PD-1,UP,x\n")
    lib, _sha, messages = fl.load_library(path)
    assert len(lib) == 1 and lib['marker_canonical'][0] == 'PD-1'
    assert len(messages) == 6


def test_positivity_and_lineage_mapping():
    gates = [_gate('B cells', '2dsep', None, 'root', ['B cells'], 'c19', 'c20'),
             _gate('Mem', '2dsep', 'B cells', 'B cells', ['Memory'], 'c27', 'c21'),
             _gate('T cells', '2dsep', None, 'root', ['T cells'], 'c3', 'c19'),
             _gate('TMem', '2dsep', 'T cells', 'T cells', ['Memory'], 'c45', 'c27'),
             _gate('CD28+ (TMem)', '1dsep', 'TMem', 'Memory', ['CD28+'], 'shared'),
             _gate('CD20+ (Mem)', '1dsep', 'Mem', 'Memory', ['CD20+'], 'shared')]
    info = fl.population_info(gates)
    assert info['CD28+ (TMem)/CD28+']['kind'] == 'positivity'
    assert info['CD28+ (TMem)/CD28+']['marker'] == 'CD28'
    pm = fl.map_populations(info)
    assert pm['Mem/Memory']['cell_type'] == 'Resting memory B cell'
    assert pm['TMem/Memory']['cell_type'] is None
    res = fl.resolve_channel_markers(info, ['TMem/Memory', 'Mem/Memory', 'T cells/T cells'],
                                     ['shared'], {'shared': 'CD20'})
    assert res[('TMem/Memory', 'shared')] == ('CD28', 'gate')
    assert res[('Mem/Memory', 'shared')] == ('CD20', 'gate')
    assert res[('T cells/T cells', 'shared')] == ('CD28', 'gate (descendant)')


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_run_enrichment_finds_planted_state(tmp_path):
    summaries, samples, groups, gates = synthetic_summaries(shift=1.2, seed=1)
    lib, sha, _m = fl.load_library(library_csv(tmp_path))
    kw = dict(group_order=['A', 'B'], reference='A', n_perm=500, seed=11,
              library_info={'sha256': sha})
    res = fs.run_enrichment(summaries, samples, groups, gates, lib, **kw)
    st = res['states'].set_index('immune_state')
    assert st.loc['Planted', 'FDR'] < 0.05 and st.loc['Planted', 'direction'] == 'gained'
    assert st.loc['Planted', 'n_tested'] == 6
    assert st.loc['Null', 'p'] > st.loc['Planted', 'p']
    assert 'not testable' in st.loc['Small', 'note']
    again = fs.run_enrichment(summaries, samples, groups, gates, lib, **kw)
    pd.testing.assert_frame_equal(res['states'], again['states'])
    back = fs.result_from_jsonable(fs.result_to_jsonable(res))
    assert list(back['states']['immune_state']) == list(res['states']['immune_state'])
    curve = fs.state_curve(back, 'B vs A', 'Planted')
    assert curve['es'] == pytest.approx(st.loc['Planted', 'ES'])


def test_run_enrichment_with_covariate_uses_residual_permutation(tmp_path):
    summaries, samples, groups, gates = synthetic_summaries(shift=1.2, seed=2)
    lib, _sha, _m = fl.load_library(library_csv(tmp_path))
    cov = pd.DataFrame({'age': [str(30 + (i * 7) % 40) for i in range(len(samples))]},
                       index=samples)
    res = fs.run_enrichment(summaries, samples, groups, gates, lib, group_order=['A', 'B'],
                            reference='A', covariates=cov, numeric_covariates=['age'],
                            n_perm=200, seed=3)
    assert res['settings']['engine'].startswith('residual')
    st = res['states'].set_index('immune_state')
    assert st.loc['Planted', 'p'] < 0.05


def test_default_anchor_follows_the_model_anchor():
    gates = [_gate('Single Cells', 'singlets', None, 'root', ['Single Cells'], 'FSC-A', 'FSC-H'),
             _gate('Viable', '1dsep', 'Single Cells', 'Single Cells', ['Live'], 'Zombie NIR', 'FSC-A'),
             _gate('T cells', '2dsep', 'Viable', 'Live', ['T cells'], 'ch_CD3', 'ch_CD19')]
    info = fl.population_info(gates)
    assert fs.default_anchor(info, gates) == 'Viable/Live'
    assert fs.default_anchor(info, gates[:1] + gates[2:]) == 'Single Cells/Single Cells'
    gates[2]['populations']['T cells']['anchor'] = True
    assert fs.default_anchor(info, gates) == 'T cells/T cells'
    assert fs.default_anchor(info) == 'root'

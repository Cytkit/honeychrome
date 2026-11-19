"""Tests for immune-state coherence and the state network (ag_coherence)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

_PLUGINS = os.path.join(os.path.dirname(__file__), '..', 'src', 'honeychrome', 'bundled_plugins')
sys.path.insert(0, os.path.abspath(_PLUGINS))

import ag_coherence as co  # noqa: E402
import ag_fsea as fs  # noqa: E402
import ag_fsea_library as fl  # noqa: E402
from test_ag_fsea import CD8_MARKERS, library_csv, synthetic_gates  # noqa: E402


def test_rank_corr_matches_spearman_without_gaps():
    X = np.random.default_rng(0).normal(size=(40, 5))
    C, n = co.rank_corr(X)
    expected = stats.spearmanr(X).statistic
    np.testing.assert_allclose(C, expected, atol=1e-12)
    assert (n == 40).all()


def test_coherence_signs_pairs_and_minimum():
    C = np.array([[1.0, 0.6, -0.4, 0.2],
                  [0.6, 1.0, -0.2, 0.0],
                  [-0.4, -0.2, 1.0, 0.0],
                  [0.2, 0.0, 0.0, 1.0]])
    allowed = np.ones((4, 4), bool)
    allowed[0, 3] = allowed[3, 0] = False
    sets = [(np.array([0, 1, 2]), np.array([1.0, 1.0, -1.0])),
            (np.array([0, 3]), np.array([1.0, 1.0]))]
    c, npairs = co.coherence(C, sets, allowed)
    assert c[0] == pytest.approx((0.6 + 0.4 + 0.2) / 3)
    assert npairs.tolist() == [3, 0] and np.isnan(c[1])


def test_structural_pairs_rules():
    pop_info = {
        'CD8/CD8': {'gate': 'CD8', 'parent_key': 'root'},
        'Mem/TEM': {'gate': 'Mem', 'parent_key': 'CD8/CD8'},
        'Mem/TCM': {'gate': 'Mem', 'parent_key': 'CD8/CD8'},
        'PD1 (CD8)/PD1+': {'gate': 'PD1 (CD8)', 'parent_key': 'CD8/CD8'},
        'PD1 (TEM)/PD1+': {'gate': 'PD1 (TEM)', 'parent_key': 'Mem/TEM'},
        'Ki67 (TEM)/Ki67+': {'gate': 'Ki67 (TEM)', 'parent_key': 'Mem/TEM'},
    }
    meta = pd.DataFrame([
        ('abundance:CD8', 'abundance', 'CD8/CD8', 'root', ''),
        ('abundance:TEM', 'abundance', 'Mem/TEM', 'CD8/CD8', ''),
        ('abundance:TCM', 'abundance', 'Mem/TCM', 'CD8/CD8', ''),
        ('subset:TEM', 'subset', 'Mem/TEM', 'CD8/CD8', ''),
        ('positivity:PD1 CD8', 'positivity', 'PD1 (CD8)/PD1+', 'CD8/CD8', 'PD-1'),
        ('positivity:PD1 TEM', 'positivity', 'PD1 (TEM)/PD1+', 'Mem/TEM', 'PD-1'),
        ('positivity:Ki67 TEM', 'positivity', 'Ki67 (TEM)/Ki67+', 'Mem/TEM', 'Ki67'),
    ], columns=['feature', 'type', 'pop_key', 'parent_key', 'marker']).set_index('feature')
    feats = list(meta.index)
    pairs = {(feats[i], feats[j]) for i, j in co.structural_pairs(feats, meta, pop_info)}
    assert pairs == {('abundance:CD8', 'abundance:TEM'), ('abundance:CD8', 'abundance:TCM'),
                     ('abundance:TEM', 'abundance:TCM'), ('abundance:TEM', 'subset:TEM'),
                     ('positivity:PD1 CD8', 'positivity:PD1 TEM')}


def test_adjust_background_removes_the_shared_shift():
    rng = np.random.default_rng(1)
    bg = rng.normal(size=60)
    R = pd.DataFrame({'pos:A': 2 * bg + rng.normal(0, 0.1, 60),
                      'pos:B': rng.normal(size=60)})
    B = pd.DataFrame({'med:1': bg + rng.normal(0, 0.05, 60), 'med:2': bg, 'med:3': bg})
    meta = pd.DataFrame({'type': ['positivity', 'positivity'], 'marker': ['CD38', 'CD39']},
                        index=['pos:A', 'pos:B'])
    out, done = co.adjust_background(R, B, {'CD38': ['med:1', 'med:2', 'med:3']}, meta)
    assert done == {'pos:A'}
    assert abs(np.corrcoef(out['pos:A'], bg)[0, 1]) < 0.05
    pd.testing.assert_series_equal(out['pos:B'], R['pos:B'])


def test_reliability_is_low_for_rare_populations():
    rng = np.random.default_rng(2)
    n = 50
    real = rng.normal(0, 0.5, n)
    big_d = np.full(n, 50000.0)
    big_c = rng.binomial(50000, 1 / (1 + np.exp(-(-1 + real))))
    tiny_d = rng.integers(1, 6, n).astype(float)
    tiny_c = rng.binomial(tiny_d.astype(int), 1 / (1 + np.exp(-(-1 + real)))).astype(float)
    c = np.column_stack([big_c, tiny_c]).astype(float)
    d = np.column_stack([big_d, tiny_d])
    y = np.log2((c + 0.5) / (d - c + 0.5))
    rel = co.reliability(y - y.mean(axis=0), co.log2_odds_noise(c, d))
    assert rel[0] > 0.95 and rel[1] < 0.3


def test_paired_relabelling_keeps_each_pair_split():
    class Recorder:
        def __init__(self):
            self.seen = []

        def compute(self, Za, Zb):
            self.seen.append((set(Za[:, 0].astype(int)), set(Zb[:, 0].astype(int))))
            return {'dC': np.zeros(1), 'dz': np.zeros((2, 2))}

    Z = np.arange(10, dtype=float)[:, None]
    rows_a, rows_b = np.arange(0, 5), np.arange(5, 10)
    pairs = [(0, 5), (1, 6), (2, 7), (3, 8)]
    rec = Recorder()
    out = co.relabel_test(Z, rows_a, rows_b, pairs, rec, 50, np.random.default_rng(3))
    for a, b in rec.seen:
        assert 4 in a
        for x, y in pairs:
            assert (x in a) != (y in a) and (x in b) != (y in b)
    assert out['p_dC'][0] == pytest.approx(1.0)
    assert co._perm_p(np.array([5.0]), np.zeros((99, 1)))[0] == pytest.approx(0.01)


def _coherent_summaries(n_per_group=25, donor_sd=(0.8, 0.0), seed=0):
    """CD8 positivity driven by a donor effect in group A only."""
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
            donor = rng.normal(0, donor_sd[gi])
            for g in gates[3:]:
                pop = next(iter(g['populations']))
                par = f"CD4 & CD8/{g['parent_popul']}"
                eta = -1.5 + rng.normal(0, 0.4)
                if g['parent_popul'] == 'CD8' and pop[:-1] in CD8_MARKERS:
                    eta += donor
                key = f"{g['gate_name']}/{pop}"
                n[key] = int(rng.binomial(n[par], 1 / (1 + np.exp(-eta))))
                parent_n[key] = n[par]
            rows = [{'key': k, 'n_events': v, 'n_parent': parent_n[k]} for k, v in n.items()]
            summaries[s] = {'populations': rows, 'markers': {}, 'n_total': 100000}
    return summaries, samples, groups, gates


def test_run_coherence_finds_a_planted_difference(tmp_path):
    import ag_fsea_plots as fp
    import ag_report

    summaries, samples, groups, gates = _coherent_summaries()
    lib, _sha, _m = fl.load_library(library_csv(tmp_path))
    res = fs.run_enrichment(summaries, samples, groups, gates, lib, group_order=['A', 'B'],
                            reference='A', n_perm=200, seed=5)
    pairing = {s: s[1:] for s in samples}
    kw = dict(group_order=['A', 'B'], reference='A', pairing=pairing, n_perm=300, n_boot=100,
              seed=7)
    block = co.run_coherence(summaries, samples, groups, gates, res, **kw)
    coh = block['coherence'].set_index('immune_state')
    planted = coh.loc['Planted']
    assert planted['group_a'] == 'B' and planted['group_b'] == 'A' and bool(planted['paired'])
    assert planted['C_b'] > 0.5 and abs(planted['C_a']) < 0.2
    assert planted['dC'] < -0.4 and planted['FDR'] < 0.05
    assert planted['C_b_lo'] < planted['C_b'] < planted['C_b_hi']
    assert not (coh.loc['Null', 'FDR'] < 0.05)
    again = co.run_coherence(summaries, samples, groups, gates, res, **kw)
    pd.testing.assert_frame_equal(block['coherence'], again['coherence'])

    res['coherence'] = block
    back = fs.result_from_jsonable(fs.result_to_jsonable(res))
    pd.testing.assert_frame_equal(back['coherence']['coherence'], block['coherence'],
                                  check_dtype=False)
    assert {'coherence', 'state_network', 'coherence_members'} <= set(fs.result_tables(back))

    view = co.pair_view(back['coherence'], 'B vs A', 'Planted', 'Null')
    assert set(view['groups']) == {'A', 'B'} and len(view['features']) == 12
    for fig in (fp.make_coherence_figure(back['coherence'], 'B vs A', 0.05),
                fp.make_state_network_figure(back['coherence'], 'B vs A', 0.05),
                fp.make_state_pair_figure(view, 'Planted', 'Null', 'B vs A', ('A', 'B'))):
        fig.canvas.draw()
    labels = [i.label for i in ag_report.coherence_report_items(back['coherence'], 0.05)]
    assert labels[0] == 'Coherence: B vs A'

    with pytest.raises(co.CoherenceBlocked):
        co.run_coherence(summaries, samples, groups, gates, res, min_group_samples=30, **kw)


@pytest.mark.ui
def test_enrichment_page_runs_and_shows_coherence(qtbot, tmp_path):
    pytest.importorskip('pytestqt')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from types import SimpleNamespace
    import ag_fsea_ui
    import automated_gating_tab as agt
    from honeychrome.view_components.event_bus import EventBus

    agt._ensure_qt_imports()
    summaries, samples, groups, gates = _coherent_summaries()
    state = agt.GatingRunState(applied_model={'gate_definitions': gates}, summaries=summaries,
                               run_samples=samples, group_names=['A', 'B'], sample_groups=groups,
                               reference_group='A', fsea_n_perm=200)
    controller = SimpleNamespace(
        experiment=SimpleNamespace(settings={'unmixed': {'event_channels_pnn': []}}, process={},
                                   samples={}),
        experiment_dir='', unmixed_transformations={})
    tab = ag_fsea_ui.EnrichmentTab(state, EventBus(), controller, samples_fn=lambda: samples,
                                   design_fn=lambda s: agt.statistics_design(state, s),
                                   antigens_fn=dict, experiment_dir_fn=lambda: '')
    qtbot.addWidget(tab)
    tab.refresh()
    assert tab.chk_coherence.isChecked() and tab.chk_background.isChecked()
    assert tab.sb_reliability.value() == pytest.approx(co.DEFAULT_MIN_RELIABILITY)
    tab._library, tab._library_sha, _m = fl.load_library(library_csv(tmp_path))
    with qtbot.waitSignal(tab.enrichment_finished, timeout=120000):
        tab._on_run()
    block = state.fsea['coherence']
    assert block['coherence']['comparison'].unique().tolist() == ['B vs A']
    assert tab.tbl_coherence.rowCount() == 2
    tabs = [tab.fig_tabs.tabText(i) for i in range(tab.fig_tabs.count())]
    assert tabs == ['Dot plot', 'Coherence', 'State network', 'State pair']
    assert tab.cb_pair.count() >= 1
    tab.show_state('Planted')
    tabs = [tab.fig_tabs.tabText(i) for i in range(tab.fig_tabs.count())]
    assert tabs[:4] == ['Dot plot', 'Coherence', 'State network', 'State pair']
    assert tabs[4:] == ['Running sum', 'Members']

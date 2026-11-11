"""
test_anova_stats.py
-------------------
The random individual effect (moderated_stats.block_correlation,
gls_lm_fit, variance_components), its use in differential_stats, and the
variance table (anova_stats.run_anova), against reference values from R:
limma 3.58 (duplicateCorrelation, lmFit with block and correlation, eBayes)
and lme4 1.1-35 (REML fits, as variancePartition makes them).

Usage:
    pytest tests/test_anova_stats.py
"""

import numpy as np
import pandas as pd
import pytest

from honeychrome.controller_components import anova_stats
from honeychrome.controller_components import differential_stats as ds
from honeychrome.controller_components import moderated_stats as ms

FEATURES = [0, 5, 11]


def _batf_layout(G=40, seed=7):
    """8 mice (5 KO, 3 WT) x 4 tissues; a mouse effect and a tissue effect,
    no genotype effect; a few missing values."""
    rng = np.random.default_rng(seed)
    mice = [f"m{i}" for i in range(8)]
    geno = ['KO'] * 5 + ['WT'] * 3
    tissues = ['LN', 'Liver', 'MLN', 'Spleen']
    mouse = [m for m in mice for _t in tissues]
    tissue = [t for _m in mice for t in tissues]
    group = [g for g in geno for _t in tissues]
    b = rng.normal(0, 0.7, (G, 8))
    tis = rng.normal(0, 1, (G, 4))
    Y = (tis[:, [tissues.index(t) for t in tissue]] + b[:, [mice.index(m) for m in mouse]]
         + rng.normal(0, 0.8, (G, 32)))
    Y[:5, 3] = np.nan
    Y[5:8, 10] = np.nan
    return Y, mouse, tissue, group


@pytest.fixture(scope='module')
def batf():
    Y, mouse, tissue, group = _batf_layout()
    samples = [f"{m}_{t}" for m, t in zip(mouse, tissue)]
    data = pd.DataFrame(Y.T, index=samples, columns=[f"f{i}" for i in range(Y.shape[0])])
    cov = pd.DataFrame({'Tissue': tissue}, index=samples)
    return data, mouse, tissue, group, cov


def _main_design(group, tissue):
    levels = sorted(set(tissue))[1:]
    return np.column_stack([np.ones(len(group)), [1.0 if g == 'WT' else 0.0 for g in group]]
                           + [[1.0 if t == lvl else 0.0 for t in tissue] for lvl in levels])


def test_block_correlation_and_gls_fit_match_limma(batf):
    data, mouse, tissue, group, _cov = batf
    Y = data.to_numpy().T
    X = _main_design(group, tissue)
    bc = ms.block_correlation(Y, X, mouse)
    assert bc.consensus == pytest.approx(0.3927367116, abs=1e-6)
    # limma stops its REML iterations (statmod, maxit = 20) a little short
    # of the maximum for some features
    assert bc.rho[FEATURES] == pytest.approx([0.18239359, 0.62047422, 0.12663117], abs=1e-5)
    fit = ms.e_bayes(ms.gls_lm_fit(Y, X, mouse, bc.consensus))
    assert fit.t[FEATURES, 1] == pytest.approx([0.46290192, 0.63313062, 1.22648851], rel=1e-5)
    assert fit.df_prior == pytest.approx(523.02118125, rel=1e-4)
    assert fit.s2_prior == pytest.approx(1.03056692, rel=1e-5)


def test_run_moderated_with_block_matches_the_gls_fit(batf):
    data, mouse, tissue, group, cov = batf
    res = ds.run_moderated(data, group, [('KO', 'WT')], 'reference', 0.05, 0.0,
                           covariates=cov, use_treat=False, block_vec=mouse)
    assert res['t'].to_numpy()[FEATURES] == pytest.approx([0.46290192, 0.63313062, 1.22648851],
                                                          rel=1e-5)
    assert any('random effect' in note for note in res.attrs['block_notes'])


def test_auto_pairing_is_random_when_individuals_are_nested_in_groups(batf):
    _data, mouse, _tissue, group, _cov = batf
    assert ds.resolve_pairing(group, mouse, 'auto')[2] == 'random'
    crossed = ['A', 'B'] * 4
    assert ds.resolve_pairing(crossed, ['d1', 'd1', 'd2', 'd2', 'd3', 'd3', 'd4', 'd4'],
                              'auto')[2] == 'fixed'
    assert ds.resolve_pairing(group, mouse, 'fixed')[2] == 'fixed'
    assert ds.resolve_pairing(group, None, 'auto') == (None, None, '')


def test_variance_table_matches_limma_and_lme4(batf):
    data, mouse, _tissue, group, cov = batf
    table = anova_stats.run_anova(data, group, block_vec=mouse, individual='Mouse',
                                  covariates=cov, group_label='Genotype')
    assert table.attrs['settings']['terms'] == ['Genotype', 'Tissue', 'Genotype × Tissue',
                                                'Mouse', 'Residuals']
    assert table.attrs['settings']['correlation'] == pytest.approx(0.3931212490, abs=1e-6)

    def term(name, column):
        rows = table[table['term'] == name].set_index('feature')
        return rows[column].to_numpy()[FEATURES]

    assert term('Genotype', 'statistic') == pytest.approx([0.21449261, 0.39217712, 1.46568401],
                                                          rel=1e-5)
    assert term('Tissue', 'statistic') == pytest.approx([16.16828830, 7.97407931, 11.96756256],
                                                        rel=1e-5)
    assert term('Genotype × Tissue', 'statistic') == pytest.approx(
        [1.61360654, 1.10891997, 0.35008723], rel=1e-5)
    assert term('Mouse', 'statistic') == pytest.approx([0.94982323, 11.00913658, 0.31526277],
                                                       rel=1e-5, abs=1e-6)
    shares = np.column_stack([term(n, 'variance.pct') for n in
                              ('Genotype', 'Tissue', 'Genotype × Tissue', 'Mouse', 'Residuals')])
    lme4 = 100 * np.array([[0.000000, 0.523040, 0.007865, 0.073584, 0.395512],
                           [0.000000, 0.344706, 0.049476, 0.375751, 0.230067],
                           [0.099488, 0.529225, 0.000000, 0.047017, 0.324270]])
    assert shares == pytest.approx(lme4, abs=0.01)
    totals = table.groupby('feature')['variance.pct'].sum()
    assert totals.to_numpy() == pytest.approx(100.0)


def test_variance_table_with_fixed_pairing_tests_the_individual():
    rng = np.random.default_rng(3)
    donors = [f"d{i}" for i in range(6) for _k in range(2)]
    group = ['before', 'after'] * 6
    data = pd.DataFrame(rng.normal(size=(12, 10)), columns=[f"f{i}" for i in range(10)])
    table = anova_stats.run_anova(data, group, pairing_vec=donors, individual='Donor')
    donor = table[table['term'] == 'Donor']
    assert set(donor['kind']) == {'fixed'}
    assert (donor['df1'] == 5).all()
    assert table.attrs['settings']['individual_mode'] == 'fixed'

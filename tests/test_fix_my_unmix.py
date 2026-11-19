"""
test_fix_my_unmix.py
--------------------
Fix My Unmix numerics (controller_components/fix_my_unmix.py) on the
synthetic panel of fmu_synthetic.py, with the main app's per-cell AF as the
background. Ported from the Fix My Unmix plugin's tests; the synthetic
recovery test, the benchmark and the scoring tables stay in the plugin.
"""

import numpy as np
import pytest

from honeychrome.controller_components import fix_my_unmix as fmc
from honeychrome.controller_components.autospectral_functions import apply_af_unmixing

from fmu_synthetic import N_DET, af_library, coexpressed_experiment, experiment, library_for, panel, projections, run


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def test_spread_thresholds_flat_and_widened():
    u = np.array([[100.0, 0.0], [0.0, 400.0]])
    thr = np.array([10.0, 20.0])
    flat = fmc.spread_thresholds(u, thr, None, 2.0)
    assert np.allclose(flat, [[10, 20], [10, 20]])
    spread = np.array([[0.0, 4.0], [1.0, 0.0]])
    upper = fmc.spread_thresholds(u, thr, spread, 2.0)
    assert np.allclose(upper, [[10.0, 20.0 + 2 * np.sqrt(400.0)], [10.0 + 2 * np.sqrt(400.0), 20.0]])
    lower = fmc.spread_thresholds(u, -thr, spread, 2.0, side='lower')
    assert np.allclose(lower, [[-10.0, -20.0 - 40.0], [-10.0 - 40.0, -20.0]])


def test_align_spread_cleans_entries():
    raw = np.array([[np.nan, -1.0], [2.0, 5.0]])
    assert np.array_equal(fmc.align_spread(raw, 2), [[0.0, 0.0], [2.0, 0.0]])


def test_half_sample_mode_ignores_the_positive_tail():
    rng = np.random.default_rng(20)
    x = np.concatenate([rng.normal(100.0, 50.0, 20_000), rng.lognormal(np.log(30_000), 0.8, 8000)])
    assert fmc.half_sample_mode(x) == pytest.approx(100.0, abs=15.0)
    assert np.isnan(fmc.half_sample_mode(np.array([np.nan])))


def test_negative_bound_mirrors_the_lower_tail():
    rng = np.random.default_rng(21)
    neg = rng.normal(0.0, 1000.0, 40_000)
    x = np.concatenate([neg, rng.lognormal(np.log(50_000), 0.5, 5000)])
    t_event = np.full(len(x), 20_000.0)
    bound = fmc.negative_bound(x, t_event, 0.001, rng)
    # the 0.1% upper tail of N(0, 1000) mirrored from the lower tail
    assert bound == pytest.approx(3090.0, rel=0.1)
    assert fmc.negative_bound(x[:50], t_event[:50], 0.001, rng) == -np.inf


def test_smoothstep_ends_and_midpoint():
    v = fmc.smoothstep(np.array([0.0, 50.0, 150.0, 250.0, 900.0]), 50, 250)
    assert np.allclose(v, [0.0, 0.0, 0.5, 1.0, 1.0])


def test_af_likeness_is_scale_free_and_sign_blind():
    lib = af_library()
    assert fmc.af_likeness(0.01 * lib[1], lib) == pytest.approx(1.0)
    assert fmc.af_likeness(-lib[2], lib) == pytest.approx(1.0)
    truth, _ = panel()
    assert fmc.af_likeness(truth[7] - truth[6], lib) < 0.3
    assert fmc.af_likeness(np.zeros(N_DET), lib) == 0.0


def test_anchored_slope_recovers_known_spillover():
    rng = np.random.default_rng(1)
    n = 30_000
    x = np.concatenate([rng.normal(0, 200, n // 2), rng.lognormal(np.log(50_000), 0.8, n // 2)])
    target_on = rng.random(n) < 0.2
    y = -0.06 * x + rng.normal(0, 200, n) + np.where(target_on, rng.lognormal(np.log(30_000), 0.5, n), 0.0)
    s = fmc.FixSettings()
    t_target = np.full((n, 1), 600.0)
    slope, count, se = fmc.anchored_slopes(x, y[:, None], t_target, np.full(n, 600.0), np.zeros(1),
                                           np.random.default_rng(0), s, -t_target)
    assert count[0] >= s.min_negative_events
    assert slope[0] == pytest.approx(-0.06, abs=0.003)
    assert 0 < se[0] < 0.003


@pytest.mark.parametrize('band', [False, True])
def test_kernel_matches_numpy_without_subsampling(band):
    if not fmc.KERNEL_AVAILABLE:
        pytest.skip('compiled kernel not built')
    rng = np.random.default_rng(4)
    n, m = 8_000, 5
    x = np.concatenate([rng.normal(0, 200, n // 2), rng.lognormal(np.log(50_000), 0.8, n // 2)])
    slopes = np.array([-0.06, 0.02, 0.0, 0.1, -0.01])
    y = np.outer(x, slopes) + rng.normal(0, 200, (n, m))
    y += np.where(rng.random((n, m)) < 0.2, rng.lognormal(np.log(30_000), 0.5, (n, m)), 0.0)
    upper = np.full((n, m), 600.0)
    lower = -upper if band else None
    args = (x, y, upper, np.full(n, 600.0), np.zeros(m))
    k_slope, k_n, k_se = fmc.anchored_slopes(*args, np.random.default_rng(0), fmc.FixSettings(), lower)
    n_slope, n_n, n_se = fmc.anchored_slopes(*args, np.random.default_rng(0), fmc.FixSettings(use_kernel=False),
                                             lower)
    assert np.array_equal(k_n, n_n)
    assert np.allclose(k_slope, n_slope, rtol=0, atol=1e-10, equal_nan=True)
    assert np.allclose(k_se, n_se, rtol=1e-8, atol=1e-12, equal_nan=True)
    # 0.1 puts every bright event above the flat boundary from a zero start: not identifiable
    assert np.isnan(k_slope[3])
    found = np.isfinite(k_slope)
    assert np.allclose(k_slope[found], slopes[found], atol=0.005)


def test_anchored_slope_is_nan_without_events():
    s = fmc.FixSettings()
    x = np.linspace(0, 1000, 50)
    slope, count, se = fmc.anchored_slopes(x, x[:, None], np.zeros((50, 1)), np.zeros(50), np.zeros(1),
                                           np.random.default_rng(0), s)
    assert np.isnan(slope[0]) and np.isnan(se[0]) and count[0] == 0


def test_extract_raw_signature_recovers_exact_row():
    rng = np.random.default_rng(2)
    truth, _names = panel()
    n = 4000
    abundance = np.zeros((n, truth.shape[0]))
    abundance[:, 2] = rng.lognormal(np.log(40_000), 0.8, n)
    abundance[:, 5] = rng.lognormal(np.log(20_000), 0.8, n) * (rng.random(n) < 0.5)
    raw = abundance @ truth + 300.0
    out = fmc.extract_raw_signature(raw, truth, abundance, 2, n_levels=60)
    assert out is not None
    assert out['stats']['joint']
    assert np.allclose(out['signature'], truth[2], rtol=0, atol=1e-7)
    assert out['stats']['explained_total'] == pytest.approx(1.0, abs=1e-6)
    assert out['stats']['resid_rel'] < 1e-6


def test_extract_raw_signature_needs_enough_bins():
    truth, _ = panel()
    with pytest.raises(ValueError):
        fmc.extract_raw_signature(np.zeros((500, N_DET)), truth, np.zeros((500, 8)), 0, n_levels=5)
    assert fmc.extract_raw_signature(np.zeros((100, N_DET)), truth, np.zeros((100, 8)), 0) is None


def test_variant_leakage_prior_shapes_and_fallback():
    rng = np.random.default_rng(3)
    truth, _ = panel()
    deltas = [rng.normal(0, 0.01, (6, N_DET)) for _ in range(truth.shape[0])]
    deltas[1] = None
    prior = fmc.variant_leakage_prior(truth, deltas)
    assert prior['variance'].shape == (8, 8)
    assert np.all(np.isnan(np.diag(prior['variance'])))
    assert prior['fallback'] == [1]
    off = ~np.eye(8, dtype=bool)
    assert np.all(prior['variance'][off] >= 0)
    assert fmc.variant_leakage_prior(truth, [None] * 8) is None


def test_stratified_sample_keeps_small_populations():
    rng = np.random.default_rng(4)
    n = 20_000
    abundance = np.zeros((n, 3))
    abundance[:100, 0] = 1000.0
    abundance[100:10_100, 1] = 1000.0
    tm = np.full((n, 3), 10.0)
    keep = fmc.stratified_sample(abundance, tm, 5000, 0.3, 2000, rng)
    assert len(np.unique(keep)) == len(keep)
    assert set(range(100)) <= set(keep.tolist())
    assert len(keep) <= 5000 + 1



# ---------------------------------------------------------------------------
# Phase-one pieces
# ---------------------------------------------------------------------------

def _bulk_confounded(rng, n_bright: int = 120, n_bulk: int = 20_000, t: float = 15_000.0):
    """
    A small positive population with true slope -0.05 into the target, and a
    bulk whose source and target noise share a +0.4 covariance slope (as a
    third dye's spread, collinearity or an AF residual gives).
    """
    common = rng.normal(0, 3000, n_bulk)
    x_bulk = common + rng.normal(0, 300, n_bulk)
    y_bulk = 0.4 * common + rng.normal(0, 300, n_bulk)
    x_bright = rng.lognormal(np.log(60_000), 0.5, n_bright)
    y_bright = -0.05 * x_bright + rng.normal(0, 300, n_bright)
    x = np.concatenate([x_bulk, x_bright])
    y = np.concatenate([y_bulk, y_bright])
    t_source = np.full(len(x), t)
    t_target = np.full((len(x), 1), 15_000.0)
    return x, y[:, None], t_source, t_target


def test_anchored_slope_ignores_bulk_covariance():
    rng = np.random.default_rng(11)
    x, Y, ts, tt = _bulk_confounded(rng)
    every, _se = fmc._huber_slope_origin(x, Y[:, 0], 0.0, 1.345, 100, 1e-4)
    anchored, n, se = fmc.anchored_slopes(x, Y, tt, ts, np.zeros(1), np.random.default_rng(0), fmc.FixSettings())
    assert every > 0.0
    assert anchored[0] == pytest.approx(-0.05, abs=0.003)
    assert n[0] >= 100 and 0 < se[0] < 0.01


def test_negative_bound_keeps_the_tilted_tail_out_of_the_fit():
    rng = np.random.default_rng(22)
    # a narrow unstained boundary sends the tilted tail of the stained negatives into the "positive" set
    x, Y, ts, tt = _bulk_confounded(rng, t=1000.0)
    s = fmc.FixSettings()
    naive, _n, _se = fmc.anchored_slopes(x, Y, tt, ts, np.zeros(1), np.random.default_rng(0), s)
    bound = fmc.negative_bound(x, ts, s.negative_tail, rng)
    clear = np.maximum(ts, bound)
    fixed, n, _se = fmc.anchored_slopes(x, Y, tt, clear, np.zeros(1), np.random.default_rng(0), s)
    assert naive[0] > 0.1
    assert 5000 < bound < 15_000
    assert fixed[0] == pytest.approx(-0.05, abs=0.005)
    assert n[0] <= 150                                     # the bright 120 and the bulk's far tail


def test_band_trims_both_tails_of_cell_to_cell_variation():
    rng = np.random.default_rng(12)
    n_bulk, n_pos = 5000, 20_000
    x_pos = rng.lognormal(np.log(80_000), 0.4, n_pos)
    # multiplicative cell-to-cell variation in the target, far wider than the boundary assumes
    y_pos = x_pos * rng.normal(0.0, 0.05, n_pos)
    x = np.concatenate([rng.normal(0, 300, n_bulk), x_pos])
    Y = np.concatenate([rng.normal(0, 300, n_bulk), y_pos])[:, None]
    ts = np.full(len(x), 1000.0)
    upper = np.full((len(x), 1), 1000.0) + 2.0 * np.sqrt(np.maximum(x, 0) * 2.0)[:, None]
    lower = -upper
    s = fmc.FixSettings()
    one_sided, _n, _se = fmc.anchored_slopes(x, Y, upper, ts, np.zeros(1), np.random.default_rng(0), s)
    band, _n, _se = fmc.anchored_slopes(x, Y, upper, ts, np.zeros(1), np.random.default_rng(0), s, lower)
    assert one_sided[0] < -0.03
    assert abs(band[0]) < 0.5 * abs(one_sided[0])


def test_source_evidence_discounts_the_unstained_rate():
    rng = np.random.default_rng(23)
    n = 50_000
    s = fmc.FixSettings(source_dominant=False)
    thresholds = np.array([1000.0, 1000.0, 1000.0])

    def sample(n_pos):
        u = rng.normal(0, 300, (n, 3))
        u[:, 2] += np.where(rng.random(n) < 0.004, 5000.0, 0.0)      # spurious AF positives in both samples
        u[:n_pos[0], 0] = rng.lognormal(np.log(30_000), 0.4, n_pos[0])
        u[:n_pos[1], 1] = rng.lognormal(np.log(30_000), 0.4, n_pos[1])
        return u

    u = sample((40, 2000))
    u0 = sample((0, 0))
    ev = fmc.source_evidence(u, u0, thresholds, None, s, rng)
    assert ev['n_clear'][2] > 150                          # many above threshold ...
    assert ev['weight'][2] == 0.0                          # ... but no more than the unstained gives
    assert ev['weight'][0] == 0.0 and 0 < ev['excess'][0] < 50
    assert ev['weight'][1] == 1.0
    assert ev['t_clear'].shape == u.shape and ev['dominant'] is None


def test_glasso_row_selects_the_true_source():
    rng = np.random.default_rng(14)
    n = 6000
    X = rng.normal(0, 1000, (n, 6))
    X[:, 2] = 0.7 * X[:, 1] + rng.normal(0, 300, n)      # correlated with the culprit only
    y = 0.08 * X[:, 1] + rng.normal(0, 50, n)
    beta, lam = fmc.glasso_row(X, y, fmc.FixSettings())
    assert lam > 0
    assert beta[1] == pytest.approx(0.08, abs=0.002)
    assert np.all(np.abs(np.delete(beta, 1)) < 0.005)
    shrunk, _lam = fmc.glasso_row(X, y, fmc.FixSettings(glasso_refit=False))
    assert 0.03 < shrunk[1] < beta[1]



def test_residual_change_is_blind_to_in_span_error():
    truth, _names = panel()
    rng = np.random.default_rng(8)
    raw = rng.lognormal(np.log(10_000), 0.6, (2000, len(truth))) @ truth + rng.normal(0, 100, (2000, N_DET))
    in_span = truth.copy()
    in_span[3] = in_span[3] + 0.1 * truth[2]
    out_span = truth.copy()
    out_span[3] = np.roll(out_span[3], 1)
    assert abs(fmc.residual_change(raw, truth, in_span, np.random.default_rng(0))) < 1e-9
    assert fmc.residual_change(raw, truth, out_span, np.random.default_rng(0)) > 0.01


# ---------------------------------------------------------------------------
# fix_my_unmix


def test_af_likeness_without_af_is_zero():
    assert fmc.af_likeness(np.ones(N_DET), np.zeros((0, N_DET))) == 0.0


# ---------------------------------------------------------------------------
# Background: the main app's per-cell AF
# ---------------------------------------------------------------------------

def test_project_events_is_the_main_app_per_cell_af():
    truth, _wrong, names, unstained, stained, _ab = experiment(n=3000)
    lib = library_for(truth, names, unstained)
    proj = fmc.project_events(stained, truth, lib['precomputed'], lib['spectra'])
    ref = apply_af_unmixing(stained, lib['precomputed'], lib['spectra'])
    assert np.array_equal(proj['abundance'], ref['unmixed'])
    assert np.array_equal(proj['af_idx'], ref['af_idx'] - 1)
    expected = stained - ref['af_scale'][:, None] * lib['spectra'][ref['af_idx'] - 1]
    assert np.allclose(proj['residual'], expected)


def test_project_events_without_af_is_ols():
    truth, _wrong, _names, _unstained, stained, _ab = experiment(n=1000)
    proj = fmc.project_events(stained, truth)
    assert np.allclose(proj['abundance'], fmc.unmix_ols(stained, truth))
    assert np.array_equal(proj['residual'], stained)
    assert np.all(proj['af_idx'] == -1) and not proj['af_scale'].any()


def test_concatenate_offsets_af_rows_into_the_stacked_library():
    truth, _wrong, names, unstained, stained, _ab = experiment(n=2000)
    lib = library_for(truth, names, unstained)
    a = fmc.project_events(stained[:1000], truth, lib['precomputed'], lib['spectra'])
    b = fmc.project_events(stained[1000:], truth, lib['precomputed'], lib['spectra'])
    c = fmc.project_events(stained[:10], truth)
    joined = fmc.concatenate([a, b, c], [0, 3, 0])
    assert joined['abundance'].shape == (2010, 8)
    assert np.array_equal(joined['af_idx'][:1000], a['af_idx'])
    assert np.array_equal(joined['af_idx'][1000:2000], b['af_idx'] + 3)
    assert np.all(joined['af_idx'][2000:] == -1)
    bg = fmc.background_for([lib['spectra'], None, lib['spectra'][1:]], N_DET)
    assert bg.n_bg == 1 and np.array_equal(bg.basis[0], lib['spectra'][0])
    assert bg.af_spectra.shape == (5, N_DET)
    empty = fmc.background_for([None], N_DET)
    assert empty.n_bg == 0 and empty.af_spectra.shape == (0, N_DET)


# ---------------------------------------------------------------------------
# fix_my_unmix
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def single_error():
    truth, wrong, names, unstained, stained, _ab = experiment()
    res = run(wrong, names, unstained, stained, truth=truth)
    return truth, wrong, names, res


def test_single_injected_error_is_recovered(single_error):
    _truth, _wrong, _names, res = single_error
    rec = {r['fluorophore']: r for r in res.recovery}
    assert rec['Dye 4']['deg_start'] > 1.0
    assert rec['Dye 4']['deg_phase1'] < 0.3 * rec['Dye 4']['deg_start']
    assert rec['Dye 4']['deg_after'] < 0.2 * rec['Dye 4']['deg_start']
    assert rec['Dye 4']['accepted']


def test_correct_rows_stay_put(single_error):
    _truth, _wrong, _names, res = single_error
    for r in res.recovery:
        if r['fluorophore'] != 'Dye 4':
            assert r['deg_phase1'] < 0.5
            assert r['deg_after'] < 0.5


def test_result_logs_are_complete(single_error):
    _truth, _wrong, names, res = single_error
    assert len(res.coefficient_log) == len(names) * (len(names) - 1)
    assert len(res.source_log) == len(names)
    assert len(res.signature_log) == len(names)
    assert res.convergence_log and res.convergence_log[0]['iter'] == 1
    assert np.allclose(np.diag(res.spillover), 1.0)
    assert np.allclose(res.spillover @ res.compensation, np.eye(len(names)), atol=1e-10)
    assert res.dominant_counts.sum() == res.n_events
    assert any(np.isfinite(r['se_fit']) for r in res.coefficient_log)
    # the log is the last iteration's; the settled rule's confirming iteration may have no candidate for the lasso
    assert any(np.isfinite(r['slope_glasso']) for r in res.coefficient_log) or \
        not any(r['trust'] > 0 for r in res.coefficient_log)
    assert res.af_assignment and sum(r['n_events'] for r in res.af_assignment) == res.n_events_fit
    assert res.settings['convergence_quantile'] == 1.0


def test_only_flagged_rows_are_remeasured_by_default(single_error):
    _truth, _wrong, names, res = single_error
    reasons = {r['fluorophore']: r['reason'] for r in res.signature_log}
    assert reasons['Dye 4'] == 'accepted'
    assert sum(r == 'not.flagged' for r in reasons.values()) >= len(names) - 3
    moved = dict(zip(names, fmc.row_angle(res.spectra_start, res.spectra)))
    assert max(m for n, m in moved.items() if n != 'Dye 4') < 0.2


def test_flagged_rows_only_are_remeasured():
    truth, wrong, names, unstained, stained, _ab = experiment()
    res = run(wrong, names, unstained, stained, fmc.FixSettings(phase_two_rows='flagged'), truth=truth)
    reasons = {r['fluorophore']: r['reason'] for r in res.signature_log}
    assert reasons['Dye 4'] == 'accepted'
    assert sum(r == 'not.flagged' for r in reasons.values()) >= len(names) - 3


def test_backsolved_adoption_takes_the_phase_one_row():
    truth, wrong, names, unstained, stained, _ab = experiment()
    adopt = run(wrong, names, unstained, stained, fmc.FixSettings(phase_two_adopt='backsolved'), truth=truth)
    f = names.index('Dye 4')
    assert adopt.accepted[f]
    assert fmc.row_angle(adopt.spectra[f], adopt.spectra_backsolved[f])[0] < 0.05


def test_no_error_gives_identity():
    truth, _wrong, names, unstained, stained, _ab = experiment(injections=())
    res = run(truth, names, unstained, stained, truth=truth)
    off = ~np.eye(len(names), dtype=bool)
    assert np.max(np.abs(res.spillover[off])) < 0.01
    assert max(r['deg_after'] for r in res.recovery) < 0.5


def test_af_row_of_the_panel_is_left_out():
    truth, wrong, names, unstained, stained, _ab = experiment(n=6000)
    pu, ps, bg = projections(wrong, names, unstained, stained)
    spectra = np.vstack([wrong, af_library()[0]])
    res = fmc.fix_my_unmix(spectra, names + ['AF'], pu, ps, bg, fmc.FixSettings(max_iter=2))
    assert res.fluor_names == names
    assert res.spectra.shape == wrong.shape


def test_inputs_are_checked():
    truth, wrong, names, unstained, stained, _ab = experiment(n=3000)
    pu, ps, bg = projections(wrong, names, unstained, stained)
    with pytest.raises(ValueError):
        fmc.fix_my_unmix(wrong[:7], names[:7], pu, ps, bg)              # unmixed with other fluorophores
    with pytest.raises(ValueError):
        fmc.fix_my_unmix(wrong[:1], names[:1], pu, ps, bg)              # one fluorophore
    with pytest.raises(ValueError):
        fmc.fix_my_unmix(wrong, names, pu, ps, fmc.background_for([af_library()[:, :30]], 30))


def test_without_af_the_design_has_no_background_row():
    truth, wrong, names, unstained, stained, _ab = experiment(n=8000)
    pu = fmc.project_events(unstained, wrong)
    ps = fmc.project_events(stained, wrong)
    res = fmc.fix_my_unmix(wrong, names, pu, ps, fmc.background_for([None], N_DET), fmc.FixSettings(max_iter=2))
    assert res.background.n_bg == 0
    assert res.af_assignment == []


def test_downsample_keeps_full_event_count():
    truth, wrong, names, unstained, stained, _ab = experiment(n=12_000)
    res = run(wrong, names, unstained, stained, fmc.FixSettings(downsample=5000, max_iter=2))
    assert res.n_events == 12_000
    assert res.n_events_fit <= 5001


@pytest.mark.parametrize('bad', [dict(hypernegative_scope='other'), dict(n_levels=5), dict(residual_gate='other'),
                                 dict(phase_two_rows='other'), dict(hypernegative_rule='other'),
                                 dict(phase_two_adopt='other'), dict(source_events_low=500),
                                 dict(negative_tail=0.0), dict(phase_two_fallback='other'), dict(stop_rule='other'),
                                 dict(stall_rule='other')])
def test_settings_are_validated(bad):
    truth, wrong, names, unstained, stained, _ab = experiment(n=3000)
    with pytest.raises(ValueError):
        run(wrong, names, unstained, stained, fmc.FixSettings(**bad))


def test_af_gate_drops_steps_that_look_like_af():
    truth, wrong, names, unstained, stained, _ab = experiment()
    every = run(wrong, names, unstained, stained, fmc.FixSettings(af_gate=True, af_gate_max=0.0, max_iter=2),
                truth=truth)
    assert np.allclose(every.spillover, np.eye(len(names)))
    assert all(r['af_gated'] for r in every.source_log if r['af_likeness'] > 0)


def test_rare_sources_get_no_trust():
    truth, wrong, names, unstained, stained, ab = experiment(n=20_000)
    keep = (ab[:, 4] == 0) | (np.cumsum(ab[:, 4] > 0) <= 40)          # 40 Dye 4 cells left
    res = run(wrong, names, unstained, stained[keep], fmc.FixSettings(max_iter=3), truth=truth)
    src = {r['source']: r for r in res.source_log}
    assert src['Dye 4']['source_weight'] == 0.0
    assert all(r['trust'] == 0.0 for r in res.coefficient_log if r['source'] == 'Dye 4')


def test_joint_l2_background_also_recovers():
    truth, wrong, names, unstained, stained, _ab = experiment()
    pu, ps, bg = projections(wrong, names, unstained, stained, envelope=False)
    res = fmc.fix_my_unmix(wrong, names, pu, ps, bg, true_spectra=truth)
    rec = {r['fluorophore']: r for r in res.recovery}
    assert rec['Dye 4']['deg_after'] < 0.3 * rec['Dye 4']['deg_start']


def test_kernel_and_numpy_give_the_same_run():
    if not fmc.KERNEL_AVAILABLE:
        pytest.skip('compiled kernel not built')
    truth, wrong, names, unstained, stained, _ab = experiment(n=8000)
    pu, ps, bg = projections(wrong, names, unstained, stained)
    k = fmc.fix_my_unmix(wrong, names, pu, ps, bg, fmc.FixSettings(max_iter=3, max_truncated_events=10**7))
    n = fmc.fix_my_unmix(wrong, names, pu, ps, bg, fmc.FixSettings(max_iter=3, max_truncated_events=10**7,
                                                                  use_kernel=False))
    assert np.allclose(k.spillover, n.spillover, atol=1e-9)
    assert np.allclose(k.spectra, n.spectra, atol=1e-9)


# ---------------------------------------------------------------------------
# Options for the noise and bead-cell findings (FMU_NOISE_STUDY.md, FMU_BEAD_CELL_BENCHMARK.md)
# ---------------------------------------------------------------------------

def test_warm_lasso_sees_a_source_its_spillover_lifts_above_the_boundary():
    rng = np.random.default_rng(0)
    n = 20_000
    u = rng.normal(0.0, 100.0, (n, 3))
    pos = rng.random(n) < 0.3
    u[pos, 0] += rng.uniform(2000.0, 50_000.0, pos.sum())
    u[pos, 1] += 0.2 * u[pos, 0]                    # every positive event is above the target's boundary
    tm, thr, nv = np.full((n, 3), 300.0), np.full(3, 300.0), np.full(3, 100.0 ** 2)
    s = fmc.FixSettings()
    assert fmc.glasso_matrix(u, tm, thr, nv, np.random.default_rng(1), s)[0, 1] == 0.0
    warm = np.zeros((3, 3))
    warm[0, 1] = 0.19
    assert abs(fmc.glasso_matrix(u, tm, thr, nv, np.random.default_rng(1), s, warm)[0, 1] - 0.2) < 0.005


def test_stop_rules_judge_every_trusted_or_applied_error():
    truth, wrong, names, unstained, stained, _ab = experiment(n=8000)
    runs = {rule: run(wrong, names, unstained, stained, fmc.FixSettings(max_iter=1, stop_rule=rule))
            for rule in fmc.STOP_RULES}
    delta = {rule: r.convergence_log[0]['delta_max'] for rule, r in runs.items()}
    assert delta['applied'] <= delta['trusted'] <= delta['all']
    assert np.array_equal(runs['applied'].spillover, runs['all'].spillover)
    assert np.array_equal(runs['trusted'].spillover, runs['all'].spillover)


def test_fallback_reject_refuses_rows_without_the_joint_fit():
    truth, wrong, names, unstained, stained, _ab = experiment(n=2500)       # ~300 events per row: 6 bins
    res = run(wrong, names, unstained, stained, fmc.FixSettings(phase_two_rows='all', phase_two_active='all',
                                                                phase_two_fallback='reject', phase_two_rescue='off'))
    not_joint = [r for r in res.signature_log if r['joint'] is False]
    assert not_joint and not any(r['accepted'] for r in not_joint)
    assert any(r['reason'] == 'not.joint' for r in not_joint)


def test_seed_streams_repeat_and_still_recover():
    truth, wrong, names, unstained, stained, _ab = experiment()
    s = fmc.FixSettings(seed_streams=True)
    a = run(wrong, names, unstained, stained, s, truth=truth)
    b = run(wrong, names, unstained, stained, s, truth=truth)
    assert np.array_equal(a.spectra, b.spectra)
    rec = {r['fluorophore']: r for r in a.recovery}
    assert rec['Dye 4']['deg_after'] < 0.2 * rec['Dye 4']['deg_start']


def test_one_full_round_of_smooth_fix_is_one_run():
    truth, wrong, names, unstained, stained, _ab = experiment(n=8000)

    def project(spectra, keep):
        pu, ps, bg = projections(spectra, names, unstained, stained[keep])
        return pu, ps, bg

    one = run(wrong, names, unstained, stained)
    smooth = fmc.smooth_fix(wrong, names, project, len(stained), rounds=1, fraction=1.0, true_spectra=truth)
    assert np.allclose(smooth.spectra, one.spectra, atol=1e-12)
    assert len(smooth.smooth_log) == 1 and np.array_equal(smooth.spectra_start, wrong)


def test_smooth_fix_recovers_the_injected_error():
    truth, wrong, names, unstained, stained, _ab = experiment()

    def project(spectra, keep):
        return projections(spectra, names, unstained, stained[keep])

    res = fmc.smooth_fix(wrong, names, project, len(stained), rounds=3, fraction=0.5, true_spectra=truth)
    rec = {r['fluorophore']: r for r in res.recovery}
    assert rec['Dye 4']['deg_after'] < 0.3 * rec['Dye 4']['deg_start']
    assert all(r['deg_after'] < 0.5 for r in res.recovery if r['fluorophore'] != 'Dye 4')
    assert [r['round'] for r in res.smooth_log] == [1, 2, 3]


def test_a_fixed_fluorophore_keeps_its_spectrum_and_the_others_still_recover():
    truth, wrong, names, unstained, stained, _ab = experiment(
        injections=(('Dye 4', 'Dye 3', 0.10), ('Dye 6', 'Dye 2', 0.10)))
    res = run(wrong, names, unstained, stained, fmc.FixSettings(fixed_fluorophores=('Dye 4', 'Not a dye')),
              truth=truth)
    i4 = res.fluor_names.index('Dye 4')
    np.testing.assert_array_equal(res.spectra[i4], wrong[names.index('Dye 4')])
    np.testing.assert_array_equal(res.spectra_backsolved[i4], wrong[names.index('Dye 4')]
                                  / wrong[names.index('Dye 4')].max())
    assert np.all(res.spillover[i4, np.arange(len(res.fluor_names)) != i4] == 0)
    sig = {r['fluorophore']: r for r in res.signature_log}
    assert sig['Dye 4']['reason'] == 'fixed' and not sig['Dye 4']['accepted']
    src = {r['source']: r for r in res.source_log}
    assert src['Dye 4']['fixed'] and src['Dye 4']['source_weight'] == 0.0
    rec = {r['fluorophore']: r for r in res.recovery}
    assert rec['Dye 6']['deg_after'] < 0.3 * rec['Dye 6']['deg_start']
    assert res.settings['fixed_fluorophores'] == ('Dye 4', 'Not a dye')


def test_excluded_fixed_channel_is_no_target_and_the_flag_alone_changes_nothing(single_error):
    truth, wrong, names, res = single_error
    _t, _w, _n, unstained, stained, _ab = experiment()
    again = run(wrong, names, unstained, stained, fmc.FixSettings(exclude_fixed_targets=True), truth=truth)
    np.testing.assert_array_equal(again.spectra, res.spectra)
    np.testing.assert_array_equal(again.spillover, res.spillover)

    truth, wrong, names, unstained, stained, _ab = experiment(
        injections=(('Dye 6', 'Dye 4', 0.10), ('Dye 2', 'Dye 5', 0.10)))
    kept = run(wrong, names, unstained, stained, fmc.FixSettings(fixed_fluorophores=('Dye 4',)), truth=truth)
    out = run(wrong, names, unstained, stained,
              fmc.FixSettings(fixed_fluorophores=('Dye 4',), exclude_fixed_targets=True), truth=truth)
    i4, i6 = out.fluor_names.index('Dye 4'), out.fluor_names.index('Dye 6')
    assert any(r['channel'] == 'Dye 4' for r in kept.coefficient_log)
    assert not any(r['channel'] == 'Dye 4' for r in out.coefficient_log)
    assert kept.spillover[i6, i4] != 0 and out.spillover[i6, i4] == 0
    np.testing.assert_array_equal(out.spectra[i4], wrong[names.index('Dye 4')])
    rec = {r['fluorophore']: r for r in out.recovery}
    assert rec['Dye 2']['deg_after'] < 0.3 * rec['Dye 2']['deg_start']


def test_no_fixed_fluorophores_is_the_default_run(single_error):
    truth, wrong, names, res = single_error
    _t, _w, _n, unstained, stained, _ab = experiment()
    again = run(wrong, names, unstained, stained, fmc.FixSettings(fixed_fluorophores=()), truth=truth)
    np.testing.assert_array_equal(again.spectra, res.spectra)
    np.testing.assert_array_equal(again.spillover, res.spillover)
    assert again.convergence_log == res.convergence_log


# ---------------------------------------------------------------------------
# Plausibility box and the shared-direction check
# ---------------------------------------------------------------------------

def test_plausible_change_is_ratio_or_floor():
    ref = np.array([1.0, 0.2, 0.004, 0.0])
    assert np.allclose(fmc.plausible_change(ref, 1.0, 0.01), [1.0, 0.2, 0.01, 0.01])
    assert np.allclose(fmc.plausible_change(ref, 0.5, 0.02), [0.5, 0.1, 0.02, 0.02])


def test_clip_to_plausible_clips_or_resets_out_of_bounds_detectors():
    ref = np.array([1.0, 0.2, 0.0, 0.0])
    allow = fmc.plausible_change(ref, 1.0, 0.01)
    inside = np.array([1.0, 0.35, 0.008, 0.0])
    row, share = fmc.clip_to_plausible(inside, ref, allow)
    assert np.allclose(row, inside) and share == 0.0
    outside = np.array([1.0, 0.5, 0.06, 0.0])
    row, share = fmc.clip_to_plausible(outside, ref, allow)
    assert np.allclose(row, [1.0, 0.4, 0.01, 0.0])
    assert share == pytest.approx(np.hypot(0.1, 0.05) / np.hypot(0.3, 0.06))
    row, share_reset = fmc.clip_to_plausible(outside, ref, allow, reset=True)
    assert np.allclose(row, ref) and share_reset == pytest.approx(share)
    # a decrease to zero is always inside the box at ratio 1
    row, share = fmc.clip_to_plausible(np.array([1.0, 0.0, 0.0, 0.0]), ref, allow)
    assert share == 0.0


def test_plausible_project_takes_an_in_span_excess_back_out():
    spectra, _names = panel()
    allow = fmc.plausible_change(spectra, 1.0, 0.01)
    spill = np.eye(len(spectra))
    spill[0, 3] = 0.2                                       # Dye 0's row gains Dye 3's peak
    out, share = fmc._plausible_project(spill, spectra, spectra, allow)
    assert share[0] > 0.5 and np.all(share[1:] == 0)
    row = np.maximum(out[0] @ spectra, 0.0)
    row /= row.max()
    assert np.max(np.abs(row - spectra[0]) - allow[0]) < 0.02
    assert np.allclose(out[1:], spill[1:]) and np.allclose(np.diag(out), 1.0)


def test_shared_direction_counts_unrelated_sources_only():
    spectra, _names = panel()
    cos = fmc.cosine_similarity(spectra)
    s = fmc.FixSettings(shared_direction=True)
    marker = np.eye(len(spectra))
    trust = np.eye(len(spectra))
    marker[0, 3] = marker[6, 3] = 0.05                       # two unrelated sources toward Dye 3
    trust[0, 3] = trust[6, 3] = 0.5
    cands = [(0, 3, 0), (6, 3, 1)]
    counts = fmc.shared_direction_counts(marker, trust, cands, cos, s)
    assert counts[0, 3] == counts[6, 3] == 2
    marker[6, 3] = -0.05                                     # opposite signs do not share a direction
    assert fmc.shared_direction_counts(marker, trust, cands, cos, s)[0, 3] == 1
    marker[6, 3] = 0.05
    related = np.where(np.eye(len(spectra)) > 0, 1.0, 0.0)
    related[0, 6] = related[6, 0] = 0.9                      # a related pair counts once
    assert fmc.shared_direction_counts(marker, trust, cands, related, s)[0, 3] == 1


def test_plausibility_box_keeps_a_real_correction():
    truth, wrong, names, unstained, stained, _ab = experiment()
    res = run(wrong, names, unstained, stained, fmc.FixSettings(), truth=truth)
    rec = {r['fluorophore']: r for r in res.recovery}
    assert rec['Dye 4']['accepted'] and rec['Dye 4']['deg_after'] < 0.2 * rec['Dye 4']['deg_start']
    assert all(r['deg_after'] < 0.5 for r in res.recovery if r['fluorophore'] != 'Dye 4')
    f = names.index('Dye 4')
    allow = fmc.plausible_change(wrong[f] / wrong[f].max(), 1.0, 0.01)
    assert np.all(np.abs(res.spectra[f] - wrong[f] / wrong[f].max()) <= allow + 1e-9)


def test_plausibility_box_limits_coexpression_booked_into_empty_detectors():
    truth, names, unstained, stained = coexpressed_experiment()
    base = run(truth, names, unstained, stained, fmc.FixSettings(max_iter=6, plausibility_ratio=0.0), truth=truth)
    boxed = run(truth, names, unstained, stained, fmc.FixSettings(max_iter=6), truth=truth)
    for dye in ('Dye 0', 'Dye 6'):
        i = names.index(dye)
        before = {r['fluorophore']: r['deg_after'] for r in base.recovery}[dye]
        after = {r['fluorophore']: r['deg_after'] for r in boxed.recovery}[dye]
        assert before > 1.0 and after < 0.5 * before
        allow = fmc.plausible_change(truth[i], 1.0, 0.01)
        assert np.all(np.abs(boxed.spectra[i] - truth[i]) <= allow + 1e-9)


def test_shared_direction_shrinks_coexpression():
    truth, names, unstained, stained = coexpressed_experiment()
    off = dict(max_iter=6, plausibility_ratio=0.0)
    base = run(truth, names, unstained, stained, fmc.FixSettings(**off), truth=truth)
    shared = run(truth, names, unstained, stained, fmc.FixSettings(shared_direction=True, **off), truth=truth)
    b = {r['fluorophore']: r['deg_after'] for r in base.recovery}
    sh = {r['fluorophore']: r['deg_after'] for r in shared.recovery}
    assert b['Dye 0'] + b['Dye 6'] > 2.0
    assert sh['Dye 0'] + sh['Dye 6'] < 0.8 * (b['Dye 0'] + b['Dye 6'])
    assert any(r.get('shared_sources', 1) > 1 for r in shared.coefficient_log)


@pytest.mark.parametrize('bad', [dict(plausibility_ratio=-1.0), dict(plausibility_reject=0.0),
                                 dict(shared_cosine=1.5), dict(shared_alignment=2.0), dict(af_penalty_width=0.0)])
def test_new_settings_are_validated(bad):
    truth, wrong, names, unstained, stained, _ab = experiment(n=3000)
    with pytest.raises(ValueError):
        run(wrong, names, unstained, stained, fmc.FixSettings(**bad))


# ---------------------------------------------------------------------------
# Settled stop rule and the anchored disagreement reference (2026-10-22)
# ---------------------------------------------------------------------------

def test_settled_stop_is_not_held_open_by_an_untrusted_slope():
    truth, wrong, names, unstained, stained, ab = experiment(injections=(('Dye 4', 'Dye 3', 0.10),
                                                                         ('Dye 6', 'Dye 5', 0.08)))
    keep = (ab[:, 6] == 0) | (np.cumsum(ab[:, 6] > 0) <= 300)        # Dye 6 too rare for more than a sliver of trust
    every = run(wrong, names, unstained, stained[keep], fmc.FixSettings(stop_rule='all'), truth=truth)
    settled = run(wrong, names, unstained, stained[keep], fmc.FixSettings(stop_rule='settled'), truth=truth)
    weak = [r['trust'] for r in settled.coefficient_log if r['source'] == 'Dye 6' and r['trust'] > 0]
    assert weak and max(weak) < 0.05
    assert settled.settings['stop_reason'] == 'converged'
    assert len(settled.convergence_log) < len(every.convergence_log)
    assert all(c['delta_quantile'] < 0.01 for c in settled.convergence_log[-2:])      # confirmed twice
    rec = {r['fluorophore']: r for r in settled.recovery}
    assert rec['Dye 4']['accepted'] and rec['Dye 4']['deg_after'] < 0.2 * rec['Dye 4']['deg_start']


def _graded_population(rng, n=20_000, slope=-0.05, coexpressed=0.0):
    """Source-positive events whose target spread grows with brightness (photon noise), optionally half of them
    carrying the target at a graded level; and a negative bulk."""
    x_pos = rng.lognormal(np.log(30_000), 0.5, n)
    y_pos = slope * x_pos + rng.normal(size=n) * np.sqrt(300.0 ** 2 + 20.0 * x_pos)
    if coexpressed:
        lift = rng.random(n) < coexpressed
        y_pos[lift] += x_pos[lift] * rng.uniform(0.0, 0.25, lift.sum())
    x = np.concatenate([rng.normal(0, 300, 20_000), x_pos])
    y = np.concatenate([rng.normal(0, 300, 20_000), y_pos])
    upper = 1000.0 + 3.0 * np.sqrt(20.0 * np.maximum(x, 0))
    return x, y[:, None], np.full(len(x), 1000.0), upper[:, None], -upper[:, None]


@pytest.mark.parametrize('coexpressed', [0.0, 0.5, 0.8])
def test_disagreement_references_flag_coexpression_but_not_spread(coexpressed):
    x, Y, ts, upper, lower = _graded_population(np.random.default_rng(31), coexpressed=coexpressed)
    out = {}
    for ref in fmc.DISAGREEMENT_REFERENCES:
        s = fmc.FixSettings(disagreement_reference=ref)
        est = fmc.envelope_slopes(x, Y, ts, upper, np.array([20.0]), np.array([300.0 ** 2]), None, np.zeros(1),
                                  np.random.default_rng(0), s, lower)
        out[ref] = est['disagreement'][0]
    if coexpressed == 0.0:
        assert out['envelope'] > 0.15                    # the lower envelope runs steeper as the spread grows
        assert out['anchored'] < 0.05 and out['envelope_corrected'] < 0.05
        assert est['slope_anchored'][0] == pytest.approx(-0.05, abs=0.005)
    elif coexpressed == 0.5:
        # co-expression inside the band moves the anchored slope with the median; the corrected envelope does not
        assert out['envelope_corrected'] > 2.0 * out['anchored']
    else:
        assert out['anchored'] > 0.5 and out['envelope_corrected'] > 0.5


@pytest.mark.parametrize('bad', [dict(stop_rule='settled', stop_trust=0.0), dict(stop_rule='settled', stop_confirm=0),
                                 dict(disagreement_reference='other'),
                                 dict(disagreement_reference='envelope_corrected', envelope_scale_quantile=0.01)])
def test_settled_and_disagreement_settings_are_validated(bad):
    truth, wrong, names, unstained, stained, _ab = experiment(n=3000)
    with pytest.raises(ValueError):
        run(wrong, names, unstained, stained, fmc.FixSettings(**bad))


def test_lower_boundary_keeps_its_own_quantile():
    # with the lower boundary at the unstained 5th percentile, about 5% of a correctly unmixed population lies
    # below it, the hypernegative check takes the trust of the true coefficient, and the error stays
    truth, wrong, names, unstained, stained, _ab = experiment()
    rec = {}
    for lower in (0.99, 0.95):
        res = run(wrong, names, unstained, stained, fmc.FixSettings(unstained_threshold=0.95,
                                                                    lower_unstained_threshold=lower), truth=truth)
        rec[lower] = {r['fluorophore']: r for r in res.recovery}['Dye 4']
    assert rec[0.99]['deg_after'] < 0.2 * rec[0.99]['deg_start']
    assert rec[0.95]['deg_after'] > 0.5 * rec[0.95]['deg_start']


@pytest.mark.parametrize('bad', [dict(unstained_threshold=0.4), dict(lower_unstained_threshold=1.0)])
def test_unstained_quantiles_are_validated(bad):
    truth, wrong, names, unstained, stained, _ab = experiment(n=3000)
    with pytest.raises(ValueError):
        run(wrong, names, unstained, stained, fmc.FixSettings(**bad))


# ---------------------------------------------------------------------------
# Present-dye joint fit, the back-solved rescue and consensus smoothing
# ---------------------------------------------------------------------------

def test_present_dyes_give_a_small_population_a_joint_fit():
    # the synthetic dyes overlap so much that every one reads as present in every population; with the
    # presence rules closed the target alone is fitted, which a population of ~400 events can carry jointly
    truth, wrong, names, unstained, stained, ab = experiment(n=12_000)
    stained = stained[(ab > 0).sum(axis=1) <= 1]                            # single positives: ~400 events per row
    every = run(wrong, names, unstained, stained, fmc.FixSettings(phase_two_rows='all', phase_two_active='all',
                                                                  phase_two_rescue='off'))
    alone = run(wrong, names, unstained, stained, fmc.FixSettings(
        phase_two_rows='all', phase_two_rescue='off', phase_two_present_frac=1.0, phase_two_present_corr=1.0,
        phase_two_present_mean=1e9), truth=truth)
    assert not any(r['joint'] for r in every.signature_log if r['reason'] != 'no.fit')
    assert all(r['joint'] for r in alone.signature_log if r['reason'] != 'no.fit')
    assert all(r['n_active'] == 1 for r in alone.signature_log if r['reason'] != 'no.fit')
    rec = {r['fluorophore']: r for r in alone.recovery}
    assert rec['Dye 4']['deg_after'] < 0.3 * rec['Dye 4']['deg_start']


def test_backsolved_row_rescues_a_row_the_remeasurement_cannot_fit(monkeypatch):
    truth, wrong, names, unstained, stained, _ab = experiment()
    monkeypatch.setattr(fmc, 'extract_raw_signature', lambda *args, **kwargs: None)
    off = run(wrong, names, unstained, stained, fmc.FixSettings(phase_two_rescue='off'), truth=truth)
    on = run(wrong, names, unstained, stained, fmc.FixSettings(), truth=truth)
    reasons_off = {r['fluorophore']: r['reason'] for r in off.signature_log}
    assert reasons_off['Dye 4'] == 'no.fit' and not any(r['accepted'] for r in off.signature_log)
    entry = {r['fluorophore']: r for r in on.signature_log}['Dye 4']
    assert entry['accepted'] and entry['rescued_from'] == 'no.fit'
    rec = {r['fluorophore']: r for r in on.recovery}
    assert rec['Dye 4']['deg_after'] < 0.2 * rec['Dye 4']['deg_start']
    f = names.index('Dye 4')
    assert fmc.row_angle(on.spectra[f:f + 1], on.spectra_backsolved[f:f + 1])[0] < 0.5     # the clipped back-solved row


def test_rescue_follows_only_its_reasons(monkeypatch):
    truth, wrong, names, unstained, stained, _ab = experiment()
    monkeypatch.setattr(fmc, 'extract_raw_signature', lambda *args, **kwargs: None)
    res = run(wrong, names, unstained, stained, fmc.FixSettings(rescue_reasons=('leakage',)))
    assert not any(r['accepted'] for r in res.signature_log)


def test_consensus_smoothing_keeps_only_agreed_rows():
    truth, wrong, names, unstained, stained, _ab = experiment()

    def project(spectra, keep):
        return projections(spectra, names, unstained, stained[keep])

    res = fmc.smooth_fix(wrong, names, project, len(stained), rounds=3, fraction=0.5,
                         settings=fmc.FixSettings(smooth_mode='consensus'), true_spectra=truth)
    rec = {r['fluorophore']: r for r in res.recovery}
    assert rec['Dye 4']['deg_after'] < 0.3 * rec['Dye 4']['deg_start']
    assert all(r['deg_after'] < 0.2 for r in res.recovery if r['fluorophore'] != 'Dye 4')
    entry = {r['fluorophore']: r for r in res.signature_log}['Dye 4']
    assert entry['accepted'] and entry['consensus_spread'] <= 0.5 * entry['consensus_move']


@pytest.mark.parametrize('bad', [dict(phase_two_active='other'), dict(phase_two_rescue='other'),
                                 dict(smooth_mode='other'), dict(phase_two_present_frac=1.5),
                                 dict(rescue_min_weight=-0.1), dict(consensus_min_rounds=0.0)])
def test_phase_two_and_consensus_settings_are_validated(bad):
    truth, wrong, names, unstained, stained, _ab = experiment(n=3000)
    with pytest.raises(ValueError):
        run(wrong, names, unstained, stained, fmc.FixSettings(**bad))


# ---------------------------------------------------------------------------
# Leakage over negative channels and the tandem parent-fraction candidate
# ---------------------------------------------------------------------------

def test_tandem_pairs_from_the_fluorophore_database():
    names = ['PE', 'PE-Cy7', 'BV711', 'Super Bright 436', 'BUV737', 'APC', 'AF']
    assert fmc.tandem_pairs(names) == [(1, 0), (2, 3)]
    assert fmc.tandem_pairs(['FITC', 'APC']) == []


def test_parent_fraction_candidate_corrects_a_tandem_the_remeasurement_cannot(monkeypatch):
    # Dye 4's injected error is 0.10 of Dye 3's row: declared a tandem of Dye 3, its parent fraction alone
    # carries the correction once the re-measurement and the back-solved rescue are out of the way
    truth, wrong, names, unstained, stained, _ab = experiment()
    monkeypatch.setattr(fmc, 'extract_raw_signature', lambda *args, **kwargs: None)
    res = run(wrong, names, unstained, stained, fmc.FixSettings(phase_two_rescue='off'), truth=truth,
              tandems=[(4, 3)])
    entry = {r['fluorophore']: r for r in res.signature_log}['Dye 4']
    assert entry['accepted'] and entry['candidate'] == 'parent' and entry['rescued_from'] == 'no.fit'
    assert entry['tandem_parent'] == 'Dye 3' and entry['tandem_coefficient'] == pytest.approx(-0.10, abs=0.02)
    rec = {r['fluorophore']: r for r in res.recovery}['Dye 4']
    assert rec['deg_after'] < 0.1 * rec['deg_start']
    off = run(wrong, names, unstained, stained, fmc.FixSettings(phase_two_rescue='off', tandem_candidate=False),
              tandems=[(4, 3)])
    assert not any(r['accepted'] for r in off.signature_log)
    small = run(wrong, names, unstained, stained, fmc.FixSettings(phase_two_rescue='off',
                                                                  tandem_min_coefficient=0.5), tandems=[(4, 3)])
    assert not any(r['accepted'] for r in small.signature_log)


def test_leakage_over_negative_channels_keeps_the_recovery(single_error):
    truth, wrong, names, res = single_error
    rec = {r['fluorophore']: r for r in res.recovery}['Dye 4']
    assert res.settings['leakage_channels'] == 'negative'
    assert rec['deg_after'] < 0.2 * rec['deg_start']
    entry = {r['fluorophore']: r for r in res.signature_log}['Dye 4']
    assert np.isfinite(entry['leak_before']) and np.isfinite(entry['leak_after'])


@pytest.mark.parametrize('bad', [dict(leakage_channels='other'), dict(tandem_min_coefficient=-0.1)])
def test_leakage_and_tandem_settings_are_validated(bad):
    truth, wrong, names, unstained, stained, _ab = experiment(n=3000)
    with pytest.raises(ValueError):
        run(wrong, names, unstained, stained, fmc.FixSettings(**bad))

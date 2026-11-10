"""
test_af_negative_envelope.py
----------------------------
Positivity-weighted (negative-envelope) AF scoring: the null variance and
its upkeep, the positivity weights and scoring, saturation re-fit and the
spillover spread, on synthetic data (no FCS files, no Qt).

Usage:
    pytest tests/test_af_negative_envelope.py -m numpy_only
"""

import math

import numpy as np
import pytest

from honeychrome.controller_components import autospectral_functions as af
from honeychrome.controller_components import spread_model as sm

N_FLUORS = 8
N_DETECTORS = 24
N_AF = 12


def _peak(centre, width, n=N_DETECTORS):
    s = np.exp(-0.5 * ((np.arange(n) - centre) / width) ** 2)
    return s / s.max()


@pytest.fixture(scope='module')
def panel():
    rng = np.random.default_rng(7)
    fluors = np.array([_peak(c, 1.8) for c in np.linspace(2, N_DETECTORS - 3, N_FLUORS)])
    af_lib = np.array([_peak(rng.uniform(2, 14), rng.uniform(4, 8)) for _ in range(N_AF)])
    af_lib = np.vstack([af_lib.mean(axis=0), af_lib])
    names = [f'F{i}' for i in range(N_FLUORS)]
    return {'fluors': fluors, 'af': af_lib, 'names': names, 'rng': rng}


def _events(panel, n, positive_fraction=0.15):
    rng = np.random.default_rng(11)
    af_lib = panel['af']
    scale = rng.lognormal(6.0, 0.6, n)
    which = rng.integers(1, len(af_lib), n)
    raw = scale[:, None] * af_lib[which] + rng.normal(0, 25, (n, N_DETECTORS))
    pos = rng.random(n) < positive_fraction
    dye = rng.integers(0, N_FLUORS, pos.sum())
    raw[pos] += rng.lognormal(7.0, 1.0, (pos.sum(), 1)) * panel['fluors'][dye]
    return raw


def _runtime_envelope(panel, raw, spread=None, ceiling=1e12):
    entry, _ = af.make_envelope_entry(raw, panel['names'], panel['fluors'], panel['af'])
    entry = dict(entry)
    return af.build_runtime_envelope(entry, spread, panel['names'], panel['fluors'], ceiling)


def _flat_null(c0, n_means=1):
    """Runtime envelope with a constant null variance c0 and no spread."""
    null_var = np.zeros((n_means, 3, np.size(c0)))
    null_var[:, 0] = c0
    return {'null_var': null_var, 'spread': None, 'mean_rows': np.arange(n_means),
            'ceiling': 1e12, 'saturated_frac': 0.999}


@pytest.mark.numpy_only
def test_infinite_null_variance_reduces_to_joint_l2(panel):
    raw = _events(panel, 4000)
    envelope = _flat_null(np.full(N_FLUORS, np.inf))
    joint = af.apply_af_unmixing(raw, af.build_af_precomputed(panel['fluors'], panel['af']), panel['af'])
    weighted = af.apply_af_unmixing(raw, af.build_af_precomputed(panel['fluors'], panel['af'], envelope), panel['af'])
    same = np.abs(joint['unmixed'] - weighted['unmixed']).max(axis=1) < 1e-6
    assert same.mean() > 0.999


@pytest.mark.numpy_only
def test_tied_scores_resolve_to_the_first_variant():
    score = np.array([
        [-3.0, -3.0, -3.0, -3.0],                    # exact ties
        [-3.0 + 4e-10, -3.0, -3.0 - 3e-10, -3.0],    # float noise
        [-1.0, -2.0, -3.0, -2.5],                    # clear minimum
        [-1.0, -3.0, -3.0 - 1e-3, -2.0],             # a later variant clearly better
    ])
    assert list(af._first_within_tolerance(score)) == [0, 0, 2, 2]


@pytest.mark.numpy_only
def test_positivity_weight_values_and_shape():
    x = np.array([[-50.0, 0.0, 1.0, 2.0, 3.0, 10.0]])
    w = af.positivity_weight(x, np.ones_like(x))[0]
    np.testing.assert_allclose(w[:2], 1.0)
    np.testing.assert_allclose(w[2:5], [0.317311, 0.0455003, 0.0026998], rtol=1e-5)
    assert 0.0 < w[5] < 1e-20
    fine = af.positivity_weight(np.linspace(-3, 6, 901)[None, :], np.ones((1, 901)))[0]
    assert np.all(np.diff(fine) <= 0) and np.abs(np.diff(fine)).max() < 0.01
    np.testing.assert_allclose(af.positivity_weight(x, 4.0 * np.ones_like(x)),
                               af.positivity_weight(x / 2.0, np.ones_like(x)))
    np.testing.assert_array_equal(af.positivity_weight(x, np.zeros_like(x))[0], [1, 1, 0, 0, 0, 0])


def _explicit_term(u, k, v_library, w, envelope, e_resid):
    """The positivity-weighted term from its definition, with an (n, n_af, n_fluors) array."""
    rows = np.arange(len(u))
    mean_rows = np.asarray(envelope['mean_rows'])
    pick = np.argmin(e_resid[:, mean_rows], axis=1)
    mean_j = mean_rows[pick]
    k_mean = k[rows, mean_j][:, None]
    position = u - k_mean * v_library.T[mean_j]
    coef = envelope['null_var'][pick]
    var = coef[:, 0] + k_mean * coef[:, 1] + k_mean ** 2 * coef[:, 2]
    if envelope['spread'] is not None:
        var = var + np.maximum(position, 0.0) @ envelope['spread']
    p = np.zeros_like(position)
    for (i, f), x in np.ndenumerate(position):
        p[i, f] = math.erf(max(x, 0.0) / math.sqrt(2.0 * var[i, f]))
    weight = w * (1.0 - p)
    x = u[:, None, :] - k[:, :, None] * v_library.T[None, :, :]
    return np.einsum('ijf,if->ij', x * x, weight) / (np.einsum('if,if->i', u * u, weight) + 1e-6)[:, None]


@pytest.mark.numpy_only
def test_positivity_term_matches_its_definition():
    rng = np.random.default_rng(3)
    n, n_af = 300, N_AF + 1
    u = rng.normal(0, 200, (n, N_FLUORS))
    k = np.abs(rng.normal(0, 800, (n, n_af)))
    v_library = rng.normal(0, 0.3, (N_FLUORS, n_af))
    w = np.abs(rng.normal(1, 0.3, N_FLUORS))
    e_resid = rng.uniform(0.1, 1.0, (n, n_af))
    spread = np.abs(rng.normal(0, 0.05, (N_FLUORS, N_FLUORS)))
    np.fill_diagonal(spread, 0.0)
    null_var = np.abs(rng.normal(0, 1, (2, 3, N_FLUORS))) * np.array([1e4, 1.0, 1e-4])[None, :, None]
    for envelope in ({'null_var': null_var[:1], 'spread': spread, 'mean_rows': np.array([0])},
                     {'null_var': null_var, 'spread': None, 'mean_rows': np.array([0, 5])}):
        got = af._positivity_fluor_term(u, k, e_resid, v_library, w, envelope)
        np.testing.assert_allclose(got, _explicit_term(u, k, v_library, w, envelope, e_resid), rtol=1e-8, atol=1e-12)


@pytest.mark.numpy_only
def test_positivity_is_judged_at_the_mean_with_the_smallest_residual():
    envelope = _flat_null(np.array([1.0]), n_means=2)
    envelope['mean_rows'] = np.array([0, 2])
    v_library = np.array([[0.0, 0.0, 1.0]])
    u = np.array([[30.0]])
    k = np.array([[0.0, 0.0, 30.0]])           # row 0 leaves 30 (positive), row 2 leaves 0 (negative)
    w = np.ones(1)
    near_row2 = af._positivity_fluor_term(u, k, np.array([[2.0, 1.0, 1.0]]), v_library, w, envelope)
    near_row0 = af._positivity_fluor_term(u, k, np.array([[1.0, 1.0, 2.0]]), v_library, w, envelope)
    assert near_row2[0, 2] < 1e-6 < near_row2[0, 0]    # negative: row 2 brings the channel to 0
    assert near_row0[0, 0] == near_row0[0, 2] == 1.0    # positive: the channel no longer counts


def _smooth_af_unstained(n=6000, seed=0):
    """AF library of five smooth, tilted shapes (mean first) and unstained events drawn from it."""
    rng = np.random.default_rng(seed)
    d = np.arange(N_DETECTORS)
    base = np.exp(-0.5 * ((d - 10) / 7.0) ** 2) + 0.2
    shapes = np.array([s / s.max() for s in (base * (1 + t * (d - 12) / 12) for t in (-0.3, -0.15, 0.0, 0.15, 0.3))])
    library = np.vstack([shapes.mean(axis=0) / shapes.mean(axis=0).max(), shapes])
    raw = rng.lognormal(np.log(5000), 0.4, n)[:, None] * shapes[rng.integers(0, 5, n)] + rng.normal(0, 40, (n, N_DETECTORS))
    return library, raw


@pytest.mark.numpy_only
def test_null_variance_follows_the_unstained_spread_at_the_mean_af(panel):
    library, raw = _smooth_af_unstained()
    coef = af.positivity_null_variance(raw, panel['fluors'], library)
    assert coef.shape == (3, N_FLUORS)
    assert np.all(coef[0] > 0) and np.all(coef[1:] >= 0)
    position, k0 = af.mean_af_position(raw, panel['fluors'], library)
    for q in (0.2, 0.5, 0.8):
        centre = np.quantile(k0, q)
        near = np.abs(k0 - centre) < 0.05 * centre
        fitted = coef[0] + centre * coef[1] + centre ** 2 * coef[2]
        observed = (1.4826 * np.median(np.abs(position[near] - np.median(position[near], axis=0)), axis=0)) ** 2
        np.testing.assert_allclose(fitted, observed, rtol=0.3)


@pytest.mark.numpy_only
def test_positivity_weighting_absorbs_fewer_dim_positives_than_joint_l2(panel):
    unstained = _events(panel, 13000, positive_fraction=0.0)
    entry, _ = af.make_envelope_entry(unstained[:5000], panel['names'], panel['fluors'], panel['af'])
    envelope = af.build_runtime_envelope(entry, None, panel['names'], panel['fluors'], 1e12)
    test = unstained[5000:]
    dye = 4
    sd = np.sqrt(np.asarray(entry['null_var'])[0, dye])
    amt = sd * np.exp(np.random.default_rng(4).uniform(np.log(2), np.log(8), len(test)))
    spiked = test + amt[:, None] * panel['fluors'][dye]
    pulled = {}
    for label, env in (('joint', None), ('positivity', envelope)):
        pre = af.build_af_precomputed(panel['fluors'], panel['af'], env)
        base = af.apply_af_unmixing(test, pre, panel['af'])['unmixed'][:, dye]
        out = af.apply_af_unmixing(spiked, pre, panel['af'])['unmixed'][:, dye]
        pulled[label] = np.mean(out - base - amt < -0.25 * amt)
    assert pulled['joint'] > 0.1
    assert pulled['positivity'] < 0.5 * pulled['joint']


@pytest.mark.numpy_only
def test_saturated_events_are_refitted_without_the_clipped_detector(panel):
    raw = _events(panel, 3000)
    ceiling = float(np.quantile(raw, 0.999))
    raw = np.minimum(raw, ceiling)
    envelope = _runtime_envelope(panel, raw, ceiling=ceiling)
    pre = af.build_af_precomputed(panel['fluors'], panel['af'], envelope)
    full = af.apply_af_unmixing(raw, pre, panel['af'])
    first_pass = af.apply_af_unmixing(raw, pre, panel['af'], handle_saturation=False)

    saturated = raw >= envelope['saturated_frac'] * ceiling
    hit = np.flatnonzero(saturated.any(axis=1))
    assert 0 < len(hit) < len(raw)
    untouched = np.setdiff1d(np.arange(len(raw)), hit)
    assert np.array_equal(full['unmixed'][untouched], first_pass['unmixed'][untouched])

    row = hit[0]
    keep = ~saturated[row]
    sub_fluors = panel['fluors'][:, keep]
    sub_af = panel['af'][:, keep]
    sub = af.build_af_precomputed(sub_fluors, sub_af, envelope)
    expected = af.apply_af_unmixing(raw[[row]][:, keep], sub, sub_af, handle_saturation=False)
    assert np.allclose(full['unmixed'][row], expected['unmixed'][0])


@pytest.mark.numpy_only
def test_unidentifiable_saturation_pattern_keeps_first_pass(panel):
    raw = _events(panel, 200)
    ceiling = float(raw.max()) * 2.0
    envelope = _runtime_envelope(panel, raw, ceiling=ceiling)
    raw = raw.copy()
    raw[0, : N_DETECTORS - N_FLUORS + 1] = ceiling
    pre = af.build_af_precomputed(panel['fluors'], panel['af'], envelope)
    full = af.apply_af_unmixing(raw, pre, panel['af'])
    first_pass = af.apply_af_unmixing(raw, pre, panel['af'], handle_saturation=False)
    assert np.array_equal(full['unmixed'][0], first_pass['unmixed'][0])


def _rotate(spectrum, degrees):
    """Spectrum moved by an exact angle, in a direction orthogonal to itself."""
    orthogonal = np.roll(spectrum, 3) - spectrum * (np.roll(spectrum, 3) @ spectrum) / (spectrum @ spectrum)
    orthogonal *= np.linalg.norm(spectrum) / np.linalg.norm(orthogonal)
    t = np.radians(degrees)
    return np.cos(t) * spectrum + np.sin(t) * orthogonal


@pytest.mark.numpy_only
def test_spectra_unchanged_uses_the_two_degree_tolerance(panel):
    moved = panel['fluors'].copy()
    moved[2] = _rotate(moved[2], 1.5)
    assert af.spectra_unchanged(panel['names'], panel['fluors'], panel['names'], moved)
    moved[2] = _rotate(panel['fluors'][2], 2.5)
    assert not af.spectra_unchanged(panel['names'], panel['fluors'], panel['names'], moved)
    assert not af.spectra_unchanged(panel['names'], panel['fluors'], panel['names'][::-1], panel['fluors'])
    assert not af.spectra_unchanged(panel['names'], panel['fluors'], panel['names'][:-1], panel['fluors'][:-1])


@pytest.mark.numpy_only
def test_envelope_sample_is_five_thousand_events_and_reproducible(panel, tmp_path):
    raw = _events(panel, 7000)
    entry, sample = af.make_envelope_entry(raw, panel['names'], panel['fluors'], panel['af'], 'p unstained.npy')
    assert sample.shape == (af.ENVELOPE_SAMPLE_EVENTS, N_DETECTORS)
    assert sample.dtype == np.float32
    assert set(entry) == {'null_var', 'fluor_names', 'fluor_spectra', 'unstained_file'}
    af.save_envelope_sample(sample, tmp_path, 'p unstained.npy')
    stored = np.load(tmp_path / af.AF_SUBDIR / 'p unstained.npy').astype(np.float64)
    null_var = af.positivity_null_variance(stored, panel['fluors'], panel['af'])
    assert np.array_equal(null_var, np.array(entry['null_var']))
    assert np.all(null_var[0] > 0)


@pytest.mark.numpy_only
def test_refresh_envelope_entry(panel, tmp_path):
    raw = _events(panel, 6000)
    entry, sample = af.make_envelope_entry(raw, panel['names'], panel['fluors'], panel['af'], 'p unstained.npy')
    af.save_envelope_sample(sample, tmp_path, 'p unstained.npy')

    kept, changed = af.refresh_envelope_entry(entry, panel['names'], panel['fluors'], panel['af'], tmp_path)
    assert kept is entry and not changed

    moved = panel['fluors'].copy()
    moved[1] = _rotate(moved[1], 1.5)
    kept, changed = af.refresh_envelope_entry(entry, panel['names'], moved, panel['af'], tmp_path)
    assert kept is entry and not changed

    moved[1] = _rotate(panel['fluors'][1], 4.0)
    rebuilt, changed = af.refresh_envelope_entry(entry, panel['names'], moved, panel['af'], tmp_path)
    assert changed
    assert np.allclose(rebuilt['fluor_spectra'], moved)
    assert not np.array_equal(rebuilt['null_var'], entry['null_var'])

    empty = tmp_path / 'empty'
    stale, changed = af.refresh_envelope_entry(entry, panel['names'], moved, panel['af'], empty)
    assert stale is entry and not changed
    dropped, changed = af.refresh_envelope_entry(entry, panel['names'][:-1], moved[:-1], panel['af'], empty)
    assert dropped is None and not changed


@pytest.mark.numpy_only
def test_earlier_entry_gains_the_null_variance_from_the_stored_sample(panel, tmp_path):
    raw = _events(panel, 6000)
    entry, sample = af.make_envelope_entry(raw, panel['names'], panel['fluors'], panel['af'], 'p unstained.npy')
    af.save_envelope_sample(sample, tmp_path, 'p unstained.npy')
    legacy = {key: value for key, value in entry.items() if key != 'null_var'}
    legacy.update({'upper': [1.0] * N_FLUORS, 'lower': [-1.0] * N_FLUORS, 'noise_sd': [1.0] * N_FLUORS})

    moved = panel['fluors'].copy()
    moved[1] = _rotate(moved[1], 1.5)
    upgraded, changed = af.refresh_envelope_entry(legacy, panel['names'], moved, panel['af'], tmp_path)
    assert changed
    np.testing.assert_allclose(upgraded['null_var'], entry['null_var'])
    assert upgraded['fluor_spectra'] == legacy['fluor_spectra']
    assert 'null_var' not in legacy

    kept, changed = af.refresh_envelope_entry(legacy, panel['names'], panel['fluors'], panel['af'], tmp_path / 'empty')
    assert kept is legacy and not changed
    assert af.build_runtime_envelope(legacy, None, panel['names'], panel['fluors'], 1e12) is None


@pytest.mark.numpy_only
def test_combined_envelope_keeps_each_profiles_null_variance(panel):
    raw = _events(panel, 3000)
    a = _runtime_envelope(panel, raw)
    b = dict(a)
    b['null_var'] = a['null_var'] * 3.0
    pre_a = af.build_af_precomputed(panel['fluors'], panel['af'], a)
    pre_b = af.build_af_precomputed(panel['fluors'], panel['af'][:5], b)
    pre_none = af.build_af_precomputed(panel['fluors'], panel['af'])
    combined = af.combine_af_precomputed([pre_a, pre_b])
    np.testing.assert_array_equal(combined['envelope']['null_var'], np.concatenate([a['null_var'], b['null_var']]))
    assert list(combined['envelope']['mean_rows']) == [0, N_AF + 1]
    assert 'envelope' not in af.combine_af_precomputed([pre_a, pre_none])
    library = np.vstack([panel['af'], panel['af'][:5]])
    out = af.apply_af_unmixing(raw, af.build_af_precomputed(panel['fluors'], library, combined['envelope']), library)
    assert np.all(np.isfinite(out['unmixed']))


@pytest.mark.numpy_only
def test_stale_spread_is_not_used(panel):
    raw = _events(panel, 3000)
    entry, _ = af.make_envelope_entry(raw, panel['names'], panel['fluors'], panel['af'])
    spread = {'fluor_names': panel['names'], 'fluor_spectra': panel['fluors'].tolist(),
              'spread': np.full((N_FLUORS, N_FLUORS), 1e-3).tolist()}
    current = af.build_runtime_envelope(entry, spread, panel['names'], panel['fluors'], 2.0 ** 18)
    assert current['spread'] is not None
    moved = panel['fluors'].copy()
    moved[0] = _rotate(moved[0], 3.0)
    stale = af.build_runtime_envelope(entry, spread, panel['names'], moved, 2.0 ** 18)
    assert stale['spread'] is None


@pytest.mark.numpy_only
def test_binned_spread_recovers_the_variance_slope():
    rng = np.random.default_rng(5)
    n_f = 4
    spectra = np.eye(n_f)
    slopes = np.array([0.0, 5e-3, 2e-2, 1e-2])
    b = rng.lognormal(7.0, 1.0, 20_000)
    noise = rng.normal(0, 1, (len(b), n_f)) * np.sqrt(10.0 ** 2 + slopes[None, :] * b[:, None])
    noise[:, 0] = 0.0
    pos = b[:, None] * spectra[0] + noise
    row = sm.residual_model_spread_one(pos, None, None, None, spectra, 0, 0, 0.0, None, False, 1e12)
    assert row is not None and row['n'] == 20_000
    assert np.isnan(row['slope'][0])
    assert np.allclose(row['slope'][1:], slopes[1:], rtol=0.3)


@pytest.mark.numpy_only
def test_compute_spread_entry_with_loaded_controls():
    rng = np.random.default_rng(9)
    n_f = 3
    spectra = np.eye(n_f)
    names = ['A', 'B', 'C']

    def loader(experiment_dir, rel_path, gate_label, layout, gating):
        n = 4000
        b = rng.lognormal(7.0, 0.8, n)
        fl = rng.normal(0, 10, (n, n_f))
        if rel_path.startswith('ctrl'):
            fl[:, int(rel_path[-1])] += b
            fl[:, (int(rel_path[-1]) + 1) % n_f] += rng.normal(0, 1, n) * np.sqrt(5e-2 * b)
        return fl, rng.normal(0, 1, (n, 2))

    snapshot = {
        'experiment_dir': 'unused', 'gating': None, 'base_gate': 'Singlets', 'unstained_path': 'unstained.fcs',
        'layout': {'ceiling': 1e12, 'n_scatter': 2}, 'fluor_names': names, 'spectra': spectra,
        'controls': [{'label': n, 'fluor_idx': i, 'path': f'ctrl{i}', 'neg_path': None, 'peak': i, 'is_cell': False}
                     for i, n in enumerate(names)],
    }
    messages = []
    entry = sm.compute_spread_entry(snapshot, messages.append, loader)
    spread = np.array(entry['spread'])
    assert entry['fluor_names'] == names and entry['n_estimated'] == 3
    assert spread.shape == (3, 3) and np.all(np.diag(spread) == 0) and np.all(spread >= 0)
    assert spread[0, 1] > 1e-2 and len(messages) == 3
    assert sm.spread_is_current(entry, names, spectra)

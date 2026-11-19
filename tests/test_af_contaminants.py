"""Tests for the AF contaminant screen (af_contaminants) and plate neighbours (plate_layout)."""
from datetime import datetime, timedelta

import numpy as np

from honeychrome.controller_components import af_contaminants as afc
from honeychrome.controller_components import plate_layout as pl

N_DET = 24
PANEL = ['D0', 'D1', 'D2', 'D3', 'D4']


def _spectra(rng):
    centres = [3, 7, 11, 15, 19]
    x = np.arange(N_DET)
    S = np.array([np.exp(-0.5 * ((x - c) / 1.5) ** 2) for c in centres])
    af = np.exp(-0.5 * ((x - 5) / 6.0) ** 2) + 0.05
    return S / S.max(axis=1, keepdims=True), af / af.max()


def _sample(rng, n, af, contaminant=None, rate=0.0, amount=0.0):
    sc = rng.lognormal(mean=[8.0, 7.0], sigma=0.3, size=(n, 2))
    k = 2000.0 * (sc[:, 1] / np.median(sc[:, 1])) ** 1.5
    fl = k[:, None] * af[None, :] + rng.normal(0, 60, size=(n, N_DET))
    truth = np.zeros(n, dtype=bool)
    if contaminant is not None:
        truth[rng.choice(n, int(rate * n), replace=False)] = True
        fl[truth] += amount * contaminant
    return fl, sc, truth


def _well(well, plate='P1', rows=8, cols=12, minute=0):
    return pl.WellInfo(well=well, plate=plate, rows=rows, cols=cols,
                       start=datetime(2026, 1, 1, 10) + timedelta(minutes=minute))


def test_neighbour_candidates_spatial_carryover_and_tube_rack():
    layout = {'unst': _well('B2', minute=10), 'ctrl_a': _well('B3', minute=12), 'ctrl_b': _well('D5', minute=8),
              'full': _well('A1', minute=1), 'far': _well('H12', minute=2), 'other_plate': _well('B1', plate='P2')}
    labels = {'ctrl_a': 'D1', 'ctrl_b': 'D3'}
    c = pl.neighbour_candidates('unst', layout, PANEL, labels, {'unst'}, {'full': 'A1 Donor 7_P1.fcs'})
    assert set(c.spatial) == {'ctrl_a', 'full'}
    assert c.carryover == ['ctrl_b']                  # acquired 2 minutes before
    assert c.single == ['D1', 'D3']
    assert c.whole_panel
    rack = {k: pl.WellInfo(v.well, v.plate, 5, 8, v.start) for k, v in layout.items()}
    empty = pl.neighbour_candidates('unst', rack, PANEL, labels, {'unst'})
    assert not empty.single and not empty.whole_panel


def test_carryover_needs_a_short_gap():
    layout = {'unst': _well('A1', minute=60), 'ctrl': _well('H12', minute=0)}
    c = pl.neighbour_candidates('unst', layout, PANEL, {'ctrl': 'D0'}, {'unst'})
    assert c.carryover == [] and c.single == []


def test_loo_median_leaves_the_event_out():
    values = np.ones((40, 3))
    values[0] = 1e6
    feats = np.zeros((40, 1))
    feats[:, 0] = np.arange(40) * 1e-3
    assert np.allclose(afc.loo_median(values, feats, k=5)[0], 1.0)


def test_screen_removes_contaminant_and_little_else():
    rng = np.random.default_rng(3)
    S, af = _spectra(rng)
    sources = []
    for i in range(3):
        fl, sc, _ = _sample(rng, 6000, af)
        z = afc.dye_z_scores(fl, sc, S)
        sources.append(afc.summarise_source(f'clean{i}', z))
    null = afc.build_null('unstained', sources, len(PANEL))
    fl, sc, truth = _sample(rng, 6000, af, S[2], rate=0.02, amount=3000.0)
    keep, removed = afc.remove_events(afc.dye_z_scores(fl, sc, S, [2]), ['D2'], null, PANEL)
    assert removed['D2'] == int((~keep).sum())
    assert afc.unexpected_dyes(afc.tail_rates(afc.dye_z_scores(fl, sc, S)), ['D2'], null, PANEL) == []
    assert (~keep[truth]).mean() > 0.95
    assert (~keep[~truth]).mean() < 0.005


def test_contaminated_null_source_is_left_out():
    rng = np.random.default_rng(5)
    S, af = _spectra(rng)
    sources = []
    for i, rate in enumerate((0.0, 0.0, 0.0, 0.03)):
        fl, sc, _ = _sample(rng, 5000, af, S[4], rate=rate, amount=4000.0)
        z = afc.dye_z_scores(fl, sc, S)
        sources.append(afc.summarise_source(f's{i}', z))
    null = afc.build_null('unstained', sources, len(PANEL))
    assert ('s3', 4) in null.excluded
    assert [src.key for src in null.sources] == ['s0', 's1', 's2']
    assert null.reference[4] < afc.CONFIRM_MIN_PER_100K


def test_guard_rejects_a_dye_laden_candidate():
    rng = np.random.default_rng(7)
    S, af = _spectra(rng)
    x = np.arange(N_DET)
    library = np.array([af] + [np.exp(-0.5 * ((x - c) / w) ** 2) + 0.05 for c, w in ((4, 5), (8, 6), (6, 4))])
    library /= library.max(axis=1, keepdims=True)
    guard = {1: 0.02}
    clean = 0.6 * library[1] + 0.4 * library[2]
    dirty = library[2] + 0.4 * S[1]
    assert not afc.guard_rejects(clean, library, guard, S)
    assert afc.guard_rejects(dirty, library, guard, S)
    assert not afc.guard_rejects(dirty, library, None, S)


def test_guard_limits_take_the_largest_clean_share():
    srcs = [afc.NullSource(key='a', rates=np.zeros(3), tail=np.zeros((1, 3)), n_events=10,
                           shares=np.array([0.01, 0.05, 0.03])),
            afc.NullSource(key='b', rates=np.zeros(3), tail=np.zeros((1, 3)), n_events=10,
                           shares=np.array([0.015, 0.20, 0.04]))]
    null = afc.NullModel(kind='unstained', sources=srcs, thresholds=np.zeros(3), reference=np.zeros(3))
    assert afc.guard_limits(null, [0, 1, 2]) == {0: afc.GUARD_FLOOR, 1: 0.20, 2: 0.04}


def test_excluded_events_round_trip(tmp_path):
    keep = np.ones(10, dtype=bool)
    keep[[2, 7]] = False
    afc.save_excluded(keep, tmp_path, afc.excluded_file_name('P'))
    profiles = {'P': {'source_fcs': 'Raw/u.fcs',
                      'contaminant_screen': {'excluded_file': afc.excluded_file_name('P')}},
                'Q': {'source_fcs': 'Raw/v.fcs'}}
    idx = afc.excluded_for_sample(tmp_path, profiles, 'Raw/u.fcs')
    assert idx.tolist() == [2, 7]
    assert afc.excluded_for_sample(tmp_path, profiles, 'Raw/v.fcs') is None
    assert afc.keep_mask(idx, 10).tolist() == keep.tolist()
    assert afc.keep_mask(None, 10) is None
    assert afc.excluded_digest(idx) == afc.excluded_digest(np.array([2, 7]))


def test_get_raw_events_leaves_out_excluded_events():
    from honeychrome.controller_components.spectral_functions import get_raw_events

    class _Sample:
        pnn_labels = ['FL1', 'FL2', 'SSC']

        def get_events(self, source, col_order=None):
            return np.arange(15, dtype=float).reshape(5, 3)

    keep = np.array([True, False, True, True, False])
    fl, sc = get_raw_events(_Sample(), [0, 1], extra_channel_ids=[2], event_keep=keep)
    assert fl[:, 0].tolist() == [0.0, 6.0, 9.0]
    assert sc[:, 0].tolist() == [2.0, 8.0, 11.0]

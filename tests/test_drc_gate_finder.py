"""
test_drc_gate_finder.py
-----------------------
Gating strategies for a cluster (drc_gate_finder) on data with known
answers.

Usage:
    pytest tests/test_drc_gate_finder.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

gf = pytest.importorskip('drc_gate_finder')

N = 20_000


def _data(d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, d)).astype(np.float32)
    sample = rng.integers(0, 8, size=N)
    return X, sample


def _two_threshold_labels(X):
    """Cluster 0: marker 0 above 1 and marker 3 below 0. Cluster 1: marker 0
    above 1, marker 3 above 0. Cluster 2: everything else."""
    labels = np.full(len(X), 2)
    labels[X[:, 0] > 1] = 1
    labels[(X[:, 0] > 1) & (X[:, 3] < 0)] = 0
    return labels


def test_two_threshold_cluster_is_confirmed_with_its_two_markers():
    X, sample = _data()
    es = gf.select_events(X, _two_threshold_labels(X), sample, 0, seed=1)
    f = gf.find_gating_strategy(es)
    assert f.status == gf.CONFIRMED and f.met
    assert sorted(f.strategy.markers()) == [0, 3]
    assert abs(f.strategy.lo[0] - 1.0) < 0.15
    assert abs(f.strategy.hi[3] - 0.0) < 0.15
    assert f.validation.recall >= 0.90 and f.validation.purity >= 0.90
    assert len(f.steps) == 1 and {f.steps[0].x, f.steps[0].y} == {0, 3}
    assert f.compromise is None and f.separability is None


def test_downsampling_weights_keep_purity_of_the_full_data():
    X, sample = _data(seed=2)
    labels = _two_threshold_labels(X)
    full = gf.find_gating_strategy(gf.select_events(X, labels, sample, 0, seed=3))
    capped = gf.find_gating_strategy(gf.select_events(X, labels, sample, 0, seed=3,
                                                      background_cap=4_000))
    assert abs(capped.validation.purity - full.validation.purity) < 0.05
    assert abs(capped.validation.recall - full.validation.recall) < 0.05


def test_tilted_border_is_not_gateable():
    X, sample = _data(d=10, seed=4)
    score = X.sum(axis=1)
    labels = (score < np.quantile(score, 0.95)).astype(int)
    f = gf.find_gating_strategy(gf.select_events(X, labels, sample, 0, seed=5))
    assert not f.met
    assert f.status == gf.NOT_GATEABLE
    assert f.separability.purity > 0.8
    assert f.compromise is not None and len(f.tradeoff) == len(gf.BETAS)


def test_random_labels_are_a_continuum():
    X, sample = _data(seed=6)
    rng = np.random.default_rng(7)
    labels = (rng.random(N) > 0.05).astype(int)
    f = gf.find_gating_strategy(gf.select_events(X, labels, sample, 0, seed=8))
    assert not f.met
    assert f.status == gf.CONTINUUM
    assert f.separability.purity < 0.2


def test_separability_off_leaves_the_cluster_not_confirmed():
    X, sample = _data(seed=6)
    rng = np.random.default_rng(7)
    labels = (rng.random(N) > 0.05).astype(int)
    f = gf.find_gating_strategy(gf.select_events(X, labels, sample, 0, seed=8),
                                compromise=False, separability=False)
    assert f.status == gf.NOT_CONFIRMED and f.separability is None


def test_max_markers_is_respected():
    X, sample = _data(seed=9)
    labels = np.where((X[:, 0] > 0.5) & (X[:, 1] > 0.5) & (X[:, 2] > 0.5), 0, 1)
    es = gf.select_events(X, labels, sample, 0, seed=10)
    capped = gf.find_gating_strategy(es, gf.Targets(max_markers=2), separability=False)
    assert len(capped.strategy.markers()) <= 2 and not capped.met
    free = gf.find_gating_strategy(es, gf.Targets(max_markers=None))
    assert free.met and sorted(free.strategy.markers()) == [0, 1, 2]


def test_per_sample_and_impurity_tables():
    """With one marker allowed the gate on marker 0 also holds cluster 1, so
    cluster 1 makes up the impurity."""
    X, sample = _data(seed=11)
    labels = np.full(N, 2)
    labels[X[:, 0] > 1] = 0
    labels[(X[:, 0] > 1) & (X[:, 1] > 0.5)] = 1
    f = gf.find_gating_strategy(gf.select_events(X, labels, sample, 0, seed=12),
                                gf.Targets(max_markers=1), compromise=False, separability=False)
    assert f.strategy.markers() == [0]
    assert {r['sample'] for r in f.per_sample} == set(range(8))
    assert all(0 <= r['recall'] <= 1 and 0 <= r['purity'] <= 1 for r in f.per_sample)
    shares = dict(f.impurity)
    assert list(shares) == [1]
    assert shares[1] == pytest.approx(1 - f.validation.purity, abs=1e-6)


def test_several_clusters_as_one_target():
    X, sample = _data(seed=13)
    labels = _two_threshold_labels(X)
    f = gf.find_gating_strategy(gf.select_events(X, labels, sample, [0, 1], seed=14))
    assert f.met and f.strategy.markers() == [0]


def test_input_checks():
    X, sample = _data(seed=15)
    labels = np.ones(N, dtype=int)
    labels[:10] = 0
    with pytest.raises(ValueError):
        gf.select_events(X, labels, sample, 0)
    with pytest.raises(ValueError):
        gf.select_events(X, _two_threshold_labels(X), np.zeros(N, dtype=int), 0,
                         holdout=gf.HOLDOUT_SAMPLES)


def test_per_cluster_sample_with_weights_matches_the_full_data():
    """A fixed number of events per cluster, each weighted by the cells it
    stands for, gives the gates and metrics of the full data."""
    X, sample = _data(seed=16)
    labels = _two_threshold_labels(X)
    full = gf.find_gating_strategy(gf.select_events(X, labels, sample, 0, seed=17))
    rng = np.random.default_rng(18)
    keep = np.concatenate([rng.choice(np.flatnonzero(labels == c),
                                      min(1500, int((labels == c).sum())), replace=False)
                           for c in np.unique(labels)])
    counts = {c: int((labels == c).sum()) for c in np.unique(labels)}
    kept = {c: int((labels[keep] == c).sum()) for c in counts}
    w = np.array([counts[c] / kept[c] for c in labels[keep]])
    sampled = gf.find_gating_strategy(gf.select_events(X[keep], labels[keep], sample[keep], 0,
                                                       weights=w, seed=17))
    assert sorted(sampled.strategy.markers()) == sorted(full.strategy.markers())
    assert abs(sampled.validation.purity - full.validation.purity) < 0.05
    assert gf.select_events(X[keep], labels[keep], sample[keep], 0, weights=w).cluster_fraction \
        == pytest.approx(float((labels == 0).mean()), rel=1e-6)


def test_saved_finding_rebuilds_the_same_gate():
    X, sample = _data()
    es = gf.select_events(X, _two_threshold_labels(X), sample, 0, seed=1)
    f = gf.find_gating_strategy(es)
    names = [f'ch{i}' for i in range(X.shape[1])]
    saved = gf.finding_to_dict(f, names)
    rebuilt = gf.strategy_from_dict(saved['strategy'], names)
    assert np.array_equal(rebuilt.inside(X), f.strategy.inside(X))
    assert saved['steps'][0]['x'] in ('ch0', 'ch3')
    assert gf.strategy_from_dict(saved['strategy'], names[:2]) is None

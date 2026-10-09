"""
test_drc_time_estimate.py
-------------------------
Event counting, probe drawing, the two-size cost fit and extrapolation in
drc_time_estimate, and the staged clustering result, with the gating pipeline
replaced by in-memory arrays.

Usage:
    pytest tests/test_drc_time_estimate.py
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

import drc_pipeline  # noqa: E402
import drc_time_estimate as te  # noqa: E402

pytestmark = pytest.mark.numpy_only


def _state(ids, n_training_events=10_000, embeddings=None):
    return types.SimpleNamespace(training_sample_ids=list(ids),
                                 n_training_events=n_training_events,
                                 embeddings=embeddings or {})


@pytest.fixture
def gated(monkeypatch):
    """Patch the pipeline to serve arrays from a dict keyed by sample name."""
    data = {}
    monkeypatch.setattr(drc_pipeline, 'sample_abs_path', lambda c, rel: rel)
    monkeypatch.setattr(drc_pipeline, 'load_unmixed_gated',
                        lambda c, s, path, af_state=None: data[path])
    monkeypatch.setattr(drc_pipeline, 'transform_selected_channels',
                        lambda c, s, g: g[:, :3])
    return data


def test_allocate_rows_proportional_and_capped():
    assert te.allocate_rows([400, 100], 10) == [8, 2]
    assert te.allocate_rows([5, 5], 100) == [5, 5]
    assert te.allocate_rows([0, 0], 10) == [0, 0]


def test_format_duration():
    assert te.format_duration(45) == '45s'
    assert te.format_duration(750) == '12.5min'
    assert te.format_duration(4320) == '1.2hr'
    assert te.format_duration(float('nan')) == 'unknown'


def test_thresholds():
    assert te.TIMING_EVENT_THRESHOLD == 50_000
    assert te.event_threshold('dr', 'UMAP') == 50_000
    assert te.event_threshold('cl', 'Leiden') == 50_000
    assert te.event_threshold('cl', 'FlowSOM') == 2_000_000


def test_fit_stage_separates_fixed_and_per_event_cost():
    fit = te.fit_stage(1000, 3.0, 2000, 5.0)
    assert fit.resolved
    assert fit.per_event == pytest.approx(0.002)
    assert fit.fixed == pytest.approx(1.0)


def test_fit_stage_falls_back_to_proportional_scaling():
    fit = te.fit_stage(1000, 5.0, 2000, 4.9)
    assert not fit.resolved
    assert fit.fixed == 0.0
    assert fit.per_event == pytest.approx(4.9 / 2000)


def test_dr_extrapolates_fit_and_embed_separately(gated):
    rng = np.random.default_rng(0)
    gated['a'] = rng.normal(size=(60_000, 5))
    gated['b'] = rng.normal(size=(20_000, 5))
    sizes = []

    def probe(data, run_events):
        sizes.append((len(data), run_events))
        return {'fit': 2.0 + 0.001 * len(data), 'embed': 0.5 + 0.0002 * len(data)}

    est = te.estimate_run(None, _state(['a', 'b'], n_training_events=100_000), 'dr',
                          'UMAP', {}, probe, probe_repeats=2)
    assert sizes == [(1000, 80_000)] * 2 + [(2000, 80_000)] * 2
    assert est.total_events == 80_000 and est.embed_events == 80_000
    # fit: 2s + 1ms/event; embed: 0.5s per sample (2 samples) + 0.2ms/event
    factor = te.RUN_TIME_FACTORS[('dr', 'UMAP')]
    assert est.seconds == pytest.approx(factor * ((2.0 + 80.0) + (1.0 + 16.0)))


def test_dr_counts_fit_pool_and_embedded_events(gated):
    rng = np.random.default_rng(0)
    gated['a'] = rng.normal(size=(40_000, 5))
    gated['b'] = rng.normal(size=(10_000, 5))
    state = _state(['a', 'b'], n_training_events=5_000)
    est = te.estimate_run(None, state, 'dr', 'UMAP', {}, lambda d, n: None)
    assert est.total_events == 10_000
    assert est.embed_events == 50_000
    assert not est.needs_confirmation       # exactly at the threshold

    gated['b'] = rng.normal(size=(10_001, 5))
    est = te.estimate_run(None, state, 'dr', 'UMAP', {}, lambda d, n: None)
    assert est.needs_confirmation


def test_small_run_is_not_probed(gated):
    gated['a'] = np.zeros((1_000, 5))
    calls = []
    est = te.estimate_run(None, _state(['a']), 'cl', 'HDBSCAN',
                          {'_event_cap': None, '_space': 'raw'},
                          lambda d, n: calls.append(len(d)))
    assert est.total_events == 1_000 and not est.needs_confirmation and calls == []


def test_flowsom_is_only_timed_from_two_million_events(gated):
    rng = np.random.default_rng(1)
    gated['a'] = rng.normal(size=(300_000, 5))
    gated['b'] = rng.normal(size=(200_000, 5))
    params = {'_event_cap': None, '_space': 'raw'}
    calls = []
    est = te.estimate_run(None, _state(['a', 'b']), 'cl', 'FlowSOM', params,
                          lambda d, n: calls.append(len(d)))
    assert est.total_events == 500_000 and not est.needs_confirmation and calls == []

    est = te.estimate_run(None, _state(['a', 'b']), 'cl', 'HDBSCAN', params,
                          lambda d, n: calls.append(len(d)))
    assert est.needs_confirmation and calls == [1000, 2000]


def test_run_time_factors_scale_the_estimate(gated):
    gated['a'] = np.random.default_rng(2).normal(size=(100_000, 5))
    params = {'_event_cap': None, '_space': 'raw'}

    def per_event(data, run_events):
        return {'fit': 0.001 * len(data)}

    def estimate(kind, algo):
        return te.estimate_run(None, _state(['a']), kind, algo, params, per_event)

    unscaled = estimate('cl', 'KMeans')
    assert unscaled.seconds == pytest.approx(100.0)
    assert 'multiplied' not in te.estimate_message(unscaled)
    for algo, factor in (('Leiden', 2), ('HDBSCAN', 2)):
        scaled = estimate('cl', algo)
        assert scaled.seconds == pytest.approx(100.0 * factor)
        assert f'multiplied by {factor}' in te.estimate_message(scaled)
    assert te.RUN_TIME_FACTORS[('dr', 'PaCMAP')] == 5.0
    assert te.RUN_TIME_FACTORS[('dr', 'UMAP')] == 2.0
    assert te.RUN_TIME_FACTORS[('cl', 'FlowSOM')] == 5.0


@pytest.mark.parametrize('repeats', [1, 2])
def test_probe_sizes_and_repeats(gated, repeats):
    rng = np.random.default_rng(1)
    gated['a'] = rng.normal(size=(60_000, 5))
    shapes = []
    te.estimate_run(None, _state(['a']), 'cl', 'HDBSCAN',
                    {'_event_cap': None, '_space': 'raw'},
                    lambda d, n: shapes.append(d.shape), probe_repeats=repeats)
    assert shapes == ([(te.TIMING_PROBE_SMALL_EVENTS, 3)] * repeats
                      + [(te.TIMING_PROBE_EVENTS, 3)] * repeats)


def test_probe_is_proportional_to_samples(gated):
    rng = np.random.default_rng(2)
    for name, tag, n in (('a', 1.0, 60_000), ('b', 2.0, 20_000)):
        arr = rng.normal(size=(n, 5))
        arr[:, 0] = tag
        gated[name] = arr
    seen = []
    te.estimate_run(None, _state(['a', 'b']), 'cl', 'HDBSCAN',
                    {'_event_cap': None, '_space': 'raw'},
                    lambda d, n: seen.append(d[:, 0].copy()))
    small, large = seen
    assert (large == 2.0).sum() == 500
    assert 200 <= (small == 2.0).sum() <= 300     # shuffled, so the slice is representative


def test_clustering_in_dr_space_uses_embeddings(gated):
    rng = np.random.default_rng(3)
    gated['a'] = rng.normal(size=(1_000, 5))
    state = _state(['a'], embeddings={'UMAP': {'a': rng.normal(size=(80_000, 2))}})
    shapes = []
    est = te.estimate_run(None, state, 'cl', 'Leiden',
                          {'_event_cap': None, '_space': 'dr', '_dr_algo': 'UMAP'},
                          lambda d, n: shapes.append(d.shape))
    assert est.total_events == 80_000 and shapes[-1] == (te.TIMING_PROBE_EVENTS, 2)


def test_probe_failure_is_reported(gated):
    gated['a'] = np.zeros((60_000, 5))

    def fail(data, run_events):
        raise ValueError('nope')

    est = te.estimate_run(None, _state(['a']), 'cl', 'Leiden',
                          {'_event_cap': None, '_space': 'raw'}, fail)
    assert est.error == 'nope' and est.needs_confirmation


def test_staged_result_leaves_state_untouched():
    import drc_clustering
    state = types.SimpleNamespace(
        cluster_labels={'old': 1}, cluster_marker_values={}, cluster_dr_positions={'p': 2},
        cluster_colors={0: 'red'}, n_clusters=3, active_clustering_algorithm='FlowSOM',
        trained_reducers={'FlowSOM': 'keep'}, embeddings={'UMAP': 5}, gated_data_cache={})
    staged = drc_clustering.StagedResult(state)
    staged.cluster_labels = {'new': 9}
    staged.cluster_colors[1] = 'blue'
    staged.n_clusters = 7
    staged.trained_reducers['Leiden'] = 'new'
    staged.gated_data_cache['k'] = 1
    assert staged.embeddings == {'UMAP': 5}              # inputs read through
    assert state.cluster_labels == {'old': 1} and state.n_clusters == 3
    assert state.cluster_colors == {0: 'red'}
    assert state.trained_reducers == {'FlowSOM': 'keep'}
    assert state.gated_data_cache == {'k': 1}            # caches are shared on purpose

    drc_clustering.commit_clustering_result(state, staged, 'Leiden')
    assert state.cluster_labels == {'new': 9} and state.n_clusters == 7
    assert state.active_clustering_algorithm == 'FlowSOM'
    assert state.trained_reducers == {'FlowSOM': 'keep', 'Leiden': 'new'}

"""
test_drc_time_estimate.py
-------------------------
The run-time model in drc_time_estimate (reference curves, per-machine
factors, stage plans, probes, learning from runs) and the staged clustering
result, with the gating pipeline replaced by in-memory arrays.

Usage:
    pytest tests/test_drc_time_estimate.py
"""

import math
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
                                 embeddings=embeddings or {}, selected_gates=['g'])


@pytest.fixture
def samples(monkeypatch):
    """Serve gated counts and features from dicts keyed by sample name."""
    counts, feats, cached = {}, {}, set()
    monkeypatch.setattr(drc_pipeline, 'estimate_gated_count',
                        lambda c, s, rel, af: (counts[rel], True))
    monkeypatch.setattr(drc_pipeline, 'is_gated_cached',
                        lambda c, s, rel, af_state=None: rel in cached)
    monkeypatch.setattr(drc_pipeline, 'sample_file_events', lambda c, rel: 10 * counts[rel])
    monkeypatch.setattr(drc_pipeline, 'preview_features',
                        lambda c, s, rel, af: feats[rel])
    monkeypatch.setattr(drc_pipeline, 'load_sample_features',
                        lambda c, s, rel, af_state=None: feats[rel])
    return types.SimpleNamespace(counts=counts, feats=feats, cached=cached)


def test_allocate_rows_proportional_and_capped():
    assert te.allocate_rows([400, 100], 10) == [8, 2]
    assert te.allocate_rows([5, 5], 100) == [5, 5]
    assert te.allocate_rows([0, 0], 10) == [0, 0]


def test_format_duration():
    assert te.format_duration(45) == '45s'
    assert te.format_duration(750) == '12.5min'
    assert te.format_duration(4320) == '1.2hr'
    assert te.format_duration(float('nan')) == 'unknown'


def test_threads_key(monkeypatch):
    monkeypatch.setattr(te.os, 'cpu_count', lambda: 8)
    assert te.threads_key('UMAP', {'n_jobs': -1}) == 't8'
    assert te.threads_key('UMAP', {'n_jobs': 1}) == 't1'
    assert te.threads_key('PaCMAP', {}) == 'any'
    assert te.threads_key('FlowSOM', {}) == 't8'
    assert te.calibration_key('UMAP', 'io', 't8') == 'io'
    assert te.calibration_key('UMAP', 'jit', 't8') == 'UMAP/fit/t8'


def test_work_scale_follows_epochs_and_iterations():
    assert te.work_scale('UMAP', 'fit', {'n_epochs': 500}) == pytest.approx(1.0)
    assert te.work_scale('UMAP', 'fit', {'n_epochs': 200}) == pytest.approx(0.24 + 0.76 * 0.4)
    assert te.work_scale('tSNE', 'fit_fft', {'n_iter': 500}) == pytest.approx(750 / 1250)
    assert te.work_scale('FlowSOM', 'som', {}) == 1.0


def test_dr_plan_splits_pool_and_embedded_events(samples):
    samples.counts.update(a=15_000, b=4_000)
    state = _state(['a', 'b'], n_training_events=10_000)
    counts = te.count_samples(None, state, ['a', 'b'], None)
    stages, n_fit, n_embed = te.plan_stages('dr', 'UMAP', {'n_jobs': -1}, state, counts,
                                            jit_warm=False)
    assert (n_fit, n_embed) == (14_000, 5_000)
    names = [s.name for s in stages]
    assert names == ['io', 'jit', 'fit', 'index', 'embed']
    embed = stages[-1]
    assert (embed.events, embed.pool, embed.calls) == (5_000, 14_000, 1)
    assert stages[3].events == 14_000             # hnswlib index on the pool
    assert stages[0].events == 190_000            # file events of uncached samples


def test_umap_embed_cost_follows_epochs():
    ref = te.reference_seconds('UMAP', 'embed', 40_000, {'n_epochs': 200}, pool=100_000, calls=1)
    more = te.reference_seconds('UMAP', 'embed', 40_000, {'n_epochs': 500}, pool=100_000, calls=1)
    per_event = ref - te.EMBED_REFERENCE['UMAP'].per_call
    assert more - te.EMBED_REFERENCE['UMAP'].per_call == pytest.approx(
        per_event * (1 + 1.5 * te.UMAP_EMBED_LAYOUT_SHARE))


def test_tsne_fit_stage_follows_gradient_method(monkeypatch):
    monkeypatch.setattr(te.drc_jit, 'opentsne_has_openmp', lambda: False)
    assert te.fit_stage_name('tSNE', 9_999) == ['fit_bh']
    assert te.fit_stage_name('tSNE', 10_000) == ['fit_fft']
    monkeypatch.setattr(te.drc_jit, 'opentsne_has_openmp', lambda: True)
    assert te.fit_stage_name('tSNE', 49_999) == ['fit_bh']
    assert te.tsne_gradient_method(49_999) == 'bh'
    assert te.fit_stage_name('tSNE', 50_000) == ['fit_fft']
    assert te.fit_stage_name('FlowSOM', 10) == ['som', 'map']


def test_opentsne_openmp_is_read_from_the_extension(monkeypatch, tmp_path):
    import importlib.util
    import types
    for blob, expected in ((b'xx___kmpc_fork_callxx', True), (b'GOMP_parallel', True),
                           (b'_vcomp_fork', True), (b'libSystem libc++', False)):
        ext = tmp_path / '_tsne.so'
        ext.write_bytes(blob)
        monkeypatch.setattr(importlib.util, 'find_spec',
                            lambda name, ext=ext: types.SimpleNamespace(origin=str(ext)))
        monkeypatch.setattr(te.drc_jit, '_opentsne_openmp', None)
        assert te.drc_jit.opentsne_has_openmp() is expected


def test_prior_then_stored_factor(samples):
    samples.counts.update(a=200_000)
    samples.cached.add('a')
    state = _state(['a'], n_training_events=200_000)
    counts = te.count_samples(None, state, ['a'], None)
    stages, *_ = te.plan_stages('dr', 'PaCMAP', {}, state, counts)
    prior = te.price_stages('PaCMAP', 'any', stages, {}, {})
    ref = te.REFERENCE[('PaCMAP', 'fit')].seconds(200_000)
    assert prior == pytest.approx(te.PRIOR_FACTOR * ref)
    cal = {'PaCMAP/fit/any': {'f': 1.5, 'n': 2, 'src': 'run', 'v': te.CALIBRATION_VERSION}}
    stages, *_ = te.plan_stages('dr', 'PaCMAP', {}, state, counts)
    assert te.price_stages('PaCMAP', 'any', stages, cal, {}) == pytest.approx(1.5 * ref)
    assert stages[0].source == '2 run(s)'


def test_old_calibration_version_is_ignored():
    cal = {'k': {'f': 2.0, 'n': 1, 'src': 'run', 'v': te.CALIBRATION_VERSION - 1}}
    assert te._stored_factor(cal, 'k') is None


def test_small_run_is_not_probed(samples):
    samples.counts.update(a=2_000)
    samples.feats['a'] = np.zeros((2_000, 5), np.float32)
    calls = []
    est = te.estimate_run(None, _state(['a']), 'cl', 'Leiden',
                          {'_event_cap': None, '_space': 'raw', 'n_neighbors': 15,
                           'resolution': 1.0}, {})
    assert not calls and not est.needs_confirmation and est.seconds < te.CONFIRM_SECONDS


def test_long_run_is_probed_and_calibrated(samples, monkeypatch):
    rng = np.random.default_rng(0)
    for name in 'abc':
        samples.counts[name] = 400_000
        samples.feats[name] = rng.normal(size=(8_000, 5)).astype(np.float32)
    params = {'xdim': 5, 'ydim': 5, 'n_metaclusters': 4, 'n_iter': 5,
              '_event_cap': None, '_space': 'raw'}
    seen = {}

    def fake_probe(algo, p):
        def probe(data):
            seen['n'] = len(data)
            return {'som': te.REFERENCE[('FlowSOM', 'som')].seconds(len(data)) * 2.0,
                    'map': te.REFERENCE[('FlowSOM', 'map')].seconds(len(data)) * 2.0}
        return probe
    monkeypatch.setattr(te, 'clustering_probe_fn', fake_probe)
    monkeypatch.setattr(te, 'CONFIRM_SECONDS', 0.0)
    est = te.estimate_run(None, _state(['a', 'b', 'c']), 'cl', 'FlowSOM', params, {})
    assert seen['n'] == te.PROBE_EVENTS['FlowSOM']
    threads = te.threads_key('FlowSOM', params)
    assert est.calibration_updates[f'FlowSOM/som/{threads}']['f'] == pytest.approx(2.0)
    som = next(s for s in est.stages if s.name == 'som')
    assert som.source == 'probe' and som.factor == pytest.approx(2.0)


def test_tsne_fft_is_calibrated_from_one_thread_barnes_hut(samples, monkeypatch):
    rng = np.random.default_rng(1)
    for name in 'ab':
        samples.counts[name] = 400_000
        samples.feats[name] = rng.normal(size=(8_000, 5)).astype(np.float32)
    params = {'perplexity': 30, 'n_iter': 500, 'n_jobs': 4}
    seen = {}

    def probe(fit_data, embed_data, single_thread=False):
        seen['single_thread'] = single_thread
        bh = te.reference_seconds('tSNE', 'fit_bh', len(fit_data), params)
        return {'fit': bh * te.TSNE_BH_ONE_THREAD * 1.2,
                'embed': 2 * te.reference_seconds('tSNE', 'embed', len(embed_data), params,
                                                  pool=len(fit_data), calls=1)}
    monkeypatch.setattr(te, 'CONFIRM_SECONDS', 0.0)
    state = _state(['a', 'b'], n_training_events=100_000)
    stored_bh = {'tSNE/fit_bh/t4': {'f': 0.4, 'n': 1, 'src': 'run', 'v': te.CALIBRATION_VERSION}}
    est = te.estimate_run(None, state, 'dr', 'tSNE', params, stored_bh, dr_probe_fn=probe)
    assert seen['single_thread'] is True
    assert est.calibration_updates['tSNE/fit_fft/t4']['f'] == pytest.approx(1.2)
    assert est.calibration_updates['tSNE/embed/t4']['f'] == pytest.approx(1.2)
    assert 'tSNE/fit_bh/t4' not in est.calibration_updates
    fit = next(s for s in est.stages if s.name == 'fit_fft')
    assert fit.source == 'probe' and fit.factor == pytest.approx(1.2)


def test_probe_with_too_few_events_keeps_the_prior(samples, monkeypatch):
    samples.counts.update(a=1_000_000)
    samples.feats['a'] = np.zeros((300, 5), np.float32)
    monkeypatch.setattr(te, 'CONFIRM_SECONDS', 0.0)
    est = te.estimate_run(None, _state(['a']), 'cl', 'Leiden',
                          {'_event_cap': None, '_space': 'raw', 'n_neighbors': 15,
                           'resolution': 1.0}, {})
    assert 'events to time' in est.error
    assert not est.calibration_updates
    assert all(s.source == 'prior' for s in est.stages)


def test_clustering_in_dr_space_counts_embeddings(samples):
    samples.counts.update(a=50, b=50)
    emb = {'a': np.zeros((30_000, 2)), 'b': np.zeros((20_000, 2))}
    state = _state(['a', 'b'], embeddings={'UMAP': emb})
    est = te.estimate_run(None, state, 'cl', 'HDBSCAN',
                          {'_event_cap': 10_000, '_space': 'dr', '_dr_algo': 'UMAP',
                           'min_cluster_size': 25, 'min_samples': 0,
                           'cluster_selection_epsilon': 0.0}, {})
    assert est.total_events == 20_000


def test_observe_run_updates_factors():
    params = {'n_jobs': 1, 'n_epochs': 200}
    ref_fit = te.reference_seconds('UMAP', 'fit', 100_000, params)
    obs = te.observe_run('UMAP', params, {}, {'fit': 3 * ref_fit, 'io': 10.0, 'embed': 0.1},
                         {'fit': 100_000, 'io': 2_000_000, 'embed': 1_000})
    assert obs['UMAP/fit/t1']['f'] == pytest.approx(3.0)
    assert obs['io']['f'] == pytest.approx(10.0 / (te.IO_SECONDS_PER_EVENT * 2_000_000))
    assert 'UMAP/embed/t1' not in obs                       # under 0.2 s: too noisy
    again = te.observe_run('UMAP', params, obs, {'fit': ref_fit}, {'fit': 100_000})
    assert again['UMAP/fit/t1']['f'] == pytest.approx(math.sqrt(3.0))
    assert again['UMAP/fit/t1']['n'] == 2
    skipped = te.observe_run('UMAP', params, {}, {'fit': ref_fit, 'embed': 5.0},
                             {'fit': 100_000, 'embed': 100_000}, pool=100_000, calls=1,
                             jit_included=True)
    assert skipped == {}


def test_probe_never_overrides_runs():
    run = {'f': 2.0, 'n': 3, 'src': 'run', 'v': te.CALIBRATION_VERSION}
    assert te.updated_entry(run, 9.0, 'probe') is run
    probe = te.updated_entry(None, 4.0, 'probe')
    after = te.updated_entry(probe, 1.0, 'run')
    assert after['src'] == 'run'
    assert after['f'] == pytest.approx(math.exp(0.25 * math.log(4.0)))


def test_staged_result_leaves_state_untouched():
    import drc_clustering
    state = types.SimpleNamespace(
        cluster_labels={'old': 1}, cluster_marker_values={}, cluster_dr_positions={'p': 2},
        cluster_colors={0: 'red'}, n_clusters=3, active_clustering_algorithm='FlowSOM',
        trained_reducers={'FlowSOM': 'keep'}, embeddings={'UMAP': 5}, gated_data_cache={},
        cluster_rows={'old': 4})
    staged = drc_clustering.StagedResult(state)
    staged.cluster_labels = {'new': 9}
    staged.cluster_rows = {'new': 8}
    staged.cluster_colors[1] = 'blue'
    staged.n_clusters = 7
    staged.trained_reducers['Leiden'] = 'new'
    staged.gated_data_cache['k'] = 1
    assert staged.embeddings == {'UMAP': 5}              # inputs read through
    assert state.cluster_labels == {'old': 1} and state.n_clusters == 3
    assert state.cluster_rows == {'old': 4}
    assert state.cluster_colors == {0: 'red'}
    assert state.trained_reducers == {'FlowSOM': 'keep'}
    assert state.gated_data_cache == {'k': 1}            # caches are shared on purpose

    drc_clustering.commit_clustering_result(state, staged, 'Leiden')
    assert state.cluster_labels == {'new': 9} and state.n_clusters == 7
    assert state.cluster_rows == {'new': 8}
    assert state.active_clustering_algorithm == 'FlowSOM'
    assert state.trained_reducers == {'FlowSOM': 'keep', 'Leiden': 'new'}

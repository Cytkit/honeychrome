"""
test_joint_unmix_export.py
--------------------------
Batch export through joint_unmix_export.JointUnmixExporter, the run-time
estimate and confirm/cancel flow (JointUnmixRun), the AutoSpectral
Optimization exporter built on them, and export_unmixed_sample()'s
extra_keywords, run on a synthetic experiment (see af_synthetic.py).

Kernel-dependent tests skip when _autospectral_opt_kernel is not built.

Usage:
    pytest tests/test_joint_unmix_export.py -m numpy_only
    pytest tests/test_joint_unmix_export.py -m ui
    pytest tests/test_joint_unmix_export.py
"""

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import honeychrome.settings as hc_settings
from honeychrome.controller_components import functions as fn
from honeychrome.controller_components.autospectral_functions import af_index_lookup

import af_synthetic as syn

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

import joint_unmix_export as jue  # noqa: E402

AF_A = hc_settings.af_abundance_channel
AF_I = hc_settings.af_index_channel


def _kernel_available():
    try:
        from autospectral_opt_kernel_wrapper import AUTOSPECTRAL_OPT_KERNEL_AVAILABLE
    except ImportError:
        return False
    return AUTOSPECTRAL_OPT_KERNEL_AVAILABLE


requires_kernel = pytest.mark.skipif(
    not _kernel_available(),
    reason='compiled AutoSpectral Optimization kernel not built '
           '(run bundled_plugins/build_autospectral_opt_kernel.py)',
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXTRA_SAMPLE_2 = 'Raw/sample_2.fcs'
EXTRA_SAMPLE_3 = 'Raw/sample_3.fcs'


def _three_af_samples(tmp_path, sizes):
    """AF_SAMPLE plus two more samples with AF assigned; PLAIN_SAMPLE has none.

    AF_SAMPLE and sample_2 share the 'AF profile' assignment; sample_3 has
    'Other profile' + 'AF profile'.
    """
    controller = syn.make_experiment(tmp_path, n_events=sizes[0])
    samples = controller.experiment.samples
    for key, n, seed in ((EXTRA_SAMPLE_2, sizes[1], 12), (EXTRA_SAMPLE_3, sizes[2], 13)):
        syn.write_raw_fcs(tmp_path / key, syn.raw_events(n, seed=seed))
        samples['all_samples'][key] = Path(key).stem
    samples['sample_af_profiles'][EXTRA_SAMPLE_2] = ['AF profile']
    samples['sample_af_profiles'][EXTRA_SAMPLE_3] = ['Other profile', 'AF profile']
    return controller


def _ols_unmix_fn(layout, af_scale=7.0, af_idx=2):
    """Deterministic stand-in for the kernel: plain OLS, constant AF values,
    and one extra channel holding twice the Time value."""
    spectra = layout.reference_spectra
    projector = np.linalg.solve(spectra @ spectra.T, spectra)
    time_col = layout.pnn_raw_export.index('Time')

    def unmix_fn(raw_fl_chunk, raw_chunk, af):
        n = len(raw_fl_chunk)
        return {
            'unmixed': raw_fl_chunk @ projector.T,
            'af_scale': np.full(n, af_scale),
            'af_idx': np.full(n, af_idx),
            'extra': 2.0 * raw_chunk[:, [time_col]],
        }

    return unmix_fn


def _run_exporter(exporter):
    errors = []
    exporter.error.connect(errors.append)
    exporter.run()
    return errors


def _f32(values):
    """Values as written to FCS (float32) and read back (float64)."""
    return np.asarray(values, dtype=np.float32).astype(np.float64)


def _text_value(text, key):
    for candidate in (key, key.lower(), f'${key}', f'${key.lower()}'):
        if candidate in text:
            return text[candidate]
    raise KeyError(key)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

@pytest.mark.numpy_only
@pytest.mark.parametrize('seconds, text', [
    (0.4, '0s'), (45, '45s'), (90, '1.5min'), (3540, '59.0min'),
    (7200, '2.0hr'), (float('nan'), 'unknown'), (float('inf'), 'unknown'), (None, 'unknown'),
])
def test_format_duration(seconds, text):
    assert jue.format_duration(seconds) == text


@pytest.mark.numpy_only
def test_allocate_probe_rows_is_proportional_and_sums_to_the_probe():
    assert jue.allocate_probe_rows([1000, 3000, 6000], 5000) == [500, 1500, 3000]
    alloc = jue.allocate_probe_rows([1, 1, 1], 2)
    assert sum(alloc) == 2 and max(alloc) == 1
    alloc = jue.allocate_probe_rows([10_000, 0, 333], 5000)
    assert alloc[1] == 0 and sum(alloc) == 5000


@pytest.mark.numpy_only
def test_allocate_probe_rows_takes_every_event_of_a_small_run():
    assert jue.allocate_probe_rows([7, 3], 5000) == [7, 3]
    assert jue.allocate_probe_rows([0, 0], 5000) == [0, 0]


@pytest.mark.numpy_only
def test_most_common_af_assignment_prefers_the_majority_then_first_seen(tmp_path):
    controller = _three_af_samples(tmp_path, (500, 500, 500))
    layout = jue.build_export_layout(controller)
    keys = [syn.PLAIN_SAMPLE, EXTRA_SAMPLE_3, syn.AF_SAMPLE, EXTRA_SAMPLE_2]
    assert jue.most_common_af_assignment(keys, layout) == ('AF profile',)

    tie = dataclasses.replace(layout, sample_af_profiles={
        syn.AF_SAMPLE: ['Other profile', 'AF profile'],
        EXTRA_SAMPLE_2: ['AF profile'],
    })
    assert jue.most_common_af_assignment([syn.AF_SAMPLE, EXTRA_SAMPLE_2], tie) == ('Other profile', 'AF profile')
    assert jue.most_common_af_assignment([syn.PLAIN_SAMPLE], layout) == ()


@pytest.mark.numpy_only
def test_af_library_needs_two_spectra_and_carries_the_experiment_index(tmp_path):
    controller = syn.make_experiment(tmp_path, n_events=200)
    layout = jue.build_export_layout(controller)
    af = jue.af_library_for_sample(layout, syn.AF_SAMPLE)
    np.testing.assert_array_equal(af.spectra, syn.af_spectra())
    np.testing.assert_array_equal(af.index_map, af_index_lookup(layout.af_profiles, ['AF profile']))
    assert jue.af_library_for_sample(layout, syn.PLAIN_SAMPLE) is None

    one_spectrum = dataclasses.replace(layout, af_profiles={'Single': {'spectra': [syn.af_spectra()[0].tolist()]}})
    assert jue.af_library_for_profiles(one_spectrum, ['Single']) is None


# ---------------------------------------------------------------------------
# Partial FCS reads
# ---------------------------------------------------------------------------

@pytest.mark.numpy_only
def test_fcs_event_count_reads_tot_from_the_header(tmp_path):
    path = tmp_path / 'events.fcs'
    syn.write_raw_fcs(path, syn.raw_events(1234, seed=4))
    assert jue.fcs_event_count(path) == 1234
    bad = tmp_path / 'not_fcs.fcs'
    bad.write_bytes(b'not an fcs file')
    assert jue.fcs_event_count(bad) is None


@pytest.mark.numpy_only
def test_read_fcs_rows_matches_a_full_read(tmp_path):
    path = tmp_path / 'events.fcs'
    syn.write_raw_fcs(path, syn.raw_events(3000, seed=5))
    pnn, full, _text = syn.read_fcs(path)
    rows = np.array([0, 1, 17, 1500, 2999])
    channels = list(reversed(syn.DET_PNN)) + ['SSC-A']

    part = jue.read_fcs_rows(path, rows, channels)
    np.testing.assert_array_equal(part, full[np.ix_(rows, [pnn.index(c) for c in channels])])


@pytest.mark.numpy_only
def test_read_fcs_rows_applies_gain_as_flowkit_raw_events_do(tmp_path):
    events = syn.raw_events(500, seed=6)
    keywords = {'$CYT': 'Synthetic'}
    for i, name in enumerate(syn.RAW_PNN, start=1):
        keywords[f'$P{i}N'] = name
        keywords[f'$P{i}B'] = '32'
        keywords[f'$P{i}E'] = '0,0'
        keywords[f'$P{i}R'] = '100' if name == 'Time' else str(int(syn.MAGNITUDE_CEILING))
    keywords[f'$P{syn.RAW_PNN.index("D1-A") + 1}G'] = '2.0'
    keywords[f'$P{syn.RAW_PNN.index("Time") + 1}G'] = '0.01'
    path = tmp_path / 'gain.fcs'
    fn.write_fcs(events, keywords, path)

    pnn, full, _text = syn.read_fcs(path)
    rows = np.arange(0, 500, 7)
    channels = ['Time', 'D1-A', 'D2-A']
    part = jue.read_fcs_rows(path, rows, channels)
    np.testing.assert_allclose(part, full[np.ix_(rows, [pnn.index(c) for c in channels])], rtol=1e-12)
    np.testing.assert_allclose(part[:, 1], _f32(events[rows, syn.RAW_PNN.index('D1-A')]) / 2.0, rtol=1e-12)


@pytest.mark.numpy_only
def test_read_fcs_rows_zero_fills_missing_channels_and_rejects_unreadable_files(tmp_path):
    path = tmp_path / 'events.fcs'
    syn.write_raw_fcs(path, syn.raw_events(100, seed=7))
    part = jue.read_fcs_rows(path, [3, 4], ['D1-A', 'Not a channel'])
    assert np.all(part[:, 1] == 0) and np.all(part[:, 0] != 0)
    with pytest.raises(IndexError):
        jue.read_fcs_rows(path, [100], ['D1-A'])

    bad = tmp_path / 'not_fcs.fcs'
    bad.write_bytes(b'not an fcs file')
    assert jue.read_fcs_rows(bad, [0], ['D1-A']) is None


# ---------------------------------------------------------------------------
# Run-time estimate
# ---------------------------------------------------------------------------

@pytest.mark.numpy_only
def test_estimate_probes_every_sample_in_proportion_with_the_common_af_set(tmp_path):
    sizes = (2000, 6000, 12000)
    controller = _three_af_samples(tmp_path, sizes)
    layout = jue.build_export_layout(controller)
    keys = [syn.AF_SAMPLE, syn.PLAIN_SAMPLE, EXTRA_SAMPLE_2, EXTRA_SAMPLE_3]

    calls = []

    def probe_fn(raw_fl, af):
        calls.append((raw_fl.copy(), af.profile_names))

    est = jue.estimate_unmix_time(keys, layout, probe_fn)

    assert len(calls) == 1
    block, profile_names = calls[0]
    assert profile_names == ('AF profile',)
    assert block.shape == (jue.PROBE_EVENTS, syn.N_DET)
    assert est.n_samples == 3 and est.total_events == sum(sizes)
    assert est.skipped_samples == (syn.PLAIN_SAMPLE,)
    assert est.probe_events == jue.PROBE_EVENTS
    assert est.af_profile_names == ('AF profile',)
    assert math.isfinite(est.seconds) and est.seconds > 0
    assert est.seconds == pytest.approx(est.total_events / est.events_per_second)

    # every probe row is a real event, and each sample contributes in
    # proportion to its event count
    source = {}
    for key in (syn.AF_SAMPLE, EXTRA_SAMPLE_2, EXTRA_SAMPLE_3):
        pnn, events, _text = syn.read_fcs(tmp_path / key)
        for row in events[:, [pnn.index(c) for c in syn.DET_PNN]]:
            source[row.tobytes()] = key
    origin = [source.get(np.ascontiguousarray(row).tobytes()) for row in block]
    assert None not in origin
    expected = dict(zip((syn.AF_SAMPLE, EXTRA_SAMPLE_2, EXTRA_SAMPLE_3),
                        jue.allocate_probe_rows(sizes, jue.PROBE_EVENTS)))
    assert {k: origin.count(k) for k in expected} == expected


@pytest.mark.numpy_only
def test_estimate_skips_the_probe_when_the_run_is_no_larger_than_it(tmp_path):
    controller = _three_af_samples(tmp_path, (1000, 1500, 2000))
    layout = jue.build_export_layout(controller)
    calls = []
    est = jue.estimate_unmix_time([syn.AF_SAMPLE, EXTRA_SAMPLE_2, EXTRA_SAMPLE_3], layout,
                                  lambda raw_fl, af: calls.append(1))
    assert not calls
    assert est.probe_events == 0 and est.total_events == 4500


@pytest.mark.numpy_only
def test_estimate_with_no_usable_sample(tmp_path):
    controller = syn.make_experiment(tmp_path, n_events=200)
    layout = jue.build_export_layout(controller)
    est = jue.estimate_unmix_time([syn.PLAIN_SAMPLE], layout, lambda raw_fl, af: None)
    assert est.n_samples == 0 and est.skipped_samples == (syn.PLAIN_SAMPLE,)


@pytest.mark.numpy_only
def test_estimate_message_reports_the_whole_run():
    est = jue.UnmixTimeEstimate(
        n_samples=6, total_events=8_400_000, probe_events=5000, probe_seconds=1.6,
        events_per_second=3125.0, seconds=2688.0, af_profile_names=('Spleen AF',),
        skipped_samples=('Raw/x.fcs',),
    )
    text = jue.estimate_message(est)
    assert '44.8min' in text and '8,400,000 events' in text and '6 sample(s)' in text
    assert 'Spleen AF' in text and '1 selected sample(s)' in text
    assert text.endswith('Start unmixing?')


# ---------------------------------------------------------------------------
# Export layout and exporter
# ---------------------------------------------------------------------------

@pytest.mark.numpy_only
def test_layout_transfer_matrix_is_uncompensated(tmp_path):
    spillover = np.eye(syn.N_FLUOR)
    spillover[0, 1] = 0.08
    controller = syn.make_experiment(tmp_path, n_events=100, spillover=spillover)
    layout = jue.build_export_layout(controller)
    unmixing_matrix = np.array(controller.experiment.process['unmixing_matrix'])
    block = layout.transfer_matrix.T[np.ix_(layout.fl_ids_unmixed, list(layout.fl_ids_raw))]
    np.testing.assert_array_equal(block, unmixing_matrix)
    np.testing.assert_array_equal(layout.spillover, spillover)


@pytest.mark.numpy_only
def test_exporter_writes_unmixed_af_and_extra_channels(tmp_path):
    controller = syn.make_experiment(tmp_path, n_events=1500)
    layout = jue.build_export_layout(controller)
    exporter = jue.JointUnmixExporter(
        [syn.AF_SAMPLE, syn.PLAIN_SAMPLE], layout, _ols_unmix_fn(layout),
        extra_channels=['FC Test'], output_root=tmp_path / 'Out', name_suffix=' FlowCode',
        unmixing_method='FlowCode', extra_keywords={'FLOWCODEUNMIX': '0.1.0'},
    )
    assert not _run_exporter(exporter)

    out = tmp_path / 'Out' / 'af_sample FlowCode (Unmixed).fcs'
    assert out.exists()
    assert not (tmp_path / 'Out' / 'plain_sample FlowCode (Unmixed).fcs').exists()

    pnn, events, text = syn.read_fcs(out)
    raw_pnn, raw, _raw_text = syn.read_fcs(tmp_path / syn.AF_SAMPLE)
    assert len(pnn) == len(set(pnn))
    assert pnn[-1] == 'FC Test' and AF_A in pnn and AF_I in pnn

    spectra = syn.fluor_spectra()
    raw_fl = raw[:, [raw_pnn.index(c) for c in syn.DET_PNN]]
    expected_fl = raw_fl @ np.linalg.solve(spectra @ spectra.T, spectra).T
    np.testing.assert_array_equal(events[:, [pnn.index(c) for c in syn.FLUOR_LABELS]], _f32(expected_fl))
    np.testing.assert_array_equal(events[:, pnn.index('FSC-A')], raw[:, raw_pnn.index('FSC-A')])
    np.testing.assert_array_equal(events[:, pnn.index('FC Test')], _f32(2.0 * raw[:, raw_pnn.index('Time')]))
    assert np.all(events[:, pnn.index(AF_A)] == 7.0)
    index_map = af_index_lookup(controller.experiment.process['af_profiles'], ['AF profile'])
    assert np.all(events[:, pnn.index(AF_I)] == index_map[1])
    assert _text_value(text, 'FLOWCODEUNMIX') == '0.1.0'


@pytest.mark.numpy_only
def test_exporter_chunking_does_not_change_the_output(tmp_path):
    controller = syn.make_experiment(tmp_path, n_events=2500)
    layout = jue.build_export_layout(controller)
    unmix_fn = _ols_unmix_fn(layout)
    written = {}
    for name, chunk_size in (('whole', jue.UNMIX_CHUNK_SIZE), ('chunked', 700)):
        exporter = jue.JointUnmixExporter([syn.AF_SAMPLE], layout, unmix_fn, extra_channels=['FC Test'],
                                          output_root=tmp_path / name, chunk_size=chunk_size)
        assert not _run_exporter(exporter)
        written[name] = syn.read_fcs(tmp_path / name / 'af_sample (Unmixed).fcs')[1]
    np.testing.assert_array_equal(written['whole'], written['chunked'])


@pytest.mark.numpy_only
def test_exporter_rejects_an_extra_channel_with_an_existing_name(tmp_path):
    controller = syn.make_experiment(tmp_path, n_events=100)
    layout = jue.build_export_layout(controller)
    exporter = jue.JointUnmixExporter([syn.AF_SAMPLE], layout, _ols_unmix_fn(layout),
                                      extra_channels=[syn.FLUOR_LABELS[0]])
    errors = _run_exporter(exporter)
    assert errors and 'same name' in errors[0]


@pytest.mark.numpy_only
def test_exporter_default_destination_is_the_unmixed_folder(tmp_path):
    controller = syn.make_experiment(tmp_path, n_events=300)
    layout = jue.build_export_layout(controller)
    exporter = jue.JointUnmixExporter([syn.AF_SAMPLE], layout, _ols_unmix_fn(layout))
    assert not _run_exporter(exporter)
    assert syn.exported_path(tmp_path, syn.AF_SAMPLE).exists()


@pytest.mark.numpy_only
def test_export_unmixed_sample_writes_extra_keywords(tmp_path):
    pnn = list(syn.UNMIXED_PNN)
    events = np.zeros((10, len(pnn)))
    fn.export_unmixed_sample(
        sample_name='kw', unmixed_folder=tmp_path, export_event_data=events, export_pnn=pnn,
        spillover=np.eye(syn.N_FLUOR), raw_keywords={}, spectral_model=[],
        unmixed_settings=syn.unmixed_settings(), raw_settings=syn.raw_settings(),
        af_spectra=None, unmixing_spectra=None, version='test',
        extra_keywords={'FLOWCODEUNMIX': '0.1.0', 'FLOWCODECOMBOS': 'FC GATA-3=AU1_C_S;'},
    )
    _pnn, _events, text = syn.read_fcs(tmp_path / 'kw (Unmixed).fcs')
    assert _text_value(text, 'FLOWCODEUNMIX') == '0.1.0'
    assert _text_value(text, 'FLOWCODECOMBOS') == 'FC GATA-3=AU1_C_S;'


# ---------------------------------------------------------------------------
# AutoSpectral Optimization exporter
# ---------------------------------------------------------------------------

@pytest.mark.numpy_only
def test_opt_exporter_times_the_run_only_with_active_variants(tmp_path):
    pytest.importorskip('pyqtgraph')
    from autospectral_optimization_tab import AutoSpectralOptExporter

    controller = syn.make_experiment(tmp_path, n_events=100)
    exporter = AutoSpectralOptExporter([syn.AF_SAMPLE], controller, None, {'n_threads': 1})
    assert exporter.probe_fn is None

    variant = syn.fluor_spectra()[:1]
    controller.autospectral_variants = {'F1': {'v_mats': variant, 'delta': variant * 0.0}}
    controller.experiment.process['autospectral_variants_meta'] = {'F1': {'active': False}}
    assert AutoSpectralOptExporter([syn.AF_SAMPLE], controller, None, {}).probe_fn is None

    controller.experiment.process['autospectral_variants_meta'] = {'F1': {'active': True}}
    assert AutoSpectralOptExporter([syn.AF_SAMPLE], controller, None, {}).probe_fn is not None


@requires_kernel
@pytest.mark.numpy_only
def test_opt_exporter_writes_exactly_the_kernel_output(tmp_path):
    """The refactored exporter must write what a direct kernel call returns."""
    pytest.importorskip('pyqtgraph')
    from autospectral_optimization_tab import AutoSpectralOptExporter
    import autospectral_optimization_functions as aof

    controller = syn.make_experiment(tmp_path, n_events=3000)
    exporter = AutoSpectralOptExporter([syn.AF_SAMPLE], controller, None, {'n_threads': 1})
    assert not _run_exporter(exporter)

    pnn, events, _text = syn.read_fcs(syn.exported_path(tmp_path, syn.AF_SAMPLE))
    raw_pnn, raw, _raw_text = syn.read_fcs(tmp_path / syn.AF_SAMPLE)
    expected = aof.unmix_autospectral_optimization(
        raw_fl_events=raw[:, [raw_pnn.index(c) for c in syn.DET_PNN]],
        reference_spectra=syn.fluor_spectra(), fluor_names=syn.FLUOR_LABELS,
        af_spectra=syn.af_spectra(), variants_meta={}, active_labels=set(),
        unmixed_pos_thresholds=np.zeros(syn.N_FLUOR), n_threads=1,
    )
    index_map = af_index_lookup(controller.experiment.process['af_profiles'], ['AF profile'])
    np.testing.assert_array_equal(events[:, [pnn.index(c) for c in syn.FLUOR_LABELS]],
                                  _f32(expected['unmixed']))
    np.testing.assert_array_equal(events[:, pnn.index(AF_A)], _f32(expected['af_scale']))
    np.testing.assert_array_equal(events[:, pnn.index(AF_I)], index_map[expected['af_idx'] - 1])


# ---------------------------------------------------------------------------
# Estimate -> confirm -> export (Qt event loop)
# ---------------------------------------------------------------------------

def _run_to_done(run, timeout_ms=60_000):
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    loop = QEventLoop()
    done = []
    run.done.connect(lambda: (done.append(True), loop.quit()))
    QTimer.singleShot(timeout_ms, loop.quit)
    run.start()
    if not done:
        loop.exec()
    assert done, 'run did not finish'


def _run_with_answer(tmp_path, answer, probe=True):
    controller = _three_af_samples(tmp_path, (2000, 3000, 4000))
    layout = jue.build_export_layout(controller)
    exporter = jue.JointUnmixExporter([syn.AF_SAMPLE, EXTRA_SAMPLE_2, EXTRA_SAMPLE_3],
                                      layout, _ols_unmix_fn(layout))
    probed = []
    probe_fn = (lambda raw_fl, af: probed.append(len(raw_fl))) if probe else None
    run = jue.JointUnmixRun(exporter, probe_fn=probe_fn)
    asked = []
    run._confirm = lambda text: (asked.append(text), answer)[1]
    outcome = []
    run.finished.connect(lambda: outcome.append('finished'))
    run.cancelled.connect(lambda: outcome.append('cancelled'))
    run.error.connect(lambda msg: outcome.append(f'error: {msg}'))
    _run_to_done(run)
    return asked, probed, outcome


@pytest.mark.ui
def test_run_cancelled_at_the_prompt_writes_nothing(tmp_path):
    asked, probed, outcome = _run_with_answer(tmp_path, answer=False)
    assert probed == [jue.PROBE_EVENTS]
    assert len(asked) == 1 and '9,000 events' in asked[0]
    assert outcome == ['cancelled']
    assert not (tmp_path / 'Unmixed').exists()


@pytest.mark.ui
def test_run_confirmed_at_the_prompt_exports_every_sample(tmp_path):
    asked, probed, outcome = _run_with_answer(tmp_path, answer=True)
    assert len(asked) == 1 and probed == [jue.PROBE_EVENTS]
    assert outcome == ['finished']
    for key in (syn.AF_SAMPLE, EXTRA_SAMPLE_2, EXTRA_SAMPLE_3):
        assert syn.exported_path(tmp_path, key).exists()


@pytest.mark.ui
def test_run_without_a_probe_exports_without_asking(tmp_path):
    asked, probed, outcome = _run_with_answer(tmp_path, answer=False, probe=False)
    assert not asked and not probed
    assert outcome == ['finished']

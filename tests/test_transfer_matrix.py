"""
The transfer matrix T (``unmixed events = raw events @ T``) is built by one
function, build_transfer_matrix(), for both the displayed data
(Controller.initialise_transfer_matrix) and the exported FCS files
(UnmixedExporter), so the two cannot drift apart. The spillover
compensation is computed by one function, unmix_method.compensation_matrix().
"""

import numpy as np
import pytest

from honeychrome.controller import Controller
from honeychrome.controller_components import functions as fn
from honeychrome.controller_components import gls_unmix, unmix_method
from honeychrome.controller_components.autospectral_functions import apply_af_transfer, build_af_precomputed
from honeychrome.controller_components.unmixed_exporter import UnmixedExporter

import af_synthetic as syn

pytestmark = pytest.mark.filterwarnings('ignore:No events bus connected')

RAW_PNN = ['Time', 'FSC-A', 'SSC-A', 'D1-A', 'D2-A', 'D3-A', 'D4-A', 'event_id']
FL_IDS = [3, 4, 5, 6]
UNMIXED_PNN = ['Time', 'event_id', 'FSC-A', 'SSC-A', 'F1', 'F2']
UNMIXING = np.array([[0.9, 0.2, -0.1, 0.0],
                     [-0.05, 0.1, 0.7, 0.3]])
SPILLOVER = np.array([[1.0, 0.2],
                      [0.1, 1.0]])
# raw columns in an order of their own, without event_id
WHITELIST = ['SSC-A', 'D2-A', 'FSC-A', 'D1-A', 'Time', 'D4-A', 'D3-A']


def _raw_settings(**overrides):
    raw = {'event_channels_pnn': list(RAW_PNN), 'whitelisted_pnn': None, 'fluorescence_channel_ids': list(FL_IDS),
           'scatter_channel_ids': [1, 2], 'time_channel_id': 0, 'event_id_channel_id': 7}
    raw.update(overrides)
    return raw


def _unmixed_settings():
    return {'event_channels_pnn': list(UNMIXED_PNN), 'fluorescence_channel_ids': [4, 5],
            'scatter_channel_ids': [2, 3], 'n_scatter_channels': 2, 'time_channel_id': 0, 'event_id_channel_id': 1}


def _raw_events(n=50):
    return np.random.default_rng(0).uniform(-100, 5000, (n, len(RAW_PNN)))


def test_unmixes_then_compensates_and_copies_scatter_time_and_event_id():
    raw = _raw_events()
    t = fn.build_transfer_matrix(_raw_settings(), _unmixed_settings(), UNMIXING, FL_IDS, spillover=SPILLOVER)

    unmixed = raw @ t
    assert t.shape == (len(RAW_PNN), len(UNMIXED_PNN))
    # unmix, then compensate the unmixed values as an FCS $SPILLOVER does: x @ inv(S)
    np.testing.assert_allclose(unmixed[:, [4, 5]], raw[:, FL_IDS] @ UNMIXING.T @ np.linalg.inv(SPILLOVER))
    np.testing.assert_array_equal(unmixed[:, [2, 3]], raw[:, [1, 2]])
    np.testing.assert_array_equal(unmixed[:, 0], raw[:, 0])
    np.testing.assert_array_equal(unmixed[:, 1], raw[:, 7])


def test_without_spillover_applies_unmixing_only():
    raw = _raw_events()
    t = fn.build_transfer_matrix(_raw_settings(), _unmixed_settings(), UNMIXING, FL_IDS)

    np.testing.assert_allclose((raw @ t)[:, [4, 5]], raw[:, FL_IDS] @ UNMIXING.T)


def test_raw_without_time_channel_leaves_unmixed_time_zero():
    with_time = fn.build_transfer_matrix(_raw_settings(), _unmixed_settings(), UNMIXING, FL_IDS)
    without_time = fn.build_transfer_matrix(_raw_settings(time_channel_id=None), _unmixed_settings(), UNMIXING,
                                            FL_IDS)

    np.testing.assert_array_equal(without_time[:, 0], 0)
    np.testing.assert_array_equal(without_time[:, 1:], with_time[:, 1:])


def test_columns_follow_the_whitelisted_raw_channel_order():
    raw_full = _raw_events()
    raw_whitelisted = raw_full[:, [RAW_PNN.index(c) for c in WHITELIST]]
    settings = _raw_settings(whitelisted_pnn=list(WHITELIST))

    t_whitelisted = fn.build_transfer_matrix(settings, _unmixed_settings(), UNMIXING, FL_IDS, spillover=SPILLOVER)
    t_full = fn.build_transfer_matrix(settings, _unmixed_settings(), UNMIXING, FL_IDS, spillover=SPILLOVER,
                                      pnn_raw=RAW_PNN)

    assert t_whitelisted.shape == (len(WHITELIST), len(UNMIXED_PNN))
    assert t_full.shape == (len(RAW_PNN), len(UNMIXED_PNN))
    from_whitelisted, from_full = raw_whitelisted @ t_whitelisted, raw_full @ t_full
    np.testing.assert_allclose(np.delete(from_whitelisted, 1, axis=1), np.delete(from_full, 1, axis=1))
    # event_id is not among the whitelisted raw columns
    np.testing.assert_array_equal(from_whitelisted[:, 1], 0)


def test_controller_transfer_matrix_is_built_by_build_transfer_matrix():
    controller = Controller()
    raw_settings = _raw_settings(whitelisted_pnn=list(WHITELIST))
    controller.experiment.settings['raw'].update(raw_settings)
    controller.experiment.settings['unmixed'].update(_unmixed_settings())
    controller.experiment.process['unmixing_matrix'] = UNMIXING.tolist()
    controller.experiment.process['spillover'] = SPILLOVER.tolist()
    controller.filtered_raw_fluorescence_channel_ids = list(FL_IDS)

    controller.initialise_transfer_matrix()

    expected = fn.build_transfer_matrix(raw_settings, _unmixed_settings(), UNMIXING, FL_IDS, spillover=SPILLOVER)
    np.testing.assert_array_equal(controller.transfer_matrix, expected)


# ---------------------------------------------------------------------------
# UnmixedExporter, on the synthetic experiment of af_synthetic.py
# ---------------------------------------------------------------------------

def _export_plain_sample(tmp_path, controller):
    UnmixedExporter('Raw', False, None, controller).run()
    pnn, events, _text = syn.read_fcs(syn.exported_path(tmp_path, syn.PLAIN_SAMPLE))
    raw_pnn, raw, _ = syn.read_fcs(tmp_path / syn.PLAIN_SAMPLE)
    return pnn, events, raw[:, [raw_pnn.index(c) for c in syn.DET_PNN]]


@pytest.mark.numpy_only
def test_export_leaves_spillover_out_of_the_event_values(tmp_path):
    """The exported values are unmixed only; the fine-tuning goes into $SPILLOVER."""
    spillover = np.eye(syn.N_FLUOR) + 0.1 * np.eye(syn.N_FLUOR, k=1)
    controller = syn.make_experiment(tmp_path, spillover=spillover)

    pnn, events, raw_fl = _export_plain_sample(tmp_path, controller)

    unmixing = np.array(controller.experiment.process['unmixing_matrix'])
    fl_cols = [pnn.index(c) for c in syn.FLUOR_LABELS]
    np.testing.assert_allclose(events[:, fl_cols], raw_fl @ unmixing.T, rtol=1e-5, atol=1e-2)


@pytest.mark.numpy_only
def test_export_of_raw_data_without_time_channel_writes_zero_time(tmp_path):
    """As on screen, the unmixed Time column stays zero when the raw data has no Time channel."""
    controller = syn.make_experiment(tmp_path)
    controller.experiment.settings['raw']['time_channel_id'] = None

    pnn, events, _raw_fl = _export_plain_sample(tmp_path, controller)

    np.testing.assert_array_equal(events[:, pnn.index('Time')], 0)


# ---------------------------------------------------------------------------
# Spillover compensation, on af_synthetic's channels
# ---------------------------------------------------------------------------

SYN_SPILLOVER = np.eye(syn.N_FLUOR) + 0.1 * np.eye(syn.N_FLUOR, k=1) + 0.05 * np.eye(syn.N_FLUOR, k=-1)
SYN_FL_IDS = list(range(3, 3 + syn.N_DET))


def _syn_unmixing(seed):
    fl = syn.fluor_spectra() + 0.02 * np.random.default_rng(seed).random((syn.N_FLUOR, syn.N_DET))
    return np.linalg.solve(fl @ fl.T, fl)


def _syn_transfer_matrix(unmixing, spillover=SYN_SPILLOVER):
    return fn.build_transfer_matrix(syn.raw_settings(), syn.unmixed_settings(), unmixing, SYN_FL_IDS,
                                    spillover=spillover)


@pytest.fixture
def compensated_with(monkeypatch):
    """Spillover matrices passed to unmix_method.compensation_matrix."""
    calls = []
    original = unmix_method.compensation_matrix

    def spy(spillover):
        calls.append(spillover)
        return original(spillover)

    monkeypatch.setattr(unmix_method, 'compensation_matrix', spy)
    return calls


def _compensated_once_with_syn_spillover(calls):
    return len(calls) == 1 and np.array_equal(calls[0], SYN_SPILLOVER)


def test_compensation_matrix_compensates_like_an_fcs_spillover():
    unmixed = np.random.default_rng(0).normal(size=(50, syn.N_FLUOR))

    compensated = (unmix_method.compensation_matrix(SYN_SPILLOVER) @ unmixed.T).T

    np.testing.assert_allclose(compensated, unmixed @ np.linalg.inv(SYN_SPILLOVER))


def test_build_transfer_matrix_compensates_through_compensation_matrix(compensated_with):
    _syn_transfer_matrix(_syn_unmixing(1))

    assert _compensated_once_with_syn_spillover(compensated_with)


def test_corrected_spectra_matrix_compensates_through_compensation_matrix(compensated_with):
    unmixing = _syn_unmixing(1)
    unmix_method.transfer_with_spectra(_syn_transfer_matrix(unmixing, spillover=None), SYN_FL_IDS,
                                       syn.unmixed_settings()['fluorescence_channel_ids'], unmixing, SYN_SPILLOVER)

    assert _compensated_once_with_syn_spillover(compensated_with)


def test_af_unmixing_compensates_through_compensation_matrix(compensated_with):
    unmixing = _syn_unmixing(1)
    transfer = _syn_transfer_matrix(unmixing, spillover=None)
    settings = {'raw': syn.raw_settings(), 'unmixed': syn.unmixed_settings()}
    apply_af_transfer(syn.raw_events(200, seed=2), transfer, build_af_precomputed(syn.fluor_spectra(), syn.af_spectra()),
                      syn.af_spectra(), settings, filtered_fl_ids_raw=SYN_FL_IDS, spillover=SYN_SPILLOVER)

    assert _compensated_once_with_syn_spillover(compensated_with)


def test_noise_model_result_compensates_through_compensation_matrix(compensated_with):
    transfer = _syn_transfer_matrix(_syn_unmixing(1), spillover=None)
    settings = {'raw': syn.raw_settings(), 'unmixed': syn.unmixed_settings()}
    result = {'unmixed': np.ones((200, syn.N_FLUOR)), 'af_idx': np.full(200, -1), 'af_scale': np.zeros(200)}
    gls_unmix.assemble_unmixed(syn.raw_events(200, seed=2), transfer, settings, result, SYN_SPILLOVER)

    assert _compensated_once_with_syn_spillover(compensated_with)


def test_matrix_for_corrected_spectra_equals_a_newly_built_one():
    """unmix_method.transfer_with_spectra (corrected spectra sets) and build_transfer_matrix agree exactly."""
    corrected = _syn_unmixing(2)

    rebuilt = unmix_method.transfer_with_spectra(_syn_transfer_matrix(_syn_unmixing(1)), SYN_FL_IDS,
                                                 syn.unmixed_settings()['fluorescence_channel_ids'], corrected,
                                                 SYN_SPILLOVER)

    np.testing.assert_array_equal(rebuilt, _syn_transfer_matrix(corrected))

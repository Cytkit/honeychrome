"""
test_af_and_control_gating.py
-----------------------------
Per-cell AF and per-sample gates outside the displayed sample:

* the Statistics tab unmixes each sample with its own AutoSpectral AF
  profile assignment, as the main window does when the sample is displayed;
* Cluster ID unmixes an unstained sample with its own AF assignment when it
  has one, and never with the displayed sample's AF matrices;
* the AF comparison plots gate 2-D gates whose axes have different bin
  counts correctly;
* spectral control profiles are computed from the control sample's own raw
  gates (per-sample overrides included), report gates with fewer than 500
  events, and Recalculate notices a moved gate.

Usage:
    pytest tests/test_af_and_control_gating.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from flowkit import Dimension, gates

from honeychrome.controller import Controller
from honeychrome.controller_components.autospectral_functions import build_af_precomputed
from honeychrome.controller_components.functions import apply_gates_in_place
from honeychrome.controller_components.per_sample_gating_functions import SampleGating
from honeychrome.controller_components.sample_loader import snapshot_unmix_state, unmix_events
from honeychrome.controller_components.transform import Transform
from honeychrome.settings import INTERNAL_NEGATIVE_SENTINEL
from test_unmix_method import fitted  # noqa: E402,F401  (fixture)

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

RAW_PNN = ['D0', 'D1', 'D2', 'D3', 'SC', 'Time']
UNMIXED_PNN = ['F1', 'F2', 'SC', 'Time']
FL_IDS = [0, 1, 2, 3]


def _rect(name, channel, lo, hi):
    return gates.RectangleGate(name, dimensions=[Dimension(channel, range_min=lo, range_max=hi)])


def _linear(top, bins=None):
    tr = Transform(scale_t=top, linear_a=0)
    if bins is not None:
        tr.scale_bins = bins
    tr.set_transform(id=0, limits=[0, 1])
    return tr


def _time(top, bins=None):
    tr = Transform()
    if bins is not None:
        tr.scale_bins = bins
    tr.set_transform(id='default', limits=[0, top * 1.05])
    return tr


# ---------------------------------------------------------------------------
# Statistics tab: per-sample AF
# ---------------------------------------------------------------------------

FLUOR = np.array([[1.0, 0.6, 0.2, 0.05], [0.1, 0.3, 1.0, 0.5]])
AF = np.array([[0.2, 1.0, 0.7, 0.4], [0.5, 1.0, 0.3, 0.1]])


def _raw_events(seed):
    """Cells with fluorophore and AF signal; SC spans 5-95, Time 0-100."""
    rng = np.random.default_rng(seed)
    n = 2000
    abundance = np.column_stack([rng.lognormal(6, 0.5, n), rng.lognormal(5, 0.5, n)])
    af = rng.lognormal(6.5, 0.4, n)[:, None] * AF[rng.integers(0, len(AF), n)]
    fl = abundance @ FLUOR + af + rng.normal(0, 5, (n, len(FL_IDS)))
    sc = rng.uniform(5, 95, n)
    t = np.linspace(0.5, 99.5, n)
    return np.column_stack([fl, sc, t])


RAW = {'Raw/A.fcs': _raw_events(1), 'Raw/B.fcs': _raw_events(2)}


def _stats_controller(shown):
    controller = Controller()
    exp = controller.experiment
    exp.settings['raw'].update({'event_channels_pnn': list(RAW_PNN), 'whitelisted_pnn': None,
                                'fluorescence_channel_ids': list(FL_IDS)})
    exp.settings['unmixed'].update({'event_channels_pnn': list(UNMIXED_PNN), 'fluorescence_channel_ids': [0, 1]})
    controller.filtered_raw_fluorescence_channel_ids = list(FL_IDS)
    tm = np.zeros((len(RAW_PNN), len(UNMIXED_PNN)))
    tm[:4, :2] = np.linalg.pinv(FLUOR)
    tm[4, 2] = 1.0
    tm[5, 3] = 1.0
    controller.transfer_matrix = tm
    exp.process['unmixing_matrix'] = np.linalg.pinv(FLUOR).T.tolist()
    exp.process['spillover'] = np.eye(2).tolist()
    exp.process['af_profiles'] = {'P': {'spectra': AF.tolist()}}
    controller.af_precomputed_cache = {'P': build_af_precomputed(FLUOR, AF)}
    exp.samples['all_samples'] = {key: Path(key).stem for key in RAW}
    exp.samples['sample_af_profiles'] = {'Raw/B.fcs': ['P']}
    controller.experiment_dir = Path('experiment')

    controller.unmixed_transformations = {'F1': _linear(1e5), 'F2': _linear(1e5), 'SC': _linear(100),
                                          'Time': _time(100)}
    controller.data_for_cytometry_plots_unmixed['pnn'] = list(UNMIXED_PNN)
    controller.unmixed_gating.add_gate(_rect('Cells', 'SC', 20, 80), gate_path=('root',))
    controller.current_sample_path = shown
    controller.calculate_lookup_tables(mode='unmixed')
    controller.data_for_cytometry_plots_unmixed.update({
        'gating': controller.unmixed_gating, 'transformations': controller.unmixed_transformations,
        'lookup_tables': controller.unmixed_lookup_tables, 'pnn_labels': {ch: ch for ch in UNMIXED_PNN}})
    exp.statistics = [{'sample_set': 'Raw', 'plot_type': 'Bar Chart', 'gate': 'Cells',
                       'statistic': 'Mean Intensity F1', 'channel': 'F1', 'depth': 1, 'data': None}]
    return controller


class _FakeSample:
    def __init__(self, events, pnn=None):
        self.events = events
        self.event_count = len(events)
        self.pnn_labels = list(pnn or RAW_PNN)

    def get_events(self, *_args, **_kwargs):
        return self.events


def _run_statistics(controller, monkeypatch):
    from honeychrome.controller_components import statistical_controller as sc

    monkeypatch.setattr(sc, 'sample_from_fcs', lambda path: _FakeSample(RAW[f'Raw/{Path(path).name}']))
    calculator = sc.StatisticsCalculator(None, controller)
    sc.StatisticsCalculator.run.__wrapped__(calculator)
    c = controller.experiment.statistics[0]
    return dict(zip(c['data']['Sample'], c['data'][c['statistic']]))


def _expected_mean_f1(controller, key, with_af):
    snap = snapshot_unmix_state(controller, include_qc=False)
    af = snap.af_for(key) if with_af else None
    unmixed = unmix_events(RAW[key], snap, af)['unmixed']
    cells = SampleGating.from_controller(controller).memberships(key, unmixed, UNMIXED_PNN)['Cells']
    return float(unmixed[cells, 0].mean())


@pytest.mark.parametrize('shown', ['Raw/A.fcs', 'Raw/B.fcs'])
def test_statistics_unmix_each_sample_with_its_own_af(shown, monkeypatch):
    controller = _stats_controller(shown)
    result = _run_statistics(controller, monkeypatch)
    assert result['A'] == pytest.approx(_expected_mean_f1(controller, 'Raw/A.fcs', with_af=False))
    assert result['B'] == pytest.approx(_expected_mean_f1(controller, 'Raw/B.fcs', with_af=True))
    # the AF correction matters on this channel
    assert result['B'] != pytest.approx(_expected_mean_f1(controller, 'Raw/B.fcs', with_af=False), rel=1e-3)


# ---------------------------------------------------------------------------
# Cluster ID: unstained samples
# ---------------------------------------------------------------------------

def _cluster_id_controller(unstained_assignment):
    af_b = AF[::-1].copy()
    samples = {
        'all_samples': {'Raw/S1.fcs': 'S1', 'Raw/S2.fcs': 'S2', 'Raw/Unstained.fcs': 'Unstained'},
        'sample_af_profiles': {'Raw/S1.fcs': ['P'], 'Raw/S2.fcs': ['P']},
        'single_stain_controls': [], 'unstained_samples': ['Raw/Unstained.fcs'],
    }
    if unstained_assignment:
        samples['sample_af_profiles']['Raw/Unstained.fcs'] = list(unstained_assignment)
    experiment = SimpleNamespace(
        samples=samples,
        process={'af_profiles': {'P': {'spectra': AF.tolist()}, 'Q': {'spectra': af_b.tolist()}}},
        settings={'raw': {'raw_samples_subdirectory': 'Raw', 'event_channels_pnn': ['D1', 'D2', 'D3'],
                          'fluorescence_channel_ids': [0, 1, 2]}},
    )
    return SimpleNamespace(
        experiment=experiment, transfer_matrix=np.eye(3), filtered_raw_fluorescence_channel_ids=[0, 1, 2],
        af_precomputed_cache={'P': build_af_precomputed(FLUOR, AF), 'Q': build_af_precomputed(FLUOR, af_b)},
    ), af_b


def test_cluster_id_unstained_sample_uses_its_own_af_assignment():
    import drc_cluster_id

    cl_run = {'training_sample_ids': ['S1.fcs', 'S2.fcs']}
    controller, af_b = _cluster_id_controller(['Q'])
    states = drc_cluster_id.resolve_unstained_af_states(controller, cl_run)
    snap, af = states['Raw/Unstained.fcs']
    np.testing.assert_array_equal(af.spectra, af_b)
    assert af.profile_names == ('Q',) and snap.method is None

    # without one, the training samples' most common assignment stands in
    controller, _ = _cluster_id_controller(None)
    states = drc_cluster_id.resolve_unstained_af_states(controller, cl_run)
    np.testing.assert_array_equal(states['Raw/Unstained.fcs'][1].spectra, AF)


def test_af_aware_unmixing_never_falls_back_to_the_displayed_sample():
    import drc_pipeline

    controller = SimpleNamespace(transfer_matrix=np.eye(3), af_precomputed={'P': None}, af_spectra=AF)
    with pytest.raises(ValueError, match='sample_key or af_state'):
        drc_pipeline.apply_unmixing_af_aware(controller, np.ones((4, 3)))


# ---------------------------------------------------------------------------
# AF comparison plots: 2-D gate with unequal bin counts
# ---------------------------------------------------------------------------

def test_comparison_gate_mask_with_unequal_bin_counts():
    from honeychrome.view_components.autospectral_tab import AfComparisonPlotWidget

    controller = Controller()
    controller.experiment.settings['unmixed']['event_channels_pnn'] = ['X', 'Time']
    controller.unmixed_transformations = {'X': _linear(100, bins=50), 'Time': _time(100, bins=400)}
    gate = gates.RectangleGate('Stable X', dimensions=[Dimension('X', range_min=20, range_max=70),
                                                        Dimension('Time', range_min=10, range_max=60)])
    controller.unmixed_gating.add_gate(gate, gate_path=('root',))
    controller.current_sample_path = 'Raw/A.fcs'
    controller.calculate_lookup_tables(mode='unmixed')

    rng = np.random.default_rng(0)
    events = np.column_stack([rng.uniform(0, 100, 5000), rng.uniform(0, 100, 5000)])
    widget = SimpleNamespace(_source_gate='Stable X', controller=controller)
    mask = AfComparisonPlotWidget._compute_gate_mask(widget, events)

    main_window = {'event_data': events, 'pnn': ['X', 'Time'],
                   'transformations': controller.unmixed_transformations,
                   'lookup_tables': controller.unmixed_lookup_tables, 'gating': controller.unmixed_gating,
                   'gate_membership': {'root': np.ones(len(events), dtype=bool)}}
    apply_gates_in_place(main_window, gates_to_calculate=['Stable X'])
    np.testing.assert_array_equal(mask, main_window['gate_membership']['Stable X'])
    inside = (events[:, 0] > 20) & (events[:, 0] < 70) & (events[:, 1] > 10) & (events[:, 1] < 60)
    assert np.mean(mask == inside) > 0.98


# ---------------------------------------------------------------------------
# Spectral controls: per-sample raw gates, low-event warning, Recalculate
# ---------------------------------------------------------------------------

CTRL_PNN = ['D0', 'D1', 'D2', 'FSC-A', 'Time']
CTRL_KEY = 'Raw/Controls/F1 Cells.fcs'


def _control_events():
    """D0 ramps 5-995; D1 rises with D0 squared, so the normalised profile
    depends on which D0 range is gated; FSC-A 50-950."""
    d0 = np.repeat(np.arange(5.0, 1000.0, 10.0), 12)
    n = len(d0)
    fsc = np.tile(np.arange(50.0, 1000.0, 100.0), n // 10 + 1)[:n]
    return np.column_stack([d0, 0.002 * d0 ** 2 + 3.0, 0.1 * d0 + 7.0, fsc, np.linspace(0.5, 50.0, n)])


CTRL_EVENTS = _control_events()


def _control_controller(monkeypatch):
    from honeychrome.controller_components import spectral_controller as scn

    controller = Controller()
    exp = controller.experiment
    exp.settings['raw'].update({'event_channels_pnn': list(CTRL_PNN), 'whitelisted_pnn': None,
                                'fluorescence_channel_ids': [0, 1, 2]})
    exp.process.update({'fluorescence_channel_filter': 'all_fluorescence', 'negative_type': 'internal',
                        'spectral_model': [], 'profiles': {}})
    exp.samples['all_samples'] = {CTRL_KEY: 'F1 Cells'}
    exp.samples['all_sample_nevents'] = {CTRL_KEY: len(CTRL_EVENTS)}
    controller.experiment_dir = Path('experiment')
    controller.raw_transformations = {ch: _linear(1000) for ch in CTRL_PNN[:4]}
    controller.raw_transformations['Time'] = _time(50)
    controller.data_for_cytometry_plots_raw['pnn'] = list(CTRL_PNN)
    gating = controller.raw_gating
    gating.add_gate(_rect('Cells', 'FSC-A', 100, 900), gate_path=('root',))
    gating.add_gate(_rect('Pos F1', 'D0', 500, 1000), gate_path=('root', 'Cells'))
    gating.add_gate(_rect('Neg F1', 'D0', 0, 200), gate_path=('root', 'Cells'))
    controller.current_sample_path = 'Raw/Other.fcs'
    controller.calculate_lookup_tables(mode='raw')
    monkeypatch.setattr(scn, 'sample_from_fcs', lambda path: _FakeSample(CTRL_EVENTS, CTRL_PNN))
    control = {'label': 'F1', 'control_type': 'Single Stained Spectral Control', 'particle_type': 'Cells',
               'gate_channel': 'D0', 'gate_channel_locked': True, 'sample_name': 'F1 Cells',
               'gate_label': 'Pos F1', 'neg_gate_label': 'Neg F1',
               'universal_negative_name': INTERNAL_NEGATIVE_SENTINEL}
    exp.process['spectral_model'].append(control)
    return controller, control, scn


def _customise(controller, gate_name, lo, hi):
    shown = controller.current_sample_path
    controller.current_sample_path = CTRL_KEY
    controller.apply_custom_sample_gates('raw')
    controller.customise_gate('raw', gate_name)
    custom = controller.raw_gating.get_gate(gate_name, sample_id=CTRL_KEY)
    custom.dimensions = [Dimension(custom.dimensions[0].id, range_min=lo, range_max=hi)]
    controller._sync_custom_gate_from_strategy('raw', gate_name)
    controller.current_sample_path = shown
    controller.calculate_lookup_tables(mode='raw')


def _profile(lo_pos, hi_pos, lo_neg=0, hi_neg=200):
    d0, fsc = CTRL_EVENTS[:, 0], CTRL_EVENTS[:, 3]
    cells = (fsc > 100) & (fsc < 900)
    pos = CTRL_EVENTS[cells & (d0 > lo_pos) & (d0 < hi_pos), :3].mean(axis=0)
    neg = CTRL_EVENTS[cells & (d0 > lo_neg) & (d0 < hi_neg), :3].mean(axis=0)
    p = pos - neg
    return p / p.max()


def test_control_profile_uses_the_control_samples_own_gate(monkeypatch):
    controller, control, scn = _control_controller(monkeypatch)
    updater = scn.ProfileUpdater(controller, None)
    assert updater.generate(control, None)
    np.testing.assert_allclose(updater.profiles['F1'], _profile(500, 1000), rtol=1e-6)

    _customise(controller, 'Pos F1', 800, 1000)
    assert updater.generate(control, None)
    np.testing.assert_allclose(updater.profiles['F1'], _profile(800, 1000), rtol=1e-6)
    assert not np.allclose(_profile(800, 1000), _profile(500, 1000), rtol=1e-3)


def test_low_event_control_gates_are_collected(monkeypatch):
    controller, control, scn = _control_controller(monkeypatch)
    updater = scn.ProfileUpdater(controller, None)
    updater.low_events = scn.LowEventLog(controller, None, collect=True)
    _customise(controller, 'Pos F1', 960, 1000)
    assert updater.generate(control, None)
    n_pos = int(np.sum((CTRL_EVENTS[:, 3] > 100) & (CTRL_EVENTS[:, 3] < 900) & (CTRL_EVENTS[:, 0] > 960)))
    assert n_pos < scn.MIN_CONTROL_GATE_EVENTS
    names = {(name, gate) for name, gate, _n in updater.low_events.entries}
    assert ('F1', 'Pos F1') in names
    text = updater.low_events.text()
    assert f'{n_pos:,} event(s) in "Pos F1"' in text

    # outside a batch, the warning goes to the active collector at once
    controller.warning_collector = []
    updater.low_events = scn.LowEventLog(controller, None, collect=False)
    assert updater.generate(control, None)
    assert len(controller.warning_collector) == 1 and '"Pos F1"' in controller.warning_collector[0]


def test_gating_signature_follows_the_control_samples_gates(monkeypatch):
    controller, control, scn = _control_controller(monkeypatch)
    updater = scn.ProfileUpdater(controller, None)
    assert updater.generate(control, None)
    assert updater.profile_gating['F1'] == updater.gating_signature(control)

    # a gate the profile does not use leaves it current
    controller.raw_gating.add_gate(_rect('Other', 'D1', 0, 10), gate_path=('root',))
    assert updater.profile_gating['F1'] == updater.gating_signature(control)

    # moving its own gate, here as an override on the control sample, does not
    _customise(controller, 'Neg F1', 0, 100)
    assert updater.profile_gating['F1'] != updater.gating_signature(control)


def test_spectral_functions_gate_with_a_sample_gating(monkeypatch):
    from honeychrome.controller_components.spectral_functions import get_raw_events

    controller, _control, scn = _control_controller(monkeypatch)
    _customise(controller, 'Cells', 500, 900)
    sample = _FakeSample(CTRL_EVENTS, CTRL_PNN)
    own = get_raw_events(sample, [0, 1, 2], gate_label='Cells', gating_strategy=scn.control_gating(controller),
                         sample_key=CTRL_KEY)
    other = get_raw_events(sample, [0, 1, 2], gate_label='Cells', gating_strategy=scn.control_gating(controller),
                           sample_key='Raw/Other.fcs')
    fsc = CTRL_EVENTS[:, 3]
    assert len(own) == int(np.sum((fsc > 500) & (fsc < 900)))
    assert len(other) == int(np.sum((fsc > 100) & (fsc < 900)))


def test_cluster_id_stand_in_is_unmixed_by_the_experiments_method(fitted, tmp_path):
    """The stand-in library of an unstained sample goes through the
    experiment's method (here M2pg), as the training samples do."""
    import drc_pipeline
    import test_unmix_method as T
    from honeychrome.controller_components import sample_loader as sl

    ctrl, _env = T.make_controller(fitted, tmp_path, {'noise_model': True})
    snap = sl.snapshot_unmix_state(ctrl, include_qc=False)
    af = snap.af_for(T.A)
    raw = T._raw(fitted, T.B)
    out = drc_pipeline.apply_unmixing_af_aware(ctrl, raw, af_state=(snap, af), sample_key=T.B)
    np.testing.assert_array_equal(out, sl.unmix_events(raw, snap, af, sample_key=T.B)['unmixed'])
    r1 = drc_pipeline.apply_unmixing_af_aware(
        ctrl, raw, af_state=(ctrl.transfer_matrix, af.precomputed, af.spectra), sample_key=T.B)
    assert not np.allclose(out, r1)

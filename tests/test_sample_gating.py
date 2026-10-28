"""
test_sample_gating.py
---------------------
``SampleGating`` gates every sample as the main window gates it when that
sample is shown: its own per-sample gate overrides, its own range on
per-sample (Time) axes, and the template everywhere else. Also checks its
users: the Statistics tab (``StatisticsCalculator``) gives the same values
whichever sample is displayed, and the Honeycluster loader
(``drc_pipeline.load_unmixed_gated``) gates per sample with a cache that
follows override edits.

Usage:
    pytest tests/test_sample_gating.py
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from flowkit import Dimension, gates

from honeychrome.controller import Controller
from honeychrome.controller_components.functions import apply_gates_in_place
from honeychrome.controller_components.per_sample_gating_functions import SampleGating
from honeychrome.controller_components.transform import Transform

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

PNN = ['X', 'Time']
X_VALUES = np.arange(5.0, 100.0, 10.0)          # 5, 15, ..., 95: never on a gate edge
GATE_NAMES = ['Cells', 'Pos', 'Stable']


def _events(time_max):
    times = np.arange(2.5, time_max, 5.0)        # 2.5, 7.5, ...: never on a gate edge
    xx, tt = np.meshgrid(X_VALUES, times, indexing='ij')
    return np.column_stack([xx.ravel(), tt.ravel()])


SAMPLES = {'Raw/A.fcs': _events(20.0), 'Raw/B.fcs': _events(100.0), 'Raw/C.fcs': _events(100.0)}


def _rect(name, channel, lo, hi):
    return gates.RectangleGate(name, dimensions=[Dimension(channel, range_min=lo, range_max=hi)])


def _time_transform(time_max):
    tr = Transform()
    tr.set_transform(id='default', limits=[0, time_max * 1.05])
    return tr


def _controller(shown='Raw/A.fcs'):
    """Controller with X (linear, 0 to 100), Time (per-sample) and three gates:
    Cells (X 10-90), Pos (X 50-90, inside Cells) and Stable (Time 5-80)."""
    controller = Controller()
    x = Transform(scale_t=100, linear_a=0)
    x.set_transform(id=0, limits=[0, 1])
    controller.unmixed_transformations = {'X': x,
                                          'Time': _time_transform(SAMPLES[shown][:, 1].max())}
    controller.data_for_cytometry_plots_unmixed['pnn'] = list(PNN)
    gating = controller.unmixed_gating
    gating.add_gate(_rect('Cells', 'X', 10, 90), gate_path=('root',))
    gating.add_gate(_rect('Pos', 'X', 50, 90), gate_path=('root', 'Cells'))
    gating.add_gate(_rect('Stable', 'Time', 5, 80), gate_path=('root',))
    controller.current_sample_path = shown
    controller.calculate_lookup_tables(mode='unmixed')
    return controller


def _override(controller, sample_key, gate_name, lo, hi, sync=True):
    """Customise *gate_name* for *sample_key* (shown) and move its range."""
    controller.current_sample_path = sample_key
    controller.apply_custom_sample_gates('unmixed')
    controller.customise_gate('unmixed', gate_name)
    custom = controller.unmixed_gating.get_gate(gate_name, sample_id=sample_key)
    custom.dimensions = [Dimension('X', range_min=lo, range_max=hi)]
    if sync:
        controller._sync_custom_gate_from_strategy('unmixed', gate_name)
    controller.calculate_lookup_tables(mode='unmixed')


def _shown_memberships(controller, sample_key):
    """Show *sample_key* in the controller as the main window does and gate
    its events with the controller's own tables."""
    events = SAMPLES[sample_key]
    controller.current_sample_path = sample_key
    controller.unmixed_transformations['Time'].set_transform(limits=[0, float(events[:, 1].max()) * 1.05])
    controller.apply_custom_sample_gates('unmixed')
    controller.calculate_lookup_tables(mode='unmixed')
    data = {'event_data': events, 'pnn': list(PNN), 'transformations': controller.unmixed_transformations,
            'lookup_tables': controller.unmixed_lookup_tables, 'gating': controller.unmixed_gating,
            'gate_membership': {'root': np.ones(len(events), dtype=bool)}}
    apply_gates_in_place(data, gates_to_calculate=GATE_NAMES)
    return data['gate_membership']


def _expected(events, cells=(10, 90)):
    x, t = events[:, 0], events[:, 1]
    in_cells = (x > cells[0]) & (x < cells[1])
    return {'Cells': in_cells, 'Pos': in_cells & (x > 50) & (x < 90), 'Stable': (t > 5) & (t < 80)}


def test_each_sample_matches_the_main_window_gating_of_that_sample():
    controller = _controller()
    _override(controller, 'Raw/B.fcs', 'Cells', 30, 90)
    controller.current_sample_path = 'Raw/A.fcs'
    controller.unmixed_transformations['Time'].set_transform(limits=[0, 17.5 * 1.05])
    controller.apply_custom_sample_gates('unmixed')
    controller.calculate_lookup_tables(mode='unmixed')

    sample_gating = SampleGating.from_controller(controller)
    ours = {key: sample_gating.memberships(key, events, PNN) for key, events in SAMPLES.items()}

    for key, events in SAMPLES.items():
        shown = _shown_memberships(controller, key)
        expected = _expected(events, cells=(30, 90) if key == 'Raw/B.fcs' else (10, 90))
        for name in GATE_NAMES:
            np.testing.assert_array_equal(ours[key][name], shown[name], err_msg=f'{key} {name}')
            np.testing.assert_array_equal(ours[key][name], expected[name], err_msg=f'{key} {name}')

    mask = SampleGating.from_controller(controller).population_mask('Raw/C.fcs', SAMPLES['Raw/C.fcs'], PNN,
                                                                    ['Pos', 'Stable'])
    np.testing.assert_array_equal(mask, ours['Raw/C.fcs']['Pos'] | ours['Raw/C.fcs']['Stable'])


def test_shown_sample_tables_are_not_reused_for_other_samples():
    """The shown sample's tables gate it alone: another sample's Time events
    beyond the shown sample's range, and its template Cells gate, differ."""
    controller = _controller()
    _override(controller, 'Raw/A.fcs', 'Cells', 30, 90)
    events = SAMPLES['Raw/C.fcs']
    shown_tables = {'event_data': events, 'pnn': list(PNN),
                    'transformations': controller.unmixed_transformations,
                    'lookup_tables': controller.unmixed_lookup_tables, 'gating': controller.unmixed_gating,
                    'gate_membership': {'root': np.ones(len(events), dtype=bool)}}
    apply_gates_in_place(shown_tables, gates_to_calculate=GATE_NAMES)

    ours = SampleGating.from_controller(controller).memberships('Raw/C.fcs', events, PNN)
    expected = _expected(events)
    for name in GATE_NAMES:
        np.testing.assert_array_equal(ours[name], expected[name], err_msg=name)
    assert not np.array_equal(shown_tables['gate_membership']['Cells'], expected['Cells'])
    assert not np.array_equal(shown_tables['gate_membership']['Stable'], expected['Stable'])


def test_live_edit_of_the_shown_sample_override_is_used():
    """ROI edits land in the strategy before ``custom_sample_gates``; the
    shown sample's live custom gate is the one used, and only for it."""
    controller = _controller()
    _override(controller, 'Raw/B.fcs', 'Cells', 30, 90, sync=False)

    sample_gating = SampleGating.from_controller(controller)
    assert set(sample_gating._seed) == {'Pos'}
    b = sample_gating.memberships('Raw/B.fcs', SAMPLES['Raw/B.fcs'], PNN)
    c = sample_gating.memberships('Raw/C.fcs', SAMPLES['Raw/C.fcs'], PNN)
    np.testing.assert_array_equal(b['Cells'], _expected(SAMPLES['Raw/B.fcs'], cells=(30, 90))['Cells'])
    np.testing.assert_array_equal(c['Cells'], _expected(SAMPLES['Raw/C.fcs'])['Cells'])


def test_snapshot_is_independent_of_later_controller_edits():
    controller = _controller()
    sample_gating = SampleGating.from_controller(controller)
    _override(controller, 'Raw/B.fcs', 'Cells', 30, 90)
    b = sample_gating.memberships('Raw/B.fcs', SAMPLES['Raw/B.fcs'], PNN)
    np.testing.assert_array_equal(b['Cells'], _expected(SAMPLES['Raw/B.fcs'])['Cells'])


def _signatures(controller):
    gating = SampleGating.from_controller(controller, copy=False)
    return {key: gating.signature([key]) for key in SAMPLES}, gating.signature()


def test_signature_changes_only_for_the_edited_sample():
    controller = _controller()
    before, before_all = _signatures(controller)
    _override(controller, 'Raw/B.fcs', 'Cells', 30, 90)
    after, after_all = _signatures(controller)
    assert after['Raw/B.fcs'] != before['Raw/B.fcs'] and after_all != before_all
    assert after['Raw/A.fcs'] == before['Raw/A.fcs'] and after['Raw/C.fcs'] == before['Raw/C.fcs']

    custom = controller.unmixed_gating.get_gate('Cells', sample_id='Raw/B.fcs')
    custom.dimensions = [Dimension('X', range_min=40, range_max=90)]
    assert _signatures(controller)[0]['Raw/B.fcs'] != after['Raw/B.fcs']


def test_missing_gate_raises_and_no_gates_keeps_everything():
    controller = _controller()
    sample_gating = SampleGating.from_controller(controller)
    events = SAMPLES['Raw/C.fcs']
    assert sample_gating.population_mask('Raw/C.fcs', events, PNN, []).all()
    with pytest.raises(ValueError, match='Nope'):
        sample_gating.population_mask('Raw/C.fcs', events, PNN, ['Nope'])
    with pytest.raises(ValueError, match='Stable'):
        sample_gating.population_mask('Raw/C.fcs', events[:, :1], ['X'], ['Stable'])


def _statistics(controller, monkeypatch):
    """Run the Statistics tab's calculator over the three samples; returns
    {(gate, statistic): {sample name: value}}."""
    from honeychrome.controller_components import statistical_controller as sc

    monkeypatch.setattr(sc, 'sample_from_fcs', lambda path: _FakeSample(SAMPLES[f'Raw/{Path(path).name}']))
    controller.experiment.settings['raw'].update({'event_channels_pnn': list(PNN), 'whitelisted_pnn': None})
    controller.filtered_raw_fluorescence_channel_ids = [0]
    controller.transfer_matrix = np.eye(len(PNN))
    controller.experiment_dir = Path('experiment')
    controller.experiment.samples['all_samples'] = {key: Path(key).stem for key in SAMPLES}
    controller.experiment.process['unmixing_matrix'] = [[1.0]]
    controller.data_for_cytometry_plots_unmixed.update({
        'gating': controller.unmixed_gating, 'transformations': controller.unmixed_transformations,
        'lookup_tables': controller.unmixed_lookup_tables, 'pnn_labels': {ch: ch for ch in PNN}})
    wanted = [('Cells', 'Number of Events'), ('Pos', '% Parent'), ('Stable', 'Number of Events')]
    controller.experiment.statistics = [{'sample_set': 'Raw', 'plot_type': 'Bar Chart', 'gate': gate,
                                         'statistic': statistic, 'depth': 1, 'data': None}
                                        for gate, statistic in wanted]
    calculator = sc.StatisticsCalculator(None, controller)
    sc.StatisticsCalculator.run.__wrapped__(calculator)
    return {(c['gate'], c['statistic']): dict(zip(c['data']['Sample'], c['data'][c['statistic']]))
            for c in controller.experiment.statistics}


def _show(controller, sample_key):
    controller.current_sample_path = sample_key
    controller.unmixed_transformations['Time'].set_transform(
        limits=[0, float(SAMPLES[sample_key][:, 1].max()) * 1.05])
    controller.apply_custom_sample_gates('unmixed')
    controller.calculate_lookup_tables(mode='unmixed')


def test_statistics_do_not_depend_on_the_displayed_sample(monkeypatch):
    controller = _controller()
    _override(controller, 'Raw/B.fcs', 'Cells', 30, 90)
    _override(controller, 'Raw/C.fcs', 'Cells', 10, 50)

    expected = {}
    for key, events in SAMPLES.items():
        cells = {'Raw/B.fcs': (30, 90), 'Raw/C.fcs': (10, 50)}.get(key, (10, 90))
        m = _expected(events, cells=cells)
        name = Path(key).stem
        expected.setdefault(('Cells', 'Number of Events'), {})[name] = int(m['Cells'].sum())
        expected.setdefault(('Pos', '% Parent'), {})[name] = 100 * m['Pos'].sum() / m['Cells'].sum()
        expected.setdefault(('Stable', 'Number of Events'), {})[name] = int(m['Stable'].sum())

    for shown in SAMPLES:
        _show(controller, shown)
        result = _statistics(controller, monkeypatch)
        for stat, values in expected.items():
            for name, value in values.items():
                assert result[stat][name] == pytest.approx(value), (shown, stat, name)


class _FakeSample:
    def __init__(self, events):
        self.events = events
        self.event_count = len(events)

    def get_events(self, *_args, **_kwargs):
        return self.events


def test_dr_loader_gates_each_sample_with_its_own_gates(tmp_path, monkeypatch):
    import drc_pipeline

    controller = _controller()
    controller.experiment_dir = tmp_path
    loads = []

    def fake_sample(path):
        loads.append(Path(path).name)
        return _FakeSample(SAMPLES[f'Raw/{Path(path).name}'])

    monkeypatch.setattr(drc_pipeline, 'sample_from_fcs', fake_sample)
    monkeypatch.setattr(drc_pipeline, 'apply_unmixing_af_aware', lambda _c, raw, **_kw: raw)
    state = types.SimpleNamespace(selected_gates=['Cells'], gated_data_cache={})

    controller.experiment.settings.setdefault('unmixed', {})['event_channels_pnn'] = list(PNN)

    def load(key):
        return drc_pipeline.load_unmixed_gated(controller, state, tmp_path / key,
                                               channels=list(PNN)).values

    for key, events in SAMPLES.items():
        np.testing.assert_array_equal(load(key), events[_expected(events)['Cells']].astype(np.float32))
    assert len(loads) == 3
    load('Raw/B.fcs')
    assert len(loads) == 3

    _override(controller, 'Raw/B.fcs', 'Cells', 30, 90)
    b = SAMPLES['Raw/B.fcs']
    np.testing.assert_array_equal(load('Raw/B.fcs'), b[_expected(b, cells=(30, 90))['Cells']].astype(np.float32))
    assert len(loads) == 4
    c = SAMPLES['Raw/C.fcs']
    np.testing.assert_array_equal(load('Raw/C.fcs'), c[_expected(c)['Cells']].astype(np.float32))
    assert len(loads) == 4

    state.selected_gates = ['Stable']
    np.testing.assert_array_equal(load('Raw/C.fcs'), c[_expected(c)['Stable']].astype(np.float32))

"""Regression guard: Time-gate lookup tables are rebuilt on every sample switch.

A gate on the Time axis uses a `default` transform whose scale is re-fitted to
each sample's own time range on load. Its lookup table must therefore be rebuilt
on every sample switch, or the gate silently selects the wrong time window on the
new sample. `Controller.apply_custom_sample_gates` does this when a sample is
loaded, via `initialise_data_for_cytometry_plots`.

This test drives that real sample-load path end to end, so it fails if the
rebuild is removed or no longer reached when a sample is loaded.
"""

import numpy as np
from flowkit import Dimension, gates

from honeychrome.controller import Controller
from honeychrome.controller_components.transform import Transform


def _time_gate(name, lo, hi):
    # No transformation_ref -> the gate compares raw Time values directly.
    return gates.RectangleGate(name, dimensions=[Dimension('Time', range_min=lo, range_max=hi)])


def _load_sample(controller, sample_path, event_data):
    """Replay the tail of Controller.load_sample, which re-fits the Time scale."""
    controller.current_sample_path = sample_path
    controller.raw_event_data = event_data
    controller.clear_data_for_cytometry_plots()
    controller.initialise_data_for_cytometry_plots()
    return controller.data_for_cytometry_plots_raw['statistics']['TimeGate']['n_events_gate']


def _events_in_window(events, lo, hi):
    return int(((events[:, 0] >= lo) & (events[:, 0] <= hi)).sum())


def test_time_gate_lookup_is_rebuilt_on_sample_switch():
    controller = Controller()
    controller.current_mode = 'raw'

    time_transform = Transform()
    time_transform.set_transform(id='default', limits=[0, 1000])
    controller.raw_transformations = {'Time': time_transform}
    controller.raw_gating.add_gate(_time_gate('TimeGate', 300, 700), gate_path=('root',))
    controller.data_for_cytometry_plots_raw.update({
        'pnn': ['Time'],
        'fluoro_indices': [],
        'transformations': controller.raw_transformations,
        'lookup_tables': controller.raw_lookup_tables,
        'gating': controller.raw_gating,
        'plots': [],
    })
    controller.data_for_cytometry_plots = controller.data_for_cytometry_plots_raw
    controller.calculate_lookup_tables(mode='raw')  # as on experiment load

    events_a = np.linspace(0, 1000, 1001).reshape(-1, 1)
    count_a = _load_sample(controller, 'A.fcs', events_a)
    assert abs(count_a - _events_in_window(events_a, 300, 700)) <= 2

    # B's run is 10x longer, so its Time scale is re-fitted to 0..10000. A stale
    # lookup table from A would select ~195 events around Time 1500..3400 instead.
    events_b = np.linspace(0, 10000, 1001).reshape(-1, 1)
    count_b = _load_sample(controller, 'B.fcs', events_b)
    truth_b = _events_in_window(events_b, 300, 700)
    assert abs(count_b - truth_b) <= 2, (
        f'Time gate mis-gated sample B after a sample switch: got {count_b} events, '
        f'expected ~{truth_b} (Time 300..700). Is the Time-gate lookup rebuild still '
        f'reached from initialise_data_for_cytometry_plots?'
    )

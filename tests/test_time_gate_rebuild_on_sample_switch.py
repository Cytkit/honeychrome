"""Regression guard for `Controller._rebuild_per_sample_lookup_tables`.

A gate on the Time axis uses a `default` transform whose scale is re-fitted to
each sample's own time range on load. Its lookup table must therefore be rebuilt
on every sample switch, or the gate silently selects the wrong time window on the
new sample. `_rebuild_per_sample_lookup_tables` (called from the full-recalc
branch of `calc_hists_and_stats`) does this.

This test drives that real entry point end to end. It fails if either the method
or its call site is removed — so a future accidental deletion is caught.
"""

import numpy as np
from flowkit import Dimension, gates

from honeychrome.controller import Controller
from honeychrome.controller_components.transform import Transform


def _time_gate(name, lo, hi):
    # No transformation_ref -> the gate compares raw Time values directly.
    return gates.RectangleGate(name, dimensions=[Dimension('Time', range_min=lo, range_max=hi)])


def _load_sample(controller, event_data, time_limits):
    """Replay the per-sample state the app sets before recalculating stats."""
    controller.raw_transformations['Time'].set_transform(id='default', limits=time_limits)  # per-sample Time scale
    controller.data_for_cytometry_plots['event_data'] = event_data
    # NB: deliberately no manual calculate_lookup_tables() here — the Time gate's
    # table must be rebuilt by _rebuild_per_sample_lookup_tables inside this call.
    controller.calc_hists_and_stats()
    return controller.data_for_cytometry_plots['statistics']['TimeGate']['n_events_gate']


def test_time_gate_lookup_is_rebuilt_on_sample_switch():
    controller = Controller()
    controller.current_mode = 'raw'

    time_transform = Transform()  # the 'default' (Time) transform, as assign_default_transforms builds it
    time_transform.set_transform(id='default', limits=[0, 1000])  # give the scale an initial fit
    controller.raw_transformations = {'Time': time_transform}
    controller.raw_gating.add_gate(_time_gate('TimeGate', 300, 700), gate_path=('root',))

    controller.data_for_cytometry_plots = {
        'event_data': None,
        'pnn': ['Time'],
        'fluoro_indices': [],
        'transformations': controller.raw_transformations,
        'lookup_tables': controller.raw_lookup_tables,  # same object calculate_lookup_tables fills
        'gating': controller.raw_gating,
        'gate_membership': {},
        'plots': [],
        'histograms': [],
        'statistics': {},
    }

    # --- Sample A: Time spans 0..1000 -----------------------------------
    controller.current_sample_path = 'A.fcs'
    events_a = np.linspace(0, 1000, 1001).reshape(-1, 1)
    controller.calculate_lookup_tables(mode='raw', top_gate='root')  # initial build (like experiment load)
    count_a = _load_sample(controller, events_a, [0, 1000])
    truth_a = int(((events_a[:, 0] >= 300) & (events_a[:, 0] <= 700)).sum())
    assert abs(count_a - truth_a) <= 2, f'Sample A gate count {count_a} != ~{truth_a}'

    # --- Switch to Sample B: Time spans 0..10000 (a 10x longer run) ------
    # Only _rebuild_per_sample_lookup_tables (inside calc_hists_and_stats) can
    # make the Time gate correct here. Remove it and this count reverts to the
    # stale window (~195 events around Time 1500..3400) and the assert fails.
    controller.current_sample_path = 'B.fcs'
    events_b = np.linspace(0, 10000, 1001).reshape(-1, 1)
    count_b = _load_sample(controller, events_b, [0, 10000])
    truth_b = int(((events_b[:, 0] >= 300) & (events_b[:, 0] <= 700)).sum())
    assert abs(count_b - truth_b) <= 2, (
        f'Time gate mis-gated Sample B after switch: got {count_b} events, '
        f'expected ~{truth_b} (Time 300..700). '
        f'Has _rebuild_per_sample_lookup_tables been removed or unwired?'
    )

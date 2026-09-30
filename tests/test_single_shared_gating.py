"""Regression tests for one shared hierarchy plus per-sample gate overrides."""

import numpy as np
import pytest
from flowkit import Dimension, gates, transforms

from honeychrome.controller import Controller
from honeychrome.controller_components import spectral_controller
from honeychrome.controller_components.gml_functions_mod_from_flowkit import from_gml
from honeychrome.controller_components.spectral_controller import SpectralAutoGenerator
from honeychrome.controller_components.transform import Transform


def _range_gate(name, lo, hi, channel='X', transformation_ref=None):
    return gates.RectangleGate(
        name,
        dimensions=[Dimension(
            channel,
            range_min=lo,
            range_max=hi,
            transformation_ref=transformation_ref,
        )],
    )


@pytest.mark.parametrize('scope', ['raw', 'unmixed'])
def test_switching_to_sample_without_override_rebuilds_template_lookup(scope):
    controller = Controller()
    transformation = Transform()
    transformation.set_transform(limits=[0, 10])
    setattr(controller, f'{scope}_transformations', {'X': transformation})
    gating = getattr(controller, f'{scope}_gating')
    lookup_tables = getattr(controller, f'{scope}_lookup_tables')
    gating.add_gate(_range_gate('Gate', 2, 8), gate_path=('root',))

    controller.current_sample_path = 'sample-A.fcs'
    controller.customise_gate(scope, 'Gate')
    custom = gating.get_gate('Gate', sample_id='sample-A.fcs')
    custom.dimensions = [Dimension('X', range_min=5, range_max=8)]
    controller.calculate_lookup_tables(mode=scope, top_gate='Gate')
    sample_a_lookup = lookup_tables['Gate'].copy()

    controller.current_sample_path = 'sample-B.fcs'
    controller.apply_custom_sample_gates(scope)
    sample_b_lookup = lookup_tables['Gate'].copy()

    controller.calculate_lookup_tables(mode=scope, top_gate='Gate')
    expected_template_lookup = lookup_tables['Gate'].copy()

    assert not np.array_equal(sample_b_lookup, sample_a_lookup)
    np.testing.assert_array_equal(sample_b_lookup, expected_template_lookup)


class _FakeSample:
    pnn_labels = ['B1-A']

    def get_events(self, *_args, **_kwargs):
        return np.array([[10.0], [20.0], [30.0]])


def test_autogenerate_handles_a_missing_base_gate_without_index_error(tmp_path, monkeypatch):
    controller = Controller()
    controller.experiment_dir = tmp_path
    controller.experiment.settings['raw'].update({
        'event_channels_pnn': ['B1-A'],
        'fluorescence_channel_ids': [0],
        'magnitude_ceiling': 100.0,
    })
    controller.experiment.process.update({
        'base_gate_priority_order': ['Cells', 'root'],
        'fluorescence_channel_filter': 'area_only',
        'spectral_model': [],
        'profiles': {},
    })
    controller.experiment.samples.update({
        'single_stain_controls': ['control.fcs'],
        'all_sample_nevents': {'control.fcs': 100},
        'all_samples': {'control.fcs': 'APC (Beads)'},
        'unstained_samples': [],
    })

    controller.raw_gating.add_transform(
        'B1-A', transforms.LinearTransform(param_t=100.0, param_a=0.0)
    )
    controller.raw_gating.add_gate(
        _range_gate('Cells', 0.0, 1.0, 'B1-A', 'B1-A'), gate_path=('root',)
    )
    controller.data_for_cytometry_plots_raw['plots'] = []
    generator = SpectralAutoGenerator(None, controller)
    assert generator.base_gate_label == 'Cells'

    controller.raw_gating.remove_gate('Cells')
    controller.raw_gating.add_gate(
        _range_gate('Beads', 0.0, 1.0, 'B1-A', 'B1-A'), gate_path=('root',)
    )

    monkeypatch.setattr(spectral_controller, 'check_fcs_matches_experiment', lambda *_: True)
    monkeypatch.setattr(spectral_controller, 'sample_from_fcs', lambda *_: _FakeSample())
    monkeypatch.setattr(
        spectral_controller, 'get_raw_events',
        lambda *_args, **_kwargs: np.array([[10.0], [20.0], [30.0]]),
    )
    monkeypatch.setattr(spectral_controller, 'find_empirical_peak', lambda *_: 0)
    monkeypatch.setattr(spectral_controller, 'match_fluorophore', lambda *_: 'APC')
    monkeypatch.setattr(spectral_controller, 'match_marker', lambda *_: None)

    with pytest.warns(UserWarning, match='base gate.*not found'):
        assert generator.generate_spectral_control(0) is False


def test_custom_sample_gate_still_round_trips_without_template_schema(tmp_path):
    path = tmp_path / 'custom-gate.kit'
    controller = Controller()
    controller.experiment.experiment_path = str(path)
    controller.experiment_dir = tmp_path
    controller.raw_transformations = {}
    controller.cleaned_events = {}
    controller.raw_gating.add_gate(_range_gate('Gate', 2, 8), gate_path=('root',))

    controller.current_sample_path = 'sample-A.fcs'
    controller.customise_gate('raw', 'Gate')
    custom = controller.raw_gating.get_gate('Gate', sample_id='sample-A.fcs')
    custom.dimensions = [Dimension('X', range_min=5, range_max=8)]
    controller._sync_custom_gate_from_strategy('raw', 'Gate')
    controller.save_experiment()

    reloaded = Controller()
    reloaded.experiment.load(path)
    reloaded.raw_gating = from_gml(reloaded.experiment.cytometry['raw_gating'])
    reloaded._load_custom_sample_gates()
    reloaded.current_sample_path = 'sample-A.fcs'
    reloaded.apply_custom_sample_gates('raw')

    restored = reloaded.raw_gating.get_gate('Gate', sample_id='sample-A.fcs')
    assert restored.dimensions[0].min == 5
    assert restored.dimensions[0].max == 8
    assert 'gating_templates' not in reloaded.experiment.cytometry

"""Regression tests for one shared hierarchy plus per-sample gate overrides."""

import json
from copy import deepcopy

import numpy as np
import pytest
from flowkit import Dimension, GatingStrategy, gates, transforms

from honeychrome.controller import Controller
from honeychrome.controller_components import spectral_controller
from honeychrome.controller_components.gml_functions_mod_from_flowkit import from_gml, to_gml
from honeychrome.controller_components.spectral_controller import SpectralAutoGenerator
from honeychrome.controller_components.transform import Transform
from honeychrome.experiment_model import ExperimentModel
from honeychrome.settings import cytometry_default, process_default, samples_default, settings_default


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


def _gml_with_gate(name):
    gating = GatingStrategy()
    gating.add_gate(_range_gate(name, 0.2, 0.8), gate_path=('root',))
    return to_gml(gating)


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


def _write_development_kit(path):
    default_raw = _gml_with_gate('DefaultCells')
    active_raw = _gml_with_gate('Beads')
    default_unmixed = _gml_with_gate('DefaultLive')
    data = {
        'settings': deepcopy(settings_default),
        'samples': deepcopy(samples_default),
        'process': deepcopy(process_default),
        'cytometry': deepcopy(cytometry_default),
        'statistics': [],
    }
    data['cytometry'].update({
        'raw_gating': default_raw,
        'gating': default_unmixed,
        'raw_plots': [],
        'plots': [],
        'gating_templates': {
            'default': {
                'raw_gml': default_raw,
                'unmixed_gml': default_unmixed,
                'raw_plots': [],
                'unmixed_plots': [],
                'dynamic_dimensions': {'raw': {}, 'unmixed': {}},
            },
        },
        'raw_gating_templates': {
            'default': {'gml': default_raw, 'plots': [], 'dynamic_dimensions': {}},
            'beads': {'gml': active_raw, 'plots': [], 'dynamic_dimensions': {}},
        },
        'unmixed_gating_templates': {
            'default': {'gml': default_unmixed, 'plots': [], 'dynamic_dimensions': {}},
        },
        'default_raw_template_name': 'default',
        'default_unmixed_template_name': 'default',
        'raw_custom_sample_gates': {},
        'unmixed_custom_sample_gates': {},
    })
    data['samples']['sample_template_assignments'] = {
        'sample.fcs': {'raw': 'beads', 'unmixed': 'default'},
    }
    path.write_text(json.dumps(data), encoding='utf-8')
    return default_raw, default_unmixed


def test_loading_a_development_kit_discards_multiple_template_state(tmp_path):
    path = tmp_path / 'old-development.kit'
    default_raw, default_unmixed = _write_development_kit(path)

    experiment = ExperimentModel()
    experiment.load(path)

    assert experiment.cytometry['raw_gating'] == default_raw
    assert experiment.cytometry['gating'] == default_unmixed
    for key in (
        'gating_templates', 'raw_gating_templates', 'unmixed_gating_templates',
        'default_raw_template_name', 'default_unmixed_template_name',
    ):
        assert key not in experiment.cytometry
    assert 'sample_template_assignments' not in experiment.samples
    assert experiment.cytometry['raw_custom_sample_gates'] == {}
    assert experiment.cytometry['unmixed_custom_sample_gates'] == {}


def test_development_kit_uses_designated_scoped_defaults_and_keeps_custom_gates(tmp_path):
    path = tmp_path / 'designated-default.kit'
    _write_development_kit(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    primary_raw = _gml_with_gate('PrimaryRaw')
    primary_unmixed = _gml_with_gate('PrimaryUnmixed')
    custom_gml = to_gml(GatingStrategy())
    data['cytometry']['raw_gating_templates']['primary'] = {
        'gml': primary_raw, 'plots': [{'id': 'primary-raw'}], 'dynamic_dimensions': {},
    }
    data['cytometry']['unmixed_gating_templates']['analysis'] = {
        'gml': primary_unmixed, 'plots': [{'id': 'primary-unmixed'}],
        'dynamic_dimensions': {},
    }
    data['cytometry']['default_raw_template_name'] = 'primary'
    data['cytometry']['default_unmixed_template_name'] = 'analysis'
    data['cytometry']['raw_custom_sample_gates'] = {
        'sample.fcs': {'Gate': custom_gml},
    }
    path.write_text(json.dumps(data), encoding='utf-8')

    experiment = ExperimentModel()
    experiment.load(path)

    assert experiment.cytometry['raw_gating'] == primary_raw
    assert experiment.cytometry['gating'] == primary_unmixed
    assert experiment.cytometry['raw_plots'] == [{'id': 'primary-raw'}]
    assert experiment.cytometry['plots'] == [{'id': 'primary-unmixed'}]
    assert experiment.cytometry['raw_custom_sample_gates'] == {
        'sample.fcs': {'Gate': custom_gml},
    }


def test_development_kit_can_collapse_unified_schema_alone(tmp_path):
    path = tmp_path / 'unified-only.kit'
    default_raw, default_unmixed = _write_development_kit(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    data['cytometry'].pop('raw_gating_templates')
    data['cytometry'].pop('unmixed_gating_templates')
    path.write_text(json.dumps(data), encoding='utf-8')

    experiment = ExperimentModel()
    experiment.load(path)

    assert experiment.cytometry['raw_gating'] == default_raw
    assert experiment.cytometry['gating'] == default_unmixed
    assert 'gating_templates' not in experiment.cytometry


def test_saving_does_not_write_multiple_template_state(tmp_path):
    path = tmp_path / 'saved.kit'
    experiment = ExperimentModel()
    experiment.experiment_path = str(path)
    experiment.cytometry.update({
        'gating_templates': {'default': {}},
        'raw_gating_templates': {'default': {}},
        'unmixed_gating_templates': {'default': {}},
        'default_raw_template_name': 'default',
        'default_unmixed_template_name': 'default',
    })
    experiment.samples['sample_template_assignments'] = {}

    experiment.save()
    saved = json.loads(path.read_text(encoding='utf-8'))

    assert not any('gating_templates' in key for key in saved['cytometry'])
    assert 'sample_template_assignments' not in saved['samples']


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

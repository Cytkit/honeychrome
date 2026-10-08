"""Unmixed gates, plots and axis settings survive a change of the unmixed
channel list (regenerating, adding or removing spectral controls), except
where they use a channel that no longer exists."""

import numpy as np
import pytest
from flowkit import Dimension, gates

from honeychrome.controller import Controller
from honeychrome.controller_components.functions import assign_default_transforms, define_quad_gates, update_transforms
from honeychrome.controller_components.gml_functions_mod_from_flowkit import from_gml, to_gml
from honeychrome.controller_components.spectral_controller import SpectralAutoGenerator, restore_user_labels
from flowkit import GatingStrategy

pytestmark = pytest.mark.filterwarnings('ignore:No events bus connected')

RAW_PNN = ['Time', 'FSC-A', 'SSC-A', 'B1-A', 'B2-A', 'B3-A', 'R1-A', 'R2-A']
FL_IDS = [3, 4, 5, 6, 7]
GATE_CHANNEL = {'FITC': 'B1-A', 'PE': 'B2-A', 'PE-Cy5': 'B3-A', 'APC': 'R1-A', 'APC-Cy7': 'R2-A'}


def _profile(peak):
    profile = np.full(len(FL_IDS), 0.05)
    profile[FL_IDS.index(RAW_PNN.index(peak))] = 1.0
    return profile.tolist()


def _set_model(controller, labels):
    model = controller.experiment.process['spectral_model']
    profiles = controller.experiment.process['profiles']
    model.clear()
    profiles.clear()
    for label in labels:
        model.append({'label': label, 'control_type': 'Channel Assignment', 'particle_type': '',
                      'gate_channel': GATE_CHANNEL[label], 'sample_name': '', 'sample_path': '',
                      'gate_label': ''})
        profiles[label] = _profile(GATE_CHANNEL[label])


def _rect(name, x, y=None):
    dims = [Dimension(x, range_min=0.2, range_max=0.8, transformation_ref=x)]
    if y:
        dims.append(Dimension(y, range_min=0.2, range_max=0.8, transformation_ref=y))
    return gates.RectangleGate(name, dimensions=dims)


def _controller(labels):
    controller = Controller()
    controller.experiment.settings['raw'].update({
        'event_channels_pnn': RAW_PNN,
        'fluorescence_channel_ids': FL_IDS,
        'scatter_channel_ids': [1, 2],
        'time_channel_id': 0,
        'magnitude_ceiling': 262144.0,
        'width_ceiling': 262144.0,
        'area_channels': [c[:-2] for c in RAW_PNN if c.endswith('-A')],
        'height_channels': [],
        'width_channels': [],
        'scatter_channels': ['FSC-A', 'SSC-A'],
        'fluorescence_channels': [RAW_PNN[i] for i in FL_IDS],
    })
    controller.experiment.process['fluorescence_channel_filter'] = 'area_only'
    controller.experiment.cytometry['raw_transforms'] = assign_default_transforms(controller.experiment.settings['raw'])
    controller.experiment.cytometry['raw_gating'] = to_gml(GatingStrategy())
    controller.initialise_ephemeral_data()
    controller.raw_gating.add_gate(_rect('Cells', 'FSC-A', 'SSC-A'), gate_path=('root',))
    controller.experiment.cytometry['raw_plots'] = [
        {'type': 'hist2d', 'channel_x': 'FSC-A', 'channel_y': 'SSC-A', 'source_gate': 'root',
         'child_gates': ['Cells']}]
    controller.data_for_cytometry_plots_raw['plots'] = controller.experiment.cytometry['raw_plots']
    _set_model(controller, labels)
    controller.refresh_spectral_process()
    return controller


def _gate_user_strategy(controller):
    """Gates and plots a user would draw on the unmixed data."""
    gating = controller.unmixed_gating
    gating.add_gate(_rect('CD4', 'FITC', 'PE'), gate_path=('root', 'Cells'))
    gating.add_gate(_rect('Activated', 'APC'), gate_path=('root', 'Cells', 'CD4'))
    gating.add_gate(_rect('Bright', 'PE-Cy5'), gate_path=('root', 'Cells'))
    gating.add_gate(_rect('Bright sub', 'FITC'), gate_path=('root', 'Cells', 'Bright'))
    quad_divs, quadrants = define_quad_gates(0.5, 0.5, 'FITC', 'APC', controller.unmixed_transformations)
    gating.add_gate(gates.QuadrantGate('Quad', quad_divs, quadrants), gate_path=('root', 'Cells'))
    gating.add_gate(_rect('DP PE', 'PE'), gate_path=('root', 'Cells', 'Quad', 'FITC+ APC+'))
    controller.unmixed_transformations['FITC'].set_transform(limits=[0.1, 0.9])
    controller.experiment.cytometry['plots'] += [
        {'type': 'hist2d', 'channel_x': 'FITC', 'channel_y': 'PE', 'source_gate': 'Cells', 'child_gates': ['CD4']},
        {'type': 'hist1d', 'channel_x': 'APC', 'source_gate': 'CD4', 'child_gates': ['Activated']},
        {'type': 'hist1d', 'channel_x': 'PE-Cy5', 'source_gate': 'Cells', 'child_gates': ['Bright']},
        {'type': 'hist2d', 'channel_x': 'FITC', 'channel_y': 'APC', 'source_gate': 'Cells', 'child_gates': ['Quad']},
    ]


def _autosave(controller):
    """The part of save_experiment that writes the live gating back to the .kit."""
    controller.experiment.cytometry['gating'] = to_gml(controller.unmixed_gating)
    update_transforms(controller.experiment.cytometry['transforms'], controller.unmixed_transformations)


def _gate_names(controller):
    return {name for name, _ in controller.unmixed_gating.get_gate_ids()}


def _plot_channels(controller):
    return [(p.get('channel_x'), p.get('channel_y')) for p in controller.experiment.cytometry['plots']]


def test_identical_model_keeps_everything():
    labels = ['FITC', 'PE', 'PE-Cy5', 'APC']
    controller = _controller(labels)
    _gate_user_strategy(controller)
    _autosave(controller)
    before = _gate_names(controller)
    _set_model(controller, labels)
    controller.refresh_spectral_process()
    assert _gate_names(controller) == before


def test_reordered_model_keeps_everything():
    controller = _controller(['FITC', 'PE', 'PE-Cy5', 'APC'])
    _gate_user_strategy(controller)
    _autosave(controller)
    before = _gate_names(controller)
    plots_before = _plot_channels(controller)
    # PE-Cy5 now peaks after APC, so the unmixed channel order changes
    GATE_CHANNEL['PE-Cy5'] = 'R2-A'
    try:
        _set_model(controller, ['FITC', 'PE', 'PE-Cy5', 'APC'])
        controller.refresh_spectral_process()
    finally:
        GATE_CHANNEL['PE-Cy5'] = 'B3-A'
    assert controller.experiment.settings['unmixed']['fluorescence_channels'] == ['FITC', 'PE', 'APC', 'PE-Cy5']
    assert _gate_names(controller) == before
    assert _plot_channels(controller) == plots_before
    assert controller.unmixed_transformations['FITC'].limits == [0.1, 0.9]


def test_added_control_keeps_everything():
    controller = _controller(['FITC', 'PE', 'PE-Cy5', 'APC'])
    _gate_user_strategy(controller)
    _autosave(controller)
    before = _gate_names(controller)
    _set_model(controller, ['FITC', 'PE', 'PE-Cy5', 'APC', 'APC-Cy7'])
    controller.refresh_spectral_process()
    assert 'APC-Cy7' in controller.experiment.settings['unmixed']['event_channels_pnn']
    assert _gate_names(controller) == before
    assert controller.unmixed_transformations['FITC'].limits == [0.1, 0.9]
    assert 'APC-Cy7' in controller.unmixed_transformations


def test_removed_channel_drops_only_its_gates_and_their_descendants():
    controller = _controller(['FITC', 'PE', 'PE-Cy5', 'APC'])
    _gate_user_strategy(controller)
    _autosave(controller)
    _set_model(controller, ['FITC', 'PE', 'APC'])
    controller.refresh_spectral_process()
    names = _gate_names(controller)
    assert {'Cells', 'CD4', 'Activated', 'Quad', 'FITC+ APC+', 'FITC- APC-', 'DP PE'} <= names
    assert 'Bright' not in names and 'Bright sub' not in names
    assert ('PE-Cy5', None) not in _plot_channels(controller)
    assert ('FITC', 'APC') in _plot_channels(controller)
    assert controller.unmixed_transformations['FITC'].limits == [0.1, 0.9]
    stored = from_gml(controller.experiment.cytometry['gating'])
    assert {name for name, _ in stored.get_gate_ids()} == names


def test_quadrant_gate_dropped_with_its_channel():
    controller = _controller(['FITC', 'PE', 'PE-Cy5', 'APC'])
    _gate_user_strategy(controller)
    _autosave(controller)
    _set_model(controller, ['FITC', 'PE', 'PE-Cy5'])
    controller.refresh_spectral_process()
    names = _gate_names(controller)
    assert not {'Quad', 'FITC+ APC+', 'DP PE', 'Activated'} & names
    assert {'Cells', 'CD4', 'Bright', 'Bright sub'} <= names


def test_unsaved_unmixed_edits_are_carried_over():
    """Gates drawn since the last autosave are read from the live gating."""
    controller = _controller(['FITC', 'PE', 'PE-Cy5', 'APC'])
    _gate_user_strategy(controller)
    assert 'CD4' not in controller.experiment.cytometry['gating']
    _set_model(controller, ['FITC', 'PE', 'PE-Cy5', 'APC', 'APC-Cy7'])
    controller.refresh_spectral_process()
    assert 'CD4' in _gate_names(controller)


def test_per_sample_overrides_follow_their_gates():
    controller = _controller(['FITC', 'PE', 'PE-Cy5', 'APC'])
    _gate_user_strategy(controller)
    for name in ('CD4', 'Bright'):
        controller.current_sample_path = 'Raw/sample.fcs'
        controller.customise_gate('unmixed', name)
    _set_model(controller, ['FITC', 'PE', 'APC'])
    controller.refresh_spectral_process()
    overrides = controller.custom_sample_gates['unmixed']['Raw/sample.fcs']
    assert set(overrides) == {'CD4'}


def test_first_unmix_copies_scatter_gates_from_raw():
    controller = _controller(['FITC', 'PE'])
    assert _gate_names(controller) == {'Cells'}
    assert _plot_channels(controller) == [('FSC-A', 'SSC-A')]


def test_plots_of_removed_populations_are_dropped():
    controller = _controller(['FITC', 'PE', 'PE-Cy5', 'APC'])
    _gate_user_strategy(controller)
    controller.experiment.cytometry['plots'].append(
        {'type': 'hist2d', 'channel_x': 'FITC', 'channel_y': 'PE', 'source_gate': 'Bright', 'child_gates': ['Bright sub']})
    _set_model(controller, ['FITC', 'PE', 'APC'])
    controller.refresh_spectral_process()
    assert all(p['source_gate'] != 'Bright' for p in controller.experiment.cytometry['plots'])


def test_removed_gates_are_reported():
    controller = _controller(['FITC', 'PE', 'PE-Cy5', 'APC'])
    _gate_user_strategy(controller)
    controller.warning_collector = []
    _set_model(controller, ['FITC', 'PE', 'APC'])
    controller.refresh_spectral_process()
    assert len(controller.warning_collector) == 1
    assert 'PE-Cy5' in controller.warning_collector[0] and 'Bright sub' in controller.warning_collector[0]


def test_falls_back_to_raw_copy_when_gating_cannot_be_pruned():
    """A Boolean gate referencing a gate that has to go blocks the removal, so
    the previous behaviour (scatter gates copied from raw) applies."""
    controller = _controller(['FITC', 'PE', 'PE-Cy5', 'APC'])
    _gate_user_strategy(controller)
    controller.unmixed_gating.add_gate(
        gates.BooleanGate('Not bright', 'not', [{'ref': 'Bright', 'path': ('root', 'Cells'), 'complement': False}]),
        gate_path=('root', 'Cells'))
    _set_model(controller, ['FITC', 'PE', 'APC'])
    controller.refresh_spectral_process()
    assert _gate_names(controller) == {'Cells'}


def _control(label, path, antigen=''):
    return {'label': label, 'antigen': antigen, 'sample_path': f'/exp/Raw/{path}.fcs',
            'sample_name': path, 'control_type': 'Single Stained Spectral Control'}


def test_restore_user_labels_and_antigens_by_tube():
    model = [_control('FITC', 'A1 FITC (Cells)', 'CD3'), _control('PE', 'A2 PE (Cells)')]
    profiles = {'FITC': [1.0], 'PE': [2.0]}
    previous = [_control('FITC Cells', 'A1 FITC (Cells)', 'CD4'), _control('PE', 'A2 PE (Cells)', 'CD8')]
    renamed = restore_user_labels(model, profiles, previous)
    assert renamed == {'FITC': 'FITC Cells'}
    assert [c['label'] for c in model] == ['FITC Cells', 'PE']
    assert [c['antigen'] for c in model] == ['CD4', 'CD8']
    assert profiles == {'FITC Cells': [1.0], 'PE': [2.0]}


def test_restore_user_labels_never_duplicates_a_label():
    model = [_control('FITC', 'A1 FITC (Cells)'), _control('PE', 'A2 PE (Cells)')]
    profiles = {'FITC': [1.0], 'PE': [2.0]}
    previous = [_control('PE', 'A1 FITC (Cells)')]
    assert restore_user_labels(model, profiles, previous) == {}
    assert [c['label'] for c in model] == ['FITC', 'PE']
    assert profiles == {'FITC': [1.0], 'PE': [2.0]}


def test_restore_user_labels_swapped_labels():
    model = [_control('FITC', 'A1 FITC (Cells)'), _control('PE', 'A2 PE (Cells)')]
    profiles = {'FITC': [1.0], 'PE': [2.0]}
    previous = [_control('PE', 'A1 FITC (Cells)'), _control('FITC', 'A2 PE (Cells)')]
    restore_user_labels(model, profiles, previous)
    assert [c['label'] for c in model] == ['PE', 'FITC']
    assert profiles == {'PE': [1.0], 'FITC': [2.0]}


def test_restore_user_labels_falls_back_to_a_unique_sample_name():
    model = [_control('FITC', 'A1 FITC (Cells)'), _control('PE', 'A2 PE (Cells)')]
    profiles = {'FITC': [1.0], 'PE': [2.0]}
    moved = _control('FITC Cells', 'A1 FITC (Cells)')
    moved['sample_path'] = '/old location/A1 FITC (Cells).fcs'
    twin_a, twin_b = _control('PE one', 'A2 PE (Cells)'), _control('PE two', 'A2 PE (Cells)')
    twin_a['sample_path'], twin_b['sample_path'] = '/a.fcs', '/b.fcs'
    restore_user_labels(model, profiles, [moved, twin_a, twin_b])
    assert [c['label'] for c in model] == ['FITC Cells', 'PE']


def test_regenerate_keeps_gates_on_a_renamed_label(monkeypatch):
    controller = _controller(['FITC', 'PE', 'APC'])
    model = controller.experiment.process['spectral_model']
    for control in model:
        control['sample_path'] = f'/exp/Raw/{control["label"]}.fcs'
        control['sample_name'] = control['label']
    # user renamed FITC after the last auto-generate
    model[0]['label'] = 'FITC Cells'
    model[0]['antigen'] = 'CD4'
    profiles = controller.experiment.process['profiles']
    profiles['FITC Cells'] = profiles.pop('FITC')
    controller.refresh_spectral_process()
    controller.unmixed_gating.add_gate(_rect('CD4+', 'FITC Cells'), gate_path=('root', 'Cells'))
    _autosave(controller)
    previous_controls = [{key: c.get(key) for key in ('label', 'antigen', 'sample_path', 'sample_name')}
                         for c in model]

    generator = SpectralAutoGenerator(None, controller, previous_controls=previous_controls)
    monkeypatch.setattr(generator, 'get_unstained_negative', lambda particle_type='Cells': False)

    def generate(n):
        label = ['FITC', 'PE', 'APC'][n]
        generator.spectral_model.append(
            {'label': label, 'antigen': '', 'control_type': 'Single Stained Spectral Control',
             'gate_channel': GATE_CHANNEL[label], 'sample_path': f'/exp/Raw/{label}.fcs', 'sample_name': label})
        generator.profiles[label] = _profile(GATE_CHANNEL[label])
        return True
    monkeypatch.setattr(generator, 'generate_spectral_control', generate)
    generator.progress_target = 3
    generator.run()
    controller.refresh_spectral_process()

    assert [c['label'] for c in model] == ['FITC Cells', 'PE', 'APC']
    assert model[0]['antigen'] == 'CD4'
    assert 'CD4+' in _gate_names(controller)

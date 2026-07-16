'''
Experiment model
----------------
Experiment has:
-settings (raw channel information, spectral settings, unmixed channels, fcs export settings)
-ephemeral list of samples and controls (refreshed from sample and control subdirectories)
-process (the spectral unmixing process)
-cytometry (i.e. the unmixed cytometry)

-list of samples and controls
-raw gates
-spectral model
-unmixing matrix
-compensation matrix
-unmixed gating hierarchy
-transforms
-list of plots and their associated channels, transforms, limits, source gate, child gates

serialise and save in pickle
(maybe use json and sqlite3 in the future in case data loss mitigation is necessary)

methods:
-create experiment
-load experiment
-save experiment
-reconsititute experiment from FCS files
-auto_generate_spectral_model
-calculate_spectral_process
-save_spectral_model_to_library
-load_spectral_library - returns data frame with spectral library - maybe change this to search for control name
-create or update gate
'''

import time
import json
import sys
from copy import deepcopy
from pathlib import Path
from flowkit import GatingStrategy, Dimension, gates
import numpy as np
import warnings
import os
from flowio import FlowData
from typing import Optional

from honeychrome.controller_components.functions import generate_transformations, assign_default_transforms
from honeychrome.controller_components.gml_functions_mod_from_flowkit import to_gml
from honeychrome.controller_components.gating_templates import migrate_cytometry_templates, migrate_scoped_gating_templates
from honeychrome.settings import settings_default, samples_default, process_default, cytometry_default, sample_name_source

import logging
logger = logging.getLogger(__name__)

def check_for_windows_junction(path):
    if os.path.isjunction(path):
        return True
    return False

def safe_save(content, filename):
    temp_name = filename + '.tmp'

    # Write to temp file
    with open(temp_name, 'w') as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())

    # Retry loop handles transient locks (antivirus, indexer, concurrent save)
    for attempt in range(5):
        try:
            os.replace(temp_name, filename)
            return
        except PermissionError as e:
            if attempt < 4:
                time.sleep(0.1 * (attempt + 1))
            else:
                logger.error(f'Failed to replace {filename}. Saved as {temp_name}. {e}')
        except Exception as e:
            logger.error(f'Failed to replace {filename}. Saved as {temp_name}. {e}')
            return

def check_fcs_matches_experiment(sample_full_path, experiment_pnn_raw, magnitude_ceiling):
    sample_metadata = FlowData(sample_full_path, only_text=True, use_header_offsets=True)
    sample_pnn = sample_metadata.pnn_labels
    # Subset check: file must contain all required channels; extra channels are allowed.
    # This permits FACSDiscover files with different derived-parameter sets to coexist.
    channels_match = set(experiment_pnn_raw).issubset(set(sample_pnn))
    return channels_match

class ExperimentModel:
    def __init__(self):
        ### experiment_path is file path and data directory (minus extension) ###
        self.experiment_path: Optional[str] = None

        ### data to save and load ###
        self.settings: dict = deepcopy(settings_default)
        self.samples: dict = deepcopy(samples_default)
        self.process: dict = deepcopy(process_default)
        self.cytometry: dict = deepcopy(cytometry_default)
        self.statistics: list = []

        self.progress_indicator = 0

    def create(self, experiment_path):
        self.experiment_path = str(experiment_path)
        ### copy defaults from experiment settings ###
        self.settings = deepcopy(settings_default)
        self.samples = deepcopy(samples_default)
        self.process = deepcopy(process_default)
        self.cytometry = deepcopy(cytometry_default)
        self.statistics = []

        ### set up default raw gating, raw transforms, raw plots ---- check that this corresponds with default channels
        self.cytometry['raw_transforms'] = assign_default_transforms(self.settings['raw'])
        raw_transformations = generate_transformations(self.cytometry['raw_transforms'])

        raw_gating = GatingStrategy()
        for label in self.settings['raw']['event_channels_pnn']:
            raw_gating.transformations[label] = raw_transformations[label].xform

        _raw = self.settings['raw']
        _scatter_pnn = _raw.get('scatter_param')  # None until FCS import resolves cytometer
        _morph_x  = _scatter_pnn[0] if _scatter_pnn else 'FSC-A'
        _morph_y  = _scatter_pnn[1] if _scatter_pnn else 'SSC-A'
        # Singlet Y: use FSC-H if available, fall back to FSC-W, then morph_x
        _pnn      = _raw.get('event_channels_pnn', [])
        _sing_y   = next((ch for ch in [_morph_x.replace('-A', '-H'), 'FSC-H', 'FSC-W'] if ch in _pnn), 'FSC-W')

        label = 'Cells'
        channel_x = _morph_x
        channel_y = _morph_y
        dim_x = Dimension(channel_x, range_min=0.2, range_max=0.8, transformation_ref=channel_x)
        dim_y = Dimension(channel_y, range_min=0.2, range_max=0.8, transformation_ref=channel_y)
        gate = gates.RectangleGate(label, dimensions=[dim_x, dim_y])
        raw_gating.add_gate(gate, gate_path=('root',))

        label = 'Singlets'
        channel_x = _morph_x
        channel_y = _sing_y
        dim_x = Dimension(channel_x, range_min=0, range_max=1, transformation_ref=channel_x)
        dim_y = Dimension(channel_y, range_min=0, range_max=1, transformation_ref=channel_y)
        vertices = [(0.2,0.2), (0.8,0.2), (0.8,0.8), (0.2,0.8)]
        gate = gates.PolygonGate(label, dimensions=[dim_x, dim_y], vertices=vertices)
        raw_gating.add_gate(gate, gate_path=('root', 'Cells'))

        # plots is list of dicts
        # type:
        # ---hist1d: channel_x, source_gate, child_gates
        # ---hist2d: channel_x, channel_y, source_gate, child_gates
        # ---ribbon: source_gate, child_gates
        time_plot = [{'type': 'hist1d', 'channel_x': 'Time', 'source_gate': 'root', 'child_gates': []}]
        morph_plot   = [{'type': 'hist2d', 'channel_x': _morph_x, 'channel_y': _morph_y, 'source_gate': 'root', 'child_gates': ['Cells']}]
        singlet_plot = [{'type': 'hist2d', 'channel_x': _morph_x, 'channel_y': _sing_y,  'source_gate': 'Cells', 'child_gates': ['Singlets']}]
        ribbon_plot = [{'type': 'ribbon', 'source_gate': 'Singlets', 'child_gates': []}]
        fluorescence_plots = [
            {'type': 'hist1d', 'channel_x': self.settings['raw']['event_channels_pnn'][i], 'source_gate': 'Singlets', 'child_gates': []} for i
            in self.settings['raw']['fluorescence_channel_ids']]
        raw_plots = time_plot + morph_plot + singlet_plot + ribbon_plot + fluorescence_plots

        self.cytometry['raw_gating'] = to_gml(raw_gating)
        self.cytometry['raw_plots'] = raw_plots

        ### per-sample gating: ensure template fields exist (backward compatible) ###
        self.cytometry['gating_templates'] = migrate_cytometry_templates(self.cytometry)
        migrate_scoped_gating_templates(self.cytometry)
        self.samples.setdefault('sample_template_assignments', {})

        ### save ###
        self.save()

    def load(self, experiment_path):
        self.experiment_path = str(experiment_path)
        with open(self.experiment_path, "rb") as f:
            file_data = json.load(f)

        self.settings = file_data['settings']
        self.samples = file_data['samples']
        self.process = file_data['process']
        self.samples.setdefault('unstained_samples', [])  # not present in older .kit files
        self.cytometry = file_data['cytometry']
        self.statistics = file_data['statistics']

        # per-sample gating (backward compatible): synthesise a default template
        # from legacy raw_gating/gating when older .kit files lack these fields.
        self.cytometry['gating_templates'] = migrate_cytometry_templates(self.cytometry)
        migrate_scoped_gating_templates(self.cytometry)
        self.samples.setdefault('sample_template_assignments', {})

    def save(self):
        if self.experiment_path is None:
            raise ValueError("No file path set for saving")
        file_data = {
            'settings':self.settings,
            'samples':self.samples,
            'process':self.process,
            'cytometry':self.cytometry,
            'statistics':self.statistics
        }
        json_string = json.dumps(file_data, indent=2)
        # json_string = json_string.replace('\\n', '\n') # to pretty print the gml sections... unfortunately not compatible with json standard

        safe_save(json_string, self.experiment_path)

    def generate_subdirs(self):
        experiment_pl = Path(self.experiment_path)
        experiment_dir = experiment_pl.parent / experiment_pl.stem
        experiment_dir_single_stain_controls = self.settings['raw']['single_stain_controls_subdirectory']
        (experiment_dir / experiment_dir_single_stain_controls).mkdir(parents=True, exist_ok=True)
        experiment_dir_raw_samples = self.settings['raw']['raw_samples_subdirectory']
        (experiment_dir / experiment_dir_raw_samples).mkdir(parents=True, exist_ok=True)
        experiment_dir_unmixed_samples = self.settings['unmixed']['unmixed_samples_subdirectory']
        (experiment_dir / experiment_dir_unmixed_samples).mkdir(parents=True, exist_ok=True)

        return experiment_dir

    def scan_sample_tree(self, sample_name_source_instance=None):
        if sample_name_source_instance is None:
            sample_name_source_instance = sample_name_source

        experiment_dir = self.generate_subdirs()
        experiment_dir_single_stain_controls = self.settings['raw']['single_stain_controls_subdirectory']
        experiment_dir_raw_samples = self.settings['raw']['raw_samples_subdirectory']
        experiment_dir_unmixed_samples = self.settings['unmixed']['unmixed_samples_subdirectory']

        single_stain_controls = [str(p.relative_to(experiment_dir)) for p in
                                 sorted((experiment_dir / experiment_dir_single_stain_controls).glob('**/*.fcs'))]
        raw_samples = [str(p.relative_to(experiment_dir)) for p in
                       sorted((experiment_dir / experiment_dir_raw_samples).glob('**/*.fcs'))]

        # add all single stain controls to raw samples if not already present
        for sample_path in single_stain_controls:
            if sample_path not in raw_samples:
                raw_samples.append(sample_path)

        # load samples one by one, print name, datetime, number of events, file location
        all_sample_nevents = {}
        all_samples = {}
        for sample_path in raw_samples:
            sample_metadata = FlowData(experiment_dir / sample_path, only_text=True, use_header_offsets=True)
            all_sample_nevents[sample_path] = sample_metadata.event_count
            if sample_name_source_instance == 'tubename':
                all_samples[sample_path] = sample_metadata.text['tubename']
            elif sample_name_source_instance == 'fil':
                all_samples[sample_path] = sample_metadata.text['fil']
            else:  # use filenames
                all_samples[sample_path] = Path(sample_path).stem

        for sample_path in single_stain_controls:
            if sample_path not in self.samples['single_stain_controls']:
                self.samples['single_stain_controls'].append(sample_path)

        saved_single_stain_controls = self.samples['single_stain_controls'].copy()
        for sample_path in saved_single_stain_controls:
            if sample_path not in single_stain_controls:
                self.samples['single_stain_controls'].remove(sample_path)

        for sample_path in all_samples:
            if sample_path not in self.samples['all_samples']:
                self.samples['all_samples'][sample_path] = all_samples[sample_path]

        saved_all_samples = self.samples['all_samples'].copy()
        for sample_path in saved_all_samples:
            if sample_path not in all_samples:
                self.samples['all_samples'].pop(sample_path)

        self.samples['all_sample_nevents'] = all_sample_nevents

        import re as _re
        # Auto-register any samples whose name matches 'unstained' (case-insensitive)
        # into unstained_samples, alongside manually-tagged ones.
        tagged = self.samples.setdefault('unstained_samples', [])
        for p, tube_name in self.samples['all_samples'].items():
            if p not in tagged:
                if (_re.search(r'unstained', tube_name, _re.IGNORECASE)
                        or _re.search(r'unstained', p, _re.IGNORECASE)):
                    tagged.append(p)

        # Prune any tagged unstained samples that no longer exist on disk
        self.samples['unstained_samples'] = [
            p for p in self.samples['unstained_samples']
            if p in self.samples['all_samples']
        ]


if __name__ == '__main__':
    '''
    Test two examples:
    1. set up new experiment
    2. reconstitute experiment from sample FCS files
    
    In each:
    create experiment
    populate ephemeral state
    save experiment
    load experiment
    check that loaded experiment is identical to saved experiment
    '''
    import os
    from deepdiff import DeepDiff
    from honeychrome.controller_components.import_fcs_controller import ImportFCSController

    base_directory = Path.home() / 'spectral_cytometry'
    os.chdir(base_directory)

    # 1. create new experiment
    e0 = ExperimentModel()
    experiment_name = base_directory / 'test experiment'
    experiment_path = experiment_name.with_suffix('.kit')
    e0.create(experiment_path)
    e0.save()

    e1 = ExperimentModel()
    e1.load(experiment_path)
    diff = DeepDiff(e0, e1, ignore_order=True, exclude_paths='root.progress_indicator')
    if diff == {}:
        print('Test 1: success')
    else:
        print('Test 1: failure')
        print(diff)

    # 2. create experiment from sample FCS files
    e0 = ExperimentModel()
    experiment_name = base_directory / '20240620 Spectral Symposium-poor cell unmixed'
    experiment_path = experiment_name.with_suffix('.kit')
    e0.create(experiment_path)
    e0.settings['raw']['single_stain_controls_subdirectory'] = 'Raw/Cell controls/Reference Group'
    import_fcs_controller = ImportFCSController(e0)
    import_fcs_controller.reconfigure_experiment_from_fcs_files()
    e0.save()

    e1 = ExperimentModel()
    e1.load(experiment_path)
    diff = DeepDiff(e0, e1, ignore_order=True, exclude_paths='root.progress_indicator')
    if diff == {}:
        print('Test 2: success')
    else:
        print('Test 2: failure')
        print(diff)

    pass
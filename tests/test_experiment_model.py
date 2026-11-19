import os
import pytest
from deepdiff import DeepDiff
from pathlib import Path
from honeychrome.controller_components.import_fcs_controller import ImportFCSController
from honeychrome.experiment_model import ExperimentModel

base_directory = Path.home() / 'spectral_cytometry'
pytestmark = pytest.mark.skipif(not base_directory.is_dir(),
                                reason=f'Example data folder not found at {base_directory}')
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
def test_new_experiment():
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
    assert not diff

def test_create_experiment_from_sample_fcs_files():
    os.chdir(base_directory)
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
    assert not diff
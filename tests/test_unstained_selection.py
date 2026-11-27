"""Tests for controller_components/unstained_selection.py (default unstained cell sample)."""

import numpy as np

from honeychrome.controller_components import unstained_selection as us


def samples():
    return {
        'all_samples': {
            'Raw/Spleen/Mix_Unstained_2.fcs': 'Mix_Unstained_2',
            'Raw/Controls/Unstained (Cells).fcs': 'Unstained (Cells)',
            'Raw/Controls/Unstained (Beads).fcs': 'Unstained (Beads)',
            'Raw/Controls/Blank.fcs': 'Blank',
            'Raw/Controls/CD4 BUV395 (Cells).fcs': 'CD4 BUV395 (Cells)',
        },
        'unstained_samples': ['Raw/Controls/Blank.fcs'],
        'bead_samples': ['Raw/Controls/Unstained (Beads).fcs'],
    }


def row(path, n_events, mismatch):
    return {'path': path, 'name': path, 'n_events': n_events, 'mismatch': mismatch}


def test_candidates_are_named_or_marked_and_not_beads():
    assert us.cell_unstained_candidates(samples()) == [
        'Raw/Spleen/Mix_Unstained_2.fcs', 'Raw/Controls/Unstained (Cells).fcs', 'Raw/Controls/Blank.fcs']


def test_default_is_stored_choice_while_still_a_candidate():
    assert us.default_cell_unstained(samples(), {'default_cell_unstained': 'Unstained (Cells)'}) == 'Unstained (Cells)'
    assert us.default_cell_unstained(samples(), {'default_cell_unstained': 'Removed'}) == 'Mix_Unstained_2'
    assert us.default_cell_unstained(samples(), {}) == 'Mix_Unstained_2'


def test_most_events_among_matched_candidates():
    rows = [row('small', 800, 0.05), row('large', 640000, 0.16), row('mismatched', 900000, 0.6)]
    assert us.choose_cell_unstained(rows) == 'large'


def test_closest_when_none_matched_and_most_events_without_reference():
    assert us.choose_cell_unstained([row('a', 1000, 0.5), row('b', 50, 0.3)]) == 'b'
    assert us.choose_cell_unstained([row('a', 1000, None), row('b', 50, None)]) == 'a'


def test_mismatch_measures_brightness_difference():
    reference = np.full(8, 1000.0)
    assert us.af_mismatch(reference * 2, reference) == 1.0
    assert us.af_mismatch(reference, reference) == 0.0
    events = np.vstack([np.full((75, 8), 5000.0), np.full((25, 8), 100.0)])
    assert np.allclose(us.dim_cells_spectrum(events), 100.0)


def test_unst_names_a_tube_unstained_but_not_its_group_or_unstim():
    assert us.is_unstained_name('D2 Unst spleen 2_Unst Samples', 'Raw/Unst Samples/D2 Unst spleen 2_Unst Samples.fcs')
    assert us.is_unstained_name('Unst Beads SCC', 'Raw/Single stain controls/Unst Beads SCC.fcs')
    assert us.is_unstained_name('A12 Unstained (Cells)_Plate_001')
    assert not us.is_unstained_name('D6 FlowCode Control_Unst Samples',
                                    'Raw/Unst Samples/FC Control/D6 FlowCode Control_Unst Samples.fcs')
    assert not us.is_unstained_name('B1 Unstim CD4_Stim', 'Raw/B1 Unstim CD4_Stim.fcs')
    assert not us.is_unstained_name('Neg ctrl', 'Raw/negative/Neg ctrl.fcs')
    assert us.is_unstained_name('Neg ctrl', 'Raw/negative/Neg ctrl.fcs', negative=True)

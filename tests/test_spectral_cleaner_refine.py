"""
test_spectral_cleaner_refine.py
-------------------------------
SpectralCleaner with refinement on synthetic single-stained cell controls:
cleaning, row refinement and the full-panel residual spillover check run
end to end, with _load_events replaced by synthetic events so no FCS files
are needed. One control is a channel assignment, so the check must use its
stored profile.

Usage:
    pytest tests/test_spectral_cleaner_refine.py
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import honeychrome.settings as hc_settings
from honeychrome.controller_components import spectral_controller as scn
from honeychrome.controller_components.spectral_refinement import AF_TARGET

from spectral_synthetic import panel_spectra, simulate, angle  # noqa: E402

CELL_PANEL = ('BUV496', 'BV421', 'BV510', 'BV605', 'PE', 'PE-Cy5', 'APC', 'APC-Fire 750')
ASSIGNED = 'BB515'


class FakeController:
    """The attributes of controller.Controller that SpectralCleaner reads."""

    def __init__(self, spectra, detectors, labels):
        self.experiment_dir = Path('.')
        self.cleaned_events = {}
        self.raw_gating = None
        self.filtered_raw_fluorescence_channel_ids = list(range(len(detectors)))
        model = [{
            'label': label, 'control_type': 'Single Stained Spectral Control',
            'particle_type': 'Cells', 'gate_channel': detectors[int(np.argmax(row))],
            'sample_name': label, 'gate_label': f'Positive {label}',
            'universal_negative_name': 'Unstained',
        } for label, row in zip(labels, spectra)]
        model.append({'label': ASSIGNED, 'control_type': 'Channel Assignment', 'particle_type': None,
                      'gate_channel': None, 'sample_name': None, 'gate_label': None})
        self.experiment = SimpleNamespace(
            samples={'all_samples': {}},
            process={'spectral_model': model, 'profiles': {}, 'negative_type': 'unstained'},
            settings={'raw': {'event_channels_pnn': list(detectors), 'magnitude_ceiling': 4e6,
                              'scatter_channel_ids': [], 'cytometer_db_col': None}},
        )

    def filter_raw_fluorescence_channels(self):
        pass


@pytest.fixture(scope='module')
def cleaned(request):
    labels = CELL_PANEL
    spectra, detectors = panel_spectra(labels + (ASSIGNED,))
    controller = FakeController(spectra[:-1], detectors, labels)
    controller.experiment.process['profiles'] = {ASSIGNED: spectra[-1].tolist()}

    rng = np.random.default_rng(3)
    unstained = simulate(None, 30_000, rng)
    events = {label: simulate(row, 30_000, rng) for label, row in zip(labels, spectra)}

    def load_events(self, control):
        label = control['label']
        return {
            'pos_events': events[label],
            'pos_scatter': rng.normal(8e4, 1e4, (len(events[label]), 2)),
            'neg_events': unstained,
            'neg_scatter': rng.normal(8e4, 1e4, (len(unstained), 2)),
            'use_internal': False,
            'negative_name': 'Unstained',
            'n_removed_saturation': 0,
            'expected_peak_ch_idx': int(np.argmax(spectra[labels.index(label)])),
        }

    patch = pytest.MonkeyPatch()
    patch.setattr(hc_settings, 'spectral_cleaning_refine_retrieved', True, raising=False)
    patch.setattr(scn.SpectralCleaner, '_load_events', load_events)
    request.addfinalizer(patch.undo)

    cleaner = scn.SpectralCleaner(None, controller)
    cleaner.run()
    return {'cleaner': cleaner, 'controller': controller, 'spectra': spectra, 'labels': labels}


def test_every_cell_control_refined_and_checked(cleaned):
    store = cleaned['controller'].cleaned_events
    labels = cleaned['labels']
    assert set(store) == set(labels)
    for label in labels:
        entry = store[label]
        assert entry['refine_log'], label
        targets = [r['target'] for r in entry['crosstalk']]
        assert targets == [t for t in labels + (ASSIGNED,) if t != label] + [AF_TARGET]
        assert entry['_fingerprint']['refine'] is True
    assert cleaned['cleaner']._check_pools == {}


def test_refined_rows_no_worse_than_cleaned(cleaned):
    store = cleaned['controller'].cleaned_events
    for label, true in zip(cleaned['labels'], cleaned['spectra']):
        entry = store[label]
        assert angle(entry['spectrum'], true) <= angle(entry['spectrum_initial'], true) + 0.1, label


def test_crosstalk_not_increased_by_refinement(cleaned):
    store = cleaned['controller'].cleaned_events
    for label in cleaned['labels']:
        rows = store[label]['crosstalk']
        worst_before = max(abs(r['slope_before']) for r in rows)
        worst_after = max(abs(r['slope_after']) for r in rows)
        assert worst_after <= worst_before + 0.005, label


def test_refinement_results_are_json_serialisable(cleaned):
    store = cleaned['controller'].cleaned_events
    for entry in store.values():
        json.dumps({k: entry[k] for k in ('spectrum', 'spectrum_initial', 'refine_log', 'crosstalk')},
                   allow_nan=False)

"""
af_synthetic.py
---------------
Synthetic spectral experiment shared by the AF export, AutoSpectral
Optimization and 3D-plot tests: Gaussian fluorophore spectra, a small AF
library, raw FCS files written with write_fcs(), and a stand-in controller
exposing only the attributes the exporters and plugins read.

Channel layout mirrors calculate_spectral_process(): Time, event_id,
scatter, fluorophores, then AF Abundance and AF Index. The experiment
stores two AF profiles; the AF sample is assigned the second, so its
experiment-wide AF Index values start after the first profile's.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import honeychrome.settings as hc_settings
from honeychrome.controller_components import functions as fn

N_DET = 12
N_FLUOR = 3
N_AF = 4
N_OTHER_AF = 3                  # spectra in a second, unassigned profile stored first
N_AF_TOTAL = N_OTHER_AF + N_AF  # experiment-wide AF Index range
MAGNITUDE_CEILING = 262144.0

DET_PNN = [f'D{i + 1}-A' for i in range(N_DET)]
RAW_PNN = ['Time', 'FSC-A', 'SSC-A'] + DET_PNN
FLUOR_LABELS = [f'F{i + 1}' for i in range(N_FLUOR)]
UNMIXED_PNN = ['Time', 'event_id', 'FSC-A', 'SSC-A'] + FLUOR_LABELS + list(hc_settings.af_channels)

AF_SAMPLE = 'Raw/af_sample.fcs'
PLAIN_SAMPLE = 'Raw/plain_sample.fcs'


def fluor_spectra():
    """(N_FLUOR, N_DET) L-inf normalised, well separated."""
    ch = np.arange(N_DET)
    s = np.array([np.exp(-0.5 * ((ch - c) / 1.5) ** 2) for c in (2, 6, 10)])
    return s / s.max(axis=1, keepdims=True)


def af_spectra(seed=0):
    """(N_AF, N_DET) broad AF variants that differ in shape."""
    ch = np.arange(N_DET)
    rows = [np.exp(-0.5 * ((ch - centre) / width) ** 2)
            for centre, width in ((2, 4.0), (4, 5.0), (6, 3.5), (3, 6.5))]
    s = np.array(rows)
    return s / s.max(axis=1, keepdims=True)


def other_af_spectra():
    """(N_OTHER_AF, N_DET) spectra of a profile no sample here is assigned."""
    ch = np.arange(N_DET)
    s = np.array([np.exp(-0.5 * ((ch - centre) / 4.0) ** 2) for centre in (5, 7, 9)])
    return s / s.max(axis=1, keepdims=True)


def near_in_span_af_library():
    """af_spectra() plus a variant lying almost inside the fluorophore span,
    whose out-of-span residual is far below 1% of the others'."""
    fl, af = fluor_spectra(), af_spectra()
    near = 0.7 * fl[0] + 0.3 * fl[1] + 0.003 * af[1]
    return np.vstack([af, near / near.max()])


def raw_events(n, seed=0, with_af=True):
    """(n, len(RAW_PNN)) events: time, scatter, fluorophores + per-cell AF."""
    rng = np.random.default_rng(seed)
    fl = fluor_spectra()
    af = af_spectra()
    abund = rng.gamma(0.8, 2000.0, (n, N_FLUOR)) * (rng.random((n, N_FLUOR)) < 0.5)
    det = abund @ fl + rng.normal(0.0, 20.0, (n, N_DET))
    if with_af:
        det += rng.gamma(2.0, 400.0, n)[:, None] * af[rng.integers(0, N_AF, n)]
    time = np.sort(rng.uniform(0, 100, n))
    scatter = rng.normal(8e4, 1e4, (n, 2))
    return np.column_stack([time, scatter, det])


def write_raw_fcs(path, events, cytometer='Synthetic'):
    """Minimal raw FCS file for RAW_PNN, written with Honeychrome's own writer."""
    keywords = {'$CYT': cytometer}
    for i, name in enumerate(RAW_PNN, start=1):
        keywords[f'$P{i}N'] = name
        keywords[f'$P{i}B'] = '32'
        keywords[f'$P{i}E'] = '0,0'
        keywords[f'$P{i}R'] = '100' if name == 'Time' else str(int(MAGNITUDE_CEILING))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fn.write_fcs(np.asarray(events, dtype=np.float64), keywords, path)


def raw_settings():
    return {
        'raw_samples_subdirectory': 'Raw',
        'event_channels_pnn': list(RAW_PNN),
        'whitelisted_pnn': list(RAW_PNN),
        'scatter_channel_ids': [1, 2],
        'time_channel_id': 0,
        'event_id_channel_id': None,
        'fluorescence_channel_ids': list(range(3, 3 + N_DET)),
        'cytometer': 'Synthetic',
        'magnitude_ceiling': MAGNITUDE_CEILING,
    }


def unmixed_settings():
    return {
        'unmixed_samples_subdirectory': 'Unmixed',
        'event_channels_pnn': list(UNMIXED_PNN),
        'time_channel_id': 0,
        'event_id_channel_id': 1,
        'scatter_channel_ids': [2, 3],
        'n_scatter_channels': 2,
        'fluorescence_channel_ids': list(range(4, 4 + N_FLUOR)),
        'n_fluorophore_channels': N_FLUOR,
        'af_channel_ids': [UNMIXED_PNN.index(c) for c in hc_settings.af_channels],
        'fluorescence_channels': list(DET_PNN),
        'magnitude_ceiling': MAGNITUDE_CEILING,
        'width_ceiling': 1024.0,
        'default_ceiling': 100.0,
        'width_channels': [],
    }


class FakeController:
    """The attributes of controller.Controller read by UnmixedExporter,
    AutoSpectralOptExporter and the 3D plot tiles."""

    def __init__(self, experiment_dir, spillover=None):
        fl = fluor_spectra()
        af = af_spectra()
        unmixing_matrix = np.linalg.solve(fl @ fl.T, fl)
        spill = np.eye(N_FLUOR) if spillover is None else np.asarray(spillover, dtype=float)
        self.experiment_dir = Path(experiment_dir)
        self.filtered_raw_fluorescence_channel_ids = list(range(3, 3 + N_DET))
        self.autospectral_variants = {}
        self.autospectral_unmixed_pos_thresholds = None
        self.raw_gating = None
        self.current_sample_path = AF_SAMPLE
        self.live_sample_path = None
        self.experiment = SimpleNamespace(
            settings={'raw': raw_settings(), 'unmixed': unmixed_settings(), 'unmixing_method': 'OLS'},
            samples={
                'all_samples': {AF_SAMPLE: 'af_sample', PLAIN_SAMPLE: 'plain_sample'},
                'sample_af_profiles': {AF_SAMPLE: ['AF profile']},
            },
            process={
                'unmixing_matrix': unmixing_matrix.tolist(),
                'spillover': spill.tolist(),
                'unmixing_weights': None,
                'af_profiles': {
                    'Other profile': {'spectra': other_af_spectra().tolist()},
                    'AF profile': {'spectra': af.tolist()},
                },
                'spectral_model': [{'label': lbl, 'antigen': f'CD{i + 1}'}
                                   for i, lbl in enumerate(FLUOR_LABELS)],
                'profiles': {lbl: fl[i].tolist() for i, lbl in enumerate(FLUOR_LABELS)},
                'spectra_matrix': fl.tolist(),
                'autospectral_variants_meta': {},
            },
        )

    def _build_fluor_spectra(self):
        return fluor_spectra()

    def get_af_index_map_for_sample(self, sample_path):
        from honeychrome.controller_components.autospectral_functions import af_index_lookup
        names = self.experiment.samples['sample_af_profiles'].get(sample_path, [])
        return af_index_lookup(self.experiment.process['af_profiles'], names)

    def n_af_spectra(self):
        profiles = self.experiment.process.get('af_profiles') or {}
        n = sum(len(p.get('spectra') or []) for p in profiles.values())
        return n or None


def make_experiment(experiment_dir, n_events=3000, spillover=None):
    """Write the two raw samples and return a FakeController for them."""
    experiment_dir = Path(experiment_dir)
    write_raw_fcs(experiment_dir / AF_SAMPLE, raw_events(n_events, seed=1))
    write_raw_fcs(experiment_dir / PLAIN_SAMPLE, raw_events(n_events, seed=2))
    return FakeController(experiment_dir, spillover=spillover)


def read_fcs(path):
    """(pnn, events float64, text keywords) via Honeychrome's FCS loader."""
    sample = fn.sample_from_fcs(path)
    return list(sample.pnn_labels), np.asarray(sample.get_events(source='raw'), dtype=np.float64), sample.get_metadata()


def exported_path(experiment_dir, sample_key):
    name = Path(sample_key).stem
    return Path(experiment_dir) / 'Unmixed' / f'{name} (Unmixed).fcs'

"""
envelope_builder.py
-------------------
Negative envelopes for stored AF profiles that have none they can keep
current (``autospectral_functions.envelope_needs_source``): profiles
extracted before envelopes existed, or whose unstained subsample is missing.
The envelope is built from the profile's source sample exactly as AF
extraction builds it (all events of the sample, ``make_envelope_entry``),
so the AF spectra, and with them the experiment's AF Index numbering, are
unchanged. The controller runs ``EnvelopeBuildWorker`` in a QThread after
the AF cache is rebuilt and stores the results on the main thread.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal

from honeychrome.controller_components.autospectral_functions import envelope_sample_name, make_envelope_entry

logger = logging.getLogger(__name__)


def build_envelope(experiment_dir, profile_name: str, profile: dict, layout: dict, fluor_names: list,
                   fluor_spectra: np.ndarray, loader=None) -> tuple:
    """
    ``(envelope entry, unstained subsample)`` of one profile from all events
    of its source sample (``profile['source_fcs']``). Raises when the sample
    cannot be read or its detectors do not match the profile.
    """
    from honeychrome.controller_components.spread_model import load_events

    loader = loader or load_events
    source = profile.get('source_fcs')
    if not source or not (Path(experiment_dir) / source).is_file():
        raise FileNotFoundError(f'source sample {source!r} not found')
    raw_fl, _sc = loader(experiment_dir, source, None, layout, None)
    af_spectra = np.asarray(profile['spectra'], dtype=np.float64)
    if raw_fl.shape[1] != af_spectra.shape[1]:
        raise ValueError(f'{raw_fl.shape[1]} detectors in {source}, {af_spectra.shape[1]} in the profile')
    if len(raw_fl) == 0:
        raise ValueError(f'{source} has no events')
    return make_envelope_entry(raw_fl, fluor_names, fluor_spectra, af_spectra, envelope_sample_name(profile_name))


def build_envelopes(job: dict, progress, loader=None) -> tuple:
    """
    Every profile of ``job['profiles']`` ({name: profile entry}); returns
    ``(built, failed)``: {name: (entry, sample)} and {name: reason}.
    """
    built, failed = {}, {}
    names = list(job['profiles'])
    for i, name in enumerate(names, start=1):
        progress(f'Building the negative envelope of {name} ({i}/{len(names)})...')
        try:
            built[name] = build_envelope(job['experiment_dir'], name, job['profiles'][name], job['layout'],
                                         job['fluor_names'], job['fluor_spectra'], loader)
        except Exception as e:
            logger.warning('AF envelope: %s not built (%s)', name, e)
            failed[name] = str(e) or type(e).__name__
    return built, failed


class EnvelopeBuildWorker(QObject):
    """Runs ``build_envelopes`` in a QThread."""
    finished = Signal(object, object, object)   # job, built, failed
    progress = Signal(str)

    def __init__(self, job: dict):
        super().__init__()
        self.job = job

    def run(self):
        built, failed = build_envelopes(self.job, self.progress.emit)
        self.finished.emit(self.job, built, failed)

import json
import re
import warnings

import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer
# from PySide6.QtWidgets import QApplication
from flowkit import Dimension, gates

from honeychrome.controller_components.functions import timer, sample_from_fcs
from honeychrome.controller_components.spectral_functions import get_profile, get_raw_events
from honeychrome.controller_components.label_matching import match_fluorophore, match_marker, get_fluorophore_db, get_marker_db
from honeychrome.controller_components.spectral_librarian import SpectralLibrary
from honeychrome.controller_components.spectral_cleaning import find_empirical_peak, cosine_filter, knn_scatter_match, exclude_saturated, CleanResult
from honeychrome.controller_components.functions import sample_from_fcs
from honeychrome.experiment_model import check_fcs_matches_experiment
from honeychrome.view_components.busy_cursor import with_busy_cursor
import honeychrome.settings as settings
from honeychrome.settings import INTERNAL_NEGATIVE_SENTINEL        

import logging
logger = logging.getLogger(__name__)

# connect to spectral library
spectral_library = SpectralLibrary()

def _find_default_unstained_tube(all_samples: dict) -> str | None:
    """Return tube name of the first sample whose path or name contains 'unstained'."""
    for path, name in all_samples.items():
        if re.search(r'unstained', path, re.IGNORECASE) or re.search(r'unstained', name, re.IGNORECASE):
            return name
    return None

class ProfileUpdater:
    def __init__(self, controller, bus):
        self.controller = controller
        self.bus = bus
        self.experiment_dir = controller.experiment_dir
        self.samples = controller.experiment.samples
        self.spectral_model = controller.experiment.process['spectral_model']
        self.profiles = controller.experiment.process['profiles']
        self.event_channels_pnn = controller.experiment.settings['raw']['event_channels_pnn']
        self.raw_gating = controller.raw_gating
        self.n_fluorophore_channels = None
        self.fluorescence_channels_pnn = []
        self.unstained_negative = None          # legacy mean-vector cache — kept for existing generate() path
        self._negative_events_cache: dict[str, np.ndarray] = {}
        self.refresh()

    def refresh(self):
        self.controller.filter_raw_fluorescence_channels()
        self.n_fluorophore_channels = len(self.controller.filtered_raw_fluorescence_channel_ids)
        self.fluorescence_channels_pnn.clear()
        self.fluorescence_channels_pnn.extend([self.event_channels_pnn[i] for i in self.controller.filtered_raw_fluorescence_channel_ids])

    def flush(self):
        self.unstained_negative = None
        self._negative_events_cache.clear()
        self.refresh()
        controls = [control['label'] for control in self.spectral_model]
        labels = list(self.profiles.keys())
        for label in labels:
            if label not in controls:
                self.profiles.pop(label)

        # n_fluorophore_channels_changed = False
        # labels = list(self.profiles.keys())
        # for label in labels:
        #     if len(self.profiles[label]) != self.n_fluorophore_channels:
        #         self.profiles.pop(label)
        #         n_fluorophore_channels_changed = True
        #
        # if n_fluorophore_channels_changed:
        #     warnings.warn('Selected number of fluorophores has changed. Previous controls have been flushed.')
        #     if self.bus:
        #         self.bus.warningMessage.emit('Selected number of fluorophores has changed. Previous controls have been flushed.')

    def get_unstained_negative(self):
        try:
            sample_path = None
            for path in self.samples.get('unstained_samples', []):
                if path in self.samples['all_samples']:
                    sample_path = path
                    break
            if sample_path is None:
                for path in self.samples['all_samples']:
                    if re.search(r'(unstained|negative)', path, re.IGNORECASE) or \
                    re.search(r'(unstained|negative)', self.samples['all_samples'][path], re.IGNORECASE):
                        sample_path = path
                        break
            if sample_path:
                full_sample_path = str(self.experiment_dir / sample_path)
                sample = sample_from_fcs(full_sample_path)

                positive_gate_label = 'Pos Unstained'
                negative_gate_label = 'Neg Unstained'
                # Use the persistent cytometry store, not the ephemeral runtime view —
                # data_for_cytometry_plots_raw['plots'] may be None if no sample is loaded.
                raw_plots = self.controller.experiment.cytometry['raw_plots']

                for target_gate_label in [positive_gate_label, negative_gate_label]:
                    if not self.raw_gating.find_matching_gate_paths(target_gate_label):
                        # Use the actual scatter channels from the morph plot, not hardcoded FSC-A/SSC-A
                        morph_plot = next(
                            (p for p in raw_plots if p['type'] == 'hist2d' and p.get('source_gate') == 'root'
                             and p['channel_x'] in self.controller.experiment.settings['raw']['event_channels_pnn']
                             and p['channel_y'] in self.controller.experiment.settings['raw']['event_channels_pnn']),
                            None
                        )
                        _sp = self.controller.experiment.settings['raw'].get('scatter_param', ['FSC-A', 'SSC-A'])
                        channel_x = morph_plot['channel_x'] if morph_plot else _sp[0]
                        channel_y = morph_plot['channel_y'] if morph_plot else _sp[1]
                        dim_x = Dimension(channel_x, range_min=0.2, range_max=0.8, transformation_ref=channel_x)
                        dim_y = Dimension(channel_y, range_min=(0.1 if target_gate_label == positive_gate_label else 0.3),
                                                    range_max=(0.7 if target_gate_label == positive_gate_label else 0.9),
                                                    transformation_ref=channel_y)
                        target_gate = gates.RectangleGate(target_gate_label, dimensions=[dim_x, dim_y])
                        base_gate_priority = self.controller.experiment.process.get('base_gate_priority_order', [])
                        raw_gate_names = [g[0].lower() for g in self.controller.raw_gating.get_gate_ids()]
                        base_gate_label = 'root'
                        for gate in base_gate_priority:
                            if gate.lower() in raw_gate_names:
                                base_gate_label = gate
                                break
                        if base_gate_label != 'root' and self.raw_gating.find_matching_gate_paths(base_gate_label):
                            base_path = tuple(list(self.raw_gating.find_matching_gate_paths(base_gate_label)[0]) + [base_gate_label])
                        else:
                            base_path = ('root',)
                        self.raw_gating.add_gate(target_gate, gate_path=base_path)

                        target_plot = None
                        for plot in raw_plots:
                            if (plot['type'] == 'hist2d'
                                    and plot['channel_x'] == channel_x
                                    and plot['channel_y'] == channel_y
                                    and plot['source_gate'] == 'root'
                                    and not set(plot['child_gates']) - {positive_gate_label, negative_gate_label}):
                                target_plot = plot
                        if not target_plot:
                            target_plot = {'type': 'hist2d', 'channel_x': channel_x, 'channel_y': channel_y,
                                        'source_gate': 'root', 'child_gates': [positive_gate_label, negative_gate_label]}
                            raw_plots.append(target_plot)
                            if self.bus:
                                self.bus.showNewPlot.emit('raw')
                        if target_gate_label not in target_plot['child_gates']:
                            target_plot['child_gates'].append(target_gate_label)

                        if self.bus:
                            self.bus.changedGatingHierarchy.emit('raw', target_gate_label)

                _col_order = self.controller.experiment.settings['raw'].get('whitelisted_pnn')
                _ecpnn = self.controller.experiment.settings['raw']['event_channels_pnn']
                try:
                    _all_events = sample.get_events('raw', col_order=_col_order)
                except (KeyError, ValueError):
                    _all_events = sample.get_events('raw')
                    _col_order = None
                _pnn = _col_order if _col_order is not None else sample.pnn_labels
                self.unstained_negative = get_profile(
                    sample, negative_gate_label, self.raw_gating,
                    self.controller.filtered_raw_fluorescence_channel_ids,
                    preloaded_events=_all_events, preloaded_pnn=_pnn,
                    event_channels_pnn=_ecpnn,
                )
                return True
            else:
                raise Exception(
                    'No unstained sample found. Name a control "Unstained", or right-click '
                    'any sample in the sample panel and choose "Mark as Unstained".'
                )
        except Exception as e:
            text = f'Failed to generate profile of unstained negative. {e}.'
            warnings.warn(text)
            if self.bus:
                self.bus.warningMessage.emit(text)
            return False


    def _get_negative_events(self, control) -> np.ndarray | None:
        """
        Resolve and return raw fluorescence event array for the negative of *control*.

        Priority order:
          1. control["universal_negative_name"] — per-control assignment
          2. Global unstained fallback (negative_type == "unstained")
          3. None — caller falls back to existing gate-mean path

        Results are cached by tube name so each FCS file is loaded at most once.
        """
        tube_name = control.get('universal_negative_name') or ''

        # Explicit per-control opt-out: use the internal (same-sample) negative
        if tube_name == INTERNAL_NEGATIVE_SENTINEL:
            return None

        # Fallback to global unstained if no per-control assignment
        if not tube_name:
            if self.controller.experiment.process.get('negative_type') == 'unstained':
                tube_name = _find_default_unstained_tube(self.samples['all_samples']) or ''
            if not tube_name:
                return None

        if tube_name in self._negative_events_cache:
            return self._negative_events_cache[tube_name]

        # Resolve path from tube name
        all_samples_reverse = {v: k for k, v in self.samples['all_samples'].items()}
        rel_path = all_samples_reverse.get(tube_name)
        if rel_path is None:
            warnings.warn(f'_get_negative_events: tube "{tube_name}" not found in samples.')
            return None

        full_path = str(self.experiment_dir / rel_path)
        try:
            sample = sample_from_fcs(full_path)
            gate_label = 'Neg Unstained' if self.raw_gating.find_matching_gate_paths('Neg Unstained') else None
            events = get_raw_events(sample, self.controller.filtered_raw_fluorescence_channel_ids,
                                    gate_label=gate_label, gating_strategy=self.raw_gating)
            self._negative_events_cache[tube_name] = events
            return events
        except Exception as e:
            warnings.warn(f'_get_negative_events: failed to load "{tube_name}": {e}')
            return None
    

    def generate(self, control, search_results):
        if control['control_type'] == 'Single Stained Spectral Control':
            try:
                if control['sample_name'] and control['gate_label']:
                    all_samples_reverse_lookup = {v: k for k, v in self.samples['all_samples'].items()}
                    tubename = control['sample_name']
                    sample_path = all_samples_reverse_lookup[tubename]
                    nevents = self.samples['all_sample_nevents'][sample_path]
                    if nevents > 0:
                        full_sample_path = str(self.experiment_dir / sample_path)
                        control['sample_path'] = full_sample_path

                        positive_gate_label = control['gate_label']
                        
                        # use cleaned event pool when available and opted in
                        cleaned_store = self.controller.cleaned_events
                        cleaned = cleaned_store.get(control['label']) if control.get('use_cleaned') is not False else None
                        if cleaned is not None and cleaned.get('spectrum'):
                            profile = np.array(cleaned['spectrum'], dtype=float)
                            if profile.max() <= 0:
                                raise Exception(f'Cleaned spectrum for "{control["label"]}" is zero — re-run Clean Controls.')
                            profile = profile / profile.max()
                            if not control.get('gate_channel_locked'):
                                control['gate_channel'] = self.fluorescence_channels_pnn[int(np.argmax(profile))]
                            self.profiles[control['label']] = profile.tolist()
                            profile_dict = dict(zip(self.fluorescence_channels_pnn, profile.tolist()))
                            spectral_library.deposit_control_with_profile_and_experiment_dir(
                                control, profile_dict, str(self.experiment_dir))
                            return True

                        # FCS load only needed for non-cleaned path
                        if not self.raw_gating.find_matching_gate_paths(positive_gate_label):
                            raise Exception(f'Positive gate label {positive_gate_label} not present in Raw Data. ')
                        sample = sample_from_fcs(full_sample_path)
                        _col_order = self.controller.experiment.settings['raw'].get('whitelisted_pnn')
                        _ecpnn = self.controller.experiment.settings['raw']['event_channels_pnn']
                        try:
                            _all_events = sample.get_events('raw', col_order=_col_order)
                        except (KeyError, ValueError):
                            _all_events = sample.get_events('raw')
                            _col_order = None
                        _pnn = _col_order if _col_order is not None else sample.pnn_labels

                        positive_profile = get_profile(
                            sample, positive_gate_label, self.raw_gating,
                            self.controller.filtered_raw_fluorescence_channel_ids,
                            preloaded_events=_all_events, preloaded_pnn=_pnn,
                            event_channels_pnn=_ecpnn,
                        )

                        universal_neg_name = control.get('universal_negative_name') or ''
                        explicit_internal = (universal_neg_name == INTERNAL_NEGATIVE_SENTINEL)
                        use_unstained = (
                            self.controller.experiment.process['negative_type'] == 'unstained'
                            and not explicit_internal
                        )

                        if use_unstained:
                            # Resolve the specific unstained FCS for this control:
                            # (1) per-control universal_negative_name, (2) global Unstained sample.
                            # If neither is available, fall back to internal negative with a warning.
                            resolved_unstained = None

                            if universal_neg_name and not explicit_internal:
                                # Per-control assignment — load it directly.
                                # Ensure Neg Unstained gate exists first (get_unstained_negative
                                # creates it if absent; it is idempotent if already present).
                                if self.unstained_negative is None:
                                    self.get_unstained_negative()
                                all_samples_rev = {v: k for k, v in self.samples['all_samples'].items()}
                                neg_rel_path = all_samples_rev.get(universal_neg_name)
                                if neg_rel_path:
                                    neg_full_path = str(self.experiment_dir / neg_rel_path)
                                    neg_sample = sample_from_fcs(neg_full_path)
                                    neg_gate_label = 'Neg Unstained'
                                    if self.raw_gating.find_matching_gate_paths(neg_gate_label):
                                        # neg_sample is a different file — load its own event block.
                                        try:
                                            _neg_events = neg_sample.get_events('raw', col_order=_col_order)
                                        except (KeyError, ValueError):
                                            _neg_events = neg_sample.get_events('raw')
                                        _neg_pnn = _col_order if _col_order is not None else neg_sample.pnn_labels
                                        resolved_unstained = get_profile(
                                            neg_sample, neg_gate_label, self.raw_gating,
                                            self.controller.filtered_raw_fluorescence_channel_ids,
                                            preloaded_events=_neg_events, preloaded_pnn=_neg_pnn,
                                            event_channels_pnn=_ecpnn,
                                        )
                                    else:
                                        warnings.warn(f'{control["label"]}: unstained negative "{universal_neg_name}" has no "Neg Unstained" gate — using internal negative.')
                                else:
                                    warnings.warn(f'{control["label"]}: unstained negative "{universal_neg_name}" not found in samples — using internal negative.')
                            else:
                                # Global unstained fallback
                                if self.unstained_negative is None:
                                    self.get_unstained_negative()
                                resolved_unstained = self.unstained_negative

                            if resolved_unstained is not None:
                                negative_profile = resolved_unstained
                            else:
                                # No unstained available for this control — fall back gracefully
                                warnings.warn(f'{control["label"]}: no unstained negative available — using internal negative.')
                                if self.bus:
                                    self.bus.warningMessage.emit(f'{control["label"]}: no unstained negative available, using internal negative.')
                                use_unstained = False  # drop through to internal path below

                        if not use_unstained:
                            if 'unstained' in control['label'].lower():
                                negative_gate_label = positive_gate_label
                            else:
                                # Prefer an explicitly set neg_gate_label; fall back to
                                # the auto-constructed name for backwards compatibility
                                # with experiments generated before this field existed.
                                negative_gate_label = (
                                    control.get('neg_gate_label')
                                    or f'Neg {control["label"]}'
                                )

                            if not self.raw_gating.find_matching_gate_paths(negative_gate_label):
                                raise Exception(f'Internal negative gate "{negative_gate_label}" not present in Raw Data. '
                                                f'Please run Auto-Generate to recreate spectral gates, '
                                                f'or select the correct Negative Gate in the Spectral Model Editor.')

                            negative_profile = get_profile(
                                sample, negative_gate_label, self.raw_gating,
                                self.controller.filtered_raw_fluorescence_channel_ids,
                                preloaded_events=_all_events, preloaded_pnn=_pnn,
                                event_channels_pnn=_ecpnn,
                            )

                        profile = positive_profile - negative_profile
                        if profile.sum() == 0:
                            profile = positive_profile

                        profile_warning = None
                        if profile.sum() > 0:
                            profile = profile / profile.max()  # max normalisation
                        else:
                            profile_warning = (
                                f'"{control["label"]}" has a negative profile — the negative '
                                f'gate has higher mean fluorescence than the positive. '
                                f'Run Control Cleaning to fix this.'
                            )
                            warnings.warn(profile_warning)
                            self.negative_profile_warnings.append(control['label'])
                            abs_max = np.abs(profile).max()
                            profile = np.abs(profile) / abs_max if abs_max > 0 else positive_profile

                        control['profile_warning'] = profile_warning
                        if not control.get('gate_channel_locked'):
                            # Orthogonalised peak search — no col_order so indices into full channel list are valid.
                            pos_events_raw = get_raw_events(
                                sample, self.controller.filtered_raw_fluorescence_channel_ids,
                                gate_label=positive_gate_label, gating_strategy=self.raw_gating,
                            )
                            neg_events_raw = self._get_negative_events(control)
                            if (pos_events_raw is not None and len(pos_events_raw) > 1
                                    and neg_events_raw is not None and len(neg_events_raw) > 1):
                                af_mean = neg_events_raw.mean(axis=0)
                                peak_idx = find_empirical_peak(pos_events_raw, af_mean)
                            else:
                                peak_idx = int(np.argmax(profile))
                            control['gate_channel'] = self.fluorescence_channels_pnn[peak_idx]
                        profile = profile.tolist()
                        self.profiles[control['label']] = profile
                        profile_dict = dict(zip(self.fluorescence_channels_pnn, profile))
                        spectral_library.deposit_control_with_profile_and_experiment_dir(control, profile_dict, str(self.experiment_dir))
                        return True

            except Exception as e:
                text = f'Failed to generate profile. {e}'
                warnings.warn(text)
                if self.bus:
                    self.bus.warningMessage.emit(text)
                return False

        elif control['control_type'] == 'Single Stained Spectral Control from Library':
            if control['sample_name'] and search_results:
                for n in search_results:
                    if control['sample_name'] == search_results[n]['current_control_list']:
                        profile = list(json.loads(search_results[n]['profile_dict']).values())
                        self.profiles[control['label']] = profile
                        return True

        elif control['control_type'] == 'Channel Assignment':
            if control['gate_channel']:
                n = self.fluorescence_channels_pnn.index(control['gate_channel'])
                profile = np.zeros(self.n_fluorophore_channels)
                profile[n] = 1
                self.profiles[control['label']] = profile.tolist()
                return True

        # remove control from profiles dict if not set correctly
        self.profiles[control['label']] = None
        self.profiles.pop(control['label'])
        return False

    def pop_control(self, label):
        self.profiles[label] = None
        self.profiles.pop(label)

class SpectralAutoGenerator(QObject):
    def __init__(self, bus, controller, preferred_unstained: str | None = None):
        super().__init__()

        # connect
        self.controller = controller
        self.bus = bus

        self.event_channels_pnn = self.controller.experiment.settings['raw']['event_channels_pnn']

        self.controller.filter_raw_fluorescence_channels()
        self.fluorescence_channel_ids = self.controller.filtered_raw_fluorescence_channel_ids
        self.fluorescence_channels_pnn = [self.event_channels_pnn[i] for i in self.fluorescence_channel_ids]

        self.raw_gating = self.controller.raw_gating
        self.raw_plots = self.controller.data_for_cytometry_plots_raw['plots']
        self.spectral_model = self.controller.experiment.process['spectral_model']
        self.profiles = self.controller.experiment.process['profiles']
        self.samples = self.controller.experiment.samples
        self.experiment_dir = self.controller.experiment_dir
        self.preferred_unstained = preferred_unstained  # tube name hint from editor

        self.spectral_model.clear()
        self.profiles.clear()
        self.unstained_negative = None
        self.unstained_negative_events = None          # cells unstained raw events
        self.unstained_negative_events_beads = None    # beads unstained raw events
        self.negative_profile_warnings = []            # labels; shown once at end of run()
        self.insufficient_events_warnings = []         # tubenames; shown once at end of run()

        self.progress_target = len(self.samples['single_stain_controls'])
        # self.progress_target = len(self.samples['single_stain_controls'][:5]) # quick test

        # (sample name, label, sample path, particle_type (cells/beads), control_type (positive only, positive and negative, autofluorescence), gate channel
        self.base_gate_label = 'root'
        raw_gate_names = [g[0].lower() for g in self.controller.raw_gating.get_gate_ids()]
        for gate in self.controller.experiment.process['base_gate_priority_order']:
            if gate.lower() in raw_gate_names:
                self.base_gate_label = gate
                break

    @with_busy_cursor
    def run(self):
        self.get_unstained_negative(particle_type='Cells')
        self.get_unstained_negative(particle_type='Beads')
        if self.unstained_negative is None:
            if self.bus:
                QTimer.singleShot(100, lambda: self.bus.statusMessage.emit(
                    'No unstained sample found — using internal negatives for profile extraction.'))
        for n in range(self.progress_target):
            if self.bus:
                self.bus.progress.emit(n, self.progress_target)
            success = self.generate_spectral_control(n)
            if not success:
                if self.bus:
                    QTimer.singleShot(100, lambda: self.bus.statusMessage.emit(
                        f'Failed to generate profile of spectral control {n}.'))
                continue
            if self.bus:
                self.bus.spectralControlAdded.emit()

        for index, plot in enumerate(self.raw_plots):
            if self.bus:
                self.bus.updateRois.emit('raw', index)

        logger.info('SpectralAutoGenerator: regenerated spectral model and raw gating hierarchy:')
        logger.info(self.raw_gating.get_gate_hierarchy(output='json'))

        if self.bus and self.negative_profile_warnings:
            labels = ', '.join(f'"{label}"' for label in self.negative_profile_warnings)
            combined_warning = (
                f'The following control(s) have a negative profile — the negative gate '
                f'has higher mean fluorescence than the positive: {labels}.\n\n'
                f'These controls have been added but will not unmix correctly. '
                f'Run Control Cleaning to fix this.'
            )
            self.bus.warningMessage.emit(combined_warning)

        if self.bus and self.insufficient_events_warnings:
            tubenames = ', '.join(f'"{name}"' for name in self.insufficient_events_warnings)
            combined_insufficient = (
                f'The following sample(s) have fewer than two events in the '
                f'"{self.base_gate_label}" base gate, so a spectral control could not be '
                f'defined for them: {tubenames}.\n\n'
                f'Adjust the "{self.base_gate_label}" gate to make sure the relevant events '
                f'for all your single stain control samples are within it.'
            )
            self.bus.warningMessage.emit(combined_insufficient)

        if self.bus:
            self.bus.progress.emit(self.progress_target, self.progress_target)
            self.bus.spectralModelUpdated.emit()
            self.bus.showSelectedProfiles.emit([])

    def get_unstained_negative(self, particle_type: str = 'Cells') -> bool:
        """
        Locate the unstained sample for *particle_type*, create the 'Neg Unstained'
        gate if absent, then cache the profile and raw events.

        particle_type='Cells': prefer a sample whose name does NOT contain 'Beads'.
        particle_type='Beads': prefer a sample whose name contains 'Beads'.

        Stores results in:
            self.unstained_negative          (profile vector, Cells only)
            self.unstained_negative_events   (raw array, Cells)
            self.unstained_negative_events_beads (raw array, Beads)

        Returns True if an unstained was found and loaded, False otherwise.
        Does NOT modify negative_type on failure.
        """
        try:
            manually_unstained = set(self.samples.get('unstained_samples', []))

            def _is_unstained(path: str, name: str) -> bool:
                return (path in manually_unstained
                        or re.search(r'(unstained|negative)', path, re.IGNORECASE)
                        or re.search(r'(unstained|negative)', name, re.IGNORECASE))

            def _is_bead(path: str, name: str) -> bool:
                return bool(re.search(r'bead', name, re.IGNORECASE)
                            or re.search(r'bead', path, re.IGNORECASE))

            sample_path = None
            # Search: manually tagged first, then regex, filtered by particle type
            candidates = [
                path for path, name in self.samples['all_samples'].items()
                if _is_unstained(path, name)
            ]
            if particle_type == 'Beads':
                bead_candidates = [p for p in candidates if _is_bead(p, self.samples['all_samples'][p])]
                sample_path = (bead_candidates or candidates or [None])[0]
            else:
                # Honour editor hint for Cells unstained when available.
                if self.preferred_unstained:
                    all_samples_rev = {v: k for k, v in self.samples['all_samples'].items()}
                    sample_path = all_samples_rev.get(self.preferred_unstained)
                if not sample_path:
                    cell_candidates = [p for p in candidates if not _is_bead(p, self.samples['all_samples'][p])]
                    sample_path = (cell_candidates or candidates or [None])[0]

            if not sample_path:
                logger.info(f'get_unstained_negative: no unstained found for particle_type={particle_type!r}.')
                return False

            full_sample_path = str(self.experiment_dir / sample_path)
            sample = sample_from_fcs(full_sample_path)

            # Create 'Neg Unstained' gate anchored under base gate
            negative_gate_label = 'Neg Unstained'
            if self.base_gate_label != 'root' and self.raw_gating.find_matching_gate_paths(self.base_gate_label):
                base_path = tuple(list(self.raw_gating.find_matching_gate_paths(self.base_gate_label)[0]) + [self.base_gate_label])
            else:
                base_path = ('root',)

            if not self.raw_gating.find_matching_gate_paths(negative_gate_label):
                channel_x = self.fluorescence_channels_pnn[0]
                dim_x = Dimension(channel_x, range_min=0.0, range_max=1.0, transformation_ref=channel_x)
                neg_gate = gates.RectangleGate(negative_gate_label, dimensions=[dim_x])
                self.raw_gating.add_gate(neg_gate, gate_path=base_path)
                if self.bus:
                    self.bus.changedGatingHierarchy.emit('raw', negative_gate_label)
                target_plot = None
                for plot in self.raw_plots:
                    if (plot['type'] == 'hist1d'
                            and plot['channel_x'] == channel_x
                            and plot['source_gate'] == self.base_gate_label
                            and not set(plot['child_gates']) - {negative_gate_label}):
                        target_plot = plot
                if not target_plot:
                    target_plot = {'type': 'hist1d', 'channel_x': channel_x,
                                   'source_gate': self.base_gate_label, 'child_gates': []}
                    self.raw_plots.append(target_plot)
                    if self.bus:
                        self.bus.showNewPlot.emit('raw')
                if negative_gate_label not in target_plot['child_gates']:
                    target_plot['child_gates'].append(negative_gate_label)

            raw_events = get_raw_events(
                sample, self.fluorescence_channel_ids,
                gate_label=self.base_gate_label,
                gating_strategy=self.raw_gating,
            )

            if particle_type == 'Beads':
                self.unstained_negative_events_beads = raw_events
            else:
                _col_order = self.controller.experiment.settings['raw'].get('whitelisted_pnn')
                _ecpnn = self.controller.experiment.settings['raw']['event_channels_pnn']
                try:
                    _all_events = sample.get_events('raw', col_order=_col_order)
                except (KeyError, ValueError):
                    _all_events = sample.get_events('raw')
                    _col_order = None
                _pnn = _col_order if _col_order is not None else sample.pnn_labels
                self.unstained_negative = get_profile(
                    sample, self.base_gate_label,
                    self.raw_gating, self.fluorescence_channel_ids,
                    preloaded_events=_all_events, preloaded_pnn=_pnn,
                    event_channels_pnn=_ecpnn,
                )
                self.unstained_negative_events = raw_events

            logger.info(
                f'get_unstained_negative: {len(raw_events)} events '
                f'from "{self.samples["all_samples"][sample_path]}" for particle_type={particle_type!r}.'
            )
            return True

        except Exception as e:
            logger.warning(f'get_unstained_negative: failed for particle_type={particle_type!r}: {e}')
            return False

    @timer
    def generate_spectral_control(self, n):
        sample_path = self.samples['single_stain_controls'][n]

        if (self.base_gate_label != 'root'
                and not self.raw_gating.find_matching_gate_paths(self.base_gate_label)):
            warnings.warn(
                f'Autogenerate: base gate "{self.base_gate_label}" not found in the '
                f'shared raw gating hierarchy.'
            )
            return False

        nevents = self.samples['all_sample_nevents'][sample_path]
        tubename = self.samples['all_samples'][sample_path]
        if nevents > 0:
            # Skip samples whose name or path contains "Unstained" or "Negative",
            # or that have been manually marked as unstained by the user.
            # Users can still add them manually via the +Add Control button.
            manually_unstained = set(self.samples.get('unstained_samples', []))
            if re.search(r'(unstained|negative)', tubename, re.IGNORECASE) or \
               re.search(r'(unstained|negative)', sample_path, re.IGNORECASE) or \
               sample_path in manually_unstained:
                logger.info(f'generate_spectral_control: skipping "{tubename}" (matches unstained/negative pattern or manually marked as unstained)')
                return True

            particle_type = 'Cells'
            match = re.findall('([Cc]ells|[Bb]eads)', tubename)
            if match:
                if 'cells' in match[0].lower():
                    particle_type = 'Cells'
                else:
                    particle_type = 'Beads'
            else:
                warnings.warn(f'Unknown particle type from name, assigning as {particle_type}')

            match = re.findall(r'^(.*?)(?=\s*[\(\[]|\s+(?:cells?|beads?)\s*$)', tubename, re.IGNORECASE)
            raw_label = match[0].strip() if match else tubename

            # Strip leading plate-position prefixes like "A1 ", "B12 " before label matching
            raw_label = re.sub(r'^[A-H]\d{1,2}\s+', '', raw_label).strip()

            canonical_fluor = match_fluorophore(raw_label, get_fluorophore_db())
            label = canonical_fluor if canonical_fluor else raw_label

            canonical_marker = match_marker(raw_label, get_marker_db())
            antigen = canonical_marker or ''

            # discard if label is already in spectral model
            if label in [control['label'] for control in self.spectral_model]:
                pass
            else:
                full_sample_path = str(self.experiment_dir / sample_path)
                if check_fcs_matches_experiment(full_sample_path, self.controller.experiment.settings['raw']['event_channels_pnn'], self.controller.experiment.settings['raw']['magnitude_ceiling']):
                    sample = sample_from_fcs(full_sample_path)

                    match = re.findall('([Uu]nstained)', label)
                    if match:
                        ...
                    else:
                        pos_events_all = get_raw_events(
                            sample, self.fluorescence_channel_ids,
                            gate_label=self.base_gate_label,
                            gating_strategy=self.raw_gating,
                        )

                        if len(pos_events_all) < 2:
                            warnings.warn(f'Control sample has less than two events in base gate {self.base_gate_label}: cannot define spectral control')
                            self.insufficient_events_warnings.append(tubename)
                            return False

                        # Determine AF reference for orthogonalisation:
                        # use particle-type-matched unstained events if available,
                        # otherwise fall back to the bottom 25 % of this sample.
                        unstained_ref = (
                            self.unstained_negative_events_beads
                            if particle_type == 'Beads'
                            else self.unstained_negative_events
                        )
                        if unstained_ref is not None and len(unstained_ref) > 1:
                            af_mean = unstained_ref.mean(axis=0)
                            af_ref_source = f'unstained ({particle_type})'
                        else:
                            n_bottom = max(2, len(pos_events_all) // 4)
                            order = np.argsort(pos_events_all.max(axis=1))
                            af_mean = pos_events_all[order[:n_bottom]].mean(axis=0)
                            af_ref_source = 'internal bottom-25%'

                        # find_empirical_peak returns an index into fluorescence_channel_ids
                        fluor_peak_idx = find_empirical_peak(pos_events_all, af_mean)
                        channel_id_best_match = self.fluorescence_channel_ids[fluor_peak_idx]
                        channel_x = self.event_channels_pnn[channel_id_best_match]
                        gate_channel = channel_x

                        logger.info(
                            f'generate_spectral_control: "{label}" empirical peak = {channel_x} '
                            f'(af_ref={af_ref_source}, fluor_peak_idx={fluor_peak_idx})'
                        )

                        # Derive gate ranges from the peak channel distribution.
                        # fl_top / fl_bottom mirror the percentile approach in get_best_channel.
                        peak_vals = pos_events_all[:, fluor_peak_idx]
                        if len(peak_vals) > 100:
                            pos_percentile = settings.spectral_positive_gate_percent_retrieved
                            neg_percentile = settings.spectral_negative_gate_percent_retrieved
                        else:
                            pos_percentile = 50
                            neg_percentile = 50
                        fl_top    = np.percentile(peak_vals, [100 - pos_percentile, 100])
                        fl_bottom = np.percentile(peak_vals, [0, neg_percentile])

                        # Clamp positive gate upper edge to on-scale.
                        sat_value = self.controller.experiment.settings['raw']['magnitude_ceiling']
                        sat_threshold = sat_value * 0.99
                        fl_top[1] = min(fl_top[1], sat_threshold)
                        fl_top[0] = min(fl_top[0], fl_top[1] * 0.95)  # keep lower edge below upper

                        pos_range_min = self.raw_gating.transformations[channel_x].apply(np.array([fl_top[0]]))[0]
                        pos_range_max = self.raw_gating.transformations[channel_x].apply(np.array([fl_top[1]]))[0]
                        pos_dim_x = Dimension(channel_x, range_min=pos_range_min, range_max=pos_range_max, transformation_ref=channel_x)
                        positive_gate_label = 'Pos ' + label
                        positive_gate = gates.RectangleGate(positive_gate_label, dimensions=[pos_dim_x])

                        neg_range_min = self.raw_gating.transformations[channel_x].apply(np.array([fl_bottom[0]]))[0]
                        neg_range_max = self.raw_gating.transformations[channel_x].apply(np.array([fl_bottom[1]]))[0]
                        neg_dim_x = Dimension(channel_x, range_min=neg_range_min, range_max=neg_range_max, transformation_ref=channel_x)
                        negative_gate_label = 'Neg ' + label
                        negative_gate = gates.RectangleGate(negative_gate_label, dimensions=[neg_dim_x])

                        if self.raw_gating.find_matching_gate_paths(positive_gate_label):
                            self.raw_gating.remove_gate(positive_gate_label)
                        if self.raw_gating.find_matching_gate_paths(negative_gate_label):
                            self.raw_gating.remove_gate(negative_gate_label)
                        self.raw_gating.add_gate(positive_gate, gate_path=tuple(list(self.raw_gating.find_matching_gate_paths(self.base_gate_label)[0]) + [self.base_gate_label]))
                        self.raw_gating.add_gate(negative_gate, gate_path=tuple(list(self.raw_gating.find_matching_gate_paths(self.base_gate_label)[0]) + [self.base_gate_label]))
                        target_plot = None
                        for n, plot in enumerate(self.raw_plots):
                            if plot['type'] == 'hist1d' and plot['channel_x'] == channel_x:
                                target_plot = plot
                        if not target_plot:
                            target_plot = {
                                'type': 'hist1d',
                                'channel_x': channel_x,
                                'source_gate': self.base_gate_label,
                                'child_gates': [positive_gate_label, negative_gate_label]
                            }
                            self.raw_plots.append(target_plot)
                            if self.bus:
                                self.bus.showNewPlot.emit('raw')
                        if positive_gate_label not in target_plot['child_gates']:
                            target_plot['child_gates'].append(positive_gate_label)
                            target_plot['child_gates'].append(negative_gate_label)

                    if self.bus is not None:
                        self.bus.changedGatingHierarchy.emit('raw', negative_gate_label)
                        self.bus.changedGatingHierarchy.emit('raw', positive_gate_label)

                    ##### uncomment this to get more info to console. consider adding to gui
                    # report = self.raw_gating.gate_sample(sample).report.set_index('gate_name')  # Note this is slow
                    # print(f'{label}: explained variance {int(explained_variance * 100)}, best channel {self.event_channels_pnn[channel_id_best_match]}, brightest events {int(best_match)}/100, gate {report.loc[positive_gate_label]['count']}/100')

                    default_unstained = (
                        self.preferred_unstained
                        or _find_default_unstained_tube(self.samples['all_samples'])
                    )
                    assigned_negative = default_unstained or ''  # default; overridden for Beads below
                    # For bead controls, only use an unstained negative if available
                    if particle_type == 'Beads':
                        # Mirror the combobox builder logic exactly: iterate all_samples,
                        # accept anything unstained (manually tagged OR regex), then
                        # filter to those whose name contains "Beads".
                        manually_tagged = set(self.samples.get('unstained_samples', []))
                        default_unstained_bead = None
                        for path, name in self.samples['all_samples'].items():
                            is_unstained = (
                                path in manually_tagged
                                or bool(re.search(r'unstained', name, re.IGNORECASE))
                                or bool(re.search(r'unstained', path, re.IGNORECASE))
                            )
                            if not is_unstained:
                                continue
                            if re.search(r'bead', name, re.IGNORECASE) or re.search(r'bead', path, re.IGNORECASE):
                                default_unstained_bead = name
                                break
                        # No bead-specific unstained available: explicitly opt out of the
                        # global-unstained fallback (which is cell-only in ProfileUpdater)
                        # rather than leaving this blank. A blank value reads to
                        # ProfileUpdater.generate() as "no per-control override, use the
                        # global default" — silently pulling in the cell AF vector.
                        assigned_negative = default_unstained_bead or INTERNAL_NEGATIVE_SENTINEL
                    control = {
                        'label': label,
                        'antigen': antigen,
                        'control_type': 'Single Stained Spectral Control', 
                        'particle_type': particle_type,
                        'gate_channel': gate_channel, 
                        'sample_name': tubename,
                        'sample_path': full_sample_path, 
                        'gate_label': positive_gate_label,
                        'neg_gate_label': negative_gate_label,
                        'universal_negative_name': assigned_negative,
                    }

                    _col_order = self.controller.experiment.settings['raw'].get('whitelisted_pnn')
                    _ecpnn = self.controller.experiment.settings['raw']['event_channels_pnn']
                    try:
                        _all_events = sample.get_events('raw', col_order=_col_order)
                    except (KeyError, ValueError):
                        _all_events = sample.get_events('raw')
                        _col_order = None
                    _pnn = _col_order if _col_order is not None else sample.pnn_labels

                    positive_profile = get_profile(
                        sample, control['gate_label'], self.raw_gating,
                        self.fluorescence_channel_ids,
                        preloaded_events=_all_events, preloaded_pnn=_pnn,
                        event_channels_pnn=_ecpnn,
                    )
                    if (self.controller.experiment.process['negative_type'] == 'unstained'
                            and self.unstained_negative is not None):
                        negative_profile = self.unstained_negative
                    else:
                        negative_profile = get_profile(
                            sample, negative_gate_label, self.raw_gating,
                            self.fluorescence_channel_ids,
                            preloaded_events=_all_events, preloaded_pnn=_pnn,
                            event_channels_pnn=_ecpnn,
                        )
                    profile = positive_profile - negative_profile
                    if profile.sum() == 0:
                        profile = positive_profile

                    profile_warning = None
                    if profile.sum() > 0:
                        profile = profile / profile.max()  # max normalisation
                    else:
                        # Negative profile: recover shape via abs() so the control
                        # still appears in the Profiles Viewer and can be cleaned.
                        profile_warning = (
                            f'"{control["label"]}" has a negative profile — the negative '
                            f'gate has higher mean fluorescence than the positive. '
                            f'The control has been added but will not unmix correctly. '
                            f'Run Control Cleaning to fix this.'
                        )
                        warnings.warn(profile_warning)
                        self.negative_profile_warnings.append(control['label'])
                        abs_max = np.abs(profile).max()
                        profile = np.abs(profile) / abs_max if abs_max > 0 else positive_profile

                    control['profile_warning'] = profile_warning  # None if clean
                    self.spectral_model.append(control)

                    profile = profile.tolist()
                    self.profiles[control['label']] = profile

                    profile_dict = dict(zip(self.fluorescence_channels_pnn, profile))
                    spectral_library.deposit_control_with_profile_and_experiment_dir(control, profile_dict, str(self.experiment_dir))
                    return True
                else:
                    warnings.warn('Control sample file does not match experiment')
                    # self.bus.warningMessage.emit('Failed to load control sample: sample channels (names and ranges) not consistent with experiment configuration.\n\n'
                    #                              'Were your controls and samples acquired by the same instrument and settings? \n'
                    #                              'If so, run Import FCS Files (from the File menu). \n'
                    #                              'Otherwise, these controls cannot be used.')
                    if self.bus:
                        self.bus.openImportFCSWidget.emit(True)
                    return False

        else:
            warnings.warn('Invalid spectral control, no events')
            return True



# ---------------------------------------------------------------------------
# SpectralCleaner
# Runs saturation exclusion + brightest-event selection for all cell controls
# that have a universal_negative_name.  Stores CleanResult in
# controller.cleaned_events[label].
# ---------------------------------------------------------------------------

TARGET_N  = 50   # minimum events needed for a reliable profile
MIN_EVENTS_FOR_CLEANING = 20   # below this, skip cleaning entirely — too few events for cosine/kNN selection to be meaningful

class SpectralCleaner(QObject):
    """
    Runs the AutoSpectral cleaning pipeline (saturation exclusion +
    brightest-event selection) for every eligible cell control.

    Instantiate, optionally move to a QThread, then call run().
    The caller is responsible for emitting bus.spectralModelUpdated after run()
    completes (e.g. via a QThread.finished signal).
    """
    cleaningFinished = Signal()

    def __init__(self, bus, controller):
        super().__init__()
        self.bus = bus
        self.controller = controller
        self.experiment_dir  = controller.experiment_dir
        self.samples         = controller.experiment.samples
        self.spectral_model  = controller.experiment.process['spectral_model']
        self.cleaned_events  = controller.cleaned_events
        self.raw_gating      = controller.raw_gating
        self.event_channels_pnn = controller.experiment.settings['raw']['event_channels_pnn']
        controller.filter_raw_fluorescence_channels()
        self.fluor_ch_ids    = controller.filtered_raw_fluorescence_channel_ids
        self.ceiling         = controller.experiment.settings['raw']['magnitude_ceiling']
        self._cytometer_key: str | None = controller.experiment.settings['raw'].get('cytometer_db_col')
        self._collected_warnings: list[str] = []   # gathered across run(), emitted as one dialog at the end

    @with_busy_cursor
    def run(self):
        self._collected_warnings = []

        eligible_external = [
            c for c in self.spectral_model
            if c.get('control_type') == 'Single Stained Spectral Control'
            and c.get('particle_type') == 'Cells'
            and c.get('universal_negative_name')
            and c.get('universal_negative_name') != INTERNAL_NEGATIVE_SENTINEL
        ]

        eligible_internal = [
            c for c in self.spectral_model
            if c.get('control_type') == 'Single Stained Spectral Control'
            and (
                c.get('particle_type') == 'Beads'
                or c.get('universal_negative_name') == INTERNAL_NEGATIVE_SENTINEL
            )
        ]

        all_eligible = eligible_external + eligible_internal
        total = len(all_eligible)

        for n, control in enumerate(eligible_external):
            if self.bus:
                self.bus.progress.emit(n, total)
            if not self._is_cleaning_current(control):
                self._clean_one(control)
            else:
                logger.info(f'SpectralCleaner: skipping "{control["label"]}" — result already current.')

        for n, control in enumerate(eligible_internal):
            if self.bus:
                self.bus.progress.emit(len(eligible_external) + n, total)
            if not self._is_cleaning_current(control):
                self._clean_one(control)
            else:
                logger.info(f'SpectralCleaner: skipping "{control["label"]}" — result already current.')

        if self.bus:
            self.bus.progress.emit(total, total)

        logger.info(f'SpectralCleaner: cleaned {len(eligible_external)} cell controls '
                    f'and {len(eligible_internal)} internal/bead controls.')

        # Emit all collected skip/failure messages as a single dialog, rather
        # than one QMessageBox per control (which can crash Qt on macOS when
        # fired repeatedly from a worker thread).
        if self._collected_warnings and self.bus:
            if len(self._collected_warnings) == 1:
                combined = f'Control Cleaning: {self._collected_warnings[0]}'
            else:
                combined = (
                    f'Control Cleaning: {len(self._collected_warnings)} control(s) had issues:\n\n'
                    + '\n'.join(f'- {w}' for w in self._collected_warnings)
                )
            self.bus.warningMessage.emit(combined)
        self._collected_warnings = []

        self.cleaningFinished.emit()

    def _cleaning_fingerprint(self, control: dict) -> dict:
        """Return a dict of all inputs that determine what _clean_one will produce.
        If this matches what is stored in cleaned_events, the result is still current."""
        return {
            'sample_name': control.get('sample_name'),
            'gate_label': control.get('gate_label'),
            'universal_negative_name': control.get('universal_negative_name'),
            'particle_type': control.get('particle_type'),
            'fluor_ch_ids': list(self.fluor_ch_ids),
        }

    def _is_cleaning_current(self, control: dict) -> bool:
        """Return True if a valid cleaned result already exists for this control
        and its fingerprint matches the current control configuration."""
        label = control.get('label') or ''
        existing = self.cleaned_events.get(label)
        if not existing:
            return False
        stored_fp = existing.get('_fingerprint')
        if stored_fp is None:
            return False
        return stored_fp == self._cleaning_fingerprint(control)
    
    def _clean_one(self, control: dict):
        label = control['label']
        try:
            # --- load base-gate fluorescence + scatter events ---
            # Use the base gate (same pool as the autogenerator), not the narrow per-control positive gate.
            # The cleaner selects its own positives via  cosine filter.
            all_samples_rev = {v: k for k, v in self.samples['all_samples'].items()}
            rel_path = all_samples_rev.get(control['sample_name'])
            if rel_path is None:
                raise ValueError(f'Sample "{control["sample_name"]}" not found.')
            full_path = str(self.experiment_dir / rel_path)
            sample = sample_from_fcs(full_path)

            base_gate_label = 'root'
            raw_gate_names = [g[0].lower() for g in self.raw_gating.get_gate_ids()]
            for gate in self.controller.experiment.process.get('base_gate_priority_order', []):
                if gate.lower() in raw_gate_names:
                    base_gate_label = gate
                    break

            scatter_ch_ids = self.controller.experiment.settings['raw']['scatter_channel_ids']
            scatter_ch_pnn = self.controller.experiment.settings['raw']['event_channels_pnn']

            pos_events, pos_scatter_all = get_raw_events(
                sample, self.fluor_ch_ids,
                gate_label=base_gate_label,
                gating_strategy=self.raw_gating,
                extra_channel_ids=scatter_ch_ids,
            )

            # Reduce scatter to the canonical morphology pair (scatter_param[0], scatter_param[1]).
            scatter_param = self.controller.experiment.settings['raw'].get('scatter_param', ['FSC-A', 'SSC-A'])
            morph_x_name, morph_y_name = scatter_param[0], scatter_param[1]

            def _fsc_ssc_cols(all_scatter: np.ndarray) -> np.ndarray:
                col_x = next((col for col, ch_id in enumerate(scatter_ch_ids)
                            if scatter_ch_pnn[ch_id] == morph_x_name), None)
                col_y = next((col for col, ch_id in enumerate(scatter_ch_ids)
                            if scatter_ch_pnn[ch_id] == morph_y_name), None)
                if col_x is None:
                    col_x = next((col for col, ch_id in enumerate(scatter_ch_ids)
                                if 'FSC' in scatter_ch_pnn[ch_id]), 0)
                if col_y is None:
                    col_y = next((col for col, ch_id in enumerate(scatter_ch_ids)
                                if 'SSC' in scatter_ch_pnn[ch_id]), 1)
                return all_scatter[:, [col_x, col_y]]
            
            pos_scatter = _fsc_ssc_cols(pos_scatter_all) if pos_scatter_all.shape[1] > 2 else pos_scatter_all

            # --- load negative events ---
            neg_name = control.get('universal_negative_name')
            has_external_neg = (
                neg_name
                and neg_name != INTERNAL_NEGATIVE_SENTINEL
                and self.controller.experiment.process.get('negative_type') != 'internal'
            )
            use_internal = not has_external_neg

            logger.info(
                f'_clean_one: "{label}" particle_type={control.get("particle_type")!r} '
                f'universal_negative_name={control.get("universal_negative_name")!r} '
                f'use_internal={use_internal}'
            )

            if use_internal:
                # Load the negative gate from the same positive sample file.
                neg_gate_lbl = f'Neg {label}'
                try:
                    if (self.raw_gating and
                            self.raw_gating.find_matching_gate_paths(neg_gate_lbl)):
                        neg_events_internal, _ = get_raw_events(
                            sample, self.fluor_ch_ids,
                            gate_label=neg_gate_lbl,
                            gating_strategy=self.raw_gating,
                        )
                    else:
                        neg_events_internal = np.empty((0, pos_events.shape[1]))
                except Exception:
                    neg_events_internal = np.empty((0, pos_events.shape[1]))
                neg_events = neg_events_internal
                neg_scatter = np.empty((0, 2))     # no scatter matching for internal
            else:
                neg_events_internal = None
                tube_name = control.get('universal_negative_name') or ''
                if not tube_name:
                    msg = f'"{label}": no negative assigned — skipping cleaning.'
                    logger.warning(f'SpectralCleaner: {msg}')
                    self._collected_warnings.append(msg)
                    return
                all_samples_rev2 = {v: k for k, v in self.samples['all_samples'].items()}
                neg_rel_path = all_samples_rev2.get(tube_name)
                if not neg_rel_path:
                    msg = f'"{label}": negative tube "{tube_name}" not found — skipping cleaning.'
                    logger.warning(f'SpectralCleaner: {msg}')
                    self._collected_warnings.append(msg)
                    return
                neg_full_path = str(self.experiment_dir / neg_rel_path)
                neg_sample = sample_from_fcs(neg_full_path)
                neg_events, neg_scatter_all = get_raw_events(
                    neg_sample, self.fluor_ch_ids,
                    gate_label=base_gate_label,
                    gating_strategy=self.raw_gating,
                    extra_channel_ids=scatter_ch_ids,
                )
                if len(neg_events) < MIN_EVENTS_FOR_CLEANING:
                    msg = (f'"{label}": only {len(neg_events)} negative event(s) — '
                           f'skipping cleaning, need at least {MIN_EVENTS_FOR_CLEANING}.')
                    logger.warning(f'SpectralCleaner: {msg}')
                    self._collected_warnings.append(msg)
                    return
                neg_scatter = _fsc_ssc_cols(neg_scatter_all) if neg_scatter_all.shape[1] > 2 else neg_scatter_all

            # Expected peak from spectral model table (fallback / QC only)
            peak_ch_name = control.get('gate_channel', '')
            try:
                expected_peak_ch_idx = self.fluor_ch_ids.index(
                    self.event_channels_pnn.index(peak_ch_name)
                )
            except (ValueError, IndexError):
                expected_peak_ch_idx = int(np.argmax(pos_events.mean(axis=0)))

            # --- Saturation exclusion ---
            pos_events, n_sat_pos, sat_mask_pos = exclude_saturated(pos_events, self.ceiling)
            pos_scatter = pos_scatter[~sat_mask_pos]
            if not use_internal:
                neg_events, n_sat_neg, sat_mask_neg = exclude_saturated(neg_events, self.ceiling)
                neg_scatter = neg_scatter[~sat_mask_neg]
            else:
                n_sat_neg = 0

            if len(pos_events) < MIN_EVENTS_FOR_CLEANING:
                msg = (f'"{label}": only {len(pos_events)} positive event(s) after saturation '
                       f'exclusion — skipping cleaning (need at least {MIN_EVENTS_FOR_CLEANING}). '
                       f'Existing profile, if any, is unaffected.')
                logger.warning(f'SpectralCleaner: {msg}')
                self._collected_warnings.append(msg)
                return  # leave cleaned_events untouched; profiles[label] is the fallback

            if use_internal:
                # --- Internal-negative path: top-100 minus bottom-10% mean ---
                n_pos       = len(pos_events)
                n_int_neg   = max(2, n_pos // 10)
                order_peak  = np.argsort(pos_events[:, expected_peak_ch_idx])
                int_neg_idx = order_peak[:n_int_neg]
                remaining   = order_peak[n_int_neg:]                      # exclude neg-selection events from signal pool
                top_idx     = remaining[-min(100, len(remaining)):]

                neg_mean     = pos_events[int_neg_idx].mean(axis=0)
                spectral_sub = pos_events[top_idx] - neg_mean

                pos_sel             = pos_events[top_idx]
                pos_scatter_sel     = (pos_scatter[top_idx]
                                       if len(pos_scatter) == len(pos_events)
                                       else np.empty((0, 2)))
                matched_neg_scatter = np.empty((0, 2))
                sel_idx             = top_idx
                cs_vals             = None
                peak_ch_idx         = expected_peak_ch_idx

                # --- Mean -> clip negatives -> normalise [0, 1] ---
                raw_spectrum = spectral_sub.mean(axis=0).clip(0, None)
            else:
                # --- External-negative path: cosine filter + kNN scatter subtraction ---
                af_median   = np.median(neg_events, axis=0)
                peak_ch_idx = expected_peak_ch_idx
                
                sel_idx, cs_vals = cosine_filter(
                    pos_events, af_median, peak_ch_idx=peak_ch_idx,
                    n_candidates=settings.spectral_cleaning_n_candidates_retrieved,
                    n_spectral=settings.spectral_cleaning_n_spectral_retrieved,
                )
                pos_sel         = pos_events[sel_idx]
                pos_scatter_sel = (pos_scatter[sel_idx]
                                   if len(pos_scatter) == len(pos_events)
                                   else np.empty((0, 2)))

                spectral_sub, matched_neg_scatter = knn_scatter_match(
                    pos_sel, pos_scatter_sel, neg_events, neg_scatter
                )

                # --- Column median -> clip negatives -> normalise [0, 1] ---
                raw_spectrum = np.median(spectral_sub, axis=0).clip(0, None)
            max_val      = raw_spectrum.max()
            spectrum     = (raw_spectrum / max_val) if max_val > 0 else raw_spectrum

            n_surviving = len(pos_sel)
            if n_surviving < TARGET_N:
                msg = f'"{label}": only {n_surviving} events after cleaning; profile may be unreliable.'
                logger.warning(f'SpectralCleaner: {msg}')
                self._collected_warnings.append(msg)

            result = CleanResult(
                spectral_sub          = spectral_sub,
                scatter_pos           = pos_scatter_sel,
                scatter_neg_matched   = matched_neg_scatter,
                n_removed_saturation  = n_sat_pos + n_sat_neg,
                n_surviving_positive  = n_surviving,
                empirical_peak_ch_idx = peak_ch_idx,
                expected_peak_ch_idx  = expected_peak_ch_idx,
                cs_vals               = cs_vals,
                cosine_selected_idx   = sel_idx,
            )

            # --- store result ---
            self.cleaned_events[label] = {
                'spectrum':              spectrum.tolist(),
                'n_removed_saturation':  result.n_removed_saturation,
                'n_surviving_positive':  result.n_surviving_positive,
                'empirical_peak_ch_idx': result.empirical_peak_ch_idx,
                'expected_peak_ch_idx':  result.expected_peak_ch_idx,
                'fluor_ch_ids':          list(self.fluor_ch_ids),
                'cytometer_key':         self._cytometer_key,
                'warnings':              result.warnings,
                '_fingerprint':          self._cleaning_fingerprint(control),
                'spectral_sub':          result.spectral_sub,
                'scatter_pos':           result.scatter_pos,
                'scatter_neg_matched':   result.scatter_neg_matched,
                'cs_vals':               result.cs_vals,
                'cosine_selected_idx':   result.cosine_selected_idx,
            }

            logger.info(
                f'SpectralCleaner: "{label}" — '
                f'{result.n_surviving_positive} cosine-selected events, '
                f'{result.n_removed_saturation} removed for saturation.'
            )

        except Exception as e:
            msg = f'"{label}": failed to clean: {e}'
            logger.error(f'SpectralCleaner: {msg}')
            self._collected_warnings.append(msg)
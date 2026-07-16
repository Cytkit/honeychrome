
'''
Experiment Control:
-This is the MVC controller
-Creates new experiment
-Loads saved experiment
-Loads sample
-Creates live sample and carries out live analysis
-Sends and receives signals to GUI

Creates new experiment / Loads saved experiment

Loads sample (e.g. first sample as current sample)
-Calculates all histograms
-Applies unmixing
-Applies compensation
-Applies gates
-Calculates masks
-Calculates stats

Creates live sample and carries out live analysis
-Receives start acquisition signal
-Initialises histograms
-Listens for new chunk, updates histogram, updates stats
-Runs spectral model process
-Runs unmixing

Sends and receives signals to GUI
'''

import warnings
from datetime import datetime
import numpy as np
from pathlib import Path
from PySide6.QtCore import QObject, Slot, QTimer
from flowkit import GatingStrategy, Sample, gates
import threading
from copy import deepcopy
from multiprocessing import shared_memory
import time

from honeychrome.experiment_model import ExperimentModel, check_fcs_matches_experiment
from honeychrome.controller_components.functions import apply_gates_in_place, apply_transfer_matrix, generate_transformations, update_transforms, initialise_hists, calc_hists, calc_stats, initialise_stats, assign_default_transforms, define_quad_gates, define_range_gate, define_polygon_gate, define_rectangle_gate, define_ellipse_gate, add_recent_file, empty_queue_nowait, define_process_plots, get_set_or_initialise_label_offset, sample_from_fcs, build_display_label_map
from honeychrome.controller_components.gml_functions_mod_from_flowkit import from_gml, to_gml
from honeychrome.controller_components.gating_templates import (
    DEFAULT_TEMPLATE_NAME,
    dynamic_key,
    empty_scoped_template,
    scoped_template_for_sample,
)
from honeychrome.controller_components.gating_instantiation import (
    replace_gates_in_place,
    resolve_dynamic_dimensions_in_place,
    sample_context_from_events,
)
from honeychrome.settings import traces_cache_size, traces_cache_dtype, adc_rate
import honeychrome.settings as settings
from honeychrome.settings import max_events_in_cache, n_channels_per_event, experiments_folder, live_data_process_repeat_time, settings_default, samples_default, channel_dict
from honeychrome.view_components.busy_cursor import with_busy_cursor
from honeychrome.controller_components.autospectral_functions import (
        get_af_spectra,
        precompute_af_matrices,
        precompute_joint_cov_extras,
        combine_af_precomputed,
        apply_af_unmixing,
        apply_af_transfer,
     )

base_directory = Path.home() / experiments_folder

import logging
logger = logging.getLogger(__name__)

# ---- cytometry_data_dictionary ----
# This dictionary comprises the ephemeral data for analysing raw or unmixed cytometry
# Copies of this dictionary are made for the processes in the different tabs of the workflow:
# raw, spectral process, unmixed, statistics, etc.
cytometry_data_dictionary = {
    'pnn': None, # list of channel names
    'pnn_labels': None, # antigen dye labels
    'fluoro_indices': None, # list of fluorescence channel indices to the list of channel names
    'lookup_tables': None, # dictionary of boolean lookup tables for each gate on the 1D or 2D plot on which the gate is defined (for fast gating)
    'event_data': None, # event data (which may be raw or unmixed depending on the copy of the dictionary)
    'transformations': None, # set of transformations for all channels
    'statistics': {}, # event statistics for each gate in the hierarchy
    'gating': GatingStrategy(), # flowkit.GatingStrategy object used to define the gating lookup tables
    'plots': [], # set of cytometry plot definitions (1D histograms, 2D histograms, ribbon plots referencing the channel names, source gates and child gates
    'histograms': [], # set of 1D and 2D histograms for plotting on the plots
    'gate_membership': {} # dictionary of gate membership for each gate, boolean array corresponding to event_data
}

class Controller(QObject):
    def __init__(self,
        events_cache_name=None,
        events_cache_lock=None,
        index_head_events_cache=None,
        index_tail_events_cache=None,
        oscilloscope_traces_queue=None,
        pipe_connection_instrument=None,
        pipe_connection_analyser=None):
        super().__init__()

        # file io
        self.experiment = ExperimentModel()
        self.experiment_dir = None
        self.current_sample = None
        self.current_sample_path = None
        self.live_sample_path = None
        self.raw_event_data = None
        self.unmixed_event_data = None
        self.af_sidecar_data = None   # (n_cells, 2) float64: col 0 = af_scale, col 1 = af_idx; None when OLS

        # ephemeral data on top of experiment
        self.filtered_raw_fluorescence_channel_ids = None
        self.raw_lookup_tables = {}
        self.unmixed_lookup_tables = {}
        self.transfer_matrix = None
        self.af_precomputed = None
        self.af_spectra = None
        self.af_precomputed_cache: dict = {}
        self.raw_transformations = None
        self.unmixed_transformations = None
        self.raw_gating = GatingStrategy()
        self.unmixed_gating = GatingStrategy()
        self.active_template = {'raw': DEFAULT_TEMPLATE_NAME, 'unmixed': DEFAULT_TEMPLATE_NAME}
        self._pending_template_scope = None
        self.data_for_cytometry_plots = deepcopy(cytometry_data_dictionary)
        self.data_for_cytometry_plots_raw = deepcopy(self.data_for_cytometry_plots)
        self.data_for_cytometry_plots_process = deepcopy(self.data_for_cytometry_plots)
        self.data_for_cytometry_plots_unmixed = deepcopy(self.data_for_cytometry_plots)
        self.current_mode = 'raw'
        self.mode_switch_in_progress = False

        # pipe connections
        self.pipe_connection_instrument = pipe_connection_instrument
        self.pipe_connection_analyser = pipe_connection_analyser

        # Events cache
        self.shm_events = None
        self.events_cache = None
        self.events_cache_name = events_cache_name
        self.events_cache_lock = events_cache_lock
        self.max_events_in_cache = max_events_in_cache
        self.n_channels_per_event = n_channels_per_event
        self.index_head_events_cache = index_head_events_cache
        self.index_tail_events_cache = index_tail_events_cache
        self.adc_rate = adc_rate

        # live data processing stop signal
        self.stop_live_data_processing = threading.Event()
        self.stop_live_data_processing.set()
        self.thread = None

        if self.events_cache_name is not None:
            self.shm_events = shared_memory.SharedMemory(name=self.events_cache_name)
            with self.events_cache_lock:
                self.events_cache = np.ndarray((self.max_events_in_cache, self.n_channels_per_event), dtype=np.int64, buffer=self.shm_events.buf)
        else:
            self.events_cache = None

        self.experiment_compatible_with_acquisition = None

        # Oscilloscope traces queue
        self.oscilloscope_traces_queue = oscilloscope_traces_queue

        # signals: note controller actions are connected in view
        self.bus = None

    def new_experiment(self, experiment_path, template_path=None):
        # get experiment_path from gui
        experiment_path = Path(experiment_path).with_suffix('.kit')
        if template_path is not None:
            self.experiment.load(template_path)
            logger.info(f'Controller: template loaded {template_path}')
            self.experiment.experiment_path = str(experiment_path)
            self.experiment.save()
        else:
            self.experiment.create(experiment_path) # initialises and saves experiment

        add_recent_file(experiment_path)
        self.experiment_dir = self.experiment.generate_subdirs()
        self.current_mode = 'raw'
        self.initialise_ephemeral_data()
        logger.info(f'Controller: new experiment created {self.experiment_dir}')

    def _load_cleaned_events(self):
        """Restore cleaned_events from .kit cleaned_meta and .cleaned.npz sidecar."""
        _ARRAY_KEYS = {
            'spectral_sub', 'scatter_pos', 'scatter_neg_matched',
            'cs_vals', 'cosine_selected_idx',
        }
        meta = self.experiment.process.get('cleaned_meta', {})
        arrays = {}
        npz_path = self.cleaned_npz_path
        if not npz_path.exists() and self._legacy_cleaned_npz_path.exists():
            # one-time migration from old alongside-.kit location
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            self._legacy_cleaned_npz_path.replace(npz_path)
            logger.info(f'Controller: migrated cleaned.npz to {npz_path}')
        if npz_path.exists():
            try:
                npz = np.load(str(npz_path), allow_pickle=False)
                arrays = dict(npz)
            except Exception as e:
                logger.warning(f'Controller: failed to load cleaned.npz: {e}')

        self.cleaned_events.clear()
        for label, json_fields in meta.items():
            entry = dict(json_fields)
            safe = label.replace('/', '_').replace(' ', '_')
            for k in _ARRAY_KEYS:
                key = f'{safe}__{k}'
                if key in arrays:
                    entry[k] = arrays[key]
            self.cleaned_events[label] = entry
        if meta:
            logger.info(f'Controller: loaded cleaned_events for {len(meta)} controls.')

    @with_busy_cursor
    def load_experiment(self, experiment_path):
        self.experiment.load(experiment_path)
        self.experiment_dir = self.experiment.generate_subdirs()
        add_recent_file(experiment_path)

        self.current_mode = 'raw'
        self.initialise_ephemeral_data()
        self._load_cleaned_events()
        self.cache_all_af_profiles()
        self.initialise_af_matrices()

        logger.info(f'Controller: experiment loaded {self.experiment_dir}')

        # # load first sample in order: #legacy: consider reinstate autoload of first sample
        # if sample_list:
        #     sample_path = self.experiment.samples['single_stain_controls'][0]
        #     self.current_sample_path = sample_path # just set path - view will trigger loading of sample
        #     # self.load_sample(sample_path)
        #     print(f'Controller: set current sample path to {self.current_sample_path}')
        # else:
        #     self.current_sample_path = None

    def _save_cleaned_events(self):
        """Write cleaned_events to JSON meta (.kit) and numpy sidecar (.cleaned.npz)."""
        _JSON_KEYS = {
            'spectrum', 'n_removed_saturation', 'n_surviving_positive',
            'empirical_peak_ch_idx', 'expected_peak_ch_idx',
            'fluor_ch_ids', 'cytometer_key', 'warnings', '_fingerprint',
        }
        _ARRAY_KEYS = {
            'spectral_sub', 'scatter_pos', 'scatter_neg_matched',
            'cs_vals', 'cosine_selected_idx',
        }
        meta = {}
        arrays = {}
        for label, entry in self.cleaned_events.items():
            safe = label.replace('/', '_').replace(' ', '_')
            meta[label] = {k: v for k, v in entry.items() if k in _JSON_KEYS}
            for k in _ARRAY_KEYS:
                arr = entry.get(k)
                if arr is not None and isinstance(arr, np.ndarray) and arr.size:
                    arrays[f'{safe}__{k}'] = arr

        self.experiment.process['cleaned_meta'] = meta
        if arrays:
            np.savez_compressed(str(self.cleaned_npz_path), **arrays)
        elif self.cleaned_npz_path.exists():
            self.cleaned_npz_path.unlink()

    @property
    def cleaned_npz_path(self) -> Path:
        cache_dir = self.experiment_dir / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / 'cleaned.npz'

    @property
    def _legacy_cleaned_npz_path(self) -> Path:
        """Pre-migration location, alongside the .kit file."""
        return Path(self.experiment.experiment_path).with_suffix('.cleaned.npz')
    
    @Slot(str)
    def save_experiment(self, experiment_path=None):
        if self.raw_transformations is None:
            logger.warning('Controller: save_experiment called before transformations initialised, skipping')
            return
        # convert ephemeral gating back to gml — persist into each scope's ACTIVE
        # template (raw / unmixed are independent template lists).
        unmixing_active = (
            self.experiment.process['unmixing_matrix'] is not None and self.unmixed_gating is not None
        )
        raw_active = self._scoped_templates('raw').setdefault(
            self._active_template_name('raw'), empty_scoped_template())
        raw_active['gml'] = to_gml(self.raw_gating)
        if unmixing_active:
            unmixed_active = self._scoped_templates('unmixed').setdefault(
                self._active_template_name('unmixed'), empty_scoped_template())
            unmixed_active['gml'] = to_gml(self.unmixed_gating)

        # Legacy fields mirror each scope's DEFAULT template so old readers still
        # see a valid hierarchy (backward compatible).
        raw_default = self._scoped_templates('raw').get(self._default_template_name('raw'), raw_active)
        self.experiment.cytometry['raw_gating'] = raw_default.get('gml') or to_gml(self.raw_gating)
        update_transforms(self.experiment.cytometry['raw_transforms'], self.raw_transformations)
        if unmixing_active:
            unmixed_default = self._scoped_templates('unmixed').get(
                self._default_template_name('unmixed'), unmixed_active)
            self.experiment.cytometry['gating'] = unmixed_default.get('gml') or to_gml(self.unmixed_gating)
            update_transforms(self.experiment.cytometry['transforms'], self.unmixed_transformations)

        if not experiment_path:
            self._save_cleaned_events()
            self.experiment.save()
            logger.info(f'Controller: experiment saved {self.experiment_dir}')
            if self.bus:
                QTimer.singleShot(100, lambda: self.bus.statusMessage.emit(f'Autosaved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}'))


        else:
            experiment_template = deepcopy(self.experiment)
            experiment_path = Path(experiment_path).with_suffix('.kit')
            experiment_template.experiment_path = str(experiment_path)
            experiment_template.samples = samples_default.copy()
            experiment_template.save()
            add_recent_file(experiment_path)
            logger.info(f'Controller: experiment template saved {experiment_path}')
            if self.bus:
                self.bus.popupMessage.emit(f'Experiment saved as a template: {experiment_path}'
                                           f'\n\n'
                                           f'(The new file has a copy of this experiment\'s settings, spectral process and cytometry; no samples.)')

    @with_busy_cursor
    @Slot(str, str)
    def on_gate_change(self, mode=None, top_gate='root'):
        # lookup tables must stay in sync with the gating strategy even when
        # the change happens on a tab the user isn't currently viewing —
        # otherwise apply_gates_in_place crashes when that tab is later shown.
        self.calculate_lookup_tables(mode=mode, top_gate=top_gate)

        if mode == self.current_mode:
            # recalculate histograms and stats
            # which gates and plots have changed?
            # but recalculate everything if using dotplots coloured by gate
            gate_ids = self.data_for_cytometry_plots['gating'].get_gate_ids()
            if settings.hist2dtype_retrieved == '2D Histogram':
                gates_to_recalculate = [top_gate]
                for gate_id in gate_ids:
                    if any([top_gate in ancestor for ancestor in gate_id[1]]):
                        gates_to_recalculate.append(gate_id[0])

            else:
                gates_to_recalculate = [g[0] for g in gate_ids]

            indices_plots_to_recalculate = []
            for n, plot in enumerate(self.data_for_cytometry_plots['plots']):
                if plot['source_gate'] in gates_to_recalculate:
                    indices_plots_to_recalculate.append(n)

            plots_to_recalculate = [self.data_for_cytometry_plots['plots'][n] for n in indices_plots_to_recalculate]
            hists = initialise_hists(plots_to_recalculate, self.data_for_cytometry_plots)
            for m, n in enumerate(indices_plots_to_recalculate):
                self.data_for_cytometry_plots['histograms'][n] = hists[m]

            self.data_for_cytometry_plots['statistics'] = initialise_stats(self.data_for_cytometry_plots['gating'])
            self.calc_hists_and_stats(gates_to_calculate=gates_to_recalculate, indices_plots_to_calculate=indices_plots_to_recalculate)

    def calculate_lookup_tables(self, mode=None, top_gate='root'):
        # apply gating strategy to unit images to produce masks
        # called on initialise ephemeral data or if gate added
        if mode is None:
            sets_to_update = [
                (self.raw_transformations, self.raw_gating, self.raw_lookup_tables),
                (self.unmixed_transformations, self.unmixed_gating, self.unmixed_lookup_tables)
            ]
        elif mode == 'raw':
            sets_to_update = [(self.raw_transformations, self.raw_gating, self.raw_lookup_tables)]
        elif mode == 'unmixed':
            sets_to_update = [(self.unmixed_transformations, self.unmixed_gating, self.unmixed_lookup_tables)]
        else:
            return

        for n in range(len(sets_to_update)):
            transformations, gating, lookup_tables = sets_to_update[n]
            if top_gate == 'root':
                lookup_tables.clear()

            if transformations and gating: #not empty
                gate_ids = gating.get_gate_ids()
                for gate_id in gate_ids:
                    if top_gate == gate_id[0] or top_gate in gate_id[1]:

                        if gating._get_gate_node(gate_id[0], gate_id[1]).gate_type != 'Quadrant': # bit of a hack. Can't find a better way of excluding Quadrants
                            gate = gating.get_gate(gate_id[0])
                            # gate = gating.get_gate(gate_id[0], gate_id[1]) # include gate path? no
                            channels = gate.get_dimension_ids()

                            if len(channels) == 1:
                                xchan = channels[0]
                                transform_x = transformations[xchan]

                                unit_1d = np.ones(transform_x.scale_bins+1, dtype=np.bool_)
                                mask_ones_1d = np.nonzero(unit_1d)

                                ### Use flowkit to make lookup table, gate by lookup table
                                scale_x = transform_x.scale
                                mask_coords = scale_x[mask_ones_1d[0] + 1]
                                mask_as_fksample = Sample(mask_coords, channel_labels=channels, sample_id='mask_as_fksample')

                                # get events in gate - generate temporary gating strategy for mask only - faster
                                temp_gating_strategy = GatingStrategy()
                                temp_gating_strategy.add_gate(gate, gate_path=('root',))
                                for channel in gate.dimensions:
                                    temp_gating_strategy.transformations[channel.id] = transformations[channel.id].xform
                                results_for_lookup_table = temp_gating_strategy.gate_sample(mask_as_fksample, verbose=True)
                                lookup_table = {gate_id[0]: results_for_lookup_table.get_gate_membership(gate_id[0])}


                            elif len(channels) == 2 and gate.gate_type != 'QuadrantGate': #2 channels
                                #todo this crashes if xchan and y chan are the same... fix
                                xchan = channels[0]
                                ychan = channels[1]
                                transform_x = transformations[xchan]
                                transform_y = transformations[ychan] #on loading/creating new experiment, sometimes crashes on this line, e.g. saying no transformations['FSC-W']

                                unit_2d = np.ones((transform_x.scale_bins + 1, transform_y.scale_bins + 1), dtype=np.bool_)
                                # unit_2d = np.ones((settings.hist_bins_retrieved+1, settings.hist_bins_retrieved+1), dtype=np.bool_)
                                mask_ones_2d = np.nonzero(unit_2d)

                                ### Use flowkit to make lookup table, gate by lookup table
                                scale_x = transform_x.scale
                                scale_y = transform_y.scale
                                mask_coords = np.column_stack((scale_x[mask_ones_2d[0]+1], scale_y[mask_ones_2d[1]+1]))
                                mask_as_fksample = Sample(mask_coords, channel_labels=channels, sample_id='mask_as_fksample')

                                # get events in gate - generate temporary gating strategy for mask only - faster
                                temp_gating_strategy = GatingStrategy()
                                temp_gating_strategy.add_gate(gate, gate_path=('root',))
                                for channel in gate.dimensions:
                                    temp_gating_strategy.transformations[channel.id] = transformations[channel.id].xform

                                results_for_lookup_table = temp_gating_strategy.gate_sample(mask_as_fksample, verbose=True)
                                lookup_table = {gate_id[0]: results_for_lookup_table.get_gate_membership(gate_id[0])}


                            else: #quad gate
                                xchan = gate.dimensions[0].dimension_ref
                                ychan = gate.dimensions[1].dimension_ref
                                channels = [xchan, ychan]
                                transform_x = transformations[xchan]
                                transform_y = transformations[ychan]

                                unit_2d = np.ones((settings.hist_bins_retrieved+1, settings.hist_bins_retrieved+1), dtype=np.bool_)
                                mask_ones_2d = np.nonzero(unit_2d)

                                ### Use flowkit to make lookup table, gate by lookup table
                                scale_x = transform_x.scale
                                scale_y = transform_y.scale
                                mask_coords = np.column_stack((scale_x[mask_ones_2d[0]+1], scale_y[mask_ones_2d[1]+1]))
                                mask_as_fksample = Sample(mask_coords, channel_labels=channels, sample_id='mask_as_fksample')

                                # get events in gate - generate temporary gating strategy for mask only - faster
                                temp_gating_strategy = GatingStrategy()
                                temp_gating_strategy.add_gate(gate, gate_path=('root',))
                                temp_gating_strategy.transformations[xchan] = transformations[xchan].xform
                                temp_gating_strategy.transformations[ychan] = transformations[ychan].xform

                                results_for_lookup_table = temp_gating_strategy.gate_sample(mask_as_fksample, verbose=True)
                                quadrant_names = gate.quadrants.keys()
                                lookup_table = {name:results_for_lookup_table.get_gate_membership(name) for name in quadrant_names}

                            lookup_tables.update(lookup_table)

        if self.bus is None:
            warnings.warn('No events bus connected')


    def _active_dynamic_specs(self, scope):
        """Dynamic dimension specs for the active scoped template, or {}.

        ``scope`` is ``'raw'`` or ``'unmixed'``. Missing template ⇒ {} (no
        dynamic dims), i.e. today's behaviour.
        """
        template = self._scoped_template(scope)
        if not template:
            return {}
        return template.get('dynamic_dimensions') or {}

    def apply_dynamic_gate_dimensions(self):
        """Resolve dynamic (e.g. Time) gate dimensions for the loaded sample.

        For gates whose dimensions are marked dynamic in the active template, the
        bounds are resolved from this sample's own channel ranges and the affected
        gates' lookup tables are rebuilt — so one template gates every sample by
        the same *relative* window (e.g. middle 90% of each run) even when the
        samples differ in duration. No dynamic specs ⇒ this is a no-op.
        """
        dfc = self.data_for_cytometry_plots
        if not dfc or dfc.get('event_data') is None or not len(dfc['event_data']):
            return
        scope = self._scope_for_mode()
        dynamic_specs = self._active_dynamic_specs(scope)
        if not dynamic_specs:
            return

        context = sample_context_from_events(dfc['event_data'], dfc['pnn'])
        affected = resolve_dynamic_dimensions_in_place(dfc['gating'], dynamic_specs, context)
        for gate_id in affected:
            self.calculate_lookup_tables(mode=scope, top_gate=gate_id)

    def _rebuild_per_sample_lookup_tables(self, scope):
        """Rebuild lookup tables only for gates that touch a per-sample (default,
        e.g. Time) transform, so they match THIS sample's scale.

        Gates on fixed transforms (fluorescence, FSC/SSC) keep their cached table
        — so a normal sample switch stays fast; only Time-involving gates pay the
        rebuild cost (and the Time transform's bins are capped so it stays cheap).
        """
        dfc = self.data_for_cytometry_plots
        gating = dfc.get('gating') if dfc else None
        transforms = (dfc.get('transformations') if dfc else None) or {}
        if gating is None:
            return
        for gate_id, gate_path in gating.get_gate_ids():
            try:
                channels = gating.get_gate(gate_id).get_dimension_ids()
            except Exception:
                continue
            if any(getattr(transforms.get(ch), 'id', None) == 'default' for ch in channels):
                self.calculate_lookup_tables(mode=scope, top_gate=gate_id)

    def set_gate_dimension_dynamic(self, scope, gate_id, channel, kind='time', params=None,
                                   template_name=None):
        """Mark a gate dimension as dynamic (per-sample) in a scoped template.

        Stored outside GatingML in the scope's template ``dynamic_dimensions``
        and re-resolved for the currently loaded sample. ``scope`` is
        ``'raw'`` or ``'unmixed'``."""
        params = params or {}
        name = template_name or self._active_template_name(scope)
        template = self._scoped_templates(scope).setdefault(name, empty_scoped_template())
        dynamic_dimensions = template.setdefault('dynamic_dimensions', {})
        dynamic_dimensions[dynamic_key(gate_id, channel)] = {'kind': kind, 'params': params}
        if scope == self._scope_for_mode():
            self.apply_dynamic_gate_dimensions()

    # --- Gating templates (per-scope: independent raw / unmixed lists) -------

    def _scope_for_mode(self):
        """Template scope for the current tab: 'raw' or 'unmixed'."""
        return 'raw' if self.current_mode == 'raw' else 'unmixed'

    def _scoped_templates(self, scope):
        """The template dict for a scope ('raw'/'unmixed'), created if absent."""
        key = 'raw_gating_templates' if scope == 'raw' else 'unmixed_gating_templates'
        return self.experiment.cytometry.setdefault(key, {})

    def _active_template_name(self, scope):
        return self.active_template.get(scope, DEFAULT_TEMPLATE_NAME)

    def _scoped_template(self, scope, name=None):
        name = name or self._active_template_name(scope)
        return self._scoped_templates(scope).get(name)

    def _gating_for_scope(self, scope):
        return self.raw_gating if scope == 'raw' else self.unmixed_gating

    def _scoped_template_gml(self, scope, name=None):
        """GML for a scope + template, falling back to legacy cytometry fields."""
        legacy = self.experiment.cytometry.get('raw_gating' if scope == 'raw' else 'gating')
        template = self._scoped_template(scope, name)
        if template is None:
            return legacy
        return template.get('gml') or legacy

    def _bind_template_plots(self, scope, name=None):
        """Point the scope's plot list at its active template's plots.

        ``cytometry['raw_plots']``/``['plots']`` and the runtime
        ``data_for_cytometry_plots_*['plots']`` become the SAME list object as
        the template's ``plots`` — so plots travel with the template, edits
        persist into it, and existing code reading those keys keeps working."""
        template = self._scoped_template(scope, name)
        if template is None:
            return
        template.setdefault('plots', [])
        if scope == 'raw':
            self.experiment.cytometry['raw_plots'] = template['plots']
            self.data_for_cytometry_plots_raw['plots'] = template['plots']
        else:
            self.experiment.cytometry['plots'] = template['plots']
            self.data_for_cytometry_plots_unmixed['plots'] = template['plots']

    def list_templates(self, scope):
        """Template names for a scope (at least ``default`` after migration)."""
        return list(self._scoped_templates(scope).keys())

    def _default_template_name(self, scope):
        """Primary template mirrored to the legacy fields; follows a rename."""
        key = 'default_raw_template_name' if scope == 'raw' else 'default_unmixed_template_name'
        return self.experiment.cytometry.get(key, DEFAULT_TEMPLATE_NAME)

    def _resolve_default_template(self, scope):
        """The scope's default/primary template name that ACTUALLY exists.

        Follows a rename via the stored pointer, then falls back to ``default``,
        then to the first available template — so a renamed default never dangles
        (which used to leave unassigned samples pointing at a missing 'default')."""
        templates = self._scoped_templates(scope)
        name = self._default_template_name(scope)
        if name in templates:
            return name
        if DEFAULT_TEMPLATE_NAME in templates:
            return DEFAULT_TEMPLATE_NAME
        return next(iter(templates), DEFAULT_TEMPLATE_NAME)

    def rename_template(self, scope, old, new):
        """Rename a template within a scope (e.g. raw ``default`` -> ``QC``)."""
        new = (new or '').strip()
        templates = self._scoped_templates(scope)
        if not new or old not in templates or new in templates or old == new:
            return
        self._scoped_templates(scope)  # ensure key exists
        key = 'raw_gating_templates' if scope == 'raw' else 'unmixed_gating_templates'
        self.experiment.cytometry[key] = {
            (new if k == old else k): v for k, v in templates.items()
        }
        assignments = self.experiment.samples.get('sample_template_assignments', {})
        for entry in assignments.values():
            if isinstance(entry, dict) and entry.get(scope) == old:
                entry[scope] = new
        if self.active_template.get(scope) == old:
            self.active_template[scope] = new
        if self._default_template_name(scope) == old:
            dkey = 'default_raw_template_name' if scope == 'raw' else 'default_unmixed_template_name'
            self.experiment.cytometry[dkey] = new
        self._emit_templates_changed()
        logger.info(f'Controller: renamed {scope} template {old!r} -> {new!r}')

    def _emit_templates_changed(self):
        """Tell each tab's picker its scope's template list + active template."""
        if self.bus is None:
            return
        for scope in ('raw', 'unmixed'):
            self.bus.templatesChanged.emit(scope, self.list_templates(scope), self._active_template_name(scope))

    def create_template(self, scope, name, copy_active=False):
        """Create a new template in a scope. Empty (root-only, no plots) by
        default; ``copy_active=True`` duplicates the active gating + plots.
        No-op if the name already exists."""
        templates = self._scoped_templates(scope)
        if name in templates:
            return name
        if copy_active:
            gml = to_gml(self._gating_for_scope(scope))
            plots = deepcopy((self.experiment.cytometry.get('raw_plots' if scope == 'raw' else 'plots')) or [])
        else:
            gml = to_gml(GatingStrategy())
            plots = []
        templates[name] = {'gml': gml, 'plots': plots, 'dynamic_dimensions': {}}
        if self.bus is not None:
            self.bus.changedGatingHierarchy.emit(scope, None)
        self._emit_templates_changed()
        return name

    def create_and_select_template(self, scope, name):
        """Create a new EMPTY template (root-only gates, no plots) in a scope and
        select it for the current sample (the ``＋`` action)."""
        self.create_template(scope, name, copy_active=False)
        self.select_template(scope, name)

    def select_template(self, scope, name, assign_current_sample=True):
        """Make ``name`` the active template for ``scope`` and the current sample.

        Loads the template's gates + plots, re-resolves dynamic dims, and
        refreshes that tab. This is the user action behind picking a template."""
        if name not in self._scoped_templates(scope):
            logger.warning(f'Controller: select_template unknown {scope} template {name!r}')
            return
        if assign_current_sample and self.current_sample_path:
            assignments = self.experiment.samples.setdefault('sample_template_assignments', {})
            entry = assignments.setdefault(self.current_sample_path, {})
            if not isinstance(entry, dict):
                entry = {}
                assignments[self.current_sample_path] = entry
            entry[scope] = name
        self._load_gating_from_template(scope, name)
        self.clear_data_for_cytometry_plots()
        self.initialise_data_for_cytometry_plots(force_recalc_histograms=True)
        if self.bus is not None:
            self.bus.changedGatingHierarchy.emit(scope, None)
        self._emit_templates_changed()
        logger.info(f'Controller: selected {scope} template {name!r} for {self.current_sample_path}')

    def _load_gating_from_template(self, scope, name):
        """Swap the scope's active gating to ``name``'s GML + plots, in place.

        Gating strategies are mutated in place (identity kept) so cached
        references stay valid. Flags a pending grid rebuild only when this scope
        is the current tab; the rebuild fires after stats are ready."""
        self.active_template[scope] = name
        gml = self._scoped_template_gml(scope, name)
        if scope == 'raw':
            replace_gates_in_place(self.raw_gating, from_gml(gml))
        elif self.experiment.process.get('unmixing_matrix') is not None and gml:
            replace_gates_in_place(self.unmixed_gating, from_gml(gml))
        self._bind_template_plots(scope, name)
        if self.raw_transformations is not None:
            self.calculate_lookup_tables(mode=scope)
        if scope == self._scope_for_mode():
            self._pending_template_scope = scope

    def ensure_template_for_current_sample(self):
        """Switch each scope's active gating to the loaded sample's assigned
        template. Called on sample load so different samples can use different
        templates. No-op for a scope already on the right template."""
        if not self.current_sample_path:
            return
        for scope in ('raw', 'unmixed'):
            desired = scoped_template_for_sample(scope, self.current_sample_path, self.experiment.samples)
            if desired not in self._scoped_templates(scope):
                desired = self._resolve_default_template(scope)
            if desired != self.active_template.get(scope) and desired in self._scoped_templates(scope):
                self._load_gating_from_template(scope, desired)

    def filter_raw_fluorescence_channels(self):
        if self.experiment.process['fluorescence_channel_filter'] == 'area_only':
            self.filtered_raw_fluorescence_channel_ids = [c for c in self.experiment.settings['raw']['fluorescence_channel_ids']
                                                          if self.experiment.settings['raw']['event_channels_pnn'][c].endswith('-A')]
        else:
            self.filtered_raw_fluorescence_channel_ids = self.experiment.settings['raw']['fluorescence_channel_ids']

    def flush_ephemeral_data(self):
        ###### reinitialise everything, just like in __init__
        self.current_sample = None
        self.current_sample_path = None
        self.live_sample_path = None
        self.raw_event_data = None
        self.unmixed_event_data = None
        self.af_sidecar_data = None

        # ephemeral data on top of experiment
        self.raw_lookup_tables = {}
        self.unmixed_lookup_tables = {}
        self.transfer_matrix = None
        self.af_precomputed_cache = {}
        self.raw_transformations = None
        self.unmixed_transformations = None
        self.raw_gating = GatingStrategy()
        self.unmixed_gating = GatingStrategy()
        self.active_template = {'raw': DEFAULT_TEMPLATE_NAME, 'unmixed': DEFAULT_TEMPLATE_NAME}
        self._pending_template_scope = None
        self.cleaned_events: dict = {}
        self.data_for_cytometry_plots = {'pnn': None, 'fluoro_indices': None, 'lookup_tables': None, 'event_data': None, 'transformations': None, 'statistics': {}, 'gating': GatingStrategy(), 'plots': [], 'histograms': [], 'gate_membership': {}}
        self.data_for_cytometry_plots_raw = deepcopy(self.data_for_cytometry_plots)
        self.data_for_cytometry_plots_process = deepcopy(self.data_for_cytometry_plots)
        self.data_for_cytometry_plots_unmixed = deepcopy(self.data_for_cytometry_plots)

    def initialise_ephemeral_data(self, scope=None):
        # called when an experiment is created or loaded (or spectral process is refreshed)
        self.experiment_compatible_with_acquisition = channel_dict['event_channels_pnn'] == self.experiment.settings['raw']['event_channels_pnn'] #todo should refer to instrument/analyst/acq config rather than defaults
        self.filter_raw_fluorescence_channels()

        if scope is None:
            scope = ['raw', 'unmixed']
            self.flush_ephemeral_data()

        # Resolve each scope's active template to its ACTUAL default (which may
        # have been renamed) so the picker, gating and plots all use the right
        # one; before a sample is loaded there is no assignment to go on.
        for _scope in scope:
            self.active_template[_scope] = self._resolve_default_template(_scope)

        # Per-sample gating: point the plot lists at the active templates' plots,
        # so plots travel with each scope's template (and edits persist into it).
        self._bind_template_plots('raw')
        self._bind_template_plots('unmixed')

        # plots is list of dicts
        # type:
        # ---hist1d: channel_x, source_gate, child_gates
        # ---hist2d: channel_x, channel_y, source_gate, child_gates
        # ---ribbon: source_gate, child_gates
        # e.g.:
        # self.experiment.cytometry['plots'] = [
        #     {'type': 'hist1d', 'channel_x': 'Time', 'source_gate': 'root', 'child_gates': []},
        #     {'type': 'hist2d', 'channel_x': 'FSC-A', 'channel_y': 'SSC-A', 'source_gate': 'root', 'child_gates': ['cells']},
        #     {'type': 'ribbon', 'source_gate': 'cells', 'child_gates': []},
        #     {'type': 'hist2d', 'channel_x': 'PE-Vio770-A', 'channel_y': 'Spark UV 387-A', 'source_gate': 'cells','child_gates': ['activated']},
        # ]

        ###### initialise data for cytometry plots
        if 'raw' in scope:
            self.raw_gating = from_gml(self._scoped_template_gml('raw'))
            self.raw_transformations = generate_transformations(self.experiment.cytometry['raw_transforms'])

            _raw = self.experiment.settings['raw']
            _display_pnn = _raw.get('whitelisted_pnn') or _raw['event_channels_pnn']
            _fl_ids_full = _raw['fluorescence_channel_ids']  # indices into full event_channels_pnn
            _fluoro_indices_display = [
                _display_pnn.index(_raw['event_channels_pnn'][i])
                for i in _fl_ids_full
                if _raw['event_channels_pnn'][i] in _display_pnn
            ]
            self.data_for_cytometry_plots_raw.update(
                {
                    'pnn': _display_pnn,
                    'fluoro_indices': _fluoro_indices_display,
                    'transformations': self.raw_transformations, 'gating': self.raw_gating,
                    'lookup_tables': self.raw_lookup_tables, 'plots': self.experiment.cytometry['raw_plots'],
                    'cytometer_db_col': _raw.get('cytometer_db_col'),
                 }
            )

        # recreate transfer matrix and compensated_unmixing_matrix if unmixing matrix is not None
        if 'unmixed' in scope:
            if self.experiment.process['unmixing_matrix']:
                self.unmixed_gating = from_gml(self._scoped_template_gml('unmixed'))
                self.unmixed_transformations = generate_transformations(self.experiment.cytometry['transforms'])
                # Call the plain impl, not reapply_fine_tuning(): this method
                # (initialise_ephemeral_data) is itself called from inside the
                # worker thread of @with_busy_cursor methods (load_experiment,
                # refresh_spectral_process). Calling the decorated
                # reapply_fine_tuning() here would re-enter with_busy_cursor
                # from a background thread — QApplication.setOverrideCursor()
                # and a nested QEventLoop off the main thread
                self._reapply_fine_tuning_impl()

                source_gate = 'root'
                unmixed_gate_names = [g[0].lower() for g in self.unmixed_gating.get_gate_ids()]
                for gate in self.experiment.process['base_gate_priority_order']:
                    if gate.lower() in unmixed_gate_names:
                        source_gate = gate
                        break
                logger.info(f'Controller: using {source_gate} as base gate for process NxN plots')
                process_plots = define_process_plots(self.experiment.settings['unmixed']['fluorescence_channels'], self.experiment.settings['unmixed']['fluorescence_channels'], source_gate=source_gate)
            else:
                self.unmixed_gating = GatingStrategy()
                self.unmixed_transformations = None
                self.unmixed_lookup_tables = {}
                process_plots = []

            unmixed_pnn = self.experiment.settings['unmixed']['event_channels_pnn']
            unmixed_pnn_labels = build_display_label_map(
                unmixed_pnn or [], self.experiment.process.get('spectral_model') # guard in case unmixed_pnn is None
            )

            self.data_for_cytometry_plots_process.update(
                {
                    'pnn': self.experiment.settings['unmixed']['event_channels_pnn'],
                    'pnn_labels': unmixed_pnn_labels,
                    'fluoro_indices': self.experiment.settings['unmixed']['fluorescence_channel_ids'],
                    'transformations': self.unmixed_transformations,
                    'gating': self.unmixed_gating,
                    'lookup_tables': self.unmixed_lookup_tables,
                    'plots': process_plots
                }
            )

            self.data_for_cytometry_plots_unmixed.update(
                {
                    'pnn': self.experiment.settings['unmixed']['event_channels_pnn'],
                    'pnn_labels': unmixed_pnn_labels,
                    'fluoro_indices': self.experiment.settings['unmixed']['fluorescence_channel_ids'],
                    'transformations': self.unmixed_transformations,
                    'gating': self.unmixed_gating,
                    'lookup_tables': self.unmixed_lookup_tables,
                    'plots': self.experiment.cytometry['plots']
                }
            )

        # extra lines here for debugging that horrible error caused by raw_transformations not being updated
        try:
            if scope is None:
                self.calculate_lookup_tables()           # full rebuild: raw + unmixed
            else:
                for s in scope:
                    self.calculate_lookup_tables(mode=s) # only the requested scope(s)
        except Exception:
            logger.warning(self.raw_transformations)
            logger.warning(self.raw_gating)
            logger.warning(self.data_for_cytometry_plots_raw['transformations'] is self.raw_transformations)
            logger.warning(self.data_for_cytometry_plots_raw['gating'] is self.raw_gating)

        # populate the template picker after (re)loading an experiment
        self._emit_templates_changed()

    def initialise_transfer_matrix(self):
        # run in intitialisation of ephemeral data or if spillover changed
        pnn_unmixed = self.experiment.settings['unmixed']['event_channels_pnn']
        # Use whitelisted PNN as the raw dimension — raw_event_data is always
        # loaded with col_order=whitelisted_pnn so its columns match this list.
        _raw = self.experiment.settings['raw']
        pnn_raw = _raw.get('whitelisted_pnn') or _raw['event_channels_pnn']
        full_pnn_raw = _raw['event_channels_pnn']

        # Re-express stored raw channel indices against pnn_raw (whitelisted).
        # Stored indices reference full_pnn_raw; map them to their position in pnn_raw.
        def _remap(ids):
            return np.array([pnn_raw.index(full_pnn_raw[i]) for i in ids])

        fl_channel_ids_raw = _remap(self.filtered_raw_fluorescence_channel_ids)
        sc_channel_ids_raw = _remap(_raw['scatter_channel_ids'])
        fl_channel_ids_unmixed = np.array(self.experiment.settings['unmixed']['fluorescence_channel_ids'])
        sc_channel_ids_unmixed = np.array(self.experiment.settings['unmixed']['scatter_channel_ids'])
        n_scatter_channels = self.experiment.settings['unmixed']['n_scatter_channels']

        unmixing_matrix = np.array(self.experiment.process['unmixing_matrix'])
        spillover = np.array(self.experiment.process['spillover'])
        # spillover[i][j] = row i spills into column j (see nxn_help_text);
        # compensation is applied as compensation @ measured (column vectors),
        # which needs the transposed inverse to match that row/column convention.
        compensation = np.linalg.inv(spillover).T
        compensated_unmixing_matrix = compensation @ unmixing_matrix
        transfer_matrix = np.zeros((len(pnn_unmixed), len(pnn_raw)))
        transfer_matrix[np.ix_(fl_channel_ids_unmixed, fl_channel_ids_raw)] = compensated_unmixing_matrix
        transfer_matrix[np.ix_(sc_channel_ids_unmixed, sc_channel_ids_raw)] = np.eye(n_scatter_channels)
        if _raw['time_channel_id'] is not None:
            raw_time_id = pnn_raw.index(full_pnn_raw[_raw['time_channel_id']])
            transfer_matrix[self.experiment.settings['unmixed']['time_channel_id'], raw_time_id] = 1
        # no Time channel in raw data — unmixed Time column stays zero

        if _raw['event_id_channel_id'] is not None:
            # event_id is not in whitelisted_pnn — omit silently (no event_id column in raw_event_data)
            if full_pnn_raw[_raw['event_id_channel_id']] in pnn_raw:
                raw_eid_id = pnn_raw.index(full_pnn_raw[_raw['event_id_channel_id']])
                transfer_matrix[self.experiment.settings['unmixed']['event_id_channel_id'], raw_eid_id] = 1

        # note transfer_matrix is transposed - multiply raw event data @ transfer_matrix to get unmixed event data in same form as raw
        transfer_matrix = transfer_matrix.T

        self.transfer_matrix = transfer_matrix
    
    def initialise_af_matrices(self):
        """
        Set self.af_precomputed and self.af_spectra for the currently loaded sample
        by combining cached per-profile matrices via np.hstack.

        No matrix algebra is performed here — only dict lookups and hstack, which
        are O(n_af * n_channels) and negligible compared to the linalg.solve that
        was previously run per sample.
        """
        af_spectra = self.get_combined_af_spectra_for_sample(self.current_sample_path)

        if af_spectra is None:
            self.af_precomputed = None
            self.af_spectra = None
            return

        # Combine per-profile cached precomputed dicts
        profile_names = (
            self.experiment.samples
            .get('sample_af_profiles', {})
            .get(self.current_sample_path, [])
        )
        cached = [
            self.af_precomputed_cache[name]
            for name in profile_names
            if name in self.af_precomputed_cache
        ]

        if not cached:
            logger.warning(
                f'initialise_af_matrices: no cache hit for sample "{self.current_sample_path}". '
                f'profile_names={profile_names}, '
                f'cache_keys={list(self.af_precomputed_cache.keys())}'
            )
            self.af_precomputed = None
            self.af_spectra = None
            return
        
        if len(cached) == 1:
            combined = cached[0]
        else:
            combined = combine_af_precomputed(cached)
            # Recompute af_error_weights across the full combined AF spectra
            if af_spectra is not None:
                combined.update(precompute_joint_cov_extras(combined, af_spectra))

        self.af_precomputed = combined
        assert af_spectra is not None
        self.af_spectra = af_spectra
        logger.info(
        f'Controller: AF matrices set for {self.current_sample_path} '
        f'({af_spectra.shape[0]} AF spectra, '
        f'{len(profile_names)} profile(s)) — from cache.'
    )

    def cache_af_profile(self, profile_name: str) -> bool:
        """
        Compute and cache the precomputed AF matrices for a single named profile.

        Call this immediately after storing a new profile in
        experiment.process['af_profiles'], so the cache is always up to date
        without needing to reload the experiment.

        Returns True on success, False if the profile or fluor_spectra are
        unavailable.
        """
        fluor_spectra = self._build_fluor_spectra()
        if fluor_spectra is None:
            logger.warning(
                f'cache_af_profile: cannot cache "{profile_name}" — '
                'fluor_spectra unavailable (spectral model not yet complete?).'
            )
            return False

        af_profiles = self.experiment.process.get('af_profiles', {})
        entry = af_profiles.get(profile_name)
        if entry is None:
            logger.warning(
                f'cache_af_profile: "{profile_name}" not found in af_profiles.'
            )
            return False

        try:
            af_spectra  = np.array(entry['spectra'])
            precomputed = precompute_af_matrices(fluor_spectra, af_spectra)
            precomputed.update(precompute_joint_cov_extras(precomputed, af_spectra))
            self.af_precomputed_cache[profile_name] = precomputed
            logger.info(
                f'Controller: cached AF precomputed matrices for "{profile_name}" '
                f'({af_spectra.shape[0]} AF spectra).'
            )
            return True
        except Exception as e:
            logger.error(f'cache_af_profile: failed for "{profile_name}": {e}')
            return False

    def cache_all_af_profiles(self):
        """
        (Re)compute precomputed matrices for every stored AF profile and replace
        the entire cache.  Called by initialise_transfer_matrix() after a spectral
        process refresh, because P depends on fluor_spectra which may have changed.
        """
        self.af_precomputed_cache = {}
        af_profiles = self.experiment.process.get('af_profiles', {})
        if not af_profiles:
            logger.debug(f'cache_all_af_profiles: experiment.process has no af_profiles — keys present: {list(self.experiment.process.keys())}')
            return

        fluor_spectra = self._build_fluor_spectra()
        if fluor_spectra is None:
            logger.warning(
                'cache_all_af_profiles: fluor_spectra unavailable — cache empty.'
            )
            return

        for name, entry in af_profiles.items():
            try:
                af_spectra  = np.array(entry['spectra'])
                precomputed = precompute_af_matrices(fluor_spectra, af_spectra)
                precomputed.update(precompute_joint_cov_extras(precomputed, af_spectra))
                self.af_precomputed_cache[name] = precomputed
            except Exception as e:
                logger.error(f'cache_all_af_profiles: failed for "{name}": {e}')

        logger.info(
            f'Controller: AF cache rebuilt — '
            f'{len(self.af_precomputed_cache)}/{len(af_profiles)} profiles cached.'
        )


    def get_combined_af_spectra_for_sample(self, sample_path) -> np.ndarray | None:
        """
        Return the vertically concatenated AF spectra ndarray for sample_path,
        or None if no profiles are assigned or sample_path is None.
        """
        if sample_path is None:
            return None

        profile_names = (
            self.experiment.samples
            .get('sample_af_profiles', {})
            .get(sample_path, [])
        )
        if not profile_names:
            return None

        af_profiles = self.experiment.process.get('af_profiles', {})
        matrices = []
        for name in profile_names:
            entry = af_profiles.get(name)
            if entry is None:
                logger.warning(
                    f'get_combined_af_spectra_for_sample: profile "{name}" '
                    f'assigned to {sample_path} not found in af_profiles — skipping.'
                )
                continue
            matrices.append(np.array(entry['spectra']))

        if not matrices:
            return None

        return np.vstack(matrices)


    def _build_fluor_spectra(self) -> np.ndarray | None:
        """Build (n_fluors, n_channels) fluorophore spectra matrix from stored profiles."""
        profiles = self.experiment.process.get('profiles')
        spectral_model = self.experiment.process.get('spectral_model', [])
        if not profiles or not spectral_model:
            return None
        labels = [c['label'] for c in spectral_model]
        try:
            rows = [profiles[label] for label in labels if label in profiles]
            if not rows:
                return None
            return np.array(rows)
        except Exception as e:
            logger.error(f'_build_fluor_spectra failed: {e}')
            return None


    def _apply_unmixing(self, raw_event_data):
        """Apply unmixing — AF-corrected if matrices are set, otherwise plain OLS/WLS.

        Always returns the unmixed event array (same shape contract as before).
        As a side-effect, populates self.af_sidecar_data with a (n_cells, 2)
        array [af_scale, af_idx] when AF correction is active, or sets it to
        None when plain OLS is used.  Callers must not rely on af_sidecar_data
        until after this method returns.
        """
        if self.af_precomputed is not None and self.af_spectra is not None:
            # Remap full-PNN fluorescence indices to whitelisted-PNN column positions.
            # raw_event_data is loaded with col_order=whitelisted_pnn, so stored
            # fluorescence_channel_ids (full-PNN) must be translated first.
            _raw = self.experiment.settings['raw']
            pnn_raw = _raw.get('whitelisted_pnn') or _raw['event_channels_pnn']
            full_pnn_raw = _raw['event_channels_pnn']
            fl_ids_remapped = [pnn_raw.index(full_pnn_raw[i])
                               for i in self.filtered_raw_fluorescence_channel_ids]
            result = apply_af_transfer(
                raw_event_data,
                self.transfer_matrix,
                self.af_precomputed,
                self.af_spectra,
                self.experiment.settings,
                filtered_fl_ids_raw=fl_ids_remapped,
                spillover=self.experiment.process.get('spillover'),
            )
            # result is now a dict; store the sidecar columns
            self.af_sidecar_data = np.column_stack(
                [result['af_scale'], result['af_idx'].astype(np.float64)]
            )
            return result['unmixed']
        else:
            self.af_sidecar_data = None
            return apply_transfer_matrix(self.transfer_matrix, raw_event_data)

    @Slot(str, str)
    def new_sample(self, sample_name, sample_type):
        if sample_type == 'single_stain_controls':
            directory = self.experiment.settings['raw']['single_stain_controls_subdirectory']
        elif sample_type:
            directory = sample_type
        else:
            directory = self.experiment.settings['raw']['raw_samples_subdirectory']

        sample_rel_path = (Path(directory) / sample_name).with_suffix('.fcs')

        if (self.experiment_dir / sample_rel_path).exists():
            if self.bus:
                self.bus.warningMessage.emit(f'{str(sample_rel_path)} already exists.')
        else:
            self.current_sample_path = str(sample_rel_path)

            # define empty sample and save
            self.current_sample = Sample(np.array([]), sample_id=sample_name, channel_labels=self.experiment.settings['raw']['event_channels_pnn'])
            self.current_sample.metadata['tubename'] = sample_name
            self.current_sample.export(self.experiment_dir / self.current_sample_path, source='raw', include_metadata=True)

            # update experiment file samples list
            if sample_type == 'single_stain_controls':
                self.experiment.samples['single_stain_controls'] += [self.current_sample_path]
            self.experiment.samples['all_samples'][self.current_sample_path] = sample_name
            self.experiment.samples['all_sample_nevents'][self.current_sample_path] = 0
            self.experiment.save()

            # initialise histograms and statistics
            self.load_sample(self.current_sample_path)

            if self.bus is not None:
                self.bus.sampleTreeUpdated.emit()
                self.bus.selectSample.emit(self.current_sample_path)
                # self.bus.selectSample.emit(str(Path(self.current_sample_path).relative_to(self.experiment_dir)))

    def batch_new_samples(self, sample_list, n_columns):
        if n_columns == 1:
            sample_dir = None
            for sample in sample_list:
                self.new_sample(sample, sample_dir)

        if n_columns == 2:
            for sample in sample_list:
                if sample[0].lower() in ['single_stain_controls', 'single stain controls']:
                    self.new_sample(sample[1], 'single_stain_controls')
                else:
                    sample_dir = Path(self.experiment.settings['raw']['raw_samples_subdirectory']) / sample[0]
                    (self.experiment_dir / sample_dir).mkdir(parents=True, exist_ok=True)
                    self.new_sample(sample[1], sample_dir)

        if n_columns == 3:
            for sample in sample_list:
                if sample[1].lower() in ['single_stain_controls', 'single stain controls']:
                    self.new_sample(sample[2], 'single_stain_controls')
                else:
                    sample_dir = Path(self.experiment.settings['raw']['raw_samples_subdirectory']) / sample[0] / sample[1]
                    (self.experiment_dir / sample_dir).mkdir(parents=True, exist_ok=True)
                    self.new_sample(sample[2], sample_dir)

    @with_busy_cursor
    def load_sample(self, sample_path):
        logger.info(f'Controller: loading sample {sample_path}')
        whitelisted_pnn = self.experiment.settings['raw'].get('whitelisted_pnn')
        check_pnn = whitelisted_pnn or self.experiment.settings['raw']['event_channels_pnn']
        if check_fcs_matches_experiment(self.experiment_dir / sample_path, check_pnn, self.experiment.settings['raw']['magnitude_ceiling']):
            self.current_sample_path = sample_path
            self.current_sample = sample_from_fcs(self.experiment_dir / self.current_sample_path, self.bus)

            if self.current_sample_path == self.live_sample_path:
                self.raw_event_data, n_events = self.copy_live_data(extent='update')
            else:
                try:
                    self.raw_event_data = self.current_sample.get_events(
                        source='raw', col_order=whitelisted_pnn
                    )
                except (KeyError, ValueError) as e:
                    logger.warning(
                        'load_sample: col_order get_events failed (%s) — reading all channels', e
                    )
                    self.raw_event_data = self.current_sample.get_events(source='raw')
                n_events = self.current_sample.event_count
                if np.any(np.isnan(self.raw_event_data)):
                    n_nan = int(np.isnan(self.raw_event_data).sum())
                    logger.warning('load_sample: %d NaN values in raw event data — replacing with 0', n_nan)
                    self.raw_event_data = np.where(np.isnan(self.raw_event_data), 0.0, self.raw_event_data)
                logger.debug('load_sample: raw_event_data shape %s', self.raw_event_data.shape)

                # Display cap — applied before unmixing so both arrays are consistently capped.
                _cap = settings.max_display_events
                if _cap and len(self.raw_event_data) > _cap:
                    _rng = np.random.default_rng(seed=42)
                    _idx = np.sort(_rng.choice(len(self.raw_event_data), _cap, replace=False))
                    self.raw_event_data = self.raw_event_data[_idx]
                    logger.info('load_sample: capped display events %d → %d', n_events, _cap)
                logger.debug('load_sample: raw_event_data shape %s', self.raw_event_data.shape)

            # apply spectral unmixing and compensation if defined
            if self.experiment.process['unmixing_matrix'] is not None:
                profiles_assigned = bool(
                    self.experiment.samples
                    .get('sample_af_profiles', {})
                    .get(sample_path)
                )
                self.initialise_af_matrices()
                if profiles_assigned and self.af_precomputed is None:
                    # Cache is still being built on the background thread
                    # (spectral process was just refreshed). AF correction
                    # will be active on the next load — reload once ready.
                    if self.bus:
                        self.bus.statusMessage.emit(
                            'AF matrices not yet cached — loaded with standard OLS. '
                            'Reload sample once caching completes.'
                        )
                self.unmixed_event_data = self._apply_unmixing(self.raw_event_data)

            if self.bus:
                self.bus.statusMessage.emit(f'Loaded sample {self.current_sample_path}: {n_events} events.')
                # self.bus.statusMessage.emit(f'{str(Path(self.current_sample_path).relative_to(self.experiment_dir))}: {n_events} events.')

            self.clear_data_for_cytometry_plots()
            self.initialise_data_for_cytometry_plots()
        else:
            if self.bus:
                self.bus.openImportFCSWidget.emit(True)
                QTimer.singleShot(500, lambda: self.bus.statusMessage.emit(f'Failed to load sample.'))

    @Slot()
    def reset_axes_reload_sample(self):
        logger.info(f"Controller: reloading sample")
        self.reset_axes_transforms_all()
        if self.current_sample_path:
            self.load_sample(self.current_sample_path)
            if self.bus:
                self.bus.statusMessage.emit(f'Axes reset, reloaded sample {self.current_sample_path}')

    def unload_sample(self):
        self.current_sample_path = None
        self.current_sample = None
        self.raw_event_data = None
        self.unmixed_event_data = None
        self.af_sidecar_data = None
        self.clear_data_for_cytometry_plots()
        self.initialise_data_for_cytometry_plots()
        logger.info(f"Controller: unloaded sample")

    def _reapply_fine_tuning_impl(self):
        self.initialise_transfer_matrix()
        if self.raw_event_data is not None:
            self.initialise_af_matrices()
            self.unmixed_event_data = self._apply_unmixing(self.raw_event_data)
            self.clear_data_for_cytometry_plots()
            self.initialise_data_for_cytometry_plots()

    # added with busy cursor since with AF integration this may be noticeable
    # only call this decorated entry point from the main thread (e.g. direct
    # UI callbacks). Anything already running inside another @with_busy_cursor
    # worker thread (e.g. initialise_ephemeral_data) must call
    # _reapply_fine_tuning_impl() directly — see note there.
    @with_busy_cursor
    def reapply_fine_tuning(self):
        self._reapply_fine_tuning_impl()

    @Slot()
    def on_gain_change(self, ch_name, value):
        logger.info(f"{ch_name} gain changed to {value}")


    def connect_instrument(self):
        self.pipe_connection_instrument.send({'command': 'connect'})
        response = self.pipe_connection_instrument.recv()
        #TODO self.view.display_instrument_status(response)
        logger.info(response)

    @Slot()
    def start_acquisition(self):
        # set live sample, switch current sample to live sample, prepare hists and stats
        if not self.current_sample or self.current_sample and self.current_sample.event_count != 0:
            # think of a name
            n = 0
            while True:
                n += 1
                sample_name = f'Sample{n}'
                if sample_name not in self.experiment.samples['all_samples'].values():
                    break
            # create sample in raw samples directory
            self.new_sample(sample_name, '')

        self.stop_live_data_processing.clear()
        self.live_sample_path = self.current_sample_path
        self.load_sample(self.live_sample_path) # this sets off thread for update_hists_and_stats through initialise_data_for_cytometry_plots call

        # start acquisition on instrument
        self.pipe_connection_instrument.send({'command': 'start'})
        response = self.pipe_connection_instrument.recv()
        logger.info(response)

        # start analyser (flush cache and start analysing incoming traces and producing events)
        self.pipe_connection_analyser.send({'command': 'start'})
        response = self.pipe_connection_analyser.recv()
        logger.info(response)

        if self.bus:
            self.bus.statusMessage.emit(f'Acquisition started')

        logger.info('Controller: acquisition started')

    @Slot()
    def stop_acquisition(self):
        # stop instrument
        self.pipe_connection_instrument.send({'command': 'stop'})
        response = self.pipe_connection_instrument.recv()
        logger.info(response)

        # stop analyser
        self.pipe_connection_analyser.send({'command': 'stop'})
        response = self.pipe_connection_analyser.recv()
        logger.info(response)

        if self.bus:
            self.bus.statusMessage.emit(f'Acquisition stopped')

        # stop live update thread
        self.stop_live_data_processing.set()
        time.sleep(0.25)
        while self.thread.is_alive():
            logger.info('Controller: waiting until live data processing is complete')
            time.sleep(0.25)
        self.thread.join()

        # save sample, load sample
        self.raw_event_data, _ = self.copy_live_data(extent='all')
        if self.raw_event_data is not None:
            self.current_sample_path = self.live_sample_path
            self.live_sample_path = None
            sample_name = Path(self.current_sample_path).stem
            self.current_sample = Sample(self.raw_event_data, sample_id=sample_name, channel_labels=self.experiment.settings['raw']['event_channels_pnn'])
            self.current_sample.metadata['tubename'] = sample_name
            self.current_sample.metadata['flow_rate'] = str(61.234) #todo get this flow rate (float)
            self.current_sample.export(self.experiment_dir / self.current_sample_path, source='raw', include_metadata=True)
            self.load_sample(self.current_sample_path)
        else:
            if self.bus:
                self.bus.statusMessage.emit(f'No data acquired')

        # empty oscilloscope traces queue
        removed = empty_queue_nowait(self.oscilloscope_traces_queue)
        logger.info(f"Controller: emptied oscilloscope traces queue ({removed} traces)")

        # update experiment file samples list
        self.experiment.samples['all_sample_nevents'][self.current_sample_path] = self.current_sample.event_count
        self.experiment.save()
        if self.bus is not None:
            self.bus.sampleTreeUpdated.emit()



    def update_instrument_settings(self):
        self.pipe_connection_instrument.send({'command': 'set', 'data': 'TODO insert settings update here'})
        response = self.pipe_connection_instrument.recv()

        logger.info(response)
        # TODO self.view.display_instrument_status(response)

    def quit_instrument_quit_analyser(self):
        self.pipe_connection_analyser.send({'command': 'quit'})  # quit analyser
        response = self.pipe_connection_analyser.recv()
        logger.info(response)

        self.pipe_connection_instrument.send({'command': 'quit'})  # quit instrument
        response = self.pipe_connection_instrument.recv()
        logger.info(response)

    def copy_live_data(self, extent='all'):
        with self.index_head_events_cache.get_lock():
            events_head = self.index_head_events_cache.value
        with self.index_tail_events_cache.get_lock():
            events_tail = self.index_tail_events_cache.value  # read only here - this is updated by trace analyser process

        n_new_events = events_tail - events_head

        if extent == 'all':
            start = 0
        else:  # extent == 'update'
            start = events_head

        if events_tail > start:
            with self.events_cache_lock:
                logger.info(f'Controller: copying live data {[start, events_tail]}')
                data = self.events_cache[start:events_tail, :].astype(np.float64)
                data[:,self.experiment.settings['raw']['time_channel_id']] /= 1000 #convert to seconds

            # update head of traces cache and tail of events cache
            events_head_new = events_tail
            logger.info(f'Controller: processed {n_new_events} events (events cache head:{events_head}, tail:{events_tail})')

            with self.index_head_events_cache.get_lock():
                self.index_head_events_cache.value = events_head_new
        else:
            data = None
            logger.info(f'Controller: awaiting events (events cache head:{events_head}, tail:{events_tail})')

        return data, n_new_events

    @with_busy_cursor
    @Slot(str)
    def set_mode(self, tab_name):
        # select set of plots: raw, process or unmixed
        logger.info(f"Controller: set mode to {tab_name}")
        self.mode_switch_in_progress = True
        try:
            if tab_name == 'Raw Data':
                self.current_mode = 'raw'
                self.data_for_cytometry_plots = self.data_for_cytometry_plots_raw
            elif tab_name == 'Spectral Process':
                self.current_mode = 'process'
                self.data_for_cytometry_plots_process['histograms'] = self.data_for_cytometry_plots_unmixed['histograms']
                self.data_for_cytometry_plots_process['statistics'] = self.data_for_cytometry_plots_unmixed['statistics']
                self.data_for_cytometry_plots = self.data_for_cytometry_plots_process
            elif tab_name == 'Unmixed Data':
                self.current_mode = 'unmixed'
                self.data_for_cytometry_plots = self.data_for_cytometry_plots_unmixed
            elif tab_name == 'Statistics':
                self.current_mode = 'statistics'
                self.data_for_cytometry_plots = self.data_for_cytometry_plots_unmixed
            else:
                # catch all for plugin tabs
                self.current_mode = tab_name
                self.data_for_cytometry_plots = None
                return

            self.initialise_data_for_cytometry_plots()
        finally:
            self.mode_switch_in_progress = False

    def clear_data_for_cytometry_plots(self):
        for data in [self.data_for_cytometry_plots_raw, self.data_for_cytometry_plots_process, self.data_for_cytometry_plots_unmixed]:
            data['histograms'] = []
            data['statistics'] = {}

    def initialise_data_for_cytometry_plots(self, force_recalc_histograms=False):
        # called on tab change (set mode), load sample, reset axes transforms all, refresh spectral process, initalise nxn grid
        # calc all hists and stats if they do not already exist or force not set
        # force only necessary for reset axes transforms all and refresh spectral process

        if self.data_for_cytometry_plots:
            # make sure data for cytometry plots is pointing at correct data
            self.data_for_cytometry_plots_raw.update({'event_data': self.raw_event_data})
            self.data_for_cytometry_plots_process.update({'event_data': self.unmixed_event_data})
            self.data_for_cytometry_plots_unmixed.update({'event_data': self.unmixed_event_data})

            # recalculate everything if it isn't already present
            if force_recalc_histograms or not self.data_for_cytometry_plots['statistics'] or not self.data_for_cytometry_plots['histograms']:
                if self.bus:
                    self.bus.statusMessage.emit(f'Initialising plots and gating statistics...')

                # for default transformations, set limits to observed data
                if self.data_for_cytometry_plots['event_data'] is not None and len(self.data_for_cytometry_plots['event_data']):
                    if self.current_sample_path != self.live_sample_path:
                        for label in (self.data_for_cytometry_plots['transformations'] or {}):
                            transformation = self.data_for_cytometry_plots['transformations'][label]
                            if transformation.id == 'default':
                                index = self.data_for_cytometry_plots['pnn'].index(label)
                                upper_limit = max(self.data_for_cytometry_plots['event_data'][:, index]) * 1.05
                                transformation.set_transform(limits=[0, upper_limit])

                # Per-sample templates: if this sample is assigned a different
                # template than the active one, switch the active gating to it.
                self.ensure_template_for_current_sample()
                self._emit_templates_changed()

                # Per-sample gating: resolve any dynamic (e.g. Time) gate dimensions
                # for this sample and rebuild just those gates' lookup tables, so a
                # single template adapts to each sample's own channel range.
                self.apply_dynamic_gate_dimensions()

                self.data_for_cytometry_plots['statistics'] = initialise_stats(self.data_for_cytometry_plots['gating'])
                self.data_for_cytometry_plots['histograms'] = initialise_hists(self.data_for_cytometry_plots['plots'], self.data_for_cytometry_plots)

                # initialise plots
                if self.bus and self.data_for_cytometry_plots['plots']:
                    self.bus.statusMessage.emit(f'Calculating {len(self.data_for_cytometry_plots['plots'])} histograms...')
                self.calc_hists_and_stats(status_message_signal=(self.bus.statusMessage if self.bus else None))

                logger.info(f'Controller: prepared hists and stats, mode: {self.current_mode}')

                # A pending template switch rebuilds the plot grid + tree AFTER
                # histograms/stats are computed, so the fresh plot widgets see
                # consistent data (their plot set differs from the old template).
                # We then re-emit histsStatsRecalculated so the just-built widgets
                # draw the already-computed (consistent) histograms — building
                # before compute would let them draw stale/mismatched data.
                if self._pending_template_scope is not None:
                    scope = self._pending_template_scope
                    self._pending_template_scope = None
                    if self.bus is not None:
                        self.bus.templateApplied.emit(scope)
                        self.bus.histsStatsRecalculated.emit(
                            self.current_mode,
                            list(range(len(self.data_for_cytometry_plots['plots'] or []))),
                        )

                if self.bus:
                    self.bus.statusMessage.emit(f'Ready.')

                # then if sample is live, start thread to calc hist and stats on updates... or calc once only
                if not self.stop_live_data_processing.is_set() and self.current_sample_path == self.live_sample_path:
                    # start live update thread
                    self.thread = threading.Thread(target=self.update_hists_and_stats, args=(), daemon=True)
                    self.thread.start()

    @with_busy_cursor
    @Slot()
    def reinitialise_data_for_process_plots(self):
        if self.current_mode == 'process':
            self.data_for_cytometry_plots.update({'event_data': self.unmixed_event_data})
            self.data_for_cytometry_plots['histograms'] = initialise_hists(self.data_for_cytometry_plots['plots'], self.data_for_cytometry_plots)
            self.calc_hists_and_stats()
            logger.info('Controller: prepared hists for process plots')

    @Slot(str, str)
    def create_new_plot(self, channel_x, channel_y):
        pnn = self.data_for_cytometry_plots['pnn']
        plots = self.data_for_cytometry_plots['plots']
        if channel_x in pnn and channel_y in pnn and channel_x != channel_y: # 2d hist
            new_plot = {'type': 'hist2d', 'channel_x': channel_x, 'channel_y': channel_y, 'source_gate': 'root', 'child_gates': []}
        elif channel_x in pnn and (channel_y == 'Count' or channel_y == channel_x or channel_y is None): # 1d hist
            new_plot = {'type': 'hist1d', 'channel_x': channel_x, 'source_gate': 'root', 'child_gates': []}
        else: # ribbon
            new_plot = {'type': 'ribbon', 'source_gate': 'root', 'child_gates': [], 'width': 3}
        plots.append(new_plot)
        if self.bus is not None:
            self.bus.showNewPlot.emit(self.current_mode)
            logger.info(f'Controller: signal emitted showNewPlot for plot={len(plots)-1}')

        self.data_for_cytometry_plots['histograms'] += initialise_hists([new_plot], self.data_for_cytometry_plots)
        indices_plots_to_calculate = [len(plots)-1]
        self.calc_hists_and_stats(indices_plots_to_calculate=indices_plots_to_calculate)

        logger.info(f'Controller: created plot {new_plot}')


    @Slot(str, int)
    def change_plot(self, mode, n_in_plot_sequence):
        if mode == self.current_mode:
            plots_to_recalculate = [self.data_for_cytometry_plots['plots'][n_in_plot_sequence]]
            hist = initialise_hists(plots_to_recalculate, self.data_for_cytometry_plots)[0]
            self.data_for_cytometry_plots['statistics'] = initialise_stats(self.data_for_cytometry_plots['gating'])
            if n_in_plot_sequence < len(self.data_for_cytometry_plots['histograms']):
                self.data_for_cytometry_plots['histograms'][n_in_plot_sequence] = hist
            else:
                self.data_for_cytometry_plots['histograms'].append(hist)

            self.calc_hists_and_stats(indices_plots_to_calculate=[n_in_plot_sequence])
            logger.info(f'Controller: changed plot {n_in_plot_sequence}')

    @Slot(str)
    def recalc_after_axis_transform(self, channel):
        # recalculate histograms, gates and stats
        # which plots have changed?
        indices_plots_to_recalculate = []
        child_gates = []
        for n, plot in enumerate(self.data_for_cytometry_plots['plots']):
            if plot['type'] == 'hist1d':
                if channel == plot['channel_x']:
                    indices_plots_to_recalculate.append(n)
                    child_gates += plot['child_gates']
            elif plot['type'] == 'hist2d':
                if channel == plot['channel_x'] or channel == plot['channel_y']:
                    indices_plots_to_recalculate.append(n)
                    child_gates += plot['child_gates']
            else:  # 'ribbon' # only recalculate if we are actually adjusting the ribbon transform
                if channel == 'ribbon':
                    indices_plots_to_recalculate.append(n)
        gates_to_recalculate = list(set(child_gates))

        gate_ids = self.data_for_cytometry_plots['gating'].get_gate_ids()
        for gate_id in gate_ids:
            if any([ancestor in child_gates for ancestor in gate_id[1]]):
                gates_to_recalculate.append(gate_id[0])

        plots_to_recalculate = [self.data_for_cytometry_plots['plots'][n] for n in indices_plots_to_recalculate]
        hists = initialise_hists(plots_to_recalculate, self.data_for_cytometry_plots)
        for m, n in enumerate(indices_plots_to_recalculate):
            self.data_for_cytometry_plots['histograms'][n] = hists[m]

        self.data_for_cytometry_plots['statistics'] = initialise_stats(self.data_for_cytometry_plots['gating'])
        self.calc_hists_and_stats(gates_to_calculate=gates_to_recalculate, indices_plots_to_calculate=indices_plots_to_recalculate)
        logger.info(f'Controller: plots recalculated {indices_plots_to_recalculate}')
        logger.info(f'Controller: gates recalculated {gates_to_recalculate}')
        # todo update all child gates too, perhaps within cytometryplotwidget

    @Slot(list)
    def reset_axes_transforms(self, channels):
        if self.current_mode in ('raw',):
            settings = self.experiment.settings['raw']
        elif self.current_mode in ('process', 'unmixed', 'statistics'):
            settings = self.experiment.settings['unmixed']
        else:
            settings = None

        transforms = assign_default_transforms(settings, channels=channels)
        transformations = generate_transformations(transforms)

        for channel in channels:
            # for default transformations, set limits to observed data
            if self.current_sample_path != self.live_sample_path:
                if transformations[channel].id == 'default':
                    index = self.data_for_cytometry_plots['pnn'].index(channel)
                    upper_limit = max(self.data_for_cytometry_plots['event_data'][:, index]) * 1.05
                    transformations[channel].set_transform(limits=[0, upper_limit])

            self.data_for_cytometry_plots['transformations'][channel] = transformations[channel]
            self.recalc_after_axis_transform(channel)
        logger.info(f'Controller: channels reset {channels}')

    def reset_axes_transforms_all(self):
        # currently used only for changing default range in experiment settings
        settings = self.experiment.settings['raw']
        raw_channels = settings.get('whitelisted_pnn') or settings['event_channels_pnn']
        transforms = assign_default_transforms(settings, channels=raw_channels)
        transformations = generate_transformations(transforms)
        self.raw_transformations.update(transformations)

        if self.experiment.process['unmixing_matrix']:
            settings = self.experiment.settings['unmixed']
            transforms = assign_default_transforms(settings)
            transformations = generate_transformations(transforms)
            self.unmixed_transformations.update(transformations)

        self.initialise_data_for_cytometry_plots(force_recalc_histograms=True) #todo check: does this reset all axes in all plots?

    def update_hists_and_stats(self):
        # update thread, calculate hists and stats
        logger.info('Controller: live update hists and stats started')
        last_update_time = time.perf_counter()
        while True:
            if self.stop_live_data_processing.is_set() or self.current_sample_path != self.live_sample_path:
                logger.info('Controller: live update hists and stats stopped')
                break

            # copy live data
            self.raw_event_data, n_new_events = self.copy_live_data(extent='update')
            self.data_for_cytometry_plots['event_data'] = self.raw_event_data

            # apply spectral unmixing and compensation if defined
            if self.current_mode == 'raw' and self.experiment.process['unmixing_matrix'] is not None:
                self.unmixed_event_data = apply_transfer_matrix(self.transfer_matrix, self.raw_event_data)
                self.data_for_cytometry_plots['event_data'] = self.unmixed_event_data

            self.calc_hists_and_stats()

            # Calculate elapsed time and sleep precisely
            new_update_time = time.perf_counter()
            elapsed = new_update_time - last_update_time
            last_update_time = new_update_time

            live_events_per_second = int(n_new_events / elapsed)
            if self.bus:
                self.bus.statusMessage.emit(f'Live acquisition rate {live_events_per_second} events/s')
            time.sleep(live_data_process_repeat_time)

    def calc_hists_and_stats(self, gates_to_calculate=None, indices_plots_to_calculate=None, status_message_signal=None):
        # guard first against this being reached by wrong path
        if self.data_for_cytometry_plots is None:
            return
        if self.data_for_cytometry_plots['event_data'] is not None:
            # apply gates to event data
            # if gates_to_calculate is none, then initialise gates_membership dict, otherwise reference it from data_for_cytometry_plots
            if not gates_to_calculate:
                gating = self.data_for_cytometry_plots['gating']
                if gating is None:
                    return
                # Full recalculation: rebuild lookup tables ONLY for gates that use
                # a per-sample (default, e.g. Time) transform, so they match this
                # sample's scale. Fixed-transform gates (fluorescence, FSC/SSC) keep
                # their cached table — rebuilding everything here was slow (laggy).
                self._rebuild_per_sample_lookup_tables(self._scope_for_mode())
                gate_membership = {'root': np.ones(len(self.data_for_cytometry_plots['event_data']), dtype=np.bool_)}
                self.data_for_cytometry_plots.update({'gate_membership': gate_membership})
                # self.data_for_cytometry_plots['gate_membership']['root'] = np.ones(len(self.data_for_cytometry_plots['event_data']), dtype=np.bool_)
                gates_to_calculate = [g[0] for g in self.data_for_cytometry_plots['gating'].get_gate_ids()]
            apply_gates_in_place(self.data_for_cytometry_plots, gates_to_calculate=gates_to_calculate)
            statistics = calc_stats(self.data_for_cytometry_plots)
            self.data_for_cytometry_plots['statistics'] = statistics

            hists = calc_hists(self.data_for_cytometry_plots,
                               indices_plots_to_calculate=indices_plots_to_calculate,
                               status_message_signal=status_message_signal,
                               density_cutoff=settings.density_cutoff_retrieved,
                               dot_plot_by_gate=settings.hist2dtype_retrieved=='Dot plot coloured by gate')
            if indices_plots_to_calculate is None:
                indices_plots_to_calculate = list(range(len(self.data_for_cytometry_plots['plots'])))

            # calc_hists skips plots whose source_gate is missing from
            # gate_membership, so hists may be shorter than indices_plots_to_calculate.
            # Zip stops at the shorter sequence, preventing an IndexError.
            for n, hist in zip(indices_plots_to_calculate, hists):
                if self.data_for_cytometry_plots['histograms'][n].shape == hist.shape:
                    self.data_for_cytometry_plots['histograms'][n] += hist
                else:
                    logger.warning(f'Controller: calc_hists_and_stats: histogram shape mismatch for plot {n} — discarding stale result')

            if self.bus is not None:
                self.bus.histsStatsRecalculated.emit(self.current_mode, indices_plots_to_calculate)
                logger.info(f'Controller: signal emitted histStatsRecalculated for plots={indices_plots_to_calculate}')

    def create_or_update_gate(self, gate_name=None, gate_type=None, gate_path=None, gate_data=None, channel_x=None, channel_y=None):
        '''
        creates or updates a gate (name, type, path, data, channel x, channel y)
        no longer used by gui - only for testing
        '''
        gating = self.data_for_cytometry_plots['gating']
        transformations = self.data_for_cytometry_plots['transformations']
        gate_paths = gating.find_matching_gate_paths(gate_name)
        if len(gate_paths) == 0:
            create = True
        else:
            create = False


        if gate_type == 'quad':
            x = gate_data['pos'][0]
            y = gate_data['pos'][1]

            quad_divs, quadrants = define_quad_gates(x, y, channel_x, channel_y, transformations)
            if create:
                gate = gates.QuadrantGate(gate_name, dividers=quad_divs, quadrants=quadrants)
                gating.add_gate(gate, gate_path=gate_path)
            else:
                gate = gating.get_gate(gate_name)
                gate.quadrants = {q.id: q for q in quadrants}

            logger.info(f'Quad gate dividers: {[q[1]._divider_ranges for q in gate.quadrants.items()]}')

        elif gate_type == 'range':
            x1 = gate_data['x1']
            x2 = gate_data['x2']

            dim_x = define_range_gate(x1, x2, channel_x, transformations)
            if create:
                gate = gates.RectangleGate(gate_name, dimensions=[dim_x])
                gating.add_gate(gate, gate_path=gate_path)
            else:
                gate = gating.get_gate(gate_name)
                gate.dimensions = [dim_x]

            logger.info(str([gate, gate.dimensions, gate.dimensions[0].min, gate.dimensions[0].max]))

        elif gate_type == 'polygon':
            origin = gate_data['origin']
            points = gate_data['points']
            points = [(p[0]+origin[0], p[1]+origin[1]) for p in points]

            vertices, dim_x, dim_y = define_polygon_gate(points, channel_x, channel_y, transformations)

            if create:
                gate = gates.PolygonGate(gate_name, [dim_x, dim_y], vertices, use_complement=False)
                gating.add_gate(gate, gate_path=gate_path)
            else:
                gate = gating.get_gate(gate_name)
                gate.vertices = vertices

            logger.info(str([gate, gate.vertices]))

        elif gate_type == 'rectangle':
            pos = np.array(gate_data['pos'])
            size = np.array(gate_data['size'])

            dim_x, dim_y = define_rectangle_gate(pos, size, channel_x, channel_y, transformations)

            if create:
                gate = gates.RectangleGate(gate_name, dimensions=[dim_x, dim_y])
                gating.add_gate(gate, gate_path=gate_path)
            else:
                gate = gating.get_gate(gate_name)
                gate.dimensions = [dim_x, dim_y]

            logger.info(str([gate, gate.dimensions, gate.dimensions[0].min, gate.dimensions[0].max, gate.dimensions[1].min,
                   gate.dimensions[1].max]))


        elif gate_type == 'ellipse':
            pos = gate_data['pos']
            size = gate_data['size']
            angle = gate_data['angle']

            dim_x, dim_y, coordinates, covariance_matrix, distance_square = define_ellipse_gate(pos, size, angle, channel_x, channel_y, transformations)

            if create:
                gate = gates.EllipsoidGate(gate_name, [dim_x, dim_y], coordinates, covariance_matrix, distance_square)
                gating.add_gate(gate, gate_path=gate_path)
            else:
                gate = gating.get_gate(gate_name)
                gate.coordinates = coordinates
                gate.covariance_matrix = covariance_matrix
                gate.distance_square = distance_square

            logger.info(str([gate, gate.coordinates, gate.covariance_matrix, gate.distance_square]))

        logger.info(gating.get_gate_ids())

    @Slot(str, tuple)
    def update_child_gate_label_offset(self, gate_name, label_offset):
        for plot in self.data_for_cytometry_plots['plots']:
            if gate_name in plot['child_gates']:
                get_set_or_initialise_label_offset(plot, gate_name, label_offset)

    def regenerate_spectral_model(self):
        from honeychrome.controller_components.spectral_controller import SpectralAutoGenerator
        spectral_auto_generator = SpectralAutoGenerator(self.bus, self)
        spectral_auto_generator.run()

    @with_busy_cursor
    @Slot()
    def refresh_spectral_process(self):
        from honeychrome.controller_components.spectral_functions import calculate_spectral_process

        # first build unmixed part of experiment: settings, process and cytometry
        raw_settings = self.experiment.settings['raw']
        spectral_model = self.experiment.process['spectral_model']
        profiles = self.experiment.process['profiles']

        # Sort spectral model by major-channel position so fluorescence_channels
        # and all downstream structures (NxN grid, heatmaps) use the correct order.
        if spectral_model:
            event_channels_pnn = raw_settings['event_channels_pnn']
            self.filter_raw_fluorescence_channels()
            fluor_pnn = [event_channels_pnn[i] for i in self.filtered_raw_fluorescence_channel_ids]
            channel_order = {ch: i for i, ch in enumerate(fluor_pnn)}
            spectral_model.sort(
                key=lambda c: channel_order.get(c.get('gate_channel') or '', len(channel_order))
            )

        spectral_model_valid = (
                set(profiles.keys()) == set([control['label'] for control in spectral_model])
                and len(profiles.keys()) == len(spectral_model)
                and all(profiles.keys())
        )
        if spectral_model and spectral_model_valid:
            if self.bus:
                self.bus.statusMessage.emit(f'Refreshing spectral process...')
            existing_spillover = self.experiment.process.get('spillover')
            unmixing_method = self.experiment.settings.get('unmixing_method', 'OLS')
            unmixed_settings, spectral_process = calculate_spectral_process(
                raw_settings, spectral_model, profiles,
                existing_spillover=existing_spillover,
                unmixing_method=unmixing_method,
                experiment_dir=self.experiment_dir,
                experiment_samples=self.experiment.samples,
                filtered_fluorescence_channel_ids=self.filtered_raw_fluorescence_channel_ids,
            )
            self.experiment.process.update(spectral_process)

            # update cytometry only if channels have changed
            if self.experiment.settings['unmixed']['event_channels_pnn'] != unmixed_settings['event_channels_pnn']:
                self.experiment.settings['unmixed'].update(unmixed_settings)

                # set up unmixed channels with default transforms, copy raw transformation if it does not belong to a fl channel
                self.experiment.cytometry['transforms'] = assign_default_transforms(unmixed_settings)
                fl_pnn = [self.experiment.settings['raw']['event_channels_pnn'][n] for n in self.experiment.settings['raw']['fluorescence_channel_ids']]
                update_transforms(self.experiment.cytometry['raw_transforms'], self.raw_transformations)
                for label in self.experiment.cytometry['raw_transforms']:
                    if label not in fl_pnn and label in self.experiment.cytometry['transforms']:
                        self.experiment.cytometry['transforms'][label].update(self.experiment.cytometry['raw_transforms'][label])

                # copy all non-fl gates from raw
                unmixed_gating = GatingStrategy()
                unmixed_transformations = generate_transformations(self.experiment.cytometry['transforms'])
                for label in self.experiment.settings['unmixed']['event_channels_pnn']:
                    unmixed_gating.transformations[label] = unmixed_transformations[label].xform

                gate_ids = self.raw_gating.get_gate_ids()
                for gate_id in gate_ids:
                    if self.raw_gating._get_gate_node(gate_id[0], gate_id[1]).gate_type != 'Quadrant':  # bit of a hack. Can't find a better way of excluding Quadrant
                        gate = self.raw_gating.get_gate(gate_id[0])
                        dimension_ids = gate.get_dimension_ids()
                        if all([dim in self.data_for_cytometry_plots_raw['pnn'] for dim in dimension_ids]):
                            if not any([self.data_for_cytometry_plots_raw['pnn'].index(dim) in self.data_for_cytometry_plots_raw['fluoro_indices'] for dim in dimension_ids]) and gate.gate_name not in ['Pos Unstained', 'Neg Unstained']:
                                unmixed_gate_names = ['root'] + [g[0] for g in unmixed_gating.get_gate_ids()]
                                if gate_id[1][-1] in unmixed_gate_names:
                                    unmixed_gating.add_gate(gate, gate_path=gate_id[1])
                self.experiment.cytometry['gating'] = to_gml(unmixed_gating)
                unmixed_gate_names = [g[0] for g in unmixed_gating.get_gate_ids()]

                # copy all non-fl plots from raw
                self.experiment.cytometry['plots'] = []
                for plot in self.data_for_cytometry_plots_raw['plots']:
                    if plot['type'] == 'hist1d':
                        if self.data_for_cytometry_plots_raw['pnn'].index(plot['channel_x']) in self.data_for_cytometry_plots_raw['fluoro_indices']:
                            continue
                    elif plot['type'] == 'hist2d':
                        if (self.data_for_cytometry_plots_raw['pnn'].index(plot['channel_x']) in self.data_for_cytometry_plots_raw['fluoro_indices']
                        or self.data_for_cytometry_plots_raw['pnn'].index(plot['channel_y']) in self.data_for_cytometry_plots_raw['fluoro_indices']):
                            continue
                    elif plot['type'] == 'ribbon':
                        continue

                    new_plot = deepcopy(plot)
                    if new_plot['source_gate'] not in unmixed_gate_names:
                        new_plot['source_gate'] = 'root'
                    EXCLUDED_GATES = {'Pos Unstained', 'Neg Unstained'}
                    new_plot_child_gates = []
                    for gate in new_plot['child_gates']:
                        if gate in unmixed_gate_names and gate not in EXCLUDED_GATES:
                            new_plot_child_gates.append(gate)
                    new_plot['child_gates'] = new_plot_child_gates
                    self.experiment.cytometry['plots'].append(new_plot)

                    if self.bus:
                        self.bus.changedGatingHierarchy.emit('unmixed', 'root')
        else:
            unmixed_settings = settings_default['unmixed'].copy()
            self.experiment.settings['unmixed'].update(unmixed_settings)

            self.experiment.process.update({'similarity_matrix': None, 'hotspot_matrix': None, 'unmixing_matrix': None, 'spillover': None})
            self.experiment.cytometry['plots'] = []
            self.experiment.cytometry['transforms'] = None
            self.experiment.cytometry['gating'] = None
            self.unmixed_event_data = None
            self.af_sidecar_data = None

        # Rebuild AF cache synchronously before reinitialising ephemeral data.
        self.cache_all_af_profiles()
        # then reinitialise ephemeral data for process and unmixed tabs
        self.initialise_ephemeral_data(scope=['unmixed'])
        # Do not recalculate histograms here: the unmixing matrix has changed so
        # any result would be immediately stale. Histograms are cleared below and
        # will be regenerated correctly on the next sample load or tab change.
        self.data_for_cytometry_plots_unmixed['histograms'].clear()
        self.data_for_cytometry_plots_unmixed['statistics'].clear()
        if self.bus is not None:
            self.bus.spectralProcessRefreshed.emit()
            # self.bus.changedGatingHierarchy.emit('unmixed', 'root')
            self.bus.statusMessage.emit(f'Spectral process refreshed.')

        logger.info(f'Controller: refreshed spectral process, unmixed settings, unmixed cytometry')



if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method("spawn")
    from multiprocessing import Lock

    '''
    1.
    open saved experiment
    open a sample
    view a sample
    calculate statistics
    save statistics

    -Calculates all histograms
    -Applies unmixing
    -Applies compensation
    -Applies gates
    -Calculates masks
    -Calculates stats
    '''

    kc = Controller()
    base_directory = Path.home() / 'spectral_cytometry'
    experiment_name = base_directory / '20240620 Spectral Symposium-poor cell unmixed'
    experiment_path = experiment_name.with_suffix('.kit')
    kc.load_experiment(experiment_path)

    # # Auto generate spectral model on loaded experiment, unmix, add plots
    kc.regenerate_spectral_model()
    kc.refresh_spectral_process()

    # note every time tab is changed (or every time plots or transforms changed), calculate all histograms and statistics
    kc.set_mode('Unmixed Data')
    kc.initialise_data_for_cytometry_plots()

    # add gates
    kc.create_or_update_gate(gate_name='activated', gate_type='rectangle', gate_path=('root', 'Cells', 'Singlets'), gate_data={'pos':[0.2, 0.2], 'size':[0.5, 0.5]}, channel_x='A2 Spark UV 387', channel_y='SSC-A')
    kc.create_or_update_gate(gate_name='quaddy', gate_type='quad', gate_path=('root', 'Cells', 'Singlets'), gate_data={'pos':[0.2, 0.2], 'size':[0.5, 0.5]}, channel_x='A2 Spark UV 387', channel_y='A10 BUV805')
    kc.create_or_update_gate(gate_name='sub++', gate_type='ellipse', gate_path=('root', 'Cells', 'Singlets', 'quaddy', 'A2 Spark UV 387+ A10 BUV805+'), gate_data={'pos':[0.2, 0.2], 'size':[0.2, 0.2], 'angle':30}, channel_x='A12 eFluor 450', channel_y='A10 BUV805')

    kc.calculate_lookup_tables()  # (re)create all lookup tabels

    print('Test gating unmixed', kc.unmixed_gating.get_gate_ids())
    print('Test gating unmixed', kc.unmixed_gating.get_gate_hierarchy())

    # add plots
    kc.data_for_cytometry_plots['plots'] += [
        {'type': 'hist2d', 'channel_x': 'A2 Spark UV 387', 'channel_y': 'SSC-A', 'source_gate': 'Singlets', 'child_gates': ['activated']},
        {'type': 'hist2d', 'channel_x': 'A2 Spark UV 387', 'channel_y': 'A10 BUV805', 'source_gate': 'Singlets', 'child_gates': ['quaddy']},
        {'type': 'hist2d', 'channel_x': 'A12 eFluor 450', 'channel_y': 'A10 BUV805', 'source_gate': 'A2 Spark UV 387+ A10 BUV805+', 'child_gates': ['sub++']}
    ]

    kc.set_mode('Raw Data')
    kc.load_sample(list(kc.experiment.samples['all_samples'])[0])

    print('Test statistics raw', kc.data_for_cytometry_plots_raw['statistics'])
    kc.set_mode('Unmixed Data')
    print('Test statistics unmixed', kc.data_for_cytometry_plots_unmixed['statistics'])

    # note every time tab is changed (or every time plots or transforms changed), calculate all histograms and statistics
    kc.set_mode('Raw Data')
    kc.new_sample('test', 'single_stain_controls')

    experiment_name = base_directory / '20240620 Spectral Symposium-poor cell unmixed'
    experiment_path = experiment_name.with_suffix('.kit')
    kc.save_experiment(experiment_path) # this saves as template (without samples)

    print('Test 1: pass')

    #1b. test whether loaded experiment is the same as current experiment
    test_experiment = ExperimentModel()
    test_experiment.load(experiment_path)
    test_experiment.scan_sample_tree() # this reloads samples

    from deepdiff import DeepDiff
    diff = DeepDiff(kc.experiment,test_experiment,ignore_order=True, exclude_paths='root.progress_indicator')
    if diff == {}:
        print('Test 1b: success')
    else:
        print('Test 1b: failure')
        print(diff)

    test_experiment.save()

    pass

    '''
    2. 
    new experiment
    create sample
    receive live data
    update thread, calculate hists and stats
    save sample

    Creates live sample and carries out live analysis
    -Receives start acquisition signal
    -Initialises histograms
    -Listens for new chunk, updates histogram, updates stats
    -Runs spectral model process
    -Runs unmixing
    '''

    # Allocate shared memory block, plus head and tail indices
    traces_cache_shm = shared_memory.SharedMemory(create=True, size=np.zeros(traces_cache_size, dtype=traces_cache_dtype).nbytes)
    traces_cache_lock = Lock()
    index_head_traces_cache = mp.Value('i', 0)
    index_tail_traces_cache = mp.Value('i', 0)

    events_cache_shm = shared_memory.SharedMemory(create=True,
                                                  size=np.zeros((max_events_in_cache, n_channels_per_event),
                                                                dtype=np.int64).nbytes)
    events_cache_lock = Lock()
    index_head_events_cache = mp.Value('i', 0)
    index_tail_events_cache = mp.Value('i', 0)

    # oscilloscope traces
    oscilloscope_traces_queue = mp.Queue()
    # command pipes
    pipe_experiment_instrument_e, pipe_experiment_instrument_i = mp.Pipe()
    pipe_experiment_analyser_e, pipe_experiment_analyser_a = mp.Pipe()

    '''
    Firstly, set up experiment controller
    '''
    kc = Controller(
            events_cache_name=events_cache_shm.name,
            events_cache_lock=events_cache_lock,
            index_head_events_cache=index_head_events_cache,
            index_tail_events_cache=index_tail_events_cache,
            oscilloscope_traces_queue=oscilloscope_traces_queue,
            pipe_connection_instrument=pipe_experiment_instrument_e,
            pipe_connection_analyser=pipe_experiment_analyser_e)

    base_directory = Path.home() / experiments_folder
    experiment_name = base_directory / 'Test experiment from new'
    experiment_path = experiment_name.with_suffix('.kit')
    kc.new_experiment(experiment_path) # this autosaves the experiment
    kc.set_mode('Raw Data')

    '''
    Then, set up instrument
    '''
    # start instrument dummy
    from honeychrome.instrument_communicator import Instrument

    instrument = Instrument(use_dummy_instrument=True, traces_cache_name=traces_cache_shm.name,
        traces_cache_lock=traces_cache_lock, index_head_traces_cache=index_head_traces_cache,
        index_tail_traces_cache=index_tail_traces_cache, pipe_connection=pipe_experiment_instrument_i)
    instrument.start()

    '''
    Then, set up analyst
    '''
    from honeychrome.trace_analyst import TraceAnalyser

    trace_analyser = TraceAnalyser(traces_cache_name=traces_cache_shm.name, traces_cache_lock=traces_cache_lock,
        index_head_traces_cache=index_head_traces_cache, index_tail_traces_cache=index_tail_traces_cache,
        events_cache_name=events_cache_shm.name, events_cache_lock=events_cache_lock,
        index_head_events_cache=index_head_events_cache, index_tail_events_cache=index_tail_events_cache,
        oscilloscope_traces_queue=oscilloscope_traces_queue,
        pipe_connection=pipe_experiment_analyser_a)
    trace_analyser.start()

    '''
    Then, send commands and read data

    connect instrument
    start instrument
    start analyser
    stop analyser
    set channels
    start analyser again
    stop instrument
    stop analyser
    quit analyser
    quit instrument
    '''
    # connect instrument
    kc.connect_instrument()

    # create blank sample
    kc.new_sample('A0 Label (Cells)', 'single_stain_controls') # this autosaves the empty sample and experiment again

    # acquire data
    kc.start_acquisition()
    # wait for a bit
    time.sleep(2)
    kc.stop_acquisition()

    # end processes, free memory
    kc.quit_instrument_quit_analyser()
    trace_analyser.join()
    instrument.join()
    traces_cache_shm.close()
    events_cache_shm.close()
    traces_cache_shm.unlink()
    events_cache_shm.unlink()

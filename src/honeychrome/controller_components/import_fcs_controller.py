import warnings
from copy import deepcopy
import re

import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer
from flowkit import GatingStrategy, Dimension, gates
from flowio import FlowData

from honeychrome.controller_components.functions import timer, apply_gates_in_place, apply_transfer_matrix, calc_stats, all_same, assign_default_transforms, generate_transformations
from honeychrome.controller_components.gml_functions_mod_from_flowkit import to_gml
from honeychrome.settings import settings_default, process_default, cytometry_default, af_channels
from honeychrome.view_components.busy_cursor import with_busy_cursor
from honeychrome.controller_components.cytometer_whitelist import resolve_cytometer_params

import logging
logger = logging.getLogger(__name__)

# $PnV (detector voltage) on most instruments; Cytek Mosaic stores the
# equivalent per-channel setting under $PnG (gain) instead. Either is
# accepted as "the detector setting for channel n".
_PNV_PATTERN = re.compile(r'^p(\d+)[vg]$', re.IGNORECASE)


# Scatter, time and width channels by name, for cytometers not in the database.
_NON_FLUOR_PATTERN = re.compile(r'FSC|SSC|Time|Width', re.IGNORECASE)
# Channels Honeychrome writes into its unmixed FCS exports; never detectors.
_HONEYCHROME_DERIVED_CHANNELS = frozenset({'event_id', *af_channels})


def unrecognised_fluorescence_ids(pnn, whitelisted):
    """Positions in pnn of the fluorescence detectors of an unrecognised cytometer.

    flowio's own classification only excludes literal "FSC"/"SSC", so
    detectors like SSC_1-4 or VSSC1-Width would count as fluorescence.
    Channels are taken as detectors when they are whitelisted, are not
    scatter, time or width by name, and are not event_id or AF Abundance /
    AF Index (present when a Honeychrome unmixed export is imported).
    """
    whitelisted = set(whitelisted)
    return [
        i for i, ch in enumerate(pnn)
        if ch in whitelisted
        and ch not in _HONEYCHROME_DERIVED_CHANNELS
        and not _NON_FLUOR_PATTERN.search(ch)
    ]


def _extract_pnv_values(text_dict, n_channels):
    """Per-channel $PnV/$PnG (detector voltage/gain) values, 1-indexed to match PnN order."""
    pnv = [None] * n_channels
    for key, value in text_dict.items():
        match = _PNV_PATTERN.match(str(key).lstrip('$'))
        if not match:
            continue
        idx = int(match.group(1)) - 1
        if 0 <= idx < n_channels:
            try:
                pnv[idx] = float(value)
            except (TypeError, ValueError):
                pnv[idx] = None
    return pnv


class ImportFCSController(QObject):
    finished = Signal()

    def __init__(self, experiment, bus=None):
        super().__init__()

        # connect
        self.experiment = experiment
        self.bus = bus

    def reconfigure_experiment_from_fcs_files(self):
        """Thin wrapper — safe to call directly from the main thread only.

        `import_fcs_files_widget.py` already runs the import on its own
        QThread, so it connects to `_reconfigure_experiment_from_fcs_files_impl`
        directly rather than this wrapper, to avoid nesting
        @with_busy_cursor's WorkerThread/QEventLoop (and its main-thread-only
        cursor calls) inside an already-background thread.
        """
        self._reconfigure_experiment_from_fcs_files_impl()

    @with_busy_cursor
    def _reconfigure_experiment_from_fcs_files_impl(self):
        self.experiment.scan_sample_tree()
        experiment_dir = self.experiment.generate_subdirs()
        # single_stain_controls = self.experiment.samples['single_stain_controls']
        all_samples = self.experiment.samples['all_samples']
        raw_samples = all_samples.keys()

        if len(raw_samples) > 0:
            try:
                # load samples one by one, print name, datetime, number of events, file location
                # all_sample_meta = {}
                # all_sample_channels = {}
                all_sample_pnn = {}
                all_sample_pnr = {}
                all_sample_pnv = {}
                first_sample_metadata = None
                for n, sample_path in enumerate(raw_samples):
                    # Replace NA keyword values before any numeric conversion.
                    _fd = FlowData(experiment_dir / sample_path, only_text=True, use_header_offsets=True)
                    for _k in list(_fd.text.keys()):
                        if str(_fd.text[_k]).strip().upper() == 'NA':
                            _fd.text[_k] = '0'
                    all_sample_pnn[sample_path] = _fd.pnn_labels
                    all_sample_pnr[sample_path] = _fd.pnr_values
                    all_sample_pnv[sample_path] = _extract_pnv_values(_fd.text, len(_fd.pnn_labels))
                    if first_sample_metadata is None:
                        first_sample_metadata = _fd

                    if self.bus:
                        self.bus.progress.emit(n, len(raw_samples))

                sample_metadata = first_sample_metadata  # keep name for downstream compat

                # Resolve cytometer and whitelisted channels from the first file.
                # This must happen before the consistency check so we can compare
                # only the whitelisted subset across files.
                representative_pnn = list(all_sample_pnn[list(raw_samples)[0]])
                cyt_info = resolve_cytometer_params(
                    all_pnn=representative_pnn,
                    text_keywords=sample_metadata.text,
                )
                channel_mismatch_note = None  # set below if files differ; folded into the single end-of-import dialog
                time_missing_note = None  # set below if no Time channel; folded into the same dialog
                voltage_mismatch_note = None  # set below if PnV/PnG differs across files; folded into the same dialog
                voltage_unavailable_note = None  # set below for ID7000, which reports no PnV/PnG at all

                if cyt_info is not None:
                    _fl_ids = cyt_info.fluorescence_channel_ids
                    _sc_base = [ch.rsplit('-', 1)[0] for ch in cyt_info.scatter_param]
                    _sc_extra_pat = (
                        re.compile('|'.join(cyt_info.scatter_extra_pat))
                        if cyt_info.scatter_extra_pat else None
                    )
                    _sc_ids = [
                        i for i, ch in enumerate(representative_pnn)
                        if ch.rsplit('-', 1)[0] in _sc_base
                        or (_sc_extra_pat and re.search(_sc_extra_pat, ch))
                    ]
                    _time_id = next(
                        (i for i, ch in enumerate(representative_pnn) if ch.lower() == 'time'), None
                    )
                    whitelisted_pnn = (
                        ([representative_pnn[_time_id]] if _time_id is not None else [])
                        + [representative_pnn[i] for i in _sc_ids]
                        + [representative_pnn[i] for i in _fl_ids]
                    )
                else:
                    # Unrecognised cytometer: restrict to channels common to every
                    # file (not just the first one scanned), so files with extra or
                    # missing derived parameters (e.g. a post-processing tool adding
                    # parameters to only some files) can all be loaded afterwards
                    # without re-triggering the "channels don't match" check.
                    _common_pnn = set(representative_pnn)
                    for sp in raw_samples:
                        _common_pnn &= set(all_sample_pnn[sp])
                    whitelisted_pnn = [ch for ch in representative_pnn if ch in _common_pnn]

                    _dropped_pnn = [ch for ch in representative_pnn if ch not in _common_pnn]
                    if _dropped_pnn:
                        _per_file_drops = {
                            sp: sorted(set(all_sample_pnn[sp]) - _common_pnn)
                            for sp in raw_samples
                            if set(all_sample_pnn[sp]) - _common_pnn
                        }
                        # Stash rather than emit immediately — combined into the single
                        # end-of-import dialog below so only one QMessageBox is shown.
                        channel_mismatch_note = (
                            'Cytometer not recognised — channels were not identical across '
                            'all imported files.\n'
                            f'Retained {len(whitelisted_pnn)} channel(s) common to every file:\n'
                            f'  {whitelisted_pnn}\n'
                            'Channels present in only some files (ignored):\n'
                            + '\n'.join(f'  {sp}: {chs}' for sp, chs in _per_file_drops.items())
                        )
                        warnings.warn(channel_mismatch_note)

                logger.debug('Whitelisted PNN (%d channels): %s', len(whitelisted_pnn), whitelisted_pnn)

                # Validate: every file must contain all whitelisted channels.
                # Extra channels (e.g. BD Chorus derived params) are ignored.
                files_missing = {
                    sp: sorted(set(whitelisted_pnn) - set(all_sample_pnn[sp]))
                    for sp in raw_samples
                    if set(whitelisted_pnn) - set(all_sample_pnn[sp])
                }
                if files_missing:
                    text = (
                        'Cannot set up the experiment file:\n'
                        'Some FCS files are missing required whitelisted channels:\n'
                        + '\n'.join(f'  {sp}: {m}' for sp, m in files_missing.items())
                    )
                    warnings.warn(text)
                    if self.bus:
                        self.bus.warningMessage.emit(text)
                    return

                # Channel consistency: check only the whitelisted subset.
                all_sample_pnn_whitelisted = [
                    [ch for ch in all_sample_pnn[sp] if ch in set(whitelisted_pnn)]
                    for sp in raw_samples
                ]
                all_sample_pnr = [list(all_sample_pnr[sample_path]) for sample_path in raw_samples]

                if True:  # always proceed — consistency guaranteed by subset check above
                    # other bits of experiment reset to default
                    self.experiment.settings['unmixed'] = deepcopy(settings_default['unmixed'])
                    self.experiment.process = deepcopy(process_default)
                    self.experiment.cytometry = deepcopy(cytometry_default)

                    # set up all raw settings
                    time_channel_id = sample_metadata.time_index
                    if time_channel_id is None:
                        time_missing_note = "No Time channel found. The FCS file does not conform to standard."
                        warnings.warn(time_missing_note)
                        # Stash rather than emit immediately — combined into the single
                        # end-of-import dialog below so only one QMessageBox is shown.

                    scatter_channel_ids = sample_metadata.scatter_indices
                    event_channels_pnn = representative_pnn  # full list from first file

                    # cyt_info already resolved above; apply its derived values here.

                    if cyt_info is not None:
                        fluorescence_channel_ids = _fl_ids
                        scatter_channel_ids = _sc_ids
                        logger.info(
                            'Cytometer identified as "%s". '
                            'Using whitelisted fluorescence channels (%d channels).',
                            cyt_info.cyt_label,
                            len(fluorescence_channel_ids),
                        )
                        self.experiment.settings['raw']['cytometer'] = cyt_info.cyt_label
                        self.experiment.settings['raw']['cytometer_db_col'] = cyt_info.db_col
                        self.experiment.settings['raw']['scatter_param'] = cyt_info.scatter_param
                    else:
                        _whitelisted_set = set(whitelisted_pnn)
                        fluorescence_channel_ids = unrecognised_fluorescence_ids(
                            representative_pnn, _whitelisted_set,
                        )
                        scatter_channel_ids = [
                            i for i in scatter_channel_ids
                            if representative_pnn[i] in _whitelisted_set
                        ]
                        cyt_kw = next(
                            (v for k, v in sample_metadata.text.items() if k.upper() in ('$CYT', 'CYT') and v),
                            ''
                        )
                        logger.warning(
                            'Cytometer not recognised ($CYT="%s"). '
                            'Falling back to flowio fluorescence channel classification. '
                            'If this file is from a FACSDiscover or Xenith, '
                            'pre-unmixed channels may be included.',
                            cyt_kw,
                        )
                        self.experiment.settings['raw']['cytometer'] = cyt_kw or 'Unknown'
                        self.experiment.settings['raw']['cytometer_db_col'] = None
                    
                    event_channels_pnn_stripped = event_channels_pnn.copy()
                    for suffix in ['-A', '-H', '-W']:
                        event_channels_pnn_stripped = [c.removesuffix(suffix) if c.endswith(suffix) else c for c in event_channels_pnn_stripped]

                    if 'event_id' in event_channels_pnn:
                        event_id_channel_id = event_channels_pnn.index('event_id')
                    else:
                        event_id_channel_id = None

                    area_channels = [s.removesuffix('-A') for s in event_channels_pnn if s.endswith("-A")]
                    height_channels = [s.removesuffix('-H') for s in event_channels_pnn if s.endswith("-H")]
                    width_channels = [s.removesuffix('-W') for s in event_channels_pnn if s.endswith("-W")]
                    pnr = all_sample_pnr[0]
                    magnitude_ceiling = self.experiment.settings['raw']['magnitude_ceiling']
                    width_ceiling = self.experiment.settings['raw']['width_ceiling']
                    default_ceiling = self.experiment.settings['raw']['default_ceiling']
                    if cyt_info is not None:
                        # Use only whitelisted fluorescence channels for the ceiling
                        magnitude_ceiling = float(cyt_info.sat_value)
                    else:
                        if len(area_channels) > 0:
                            magnitude_ceiling = float(np.max([pnr[event_channels_pnn_stripped.index(c)] for c in area_channels]))
                        if len(height_channels) > 0:
                            magnitude_ceiling = float(np.max([pnr[event_channels_pnn_stripped.index(c)] for c in height_channels]))
                    if len(width_channels) > 0:
                        width_ceiling = float(np.max([pnr[event_channels_pnn_stripped.index(c)] for c in width_channels]))
                    n_scatter_channels = len(scatter_channel_ids)
                    n_fluorophore_channels = len(fluorescence_channel_ids)

                    # Voltage consistency check — fluorescence channels only.
                    # Some instruments (e.g. ID7000) allow controls/samples to
                    # be run at different voltages; Honeychrome does not yet
                    # adjust for this, so flag it rather than silently
                    # unmixing across mismatched gains.
                    fluor_names = [event_channels_pnn[i] for i in fluorescence_channel_ids]
                    voltage_by_file = {
                        sp: dict(zip(all_sample_pnn[sp], all_sample_pnv[sp]))
                        for sp in raw_samples
                    }
                    mismatched_voltages = {}
                    for ch in fluor_names:
                        seen = {
                            voltage_by_file[sp][ch]
                            for sp in raw_samples
                            if voltage_by_file[sp].get(ch) is not None
                        }
                        if len(seen) > 1:
                            mismatched_voltages[ch] = {
                                sp: voltage_by_file[sp].get(ch) for sp in raw_samples
                            }
                    if mismatched_voltages:
                        logger.debug(
                            'Voltage/gain mismatch detail (%d of %d whitelisted '
                            'fluorescence channels): %s',
                            len(mismatched_voltages), len(fluor_names), mismatched_voltages,
                        )
                        voltage_mismatch_note = (
                            f'Voltage/gain mismatch detected across imported files in '
                            f'{len(mismatched_voltages)} of {len(fluor_names)} critical '
                            f'fluorescence channel(s).\n'
                            'Honeychrome does not currently adjust for voltage '
                            'differences between controls and samples — unmixing '
                            'accuracy may be affected.'
                        )
                        warnings.warn(voltage_mismatch_note)

                    # ID7000 files carry no $PnV/$PnG keywords at all, so a
                    # mismatch can never be detected either way — warn
                    # unconditionally instead.
                    if cyt_info is not None and cyt_info.db_col == 'ID7000':
                        voltage_unavailable_note = (
                            'Unmixing in Honeychrome assumes identical settings '
                            'between controls and samples. ID7000 files run on '
                            'One-Max or All-Max may exhibit inaccurate unmixing.'
                        )
                        warnings.warn(voltage_unavailable_note)

                    self.experiment.settings['raw']['area_channels'] = area_channels
                    self.experiment.settings['raw']['height_channels'] = height_channels
                    self.experiment.settings['raw']['width_channels'] = width_channels
                    self.experiment.settings['raw']['scatter_channels'] = list(set([event_channels_pnn_stripped[i] for i in scatter_channel_ids]))
                    self.experiment.settings['raw']['fluorescence_channels'] = list(set([event_channels_pnn_stripped[i] for i in fluorescence_channel_ids]))
                    self.experiment.settings['raw']['event_channels_pnn'] = event_channels_pnn
                    self.experiment.settings['raw']['whitelisted_pnn'] = whitelisted_pnn
                    self.experiment.settings['raw']['width_ceiling'] = width_ceiling
                    self.experiment.settings['raw']['magnitude_ceiling'] = magnitude_ceiling
                    self.experiment.settings['raw']['default_ceiling'] = default_ceiling
                    self.experiment.settings['raw']['time_channel_id'] = time_channel_id
                    self.experiment.settings['raw']['event_id_channel_id'] = event_id_channel_id
                    self.experiment.settings['raw']['scatter_channel_ids'] = scatter_channel_ids
                    self.experiment.settings['raw']['n_scatter_channels'] = n_scatter_channels
                    self.experiment.settings['raw']['fluorescence_channel_ids'] = fluorescence_channel_ids
                    self.experiment.settings['raw']['n_fluorophore_channels'] = n_fluorophore_channels
                    self.experiment.settings['raw']['channel_pnr'] = [float(v) for v in pnr]
                    self.experiment.settings['raw']['scatter_display_ceiling'] = (
                        cyt_info.scatter_display_ceiling if cyt_info is not None else {}
                    )

                    # set up raw transforms — restrict to whitelisted channels to avoid
                    # building hundreds of transforms for FACSDiscover derived parameters
                    self.experiment.cytometry['raw_transforms'] = assign_default_transforms(
                        self.experiment.settings['raw'],
                        channels=whitelisted_pnn,
                    )
                    raw_transformations = generate_transformations(self.experiment.cytometry['raw_transforms'])

                    # set up raw gating — register only whitelisted channel transforms
                    raw_gating = GatingStrategy()
                    for label in whitelisted_pnn:
                        if label in raw_transformations:
                            raw_gating.transformations[label] = raw_transformations[label].xform

                    if cyt_info is not None and len(cyt_info.scatter_param) >= 2:
                        # Use cytometer-specific canonical scatter channel names
                        morph_x = cyt_info.scatter_param[0] if cyt_info.scatter_param[0] in event_channels_pnn else None
                        morph_y = cyt_info.scatter_param[1] if cyt_info.scatter_param[1] in event_channels_pnn else None
                    else:
                        # Fallback: original hardcoded FSC/SSC detection
                        if 'FSC' in area_channels:
                            morph_x = 'FSC-A'
                        elif 'FSC' in height_channels:
                            morph_x = 'FSC-H'
                        else:
                            morph_x = None

                        if morph_x is not None:
                            if 'SSC' in area_channels:
                                morph_y = 'SSC-A'
                            elif 'SSC' in height_channels:
                                morph_y = 'SSC-H'
                            else:
                                morph_y = None

                    # default gates need a raw transform, which exists only for whitelisted channels
                    if morph_x not in whitelisted_pnn or morph_y not in whitelisted_pnn:
                        morph_x = morph_y = None

                    if (morph_x is not None) and (morph_y is not None):
                        sing_x = morph_x
                        sing_y = None
                        preferred = cyt_info.singlet_y_preference if cyt_info is not None else 'FSC-W'
                        if preferred in event_channels_pnn:
                            sing_y = preferred
                        elif preferred == 'FSC-H' and 'FSC' in width_channels:
                            sing_y = 'FSC-H'
                        elif preferred == 'FSC-W' and 'FSC' in width_channels:
                            sing_y = 'FSC-W'
                        elif 'FSC' in width_channels:
                            sing_y = 'FSC-W'
                        elif 'FSC' in height_channels and 'FSC-H' in event_channels_pnn:
                            sing_y = 'FSC-H'
                        else:
                            sing_y = None
                        if sing_y not in whitelisted_pnn:
                            sing_y = None

                    time_plot = None
                    morph_plot = None
                    singlet_plot = None
                    label = 'root'
                    # Resolve per-channel display ceilings for gate placement.
                    # Gates are positioned as fractions of the display ceiling (not the
                    # full PNR), so they land on-screen when a scatter_display_ceiling
                    # override is active (e.g. FACSDiscover FSC-A).
                    scatter_display_ceiling = self.experiment.settings['raw'].get('scatter_display_ceiling', {})

                    def _display_ceil(channel):
                        """Raw value at the top of the initial display viewport for channel."""
                        return scatter_display_ceiling.get(channel, pnr[event_channels_pnn.index(channel)])

                    if (morph_x is not None) and (morph_y is not None):
                        label = 'Cells'
                        morph_x_ceil = _display_ceil(morph_x)
                        morph_y_ceil = _display_ceil(morph_y)
                        morph_x_pnr  = pnr[event_channels_pnn.index(morph_x)]
                        morph_y_pnr  = pnr[event_channels_pnn.index(morph_y)]
                        dim_x = Dimension(morph_x, range_min=0.2 * morph_x_ceil / morph_x_pnr,
                                          range_max=0.8 * morph_x_ceil / morph_x_pnr, transformation_ref=morph_x)
                        dim_y = Dimension(morph_y, range_min=0.2 * morph_y_ceil / morph_y_pnr,
                                          range_max=0.8 * morph_y_ceil / morph_y_pnr, transformation_ref=morph_y)
                        gate = gates.RectangleGate(label, dimensions=[dim_x, dim_y])
                        raw_gating.add_gate(gate, gate_path=('root',))
                        morph_plot = [{'type': 'hist2d', 'channel_x': morph_x, 'channel_y': morph_y, 'source_gate': 'root', 'child_gates': ['Cells']}]

                        if (sing_x is not None) and (sing_y is not None):
                            label = 'Singlets'
                            sing_x_pnr = pnr[event_channels_pnn.index(sing_x)]
                            sing_y_pnr = pnr[event_channels_pnn.index(sing_y)]
                            sing_x_ceil = _display_ceil(sing_x)
                            sing_y_ceil = _display_ceil(sing_y)
                            
                            if sing_y == 'FSC-W':
                                dim_x = Dimension(sing_x, range_min=0.2 * sing_x_ceil / sing_x_pnr,
                                                  range_max=0.8 * sing_x_ceil / sing_x_pnr, transformation_ref=sing_x)
                                dim_y = Dimension(sing_y, range_min=0.2 * sing_y_ceil / sing_y_pnr,
                                                  range_max=0.8 * sing_y_ceil / sing_y_pnr, transformation_ref=sing_y)
                                gate = gates.RectangleGate(label, dimensions=[dim_x, dim_y])
                            else:  # FSC-H — vertices in display space (post-transform)
                                tr_x = raw_transformations[sing_x]
                                tr_y = raw_transformations[sing_y]
                                def _tx(v): return float(tr_x.xform.apply(np.array([v]))[0])
                                def _ty(v): return float(tr_y.xform.apply(np.array([v]))[0])
                                dim_x = Dimension(sing_x, range_min=0, range_max=1, transformation_ref=sing_x)
                                dim_y = Dimension(sing_y, range_min=0, range_max=1, transformation_ref=sing_y)
                                vertices = [
                                    (_tx(0.2 * sing_x_ceil), _ty(0.1 * sing_y_ceil)),
                                    (_tx(0.8 * sing_x_ceil), _ty(0.7 * sing_y_ceil)),
                                    (_tx(0.8 * sing_x_ceil), _ty(0.9 * sing_y_ceil)),
                                    (_tx(0.2 * sing_x_ceil), _ty(0.3 * sing_y_ceil)),
                                ]
                                gate = gates.PolygonGate(label, dimensions=[dim_x, dim_y], vertices=vertices)
                            raw_gating.add_gate(gate, gate_path=('root', 'Cells'))
                            singlet_plot = [{'type': 'hist2d', 'channel_x': sing_x, 'channel_y': sing_y, 'source_gate': 'Cells', 'child_gates': ['Singlets']}]

                    # set up raw plots
                    # plots is list of dicts
                    # type:
                    # ---hist1d: channel_x, source_gate, child_gates
                    # ---hist2d: channel_x, channel_y, source_gate, child_gates
                    # ---ribbon: source_gate, child_gates
                    if time_channel_id is not None:
                        time_plot = [{'type': 'hist1d', 'channel_x': event_channels_pnn[time_channel_id], 'source_gate': 'root', 'child_gates': []}]
                    ribbon_plot = [{'type': 'ribbon', 'source_gate': label, 'child_gates': []}]
                    fluorescence_plots = [{'type': 'hist1d', 'channel_x': event_channels_pnn[i], 'source_gate': label, 'child_gates': []} for i in fluorescence_channel_ids]

                    raw_plots = []
                    if time_plot:
                        raw_plots += time_plot
                    if morph_plot:
                        raw_plots += morph_plot
                    if singlet_plot:
                        raw_plots += singlet_plot
                    raw_plots += ribbon_plot
                    raw_plots += fluorescence_plots

                    self.experiment.cytometry['raw_gating'] = to_gml(raw_gating)
                    self.experiment.cytometry['raw_plots'] = raw_plots
                    self.experiment.cytometry['gating'] = None
                    self.experiment.save()
                    # otb: suggest not listing all the channels, as this can be too long
                    text = (f'Experiment successfully configured for imported FCS files.\n\n'
                            f'Number of FCS files imported: {len(raw_samples)}\n'
                            f'Cytometer: {self.experiment.settings['raw']['cytometer']}\n'
                            f'Number of scatter channels: {n_scatter_channels}\n'
                            f'Number of fluorescence channels: {n_fluorophore_channels}\n'
                            #f'Area channels: {self.experiment.settings['raw']['area_channels']}\n'
                            #f'Height channels: {self.experiment.settings['raw']['height_channels']}\n'
                            #f'Width channels: {self.experiment.settings['raw']['width_channels']}\n'
                            f'Scatter channels: {self.experiment.settings['raw']['scatter_channels']}\n'
                            #f'Fluorescence channels: {self.experiment.settings['raw']['fluorescence_channels']}\n'
                            f'Magnitude ceiling: {self.experiment.settings['raw']['magnitude_ceiling']}\n'
                            f'Time channel ID: {self.experiment.settings['raw']['time_channel_id']}\n'
                            f'Event ID channel ID: {self.experiment.settings['raw']['event_id_channel_id']}\n'
                            )
                    # Fold any collected notes (channel mismatch, missing Time channel,
                    # voltage mismatch, ID7000 voltage-unavailable) into this single
                    # dialog instead of emitting separately
                    notes = [
                        n for n in (
                            channel_mismatch_note,
                            time_missing_note,
                            voltage_mismatch_note,
                            voltage_unavailable_note,
                        ) if n
                    ]
                    if notes:
                        text = '\n\n'.join(notes) + '\n\n' + text
                    logger.info(text)
                    if self.bus:
                        self.bus.reloadExpRequested.emit()
                        if notes:
                            self.bus.warningMessage.emit(text)
                        else:
                            self.bus.popupMessage.emit(text)
                else:
                    text = ('Cannot set up the experiment file: \n'
                            'the FCS files supplied do not have a consistent set of channels (channel names and ranges). \n'
                            'Please make sure the FCS files are all from a single experiment on a single instrument.')
                    warnings.warn(text)
                    if self.bus:
                        self.bus.warningMessage.emit(text)

            except Exception as e:
                text = (f'Cannot set up the experiment file: \n'
                        f'the FCS files supplied do not have a consistent set of channels (channel names and ranges). \n'
                        f'Please make sure the FCS files are all from a single experiment on a single instrument.\n'
                        f'Exception:\n'
                        f'{e}')
                warnings.warn(text)
                if self.bus:
                    self.bus.warningMessage.emit(text)

        else:
            text = 'No FCS files found in the experiment <tt>Raw</tt> folder.'
            warnings.warn(text)
            if self.bus:
                self.bus.warningMessage.emit(text)

        self.finished.emit()



if __name__ == "__main__":
    from pathlib import Path
    from honeychrome.controller import Controller

    kc = Controller()
    base_directory = Path.home() / 'spectral_cytometry'
    experiment_name = base_directory / '20240620 Spectral Symposium-poor cell unmixed'
    experiment_path = experiment_name.with_suffix('.kit')
    kc.load_experiment(experiment_path) # note this loads first sample too and runs calculate all histograms and statistics
    import_fcs_controller = ImportFCSController(kc.experiment, None)
    import_fcs_controller.reconfigure_experiment_from_fcs_files()


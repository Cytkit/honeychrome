import json
import math
import warnings
from pathlib import Path
# from unicodedata import category

import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer

from honeychrome.controller_components.fcs_preview import read_preview_metadata
from honeychrome.controller_components.functions import timer, calc_hist1d
from honeychrome.controller_components.per_sample_gating_functions import SampleGating
from honeychrome.controller_components.sample_loader import (snapshot_unmix_state, load_unmixed,
                                                             seeded_subsample_indices)
from honeychrome.view_components.busy_cursor import with_busy_cursor

import logging
logger = logging.getLogger(__name__)

# Intensity statistics (heatmap medians and mean intensities) are calculated on
# a seeded random subsample of each file rather than on every event. A first
# subsample of INTENSITY_PILOT_EVENTS events shows how many events fall in each
# plotted gate. A gate holding fewer than INTENSITY_TARGET_EVENTS of them gets
# a larger subsample, sized from its share of the first one to give it about
# INTENSITY_TARGET_EVENTS events, or the whole file when that would need more
# than WHOLE_FILE_FRACTION of it. Each gate's subsample depends only on that
# gate, so its values do not change with the other plots calculated alongside
# it. The
# standard error of a median from n events is about 1.25 * SD / sqrt(n), so
# 10,000 events place a gate's median within about 1% of its spread. Counts,
# percentages and histograms use every event.
INTENSITY_PILOT_EVENTS = 20_000
INTENSITY_TARGET_EVENTS = 10_000
WHOLE_FILE_FRACTION = 0.5
COUNT_STATISTICS = ('% Total Events', '% Parent', 'Event Concentration', 'Number of Events')


def _uses_subsample(statistics_comparison) -> bool:
    """True for statistics calculated on a subsample of events: heatmap
    medians and mean intensities."""
    return (statistics_comparison['plot_type'] == 'Heatmap'
            or statistics_comparison['statistic'].startswith('Mean'))


class StatisticsCalculator(QObject):
    finished = Signal()

    def __init__(self, bus, controller, only_new=False):
        super().__init__()

        # connect
        self.controller = controller
        self.bus = bus
        # With only_new, only comparisons without results (data None) are
        # calculated; the others keep theirs.
        self.only_new = only_new
        # Taken here, on the main thread: each sample is gated with its own
        # gates (its per-sample overrides and its own Time range), not with
        # the lookup tables of the sample on display.
        self.sample_gating = SampleGating.from_controller(controller)
        # Likewise each sample is unmixed with its own AutoSpectral AF
        # profile assignment, as the main window unmixes it when displayed.
        self.unmix_snapshot = (snapshot_unmix_state(controller, include_qc=False)
                               if controller.transfer_matrix is not None else None)

    @with_busy_cursor
    def run(self):
        """Calculate every comparison; ``finished`` is emitted on every path, failures included."""
        try:
            self._calculate()
        except Exception as e:
            logger.exception('StatisticsCalculator: failed')
            if self.bus:
                self.bus.progress.emit(0, 0)
                self.bus.statusMessage.emit(f'Statistics failed: {e}')
            self.finished.emit()

    def _calculate(self):
        all_samples = self.controller.experiment.samples['all_samples']
        experiment_statistics = self.controller.experiment.statistics
        comparisons = [statistics_comparison for statistics_comparison in experiment_statistics
                       if not self.only_new or statistics_comparison.get('data') is None]
        sample_sets = [statistics_comparison['sample_set'] for statistics_comparison in comparisons]
        samples_to_calculate = [sample for sample in all_samples if any([Path(sample).is_relative_to(folder) for folder in sample_sets])]
        samples_by_set = {sample_set:[sample for sample in samples_to_calculate if Path(sample).is_relative_to(sample_set)] for sample_set in sample_sets}

        if self.controller.experiment.process['unmixing_matrix'] is not None and self.unmix_snapshot is not None:
            # Gates that statistics can be calculated for, with the gate each
            # one's '% Parent' refers to. Quadrant gates are containers: their
            # quadrants are reported against the quadrant gate's parent.
            gating = self.sample_gating.gating
            parents = {}
            for gate_id, gate_path in gating.get_gate_ids():
                if gating._get_gate_node(gate_id, gate_path).gate_type == 'QuadrantGate':
                    continue
                parent_id = gating.get_parent_gate_id(gate_id, gate_path=gate_path)
                if parent_id is None:
                    parents[gate_id] = 'root'
                elif gating._get_gate_node(parent_id[0], parent_id[1]).gate_type != 'QuadrantGate':
                    parents[gate_id] = parent_id[0]
                else:
                    parents[gate_id] = parent_id[1][-1]

            for statistics_comparison in comparisons:
                gate_name = statistics_comparison['gate']
                if gate_name != 'root' and gate_name not in parents:
                    text = (f'Cannot calculate statistics: gate "{gate_name}" no longer exists.  '
                            f'Please either create the gate again or delete any statistics plots below that reference {gate_name}.')
                    warnings.warn(text)
                    if self.bus:
                        self.bus.warningMessage.emit(text)
                        self.bus.progress.emit(0,0)
                        self.bus.statusMessage.emit(text)
                    self.finished.emit()
                    return

            # Channel transforms for intensities and histograms are shared by
            # every sample, so values stay comparable across samples.
            data_for_statistics_comparison = self.controller.data_for_cytometry_plots_unmixed.copy()
            snap = self.unmix_snapshot
            # set up data first by sample
            data_by_sample = {}
            for n in range(len(samples_to_calculate)):
                logger.info(f'StatisticsCalculator: sample {n+1}/{len(samples_to_calculate)}')
                if self.bus:
                    self.bus.progress.emit(n, len(samples_to_calculate))

                sample_key = samples_to_calculate[n]
                sample_name = all_samples[sample_key]
                path_components = Path(sample_key).parts
                depth = len(path_components)

                if depth > 2:
                    group_name = path_components[-2]
                else:
                    group_name = None

                if depth > 3:
                    category_name = '/'.join(path_components[1:-2])
                else:
                    category_name = None

                full_sample_path = self.controller.experiment_dir / sample_key
                try:
                    metadata = read_preview_metadata(full_sample_path)
                except Exception as e:
                    logger.debug('StatisticsCalculator: no TEXT segment preview for %s (%s)', full_sample_path, e)
                    metadata = None
                missing = ([ch for ch in snap.pnn_raw if ch not in metadata.pnn_labels]
                           if metadata is not None and snap.whitelisted else [])
                if missing:
                    self._warn_skipped(sample_name, full_sample_path, f'missing channels {missing}')
                    continue

                data_by_sample[sample_key] = {'Sample': sample_name, 'Group': group_name, 'Category': category_name, 'Statistics': {}}
                if metadata is not None and metadata.event_count == 0:
                    continue
                sample_comparisons = [statistics_comparison for statistics_comparison in comparisons
                                      if sample_key in samples_by_set[statistics_comparison['sample_set']]]
                try:
                    data_by_sample[sample_key]['Statistics'] = self._sample_statistics(
                        sample_key, sample_comparisons, data_for_statistics_comparison, parents)
                except (KeyError, ValueError) as e:
                    data_by_sample.pop(sample_key)
                    self._warn_skipped(sample_name, full_sample_path, e)

            # assemble the data into experiment statistics data table
            for m, statistics_comparison in enumerate(comparisons):
                calculated_samples = [sample for sample in samples_by_set[sample_sets[m]] if sample in data_by_sample]
                if statistics_comparison['plot_type'] == 'Heatmap':
                    # Heatmap data is a sample x marker table: one 'Sample' list
                    # plus one list per marker column.
                    key = (statistics_comparison['gate'], statistics_comparison['statistic'])
                    samples_list = []
                    marker_columns = {}
                    for sample in calculated_samples:
                        row = data_by_sample[sample]['Statistics'].get(key, {})
                        samples_list.append(data_by_sample[sample]['Sample'])
                        for marker, marker_value in row.items():
                            marker_columns.setdefault(marker, []).append(marker_value)
                    heatmap_data = {'Sample': samples_list}
                    heatmap_data.update(marker_columns)
                    statistics_comparison['data'] = heatmap_data if samples_list else {}
                    statistics_comparison['depth'] = 1
                    continue
                data_list = []
                for sample in calculated_samples:
                    key = (statistics_comparison['gate'], statistics_comparison['statistic'])
                    if key in data_by_sample[sample]['Statistics'].keys():
                        value = data_by_sample[sample]['Statistics'][key]
                    else:
                        value = np.nan

                    data_list.append({
                        'Sample': data_by_sample[sample]['Sample'],
                        'Group': data_by_sample[sample]['Group'],
                        'Category': data_by_sample[sample]['Category'],
                        statistics_comparison['statistic']: value
                    })

                data_dict_of_lists = {}
                for d in data_list:
                    for key, value in d.items():
                        data_dict_of_lists.setdefault(key, []).append(value)

                statistics_comparison['data'] = data_dict_of_lists
                if statistics_comparison['data']: # empty if there are no samples
                    depth = 3
                    if None in statistics_comparison['data']['Category'] or len(set(statistics_comparison['data']['Category']))==1:
                        statistics_comparison['data'].pop('Category')
                        depth = 2
                    if None in statistics_comparison['data']['Group'] or len(set(statistics_comparison['data']['Group']))==1:
                        statistics_comparison['data'].pop('Group')
                        depth = 1
                    statistics_comparison['depth'] = depth

            logger.info(f'StatisticsCalculator: calculated statistics {json.dumps(experiment_statistics, indent=2)}')

            if self.bus:
                self.bus.progress.emit(len(samples_to_calculate), len(samples_to_calculate))

        self.finished.emit()

    def _warn_skipped(self, sample_name, full_sample_path, reason):
        logger.warning('StatisticsCalculator: skipping %s (%s)', full_sample_path, reason)
        if self.bus:
            self.bus.warningMessage.emit(
                f'Skipped "{sample_name}" when calculating statistics: '
                f'its channels do not match the experiment whitelist or it could not be read ({reason}).'
            )

    def _sample_statistics(self, sample_key, sample_comparisons, data_for_statistics_comparison, parents) -> dict:
        """{(gate, statistic): value} for one sample's comparisons.

        Counts, percentages and histograms are calculated on every event of
        the file, read through the experiment's unmixing cache. Heatmap
        medians and mean intensities are calculated on the subsample chosen
        by ``_intensity_events``.
        """
        pnn = data_for_statistics_comparison['pnn']
        transformations = data_for_statistics_comparison['transformations']
        gate_names = sorted({statistics_comparison['gate'] for statistics_comparison in sample_comparisons})
        full_comparisons = [c for c in sample_comparisons if not _uses_subsample(c)]
        subsample_comparisons = [c for c in sample_comparisons if _uses_subsample(c)]
        statistics = {}

        full_events = full_membership = None
        if full_comparisons:
            full_events = load_unmixed(self.controller.experiment_dir, sample_key, self.unmix_snapshot,
                                       use_cache=True)['unmixed']
            full_membership = self.sample_gating.memberships(sample_key, full_events, pnn, gate_names)
            n_events_total = len(full_events)
            no_events = np.zeros(n_events_total, dtype=bool)
            for statistics_comparison in full_comparisons:
                gate_name = statistics_comparison['gate']
                statistic = statistics_comparison['statistic']
                gate_membership = full_membership.get(gate_name, no_events)
                if statistic in COUNT_STATISTICS:
                    n_events_gate = int(gate_membership.sum())
                    if gate_name == 'root':
                        p_gate_total = p_gate_parent = 1.
                    else:
                        p_gate_total = n_events_gate / n_events_total if n_events_total else 0
                        n_events_parent = int(full_membership.get(parents[gate_name], no_events).sum())
                        p_gate_parent = n_events_gate / n_events_parent if n_events_parent else 0
                    if statistic == "% Total Events":
                        value = p_gate_total * 100
                    elif statistic == "% Parent":
                        value = p_gate_parent * 100
                    elif statistic == "Event Concentration":
                        value = n_events_gate / 60
                    else: # statistic == "Number of Events"
                        value = n_events_gate
                else: # statistic == "Intensity...": histogram
                    channel_index = pnn.index(statistics_comparison['channel'])
                    transform = transformations[statistics_comparison['channel']]
                    value = calc_hist1d(full_events, gate_membership, channel_index, transform).tolist()
                statistics[(gate_name, statistic)] = value

        if subsample_comparisons:
            intensity_gates = {statistics_comparison['gate'] for statistics_comparison in subsample_comparisons}
            events_by_gate = self._intensity_events(sample_key, pnn, gate_names, intensity_gates,
                                                    full_events, full_membership)
            for statistics_comparison in subsample_comparisons:
                gate_name = statistics_comparison['gate']
                event_data, membership = events_by_gate[gate_name]
                gate_membership = membership.get(gate_name, np.zeros(len(event_data), dtype=bool))
                if statistics_comparison['plot_type'] == 'Heatmap':
                    # Median (display-scale) intensity of every marker on this
                    # gate. Channels without a display transform (e.g. Time)
                    # are skipped so the colour scale stays comparable across
                    # markers.
                    value = {}
                    for ch_pnn, ch_label in data_for_statistics_comparison['pnn_labels'].items():
                        transform = transformations.get(ch_pnn)
                        if transform is None or transform.xform is None:
                            continue
                        channel_index = pnn.index(ch_pnn)
                        channel_values = event_data[gate_membership, channel_index]
                        value[ch_label] = float(np.median(transform.xform.apply(channel_values))) if channel_values.size else float('nan')
                else: # statistic == "Mean Intensity..."
                    channel_index = pnn.index(statistics_comparison['channel'])
                    value = float(event_data[gate_membership, channel_index].mean())
                statistics[(gate_name, statistics_comparison['statistic'])] = value

        return statistics

    def _intensity_events(self, sample_key, pnn, gate_names, intensity_gates,
                          full_events=None, full_membership=None):
        """{gate: (unmixed events, gate memberships)} for one sample's intensity statistics.

        A seeded subsample of INTENSITY_PILOT_EVENTS events is taken first. A
        gate holding fewer than INTENSITY_TARGET_EVENTS of them gets a larger
        seeded subsample, sized from its share (with a 10% margin for the
        sampling noise of the share) to give it about INTENSITY_TARGET_EVENTS
        events, or the whole file when that would need more than
        WHOLE_FILE_FRACTION of it. Each gate's subsample depends only on that
        gate; gates needing the same subsample share one read. With
        *full_events* (the whole sample, already unmixed and gated in
        *full_membership*) the same rows are taken from memory; otherwise only
        the selected rows are read from the file. A sample's subsamples are
        the same on every run.
        """
        def take(cap):
            if full_events is not None:
                n_events = len(full_events)
                rows = seeded_subsample_indices(n_events, cap, str(sample_key))
                return full_events[rows], {name: mask[rows] for name, mask in full_membership.items()}, n_events
            result = load_unmixed(self.controller.experiment_dir, sample_key, self.unmix_snapshot,
                                  max_events=cap, use_cache=cap is None)
            event_data = result['unmixed']
            return (event_data, self.sample_gating.memberships(sample_key, event_data, pnn, gate_names),
                    result['n_events_file'])

        event_data, membership, n_events_file = take(INTENSITY_PILOT_EVENTS)
        reads = {INTENSITY_PILOT_EVENTS: (event_data, membership)}
        events_by_gate = {}
        for gate_name in intensity_gates:
            cap = INTENSITY_PILOT_EVENTS
            in_gate = int(membership[gate_name].sum()) if gate_name in membership else None
            if len(event_data) < n_events_file and in_gate is not None and in_gate < INTENSITY_TARGET_EVENTS:
                needed = (math.ceil(1.1 * INTENSITY_TARGET_EVENTS * len(event_data) / in_gate)
                          if in_gate else n_events_file)
                cap = None if needed > WHOLE_FILE_FRACTION * n_events_file else needed
            if cap not in reads:
                reads[cap] = take(cap)[:2]
            events_by_gate[gate_name] = reads[cap]
        return events_by_gate





if __name__ == "__main__":
    from pathlib import Path
    from honeychrome.controller import Controller
    # from honeychrome.view_components.event_bus import EventBus

    kc = Controller()
    # base_directory = Path.home() / 'spectral_cytometry'
    # experiment_name = base_directory / '20240620 Spectral Symposium-poor cell unmixed'
    base_directory = Path.home() / 'Experiments'
    experiment_name = base_directory / 'AutoSpectral Full Workflow Imported'
    experiment_path = experiment_name.with_suffix('.kit')
    kc.load_experiment(experiment_path) # note this loads first sample too and runs calculate all histograms and statistics
    kc.set_mode('Unmixed Data')
    kc.initialise_data_for_cytometry_plots()
    kc.set_mode('Simple Stats')

    kc.experiment.statistics = [
        {
            "sample_set": "Raw/Set1/Stained",
            "plot_type": "Bar Chart",
            "gate": "A3 CD45 BUV395+ A4 CD11b BUV805+",
            "statistic": "% Parent",
            "depth": 1,
            'data':None
        },
        {
            "sample_set": "Raw/Set1",
            "plot_type": "Box and Whisker Chart",
            "gate": "Singlets",
            "statistic": "% Parent",
            "depth": 2,
            'data':None
        },
        {
            'sample_set': 'Raw/Set1/Stained',
            'plot_type': '1D Histogram Overlay',
            'gate': 'Singlets',
            'channel': 'A6 Siglec F PE',
            'statistic': 'Intensity A6 Siglec F PE',
            'depth': 1,
            'data': None}
    ]

    statistics_calculator = StatisticsCalculator(None, kc)
    statistics_calculator.run()
    print(json.dumps(kc.experiment.statistics, indent=2))


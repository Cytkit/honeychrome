"""
Honeychrome Plugin Template
---------------------------
This module defines the interface for a Honeychrome tabbed plugin.

Required Attributes:
    plugin_name (str): The display name used for the tab in the main window.
    PluginWidget (class): the widget to be displayed in the tab

Technical Requirements:
    - Framework: PySide6 (Qt for Python)
"""
from datetime import datetime
from pathlib import Path
import colorcet as cc
import numpy as np

from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QPushButton, QLabel, QComboBox
from PySide6.QtCore import Qt
from honeychrome.controller_components.per_sample_gating_functions import SampleGating
from honeychrome.controller_components.sample_loader import (
    sample_key_for_path, snapshot_unmix_state, read_raw_events, unmix_events,
)
from honeychrome.view_components.busy_cursor import with_busy_cursor
from honeychrome.view_components.clear_layout import clear_layout
from honeychrome.view_components.exportable_plot_widget import ExportablePlotWidget
from honeychrome.view_components.ordered_multi_sample_picker import OrderedMultiSamplePicker
from honeychrome.view_components.copyable_table_widget import CopyableTableWidget

plugin_name = 'Data Processing Example Plugin'
table_headers = ['Index', 'Colour', 'Count']


def apply_gate(sample_gating, sample_key, event_data, pnn, gate_name):
    """
    apply_gate:
    This function returns the events of one sample that belong to a gate, using the fast method (lookup tables).
    The sample is gated with its own gates: its per-sample gate overrides and its own Time range.
    Do not reuse the lookup tables in controller.data_for_cytometry_plots_unmixed for other samples:
    they belong to the sample on display.
    :param sample_gating: SampleGating.from_controller(controller), taken once for a batch of samples
    :param sample_key: the sample's key in experiment.samples['all_samples'] (path relative to the experiment)
    :param event_data: unmixed event data of the sample
    :param pnn: channel names of the event data columns
    :param gate_name: name of gate (string)
    :return: gated_sample_data: set of event data that belongs to gate
    """
    gate_membership = sample_gating.population_mask(sample_key, event_data, pnn, [gate_name])
    gated_sample_data = event_data[gate_membership]
    return gated_sample_data

class PluginWidget(QWidget):
    """
    The main UI container for the plugin.

    Required arguments:
        bus: the signals to communicate with the rest of the honeychrome app
        controller: the honeychrome controller including all ephemeral data, the experiment model and sample. In particular:
            controller.experiment: the experiment model (the honeychrome data)
            controller.current_sample: flowkit.Sample object containing the current sample (raw data)
            cytometry data dictionaries: (see definition in controller)
                controller.data_for_cytometry_plots_raw: ephemeral data for raw cytometry
                controller.data_for_cytometry_plots_process: ephemeral data for spectral process cytometry
                controller.data_for_cytometry_plots_unmixed: ephemeral data for unmixed cytometry

    This plugin processes a selection of sample FCS files according to a gate in the unmixed data. It trains a UMAP reducer model on the selection,
    and a clusterer model on top of the UMAP embedding. Then embedding and clusters can be predicted for any selected sample.

    Methods:
        initialise_gui: runs when tab is selected, updates gate combobox and sample picker
        train_model: runs umap on training data, plots results as scatter plot and table
        predict_sample: runs umap on selected sample, plots results as scatter plot and table

    """
    def __init__(self, bus=None, controller=None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.controller = controller

        # --- Create widget, scroll area and layouts to hold the plugin content ---
        self.label = QLabel('Data Processing Example')
        self.label_disabled = QLabel('Data Processing Example: unmixed data not available. Set up the spectral model first.')

        # the content widget goes in a scroll widget, which goes in the PluginWidget
        self.content_widget = QWidget()
        main_layout = QVBoxLayout(self.content_widget)

        # make this widget scrollable and resizeable
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self.content_widget)

        overall_layout = QVBoxLayout(self)
        overall_layout.addWidget(self.label_disabled)
        overall_layout.addWidget(scroll)

        # --- Add gui objects for a data processing workflow ---

        # Add sample picker
        self.picker = OrderedMultiSamplePicker(title="Choose Source Samples for Processing")
        # Add gate selection combobox
        self.gate_combo = QComboBox()
        self.gate_combo.addItem("Select Gate:")  # placeholder for "no selection"
        output_widget = QWidget()
        self.output_layout = QVBoxLayout(output_widget)

        # --- Add gui elements ---
        self.train_button = QPushButton('Train model')
        self.train_button.setToolTip('Runs the training process on concatenated selected samples')
        self.train_button.clicked.connect(self.train_model)

        training_widget = QWidget()
        self.training_layout = QVBoxLayout(training_widget)

        prediction_widget = QWidget()
        self.prediction_layout = QVBoxLayout(prediction_widget)

        main_layout.addWidget(self.label)
        main_layout.addWidget(self.gate_combo)
        main_layout.addWidget(self.picker)
        main_layout.addWidget(self.train_button)
        main_layout.addWidget(training_widget)
        main_layout.addWidget(prediction_widget)
        main_layout.addWidget(output_widget)
        main_layout.addStretch()

        # connect signals:
        # modeChangeRequested: if user switches to this plugin tab, run initialise_gui
        # loadSampleRequested: if user selects a sample from sample browser, run predict_sample
        self.bus.modeChangeRequested.connect(self.initialise_gui)
        self.bus.loadSampleRequested.connect(self.predict_sample)

        self.reducer = None
        self.clusterer = None
        self.label_to_color = None

    def initialise_gui(self, mode):
        # re-iniitialise if user selects this tab
        if mode == plugin_name:
            # only if unmixing already done
            if self.controller.experiment.process['unmixing_matrix'] is not None:
                self.label_disabled.setVisible(False)
                self.content_widget.setVisible(True)

                # populate sample picker if selection wasn't already made; use sample paths relative to raw sample subdirectory
                selection = self.picker.get_ordered_list()
                if not selection:
                    all_samples = self.controller.experiment.samples['all_samples']
                    raw_samples_subdirectory = self.controller.experiment.settings['raw']['raw_samples_subdirectory']
                    source_samples_relative_to_raw = [str(Path(sample).relative_to(raw_samples_subdirectory)) for sample in all_samples if Path(sample).is_relative_to(raw_samples_subdirectory)]
                    self.picker.set_items(source_samples_relative_to_raw)

                # populate gate selection if selection wasn't already made
                if self.gate_combo.currentText() == "Select Gate:":
                    self.gate_combo.clear()
                    self.gate_combo.addItem("Select Gate:")  # placeholder for "no selection"
                    unmixed_gate_names = ['root'] + [g[0] for g in self.controller.unmixed_gating.get_gate_ids()]
                    self.gate_combo.addItems(unmixed_gate_names)

            else:
                self.label_disabled.setVisible(True)
                self.content_widget.setVisible(False)

    def progress_message(self, text):
        # output to stdout
        print(text)

        # output to status bar
        self.bus.statusMessage.emit(text)

        # output within tabbed window
        label = QLabel(text)
        label.setTextFormat(Qt.RichText)
        label.setWordWrap(True)
        self.output_layout.addWidget(label)

    @with_busy_cursor
    def umap_fit_transform(self, data):
        # umap is a heavy function, so put in its own method with busy cursor; run fit and transform to generate reducer and embedding
        import umap
        self.reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, transform_queue_size=1.0, n_components=2).fit(data)
        embedding = self.reducer.transform(data)
        return embedding

    @with_busy_cursor
    def generate_clusterer(self, embedding):
        # Cluster the UMAP output with HDBSCAN
        import hdbscan
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=50, prediction_data=True).fit(embedding)

    @with_busy_cursor
    def predict_new_data(self, data):
        # based on trained model, predict embedding and labels for a new sample
        import hdbscan
        new_embedding = self.reducer.transform(data)
        new_labels, strengths = hdbscan.approximate_predict(self.clusterer, new_embedding)
        return new_embedding, new_labels, strengths

    def train_model(self):
        clear_layout(self.training_layout)

        self.progress_message(f'{datetime.now():%H:%M:%S} Started model training...')

        # get gate, return if not selected
        gate_name = self.gate_combo.currentText()
        if gate_name == "Select Gate:":
            self.progress_message('Please select a gate.')
            return
        else:
            self.progress_message(f'{datetime.now():%H:%M:%S} Using {gate_name} gate')

        # get source samples, return if not selected
        source_samples = self.picker.get_ordered_list()
        if source_samples:
            self.progress_message(f'{datetime.now():%H:%M:%S} Loading samples from FCS files: {source_samples}')
        else:
            self.progress_message(f'Please select at least one source file.')
            return

        # load sample FCS files, concatenate data, apply gating
        # take a copy of the unmixed gating (gates, axis transforms and per-sample gate overrides),
        # so that each sample is gated with its own gates
        sample_gating = SampleGating.from_controller(self.controller)
        # and of the unmixing state: the transfer matrix and each sample's own AutoSpectral AF
        # profile assignment, so that each sample is unmixed as the main window unmixes it
        unmix_snapshot = snapshot_unmix_state(self.controller, include_qc=False)
        pnn = self.controller.data_for_cytometry_plots_unmixed['pnn']
        gated_sample_data_all = []
        for sample_name in source_samples:
            full_sample_path = str(self.controller.experiment_dir / self.controller.experiment.settings['raw']['raw_samples_subdirectory'] / sample_name)
            sample_key = sample_key_for_path(self.controller.experiment_dir, full_sample_path)
            # read_raw_events reads the experiment's raw channels in order (with the channel whitelist when set)
            raw_event_data = read_raw_events(full_sample_path, unmix_snapshot)
            n_events = len(raw_event_data)

            if n_events > 0:
                # apply unmixing: transfer matrix, with per-cell AF correction when the sample has an AF profile
                unmixed_event_data = unmix_events(raw_event_data, unmix_snapshot,
                                                  unmix_snapshot.af_for(sample_key))['unmixed']
                gated_sample_data = apply_gate(sample_gating, sample_key, unmixed_event_data, pnn, gate_name)
                gated_sample_data_all.append(gated_sample_data)
                self.progress_message(f'{datetime.now():%H:%M:%S} Loaded {sample_name}: {len(gated_sample_data)}/{n_events} events within {gate_name}')
            else:
                self.progress_message(f'Warning: 0 events in {sample_name}.')

        gated_sample_data_all = np.concatenate(gated_sample_data_all)
        self.progress_message(f'{datetime.now():%H:%M:%S} Concatenated source data {len(gated_sample_data_all)} events')

        try:
            # train UMAP model fit and transform, then run clusterer
            data = gated_sample_data_all
            self.progress_message(f'{datetime.now():%H:%M:%S} Calculating UMAP reducer and embedding')
            embedding = self.umap_fit_transform(data)
            self.generate_clusterer(embedding)
            self.progress_message(f'{datetime.now():%H:%M:%S} Clustering with HDBSCAN')
            labels = self.clusterer.labels_

            # ---- prepare plot and table output-----
            self.progress_message(f'{datetime.now():%H:%M:%S} Preparing UMAP plot')
            # unique_labels = np.unique(labels).astype(int)
            unique_labels = np.arange(-1, max(labels)+1)
            palette = cc.glasbey

            # use a table as a legend of the clusters: index, colour and count of events
            self.label_to_color = {}
            table_data = []
            for i, unique_label in enumerate(unique_labels):
                l = int(unique_label)
                if l == -1:
                    self.label_to_color[l] = "#7f7f7f"  # Standard Gray for noise
                else:
                    # Use modulo to wrap around if there are > 256 clusters
                    self.label_to_color[l] = palette[i % len(palette)]

                table_data.append({'Index': l, 'Colour': self.label_to_color[l], 'Count': int(np.sum(labels == l))})

            # make a scatter plot of the UMAP embedding with clusters
            plot_colors = [self.label_to_color[l] for l in labels]
            from matplotlib import pyplot as plt
            figure, ax = plt.subplots(1)
            ax.axis('equal')
            ax.scatter(embedding[:, 0], embedding[:, 1], c=plot_colors, s=5)

            self.training_layout.addWidget(QLabel("Training UMAP and clusters"))

            # put plot in a widget and add to the layout
            plot_widget = ExportablePlotWidget(figure, title="UMAP training data and clustering", experiment_dir=self.controller.experiment_dir)
            self.training_layout.addWidget(plot_widget)

            # generate and add table widget
            table_widget = CopyableTableWidget(table_data, table_headers)
            self.training_layout.addWidget(table_widget)

            self.progress_message(f'{datetime.now():%H:%M:%S} Finished model training.')
            self.progress_message('Click on a sample (in the sample browser) to view it in the same UMAP embedding and clusters.')

        except Exception as e:
            self.progress_message(f'Exception: {e}')


    def predict_sample(self, sample_path):
        if self.controller.current_mode == plugin_name:
            if self.reducer and self.clusterer:
                clear_layout(self.prediction_layout)

                gate_name = self.gate_combo.currentText()
                if gate_name == "Select Gate:":
                    self.progress_message('Please select a gate.')
                    return

                sample_gating = SampleGating.from_controller(self.controller)
                pnn = self.controller.data_for_cytometry_plots_unmixed['pnn']

                full_sample_path = str(self.controller.experiment_dir / sample_path)
                sample_key = sample_key_for_path(self.controller.experiment_dir, full_sample_path)
                unmix_snapshot = snapshot_unmix_state(self.controller, [sample_key], include_qc=False)
                raw_event_data = read_raw_events(full_sample_path, unmix_snapshot)
                n_events = len(raw_event_data)

                if n_events > 0:
                    unmixed_event_data = unmix_events(raw_event_data, unmix_snapshot,
                                                      unmix_snapshot.af_for(sample_key))['unmixed']
                    gated_sample_data = apply_gate(sample_gating, sample_key, unmixed_event_data, pnn, gate_name)

                    self.progress_message(f'{datetime.now():%H:%M:%S} Loaded {sample_path}: {len(gated_sample_data)}/{n_events} events within {gate_name}')
                else:
                    self.progress_message(f'Cannot process sample: 0 events in {sample_path}.')
                    return

            try:
                self.progress_message(f'{datetime.now():%H:%M:%S} Calculating UMAP embedding for {sample_path}')
                embedding, labels, strengths = self.predict_new_data(gated_sample_data)

                # ---- prepare plot and table output-----
                self.progress_message(f'{datetime.now():%H:%M:%S} Preparing UMAP plot')
                table_data = []
                unique_labels = self.label_to_color.keys()
                for i, l in enumerate(unique_labels):
                    table_data.append({'Index': l, 'Colour': self.label_to_color[l], 'Count': int(np.sum(labels == l))})

                plot_colors = [self.label_to_color[l] for l in labels]
                from matplotlib import pyplot as plt
                figure, ax = plt.subplots(1)
                ax.axis('equal')
                ax.scatter(embedding[:, 0], embedding[:, 1], c=plot_colors, s=5)

                # put plot in a widget and add to the layout
                self.prediction_layout.addWidget(QLabel("Prediction UMAP and clusters"))
                plot_widget = ExportablePlotWidget(figure, title=f"{str(Path(sample_path).stem)} UMAP prediction data and clustering", experiment_dir=self.controller.experiment_dir)
                self.prediction_layout.addWidget(plot_widget)

                # generate and add table widget
                table_widget = CopyableTableWidget(table_data, table_headers)
                self.prediction_layout.addWidget(table_widget)

                self.progress_message(f'{datetime.now():%H:%M:%S} Finished prediction of {sample_path}.')

            except Exception as e:
                self.progress_message(f'Exception: {e}')





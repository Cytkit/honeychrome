from PySide6.QtCore import QThread, QTimer, Qt, QSettings
from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QComboBox, QLabel, QVBoxLayout, QScrollArea, QMessageBox, QLayout
from pathlib import Path
from honeychrome.controller_components.functions import get_all_subfolders_recursive
from honeychrome.controller_components.statistical_controller import StatisticsCalculator
import honeychrome.settings as settings
from honeychrome.view_components.busy_cursor import with_busy_cursor

from PySide6.QtCore import Qt, QPoint, QRect, QSize

class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, spacing=-1):
        super().__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self.items = []

    def __del__(self):
        del self.items

    def addItem(self, item):
        self.items.append(item)

    def count(self):
        return len(self.items)

    def itemAt(self, index):
        if 0 <= index < len(self.items):
            return self.items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self.items):
            return self.items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientations(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), apply_geometry=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, apply_geometry=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.items:
            size = size.expandedTo(item.minimumSize())
        margin = self.contentsMargins().left()
        size += QSize(2 * margin, 2 * margin)
        return size

    def _do_layout(self, rect, apply_geometry):
        """ The magic happens here: calculating x and y positions """
        x = rect.x()
        y = rect.y()
        line_height = 0

        for item in self.items:
            wid = item.widget()
            space_x = self.spacing()
            space_y = self.spacing()

            next_x = x + item.sizeHint().width() + space_x
            if next_x - space_x > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0

            if apply_geometry:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = next_x
            line_height = max(line_height, item.sizeHint().height())

        return y + line_height - rect.y()

class StatisticalComparisonWidget(QWidget):
    def __init__(self, bus=None, controller=None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.controller = controller
        self.thread = None
        self.statistics_calculator = None
        self.folders = None

        # --- Widgets ---
        content_widget = QWidget()
        main_layout = QVBoxLayout(content_widget)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)  # important!
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll.setWidget(content_widget)

        overall_layout = QVBoxLayout(self)
        overall_layout.addWidget(scroll)

        # refresh and clear all
        self.refresh_button = QPushButton('Recalculate All Plots')
        self.refresh_button.setToolTip('Recalculates all statistical comparisons from all sample files and current gating hierarchy')
        self.refresh_button.clicked.connect(self.refresh_statistical_comparisons)
        self.clear_button = QPushButton('Clear All Plots')
        self.clear_button.setToolTip('Deletes all statistical comparison plots')
        self.clear_button.clicked.connect(self.clear_all_plots)
        self.buttons = QWidget()
        main_layout.addWidget(self.buttons)
        btn_layout = QHBoxLayout(self.buttons)
        btn_layout.addWidget(self.refresh_button)
        btn_layout.addWidget(self.clear_button)
        btn_layout.addStretch()
        self.buttons.setVisible(False)

        # --- statistical plots container
        statistical_plots_container = QWidget()
        self.statistical_plots_container_layout = QVBoxLayout(statistical_plots_container)
        main_layout.addWidget(statistical_plots_container)

        # --- new plot menu ---
        menu = QWidget()
        main_layout.addWidget(menu)
        main_layout.addStretch(100)

        self.sample_set_combo = QComboBox()
        self.plot_type_combo = QComboBox()
        self.gate_combo = QComboBox()
        self.statistic_combo = QComboBox()
        self.channel_combo = QComboBox()
        self.channel_combo.setMaxVisibleItems(100)
        self.channel_combo.setStyleSheet("""
            QComboBox { 
                combobox-popup: 0; max-height: 700px; min-width: 150px; 
            }
        """)

        self.create_button = QPushButton("Calculate!")

        # menu_layout = QHBoxLayout(menu)
        menu_layout = FlowLayout(menu)
        menu_layout.addWidget(QLabel('New Statistical Comparison:'))
        menu_layout.addWidget(self.sample_set_combo)
        menu_layout.addWidget(self.plot_type_combo)
        menu_layout.addWidget(self.gate_combo)
        menu_layout.addWidget(self.statistic_combo)
        menu_layout.addWidget(self.channel_combo)
        menu_layout.addWidget(self.create_button)
        # menu_layout.addStretch()

        # --- Connections ---
        self.sample_set_combo.currentTextChanged.connect(self.on_sample_set_selected)
        self.plot_type_combo.currentTextChanged.connect(self.on_plot_type_selected)
        self.gate_combo.currentTextChanged.connect(self.on_gate_selected)
        self.statistic_combo.currentTextChanged.connect(self.on_statistic_selected)
        self.channel_combo.currentTextChanged.connect(self.on_channel_selected)
        self.create_button.clicked.connect(self.create_statistical_comparison)

        self.initialise()

    def initialise(self):
        # initialise plots (if statistics already defined) and initialise menu
        if self.controller.experiment.statistics:
            self.buttons.setVisible(True)

        #### initialise plots
        old_statistics_plot_widgets = self.findChildren(StatisticsPlotWidget)
        for w in old_statistics_plot_widgets:
            w.deleteLater()

        # while self.statistical_plots_container_layout.count():
        #     plot = self.statistical_plots_container_layout.takeAt(0)
        #     if plot.widget():
        #         plot.widget().deleteLater()

        for statistics_comparison in self.controller.experiment.statistics:
            self.add_statistics_plot(statistics_comparison)

        #### initialise menu
        self.sample_set_combo.clear()
        self.sample_set_combo.addItem("Select Sample Set:")  # placeholder for "no selection"
        source_folder = str(self.controller.experiment_dir / self.controller.experiment.settings['raw']['raw_samples_subdirectory'])
        self.folders = get_all_subfolders_recursive(source_folder, self.controller.experiment_dir)
        all_samples = self.controller.experiment.samples['all_samples']
        samples_by_folder = {str(folder): [sample for sample in all_samples if sample.startswith(str(folder))] for folder in self.folders}
        self.folders = [folder for folder in self.folders if samples_by_folder[str(folder)]]
        sample_sets = [str(folder.relative_to(self.controller.experiment.settings['raw']['raw_samples_subdirectory'])) for folder in self.folders]
        if sample_sets:
            sample_sets[0] = '[All FCS files in experiment folder]'
        self.sample_set_combo.addItems(sample_sets)

        self.plot_type_combo.setVisible(False)
        self.gate_combo.setVisible(False)
        self.statistic_combo.setVisible(False)
        self.channel_combo.setVisible(False)
        self.create_button.setVisible(False)

    def clear_all_plots(self):
        reply = QMessageBox.question(self, "Clear all statistical comparisons", f"This will clear all plots. Are you sure you wish to continue?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.controller.experiment.statistics = []
            self.initialise()

    def add_statistics_plot(self, statistics_comparison):
        plot_widget = StatisticsPlotWidget(statistics_comparison, self.controller, parent=self)
        self.statistical_plots_container_layout.addWidget(plot_widget)

    def on_sample_set_selected(self, text: str):
        if text != 'Select Sample Set:':
            self.plot_type_combo.clear()
            self.plot_type_combo.addItem("Select Chart Type:")  # placeholder for "no selection"
            self.plot_type_combo.addItem("Bar Chart")
            self.plot_type_combo.addItem("Box and Whisker Chart")
            self.plot_type_combo.addItem("1D Histogram Overlay")
            self.plot_type_combo.addItem("Heatmap")
            self.plot_type_combo.setVisible(True)
        else:
            self.plot_type_combo.setVisible(False)
            self.gate_combo.setVisible(False)
            self.statistic_combo.setVisible(False)
            self.channel_combo.setVisible(False)
            self.create_button.setVisible(False)

    def on_plot_type_selected(self, text: str):
        if text != 'Select Chart Type:':
            self.gate_combo.clear()
            self.gate_combo.addItem("Select Gate:")  # placeholder for "no selection"
            self.gate_combo.addItems(self.controller.data_for_cytometry_plots_unmixed['statistics'].keys())
            self.gate_combo.setVisible(True)
        else:
            self.gate_combo.setVisible(False)
            self.statistic_combo.setVisible(False)
            self.channel_combo.setVisible(False)
            self.create_button.setVisible(False)

    def on_gate_selected(self, text: str):
        if text != 'Select Gate:':
            self.statistic_combo.clear()
            if self.plot_type_combo.currentText() == '1D Histogram Overlay':
                self.statistic_combo.addItem("Intensity")
                self.on_statistic_selected('Intensity')
                self.statistic_combo.setVisible(False)
            elif self.plot_type_combo.currentText() == 'Heatmap':
                # Heatmap shows the median intensity of every marker (columns)
                # for each sample (rows) on the selected gate. No per-channel
                # selection is needed, so go straight to the Calculate button.
                self.statistic_combo.addItem("Median Intensity")
                self.statistic_combo.setVisible(False)
                self.channel_combo.setVisible(False)
                self.create_button.setVisible(True)
            else:
                self.statistic_combo.addItem("Select Statistic:")  # placeholder for "no selection"
                self.statistic_combo.addItem("% Total Events")
                self.statistic_combo.addItem("% Parent")
                # self.statistic_combo.addItem("Event Concentration") # disable until standardised
                self.statistic_combo.addItem("Number of Events")
                self.statistic_combo.addItem("Mean Intensity")
                self.statistic_combo.setVisible(True)
        else:
            self.statistic_combo.setVisible(False)
            self.channel_combo.setVisible(False)
            self.create_button.setVisible(False)

    def on_statistic_selected(self, text: str):
        if text == 'Select Statistic:':
            self.channel_combo.setVisible(False)
            self.create_button.setVisible(False)
        elif text == 'Mean Intensity' or text == 'Intensity':
            self.channel_combo.clear()
            self.channel_combo.addItem("Select Channel:")  # placeholder for "no selection"
            self.channel_combo.addItems(self.controller.data_for_cytometry_plots_unmixed['pnn_labels'].values())
            self.channel_combo.setVisible(True)
            self.create_button.setVisible(False)
        else:
            self.channel_combo.setVisible(False)
            self.create_button.setVisible(True)

    def on_channel_selected(self, text: str):
        if text != 'Select Channel:':
            self.create_button.setVisible(True)
        else:
            self.create_button.setVisible(False)

    def create_statistical_comparison(self):
        sample_set = str(self.folders[self.sample_set_combo.currentIndex()-1])
        plot_type = self.plot_type_combo.currentText()
        gate = self.gate_combo.currentText()
        statistics_comparison = {'sample_set': sample_set, 'plot_type': plot_type, 'gate': gate, 'data':None}
        statistic = self.statistic_combo.currentText()
        if statistic == 'Mean Intensity' or statistic == 'Intensity':
            channel = [k for k, v in self.controller.data_for_cytometry_plots_unmixed['pnn_labels'].items() if v == self.channel_combo.currentText()][0]
            statistic += ' ' + self.channel_combo.currentText()
            statistics_comparison['channel'] = channel
            statistics_comparison['antigen channel'] = self.channel_combo.currentText()
        statistics_comparison['statistic'] = statistic

        # add to statistics specification
        self.controller.experiment.statistics.append(statistics_comparison)
        self.setEnabled(False)
        self.refresh_statistical_comparisons()

    def refresh_statistical_comparisons(self):
        self.thread = QThread()
        self.statistics_calculator = StatisticsCalculator(self.bus, self.controller)
        self.statistics_calculator.moveToThread(self.thread)
        self.thread.started.connect(self.statistics_calculator.run)
        self.statistics_calculator.finished.connect(self.thread.quit)
        self.statistics_calculator.finished.connect(self.statistics_calculator.deleteLater)
        self.thread.finished.connect(self._on_thread_finished)
        self.thread.start()

    def _on_thread_finished(self):
        self.thread.deleteLater()
        self.setEnabled(True)
        self.initialise()
        self.bus.autoSaveRequested.emit()




class StatisticsPlotWidget(QWidget):
    def __init__(self, statistics_comparison, controller, parent=None):
        super().__init__(parent)
        self.statistics_comparison = statistics_comparison
        self.controller = controller

        export_filename = self.statistics_comparison['plot_type']
        export_filename += ' ' + '_'.join(Path(self.statistics_comparison['sample_set']).parts)
        export_filename += ' ' + self.statistics_comparison['gate']
        export_filename += ' ' + self.statistics_comparison['statistic']
        if self.statistics_comparison['statistic'] == 'Mean Intensity':
            export_filename += ' ' + self.statistics_comparison['channel']
        self.export_filename = export_filename

        plot_label = QLabel(f'''
            <p>
            Plot type: {self.statistics_comparison['plot_type']}<br/>
            Sample set: {
                'All Samples' if self.statistics_comparison['sample_set'] == self.controller.experiment.settings['raw']['raw_samples_subdirectory'] 
                else str(Path(self.statistics_comparison['sample_set']).relative_to(Path(self.controller.experiment.settings['raw']['raw_samples_subdirectory']), walk_up=True))
            }<br/>
            Gate: {self.statistics_comparison['gate']}<br/>
            Statistic: {self.statistics_comparison['statistic']}<br/>
            {'Channel: ' + self.statistics_comparison['channel'] if self.statistics_comparison['statistic'].startswith('Mean Intensity') 
                                                                    or self.statistics_comparison['statistic'].startswith('Intensity') else ''}
            </p>
        ''')
        plot_label.setTextFormat(Qt.RichText)
        plot_label.setWordWrap(True)
        plot_label.setMinimumWidth(200)
        plot_label.setMaximumWidth(500)

        # Matplotlib imports for embedding
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        # Create matplotlib figure + canvas
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setFixedSize(800, 600)

        self.delete_button = QPushButton('Delete')
        self.delete_button.clicked.connect(self.delete_plot)
        self.export_graphic_button = QPushButton('Export Graphic')
        self.export_graphic_button.clicked.connect(self.export_graphic)
        self.export_csv_button = QPushButton('Export CSV')
        self.export_csv_button.clicked.connect(self.export_csv)
        buttons_layout = QVBoxLayout()
        buttons_layout.addWidget(plot_label)
        buttons_layout.addWidget(self.delete_button)
        buttons_layout.addWidget(self.export_graphic_button)
        buttons_layout.addWidget(self.export_csv_button)
        buttons_layout.addStretch()

        # Layout to hold the canvas
        layout = QHBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.addLayout(buttons_layout)
        layout.addStretch()

        self.draw_plot()

    def delete_plot(self):
        reply = QMessageBox.question(self, "Delete Plot", f"This will delete the current statistical comparison. Are you sure you wish to continue?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.controller.experiment.statistics.remove(self.statistics_comparison)
            self.deleteLater()

    def export_graphic(self):
        self.figure.savefig(f"{self.controller.experiment_dir /self.export_filename}.{settings.graphics_export_format_retrieved}", bbox_inches="tight")
        QMessageBox.information(self, "Exported", f"Exported {settings.graphics_export_format_retrieved} graphic file: \n{self.export_filename}.{settings.graphics_export_format_retrieved}\nto {self.controller.experiment_dir}")

    def export_csv(self):
        from pandas import DataFrame
        DataFrame(self.statistics_comparison['data']).to_csv(f"{self.controller.experiment_dir /self.export_filename}.csv")
        QMessageBox.information(self, "Exported", f"Exported CSV file: \n{self.export_filename}.csv\nto {self.controller.experiment_dir}")


    def draw_plot(self):
        import numpy as np
        import seaborn as sns

        # Clean appearance for publication-like aesthetics
        sns.set_theme(style="whitegrid", font_scale=1.2)
        import matplotlib as mpl
        mpl.rcParams['font.sans-serif'] = ['DejaVu Sans'] + [
            f for f in mpl.rcParams['font.sans-serif'] if f != 'DejaVu Sans'
        ]


        if self.statistics_comparison['data']:
            ax = self.figure.add_subplot(111)
            ax.set_title(self.statistics_comparison['gate'])
            if self.statistics_comparison['plot_type'] == '1D Histogram Overlay':
                import matplotlib.ticker as ticker
                if self.controller.data_for_cytometry_plots_unmixed['transformations'] and self.statistics_comparison['channel'] in self.controller.data_for_cytometry_plots_unmixed['transformations']:
                    transformation = self.controller.data_for_cytometry_plots_unmixed['transformations'][self.statistics_comparison['channel']]
                    x = transformation.step_scale[:-1]
                    for n, sample in enumerate(self.statistics_comparison['data']['Sample']):
                        y = self.statistics_comparison['data'][self.statistics_comparison['statistic']][n]
                        colour = settings.line_colors[n % len(settings.line_colors)]
                        ax.plot(x, y, color=colour, linewidth=2, label=sample)
                        ax.fill_between(x, y, color=colour, alpha=0.3)

                    # Some transforms (e.g. the default transform used for the
                    # Time channel) have no custom tick scheme and return None
                    # from ticks(). Fall back to matplotlib's default ticks in
                    # that case instead of crashing on unpacking None.
                    ticks = transformation.ticks()
                    if ticks is not None:
                        [minor_ticks, major_ticks] = ticks
                        major_tick_values = [val for (val, name) in major_ticks]
                        major_tick_labels = [name for (val, name) in major_ticks]
                        minor_tick_values = [val for (val, name) in minor_ticks]
                        ax.xaxis.set_major_locator(ticker.FixedLocator(major_tick_values))
                        ax.xaxis.set_major_formatter(ticker.FixedFormatter(major_tick_labels))
                        ax.xaxis.set_minor_locator(ticker.FixedLocator(minor_tick_values))

                        ax.tick_params(axis='x', which='major', length=10, width=2, color='black')
                        ax.tick_params(axis='x', which='minor', length=5, width=1, color='gray')

                        ax.grid(which='major', linestyle='-', alpha=0.8)
                        ax.grid(which='minor', linestyle='-', alpha=0.4)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                    ax.set_xlabel(self.statistics_comparison['statistic'])
                    ax.set_ylabel('Count')
                    ax.set_xlim([transformation.step_scale[0], transformation.step_scale[-1]])
                else:
                    ax.text(0,0, f"Error: transform missing for {self.statistics_comparison['channel']}. \nHas it been deleted from the spectral model?")
            elif self.statistics_comparison['plot_type'] == 'Heatmap':
                import pandas as pd
                data = self.statistics_comparison['data']
                samples = data.get('Sample', [])
                markers = [k for k in data.keys() if k not in ('Sample', 'Group', 'Category')]
                if samples and markers:
                    matrix = pd.DataFrame({marker: data[marker] for marker in markers}, index=samples)
                    sns.heatmap(
                        matrix, ax=ax, cmap='viridis',
                        cbar_kws={'label': self.statistics_comparison['statistic']},
                    )
                    ax.set_xlabel('Marker')
                    ax.set_ylabel('Sample')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                else:
                    ax.text(0.5, 0.5, 'Heatmap: no marker data available.', ha='center', va='center')
            else:
                if self.statistics_comparison['depth'] == 3:
                    x = 'Category'
                    hue = 'Group'
                    palette = None
                elif self.statistics_comparison['depth'] == 2:
                    if self.statistics_comparison['plot_type'] == 'Box and Whisker Chart':
                        x = 'Group'
                        hue = None
                        palette = None
                    else:
                        x = 'Group'
                        hue = 'Sample'
                        palette = None
                else:
                    x = 'Sample'
                    hue = None
                    palette = 'Set2'

                max_length = max([len(x_label) for x_label in self.statistics_comparison['data'][x]])
                if max_length > 20:
                    rotation = 60
                elif max_length > 16:
                    rotation = 40
                elif max_length > 12:
                    rotation = 20
                else:
                    rotation = 0

                y = self.statistics_comparison['statistic']

                if self.statistics_comparison['plot_type'] == 'Box and Whisker Chart':
                    # compatibility with seaborn 0.14
                    sns.boxplot(
                        data=self.statistics_comparison['data'],
                        x=x,
                        y=y,
                        hue=hue if hue is not None else x,   # use x for colouring when no hue grouping
                        legend=False if hue is None else True,
                        showfliers=True,
                        width=0.6,
                        palette="Set2",
                        ax=ax,
                    )
                else:
                    sns.barplot(
                        data=self.statistics_comparison['data'],
                        x=x,
                        y=y,
                        hue=hue,
                        width=0.6,
                        palette=palette,
                        ax=ax,
                    )

                if self.statistics_comparison['depth'] == 3 or (self.statistics_comparison['depth'] == 2 and self.statistics_comparison['plot_type'] == 'Box and Whisker Chart'):
                    sns.swarmplot(
                        data=self.statistics_comparison['data'],
                        x=x,
                        y=y,
                        hue=hue,
                        dodge=True,
                        color="k",
                        alpha=0.55,
                        size=10,
                        ax=ax,
                    )

                # Remove duplicate legends (swarmplot adds more)
                handles, labels = ax.get_legend_handles_labels()

                if hue:
                    indices = [labels.index(label) for label in set(labels)]
                    ax.legend([handles[index] for index in indices], [labels[index] for index in indices], title=hue, bbox_to_anchor=(1.05, 1), loc="upper left")

                # ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right')
                ax.tick_params(axis='x', rotation=rotation, labelrotation=rotation)
                ax.set_xlabel("")

        else:
            self.figure.suptitle('No samples in selected sample set!', color='red')

        self.canvas.draw()



if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    from pathlib import Path
    from honeychrome.controller import Controller
    from honeychrome.view_components.event_bus import EventBus

    app = QApplication([])

    kc = Controller()
    base_directory = Path.home() / 'spectral_cytometry'
    experiment_name = base_directory / '20240620 Spectral Symposium-poor cell unmixed'
    experiment_path = experiment_name.with_suffix('.kit')
    kc.load_experiment(experiment_path) # note this loads first sample too and runs calculate all histograms and statistics
    kc.experiment.statistics = [
      {
        "sample_set": "Raw/Samples/Lung",
        "plot_type": "Bar Chart",
        "gate": "activated",
        "data": {
          "Sample": [
            "C1 Lung_WT_001_Samples",
            "C2 Lung_WT_002_Samples",
            "C3 Lung_GFP_003_Samples",
            "C4 Lung_GFP_004_Samples"
          ],
          "Mean Intensity A10 BUV805": [
            16578.85450367425,
            14476.030268458402,
            12903.001374391868,
            11949.678970245182
          ]
        },
        "statistic": "Mean Intensity A10 BUV805",
        "channel": "A10 BUV805",
        "depth": 1
      },
        {"sample_set": "Raw/Samples", "plot_type": "Bar Chart", "gate": "Singlets", "data": {"Sample": ["E1 Brain_WT_001_Samples", "E2 Brain_WT_002_Samples", "E3 Brain_GFP_003_Samples", "E4 Brain_GFP_004_Samples", "C1 Lung_WT_001_Samples", "C2 Lung_WT_002_Samples", "C3 Lung_GFP_003_Samples", "C4 Lung_GFP_004_Samples", "A1 Spleen_WT_001_Samples", "A2 Spleen_WT_002_Samples", "A3 Spleen_GFP_003_Samples", "A4 Spleen_GFP_004_Samples"], "Group": ["Brain", "Brain", "Brain", "Brain", "Lung", "Lung", "Lung", "Lung", "Spleen", "Spleen", "Spleen", "Spleen"], "Number of Events": [92469, 313448, 185281, 185303, 387313, 262606, 322113, 291412, 249984, 275581, 221640, 201272]}, "statistic": "Number of Events", "depth": 2},
        {"sample_set": "Raw/Samples", "plot_type": "Box and Whisker Chart", "gate": "activated", "data": {"Sample": ["E1 Brain_WT_001_Samples", "E2 Brain_WT_002_Samples", "E3 Brain_GFP_003_Samples", "E4 Brain_GFP_004_Samples", "C1 Lung_WT_001_Samples", "C2 Lung_WT_002_Samples", "C3 Lung_GFP_003_Samples", "C4 Lung_GFP_004_Samples", "A1 Spleen_WT_001_Samples", "A2 Spleen_WT_002_Samples", "A3 Spleen_GFP_003_Samples", "A4 Spleen_GFP_004_Samples"], "Group": ["Brain", "Brain", "Brain", "Brain", "Lung", "Lung", "Lung", "Lung", "Spleen", "Spleen", "Spleen", "Spleen"], "Mean Intensity A10 BUV805": [19777.12693217841, 20099.32931663595, 20105.138540011478, 19716.291252580784, 16578.85450367425, 14476.030268458402, 12903.001374391868, 11949.678970245182, 11856.829612816671, 13037.226422102669, 16893.346546687982, 16695.930836221964]}, "channel": "A10 BUV805", "statistic": "Mean Intensity A10 BUV805", "depth": 2},
        {"sample_set": "Raw/Samples", "plot_type": "Box and Whisker Chart", "gate": "P1", "data": {"Sample": ["E1 Brain_WT_001_Samples", "E2 Brain_WT_002_Samples", "E3 Brain_GFP_003_Samples", "E4 Brain_GFP_004_Samples", "C1 Lung_WT_001_Samples", "C2 Lung_WT_002_Samples", "C3 Lung_GFP_003_Samples", "C4 Lung_GFP_004_Samples", "A1 Spleen_WT_001_Samples", "A2 Spleen_WT_002_Samples", "A3 Spleen_GFP_003_Samples", "A4 Spleen_GFP_004_Samples"], "Group": ["Brain", "Brain", "Brain", "Brain", "Lung", "Lung", "Lung", "Lung", "Spleen", "Spleen", "Spleen", "Spleen"], "% Parent": [7.925229558373414e-05, 3.1504927370640767e-06, 3.2453834420536785e-05, 8.122553080884383e-06, 0.002252513595052467, 0.003785737084492271, 0.0009384225252237184, 0.00402574728368941, 2.0293305914822232e-05, 1.5132103261472656e-05, 2.5617776563305406e-05, 3.9583548754006825e-05]}, "statistic": "% Parent", "depth": 2}
    ]

    kc.set_mode('Statistics')

    widget = StatisticalComparisonWidget(
        bus=EventBus(), controller=kc
    )
    widget.show()
    app.exec()

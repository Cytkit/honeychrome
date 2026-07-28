import functools
import os
from pathlib import Path

from honeychrome.controller_components.functions import q_settings
# load_tabbed_plugins is imported inside _PluginLoaderWorker.run() to keep it off the main thread
# from honeychrome.controller_components.plugin_loaders import load_tabbed_plugins
from honeychrome.settings import file_extension, experiments_folder
from honeychrome.view_components.configuration_dialogs import AppConfigDialog, ExperimentSettings, InstrumentConfigDialog
from honeychrome.view_components.help_toggle_widget import HelpToggleWidget
from honeychrome.view_components.import_fcs_files_widget import ImportFCSFilesWidget
from honeychrome.view_components.oscilloscope_widget import OscilloscopeWidget
from honeychrome.view_components.statistical_plotter import StatisticalComparisonWidget

os.environ["QT_LOGGING_RULES"] = "qt.core.qobject.connect=false" #suppress pyqtgraph graphicslayoutwidget warning

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow, QFileDialog, QWidget, QHBoxLayout, QSplitter, QTabWidget, QStatusBar, QVBoxLayout, QLabel, QProgressBar, QScrollArea, QPushButton, QMenu, QGroupBox, QSpinBox
from PySide6.QtCore import Qt, QSettings, QByteArray, Slot, QTimer, QThread, Signal as QtSignal

from honeychrome.view_components.gating_hierarchy_widget import GatingHierarchyWidget
from honeychrome.view_components.heatmap_viewedit import HeatmapViewEditor
from honeychrome.view_components.nxn_grid import NxNGrid
from honeychrome.view_components.profiles_viewer import ProfilesViewer
from honeychrome.view_components.scatter_cleaning_viewer import ScatterCleaningViewer
from honeychrome.view_components.cosine_qc_viewer import CosineQCViewer
from honeychrome.view_components.spectral_model_editor import SpectralControlsEditor
from honeychrome.view_components.cytometry_grid_widget import CytometryGridWidget
from honeychrome.view_components.cytometry_toolbar import CytometryToolbar
from honeychrome.view_components.acquisition_widget import AcquisitionWidget
from honeychrome.view_components.gains_widget import GainsWidget
from honeychrome.view_components.icon_loader import icon
from honeychrome.view_components.sample_widget import SampleWidget
from honeychrome.view_components.help_texts import process_help_text
from honeychrome.view_components.autospectral_tab import AutoSpectralTab

base_directory = Path.home() / experiments_folder

import logging
logger = logging.getLogger(__name__)

class _PluginLoaderWorker(QThread):
    """Imports plugin modules + runs _bootstrap() on a background thread."""
    finished_loading = QtSignal(dict)   # emits {module_name: module}

    def __init__(self, parent=None):
        super().__init__(parent)
        #self._bus = bus
        #self._controller = controller

    def run(self):
        from honeychrome.controller_components.plugin_loaders import load_plugin_modules
        loaded_modules = load_plugin_modules()
        # emit modules only, no widgets yet
        self.finished_loading.emit(loaded_modules)

class MainWindow(QMainWindow):
    def __init__(self, bus=None, controller=None, parent=None, is_dark=False):
        super().__init__(parent)

        self.is_dark = is_dark
        # connect all signals
        self.bus = bus
        # connect to data
        self.controller = controller
        # connect to QSettings
        self.settings = QSettings("honeychrome", "MainWindow")

        # autosave if window closed
        self.destroyed.connect(lambda: self.controller.save_experiment())

        ##### build menus/buttons/widgets
        # left splitter
        self.sample_tree = SampleWidget(bus, parent=self, controller=self.controller)
        self.acquisition_widget = AcquisitionWidget(bus, parent=self)
        self.gains_widget = GainsWidget(bus, parent=self)
        self.sample_tree.set_selection(self.controller.current_sample_path)

        # raw
        self.gating_tree_raw = GatingHierarchyWidget(bus, mode='raw')
        self.cytometry_grid_raw = CytometryGridWidget(bus, parent=self, mode='raw', gating_tree=self.gating_tree_raw)
        self.cytometry_toolbar_raw = CytometryToolbar(bus, parent=self.cytometry_grid_raw)
        self.cytometry_grid_raw.set_toolbar(self.cytometry_toolbar_raw)
        self.gating_tree_raw.set_toolbar(self.cytometry_toolbar_raw)

        # process
        self.help_spectral_process = HelpToggleWidget(text=process_help_text, parent=self)
        self.spectral_controls_editor = SpectralControlsEditor(bus, self.controller, parent=self)
        self.profiles_viewer = ProfilesViewer(bus, self.controller, parent=self)
        self.scatter_cleaning_viewer = ScatterCleaningViewer(bus, self.controller, parent=self)
        self.cosine_qc_viewer = CosineQCViewer(bus, self.controller, parent=self)
        self.similarity_viewer = HeatmapViewEditor(bus, self.controller, 'similarity_matrix', is_dark, parent=self)
        self.hotspot_viewer = HeatmapViewEditor(bus, self.controller, 'hotspot_matrix', is_dark, parent=self)
        self.unmixing_viewer = HeatmapViewEditor(bus, self.controller, 'unmixing_matrix', is_dark, parent=self)
        self.compensation_editor = HeatmapViewEditor(bus, self.controller, 'spillover', is_dark, parent=self)
        self.nxn_viewer = NxNGrid(bus, self.controller, is_dark=is_dark, parent=self)

        # unmixed
        self.gating_tree_unmixed = GatingHierarchyWidget(bus, mode='unmixed')
        self.cytometry_grid_unmixed = CytometryGridWidget(bus, parent=self, mode='unmixed', gating_tree=self.gating_tree_unmixed)
        self.tip_unmixed = QWidget()
        container_layout = QVBoxLayout(self.tip_unmixed)
        container_layout.addWidget(QLabel('Unmixed data requires a spectral model.\nTo see unmixed data, first define the spectral controls (spectral process tab) and calculate the unmixing matrix.'))
        container_layout.addStretch(100)
        self.cytometry_toolbar_unmixed = CytometryToolbar(bus, parent=self.cytometry_grid_unmixed)
        self.cytometry_grid_unmixed.set_toolbar(self.cytometry_toolbar_unmixed)
        self.gating_tree_unmixed.set_toolbar(self.cytometry_toolbar_unmixed)

        # statistics
        self.tip_statistics = QWidget()
        container_layout = QVBoxLayout(self.tip_statistics)
        container_layout.addWidget(QLabel('Statistical comparison of samples requires unmixed data.\nTo make a statistical comparison, first define the spectral model (spectral process tab) and define at least one gate on the unmixed data (unmixed data tab).'))
        container_layout.addStretch(100)
        self.statistical_comparison_widget = StatisticalComparisonWidget(bus, self.controller)

        # --- Menu Bar ---
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        edit_menu = menubar.addMenu("&Edit")
        view_menu = menubar.addMenu("&View")
        help_menu = menubar.addMenu("&Help")

        # file menu actions
        action_new = QAction(icon('file'), "New Experiment (default template)", self)
        action_new.triggered.connect(self.bus.newExpRequested)
        action_new_from_this_template = QAction(icon('file-plus'), "New Experiment (this experiment as template)", self)
        action_new_from_this_template.triggered.connect(self.bus.newExpRequestedFromThisTemplate)
        action_new_from_choose_template = QAction(icon("file-spark"), "New Experiment (choose template)", self)
        action_new_from_choose_template.triggered.connect(self.bus.newExpRequestedFromTemplate)
        action_open = QAction(icon('file-upload'), "Open Experiment", self)
        action_open.triggered.connect(self.bus.openExpRequested)
        action_save = QAction(icon('file-download'), "Save Experiment As Template", self)
        action_save.triggered.connect(self.bus.saveAsTemplateRequested)
        action_new_sample = QAction(icon('square-plus'), "New Sample", self)
        action_new_sample.triggered.connect(self.bus.showNewSampleWidget)
        action_batch_add_samples = QAction(icon('files'), "Batch Add Samples from CSV", self)
        action_batch_add_samples.triggered.connect(self.bus.batchAddSamples)
        action_import_fcs_files = QAction(icon('files'), "Import FCS Files", self)
        action_import_fcs_files.triggered.connect(self.open_import_fcs_files_widget)
        action_export_all = QAction(icon('files'), "Batch Export FCS Files", self)
        action_export_all.triggered.connect(self.bus.showExportModal)
        action_generate_report = QAction(icon('file-type-docx'), "Generate Sample Report", self)
        action_generate_report.triggered.connect(self.bus.generateSampleReport)
        action_quit = QAction(icon('logout'), "Quit", self)
        action_quit.triggered.connect(self.close)


        # Create Recent Files submenu
        recent_menu = QMenu("Open Recent Experiment", self)
        self.populate_recent_menu(recent_menu)

        file_menu.addAction(action_new)
        file_menu.addAction(action_new_from_this_template)
        file_menu.addAction(action_new_from_choose_template)
        file_menu.addAction(action_open)
        file_menu.addMenu(recent_menu)
        file_menu.addAction(action_save)
        file_menu.addAction(action_new_sample)
        file_menu.addAction(action_batch_add_samples)
        file_menu.addAction(action_import_fcs_files)
        file_menu.addAction(action_export_all)
        file_menu.addAction(action_generate_report)
        file_menu.addAction(action_quit)

        # edit menu
        action_app_config = QAction(icon('adjustments'), "App Configuration", self)
        action_app_config.triggered.connect(self.app_config)
        action_instrument_config = QAction(icon('tool'), "Instrument Configuration", self)
        action_instrument_config.triggered.connect(self.instrument_config)
        action_experiment_settings = QAction(icon('settings'), "Experiment Settings", self)
        action_experiment_settings.triggered.connect(self.expt_settings)
        edit_menu.addAction(action_app_config)
        edit_menu.addAction(action_instrument_config)
        edit_menu.addAction(action_experiment_settings)

        # view menu
        # action_acquire = QAction(icon('player-play'), "Acquisition Panel", self)
        action_acquire = QAction("Acquisition Panel", self)
        action_acquire.setCheckable(True)
        action_acquire.setChecked(self.settings.value("acquisition_visible", True, type=bool))
        action_acquire.triggered.connect(lambda : self.acquisition_widget.setVisible(action_acquire.isChecked()))
        action_acquire.triggered.connect(lambda : self._save_visibility_state())
        view_menu.addAction(action_acquire)

        # action_gains = QAction(icon('adjustments'), "Gains Panel", self)
        action_gains = QAction("Gains Panel", self)
        action_gains.setCheckable(True)
        action_gains.setChecked(self.settings.value("gains_visible", True, type=bool))
        action_gains.triggered.connect(lambda : self.gains_widget.setVisible(action_gains.isChecked()))
        action_gains.triggered.connect(lambda : self._save_visibility_state())
        view_menu.addAction(action_gains)

        action_oscilloscope = QAction(icon('wave-sine'), "Oscilloscope Viewer", self)
        action_oscilloscope.triggered.connect(self.open_oscilloscope_widget)
        view_menu.addAction(action_acquire)
        view_menu.addAction(action_gains)
        view_menu.addAction(action_oscilloscope)

        # help menu
        action_forum = QAction(icon('bubble-text'), "Discussions", self)
        action_forum.triggered.connect(lambda: self.bus.popupMessage.emit('Do you want to ask a question, or discuss an issue or a feature request? Visit the <a href="https://github.com/salmansamson/honeychrome/discussions">Honeychrome discussions page on Github</a>.'))
        action_about = QAction(icon('carambola'), "About Honeychrome", self)
        action_about.triggered.connect(self.bus.aboutHoneychrome.emit)
        help_menu.addAction(action_forum)
        help_menu.addAction(action_about)


        # --- Central Widget Layout ---
        central_widget = QWidget()
        main_layout = QHBoxLayout()

        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel (samples list, acquisition panel, gains, dock)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)

        left_layout.addWidget(self.sample_tree)
        left_layout.addWidget(self.acquisition_widget)
        left_layout.addWidget(self.gains_widget)

        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self.splitter.addWidget(left_container)

        # Right panel (tab widget)
        self.tabs = QTabWidget()

        # --- Raw ---
        self.raw_tab = QWidget()
        self.raw_layout = QVBoxLayout()
        self.raw_layout.setContentsMargins(0, 0, 0, 0)
        self.raw_layout.setSpacing(0)
        self.raw_tab.setLayout(self.raw_layout)
        self.raw_layout.addWidget(self.cytometry_toolbar_raw)
        self.raw_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.raw_splitter.addWidget(self.cytometry_grid_raw)
        self.raw_splitter.addWidget(self.gating_tree_raw)
        self.gating_tree_raw.setParent(self.raw_splitter)
        self.raw_splitter.setStretchFactor(0, 1)
        self.raw_splitter.setStretchFactor(1, 0)
        self.raw_layout.addWidget(self.raw_splitter)
        self.tabs.addTab(self.raw_tab, "Raw Data")

        # --- Process ---
        self.process_tab = QWidget()
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.process_tab)
        scroll_area.setWidgetResizable(True)
        self.process_layout = QVBoxLayout()
        self.process_layout.addWidget(self.help_spectral_process)
        self.process_layout.addWidget(self.spectral_controls_editor)
        self.process_layout.addWidget(self.profiles_viewer)

        # ---- AutoSpectral Control Cleaning boxed section ----
        self.cleaning_section = QGroupBox()
        self.cleaning_section.setStyleSheet("""
            QGroupBox {
                border: 2px solid #cccccc;
                border-radius: 6px;
                margin-top: 4px;
                padding: 8px;
            }
        """)
        cleaning_box_layout = QVBoxLayout(self.cleaning_section)
        cleaning_box_layout.setContentsMargins(8, 8, 8, 8)
        cleaning_box_layout.setSpacing(6)

        from honeychrome.settings import heading_style
        cleaning_title = QLabel("AutoSpectral Control Cleaning")
        cleaning_title.setStyleSheet(heading_style)
        cleaning_box_layout.addWidget(cleaning_title)

        clean_btn_row = QHBoxLayout()
        clean_btn_row.addWidget(self.spectral_controls_editor.clean_controls_btn)
        clean_btn_row.addStretch()
        cleaning_box_layout.addLayout(clean_btn_row)

        cleaning_box_layout.addWidget(self.profiles_viewer._hist_toggle)
        cleaning_box_layout.addWidget(self.profiles_viewer._hist_panel)
        cleaning_box_layout.addWidget(self.scatter_cleaning_viewer)
        cleaning_box_layout.addWidget(self.cosine_qc_viewer)

        self.cleaning_section.setVisible(False)
        self.process_layout.addWidget(self.cleaning_section)
        # ---- end cleaning section ----

        # --- Heatmap zoom control ---
        zoom_row = QHBoxLayout()
        zoom_label = QLabel("Heatmap cell size:")
        self._heatmap_cell_size = 60
        self._heatmap_zoom_minus = QPushButton("−")
        self._heatmap_zoom_plus = QPushButton("+")
        self._heatmap_zoom_minus.setFixedWidth(30)
        self._heatmap_zoom_plus.setFixedWidth(30)
        self._heatmap_zoom_minus.setToolTip("Decrease heatmap cell size")
        self._heatmap_zoom_plus.setToolTip("Increase heatmap cell size")
        self._heatmap_zoom_minus.clicked.connect(self._heatmap_zoom_out)
        self._heatmap_zoom_plus.clicked.connect(self._heatmap_zoom_in)
        zoom_row.addWidget(zoom_label)
        zoom_row.addWidget(self._heatmap_zoom_minus)
        zoom_row.addWidget(self._heatmap_zoom_plus)
        zoom_row.addStretch()
        self.process_layout.addLayout(zoom_row)
        # --- end zoom control ---

        self.process_layout.addWidget(self.similarity_viewer)
        self.process_layout.addWidget(self.hotspot_viewer)
        self.process_layout.addWidget(self.unmixing_viewer)
        self.process_layout.addWidget(self.compensation_editor)
        self.process_layout.addWidget(self.nxn_viewer)
        self.process_layout.addStretch()
        self.process_tab.setLayout(self.process_layout)
        self.tabs.addTab(scroll_area, "Spectral Process")

        # --- AutoSpectral AF tab ---
        self.autospectral_widget = AutoSpectralTab(bus, controller, parent=self)
        autospectral_tab_widget = QWidget()
        autospectral_tab_layout = QVBoxLayout()
        autospectral_tab_layout.setContentsMargins(0, 0, 0, 0)
        autospectral_tab_layout.setSpacing(0)
        autospectral_tab_widget.setLayout(autospectral_tab_layout)
        autospectral_tab_layout.addWidget(self.autospectral_widget)
        self.tabs.addTab(autospectral_tab_widget, "AutoSpectral AF")

        # --- Unmixed ---
        self.unmixed_tab = QWidget()
        self.unmixed_layout = QVBoxLayout()
        self.unmixed_layout.setContentsMargins(0, 0, 0, 0)
        self.unmixed_layout.setSpacing(0)
        self.unmixed_tab.setLayout(self.unmixed_layout)
        self.unmixed_layout.addWidget(self.tip_unmixed)
        self.unmixed_layout.addWidget(self.cytometry_toolbar_unmixed)
        self.unmixed_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.unmixed_splitter.addWidget(self.cytometry_grid_unmixed)
        self.unmixed_splitter.addWidget(self.gating_tree_unmixed)
        self.gating_tree_unmixed.setParent(self.unmixed_splitter)
        self.unmixed_splitter.setStretchFactor(0, 1)
        self.unmixed_splitter.setStretchFactor(1, 0)
        self.unmixed_layout.addWidget(self.unmixed_splitter)
        self.tabs.addTab(self.unmixed_tab, "Unmixed Data")

        # --- Statistics ---
        self.statistics_tab = QWidget()
        self.statistics_layout = QVBoxLayout()
        self.statistics_layout.setContentsMargins(0, 0, 0, 0)
        self.statistics_layout.setSpacing(0)
        self.statistics_tab.setLayout(self.statistics_layout)
        self.statistics_layout.addWidget(self.tip_statistics)
        self.statistics_layout.addWidget(self.statistical_comparison_widget)
        self.tabs.addTab(self.statistics_tab, "Statistics")

        # --- visibility wiring ---
        self.bus.cleaningActivated.connect(self.cleaning_section.setVisible)
        self.bus.cleaningResultsReady.connect(self._on_cleaning_results_ready)

        # Restore cleaning UI state
        from PySide6.QtCore import QSettings as _QSettings
        _qs = _QSettings("honeychrome", "app_configuration")
        if _qs.value("show_autospectral_cleaning", False, type=bool):
            _help_expanded = _qs.value("cleaning_help_expanded", True, type=bool)
            self.spectral_controls_editor.activate_cleaning_checkbox.setChecked(True)
            # Restore help expanded/collapsed state (setChecked auto-expands help;
            # override here if the user had collapsed it last time)
            self.spectral_controls_editor.cleaning_help_widget.help_label.setVisible(_help_expanded)
            self.spectral_controls_editor.cleaning_help_widget.help_button.setText(
                "Hide Help" if _help_expanded else "Show Cleaning Help"
            )

        # --- Plugin tabs (placeholders + background load) ---
        self._plugin_placeholders = {}
        self._bus_ref = bus
        self._controller_ref = controller
        import ast
        _ps = QSettings("honeychrome", "app_configuration")
        _plugin_dir = Path.home() / experiments_folder / "plugins"
        for _fp in Path(_plugin_dir).glob("*_tab.py"):
            if _ps.value(f"EnablePlugin_{_fp}", False, type=bool):
                _name = _fp.stem.replace('_tab', '').replace('_', ' ').title()
                try:
                    _tree = ast.parse(_fp.read_text(encoding='utf-8'))
                    for _node in ast.walk(_tree):
                        if isinstance(_node, ast.Assign):
                            for _t in _node.targets:
                                if isinstance(_t, ast.Name) and _t.id == 'plugin_name':
                                    if isinstance(_node.value, ast.Constant):
                                        _name = _node.value.value
                except Exception:
                    pass
                _ph = QWidget()
                _ph_layout = QVBoxLayout(_ph)
                _ph_layout.setContentsMargins(20, 20, 20, 20)
                _ph_label = QLabel("Loading…")
                _ph_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
                _ph_layout.addWidget(_ph_label)
                _ph_layout.addStretch()
                _idx = self.tabs.addTab(_ph, _name)
                _module_name = f"{_plugin_dir.name}.{_fp.stem}"
                self._plugin_placeholders[_module_name] = _idx
        self._plugin_loader = _PluginLoaderWorker(parent=self)
        self._plugin_loader.finished_loading.connect(self._on_plugins_loaded)
        self._plugin_loader.start()

        # Connect to tab change
        self.tabs.currentChanged.connect(self.on_tab_changed)

        # add to splitter
        self.splitter.addWidget(self.tabs)

        main_layout.addWidget(self.splitter)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # --- Status Bar ---
        status = QStatusBar()
        self.setStatusBar(status)
        self.statusBar().showMessage("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximumWidth(250)
        status.addPermanentWidget(self.progress_bar)
        self.progress_bar.setVisible(False)  # Hide initially
        self.bus.progress.connect(self.update_progress)
        self.bus.statusMessage.connect(self.status_message)

        # widgets visibility state
        self._restore_visibility_state()

        if not self.controller.experiment_compatible_with_acquisition:
            action_new_sample.setEnabled(False)
            action_batch_add_samples.setEnabled(False)
            action_acquire.setEnabled(False)
            action_gains.setEnabled(False)
            action_oscilloscope.setEnabled(False)
            self.acquisition_widget.setVisible(False)
            self.gains_widget.setVisible(False)

        self.bus.openImportFCSWidget.connect(self.open_import_fcs_files_widget)


    @Slot(dict)
    def _on_plugins_loaded(self, loaded_modules):
        from honeychrome.controller_components.plugin_loaders import instantiate_plugin_widgets
        tab_plugins = instantiate_plugin_widgets(loaded_modules, self._bus_ref, self._controller_ref)
        self.tab_plugins = tab_plugins
        for module_name, plugin in tab_plugins.items():
            widget = plugin['widget']
            tab = QWidget()
            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            tab.setLayout(layout)
            layout.addWidget(widget)
            idx = self._plugin_placeholders.get(module_name)
            if idx is not None:
                self.tabs.removeTab(idx)
                self.tabs.insertTab(idx, tab, plugin['module'].plugin_name)
            else:
                self.tabs.addTab(tab, plugin['module'].plugin_name)
        logger.info(f'MainWindow: {len(tab_plugins)} plugin tab(s) loaded')


    def populate_recent_menu(self, recent_menu: QMenu):
        recent_menu.clear()  # Clear old actions before repopulating
        recent_files = q_settings.value("recent_files", [])
        recent_files = [file for file in recent_files if Path(file).exists()]

        if not recent_files:
            # Add disabled placeholder
            empty_action = QAction("(No recent experiments)", self)
            empty_action.setEnabled(False)
            recent_menu.addAction(empty_action)
            return

        for path in recent_files:
            action = QAction(path, self)
            # Use functools.partial to bind filename to slot
            action.triggered.connect(functools.partial(self.bus.loadExpRequested.emit, path))
            recent_menu.addAction(action)

    @Slot(int, int)
    def update_progress(self, n, m):
        self.progress_bar.setMaximum(m)
        self.progress_bar.setValue(n)
        if n != m:
            self.progress_bar.setVisible(True)
            self.statusBar().showMessage(f"Processing {n}/{m}")
        else:
            self.statusBar().showMessage("Ready")
            QTimer.singleShot(100, lambda: self.progress_bar.setVisible(False))

    @Slot(str)
    def status_message(self, message):
        self.statusBar().showMessage(message)

    def on_tab_changed(self, index):
        tab_name = self.tabs.tabText(index)
        logger.info(f'View: tab changed {tab_name}')
        self.bus.modeChangeRequested.emit(tab_name)

    def closeEvent(self, event):
        self.save_state()

        # Flush every plugin's own persisted state before tearing down.
        for plugin in getattr(self, 'tab_plugins', {}).values():
            widget = plugin.get('widget')
            if widget is not None and hasattr(widget, 'save_state'):
                try:
                    widget.save_state()
                except Exception:
                    logger.exception(
                        f"Failed to save state for plugin {plugin.get('module')}"
                    )

        # Clean up child widgets
        for child in self.findChildren(QWidget):
            child.deleteLater()

        # Disconnect all signals
        try:
            self.blockSignals(True)
        except:
            pass

        super().closeEvent(event)

        # Schedule widget for deletion
        self.deleteLater()
        event.accept()

    def app_config(self):
        dialog = AppConfigDialog(self, self.bus)
        dialog.exec()

    def instrument_config(self):
        dialog = InstrumentConfigDialog(self)
        dialog.exec()

    def expt_settings(self):
        dialog = ExperimentSettings(self.controller.experiment, self.bus, self)
        dialog.exec()

    def open_oscilloscope_widget(self):
        widget = OscilloscopeWidget(self.controller, self.bus)
        if widget.isHidden():
            widget.show()
        else:
            widget.raise_()  # Bring to top
            widget.activateWindow()  # Give focus

    @Slot(bool)
    def open_import_fcs_files_widget(self, failed_to_load_sample_warning=False):
        dialog = ImportFCSFilesWidget(self, self.bus, self.controller, failed_to_load_sample_warning)
        dialog.show()

    def save_state(self):
        # QByteArray returned by saveState() — stored directly in QSettings
        self.settings.setValue("main_splitter_state", self.splitter.saveState())
        self.settings.setValue("raw_splitter_state", self.raw_splitter.saveState())
        self.settings.setValue("unmixed_splitter_state", self.unmixed_splitter.saveState())
        """Save window geometry and maximized state to QSettings."""
        if self.isMaximized():
            self.settings.setValue("window/maximized", True)
        else:
            self.settings.setValue("window/maximized", False)
            self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.sync()

    def restore_state(self):
        state = self.settings.value("main_splitter_state")
        raw_state = self.settings.value("raw_splitter_state")
        unmixed_state = self.settings.value("unmixed_splitter_state")
        if state is not None:
            self.splitter.restoreState(state)
        if raw_state is not None:
            self.raw_splitter.restoreState(raw_state)
        if unmixed_state is not None:
            self.unmixed_splitter.restoreState(unmixed_state)
        """Restore window geometry and maximized state from QSettings."""
        maximized = self.settings.value("window/maximized", False, type=bool)
        geometry = self.settings.value("window/geometry")

        if geometry is not None:
            self.restoreGeometry(QByteArray(geometry))
        else:
            self.resize(1000, 600)

        if maximized:
            self.showMaximized()
        else:
            self.showNormal()

    _HEATMAP_CELL_SIZES = [30, 40, 50, 60, 70, 80, 100, 120]

    def _heatmap_zoom_in(self):
        sizes = MainWindow._HEATMAP_CELL_SIZES
        idx = min(sizes.index(self._heatmap_cell_size) + 1, len(sizes) - 1) if self._heatmap_cell_size in sizes else 0
        self._heatmap_cell_size = sizes[idx]
        self._apply_heatmap_cell_size()

    def _heatmap_zoom_out(self):
        sizes = MainWindow._HEATMAP_CELL_SIZES
        idx = max(sizes.index(self._heatmap_cell_size) - 1, 0) if self._heatmap_cell_size in sizes else 0
        self._heatmap_cell_size = sizes[idx]
        self._apply_heatmap_cell_size()

    def _apply_heatmap_cell_size(self):
        for viewer in (self.similarity_viewer, self.hotspot_viewer,
                       self.unmixing_viewer, self.compensation_editor):
            viewer.view.set_cell_size(self._heatmap_cell_size)

    def _save_visibility_state(self):
        for name, widget in {"gains": self.gains_widget, "acquisition": self.acquisition_widget}.items():
            self.settings.setValue(f"{name}_visible", widget.isVisible())

    def _restore_visibility_state(self):
        for name, widget in {"gains": self.gains_widget, "acquisition": self.acquisition_widget}.items():
            visible = self.settings.value(f"{name}_visible", True, type=bool)
            widget.setVisible(visible)

    @Slot()
    def _on_cleaning_results_ready(self):
        """Refresh the peak-channel histogram immediately after Clean Controls finishes."""
        if self.profiles_viewer._hist_toggle.isChecked():
            logger.info(f'MainWindow: calling _refresh_hist_combo and _refresh_histogram')
            self.profiles_viewer._refresh_hist_combo()
            self.profiles_viewer._refresh_histogram()

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    from honeychrome.controller import Controller
    from pathlib import Path
    from honeychrome.view_components.event_bus import EventBus

    app =  QApplication(sys.argv)

    kc = Controller()
    base_directory = Path.home() / 'spectral_cytometry'
    experiment_name = base_directory / '20240620 Spectral Symposium-poor cell unmixed'
    experiment_path = experiment_name.with_suffix('.kit')
    kc.load_experiment(experiment_path) # note this loads first sample too and runs calculate all histograms and statistics

    main_window = MainWindow(bus=EventBus(), controller=kc)
    main_window.restore_state()
    # main_window.show()

    main_window.gating_tree_raw.init_data(kc.data_for_cytometry_plots)

    exit_code = app.exec()
    sys.exit(exit_code)

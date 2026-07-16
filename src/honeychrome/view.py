'''
GUI:
-This is the MVC View
-Creates GUI widgets
-Serves and updates data from experiment control
--sample list
--settings
--plots
--histograms
--gates
--spectral model
--instrument control
--oscilloscope


Gui structure
Splash screen
Main window
Menus
Sample panel
Cytometry panel with tabs
Docking area- instrument control, statistics, oscilloscope, console

Serves sample list
Serves settings
Serves plots
Serves histograms
Serves oscilloscope
Instrument control buttons+status
Signals start and stop
Updates plots
Updates gating hierarchy
Updates transforms and dimensions
Updates spectral model

Sample panel has tree view, controls, samples, new button, make control/sample
Filled sample has fcs file (raw data)

Cytometry panel is grid, each plot widget occupies one tile.

Create list of icps, all share underlying data, so updated simultaneously. Icps have axes and rois. Icps can update gating and axes
'''
import os
import warnings
from pathlib import Path

from PySide6.QtGui import QPalette, QIcon
from PySide6.QtWidgets import QMessageBox, QWidget, QFileDialog, QApplication
from PySide6.QtCore import Signal, QObject, Slot, QTimer, Qt

import honeychrome.settings as settings
from honeychrome.view_components.busy_cursor import with_busy_cursor
from honeychrome.view_components.event_bus import EventBus
from honeychrome.view_components.splash_dialog import SplashScreen
from honeychrome.view_components.main_window import MainWindow
from honeychrome.controller_components.functions import add_recent_file
from honeychrome.settings import experiments_folder, file_extension
from honeychrome.view_components.new_file_dialog import NewFileDialog
from honeychrome import __version__

import pyqtgraph as pg

base_directory = str(Path.home() / experiments_folder)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

logo_icon = resource_path(str(Path(__file__).resolve().parent / 'view_components' / 'assets' / 'cytkit_web_logo.ico'))

class View(QObject):
    def __init__(self, controller=None):
        super().__init__()
        self.bus = EventBus()
        self.main_window = None
        self.splash = SplashScreen(self)
        self.current_window = self.splash
        self.splash.show()
        self.splash.setWindowIcon(QIcon(logo_icon))
        self.controller = controller

        # debounce to avoid autosaving too frequently
        self.autosave_debounce_timer = QTimer(parent=self)
        self.autosave_debounce_timer.setSingleShot(True)
        self.autosave_debounce_timer.timeout.connect(lambda: self.controller.save_experiment(''))

        # set theme
        app = QApplication.instance()

        palette = app.palette()
        base_color = palette.color(QPalette.ColorRole.Base)
        self.is_dark = base_color.value() < 128


        if self.is_dark:  # dark theme
            pg.setConfigOptions(background='black', foreground='white')
        else:  # light theme
            pg.setConfigOptions(background='white', foreground='black')

        if self.controller is not None:
            self._connect_signals()

    def _connect_signals(self):
        # file actions
        self.bus.newExpRequested.connect(self.new_experiment)
        self.bus.newExpRequestedFromTemplate.connect(self.new_from_template)
        self.bus.newExpRequestedFromThisTemplate.connect(self.new_from_this_template)
        self.bus.openExpRequested.connect(self.open_experiment)
        self.bus.saveAsTemplateRequested.connect(self.save_as_template)
        self.bus.reloadExpRequested.connect(self.reload_main_window)
        self.bus.loadExpRequested.connect(self.load_main_window_with_experiment_and_template)
        self.bus.saveExpRequested.connect(self.controller.save_experiment)
        self.bus.autoSaveRequested.connect(lambda: self.autosave_debounce_timer.start(3000))

        # sample actions
        self.bus.loadSampleRequested.connect(self.controller.load_sample)
        self.bus.newSampleRequested.connect(self.controller.new_sample)

        # view
        # update oscilloscope

        # instrument control
        self.bus.startAcquisition.connect(self.controller.start_acquisition)
        self.bus.stopAcquisition.connect(self.controller.stop_acquisition)
        # clear / restart recording
        # flush
        # backflush
        self.bus.gainChanged.connect(self.controller.on_gain_change)

        # update instrument configuration
        # update experiment preferences
        self.bus.resetAxisReloadSample.connect(self.controller.reset_axes_reload_sample)

        self.bus.modeChangeRequested.connect(self.controller.set_mode)
        # self.bus.tab_change_requested.connect(self.controller.on_tab_change, Qt.DirectConnection) # consider this if plots are refreshed before data is available
        self.bus.newPlotRequested.connect(self.controller.create_new_plot)
        self.bus.plotChangeRequested.connect(self.controller.change_plot)
        self.bus.changedGatingHierarchy.connect(self.controller.on_gate_change)
        self.bus.axisTransformed.connect(self.controller.recalc_after_axis_transform)
        self.bus.axesReset.connect(self.controller.reset_axes_transforms)
        self.bus.updateChildGateLabelOffset.connect(self.controller.update_child_gate_label_offset)

        # gating templates (per-sample gating)
        self.bus.selectTemplateRequested.connect(self.controller.select_template)
        self.bus.createTemplateRequested.connect(self.controller.create_and_select_template)
        self.bus.renameTemplateRequested.connect(self.controller.rename_template)
        self.bus.templateApplied.connect(self.init_plot_grids_and_gating_trees)

        # change spectral model, unmix!, change fine tuning matrix
        self.bus.spectralModelUpdated.connect(self.controller.refresh_spectral_process)
        self.bus.spectralProcessRefreshed.connect(lambda : self.init_plot_grids_and_gating_trees('unmixed'))
        self.bus.requestUpdateProcessHists.connect(self.controller.reinitialise_data_for_process_plots)

        self.bus.setMainWindowTitle.connect(self.set_main_window_title)
        self.bus.popupMessage.connect(self.popup_message)
        self.bus.warningMessage.connect(self.warning_message)
        self.bus.aboutHoneychrome.connect(self.about_honeychrome)

        ### changing experiment triggers autosave
        self.bus.sampleTreeUpdated.connect(lambda: self.bus.autoSaveRequested.emit())
        self.bus.newPlotRequested.connect(lambda: self.bus.autoSaveRequested.emit())
        self.bus.plotChangeRequested.connect(lambda: self.bus.autoSaveRequested.emit())
        self.bus.updateSourceChildGates.connect(lambda: self.bus.autoSaveRequested.emit())
        self.bus.updateChildGateLabelOffset.connect(lambda: self.bus.autoSaveRequested.emit())
        self.bus.changedGatingHierarchy.connect(lambda: self.bus.autoSaveRequested.emit())
        self.bus.axisTransformed.connect(lambda: self.bus.autoSaveRequested.emit())
        self.bus.axesReset.connect(lambda: self.bus.autoSaveRequested.emit())
        self.bus.updateRois.connect(lambda: self.bus.autoSaveRequested.emit())
        self.bus.selectTemplateRequested.connect(lambda: self.bus.autoSaveRequested.emit())
        self.bus.createTemplateRequested.connect(lambda: self.bus.autoSaveRequested.emit())
        self.bus.renameTemplateRequested.connect(lambda: self.bus.autoSaveRequested.emit())
        self.bus.spectralModelUpdated.connect(lambda: self.bus.autoSaveRequested.emit())
        self.bus.spectralProcessRefreshed.connect(lambda: self.bus.autoSaveRequested.emit())

    @Slot()
    def about_honeychrome(self):
        self.popup_message(f'''
        <html>
        <body>
        <h3>Honeychrome<sup>TM</sup> is part of the Cytkit<sup>TM</sup> project</h3>
        <p>Version {__version__}</p>
        <p>This software is licensed as open source under the GNU Lesser General Public License (LGPL) V3.0</p>
        
        <p>
        For too long, our field lacked a general purpose, 
        free and open-source cytometry software package. 
        We are plugging that gap to provide a software package that is useful for everybody: 
        power cytometrists, biologists, students, innovators. 
        It is also fully open source, to provide a platform that innovators can exploit, 
        for developing both new instrumentation and new methods in data analysis/visualisation.
        <p>
                
        <p>Visit <a href="https://honeychrome.cytkit.com">https://honeychrome.cytkit.com</a> for more info.</p>
        <p>Send your feedback to <a href="mailto:hello@cytkit.com">hello@cytkit.com</a>.</p>
        </body>
        </html>
        ''')

    @Slot()
    def reload_main_window(self):
        self.load_main_window_with_experiment_and_template(self.controller.experiment.experiment_path)

    @Slot(str)
    def load_main_window_with_experiment_and_template(self, experiment_file, new=False, template_path=None):
        try:
            self.current_window.blockSignals(True)

            if new:
                self.controller.new_experiment(experiment_file, template_path=template_path)
            else:
                self.controller.load_experiment(experiment_file)

            self.main_window = MainWindow(bus=self.bus, controller=self.controller, is_dark=self.is_dark)
            self.main_window.restore_state()
            if self.controller.cleaned_events:
                QTimer.singleShot(0, self.bus.cleaningResultsReady.emit)
            self.current_window.close() #used to crash on this line complaining Internal C++ object (MainWindow) already deleted, cytometry_grid_widget.py line 245, in init_grid, 'NoneType' object is not subscriptable

            self.init_plot_grids_and_gating_trees()
            self.controller.set_mode('Raw Data')
            self.current_window = self.main_window

            self.bus.setMainWindowTitle.emit(str(self.controller.experiment_dir))
            if self.controller.experiment.samples['all_samples']:
                status = 'Experiment loaded. Select a sample to view plots.'
            else:
                status = 'Experiment ready.'
            self.bus.statusMessage.emit(status)
            # QTimer.singleShot(0, lambda: self.bus.statusMessage.emit(f'Failed to load sample.'))

        except FileNotFoundError as e:
            warnings.warn(f'{e}')
            if self.bus:
                self.bus.warningMessage.emit(f'{e}')


    @Slot(str)
    def set_main_window_title(self, title):
        if self.main_window:
            self.main_window.setWindowTitle(f'Honeychrome by Cytkit: {title}')

    @Slot(str)
    def init_plot_grids_and_gating_trees(self, scope=''):
        if self.main_window is None:
            return
        # reinitialises widgets
        # called for new/load experiment, or on spectral process updated (just for scope = 'unmixed')
        # run this for first initialisation, then for updates
        if scope == '' or scope == 'raw':
            self.main_window.cytometry_grid_raw.init_plots(self.controller.data_for_cytometry_plots_raw)
            self.main_window.gating_tree_raw.init_data(self.controller.data_for_cytometry_plots_raw)
        if scope == '' or scope == 'unmixed':
            if self.controller.experiment.process['unmixing_matrix']:
                self.main_window.cytometry_grid_unmixed.setVisible(True)
                self.main_window.gating_tree_unmixed.setVisible(True)
                self.main_window.cytometry_toolbar_unmixed.setVisible(True)
                self.main_window.unmixed_splitter.setVisible(True)
                self.main_window.tip_unmixed.setVisible(False)
            else:
                self.main_window.cytometry_grid_unmixed.setVisible(False)
                self.main_window.gating_tree_unmixed.setVisible(False)
                self.main_window.cytometry_toolbar_unmixed.setVisible(False)
                self.main_window.unmixed_splitter.setVisible(False)
                self.main_window.tip_unmixed.setVisible(True)


            self.main_window.cytometry_grid_unmixed.init_plots(self.controller.data_for_cytometry_plots_unmixed)
            self.main_window.gating_tree_unmixed.init_data(self.controller.data_for_cytometry_plots_unmixed)
            self.main_window.similarity_viewer.refresh_heatmap()
            self.main_window.hotspot_viewer.refresh_heatmap()
            self.main_window.unmixing_viewer.refresh_heatmap()
            self.main_window.compensation_editor.refresh_heatmap()
            self.main_window.nxn_viewer.initialise()

        if self.controller.experiment.process['unmixing_matrix']:
            self.main_window.tip_statistics.setVisible(False)
            self.main_window.statistical_comparison_widget.setVisible(True)
        else:
            self.main_window.tip_unmixed.setVisible(True)
            self.main_window.statistical_comparison_widget.setVisible(False)

        # Restore profile display for all controls (including library controls) whose profiles were saved in the kit file.
        if self.controller.experiment.process.get('profiles'):
            QTimer.singleShot(100, lambda: self.bus.showSelectedProfiles.emit([]))


    @Slot(str)
    def popup_message(self, message):
        QMessageBox.information(self.main_window, "Honeychrome by Cytkit", f"{message}")

    @Slot(str)
    def warning_message(self, message):
        QMessageBox.warning(self.main_window, "Warning", f"{message}")

    @Slot()
    def new_experiment(self):
        new_file_dialog = NewFileDialog(self.current_window)
        new_file_dialog.finished.connect(self.save_new_file)
        new_file_dialog.show()

    def save_new_file(self, result):
        new_file_dialog = self.sender()   # the dialog instance
        if result == QFileDialog.Accepted:
            file = Path(new_file_dialog.selectedFiles()[0]).with_suffix('.kit')
            add_recent_file(file)
            self.load_main_window_with_experiment_and_template(file, new=True)

    @Slot()
    def open_experiment(self):
        file, _ = QFileDialog.getOpenFileName(self.current_window, "Open Experiment", base_directory, f"Experiment File (*.{file_extension})")
        if file:
            add_recent_file(file)
            self.load_main_window_with_experiment_and_template(file, new=False)

    @Slot()
    def new_from_template(self):
        template, _ = QFileDialog.getOpenFileName(self.current_window, "Select Template Experiment", base_directory, f"Experiment File (*.{file_extension})")
        if template:
            file, _ = QFileDialog.getSaveFileName(self.current_window, f"New Experiment from \"{Path(template)}\"", base_directory, f"Experiment File (*.{file_extension})")
            if file:
                file = Path(file).with_suffix('.kit')
                add_recent_file(file)
                self.load_main_window_with_experiment_and_template(file, new=True, template_path=template)

    @Slot()
    def new_from_this_template(self):
        path = self.controller.experiment_dir.with_suffix('.kit')
        new_file, _ = QFileDialog.getSaveFileName(self.current_window, f"New Experiment from \"{Path(path)}\"", base_directory, f"Experiment File (*.{file_extension})")
        if new_file:
            file = Path(new_file).with_suffix('.kit')
            add_recent_file(file)
            self.load_main_window_with_experiment_and_template(file, new=True, template_path=path)

    @Slot()
    def save_as_template(self):
        path, _ = QFileDialog.getSaveFileName(self.current_window, "Save File", str(base_directory), f"Experiment File (*.{file_extension})")
        if path:
            self.bus.saveExpRequested.emit(path)


if __name__ == "__main__":
    import sys
    from controller import Controller

    app =  QApplication(sys.argv)

    controller = Controller()
    view = View(controller=controller)
    controller.bus = view.bus

    exit_code = app.exec()
    sys.exit(exit_code)


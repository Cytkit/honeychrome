from PySide6.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout, QRadioButton, QButtonGroup, QLineEdit,
                               QLabel, QPushButton, QDialogButtonBox, QCheckBox)
from PySide6.QtGui import QRegularExpressionValidator, QDesktopServices
from PySide6.QtCore import QRegularExpression, Qt, QUrl, QThread
import sys

from honeychrome.controller_components.functions import get_all_subfolders_recursive
from honeychrome.controller_components.import_fcs_controller import ImportFCSController
from honeychrome.controller_components.job_gate import acquire_job, guard_while_busy
from honeychrome.view_components.configuration_dialogs import ExperimentSettings
from honeychrome.view_components.icon_loader import icon


class ImportFCSFilesWidget(QDialog):
    def __init__(self, parent, bus=None, controller=None, failed_to_load_sample_warning=False):
        super().__init__(parent=parent, modal=True)
        self.bus = bus
        self.controller = controller
        self.setWindowTitle("Import FCS Files")

        self.thread = None
        self.import_fcs_controller = None
        self._job_token = None

        layout = QVBoxLayout(self)

        if failed_to_load_sample_warning:
            self.setGeometry(200, 200, 450, 200)
            label = QLabel('''
            <p><b>Warning: sample channels (names and ranges) do not match the experiment configuration.</b></p>
            <p>Does this sample belong with the others? If so, update the experiment configuration below. If not, delete this sample.</p>
            ''')
            label.setTextFormat(Qt.RichText)
            label.setWordWrap(True)

            layout.addWidget(label)
            layout.addStretch()

        else:
            self.setGeometry(200, 200, 900, 700)
            part_oneA_label = QLabel('''
            <h2>To import FCS files, you have two options.</h2>
            <h4>Option A: Copy/Move FCS Files</h4>
            <p>Copy or move FCS files into the experiment's "Raw" and "Single stain controls" folders using your file browser.</p>
            ''')
            part_oneA_label.setTextFormat(Qt.RichText)
            part_oneA_label.setWordWrap(True)

            part_oneA_button = QPushButton(icon('folder-search'), "Open Raw Subfolder")
            part_oneA_button.clicked.connect(self.open_samples_folder)
            part_oneA_button_layout = QHBoxLayout()

            part_oneB_label = QLabel('''
            <h4>Option B: Link to Existing Data Folders</h4>
            <p>Set the "Raw" and "Single stain controls" folders in Experiment Settings to find your FCS files. (This creates a symbolic link if the target folder is outside the experiments folder.)</p>
            ''')
            part_oneB_label.setTextFormat(Qt.RichText)
            part_oneB_label.setWordWrap(True)

            part_oneB_button = QPushButton(icon('settings'), "Open Experiment Settings")
            part_oneB_button.clicked.connect(self.open_experiment_settings)
            part_oneB_button_layout = QHBoxLayout()


            part_two_label = QLabel('''
            <p>Tip: you can organise your FCS files into a set of subfolders, which Honeychrome will treat as groups or categories for statistical comparisons.</p>
            <h4>Final Step: Automatically Update Experiment Configuration</h4>
            <p>The experiment configuration must match your FCS files 
            (set channel names and ranges, and check for consistency between the FCS files). 
            Note that all plots, transforms, gating and spectral process in the experiment will be reset.</p>
            ''')
            part_two_label.setTextFormat(Qt.RichText)
            part_two_label.setWordWrap(True)

            layout.addWidget(part_oneA_label)
            layout.addStretch()
            layout.addLayout(part_oneA_button_layout)
            part_oneA_button_layout.addStretch()
            part_oneA_button_layout.addWidget(part_oneA_button)
            part_oneA_button_layout.addStretch()

            layout.addStretch()
            layout.addWidget(part_oneB_label)
            layout.addStretch()
            layout.addLayout(part_oneB_button_layout)
            part_oneB_button_layout.addStretch()
            part_oneB_button_layout.addWidget(part_oneB_button)
            part_oneB_button_layout.addStretch()

            layout.addStretch()
            layout.addWidget(part_two_label)
            layout.addStretch()

        # --- Buttons ---
        button_box = QDialogButtonBox()
        self.btn_cancel = button_box.addButton("Cancel", QDialogButtonBox.RejectRole)
        self.btn_submit = button_box.addButton("Update Experiment Configuration", QDialogButtonBox.AcceptRole)

        layout.addWidget(button_box)

        # Connections
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_submit.clicked.connect(self.submit)
        guard_while_busy(self.controller, self.btn_submit)

    def submit(self):
        if self.thread is not None:
            return  # one reconfiguration per dialog
        self._job_token = acquire_job(self.controller, 'Import FCS files')
        if self._job_token is None:
            return
        self.btn_submit.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.thread = QThread(QApplication.instance())  # survives the main window rebuild this import triggers
        self.import_fcs_controller = ImportFCSController(self.controller.experiment, bus=self.bus)
        self.import_fcs_controller.moveToThread(self.thread)
        self.thread.started.connect(self.import_fcs_controller._reconfigure_experiment_from_fcs_files_impl)
        self.import_fcs_controller.finished.connect(self.thread.quit)
        self.thread.finished.connect(QApplication.restoreOverrideCursor)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._on_thread_finished)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.thread.start()

    def _on_thread_finished(self):
        self._job_token.release()
        bus, result = self.bus, self.import_fcs_controller.reload_message
        self.accept()
        if result is not None and bus is not None:
            text, is_warning = result
            bus.reloadExpRequested.emit()
            (bus.warningMessage if is_warning else bus.popupMessage).emit(text)

    def open_samples_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.controller.experiment_dir / self.controller.experiment.settings['raw']['raw_samples_subdirectory']))

    def open_experiment_settings(self):
        dialog = ExperimentSettings(self.controller.experiment, self.bus, self)
        dialog.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)


    from honeychrome.controller import Controller
    from pathlib import Path
    from event_bus import EventBus

    bus = EventBus()
    kc = Controller()
    kc.bus = bus
    base_directory = Path.home() / 'spectral_cytometry'
    experiment_name = base_directory / '20240620 Spectral Symposium-poor cell unmixed'
    experiment_path = experiment_name.with_suffix('.kit')
    kc.load_experiment(experiment_path) # note this loads first sample too and runs calculate all histograms and statistics

    dialog = ImportFCSFilesWidget(None, bus, kc)
    dialog.exec()
    sys.exit(0)

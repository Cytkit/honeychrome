import importlib
import os
import sys
from pathlib import Path
import subprocess
import re

from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QWidget, QVBoxLayout, QDialog, QScrollArea, QFormLayout, QLineEdit, QCheckBox, QSpinBox, QDoubleSpinBox, QDialogButtonBox, QComboBox, QLabel, QButtonGroup, QFileDialog, QFrame, QMessageBox)
from PySide6.QtCore import Qt, QSettings

from honeychrome.settings import (colourmap_choice, graphics_export_formats, hist2dtype, colormap_name, graphics_export_format, cytometry_plot_width_target,
                                  tile_size_nxn_grid, subsample, max_display_events, hist_bins, density_cutoff, trigger_channel, adc_channels, width_channels, height_channels,
                                  use_dummy_instrument, sample_pump_steps_per_microlitre, magnitude_ceilings, magnitude_ceiling, raw_settings, unmixed_settings, experiments_folder,
                                  magnitude_ceilings_int, spectral_positive_gate_percent, spectral_negative_gate_percent, report_include_raw, report_include_unmixed, report_include_process, send_debug_data,
                                  heatmap_colourmap_name, heatmap_colourmap_choice, spectral_cleaning_n_candidates, spectral_cleaning_n_spectral)
import honeychrome.settings as settings


import numpy as np
import colorcet as cc

from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

PLUGIN_DIR = Path.home() / experiments_folder / "plugins"
plugins_path = Path(PLUGIN_DIR)

# Plugin enable checkboxes only if run in full python environment
meipass = getattr(sys, '_MEIPASS', None)

# Bundled/approved plugins — always available, dev and frozen. Mirrors the
# resolution in plugin_loaders.py.
if meipass:
    BUNDLED_PLUGIN_DIR = Path(meipass) / "honeychrome" / "bundled_plugins"
else:
    BUNDLED_PLUGIN_DIR = Path(__file__).resolve().parent.parent / "bundled_plugins"
bundled_plugins_path = Path(BUNDLED_PLUGIN_DIR)

def path_to_folder_name_readable(original_path):
    # 1. Convert to string and resolve
    path_str = str(Path(original_path).resolve())

    # 2. Replace common illegal characters with underscores
    # This handles :, /, \, *, ?, ", <, >, |
    sanitized = re.sub(r'[\\/:*?"<>|]', '_', path_str)

    # 3. Clean up leading/trailing dots or spaces (Windows hates these)
    return sanitized.strip('. ')

def is_subfolder(child, parent):
    child = Path(child).resolve()
    parent = Path(parent).resolve()

    # Returns True if child is a subfolder (or the folder itself)
    return child.is_relative_to(parent)

def create_folder_link(experiment_path, link_name, target_path):
    """
    Creates a symbolic link to a directory.
    On Windows, if symlink creation fails (usually due to permissions),
    it attempts to create a Directory Junction instead.
    """
    target = Path(target_path).resolve()
    link = Path(experiment_path) / link_name

    # 1. Validation
    if not target.is_dir():
        raise ValueError(f"Target is not a directory: {target}")

    if link.exists():
        print(f"Link path already exists: {link}.")
        link.replace(link.with_suffix('.old'))

    # 2. Try Standard Symlink (Works on Linux, macOS, and Windows with Dev Mode/Admin)
    try:
        link.symlink_to(target, target_is_directory=True)
        print(f"Symbolic link created: {link} -> {target}")
        return
    except OSError as e:
        # If not on Windows, or error isn't permission related, re-raise
        if sys.platform != "win32" or e.winerror not in (1314, 5):
            raise e

        print("Standard symlink failed (Privilege Error). Attempting Windows Junction...")

    # 3. Windows Fallback: Create a Junction (/J)
    # Junctions don't require Admin privileges.
    # — but only if target is not a UNC path
    if str(target).startswith('\\\\'):
        # Junctions don't support UNC targets; warn the user
        QMessageBox.warning(
            None, 'Network path not supported',
            f"Honeychrome cannot create a junction to a UNC network path: {str(target)}. \n\nPlease map the network drive and use the mapped drive letter instead.")
        return
    try:
        # mklink is a shell built-in, so shell=True is required
        cmd = f'mklink /J "{link}" "{target}"'
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        print(f"Junction created: {link} -> {target}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to create junction: {e.stderr.decode().strip()}")
        raise

def colormap_to_qimage(cmap_name, width=256, height=20):
    """
    Convert a colorcet colormap to a horizontal swatch QImage.
    """
    cmap = cc.cm[cmap_name]
    # colorcet returns RGB tuples in 0–1 range → convert to uint8
    data = np.array([cmap(i) for i in np.linspace(0, 1, width)])
    data = (data[:, :3] * 255).astype(np.uint8)  # drop alpha if present

    # Repeat rows vertically to form a swatch
    img_array = np.repeat(data[np.newaxis, :, :], height, axis=0)

    # Convert to bytes for QImage
    h, w, _ = img_array.shape
    bytes_per_line = w * 3
    qimg = QImage(
        img_array.data,
        w,
        h,
        bytes_per_line,
        QImage.Format_RGB888
    )
    return qimg.copy()  # copy to detach from numpy buffer


class AppConfigDialog(QDialog):
    def __init__(self, parent=None, bus=None):
        super().__init__(parent)
        self.setWindowTitle("Application Configuration")
        self.resize(1000, 700)
        self.settings = QSettings("honeychrome", "app_configuration")
        self.bus = bus

        main_layout = QVBoxLayout(self)

        # --- Scroll Area ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        form = QFormLayout(container)
        form.setSpacing(10)

        # --- Settings Widgets ---

        self.hist2dtype_combo = QComboBox()
        self.hist2dtype_combo.addItems(settings.hist2dtypes)
        form.addRow("2D plot type:", self.hist2dtype_combo)

        self.colourmap_combo = QComboBox()
        self.colourmap_combo.addItems(colourmap_choice)
        form.addRow("Colourmap (visit colorcet.com for definitions):", self.colourmap_combo)

        self.swatch = QLabel()
        form.addWidget(self.swatch)
        self.colourmap_combo.currentTextChanged.connect(self.set_swatch)

        self.heatmap_colourmap_combo = QComboBox()
        self.heatmap_colourmap_combo.addItems(heatmap_colourmap_choice)
        form.addRow("Heatmap colour palette (colorcet.com):", self.heatmap_colourmap_combo)
        self.heatmap_swatch = QLabel()
        form.addWidget(self.heatmap_swatch)
        self.heatmap_colourmap_combo.currentTextChanged.connect(self.set_heatmap_swatch)

        self.graphics_export_format_combo = QComboBox()
        self.graphics_export_format_combo.addItems(graphics_export_formats)
        form.addRow("Graphics Export Format:", self.graphics_export_format_combo)

        self.cytometry_plot_size_spin = QSpinBox()
        self.cytometry_plot_size_spin.setRange(100, 600)
        self.cytometry_plot_size_spin.setSingleStep(50)
        form.addRow("Cytometry Plot Size (pixels):", self.cytometry_plot_size_spin)

        self.nxn_tile_size_spin = QSpinBox()
        self.nxn_tile_size_spin.setRange(20, 300)
        self.nxn_tile_size_spin.setSingleStep(10)
        form.addRow("NxN plots tile size (pixels):", self.nxn_tile_size_spin)

        self.subsample_number_spin = QSpinBox()
        self.subsample_number_spin.setRange(1000, 1_000_000)
        self.subsample_number_spin.setSingleStep(1000)
        form.addRow("Number of events to export if 'Subsample' enabled (pixels):", self.subsample_number_spin)

        self.max_display_events_spin = QSpinBox()
        self.max_display_events_spin.setRange(10_000, 2_000_000)
        self.max_display_events_spin.setSingleStep(50_000)
        self.max_display_events_spin.setToolTip('Maximum events shown in cytometry display plots. Does not affect spectral process or unmixing.')
        form.addRow("Max display events:", self.max_display_events_spin)

        self.histogram_resolution_spin = QSpinBox()
        self.histogram_resolution_spin.setRange(50, 400)
        self.histogram_resolution_spin.setSingleStep(50)
        form.addRow("Histogram resolution (bin count):", self.histogram_resolution_spin)

        self.density_cutoff_spin = QSpinBox()
        self.density_cutoff_spin.setRange(0, 1000)
        self.density_cutoff_spin.setSingleStep(10)
        form.addRow("Density cutoff (bin value):", self.density_cutoff_spin)
        self.density_cutoff_spin.setToolTip('Bin value threshold for first level of colourmap (below this level is transparent). Set to 0 for 1/255 of maximum value.')

        self.spectral_positive_gate_percent_spin = QSpinBox()
        self.spectral_positive_gate_percent_spin.setRange(0, 50)
        self.spectral_positive_gate_percent_spin.setSingleStep(1)
        form.addRow("Spectral control positive gate (brightest % of events):", self.spectral_positive_gate_percent_spin)

        self.spectral_negative_gate_percent_spin = QSpinBox()
        self.spectral_negative_gate_percent_spin.setRange(0, 50)
        self.spectral_negative_gate_percent_spin.setSingleStep(1)
        form.addRow("Spectral control negative gate (brightest % of events):", self.spectral_negative_gate_percent_spin)

        self.spectral_cleaning_n_candidates_spin = QSpinBox()
        self.spectral_cleaning_n_candidates_spin.setRange(50, 5000)
        self.spectral_cleaning_n_candidates_spin.setSingleStep(50)
        form.addRow("Control cleaning: candidate pool size (top-N by Major Channel):", self.spectral_cleaning_n_candidates_spin)
        self.spectral_cleaning_n_candidates_spin.setToolTip('Number of brightest events (by Major Channel) considered as cosine-similarity candidates during Clean Controls.')

        self.spectral_cleaning_n_spectral_spin = QSpinBox()
        self.spectral_cleaning_n_spectral_spin.setRange(10, 2000)
        self.spectral_cleaning_n_spectral_spin.setSingleStep(10)
        form.addRow("Control cleaning: events retained (least AF-similar):", self.spectral_cleaning_n_spectral_spin)
        self.spectral_cleaning_n_spectral_spin.setToolTip('Number of candidate events retained for the cleaned spectral profile, ranked by lowest cosine similarity to the unstained AF reference.')

        self.report_include_raw_cb = QCheckBox("Raw Data")
        self.report_include_unmixed_cb = QCheckBox("Unmixed Data")
        self.report_include_process_cb = QCheckBox("Spectral Process")
        form.addRow("Include raw data in sample report", self.report_include_raw_cb)
        form.addRow("Include unmixed data in sample report", self.report_include_unmixed_cb)
        form.addRow("Include spectral process in sample report", self.report_include_process_cb)

        # Bundled/approved plugins — always shown, dev and frozen
        self.enable_bundled_plugin = {}
        for file_path in bundled_plugins_path.glob("*_tab.py"):
            self.enable_bundled_plugin[file_path] = QCheckBox(f"Enable {file_path.stem}")
            form.addRow("Bundled plugin", self.enable_bundled_plugin[file_path])

        # Arbitrary user-added plugins — full Python (dev) environment only
        if not meipass:
            self.enable_plugin = {}
            for file_path in plugins_path.glob("*_tab.py"):
                self.enable_plugin[file_path] = QCheckBox(f"Enable {file_path.stem}")
                form.addRow(f"Tabbed plugin", self.enable_plugin[file_path])
        else:
            form.addRow(QLabel("Note: custom user plugins require running Honeychrome in a full Python environment. Bundled plugins above are still available."))


        frame = QFrame(self)
        frame_layout = QVBoxLayout()
        frame.setLayout(frame_layout)
        frame_layout.setContentsMargins(10, 10, 10, 10)
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setLineWidth(1)

        frame_layout.addWidget(QLabel('Honeychrome does not collect personal identifiable information'))
        self.send_debug_data = QCheckBox("Send debug data to improve Honeychrome")
        self.send_debug_data_label = QLabel('<a href="https://honeychrome.cytkit.com/docs/send_debug_data">What data is collected?</a>')

        self.send_debug_data_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.send_debug_data_label.setOpenExternalLinks(True)

        frame_layout.addWidget(self.send_debug_data)
        frame_layout.addWidget(self.send_debug_data_label)
        form.addRow(frame)

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok |
            QDialogButtonBox.Cancel |
            QDialogButtonBox.RestoreDefaults
        )
        buttons.accepted.connect(self.handle_accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.reset_to_defaults)
        main_layout.addWidget(buttons, alignment=Qt.AlignRight)

        # Load settings on start
        self.load_settings()

    # ----------------------------
    # Settings Load / Save Methods
    # ----------------------------

    def set_swatch(self, cmap_name):
        qimage = colormap_to_qimage(cmap_name)
        qpixmap = QPixmap.fromImage(qimage)
        self.swatch.setPixmap(qpixmap)

    def set_heatmap_swatch(self, cmap_name):
        qimage = colormap_to_qimage(cmap_name)
        qpixmap = QPixmap.fromImage(qimage)
        self.heatmap_swatch.setPixmap(qpixmap)
    
    def load_settings(self):
        hist2dtype_retrieved = str(self.settings.value("hist2dtype", hist2dtype))
        index = self.hist2dtype_combo.findText(hist2dtype_retrieved)
        if index >= 0:
            self.hist2dtype_combo.setCurrentIndex(index)

        colourmap_retrieved = str(self.settings.value("colourmap", colormap_name))
        index = self.colourmap_combo.findText(colourmap_retrieved)
        if index >= 0:
            self.colourmap_combo.setCurrentIndex(index)

        heatmap_colourmap_retrieved = str(self.settings.value("heatmap_colourmap", heatmap_colourmap_name))
        index = self.heatmap_colourmap_combo.findText(heatmap_colourmap_retrieved)
        if index >= 0:
            self.heatmap_colourmap_combo.setCurrentIndex(index)
        self.set_heatmap_swatch(self.heatmap_colourmap_combo.currentText())

        graphics_export_format_retrieved = str(self.settings.value("graphics_export_format", graphics_export_format))
        index = self.graphics_export_format_combo.findText(graphics_export_format_retrieved)
        if index >= 0:
            self.graphics_export_format_combo.setCurrentIndex(index)

        self.cytometry_plot_size_spin.setValue(self.settings.value("cytometry_plot_size", cytometry_plot_width_target, type=int))
        self.nxn_tile_size_spin.setValue(self.settings.value("nxn_tile_size", tile_size_nxn_grid, type=int))
        self.subsample_number_spin.setValue(self.settings.value("subsample_number", subsample, type=int))
        self.max_display_events_spin.setValue(self.settings.value("max_display_events", max_display_events, type=int))
        self.histogram_resolution_spin.setValue(self.settings.value("histogram_resolution", hist_bins, type=int))
        self.density_cutoff_spin.setValue(self.settings.value("density_cutoff", density_cutoff, type=int))
        self.spectral_positive_gate_percent_spin.setValue(self.settings.value("spectral_positive_gate_percent", spectral_positive_gate_percent, type=int))
        self.spectral_negative_gate_percent_spin.setValue(self.settings.value("spectral_negative_gate_percent", spectral_negative_gate_percent, type=int))
        self.spectral_cleaning_n_candidates_spin.setValue(self.settings.value("spectral_cleaning_n_candidates", spectral_cleaning_n_candidates, type=int))
        self.spectral_cleaning_n_spectral_spin.setValue(self.settings.value("spectral_cleaning_n_spectral", spectral_cleaning_n_spectral, type=int))

        self.report_include_raw_cb.setChecked(self.settings.value("report_include_raw", report_include_raw, type=bool))
        self.report_include_unmixed_cb.setChecked(self.settings.value("report_include_unmixed", report_include_unmixed, type=bool))
        self.report_include_process_cb.setChecked(self.settings.value("report_include_process", report_include_process, type=bool))

        self.send_debug_data.setChecked(self.settings.value("send_debug_data", send_debug_data, type=bool))

        for file_path in bundled_plugins_path.glob("*_tab.py"):
            self.enable_bundled_plugin[file_path].setChecked(self.settings.value(f"EnableBundledPlugin_{file_path.stem}", False, type=bool))

        if not meipass:
            for file_path in plugins_path.glob("*_tab.py"):
                self.enable_plugin[file_path].setChecked(self.settings.value(f"EnablePlugin_{file_path}", False, type=bool))

    def save_settings(self):
        self.settings.setValue("hist2dtype", self.hist2dtype_combo.currentText())
        self.settings.setValue("colourmap", self.colourmap_combo.currentText())
        self.settings.setValue("heatmap_colourmap", self.heatmap_colourmap_combo.currentText())
        self.settings.setValue("graphics_export_format", self.graphics_export_format_combo.currentText())
        self.settings.setValue("cytometry_plot_size", self.cytometry_plot_size_spin.value())
        self.settings.setValue("nxn_tile_size", self.nxn_tile_size_spin.value())
        self.settings.setValue("subsample_number", self.subsample_number_spin.value())
        self.settings.setValue("max_display_events", self.max_display_events_spin.value())
        self.settings.setValue("histogram_resolution", self.histogram_resolution_spin.value())
        self.settings.setValue("density_cutoff", self.density_cutoff_spin.value())
        self.settings.setValue("spectral_positive_gate_percent", self.spectral_positive_gate_percent_spin.value())
        self.settings.setValue("spectral_negative_gate_percent", self.spectral_negative_gate_percent_spin.value())
        self.settings.setValue("spectral_cleaning_n_candidates", self.spectral_cleaning_n_candidates_spin.value())
        self.settings.setValue("spectral_cleaning_n_spectral", self.spectral_cleaning_n_spectral_spin.value())
        self.settings.setValue("report_include_raw", self.report_include_raw_cb.isChecked())
        self.settings.setValue("report_include_unmixed", self.report_include_unmixed_cb.isChecked())
        self.settings.setValue("report_include_process", self.report_include_process_cb.isChecked())
        self.settings.setValue("send_debug_data", self.send_debug_data.isChecked())

        for file_path in bundled_plugins_path.glob("*_tab.py"):
            self.settings.setValue(f"EnableBundledPlugin_{file_path.stem}", self.enable_bundled_plugin[file_path].isChecked())

        if not meipass:
            for file_path in plugins_path.glob("*_tab.py"):
                self.settings.setValue(f"EnablePlugin_{file_path}", self.enable_plugin[file_path].isChecked())

    def handle_accept(self):
        self.save_settings()
        self.accept()
        importlib.reload(settings)
        self.bus.reloadExpRequested.emit()

    def reset_to_defaults(self):
        index = self.hist2dtype_combo.findText(hist2dtype)
        if index >= 0:
            self.hist2dtype_combo.setCurrentIndex(index)
        index = self.colourmap_combo.findText(colormap_name)
        if index >= 0:
            self.colourmap_combo.setCurrentIndex(index)
        index = self.heatmap_colourmap_combo.findText(heatmap_colourmap_name)
        if index >= 0:
            self.heatmap_colourmap_combo.setCurrentIndex(index)
        index = self.graphics_export_format_combo.findText(graphics_export_format)
        if index >= 0:
            self.graphics_export_format_combo.setCurrentIndex(index)
        self.cytometry_plot_size_spin.setValue(cytometry_plot_width_target)
        self.nxn_tile_size_spin.setValue(tile_size_nxn_grid)
        self.subsample_number_spin.setValue(subsample)
        self.max_display_events_spin.setValue(max_display_events)
        self.histogram_resolution_spin.setValue(hist_bins)
        self.density_cutoff_spin.setValue(density_cutoff)
        self.spectral_positive_gate_percent_spin.setValue(spectral_positive_gate_percent)
        self.spectral_negative_gate_percent_spin.setValue(spectral_negative_gate_percent)
        self.spectral_cleaning_n_candidates_spin.setValue(spectral_cleaning_n_candidates)
        self.spectral_cleaning_n_spectral_spin.setValue(spectral_cleaning_n_spectral)
        self.report_include_raw_cb.setChecked(report_include_raw)
        self.report_include_unmixed_cb.setChecked(report_include_unmixed)
        self.report_include_process_cb.setChecked(report_include_process)
        self.send_debug_data.setChecked(send_debug_data)

        for file_path in bundled_plugins_path.glob("*_tab.py"):
            self.enable_bundled_plugin[file_path].setChecked(False)

        if not meipass:
            for file_path in plugins_path.glob("*_tab.py"):
                self.enable_plugin[file_path].setChecked(False)


class InstrumentConfigDialog(QDialog):
    def __init__(self, parent=None, bus=None):
        super().__init__(parent)
        self.bus = bus

        self.setWindowTitle("Instrument Configuration: restart application for changes to take effect")
        self.resize(700, 700)
        self.settings = QSettings("honeychrome", "instrument_configuration")

        main_layout = QVBoxLayout(self)

        # --- Scroll Area ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        form = QFormLayout(container)
        form.setSpacing(10)

        # --- Settings Widgets ---
        self.trigger_channel_combo = QComboBox()
        self.trigger_channel_combo.addItems(adc_channels)
        form.addRow("Trigger Channel:", self.trigger_channel_combo)

        self.width_channel_combo = QComboBox()
        self.width_channel_combo.addItems(adc_channels)
        form.addRow("Width Channel:", self.width_channel_combo)

        self.height_channel_combo = QComboBox()
        self.height_channel_combo.addItems(adc_channels)
        form.addRow("Height Channel:", self.height_channel_combo)

        # self.use_dummy_instrument = QCheckBox("Use Dummy Instrument:")
        # form.addRow(self.use_dummy_instrument)

        self.steps_per_microlitre_spin = QDoubleSpinBox(minimum=0, maximum=10000, decimals=1)
        form.addRow("Volume calibration of the sample pump (steps per microlitre):", self.steps_per_microlitre_spin)

        form.addRow(QLabel("Additional settings for Cytkit can be found in the plugin tab Cytkit Hardware"))

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok |
            QDialogButtonBox.Cancel |
            QDialogButtonBox.RestoreDefaults
        )
        buttons.accepted.connect(self.handle_accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.reset_to_defaults)
        main_layout.addWidget(buttons, alignment=Qt.AlignRight)

        # Load settings on start
        self.load_settings()

    # ----------------------------
    # Settings Load / Save Methods
    # ----------------------------

    def load_settings(self):
        trigger_channel_retrieved = str(self.settings.value("trigger_channel", trigger_channel)) # there can only be one trigger channel
        index = self.trigger_channel_combo.findText(trigger_channel_retrieved)
        if index >= 0:
            self.trigger_channel_combo.setCurrentIndex(index)

        width_channel_retrieved = str(self.settings.value("width_channel", width_channels[0])) # there can be more than one width channel, but currently only allowing one
        index = self.width_channel_combo.findText(width_channel_retrieved)
        if index >= 0:
            self.width_channel_combo.setCurrentIndex(index)

        height_channel_retrieved = str(self.settings.value("height_channel", height_channels)) # there can be more than one height channel, but currently only allowing one
        index = self.height_channel_combo.findText(height_channel_retrieved)
        if index >= 0:
            self.height_channel_combo.setCurrentIndex(index)

        sample_pump_steps_per_microlitre_retrieved = self.settings.value("sample_pump_steps_per_microlitre", sample_pump_steps_per_microlitre)
        self.steps_per_microlitre_spin.setValue(float(sample_pump_steps_per_microlitre_retrieved))

        # self.use_dummy_instrument.setChecked(self.settings.value("use_dummy_instrument", use_dummy_instrument, type=bool))

    def save_settings(self):
        self.settings.setValue("trigger_channel", self.trigger_channel_combo.currentText())
        self.settings.setValue("width_channel", self.width_channel_combo.currentText())
        self.settings.setValue("height_channel", self.height_channel_combo.currentText())
        self.settings.setValue("sample_pump_steps_per_microlitre", self.steps_per_microlitre_spin.value())
        self.settings.setValue("temperature_set_point", self.steps_per_microlitre_spin.value())
        self.settings.setValue("", self.steps_per_microlitre_spin.value())

        if self.bus:
            self.bus.setSamplePumpFlowRate.emit(0)

        # self.settings.setValue("use_dummy_instrument", self.use_dummy_instrument.isChecked())

    def handle_accept(self):
        self.save_settings()
        self.accept()

    def reset_to_defaults(self):
        index = self.trigger_channel_combo.findText(trigger_channel)
        if index >= 0:
            self.trigger_channel_combo.setCurrentIndex(index)
        index = self.width_channel_combo.findText(width_channels[0])
        if index >= 0:
            self.width_channel_combo.setCurrentIndex(index)
        index = self.height_channel_combo.findText(height_channels[0])
        if index >= 0:
            self.height_channel_combo.setCurrentIndex(index)
        self.steps_per_microlitre_spin.setValue(sample_pump_steps_per_microlitre)
        # self.use_dummy_instrument.setChecked(use_dummy_instrument)


class FolderSelectorLineEdit(QLineEdit):
    def __init__(self, parent=None, requirement=None, experiment_path=None, current_path=None):
        super().__init__(parent)
        self.setPlaceholderText("Click to select a folder...")
        self.setReadOnly(True)  # Prevents manual typing
        self.current_path = current_path
        self.experiment_path = experiment_path
        self.requirement = requirement

    def mousePressEvent(self, event):
        # Trigger the dialog on click
        folder_path = QFileDialog.getExistingDirectory(self, "Select Directory", self.current_path)

        if folder_path:
            # if folder path is within experiment path, convert to relative path based on Current Working Directory
            # else create symlink and insert that
            if is_subfolder(folder_path, self.experiment_path):
                relative_path = os.path.relpath(folder_path, self.experiment_path)
            else:
                if self.requirement:
                    name = self.requirement
                else:
                    name = path_to_folder_name_readable(folder_path)
                relative_path = 'Link_to_'+name
                create_folder_link(self.experiment_path, relative_path, folder_path)

            self.setText(relative_path)

        super().mousePressEvent(event)

class ExperimentSettings(QDialog):
    def __init__(self, experiment, bus, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Experiment Settings")
        self.resize(900, 700)
        self.settings = experiment.settings
        self.bus = bus

        main_layout = QVBoxLayout(self)

        # --- Scroll Area ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        form = QFormLayout(container)
        form.setSpacing(10)

        # --- Settings Widgets ---
        experiment_path = Path(experiment.experiment_path)
        experiment_path = experiment_path.parent / experiment_path.stem

        self.raw_samples_subdirectory_lineedit = FolderSelectorLineEdit(requirement='raw', experiment_path=str(experiment_path), current_path=str(experiment_path / self.settings['raw']['raw_samples_subdirectory']))
        form.addRow("Raw Samples Folder", self.raw_samples_subdirectory_lineedit)

        self.single_stain_controls_subdirectory_lineedit = FolderSelectorLineEdit(requirement='single_stain_controls', experiment_path=str(experiment_path), current_path=str(experiment_path / self.settings['raw']['single_stain_controls_subdirectory']))
        form.addRow("Single Stain Controls Folder:", self.single_stain_controls_subdirectory_lineedit)

        self.unmixed_samples_subdirectory_lineedit = FolderSelectorLineEdit(requirement='unmixed', experiment_path=str(experiment_path), current_path=str(experiment_path / self.settings['unmixed']['unmixed_samples_subdirectory']))
        form.addRow("Unmixed Samples Folder:", self.unmixed_samples_subdirectory_lineedit)

        self.magnitude_ceiling_combo = QComboBox()
        self.magnitude_ceiling_combo.addItems(magnitude_ceilings)
        form.addRow("Magnitude Ceiling of Raw Channels (traditionally a large power of 2):", self.magnitude_ceiling_combo)


        scroll.setWidget(container)
        main_layout.addWidget(scroll)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok |
            QDialogButtonBox.Cancel |
            QDialogButtonBox.RestoreDefaults
        )
        buttons.accepted.connect(self.handle_accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.reset_to_defaults)
        main_layout.addWidget(buttons, alignment=Qt.AlignRight)

        # Load settings on start
        self.load_settings()

    # ----------------------------
    # Settings Load / Save Methods
    # ----------------------------

    def load_settings(self):
        self.raw_samples_subdirectory_lineedit.setText(self.settings['raw']['raw_samples_subdirectory'])
        self.single_stain_controls_subdirectory_lineedit.setText(self.settings['raw']['single_stain_controls_subdirectory'])
        self.unmixed_samples_subdirectory_lineedit.setText(self.settings['unmixed']['unmixed_samples_subdirectory'])

        if self.settings['raw']['magnitude_ceiling'] in magnitude_ceilings_int:
            index = magnitude_ceilings_int.index(self.settings['raw']['magnitude_ceiling'])
            if index >= 0:
                self.magnitude_ceiling_combo.setCurrentIndex(index)

    def save_settings(self):
        self.settings['raw']['raw_samples_subdirectory'] = self.raw_samples_subdirectory_lineedit.text()
        self.settings['raw']['single_stain_controls_subdirectory'] = self.single_stain_controls_subdirectory_lineedit.text()
        self.settings['unmixed']['unmixed_samples_subdirectory'] = self.unmixed_samples_subdirectory_lineedit.text()

        magnitude_ceiling_choice = magnitude_ceilings_int[self.magnitude_ceiling_combo.currentIndex()]
        self.settings['raw']['magnitude_ceiling'] = magnitude_ceiling_choice
        self.settings['unmixed']['magnitude_ceiling'] = magnitude_ceiling_choice

    def handle_accept(self):
        self.save_settings()
        if self.bus:
            self.bus.sampleTreeUpdated.emit()
            self.bus.resetAxisReloadSample.emit()

            # don't do the following any more... too slow
            # self.bus.axesReset.emit([self.settings['raw']['event_channels_pnn'][index] for index in self.settings['raw']['fluorescence_channel_ids']])
            # self.bus.axesReset.emit([self.settings['raw']['event_channels_pnn'][index] for index in self.settings['raw']['scatter_channel_ids']])
        self.accept()

    def reset_to_defaults(self):
        self.raw_samples_subdirectory_lineedit.setText(raw_settings['raw_samples_subdirectory'])
        self.single_stain_controls_subdirectory_lineedit.setText(raw_settings['single_stain_controls_subdirectory'])
        self.unmixed_samples_subdirectory_lineedit.setText(unmixed_settings['unmixed_samples_subdirectory'])

        index = magnitude_ceilings_int.index(raw_settings['magnitude_ceiling'])
        if index >= 0:
            self.magnitude_ceiling_combo.setCurrentIndex(index)

if __name__ == '__main__':
    from honeychrome.experiment_model import ExperimentModel
    from honeychrome.view_components.event_bus import EventBus

    base_directory = Path.home() / experiments_folder

    experiment = ExperimentModel()
    experiment_name = base_directory / 'test experiment'
    experiment_path = experiment_name.with_suffix('.kit')
    experiment.create(experiment_path)

    bus = EventBus()

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Settings Dialogs Test")
            self.resize(700, 700)

            btn0 = QPushButton("Application Configuration")
            btn0.clicked.connect(self.app_config)

            btn1 = QPushButton("Instrument Configuration")
            btn1.clicked.connect(self.instrument_config)

            btn2 = QPushButton("Experiment Settings")
            btn2.clicked.connect(self.expt_settings)

            central = QWidget()
            layout = QVBoxLayout(central)
            layout.addWidget(btn0)
            layout.addWidget(btn1)
            layout.addWidget(btn2)
            self.setCentralWidget(central)

        def app_config(self):
            dialog = AppConfigDialog(self)
            dialog.exec()

        def instrument_config(self):
            dialog = InstrumentConfigDialog(self)
            dialog.exec()

        def expt_settings(self):
            dialog = ExperimentSettings(experiment, bus, self)
            dialog.exec()

    app = QApplication(sys.argv)

    # Optional: define organization/application for QSettings globally
    QSettings.setDefaultFormat(QSettings.IniFormat)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())

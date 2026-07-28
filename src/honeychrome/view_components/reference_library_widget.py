"""Reference Library workspace window (§4.2 of the feature plan).

A management UI over the curated SpectralReferenceLibrary store: pick an
instrument, browse its saved/shipped reference spectra, flag the Reference
(Spectral Process) and QC Target (Spectral QC) per fluorophore, plot/compare
spectra, and import/export/delete. First cut — see _local_docs plan for the
full spec (double-click rename, compatibility badge polish, etc.).
"""
import datetime

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QSplitter, QHeaderView, QFileDialog,
    QMessageBox, QAbstractItemView,
)

from honeychrome.controller_components.spectral_reference_library import (
    SpectralReferenceLibrary, compute_config_key,
)
from honeychrome.controller_components.cytometer_whitelist import (
    get_detector_laser_map, LASER_LABEL_COLORS,
)

import logging
logger = logging.getLogger(__name__)

_COLS = ['Fluorophore', 'Origin', 'Config', 'Reference', 'QC Target', 'Source', 'Created', 'Compatible']
_COL_REFERENCE = _COLS.index('Reference')
_COL_QC = _COLS.index('QC Target')

# a small line-colour cycle for overlaying multiple spectra
_LINE_COLORS = ['#328FE7', '#E74C3C', '#ACF312', '#7F00FF', '#D886F9', '#F39C12', '#16A085']


class ReferenceLibraryWidget(QWidget):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, controller, bus):
        if getattr(self, '_initialized', False):
            return
        super().__init__()
        self._initialized = True
        self.controller = controller
        self.bus = bus
        self.library = SpectralReferenceLibrary()
        self.library.ensure_honeychrome_rows_populated()

        self._row_profiles = []   # row index -> ReferenceProfile
        self._loading = False     # guard itemChanged during table (re)build

        self.setWindowTitle('Reference Library')
        self.setGeometry(40, 40, 1000, 620)
        self._build_ui()
        self._reload_instruments()

    # --- UI construction -----------------------------------------------------
    def _build_ui(self):
        outer = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel('Instrument:'))
        self.instrument_combo = QComboBox()
        self.instrument_combo.currentTextChanged.connect(self._reload_table)
        top.addWidget(self.instrument_combo)
        top.addStretch(1)
        self.import_button = QPushButton('Import CSV…')
        self.import_button.clicked.connect(self._on_import)
        self.export_button = QPushButton('Export CSV…')
        self.export_button.clicked.connect(self._on_export)
        self.delete_button = QPushButton('Delete')
        self.delete_button.clicked.connect(self._on_delete)
        top.addWidget(self.import_button)
        top.addWidget(self.export_button)
        top.addWidget(self.delete_button)
        outer.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(16)

        self.table = QTableWidget(0, len(_COLS))
        self.table.setHorizontalHeaderLabels(_COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        splitter.addWidget(self.table)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Normalised intensity')
        self.plot_widget.setLabel('bottom', 'Detector channel')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()
        splitter.addWidget(self.plot_widget)
        splitter.setSizes([340, 260])
        outer.addWidget(splitter)

        self.status = QLabel('')
        outer.addWidget(self.status)

    # --- current experiment config (for the Compatible column) ---------------
    def _current_config_key(self):
        try:
            raw = self.controller.experiment.settings.get('raw', {})
            db_col = raw.get('cytometer_db_col')
            event_pnn = raw.get('event_channels_pnn') or []
            fl_ids = raw.get('fluorescence_channel_ids') or []
            channels = [event_pnn[i] for i in fl_ids if i < len(event_pnn)]
            if not channels:
                return None
            return compute_config_key(db_col, channels)
        except Exception:
            return None

    # --- populate ------------------------------------------------------------
    def _reload_instruments(self):
        self.instrument_combo.blockSignals(True)
        self.instrument_combo.clear()
        self.instrument_combo.addItems(self.library.list_cytometer_keys())
        self.instrument_combo.blockSignals(False)
        self._reload_table()

    def _reload_table(self):
        self._loading = True
        cytometer_key = self.instrument_combo.currentText()
        profiles = self.library.list_profiles(cytometer_key) if cytometer_key else []
        self._row_profiles = profiles
        current_cfg = self._current_config_key()

        self.table.setRowCount(len(profiles))
        for r, p in enumerate(profiles):
            self._set_text(r, 0, p.display_name)
            self._set_text(r, 1, 'Honeychrome' if p.origin == 'honeychrome' else 'User')
            self._set_text(r, 2, p.config_key.split('::')[-1])
            self._set_check(r, _COL_REFERENCE, p.is_reference)
            self._set_check(r, _COL_QC, p.is_qc_target)
            self._set_text(r, 5, p.source_sample_name or '')
            self._set_text(r, 6, datetime.datetime.fromtimestamp(p.created_at).strftime('%Y-%m-%d'))
            compat = '✓' if (current_cfg and p.config_key == current_cfg) else ''
            self._set_text(r, 7, compat)
        self._loading = False
        self.status.setText(f'{len(profiles)} profiles · instrument: {cytometer_key}')
        self._update_delete_enabled()
        self.plot_widget.clear()

    def _set_text(self, r, c, text):
        item = QTableWidgetItem(str(text))
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(r, c, item)

    def _set_check(self, r, c, checked):
        item = QTableWidgetItem('')
        item.setFlags((item.flags() | Qt.ItemFlag.ItemIsUserCheckable) & ~Qt.ItemFlag.ItemIsEditable)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.table.setItem(r, c, item)

    # --- interactions --------------------------------------------------------
    @Slot('QTableWidgetItem*')
    def _on_item_changed(self, item):
        if self._loading:
            return
        row, col = item.row(), item.column()
        if row >= len(self._row_profiles):
            return
        profile = self._row_profiles[row]
        checked = item.checkState() == Qt.CheckState.Checked
        if col == _COL_REFERENCE:
            self.library.set_reference(profile.id, checked)
            self._refresh_flags()
        elif col == _COL_QC:
            self.library.set_qc_target(profile.id, checked)
            self._refresh_flags()

    def _refresh_flags(self):
        """Re-read the Reference/QC flags and update the checkboxes in place,
        without rebuilding the table — so the selection and plot are preserved."""
        self._loading = True
        profiles = self.library.list_profiles(self.instrument_combo.currentText())
        self._row_profiles = profiles
        for r, p in enumerate(profiles):
            self.table.item(r, _COL_REFERENCE).setCheckState(
                Qt.CheckState.Checked if p.is_reference else Qt.CheckState.Unchecked)
            self.table.item(r, _COL_QC).setCheckState(
                Qt.CheckState.Checked if p.is_qc_target else Qt.CheckState.Unchecked)
        self._loading = False

    def _selected_profiles(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        return [self._row_profiles[r] for r in rows if r < len(self._row_profiles)]

    def _on_selection_changed(self):
        self._update_delete_enabled()
        self._plot_selected()

    def _update_delete_enabled(self):
        selected = self._selected_profiles()
        self.delete_button.setEnabled(bool(selected) and all(p.is_deletable for p in selected))

    def _plot_selected(self):
        self.plot_widget.clear()
        selected = self._selected_profiles()
        if not selected:
            return
        cytometer_key = self.instrument_combo.currentText()
        laser_map = get_detector_laser_map(cytometer_key)
        for i, p in enumerate(selected):
            channels = p.channel_names
            x = np.arange(len(channels))
            y = np.array([p.profile.get(c, 0.0) for c in channels], dtype=float)
            colour = _LINE_COLORS[i % len(_LINE_COLORS)]
            self.plot_widget.plot(x, y, pen=pg.mkPen(colour, width=2), name=p.display_name)
        # colour x tick labels per laser (like the QC viewer)
        if selected:
            channels = selected[0].channel_names
            ticks = []
            for idx, ch in enumerate(channels):
                laser = laser_map.get(ch)
                ticks.append((idx, ch.removesuffix('-A')))
            stride = max(1, round(len(channels) / 40))
            self.plot_widget.getAxis('bottom').setTicks([
                [(m, lab) for j, (m, lab) in enumerate(ticks) if j % stride == 0], []
            ])

    def _on_delete(self):
        selected = self._selected_profiles()
        if not selected:
            return
        names = ', '.join(p.display_name for p in selected)
        if QMessageBox.question(self, 'Delete profiles', f'Delete: {names}?') != QMessageBox.StandardButton.Yes:
            return
        for p in selected:
            try:
                self.library.delete_profile(p.id)
            except ValueError as e:
                QMessageBox.warning(self, 'Cannot delete', str(e))
        self._reload_table()

    def _on_import(self):
        cytometer_key = self.instrument_combo.currentText()
        if not cytometer_key:
            QMessageBox.information(self, 'Import', 'Select an instrument first.')
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Import reference CSV', '', 'CSV files (*.csv)')
        if not path:
            return
        try:
            df = pd.read_csv(path, index_col=0)
            channel_names = [str(c) for c in df.columns]
            config_key = compute_config_key(cytometer_key, channel_names)
            for fluor, row in df.iterrows():
                profile = {c: float(row[c]) for c in channel_names}
                self.library.save_profile(
                    fluorophore=str(fluor), profile=profile, cytometer_key=cytometer_key,
                    config_key=config_key, channel_names=channel_names, origin='user',
                )
        except Exception as e:
            QMessageBox.critical(self, 'Import failed', str(e))
            return
        self._reload_table()

    def _on_export(self):
        cytometer_key = self.instrument_combo.currentText()
        profiles = self.library.list_profiles(cytometer_key)
        if not profiles:
            QMessageBox.information(self, 'Export', 'Nothing to export.')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Export reference CSV', f'{cytometer_key}_reference.csv', 'CSV files (*.csv)')
        if not path:
            return
        try:
            rows = {p.display_name: p.profile for p in profiles}
            pd.DataFrame.from_dict(rows, orient='index').to_csv(path)
        except Exception as e:
            QMessageBox.critical(self, 'Export failed', str(e))

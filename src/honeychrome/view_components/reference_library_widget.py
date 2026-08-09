"""Reference Library workspace window (§4.2 of the feature plan).

A management UI over the curated SpectralReferenceLibrary store: pick an
instrument, browse its saved/shipped reference spectra, flag the Reference
(Spectral Process) and QC Target (Spectral QC) per fluorophore, plot/compare
spectra, and import/export/delete.
"""
import colorsys
import datetime

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QSplitter, QHeaderView, QFileDialog,
    QMessageBox, QAbstractItemView, QScrollArea, QColorDialog, QSizePolicy,
)

from honeychrome.controller_components.spectral_reference_library import (
    SpectralReferenceLibrary, compute_config_key,
)
from honeychrome.controller_components.cytometer_whitelist import (
    get_detector_laser_map, LASER_LABEL_COLORS,
)
from honeychrome.view_components.profiles_viewer import (
    BottomAxisVerticalTickLabels, FlowLayout,
)

import logging
logger = logging.getLogger(__name__)


# (header, attribute, tooltip, editable)
_COLUMNS = [
    ('Fluorophore', 'display_name',
     'The fluorophore this spectrum belongs to.', True),
    ('Antigen', 'antigen',
     'The marker this antibody-fluorophore conjugate targets (e.g. CD3). Filled in '
     'from the Spectral Process tab when a reference is saved; editable here.', True),
    ('Peak Detector', 'gate_channel',
     'The detector where this spectrum peaks (its major channel).', False),
    ('Bead/Cell', 'particle_type',
     'Whether the control was run on beads or cells. Filled in from the Spectral '
     'Process tab when a reference is saved.', False),
    ('Origin', 'origin',
     'Honeychrome = shipped with the app (cannot be deleted). User = saved or '
     'imported by you.', False),
    ('Config', 'config_key',
     'The exact detector configuration this spectrum was measured on.', False),
    ('Reference', 'is_reference',
     'Use this spectrum for unmixing in Spectral Process: it is offered first for a '
     '"from Library" control. One per fluorophore per detector configuration.', False),
    ('QC Target', 'is_qc_target',
     'Use this spectrum as the target for the Spectral QC cosine-similarity check on '
     'this instrument, instead of the shipped one. One per fluorophore per instrument.', False),
    ('Lot Number', 'lot_number',
     'Reagent lot number. Free text — double-click to edit.', True),
    ('Notes', 'notes',
     'Anything worth recording: catalogue number, fixed/fresh, tissue/PBMC, etc. '
     'Free text — double-click to edit.', True),
    ('Source Sample', 'source_sample_name',
     'The tube this spectrum was generated from (blank for shipped spectra).', False),
    ('Created', 'created_at', 'When this entry was added to the library.', False),
    ('Compatible', '_compatible',
     'Ticked when this spectrum was measured on the same detector configuration as the '
     'currently open experiment, so it can be used for unmixing directly.', False),
]
_HEADERS = [c[0] for c in _COLUMNS]
_COL_REFERENCE = _HEADERS.index('Reference')
_COL_QC = _HEADERS.index('QC Target')
_EDITABLE_COLS = {i: c[1] for i, c in enumerate(_COLUMNS) if c[3]}

_ID_ROLE = Qt.ItemDataRole.UserRole
_SORT_ROLE = Qt.ItemDataRole.UserRole + 1

# Minimum horizontal space per detector. Below this the 45-degree labels start to
# collide, so instead of squeezing (or dropping) labels the plot grows wider and
# scrolls horizontally — important for high-laser-count instruments (100+ detectors).
_MIN_PX_PER_CHANNEL = 14


def _distinct_colours(n: int) -> list[str]:
    """``n`` visually distinct colours: a hand-picked set first, then golden-angle
    hue steps (alternating lightness) so large selections keep separating instead
    of cycling through the same few colours."""
    base = [
        '#328FE7', '#E74C3C', '#2ECC71', '#7F00FF', '#F39C12', '#16A085',
        '#D886F9', '#795548', '#00BCD4', '#C0392B', '#8BC34A', '#3F51B5',
        '#FF6F00', '#009688', '#9C27B0', '#607D8B', '#E91E63', '#4CAF50',
        '#FFC107', '#2196F3',
    ]
    if n <= len(base):
        return base[:n]
    colours = list(base)
    golden = 0.618033988749895
    hue = 0.0
    while len(colours) < n:
        hue = (hue + golden) % 1.0
        light = 0.45 if len(colours) % 2 else 0.68
        r, g, b = colorsys.hls_to_rgb(hue, light, 0.75)
        colours.append('#%02X%02X%02X' % (int(r * 255), int(g * 255), int(b * 255)))
    return colours


class _SortableItem(QTableWidgetItem):
    """Table item that sorts on an explicit sort key (so Created sorts
    chronologically and the tick columns sort by state, not by text)."""

    def __lt__(self, other):
        mine = self.data(_SORT_ROLE)
        theirs = other.data(_SORT_ROLE) if isinstance(other, QTableWidgetItem) else None
        if mine is None or theirs is None:
            return super().__lt__(other)
        try:
            return mine < theirs
        except TypeError:
            return str(mine) < str(theirs)


class LegendSwatch(QWidget):
    """One legend entry below the plot.

    Hovering highlights the matching trace (and fades the rest); double-clicking
    opens a colour picker for it.
    """

    def __init__(self, colour: str, text: str, profile_id, on_recolour, on_hover):
        super().__init__()
        self._on_recolour = on_recolour
        self._on_hover = on_hover
        self.profile_id = profile_id
        self.text = text
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(5)

        self.swatch = QLabel()
        self.swatch.setFixedSize(14, 14)
        self._set_swatch_colour(colour)
        layout.addWidget(self.swatch)
        layout.addWidget(QLabel(text))

        self.setToolTip(f'{text}\nHover to highlight this spectrum · double-click to change its colour')
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)

    def _set_swatch_colour(self, colour: str):
        self.swatch.setStyleSheet(f'background-color: {colour}; border:1px solid #444;')

    def enterEvent(self, event):
        self._on_hover(self.profile_id)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._on_hover(None)
        super().leaveEvent(event)

    def mouseDoubleClickEvent(self, event):
        colour = QColorDialog.getColor(parent=self, title=f'Colour for {self.text}')
        if colour.isValid():
            self._set_swatch_colour(colour.name())
            self._on_recolour(self.text, colour.name())


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

        self._profiles_by_id = {}    # id -> ReferenceProfile
        self._colour_overrides = {}  # display_name -> '#RRGGBB' chosen by the user
        self._curves = {}            # profile id -> (PlotDataItem, colour) currently drawn
        self._hovered_id = None      # profile id currently highlighted, if any
        self._loading = False        # guard itemChanged during table (re)build

        self.setWindowTitle('Reference Library')
        self.setGeometry(40, 40, 1100, 700)
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

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_HEADERS)
        for i, (_, _, tooltip, _) in enumerate(_COLUMNS):
            self.table.horizontalHeaderItem(i).setToolTip(tooltip)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        splitter.addWidget(self.table)

        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        # 45-degree, laser-coloured detector labels (same colour convention as the
        # Raw Data / Spectral Process plots, but angled so they never overlap)
        self._axis_bottom = BottomAxisVerticalTickLabels(angle=45)
        self.plot_widget = pg.PlotWidget(axisItems={'bottom': self._axis_bottom})
        self.plot_widget.setLabel('left', 'Normalised intensity')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        # wheel zoom makes the plot the wrong size far too easily
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.getViewBox().setMenuEnabled(False)

        # scroll horizontally rather than cramming every detector into the window
        self._plot_scroll = QScrollArea()
        self._plot_scroll.setWidgetResizable(True)
        self._plot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._plot_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._plot_scroll.setWidget(self.plot_widget)
        plot_layout.addWidget(self._plot_scroll)

        # legend lives BELOW the plot so long selections don't run off it
        self._legend_area = QScrollArea()
        self._legend_area.setWidgetResizable(True)
        self._legend_area.setFixedHeight(76)
        self._legend_host = QWidget()
        self._legend_layout = FlowLayout(self._legend_host)
        self._legend_area.setWidget(self._legend_host)
        plot_layout.addSpacing(16)
        plot_layout.addWidget(self._legend_area)

        splitter.addWidget(plot_container)
        splitter.setSizes([330, 340])
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
        self.table.setSortingEnabled(False)
        cytometer_key = self.instrument_combo.currentText()
        profiles = self.library.list_profiles(cytometer_key) if cytometer_key else []
        self._profiles_by_id = {p.id: p for p in profiles}
        current_cfg = self._current_config_key()

        self.table.setRowCount(len(profiles))
        for r, p in enumerate(profiles):
            created = datetime.datetime.fromtimestamp(p.created_at)
            values = {
                'display_name': p.display_name,
                'antigen': p.antigen or '',
                'gate_channel': p.gate_channel or '',
                'particle_type': p.particle_type or '',
                'origin': 'Honeychrome' if p.origin == 'honeychrome' else 'User',
                'config_key': p.config_key.split('::')[-1],
                'lot_number': p.lot_number or '',
                'notes': p.notes or '',
                'source_sample_name': p.source_sample_name or '',
                'created_at': created.strftime('%Y-%m-%d'),
                '_compatible': '✓' if (current_cfg and p.config_key == current_cfg) else '',
            }
            for c, (_, attr, _, editable) in enumerate(_COLUMNS):
                if c == _COL_REFERENCE:
                    self._set_check(r, c, p.is_reference, p.id)
                elif c == _COL_QC:
                    self._set_check(r, c, p.is_qc_target, p.id)
                else:
                    sort_value = p.created_at if attr == 'created_at' else values[attr].lower()
                    self._set_text(r, c, values[attr], p.id, editable, sort_value)
        self.table.setSortingEnabled(True)
        self._loading = False
        self.status.setText(f'{len(profiles)} profiles · instrument: {cytometer_key}')
        self._update_delete_enabled()
        self._plot_selected()

    def _set_text(self, r, c, text, profile_id, editable=False, sort_value=None):
        item = _SortableItem(str(text))
        if not editable:
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        item.setData(_ID_ROLE, profile_id)
        item.setData(_SORT_ROLE, sort_value if sort_value is not None else str(text).lower())
        self.table.setItem(r, c, item)

    def _set_check(self, r, c, checked, profile_id):
        item = _SortableItem('')
        item.setFlags((item.flags() | Qt.ItemFlag.ItemIsUserCheckable) & ~Qt.ItemFlag.ItemIsEditable)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        item.setData(_ID_ROLE, profile_id)
        item.setData(_SORT_ROLE, 0 if checked else 1)
        self.table.setItem(r, c, item)

    # --- interactions --------------------------------------------------------
    def _profile_id_at(self, row):
        item = self.table.item(row, 0)
        return item.data(_ID_ROLE) if item else None

    @Slot('QTableWidgetItem*')
    def _on_item_changed(self, item):
        if self._loading:
            return
        profile_id = item.data(_ID_ROLE)
        if profile_id is None:
            return
        col = item.column()
        if col in (_COL_REFERENCE, _COL_QC):
            checked = item.checkState() == Qt.CheckState.Checked
            if col == _COL_REFERENCE:
                self.library.set_reference(profile_id, checked)
            else:
                self.library.set_qc_target(profile_id, checked)
            self._refresh_flags()
        elif col in _EDITABLE_COLS:
            self.library.update_fields(profile_id, **{_EDITABLE_COLS[col]: item.text()})
            profile = self.library.get_profile(profile_id)
            if profile:
                self._profiles_by_id[profile_id] = profile

    def _refresh_flags(self):
        """Re-read the Reference/QC flags and update the checkboxes in place, without
        rebuilding the table — so the selection and the plot are preserved."""
        self._loading = True
        profiles = self.library.list_profiles(self.instrument_combo.currentText())
        self._profiles_by_id = {p.id: p for p in profiles}
        for r in range(self.table.rowCount()):
            profile = self._profiles_by_id.get(self._profile_id_at(r))
            if profile is None:
                continue
            self.table.item(r, _COL_REFERENCE).setCheckState(
                Qt.CheckState.Checked if profile.is_reference else Qt.CheckState.Unchecked)
            self.table.item(r, _COL_QC).setCheckState(
                Qt.CheckState.Checked if profile.is_qc_target else Qt.CheckState.Unchecked)
        self._loading = False

    def _selected_profiles(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        profiles = [self._profiles_by_id.get(self._profile_id_at(r)) for r in rows]
        return [p for p in profiles if p is not None]

    def _on_selection_changed(self):
        self._update_delete_enabled()
        self._plot_selected()

    def _update_delete_enabled(self):
        selected = self._selected_profiles()
        self.delete_button.setEnabled(bool(selected) and all(p.is_deletable for p in selected))

    # --- plotting ------------------------------------------------------------
    def _plot_selected(self):
        self.plot_widget.clear()
        self._clear_legend()
        self._curves = {}
        self._hovered_id = None
        selected = self._selected_profiles()
        if not selected:
            self._axis_bottom.tick_colors = {}
            self._axis_bottom.setTicks([[], []])
            return

        palette = _distinct_colours(len(selected))
        for i, p in enumerate(selected):
            channels = p.channel_names
            x = np.arange(len(channels))
            y = np.array([p.profile.get(c, 0.0) for c in channels], dtype=float)
            colour = self._colour_overrides.get(p.display_name, palette[i])
            curve = self.plot_widget.plot(x, y, pen=pg.mkPen(colour, width=2))
            self._curves[p.id] = (curve, colour)
            self._legend_layout.addWidget(
                LegendSwatch(colour, p.display_name, p.id, self._on_recolour, self._on_hover_profile))

        # laser-coloured, 45-degree detector labels
        channels = selected[0].channel_names
        laser_map = get_detector_laser_map(self.instrument_combo.currentText())
        pairs = [(i, ch.removesuffix('-A')) for i, ch in enumerate(channels)]
        self._axis_bottom.tick_colors = {
            label: LASER_LABEL_COLORS[laser]
            for ch, (_, label) in zip(channels, pairs)
            if (laser := laser_map.get(ch)) in LASER_LABEL_COLORS
        }
        # the plot is widened below to guarantee room, so every detector is labelled
        self._axis_bottom.setTicks([pairs, []])
        self.plot_widget.setMinimumWidth(_MIN_PX_PER_CHANNEL * len(channels) + 120)
        self.plot_widget.setXRange(-0.5, len(channels) - 0.5, padding=0.02)

    def _clear_legend(self):
        self._legend_layout.clear()

    def _on_recolour(self, display_name, colour):
        self._colour_overrides[display_name] = colour
        self._plot_selected()

    def _on_hover_profile(self, profile_id):
        """Highlight one spectrum (and fade the others) while its legend entry is
        hovered — with many overlaid spectra colour alone is not enough to pick
        one out. ``profile_id`` of None restores every trace."""
        if profile_id == self._hovered_id:
            return
        self._hovered_id = profile_id

        for pid, (curve, colour) in self._curves.items():
            if profile_id is None:
                curve.setPen(pg.mkPen(colour, width=2))
                curve.setZValue(0)
            elif pid == profile_id:
                curve.setPen(pg.mkPen(colour, width=4))
                curve.setZValue(1)
            else:
                faded = QColor(colour)
                faded.setAlpha(50)
                curve.setPen(pg.mkPen(faded, width=1))
                curve.setZValue(0)

        profile = self._profiles_by_id.get(profile_id)
        if profile is None:
            self._set_default_status()
        else:
            peak = f' · peak {profile.gate_channel}' if profile.gate_channel else ''
            self.status.setText(f'Highlighting: {profile.display_name}{peak}')

    def _set_default_status(self):
        cytometer_key = self.instrument_combo.currentText()
        self.status.setText(
            f'{self.table.rowCount()} profiles · instrument: {cytometer_key}')

    # --- actions -------------------------------------------------------------
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
            imported = self.library.import_csv(path, cytometer_key)
        except Exception as e:
            QMessageBox.critical(self, 'Import failed', str(e))
            return
        self._reload_table()
        self.status.setText(f'Imported {len(imported)} spectra into {cytometer_key}.')

    def _on_export(self):
        cytometer_key = self.instrument_combo.currentText()
        profiles = self._selected_profiles() or self.library.list_profiles(cytometer_key)
        if not profiles:
            QMessageBox.information(self, 'Export', 'Nothing to export.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export reference CSV', f'{cytometer_key}_reference.csv', 'CSV files (*.csv)')
        if not path:
            return
        try:
            self.library.export_csv(path, profiles)
        except Exception as e:
            QMessageBox.critical(self, 'Export failed', str(e))
            return
        self.status.setText(f'Exported {len(profiles)} spectra to {path}.')

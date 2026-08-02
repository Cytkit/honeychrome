import json
import sys
import re
from datetime import datetime
from typing import List, Any, Dict
from PySide6 import QtCore
from PySide6.QtCore import Qt, QModelIndex, QTimer, QThread, Slot, QObject, QEvent, QSize, Signal, QSettings
from PySide6.QtWidgets import (QApplication, QFrame, QVBoxLayout, QHBoxLayout, QTableView, QPushButton, QStyledItemDelegate, QComboBox, QLineEdit, QMessageBox, QHeaderView, QLabel, QWidget, QCheckBox, QMenu, QInputDialog)

from honeychrome.controller_components.functions import raw_gates_list
from honeychrome.controller_components.spectral_controller import SpectralAutoGenerator, ProfileUpdater, SpectralCleaner, spectral_library
from honeychrome.controller_components.spectral_reference_library import (
    cosine_similarity_to_reference, SpectralReferenceLibrary, compute_config_key,
)
from honeychrome.view_components.cosine_qc_viewer import COSINE_QC_WARNING_THRESHOLD
from honeychrome.controller_components.spectral_functions import sanitise_control_in_place, _find_default_unstained
from honeychrome.view_components.icon_loader import icon
from honeychrome.settings import spectral_model_column_labels, heading_style, INTERNAL_NEGATIVE_SENTINEL
from honeychrome.view_components.busy_cursor import with_busy_cursor
from honeychrome.view_components.help_toggle_widget import WheelBlocker, HelpToggleWidget
from honeychrome.view_components.help_texts import autospectral_cleaning_help_text
from honeychrome.controller_components.gml_functions_mod_from_flowkit import _rename_channel_in_gml

import logging
logger = logging.getLogger(__name__)

COLUMNS = list(spectral_model_column_labels.keys())
CONTROL_TYPES = ["Single Stained Spectral Control", "Single Stained Spectral Control from Library", "Channel Assignment"]
PARTICLE_TYPES = ["Beads", "Cells"]
NEGATIVE_TYPES = ["Internal Negative", "Unstained Negative"]



class ResizingTable(QTableView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # No scrollbars
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # multiselect
        self.setSelectionBehavior(QTableView.SelectRows)
        self.setSelectionMode(QTableView.ExtendedSelection)

        # fixed row height
        self.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.setWordWrap(False)

        # no sorting, turn on row indices
        self.setSortingEnabled(False)
        self.verticalHeader().setVisible(True)


    def sizeHint(self):
        """
        Always compute the exact size needed for all rows and columns.
        """
        self.resizeRowsToContents()
        self.resizeColumnsToContents()

        h_scrollbar = self.horizontalScrollBar().height() if self.horizontalScrollBar().isVisible() else 0

        width = self.verticalHeader().width() + self.horizontalHeader().length()
        height = self.horizontalHeader().height() + self.verticalHeader().length() + h_scrollbar

        # Add a small margin
        return QSize(width + 4, max([height + 4, 60]))

    def resizeToFit(self):
        self.resizeRowsToContents()

        total_height = 0

        # Add height of all rows
        for row in range(self.model().rowCount()):
            total_height += self.rowHeight(row)

        # Add horizontal header height and scrollbar
        h_scrollbar = self.horizontalScrollBar().height() if self.horizontalScrollBar().isVisible() else 0
        total_height += self.horizontalHeader().height() + h_scrollbar

        # Add frame width (borders)
        total_height += 2 * self.frameWidth()

        # Apply the new height
        self.setMinimumHeight(total_height)
        # self.setMaximumHeight(total_height)

    # def wheelEvent(self, event):
    #     # We tell Qt we didn't handle this.
    #     # It will then look at the parent widget to handle it.
    #     event.ignore()


class _RecalcWorker(QObject):
    finished = Signal()

    def __init__(self, profile_updater, controls, search_results):
        super().__init__()
        self._profile_updater = profile_updater
        self._controls = controls
        self._search_results = search_results

    @Slot()
    def run(self):
        self._profile_updater.flush()
        for control in self._controls:
            self._profile_updater.generate(control, self._search_results)
        self.finished.emit()

class ListTableModel(QtCore.QAbstractTableModel):
    dataEditedSignal = QtCore.Signal(int, int) # changed to track cosmetic vs function changes
    dataDeletedSignal = QtCore.Signal(list)

    def __init__(self, data: List[Dict[str, Any]], fluorescence_channels_pnn, parent=None):
        super().__init__(parent)
        self._data = data
        self.fluorescence_channels_pnn = fluorescence_channels_pnn

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(COLUMNS)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        key = COLUMNS[col]
        val = self._data[row].get(key, None)
        if role in (Qt.DisplayRole, Qt.EditRole):
            if key == "use_cleaned":
                return None   # checkbox widget handles display; suppress cell text
            return "" if val is None else str(val)
        return None

    _COLUMN_TOOLTIPS = {
        'universal_negative_name': (
            'The unstained sample used as the negative reference for this control.\n'
            'To add a sample here: right-click any sample in the Sample panel\n'
            'and choose "Mark as Unstained".'
        ),
    }

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal:
            if role == Qt.DisplayRole:
                return spectral_model_column_labels[COLUMNS[section]]
            if role == Qt.ToolTipRole:
                return self._COLUMN_TOOLTIPS.get(COLUMNS[section])
        if role == Qt.DisplayRole:
            return section + 1
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemIsEnabled
        colname = COLUMNS[index.column()]
        base = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        if colname in ("label", "antigen"):
            base |= Qt.ItemIsEditable
        return base

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole or not index.isValid():
            return False
        row, col = index.row(), index.column()
        key = COLUMNS[col]
        # if key == "gate_channel":
        #     try:
        #         value = int(value)
        #     except ValueError:
        #         return False
        self._data[row][key] = value
        self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
        # changed to return col so we can track cosmetic vs functional changes
        self.dataEditedSignal.emit(row, col)
        return True

    def insertRow(self, position, parent=QModelIndex()):
        self.beginInsertRows(QModelIndex(), position, position)
        new_row = {c: None for c in COLUMNS}

        # if entering channel assignment, give the next unused raw channel
        if position > 0:
            if self._data[-1]['control_type'] == 'Channel Assignment':
                new_row['control_type'] = 'Channel Assignment'
                unused_raw_channels = self.unused_raw_channels()
                new_row['gate_channel'] = unused_raw_channels[0] if self.unused_raw_channels() else None

        self._data.insert(position, new_row)
        self.endInsertRows()
        # self.dataEditedSignal.emit(position)
        return True

    def unused_raw_channels(self, exception=None):
        allowed_channels = set(self.fluorescence_channels_pnn) - ({row['gate_channel'] for row in self._data} - {exception})
        return [c for c in self.fluorescence_channels_pnn if c in allowed_channels]

    def removeRows(self, position, rows=1, parent=QModelIndex()):
        if rows <= 0:
            return False
        self.beginRemoveRows(QModelIndex(), position, position + rows - 1)
        labels = []
        for i in range(position, position + rows):
            labels.append(self._data[i]['label'])
        del self._data[position:position + rows]
        self.endRemoveRows()
        self.dataDeletedSignal.emit(labels)
        return True

    # this is the one that actually seems to be used
    def delete_rows_by_indices(self, indices: List[int]):
        if not indices:
            return
        to_drop = sorted(set(indices), reverse=True)
        self.beginResetModel()
        labels = []
        for i in to_drop:
            labels.append(self._data[i]['label'])
            del self._data[i]
        self.endResetModel()
        self.dataDeletedSignal.emit(labels)

class LabelDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        return QLineEdit(parent)

    def setEditorData(self, editor, index):
        val = index.model().data(index, Qt.EditRole)
        editor.setText(val)

    def setModelData(self, editor, model, index):
        text = editor.text()
        model.setData(index, text, Qt.EditRole)

class SpectralControlsEditor(QFrame):
    def __init__(self, bus, controller, parent=None):
        super().__init__(parent)

        # connect
        self.bus = bus
        self.controller = controller

        # initialise data
        self.fluorescence_channels_pnn = []
        self.update_fluorescence_channels_pnn()
        self.model = ListTableModel(self.controller.experiment.process['spectral_model'], self.fluorescence_channels_pnn)
        self.samples = self.controller.experiment.samples
        self.raw_gating = self.controller.raw_gating

        self.proxy = QtCore.QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.proxy.setFilterKeyColumn(-1)

        self.view = ResizingTable()
        self.view.setModel(self.proxy)
        self.bus.spectralModelUpdated.connect(self.view.resizeToFit)
        self.bus.spectralModelUpdated.connect(self._sort_model_by_gate_channel)
        self.bus.spectralControlAdded.connect(self.view.resizeToFit) #extends the table as autogeneration runs... looks interesting but a bit wonky
        self.bus.sampleTreeUpdated.connect(self.refresh_comboboxes) # check for changes to unstained samples
        self.bus.rawGateRenamed.connect(self._on_raw_gate_renamed)
        self.bus.rawGateRenamed.connect(lambda old_name, new_name: self.refresh_comboboxes())
        self.view.selectionModel().selectionChanged.connect(self._show_selected_profiles)

        # right-click a control to save its spectrum to the Reference Library
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._show_row_context_menu)

        # Different resize modes for different columns
        # Column 0: label
        # Column 1, 2, 3...: control_type particle_type gate_channel gate_label
        # Column 4: sample_name
        # Column 5: gate_label
        # Column 6: universal_negative_name
        header = self.view.horizontalHeader()
        header.setMinimumSectionSize(100)
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.Interactive)   # label
        header.setSectionResizeMode(1, QHeaderView.Interactive)   # antigen
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # control_type
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # particle_type
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # gate_channel
        header.setSectionResizeMode(5, QHeaderView.Fixed)  # sample_name
        header.setDefaultSectionSize(200)
        header.resizeSection(5, 200)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # gate_label
        header.setSectionResizeMode(7, QHeaderView.Fixed)  # universal_negative_name
        header.resizeSection(7, 200)
        header.setSectionResizeMode(8, QHeaderView.ResizeToContents)  # use_cleaned

        self.label_delegate = LabelDelegate()
        self.view.setItemDelegateForColumn(COLUMNS.index("label"), self.label_delegate)
        self.antigen_delegate = LabelDelegate()
        self.view.setItemDelegateForColumn(COLUMNS.index("antigen"), self.antigen_delegate)

        self.model.dataDeletedSignal.connect(self._on_delete_controls)

        # Debounce rapid edits and guard against re-entrant generate calls
        self._pending_update_index = None
        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(300)
        self._update_timer.timeout.connect(self._do_update_control)
        self.model.dataEditedSignal.connect(self._on_update_control)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter table...")
        self.filter_edit.textChanged.connect(self.proxy.setFilterFixedString)

        btn_top_layout = QHBoxLayout()
        self.auto_generate_button = QPushButton(icon('sparkles'), "Auto generate spectral controls")
        self.auto_generate_button.setToolTip('Loop through all single stained control samples (cells and beads, prioritising cells); \nautomatically each control with positive/negative gates in the relevant channel. \nAny previous controls will be cleared.')

        self.negatives_combo = QComboBox()
        self.negatives_combo.addItems(['Using unstained negative', 'Using internal negatives'])
        self.negatives_combo.installEventFilter(WheelBlocker(self.negatives_combo))
        self.negatives_combo.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.fluorescence_channel_filter_combo = QComboBox()
        self.fluorescence_channel_filter_combo.addItems(['Using area channels only', 'Using all fluorescence channels'])
        self.fluorescence_channel_filter_combo.installEventFilter(WheelBlocker(self.fluorescence_channel_filter_combo))
        self.fluorescence_channel_filter_combo.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.unmixing_method_combo = QComboBox()
        self.unmixing_method_combo.addItems(['OLS', 'WLS'])
        self.unmixing_method_combo.installEventFilter(WheelBlocker(self.unmixing_method_combo))
        self.unmixing_method_combo.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.unmixing_method_combo.setToolTip(
            'OLS: uniform weights (default).\n'
            'WLS: shot-noise model'
        )
        self.update_combos()
        self.bus.spectralModelUpdated.connect(self.update_combos)

        self.force_recalc_btn = QPushButton(icon('refresh'), "Recalculate")
        self.force_recalc_btn.setToolTip('Recalculate the unmixing matrix; reset spillover and unmixed plots and gates. \n(Useful if the control data has been replaced or control gates have moved under the same names.) \nDoes not change the spectral controls table.')

        self.clean_controls_btn = QPushButton(icon('sparkles'), "Clean Controls")
        self.clean_controls_btn.setToolTip(
            'Run the cleaning pipeline for all cell controls that have an Unstained Negative assigned:\n'
            '  • Saturation exclusion\n'
            '  • Cosine-similarity event selection (least AF-like events)\n'
            '  • kNN scatter-matched AF subtraction\n'
            'Once complete, each control will have a "Use Cleaned" checkbox.\n'
            'Cleaned controls use the kNN-subtracted spectrum directly.'
        )
        self.clean_controls_btn.clicked.connect(self._on_clean_controls)

        btn_top_layout.addWidget(self.auto_generate_button)
        btn_top_layout.addWidget(self.negatives_combo)
        btn_top_layout.addWidget(self.fluorescence_channel_filter_combo)
        btn_top_layout.addWidget(self.unmixing_method_combo)
        btn_top_layout.addWidget(self.force_recalc_btn)
        btn_top_layout.addStretch()

        # ---- Bulk unstained-negative row ----
        bulk_neg_layout = QHBoxLayout()
        bulk_neg_layout.addWidget(QLabel("Set negative for all:"))

        self.bulk_particle_type_combo = QComboBox()
        self.bulk_particle_type_combo.addItems(["Cells", "Beads"])
        self.bulk_particle_type_combo.installEventFilter(WheelBlocker(self.bulk_particle_type_combo))
        self.bulk_particle_type_combo.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        bulk_neg_layout.addWidget(self.bulk_particle_type_combo)

        self.bulk_negative_combo = QComboBox()
        self.bulk_negative_combo.installEventFilter(WheelBlocker(self.bulk_negative_combo))
        self.bulk_negative_combo.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        bulk_neg_layout.addWidget(self.bulk_negative_combo)

        self.bulk_apply_btn = QPushButton("Apply to all")
        self.bulk_apply_btn.setToolTip(
            'Set the Unstained Negative for every control whose particle type\n'
            'matches the selection on the left.'
        )
        self.bulk_apply_btn.clicked.connect(self._on_bulk_apply_negative)
        bulk_neg_layout.addWidget(self.bulk_apply_btn)
        bulk_neg_layout.addStretch()

        # Wrap in a widget so it can be shown/hidden as a unit
        self._bulk_neg_widget = QWidget()
        self._bulk_neg_widget.setLayout(bulk_neg_layout)
        self.bulk_particle_type_combo.currentTextChanged.connect(self._refresh_bulk_negative_combo)

        self.add_row_btn = QPushButton(icon('plus'), "Add Control")
        self.select_all_btn = QPushButton("Select All")
        self.select_none_btn = QPushButton("Select None")
        self.delete_btn = QPushButton("Delete Selected")

        self.auto_generate_button.clicked.connect(self.auto_generate)
        self._negative_type_timer = QTimer()
        self._negative_type_timer.setSingleShot(True)
        self._negative_type_timer.setInterval(250)
        self._negative_type_timer.timeout.connect(self.set_negative_type)
        self.negatives_combo.currentTextChanged.connect(self._negative_type_timer.start)
        self.fluorescence_channel_filter_combo.currentTextChanged.connect(self.set_fluorescence_channel_filter)
        _METHOD_MAP = {'OLS': 'OLS', 'WLS': 'WLS'}
        self.unmixing_method_combo.currentTextChanged.connect(
            lambda text: self._on_unmixing_method_changed(_METHOD_MAP[text])
        )
        self.force_recalc_btn.clicked.connect(self._on_force_recalc)
        self.add_row_btn.clicked.connect(self.add_row)
        self.select_all_btn.clicked.connect(self.view.selectAll)
        self.select_none_btn.clicked.connect(self.view.clearSelection)
        self.delete_btn.clicked.connect(self.delete_selected_rows)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.add_row_btn)
        btn_layout.addWidget(self.select_all_btn)
        btn_layout.addWidget(self.select_none_btn)
        btn_layout.addWidget(self.delete_btn)
        btn_layout.addStretch()

        # ---- AutoSpectral Cleaning activation ----
        self.activate_cleaning_checkbox = QCheckBox("Show AutoSpectral Cleaning")
        self.activate_cleaning_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 13pt;
                font-weight: bold;
                spacing: 8px;
                padding: 6px 0px;
            }
        """)
        self.activate_cleaning_checkbox.setToolTip(
            'Show AutoSpectral Control Cleaning options: "Clean Controls" button, '
            '"Use Cleaned" and "Exclude noise" table columns, peak-channel histograms, '
            'scatter-matching and noise exclusion diagnostic plots.'
        )

        self.cleaning_help_widget = HelpToggleWidget(
            title="Show Cleaning Help",
            text=autospectral_cleaning_help_text,
        )
        self.cleaning_help_widget.setVisible(False)

        # Persist cleaning help expanded/collapsed state whenever toggled
        self.cleaning_help_widget.help_button.clicked.connect(self._save_cleaning_help_state)

        # ---- Wiring ----
        self.activate_cleaning_checkbox.toggled.connect(self._on_activate_cleaning_toggled)

        # ---- Main layout ----
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Spectral Model Editor")
        layout.addWidget(title)
        title.setStyleSheet(heading_style)
        layout.addWidget(QLabel("Note: editing this table resets unmixed cytometry (plots, gates and fine-tuning (spillover) matrix)"))
        layout.addWidget(self.activate_cleaning_checkbox)
        layout.addWidget(self.cleaning_help_widget)
        layout.addLayout(btn_top_layout)
        layout.addWidget(self._bulk_neg_widget)
        layout.addWidget(self.filter_edit)
        layout.addWidget(self.view)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        # Build combobox widgets after layout setup
        self.refresh_comboboxes()
        # Hide cleaning columns by default
        self._set_cleaning_columns_visible(False)
        self.thread = None
        self.spectral_auto_generator = None
        self.profile_updater = ProfileUpdater(self.controller, self.bus)
        self.spectral_library_search_results = None
        # Populate bulk-negative combo now that all widgets exist
        self._refresh_bulk_negative_combo()

    def _set_cleaning_columns_visible(self, visible: bool):
        """Show or hide the cleaning-only table columns."""
        for col_name in ("use_cleaned",):
            col_idx = COLUMNS.index(col_name)
            header = self.view.horizontalHeader()
            if visible:
                header.showSection(col_idx)
            else:
                header.hideSection(col_idx)

    def _on_activate_cleaning_toggled(self, checked: bool):
        self.cleaning_help_widget.setVisible(checked)
        if checked:
            # Auto-expand help text when cleaning section is first shown
            self.cleaning_help_widget.help_label.setVisible(True)
            self.cleaning_help_widget.help_button.setText("Hide Help")
        self._set_cleaning_columns_visible(checked)
        QSettings("honeychrome", "app_configuration").setValue("show_autospectral_cleaning", checked)
        if self.bus:
            self.bus.cleaningActivated.emit(checked)

    def _save_cleaning_help_state(self):
        # Called after toggle_help has already flipped the label, so isVisible()
        # reflects the new state at the time this fires.
        QSettings("honeychrome", "app_configuration").setValue(
            "cleaning_help_expanded",
            self.cleaning_help_widget.help_label.isVisible()
        )
    
    def update_fluorescence_channels_pnn(self):
        event_channels_pnn = self.controller.experiment.settings['raw']['event_channels_pnn']
        fluorescence_channel_ids = self.controller.filtered_raw_fluorescence_channel_ids
        self.fluorescence_channels_pnn.clear()
        self.fluorescence_channels_pnn.extend([event_channels_pnn[i] for i in fluorescence_channel_ids])

    def set_negative_type(self):
        # Cancel any pending per-control update — the full refresh below supersedes it
        self._update_timer.stop()
        if self.negatives_combo.currentText() == 'Using unstained negative':
            self.controller.experiment.process['negative_type'] = 'unstained'
            self.negatives_combo.setToolTip('Negative set to "Neg Unstained" gate on Unstained sample')
            default_unstained = _find_default_unstained(self.samples['all_samples'])
            if not default_unstained:
                unstained_paths = self.samples.get('unstained_samples', [])
                if unstained_paths:
                    default_unstained = self.samples['all_samples'].get(unstained_paths[0])

            # Find a bead-specific unstained: must be manually tagged AND match bead regex.
            manually_tagged = set(self.samples.get('unstained_samples', []))
            default_unstained_bead = None
            for path, name in self.samples['all_samples'].items():
                is_unstained = (
                    path in manually_tagged
                    or bool(re.search(r'unstained', name, re.IGNORECASE))
                    or bool(re.search(r'unstained', path, re.IGNORECASE))
                )
                if not is_unstained:
                    continue
                if re.search(r'bead', name, re.IGNORECASE) or re.search(r'bead', path, re.IGNORECASE):
                    default_unstained_bead = name
                    break
                
            for control in self.model._data:
                if control.get('control_type') == 'Single Stained Spectral Control':
                    if not control.get('universal_negative_name'):
                        if control.get('particle_type') == 'Cells' and default_unstained:
                            control['universal_negative_name'] = default_unstained
                        elif control.get('particle_type') == 'Beads' and default_unstained_bead:
                            control['universal_negative_name'] = default_unstained_bead
                        else:
                            control['universal_negative_name'] = INTERNAL_NEGATIVE_SENTINEL
        else:
            self.controller.experiment.process['negative_type'] = 'internal'
            self.negatives_combo.setCurrentText('Using internal negatives')
            self.negatives_combo.setToolTip('Negative set to bottom percentile of each control sample')
        logger.info(f'SpectralModelEditor: set negative type to {self.controller.experiment.process["negative_type"]}')
        self._update_universal_negative_column_visibility()
        self.refresh_comboboxes()  # rebuilds all comboboxes, making layoutChanged redundant

        # Changing the negative type invalidates all existing profiles. Clear them
        # so that when the user presses Recalculate, the skip-if-valid logic in
        # _on_force_recalc cannot retain stale profiles computed with the old
        # negative type. Do not trigger recalculation automatically — the user
        # must press Recalculate to recompute with the new setting.
        self.profile_updater.profiles.clear()
        if self.bus:
            self.bus.statusMessage.emit(
                'Negative type changed — press Recalculate to update spectral profiles.'
            )

    def set_fluorescence_channel_filter(self):
        if self.model.rowCount():
            reply = QMessageBox.question(self,
                                         f"Change to {self.fluorescence_channel_filter_combo.currentText()}",
                                         f"This will clear the spectral model. Are you sure you wish to continue?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.model.delete_rows_by_indices(list(range(len(self.model._data))))
            else:
                self.update_combos()
                return

        if self.fluorescence_channel_filter_combo.currentText() == 'Using all fluorescence channels':
            self.controller.experiment.process['fluorescence_channel_filter'] = 'all_fluorescence'
            self.fluorescence_channel_filter_combo.setToolTip('Including both area and height channels in spectral model')
        else:
            self.controller.experiment.process['fluorescence_channel_filter'] = 'area_only'
            self.fluorescence_channel_filter_combo.setToolTip('Ignoring height channels in spectral model')
        logger.info(f'SpectralModelEditor: set fluorescence channel filter to {self.controller.experiment.process['fluorescence_channel_filter']}')

        # Refresh filtered channel state so -H/-W channels become
        # addable/removable immediately, matching auto_conventional().
        self.controller.filter_raw_fluorescence_channels()
        self.update_fluorescence_channels_pnn()

        if self.bus:
            self.bus.spectralModelUpdated.emit()
            self.bus.showSelectedProfiles.emit(None)

    def update_combos(self):
        # Block signals so that programmatic sync does not fire set_negative_type
        # or set_fluorescence_channel_filter, which would clear profiles and
        # trigger unnecessary recalculation.
        self.negatives_combo.blockSignals(True)
        self.fluorescence_channel_filter_combo.blockSignals(True)
        self.unmixing_method_combo.blockSignals(True)
        try:
            if self.controller.experiment.process['negative_type'] == 'unstained':
                self.negatives_combo.setCurrentText('Using unstained negative')
                self.negatives_combo.setToolTip('Negative set to "Neg Unstained" gate on Unstained sample')
            elif self.controller.experiment.process['negative_type'] == 'internal':
                self.negatives_combo.setCurrentText('Using internal negatives')
                self.negatives_combo.setToolTip('Negative set to bottom percentile of each control sample')

            if self.controller.experiment.process['fluorescence_channel_filter'] == 'area_only':
                self.fluorescence_channel_filter_combo.setCurrentText('Using area channels only')
                self.fluorescence_channel_filter_combo.setToolTip('Ignoring height channels in spectral model')
            elif self.controller.experiment.process['fluorescence_channel_filter'] == 'all_fluorescence':
                self.fluorescence_channel_filter_combo.setCurrentText('Using all fluorescence channels')
                self.fluorescence_channel_filter_combo.setToolTip('Including both area and height channels in spectral model')

            _REVERSE_METHOD_MAP = {'OLS': 'OLS', 'WLS': 'WLS'}
            saved_method = self.controller.experiment.settings.get('unmixing_method', 'OLS')
            self.unmixing_method_combo.setCurrentText(_REVERSE_METHOD_MAP.get(saved_method, 'OLS'))

            self._update_universal_negative_column_visibility()
        finally:
            self.negatives_combo.blockSignals(False)
            self.fluorescence_channel_filter_combo.blockSignals(False)
            self.unmixing_method_combo.blockSignals(False)

    def _update_universal_negative_column_visibility(self):
        using_unstained = self.controller.experiment.process.get('negative_type') == 'unstained'
        header = self.view.horizontalHeader()
        # Unstained Negative column — visible only when using unstained negatives
        col_unstained = COLUMNS.index("universal_negative_name")
        if using_unstained:
            header.showSection(col_unstained)
        else:
            header.hideSection(col_unstained)
        # Negative Gate column — visible only when using internal negatives
        col_neg = COLUMNS.index("neg_gate_label")
        if using_unstained:
            header.hideSection(col_neg)
        else:
            header.showSection(col_neg)
        # Bulk-negative row follows the same visibility as the column
        if hasattr(self, '_bulk_neg_widget'):
            self._bulk_neg_widget.setVisible(using_unstained)
            if using_unstained:
                self._refresh_bulk_negative_combo()

    def _refresh_bulk_negative_combo(self):
        """Populate bulk_negative_combo with options appropriate for the selected particle type."""
        particle_type = self.bulk_particle_type_combo.currentText()  # "Cells" or "Beads"
        all_samples = self.samples.get('all_samples', {})
        manually_tagged = set(self.samples.get('unstained_samples', []))

        options = []
        for p, tube_name in all_samples.items():
            is_unstained = (
                p in manually_tagged
                or bool(re.search(r'unstained', tube_name, re.IGNORECASE))
                or bool(re.search(r'unstained', p, re.IGNORECASE))
            )
            if not is_unstained:
                continue
            is_bead = bool(re.search(r'[Bb]eads', tube_name))
            if particle_type == 'Beads' and not is_bead:
                continue
            if particle_type == 'Cells' and is_bead:
                continue
            options.append(tube_name)

        prev = self.bulk_negative_combo.currentText()
        self.bulk_negative_combo.blockSignals(True)
        self.bulk_negative_combo.clear()
        self.bulk_negative_combo.addItems([INTERNAL_NEGATIVE_SENTINEL] + options)
        # Restore previous selection if still available.
        # On first population (or if prev was the sentinel), default to the
        # first available unstained sample instead of Internal Negative.
        idx = self.bulk_negative_combo.findText(prev) if prev and prev != INTERNAL_NEGATIVE_SENTINEL else -1
        if idx >= 0:
            self.bulk_negative_combo.setCurrentIndex(idx)
        elif options:
            self.bulk_negative_combo.setCurrentIndex(1)   # first unstained entry
        # else: leave at 0 (sentinel) — no unstained samples available
        self.bulk_negative_combo.blockSignals(False)

    def _has_unstained_cell_sample(self) -> bool:
        """True if at least one loaded sample is tagged/named "unstained"
        and is not a bead sample.

        Clean Controls needs a genuine unstained CELL control as the
        negative reference for cosine-similarity selection and kNN
        scatter-matched AF subtraction. Without one, cell controls would
        otherwise fall back to the internal-negative approximation
        (top-bright minus bottom-dim within the same tube), which is not
        an adequate substitute.
        """
        all_samples = self.samples.get('all_samples', {})
        manually_tagged = set(self.samples.get('unstained_samples', []))
        for path, name in all_samples.items():
            is_unstained = (
                path in manually_tagged
                or bool(re.search(r'unstained', name, re.IGNORECASE))
                or bool(re.search(r'unstained', path, re.IGNORECASE))
            )
            if not is_unstained:
                continue
            if re.search(r'bead', name, re.IGNORECASE) or re.search(r'bead', path, re.IGNORECASE):
                continue
            return True
        return False

    def _update_clean_controls_enabled(self):
        """Enable/disable Clean Controls based on unstained-cell-sample
        availability. Deliberately surfaced via button state + tooltip,
        never a popup: SpectralCleaner already emits bus.warningMessage
        (-> QMessageBox.warning) per failing control from its background
        thread, and multiple modal dialogs from that thread have previously
        caused native heap-corruption crashes on macOS/PySide6.
        """
        has_unstained_cell = self._has_unstained_cell_sample()
        self.clean_controls_btn.setEnabled(has_unstained_cell)
        if has_unstained_cell:
            self.clean_controls_btn.setToolTip(
                'Run the cleaning pipeline for all cell controls that have an Unstained Negative assigned:\n'
                '  • Saturation exclusion\n'
                '  • Cosine-similarity event selection (least AF-like events)\n'
                '  • kNN scatter-matched AF subtraction\n'
                'Once complete, each control will have a "Use Cleaned" checkbox.\n'
                'Cleaned controls use the kNN-subtracted spectrum directly.'
            )
        else:
            self.clean_controls_btn.setToolTip(
                'Disabled: no unstained cell sample found in this experiment.\n'
                'Clean Controls needs a genuine unstained cell control as the\n'
                'negative reference. Tag a sample as unstained (or include\n'
                '"unstained" in its tube name) to enable this.'
            )
    
    def _on_bulk_apply_negative(self):
        """Set universal_negative_name for all controls matching the selected particle type."""
        particle_type = self.bulk_particle_type_combo.currentText()
        new_negative = self.bulk_negative_combo.currentText()
        if not new_negative:
            return

        changed = False
        for row, control in enumerate(self.model._data):
            if control.get('particle_type') != particle_type:
                continue
            if control.get('control_type') != 'Single Stained Spectral Control':
                continue
            control['universal_negative_name'] = new_negative
            changed = True

        if changed:
            # Rebuild per-row comboboxes so they reflect the new value
            self.refresh_comboboxes()
            # Clear cached profiles — they are now stale
            self.profile_updater.profiles.clear()
            if self.bus:
                self.bus.statusMessage.emit(
                    f'Unstained negative set to "{new_negative}" for all {particle_type} controls'
                    ' — press Recalculate to update spectral profiles.'
                )

    def _emit_spectral_model_updated_when_mode_switch_done(self):
        # Guards against emitting while controller.set_mode is mid-flight on its
        # own worker thread (e.g. a tab switch), which previously raced with
        # refresh_spectral_process and corrupted shared state / crashed.
        if getattr(self.controller, 'mode_switch_in_progress', False):
            QTimer.singleShot(50, self._emit_spectral_model_updated_when_mode_switch_done)
        else:
            self.bus.spectralModelUpdated.emit()

    def refresh_comboboxes(self):
        if getattr(self, '_refreshing_comboboxes', False):
            return
        self._refreshing_comboboxes = True
        try:
            for row in range(self.model.rowCount()):
                self._add_comboboxes_to_row(row)
        finally:
            self._refreshing_comboboxes = False
        self._update_clean_controls_enabled()

    def _add_or_replace_combobox_if_enabled(self, idx, should_have_combobox, items,
                                         searchable=False, row=None):
        proxy_index = self.proxy.mapFromSource(idx)

        # remove existing index widget (if any)
        old_widget = self.view.indexWidget(proxy_index)
        if old_widget is not None:
            self.view.setIndexWidget(proxy_index, None)  # unparents and hides immediately
            old_widget.hide()
            old_widget.setParent(None)                   # synchronous destruction path

        # now safely create and add a new combobox (if needed)
        if should_have_combobox:
            cb = QComboBox()
            cb.installEventFilter(WheelBlocker(cb))
            cb.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
            cb.addItems(items)

            if searchable:
                from PySide6.QtWidgets import QCompleter
                cb.setEditable(True)
                cb.setInsertPolicy(QComboBox.NoInsert)
                completer = QCompleter(items, cb)
                completer.setFilterMode(Qt.MatchContains)
                completer.setCaseSensitivity(Qt.CaseInsensitive)
                cb.setCompleter(completer)

            self.view.setIndexWidget(proxy_index, cb)

            current_val = self.model.data(idx, Qt.EditRole)
            i = cb.findText(str(current_val))
            if i >= 0:
                cb.setCurrentIndex(i)

            if searchable and row is not None:
                def _on_gate_channel_changed(val, i=idx, r=row):
                    self.model.setData(i, val)
                    if val:
                        self.model._data[r]['gate_channel_locked'] = True
                    else:
                        self.model._data[r].pop('gate_channel_locked', None)
                cb.currentTextChanged.connect(_on_gate_channel_changed)
            else:
                cb.currentTextChanged.connect(lambda val, i=idx: self.model.setData(i, val))


    def _add_comboboxes_to_row(self, row):
        current_control_list = []
        unused_raw_channels = []
        if self.model._data[row]['control_type'] == 'Single Stained Spectral Control':
            enable_particle_types_cb = True
            enable_gate_channel_cb = True # user may override the auto-detected major channel
            if self.model._data[row]['particle_type'] == 'Cells':
                current_control_list = [self.samples['all_samples'][sample_path] for sample_path in self.samples['single_stain_controls']]
                enable_sample_name_cb = True
                enable_gate_label_cb = True
            elif self.model._data[row]['particle_type'] == 'Beads':
                current_control_list = [self.samples['all_samples'][sample_path] for sample_path in self.samples['single_stain_controls']]
                enable_sample_name_cb = True
                enable_gate_label_cb = True
            else:
                enable_sample_name_cb = False
                enable_gate_label_cb = False

        elif self.model._data[row]['control_type'] == 'Channel Assignment':
            enable_particle_types_cb = False
            enable_gate_channel_cb = True
            enable_sample_name_cb = False
            enable_gate_label_cb = False
            unused_raw_channels = self.model.unused_raw_channels(self.model._data[row]['gate_channel'])

        elif self.model._data[row]['control_type'] == 'Single Stained Spectral Control from Library':
            enable_particle_types_cb = False
            enable_gate_channel_cb = False
            if self.model._data[row]['label']:
                enable_gate_label_cb = False
                search_results = spectral_library.search_for_label(self.model._data[row]['label'].strip())
                if search_results:
                    enable_sample_name_cb = True
                    # Annotate each result with its display string (or None if channels
                    # don't match this instrument), then derive the combobox list from
                    # those annotations.  Using a single loop avoids the index-mismatch
                    # that occurred when results were filtered but indices weren't remapped.
                    for index in search_results:
                        if list(json.loads(search_results[index]['profile_dict']).keys()) == self.fluorescence_channels_pnn:
                            search_results[index]['current_control_list'] = (
                                '[Spectral Library] '
                                + search_results[index]['sample_name'] + ', '
                                + ('Major Channel: ' + search_results[index]['gate_channel'] + ', ' if search_results[index]['gate_channel'] else '')
                                + 'Experiment: ' + search_results[index]['experiment_root_directory'] + ', '
                                + datetime.fromtimestamp(search_results[index]['timestamp']).strftime('%Y-%m-%d %H:%M:%S') + ' '
                            )
                        else:
                            search_results[index]['current_control_list'] = None

                    current_control_list = [
                        search_results[index]['current_control_list']
                        for index in search_results
                        if search_results[index]['current_control_list'] is not None
                    ]
                    self.spectral_library_search_results = search_results # store search_results for profile generator
                else:
                    enable_sample_name_cb = False
            else:
                enable_particle_types_cb = False
                enable_gate_channel_cb = False
                enable_sample_name_cb = False
                enable_gate_label_cb = False

        else:
            enable_particle_types_cb = False
            enable_gate_channel_cb = False
            enable_sample_name_cb = False
            enable_gate_label_cb = False

        # Build list of available FCS file names for the universal negative combobox.
        # The sentinel option lets users explicitly opt back into the internal negative
        # for a specific control even when the global toggle is "Using unstained negative".
        all_samples = self.samples.get('all_samples', {})
        manually_tagged = set(self.samples.get('unstained_samples', []))
        particle_type = self.model._data[row].get('particle_type', '')

        # Restrict to samples from locations that are unstained (manually tagged
        # or regex-matched) and match the particle type of this control.
        unstained_options = []
        for p, tube_name in all_samples.items():
            is_unstained = (
                p in manually_tagged
                or bool(re.search(r'unstained', tube_name, re.IGNORECASE))
                or bool(re.search(r'unstained', p, re.IGNORECASE))
            )
            if not is_unstained:
                continue
            is_bead = bool(re.search(r'[Bb]eads', tube_name))
            if particle_type == 'Beads' and not is_bead:
                continue
            if particle_type == 'Cells' and is_bead:
                continue
            unstained_options.append(tube_name)

        universal_negative_options = [INTERNAL_NEGATIVE_SENTINEL] + unstained_options
        is_cell_single_stain = (
            self.model._data[row]['control_type'] == 'Single Stained Spectral Control'
            and self.model._data[row]['particle_type'] == 'Cells'
        )
        is_bead_single_stain = (
            self.model._data[row]['control_type'] == 'Single Stained Spectral Control'
            and self.model._data[row]['particle_type'] == 'Beads'
        )
        enable_universal_negative_cb = is_cell_single_stain or is_bead_single_stain

        using_internal = self.controller.experiment.process.get('negative_type') == 'internal'
        enable_neg_gate_label_cb = enable_gate_label_cb and using_internal

        for col_name in ["control_type", "particle_type", "gate_channel", "sample_name", "gate_label", "neg_gate_label", "universal_negative_name"]:
            idx = self.model.index(row, COLUMNS.index(col_name))
            if col_name == "control_type":
                self._add_or_replace_combobox_if_enabled(idx, True, [""] + CONTROL_TYPES)
            elif col_name == "particle_type":
                self._add_or_replace_combobox_if_enabled(idx, enable_particle_types_cb, [""] + PARTICLE_TYPES)
            elif col_name == "gate_channel":
                self.update_fluorescence_channels_pnn()
                is_ssc = self.model._data[row].get('control_type') == 'Single Stained Spectral Control'
                if is_ssc:
                    channel_items = [""] + self.fluorescence_channels_pnn
                else:
                    channel_items = [""] + unused_raw_channels
                self._add_or_replace_combobox_if_enabled(
                    idx, enable_gate_channel_cb, channel_items,
                    searchable=is_ssc, row=row
                )
            elif col_name == "sample_name":
                self._add_or_replace_combobox_if_enabled(idx, enable_sample_name_cb, [""] + current_control_list)
            elif col_name == "gate_label":
                self._add_or_replace_combobox_if_enabled(idx, enable_gate_label_cb, [""] + raw_gates_list(self.raw_gating))
            elif col_name == "neg_gate_label":
                self._add_or_replace_combobox_if_enabled(idx, enable_neg_gate_label_cb, [""] + raw_gates_list(self.raw_gating))
            elif col_name == "universal_negative_name":
                self._add_or_replace_combobox_if_enabled(idx, enable_universal_negative_cb, universal_negative_options)

        # "Use Cleaned" checkbox — visible only when cleaned data exist for this control
        label = self.model._data[row].get('label') or ''
        cleaned_available = label in self.controller.cleaned_events
        uc_col = COLUMNS.index("use_cleaned")
        uc_idx = self.model.index(row, uc_col)
        proxy_uc_idx = self.proxy.mapFromSource(uc_idx)

        old = self.view.indexWidget(proxy_uc_idx)
        if old is not None:
            self.view.setIndexWidget(proxy_uc_idx, None)
            old.hide()
            old.setParent(None)

        if cleaned_available:
            cb = QCheckBox()
            cb.installEventFilter(WheelBlocker(cb))
            # Default to checked if use_cleaned is True or not yet set
            current_val = self.model._data[row].get('use_cleaned')
            cb.setChecked(current_val is not False)  # True or None → checked; False → unchecked
            cb.setToolTip('Use cleaned event pool for profile extraction.\nUncheck to revert to the standard gate-mean method for this control.')

            def _on_toggle(checked, idx=uc_idx, row=row):
                self.model._data[row]['use_cleaned'] = checked
                # Re-run generate() for this control immediately
                self.setEnabled(False)
                self.profile_updater.flush()
                control_valid = self.profile_updater.generate(
                    self.model._data[row], self.spectral_library_search_results
                )

                def _finish_toggle():
                    # refresh_comboboxes() rebuilds this row's checkbox, tearing
                    # down the current one (setIndexWidget(..., None) + setParent(None)).
                    # Doing that synchronously would delete this checkbox while its
                    # own toggled() signal is still unwinding on the call stack —
                    # a native-crash-causing reentrancy hazard. Defer to the next
                    # event-loop tick so toggled() finishes first.
                    self.refresh_comboboxes()
                    self.setEnabled(True)
                    if control_valid:
                        self._emit_spectral_model_updated_when_mode_switch_done()

                QTimer.singleShot(0, _finish_toggle)

            cb.toggled.connect(_on_toggle)
            self.view.setIndexWidget(proxy_uc_idx, cb)
                

    def add_row(self):
        pos = self.model.rowCount()
        self.model.insertRow(pos)
        self.refresh_comboboxes()
        self.view.scrollToBottom()
        self.view.resizeToFit()

        # focus and start editing the new label cell
        label_index = self.proxy.mapFromSource(self.model.index(pos, COLUMNS.index("label")))
        self.view.setCurrentIndex(label_index)
        self.view.edit(label_index)

        # self._on_spectral_control_added()

    def delete_selected_rows(self):
        sel = self.view.selectionModel().selectedRows()
        if not sel:
            QMessageBox.information(self, "No selection", "No rows selected to delete.")
            return
        indices = sorted([self.proxy.mapToSource(s).row() for s in sel])
        reply = QMessageBox.question(self, "Confirm delete", f"Are you sure you want to delete {len(indices)} row(s)?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.model.delete_rows_by_indices(indices)
            self.refresh_comboboxes()
            self.bus.spectralModelUpdated.emit() #spectral model already scrubbed at this point - this will clear unmixing matrix
            self.bus.showSelectedProfiles.emit(None)

    def auto_generate(self):
        # if spectral model already exists, ask if user wants it cleared
        if self.model.rowCount():
            reply = QMessageBox.question(self, "Auto generate spectral model", f"This will overwrite the spectral model. Are you sure you wish to continue?", QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.model.delete_rows_by_indices(list(range(len(self.model._data))))
            else:
                return

        # then if the user has samples but no single stain controls, ask if channel assignment is desired
        if not self.controller.experiment.samples['single_stain_controls']:
            if self.controller.experiment.samples['all_samples']:
                reply = QMessageBox.question(self, "Do conventional cytometry?",
                                             "You have no spectral controls defined."
                                             "\nDo you wish to assign fluorophore names "
                                             "\nto all your raw FCS channels?"
                                             "\nYou can then edit the names.",
                                             QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self.auto_conventional()
                return

            # then if the user has no samples at all, just warn
            elif not self.controller.experiment.samples['all_samples'] and not self.controller.experiment.samples['single_stain_controls']:
                self.bus.warningMessage.emit('Can\'t autogenerate: your list of single stain controls is empty.'
                                             '\n\nPlease add your single stain controls or set the correct folder for '
                                             'single stain controls in experiment configuration. '
                                             '\n\nAlternatively, you can add controls from the spectral library (if previously added) '
                                             'or assign channels (for conventional cytometry).')
                return

        # if user gets through all above, run spectral auto generator
        self.setEnabled(False)
        self.thread = QThread()
        # Read bulk_negative_combo directly
        if self.bulk_particle_type_combo.currentText() != 'Cells':
            self.bulk_particle_type_combo.setCurrentText('Cells')
        preferred = self.bulk_negative_combo.currentText()
        if not preferred or preferred == INTERNAL_NEGATIVE_SENTINEL:
            preferred = None
        self.spectral_auto_generator = SpectralAutoGenerator(self.bus, self.controller,
                                                             preferred_unstained=preferred)

        self.spectral_auto_generator.moveToThread(self.thread)
        self.bus.spectralControlAdded.connect(self._on_spectral_control_added_by_autogenerator)
        self.thread.started.connect(self.spectral_auto_generator.run)
        self.bus.spectralModelUpdated.connect(self.thread.quit)
        self.thread.finished.connect(self.refresh_table_and_enable)
        self.thread.start()

    def auto_conventional(self):
        # first make sure all channels selected
        self.fluorescence_channel_filter_combo.blockSignals(True)
        self.controller.experiment.process['fluorescence_channel_filter'] = 'all_fluorescence'
        self.update_combos()
        self.controller.filter_raw_fluorescence_channels()
        self.profile_updater.refresh()
        self.update_fluorescence_channels_pnn()
        self.fluorescence_channel_filter_combo.blockSignals(False)

        # generate profiles
        spectral_model = self.controller.experiment.process['spectral_model']
        for n, channel in enumerate(self.fluorescence_channels_pnn):
            control = {'label': channel, 'control_type': 'Channel Assignment', 'particle_type': '', 'gate_channel': channel, 'sample_name': '', 'sample_path': '', 'gate_label': ''}
            spectral_model.append(control)
            self.profile_updater.generate(control, self.spectral_library_search_results)  # pass in search results in case control is from library

        # refresh and update
        self.refresh_table_and_enable() #comboboxes
        self.model.layoutChanged.emit() #table view
        self.bus.spectralModelUpdated.emit() #unmixing matrix etc

    def _on_spectral_control_added_by_autogenerator(self):
        self.model.layoutChanged.emit()
        # control = self.model._data[-1]['label']
        # if control:
        #     if control in self.controller.experiment.process['profiles']:
        #         self.bus.showSelectedProfiles.emit([control])

    def refresh_table_and_enable(self):
        # self.model.layoutChanged.emit() # consider refreshing at this point... but it may cause crash
        self.refresh_comboboxes()
        self.setEnabled(True)
        QTimer.singleShot(300, lambda: self.bus.showSelectedProfiles.emit([]))

    def _show_selected_profiles(self):
        selected_rows = set()
        for index in self.view.selectionModel().selectedRows():
            selected_rows.add(index.row())
        selected_rows = sorted(selected_rows)
        profile_list = [self.model._data[row]['label'] for row in selected_rows if self.model._data[row]['label']]
        self.bus.showSelectedProfiles.emit(profile_list)

    @Slot(int)
    def _on_update_control(self, index, col):
        self._pending_update_index = index
        self._pending_update_col = col # track cosmetic vs functional changes
        self._update_timer.start()  # restarts timer on each rapid edit

    def _do_update_control(self):
        index = self._pending_update_index
        col = self._pending_update_col
        changed_col = COLUMNS[col]
        if index is None:
            return
        if changed_col in ('label', 'antigen'):
            control = self.model._data[index]

            if changed_col in ('label', 'antigen'):
                control = self.model._data[index]

                if changed_col == 'label':
                    attempted_label = control['label']
                    # Check for duplicate labels before mutating anything.
                    all_labels = [c['label'] for c in self.model._data if c['label']]
                    if len(all_labels) != len(set(all_labels)):
                        # Revert by restoring the old label from the profiles dict —
                        # the profiles key for this row hasn't been renamed yet so it
                        # still holds the pre-edit value.
                        old_label = next(
                            (k for k in self.profile_updater.profiles
                            if k not in [c['label'] for c in self.model._data]),
                            None
                        )
                        if old_label is not None:
                            control['label'] = old_label
                        self.model.dataChanged.emit(
                            self.model.index(index, col),
                            self.model.index(index, col),
                            [Qt.DisplayRole, Qt.EditRole]
                        )
                        self.bus.warningMessage.emit(
                            f'Label "{attempted_label}" is already in use. '
                            f'Each control must have a unique label.'
                        )
                        return
            
            # Rename the profiles dict key to match the new label. Include any labels where either label or antigen has been changed
            labels_from_profiles = self.profile_updater.profiles.keys()
            labels_from_spectral_model = [c['label'] for c in self.model._data]
            old_labels = [k for k in labels_from_profiles if k not in labels_from_spectral_model]
            new_labels = [c for c in labels_from_spectral_model if c not in labels_from_profiles]
            for old, new in zip(old_labels, new_labels):
                self.profile_updater.profiles[new] = self.profile_updater.profiles.pop(old)

            # append if label - antigen has been changed f'{antigen} {name}'.strip() if antigen else name
            if self.controller.data_for_cytometry_plots_unmixed.get('pnn_labels'):
                antigen_labels = [(f'{c.get("antigen") or ""} {c["label"]}'.strip() or c['label']) for c in self.model._data]
                full_labels_from_cytometry_plots = list(self.controller.data_for_cytometry_plots_unmixed['pnn_labels'].values())
                changed_antigen_label_indices_in_spectral_model = [n for n, k in enumerate(antigen_labels) if k not in full_labels_from_cytometry_plots]
                for n in changed_antigen_label_indices_in_spectral_model:
                    l = labels_from_spectral_model[n]
                    if l not in new_labels:
                        old_labels.append(l)
                        new_labels.append(l)

            if changed_col in ('label', 'antigen') and old_labels and new_labels:
                # A label rename leaves four downstream structures stale.
                # Propagate atomically without recalculating the unmixing matrix.
                for old, new in zip(old_labels, new_labels):
                    self._propagate_label_rename(old, new)
                # Rebuild ephemeral gating/transform objects from the updated experiment
                # state and refresh the UI. Does not touch the unmixing matrix,
                # spillover, fine-tuning, or AF cache.
                self.controller.initialise_ephemeral_data(scope=['unmixed'])
                if self.bus:
                    self.bus.spectralProcessRefreshed.emit()

            self.bus.showSelectedProfiles.emit([control['label']])
            logger.info(f'SpectralModelEditor: cosmetic edit on row {index}, col "{changed_col}" — skipping regeneration')
            return
        self.setEnabled(False)  # block further interaction during generate
        control = self.model._data[index]
        unused_raw_channels = self.model.unused_raw_channels(control['gate_channel'])
        if control['control_type'] == 'Channel Assignment' and not control['gate_channel'] and unused_raw_channels:
            control['gate_channel'] = unused_raw_channels[0]

        # If the user changed the major channel on an SSC, move the Pos/Neg gates
        # to the new channel before regenerating the profile.
        if (changed_col == 'gate_channel'
                and control.get('control_type') == 'Single Stained Spectral Control'
                and control.get('gate_channel')):
            # Guard against re-entry: only rebuild gates if the channel has
            # actually changed from what the current Pos/Neg gates are on.
            pos_gate_label = control.get('gate_label') or f'Pos {control["label"]}'
            existing_gate_paths = self.controller.raw_gating.find_matching_gate_paths(pos_gate_label)
            if existing_gate_paths:
                existing_gate = self.controller.raw_gating.get_gate(pos_gate_label)
                current_gate_channel = existing_gate.dimensions[0].id
            else:
                current_gate_channel = None
            if control['gate_channel'] != current_gate_channel:
                self._move_pos_neg_gates_to_channel(control)

        sanitise_control_in_place(control)
        self.profile_updater.flush()
        control_valid = self.profile_updater.generate(control, self.spectral_library_search_results)
        self.refresh_comboboxes()
        self.setEnabled(True)  # re-enable after generate completes
        if control_valid:
            self.bus.showSelectedProfiles.emit([control['label']])
            if changed_col == 'gate_channel':
                # Gates were moved to a new channel — update raw gating lookup
                # tables and histograms for just the affected gates, then emit
                # spectralModelUpdated once to recalculate the unmixing matrix.
                pos_gate_label = control.get('gate_label') or f'Pos {control["label"]}'
                self.bus.changedGatingHierarchy.emit('raw', pos_gate_label)
            self.bus.spectralModelUpdated.emit()
        logger.info(f'SpectralModelEditor: updated {"valid" if control_valid else "invalid"} control {control}')

    @with_busy_cursor
    def _move_pos_neg_gates_to_channel(self, control: dict) -> None:
        """
        Destroy the existing Pos/Neg gates for an SSC control and recreate them
        on the user-selected major channel, preserving gate labels so all downstream
        references (gate_label, neg_gate_label, plot child_gates) remain valid.
        """
        from flowkit import Dimension, gates
        import numpy as np
        from honeychrome.controller_components.functions import sample_from_fcs
        from honeychrome.controller_components.spectral_functions import get_raw_events
        import honeychrome.settings as settings

        label = control.get('label', '')
        channel_x = control.get('gate_channel', '')
        if not label or not channel_x:
            return

        pos_gate_label = control.get('gate_label') or f'Pos {label}'
        neg_gate_label = control.get('neg_gate_label') or f'Neg {label}'
        raw_gating = self.controller.raw_gating
        xform = self.controller.raw_gating.transformations.get(channel_x)
        if xform is None:
            logger.warning(f'_move_pos_neg_gates: no transform for "{channel_x}", aborting')
            return

        # Resolve base gate (same priority order as SpectralAutoGenerator)
        base_gate_priority = self.controller.experiment.process.get('base_gate_priority_order', [])
        raw_gate_names = [g[0].lower() for g in raw_gating.get_gate_ids()]
        base_gate_label = 'root'
        for gate in base_gate_priority:
            if gate.lower() in raw_gate_names:
                base_gate_label = gate
                break

        if base_gate_label != 'root' and raw_gating.find_matching_gate_paths(base_gate_label):
            base_path = tuple(list(raw_gating.find_matching_gate_paths(base_gate_label)[0]) + [base_gate_label])
        else:
            base_path = ('root',)

        # Derive gate bounds from the control sample on the new channel
        all_samples_reverse = {v: k for k, v in self.controller.experiment.samples['all_samples'].items()}
        sample_rel_path = all_samples_reverse.get(control.get('sample_name', ''))
        try:
            if sample_rel_path is None:
                logger.warning(f'SpectralModelEditor: _move_pos_neg_gates: sample "{control.get("sample_name")}" not in experiment samples — using fallback bounds')
                raise ValueError("sample not found")  # caught by the existing except block

            sample_full_path = str(self.controller.experiment_dir / sample_rel_path)
            sample = sample_from_fcs(sample_full_path)
            fluor_ids = self.controller.filtered_raw_fluorescence_channel_ids
            event_channels_pnn = self.controller.experiment.settings['raw']['event_channels_pnn']
            ch_global_idx = event_channels_pnn.index(channel_x)
            fl_col = fluor_ids.index(ch_global_idx)

            gate_label_for_load = base_gate_label if base_gate_label != 'root' else None
            fl_events = get_raw_events(sample, fluor_ids,
                                    gate_label=gate_label_for_load,
                                    gating_strategy=raw_gating)
            ch_values = fl_events[:, fl_col]

            pos_pct = settings.spectral_positive_gate_percent_retrieved
            neg_pct = settings.spectral_negative_gate_percent_retrieved
            fl_top    = np.percentile(ch_values, [100 - pos_pct, 100])
            fl_bottom = np.percentile(ch_values, [0, neg_pct])
        except Exception as e:
            logger.warning(f'_move_pos_neg_gates: could not derive bounds for "{label}": {e} — using full-range fallback')
            transformation = self.controller.raw_transformations.get(channel_x)
            if transformation is None:
                return
            lo, hi = transformation.limits
            fl_top    = np.array([lo + 0.6 * (hi - lo), hi])
            fl_bottom = np.array([lo, lo + 0.25 * (hi - lo)])

        # Convert raw-space bounds to transformed space — mirrors the auto-generator exactly
        if xform is not None:
            pos_min = xform.apply(np.array([fl_top[0]]))[0]
            pos_max = xform.apply(np.array([fl_top[1]]))[0]
            neg_min = xform.apply(np.array([fl_bottom[0]]))[0]
            neg_max = xform.apply(np.array([fl_bottom[1]]))[0]
        else:
            pos_min, pos_max = float(fl_top[0]),    float(fl_top[1])
            neg_min, neg_max = float(fl_bottom[0]), float(fl_bottom[1])

        # Remove old gates and recreate on the new channel
        for gate_lbl in (pos_gate_label, neg_gate_label):
            if raw_gating.find_matching_gate_paths(gate_lbl):
                raw_gating.remove_gate(gate_lbl)

        pos_gate = gates.RectangleGate(pos_gate_label,
                        dimensions=[Dimension(channel_x, range_min=pos_min, range_max=pos_max,
                                            transformation_ref=channel_x)])
        neg_gate = gates.RectangleGate(neg_gate_label,
                        dimensions=[Dimension(channel_x, range_min=neg_min, range_max=neg_max,
                                            transformation_ref=channel_x)])
        raw_gating.add_gate(pos_gate, gate_path=base_path)
        raw_gating.add_gate(neg_gate, gate_path=base_path)

        # Update raw plots: remove gate refs from any old-channel plot, add to new-channel plot
        raw_plots = self.controller.experiment.cytometry.get('raw_plots', [])
        for n, plot in enumerate(raw_plots):
            for lbl in (pos_gate_label, neg_gate_label):
                if lbl in plot.get('child_gates', []):
                    plot['child_gates'].remove(lbl)
                    if self.bus:
                        self.bus.updateRois.emit('raw', n)

        target_plot = next(
            (p for p in raw_plots
            if p.get('type') == 'hist1d' and p.get('channel_x') == channel_x),
            None
        )
        if not target_plot:
            target_plot = {
                'type': 'hist1d',
                'channel_x': channel_x,
                'source_gate': base_gate_label,
                'child_gates': []
            }
            raw_plots.append(target_plot)
            if self.bus:
                self.bus.showNewPlot.emit('raw')

        for lbl in (pos_gate_label, neg_gate_label):
            if lbl not in target_plot['child_gates']:
                target_plot['child_gates'].append(lbl)
                n = raw_plots.index(target_plot)
                if self.bus:
                    self.bus.updateRois.emit('raw', n)


        if self.bus:
            self.bus.changedGatingHierarchy.emit('raw', pos_gate_label)
            self.bus.changedGatingHierarchy.emit('raw', neg_gate_label)

        logger.info(f'_move_pos_neg_gates: moved "{pos_gate_label}" and "{neg_gate_label}" to "{channel_x}"')

    def _propagate_label_rename(self, old: str, new: str):
        """
        Rename a fluorescence channel label in all experiment structures that
        reference it by name, without touching the unmixing matrix or any
        computed numerical state.

        Structures updated:
          - experiment.settings['unmixed']['fluorescence_channels']
          - experiment.settings['unmixed']['event_channels_pnn']
          - experiment.cytometry['transforms']  (dict keyed by channel name)
          - experiment.cytometry['gating']      (GML string — replace as text)
        """
        exp = self.controller.experiment

        # 1. fluorescence_channels list
        fl = exp.settings['unmixed'].get('fluorescence_channels', [])
        exp.settings['unmixed']['fluorescence_channels'] = [
            new if ch == old else ch for ch in fl
        ]

        # 2. event_channels_pnn list
        pnn = exp.settings['unmixed'].get('event_channels_pnn', [])
        exp.settings['unmixed']['event_channels_pnn'] = [
            new if ch == old else ch for ch in pnn
        ]

        # 3. cytometry transforms dict (rename the key)
        transforms = exp.cytometry.get('transforms') or {}
        if old in transforms:
            transforms[new] = transforms.pop(old)

        # 4. GML gating string — parse as XML and rename only the fcs-dimension
        #    name attributes that match exactly, avoiding substring corruption
        #    (e.g. renaming "PE" must not touch "PE-Cy7", "PE-CF594", etc.)
        gating_gml = exp.cytometry.get('gating')
        if gating_gml:
            exp.cytometry['gating'] = _rename_channel_in_gml(gating_gml, old, new)

        # 5. cytometry plots list — channel_x and channel_y are stored by name
        for plot in exp.cytometry.get('plots') or []:
            if plot.get('channel_x') == old:
                plot['channel_x'] = new
            if plot.get('channel_y') == old:
                plot['channel_y'] = new

        # 6. Update the live ephemeral dicts immediately so any widget that
        #    re-renders before spectralProcessRefreshed is handled sees
        #    consistent channel names. initialise_ephemeral_data will
        #    overwrite these properly afterwards.
        for data_dict in (self.controller.data_for_cytometry_plots_process,
                          self.controller.data_for_cytometry_plots_unmixed):
            pnn = data_dict.get('pnn')
            if pnn:
                data_dict['pnn'] = [new if ch == old else ch for ch in pnn]
            for plot in data_dict.get('plots') or []:
                if plot.get('channel_x') == old:
                    plot['channel_x'] = new
                if plot.get('channel_y') == old:
                    plot['channel_y'] = new
            transformations = data_dict.get('transformations')
            if transformations and old in transformations:
                transformations[new] = transformations.pop(old)

        logger.info(f'SpectralModelEditor: propagated label rename "{old}" -> "{new}"')

    def _sort_model_by_gate_channel(self):
        """Re-order the spectral model table rows by major channel order
        (position of gate_channel in fluorescence_channels_pnn)."""
        if not self.model._data:
            return
        channel_order = {ch: i for i, ch in enumerate(self.fluorescence_channels_pnn)}
        self.model.beginResetModel()
        self.model._data.sort(
            key=lambda c: channel_order.get(c.get('gate_channel') or '', len(channel_order))
        )
        self.model.endResetModel()
        self.refresh_comboboxes()

    @Slot(str, str)
    def _on_raw_gate_renamed(self, old_name: str, new_name: str):
        """
        A gate in raw gating was renamed by the user via the plot ROI label.
        Update any spectral model control whose gate_label matches the old name,
        so that generate() can still locate the gate without requiring Auto-Generate.
        Also updates raw_plots source_gate references in case the renamed gate is
        used as a plot source.
        """
        changed = False
        for control in self.model._data:
            if control.get('gate_label') == old_name:
                control['gate_label'] = new_name
                changed = True
                logger.info(
                    f'SpectralModelEditor: gate rename propagated to spectral model '
                    f'"{old_name}" -> "{new_name}" for control "{control["label"]}"'
                )
            if control.get('neg_gate_label') == old_name:
                control['neg_gate_label'] = new_name
                changed = True
                logger.info(
                    f'SpectralModelEditor: gate rename propagated to neg_gate_label '
                    f'"{old_name}" -> "{new_name}" for control "{control["label"]}"'
                )

        # Also fix any raw_plots that use the old name as source_gate
        for plot in self.controller.experiment.cytometry.get('raw_plots') or []:
            if plot.get('source_gate') == old_name:
                plot['source_gate'] = new_name
                changed = True

        if changed:
            self.model.dataChanged.emit(
                self.model.index(0, 0),
                self.model.index(self.model.rowCount() - 1, self.model.columnCount() - 1),
            )

    @Slot()
    def _on_force_recalc(self):
        self.setEnabled(False)
        self.profile_updater.flush()  # remove profiles that are not in the model

        for index, control in enumerate(self.model._data):
            label = control.get('label', '')

            # Skip this control if its profile is already present and the gate
            # it references still exists in raw gating (i.e. nothing has changed
            # that would invalidate it).  This avoids unnecessary re-extraction
            # when only one or two controls have changed.
            if label and label in self.profile_updater.profiles:
                # Library controls depend on spectral_library_search_results which is
                # only populated after refresh_comboboxes runs — always regenerate them
                # so the search_results lookup is performed with current data.
                is_library = control.get('control_type') == 'Single Stained Spectral Control from Library'
                gate_label = control.get('gate_label', '')
                gate_still_valid = (
                    not gate_label
                    or bool(self.raw_gating.find_matching_gate_paths(gate_label))
                )
                if gate_still_valid and not is_library:
                    logger.info(f'SpectralModelEditor: skipping recalc for "{label}" — profile present and gate valid')
                    continue

            control_valid = self.profile_updater.generate(control, self.spectral_library_search_results)
            if not control_valid:
                logger.warning(f'SpectralModelEditor: recalc failed for control at index {index}')
                # Do not break — attempt remaining controls so the user sees all
                # failures at once rather than stopping at the first one.

        self.bus.spectralModelUpdated.emit()
        self.refresh_table_and_enable()
        logger.info(f'SpectralModelEditor: forced recalculation complete')

    def _on_unmixing_method_changed(self, method_key: str):
        self.controller.experiment.settings['unmixing_method'] = method_key
        self.controller.experiment.save()
        self.controller.refresh_spectral_process()
        if self.controller.current_sample_path:
            path = self.controller.current_sample_path
            QTimer.singleShot(0, lambda: self.controller.load_sample(path))

    @Slot()
    def _on_clean_controls(self):
        """Launch SpectralCleaner in a QThread. On completion, set use_cleaned=True
        for all successfully cleaned controls and trigger a full recalculate."""
        if not self._has_unstained_cell_sample():
            # Button should already be disabled in this state — this guard only
            # protects against a stale-enabled race. statusMessage only reaches the
            # status bar (see main_window.status_message), never a QMessageBox, so
            # it carries none of the modal-stacking risk warningMessage would.
            if self.bus:
                self.bus.statusMessage.emit(
                    'Clean Controls needs an unstained cell sample as the negative reference — none found.'
                )
            return

        self.setEnabled(False)
        self.clean_controls_btn.setText("Cleaning…")

        self.cleaner_thread = QThread()
        self.spectral_cleaner = SpectralCleaner(self.bus, self.controller)
        self.spectral_cleaner.moveToThread(self.cleaner_thread)

        self.cleaner_thread.started.connect(self.spectral_cleaner.run)
        self.spectral_cleaner.cleaningFinished.connect(self.cleaner_thread.quit)
        self.cleaner_thread.finished.connect(self._on_clean_controls_finished)
        self.cleaner_thread.start()

    @Slot()
    def _on_clean_controls_finished(self):
        """After SpectralCleaner.run() completes: mark controls as use_cleaned=True,
        recalculate profiles, refresh UI."""
        for control in self.model._data:
            label = control.get('label') or ''
            if label in self.controller.cleaned_events:
                # Only set the default if not already explicitly set by the user
                if control.get('use_cleaned') is None:
                    control['use_cleaned'] = True

        self.clean_controls_btn.setText("Clean Controls")

        # Rebuild comboboxes now so that spectral_library_search_results is populated before the recalc worker runs
        self.refresh_comboboxes()

        # Run profile regeneration in a background thread so the main thread
        # stays responsive. _on_force_recalc is not thread-safe (it touches Qt
        # widgets directly), so we do only the pure computation here and defer
        # the UI update to the main thread via a signal.
        self._recalc_thread = QThread()
        self._recalc_worker = _RecalcWorker(self.profile_updater, self.model._data, self.spectral_library_search_results)
        self._recalc_worker.moveToThread(self._recalc_thread)
        self._recalc_thread.started.connect(self._recalc_worker.run)
        self._recalc_worker.finished.connect(self._recalc_thread.quit)
        self._recalc_thread.finished.connect(self._on_recalc_finished)
        self._recalc_thread.start()

    @Slot()
    def _on_recalc_finished(self):
        self.bus.spectralModelUpdated.emit()
        self.refresh_table_and_enable()
        if self.bus:
            self.bus.cleaningResultsReady.emit()
        logger.info('SpectralControlsEditor: Clean Controls run complete.')
        self._warn_low_cosine_controls()

    def _show_row_context_menu(self, position):
        """Right-click menu on a control row: save its spectrum to the Reference
        Library, carrying the antigen / particle type / major channel / source
        sample across from this tab."""
        index = self.view.indexAt(position)
        if not index.isValid():
            return
        row = self.proxy.mapToSource(index).row()
        control = self.model._data[row]
        profile = (self.controller.experiment.process.get('profiles') or {}).get(control.get('label'))

        menu = QMenu(self)
        action = menu.addAction('Save to Reference Library')
        action.setEnabled(bool(profile))
        if not profile:
            action.setToolTip('Generate a profile for this control first')
        action.triggered.connect(lambda: self._save_control_to_reference_library(control, profile))
        menu.exec(self.view.viewport().mapToGlobal(position))

    def _save_control_to_reference_library(self, control, profile):
        """Persist one generated control spectrum as a curated reference profile."""
        label = (control.get('label') or '').strip()
        channels = list(self.fluorescence_channels_pnn)
        if not label or not profile or len(profile) != len(channels):
            QMessageBox.warning(self, 'Save to Reference Library',
                                'This control has no profile matching the current channels.')
            return

        name, ok = QInputDialog.getText(self, 'Save to Reference Library',
                                        'Fluorophore name:', text=label)
        name = (name or '').strip()
        if not ok or not name:
            return
        notes, _ = QInputDialog.getText(self, 'Save to Reference Library', 'Notes (optional):')

        raw = self.controller.experiment.settings['raw']
        cytometer_key = raw.get('cytometer_db_col') or ''
        try:
            library = SpectralReferenceLibrary()
            library.save_profile(
                fluorophore=name,
                profile=dict(zip(channels, [float(v) for v in profile])),
                cytometer_key=cytometer_key,
                config_key=compute_config_key(cytometer_key, channels),
                channel_names=channels,
                antigen=control.get('antigen') or None,
                particle_type=control.get('particle_type') or None,
                gate_channel=control.get('gate_channel') or None,
                source_sample_name=control.get('sample_name') or None,
                source_experiment_dir=str(self.controller.experiment_dir or ''),
                notes=(notes or '').strip() or None,
            )
        except Exception as exc:
            logger.warning(f'Save to Reference Library failed: {exc}')
            QMessageBox.critical(self, 'Save to Reference Library', str(exc))
            return

        message = f'Saved "{name}" to the Reference Library.'
        if not cytometer_key:
            message += ' (Instrument not identified — saved as "unknown".)'
        if self.bus is not None:
            self.bus.statusMessage.emit(message)
        logger.info(f'SpectralControlsEditor: {message}')

    def _warn_low_cosine_controls(self):
        """Show a single warning popup listing controls with cosine similarity
        below COSINE_QC_WARNING_THRESHOLD.  Controls without a reference entry
        are silently skipped."""
        cleaned = self.controller.cleaned_events
        pnn     = self.controller.experiment.settings['raw']['event_channels_pnn']

        low: list[tuple[str, float]] = []
        for label, entry in cleaned.items():
            if not entry.get('spectrum'):
                continue
            cytometer_key = entry.get('cytometer_key')
            if not cytometer_key:
                continue
            fluor_ch_ids = (entry.get('fluor_ch_ids')
                            or self.controller.filtered_raw_fluorescence_channel_ids)
            try:
                fluor_pnn = [pnn[i] for i in fluor_ch_ids]
            except (IndexError, KeyError):
                continue
            import numpy as np
            spectrum = np.array(entry['spectrum'], dtype=float)
            cs = cosine_similarity_to_reference(spectrum, fluor_pnn, label, cytometer_key)
            if cs is None:
                continue  # no reference available — skip
            if cs < COSINE_QC_WARNING_THRESHOLD:
                low.append((label, cs))

        if not low:
            return

        lines = '\n'.join(
            f'  • {label}  (cosine = {cs:.4f})'
            for label, cs in low
        )
        msg = QMessageBox(self)
        msg.setWindowTitle('Cosine QC Warning')
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText(
            f'The following {len(low)} control(s) have a cosine similarity to the '
            f'reference library below {COSINE_QC_WARNING_THRESHOLD}:\n\n'
            f'{lines}\n\n'
            f'Check the Cosine QC plots for details. If the Major Channel is '
            f'incorrect, update it in the Spectral Model table and re-run '
            f'Clean Controls.'
        )
        msg.exec()


    @Slot(list)
    def _on_delete_controls(self, labels):
        for label in labels:
            self.profile_updater.pop_control(label)


if __name__ == "__main__":
    controls = [
        {'label': 'PE-Fire 810', 'control_type': 'Single Stained Spectral Control', 'particle_type': 'Cells', 'gate_channel': 'B6-A',
            'sample_name': 'PE-Fire 810 (Cells)',
            'sample_path': '/home/ssr/spectral_cytometry/20240620 Spectral Symposium-poor cell unmixed/Raw/Cell controls/Reference Group/G1 PE-Fire 810 (Cells)_Cell controls.fcs',
            'gate_label': 'Positive PE-Fire 810'},
        {'label': 'Spark', 'control_type': 'Single Stained Spectral Control', 'particle_type': 'Cells', 'gate_channel': 'B2-A',
            'sample_name': 'Spark (Beads)',
            'sample_path': '/home/ssr/spectral_cytometry/20240620 Spectral Symposium-poor cell unmixed/Raw/Bead controls/Reference Group/Spark (Beads)_Beads controls.fcs',
            'gate_label': 'Positive Spark'},
        {'label': 'FITC', 'control_type': 'Channel Assignment', 'particle_type': None, 'gate_channel': 'B1-A',
            'sample_name': None,
            'sample_path': None,
            'gate_label': None}
    ]

    samples = {'single_stain_controls': ['Raw/Cell controls/Reference Group/A1 Unstained (Cells)_Cell controls.fcs',
                                 'Raw/Cell controls/Reference Group/A10 BUV805 (Cells)_Cell controls.fcs',
                                 'Raw/Cell controls/Reference Group/A11 Super Bright 436 (Cells)_Cell controls.fcs',
                                 'Raw/Cell controls/Reference Group/A12 eFluor 450 (Cells)_Cell controls.fcs',
                                 'Raw/Cell controls/Reference Group/A2 Spark UV 387 (Cells)_Cell controls.fcs',
                                 'Raw/Cell controls/Reference Group/G7 APC-Fire 810 (Cells)_Cell controls.fcs']
    }

    gates = ['root', 'Cells', 'Singlets', 'Positive PE-Fire 810', 'Positive Spark']

    app = QApplication(sys.argv)

    from honeychrome.controller import Controller
    from pathlib import Path
    from event_bus import EventBus

    bus = EventBus()
    kc = Controller()
    kc.bus = bus
    # base_directory = Path.home() / 'spectral_cytometry'
    # experiment_name = base_directory / '20240620 Spectral Symposium-poor cell unmixed'
    # experiment_path = experiment_name.with_suffix('.kit')
    base_directory = Path.home() / 'Experiments'
    experiment_name = base_directory / 'Oleg K BD FACSAria III'
    experiment_path = experiment_name.with_suffix('.kit')
    kc.load_experiment(experiment_path) # note this loads first sample too and runs calculate all histograms and statistics

    frame = SpectralControlsEditor(bus, kc)
    frame.show()
    sys.exit(app.exec())


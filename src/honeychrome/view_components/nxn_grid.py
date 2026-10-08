import numpy as np

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject, QEvent, Slot, QSize, QTimer, QSettings, QItemSelectionModel, QSignalBlocker
from PySide6.QtWidgets import QTableView, QStyledItemDelegate, QFrame, QVBoxLayout, QLabel, QApplication, QComboBox, QHBoxLayout, QStyle, QAbstractItemView
from PySide6.QtGui import QColor, QPalette, QImage, QPixmap, QPen

import colorcet as cc

import honeychrome.settings as settings
from honeychrome.controller_components.functions import define_process_plots, resolve_base_gate
from honeychrome.view_components.help_texts import nxn_help_text
from honeychrome.view_components.help_toggle_widget import HelpToggleWidget

import logging
logger = logging.getLogger(__name__)

# ---------------------------
# Model
# ---------------------------
class HeatmapGridModel(QAbstractTableModel):
    def __init__(self, controller, is_dark):
        super().__init__()
        self._data = []
        self.horizontal_headers = []
        self.vertical_headers = []
        self.controller = controller

        self._pixmap_cache = {}
        self.pnn_labels = {}
        self._pixmap_size = QSize(settings.tile_size_nxn_grid_retrieved, settings.tile_size_nxn_grid_retrieved)

        self.colormap = self.get_colorcet_colormap(settings.colourmap_name_retrieved)

        if is_dark:
            background_colour = QColor(0,0,0,255)
        else:
            background_colour = QColor(255,255,255,255)
        self.colormap[0] = background_colour

        # Create color lookup table
        self.color_table = np.array([[c.red(), c.green(), c.blue(), c.alpha()] for c in self.colormap], dtype=np.uint8).astype(np.uint32)


    def get_colorcet_colormap(self, name):
        """Get a colormap from colorcet and convert to Qt-friendly format"""
        cmap_colors = getattr(cc, name)
        qt_colors = []
        for color in cmap_colors:
            if isinstance(color, str):
                qt_colors.append(QColor(color))
            else:
                r, g, b = [int(c * 255) for c in color]
                qt_colors.append(QColor(r, g, b))
        return qt_colors


    def update_data(self, data, horizontal_headers, vertical_headers, pnn_labels=None):
        self.beginResetModel()  # Notify view that model is about to be reset
        self._data = data
        self._pixmap_cache = {}
        self.horizontal_headers = horizontal_headers
        self.vertical_headers = vertical_headers
        self.pnn_labels = pnn_labels if pnn_labels is not None else {}
        self.endResetModel()  # Notify view that model has been reset

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(self._data[0]) if self._data else 0

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        if role == Qt.DecorationRole:
            r = index.row()
            c = index.column()

            if self.vertical_headers[r] != self.horizontal_headers[c]:
                # return self.get_cached_pixmap(index)
                pixmap = self.create_heatmap_pixmap(self._data[r][c])
                return pixmap
            else:
                return None

        elif role == Qt.ToolTipRole:
            r = index.row()
            c = index.column()

            if r >= len(self.vertical_headers) or c >= len(self.horizontal_headers):
                return None

            return ("Spillover:\n"
                    f"{self.vertical_headers[r]} --- {self.horizontal_headers[c]}: {self.controller.experiment.process['spillover'][r][c]:0.3f}\n"
                    f"{self.horizontal_headers[c]} --- {self.vertical_headers[r]}: {self.controller.experiment.process['spillover'][c][r]:0.3f}\n"
                    )

        return None

    def get_cached_pixmap(self, index):
        cache_key = (index.row(), index.column())
        if cache_key not in self._pixmap_cache:
            heatmap_data = self._data[index.row()][index.column()]
            pixmap = self.create_heatmap_pixmap(heatmap_data)
            self._pixmap_cache[cache_key] = pixmap
        return self._pixmap_cache[cache_key]

    def create_heatmap_pixmap(self, data):
        """Fastest approach using pure NumPy operations"""
        height, width = data.shape

        # Normalize
        data_max = np.max(data)

        if data_max > 0:
            normalized = np.flipud((data / data_max).T)
        else:
            normalized = np.full_like(data, 0.5)

        indices = (normalized * (len(self.colormap) - 1)).astype(np.int32)
        indices = np.clip(indices, 0, len(self.colormap) - 1) # not necessary to clip as already scaled # ssr putting this back in... otherwise we get crashes

        # Vectorized lookup
        rgb_array = self.color_table[indices]

        # Convert to ARGB32 format expected by QImage
        argb_array = (rgb_array[:, :, 3] << 24) | (rgb_array[:, :, 0] << 16) | (rgb_array[:, :, 1] << 8) | (rgb_array[:, :, 2])

        scaled_image = QImage(argb_array.data, width, height, QImage.Format_ARGB32).scaled(self._pixmap_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # Create QImage from memory
        return QPixmap.fromImage(scaled_image)

    def setData(self, index, value, role):
        if role == Qt.EditRole:
            try:
                v = float(value)
            except ValueError:
                return False

            self._data[index.row(), index.column()] = v
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.BackgroundRole])
            return True
        return False

    def flags(self, index):
        r, c = index.row(), index.column()

        if r == c:  # hide & disable
            return Qt.ItemIsEnabled

        return Qt.ItemIsSelectable | Qt.ItemIsEditable | Qt.ItemIsEnabled

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.ToolTipRole:
            return self.horizontal_headers[section]

        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                ch = self.horizontal_headers[section]
                return self.pnn_labels.get(ch, ch)
            elif orientation == Qt.Vertical:
                ch = self.vertical_headers[section]
                return self.pnn_labels.get(ch, ch)
        return None


# ---------------------------
# Delegate: select text on editing
# ---------------------------
class HeatmapDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)

    def paint(self, painter, option, index):
        # Hide diagonal: paint background color of table with no text
        if not index.data(Qt.DecorationRole):
            painter.fillRect(option.rect, option.palette.window())  # blank area
            return  # skip default painting

        # 1. Clear the "Selected" state so the default renderer
        # doesn't paint the blue background fill.
        custom_option = option
        is_selected = option.state & QStyle.State_Selected
        if is_selected:
            custom_option.state &= ~QStyle.State_Selected

        # 2. Draw the standard cell content (text, etc.)
        # Normal painting for all other cells
        super().paint(painter, option, index)

        # 3. Manually draw the border if the cell is selected
        if is_selected:
            painter.save()

            # Setup the pen (color and thickness)
            pen = QPen(QColor("#3498db"), 2)  # Modern Blue
            painter.setPen(pen)

            # Adjust the rect slightly so the border isn't clipped
            rect = option.rect.adjusted(1, 1, -1, -1)
            painter.drawRect(rect)

            painter.restore()

class ResizingTable(QTableView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # No scrollbars
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # fixed row height
        # Equal column widths
        self.horizontalHeader().setDefaultSectionSize(settings.tile_size_nxn_grid_retrieved)
        self.verticalHeader().setDefaultSectionSize(settings.tile_size_nxn_grid_retrieved)
        self.setWordWrap(True)

        #single selection
        # 1. Allow only one item to be selected at a time
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        # 2. Ensure selection happens at the cell level (not the whole row)
        self.setSelectionBehavior(QAbstractItemView.SelectItems)

    def setModel(self, model):
        super().setModel(model)
        # Connect to model's dataChanged signal
        if model:
            model.layoutChanged.connect(self._on_model_changed)
            model.modelReset.connect(self._on_model_changed)

    def _on_model_changed(self):
        self.updateGeometry()

    def sizeHint(self):
        if not self.model() or self.model().rowCount() == 0 or self.model().columnCount() == 0:
            return super().sizeHint()

        # Get header sizes
        horizontal_header = self.horizontalHeader()
        vertical_header = self.verticalHeader()

        # Calculate total width: vertical header + all columns + frame
        width = (vertical_header.width() if vertical_header.isVisible() else 0)
        width += horizontal_header.length()

        # Calculate total height: horizontal header + all rows + frame
        height = (horizontal_header.height() if horizontal_header.isVisible() else 0)
        height += vertical_header.length()

        # Add frame width (usually 1px per side)
        frame_width = self.frameWidth() * 2

        # Check if horizontal scrollbar is visible and add its height
        horizontal_scrollbar = self.horizontalScrollBar()
        if horizontal_scrollbar and horizontal_scrollbar.isVisible():
            height += horizontal_scrollbar.height()

        # Check if vertical scrollbar is visible and add its width
        vertical_scrollbar = self.verticalScrollBar()
        if vertical_scrollbar and vertical_scrollbar.isVisible():
            width += vertical_scrollbar.width()

        return QSize(width + frame_width, height + frame_width)

# -----------------------------------------------------
# Main Application
# -----------------------------------------------------

class NxNGrid(QFrame):
    def __init__(self, bus, controller, is_dark=False, parent=None):
        super().__init__(parent)

        # connect
        self.bus = bus
        self.controller = controller
        self.heatmaps = None
        self.horizontal_headers = []
        self.vertical_headers = []

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.view = ResizingTable()
        self.title = QLabel('NxN Plots')
        self.help_nxn = HelpToggleWidget(text=nxn_help_text)

        self.layout.addWidget(self.title)
        self.title.setStyleSheet(settings.heading_style)
        self.layout.addWidget(self.help_nxn)
        source_gate_combo_layout = QHBoxLayout()
        self.source_gate_combo = QComboBox(self)
        self.source_gate_combo.addItem("root")  # placeholder for "no selection"
        source_gate_combo_layout.addWidget(self.source_gate_combo)
        source_gate_combo_layout.addStretch()
        self.layout.addLayout(source_gate_combo_layout)

        self.model = HeatmapGridModel(self.controller, is_dark)
        self.view.setModel(self.model)
        delegate = HeatmapDelegate()
        self.view.setItemDelegate(delegate)

        self.layout.addWidget(self.view)

        # Spillover and wheel editor
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._process_spillover_change)
        self.view.viewport().installEventFilter(self)
        self.view.selectionModel().currentChanged.connect(self.selected_cell_changed)

        if self.bus is not None: # nxn grid is in the gui - connect signals and initialise in the normal way
            self.source_gate_combo.currentTextChanged.connect(self.request_update_process_plots)
            self.bus.showSelectedProfiles.connect(self.show_selected_rows)
            self.bus.histsStatsRecalculated.connect(self.refresh_heatmaps)
            self.bus.spectralProcessRefreshed.connect(self.refresh_source_combo)
            self.bus.changedGatingHierarchy.connect(self.refresh_source_combo)
            self.bus.spilloverSelectedCellChanged.connect(self.set_selected_cell)

        else: # nxn grid is in the exporter - just update the plots, histograms and generate the model and view
            # refresh list of plots with preferred source gate
            source_gate, _ = resolve_base_gate(self.controller.unmixed_gating, self.controller.experiment.process['base_gate_priority_order'])
            logger.info(f'NxN Grid: using {source_gate} as base gate for process NxN plots')
            process_plots = define_process_plots(self.controller.experiment.settings['unmixed']['fluorescence_channels'], self.controller.experiment.settings['unmixed']['fluorescence_channels'], source_gate=source_gate)
            self.controller.data_for_cytometry_plots_process.update({'plots': process_plots})

            # calculate histograms
            self.controller.initialise_data_for_cytometry_plots(force_recalc_histograms=True) # make sure all histograms are present before constructing grid
            self.set_headers_to_all_labels()
            self.refresh_heatmaps() #produces dummy hists



    def initialise(self):
        if self.controller.data_for_cytometry_plots_process['plots']:
            self.setVisible(True)
            self.set_headers_to_all_labels()
            self.refresh_source_combo(mode='unmixed') # populate the source combo with all unmixed gates
            self.refresh_heatmaps() #produces dummy hists
        else:
            self.setVisible(False)

    def set_headers_to_all_labels(self):
        # initialise headers - they will be filtered later
        pnn = self.controller.data_for_cytometry_plots_process['pnn']
        if pnn is not None:
            fl_ids = self.controller.data_for_cytometry_plots_process['fluoro_indices']
            fl_pnn = [pnn[n] for n in fl_ids]
            self.horizontal_headers = fl_pnn
            self.vertical_headers = fl_pnn

    def show_context_menu(self, event):
        # Empty method to completely disable context menu
        pass

    @Slot(str, str)
    def refresh_source_combo(self, mode='unmixed', gate=None): #gate not used, just for connecting signal
        # called on initialisation or update of spectral process or unmixed gating hierarchy
        if mode == 'unmixed' or mode == 'process':
            if self.controller.data_for_cytometry_plots_process['gating'] and self.controller.data_for_cytometry_plots_process['plots']:
                gate_names = ['root'] + [g[0] for g in self.controller.data_for_cytometry_plots_process['gating'].get_gate_ids()]
                current_source_combo_items = [self.source_gate_combo.itemText(i) for i in range(self.source_gate_combo.count())]
                if set(current_source_combo_items) != set(gate_names):
                    self.source_gate_combo.clear()
                    self.source_gate_combo.addItems(gate_names)
                source_gate = self.controller.data_for_cytometry_plots_process['plots'][0]['source_gate']
                index = self.source_gate_combo.findText(source_gate)
                if index >= 0:
                    self.source_gate_combo.blockSignals(True)
                    self.source_gate_combo.setCurrentIndex(index)
                    self.source_gate_combo.blockSignals(False)

    def _process_spillover_change(self):
        # recalculate histograms only for source gate and plots in selected rows
        source_gate = self.source_gate_combo.currentText()
        self.controller.reapply_fine_tuning()
        self.request_update_process_plots(source_gate)

    @Slot(list)
    def show_selected_rows(self, selected_label_list):
        # whenever profiles selection is changed, set up new vertical and horizontal headers
        # request update of process plots since plots and hists may not be available
        old_vertical_headers = self.vertical_headers.copy()
        if selected_label_list:
            self.vertical_headers = selected_label_list
        else:
            self.set_headers_to_all_labels()

        # only emit if selection changed
        if self.vertical_headers != old_vertical_headers:
            source_gate = self.source_gate_combo.currentText()
            self.request_update_process_plots(source_gate)

    @Slot(str)
    def request_update_process_plots(self, source_gate):
        # runs if source combo selection is changed, label selection changed or spillover changed,
        # unless spectral model has changed (in that case wait for reinitialisation)
        # redefines process plots and forces reinitialisation of data,
        # ultimately signalling histstatsrecalculated
        if self.controller.experiment.settings['unmixed']['fluorescence_channels']:
            if set(self.controller.experiment.process['profiles'].keys()) == set(self.horizontal_headers):
                process_plots = define_process_plots(self.horizontal_headers, self.vertical_headers, source_gate=source_gate)
                self.controller.data_for_cytometry_plots_process.update({'plots': process_plots})
                if self.controller.current_mode == 'process':
                    if source_gate:
                        self.bus.requestUpdateProcessHists.emit()

    @Slot(str)
    def refresh_heatmaps(self, mode='process'):
        # called when nxn grid initialised, when experiment loaded, and on histstatsrecalculated
        # triggers update of model and view
        # access data_for_cytometry_plots_process instead of data_for_cytometry_plots since mode may be different (probably on raw) when initialised
        try:

            if mode == 'process':
                if self.horizontal_headers:
                    plots = self.controller.data_for_cytometry_plots_process['plots']
                    if plots:
                        self.setVisible(True)
                        histograms = self.controller.data_for_cytometry_plots_process['histograms']
                        if not histograms:
                            dummy_hist = np.zeros((settings.tile_size_nxn_grid_retrieved, settings.tile_size_nxn_grid_retrieved))
                            histograms += [dummy_hist for plot in plots]

                        # build data arrays
                        # vertical headers are the unmixed filtered set
                        # horizontal headers are the raw fluorescence set
                        self.heatmaps = []
                        for r in self.vertical_headers:
                            row = []
                            for c in self.horizontal_headers:
                                if c != r:
                                    index = [plots.index(plot) for plot in plots if plot['type'] == 'hist2d' and plot['channel_x'] == c and plot['channel_y'] == r][0]
                                    row.append(histograms[index])
                                else:
                                    row.append(None)
                            self.heatmaps.append(row)

                        # self.view.setModel(self.model)

                        # if there is a current selection, look it up and set it after the model is reset
                        index = self.view.selectionModel().currentIndex()
                        if index.isValid() and self.vertical_headers and self.horizontal_headers:
                            selected_row_chan = self.vertical_headers[index.row()]
                            selected_col_chan = self.horizontal_headers[index.column()]

                        pnn_labels = self.controller.data_for_cytometry_plots_process.get('pnn_labels') or {}
                        self.model.update_data(self.heatmaps, self.horizontal_headers, self.vertical_headers, pnn_labels=pnn_labels)

                        if index.isValid():
                            QTimer.singleShot(0, lambda : self.set_selected_cell(selected_row_chan, selected_col_chan))

                    else:
                        self.setVisible(False)
                else:
                    self.setVisible(False)
        except Exception as e:
            logger.warning(f'NxN Grid: refresh_heatmaps failed ({e}), skipping update.')

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            pos = event.position().toPoint()
            index = self.view.indexAt(pos)

            if not index.isValid():
                return False

            # note that row number is not the same as the index of the fluorescence pnn if profiles have been selected. convert to the right row rr
            r = index.row()
            if not self.vertical_headers or r >= len(self.vertical_headers):
                return False

            # guard against uncalculated profiles
            v_label = self.vertical_headers[r]
            if v_label not in self.horizontal_headers:
                return False
            rr = self.horizontal_headers.index(v_label)
            c = index.column()
            # ignore diagonal cells
            if rr == c:
                return super().eventFilter(obj, event)

            # ignore if not selected
            selection_model = self.view.selectionModel()
            if not selection_model.isSelected(index):
                return super().eventFilter(obj, event)

            old = self.controller.experiment.process['spillover'][rr][c]
            step = -settings.wheel_speed if event.angleDelta().y() > 0 else settings.wheel_speed
            new_value = float(old) + step
            self.controller.experiment.process['spillover'][rr][c] = new_value
            self._timer.start()

            return True  # consume wheel event

        return super().eventFilter(self.view.viewport(), event)

    @Slot(QModelIndex, QModelIndex)
    def selected_cell_changed(self, current, previous):
        if current.isValid():
            if current.row() >= len(self.vertical_headers) or current.column() >= len(self.horizontal_headers):
                return

            row_chan = self.vertical_headers[current.row()]
            col_chan = self.horizontal_headers[current.column()]
            self.bus.spilloverSelectedCellChanged.emit(row_chan, col_chan)

    @Slot(str, str)
    def set_selected_cell(self, row_chan, col_chan):#
        if row_chan in self.vertical_headers and col_chan in self.horizontal_headers:
            self.view.selectionModel().setCurrentIndex(
                self.model.index(self.vertical_headers.index(row_chan),
                                 self.horizontal_headers.index(col_chan)
                                 ), QItemSelectionModel.ClearAndSelect)
        else:
            with QSignalBlocker(self.view.selectionModel()):
                self.view.selectionModel().clearSelection()
                self.view.selectionModel().clearCurrentIndex()  # Optional: also clears the focus anchor

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)

    from honeychrome.controller import Controller
    from pathlib import Path
    from event_bus import EventBus

    bus = EventBus()
    kc = Controller()
    base_directory = Path.home() / 'spectral_cytometry'
    experiment_name = base_directory / '20240620 Spectral Symposium-poor cell unmixed'
    experiment_path = experiment_name.with_suffix('.kit')
    kc.load_experiment(experiment_path)
    kc.set_mode('Spectral Process')
    kc.load_sample(kc.experiment.samples['single_stain_controls'][0])

    frame = NxNGrid(bus, kc)

    # Plot the profiles
    frame.refresh_heatmaps()
    frame.show()

    sys.exit(app.exec())


import sys
import numpy as np

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject, QEvent, Slot, QSize, QTimer, QItemSelectionModel, QSignalBlocker, QSettings, QRect
from PySide6.QtWidgets import QApplication, QTableView, QStyledItemDelegate, QLineEdit, QFrame, QVBoxLayout, QHBoxLayout, QLabel, QHeaderView, QStyle, QAbstractItemView, QWidget, QPushButton, QCheckBox, QSpinBox, QStyleOptionHeader
from PySide6.QtGui import QColor, QPen, QPalette

import pyqtgraph as pg
import colorcet as cc

from honeychrome.settings import heading_style, wheel_speed, heatmap_colourmap_name

# ---------------------------
# Model
# ---------------------------
class HeatmapModel(QAbstractTableModel):
    def __init__(self, is_dark, heatmap_range):
        super().__init__()
        self._data = None
        self.horizontal_headers = None
        self.vertical_headers = None
        self.heatmap_range = heatmap_range

        q = QSettings("honeychrome", "app_configuration")
        palette = str(q.value("heatmap_colourmap", heatmap_colourmap_name))
        colors = cc.palette[palette]
        self.cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), color=colors)

    def value_to_cet_color(self, value):
        vmin, vmax = self.heatmap_range
        t = (value - vmin) / (vmax - vmin)
        r, g, b, _ = self.cmap.map(np.array([t]))[0]
        return QColor(int(r), int(g), int(b))

    def update_data(self, data, horizontal_headers, vertical_headers):
        self.beginResetModel()  # Notify view that model is about to be reset
        self._data = np.array(data)
        self.horizontal_headers = horizontal_headers
        self.vertical_headers = vertical_headers
        self.endResetModel()  # Notify view that model has been reset

    def rowCount(self, parent=None):
        if self._data is not None:
            return self._data.shape[0]
        else:
            return 0

    def columnCount(self, parent=None):
        if self._data is not None:
            return self._data.shape[1]
        else:
            return 0

    def data(self, index, role):
        if not index.isValid():
            return None

        value = self._data[index.row(), index.column()]

        if role == Qt.DisplayRole:
            return f"{value:.3f}"

        if role == Qt.EditRole:
            return value

        if role == Qt.BackgroundRole:
            return self.value_to_cet_color(value)

        if role == Qt.ToolTipRole:
            return (f"{self.vertical_headers[index.row()]} / "
                    f"{self.horizontal_headers[index.column()]}: {value:.4f}")

        return None

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
        """Channel label of a row or column. None for a section outside the
        labels: while a reset that changes the matrix size is propagating,
        a header can still ask for sections of the previous size."""
        if role not in (Qt.DisplayRole, Qt.ToolTipRole):
            return None
        labels = self.horizontal_headers if orientation == Qt.Horizontal else self.vertical_headers
        if labels is None or not 0 <= section < len(labels):
            return None
        return labels[section]


# ---------------------------
# Delegate: select text on editing
# ---------------------------
class HeatmapDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, enable_editor=False, disable_diagonal=False):
        super().__init__(parent)
        self.enable_editor = enable_editor
        self.disable_diagonal = disable_diagonal
        # row and column under the mouse, outlined as a crosshair (-1 = none)
        self.hover_row = -1
        self.hover_col = -1

    def createEditor(self, parent, option, index):
        if not self.enable_editor:
            return super().createEditor(parent, option, index)

        editor = QLineEdit(parent)
        return editor

    def setEditorData(self, editor, index):
        if self.enable_editor:
            val = index.model().data(index, Qt.EditRole)
            # editor.setText(str(val))
            editor.setText(f"{val:.3f}")
            editor.selectAll()
        else:
            super().setEditorData(editor, index)

    def initStyleOption(self, option, index):
        # Show colour only when the number would not fit in the cell; the value
        # remains available as the cell tooltip.
        super().initStyleOption(option, index)
        if option.text and option.fontMetrics.horizontalAdvance(option.text) > option.rect.width() - 4:
            option.text = ''

    def paint(self, painter, option, index):
        if self.disable_diagonal:
            r, c = index.row(), index.column()

            # Hide diagonal: paint background color of table with no text
            if r == c:
                painter.fillRect(option.rect, option.palette.window())  # blank area
                self._paint_crosshair(painter, option, index)
                return  # skip default painting

        # 1. Clear the "Selected" state so the default renderer
        # doesn't paint the blue background fill.
        custom_option = option
        is_selected = option.state & QStyle.State_Selected
        if is_selected:
            custom_option.state &= ~QStyle.State_Selected

        # 2. Draw the standard cell content (text, etc.)
        super().paint(painter, custom_option, index)

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

        self._paint_crosshair(painter, option, index)

    def _paint_crosshair(self, painter, option, index):
        """Outline the cells in the hovered row and column so a value can be traced back to both labels."""
        in_row = index.row() == self.hover_row
        in_col = index.column() == self.hover_col
        if not (in_row or in_col):
            return
        painter.save()
        painter.setPen(QPen(option.palette.color(QPalette.Text), 1))
        rect = option.rect.adjusted(0, 0, -1, -1)
        if in_row:
            painter.drawLine(rect.topLeft(), rect.topRight())
            painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        if in_col:
            painter.drawLine(rect.topLeft(), rect.bottomLeft())
            painter.drawLine(rect.topRight(), rect.bottomRight())
        painter.restore()


# ---------------------------
# Headers: hover highlight and rotated column labels
# ---------------------------
class HighlightHeaderView(QHeaderView):
    """Header that shades the section matching the hovered row or column."""
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setSectionsClickable(True)
        self.setHighlightSections(True)
        self._highlight = -1

    def set_highlight(self, logical_index):
        if logical_index != self._highlight:
            self._highlight = logical_index
            self.viewport().update()

    def paintSection(self, painter, rect, logical_index):
        painter.save()
        super().paintSection(painter, rect, logical_index)
        painter.restore()
        self._paint_highlight(painter, rect, logical_index)

    def _paint_highlight(self, painter, rect, logical_index):
        if logical_index == self._highlight:
            color = QColor(self.palette().color(QPalette.Highlight))
            color.setAlpha(90)
            painter.fillRect(rect, color)


class RotatedHeaderView(HighlightHeaderView):
    """Column header with labels written bottom-to-top, so columns can be as narrow as the
    text height instead of the label length. Height follows the longest label, capped."""
    PADDING = 4
    MAX_HEIGHT = 220

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)

    def _label(self, logical_index):
        model = self.model()
        text = model.headerData(logical_index, Qt.Horizontal, Qt.DisplayRole) if model else None
        return '' if text is None else str(text)

    def sizeHint(self):
        hint = super().sizeHint()
        metrics = self.fontMetrics()
        longest = max((metrics.horizontalAdvance(self._label(i)) for i in range(self.count())), default=0)
        hint.setHeight(min(self.MAX_HEIGHT, longest + 3 * self.PADDING))
        return hint

    def paintSection(self, painter, rect, logical_index):
        if not rect.isValid():
            return
        painter.save()
        option = QStyleOptionHeader()
        self.initStyleOption(option)
        option.rect = rect
        option.section = logical_index
        option.text = ''
        self.style().drawControl(QStyle.CE_Header, option, painter, self)

        length = rect.height() - 2 * self.PADDING
        text = self.fontMetrics().elidedText(self._label(logical_index), Qt.ElideRight, length)
        painter.setPen(self.palette().color(QPalette.ButtonText))
        painter.translate(rect.left(), rect.bottom() - self.PADDING)
        painter.rotate(-90)
        painter.drawText(QRect(0, 0, length, rect.width()), Qt.AlignLeft | Qt.AlignVCenter, text)
        painter.restore()
        self._paint_highlight(painter, rect, logical_index)


def install_readable_headers(view):
    """Give a heatmap table rotated column labels and hover-highlighting row labels,
    keeping the current cell size."""
    size = view.horizontalHeader().defaultSectionSize()
    view.setHorizontalHeader(RotatedHeaderView(view))
    view.setVerticalHeader(HighlightHeaderView(Qt.Vertical, view))
    for header in (view.horizontalHeader(), view.verticalHeader()):
        header.setDefaultSectionSize(size)


class CrosshairTracker(QObject):
    """Follow the mouse over a heatmap table and highlight the row and column under it,
    in the cells (via the delegate) and in both headers."""
    def __init__(self, view):
        super().__init__(view)
        self.view = view
        view.setMouseTracking(True)
        view.viewport().installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseMove:
            index = self.view.indexAt(event.position().toPoint())
            if index.isValid():
                self._set(index.row(), index.column())
            else:
                self._set(-1, -1)
        elif event.type() == QEvent.Leave:
            self._set(-1, -1)
        return False

    def _set(self, row, col):
        delegate = self.view.itemDelegate()
        if (delegate.hover_row, delegate.hover_col) == (row, col):
            return
        delegate.hover_row, delegate.hover_col = row, col
        self.view.viewport().update()
        if isinstance(self.view.horizontalHeader(), HighlightHeaderView):
            self.view.horizontalHeader().set_highlight(col)
        if isinstance(self.view.verticalHeader(), HighlightHeaderView):
            self.view.verticalHeader().set_highlight(row)

# ---------------------------
# Wheel handler (event filter)
# ---------------------------
class WheelEditor(QObject):
    def __init__(self, view, model):
        super().__init__()
        self.view = view
        self.model = model

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            pos = event.position().toPoint()
            index = self.view.indexAt(pos)

            if not index.isValid():
                return False

            # ignore diagonal cells
            if index.row() == index.column():
                return super().eventFilter(obj, event)

            old = self.model.data(index, Qt.EditRole)
            if old is None:
                return super().eventFilter(obj, event)

            # ignore if not selected
            selection_model = self.view.selectionModel()
            if not selection_model.isSelected(index):
                return super().eventFilter(obj, event)

            step = wheel_speed if event.angleDelta().y() > 0 else -wheel_speed
            new_value = float(old) + step

            self.model.setData(index, new_value, Qt.EditRole)
            return True  # consume wheel event

        return False



class ResizingTable(QTableView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # No scrollbars
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.horizontalHeader().setDefaultSectionSize(60)
        self.verticalHeader().setDefaultSectionSize(60)
        self.setWordWrap(True)

    def set_cell_size(self, size):
        self.horizontalHeader().setDefaultSectionSize(size)
        self.verticalHeader().setDefaultSectionSize(size)
        self.updateGeometry()

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

class HeatmapPopoutWindow(QWidget):
    """Separate, resizable window showing one Spectral Process heatmap.

    The table shares the model and the selection with the docked heatmap, so
    values, spillover edits and the selected cell stay in sync between the two.
    With "Fit to window" ticked the cells are resized to fill the window, so the
    whole matrix can be seen at once (for example maximised on a second screen);
    unticked, the cell size is set manually and the table scrolls with its
    channel labels kept in view. Numbers are hidden when a cell is too small to
    show them, and every cell's value is available as a tooltip.
    """
    MIN_CELL = 16
    MAX_CELL = 160

    def __init__(self, editor, title):
        super().__init__(editor, Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(title)
        self.editor = editor

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.fit_check = QCheckBox('Fit to window')
        self.fit_check.setChecked(True)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(self.MIN_CELL, self.MAX_CELL)
        self.size_spin.setSingleStep(4)
        self.size_spin.setSuffix(' px')
        self.size_spin.setValue(editor.view.horizontalHeader().defaultSectionSize())
        self.size_spin.setEnabled(False)
        controls.addWidget(self.fit_check)
        controls.addWidget(QLabel('Cell size:'))
        controls.addWidget(self.size_spin)
        controls.addStretch()
        layout.addLayout(controls)

        self.view = QTableView()
        self.view.setModel(editor.model)
        self.view.setSelectionModel(editor.view.selectionModel())
        self.view.setItemDelegate(HeatmapDelegate(self.view, **editor.delegate_kwargs))
        self.view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.view.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.view.setEditTriggers(editor.view.editTriggers())
        self.view.setWordWrap(True)
        install_readable_headers(self.view)
        for header in (self.view.horizontalHeader(), self.view.verticalHeader()):
            header.setMinimumSectionSize(self.MIN_CELL)
        self.crosshair = CrosshairTracker(self.view)
        layout.addWidget(self.view)

        if editor.process_key == 'spillover':
            self.wheel_handler = WheelEditor(self.view, editor.model)
            self.view.viewport().installEventFilter(self.wheel_handler)

        self.sync_hidden_rows()
        self._set_cell_size(self.size_spin.value())

        self.fit_check.toggled.connect(self._on_fit_toggled)
        self.size_spin.valueChanged.connect(self._set_cell_size)
        editor.model.modelReset.connect(self._refit)
        self.view.installEventFilter(self)

        screen = editor.screen().availableGeometry()
        self.resize(int(screen.width() * 0.8), int(screen.height() * 0.8))

    def sync_hidden_rows(self):
        """Show the same rows as the docked heatmap (profiles selected above)."""
        for row in range(self.editor.model.rowCount()):
            self.view.setRowHidden(row, self.editor.view.isRowHidden(row))
        self._refit()

    def eventFilter(self, obj, event):
        if obj is self.view and event.type() == QEvent.Resize:
            self._refit()
        return False

    def _on_fit_toggled(self, checked):
        self.size_spin.setEnabled(not checked)
        self._refit()

    def _refit(self):
        if self.fit_check.isChecked():
            self.fit_to_window()

    def fit_to_window(self):
        model = self.view.model()
        n_cols = model.columnCount()
        n_rows = sum(not self.view.isRowHidden(row) for row in range(model.rowCount()))
        if n_cols == 0 or n_rows == 0:
            return
        frame = 2 * self.view.frameWidth()
        avail_w = self.view.width() - self.view.verticalHeader().width() - frame
        avail_h = self.view.height() - self.view.horizontalHeader().height() - frame
        size = int(min(avail_w / n_cols, avail_h / n_rows))
        size = max(self.MIN_CELL, min(self.MAX_CELL, size))
        with QSignalBlocker(self.size_spin):
            self.size_spin.setValue(size)
        self._set_cell_size(size)

    def _set_cell_size(self, size):
        self.view.horizontalHeader().setDefaultSectionSize(size)
        self.view.verticalHeader().setDefaultSectionSize(size)


# -----------------------------------------------------
# Main Application
# -----------------------------------------------------

class HeatmapViewEditor(QFrame):
    def __init__(self, bus, controller, process_key, is_dark, parent=None):
        super().__init__(parent)

        # connect
        self.bus = bus
        self.controller = controller
        self.process_key = process_key
        self.matrix = None
        if self.bus:
            self.bus.showSelectedProfiles.connect(self.show_selected_rows)

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._emit_now)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.view = ResizingTable()
        self.title = None
        self._popout = None
        self.delegate_kwargs = {}
        if process_key == 'similarity_matrix':
            self.short_title = 'Similarity Matrix'
            self.title = QLabel(self.short_title)
            self.delegate_kwargs = dict(disable_diagonal=True)
            heatmap_range = (-1,1)
        elif process_key == 'hotspot_matrix':
            self.short_title = 'Hotspot Matrix'
            self.title = QLabel(self.short_title)
            self.delegate_kwargs = dict(disable_diagonal=True)
            heatmap_range = (-10,10)
        elif process_key == 'unmixing_matrix':
            self.short_title = 'Unmixing Matrix'
            self.title = QLabel(self.short_title)
            heatmap_range = (-1,1)
        elif self.process_key == 'spillover':
            self.short_title = 'Spillover (Fine Tuning) Matrix'
            self.title = QLabel('Spillover (Fine Tuning) Matrix: double click to edit, or click to select then roll scroll wheel')
            self.delegate_kwargs = dict(enable_editor=True, disable_diagonal=True)
            heatmap_range = (-0.5,0.5)
        delegate = HeatmapDelegate(**self.delegate_kwargs)

        if self.bus:
            title_row = QHBoxLayout()
            title_row.addWidget(self.title)
            title_row.addStretch()
            popout_button = QPushButton('⤢ Pop Out')
            popout_button.setToolTip('Open this matrix in a separate, resizable window')
            popout_button.clicked.connect(self.open_popout)
            title_row.addWidget(popout_button)
            self.layout.addLayout(title_row)
        else:
            self.layout.addWidget(self.title)
        self.title.setStyleSheet(heading_style)
        self.model = HeatmapModel(is_dark, heatmap_range)
        self.view.setModel(self.model)
        self.view.setItemDelegate(delegate)

        # Wheel editor
        if process_key == 'spillover':
            self.wheel_handler = WheelEditor(self.view, self.model)
            self.view.viewport().installEventFilter(self.wheel_handler)
            self.model.dataChanged.connect(self._on_edit)
            self.view.selectionModel().currentChanged.connect(self.selected_cell_changed)
            if self.bus:
                self.bus.requestUpdateProcessHists.connect(self.refresh_heatmap)
                self.bus.spilloverSelectedCellChanged.connect(self.set_selected_cell)


        self.layout.addWidget(self.view)

        # fit-to-width state, switched on from the Spectral Process tab
        self.fit_width = False
        self.fit_min_cell = 16
        self.fit_max_cell = 120
        self.model.modelReset.connect(lambda: QTimer.singleShot(0, self._refit_width))
        if self.bus:
            install_readable_headers(self.view)
            for header in (self.view.horizontalHeader(), self.view.verticalHeader()):
                header.setMinimumSectionSize(self.fit_min_cell)
            self.crosshair = CrosshairTracker(self.view)

        self.refresh_heatmap()

    def set_fit_width(self, enabled):
        """Size cells so all columns fit the width of the panel, within fit_min_cell to fit_max_cell pixels."""
        self.fit_width = enabled
        self._refit_width()

    def _refit_width(self):
        if not self.fit_width:
            return
        n_cols = self.model.columnCount()
        if n_cols == 0:
            return
        available = self.width() - self.view.verticalHeader().width() - 2 * self.view.frameWidth()
        size = max(self.fit_min_cell, min(self.fit_max_cell, available // n_cols))
        if size != self.view.horizontalHeader().defaultSectionSize():
            self.view.set_cell_size(size)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refit_width()

    @Slot(list)
    def show_selected_rows(self, selected_label_list):
        if self.model.vertical_headers:
            if selected_label_list:
                pass
            else:
                selected_label_list = self.model.vertical_headers

            for row, label in enumerate(self.model.vertical_headers):
                visible = label in selected_label_list
                self.view.setRowHidden(row, not visible)
            self.view.updateGeometry()
            if self._popout is not None:
                self._popout.sync_hidden_rows()

    def open_popout(self):
        """Show this matrix in its own window, or raise the window if already open."""
        if self._popout is None:
            self._popout = HeatmapPopoutWindow(self, self.short_title)
            self._popout.destroyed.connect(self._on_popout_destroyed)
        self._popout.show()
        self._popout.raise_()
        self._popout.activateWindow()

    def _on_popout_destroyed(self):
        self._popout = None

    def to_dataframe(self):
        """The full matrix (including rows hidden by profile selection) with channel labels, or None if not calculated."""
        if not self.controller.experiment.process.get(self.process_key) or self.model._data is None:
            return None
        import pandas as pd
        return pd.DataFrame(self.model._data, index=self.model.vertical_headers,
                            columns=self.model.horizontal_headers)

    def _emit_now(self):
        self.controller.reapply_fine_tuning()
        self.bus.requestUpdateProcessHists.emit()

    def _on_edit(self, index1, index2, role):
        if index1 == index2:
            r = index1.row()
            c = index1.column()
            self.matrix[r][c] = self.model._data[r][c]
            self._timer.start()

    def show_context_menu(self, event):
        # Empty method to completely disable context menu
        pass

    def refresh_heatmap(self):
        # insert key in case of backward compatibility problems
        if self.process_key not in self.controller.experiment.process:
            self.controller.experiment.process[self.process_key] = None

        self.matrix = self.controller.experiment.process[self.process_key]

        if self.matrix:
            if self.process_key == 'similarity_matrix' or self.process_key == 'hotspot_matrix':
                pnn = self.controller.experiment.settings['unmixed']['event_channels_pnn']
                fl_ids = self.controller.experiment.settings['unmixed']['fluorescence_channel_ids']
                fl_pnn = [pnn[n] for n in fl_ids]
                horizontal_headers = fl_pnn
                vertical_headers = fl_pnn
                self.model.update_data(self.matrix, horizontal_headers, vertical_headers)
                self.setVisible(True)
            elif self.process_key == 'unmixing_matrix':
                pnn = self.controller.experiment.settings['raw']['event_channels_pnn']
                fl_ids = self.controller.filtered_raw_fluorescence_channel_ids
                horizontal_headers = [pnn[n] for n in fl_ids]
                pnn = self.controller.experiment.settings['unmixed']['event_channels_pnn']
                fl_ids = self.controller.experiment.settings['unmixed']['fluorescence_channel_ids']
                vertical_headers = [pnn[n] for n in fl_ids]
                self.model.update_data(self.matrix, horizontal_headers, vertical_headers)
                self.setVisible(True)
            elif self.process_key == 'spillover':
                pnn = self.controller.experiment.settings['unmixed']['event_channels_pnn']
                fl_ids = self.controller.experiment.settings['unmixed']['fluorescence_channel_ids']
                fl_pnn = [pnn[n] for n in fl_ids]
                horizontal_headers = fl_pnn
                vertical_headers = fl_pnn

                # if there is a current selection, look it up and set it after the model is reset
                index = self.view.selectionModel().currentIndex()
                if index.isValid() and vertical_headers and horizontal_headers:
                    selected_row_chan = vertical_headers[index.row()]
                    selected_col_chan = horizontal_headers[index.column()]

                self.model.update_data(self.matrix, horizontal_headers, vertical_headers)
                self.setVisible(True)

                if index.isValid() and self.process_key == 'spillover':
                    QTimer.singleShot(0, lambda: self.set_selected_cell(selected_row_chan, selected_col_chan))

        else:
            self.setVisible(False)


    @Slot(QModelIndex, QModelIndex)
    def selected_cell_changed(self, current, previous):
        if current.isValid():
            row_chan = self.model.vertical_headers[current.row()]
            col_chan = self.model.horizontal_headers[current.column()]
            self.bus.spilloverSelectedCellChanged.emit(row_chan, col_chan)

    # guard against header changes and missing headers by checking if row_chan and col_chan are still in the model's headers before trying to find their indices. If not, clear selection.
    @Slot(str, str)
    def set_selected_cell(self, row_chan, col_chan):
        if (self.model.vertical_headers and self.model.horizontal_headers
                and row_chan in self.model.vertical_headers and col_chan in self.model.horizontal_headers):
            self.view.selectionModel().setCurrentIndex(
                self.model.index(self.model.vertical_headers.index(row_chan),
                                 self.model.horizontal_headers.index(col_chan)
                                 ), QItemSelectionModel.ClearAndSelect)
        else:
            with QSignalBlocker(self.view.selectionModel()):
                self.view.selectionModel().clearSelection()
                self.view.selectionModel().clearCurrentIndex()  # Optional: also clears the focus anchor




if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
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
    kc.load_experiment(experiment_path) # note this loads first sample too and runs calculate all histograms and statistics

    frame = HeatmapViewEditor(bus, kc, 'similarity_matrix')
    # frame = HeatmapViewEditor(bus, kc, 'unmixing_matrix')
    # frame = HeatmapViewEditor(bus, kc, 'spillover')

    # Plot the profiles
    frame.refresh_heatmap()
    frame.show()

    frame.resize(950, 600)
    sys.exit(app.exec())


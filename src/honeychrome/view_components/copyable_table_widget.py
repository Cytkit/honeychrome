import functools
import math
import re

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QTableWidgetItem

from PySide6 import QtWidgets, QtGui

_NUMBER, _TEXT, _EMPTY = 0, 1, 2
# A number as tables show it: an optional comparison sign, thousands
# separators, a decimal part, an exponent, and a trailing % or degree sign.
_NUMBER_TEXT = re.compile(r'^[<>≤≥~]?\s*([+\-−]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?:[eE][+\-]?\d+)?)\s*[%°]?$')


@functools.lru_cache(maxsize=65536)
def _text_sort_key(text: str) -> tuple:
    text = text.strip()
    if not text or text.lower() == 'nan':
        return (_EMPTY, 0.0)
    if text.lower().lstrip('+-−') == 'inf':
        return (_NUMBER, -math.inf if text[0] in '-−' else math.inf)
    match = _NUMBER_TEXT.match(text)
    if match and any(ch.isdigit() for ch in match.group(1)):
        return (_NUMBER, float(match.group(1).replace(',', '').replace('−', '-')))
    return (_TEXT, text)


class SortableTableItem(QTableWidgetItem):
    """
    A table cell that sorts as a number when it shows one, so that 10 comes
    after 2. A cell holds a number when its value is one, or when its text
    reads as one: '1,234', '+0.52', '3.1e-05', '12.5%', '45°', '<0.001'.
    In ascending order numbers come first, then text, then empty cells and
    'nan'. Use it wherever a sortable table is filled with formatted text.
    The comparison never falls back on ``QTableWidgetItem.__lt__``: called
    from this override, it calls the override again.
    """

    def _sort_key(self) -> tuple:
        value = self.data(Qt.EditRole)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return (_EMPTY, 0.0) if math.isnan(value) else (_NUMBER, float(value))
        return _text_sort_key(self.text())

    def __lt__(self, other):
        theirs = other._sort_key() if isinstance(other, SortableTableItem) else _text_sort_key(other.text())
        return self._sort_key() < theirs


class CopyableTableWidget(QtWidgets.QTableWidget):
    def __init__(self, list_of_dicts, headers):
        """
        list_of_dicts: list where each element is a dict containing the data for a row of the table
        headers: ordered list of columns (strings)
        """
        rows, columns = len(list_of_dicts), len(headers)
        super().__init__(rows, columns)

        self.setHorizontalHeaderLabels(headers)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSortingEnabled(False)  # Allow sorting by count or ID

        # 3. Populate Rows
        for i, row_data in enumerate(list_of_dicts):
            for j, column_name in enumerate(headers):
                value = row_data.get(column_name, "")
                item = SortableTableItem()

                # Check if the value is a Hex Color string
                if isinstance(value, str) and value.startswith("#") and len(value) == 7:
                    try:
                        # Set the background color
                        item.setBackground(QColor(value))
                        # Optionally, set the text to the hex code or leave it empty
                        item.setData(Qt.EditRole, value)
                    except Exception:
                        # Fallback if the string isn't a valid color
                        item.setData(Qt.EditRole, value)
                else:
                    item.setData(Qt.EditRole, value)

                self.setItem(i, j, item)

        self.setSortingEnabled(True)  # Allow sorting by count or ID
        self.resizeColumnsToContents()
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

    def keyPressEvent(self, event):
        """Handles Ctrl+C to copy selected rows as TSV"""
        if event.matches(QtGui.QKeySequence.Copy):
            self.copy_to_clipboard()
        else:
            super().keyPressEvent(event)

    def copy_to_clipboard(self):
        selection = self.selectedRanges()
        if not selection:
            return

        output = []
        # Support multi-range selection
        for r_range in selection:
            for r in range(r_range.topRow(), r_range.bottomRow() + 1):
                row_data = [self.item(r, c).text() for c in range(self.columnCount())]
                output.append("\t".join(row_data))

        QtWidgets.QApplication.clipboard().setText("\n".join(output))



def _value_sort_key(value) -> tuple:
    """The sort key SortableTableItem gives a cell holding *value*."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (_EMPTY, 0.0) if math.isnan(value) else (_NUMBER, float(value))
    return _text_sort_key('' if value is None else str(value))


def _is_hex_colour(value) -> bool:
    return isinstance(value, str) and value.startswith("#") and len(value) == 7


class _RowsTableModel(QAbstractTableModel):
    """Rows of values (one list per row) behind CopyableTableView. Sorting
    reorders a permutation of the rows using keys computed once per column,
    so it stays fast with hundreds of thousands of rows."""

    def __init__(self, rows: list, headers: list, parent=None):
        super().__init__(parent)
        self._headers = list(headers)
        self._rows = rows
        self._order = list(range(len(rows)))
        self._keys: dict[int, list] = {}

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._order)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._headers)

    def value(self, row: int, col: int):
        """The value shown at view row *row*, column *col*."""
        return self._rows[self._order[row]][col]

    def text(self, row: int, col: int) -> str:
        value = self.value(row, col)
        return '' if value is None else str(value)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        value = self.value(index.row(), index.column())
        if role in (Qt.DisplayRole, Qt.EditRole):
            return value
        if role == Qt.BackgroundRole and _is_hex_colour(value):
            return QColor(value)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self._headers[section]
        return None

    def sort(self, column: int, order=Qt.AscendingOrder):
        if column < 0 or column >= len(self._headers):
            return
        keys = self._keys.get(column)
        if keys is None:
            keys = [_value_sort_key(r[column]) for r in self._rows]
            self._keys[column] = keys
        self.layoutAboutToBeChanged.emit()
        self._order.sort(key=keys.__getitem__, reverse=order == Qt.DescendingOrder)
        self.layoutChanged.emit()


class CopyableTableView(QtWidgets.QTableView):
    """
    CopyableTableWidget for large tables: the same constructor, sorting
    (numbers as numbers, then text, then empty cells) and Ctrl+C copying
    of the selected rows as TSV, but backed by a model, so tens of
    thousands of rows are built and sorted without delay. Column widths
    are fitted to the header and the first rows only.
    """

    WIDTH_SAMPLE_ROWS = 200

    def __init__(self, list_of_dicts, headers):
        super().__init__()
        rows = [[row.get(h, "") for h in headers] for row in list_of_dicts]
        self._model = _RowsTableModel(rows, headers, self)
        self.setModel(self._model)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self.verticalHeader().setDefaultSectionSize(self.fontMetrics().height() + 8)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setWordWrap(False)
        # Rows keep the order given until a header is clicked.
        self.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        self.setSortingEnabled(True)
        self._fit_columns(headers, rows)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        self.horizontalHeader().setStretchLastSection(True)

    def _fit_columns(self, headers: list, rows: list):
        fm = self.fontMetrics()
        sample = rows[:self.WIDTH_SAMPLE_ROWS]
        for j, head in enumerate(headers):
            texts = [str(r[j]) for r in sample if r[j] is not None]
            width = max([fm.horizontalAdvance(t) for t in texts] + [fm.horizontalAdvance(head) + 24])
            self.setColumnWidth(j, min(width + 16, 600))

    def keyPressEvent(self, event):
        """Handles Ctrl+C to copy selected rows as TSV"""
        if event.matches(QtGui.QKeySequence.Copy):
            self.copy_to_clipboard()
        else:
            super().keyPressEvent(event)

    def copy_to_clipboard(self):
        rows = sorted({i.row() for i in self.selectionModel().selectedRows()})
        if not rows:
            return
        n_cols = self._model.columnCount()
        output = ["\t".join(self._model.text(r, c) for c in range(n_cols)) for r in rows]
        QtWidgets.QApplication.clipboard().setText("\n".join(output))

def show_frame_dialog(parent, title: str, frame, default_path: str = '', digits: int = 3):
    """
    A non-modal window listing a pandas DataFrame in a CopyableTableWidget
    (sortable, Ctrl+C copies the selected rows), with a button saving it as
    CSV (full precision; *default_path* proposes the file name). Numbers
    are shown to *digits* significant figures.
    """
    import numpy as np

    def shown(value):
        if isinstance(value, (float, np.floating)):
            return '' if not np.isfinite(value) else float(f"{value:.{digits}g}")
        if isinstance(value, (bool, np.bool_)):
            return 'yes' if value else 'no'
        return value

    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    layout = QtWidgets.QVBoxLayout(dialog)
    headers = [str(c) for c in frame.columns]
    rows = [{h: shown(v) for h, v in zip(headers, record)}
            for record in frame.itertuples(index=False, name=None)]
    layout.addWidget(CopyableTableWidget(rows, headers), stretch=1)
    footer = QtWidgets.QHBoxLayout()
    footer.addStretch()
    save = QtWidgets.QPushButton("Save CSV…")

    def _save():
        path, _ = QtWidgets.QFileDialog.getSaveFileName(dialog, f"Save {title}", default_path,
                                                        "CSV (*.csv)")
        if path:
            frame.to_csv(path, index=False)

    save.clicked.connect(_save)
    footer.addWidget(save)
    layout.addLayout(footer)
    dialog.resize(760, 480)
    dialog.show()
    return dialog

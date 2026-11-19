from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QTableWidgetItem

from PySide6 import QtWidgets, QtGui


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
                item = QTableWidgetItem()

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

"""
Table header with a tick box under the title of chosen columns.

``CheckableHeaderView`` is a horizontal header for a QTableWidget or
QTableView. Sections registered with ``set_checks`` show a labelled tick
box below their title; clicking it toggles the state and emits
``check_toggled(section, checked)``. Clicks elsewhere on the header behave
as usual, including dragging a section edge to resize it.

    header = CheckableHeaderView(table, check_label="Numeric")
    table.setHorizontalHeader(header)
    header.set_checks({3: False, 4: True})
    header.check_toggled.connect(on_toggled)
"""

from PySide6.QtCore import QEvent, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import QApplication, QHeaderView, QStyle, QStyleOptionButton, QToolTip

__all__ = ['CheckableHeaderView']

_CHECK_ROW_HEIGHT = 22


class CheckableHeaderView(QHeaderView):
    """Horizontal header whose chosen sections carry a tick box."""

    check_toggled = Signal(int, bool)

    def __init__(self, parent=None, check_label: str = "", tooltip: str = ""):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._check_label = check_label
        self._check_tooltip = tooltip
        self._checks: dict[int, bool] = {}
        self.setSectionsClickable(True)
        self.setMouseTracking(True)

    def set_checks(self, checks: dict[int, bool]):
        """Show a tick box under exactly the sections in *checks*, ticked
        where the value is True. Replaces any earlier set."""
        self._checks = dict(checks)
        self.viewport().update()
        QTimer.singleShot(0, self._refresh_layout)

    def _refresh_layout(self):
        """Make the table re-measure its header height, which grows by one
        row when any section carries a tick box."""
        self.updateGeometry()
        view = self.parentWidget()
        if view is not None:
            QApplication.sendEvent(view, QResizeEvent(view.size(), view.size()))

    def checks(self) -> dict[int, bool]:
        return dict(self._checks)

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        if self._checks:
            hint.setHeight(hint.height() + _CHECK_ROW_HEIGHT)
        return hint

    def _check_rect(self, section: int) -> QRect:
        left = self.sectionViewportPosition(section)
        return QRect(left, self.height() - _CHECK_ROW_HEIGHT,
                     self.sectionSize(section), _CHECK_ROW_HEIGHT)

    def paintSection(self, painter, rect, logical_index):
        if logical_index not in self._checks:
            super().paintSection(painter, rect, logical_index)
            return
        title_rect = QRect(rect.left(), rect.top(), rect.width(),
                           max(rect.height() - _CHECK_ROW_HEIGHT, 1))
        painter.save()
        painter.fillRect(rect, self.palette().button())
        super().paintSection(painter, title_rect, logical_index)
        painter.restore()
        option = QStyleOptionButton()
        option.state = QStyle.StateFlag.State_Enabled | (
            QStyle.StateFlag.State_On if self._checks[logical_index]
            else QStyle.StateFlag.State_Off)
        option.text = self._check_label
        option.rect = QRect(rect.left() + 6, rect.bottom() - _CHECK_ROW_HEIGHT + 2,
                            max(rect.width() - 8, 1), _CHECK_ROW_HEIGHT - 2)
        option.palette = self.palette()
        self.style().drawControl(QStyle.ControlElement.CE_CheckBox, option, painter, self)

    def mousePressEvent(self, event):
        section = self.logicalIndexAt(event.position().toPoint())
        if section in self._checks and self._check_rect(section).contains(
                event.position().toPoint()):
            self._checks[section] = not self._checks[section]
            self.viewport().update()
            self.check_toggled.emit(section, self._checks[section])
            event.accept()
            return
        super().mousePressEvent(event)

    def event(self, event):
        if event.type() == QEvent.Type.ToolTip and self._check_tooltip:
            section = self.logicalIndexAt(event.pos())
            if section in self._checks and self._check_rect(section).contains(event.pos()):
                QToolTip.showText(event.globalPos(), self._check_tooltip, self)
                return True
        return super().event(event)

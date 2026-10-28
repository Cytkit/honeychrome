"""
Vertical section navigation for multi-page plugins.

``SideTabWidget`` shows a plugin's sections as a list down the left-hand
side (an icon and a horizontal label each) beside a page stack that fills
the rest of the window, so a plugin's own sections read as distinct from the
main window's tab bar and the page in use gets the whole width. With
``scroll_pages=True`` each page sits in its own scroll area, so a page wider
or taller than the window scrolls instead of setting a minimum size for the
whole main window (which would stop the sample panel's splitter). It exposes
the part of the QTabWidget API plugins use: addTab, setTabText,
currentIndex, setCurrentIndex, currentWidget, setCurrentWidget, widget,
indexOf, count and currentChanged.
"""

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QFrame, QHBoxLayout, QListWidget, QListWidgetItem,
    QScrollArea, QSizePolicy, QStackedWidget, QWidget,
)

__all__ = ['SideTabWidget']


class SideTabWidget(QWidget):
    """Vertical navigation list beside a page stack."""

    currentChanged = Signal(int)

    def __init__(self, parent=None, scroll_pages: bool = False):
        super().__init__(parent)
        self._scroll_pages = scroll_pages
        # The pages in stack order; with scroll_pages the stack holds the
        # scroll area around each page instead of the page itself.
        self._pages: list[QWidget] = []
        self._nav = QListWidget()
        self._nav.setIconSize(QSize(20, 20))
        self._nav.setSelectionMode(QAbstractItemView.SingleSelection)
        self._nav.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._nav.setFrameShape(QFrame.NoFrame)
        self._nav.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self._nav.setStyleSheet("QListWidget::item { padding: 8px 14px 8px 8px; }")

        self._stack = QStackedWidget()

        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setFrameShadow(QFrame.Sunken)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._nav)
        layout.addWidget(divider)
        layout.addWidget(self._stack, stretch=1)

        self._nav.currentRowChanged.connect(self._on_row_changed)

    def addTab(self, widget: QWidget, label: str, icon_name: str | None = None) -> int:
        """Add a page; *icon_name* is a Tabler icon from the app's assets."""
        holder = widget
        if self._scroll_pages:
            holder = QScrollArea()
            holder.setWidgetResizable(True)
            holder.setFrameShape(QFrame.NoFrame)
            holder.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            holder.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            holder.setWidget(widget)
        index = self._stack.addWidget(holder)
        self._pages.append(widget)
        item = QListWidgetItem(label)
        if icon_name:
            from honeychrome.view_components.icon_loader import icon
            item.setIcon(icon(icon_name))
        self._nav.addItem(item)
        if self._nav.currentRow() < 0:
            self._nav.setCurrentRow(0)
        # Polish first so the item padding from the style sheet is measured.
        self._nav.ensurePolished()
        self._nav.setFixedWidth(self._nav.sizeHintForColumn(0) + 2 * self._nav.frameWidth() + 12)
        return index

    def setTabText(self, index: int, text: str):
        item = self._nav.item(index)
        if item is not None:
            item.setText(text)

    def currentIndex(self) -> int:
        return self._stack.currentIndex()

    def setCurrentIndex(self, index: int):
        self._nav.setCurrentRow(index)

    def currentWidget(self) -> QWidget | None:
        return self.widget(self._stack.currentIndex())

    def setCurrentWidget(self, widget: QWidget):
        index = self.indexOf(widget)
        if index >= 0:
            self.setCurrentIndex(index)

    def widget(self, index: int) -> QWidget | None:
        return self._pages[index] if 0 <= index < len(self._pages) else None

    def indexOf(self, widget: QWidget) -> int:
        for i, page in enumerate(self._pages):
            if page is widget:
                return i
        return -1

    def count(self) -> int:
        return self._stack.count()

    def _on_row_changed(self, row: int):
        # Clearing the list's current row (e.g. ctrl-click) must not blank the page.
        if row < 0:
            self._nav.setCurrentRow(self._stack.currentIndex())
            return
        if row == self._stack.currentIndex():
            return
        self._stack.setCurrentIndex(row)
        self.currentChanged.emit(row)

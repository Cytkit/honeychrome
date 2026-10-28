import sys
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout,
                               QScrollArea, QFrame)
from PySide6.QtCore import Qt, QObject, QEvent

from honeychrome.view_components.help_texts import process_help_text
from honeychrome.view_components.icon_loader import icon

class HelpToggleWidget(QWidget):
    """A "Show Help" button that reveals rich-text help below it.

    With ``scrolling=True`` the text sits in its own scroll area: when shown
    it takes the height it needs, up to ``SCROLL_HEIGHT_FRACTION`` of the
    window (at least ``SCROLL_MIN_HEIGHT`` pixels), and scrolls beyond that;
    the height follows the window as it is resized.
    Use it where the help sits above a page that fills the window and is not
    itself inside a scroll area, so long help is not cropped.
    """

    SCROLL_HEIGHT_FRACTION = 0.45
    SCROLL_MIN_HEIGHT = 160

    def __init__(self, title="Show Help", text='', parent=None, scrolling=False):
        super().__init__(parent)
        self.title = title
        self.help_scroll = None
        self._watched_window = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toggle button
        self.help_button = QPushButton(icon('help'), title)
        self.help_button.clicked.connect(self.toggle_help)

        btn_top_layout = QHBoxLayout()
        btn_top_layout.addWidget(self.help_button)
        btn_top_layout.addStretch()

        # Help text (rich text / HTML)
        self.help_label = QLabel(text)
        self.help_label.setTextFormat(Qt.RichText)
        self.help_label.setWordWrap(True)
        self.help_label.setVisible(False)  # start hidden
        # references open in the browser
        self.help_label.setTextInteractionFlags(Qt.LinksAccessibleByMouse)
        self.help_label.setOpenExternalLinks(True)
        # self.help_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        font = self.help_label.font()
        font.setPointSize(14)  # Set your desired font size
        self.help_label.setFont(font)

        layout.addLayout(btn_top_layout)
        if scrolling:
            self.help_label.setVisible(True)
            self.help_scroll = QScrollArea()
            self.help_scroll.setWidgetResizable(True)
            self.help_scroll.setFrameShape(QFrame.NoFrame)
            self.help_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.help_scroll.setWidget(self.help_label)
            self.help_scroll.setVisible(False)
            layout.addWidget(self.help_scroll)
        else:
            layout.addWidget(self.help_label)

    def toggle_help(self):
        if self.help_scroll is None:
            is_visible = self.help_label.isVisible()
            self.help_label.setVisible(not is_visible)
            self.help_button.setText("Hide Help" if not is_visible else self.title)
            return
        shown = self.help_scroll.isHidden()
        if shown:
            window = self.window()
            if window is not self and window is not self._watched_window:
                window.installEventFilter(self)
                self._watched_window = window
            self._fit_scroll_height()
        self.help_scroll.setVisible(shown)
        self.help_button.setText("Hide Help" if shown else self.title)

    def _fit_scroll_height(self):
        """Size the scroll area to the text at the current width, capped
        at a fraction of the window height."""
        cap = max(self.SCROLL_MIN_HEIGHT,
                  int(self.SCROLL_HEIGHT_FRACTION * self.window().height()))
        bar = self.help_scroll.verticalScrollBar().sizeHint().width()
        width = max(200, self.width() - bar - 4)
        self.help_scroll.setFixedHeight(min(cap, self.help_label.heightForWidth(width) + 8))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.help_scroll is not None and not self.help_scroll.isHidden():
            self._fit_scroll_height()

    def eventFilter(self, obj, event):
        """The window was resized: refit the scrolling help to it."""
        if (obj is self._watched_window and event.type() == QEvent.Type.Resize
                and self.help_scroll is not None and not self.help_scroll.isHidden()):
            self._fit_scroll_height()
        return False


class WheelBlocker(QObject):
    """Event filter that ignores wheel events, preventing scroll from being
    captured by pyqtgraph viewports inside a scrollable parent."""
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            event.ignore()
            return True
        return False
    

if __name__ == "__main__":
    title = "Show Help"
    text = process_help_text



    app = QApplication(sys.argv)
    window = HelpToggleWidget(title, text)
    window.resize(400, 300)
    window.show()
    sys.exit(app.exec())
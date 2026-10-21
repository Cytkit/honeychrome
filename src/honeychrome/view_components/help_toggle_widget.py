import sys
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout)
from PySide6.QtCore import Qt, QObject, QEvent

from honeychrome.view_components.help_texts import process_help_text
from honeychrome.view_components.icon_loader import icon

class HelpToggleWidget(QWidget):
    def __init__(self, title="Show Help", text='', parent=None):
        super().__init__(parent)
        self.title = title
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
        layout.addWidget(self.help_label)

    def toggle_help(self):
        is_visible = self.help_label.isVisible()
        self.help_label.setVisible(not is_visible)
        self.help_button.setText("Hide Help" if not is_visible else self.title)


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
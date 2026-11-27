from pathlib import Path

# path to your icons
ICON_PATH = Path(__file__).parent / 'assets' / 'tabler_icons'

# def icon(name: str) -> QIcon:
#     """Load a Tabler icon by filename (without .svg)."""
#     path = ICON_PATH / f"{name}.svg"
#     if path.exists():
#         return QIcon(str(path))
#     return QIcon()  # null icon if missing

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QPalette, QLinearGradient, QBrush
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtCore import Qt, QSize

def icon(svg_name: str, size: QSize = QSize(24, 24), colour=None) -> QIcon:
    """Return a QIcon automatically tinted based on light/dark theme.
    colour may be a pair of colours: the icon is then split on a diagonal
    slash, the first colour above it and the second below."""
    app = QApplication.instance()
    if colour is None:
        palette = app.palette()
        base_color = palette.color(QPalette.ColorRole.Base)
        is_dark = base_color.value() < 128
        color = QColor("white" if is_dark else "black")
    elif isinstance(colour, (tuple, list)):
        split = QLinearGradient(0, 0, size.width(), size.height())
        split.setColorAt(0.0, QColor(colour[0]))
        split.setColorAt(0.5, QColor(colour[0]))
        split.setColorAt(0.5001, QColor(colour[1]))
        split.setColorAt(1.0, QColor(colour[1]))
        color = QBrush(split)
    else:
        color = QColor(colour)

    svg_path = ICON_PATH / f"{svg_name}.svg"
    renderer = QSvgRenderer(str(svg_path))
    pixmap = QPixmap(size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), color)
    painter.end()

    return QIcon(pixmap)

"""
Embedded matplotlib figure with a size slider, pop-out and export buttons.

``ScalableFigureWidget`` shows one figure at a chosen percentage of its
natural size (25% to 400%) inside a scroll area, so a large heatmap can be
enlarged to read its labels and a small plot shrunk to fit beside others.
The mouse wheel is left to the scroll area rather than being used by
matplotlib.

The figure's canvas is available as ``widget.canvas`` for connecting
matplotlib events (hover, click).
"""

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea, QSizePolicy,
    QSlider, QVBoxLayout, QWidget,
)

import honeychrome.settings as settings
from honeychrome.view_components.exportable_plot_widget import apply_plot_theme

__all__ = ['ScalableFigureWidget', 'MIN_SCALE_PCT', 'MAX_SCALE_PCT', 'DEFAULT_SCALE_PCT',
           'initial_scale_for']

MIN_SCALE_PCT = 25
MAX_SCALE_PCT = 400
DEFAULT_SCALE_PCT = 100
RESIZE_DELAY_MS = 120

_canvas_class = None


def initial_scale_for(figure, preferred: int = DEFAULT_SCALE_PCT, max_height_px: int = 8000) -> int:
    """A starting size (percent) for *figure*: *preferred*, reduced so the
    canvas is at most *max_height_px* tall, and never below MIN_SCALE_PCT."""
    height = figure.get_figheight() * figure.get_dpi()
    pct = preferred
    if height * pct / 100 > max_height_px:
        pct = int(100 * max_height_px / max(height, 1))
    return max(MIN_SCALE_PCT, min(MAX_SCALE_PCT, pct))


def scrollable_canvas_class():
    """FigureCanvasQTAgg subclass that passes wheel events to the enclosing
    scroll area instead of consuming them (imported on first use so that
    importing this module does not load matplotlib's Qt backend)."""
    global _canvas_class
    if _canvas_class is None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        class _ScrollableFigureCanvas(FigureCanvasQTAgg):
            def wheelEvent(self, event):
                event.ignore()

        _canvas_class = _ScrollableFigureCanvas
    return _canvas_class


class ScalableFigureWidget(QWidget):
    """
    A figure with a row of controls above it: Pop Out (when *pop_out* is
    given), Export, and a Size slider.

    figure         : the matplotlib Figure to show.
    title          : used for the export file name.
    experiment_dir : folder the exported graphic is written to.
    pop_out        : optional callable with no arguments that opens the
                     figure in its own window.
    initial_scale  : starting size in percent of the natural size (large
                     heatmaps start smaller so they draw quickly).
    """

    def __init__(self, figure, title: str = 'figure', experiment_dir='',
                 pop_out=None, parent=None, initial_scale: int = DEFAULT_SCALE_PCT):
        super().__init__(parent)
        self.figure = figure
        self.export_stem = title.replace(' ', '_').replace(':', '')
        self.experiment_dir = experiment_dir

        # Natural size in logical pixels, taken before the canvas adjusts
        # the figure for a high-density display.
        dpi = figure.get_dpi()
        self._natural_w = int(figure.get_figwidth() * dpi)
        self._natural_h = int(figure.get_figheight() * dpi)

        apply_plot_theme()
        self.canvas = scrollable_canvas_class()(figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self._scroll = QScrollArea()
        self._scroll.setWidget(self.canvas)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        if pop_out is not None:
            pop_out_button = QPushButton("⤢ Pop Out")
            pop_out_button.setToolTip("Open this plot in a larger window with zoom and save tools.")
            pop_out_button.clicked.connect(lambda _checked=False: pop_out())
            controls.addWidget(pop_out_button)
        export_button = QPushButton("Export Graphic")
        export_button.clicked.connect(self.export_graphic)
        controls.addWidget(export_button)
        controls.addSpacing(16)
        controls.addWidget(QLabel("Size:"))
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(MIN_SCALE_PCT, MAX_SCALE_PCT)
        self.size_slider.setSingleStep(5)
        self.size_slider.setPageStep(25)
        initial_scale = int(min(MAX_SCALE_PCT, max(MIN_SCALE_PCT, initial_scale)))
        self.size_slider.setValue(initial_scale)
        self.size_slider.setFixedWidth(200)
        self.size_slider.setToolTip(
            f"Show this plot at {MIN_SCALE_PCT}% to {MAX_SCALE_PCT}% of its natural size. "
            "Pop Out and Export always use the full figure."
        )
        controls.addWidget(self.size_slider)
        self._pct_label = QLabel()
        self._pct_label.setFixedWidth(44)
        controls.addWidget(self._pct_label)
        controls.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addLayout(controls)
        layout.addWidget(self._scroll, stretch=1)

        # The canvas is resized once the slider has paused, so dragging it
        # does not redraw a large figure at every step.
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(RESIZE_DELAY_MS)
        self._resize_timer.timeout.connect(self._apply_scale)
        self.size_slider.valueChanged.connect(self._on_slider_changed)

        self._pct_label.setText(f"{initial_scale}%")
        self._apply_scale()

    def scale_percent(self) -> int:
        """The percentage of natural size currently selected."""
        return self.size_slider.value()

    def _on_slider_changed(self, pct: int):
        self._pct_label.setText(f"{pct}%")
        self._resize_timer.start()

    def _apply_scale(self):
        pct = self.size_slider.value()
        self.canvas.setFixedSize(max(80, int(self._natural_w * pct / 100)),
                                 max(80, int(self._natural_h * pct / 100)))

    def export_graphic(self):
        """Save the figure in the application's graphics export format."""
        fmt = settings.graphics_export_format_retrieved
        if not self.experiment_dir:
            QMessageBox.warning(self, "Export", "There is no experiment folder to export to.")
            return
        path = Path(self.experiment_dir) / f"{self.export_stem}.{fmt}"
        try:
            self.figure.savefig(path, bbox_inches="tight")
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        QMessageBox.information(
            self, "Exported",
            f"Exported {fmt} graphic file:\n{path.name}\nto {path.parent}")

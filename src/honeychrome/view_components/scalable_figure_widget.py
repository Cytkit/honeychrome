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
    QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

import honeychrome.settings as settings
from honeychrome.view_components.exportable_plot_widget import apply_plot_theme

__all__ = ['ScalableFigureWidget', 'MIN_SCALE_PCT', 'MAX_SCALE_PCT', 'DEFAULT_SCALE_PCT',
           'initial_scale_for']

MIN_SCALE_PCT = 25
MAX_SCALE_PCT = 400
DEFAULT_SCALE_PCT = 100
RESIZE_DELAY_MS = 120
EXPORT_DPI = 300

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
    options        : optional figure_options.FigureOptionsRef: the figure is
                     drawn with its options, and an Options button opens a
                     side panel that changes them (see figure_options_panel).
    """

    def __init__(self, figure, title: str = 'figure', experiment_dir='',
                 pop_out=None, parent=None, initial_scale: int = DEFAULT_SCALE_PCT,
                 options=None):
        super().__init__(parent)
        self.figure = figure
        self.options = options
        if options is not None:
            options.apply(figure)
        self.export_stem = title.replace(' ', '_').replace(':', '')
        self.experiment_dir = experiment_dir

        # Natural size in logical pixels, taken before the canvas adjusts
        # the figure for a high-density display.
        dpi = figure.get_dpi()
        self._natural_w = int(figure.get_figwidth() * dpi)
        self._natural_h = int(figure.get_figheight() * dpi)
        # The slider resizes the figure itself; exports restore this size.
        self._natural_inches = tuple(figure.get_size_inches())

        apply_plot_theme()
        self.canvas = scrollable_canvas_class()(figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.canvas.mpl_connect('figure_leave_event', self._on_leave)

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

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.addWidget(self._scroll, stretch=1)
        if options is not None:
            from honeychrome.view_components.figure_options_panel import (
                FigureOptionsPanel, options_toggle_button,
            )
            from honeychrome.view_components.figure_options import has_axis_range
            self.options_panel = FigureOptionsPanel(options, self, axis_range=has_axis_range(figure))
            controls.addWidget(options_toggle_button(self.options_panel))
            body.addWidget(self.options_panel)
            options.store.subscribe(self._on_options_changed)
            self.destroyed.connect(lambda _obj=None, store=options.store,
                                   cb=self._on_options_changed: store.unsubscribe(cb))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addLayout(controls)
        layout.addLayout(body, stretch=1)

        # The canvas is resized once the slider has paused, so dragging it
        # does not redraw a large figure at every step.
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(RESIZE_DELAY_MS)
        self._resize_timer.timeout.connect(self._apply_scale)
        self.size_slider.valueChanged.connect(self._on_slider_changed)

        self._pct_label.setText(f"{initial_scale}%")
        self._apply_scale()

    def _on_options_changed(self, scope: str):
        if self.options is not None and scope == self.options.scope:
            self.options.apply(self.figure)
            self.canvas.draw_idle()

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
            self.save_full_size(path)
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        QMessageBox.information(
            self, "Exported",
            f"Exported {fmt} graphic file:\n{path.name}\nto {path.parent}")

    def save_full_size(self, path, dpi: int = EXPORT_DPI):
        """Save the figure at its natural size, whatever size it is shown at,
        at *dpi* and trimmed to its contents, without hover labels."""
        from honeychrome.view_components.differential_plots import hide_hover_annotations
        shown = tuple(self.figure.get_size_inches())
        hide_hover_annotations(self.figure)
        self.figure.set_size_inches(*self._natural_inches, forward=False)
        try:
            self.figure.savefig(path, dpi=dpi, bbox_inches='tight')
        finally:
            self.figure.set_size_inches(*shown, forward=False)
            self.canvas.draw_idle()

    def save_full_size_as(self):
        """Ask for a file name, then save as save_full_size does."""
        fmt = settings.graphics_export_format_retrieved
        start = str(Path(self.experiment_dir or '.') / f"{self.export_stem}.{fmt}")
        path, _selected = QFileDialog.getSaveFileName(
            self, "Save figure", start, "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;All files (*)")
        if not path:
            return
        if not Path(path).suffix:
            path += f".{fmt}"
        try:
            self.save_full_size(path)
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))

    def make_toolbar(self, parent=None):
        """Matplotlib's navigation toolbar for this figure, whose Save button
        saves the full figure as Export does."""
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
        widget = self

        class _FullSizeSaveToolbar(NavigationToolbar2QT):
            def save_figure(self, *_args):
                widget.save_full_size_as()

        return _FullSizeSaveToolbar(self.canvas, parent)

    def _on_leave(self, _event):
        from honeychrome.view_components.differential_plots import hide_hover_annotations
        if hide_hover_annotations(self.figure):
            self.canvas.draw_idle()

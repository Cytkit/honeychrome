"""
Embedded matplotlib figure with a size slider, pop-out and export buttons.

``ScalableFigureWidget`` shows one figure at a chosen percentage of its
natural size (25% to 400%) inside a scroll area, so a large heatmap can be
enlarged to read its labels and a small plot shrunk to fit beside others.
The mouse wheel is left to the scroll area rather than being used by
matplotlib.

The figure's canvas is available as ``widget.canvas`` for connecting
matplotlib events (hover, click).

The module also holds the figure-saving functions shared by the analysis
plugins, so that every Export button and every pop-out window's Save button
behaves the same way: figures are written at EXPORT_DPI, trimmed to their
contents, into the plugin's results folder (``results_folder``) by default,
and the file written is reported in the main window's status bar.
"""

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

import honeychrome.settings as settings
from honeychrome.view_components.exportable_plot_widget import apply_plot_theme

__all__ = ['ScalableFigureWidget', 'MIN_SCALE_PCT', 'MAX_SCALE_PCT', 'DEFAULT_SCALE_PCT',
           'EXPORT_DPI', 'initial_scale_for', 'results_folder', 'start_path', 'save_figure',
           'report_saved', 'ask_figure_path', 'figure_toolbar']

MIN_SCALE_PCT = 25
MAX_SCALE_PCT = 400
DEFAULT_SCALE_PCT = 100
RESIZE_DELAY_MS = 120
EXPORT_DPI = 300

_canvas_class = None
_FIGURE_FILTERS = {'png': "PNG (*.png)", 'pdf': "PDF (*.pdf)", 'svg': "SVG (*.svg)"}


def results_folder(experiment_dir, subfolder) -> Path | str:
    """The plugin's results folder *subfolder* inside *experiment_dir*, or
    '' when no experiment is open."""
    return Path(experiment_dir) / subfolder if experiment_dir else ''


def start_path(folder, filename: str) -> str:
    """*folder*/*filename* as the starting path of a save dialog, creating
    *folder* first so the dialog opens in it ('.' when there is no folder)."""
    if folder:
        try:
            Path(folder).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    return str(Path(folder or '.') / filename)


def save_figure(figure, path, dpi: int = EXPORT_DPI):
    """Write *figure* to *path* (format from its suffix) at *dpi*, trimmed
    to its contents and without hover labels, creating the folder when
    missing."""
    from honeychrome.view_components.differential_plots import hide_hover_annotations
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hide_hover_annotations(figure)
    figure.savefig(path, dpi=dpi, bbox_inches='tight')


def report_saved(widget, what):
    """Show "Saved <what>" in the status bar of the main window that
    *widget* (or the window it sits in) belongs to."""
    window = widget
    while window is not None and not isinstance(window, QMainWindow):
        window = window.parentWidget()
    if window is not None:
        window.statusBar().showMessage(f"Saved {what}")


def ask_figure_path(parent, folder, stem: str, title: str = "Save figure") -> Path | None:
    """Ask where to save a figure, starting at *folder*/*stem* in the
    application's graphics export format; the folder is created first so
    the dialog opens in it. Returns the chosen path, with a suffix, or None
    when cancelled."""
    fmt = str(settings.graphics_export_format_retrieved or 'png').lower()
    if fmt not in _FIGURE_FILTERS:
        fmt = 'png'
    start = start_path(folder, f"{stem}.{fmt}")
    path, selected = QFileDialog.getSaveFileName(
        parent, title, start, ";;".join(_FIGURE_FILTERS.values()), _FIGURE_FILTERS[fmt])
    if not path:
        return None
    if Path(path).suffix.lower().lstrip('.') not in _FIGURE_FILTERS:
        path += '.' + (selected.split('*.')[-1].rstrip(')') if '*.' in selected else fmt)
    return Path(path)


def figure_toolbar(canvas, parent, folder, stem: str, save=None):
    """
    Matplotlib's navigation toolbar for *canvas*, whose Save button asks for
    a file name (starting in *folder*, named *stem*), saves at EXPORT_DPI
    and reports the file in the status bar.

    save : optional callable taking the chosen path, used in place of
           saving the canvas's figure as it is (e.g. to save it at its
           natural size).
    """
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

    class _ExportToolbar(NavigationToolbar2QT):
        def save_figure(self, *_args):
            widget = self.canvas
            if widget is None:
                return
            path = ask_figure_path(widget, folder, stem)
            if path is None:
                return
            try:
                if save is not None:
                    save(path)
                else:
                    save_figure(widget.figure, path)
                    widget.draw_idle()
            except Exception as exc:
                QMessageBox.warning(widget, "Save failed", str(exc))
                return
            report_saved(widget, path)

    return _ExportToolbar(canvas, parent)


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
    experiment_dir : folder the exported graphic is written to (the
                     plugin's results folder, see results_folder); also
                     where the pop-out toolbar's Save dialog starts.
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
        """Save the figure in the application's graphics export format to
        the export folder, and report the file in the status bar."""
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
        report_saved(self, path)

    def save_full_size(self, path, dpi: int = EXPORT_DPI):
        """Save the figure at its natural size, whatever size it is shown at,
        as save_figure does."""
        shown = tuple(self.figure.get_size_inches())
        self.figure.set_size_inches(*self._natural_inches, forward=False)
        try:
            save_figure(self.figure, path, dpi)
        finally:
            self.figure.set_size_inches(*shown, forward=False)
            self.canvas.draw_idle()

    def make_toolbar(self, parent=None):
        """Matplotlib's navigation toolbar for this figure, whose Save button
        saves the full figure as Export does, starting in the export folder."""
        return figure_toolbar(self.canvas, parent, self.experiment_dir, self.export_stem,
                              save=self.save_full_size)

    def _on_leave(self, _event):
        from honeychrome.view_components.differential_plots import hide_hover_annotations
        if hide_hover_annotations(self.figure):
            self.canvas.draw_idle()

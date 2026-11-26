"""
Embedded matplotlib figure with a size slider, pop-out and export buttons.

``ScalableFigureWidget`` shows one figure at a chosen percentage of its
natural size (25% to 400%) inside a scroll area, so a large heatmap can be
enlarged to read its labels and a small plot shrunk to fit beside others.
The mouse wheel is left to the scroll area rather than being used by
matplotlib. With ``fit`` the widget also offers a Fit menu: Width sizes the
figure so its full width fits the panel, Height so its full height does,
and Whole figure so all of it is visible without scrolling; the fit follows
the panel as it is resized. With ``fit_key`` the chosen fit is remembered
across sessions for figures that share the key.

The figure's canvas is available as ``widget.canvas`` for connecting
matplotlib events (hover, click).

The module also holds the figure-saving functions shared by the analysis
plugins, so that every Export button and every pop-out window's Save button
behaves the same way: figures are written at EXPORT_DPI, trimmed to their
contents, into the plugin's results folder (``results_folder``) by default,
and the file written is reported in the main window's status bar.
"""

from pathlib import Path

from PySide6.QtCore import QEvent, QSettings, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

import honeychrome.settings as settings
from honeychrome.view_components.exportable_plot_widget import apply_plot_theme

__all__ = ['ScalableFigureWidget', 'MIN_SCALE_PCT', 'MAX_SCALE_PCT', 'DEFAULT_SCALE_PCT',
           'EXPORT_DPI', 'FIT_OFF', 'FIT_WIDTH', 'FIT_HEIGHT', 'FIT_WHOLE', 'initial_scale_for',
           'results_folder', 'start_path', 'save_figure', 'report_saved', 'ask_figure_path',
           'figure_toolbar']

MIN_SCALE_PCT = 25
MAX_SCALE_PCT = 400
DEFAULT_SCALE_PCT = 100
RESIZE_DELAY_MS = 120
EXPORT_DPI = 300

FIT_OFF = 'off'
FIT_WIDTH = 'width'
FIT_HEIGHT = 'height'
FIT_WHOLE = 'whole'
FIT_MODES = {FIT_OFF: 'Off (Size slider)', FIT_WIDTH: 'Width', FIT_HEIGHT: 'Height', FIT_WHOLE: 'Whole figure'}
# Remembered fits (``fit_key``), shared by every plugin's figures.
_FIT_SETTINGS = ('honeychrome', 'figure_fit')

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
    fit            : None for no Fit menu; else the fit it starts with
                     (FIT_OFF, FIT_WIDTH, FIT_HEIGHT or FIT_WHOLE). While a
                     fit is chosen, the Size slider is set (within its range)
                     so the figure's width, height or both fit the panel.
                     ``fitModeChanged`` reports the user's changes.
    fit_key        : with *fit*, the key under which the user's fit is
                     remembered; a remembered fit replaces *fit*.
    """

    fitModeChanged = Signal(str)

    def __init__(self, figure, title: str = 'figure', experiment_dir='',
                 pop_out=None, parent=None, initial_scale: int = DEFAULT_SCALE_PCT,
                 options=None, fit: str | None = None, fit_key: str | None = None):
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
        self._controls = controls
        self._extra_at = controls.count()
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
        self.fit_combo = None
        self._fit_key = fit_key if fit is not None else None
        if fit is not None:
            if self._fit_key:
                stored = QSettings(*_FIT_SETTINGS).value(self._fit_key)
                fit = stored if stored in FIT_MODES else fit
            fit = fit if fit in FIT_MODES else FIT_OFF
            controls.addWidget(QLabel("Fit:"))
            self.fit_combo = QComboBox()
            for mode, text in FIT_MODES.items():
                self.fit_combo.addItem(text, mode)
            self.fit_combo.setCurrentIndex(self.fit_combo.findData(fit))
            self.fit_combo.setToolTip("Size the plot to the panel: its full width, its full height, or the whole "
                                      "figure without scrolling. Off uses the Size slider. Pop Out and Export "
                                      "always use the full figure.")
            self.size_slider.setEnabled(fit == FIT_OFF)
            controls.addWidget(self.fit_combo)
        controls.addStretch()

        self._held_label = QLabel()
        self._held_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._held_label.setStyleSheet("color: grey; font-style: italic;")
        self._held_label.hide()
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.addWidget(self._scroll, stretch=1)
        body.addWidget(self._held_label, stretch=1)
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
        if self.fit_combo is not None:
            self.fit_combo.currentIndexChanged.connect(self._on_fit_changed)
            self._scroll.installEventFilter(self)

        self._pct_label.setText(f"{initial_scale}%")
        self._apply_scale()

    def add_control(self, widget):
        """Add *widget* (a button, say) to the control row, after Export
        Graphic and any controls added before it."""
        self._controls.insertWidget(self._extra_at, widget)
        self._extra_at += 1

    def hold_while_open(self, window, text: str):
        """Show *text* in place of the figure while *window* (a pop-out of
        it) is open, and the figure again once the window is destroyed."""
        self._held_label.setText(text)
        self._scroll.hide()
        self._held_label.show()

        def restore(_obj=None):
            try:
                self._held_label.hide()
                self._scroll.show()
                self._apply_scale()
            except RuntimeError:
                pass    # this widget went first
        window.destroyed.connect(restore)

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

    def _on_fit_changed(self, _index: int):
        mode = self.fit_mode()
        self.size_slider.setEnabled(mode == FIT_OFF)
        if self._fit_key:
            QSettings(*_FIT_SETTINGS).setValue(self._fit_key, mode)
        self._apply_scale()
        self.fitModeChanged.emit(mode)

    def fit_mode(self) -> str:
        """The chosen fit (FIT_OFF when there is no Fit menu)."""
        return self.fit_combo.currentData() if self.fit_combo is not None else FIT_OFF

    def set_fit_mode(self, mode: str):
        """Choose a fit, as the user would in the Fit menu."""
        if self.fit_combo is not None and mode in FIT_MODES:
            self.fit_combo.setCurrentIndex(self.fit_combo.findData(mode))

    def eventFilter(self, obj, event):
        if (obj is self._scroll and event.type() in (QEvent.Type.Resize, QEvent.Type.Show)
                and self.fit_mode() != FIT_OFF):
            self._resize_timer.start()
        return False

    def _fit_percent(self, mode: str) -> int:
        """The size (percent) at which the figure fits the scroll area for
        *mode*. Fitting one side leaves room for the scroll bar of the other,
        which can then appear, so that its appearing does not change the fit;
        the whole figure fits without either."""
        frame = 2 * self._scroll.frameWidth()
        width = self._scroll.width() - frame
        height = self._scroll.height() - frame
        if mode == FIT_WIDTH:
            width -= self._scroll.verticalScrollBar().sizeHint().width()
        elif mode == FIT_HEIGHT:
            height -= self._scroll.horizontalScrollBar().sizeHint().height()
        by_width = 100 * width / max(self._natural_w, 1)
        by_height = 100 * height / max(self._natural_h, 1)
        pct = {FIT_WIDTH: by_width, FIT_HEIGHT: by_height}.get(mode, min(by_width, by_height))
        return max(MIN_SCALE_PCT, min(MAX_SCALE_PCT, int(pct)))

    def _apply_scale(self):
        mode = self.fit_mode()
        if mode != FIT_OFF and self._scroll.isVisible():
            pct = self._fit_percent(mode)
            if pct != self.size_slider.value():
                self.size_slider.blockSignals(True)
                self.size_slider.setValue(pct)
                self.size_slider.blockSignals(False)
                self._pct_label.setText(f"{pct}%")
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

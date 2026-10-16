"""
transforms_panel.py — axis transform inspector shared by the analysis plugins
============================================================================
A plugin's own set of axis transforms, shown and adjusted as in the main
window's plots:

  • TransformInspector  — a channel list, a 1-D histogram of the selected
                          channel and square biplots against it. Dragging an
                          axis adjusts the transform exactly as on the main
                          cytometry plots (lower half of a logicle axis: the
                          width basis W; upper half: the displayed range).
                          Auto-transform sets W per channel from the events
                          shown (Parks, Roederer & Moore 2006), and the
                          parameters can be saved to and loaded from CSV.
                          Subclasses provide the events and what happens
                          when a transform changes.
  • PopulationTransformsPanel
                        — the inspector for the gating plugins: the events
                          are one population of the experiment's gating
                          hierarchy, pooled over a few samples that are
                          loaded and unmixed in the background.
  • TransformSet        — a plugin's transforms as Honeychrome Transform
                          objects, built from and saved as parameter dicts.

The Honeycluster (DR/Clustering) plugin's Transforms tab is a
TransformInspector too. The file name does not end in ``_tab.py``, so the
plugin loader does not treat it as a tab.
"""

from __future__ import annotations

import csv
import math
from copy import deepcopy

import numpy as np

from PySide6.QtCore import QEvent, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout,
    QWidget,
)

import honeychrome.settings as hc_settings

__all__ = ['TransformInspector', 'PopulationTransformsPanel', 'TransformSet', 'BiplotTile',
           'transform_params', 'transform_from_params', 'parks_moore_w', 'magnitude_decades',
           'antigen_dash_labels', 'default_population', 'SKIP_CHANNELS']

SKIP_CHANNELS = frozenset({'ribbon', 'Time', 'event_id'})
TRANSFORM_KEYS = ('id', 'scale_t', 'linear_a', 'logicle_w', 'logicle_m', 'logicle_a',
                  'log_m', 'limits')
_ZOOM_RATE = 1.04

# ---------------------------------------------------------------------------
# Deferred Qt imports: pyqtgraph and the cytometry plot components create Qt
# objects on import, so they are imported on the main thread by
# ensure_qt_imports() (called from each widget's constructor).
# ---------------------------------------------------------------------------
cc = None
pg = None
ZoomAxis = None
NoPanViewBox = None
TransparentGraphicsLayoutWidget = None
InteractiveLabel = None
HelpToggleWidget = None
_qt_imports_done = False


def ensure_qt_imports():
    """Import pyqtgraph, colorcet and the plot components (main thread only)."""
    global cc, pg, ZoomAxis, NoPanViewBox, TransparentGraphicsLayoutWidget, InteractiveLabel
    global HelpToggleWidget, _qt_imports_done
    if _qt_imports_done:
        return
    import colorcet as _cc
    import pyqtgraph as _pg
    from honeychrome.view_components.cytometry_plot_components import (
        ZoomAxis as _ZA, NoPanViewBox as _NPV,
        TransparentGraphicsLayoutWidget as _TGLW, InteractiveLabel as _IL,
    )
    from honeychrome.view_components.help_toggle_widget import HelpToggleWidget as _HTW
    cc, pg = _cc, _pg
    ZoomAxis, NoPanViewBox, TransparentGraphicsLayoutWidget, InteractiveLabel = _ZA, _NPV, _TGLW, _IL
    HelpToggleWidget = _HTW
    _qt_imports_done = True


# ---------------------------------------------------------------------------
# Transform helpers (no Qt)
# ---------------------------------------------------------------------------

def transform_params(transform) -> dict:
    """JSON-safe parameters of a Honeychrome Transform object."""
    out = {}
    for key in TRANSFORM_KEYS:
        value = getattr(transform, key, None)
        if isinstance(value, np.generic):
            value = value.item()
        if key == 'limits' and value is not None:
            value = [float(v) for v in value]
        out[key] = value
    return out


def transform_from_params(params: dict, template=None):
    """A Honeychrome Transform with *params*. *template*, when given, is
    copied first, so anything the parameters do not cover is kept."""
    from honeychrome.controller_components.transform import Transform
    tr = deepcopy(template) if template is not None else Transform()
    for key in ('scale_t', 'linear_a', 'logicle_w', 'logicle_m', 'logicle_a', 'log_m'):
        if params.get(key) is not None:
            setattr(tr, key, params[key])
    limits = params.get('limits')
    tr.set_transform(id=params.get('id', getattr(tr, 'id', None)),
                     limits=list(limits) if limits is not None else list(tr.limits))
    return tr


def magnitude_decades(controller) -> float:
    """Logicle M (decades) from the cytometer's range: log10(ceiling) − 1,
    so the top-of-scale decade is excluded as in the Gating-ML spec (a
    1 × 10⁶ instrument gives M = 5)."""
    try:
        ceiling = controller.experiment.settings['raw']['magnitude_ceiling']
        if ceiling and ceiling > 0:
            return math.log10(float(ceiling)) - 1.0
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    return 4.5


def parks_moore_w(arrays, top: float, decades: float) -> float:
    """Logicle width basis for one channel (Parks, Roederer & Moore 2006):
    W = (M − log10(T / |q|)) / 2, with q the 25th percentile of the negative
    events pooled over *arrays*; 0.5 without negative events. Clamped to
    [0, M/2]."""
    negatives = [np.asarray(a)[np.asarray(a) < 0] for a in arrays]
    negatives = [a for a in negatives if len(a)]
    if not negatives:
        return 0.5
    q = float(np.percentile(np.concatenate(negatives), 25))
    try:
        w = (decades - math.log10(top / abs(q))) / 2.0
    except (ValueError, ZeroDivisionError):
        w = 0.5
    return max(0.0, min(decades / 2.0, w))


def antigen_dash_labels(controller) -> dict:
    """{channel: 'Antigen - Channel'}, or the channel name without an antigen."""
    try:
        pnn = controller.experiment.settings.get('unmixed', {}).get('event_channels_pnn') or []
        spectral_model = controller.experiment.process.get('spectral_model') or []
    except (AttributeError, KeyError):
        return {}
    antigen_of = {c.get('label'): (c.get('antigen') or '') for c in spectral_model}
    return {ch: (f"{antigen_of[ch]} - {ch}" if antigen_of.get(ch) else ch) for ch in pnn}


def default_population(gate_names) -> str:
    """The population a transform view opens on: Singlets, else Cells, else
    the root (names matched without regard to case)."""
    names = list(gate_names or [])
    lower = {n.lower(): n for n in names}
    for wanted in ('singlets', 'single cells'):
        if wanted in lower:
            return lower[wanted]
    for n in names:
        if 'singlet' in n.lower():
            return n
    if 'cells' in lower:
        return lower['cells']
    for n in names:
        if 'cells' in n.lower():
            return n
    return 'root'


class TransformSet:
    """A plugin's own axis transforms: {channel: Honeychrome Transform}.

    The objects are edited in place by the views that show them; ``params``
    and ``load`` convert to and from JSON-safe parameter dicts.
    """

    def __init__(self):
        self.objects: dict = {}

    def params(self) -> dict:
        return {ch: transform_params(tr) for ch, tr in self.objects.items()}

    def load(self, params: dict | None, fallback: dict):
        """Rebuild from *params* ({channel: parameter dict}); channels without
        parameters are copied from *fallback* ({channel: Transform}, e.g. the
        experiment's transforms)."""
        params = params or {}
        out = {}
        for ch, tr in (fallback or {}).items():
            if ch in params and params[ch]:
                try:
                    out[ch] = transform_from_params(params[ch], template=tr)
                    continue
                except Exception:
                    pass
            out[ch] = deepcopy(tr)
        for ch, p in params.items():
            if ch not in out and p:
                try:
                    out[ch] = transform_from_params(p)
                except Exception:
                    continue
        self.objects = out

    def set_channel(self, channel: str, params: dict):
        """Replace one channel's transform by *params*, in place."""
        tr = self.objects.get(channel)
        if tr is None:
            self.objects[channel] = transform_from_params(params)
            return
        for key in ('scale_t', 'linear_a', 'logicle_w', 'logicle_m', 'logicle_a', 'log_m'):
            if params.get(key) is not None:
                setattr(tr, key, params[key])
        limits = params.get('limits')
        tr.set_transform(id=params.get('id', tr.id),
                         limits=list(limits) if limits is not None else list(tr.limits))


def _lookup_table():
    """RGBA lookup table of the app's density colour map, zero transparent."""
    try:
        colors = cc.palette[hc_settings.colourmap_name_retrieved]
    except Exception:
        colors = cc.palette['bjy']
    cmap = pg.ColorMap(pos=0.9 * np.linspace(0, 1, len(colors)) ** 2
                       + 0.1 * np.linspace(0, 1, len(colors)), color=colors)
    lut = cmap.getLookupTable(alpha=True)
    lut[0, 3] = 0
    return lut


# ---------------------------------------------------------------------------
# Biplot tile
# ---------------------------------------------------------------------------

class BiplotTile(QWidget):
    """
    A single square biplot tile — heatmap of x_channel vs y_channel.
    x_channel is always the inspector's active channel (set externally).
    y_channel is chosen by clicking the y-axis InteractiveLabel (same
    mechanism as CytometryPlotWidget). Drag the axes to adjust W / zoom,
    exactly as in CytometryPlotWidget.
    """

    # InteractiveLabel.mousePressEvent calls parent_plot.select_plot_on_parent_grid().
    def select_plot_on_parent_grid(self):
        pass

    def __init__(self, rgba_lut, parent_tab, y_channel: str = '', parent=None):
        super().__init__(parent)
        self._parent_tab = parent_tab
        self._y_channel = y_channel
        self._rgba_lut = rgba_lut

        size = hc_settings.cytometry_plot_width_target_retrieved
        self.setMinimumSize(size, size)
        self.setMaximumSize(size, size)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        # The y-channel store, driven by the InteractiveLabel click menu.
        self._y_combo = QComboBox()
        self._y_combo.setMaximumHeight(22)
        self._y_combo.setVisible(False)
        self._y_combo.currentTextChanged.connect(self._on_y_changed)
        outer.addWidget(self._y_combo)

        self._gw = TransparentGraphicsLayoutWidget()
        self._gw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gl = self._gw.ci.layout
        gl.setHorizontalSpacing(0)
        gl.setVerticalSpacing(0)

        self._label_y = InteractiveLabel('', parent_plot=self, angle=-90, size='9pt')
        self._label_y.leftClickMenuFunction = self._set_y_from_label
        self._gw.addItem(self._label_y, row=0, col=0)

        self._vb = NoPanViewBox()
        self._gw.addItem(self._vb, row=0, col=2)

        self._axis_x = ZoomAxis('bottom', self._vb)
        self._axis_y = ZoomAxis('left', self._vb)
        self._gw.addItem(self._axis_x, row=1, col=2)
        self._gw.addItem(self._axis_y, row=0, col=1)
        self._axis_x.linkToView(self._vb)
        self._axis_y.linkToView(self._vb)

        # x is shared by every tile; the menu routes through the inspector.
        self._label_x = InteractiveLabel('', parent_plot=self, size='9pt')
        self._label_x.leftClickMenuFunction = self._set_x_from_label
        self._gw.addItem(self._label_x, row=2, col=2)

        self._vb.setMouseEnabled(x=False, y=False)
        self._vb.raiseContextMenu = lambda ev: None

        self._img = pg.ImageItem()
        self._img.setLookupTable(rgba_lut)
        self._vb.addItem(self._img)

        self._axis_x.zoom_timer.timeout.connect(lambda: self._parent_tab._apply_tile_zoom(self, 'x'))
        self._axis_y.zoom_timer.timeout.connect(lambda: self._parent_tab._apply_tile_zoom(self, 'y'))

        outer.addWidget(self._gw)

    def _set_y_from_label(self, item_index, _parent):
        channels = [self._y_combo.itemText(i) for i in range(self._y_combo.count())]
        if 0 <= item_index < len(channels):
            self._label_y.leftItemSelected = item_index
            self._y_combo.setCurrentText(channels[item_index])

    def _set_x_from_label(self, item_index, _parent):
        channels = [self._y_combo.itemText(i) for i in range(self._y_combo.count())]
        if 0 <= item_index < len(channels):
            self._parent_tab._set_active_channel_from_tile(channels[item_index])

    def set_channels(self, available: list[str], y_channel: str = ''):
        self._y_combo.blockSignals(True)
        current = y_channel or self._y_combo.currentText()
        self._y_combo.clear()
        self._y_combo.addItems(available)
        if current in available:
            self._y_combo.setCurrentText(current)
        self._y_combo.blockSignals(False)
        self._y_channel = self._y_combo.currentText()

        # Menus show antigen labels; the selection is by index.
        labels = self._parent_tab.channel_labels()
        self._label_y.leftClickMenuItems = [labels.get(ch, ch) for ch in available]
        ch = self._y_combo.currentText()
        self._label_y.leftItemSelected = available.index(ch) if ch in available else 0
        self._label_x.leftClickMenuItems = [labels.get(ch, ch) for ch in available]
        x_ch = self._parent_tab._active_channel
        self._label_x.leftItemSelected = available.index(x_ch) if x_ch in available else 0

    def y_channel(self) -> str:
        return self._y_combo.currentText()

    def set_x_selected(self, x_ch: str):
        """Highlight the inspector's active channel in the x-axis menu."""
        channels = [self._y_combo.itemText(i) for i in range(self._y_combo.count())]
        if x_ch in channels:
            self._label_x.leftItemSelected = channels.index(x_ch)

    def _on_y_changed(self, ch: str):
        self._y_channel = ch
        labels = self._parent_tab.channel_labels()
        self._label_y.setText(labels.get(ch, ch))
        channels = [self._y_combo.itemText(i) for i in range(self._y_combo.count())]
        if ch in channels:
            self._label_y.leftItemSelected = channels.index(ch)
        self._parent_tab._configure_tile_axes(self)
        self._parent_tab._draw_tile(self)

    def configure_axes(self, tr_x, tr_y, x_label: str = '', y_label: str = ''):
        """Apply transform ticks/limits and axis labels (ticks before ranges,
        so real-value labels appear at once)."""
        if x_label:
            self._label_x.setText(x_label)
        if y_label:
            self._label_y.setText(y_label)
            self._y_channel = y_label

        if tr_x.ticks:
            self._axis_x.setTicks(tr_x.ticks())
        self._axis_x.zoomZero = tr_x.zero
        self._axis_x.fullRange = (0, 1.1)
        self._axis_x.limits = tuple(tr_x.limits)
        self._vb.setXRange(tr_x.limits[0], tr_x.limits[1], padding=0)

        if tr_y.ticks:
            self._axis_y.setTicks(tr_y.ticks())
        self._axis_y.zoomZero = tr_y.zero
        self._axis_y.fullRange = (0, 1.1)
        self._axis_y.limits = tuple(tr_y.limits)
        self._vb.setYRange(tr_y.limits[0], tr_y.limits[1], padding=0)

    def draw(self, heatmap, tr_x, tr_y):
        from PySide6.QtCore import QRectF
        self._img.setImage(heatmap)
        self._img.setRect(QRectF(tr_x.limits[0], tr_y.limits[0],
                                 tr_x.limits[1] - tr_x.limits[0],
                                 tr_y.limits[1] - tr_y.limits[0]))

    def clear(self):
        self._img.clear()


# ---------------------------------------------------------------------------
# Inspector
# ---------------------------------------------------------------------------

class TransformInspector(QWidget):
    """
    Channel list, 1-D histogram and biplots of a set of transforms.

    The transforms (``self._transforms``, {channel: Transform}) are edited
    in place. Subclasses supply the events (``self._raw_data``, {channel:
    array}), widgets above the toolbar (``_build_selection``), extra toolbar
    buttons (``_extend_toolbar``), the events used by Auto-transform
    (``auto_transform_data``) and what follows a change
    (``_on_transform_edited``, which by default emits
    ``transformsChanged(channel, parameters before)``).
    """

    transformsChanged = Signal(str, object)

    _XFORM_LABELS = {0: 'Linear', 1: 'Logicle', 2: 'Log', 'default': 'Default'}

    def __init__(self, controller, help_text: str = '', parent=None):
        super().__init__(parent)
        ensure_qt_imports()
        self.controller = controller
        self._help_text = help_text
        self._active_channel: str | None = None
        self._transforms: dict = {}
        self._raw_data: dict = {}
        self._pnn: list[str] = []

        # Redraws 200 ms after the last change.
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(200)
        self._redraw_timer.timeout.connect(self._redraw_plots)

        self._rgba_lut = _lookup_table()
        self._build_ui()

    # -- hooks ------------------------------------------------------------

    def _build_selection(self, layout: QVBoxLayout):
        """Add widgets above the toolbar (e.g. the population or gates shown)."""

    def _extend_toolbar(self, toolbar: QHBoxLayout):
        """Add buttons at the start of the toolbar."""

    def channel_labels(self) -> dict:
        return antigen_dash_labels(self.controller)

    def auto_transform_data(self):
        """({channel: [arrays]}, description of their source) for
        Auto-transform, or (None, '') when there is nothing to use."""
        if not self._raw_data:
            return None, ''
        return {ch: [arr] for ch, arr in self._raw_data.items()}, "the events shown"

    def _on_transform_edited(self, channel: str, tr, before: dict):
        """A channel's transform changed (axis drag, Auto-transform, CSV)."""
        self.transformsChanged.emit(channel, before)

    def _after_auto_transform(self, n_updated: int, source_note: str):
        QMessageBox.information(
            self, "Auto-Transform",
            f"W set for {n_updated} logicle channel(s) from {source_note} "
            "(Parks, Roederer & Moore: from the lower quartile of the negative events).")

    def magnitude_decades(self) -> float:
        return magnitude_decades(self.controller)

    def _available_channels(self) -> list[str]:
        return [ch for ch in self._transforms if ch not in SKIP_CHANNELS]

    # -- UI ---------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        if self._help_text:
            self.help_widget = HelpToggleWidget(text=self._help_text)
            outer.addWidget(self.help_widget)

        # Selection, controls and plots flow down the left; the channel list
        # sits at full height on the right.
        body = QHBoxLayout()
        body.setSpacing(8)
        outer.addLayout(body, stretch=1)
        main_col = QVBoxLayout()
        main_col.setSpacing(6)
        body.addLayout(main_col, stretch=1)

        self._build_selection(main_col)

        toolbar = QHBoxLayout()
        self._extend_toolbar(toolbar)
        self.auto_btn = QPushButton("✦ Auto-transform (all channels)")
        self.auto_btn.setToolTip(
            "Set the logicle width basis W of every logicle channel from the events "
            "(Parks, Roederer & Moore 2006).")
        self.auto_btn.clicked.connect(self._auto_transform_all)
        toolbar.addWidget(self.auto_btn)
        self.save_csv_btn = QPushButton("⬇ Save CSV")
        self.save_csv_btn.setToolTip("Export the transform parameters to a CSV file.\n"
                                     "Columns: channel, id, T, W, M, A")
        self.save_csv_btn.clicked.connect(self._save_transforms_csv)
        toolbar.addWidget(self.save_csv_btn)
        self.load_csv_btn = QPushButton("⬆ Load CSV")
        self.load_csv_btn.setToolTip(
            "Import transform parameters from a CSV file.\n"
            "Channel names are validated against the current experiment.\n"
            "Unrecognised channels are skipped; missing channels keep their\n"
            "current parameters.")
        self.load_csv_btn.clicked.connect(self._load_transforms_csv)
        toolbar.addWidget(self.load_csv_btn)
        toolbar.addStretch()
        main_col.addLayout(toolbar)

        plots_scroll = QScrollArea()
        plots_scroll.setWidgetResizable(True)
        plots_scroll.setFrameShape(QFrame.NoFrame)
        plots_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.plots_widget = QWidget()
        self.plots_layout = QVBoxLayout(self.plots_widget)
        self.plots_layout.setContentsMargins(0, 0, 0, 0)
        self.plots_layout.setSpacing(8)
        plots_scroll.setWidget(self.plots_widget)
        main_col.addWidget(plots_scroll, stretch=1)

        channels_col = QVBoxLayout()
        channels_col.setContentsMargins(0, 0, 0, 0)
        channels_col.addWidget(QLabel("Channels"))
        self.channel_list = QListWidget()
        self.channel_list.setFixedWidth(180)
        self.channel_list.currentItemChanged.connect(self._on_channel_item_changed)
        channels_col.addWidget(self.channel_list, stretch=1)
        self.xform_type_label = QLabel("—")
        self.xform_type_label.setStyleSheet("font-style: italic; color: grey; padding: 4px;")
        channels_col.addWidget(self.xform_type_label)
        body.addLayout(channels_col)

        # 1-D histogram, built once and updated in place.
        self.plots_layout.addWidget(QLabel("1-D Histogram  (drag axis to adjust transform)"))
        self._hist_gw = TransparentGraphicsLayoutWidget()
        self._hist_gw.setMinimumHeight(180)
        self._hist_gw.setMaximumHeight(280)
        self._hist_gw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        gl = self._hist_gw.ci.layout
        gl.setHorizontalSpacing(0)
        gl.setVerticalSpacing(0)
        self._hist_vb = NoPanViewBox()
        self._hist_vb.enableAutoRange(axis=self._hist_vb.YAxis, enable=True)
        self._hist_gw.addItem(self._hist_vb, row=0, col=1)
        self._hist_axis_x = ZoomAxis('bottom', self._hist_vb)
        self._hist_axis_y = ZoomAxis('left', self._hist_vb)
        self._hist_gw.addItem(self._hist_axis_x, row=1, col=1)
        self._hist_gw.addItem(self._hist_axis_y, row=0, col=0)
        self._hist_axis_x.linkToView(self._hist_vb)
        self._hist_axis_y.linkToView(self._hist_vb)
        self._hist_vb.setMouseEnabled(x=False, y=False)
        self._hist_vb.raiseContextMenu = lambda ev: None
        self._hist_curve = pg.PlotDataItem(stepMode='center', fillLevel=0,
                                           brush=(100, 100, 250, 150))
        self._hist_vb.addItem(self._hist_curve)
        self._hist_axis_x.zoom_timer.timeout.connect(self._apply_hist_zoom)
        self._hist_axis_y.zoom_timer.timeout.connect(lambda: None)  # y auto-ranges
        self.plots_layout.addWidget(self._hist_gw)

        # Biplots: x is the active channel; y is chosen per tile.
        biplot_header = QHBoxLayout()
        biplot_header.addWidget(QLabel("Biplots  (drag axes to adjust transforms)"))
        biplot_header.addStretch()
        add_btn = QPushButton("＋ Add biplot")
        add_btn.setFixedHeight(24)
        add_btn.clicked.connect(self._add_biplot)
        biplot_header.addWidget(add_btn)
        remove_btn = QPushButton("－ Remove last")
        remove_btn.setFixedHeight(24)
        remove_btn.clicked.connect(self._remove_biplot)
        biplot_header.addWidget(remove_btn)
        self.plots_layout.addLayout(biplot_header)

        self._biplot_scroll = QScrollArea()
        self._biplot_scroll.setWidgetResizable(True)
        self._biplot_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._biplot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._biplot_scroll.setMinimumHeight(hc_settings.cytometry_plot_width_target_retrieved + 40)
        self._biplot_scroll.setMaximumHeight(2 * hc_settings.cytometry_plot_width_target_retrieved + 50)
        self._biplot_container = QWidget()
        self._biplot_grid = QGridLayout(self._biplot_container)
        self._biplot_grid.setContentsMargins(0, 0, 0, 0)
        self._biplot_grid.setSpacing(4)
        self._biplot_scroll.setWidget(self._biplot_container)
        self.plots_layout.addWidget(self._biplot_scroll)

        # Tiles reflow when the viewport width changes (debounced).
        self._tile_relayout_timer = QTimer(self)
        self._tile_relayout_timer.setSingleShot(True)
        self._tile_relayout_timer.setInterval(150)
        self._tile_relayout_timer.timeout.connect(self._relayout_tiles)
        self._biplot_scroll.viewport().installEventFilter(self)

        self._biplot_tiles: list[BiplotTile] = []
        self.plots_layout.addStretch()

    # -- channels ---------------------------------------------------------

    def _channel_order(self) -> list[str]:
        """Channels in the cytometer's PNN order, then any others."""
        pnn_order = [ch for ch in self._pnn if ch in self._transforms and ch not in SKIP_CHANNELS]
        rest = [ch for ch in self._transforms if ch not in SKIP_CHANNELS and ch not in pnn_order]
        return pnn_order + rest

    def show_channels(self):
        """Fill the channel list and the tiles' menus, keeping the selection."""
        channels = self._channel_order()
        self.channel_list.blockSignals(True)
        prev = self.channel_list.currentItem()
        prev_channel = prev.data(Qt.UserRole) if prev else None
        self.channel_list.clear()
        labels = self.channel_labels()
        for ch in channels:
            item = QListWidgetItem(labels.get(ch, ch))
            item.setData(Qt.UserRole, ch)
            self.channel_list.addItem(item)
        self.channel_list.blockSignals(False)
        if not self._biplot_tiles:
            self._add_biplot()
        for tile in self._biplot_tiles:
            tile.set_channels(channels)
        restored = False
        if prev_channel and prev_channel in channels:
            for i in range(self.channel_list.count()):
                if self.channel_list.item(i).data(Qt.UserRole) == prev_channel:
                    self.channel_list.setCurrentRow(i)
                    restored = True
                    break
        if not restored and channels:
            self.channel_list.setCurrentRow(0)

    def _on_channel_item_changed(self, current, previous):
        channel = current.data(Qt.UserRole) if current else None
        self._on_channel_selected(channel)

    def _on_channel_selected(self, channel: str):
        if not channel or channel not in self._transforms:
            return
        self._active_channel = channel
        tr = self._transforms[channel]
        self.xform_type_label.setText(self._XFORM_LABELS.get(tr.id, str(tr.id)))
        self._configure_hist_axes()
        self._configure_tile_axes_all()
        self._schedule_redraw()

    def _set_active_channel_from_tile(self, channel: str):
        """A tile's x-axis label was clicked: select that channel in the list."""
        for i in range(self.channel_list.count()):
            if self.channel_list.item(i).data(Qt.UserRole) == channel:
                self.channel_list.setCurrentRow(i)
                return

    # -- tiles -----------------------------------------------------------

    def eventFilter(self, obj, event):
        if obj is self._biplot_scroll.viewport() and event.type() == QEvent.Resize:
            if event.oldSize().width() != event.size().width():
                self._tile_relayout_timer.start()
        return super().eventFilter(obj, event)

    def _relayout_tiles(self):
        """Place the tiles in the grid, wrapping to the available width."""
        for i in range(self._biplot_grid.count()):
            item = self._biplot_grid.itemAt(i)
            if item and item.widget():
                self._biplot_grid.removeWidget(item.widget())
        tile_size = hc_settings.cytometry_plot_width_target_retrieved
        available_w = self._biplot_scroll.viewport().width() or (tile_size + 4)
        cols = max(1, available_w // (tile_size + 4))
        for n, tile in enumerate(self._biplot_tiles):
            row, col = divmod(n, cols)
            self._biplot_grid.addWidget(tile, row, col)
        n_rows = max(1, (len(self._biplot_tiles) + cols - 1) // cols) if self._biplot_tiles else 1
        self._biplot_scroll.setMinimumHeight(min(n_rows, 2) * (tile_size + 30) + 10)

    def _add_biplot(self):
        channels = self._available_channels()
        used_y = {t.y_channel() for t in self._biplot_tiles}
        x_ch = self._active_channel or (channels[0] if channels else '')
        candidates = [ch for ch in channels if ch != x_ch and ch not in used_y]
        default_y = candidates[0] if candidates else (
            channels[1] if len(channels) > 1 else (channels[0] if channels else ''))
        tile = BiplotTile(self._rgba_lut, parent_tab=self)
        tile.set_channels(channels, y_channel=default_y)
        self._biplot_tiles.append(tile)
        self._relayout_tiles()
        self._configure_tile_axes(tile)
        self._draw_tile(tile)

    def _remove_biplot(self):
        if not self._biplot_tiles:
            return
        tile = self._biplot_tiles.pop()
        self._biplot_grid.removeWidget(tile)
        tile.deleteLater()
        self._relayout_tiles()

    def _configure_tile_axes(self, tile: BiplotTile):
        x_ch = self._active_channel
        y_ch = tile.y_channel()
        if not x_ch or not y_ch or x_ch not in self._transforms or y_ch not in self._transforms:
            return
        labels = self.channel_labels()
        tile.configure_axes(self._transforms[x_ch], self._transforms[y_ch],
                            x_label=labels.get(x_ch, x_ch), y_label=labels.get(y_ch, y_ch))
        tile.set_x_selected(x_ch)

    def _configure_tile_axes_all(self):
        for tile in self._biplot_tiles:
            self._configure_tile_axes(tile)

    def _configure_hist_axes(self):
        """Ticks and limits of the histogram's x-axis (ticks before range)."""
        ch = self._active_channel
        if not ch or ch not in self._transforms:
            return
        tr = self._transforms[ch]
        if tr.ticks is None:
            return
        ticks = tr.ticks()
        if ticks:
            self._hist_axis_x.setTicks(ticks)
        self._hist_axis_x.zoomZero = tr.zero
        self._hist_axis_x.fullRange = (0, 1.1)
        self._hist_axis_x.limits = tuple(tr.limits)
        self._hist_vb.setXRange(tr.limits[0], tr.limits[1], padding=0)

    def _draw_tile(self, tile: BiplotTile):
        x_ch = self._active_channel
        y_ch = tile.y_channel()
        if (not x_ch or not y_ch or x_ch == y_ch
                or x_ch not in self._transforms or y_ch not in self._transforms
                or x_ch not in self._raw_data or y_ch not in self._raw_data):
            tile.clear()
            return
        tr_x, tr_y = self._transforms[x_ch], self._transforms[y_ch]
        try:
            heatmap, _, _ = np.histogram2d(self._raw_data[x_ch], self._raw_data[y_ch],
                                           bins=[tr_x.scale, tr_y.scale])
        except Exception:
            tile.clear()
            return
        try:
            cutoff = hc_settings.density_cutoff_retrieved
        except Exception:
            cutoff = 1
        heatmap[heatmap < cutoff] = 0
        tile.draw(heatmap, tr_x, tr_y)

    # -- axis drags -------------------------------------------------------

    def _zoom(self, axis, vb, tr, axis_name: str, set_range) -> bool:
        """Apply a pending axis drag to *tr*; True when it was applied."""
        if axis._pending_delta == 0:
            return False
        step = axis._pending_delta
        axis._pending_delta = 0
        if abs(step) < 1:
            return False
        factor = (1 / _ZOOM_RATE) if step > 0 else _ZOOM_RATE
        if axis_name == 'y':
            factor = 1 / factor
        idx = 0 if axis_name == 'x' else 1
        vmin, vmax = vb.viewRange()[idx]
        if axis.initial_pos:
            mapped = vb.mapToView(axis.initial_pos)
            map_pos = mapped.x() if axis_name == 'x' else mapped.y()
        else:
            map_pos = 0.5
        if tr.id == 1 and map_pos < 0.5 * vmax:
            tr.logicle_w = max(0.01, tr.logicle_w / factor)
            tr.set_transform()
        else:
            new_max = (vmax - axis.zoomZero) * factor + axis.zoomZero
            new_min = (vmin - axis.zoomZero) * factor + axis.zoomZero
            if new_max < axis.fullRange[1] * 1.01:
                set_range(new_min, new_max, padding=0)
            axis.limits = (new_min, new_max)
            tr.set_transform(limits=list(axis.limits))
        axis.zoomZero = tr.zero
        if tr.ticks:
            axis.setTicks(tr.ticks())
        return True

    def _apply_tile_zoom(self, tile: BiplotTile, axis_name: str):
        """Axis drag on a tile — as CytometryPlotWidget.apply_zoom."""
        if axis_name == 'x':
            axis, channel, set_range = tile._axis_x, self._active_channel, tile._vb.setXRange
        else:
            axis, channel, set_range = tile._axis_y, tile.y_channel(), tile._vb.setYRange
        if not channel or channel not in self._transforms:
            axis._pending_delta = 0
            return
        tr = self._transforms[channel]
        before = transform_params(tr)
        if not self._zoom(axis, tile._vb, tr, axis_name, set_range):
            return
        for other in self._biplot_tiles:
            if other is not tile:
                self._configure_tile_axes(other)
        if channel == self._active_channel:
            self._configure_hist_axes()
        self._on_transform_edited(channel, tr, before)
        self._schedule_redraw()

    def _apply_hist_zoom(self):
        ch = self._active_channel
        if not ch or ch not in self._transforms:
            self._hist_axis_x._pending_delta = 0
            return
        tr = self._transforms[ch]
        before = transform_params(tr)
        if not self._zoom(self._hist_axis_x, self._hist_vb, tr, 'x', self._hist_vb.setXRange):
            return
        self._on_transform_edited(ch, tr, before)
        self._configure_tile_axes_all()
        self._schedule_redraw()

    # -- drawing ----------------------------------------------------------

    def _schedule_redraw(self):
        self._redraw_timer.start()

    def _redraw_plots(self):
        self._draw_histogram()
        for tile in self._biplot_tiles:
            self._draw_tile(tile)

    def _draw_histogram(self):
        ch = self._active_channel
        if not ch or ch not in self._transforms or ch not in self._raw_data:
            self._hist_curve.setData([], [])
            return
        tr = self._transforms[ch]
        try:
            count, _ = np.histogram(self._raw_data[ch], bins=tr.scale)
        except Exception:
            self._hist_curve.setData([], [])
            return
        self._hist_curve.setData(tr.step_scale, count)
        self._configure_hist_axes()

    def redraw_all(self):
        """Axes, histogram and tiles after the transforms were replaced."""
        if self._active_channel in self._transforms:
            tr = self._transforms[self._active_channel]
            self.xform_type_label.setText(self._XFORM_LABELS.get(tr.id, str(tr.id)))
        self._configure_hist_axes()
        self._configure_tile_axes_all()
        self._schedule_redraw()

    # -- CSV --------------------------------------------------------------

    def _save_transforms_csv(self):
        """Write channel, id, T, W, M, A for every channel (id: 0 linear,
        1 logicle, 2 log)."""
        if not self._transforms:
            QMessageBox.information(self, "Save Transforms", "No transforms loaded yet.")
            return
        default_dir = str(getattr(self.controller, 'experiment_dir', ''))
        path, _ = QFileDialog.getSaveFileName(self, "Save Transforms CSV", default_dir,
                                              "CSV files (*.csv);;All files (*)")
        if not path:
            return
        if not path.lower().endswith('.csv'):
            path += '.csv'
        rows = [{'channel': ch, 'id': tr.id, 'T': tr.scale_t, 'W': round(tr.logicle_w, 6),
                 'M': round(tr.logicle_m, 6), 'A': round(tr.logicle_a, 6)}
                for ch, tr in self._transforms.items() if ch not in SKIP_CHANNELS]
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['channel', 'id', 'T', 'W', 'M', 'A'])
                writer.writeheader()
                writer.writerows(rows)
            QMessageBox.information(self, "Save Transforms", f"Saved {len(rows)} channels to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _load_transforms_csv(self):
        """Read transform parameters from a CSV file (columns channel and W
        required; T, M, A and id optional). Unknown channels are reported
        and skipped; W is clamped to [0, M/2]."""
        default_dir = str(getattr(self.controller, 'experiment_dir', ''))
        path, _ = QFileDialog.getOpenFileName(self, "Load Transforms CSV", default_dir,
                                              "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            with open(path, newline='', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Could not read CSV:\n{e}")
            return
        if not rows:
            QMessageBox.warning(self, "Load Transforms", "CSV file is empty.")
            return
        required = {'channel', 'W'}
        if not required.issubset(set(rows[0].keys())):
            QMessageBox.critical(self, "Load Transforms",
                                 f"CSV must contain at least columns: {sorted(required)}\n"
                                 f"Found: {sorted(rows[0].keys())}")
            return
        updated, skipped_unknown, skipped_invalid = [], [], []
        for row in rows:
            ch = (row.get('channel') or '').strip()
            if not ch:
                continue
            if ch not in self._transforms:
                skipped_unknown.append(ch)
                continue
            tr = self._transforms[ch]
            before = transform_params(tr)
            try:
                w = float(row['W'])
            except (ValueError, KeyError):
                skipped_invalid.append(f"{ch} (W invalid)")
                continue
            if (row.get('T') or '').strip():
                try:
                    t = float(row['T'])
                    if t > 0:
                        tr.scale_t = t
                except ValueError:
                    pass
            if (row.get('M') or '').strip():
                try:
                    m = float(row['M'])
                    if m > 0:
                        tr.logicle_m = m
                except ValueError:
                    pass
            if (row.get('A') or '').strip():
                try:
                    tr.logicle_a = float(row['A'])
                except ValueError:
                    pass
            if (row.get('id') or '').strip():
                try:
                    tr.id = int(row['id'])
                except ValueError:
                    pass
            tr.logicle_w = round(max(0.0, min(tr.logicle_m / 2.0, w)), 6)
            try:
                tr.set_transform()
            except Exception as e:
                skipped_invalid.append(f"{ch} (set_transform failed: {e})")
                continue
            self._on_transform_edited(ch, tr, before)
            updated.append(ch)
        parts = [f"Updated {len(updated)} channel(s)."]
        if skipped_unknown:
            parts.append(f"\nSkipped {len(skipped_unknown)} unrecognised channel(s):\n"
                         + ", ".join(skipped_unknown[:10]) + (" …" if len(skipped_unknown) > 10 else ""))
        if skipped_invalid:
            parts.append(f"\nSkipped {len(skipped_invalid)} invalid row(s):\n"
                         + ", ".join(skipped_invalid[:10]))
        QMessageBox.information(self, "Load Transforms", "".join(parts))
        if updated:
            self.redraw_all()

    # -- Auto-transform ---------------------------------------------------

    def _auto_transform_all(self):
        """Set W of every logicle channel from auto_transform_data()."""
        data, source_note = self.auto_transform_data()
        if data is None:
            return
        decades = self.magnitude_decades()
        n_updated = 0
        for ch, tr in self._transforms.items():
            if tr.id != 1:
                continue
            arrays = data.get(ch) or []
            if not arrays:
                continue
            before = transform_params(tr)
            tr.logicle_w = round(parks_moore_w(arrays, tr.scale_t, decades), 3)
            tr.set_transform()
            self._on_transform_edited(ch, tr, before)
            n_updated += 1
        self.redraw_all()
        self._after_auto_transform(n_updated, source_note)


# ---------------------------------------------------------------------------
# Population view for the gating plugins
# ---------------------------------------------------------------------------

class _PopulationLoader(QThread):
    """Load, unmix and gate a few samples; emits finished(ok, message,
    (events, n_per_sample)) with the events of one population pooled."""

    finished = Signal(bool, str, object)

    def __init__(self, experiment_dir, keys, snapshot, gating, pnn, population, per_sample,
                 seed: int = 42, parent=None):
        super().__init__(parent)
        self._dir = experiment_dir
        self._keys = list(keys)
        self._snap = snapshot
        self._gating = gating
        self._pnn = list(pnn)
        self._population = population
        self._per_sample = int(per_sample)
        self._seed = int(seed)

    def run(self):
        from honeychrome.controller_components.sample_loader import load_unmixed
        try:
            chunks, counts = [], {}
            for key in self._keys:
                out = load_unmixed(self._dir, key, self._snap, max_events=self._per_sample,
                                   seed=self._seed)
                events = np.asarray(out['unmixed'], dtype=np.float32)
                if self._population and self._population != 'root' and self._gating is not None:
                    mask = self._gating.population_mask(key, events, self._pnn, [self._population])
                    events = events[mask]
                chunks.append(events)
                counts[key] = int(len(events))
            pooled = np.concatenate(chunks, axis=0) if chunks else np.empty((0, len(self._pnn)),
                                                                            dtype=np.float32)
            self.finished.emit(True, '', (pooled, counts))
        except Exception as exc:
            self.finished.emit(False, str(exc), None)


class PopulationTransformsPanel(TransformInspector):
    """Transforms of a gating plugin, shown on one population.

    The population is a gate of the experiment's own gating hierarchy (the
    .kit gates of the Unmixed Data tab); its events are pooled from a few
    samples, each loaded with its own unmixing and gated with its own gates
    (per-sample overrides included). The transforms are the plugin's
    TransformSet, edited in place.

    sample_keys       : callable returning the samples to pool (the first
                        MAX_SAMPLES are used).
    reset_label/reset : text of the button that restores the default
                        transforms, and the callable that does it (the host
                        replaces the TransformSet's objects and calls
                        set_transforms).
    populationChanged(name) is emitted when the user picks a population.
    """

    populationChanged = Signal(str)

    MAX_SAMPLES = 6
    TARGET_EVENTS = 200_000
    MIN_EVENTS_PER_SAMPLE = 20_000

    def __init__(self, controller, transform_set: TransformSet, sample_keys, help_text: str = '',
                 reset_label: str = '', reset=None, parent=None):
        super().__init__(controller, help_text=help_text, parent=parent)
        self._set = transform_set
        self._sample_keys = sample_keys
        self._reset = reset
        self._population = 'root'
        self._loader = None
        self._wanted = None
        self._loaded = None
        self.reset_btn.setText(f"↺ {reset_label}")
        self.reset_btn.setVisible(reset is not None)

    def _build_selection(self, layout: QVBoxLayout):
        row = QHBoxLayout()
        row.addWidget(QLabel("Population:"))
        self.combo_population = QComboBox()
        self.combo_population.setMinimumWidth(200)
        self.combo_population.setToolTip(
            "Population of the experiment's gating hierarchy (the gates drawn on the "
            "Unmixed Data tab) whose events are shown and used by Auto-transform.")
        self.combo_population.currentIndexChanged.connect(self._on_population_picked)
        row.addWidget(self.combo_population)
        self.lbl_samples = QLabel("")
        self.lbl_samples.setStyleSheet("color: #555; font-style: italic;")
        self.lbl_samples.setWordWrap(True)
        row.addWidget(self.lbl_samples, stretch=1)
        self.btn_reload = QPushButton("Reload events")
        self.btn_reload.setToolTip("Load the samples' events again (after a change of samples "
                                   "or gates).")
        self.btn_reload.clicked.connect(lambda: self.load_events(force=True))
        row.addWidget(self.btn_reload)
        layout.addLayout(row)

    def _extend_toolbar(self, toolbar: QHBoxLayout):
        self.reset_btn = QPushButton("↺ Reset")
        self.reset_btn.clicked.connect(self._on_reset)
        toolbar.addWidget(self.reset_btn)

    def _on_reset(self):
        if self._reset is not None:
            self._reset()

    # -- host API ---------------------------------------------------------

    def set_transforms(self):
        """Show the TransformSet's current objects (after the host replaced them)."""
        self._transforms = self._set.objects
        self._pnn = list(self.controller.experiment.settings.get('unmixed', {})
                         .get('event_channels_pnn') or [])
        self.show_channels()
        self.redraw_all()

    def population(self) -> str:
        return self._population

    def refresh(self, population: str | None = None):
        """Fill the population list (selecting *population*, else the
        default), show the transforms and load the events when needed."""
        names = self._gate_names()
        wanted = population if population in names or population == 'root' else None
        if wanted is None:
            wanted = self._population if self._population in names else default_population(names)
        self.combo_population.blockSignals(True)
        self.combo_population.clear()
        self.combo_population.addItem("All events (root)", 'root')
        for name, depth in self._gate_names(with_depth=True):
            self.combo_population.addItem(("   " * depth) + name, name)
        i = self.combo_population.findData(wanted)
        self.combo_population.setCurrentIndex(max(i, 0))
        self.combo_population.blockSignals(False)
        self._population = self.combo_population.currentData() or 'root'
        self.set_transforms()
        self.load_events()

    def _gate_names(self, with_depth: bool = False):
        gating = getattr(self.controller, 'unmixed_gating', None)
        if gating is None:
            return []
        out = []
        try:
            for gid, path in gating.get_gate_ids():
                out.append((gid, max(0, len(path) - 1)))
        except Exception:
            return []
        return out if with_depth else [n for n, _d in out]

    def _on_population_picked(self, _index):
        self._population = self.combo_population.currentData() or 'root'
        self.populationChanged.emit(self._population)
        self.load_events(force=True)

    def load_events(self, force: bool = False):
        """Load the population's events from the samples in the background."""
        from honeychrome.controller_components.sample_loader import (
            analysis_sample_keys, snapshot_unmix_state,
        )
        from honeychrome.controller_components.per_sample_gating_functions import SampleGating
        keys = list(self._sample_keys() or [])
        if not keys:
            current = getattr(self.controller, 'current_sample_path', None)
            available = analysis_sample_keys(self.controller)
            keys = [k for k in available if current and str(current).endswith(str(k))][:1] \
                or available[:1]
        if len(keys) > self.MAX_SAMPLES:
            picks = np.linspace(0, len(keys) - 1, self.MAX_SAMPLES).round().astype(int)
            keys = [keys[i] for i in sorted(set(picks.tolist()))]
        request = (tuple(keys), self._population)
        if not keys:
            self.lbl_samples.setText("No samples to show.")
            return
        if not force and request == self._loaded:
            return
        self._wanted = request
        if self._loader is not None:
            return
        try:
            snap = snapshot_unmix_state(self.controller, keys)
            gating = SampleGating.from_controller(self.controller) \
                if self._population != 'root' else None
        except Exception as exc:
            self.lbl_samples.setText(f"Cannot load the samples: {exc}")
            return
        pnn = list(self.controller.experiment.settings.get('unmixed', {})
                   .get('event_channels_pnn') or [])
        per_sample = max(self.TARGET_EVENTS // len(keys), self.MIN_EVENTS_PER_SAMPLE)
        self.lbl_samples.setText(f"Loading {len(keys)} sample(s) …")
        loader = _PopulationLoader(self.controller.experiment_dir, keys, snap, gating, pnn,
                                   self._population, per_sample, parent=self)
        loader.finished.connect(lambda ok, msg, out, r=request: self._on_loaded(ok, msg, out, r))
        self._loader = loader
        loader.start()

    def _on_loaded(self, ok: bool, message: str, out, request):
        self._loader = None
        if self._wanted is not None and self._wanted != request:
            self.load_events(force=True)
            return
        if not ok:
            self.lbl_samples.setText(f"Could not load the samples: {message}")
            return
        events, counts = out
        self._loaded = request
        pnn = list(self.controller.experiment.settings.get('unmixed', {})
                   .get('event_channels_pnn') or [])
        self._raw_data = {ch: events[:, i] for i, ch in enumerate(pnn) if i < events.shape[1]}
        names = ", ".join(str(k).replace('\\', '/').rsplit('/', 1)[-1].rsplit('.', 1)[0]
                          for k in counts)
        self.lbl_samples.setText(f"{len(events):,} events of {request[1]} pooled from "
                                 f"{len(counts)} sample(s): {names}")
        self._schedule_redraw()

    def auto_transform_data(self):
        if not self._raw_data:
            QMessageBox.information(self, "Auto-Transform", "The events are still loading.")
            return None, ''
        return ({ch: [arr] for ch, arr in self._raw_data.items()},
                f"{self._population} ({self.lbl_samples.text().split(':')[0]})")

"""
autospectral_tab.py
--------------------
Honeychrome integrated tab: AutoSpectral AF Extraction.

Location: src/honeychrome/view_components/autospectral_tab.py

Sections
--------
1  Extract AF profile from an unstained sample (KMeans).
2  Manage stored AF profiles (spectral plot, CSV load/save/delete).
3  Assign AF profiles to samples (grid: rows=non-SSC samples, columns=profiles).
4  Side-by-side OLS vs AF-corrected biplot comparison, using the same
   density-heatmap rendering, logicle transforms, and axis-drag zoom as the
   Unmixed Data tab.
"""

import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import colorcet as cc
from PySide6.QtCore import Qt, QThread, Signal, QObject, QRectF, QSettings, QTimer, QEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel,
    QPushButton, QSpinBox, QComboBox, QGroupBox,
    QFormLayout, QSizePolicy, QFileDialog, QListWidget, QListWidgetItem,
    QAbstractItemView, QSplitter, QCheckBox, QGridLayout,
)
import pyqtgraph as pg
import warnings

from honeychrome.controller_components.functions import (
    sample_from_fcs,
    apply_transfer_matrix,
    build_display_label_map,
)
from honeychrome.controller_components.autospectral_functions import (
    get_af_spectra,
    precompute_af_matrices,
    apply_af_transfer,
    save_af_profile_csv,
    load_af_profile_csv,
)
from honeychrome.controller_components.transform import Transform
from honeychrome.view_components.cytometry_plot_components import (
    InteractiveLabel,
    NoPanViewBox,
    ZoomAxis,
    TransparentGraphicsLayoutWidget,
)
from honeychrome.view_components.profiles_viewer import BottomAxisVerticalTickLabels
from honeychrome.view_components.help_toggle_widget import WheelBlocker
from honeychrome.controller_components.cytometer_whitelist import (
    get_detector_laser_map, LASER_LABEL_COLORS,
)
import honeychrome.settings as settings

import logging

from honeychrome.view_components.help_texts import autospectral_af_help_text

logger = logging.getLogger(__name__)

TAB_NAME = 'AutoSpectral AF'


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class AfTrainingWorker(QObject):
    """Runs get_af_spectra in a QThread."""
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, unstained_raw, fluor_spectra, n_clusters, source_fcs_path):
        super().__init__()
        self.unstained_raw = unstained_raw
        self.fluor_spectra = fluor_spectra
        self.n_clusters = n_clusters
        self.source_fcs_path = source_fcs_path   # stored so the slot can read it

    def run(self):
        try:
            self.progress.emit('Fitting KMeans clusters to unstained sample...')
            af_spectra = get_af_spectra(
                self.unstained_raw,
                self.fluor_spectra,
                n_clusters=self.n_clusters,
            )
            self.finished.emit(af_spectra)
        except Exception as e:
            self.error.emit(str(e))


class ComparisonWorker(QObject):
    """Computes OLS and AF-corrected unmixed arrays in a QThread."""
    finished = Signal(object, object)   # (ols_data, af_data) full arrays
    error = Signal(str)

    def __init__(self, raw_event_data, transfer_matrix,
                 af_precomputed, af_spectra, exp_settings,
                 filtered_fl_ids_raw, spillover=None):
        super().__init__()
        self.raw_event_data = raw_event_data
        self.transfer_matrix = transfer_matrix
        self.af_precomputed = af_precomputed
        self.af_spectra = af_spectra
        self.exp_settings = exp_settings
        self.filtered_fl_ids_raw = filtered_fl_ids_raw
        self.spillover = spillover

    def run(self):
        try:
            ols_data = apply_transfer_matrix(
                self.transfer_matrix, self.raw_event_data
            )
            af_result = apply_af_transfer(
                self.raw_event_data,
                self.transfer_matrix,
                self.af_precomputed,
                self.af_spectra,
                self.exp_settings,
                filtered_fl_ids_raw=self.filtered_fl_ids_raw,
                spillover=self.spillover,
            )
            self.finished.emit(ols_data, af_result['unmixed'])
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# AfComparisonPlotWidget
# ---------------------------------------------------------------------------

class AfComparisonPlotWidget(QWidget):
    """
    A single density-heatmap biplot that mirrors the style of CytometryPlotWidget
    (dark background, colorcet colourmap, logicle transforms, ZoomAxis drag zoom)
    but operates on a locally held event array rather than the shared controller
    data pipeline.

    Parameters
    ----------
    title : str
        Label shown above the plot.
    controller : Controller
        Used to read unmixed_transformations for initial transform params,
        unmixed pnn/fl_ids for channel menus, and unmixed_lookup_tables for gating.
    """

    # Emitted when the source gate is changed so the parent can redraw both plots
    # with a consistent gate mask.
    sourceGateChanged = Signal(str)
    # Emitted when either channel is changed, so the sibling plot can mirror it.
    channelChanged = Signal(str, str)   # (channel_x, channel_y)
    # Emitted when a zoom/scaling is applied on one axis, so the sibling can mirror.
    scalingChanged = Signal(str, object)  # (axis_name, Transform)

    def __init__(self, title: str, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._title_text = title

        # Local copies of Transform objects (not shared with Unmixed Data tab)
        self._transformations: dict[str, Transform] = {}
        # Event data for this plot (full array, gate mask applied only for display)
        self._event_data: np.ndarray | None = None
        # Current source gate name
        self._source_gate: str = 'root'
        # Current channel names
        self._channel_x: str | None = None
        self._channel_y: str | None = None

        # Build colourmap LUT (same as CytometryPlotWidget)
        colors = cc.palette[settings.colourmap_name_retrieved]
        cmap = pg.ColorMap(
            pos=0.9 * np.linspace(0.0, 1.0, len(colors)) ** 2
                + 0.1 * np.linspace(0.0, 1.0, len(colors)),
            color=colors,
        )
        rgba_lut = cmap.getLookupTable(alpha=True)
        rgba_lut[0, 3] = 0   # fully transparent for zero-count bins
        self._rgba_lut = rgba_lut

        # ---- Layout ----
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.graphics_widget = TransparentGraphicsLayoutWidget(parent=self)
        gl = self.graphics_widget.ci.layout
        gl.setHorizontalSpacing(0)
        gl.setVerticalSpacing(0)

        # Title label (gate source selector)
        self.plot_title = InteractiveLabel(title, parent_plot=self)
        self.plot_title.setParent(self)
        self.graphics_widget.addItem(self.plot_title, row=0, col=2)

        # ViewBox
        self.vb = NoPanViewBox()
        self.vb.setParent(self)
        self.graphics_widget.addItem(self.vb, row=1, col=2)

        # Axes and labels
        self.label_y = InteractiveLabel('Y Axis', parent_plot=self, angle=-90)
        self.graphics_widget.addItem(self.label_y, row=1, col=0)
        self.axis_left = ZoomAxis('left', self.vb)
        self.graphics_widget.addItem(self.axis_left, row=1, col=1)

        self.axis_bottom = ZoomAxis('bottom', self.vb)
        self.graphics_widget.addItem(self.axis_bottom, row=2, col=2)
        self.label_x = InteractiveLabel('X Axis', parent_plot=self)
        self.graphics_widget.addItem(self.label_x, row=3, col=2)

        self.axis_left.linkToView(self.vb)
        self.axis_bottom.linkToView(self.vb)
        self.label_x.setParent(self.axis_bottom)
        self.label_y.setParent(self.axis_left)
        self.axis_bottom.setParent(self)
        self.axis_left.setParent(self)

        self.axis_bottom.zoom_timer.timeout.connect(lambda: self._apply_zoom('x'))
        self.axis_left.zoom_timer.timeout.connect(lambda: self._apply_zoom('y'))

        # Disable right-click context menu (no gates in comparison plots)
        self.vb.raiseContextMenu = lambda ev: None

        # Heatmap image item
        self.img = pg.ImageItem(parent=self)
        self.img.setLookupTable(self._rgba_lut)
        self.vb.addItem(self.img)

        # Status label (shown when computing or on error)
        self._status_label = QLabel('', alignment=Qt.AlignCenter)
        self._status_label.setStyleSheet('color: #aaaaaa;')

        main_layout.addWidget(self.graphics_widget)
        main_layout.addWidget(self._status_label)

        # Axis label left-click: change channel
        self.label_x.leftClickMenuFunction = self._set_channel_x
        self.label_y.leftClickMenuFunction = self._set_channel_y

        # Title left-click: change source gate
        self.plot_title.leftClickMenuFunction = self._set_source_gate

        self.setMinimumHeight(280)
        self.setMinimumWidth(280)

    # ------------------------------------------------------------------
    # Square aspect ratio — always keeps width == height
    # ------------------------------------------------------------------

    def sizeHint(self):
        from PySide6.QtCore import QSize
        side = max(self.minimumHeight(), self.width())
        return QSize(side, side)

    def resizeEvent(self, event):
        # Force square: set height to match width, clamped to minimum
        side = max(self.minimumHeight(), event.size().width())
        if self.height() != side:
            self.setFixedHeight(side)
        super().resizeEvent(event)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_status(self, text: str):
        self._status_label.setText(text)

    def set_event_data(self, event_data: np.ndarray | None):
        """Set the full (ungated) unmixed event array for this plot."""
        self._event_data = event_data

    def initialise_from_controller(self, channel_x: str, channel_y: str):
        """
        Copy Transform objects from controller.unmixed_transformations,
        set initial channels, and configure axes.  Call after set_event_data().
        """
        src = self.controller.unmixed_transformations or {}
        self._transformations = {}
        for ch, tr in src.items():
            self._transformations[ch] = deepcopy(tr)

        self._channel_x = channel_x
        self._channel_y = channel_y
        self._source_gate = 'root'
        self._configure_axes()

    def set_channels(self, channel_x: str, channel_y: str):
        """Change the displayed channels and reconfigure axes."""
        self._channel_x = channel_x
        self._channel_y = channel_y
        self._configure_axes()
        self._draw()

    def set_scaling(self, axis_name: str, tr):
        """
        Mirror scaling/zoom from the sibling plot.
        tr is the Transform object that was already modified by the sibling's _apply_zoom.
        We copy its parameters into our own Transform and update the view accordingly.
        """
        if axis_name == 'x':
            channel = self._channel_x
            axis = self.axis_bottom
            vb_set_range = self.vb.setXRange
        else:
            channel = self._channel_y
            axis = self.axis_left
            vb_set_range = self.vb.setYRange

        if channel not in self._transformations:
            return

        my_tr = self._transformations[channel]
        # Copy the transform state from the sibling
        my_tr.id = tr.id
        my_tr.logicle_w = tr.logicle_w
        my_tr.logicle_a = tr.logicle_a
        my_tr.limits = tr.limits
        my_tr.set_transform(my_tr.id, my_tr.limits)

        vb_set_range(*my_tr.limits, padding=0)
        axis.zoomZero = my_tr.zero
        axis.limits = my_tr.limits
        axis.setTicks(my_tr.ticks())
        self._draw()

    def set_source_gate(self, gate_name: str):
        """Change the source gate (called from external sync)."""
        self._source_gate = gate_name
        self._configure_title()
        self._draw()

    def redraw(self):
        self._draw()

    def clear(self):
        self.img.clear()
        self._event_data = None
        self._status_label.setText('')

    # ------------------------------------------------------------------
    # Internal: axis configuration
    # ------------------------------------------------------------------

    def _configure_axes(self):
        if (not self._transformations
                or self._channel_x not in self._transformations
                or self._channel_y not in self._transformations):
            return

        pnn = self.controller.experiment.settings['unmixed'].get('event_channels_pnn', [])
        fl_ids = self.controller.experiment.settings['unmixed'].get('fluorescence_channel_ids', [])
        fl_names = [pnn[i] for i in fl_ids] if pnn and fl_ids else []
        pnn_labels = build_display_label_map(
            pnn, self.controller.experiment.process.get('spectral_model')
        )
        fl_display = [pnn_labels.get(n, n) for n in fl_names]

        # X axis
        self.label_x.setText(pnn_labels.get(self._channel_x, self._channel_x))
        self.label_x.leftClickMenuItems = fl_display
        self.label_x.leftItemSelected = (
            fl_names.index(self._channel_x) if self._channel_x in fl_names else 0
        )
        tr_x = self._transformations[self._channel_x]
        self.axis_bottom.setTicks(tr_x.ticks())
        self.axis_bottom.zoomZero = tr_x.zero
        self.axis_bottom.fullRange = (0, 1.1)
        self.axis_bottom.limits = tr_x.limits
        self.vb.setXRange(*tr_x.limits, padding=0)

        # Y axis
        self.label_y.setText(pnn_labels.get(self._channel_y, self._channel_y))
        self.label_y.leftClickMenuItems = fl_display
        self.label_y.leftItemSelected = (
            fl_names.index(self._channel_y) if self._channel_y in fl_names else 0
        )
        tr_y = self._transformations[self._channel_y]
        self.axis_left.setTicks(tr_y.ticks())
        self.axis_left.zoomZero = tr_y.zero
        self.axis_left.fullRange = (0, 1.1)
        self.axis_left.limits = tr_y.limits
        self.vb.setYRange(*tr_y.limits, padding=0)

        self._configure_title()

    def _configure_title(self):
        """Populate gate source menu from the unmixed gating strategy."""
        gating = self.controller.unmixed_gating
        if gating is None:
            gate_names = ['root']
        else:
            gate_ids = [
                g for g in gating.get_gate_ids()
                if gating._get_gate_node(g[0], g[1]).gate_type != 'QuadrantGate'
            ]
            gate_names = ['root'] + [g[0] for g in gate_ids]

        # Validate current gate still exists
        if self._source_gate not in gate_names:
            self._source_gate = 'root'

        self.plot_title.setText(f'{self._title_text}  [{self._source_gate}]')
        self.plot_title.leftClickMenuItems = gate_names
        self.plot_title.leftClickMenuFunction = self._set_source_gate
        self.plot_title.leftItemSelected = gate_names.index(self._source_gate)

    # ------------------------------------------------------------------
    # Internal: drawing
    # ------------------------------------------------------------------

    def _draw(self):
        if self._event_data is None:
            self.img.clear()
            return
        if (self._channel_x not in self._transformations
                or self._channel_y not in self._transformations):
            return

        pnn = self.controller.experiment.settings['unmixed'].get('event_channels_pnn', [])
        if self._channel_x not in pnn or self._channel_y not in pnn:
            return

        x_col = pnn.index(self._channel_x)
        y_col = pnn.index(self._channel_y)

        # Start with all events; gate masking will narrow this down if needed.
        event_data = self._event_data

        # Compute a per-event gate mask using the same mechanism as
        # apply_gates_in_place in functions.py.  The unmixed_lookup_tables,
        # unmixed_transformations, and unmixed_gating are all populated on the
        # controller regardless of which tab is currently active, so this works
        # from the AutoSpectral tab.
        mask = self._compute_gate_mask(event_data)
        if mask is not None:
            event_data = event_data[mask]

        tr_x = self._transformations[self._channel_x]
        tr_y = self._transformations[self._channel_y]

        try:
            heatmap, _, _ = np.histogram2d(
                event_data[:, x_col],
                event_data[:, y_col],
                bins=[tr_x.scale, tr_y.scale],
            )
        except Exception as e:
            logger.error(f'AfComparisonPlotWidget._draw histogram failed: {e}')
            return

        # Apply density cutoff (match CytometryPlotWidget: bins below cutoff → 0)
        cutoff = settings.density_cutoff_retrieved
        heatmap[heatmap < cutoff] = 0

        # Rescale border bins to interior max — mirrors calc_hist2d() in functions.py
        # so that edge overflow events don't compress the interior colour range.
        if heatmap.shape[0] > 2 and heatmap.shape[1] > 2:
            global_max = heatmap.max()
            inside_max = heatmap[1:-1, 1:-1].max()
            if inside_max > 0 and inside_max < global_max:
                scale = inside_max / global_max
                heatmap[0, :] *= scale
                heatmap[-1, :] *= scale
                heatmap[1:-1, 0] *= scale
                heatmap[1:-1, -1] *= scale

        self.img.setImage(heatmap)
        self.img.setRect(QRectF(
            tr_x.limits[0], tr_y.limits[0],
            tr_x.limits[1] - tr_x.limits[0],
            tr_y.limits[1] - tr_y.limits[0],
        ))

    # ------------------------------------------------------------------
    # Internal: gate membership computation
    # ------------------------------------------------------------------

    def _compute_gate_mask(self, event_data: np.ndarray) -> np.ndarray | None:
        """
        Compute a per-event boolean mask for self._source_gate applied to
        event_data, using the controller's unmixed_lookup_tables,
        unmixed_transformations, and unmixed_gating.

        This replicates the logic of apply_gates_in_place() from functions.py,
        walking the full gate ancestry so that child gates correctly inherit
        their parent's membership.

        Returns None if 'root' is selected or if gating is unavailable.
        Returns a boolean ndarray of length len(event_data) otherwise.
        """
        if self._source_gate == 'root':
            return None

        gating = self.controller.unmixed_gating
        lookup_tables = self.controller.unmixed_lookup_tables
        transforms = self.controller.unmixed_transformations
        pnn = self.controller.experiment.settings['unmixed'].get('event_channels_pnn', [])

        if gating is None or not lookup_tables or not transforms or not pnn:
            return None

        # Build the full ancestry path to the target gate, root-first
        try:
            paths = gating.find_matching_gate_paths(self._source_gate)
        except Exception:
            return None
        if not paths:
            return None

        # Ancestry: all gates in the path from root to target (exclusive of
        # 'root' itself since every event is in root), then the target gate.
        ancestry_path = list(paths[0])   # e.g. ('root', 'Cells')
        gate_sequence = [g for g in ancestry_path if g != 'root'] + [self._source_gate]
        # Deduplicate while preserving order (target may already be last)
        seen = set()
        gate_sequence_dedup = []
        for g in gate_sequence:
            if g not in seen:
                seen.add(g)
                gate_sequence_dedup.append(g)

        n_events = len(event_data)
        # Start with all-True (every event is in 'root')
        cumulative_mask = np.ones(n_events, dtype=bool)

        for gate_name in gate_sequence_dedup:
            if gate_name not in lookup_tables:
                # Gate not found in lookup tables — skip
                logger.warning(
                    f'AfComparisonPlotWidget._compute_gate_mask: '
                    f'"{gate_name}" not in unmixed_lookup_tables'
                )
                return None

            try:
                gate = gating.get_gate(gate_name)
                channels = gate.get_dimension_ids()

                if len(channels) == 1:
                    xchan = channels[0]
                    if xchan not in pnn:
                        return None
                    ix = pnn.index(xchan)
                    x = event_data[:, ix]
                    scale_x = transforms[xchan].scale
                    idx = np.searchsorted(scale_x, x) - 2
                    lt = lookup_tables[gate_name]
                    idx = np.clip(idx, 0, len(lt) - 1)
                    gate_mask = lt[idx]

                else:  # 2-channel gate
                    if gate.gate_type == 'QuadrantGate':
                        xchan = gate.dimensions[0].dimension_ref
                        ychan = gate.dimensions[1].dimension_ref
                    else:
                        xchan = channels[0]
                        ychan = channels[1]
                    if xchan not in pnn or ychan not in pnn:
                        return None
                    ix = pnn.index(xchan)
                    iy = pnn.index(ychan)
                    x = event_data[:, ix]
                    y = event_data[:, iy]
                    scale_x = transforms[xchan].scale
                    scale_y = transforms[ychan].scale
                    idx_x = np.searchsorted(scale_x, x) - 2
                    idx_y = np.searchsorted(scale_y, y) - 2
                    bins_x = transforms[xchan].scale_bins + 1
                    flat_idx = idx_x * bins_x + idx_y
                    lt = lookup_tables[gate_name]
                    flat_idx = np.clip(flat_idx, 0, len(lt) - 1)
                    gate_mask = lt[flat_idx]

                cumulative_mask = cumulative_mask & gate_mask.astype(bool)

            except Exception as e:
                logger.error(
                    f'AfComparisonPlotWidget._compute_gate_mask: '
                    f'error computing mask for "{gate_name}": {e}'
                )
                return None

        return cumulative_mask

    # ------------------------------------------------------------------
    # Internal: zoom (mirrors CytometryPlotWidget.apply_zoom)
    # ------------------------------------------------------------------

    def _apply_zoom(self, axis_name: str):
        if axis_name == 'x':
            axis = self.axis_bottom
            channel = self._channel_x
            vb_set_range = self.vb.setXRange
            vb_range_ind = 0
            map_pos = self.vb.mapToView(axis.initial_pos).x() if axis.initial_pos else 0.5
            factor_flip = False
        else:
            axis = self.axis_left
            channel = self._channel_y
            vb_set_range = self.vb.setYRange
            vb_range_ind = 1
            map_pos = self.vb.mapToView(axis.initial_pos).y() if axis.initial_pos else 0.5
            factor_flip = True

        if axis._pending_delta == 0 or channel not in self._transformations:
            return

        step = axis._pending_delta
        axis._pending_delta = 0
        if abs(step) < 1:
            return

        zoom_rate = 1.04
        factor = (1 / zoom_rate) if step > 0 else zoom_rate
        if factor_flip:
            factor = 1 / factor

        tr = self._transformations[channel]
        vmin, vmax = self.vb.viewRange()[vb_range_ind]

        if tr.id == 1:  # logicle
            if map_pos < 0.5 * vmax:
                tr.logicle_w = tr.logicle_w / factor
                tr.set_transform()
            else:
                new_max = (vmax - axis.zoomZero) * factor + axis.zoomZero
                new_min = (vmin - axis.zoomZero) * factor + axis.zoomZero
                if new_max < axis.fullRange[1] * 1.01:
                    vb_set_range(new_min, new_max, padding=0)
                axis.limits = (new_min, new_max)
                tr.set_transform(limits=axis.limits)
        else:  # linear or log
            new_max = (vmax - axis.zoomZero) * factor + axis.zoomZero
            new_min = (vmin - axis.zoomZero) * factor + axis.zoomZero
            if new_max < axis.fullRange[1] * 1.01:
                vb_set_range(new_min, new_max, padding=0)
            axis.limits = (new_min, new_max)
            tr.set_transform(limits=axis.limits)

        axis.zoomZero = tr.zero
        axis.setTicks(tr.ticks())
        self._draw()
        self.scalingChanged.emit(axis_name, tr)

    # ------------------------------------------------------------------
    # Slot: channel changed via label click
    # ------------------------------------------------------------------

    def _set_channel_x(self, n, _parent):
        pnn = self.controller.experiment.settings['unmixed'].get('event_channels_pnn', [])
        fl_ids = self.controller.experiment.settings['unmixed'].get('fluorescence_channel_ids', [])
        fl_names = [pnn[i] for i in fl_ids] if pnn and fl_ids else []
        if 0 <= n < len(fl_names):
            self._channel_x = fl_names[n]
            self._configure_axes()
            self._draw()
            self.channelChanged.emit(self._channel_x, self._channel_y)

    def _set_channel_y(self, n, _parent):
        pnn = self.controller.experiment.settings['unmixed'].get('event_channels_pnn', [])
        fl_ids = self.controller.experiment.settings['unmixed'].get('fluorescence_channel_ids', [])
        fl_names = [pnn[i] for i in fl_ids] if pnn and fl_ids else []
        if 0 <= n < len(fl_names):
            self._channel_y = fl_names[n]
            self._configure_axes()
            self._draw()
            self.channelChanged.emit(self._channel_x, self._channel_y)

    def _set_source_gate(self, n, _parent):
        gate_names = self.plot_title.leftClickMenuItems
        if 0 <= n < len(gate_names):
            self._source_gate = gate_names[n]
            self._configure_title()
            self._draw()
            self.sourceGateChanged.emit(self._source_gate)

    # ------------------------------------------------------------------
    # Required by InteractiveLabel / CytometryPlotWidget contract
    # ------------------------------------------------------------------

    def select_plot_on_parent_grid(self):
        pass   # No parent grid — no-op


# ---------------------------------------------------------------------------
# Main tab widget
# ---------------------------------------------------------------------------

class AutoSpectralTab(QWidget):
    """Permanent integrated tab for AutoSpectral AF extraction and evaluation."""

    def __init__(self, bus=None, controller=None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.controller = controller

        self._train_thread = None
        self._train_worker = None
        self._cmp_thread = None
        self._cmp_worker = None
        self._cmp_running = False
        # Tuple describing what the last successful comparison was computed for.
        # Format: (sample_path, profile_key, id(af_precomputed))
        # If all three match the current request, we skip the worker and just
        # redraw from the cached arrays.
        self._last_cmp_state: tuple | None = None
        self._pending_cmp_state: tuple | None = None

        # Full unmixed arrays (all events) from the worker
        self._ols_data: np.ndarray | None = None
        self._af_data: np.ndarray | None = None

        # Outer layout
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._label_disabled = QLabel(
            'AutoSpectral AF: set up the spectral model and calculate the '
            'unmixing matrix (Spectral Process tab) before using this tab.'
        )
        self._label_disabled.setWordWrap(True)
        self._label_disabled.setContentsMargins(8, 8, 8, 8)
        outer.addWidget(self._label_disabled)

        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setAlignment(Qt.AlignTop)
        content_layout.setSpacing(10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self._content)
        outer.addWidget(scroll)

        from honeychrome.settings import heading_style
        title_label = QLabel("AutoSpectral AF")
        title_label.setStyleSheet(heading_style)
        content_layout.addWidget(title_label)

        # Help text (rich text / HTML)
        self.help_label = QLabel(autospectral_af_help_text)
        self.help_label.setTextFormat(Qt.RichText)
        self.help_label.setWordWrap(True)
        # self.help_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        font = self.help_label.font()
        font.setPointSize(14)  # Set your desired font size
        self.help_label.setFont(font)
        content_layout.addWidget(self.help_label)

        # toggle button and content
        self.toggle_content_button = QCheckBox('Show AutoSpectral AF process')
        self.settings = QSettings("honeychrome", "app_configuration")
        saved_state = self.settings.value("show_autospectral_af", "false") == "true"
        self.toggle_content_button.setCheckable(True)
        self.toggle_content_button.setChecked(saved_state)
        content_layout.addWidget(self.toggle_content_button)

        self.toggle_content = QWidget()
        toggle_content_layout = QVBoxLayout(self.toggle_content)
        self._build_extraction_section(toggle_content_layout)
        self._build_profile_manager_section(toggle_content_layout)
        self._build_assignment_section(toggle_content_layout)
        self._build_comparison_section(toggle_content_layout)
        content_layout.addWidget(self.toggle_content)
        self.toggle_content.setVisible(saved_state)

        self.toggle_content_button.toggled.connect(self.toggle_content.setVisible)
        self.toggle_content_button.toggled.connect(self.save_visibility)

        self.toggle_content_button.setStyleSheet("""
                    QCheckBox {
                        font-size: 14pt;
                        spacing: 10px;      /* Gap between checkmark and text */
                        padding: 10px;      /* Internal padding */
                    }
                """)


        content_layout.addStretch()

        if self.bus:
            self.bus.modeChangeRequested.connect(self._on_mode_change)
            self.bus.loadSampleRequested.connect(self._on_sample_loaded)
            self.bus.spectralProcessRefreshed.connect(self._on_process_refreshed)
            self.bus.sampleTreeUpdated.connect(self._populate_sample_combo)

    def save_visibility(self, checked: bool):
        self.settings.setValue("show_autospectral_af", "true" if checked else "false")

    # ======================================================================
    # Section builders
    # ======================================================================

    def _build_extraction_section(self, parent_layout):
        grp = QGroupBox('Step 1 — Extract AF Profile from Unstained Sample')
        layout = QFormLayout(grp)
        layout.setRowWrapPolicy(QFormLayout.DontWrapRows)

        self._sample_combo = QComboBox()
        self._sample_combo.setToolTip('Select the unstained control sample.')
        self._sample_combo.installEventFilter(WheelBlocker(self._sample_combo))
        self._sample_combo.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        layout.addRow('Unstained sample:', self._sample_combo)

        self._n_clusters_spin = QSpinBox()
        self._n_clusters_spin.setRange(4, 1000)
        self._n_clusters_spin.setValue(200)
        self._n_clusters_spin.setToolTip(
            'KMeans cluster count (equivalent to som.dim² in R AutoSpectral).'
        )
        self._n_clusters_spin.installEventFilter(WheelBlocker(self._n_clusters_spin))
        self._n_clusters_spin.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        layout.addRow('AF clusters:', self._n_clusters_spin)

        self._extract_btn = QPushButton('Extract AF Profile')
        self._extract_btn.clicked.connect(self._run_extraction)
        layout.addRow(self._extract_btn)

        self._extract_status = QLabel('')
        layout.addRow(self._extract_status)

        parent_layout.addWidget(grp)

    def _build_profile_manager_section(self, parent_layout):
        grp = QGroupBox('Stored AF Profiles')
        layout = QVBoxLayout(grp)

        self._profile_plot_axis = BottomAxisVerticalTickLabels(angle=45)
        self._profile_plot = pg.PlotWidget(axisItems={'bottom': self._profile_plot_axis})
        self._profile_plot.setLabel('left', 'Intensity')
        self._profile_plot.showGrid(x=True, y=True, alpha=0.3)
        self._profile_plot.setMaximumHeight(440)
        self._profile_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # Disable mouse wheel rescaling
        self._profile_plot.setMouseEnabled(x=False, y=False)
        # prevent mouse scroll catching
        self._wheel_blocker = WheelBlocker(self)
        self._profile_plot.viewport().installEventFilter(self._wheel_blocker)
        layout.addWidget(self._profile_plot)

        self._profile_list = QListWidget()
        self._profile_list.setMaximumHeight(120)
        self._profile_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._profile_list.currentRowChanged.connect(self._on_profile_row_changed)
        layout.addWidget(self._profile_list)

        btn_row = QHBoxLayout()

        load_btn = QPushButton('Load CSV…')
        load_btn.setToolTip('Load an AF profile from a CSV file.')
        load_btn.clicked.connect(self._load_profile_csv)
        btn_row.addWidget(load_btn)

        self._save_btn = QPushButton('Save Selected CSV…')
        self._save_btn.setToolTip('Re-export the selected profile to a CSV file.')
        self._save_btn.clicked.connect(self._save_selected_profile_csv)
        self._save_btn.setEnabled(False)
        btn_row.addWidget(self._save_btn)

        self._delete_btn = QPushButton('Delete Selected')
        self._delete_btn.setToolTip(
            'Remove the selected profile and clear any sample assignments to it.'
        )
        self._delete_btn.clicked.connect(self._delete_selected_profile)
        self._delete_btn.setEnabled(False)
        btn_row.addWidget(self._delete_btn)

        layout.addLayout(btn_row)
        parent_layout.addWidget(grp)

    def _build_assignment_section(self, parent_layout):
        grp = QGroupBox('Step 2 — Assign AF Profiles to Samples')
        layout = QVBoxLayout(grp)

        info = QLabel(
            'Tick the AF profile(s) for each sample. '
            'Multiple profiles are concatenated before unmixing. '
            'Unticked samples use standard OLS unmixing. '
            'Single Stain Controls are excluded.'
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Frozen header row (profile names) — does not scroll vertically.
        # Its own horizontal scrollbar is hidden and driven by the body's.
        self._assign_header_scroll = QScrollArea()
        self._assign_header_scroll.setWidgetResizable(True)
        self._assign_header_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._assign_header_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._assign_header_scroll.setFrameShape(QScrollArea.NoFrame)
        self._assign_header_widget = QWidget()
        self._assign_header_layout = QGridLayout(self._assign_header_widget)
        self._assign_header_layout.setSpacing(6)
        self._assign_header_layout.setContentsMargins(0, 0, 0, 0)
        self._assign_header_layout.setAlignment(Qt.AlignLeft)
        self._assign_header_scroll.setWidget(self._assign_header_widget)
        layout.addWidget(self._assign_header_scroll)

        # Scrollable grid: rows = non-SSC samples, columns = AF profiles
        # Min/max here are just the pre-populate defaults; _rebuild_assignment_grid()
        # re-sizes this to fit the actual sample count each time it's called.
        self._assign_scroll = QScrollArea()
        self._assign_scroll.setWidgetResizable(True)
        self._assign_scroll.setMinimumHeight(240)
        self._assign_scroll.setMaximumHeight(900)
        self._assign_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._assign_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._assign_grid_widget = QWidget()
        self._assign_grid_layout = QGridLayout(self._assign_grid_widget)
        self._assign_grid_layout.setSpacing(6)
        self._assign_grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._assign_scroll.setWidget(self._assign_grid_widget)
        layout.addWidget(self._assign_scroll)

        # Keep header columns aligned with body columns when scrolling sideways.
        self._assign_scroll.horizontalScrollBar().valueChanged.connect(
            self._assign_header_scroll.horizontalScrollBar().setValue
        )

        self._assign_checkboxes: dict = {}

        self._assign_status = QLabel('')
        layout.addWidget(self._assign_status)

        btn_row = QHBoxLayout()

        assign_all_btn = QPushButton('Assign Selected Profile to All Samples')
        assign_all_btn.setToolTip(
            'Assign the profile currently selected in the Stored AF Profiles list '
            'to all non-SSC samples. Already-assigned profiles are not duplicated.'
        )
        assign_all_btn.clicked.connect(self._assign_selected_profile_to_all)
        btn_row.addWidget(assign_all_btn)

        clear_sample_btn = QPushButton('Clear AF for Current Sample')
        clear_sample_btn.setToolTip(
            'Remove all AF assignments for the currently loaded sample.'
        )
        clear_sample_btn.clicked.connect(self._clear_af_for_current_sample)
        btn_row.addWidget(clear_sample_btn)

        clear_all_btn = QPushButton('Clear All AF Assignments')
        clear_all_btn.setToolTip('Remove all AF assignments for all samples.')
        clear_all_btn.clicked.connect(self._clear_all_af)
        btn_row.addWidget(clear_all_btn)

        layout.addLayout(btn_row)
        parent_layout.addWidget(grp)

    def _build_comparison_section(self, parent_layout):
        grp = QGroupBox('Step 3 — Side-by-Side Comparison (Current Sample)')
        layout = QVBoxLayout(grp)

        # Top control row: AF profile selector + Update button
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel('AF profile (right plot):'))
        self._cmp_profile_combo = QComboBox()
        self._cmp_profile_combo.installEventFilter(WheelBlocker(self._cmp_profile_combo))
        self._cmp_profile_combo.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._cmp_profile_combo.setToolTip(
            '"Assigned to sample" uses the profiles assigned in Step 2. '
            'Select an individual profile to preview it without changing assignments.'
        )
        ctrl_row.addWidget(self._cmp_profile_combo)
        ctrl_row.addStretch()
        run_btn = QPushButton('Update Plots')
        run_btn.setToolTip(
            'Re-run unmixing on the current sample and refresh both plots.'
        )
        run_btn.clicked.connect(self._run_comparison)
        ctrl_row.addWidget(run_btn)
        layout.addLayout(ctrl_row)

        self._cmp_status = QLabel('')
        layout.addWidget(self._cmp_status)

        # Side-by-side plot widgets
        plot_splitter = QSplitter(Qt.Horizontal)

        self._plot_ols = AfComparisonPlotWidget(
            'OLS (no AF)', self.controller, parent=self
        )
        self._plot_ols.sourceGateChanged.connect(self._on_ols_gate_changed)
        self._plot_ols.channelChanged.connect(self._on_ols_channel_changed)
        self._plot_ols.scalingChanged.connect(self._on_ols_scaling_changed)

        self._plot_af = AfComparisonPlotWidget(
            'AF-corrected', self.controller, parent=self
        )
        self._plot_af.sourceGateChanged.connect(self._on_af_gate_changed)
        self._plot_af.channelChanged.connect(self._on_af_channel_changed)
        self._plot_af.scalingChanged.connect(self._on_af_scaling_changed)

        plot_splitter.addWidget(self._plot_ols)
        plot_splitter.addWidget(self._plot_af)
        plot_splitter.setSizes([500, 500])
        layout.addWidget(plot_splitter)

        parent_layout.addWidget(grp)

    # ======================================================================
    # Bus signal handlers
    # ======================================================================

    def _on_mode_change(self, mode):
        if mode == TAB_NAME:
            self._refresh_ui()
            if self.controller.raw_event_data is not None:
                QTimer.singleShot(0, self._run_comparison)

    def _on_sample_loaded(self, _sample_path):
        # Rebuild grid and controls immediately, then defer the comparison run
        # with a zero-delay timer so that load_sample() has fully completed
        # (raw_event_data, transfer_matrix etc. are all set) before we read them.
        if self.controller.current_mode == TAB_NAME:
            self._rebuild_assignment_grid()
            self._refresh_comparison_controls()
            self._clear_comparison_plots()
            QTimer.singleShot(0, self._run_comparison)

    def _on_process_refreshed(self):
        self._refresh_ui()

    # ======================================================================
    # Full UI refresh
    # ======================================================================

    def _refresh_ui(self):
        has_unmixing = (
            self.controller.experiment.process.get('unmixing_matrix') is not None
        )
        self._label_disabled.setVisible(not has_unmixing)
        self._content.setVisible(has_unmixing)

        if not has_unmixing:
            return

        self._populate_sample_combo()
        self._refresh_profile_list()
        self._rebuild_assignment_grid()
        self._refresh_comparison_controls()

    # ======================================================================
    # Section 1 — extraction
    # ======================================================================

    def _populate_sample_combo(self):
        """Populate unstained-sample picker with cell controls that are marked or
        regex-identified as unstained."""
        self._sample_combo.blockSignals(True)
        self._sample_combo.clear()
        self._sample_combo.addItem('— select unstained sample —', userData=None)

        samples = self.controller.experiment.samples
        all_samples = samples.get('all_samples', {})
        manually_tagged = set(samples.get('unstained_samples', []))

        # Search all samples for unstained negatives, excluding bead files.
        candidates = [
            p for p in all_samples
            if not re.search(r'[Bb]eads', all_samples.get(p, ''), re.IGNORECASE)
            and (
                p in manually_tagged
                or re.search(r'unstained', all_samples.get(p, ''), re.IGNORECASE)
                or re.search(r'unstained', p, re.IGNORECASE)
            )
        ]

        preselect = 0
        for i, path in enumerate(candidates, start=1):
            display = Path(path).stem
            self._sample_combo.addItem(display, userData=path)
            if 'unstained' in display.lower() or path in manually_tagged:
                preselect = preselect or i

        if preselect:
            self._sample_combo.setCurrentIndex(preselect)
        self._sample_combo.blockSignals(False)

    def _run_extraction(self):
        sample_path = self._sample_combo.currentData()
        if sample_path is None:
            self._extract_status.setText('Please select an unstained sample.')
            return

        spectral_model = self.controller.experiment.process.get('spectral_model', [])
        all_samples = self.controller.experiment.samples
        unstained_paths = set(all_samples.get('unstained_samples', []))
        all_sample_paths = all_samples.get('all_samples', {})

        # Reverse map: sample display name → path
        name_to_path = {v: k for k, v in all_sample_paths.items()}

        # Check if any control in the spectral model points to an unstained sample
        offending = [
            c.get('label', '?')
            for c in spectral_model
            if name_to_path.get(c.get('sample_name', '')) in unstained_paths
        ]

        if offending:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Unstained Sample in Spectral Model",
                "Warning: the following spectral model controls appear to use an unstained sample:\n\n"
                + "\n".join(f"  • {lbl}" for lbl in offending)
                + "\n\nUnstained samples should not be added as spectral controls when using AutoSpectral AF. "
                "They should only be used as the negative reference (Unstained Negative column). "
                "Please remove these controls from the Spectral Process table before proceeding."
            )
            return
        
        try:
            full_path = str(self.controller.experiment_dir / sample_path)
            sample = sample_from_fcs(full_path, self.bus)
            raw_event_data = sample.get_events(source='raw')
        except Exception as e:
            self._extract_status.setText(f'Failed to load sample: {e}')
            return

        if raw_event_data is None or raw_event_data.shape[0] == 0:
            self._extract_status.setText('Sample has no events.')
            return

        fl_ids = np.array(
            self.controller.filtered_raw_fluorescence_channel_ids
        )
        raw_fl = raw_event_data[:, fl_ids]

        fluor_spectra = self._build_fluor_spectra()
        if fluor_spectra is None:
            self._extract_status.setText(
                'Fluorophore profiles not found — ensure the spectral model is complete.'
            )
            return

        self._extract_btn.setEnabled(False)
        self._extract_status.setText('Running KMeans clustering...')

        self._train_thread = QThread()
        self._train_worker = AfTrainingWorker(
            raw_fl, fluor_spectra, self._n_clusters_spin.value(), sample_path
        )
        self._train_worker.moveToThread(self._train_thread)
        self._train_thread.started.connect(self._train_worker.run)
        self._train_worker.finished.connect(self._on_extraction_finished)
        self._train_worker.error.connect(self._on_extraction_error)
        self._train_worker.progress.connect(self._extract_status.setText)
        self._train_worker.finished.connect(self._train_thread.quit)
        self._train_worker.error.connect(self._train_thread.quit)
        self._train_thread.finished.connect(self._train_thread.deleteLater)
        self._train_thread.start()

    def _on_extraction_finished(self, af_spectra: np.ndarray):
        # Read source_fcs_path from the worker object (set before thread started)
        source_fcs_path = self._train_worker.source_fcs_path
        self._extract_btn.setEnabled(True)
        n_af = af_spectra.shape[0]

        pnn_raw = self.controller.experiment.settings['raw']['event_channels_pnn']
        fl_ids = self.controller.filtered_raw_fluorescence_channel_ids
        channel_names = [pnn_raw[i] for i in fl_ids]

        try:
            profile_name = save_af_profile_csv(
                af_spectra, channel_names, source_fcs_path,
                self.controller.experiment_dir,
            )
        except Exception as e:
            logger.error(f'AutoSpectral: failed to save CSV: {e}')
            profile_name = Path(source_fcs_path).stem + ' AutoSpectral AF'

        af_profiles = self.controller.experiment.process.get('af_profiles', {})
        af_profiles[profile_name] = {
            'spectra': af_spectra.tolist(),
            'source_fcs': str(source_fcs_path),
            'channel_names': channel_names,
        }
        self.controller.experiment.process['af_profiles'] = af_profiles
        self.bus.autoSaveRequested.emit()

        # Cache precomputed matrices for the new profile immediately —
        # this is the only linalg.solve call needed; sample loading just does hstack.
        self.controller.cache_af_profile(profile_name)

        self._extract_status.setText(
            f'Done. Profile "{profile_name}" stored ({n_af} AF spectra).'
        )
        self._refresh_profile_list()
        self._rebuild_assignment_grid()

        if self.bus:
            self.bus.statusMessage.emit(
                f'AutoSpectral: "{profile_name}" stored ({n_af} spectra).'
            )

    def _on_extraction_error(self, msg: str):
        self._extract_btn.setEnabled(True)
        self._extract_status.setText(f'Error: {msg}')
        logger.error(f'AutoSpectral extraction error: {msg}')

    # ======================================================================
    # Section 2 — profile list and spectral plot
    # ======================================================================

    def _refresh_profile_list(self):
        """Rebuild profile list, restore or default selection, and draw the plot."""
        current_name = (
            self._profile_list.currentItem().text()
            if self._profile_list.currentItem() else None
        )

        self._profile_list.blockSignals(True)
        self._profile_list.clear()

        af_profiles = self.controller.experiment.process.get('af_profiles', {})
        for name in af_profiles:
            self._profile_list.addItem(name)

        # Restore previous selection, or default to most recent (last) item
        restored = False
        if current_name:
            items = self._profile_list.findItems(current_name, Qt.MatchExactly)
            if items:
                self._profile_list.setCurrentItem(items[0])
                restored = True
        if not restored and self._profile_list.count() > 0:
            # Select the last item — most recently extracted profile
            self._profile_list.setCurrentRow(self._profile_list.count() - 1)

        self._profile_list.blockSignals(False)

        # Always draw regardless of whether signals fired
        self._draw_selected_profile()
        self._refresh_comparison_controls()

    def _on_profile_row_changed(self, _row: int):
        if not hasattr(self, '_save_btn'):
            return
        has = self._profile_list.currentItem() is not None
        self._save_btn.setEnabled(has)
        self._delete_btn.setEnabled(has)
        self._draw_selected_profile()

    def _draw_selected_profile(self):
        """Draw the spectral plot for the currently selected profile."""
        if not hasattr(self, '_save_btn'):
            return
        self._profile_plot.clear()
        item = self._profile_list.currentItem()
        has = item is not None
        self._save_btn.setEnabled(has)
        self._delete_btn.setEnabled(has)
        if not has:
            return

        af_profiles = self.controller.experiment.process.get('af_profiles', {})
        entry = af_profiles.get(item.text())
        if entry is None:
            return

        af_spectra = np.array(entry['spectra'])
        n_af, n_ch = af_spectra.shape
        colours = pg.colormap.get('viridis').getLookupTable(
            nPts=max(n_af, 2), alpha=False
        )
        x = np.arange(n_ch)

        # Set detector labels on x-axis with per-laser colour coding
        channel_names = entry.get('channel_names', [])
        if channel_names and len(channel_names) == n_ch:
            # channel_names carry the '-A' suffix; strip for display
            display_names = [n.removesuffix('-A') for n in channel_names]
            ticks = [[(i, dn) for i, dn in enumerate(display_names)], []]
            self._profile_plot_axis.setTicks(ticks)

            db_col = self.controller.experiment.settings['raw'].get('cytometer_db_col')
            if db_col:
                detector_laser_map = get_detector_laser_map(db_col)
                tick_colors = {
                    dn: LASER_LABEL_COLORS[laser]
                    for dn, name in zip(display_names, channel_names)
                    if (laser := detector_laser_map.get(name)) in LASER_LABEL_COLORS
                }
            else:
                tick_colors = {}
            self._profile_plot_axis.tick_colors = tick_colors
        else:
            self._profile_plot_axis.setTicks(None)
            self._profile_plot_axis.tick_colors = {}

        for i, row in enumerate(af_spectra):
            pen = (pg.mkPen(color=(0, 0, 0), width=2) if i == 0
                else pg.mkPen(color=colours[i], width=1))
            self._profile_plot.plot(x, np.maximum(row, 0), pen=pen)

    def _save_selected_profile_csv(self):
        item = self._profile_list.currentItem()
        if item is None:
            return
        name = item.text()
        entry = self.controller.experiment.process.get('af_profiles', {}).get(name)
        if entry is None:
            return

        default_path = str(
            self.controller.experiment_dir / 'AutoSpectral' / f'{name}.csv'
        )
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save AF Profile CSV', default_path, 'CSV files (*.csv)'
        )
        if not path:
            return

        try:
            import pandas as pd
            af_spectra = np.array(entry['spectra'])
            channel_names = entry['channel_names']
            n_af = af_spectra.shape[0]
            row_labels = ['mean'] + [str(i) for i in range(1, n_af)]
            df = pd.DataFrame(af_spectra, columns=channel_names)
            df.insert(0, 'AF_index', row_labels)
            df.to_csv(path, index=False)
            if self.bus:
                self.bus.statusMessage.emit(f'AutoSpectral: saved "{name}" to {path}')
        except Exception as e:
            logger.error(f'AutoSpectral: CSV save failed: {e}')

    def _load_profile_csv(self):
        default_dir = str(self.controller.experiment_dir / 'AutoSpectral')
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load AF Profile CSV', default_dir, 'CSV files (*.csv)'
        )
        if not path:
            return
        try:
            profile_name, af_spectra, channel_names = load_af_profile_csv(path)
        except Exception as e:
            logger.error(f'AutoSpectral: CSV load failed: {e}')
            return

        af_profiles = self.controller.experiment.process.get('af_profiles', {})
        af_profiles[profile_name] = {
            'spectra': af_spectra.tolist(),
            'source_fcs': path,
            'channel_names': channel_names,
        }
        self.controller.experiment.process['af_profiles'] = af_profiles
        self.bus.autoSaveRequested.emit()

        # Cache precomputed matrices for the loaded profile.
        self.controller.cache_af_profile(profile_name)

        self._refresh_profile_list()
        self._rebuild_assignment_grid()
        if self.bus:
            self.bus.statusMessage.emit(
                f'AutoSpectral: loaded profile "{profile_name}" from CSV.'
            )

    def _delete_selected_profile(self):
        item = self._profile_list.currentItem()
        if item is None:
            return
        name = item.text()

        af_profiles = self.controller.experiment.process.get('af_profiles', {})
        af_profiles.pop(name, None)
        self.controller.experiment.process['af_profiles'] = af_profiles

        sample_af = self.controller.experiment.samples.get('sample_af_profiles', {})
        for sp in sample_af:
            if name in sample_af[sp]:
                sample_af[sp].remove(name)
        self.controller.experiment.samples['sample_af_profiles'] = sample_af
        self.bus.autoSaveRequested.emit()

        self.controller.initialise_af_matrices()
        self._refresh_profile_list()
        self._rebuild_assignment_grid()
        if self.bus:
            self.bus.statusMessage.emit(f'AutoSpectral: deleted profile "{name}".')

    # ======================================================================
    # Section 3 — assignment grid (non-SSC samples only)
    # ======================================================================

    def _rebuild_assignment_grid(self):
        """
        Rebuild the samples × profiles checkbox grid.
        Only non-Single-Stain-Control samples are shown.
        The profile-name header lives in a separate frozen widget/layout
        (self._assign_header_layout) above the scrollable body grid, kept
        column-aligned with it. Body row 0 = first sample (no header row
        inside the scrolling grid anymore).
        """
        self._assign_checkboxes.clear()
        while self._assign_header_layout.count():
            item = self._assign_header_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        while self._assign_grid_layout.count():
            item = self._assign_grid_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        af_profiles = self.controller.experiment.process.get('af_profiles', {})
        profile_names = list(af_profiles.keys())
        all_samples = self.controller.experiment.samples.get('all_samples', {})
        ssc = set(self.controller.experiment.samples.get('single_stain_controls', []))
        # Exclude SSC from the assignment grid
        non_ssc = {p: n for p, n in all_samples.items() if p not in ssc}
        sample_af = self.controller.experiment.samples.get('sample_af_profiles', {})

        if not profile_names or not non_ssc:
            self._assign_grid_layout.addWidget(
                QLabel('No profiles or non-SSC samples available yet.'), 0, 0
            )
            self._assign_header_scroll.setFixedHeight(1)
            self._assign_scroll.setMinimumHeight(60)
            self._assign_scroll.setMaximumHeight(60)
            return

        # Header (frozen row, own layout — not row 0 of the scrolling body)
        corner = QLabel('Sample')
        corner.setStyleSheet('font-weight: bold;')
        self._assign_header_layout.addWidget(corner, 0, 0, Qt.AlignLeft)

        AF_SUFFIX = ' AutoSpectral AF'
        for col, pname in enumerate(profile_names, start=1):
            display = pname
            if display.endswith(AF_SUFFIX):
                display = display[:-len(AF_SUFFIX)]
            if len(display) > 28:
                display = display[:26] + '…'
            lbl = QLabel(display)
            lbl.setStyleSheet('font-weight: bold;')
            lbl.setToolTip(pname)
            self._assign_header_layout.addWidget(lbl, 0, col, Qt.AlignCenter)

        # Sample rows (scrolling body — row 0 is now the first sample)
        for row, (sample_path, display_name) in enumerate(non_ssc.items()):
            name_lbl = QLabel(display_name)
            name_lbl.setToolTip(sample_path)
            if sample_path == self.controller.current_sample_path:
                name_lbl.setStyleSheet('font-weight: bold;')
            self._assign_grid_layout.addWidget(name_lbl, row, 0, Qt.AlignLeft)

            assigned = sample_af.get(sample_path, [])
            for col, pname in enumerate(profile_names, start=1):
                cb = QCheckBox()
                cb.setChecked(pname in assigned)
                cb.setToolTip(f'Apply "{pname}" to {display_name}')
                cb.checkStateChanged.connect(
                    self._make_assignment_handler(sample_path, pname, cb)
                )
                self._assign_grid_layout.addWidget(cb, row, col, Qt.AlignCenter)
                self._assign_checkboxes[(sample_path, pname)] = cb

        # Force header and body columns to the same pixel width so checkboxes
        # line up under their profile-name label, regardless of which is wider.
        n_cols = len(profile_names) + 1
        for col in range(n_cols):
            header_item = self._assign_header_layout.itemAtPosition(0, col)
            body_item = self._assign_grid_layout.itemAtPosition(0, col)
            header_w = header_item.widget().sizeHint().width() if header_item else 0
            body_w = body_item.widget().sizeHint().width() if body_item else 0
            width = max(header_w, body_w)
            self._assign_header_layout.setColumnMinimumWidth(col, width)
            self._assign_grid_layout.setColumnMinimumWidth(col, width)

        # Size the header strip to fit one row of (bold) labels exactly.
        self._assign_header_layout.activate()
        row_height = max(
            self._assign_header_layout.itemAtPosition(0, col).widget().sizeHint().height()
            for col in range(n_cols)
        )
        margins = self._assign_header_layout.contentsMargins()
        self._assign_header_scroll.setFixedHeight(
            row_height + margins.top() + margins.bottom() + 10
        )

        # Size the body strip to fit ALL sample rows (up to a generous cap),
        # so the grid doesn't need its own inner scrollbar for typical sample
        # counts — the tab's outer QScrollArea handles page-level scrolling
        # instead. Beyond the cap, the inner scrollbar takes back over.
        self._assign_grid_layout.activate()
        n_rows = len(non_ssc)
        row_heights = [
            self._assign_grid_layout.itemAtPosition(r, 0).widget().sizeHint().height()
            for r in range(n_rows)
        ]
        grid_margins = self._assign_grid_layout.contentsMargins()
        content_height = (
            sum(row_heights)
            + self._assign_grid_layout.spacing() * max(n_rows - 1, 0)
            + grid_margins.top() + grid_margins.bottom()
        )

        MIN_ASSIGN_HEIGHT = 240   # doubled from the old fixed 120px minimum
        MAX_ASSIGN_HEIGHT = 900   # safety cap for very large sample counts
        target_height = max(MIN_ASSIGN_HEIGHT, min(content_height, MAX_ASSIGN_HEIGHT))
        self._assign_scroll.setMinimumHeight(MIN_ASSIGN_HEIGHT)
        self._assign_scroll.setMaximumHeight(MAX_ASSIGN_HEIGHT)
        self._assign_scroll.setFixedHeight(target_height)

    def _make_assignment_handler(self, sample_path: str, profile_name: str, cb: QCheckBox):
        def handler(_state):
            sample_af = self.controller.experiment.samples.get('sample_af_profiles', {})
            assigned = list(sample_af.get(sample_path, []))
            if cb.isChecked():
                if profile_name not in assigned:
                    assigned.append(profile_name)
            else:
                if profile_name in assigned:
                    assigned.remove(profile_name)
            sample_af[sample_path] = assigned
            self.controller.experiment.samples['sample_af_profiles'] = sample_af
            self.bus.autoSaveRequested.emit()

            # Update the controller's cached AF matrices for this sample so
            # that the next load_sample() call picks up the new assignment
            # without needing to recompute anything.
            if sample_path == self.controller.current_sample_path:
                self.controller.initialise_af_matrices()
                # Re-apply unmixing with the new AF assignment so the Unmixed
                # Data tab updates immediately without requiring a sample reload.
                if self.controller.raw_event_data is not None:
                    self.controller.unmixed_event_data = self.controller._apply_unmixing(
                        self.controller.raw_event_data
                    )
                    self.controller.clear_data_for_cytometry_plots()
                    self.controller.initialise_data_for_cytometry_plots()

            n = len(assigned)
            if sample_path == self.controller.current_sample_path:
                self._assign_status.setText(
                    f'Current sample: {n} profile(s) assigned — Unmixed Data tab updated.'
                    if n else
                    'Current sample: no AF assigned — reverted to standard OLS unmixing.'
                )
            else:
                self._assign_status.setText(
                    f'{n} profile(s) assigned to {Path(sample_path).stem}.'
                    if n else
                    f'No AF assigned to {Path(sample_path).stem}.'
                )
        return handler

    def _clear_af_for_current_sample(self):
        sample_path = self.controller.current_sample_path
        if sample_path is None:
            self._assign_status.setText('No sample currently loaded.')
            return

        sample_af = self.controller.experiment.samples.get('sample_af_profiles', {})
        sample_af[sample_path] = []
        self.controller.experiment.samples['sample_af_profiles'] = sample_af
        self.bus.autoSaveRequested.emit()

        self.controller.initialise_af_matrices()

        # Re-apply unmixing immediately so the Unmixed Data tab reflects the change.
        if self.controller.raw_event_data is not None:
            self.controller.unmixed_event_data = self.controller._apply_unmixing(
                self.controller.raw_event_data
            )
            self.controller.clear_data_for_cytometry_plots()
            self.controller.initialise_data_for_cytometry_plots()

        self._rebuild_assignment_grid()
        self._assign_status.setText(
            'AF cleared for current sample — reverted to standard OLS unmixing.'
        )
        if self.bus:
            self.bus.statusMessage.emit(
                f'AutoSpectral: AF cleared for {Path(sample_path).stem}.'
            )

    def _assign_selected_profile_to_all(self):
        """Assign the profile selected in the profile list to every non-SSC sample."""
        item = self._profile_list.currentItem()
        if item is None:
            self._assign_status.setText(
                'No profile selected — pick one from the Stored AF Profiles list first.'
            )
            return

        profile_name = item.text()
        all_samples = self.controller.experiment.samples.get('all_samples', {})
        ssc = set(self.controller.experiment.samples.get('single_stain_controls', []))
        non_ssc = [p for p in all_samples if p not in ssc]

        if not non_ssc:
            self._assign_status.setText('No non-SSC samples available.')
            return

        sample_af = self.controller.experiment.samples.get('sample_af_profiles', {})
        for sample_path in non_ssc:
            assigned = list(sample_af.get(sample_path, []))
            if profile_name not in assigned:
                assigned.append(profile_name)
            sample_af[sample_path] = assigned

        self.controller.experiment.samples['sample_af_profiles'] = sample_af
        self.bus.autoSaveRequested.emit()

        # If the current sample was affected, reinitialise its AF matrices and
        # re-apply unmixing so the Unmixed Data tab updates immediately.
        if self.controller.current_sample_path in non_ssc:
            self.controller.initialise_af_matrices()
            if self.controller.raw_event_data is not None:
                self.controller.unmixed_event_data = self.controller._apply_unmixing(
                    self.controller.raw_event_data
                )
                self.controller.clear_data_for_cytometry_plots()
                self.controller.initialise_data_for_cytometry_plots()

        self._rebuild_assignment_grid()
        self._assign_status.setText(
            f'"{profile_name}" assigned to {len(non_ssc)} sample(s).'
        )
        if self.bus:
            self.bus.statusMessage.emit(
                f'AutoSpectral: "{profile_name}" assigned to all {len(non_ssc)} non-SSC sample(s).'
            )

    def _clear_all_af(self):
        self.controller.experiment.samples['sample_af_profiles'] = {}
        self.controller.experiment.save()

        self.controller.initialise_af_matrices()

        # Re-apply unmixing immediately so the Unmixed Data tab reflects the change.
        if self.controller.raw_event_data is not None:
            self.controller.unmixed_event_data = self.controller._apply_unmixing(
                self.controller.raw_event_data
            )
            self.controller.clear_data_for_cytometry_plots()
            self.controller.initialise_data_for_cytometry_plots()

        self._rebuild_assignment_grid()
        self._assign_status.setText(
            'All AF assignments cleared — reverted to standard OLS unmixing.'
        )
        if self.bus:
            self.bus.statusMessage.emit('AutoSpectral: all AF assignments cleared.')

    # ======================================================================
    # Section 4 — comparison plots
    # ======================================================================

    def _refresh_comparison_controls(self):
        """Populate the AF profile combo."""
        self._cmp_profile_combo.blockSignals(True)
        current = self._cmp_profile_combo.currentText()
        self._cmp_profile_combo.clear()
        self._cmp_profile_combo.addItem('Assigned to sample', userData='__assigned__')
        af_profiles = self.controller.experiment.process.get('af_profiles', {})
        for name in af_profiles:
            self._cmp_profile_combo.addItem(name, userData=name)
        idx = self._cmp_profile_combo.findText(current)
        if idx >= 0:
            self._cmp_profile_combo.setCurrentIndex(idx)
        self._cmp_profile_combo.blockSignals(False)

    def _clear_comparison_plots(self):
        self._plot_ols.clear()
        self._plot_af.clear()
        self._ols_data = None
        self._af_data = None
        self._last_cmp_state = None
        self._pending_cmp_state = None
        self._cmp_status.setText('')

    def _run_comparison(self):
        # Guard: don't start a new comparison if one is already running.
        if self._cmp_running:
            self._cmp_status.setText(
                'Comparison already running — please wait.'
            )
            return

        if self.controller.raw_event_data is None:
            self._cmp_status.setText('No sample loaded.')
            return
        if self.controller.transfer_matrix is None:
            self._cmp_status.setText('No unmixing matrix — run spectral process first.')
            return

        profile_key = self._cmp_profile_combo.currentData()
        if profile_key == '__assigned__':
            af_spectra = self.controller.get_combined_af_spectra_for_sample(
                self.controller.current_sample_path
            )
        else:
            entry = self.controller.experiment.process.get('af_profiles', {}).get(profile_key)
            af_spectra = np.array(entry['spectra']) if entry else None

        if af_spectra is None:
            self._cmp_status.setText(
                'No AF profile available. '
                'Extract a profile (Step 1) and assign it, or pick one from the dropdown.'
            )
            return

        fluor_spectra = self._build_fluor_spectra()
        if fluor_spectra is None:
            self._cmp_status.setText('Fluorophore profiles unavailable.')
            return

        # Use cached precomputed matrices where possible.
        # For the "assigned" case, controller.af_precomputed is already the
        # correctly combined result — no recomputation needed.
        # For a single named profile, use its cached entry if available.
        if profile_key == '__assigned__' and self.controller.af_precomputed is not None:
            af_precomputed = self.controller.af_precomputed
        elif profile_key != '__assigned__' and profile_key in self.controller.af_precomputed_cache:
            af_precomputed = self.controller.af_precomputed_cache[profile_key]
        else:
            # Fallback: compute now (e.g. profile was just loaded from CSV and
            # cache hasn't been populated yet, or fluor_spectra changed).
            try:
                af_precomputed = precompute_af_matrices(fluor_spectra, af_spectra)
            except Exception as e:
                self._cmp_status.setText(f'Precompute error: {e}')
                return

        self._plot_ols.set_status('Computing...')
        self._plot_af.set_status('Computing...')
        self._cmp_status.setText('Running unmixing on current sample...')

        # Build a state key describing exactly what this run will compute.
        # If it matches the last completed run, the cached arrays are still
        # valid — skip the worker and just redraw.
        spillover = self.controller.experiment.process.get('spillover')
        spillover_key = tuple(np.array(spillover).ravel()) if spillover is not None else None
        # Use a content hash of the precomputed P matrix rather than id()
        p_matrix = af_precomputed.get('P') if af_precomputed is not None else None
        af_key = bytes(p_matrix.data) if p_matrix is not None else None
        state_key = (
            self.controller.current_sample_path,
            profile_key,
            af_key,
            spillover_key,
        )
        if state_key == self._last_cmp_state and self._ols_data is not None:
            self._plot_ols.set_status('')
            self._plot_af.set_status('')
            self._cmp_status.setText(
                f'No change — redrawing from cached result '
                f'({self._ols_data.shape[0]:,} events).'
            )
            self._initialise_comparison_plots()
            return

        self._pending_cmp_state = state_key
        self._cmp_running = True
        self._cmp_thread = QThread()
        _raw = self.controller.experiment.settings['raw']
        _pnn_raw = _raw.get('whitelisted_pnn') or _raw['event_channels_pnn']
        _full_pnn_raw = _raw['event_channels_pnn']
        _fl_ids_remapped = [_pnn_raw.index(_full_pnn_raw[i])
                            for i in self.controller.filtered_raw_fluorescence_channel_ids]
        self._cmp_worker = ComparisonWorker(
            self.controller.raw_event_data,
            self.controller.transfer_matrix,
            af_precomputed,
            af_spectra,
            self.controller.experiment.settings,
            _fl_ids_remapped,
            spillover=self.controller.experiment.process.get('spillover'),
        )
        self._cmp_worker.moveToThread(self._cmp_thread)
        self._cmp_thread.started.connect(self._cmp_worker.run)
        self._cmp_worker.finished.connect(self._on_comparison_finished)
        self._cmp_worker.error.connect(self._on_comparison_error)
        self._cmp_worker.finished.connect(self._cmp_thread.quit)
        self._cmp_worker.error.connect(self._cmp_thread.quit)
        self._cmp_thread.finished.connect(self._cmp_thread.deleteLater)
        self._cmp_thread.finished.connect(self._on_cmp_thread_finished)
        self._cmp_thread.start()

    def _on_comparison_finished(self, ols_data: np.ndarray, af_data: np.ndarray):
        self._ols_data = ols_data
        self._af_data = af_data
        self._last_cmp_state = self._pending_cmp_state
        self._cmp_status.setText(f'Done. {ols_data.shape[0]:,} events.')
        self._plot_ols.set_status('')
        self._plot_af.set_status('')
        self._initialise_comparison_plots()

    def _on_comparison_error(self, msg: str):
        self._plot_ols.set_status('Error')
        self._plot_af.set_status('Error')
        self._cmp_status.setText(f'Error: {msg}')
        logger.error(f'AutoSpectral comparison error: {msg}')

    def _on_cmp_thread_finished(self):
        """Called when the QThread finishes (after deleteLater is queued).
        Clears the running flag and drops the Python reference to the thread
        so the guard in _run_comparison works correctly next time."""
        self._cmp_running = False
        self._cmp_thread = None
        self._cmp_worker = None

    def _initialise_comparison_plots(self):
        """
        Feed event data to both plot widgets and initialise their transforms
        from the live unmixed_transformations.  Pick default X/Y channels.
        Existing channel, scaling, and gate selections are preserved.
        """
        if self._ols_data is None or self._af_data is None:
            return

        pnn = self.controller.experiment.settings['unmixed'].get('event_channels_pnn', [])
        fl_ids = self.controller.experiment.settings['unmixed'].get('fluorescence_channel_ids', [])
        if not pnn or not fl_ids or len(fl_ids) < 2:
            return

        fl_names = [pnn[i] for i in fl_ids]
        # Default: first two fluorescence channels
        ch_x = fl_names[0]
        ch_y = fl_names[1]

        # Preserve existing channel selections if both plots already have them
        if (self._plot_ols._channel_x in fl_names
                and self._plot_ols._channel_y in fl_names):
            ch_x = self._plot_ols._channel_x
            ch_y = self._plot_ols._channel_y

        # Preserve per-channel transform state (limits, logicle_w, etc.) and gate
        # so that "Update Plots" does not reset zooming/scaling or gating.
        def _snapshot_transforms(plot):
            """Return a dict of {channel: (id, limits, logicle_w, logicle_a)} for current channels."""
            snap = {}
            for ch in (plot._channel_x, plot._channel_y):
                if ch and ch in plot._transformations:
                    tr = plot._transformations[ch]
                    snap[ch] = (tr.id, tr.limits, tr.logicle_w, tr.logicle_a)
            return snap

        def _restore_transforms(plot, snap):
            for ch, (tid, limits, lw, la) in snap.items():
                if ch in plot._transformations:
                    tr = plot._transformations[ch]
                    tr.logicle_w = lw
                    tr.logicle_a = la
                    tr.set_transform(tid, limits)

        snap_ols = _snapshot_transforms(self._plot_ols)
        snap_af = _snapshot_transforms(self._plot_af)
        gate_ols = self._plot_ols._source_gate
        gate_af = self._plot_af._source_gate

        self._plot_ols.set_event_data(self._ols_data)
        self._plot_ols.initialise_from_controller(ch_x, ch_y)
        _restore_transforms(self._plot_ols, snap_ols)
        self._plot_ols._source_gate = gate_ols
        self._plot_ols._configure_axes()
        self._plot_ols.redraw()

        self._plot_af.set_event_data(self._af_data)
        self._plot_af.initialise_from_controller(ch_x, ch_y)
        _restore_transforms(self._plot_af, snap_af)
        self._plot_af._source_gate = gate_af
        self._plot_af._configure_axes()
        self._plot_af.redraw()

    def _on_ols_gate_changed(self, gate_name: str):
        """Sync the AF plot's source gate when the OLS plot's gate changes."""
        self._plot_af.set_source_gate(gate_name)

    def _on_af_gate_changed(self, gate_name: str):
        """Sync the OLS plot's source gate when the AF plot's gate changes."""
        self._plot_ols.set_source_gate(gate_name)

    def _on_ols_channel_changed(self, ch_x: str, ch_y: str):
        """Mirror channel selection from OLS to AF plot."""
        self._plot_af.blockSignals(True)
        self._plot_af.set_channels(ch_x, ch_y)
        self._plot_af.blockSignals(False)

    def _on_af_channel_changed(self, ch_x: str, ch_y: str):
        """Mirror channel selection from AF to OLS plot."""
        self._plot_ols.blockSignals(True)
        self._plot_ols.set_channels(ch_x, ch_y)
        self._plot_ols.blockSignals(False)

    def _on_ols_scaling_changed(self, axis_name: str, tr):
        """Mirror zoom/scaling from OLS to AF plot."""
        self._plot_af.blockSignals(True)
        self._plot_af.set_scaling(axis_name, tr)
        self._plot_af.blockSignals(False)

    def _on_af_scaling_changed(self, axis_name: str, tr):
        """Mirror zoom/scaling from AF to OLS plot."""
        self._plot_ols.blockSignals(True)
        self._plot_ols.set_scaling(axis_name, tr)
        self._plot_ols.blockSignals(False)

    # ======================================================================
    # Shared helper
    # ======================================================================

    def _build_fluor_spectra(self) -> np.ndarray | None:
        profiles = self.controller.experiment.process.get('profiles')
        spectral_model = self.controller.experiment.process.get('spectral_model', [])
        if not profiles or not spectral_model:
            return None
        labels = [c['label'] for c in spectral_model]
        try:
            rows = [profiles[label] for label in labels if label in profiles]
            return np.array(rows) if rows else None
        except Exception as e:
            logger.error(f'AutoSpectral: _build_fluor_spectra: {e}')
            return None

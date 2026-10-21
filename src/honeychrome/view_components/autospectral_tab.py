"""
autospectral_tab.py
--------------------
Honeychrome integrated tab: AutoSpectral.

Location: src/honeychrome/view_components/autospectral_tab.py

Pages (side navigation)
-----------------------
Autofluorescence    AutofluorescencePage (this module)
    1  Extract AF profile from an unstained sample (SOM, KMeans fallback).
    2  Manage stored AF profiles (spectral plot, CSV load/save/delete; right-click
       a trace to delete that spectrum).
    3  Assign AF profiles to samples (grid: rows=samples, columns=profiles; unstained
       and single-stained controls optionally included), to all samples, to the
       samples of a folder, or automatically from scatter (af_auto_assign).
    4  Side-by-side OLS vs AF-corrected biplot comparison, using the same
       density-heatmap rendering, logicle transforms, and axis-drag zoom as the
       Unmixed Data tab.
Noise               NoisePage (autospectral_noise_page.py)
Spectral Variants   SpectralVariantsPage (autospectral_variants_page.py)
Spectral Correction SpectralCorrectionPage (autospectral_correction_page.py)

Comparison plots open on the Singlets gate, else Cells, else root
(default_comparison_gate).
"""

import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import colorcet as cc
from PySide6.QtCore import Qt, QThread, Signal, QObject, QRectF, QSettings, QTimer, QEvent
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel,
    QPushButton, QSpinBox, QComboBox, QGroupBox,
    QFormLayout, QSizePolicy, QFileDialog, QListWidget, QListWidgetItem,
    QAbstractItemView, QSplitter, QCheckBox, QHeaderView,
    QTableWidget, QTableWidgetItem, QMenu, QMessageBox,
)
import pyqtgraph as pg
import warnings

from honeychrome.controller_components.functions import (
    sample_from_fcs,
    apply_transfer_matrix,
    apply_gates_in_place,
    build_display_label_map,
)
from honeychrome.controller_components.autospectral_functions import (
    get_af_spectra,
    apply_af_transfer,
    save_af_profile_csv,
    load_af_profile_csv,
    af_index_lookup,
    make_envelope_entry,
    envelope_sample_name,
    save_envelope_sample,
)
from honeychrome.controller_components.spread_model import (
    spread_is_current,
    spread_snapshot,
    compute_spread_entry,
)
from honeychrome.controller_components.transform import Transform
from honeychrome.controller_components.job_gate import acquire_job, guard_while_busy, wait_if_busy
from honeychrome.controller_components.sample_loader import resolve_af_for_profiles
from honeychrome.controller_components.particle_type import (
    is_bead_sample, scatter_channel_pair, scatter_ceilings,
)
from honeychrome.controller_components.af_auto_assign import (
    POOR_FIT_SHARE, folder_groups, reference_sources, suggest_af_profiles,
)
from honeychrome.view_components.cytometry_plot_components import (
    InteractiveLabel,
    NoPanViewBox,
    ZoomAxis,
    TransparentGraphicsLayoutWidget,
)
from honeychrome.view_components.profiles_viewer import BottomAxisVerticalTickLabels
from honeychrome.view_components.help_toggle_widget import HelpToggleWidget, WheelBlocker
from honeychrome.view_components.side_tab_widget import SideTabWidget
from honeychrome.controller_components.cytometer_whitelist import (
    get_detector_laser_map, LASER_LABEL_COLORS,
)
import honeychrome.settings as settings

import logging

from honeychrome.view_components.help_texts import autospectral_af_help_text

logger = logging.getLogger(__name__)

TAB_NAME = 'AutoSpectral'
PAGE_SETTING = 'autospectral_page'

# Comparison plots open on the first of these gates that exists (any case).
COMPARISON_GATE_PREFERENCE = ('Singlets', 'Cells')
# A right-click this close to a profile trace (pixels) picks that spectrum.
PROFILE_PICK_TOLERANCE_PX = 8

# Assignment table column widths (pixels): defaults fit the text up to the
# caps; the user can drag any divider down to the minimum.
ASSIGN_MIN_COL_WIDTH = 40
ASSIGN_MAX_NAME_WIDTH = 320
ASSIGN_MAX_PROFILE_WIDTH = 200
ASSIGN_COL_PADDING = 24
ASSIGN_MIN_HEIGHT = 240
ASSIGN_MAX_HEIGHT = 900
AF_PROFILE_SUFFIX = ' AutoSpectral AF'


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class AfTrainingWorker(QObject):
    """
    Runs get_af_spectra in a QThread, then builds the negative envelope and,
    when spread_job is given, estimates the spillover spread. Emits a dict
    with af_spectra, envelope, envelope_sample and spread.
    """
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, unstained_raw, fluor_spectra, som_dim, source_fcs_path,
                 fluor_names=None, spread_job=None):
        super().__init__()
        self.unstained_raw = unstained_raw
        self.fluor_spectra = fluor_spectra
        self.som_dim = som_dim
        self.source_fcs_path = source_fcs_path   # stored so the slot can read it
        self.fluor_names = fluor_names
        self.spread_job = spread_job

    def run(self):
        try:
            self.progress.emit('Clustering unstained sample and refining AF spectra...')
            af_spectra = get_af_spectra(
                self.unstained_raw,
                self.fluor_spectra,
                som_dim=self.som_dim,
            )
            envelope, sample, spread = None, None, None
            if self.fluor_names is not None:
                self.progress.emit('Building the negative envelope')
                envelope, sample = make_envelope_entry(
                    self.unstained_raw, self.fluor_names, self.fluor_spectra, af_spectra)
                if self.spread_job is not None:
                    try:
                        spread = self.spread_job(self.progress.emit)
                    except Exception as e:
                        logger.warning(f'AutoSpectral: spillover spread not estimated: {e}')
            self.finished.emit({'af_spectra': af_spectra, 'envelope': envelope,
                                'envelope_sample': sample, 'spread': spread})
        except Exception as e:
            self.error.emit(str(e))


class AfAutoAssignWorker(QObject):
    """Runs af_auto_assign.suggest_af_profiles in a QThread; emits its result."""
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, experiment_dir, references, targets, pair, ceilings):
        super().__init__()
        self.experiment_dir = experiment_dir
        self.references = references
        self.targets = targets
        self.pair = pair
        self.ceilings = ceilings

    def run(self):
        try:
            result = suggest_af_profiles(self.experiment_dir, self.references, self.targets,
                                         self.pair, self.ceilings, progress=self.progress.emit)
            self.finished.emit(result)
        except Exception as e:
            logger.exception('AutoSpectral: auto-assign failed')
            self.error.emit(str(e) or type(e).__name__)


class ComparisonWorker(QObject):
    """Computes OLS and AF-corrected unmixed arrays in a QThread."""
    finished = Signal(object, object)   # (ols_data, af_data) full arrays
    error = Signal(str)

    def __init__(self, raw_event_data, transfer_matrix,
                 af_precomputed, af_spectra, exp_settings,
                 filtered_fl_ids_raw, spillover=None, af_index_map=None):
        super().__init__()
        self.raw_event_data = raw_event_data
        self.transfer_matrix = transfer_matrix
        self.af_precomputed = af_precomputed
        self.af_spectra = af_spectra
        self.exp_settings = exp_settings
        self.filtered_fl_ids_raw = filtered_fl_ids_raw
        self.spillover = spillover
        self.af_index_map = af_index_map

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
                af_index_map=self.af_index_map,
            )
            self.finished.emit(ols_data, af_result['unmixed'])
        except Exception as e:
            self.error.emit(str(e))


def pick_most_affected_channels(
    data_a: np.ndarray,
    data_b: np.ndarray,
    pnn: list[str],
    fl_names: list[str],
    min_scale: float = 1e-6,
) -> tuple[str, str] | None:
    """
    Return the two fluorescence channels with the largest *proportional*
    change between data_a and data_b (e.g. OLS vs AF-corrected, or
    AF-corrected vs Optimization).

    Raw mean(|a - b|) is biased toward whichever channels happen to have the
    largest dynamic range. Instead, each channel's mean absolute difference
    is normalised by that channel's own robust spread (5th-95th percentile
    of data_a), so a small absolute shift in an otherwise quiet channel can
    outrank a larger absolute shift in a channel that is naturally bright/
    noisy. min_scale floors the denominator to avoid blow-ups on flat or
    all-zero channels.

    Returns None if fewer than two fluorescence channels are available or the
    arrays don't align with pnn.
    """
    if len(fl_names) < 2:
        return None
    try:
        cols = [pnn.index(ch) for ch in fl_names]
    except ValueError:
        return None
    a = data_a[:, cols]
    b = data_b[:, cols]
    diff = np.abs(a - b).mean(axis=0)
    scale = np.percentile(a, 95, axis=0) - np.percentile(a, 5, axis=0)
    scale = np.maximum(scale, min_scale)
    proportional_diff = diff / scale
    top2 = np.argsort(proportional_diff)[-2:][::-1]
    return fl_names[top2[0]], fl_names[top2[1]]


def default_comparison_gate(gate_names) -> str:
    """The first of COMPARISON_GATE_PREFERENCE in gate_names (any case), else 'root'."""
    by_lower = {str(g).lower(): g for g in gate_names}
    for name in COMPARISON_GATE_PREFERENCE:
        if name.lower() in by_lower:
            return by_lower[name.lower()]
    return 'root'


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

    def __init__(self, title: str, controller, parent=None, has_af_channels: bool = True):
        super().__init__(parent)
        self.controller = controller
        self._title_text = title
        # False for plain (OLS) unmixed data, which has no AF Abundance or AF
        # Index values: a source gate drawn on those channels cannot be
        # applied, and the plot says so instead of drawing.
        self._has_af_channels = has_af_channels
        self._af_gate_message = ''

        # Local copies of Transform objects (not shared with Unmixed Data tab)
        self._transformations: dict[str, Transform] = {}
        # Event data for this plot (full array, gate mask applied only for display)
        self._event_data: np.ndarray | None = None
        # Current source gate name; None until _configure_title picks the default.
        self._source_gate: str | None = None
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
        The source gate is kept (default_comparison_gate the first time).
        """
        src = self.controller.unmixed_transformations or {}
        self._transformations = {}
        for ch, tr in src.items():
            self._transformations[ch] = deepcopy(tr)

        self._channel_x = channel_x
        self._channel_y = channel_y
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
            self._source_gate = default_comparison_gate(gate_names)

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

        blocked = None if self._has_af_channels else self._gate_using_af_channels()
        if blocked is not None:
            gate_name, channel = blocked
            self._af_gate_message = (f"Gate '{gate_name}' uses {channel}, which unmixing "
                                     f"without AF correction does not produce.")
            self.img.clear()
            self.set_status(self._af_gate_message)
            return
        if self._af_gate_message:
            self._af_gate_message = ''
            self.set_status('')

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

    def _gate_using_af_channels(self) -> tuple[str, str] | None:
        """(gate, channel) for the first gate from root to the source gate
        that is drawn on an AF channel, or None."""
        if self._source_gate in (None, 'root'):
            return None
        gating = self.controller.unmixed_gating
        if gating is None:
            return None
        try:
            paths = gating.find_matching_gate_paths(self._source_gate)
        except Exception:
            return None
        if not paths:
            return None
        for gate_name in [g for g in paths[0] if g != 'root'] + [self._source_gate]:
            try:
                gate = gating.get_gate(gate_name)
                if gate.gate_type == 'QuadrantGate':
                    channels = [d.dimension_ref for d in gate.dimensions]
                else:
                    channels = list(gate.get_dimension_ids())
            except Exception:
                continue
            for ch in channels:
                if ch in settings.af_channels:
                    return gate_name, ch
        return None

    def _compute_gate_mask(self, event_data: np.ndarray) -> np.ndarray | None:
        """
        Compute a per-event boolean mask for self._source_gate applied to
        event_data, with the controller's unmixed_lookup_tables,
        unmixed_transformations and unmixed_gating: the gating the main
        window applies to the displayed sample, whose events these are.

        The gate and its ancestors are evaluated with apply_gates_in_place,
        so child gates inherit their parent's membership and quadrants are
        resolved through their quadrant gate.

        Returns None if 'root' is selected or if gating is unavailable.
        Returns a boolean ndarray of length len(event_data) otherwise.
        """
        if self._source_gate in (None, 'root'):
            return None

        gating = self.controller.unmixed_gating
        lookup_tables = self.controller.unmixed_lookup_tables
        transforms = self.controller.unmixed_transformations
        pnn = self.controller.experiment.settings['unmixed'].get('event_channels_pnn', [])

        if gating is None or not lookup_tables or not transforms or not pnn:
            return None

        try:
            paths = gating.find_matching_gate_paths(self._source_gate)
        except Exception:
            return None
        if not paths:
            return None
        gate_sequence = [g for g in paths[0] if g != 'root'] + [self._source_gate]

        data = {'event_data': event_data, 'pnn': list(pnn), 'transformations': transforms,
                'lookup_tables': lookup_tables, 'gating': gating,
                'gate_membership': {'root': np.ones(len(event_data), dtype=bool)}}
        try:
            apply_gates_in_place(data, gates_to_calculate=gate_sequence)
        except Exception as e:
            logger.error(
                f'AfComparisonPlotWidget._compute_gate_mask: '
                f'error computing mask for "{self._source_gate}": {e}'
            )
            return None
        mask = data['gate_membership'].get(self._source_gate)
        if mask is None:
            logger.warning(
                f'AfComparisonPlotWidget._compute_gate_mask: '
                f'"{self._source_gate}" could not be evaluated on these events'
            )
            return None
        return np.asarray(mask, dtype=bool)

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

class AutofluorescencePage(QWidget):
    """AutoSpectral tab page: AF extraction, stored profiles, assignment and comparison."""

    def __init__(self, bus=None, controller=None, active_fn=None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.controller = controller
        # active_fn(page) -> bool: whether the page is on screen.
        self._active_fn = active_fn
        # Plotted traces of the selected profile, one per spectrum row.
        self._profile_curves: list = []

        self._train_thread = None
        self._train_worker = None
        self._train_token = None
        self._auto_thread = None
        self._auto_worker = None
        self._auto_running = False
        self._auto_token = None
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

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setWidget(self._content)
        outer.addWidget(self._scroll)

        content_layout.addWidget(HelpToggleWidget(text=autospectral_af_help_text))
        self._build_extraction_section(content_layout)
        self._build_profile_manager_section(content_layout)
        self._build_assignment_section(content_layout)
        self._build_comparison_section(content_layout)
        content_layout.addStretch()

        if self.bus:
            self.bus.modeChangeRequested.connect(self._on_mode_change)
            self.bus.loadSampleRequested.connect(self._on_sample_loaded)
            self.bus.spectralProcessRefreshed.connect(self._on_process_refreshed)
            self.bus.sampleTreeUpdated.connect(self._populate_sample_combo)

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

        self._som_dim_spin = QSpinBox()
        self._som_dim_spin.setRange(2, 30)
        self._som_dim_spin.setValue(10)
        self._som_dim_spin.setToolTip(
            'Side length of the square SOM grid (som.dim in R AutoSpectral); '
            'the grid has som.dim² nodes. KMeans with the same number of '
            'clusters is used when the compiled SOM kernel is unavailable.'
        )
        self._som_dim_spin.installEventFilter(WheelBlocker(self._som_dim_spin))
        self._som_dim_spin.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        layout.addRow('SOM grid size:', self._som_dim_spin)

        self._extract_btn = QPushButton('Extract AF Profile')
        self._extract_btn.clicked.connect(self._run_extraction)
        layout.addRow(self._extract_btn)
        guard_while_busy(self.controller, self._extract_btn)

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
        # Right-click on a trace offers to delete that spectrum.
        self._profile_plot.getPlotItem().setMenuEnabled(False)
        self._profile_plot.scene().sigMouseClicked.connect(self._on_profile_plot_clicked)
        self._profile_plot.setToolTip('Right-click a spectrum to delete it from the profile.')
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
        # profiles and their assignment decide the unmixing: not changed under a running job
        guard_while_busy(self.controller, load_btn, self._delete_btn)

        layout.addLayout(btn_row)
        parent_layout.addWidget(grp)

    def _build_assignment_section(self, parent_layout):
        grp = QGroupBox('Step 2 — Assign AF Profiles to Samples')
        layout = QVBoxLayout(grp)

        info = QLabel(
            'Tick the AF profile(s) for each sample. '
            'Multiple profiles are concatenated before unmixing. '
            'Unticked samples use standard OLS unmixing. '
            'Unstained controls are listed first and are assigned the AF '
            'profile extracted from their own data by default. '
            'Auto-assign gives each unassigned sample the profiles of the unstained '
            'controls whose forward and side scatter it most resembles.'
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        filter_row = QHBoxLayout()
        self._include_unstained_cb = QCheckBox('Include unstained controls')
        self._include_unstained_cb.setChecked(True)
        self._include_unstained_cb.setToolTip(
            'Show unstained controls (listed at the top of the grid).'
        )
        filter_row.addWidget(self._include_unstained_cb)

        self._include_ssc_cb = QCheckBox('Include single-stained controls')
        self._include_ssc_cb.setChecked(False)
        self._include_ssc_cb.setToolTip(
            'Show single-stained controls (listed after the samples). '
            'Also includes unstained controls.'
        )
        filter_row.addWidget(self._include_ssc_cb)
        filter_row.addStretch(1)
        layout.addLayout(filter_row)
        self._include_unstained_cb.toggled.connect(self._on_assignment_filter_changed)
        self._include_ssc_cb.toggled.connect(self._on_assignment_filter_changed)

        # Rows = samples; column 0 = sample name, then one tick-box column per
        # profile. One header, so labels stay over their columns; dividers
        # are draggable and names elide in the middle when a column is narrow.
        self._assign_table = QTableWidget()
        self._assign_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._assign_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._assign_table.setFocusPolicy(Qt.NoFocus)
        self._assign_table.setWordWrap(False)
        self._assign_table.setTextElideMode(Qt.ElideMiddle)
        self._assign_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self._assign_table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        rows = self._assign_table.verticalHeader()
        rows.setVisible(False)
        rows.setSectionResizeMode(QHeaderView.Fixed)
        rows.setDefaultSectionSize(
            max(QCheckBox().sizeHint().height(), self._assign_table.fontMetrics().height()) + 6
        )
        header = self._assign_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setTextElideMode(Qt.ElideMiddle)
        header.setMinimumSectionSize(ASSIGN_MIN_COL_WIDTH)
        header.setStretchLastSection(False)
        header.setSectionsMovable(False)
        header.sectionResized.connect(self._on_assign_column_resized)
        layout.addWidget(self._assign_table)

        self._assign_empty_label = QLabel('No profiles or samples available yet.')
        self._assign_empty_label.setVisible(False)
        layout.addWidget(self._assign_empty_label)

        # Widths the user dragged, by column key ('' = sample column, else
        # profile name), kept across rebuilds for the session.
        self._assign_col_widths: dict = {}
        self._assign_col_keys: list = []
        self._assign_rebuilding = False

        self._assign_checkboxes: dict = {}

        self._assign_status = QLabel('')
        layout.addWidget(self._assign_status)

        group_row = QHBoxLayout()
        group_row.addWidget(QLabel('Folder:'))
        self._folder_combo = QComboBox()
        self._folder_combo.setMinimumWidth(220)
        self._folder_combo.setToolTip(
            'Samples shown in the grid that are in this folder or its subfolders.'
        )
        group_row.addWidget(self._folder_combo)
        self._assign_folder_btn = QPushButton('Assign Selected Profile to Folder')
        self._assign_folder_btn.setToolTip(
            'Assign the profile currently selected in the Stored AF Profiles list '
            'to every sample shown in the grid that is in the chosen folder. '
            'Already-assigned profiles are not duplicated.'
        )
        self._assign_folder_btn.clicked.connect(self._assign_selected_profile_to_folder)
        group_row.addWidget(self._assign_folder_btn)
        group_row.addStretch(1)
        self._auto_assign_btn = QPushButton('Auto-assign from Scatter')
        self._auto_assign_btn.setToolTip(
            'For every sample shown in the grid with no AF profile (bead samples '
            'excepted), choose the profile(s) of the unstained control(s) whose '
            'FSC/SSC distribution best explains its own: Gaussian mixture models '
            'of each unstained sample, combined by BIC. Existing assignments are '
            'not changed. Review the result before unmixing.'
        )
        self._auto_assign_btn.clicked.connect(self._run_auto_assign)
        group_row.addWidget(self._auto_assign_btn)
        layout.addLayout(group_row)

        btn_row = QHBoxLayout()

        assign_all_btn = QPushButton('Assign Selected Profile to All Samples')
        assign_all_btn.setToolTip(
            'Assign the profile currently selected in the Stored AF Profiles list '
            'to every sample shown in the grid. Already-assigned profiles are not duplicated.'
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
        guard_while_busy(self.controller, self._assign_table, self._folder_combo, self._assign_folder_btn,
                         self._auto_assign_btn, assign_all_btn, clear_sample_btn, clear_all_btn)

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
            'OLS (no AF)', self.controller, parent=self, has_af_channels=False
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

    def is_active(self) -> bool:
        """True while the page is on screen."""
        if self._active_fn is None:
            return self.controller.current_mode == TAB_NAME
        return self._active_fn(self)

    def on_activated(self):
        """The page came on screen: refresh it and bring the comparison up to date."""
        self._refresh_ui()
        if self.controller.raw_event_data is not None:
            QTimer.singleShot(0, self._run_comparison)

    def _on_mode_change(self, mode):
        if mode == TAB_NAME and self.is_active():
            self.on_activated()

    def _on_sample_loaded(self, _sample_path):
        # Rebuild grid and controls immediately, then defer the comparison run
        # with a zero-delay timer so that load_sample() has fully completed
        # (raw_event_data, transfer_matrix etc. are all set) before we read them.
        if self.controller.current_mode != TAB_NAME:
            return
        if not self.is_active():
            self._clear_comparison_plots()
            return
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
        self._scroll.setVisible(has_unmixing)

        if not has_unmixing:
            return

        self._populate_sample_combo()
        self._refresh_profile_list()
        self._assign_unstained_own_profiles()
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
            if not is_bead_sample(samples, p)
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

        token = acquire_job(self.controller, 'AutoSpectral: AF extraction')
        if token is None:
            self._extract_status.setText(self.controller.job_gate.busy_text())
            return
        self._train_token = token
        try:
            full_path = str(self.controller.experiment_dir / sample_path)
            sample = sample_from_fcs(full_path, self.bus)
            raw_event_data = sample.get_events(source='raw')
        except Exception as e:
            token.release()
            self._extract_status.setText(f'Failed to load sample: {e}')
            return

        if raw_event_data is None or raw_event_data.shape[0] == 0:
            token.release()
            self._extract_status.setText('Sample has no events.')
            return

        fl_ids = np.array(
            self.controller.filtered_raw_fluorescence_channel_ids
        )
        raw_fl = raw_event_data[:, fl_ids]

        fluor_spectra = self._build_fluor_spectra()
        if fluor_spectra is None:
            token.release()
            self._extract_status.setText(
                'Fluorophore profiles not found — ensure the spectral model is complete.'
            )
            return

        fluor_names = self.controller._build_fluor_names()
        spread_job = None
        if fluor_names is not None and not spread_is_current(
                self.controller.experiment.process.get('af_spread'), fluor_names, fluor_spectra):
            snapshot = spread_snapshot(self.controller, fluor_names, fluor_spectra, sample_path)

            def spread_job(progress):
                return compute_spread_entry(snapshot, progress)

        self._extract_btn.setEnabled(False)
        self._extract_status.setText('Extracting AF spectra...')

        self._train_thread = QThread()
        self._train_worker = AfTrainingWorker(
            raw_fl, fluor_spectra, self._som_dim_spin.value(), sample_path,
            fluor_names=fluor_names, spread_job=spread_job,
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

    def _on_extraction_finished(self, result: dict):
        self._train_token.release()
        # Read source_fcs_path from the worker object (set before thread started)
        source_fcs_path = self._train_worker.source_fcs_path
        self._extract_btn.setEnabled(True)
        af_spectra = result['af_spectra']
        n_af = af_spectra.shape[0]

        pnn_raw = self.controller.experiment.settings['raw']['event_channels_pnn']
        fl_ids = self.controller.filtered_raw_fluorescence_channel_ids
        channel_names = [pnn_raw[i] for i in fl_ids]

        csv_save_error = None
        try:
            profile_name = save_af_profile_csv(
                af_spectra, channel_names, source_fcs_path,
                self.controller.experiment_dir,
            )
        except Exception as e:
            logger.error(f'AutoSpectral: failed to save CSV: {e}')
            csv_save_error = str(e)
            profile_name = Path(source_fcs_path).stem + ' AutoSpectral AF'

        af_profiles = self.controller.experiment.process.get('af_profiles', {})
        af_profiles[profile_name] = {
            'spectra': af_spectra.tolist(),
            'source_fcs': str(source_fcs_path),
            'channel_names': channel_names,
        }
        envelope = result['envelope']
        if envelope is not None:
            envelope['unstained_file'] = envelope_sample_name(profile_name)
            try:
                save_envelope_sample(result['envelope_sample'], self.controller.experiment_dir,
                                     envelope['unstained_file'])
            except OSError as e:
                logger.warning(f'AutoSpectral: failed to save the unstained subsample: {e}')
            af_profiles[profile_name]['envelope'] = envelope
        if result['spread'] is not None:
            self.controller.experiment.process['af_spread'] = result['spread']
        self.controller.experiment.process['af_profiles'] = af_profiles
        self.bus.autoSaveRequested.emit()

        # Cache precomputed matrices for the new profile immediately; a new
        # spread applies to every profile with an envelope.
        if result['spread'] is not None:
            self.controller.cache_all_af_profiles()
        else:
            self.controller.cache_af_profile(profile_name)
        self.controller.on_af_profiles_changed()

        self._assign_profile_to_sample(str(source_fcs_path), profile_name)

        if csv_save_error:
            self._extract_status.setText(
                f'Profile "{profile_name}" stored in-memory, but saving its CSV '
                f'failed: {csv_save_error}'
            )
        else:
            spread_note = (f' Spillover spread estimated from {result["spread"]["n_estimated"]} controls.'
                           if result['spread'] is not None else '')
            self._extract_status.setText(
                f'Done. Profile "{profile_name}" stored ({n_af} AF spectra).{spread_note}'
            )
        self._refresh_profile_list(select_name=profile_name)
        self._rebuild_assignment_grid()

        if self.bus:
            self.bus.statusMessage.emit(
                f'AutoSpectral: "{profile_name}" stored ({n_af} spectra).'
            )

    def _on_extraction_error(self, msg: str):
        self._train_token.release()
        self._extract_btn.setEnabled(True)
        self._extract_status.setText(f'Error: {msg}')
        logger.error(f'AutoSpectral extraction error: {msg}')

    # ======================================================================
    # Section 2 — profile list and spectral plot
    # ======================================================================

    def _refresh_profile_list(self, select_name: str | None = None):
        """Rebuild profile list, restore or default selection, and draw the plot.

        select_name, when given, takes priority over whatever was previously
        selected. Callers that just (re-)created or loaded a specific profile
        (extraction, CSV load) pass it here so that profile is the one shown —
        otherwise, if a *different* profile happened to be selected in the
        list already, the "restore previous selection" behaviour below would
        silently keep showing that unrelated profile and the just-updated
        spectra would never appear on screen.
        """
        current_name = select_name or (
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

        self._profile_curves = []
        for i, row in enumerate(af_spectra):
            pen = (pg.mkPen(color=(0, 0, 0), width=2) if i == 0
                else pg.mkPen(color=colours[i], width=1))
            self._profile_curves.append(self._profile_plot.plot(x, np.maximum(row, 0), pen=pen))

    def _profile_row_at(self, scene_pos) -> int | None:
        """Spectrum row of the selected profile drawn nearest scene_pos, if
        within PROFILE_PICK_TOLERANCE_PX (distance to the trace's segments)."""
        item = self._profile_list.currentItem()
        if item is None:
            return None
        entry = self.controller.experiment.process.get('af_profiles', {}).get(item.text())
        if entry is None:
            return None
        vb = self._profile_plot.getViewBox()
        if not vb.sceneBoundingRect().contains(scene_pos):
            return None
        point = vb.mapSceneToView(scene_pos)
        px_w, px_h = vb.viewPixelSize()
        if px_w <= 0 or px_h <= 0:
            return None
        spectra = np.maximum(np.asarray(entry['spectra'], dtype=float), 0)
        n_ch = spectra.shape[1]
        if n_ch < 2:
            return None
        # Trace vertices relative to the click, in pixels.
        sx = (np.arange(n_ch) - point.x()) / px_w
        sy = (spectra - point.y()) / px_h
        ax, dx = sx[:-1], np.diff(sx)
        ay, dy = sy[:, :-1], np.diff(sy, axis=1)
        t = np.clip(-(ax * dx + ay * dy) / np.maximum(dx * dx + dy * dy, 1e-12), 0.0, 1.0)
        distance = np.hypot(ax + t * dx, ay + t * dy).min(axis=1)
        row = int(np.argmin(distance))
        return row if distance[row] <= PROFILE_PICK_TOLERANCE_PX else None

    def _on_profile_plot_clicked(self, event):
        """Right-click on a trace: menu to delete that spectrum (not the mean)."""
        if event.button() != Qt.MouseButton.RightButton:
            return
        item = self._profile_list.currentItem()
        if item is None:
            return
        row = self._profile_row_at(event.scenePos())
        if row is None:
            return
        event.accept()
        gate = getattr(self.controller, 'job_gate', None)
        if gate is not None and gate.is_held():
            self._extract_status.setText(f'{gate.busy_text()}: AF spectra cannot be deleted until it has finished.')
            return
        name = item.text()
        curve = self._profile_curves[row] if row < len(self._profile_curves) else None
        old_pen = curve.opts['pen'] if curve is not None else None
        if curve is not None:
            curve.setPen(pg.mkPen(color=(230, 60, 60), width=3))
        menu = QMenu(self)
        if row == 0:
            action = menu.addAction('The profile mean (row 0) cannot be deleted')
            action.setEnabled(False)
        else:
            action = menu.addAction(f'Delete spectrum {row}…')
        chosen = menu.exec(event.screenPos().toPoint())
        if curve is not None:
            curve.setPen(old_pen)
        if row > 0 and chosen is action:
            self._delete_profile_spectrum(name, row)

    def _delete_profile_spectrum(self, name: str, row: int):
        """Remove spectrum *row* (not the mean, row 0) from a stored profile, after confirmation."""
        af_profiles = self.controller.experiment.process.get('af_profiles', {})
        entry = af_profiles.get(name)
        if entry is None or not 0 < row < len(entry['spectra']):
            return
        af_index = af_index_lookup(af_profiles, [name])
        index_text = '' if af_index is None else f' (AF Index {int(af_index[row])})'
        answer = QMessageBox.question(
            self, 'Delete AF spectrum',
            f'Delete spectrum {row}{index_text} from "{name}"?\n\n'
            'The experiment\'s AF Index numbering changes, so gates drawn on AF Index '
            'may select different spectra afterwards. Re-extract the profile to restore it.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        spectra = list(entry['spectra'])
        del spectra[row]
        entry['spectra'] = spectra
        if self.bus:
            self.bus.autoSaveRequested.emit()
        self.controller.cache_af_profile(name)
        self.controller.on_af_profiles_changed()
        self._refresh_profile_list(select_name=name)
        if self.bus:
            self.bus.statusMessage.emit(
                f'AutoSpectral: deleted spectrum {row} from "{name}" ({len(spectra)} remain).'
            )

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
        self.controller.on_af_profiles_changed()

        self._refresh_profile_list(select_name=profile_name)
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

        self.controller.on_af_profiles_changed()
        self._refresh_profile_list()
        self._rebuild_assignment_grid()
        if self.bus:
            self.bus.statusMessage.emit(f'AutoSpectral: deleted profile "{name}".')

    # ======================================================================
    # Section 3 — assignment grid
    # ======================================================================

    def _is_unstained_sample(self, sample_path: str, display_name: str = '') -> bool:
        """True if the sample is tagged as unstained or its name says so."""
        samples = self.controller.experiment.samples
        if sample_path in samples.get('unstained_samples', []):
            return True
        name = display_name or samples.get('all_samples', {}).get(sample_path, '')
        return bool(
            re.search(r'unstained', name, re.IGNORECASE)
            or re.search(r'unstained', str(sample_path), re.IGNORECASE)
        )

    def _assignment_samples(self) -> dict:
        """
        Samples shown in the assignment grid, as {path: display name}.
        Order: unstained controls, then ordinary samples, then single-stained
        controls. Unstained controls are shown when either control checkbox is
        ticked; single-stained controls only when their own checkbox is ticked.
        """
        samples = self.controller.experiment.samples
        all_samples = samples.get('all_samples', {})
        ssc = set(samples.get('single_stain_controls', []))
        include_ssc = self._include_ssc_cb.isChecked()
        include_unstained = include_ssc or self._include_unstained_cb.isChecked()

        unstained_rows = {}
        sample_rows = {}
        ssc_rows = {}
        for path, name in all_samples.items():
            if self._is_unstained_sample(path, name):
                if include_unstained:
                    unstained_rows[path] = name
            elif path in ssc:
                if include_ssc:
                    ssc_rows[path] = name
            else:
                sample_rows[path] = name
        return {**unstained_rows, **sample_rows, **ssc_rows}

    def _on_assignment_filter_changed(self, _checked=False):
        """Ticking single-stained controls forces unstained controls on."""
        include_ssc = self._include_ssc_cb.isChecked()
        if include_ssc:
            self._include_unstained_cb.blockSignals(True)
            self._include_unstained_cb.setChecked(True)
            self._include_unstained_cb.blockSignals(False)
        self._include_unstained_cb.setEnabled(not include_ssc)
        self._rebuild_assignment_grid()

    def _profiles_from_sample(self, sample_path: str) -> list:
        """Names of stored AF profiles that were extracted from sample_path."""
        af_profiles = self.controller.experiment.process.get('af_profiles', {})
        return [
            name for name, entry in af_profiles.items()
            if str(entry.get('source_fcs', '')) == str(sample_path)
        ]

    def _reapply_unmixing_if_current(self, sample_paths):
        """Refresh AF matrices and unmixing when the loaded sample is in sample_paths."""
        if self.controller.current_sample_path not in sample_paths:
            return
        self.controller.initialise_af_matrices()
        if self.controller.raw_event_data is not None:
            self.controller.unmixed_event_data = self.controller._apply_unmixing(
                self.controller.raw_event_data
            )
            self.controller.clear_data_for_cytometry_plots()
            self.controller.initialise_data_for_cytometry_plots()

    def _assign_profile_to_sample(self, sample_path: str, profile_name: str):
        """Add profile_name to the sample's assignment if not already present."""
        sample_af = self.controller.experiment.samples.get('sample_af_profiles', {})
        assigned = list(sample_af.get(sample_path, []))
        if profile_name not in assigned:
            assigned.append(profile_name)
        sample_af[sample_path] = assigned
        self.controller.experiment.samples['sample_af_profiles'] = sample_af
        self.bus.autoSaveRequested.emit()
        self._reapply_unmixing_if_current([sample_path])

    def _assign_unstained_own_profiles(self):
        """
        Give each unstained control the AF profile(s) extracted from its own
        data. Only samples with no assignment record are touched, so an
        assignment the user has edited or cleared is never overwritten.
        """
        samples = self.controller.experiment.samples
        sample_af = samples.get('sample_af_profiles', {})
        changed = []
        for path, name in samples.get('all_samples', {}).items():
            if path in sample_af or not self._is_unstained_sample(path, name):
                continue
            own = self._profiles_from_sample(path)
            if own:
                sample_af[path] = own
                changed.append(path)
        if not changed:
            return
        samples['sample_af_profiles'] = sample_af
        self.bus.autoSaveRequested.emit()
        self._reapply_unmixing_if_current(changed)

    def _rebuild_assignment_grid(self):
        """
        Rebuild the samples × profiles table.
        Rows are chosen by _assignment_samples(): unstained controls first,
        then samples, then single-stained controls if requested. Column 0 is
        the sample name, then one tick-box column per profile. Column widths
        default to the text (capped) unless the user has dragged them.
        """
        self._assign_checkboxes.clear()
        table = self._assign_table
        self._assign_rebuilding = True
        table.clear()
        table.setRowCount(0)
        table.setColumnCount(0)

        af_profiles = self.controller.experiment.process.get('af_profiles', {})
        profile_names = list(af_profiles.keys())
        shown_samples = self._assignment_samples()
        sample_af = self.controller.experiment.samples.get('sample_af_profiles', {})
        self._refresh_folder_combo(shown_samples)

        if not profile_names or not shown_samples:
            self._assign_rebuilding = False
            table.setVisible(False)
            self._assign_empty_label.setVisible(True)
            return
        self._assign_empty_label.setVisible(False)
        table.setVisible(True)

        table.setColumnCount(len(profile_names) + 1)
        table.setRowCount(len(shown_samples))
        bold = table.font()
        bold.setBold(True)
        header_fm = QFontMetrics(bold)
        body_fm = QFontMetrics(table.font())

        corner = QTableWidgetItem('Sample')
        corner.setFont(bold)
        corner.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        table.setHorizontalHeaderItem(0, corner)
        defaults = [header_fm.horizontalAdvance('Sample')]
        for col, pname in enumerate(profile_names, start=1):
            display = pname[:-len(AF_PROFILE_SUFFIX)] if pname.endswith(AF_PROFILE_SUFFIX) else pname
            item = QTableWidgetItem(display)
            item.setFont(bold)
            item.setToolTip(pname)
            table.setHorizontalHeaderItem(col, item)
            defaults.append(min(header_fm.horizontalAdvance(display) + ASSIGN_COL_PADDING,
                                ASSIGN_MAX_PROFILE_WIDTH))

        for row, (sample_path, display_name) in enumerate(shown_samples.items()):
            name_item = QTableWidgetItem(display_name)
            name_item.setToolTip(f'{display_name}\n{sample_path}')
            name_item.setFlags(Qt.ItemIsEnabled)
            if sample_path == self.controller.current_sample_path:
                name_item.setFont(bold)
                width = header_fm.horizontalAdvance(display_name)
            else:
                width = body_fm.horizontalAdvance(display_name)
            defaults[0] = max(defaults[0], width)
            table.setItem(row, 0, name_item)

            assigned = sample_af.get(sample_path, [])
            for col, pname in enumerate(profile_names, start=1):
                cb = QCheckBox()
                cb.setChecked(pname in assigned)
                cb.setToolTip(f'Apply "{pname}" to {display_name}')
                cb.checkStateChanged.connect(
                    self._make_assignment_handler(sample_path, pname, cb)
                )
                cell = QWidget()
                cell_layout = QHBoxLayout(cell)
                cell_layout.setContentsMargins(0, 0, 0, 0)
                cell_layout.setAlignment(Qt.AlignCenter)
                cell_layout.addWidget(cb)
                table.setCellWidget(row, col, cell)
                self._assign_checkboxes[(sample_path, pname)] = cb
        defaults[0] = min(defaults[0] + ASSIGN_COL_PADDING, ASSIGN_MAX_NAME_WIDTH)

        self._assign_col_keys = [''] + profile_names
        for col, key in enumerate(self._assign_col_keys):
            width = self._assign_col_widths.get(key, defaults[col])
            table.setColumnWidth(col, max(width, ASSIGN_MIN_COL_WIDTH))
        self._assign_rebuilding = False

        # Fit every row up to the cap; the tab's outer scroll area handles
        # page-level scrolling, the table's own scrollbar takes over beyond it.
        content_height = (table.horizontalHeader().sizeHint().height()
                          + table.verticalHeader().length()
                          + 2 * table.frameWidth()
                          + table.horizontalScrollBar().sizeHint().height())
        table.setFixedHeight(max(ASSIGN_MIN_HEIGHT, min(content_height, ASSIGN_MAX_HEIGHT)))

    def _on_assign_column_resized(self, index: int, _old: int, new: int):
        """Remember a width the user dragged, so rebuilds keep it."""
        if self._assign_rebuilding or index >= len(self._assign_col_keys):
            return
        self._assign_col_widths[self._assign_col_keys[index]] = new

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

    def _refresh_folder_combo(self, shown_samples):
        """List the folders holding samples shown in the grid, keeping the selection."""
        current = self._folder_combo.currentData()
        groups = folder_groups(shown_samples)
        self._folder_combo.blockSignals(True)
        self._folder_combo.clear()
        for folder, paths in groups.items():
            self._folder_combo.addItem(f'{folder} ({len(paths)})', userData=folder)
        if not groups:
            self._folder_combo.addItem('No folders', userData=None)
        index = self._folder_combo.findData(current) if current is not None else -1
        self._folder_combo.setCurrentIndex(max(index, 0))
        self._folder_combo.blockSignals(False)
        self._folder_combo.setEnabled(bool(groups))
        self._assign_folder_btn.setEnabled(bool(groups))

    def _assign_selected_profile(self, target_samples: list, where: str = ''):
        """Assign the profile selected in the profile list to target_samples."""
        item = self._profile_list.currentItem()
        if item is None:
            self._assign_status.setText(
                'No profile selected — pick one from the Stored AF Profiles list first.'
            )
            return

        profile_name = item.text()

        if not target_samples:
            self._assign_status.setText('No samples available.')
            return

        sample_af = self.controller.experiment.samples.get('sample_af_profiles', {})
        for sample_path in target_samples:
            assigned = list(sample_af.get(sample_path, []))
            if profile_name not in assigned:
                assigned.append(profile_name)
            sample_af[sample_path] = assigned

        self.controller.experiment.samples['sample_af_profiles'] = sample_af
        self.bus.autoSaveRequested.emit()

        # If the current sample was affected, reinitialise its AF matrices and
        # re-apply unmixing so the Unmixed Data tab updates immediately.
        self._reapply_unmixing_if_current(target_samples)

        self._rebuild_assignment_grid()
        message = f'"{profile_name}" assigned to {len(target_samples)} sample(s){where}.'
        self._assign_status.setText(message)
        if self.bus:
            self.bus.statusMessage.emit(f'AutoSpectral: {message}')

    def _assign_selected_profile_to_all(self):
        """Assign the profile selected in the profile list to every sample shown in the grid."""
        self._assign_selected_profile(list(self._assignment_samples()))

    def _assign_selected_profile_to_folder(self):
        """Assign the selected profile to the shown samples in the chosen folder."""
        folder = self._folder_combo.currentData()
        if folder is None:
            self._assign_status.setText('No folder selected.')
            return
        targets = folder_groups(self._assignment_samples()).get(folder, [])
        self._assign_selected_profile(targets, where=f' in {folder}')

    def _run_auto_assign(self):
        """Suggest profiles for the unassigned samples shown in the grid, from scatter."""
        if self._auto_running:
            return
        experiment = self.controller.experiment
        samples = experiment.samples
        sample_af = samples.get('sample_af_profiles', {})
        all_samples = samples.get('all_samples', {})
        targets = [p for p, name in self._assignment_samples().items()
                   if not sample_af.get(p) and not is_bead_sample(samples, p, name)]
        if not targets:
            self._assign_status.setText(
                'Auto-assign: every sample shown in the grid already has an AF profile '
                '(bead samples are not auto-assigned).'
            )
            return
        beads = [p for p in all_samples if is_bead_sample(samples, p)]
        references = reference_sources(experiment.process.get('af_profiles', {}),
                                       all_samples, exclude=beads)
        raw_settings = experiment.settings['raw']
        pair = scatter_channel_pair(raw_settings, experiment.cytometry.get('raw_plots'))
        from PySide6.QtWidgets import QMessageBox
        if not references or pair is None:
            QMessageBox.information(
                self, 'Auto-assign',
                'No AF profile was extracted from an unstained cell sample of this experiment.'
                if not references else
                'The forward and side scatter channels could not be identified.'
            )
            return
        token = acquire_job(self.controller, 'AutoSpectral: AF auto-assign')
        if token is None:
            self._assign_status.setText(self.controller.job_gate.busy_text())
            return
        self._auto_token = token

        self._auto_running = True
        self._auto_assign_btn.setEnabled(False)
        self._assign_status.setText(f'Auto-assign: scoring {len(targets)} sample(s)…')
        self._auto_thread = QThread()
        self._auto_worker = AfAutoAssignWorker(
            self.controller.experiment_dir, references, targets, pair,
            scatter_ceilings(raw_settings, pair),
        )
        self._auto_worker.moveToThread(self._auto_thread)
        self._auto_thread.started.connect(self._auto_worker.run)
        self._auto_worker.finished.connect(self._on_auto_assign_finished)
        self._auto_worker.error.connect(self._on_auto_assign_error)
        self._auto_worker.progress.connect(self._assign_status.setText)
        self._auto_worker.finished.connect(self._auto_thread.quit)
        self._auto_worker.error.connect(self._auto_thread.quit)
        self._auto_thread.finished.connect(self._auto_thread.deleteLater)
        self._auto_thread.start()

    def _on_auto_assign_error(self, msg: str):
        self._auto_token.release()
        self._auto_running = False
        self._auto_assign_btn.setEnabled(True)
        self._assign_status.setText(f'Auto-assign failed: {msg}')

    def _on_auto_assign_finished(self, result: dict):
        """Apply the suggestions to samples still unassigned, then ask for a review."""
        from PySide6.QtWidgets import QMessageBox

        self._auto_token.release()
        self._auto_running = False
        self._auto_assign_btn.setEnabled(True)
        samples = self.controller.experiment.samples
        all_samples = samples.get('all_samples', {})
        af_profiles = self.controller.experiment.process.get('af_profiles', {})
        sample_af = samples.get('sample_af_profiles', {})

        changed = []
        for path, names in result['assignments'].items():
            names = [n for n in names if n in af_profiles]
            if path in all_samples and names and not sample_af.get(path):
                sample_af[path] = names
                changed.append(path)
        if changed:
            samples['sample_af_profiles'] = sample_af
            self.bus.autoSaveRequested.emit()
            self._reapply_unmixing_if_current(changed)
        self._rebuild_assignment_grid()

        def label(path):
            return all_samples.get(path) or Path(path).stem

        n_scored = len(result['assignments']) + len(result['skipped'])
        counts = {}
        for path in changed:
            key = ' + '.join(sample_af[path])
            counts[key] = counts.get(key, 0) + 1
        lines = [f'AF profiles were assigned to {len(changed)} of {n_scored} unassigned '
                 'sample(s) from their forward and side scatter.']
        lines += [f'  • {key}: {n} sample(s)' for key, n in sorted(counts.items(), key=lambda kv: -kv[1])]
        if result['skipped']:
            lines.append(f'{len(result["skipped"])} sample(s) were left unassigned: '
                         + ', '.join(label(p) for p in result['skipped']) + '.')
        if result['unused_references']:
            lines.append('Unstained samples not used: '
                         + ', '.join(label(p) for p in result['unused_references']) + '.')
        poor = sorted(((d['outside'], p) for p, d in result['details'].items()
                       if p in changed and d['outside'] >= POOR_FIT_SHARE), reverse=True)
        if poor:
            lines.append(f'{len(poor)} sample(s) have at least {POOR_FIT_SHARE:.0%} of events '
                         'outside the scatter of their unstained controls (check these first): '
                         + ', '.join(f'{label(p)} ({f:.0%})' for f, p in poor) + '.')
        lines.append('')
        lines.append('Please review the assignments in the grid. A sample is matched to the '
                     'unstained controls it most resembles in scatter, which does not guarantee '
                     'that its autofluorescence matches.')

        details = []
        for path in changed:
            d = result['details'][path]
            refs = ', '.join(f'{label(r)} {w:.0%}' for r, w in zip(d['references'], d['weights']))
            details.append(f'{label(path)}: {" + ".join(sample_af[path])} [{refs}; '
                           f'ΔBIC {d["delta_bic"]:.0f}; {d["outside"]:.1%} of events outside; '
                           f'{d["n_events"]:,} events]')
        for path, reason in result['skipped'].items():
            details.append(f'{label(path)}: not assigned ({reason})')
        for path, reason in result['unused_references'].items():
            details.append(f'{label(path)} (unstained): not used ({reason})')

        self._assign_status.setText(
            f'Auto-assign: {len(changed)} sample(s) assigned — please review the grid.'
        )
        if self.bus:
            self.bus.statusMessage.emit(f'AutoSpectral: auto-assigned {len(changed)} sample(s).')
        box = QMessageBox(QMessageBox.Icon.Information, 'Auto-assign complete', '\n'.join(lines),
                          QMessageBox.StandardButton.Ok, self)
        if details:
            box.setDetailedText('\n'.join(details))
        box.exec()

    def _clear_all_af(self):
        all_samples = self.controller.experiment.samples.get('all_samples', {})
        self.controller.experiment.samples['sample_af_profiles'] = {p: [] for p in all_samples}
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
        if wait_if_busy(self.controller, ('AutoSpectral comparison', id(self)),
                        lambda: self.is_active() and self._run_comparison(), self._cmp_status.setText):
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
            af_index_map = self.controller.get_af_index_map_for_sample(
                self.controller.current_sample_path
            )
        else:
            af_profiles = self.controller.experiment.process.get('af_profiles', {})
            entry = af_profiles.get(profile_key)
            af_spectra = np.array(entry['spectra']) if entry else None
            af_index_map = af_index_lookup(af_profiles, [profile_key])

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

        # The AF matrices the main window unmixes with, negative envelope
        # included: the sample's combined assignment, or the chosen profile
        # (computed and cached now if it is not cached yet).
        if profile_key == '__assigned__':
            assigned = (self.controller.experiment.samples
                        .get('sample_af_profiles', {})
                        .get(self.controller.current_sample_path, []))
            af = resolve_af_for_profiles(self.controller, assigned)
            af_precomputed = None if af is None else af.precomputed
            if af is not None:
                af_spectra, af_index_map = af.spectra, af.index_map
        else:
            af_precomputed = self.controller.precomputed_for_profile(profile_key)
        if af_precomputed is None:
            self._cmp_status.setText(
                'AF matrices are not available yet for this profile. '
                'Wait for the spectral process to finish, then run the comparison again.'
            )
            return

        self._plot_ols.set_status('Computing...')
        self._plot_af.set_status('Computing...')
        self._cmp_status.setText('Running unmixing on current sample...')

        # Build a state key describing exactly what this run will compute.
        # If it matches the last completed run, the cached arrays are still
        # valid — skip the worker and just redraw.
        spillover = self.controller.experiment.process.get('spillover')
        spillover_key = tuple(np.array(spillover).ravel()) if spillover is not None else None
        # Use a content hash of the AF spectra actually being applied, not P:
        # P = solve(fluor_spectra @ fluor_spectra.T, fluor_spectra) depends only
        # on the fluorophore library, not on which AF profile(s) are assigned
        # (see combine_af_precomputed's docstring — P is shared across profiles),
        # so keying on P made this cache blind to AF-assignment changes while
        # staying on "Assigned to sample": ticking a different profile changed
        # af_spectra but never af_key, so a stale cached result was reused and
        # "No change — redrawing..." was reported incorrectly.
        af_key = bytes(np.ascontiguousarray(af_spectra).data) if af_spectra is not None else None
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
            af_index_map=af_index_map,
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

        # Better default: the two channels most affected by AF correction
        # (largest mean |OLS - AF| difference).
        try:
            picked = pick_most_affected_channels(
                self._ols_data, self._af_data, pnn, fl_names
            )
            if picked is not None:
                ch_x, ch_y = picked
        except Exception as e:
            logger.debug(f'AutoSpectral comparison: channel auto-pick failed: {e}')

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


# ---------------------------------------------------------------------------
# The AutoSpectral tab
# ---------------------------------------------------------------------------

class AutoSpectralTab(QWidget):
    """
    The AutoSpectral tab: Autofluorescence, Noise, Spectral Variants and
    Spectral Correction pages beside a side navigation list. A page with
    ``on_activated`` is told when it comes on screen; ``page_is_active``
    tells a page whether it is on screen.
    """

    def __init__(self, bus=None, controller=None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.controller = controller

        from honeychrome.view_components.autospectral_correction_page import SpectralCorrectionPage
        from honeychrome.view_components.autospectral_noise_page import NoisePage
        from honeychrome.view_components.autospectral_variants_page import SpectralVariantsPage

        self._pages = SideTabWidget()
        self.af_page = AutofluorescencePage(bus, controller, active_fn=self.page_is_active, parent=self)
        self.noise_page = NoisePage(bus, controller, active_fn=self.page_is_active, parent=self)
        self.variants_page = SpectralVariantsPage(bus, controller, active_fn=self.page_is_active,
                                                  parent=self)
        self.correction_page = SpectralCorrectionPage(bus, controller, active_fn=self.page_is_active, parent=self)
        for page, label, icon_name in (
                (self.af_page, 'Autofluorescence', 'sparkles'),
                (self.noise_page, 'Noise', 'wave-sine'),
                (self.variants_page, 'Spectral Variants', 'shape'),
                (self.correction_page, 'Spectral Correction', 'adjustments')):
            self._pages.addTab(page, label, icon_name)

        self._qsettings = QSettings('honeychrome', 'app_configuration')
        saved = self._qsettings.value(PAGE_SETTING, 0, type=int)
        if 0 <= saved < self._pages.count():
            self._pages.setCurrentIndex(saved)
        self._pages.currentChanged.connect(self._on_page_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._pages)

    def page_is_active(self, page) -> bool:
        """True when the AutoSpectral tab is shown with *page* selected."""
        return (self.controller is not None
                and self.controller.current_mode == TAB_NAME
                and self._pages.currentWidget() is page)

    def _on_page_changed(self, index: int):
        self._qsettings.setValue(PAGE_SETTING, index)
        page = self._pages.widget(index)
        if hasattr(page, 'on_activated') and self.page_is_active(page):
            page.on_activated()

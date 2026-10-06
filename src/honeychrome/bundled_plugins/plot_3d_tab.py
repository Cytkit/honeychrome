"""
plot_3d_tab.py
================

Module layout:
  1. Imports / guarded pyqtgraph.opengl import
  2. Constants
  3. Theme + transform/normalisation helpers
  4. Tick-label helpers
  5. Colour-mode builders (density + categorical-by-gate)
  6. AxisControlStrip, GateTitleBar (per-tile controls)
  7. Grid / density-lookup / gate-chain helpers
  8. Plot3DGLView (GLViewWidget subclass: orbit/pan + click-select + double-click)
  9. Plot3DPlotWidget (one tile: title bar + canvas + axis strips + interactivity)
  10. NewPlot3DTile (placeholder "add plot" tile)
  11. Plot3DToolbar
  12. Plot3DGridWidget (workspace: bin-packing layout, add/delete/move/pop-out)
  13. PluginWidget + plugin_name (top-level tab, lifecycle + signal wiring)
  14. Standalone demo harness (`python plot_3d_tab.py`)

IMPORTANT - a core-code landmine found while wiring up step 8, and how this
file works around it without touching core code:

  controller.set_mode()'s catch-all branch for plugin tabs sets
  `self.current_mode = tab_name` and `self.data_for_cytometry_plots = None`.
  That means *whenever this plugin's tab is the active tab* (i.e. exactly
  when the user is dragging an AxisControlStrip or using the right-click
  menu), `controller.data_for_cytometry_plots` is None. Both
  `controller.recalc_after_axis_transform()` (connected to
  `bus.axisTransformed`) and `controller.reset_axes_transforms()` (connected
  to `bus.axesReset`) unconditionally index into that dict - so emitting
  either signal while this tab is active would raise inside core code this
  plugin isn't allowed to touch.

  Fix used throughout this file: this tile mutates its own (shared)
  Transform objects directly for its own redraw - exactly like the 2D
  CytometryPlotWidget already does for drag-zoom/fit-axes - so this tab's
  own rendering never depends on the bus round-trip. The bus emission that
  exists purely to tell the 2D Unmixed Data tab "this channel changed, your
  cached histogram is stale" is *queued* (Plot3DGridWidget.dirty_axis_
  transformed / dirty_axes_reset) and only actually flushed once
  bus.modeChangeRequested reports the user has navigated to a real
  cytometry-data tab (see PluginWidget._handle_mode_changed) - at which
  point controller.data_for_cytometry_plots is guaranteed to be the correct,
  non-None dict again. "Reset Axes" goes one step further and reuses
  controller_components.functions.assign_default_transforms /
  generate_transformations directly (the same helpers
  controller.reset_axes_transforms() calls) so this tab gets the reset
  applied immediately rather than waiting for a tab switch.
"""

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import colorcet as cc
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)

logging.captureWarnings(True)

try:
    import pyqtgraph.opengl as gl
except ImportError as exc:
    raise ImportError(
        "plot_3d_tab.py requires pyqtgraph.opengl, which needs PyOpenGL. "
        "Install with: pip install PyOpenGL"
    ) from exc

from OpenGL import GL  # noqa: E402  (safe: pyqtgraph.opengl already required PyOpenGL above)

import honeychrome.settings as settings
from honeychrome.controller_components.transform import transforms_menu_items
from honeychrome.controller_components.functions import assign_default_transforms, generate_transformations
from honeychrome.view_components.icon_loader import icon


plugin_name = '3D Plots'

# tab names whose data_for_cytometry_plots dict is real (non-None) - see the
# core-code landmine note above. Any other tab name (including every other
# plugin tab) leaves controller.data_for_cytometry_plots as None.
DATA_TAB_NAMES = {'Raw Data', 'Spectral Process', 'Unmixed Data', 'Statistics'}

# subset of DATA_TAB_NAMES where controller.set_mode() points
# data_for_cytometry_plots at data_for_cytometry_plots_unmixed specifically
# (see set_mode() in controller.py) - calc_hists_and_stats() must only be
# flushed once one of these is the active tab, or it would recalculate the
# wrong dict (raw/process instead of unmixed).
UNMIXED_DATA_TAB_NAMES = {'Unmixed Data', 'Statistics'}


# --------------------------------------------------------------------------
# Constants (plugin-local)
# --------------------------------------------------------------------------
DEFAULT_MAX_POINTS = 100_000   # workspace-wide toolbar setting
N_DENSITY_BINS = 48            # lugin-local constant, not settings.hist_bins_retrieved
POINT_SIZE = 4.0
GRID_DEFAULT_SIZE = 20.0       # pyqtgraph GLGridItem native size
GRID_RESIZE_METHOD = 'scale'
DEFAULT_DISPLAY_RANGE = 10.0
TILE_WIDTH_TARGET = 420        # plugin-local equivalent of settings.cytometry_plot_width_target_retrieved
AXIS_NAME_OFFSET = 1.45        # axis-name distance from cube centre, in half-extents
TICK_FRONT_ALPHA = 0.12        # tick alpha fraction when in front of the data

# --------------------------------------------------------------------------
# Theme helpers - mirrors view.py's pg.setConfigOptions convention.
# is_dark itself is computed by PluginWidget (needs QApplication.instance()).
# --------------------------------------------------------------------------
def gl_background_color(is_dark):
    return 'black' if is_dark else 'white'


def gl_foreground_rgba(is_dark):
    return (235, 235, 235, 255) if is_dark else (20, 20, 20, 255)


def gl_tick_rgba(is_dark):
    return (150, 150, 150, 255) if is_dark else (110, 110, 110, 255)


# --------------------------------------------------------------------------
# Biexponential transform + display-space normalisation
# --------------------------------------------------------------------------
def to_display(raw_values, tr):
    """Raw event values -> display space, same convention used everywhere
    else in the app. 'default' transform (e.g. Time) has tr.xform is None,
    so raw values pass through unchanged."""
    return tr.xform.apply(raw_values) if tr.xform is not None else raw_values


def normalise_value(display_value, lo, hi, display_range=DEFAULT_DISPLAY_RANGE):
    """Map a display-space value into the GL scene's normalised cube, so
    mismatched axis magnitudes (e.g. a [0,1] logicle channel next to a
    Time channel up to settings.default_ceiling) frame sensibly together."""
    span = hi - lo
    if span <= 0:
        span = 1.0
    return (display_value - lo) / span * display_range - display_range / 2.0


# --------------------------------------------------------------------------
# Tick labels - GLTextItem with a 2D-legend fallback
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def gltext_supported():
    """One-time capability probe - confirmed working in the throwaway
    prototype against this project's pinned pyqtgraph + PySide6, but kept as
    a real probe (not assumed) for robustness against environment
    differences. Cached - the same process/install always gives the same
    answer, so re-probing it once per tile was wasted work."""
    try:
        item = gl.GLTextItem(pos=(0, 0, 0), text='test', color=(255, 255, 255, 255))
        del item
        return True
    except Exception as exc:
        logger.info(f"GLTextItem unavailable ({exc}); falling back to 2D legend.")
        return False


def ticks_for_axis(tr, display_range=DEFAULT_DISPLAY_RANGE, n_fallback_ticks=6):
    """
    Build (normalised_position, label) pairs for one axis from the channel's
    real Transform.ticks() - the same (value, label) pairs used by the 2D
    plots' AxisItem. tr.ticks() returns [minor_ticks, major_ticks]; the major
    list carries the real labels (some entries still have '' for unlabeled
    gridlines - those are skipped here, since an empty GLTextItem is wasted).

    'default'-transform channels (e.g. Time) have tr.ticks() is None, since
    Transform.default_ticks() relies on pg.AxisItem's own auto-ticking in the
    2D plots - the GL canvas has no AxisItem, so this falls back to evenly
    spaced raw-value ticks across tr.limits.
    """
    lo, hi = tr.limits
    raw_ticks = tr.ticks() if tr.ticks is not None else None

    pairs = None
    if raw_ticks:
        for level in reversed(raw_ticks):  # major (labelled) list last
            labelled = [(value, label) for value, label in level if label]
            if labelled:
                pairs = labelled
                break

    if not pairs:
        raw_vals = np.linspace(lo, hi, n_fallback_ticks)
        pairs = [(v, f"{v:,.0f}") for v in raw_vals]

    return [(normalise_value(value, lo, hi, display_range), label) for value, label in pairs]


class DepthFadedTextItem(gl.GLTextItem):
    """
    GLTextItem whose alpha falls as its anchor moves to the viewer's side of
    the cube centre. GLTextItem paints with QPainter after the GL pass, so
    it has no depth test and always draws over the data; fading by position
    along the view direction keeps tick text from competing with points
    that sit between the camera and the axis.
    """

    def __init__(self, display_range=DEFAULT_DISPLAY_RANGE, front_alpha=TICK_FRONT_ALPHA, **kwds):
        self._depth_scale = display_range / 2.0 * np.sqrt(3.0)  # corner distance from centre
        self._front_alpha = front_alpha
        super().__init__(**kwds)

    def _alpha_scale(self):
        cam = self.view().cameraPosition()
        to_cam = np.array([cam.x(), cam.y(), cam.z()])
        dist = np.linalg.norm(to_cam)
        if dist == 0.0:
            return 1.0
        depth = float(np.dot(np.asarray(self.pos, dtype=float), to_cam / dist)) / self._depth_scale
        t = float(np.clip((depth + 0.1) / 0.5, 0.0, 1.0))
        smooth = t * t * (3.0 - 2.0 * t)
        return 1.0 - (1.0 - self._front_alpha) * smooth

    def paint(self):
        if len(self.text) < 1:
            return
        self.setupGLState()

        color = QtGui.QColor(self.color)
        color.setAlphaF(color.alphaF() * self._alpha_scale())
        if color.alpha() < 3:
            return

        project = self.compute_projection()
        vec3 = QtGui.QVector3D(*self.pos)
        text_pos = self.align_text(project.map(vec3).toPointF())

        painter = QtGui.QPainter(self.view())
        painter.setPen(color)
        painter.setFont(self.font)
        painter.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing)
        painter.drawText(text_pos, self.text)
        painter.end()


def build_tick_items(ticks_per_axis, display_range, fg_color):
    """
    ticks_per_axis: list of 3 lists of (normalised_position, label) pairs,
    one per axis (x, y, z) - e.g. from ticks_for_axis(). One text item per
    tick, positioned at the tick's coordinate on its own axis and pinned to
    the cube floor on the other two. Each fades when it lies between the
    camera and the data (see DepthFadedTextItem). No glOptions override -
    GLTextItem's own 'additive' default is the only blend mode it actually
    renders with in this environment.
    """
    items = []
    half = display_range / 2.0
    for axis_i, ticks in enumerate(ticks_per_axis):
        for norm_pos, label in ticks:
            pos = [-half, -half, -half]
            pos[axis_i] = norm_pos
            items.append(DepthFadedTextItem(
                display_range=display_range, pos=tuple(pos), text=label, color=fg_color,
            ))
    return items

def build_axis_name_items(channel_labels, display_range, fg_color):
    """
    One GLTextItem per axis (x, y, z), showing that axis's antigen:marker
    label - separate from build_tick_items()'s per-gridline tick values.
    Centred on the axis midpoint and pushed outward from the cube edge on
    the other two axes, so it clears the tick text at every viewing angle
    and does not overlap data points. Floats in the GL scene like the tick
    text, so it stays attached to its axis as the plot is rotated - unlike
    the legend_label fallback, which is a flat 2D widget under the canvas
    and does not rotate with the scene.
    """
    items = []
    half = display_range / 2.0
    outer = -half * AXIS_NAME_OFFSET
    centred = QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter
    for axis_i, label in enumerate(channel_labels):
        pos = [outer, outer, outer]
        pos[axis_i] = 0.0
        items.append(gl.GLTextItem(
            pos=tuple(pos), text=label, color=fg_color, alignment=centred,
        ))
    return items

def build_legend_text(ticks_per_axis, channel_names):
    """2D fallback when gltext_supported() is False (RISK 4)."""
    lines = ["Axis ticks (2D legend fallback - GLTextItem unavailable):"]
    for name, ticks in zip(channel_names, ticks_per_axis):
        vals_str = ", ".join(label for _pos, label in ticks if label)
        lines.append(f"  {name}: {vals_str}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Colour modes - density via continuous ColorMap.map(), categorical
# via a discrete lookup table. Both mirror the existing 2D colour patterns
# (cytometry_plot_widget.py's rainbow4 quadratic warp / glasbey_bw_minc_20).
# --------------------------------------------------------------------------
def build_density_colormap():
    colors_hex = cc.palette['rainbow4']
    n = len(colors_hex)
    pos = 0.9 * np.linspace(0.0, 1.0, n) ** 2 + 0.1 * np.linspace(0.0, 1.0, n)
    return pg.ColorMap(pos=pos, color=colors_hex)


def colors_from_density(values_0to1, cmap=None):
    """values_0to1: per-point density, already normalised/clipped by the
    caller (np.histogramdd + 99.9th-percentile clip, mirroring calc_hist2d).
    Returns an (N,4) float array in 0..1, ready for GLScatterPlotItem."""
    cmap = cmap or build_density_colormap()
    return cmap.map(values_0to1, mode='float')


def build_categorical_colormap():
    colors_hex = cc.palette['glasbey_bw_minc_20']
    return pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors_hex)), color=colors_hex)


def categorical_keys_from_chain(displayed_indices, gate_membership, chain):
    """
    0 = background, 1..len(chain) = gates in 'chain' order - later entries
    overwrite earlier ones for points matching both, same precedence rule as
    calc_dotplot2d's gate-mode dot plots.
    chain: e.g. [colour_mode_gate] + its descendant gate ids, resolved by
    the caller from the real gating hierarchy (not this function's concern).
    """
    keys = np.zeros(len(displayed_indices), dtype=int)
    for level, gate in enumerate(chain, start=1):
        mask = gate_membership.get(gate)
        if mask is None:
            # same startup/recalc race as Plot3DPlotWidget._gate_mask() -
            # gate listed in 'gating' but its mask not yet computed. Treat
            # as contributing no points at this level rather than crashing.
            logger.warning(
                f"Plot3DPlotWidget: gate '{gate}' not yet in "
                f"gate_membership, treating as empty for colouring"
            )
            continue
        keys[mask[displayed_indices]] = level
    return keys


def colors_from_categorical_keys(keys, n_levels, cmap=None, background_rgba=(120, 120, 120, 255)):
    """keys: from categorical_keys_from_chain(). Returns an (N,4) float
    array in 0..1, ready for GLScatterPlotItem.

    Superseded by colors_from_gate_chain() below for the 3D plugin's own
    gate-mode colouring (request #3) - left in place since nothing else in
    this file calls it, but no longer wired into rebuild_colors()."""
    cmap = cmap or build_categorical_colormap()
    lut = cmap.getLookupTable(nPts=max(n_levels + 1, 2), alpha=True).astype(float)
    lut[0] = list(background_rgba)
    return lut[keys] / 255.0


def default_gate_colours(gate_names):
    """gate_names: e.g. selectable_gate_names(gating) - 'root' first. Builds
    ONE fixed RGBA (0..1) per gate name from the same glasbey_bw_minc_20
    palette already used elsewhere in the app, indexed by each gate's
    position in the hierarchy - not resampled per source-gate selection like
    the old categorical_keys_from_chain/colors_from_categorical_keys pair
    was, so a given population always gets the same colour regardless of
    which gate a tile is currently colouring by."""
    cmap = build_categorical_colormap()
    lut = cmap.getLookupTable(nPts=max(len(gate_names), 2), alpha=True).astype(float) / 255.0
    return {name: tuple(lut[i]) for i, name in enumerate(gate_names)}


def colors_from_gate_chain(keys, chain, gate_colours, background_rgba=(120, 120, 120, 255)):
    """keys: from categorical_keys_from_chain() - 0=background, 1..len(chain)
    map to chain[i-1]. gate_colours: the shared, user-editable workspace
    palette (Plot3DGridWidget.gate_colours) - looked up directly by gate
    name rather than by resampled chain position, so colours stay stable."""
    lut = np.empty((len(chain) + 1, 4), dtype=float)
    lut[0] = np.array(background_rgba, dtype=float) / 255.0
    for i, gate in enumerate(chain, start=1):
        lut[i] = gate_colours[gate]
    return lut[keys]


def selectable_gate_names(gating):
    """All non-QuadrantGate gate names, 'root' first. QuadrantGates excluded
    to match the 2D source-gate convention (CytometryPlotWidget.configure_
    title) - a quadrant gate doesn't have a single coherent membership mask
    to use as a 3D source gate or colour-by chain link."""
    if gating is None:
        return ['root']
    gate_ids = [g for g in gating.get_gate_ids() if gating._get_gate_node(g[0], g[1]).gate_type != 'QuadrantGate']
    return ['root'] + [g[0] for g in gate_ids]


# --------------------------------------------------------------------------
# AxisControlStrip
#
# The 3D canvas has no GraphicsLayoutWidget scene, so the existing 2D
# pattern's two separate pieces - InteractiveLabel (cytometry_plot_components.py,
# click -> channel/transform QMenu) and ZoomAxis (same file, drag -> zoom via
# a 60ms QTimer + _pending_delta) - are ported into one plain QWidget here.
# A press/release pixel-distance threshold (same technique used for tile
# click-select in Plot3DGLView) distinguishes a click (opens a menu) from a
# drag (accumulates zoom); the zoom_timer runs throughout a held press either
# way, exactly like ZoomAxis, so a true click naturally produces zero
# accumulated delta and no zoom.
#
# All three axis strips (X, Y, Z) drag horizontally only, per the plan - so
# unlike ZoomAxis there's no vertical-orientation branch.
#
# apply_zoom() itself (reading _pending_delta and updating a real Transform)
# is NOT implemented here - that's Plot3DPlotWidget's job, since it needs the
# shared Transform object this strip doesn't hold.
# --------------------------------------------------------------------------
class AxisControlStrip(QtWidgets.QLabel):
    CLICK_THRESHOLD_PX = 4
    ZOOM_TIMER_INTERVAL_MS = 60

    def __init__(self, axis_role='x', parent=None):
        super().__init__(parent)
        self.axis_role = axis_role  # 'x' / 'y' / 'z' - for the host's apply_zoom(axis_role)

        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFrameStyle(QtWidgets.QFrame.Shape.Panel | QtWidgets.QFrame.Shadow.Raised)
        self.setMinimumHeight(28)

        self._default_font = self.font()
        self._hover_font = QtGui.QFont(self._default_font)
        self._hover_font.setUnderline(True)

        # left-click channel menu - mirrors InteractiveLabel's naming
        self.leftClickMenuItems = []
        self.leftItemSelected = None
        self.leftClickMenuFunction = None

        # right-click transform menu - mirrors InteractiveLabel's naming
        self.rightClickMenuItems = []
        self.rightItemSelected = None
        self.rightClickMenuFunction = None

        # drag-to-zoom state - ported from ZoomAxis
        self._press_pos = None
        self._last_pos = None
        self._pending_delta = 0
        self.zoom_timer = QtCore.QTimer(self)
        self.zoom_timer.setInterval(self.ZOOM_TIMER_INTERVAL_MS)

    def set_label(self, text):
        self.setText(text)

    # -- hover (mirrors InteractiveLabel's underline + cursor) -------------
    def enterEvent(self, event):
        self.setFont(self._hover_font)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setFont(self._default_font)
        QtWidgets.QApplication.restoreOverrideCursor()
        super().leaveEvent(event)

    # -- mouse handling (click -> menu, drag -> zoom accumulation) --------
    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            self._show_transform_menu(ev.globalPosition().toPoint())
            return
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self._press_pos = ev.position()
            self._last_pos = self._press_pos
            self._pending_delta = 0
            self.zoom_timer.start()

    def mouseMoveEvent(self, ev):
        if self._last_pos is None:
            return
        pos = ev.position()
        delta = pos - self._last_pos
        self._last_pos = pos
        self._pending_delta += delta.x()  # all-horizontal-drag, per the plan
        if QtWidgets.QApplication.overrideCursor() == QtCore.Qt.CursorShape.PointingHandCursor:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.ClosedHandCursor)

    def mouseReleaseEvent(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        self.zoom_timer.stop()
        was_click = False
        if self._press_pos is not None:
            dist = (ev.position() - self._press_pos).manhattanLength()
            was_click = dist < self.CLICK_THRESHOLD_PX
        self._press_pos = None
        self._last_pos = None
        self._pending_delta = 0
        QtWidgets.QApplication.restoreOverrideCursor()
        if was_click:
            self._show_channel_menu(ev.globalPosition().toPoint())

    # -- menus (mirrors InteractiveLabel's selectable_menu_activates_function
    # / show_right_context_menu) -------------------------------------------
    def _show_channel_menu(self, global_pos):
        menu = QtWidgets.QMenu(self)
        actions = {}
        for n, item in enumerate(self.leftClickMenuItems):
            action = menu.addAction(item)
            action.setCheckable(True)
            if n == self.leftItemSelected:
                action.setChecked(True)
            actions[action] = n
        chosen = menu.exec(global_pos)
        if chosen is not None:
            self.leftItemSelected = actions[chosen]
            if self.leftClickMenuFunction is not None:
                self.leftClickMenuFunction(self.leftItemSelected, self)
                logger.info(f"AxisControlStrip[{self.axis_role}] channel -> {self.leftItemSelected}")

    def _show_transform_menu(self, global_pos):
        menu = QtWidgets.QMenu(self)
        actions = {}
        for n, item in enumerate(self.rightClickMenuItems):
            action = menu.addAction(item)
            action.setCheckable(True)
            if n == self.rightItemSelected:
                action.setChecked(True)
            actions[action] = n
        chosen = menu.exec(global_pos)
        if chosen is not None:
            self.rightItemSelected = actions[chosen]
            if self.rightClickMenuFunction is not None:
                self.rightClickMenuFunction(self.rightItemSelected, self)
                logger.info(f"AxisControlStrip[{self.axis_role}] transform -> {self.rightItemSelected}")


# --------------------------------------------------------------------------
# GateTitleBar - tile header: source-gate selector +
# colour-mode combo ('Density' + every selectable gate name).
# --------------------------------------------------------------------------
class GateTitleBar(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        layout.addWidget(QtWidgets.QLabel("Source gate:"))
        self.gate_combo = QtWidgets.QComboBox()
        layout.addWidget(self.gate_combo)

        layout.addStretch(1)

        layout.addWidget(QtWidgets.QLabel("Colour by:"))
        self.colour_combo = QtWidgets.QComboBox()
        layout.addWidget(self.colour_combo)

        self.empty_gate_label = QtWidgets.QLabel("No events in gate")
        self.empty_gate_label.setStyleSheet("color: #d9534f; font-weight: bold;")
        self.empty_gate_label.setVisible(False)
        layout.addWidget(self.empty_gate_label)

    def set_gate_options(self, gate_names, current=None):
        gate_names = list(gate_names)
        self.gate_combo.blockSignals(True)
        self.gate_combo.clear()
        self.gate_combo.addItems(gate_names)
        if current in gate_names:
            self.gate_combo.setCurrentText(current)
        self.gate_combo.blockSignals(False)

    def set_colour_options(self, gate_names, current='Density'):
        options = ['Density'] + list(gate_names)
        self.colour_combo.blockSignals(True)
        self.colour_combo.clear()
        self.colour_combo.addItems(options)
        if current in options:
            self.colour_combo.setCurrentText(current)
        self.colour_combo.blockSignals(False)


# --------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------
def make_grids(extent):
    half = extent / 2.0
    specs = [
        (None, (0, 0, -half)),                  # xy plane
        ((90, 1, 0, 0), (0, -half, 0)),         # xz plane
        ((90, 0, 1, 0), (-half, 0, 0)),         # yz plane
    ]
    grids = []
    for rotation, offset in specs:
        g = gl.GLGridItem()
        try:
            g.setColor((200, 200, 200, 60))
        except Exception:
            pass  # cosmetic only - not all pyqtgraph versions expose setColor here
        if rotation is not None:
            angle, ax, ay, az = rotation
            g.rotate(angle, ax, ay, az)
        g.translate(*offset)
        if GRID_RESIZE_METHOD == 'setSize':
            g.setSize(x=extent, y=extent, z=1)
        else:
            factor = extent / GRID_DEFAULT_SIZE
            g.scale(factor, factor, 1)
        grids.append(g)
    return grids


# --------------------------------------------------------------------------
# Density colour mode - 48-bin histogram per axis, with bin edges
# spaced evenly in *display* (transformed) space rather than raw event
# space - same construction as Transform.scale / calc_hist2d's
# bins=transform.scale pattern (functions.py), just at a smaller,
# plugin-local bin count instead of settings.hist_bins_retrieved (a full
# hist_bins_retrieved**3 grid would be far too much memory for a 3D
# histogram).
# --------------------------------------------------------------------------
def density_bin_edges(tr, n_bins=N_DENSITY_BINS):
    """One axis's histogramdd bin edges - subsampled from tr.scale, the
    same already-validated bin-edge array calc_hist1d/calc_hist2d use
    (functions.py), instead of recomputing tr.xform.inverse() from scratch
    at a coarser resolution. Recomputing independently at n_bins=48 hit
    non-monotonic/duplicate values from xform.inverse() that tr.scale's own
    resolution (settings.hist_bins_retrieved) doesn't - FlowKit's logicle
    inverse flattens out near its boundary, and 48 widely-spaced samples
    are far more likely to land twice in that flat region than tr.scale's
    finer sampling. tr.scale already carries the -inf/+inf caps
    (transform.py), so this only needs to evenly subsample its interior.

    Picking unique *indices* into tr.scale doesn't guarantee unique
    *values* - in a flattened region of the inverse, two different indices
    can round to the same float64 value once subsampled this coarsely,
    which breaks np.histogramdd's strictly-increasing bin requirement.
    Deduplicating the sampled values (not just the indices) after
    subsampling guards against that; tr.scale is already sorted, so
    np.unique on the subsample is equivalent to a stable dedup."""
    inner = tr.scale[1:-1]
    if len(inner) > n_bins:
        idx = np.unique(np.linspace(0, len(inner) - 1, n_bins).round().astype(int))
        inner = inner[idx]
    inner = np.unique(inner)  # dedup values, not just indices
    return np.concatenate(([-np.inf], inner, [np.inf]))


def compute_density_lookup(gated_pts_3ch, bin_edges):
    hist, edges = np.histogramdd(gated_pts_3ch, bins=bin_edges)
    nonzero = hist[hist > 0]
    clip_val = float(np.percentile(nonzero, 99.9)) if nonzero.size else 1.0
    hist_clipped = np.clip(hist, 0, clip_val)
    return hist_clipped, edges


def lookup_density_for_points(points_3ch, hist, edges):
    if points_3ch.shape[0] == 0:
        return np.empty(0, dtype=float)
    idx = []
    for axis_i in range(3):
        ax_idx = np.searchsorted(edges[axis_i], points_3ch[:, axis_i], side='right') - 1
        ax_idx = np.clip(ax_idx, 0, hist.shape[axis_i] - 1)
        idx.append(ax_idx)
    vals = hist[idx[0], idx[1], idx[2]]
    peak = vals.max()
    if peak > 0:
        vals = vals / peak
    return vals


# --------------------------------------------------------------------------
# Gate colour mode - chain resolution ported from the existing
# dot-plot-by-gate pattern (functions.py calc_dotplot2d call sites): all
# descendants of a gate share its path-tuple as a substring, with the
# project's documented QuadrantGate container filter applied.
# --------------------------------------------------------------------------
def resolve_gate_chain(gating, colour_mode_gate):
    """[colour_mode_gate] + every descendant, shallowest first."""
    gate_ids = [
        g for g in gating.get_gate_ids()
        if gating._get_gate_node(g[0], g[1]).gate_type != 'QuadrantGate'
    ]
    descendants = [g for g in gate_ids if colour_mode_gate in g[1]]
    descendants.sort(key=lambda g: len(g[1]))
    return [colour_mode_gate] + [g[0] for g in descendants]


# --------------------------------------------------------------------------
# Tile image capture - .grab(), not .render(): GLViewWidget is a
# QOpenGLWidget, and the existing 2D get_widget_pixmap() (QPainter + widget.
# render()) isn't guaranteed to capture composited OpenGL content correctly.
# .grab() is Qt's own OpenGL-aware screenshot call and works for ordinary
# widgets too, so grabbing the whole tile (title bar + canvas + axis strips)
# in one call is simplest.
# --------------------------------------------------------------------------
def get_tile_pixmap(tile):
    return tile.grab()


def export_tile_png(tile, filename):
    filename_png = str(Path(filename).with_suffix('.png'))
    get_tile_pixmap(tile).save(filename_png)
    logger.info(f"Plot3DPlotWidget: exported PNG to {filename_png}")


def copy_tile_to_clipboard(tile):
    pm = get_tile_pixmap(tile)
    clipboard = QtWidgets.QApplication.clipboard()
    mime_data = QtCore.QMimeData()
    image = pm.toImage()

    png_data = QtCore.QByteArray()
    png_buffer = QtCore.QBuffer(png_data)
    png_buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    image.save(png_buffer, "PNG")
    mime_data.setData("image/png", png_data)
    mime_data.setImageData(image)

    clipboard.setMimeData(mime_data)


# --------------------------------------------------------------------------
# Plot3DGLView - GLViewWidget subclass: orbit/pan are already native
# (GLViewWidget.mouseMoveEvent calls self.orbit()/self.pan() on left/middle
# drag - confirmed from source per the plan §1, nothing to add for those).
# What's added here: always call super() first so built-in orbit/pan keep
# working unmodified, then layer on click-select (small press-to-release
# pixel distance = a plain click, not a drag) and double-click pop-out,
# plus camera-state persistence on every release.
# --------------------------------------------------------------------------
class Plot3DGLView(gl.GLViewWidget):
    CLICK_THRESHOLD_PX = 4

    def __init__(self, tile, parent=None):
        super().__init__(parent)
        self.tile = tile
        self._press_pos = None
        self.in_modal = False  # set True while popped out, to suppress nested select/pop-out

    def mousePressEvent(self, ev):
        super().mousePressEvent(ev)
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            self.tile.show_context_menu(ev.globalPosition().toPoint())
            return
        self._press_pos = ev.position()

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        self.tile.persist_camera_state()
        if (not self.in_modal) and ev.button() == QtCore.Qt.MouseButton.LeftButton and self._press_pos is not None:
            dist = (ev.position() - self._press_pos).manhattanLength()
            if dist < self.CLICK_THRESHOLD_PX:
                self.tile.select_plot_on_parent_grid()
        self._press_pos = None

    def mouseDoubleClickEvent(self, ev):
        super().mouseDoubleClickEvent(ev)
        if (not self.in_modal) and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.tile.select_plot_on_parent_grid()
            QtCore.QTimer.singleShot(0, self.tile.open_in_modal)

    def _zoom_by_scroll(self, delta):
        self.opts['distance'] *= 0.999 ** delta
        self.update()

    def wheelEvent(self, ev):
        """
        Mouse wheel (angle deltas only) zooms. Trackpad scrolling carries
        pixel deltas on both axes plus momentum events, so it pans instead
        (Ctrl + scroll zooms); pinch is handled in event().
        """
        pixel = ev.pixelDelta()
        if pixel.isNull():
            angle = ev.angleDelta()
            self._zoom_by_scroll(angle.y() or angle.x())
        elif ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            self._zoom_by_scroll(pixel.y())
        else:
            self.pan(pixel.x(), pixel.y(), 0, relative='view-upright')
        self.tile.persist_camera_state()
        ev.accept()

    def event(self, ev):
        if (ev.type() == QtCore.QEvent.Type.NativeGesture
                and ev.gestureType() == QtCore.Qt.NativeGestureType.ZoomNativeGesture):
            self.opts['distance'] /= max(1.0 + ev.value(), 0.1)
            self.update()
            self.tile.persist_camera_state()
            ev.accept()
            return True
        return super().event(ev)


# --------------------------------------------------------------------------
# Plot3DPlotWidget - rendering pipeline + full interactivity: GateTitleBar,
# AxisControlStrips, drag-to-zoom, rotation persistence, click-select / double-click
# pop-out, six-item right-click menu.
# --------------------------------------------------------------------------
class Plot3DPlotWidget(QtWidgets.QFrame):
    """
    Single 3D plot tile.

    bus: shared honeychrome EventBus - only existing signals are read or
        (queued-then-)emitted; no new signals are added.
    grid: the owning Plot3DGridWidget - used for tile selection, pop-out,
        and to queue axisTransformed/axesReset emissions until it's safe to
        fire them (see module docstring + Plot3DGridWidget.mark_channel_
        transformed / mark_channels_reset).
    controller: the honeychrome Controller - used only for "Reset Axes"
        (assign_default_transforms needs controller.experiment.settings)
        and for the live/current-sample check it already uses centrally.
    data_for_cytometry_plots_unmixed: dict borrowed directly (non-copy) from
        the controller - keys 'event_data', 'pnn', 'pnn_labels',
        'fluoro_indices', 'transformations', 'gating', 'gate_membership'.
    plot: plot-spec dict - mutated in place as
        the user interacts with the tile.
    """

    AXIS_ROLES = ('x', 'y', 'z')

    def __init__(self, bus, grid, controller, data_for_cytometry_plots_unmixed, plot,
                 max_points=DEFAULT_MAX_POINTS, is_dark=False, seed=None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.grid = grid
        self.controller = controller
        self.data_for_cytometry_plots = data_for_cytometry_plots_unmixed
        self.plot = plot
        self.max_points = max_points
        self.is_dark = is_dark
        self.display_range = DEFAULT_DISPLAY_RANGE
        self._rng = np.random.default_rng(seed)

        self.pnn = self.data_for_cytometry_plots['pnn']
        self.pnn_labels = self.data_for_cytometry_plots.get('pnn_labels') or {}
        self.transformations = self.data_for_cytometry_plots['transformations']
        self.gate_membership = self.data_for_cytometry_plots['gate_membership']
        self.event_data = self.data_for_cytometry_plots['event_data']
        self.gating = self.data_for_cytometry_plots.get('gating')

        self.channels = [self.plot['channel_x'], self.plot['channel_y'], self.plot['channel_z']]
        self.id_channels = [self.pnn.index(ch) for ch in self.channels]
        self.refresh_transforms()

        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setLineWidth(1)

        self._build_ui()
        self._build_grids()
        self.resample()
        self.rebuild_colors()
        self.rebuild_positions()
        self._build_scatter()
        self._build_ticks()
        self._set_initial_camera()
        self.refresh_gate_options()
        self._update_axis_strip_labels()

    # -- UI shell -------------------------------------------------------------
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self.title_bar = GateTitleBar()
        layout.addWidget(self.title_bar)

        self.gl_view = Plot3DGLView(self)
        # Forced dark/bright regardless of self.is_dark (the app's theme) -
        # see the "On-plot axis-name labels" note: GLTextItem's only
        # confirmed-working glOptions here is its default 'additive', which
        # can only brighten a pixel, never darken one, so bright text on a
        # light/white canvas is invisible no matter the colour chosen.
        # Axis-name labels need to stay visible while the plot is rotated,
        # which a flat 2D legend can't do, so the 3D canvas always uses the
        # dark pairing now, independent of the rest of the app's theme.
        self.gl_view.setBackgroundColor(gl_background_color(True))
        self.gl_view.setMinimumSize(300, 300)
        layout.addWidget(self.gl_view, stretch=1)

        self.legend_label = QtWidgets.QLabel()
        self.legend_label.setStyleSheet("font-family: monospace; font-size: 10px;")
        layout.addWidget(self.legend_label)

        self.tick_items = []
        self.axis_name_items = []
        self._ticks_available = gltext_supported()
        logger.info(f"Plot3DPlotWidget: ticks_available -> {self._ticks_available}")
        self.legend_label.setVisible(not self._ticks_available)

        strips_layout = QtWidgets.QHBoxLayout()
        self.axis_strips = []
        for role in self.AXIS_ROLES:
            strip = AxisControlStrip(axis_role=role)
            strip.leftClickMenuFunction = self._on_channel_selected
            strip.rightClickMenuFunction = self._on_transform_selected
            strip.zoom_timer.timeout.connect(lambda r=role: self.apply_zoom(r))
            strips_layout.addWidget(strip)
            self.axis_strips.append(strip)
        layout.addLayout(strips_layout)

        self.title_bar.gate_combo.currentTextChanged.connect(self._on_source_gate_changed)
        self.title_bar.colour_combo.currentTextChanged.connect(self._on_colour_mode_changed)

    def _build_grids(self):
        self.grids = make_grids(self.display_range)
        show_grid = self.plot.get('show_grid', True)
        for g in self.grids:
            g.setVisible(show_grid)
            self.gl_view.addItem(g)

    # -- gate options (title bar combos) -------------------------------------
    def refresh_gate_options(self):
        # re-fetch rather than trust the cached reference - controller
        # replaces 'gate_membership' (and 'gating') with a brand new
        # dict/object on gating recalculation instead of mutating in place,
        # so a reference captured once in __init__ goes stale the moment a
        # gate is added/removed/renamed (root cause of the 'CD3' KeyError).
        self.gate_membership = self.data_for_cytometry_plots['gate_membership']
        self.gating = self.data_for_cytometry_plots.get('gating')

        gate_names = selectable_gate_names(self.gating)

        stale_source_gate = self.plot.get('source_gate') not in gate_names
        if stale_source_gate:
            logger.warning(
                f"Plot3DPlotWidget: source_gate '{self.plot.get('source_gate')}' "
                f"no longer exists, falling back to root"
            )
            self.plot['source_gate'] = 'root'

        stale_colour_mode = self.plot.get('colour_mode') not in (['Density'] + gate_names)
        if stale_colour_mode:
            logger.warning(
                f"Plot3DPlotWidget: colour_mode '{self.plot.get('colour_mode')}' "
                f"no longer exists, falling back to Density"
            )
            self.plot['colour_mode'] = 'Density'

        self.title_bar.set_gate_options(gate_names, current=self.plot['source_gate'])
        self.title_bar.set_colour_options(gate_names, current=self.plot['colour_mode'])

        # always re-resample/recolour - gate_membership's *content* can
        # change (boundary edit) without any gate name becoming invalid, and
        # that case needs the same rebuild stale_source_gate/colour_mode
        # already get below.
        self.resample()
        self.rebuild_colors()
        self.rebuild_positions()
        self._refresh_scatter()

    def _on_source_gate_changed(self, text):
        if not text or text == self.plot.get('source_gate'):
            return
        self.plot['source_gate'] = text
        self.resample()
        self.rebuild_colors()
        self.rebuild_positions()
        self._refresh_scatter()
        self._build_ticks()

    def _on_colour_mode_changed(self, text):
        if not text or text == self.plot.get('colour_mode'):
            return
        self.plot['colour_mode'] = text
        self.rebuild_colors()
        self._refresh_scatter()

    # -- axis strip labels / menus --------------------------------------------
    def _update_axis_strip_labels(self):
        for axis_i, strip in enumerate(self.axis_strips):
            ch = self.channels[axis_i]
            strip.set_label(self.pnn_labels.get(ch, ch))
            strip.leftClickMenuItems = [self.pnn_labels.get(c, c) for c in self.pnn]
            strip.leftItemSelected = self.id_channels[axis_i]
            strip.rightClickMenuItems = transforms_menu_items
            tr_id = self.transformations[ch].id
            strip.rightItemSelected = tr_id if isinstance(tr_id, int) else None

    def _on_channel_selected(self, n, strip):
        axis_i = self.AXIS_ROLES.index(strip.axis_role)
        new_channel = self.pnn[n]
        self.channels[axis_i] = new_channel
        self.id_channels[axis_i] = n
        self.plot[f'channel_{strip.axis_role}'] = new_channel
        self.refresh_transforms()
        self.rebuild_positions()
        self.rebuild_colors()
        self._refresh_scatter()
        self._build_ticks()
        self._update_axis_strip_labels()
        logger.info(f'Plot3DPlotWidget: channel_{strip.axis_role} -> {new_channel}')

    def _on_transform_selected(self, n, strip):
        axis_i = self.AXIS_ROLES.index(strip.axis_role)
        channel = self.channels[axis_i]
        self.transformations[channel].set_transform(id=n)
        self.refresh_transforms()
        self.rebuild_positions()
        self._refresh_scatter()
        self._build_ticks()
        self._update_axis_strip_labels()
        self.grid.mark_channel_transformed(channel)
        logger.info(f'Plot3DPlotWidget: {channel} transform -> {n}')

    def apply_zoom(self, axis_role):
        """Drag-to-zoom, fired every ZOOM_TIMER_INTERVAL_MS while a strip is
        held - ported from CytometryPlotWidget.apply_zoom
        (cytometry_plot_widget.py), adapted for AxisControlStrip's plain
        QLabel + horizontal-only drag: there's no ViewBox/AxisItem here to
        read a value-space click position from, so the logicle w-vs-limits
        split uses the press position's fraction across the strip's own
        width as the equivalent signal."""
        axis_i = self.AXIS_ROLES.index(axis_role)
        strip = self.axis_strips[axis_i]
        channel = self.channels[axis_i]
        tr = self.transformations[channel]

        step = strip._pending_delta
        strip._pending_delta = 0
        if step == 0:
            return

        threshold = 1  # pixels
        if abs(step) < threshold:
            return

        zoom_rate = 1.04
        factor = (1 / zoom_rate) if step > 0 else zoom_rate

        lo, hi = tr.limits
        if tr.id == 0 or tr.id == 2:  # linear or log
            new_min = (lo - tr.zero) * factor + tr.zero
            new_max = (hi - tr.zero) * factor + tr.zero
            tr.set_transform(limits=[new_min, new_max])
        elif tr.id == 1:  # logicle
            press_frac = 0.5
            if strip._press_pos is not None and strip.width() > 0:
                press_frac = strip._press_pos.x() / strip.width()
            value_at_press = lo + press_frac * (hi - lo)
            if value_at_press < 0.5 * hi:
                tr.logicle_w = tr.logicle_w / factor
                tr.set_transform()
            else:
                new_min = (lo - tr.zero) * factor + tr.zero
                new_max = (hi - tr.zero) * factor + tr.zero
                tr.set_transform(limits=[new_min, new_max])
        else:
            return  # 'default' (e.g. Time) - no zoom, matches CytometryPlotWidget

        self.refresh_transforms()
        self.rebuild_positions()
        self._refresh_scatter()
        self._build_ticks()
        self._update_axis_strip_labels()
        self.grid.mark_channel_transformed(channel)
        logger.info(f'Plot3DPlotWidget: {channel} zoom -> limits {tr.limits}')

    def refresh_transforms(self):
        # Re-lookup by name rather than trusting a cached list
        self.transforms = [self.transformations[ch] for ch in self.channels]

    def _gate_mask(self, gate_name):
        """Safe gate_membership lookup - gate_membership can lack ANY key,
        including 'root', until controller.calc_hists_and_stats() has run at
        least once for the current sample/mode (data_for_cytometry_plots
        starts as {} - controller.py - and 'gating' can already list a gate
        by name before apply_gates_in_place() has computed its mask). Falls
        back to 'root', then to a synthetic all-events mask. Self-corrects:
        refresh_after_data_change() (wired to bus.histsStatsRecalculated)
        re-resamples every tile the moment the real gate_membership lands."""
        mask = self.gate_membership.get(gate_name)
        if mask is None and gate_name != 'root':
            logger.warning(
                f"Plot3DPlotWidget: gate '{gate_name}' not yet in "
                f"gate_membership, falling back to root"
            )
            mask = self.gate_membership.get('root')
        if mask is None:
            logger.warning(
                "Plot3DPlotWidget: gate_membership has no entries yet "
                "(calc_hists_and_stats hasn't run for this sample/mode) - "
                "showing all events untriaged"
            )
            mask = np.ones(len(self.event_data), dtype=np.bool_)
        return mask

    def resample(self):
        if self.plot['source_gate'] not in self.gate_membership:
            self.plot['source_gate'] = 'root'
        mask = self._gate_mask(self.plot['source_gate'])
        source_indices = np.nonzero(mask)[0]
        n = min(self.max_points, len(source_indices))
        if n < len(source_indices):
            self.displayed_indices = self._rng.choice(source_indices, size=n, replace=False)
        else:
            self.displayed_indices = source_indices

        is_empty = len(source_indices) == 0
        self.title_bar.empty_gate_label.setVisible(is_empty)
        if is_empty:
            logger.warning(
                f"Plot3DPlotWidget: gate '{self.plot['source_gate']}' has "
                f"no events for the current sample - showing empty plot"
            )

    def rebuild_positions(self):
        raw_pts = self.event_data[self.displayed_indices][:, self.id_channels]
        positions = np.empty_like(raw_pts, dtype=float)
        for axis_i, tr in enumerate(self.transforms):
            display_vals = to_display(raw_pts[:, axis_i], tr)
            lo, hi = tr.limits
            positions[:, axis_i] = normalise_value(display_vals, lo, hi, self.display_range)
        self.positions = positions

    def rebuild_colors(self):
        colour_mode = self.plot.get('colour_mode', 'Density')
        source_mask = self._gate_mask(self.plot['source_gate'])

        if colour_mode == 'Density':
            gated_pts = self.event_data[source_mask][:, self.id_channels]
            bin_edges = [density_bin_edges(tr) for tr in self.transforms]
            hist, edges = compute_density_lookup(gated_pts, bin_edges)
            displayed_pts = self.event_data[self.displayed_indices][:, self.id_channels]
            vals = lookup_density_for_points(displayed_pts, hist, edges)
            self.colors = colors_from_density(vals)
        else:
            if self.gating is None:
                raise ValueError(
                    f"colour_mode '{colour_mode}' needs a real gating object "
                    f"to resolve descendants - 'gating' was None."
                )
            chain = resolve_gate_chain(self.gating, colour_mode)
            keys = categorical_keys_from_chain(self.displayed_indices, self.gate_membership, chain)
            self.colors = colors_from_gate_chain(keys, chain, self.grid.gate_colours)

    # -- scene objects --------------------------------------------------------
    def _build_scatter(self):
        # explicit 'opaque' (real depth test, no blending) instead of
        # GLScatterPlotItem's additive default - additive sums every
        # overlapping point's colour with no depth test, which saturates to
        # white well before 100k points overlap at this tile size regardless
        # of which colour was assigned.
        self.scatter = gl.GLScatterPlotItem(
            pos=self.positions, color=self.colors, size=POINT_SIZE, pxMode=True, glOptions='opaque',
        )
        self.gl_view.addItem(self.scatter)

    def _refresh_scatter(self):
        self.scatter.setData(pos=self.positions, color=self.colors, size=POINT_SIZE, pxMode=True)

    def rebuild_gl_items(self):
        """Discard and recreate every GL item on this tile (grids, scatter,
        tick text) - call after gl_view has been reparented across a
        top-level-window boundary (pop-out-to-modal / pop-back-in). Without
        Qt.AA_ShareOpenGLContexts set app-wide (main.py), that kind of
        reparent can destroy/replace the QOpenGLWidget's context, leaving
        previously-compiled shader programs cached on the old item objects
        invalid in the new context (GLError 1281 on glUseProgram). Building
        fresh item objects sidesteps that - each one lazily compiles its
        shader on its own next paint, against whatever context is current
        then."""
        for item in list(self.gl_view.items):
            self.gl_view.removeItem(item)
        self.tick_items = []
        self.axis_name_items = []
        self._build_grids()
        self._build_scatter()
        self._build_ticks()

    def _build_ticks(self):
        fg = gl_foreground_rgba(True)  # canvas is forced dark - see _build_ui
        tick_fg = gl_tick_rgba(True)
        ticks_per_axis = [ticks_for_axis(tr, self.display_range) for tr in self.transforms]
        channel_labels = [self.pnn_labels.get(ch, ch) for ch in self.channels]
        if self._ticks_available:
            for item in self.tick_items:
                self.gl_view.removeItem(item)
            self.tick_items = build_tick_items(ticks_per_axis, self.display_range, tick_fg)

            for item in self.axis_name_items:
                self.gl_view.removeItem(item)
            self.axis_name_items = build_axis_name_items(channel_labels, self.display_range, fg)

            for item in self.tick_items + self.axis_name_items:
                self.gl_view.addItem(item)
            logger.info(
                f"Plot3DPlotWidget: built {len(self.tick_items)} tick + "
                f"{len(self.axis_name_items)} axis-name GLTextItem(s)"
            )
        else:
            self.legend_label.setText(build_legend_text(ticks_per_axis, channel_labels))

    def _set_initial_camera(self):
        elevation = self.plot.get('elevation', 20.0)
        azimuth = self.plot.get('azimuth', -60.0)
        distance = self.plot.get('distance') or self.display_range * 2.2
        self.gl_view.setCameraPosition(distance=distance, elevation=elevation, azimuth=azimuth)

    # -- rotation persistence / click-select / pop-out ---------------
    def persist_camera_state(self):
        opts = self.gl_view.opts
        self.plot['elevation'] = opts.get('elevation', self.plot.get('elevation', 20.0))
        self.plot['azimuth'] = opts.get('azimuth', self.plot.get('azimuth', -60.0))
        self.plot['distance'] = opts.get('distance', self.plot.get('distance'))

    def reset_rotation(self):
        distance = self.display_range * 2.2
        self.gl_view.setCameraPosition(distance=distance, elevation=20.0, azimuth=-60.0)
        self.persist_camera_state()

    def select_plot_on_parent_grid(self):
        if self.grid is not None:
            self.grid.select_plot(self)

    def open_in_modal(self):
        if self.grid is not None:
            self.grid.open_plot_in_modal(self)

    # -- grid-lines toggle --------------------------------------------
    def toggle_grid_lines(self, checked):
        self.plot['show_grid'] = checked
        for g in self.grids:
            g.setVisible(checked)

    # -- right-click menu ---------------------------------------------
    def show_context_menu(self, global_pos):
        menu = QtWidgets.QMenu(self)
        menu.addAction("Fit Axes to Data", self.fit_axes_to_data)
        menu.addAction("Reset Axes", self.reset_axes_transforms)
        menu.addAction("Reset Rotation", self.reset_rotation)
        grid_action = menu.addAction("Toggle Grid Lines")
        grid_action.setCheckable(True)
        grid_action.setChecked(self.plot.get('show_grid', True))
        grid_action.toggled.connect(self.toggle_grid_lines)
        menu.addAction("Export Image", self.export_image)
        menu.addAction("Copy to Clipboard", lambda: copy_tile_to_clipboard(self))
        menu.exec(global_pos)

    def export_image(self):
        base_directory = Path.home() / settings.experiments_folder
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export 3D plot as image", str(base_directory), "PNG Image File (*.png)"
        )
        if path:
            export_tile_png(self, path)

    # -- fit / reset axes (ported from CytometryPlotWidget) -----------
    def fit_axes_to_data(self):
        fitted_channels = []
        for axis_i, channel in enumerate(self.channels):
            tr = self.transformations[channel]
            col_idx = self.id_channels[axis_i]
            channel_min, channel_max = np.percentile(self.event_data[:, col_idx], [1, 99])
            if not channel_max > 0:  # e.g. AF channels on a sample without AF correction
                continue

            if tr.id == 1:  # logicle
                tr.scale_t = 1.5 * channel_max
                if channel_min < -10 ** tr.logicle_w:
                    logicle_a = min(
                        max(np.log10(-channel_min) - tr.logicle_w, 0),
                        0.5 * np.log10(tr.scale_t),
                    )
                else:
                    logicle_a = 0
                tr.logicle_a = logicle_a
                tr.set_transform(limits=[0, 1])
                fitted_channels.append(channel)
            elif tr.id == 0:  # linear
                tr.scale_t = 1.5 * channel_max
                tr.linear_a = max(-channel_min * 2, 0)
                tr.set_transform(limits=[0, 1])
                fitted_channels.append(channel)

        if fitted_channels:
            self.refresh_transforms()
            self.rebuild_positions()
            self._refresh_scatter()
            self._build_ticks()
            self._update_axis_strip_labels()
            for channel in fitted_channels:
                self.grid.mark_channel_transformed(channel)

    def reset_axes_transforms(self):
        # See the module docstring's "core-code landmine" note: this re-uses
        # the exact helpers controller.reset_axes_transforms() would call,
        # applying the reset locally (immediate, correct feedback on this
        # tile) instead of going through bus.axesReset while this tab is
        # active (which would crash inside core code this plugin can't
        # touch). bus.axesReset is still queued purely so the 2D Unmixed
        # Data tab's cached histograms get recalculated next time it's
        # visited.
        settings_unmixed = self.controller.experiment.settings['unmixed']
        new_transforms = generate_transformations(assign_default_transforms(
            settings_unmixed, channels=self.channels, n_af_spectra=self.controller.n_af_spectra()))

        for channel in self.channels:
            if self.controller.current_sample_path != self.controller.live_sample_path:
                if new_transforms[channel].id is None:  # 'default' (e.g. Time)
                    col_idx = self.pnn.index(channel)
                    upper_limit = max(self.event_data[:, col_idx]) * 1.05
                    new_transforms[channel].set_transform(limits=[0, upper_limit])
            self.transformations[channel] = new_transforms[channel]

        self.refresh_transforms()
        self.rebuild_positions()
        self.rebuild_colors()
        self._refresh_scatter()
        self._build_ticks()
        self._update_axis_strip_labels()
        self.grid.mark_channels_reset(self.channels)

    # -- workspace-wide max-points setting ----------------------------
    def set_max_points(self, value):
        self.max_points = value
        self.resample()
        self.rebuild_colors()
        self.rebuild_positions()
        self._refresh_scatter()

    # -- full refresh after external data change (sample load / gate edit) --
    def refresh_after_data_change(self):
        self.event_data = self.data_for_cytometry_plots['event_data']
        self.gate_membership = self.data_for_cytometry_plots['gate_membership']
        self.gating = self.data_for_cytometry_plots.get('gating')
        self.refresh_transforms()
        self.refresh_gate_options()
        self.resample()
        self.rebuild_colors()
        self.rebuild_positions()
        self._refresh_scatter()
        self._build_ticks()
        self._update_axis_strip_labels()


# --------------------------------------------------------------------------
# NewPlot3DTile
# --------------------------------------------------------------------------
class NewPlot3DTile(QtWidgets.QFrame):
    def __init__(self, grid, parent=None):
        super().__init__(parent)
        self.grid = grid
        pnn = grid.data_for_cytometry_plots['pnn']
        pnn_labels = grid.data_for_cytometry_plots.get('pnn_labels') or {}
        gate_names = selectable_gate_names(grid.data_for_cytometry_plots['gating'])
        channel_labels = [pnn_labels.get(ch, ch) for ch in pnn]

        self.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)

        self.gate_combo = QtWidgets.QComboBox()
        self.gate_combo.addItem("")
        self.gate_combo.addItems(gate_names)

        self.x_combo = QtWidgets.QComboBox()
        self.x_combo.addItem("")
        self.x_combo.addItems(channel_labels)
        self.y_combo = QtWidgets.QComboBox()
        self.y_combo.addItem("")
        self.y_combo.addItems(channel_labels)
        self.z_combo = QtWidgets.QComboBox()
        self.z_combo.addItem("")
        self.z_combo.addItems(channel_labels)

        self._pnn = pnn

        layout = QtWidgets.QVBoxLayout(self)
        layout.addStretch()
        for label_text, combo in (
            ("Source gate:", self.gate_combo),
            ("X axis:", self.x_combo),
            ("Y axis:", self.y_combo),
            ("Z axis:", self.z_combo),
        ):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(label_text))
            row.addWidget(combo)
            layout.addLayout(row)

        self.create_button = QtWidgets.QPushButton("Create Plot")
        self.create_button.setEnabled(False)
        layout.addWidget(self.create_button)
        layout.addStretch()

        for combo in (self.gate_combo, self.x_combo, self.y_combo, self.z_combo):
            combo.currentTextChanged.connect(self._update_button_state)
        self.create_button.clicked.connect(self._emit_create)

    def _update_button_state(self, _text=None):
        ready = all([
            self.gate_combo.currentText(),
            self.x_combo.currentText(),
            self.y_combo.currentText(),
            self.z_combo.currentText(),
        ])
        self.create_button.setEnabled(ready)

    def _emit_create(self):
        plot = {
            'type': 'scatter3d',
            'channel_x': self._pnn[self.x_combo.currentIndex() - 1],
            'channel_y': self._pnn[self.y_combo.currentIndex() - 1],
            'channel_z': self._pnn[self.z_combo.currentIndex() - 1],
            'source_gate': self.gate_combo.currentText(),
            'colour_mode': 'Density',
            'show_grid': self.grid.default_show_grid,
            'elevation': 20.0,
            'azimuth': -60.0,
            'distance': None,
            'width': self.grid.default_plot_size,
            'height': self.grid.default_plot_size,
        }
        self.grid.add_new_plot(plot)


# --------------------------------------------------------------------------
# Plot3DToolbar
# --------------------------------------------------------------------------
class Plot3DToolbar(QtWidgets.QToolBar):
    def __init__(self, grid, parent=None):
        super().__init__(parent)
        self.grid = grid
        self.setMovable(False)

        self.action_add_plot = QtGui.QAction(icon('plus'), "Add Plot", self)
        self.action_delete_plot = QtGui.QAction(icon('x'), "Delete Plot", self)
        self.action_move_to_start = QtGui.QAction(icon('chevrons-left'), "Move Plot to Start", self)
        self.action_move_left = QtGui.QAction(icon('chevron-left'), "Move Plot Left", self)
        self.action_move_right = QtGui.QAction(icon('chevron-right'), "Move Plot Right", self)
        self.action_move_to_end = QtGui.QAction(icon('chevrons-right'), "Move Plot to End", self)
        self.action_toggle_grid = QtGui.QAction(icon('border-all'), "Grid Lines for New Plots", self)
        self.action_toggle_grid.setCheckable(True)
        self.action_toggle_grid.setChecked(True)
        self.action_edit_gate_colours = QtGui.QAction("Edit Gate Colours...", self)

        self.addAction(self.action_add_plot)
        self.addAction(self.action_delete_plot)
        self.addSeparator()
        self.addAction(self.action_move_to_start)
        self.addAction(self.action_move_left)
        self.addAction(self.action_move_right)
        self.addAction(self.action_move_to_end)
        self.addSeparator()
        self.addAction(self.action_toggle_grid)
        self.addAction(self.action_edit_gate_colours)
        self.addSeparator()
        self.addWidget(QtWidgets.QLabel("Max points displayed:  "))
        self.max_points_spin = QtWidgets.QSpinBox()
        self.max_points_spin.setRange(1_000, 1_000_000)
        self.max_points_spin.setSingleStep(10_000)
        self.max_points_spin.setValue(DEFAULT_MAX_POINTS)
        self.addWidget(self.max_points_spin)

        self.addWidget(QtWidgets.QLabel("  New plot size:  "))
        self.plot_size_spin = QtWidgets.QSpinBox()
        self.plot_size_spin.setRange(1, 6)
        self.plot_size_spin.setSingleStep(1)
        self.plot_size_spin.setValue(self.grid.default_plot_size)
        self.plot_size_spin.setToolTip("Grid cells per side for new plots (square)")
        self.addWidget(self.plot_size_spin)

        self.action_add_plot.triggered.connect(self.grid.show_new_plot_widget)
        self.action_delete_plot.triggered.connect(self.grid.delete_current_plot)
        self.action_move_to_start.triggered.connect(lambda: self.grid.move_tile('start'))
        self.action_move_left.triggered.connect(lambda: self.grid.move_tile('left'))
        self.action_move_right.triggered.connect(lambda: self.grid.move_tile('right'))
        self.action_move_to_end.triggered.connect(lambda: self.grid.move_tile('end'))
        self.action_toggle_grid.toggled.connect(self.grid.set_default_show_grid)
        self.action_edit_gate_colours.triggered.connect(self.grid.open_gate_colour_editor)
        self.max_points_spin.valueChanged.connect(self.grid.set_max_points)
        self.plot_size_spin.valueChanged.connect(self.grid.set_default_plot_size)

        self.update_button_state(False)

    def update_button_state(self, has_selection):
        self.action_delete_plot.setEnabled(has_selection)
        self.action_move_to_start.setEnabled(has_selection)
        self.action_move_left.setEnabled(has_selection)
        self.action_move_right.setEnabled(has_selection)
        self.action_move_to_end.setEnabled(has_selection)


# --------------------------------------------------------------------------
# GateColourEditorDialog: view the whole gating
# hierarchy and override the colour used for each population in 'colour by
# gate' mode. Reads/writes the one shared, workspace-wide
# Plot3DGridWidget.gate_colours dict, so every tile's gate-mode colouring
# stays consistent and stable regardless of which gate that tile is
# currently colouring by. Built from gating.get_gate_hierarchy(output='dict').
# --------------------------------------------------------------------------
class GateColourEditorDialog(QtWidgets.QDialog):
    SWATCH_SIZE = 20

    def __init__(self, grid, parent=None):
        super().__init__(parent)
        self.grid = grid
        self.setWindowTitle("Edit Gate Colours")
        self.resize(420, 520)

        layout = QtWidgets.QVBoxLayout(self)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Population", "Colour"])
        self.tree.setColumnWidth(0, 280)
        layout.addWidget(self.tree, stretch=1)

        button_row = QtWidgets.QHBoxLayout()
        self.reset_button = QtWidgets.QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self._on_reset)
        button_row.addWidget(self.reset_button)
        button_row.addStretch(1)
        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        button_row.addWidget(self.close_button)
        layout.addLayout(button_row)

        self._swatch_buttons = {}  # gate_name -> QPushButton
        self._rebuild_tree()

    def _rebuild_tree(self):
        self.tree.clear()
        self._swatch_buttons = {}
        gating = None
        if self.grid.data_for_cytometry_plots:
            gating = self.grid.data_for_cytometry_plots.get('gating')

        if gating is None:
            root_item = QtWidgets.QTreeWidgetItem(["root"])
            self.tree.addTopLevelItem(root_item)
            self._add_swatch(root_item, 'root')
            return

        hierarchy = gating.get_gate_hierarchy(output='dict')
        self._add_hierarchy_node(hierarchy, parent_item=None)
        self.tree.expandAll()

    def _add_hierarchy_node(self, node, parent_item):
        name = node['name']
        item = QtWidgets.QTreeWidgetItem([name])
        if parent_item is None:
            self.tree.addTopLevelItem(item)
        else:
            parent_item.addChild(item)
        # QuadrantGates are excluded from selectable_gate_names() (no single
        # coherent membership mask) so they have no swatch here either - the
        # row just shows the name with no colour control.
        if name in self.grid.gate_colours:
            self._add_swatch(item, name)
        for child in node.get('children', []):
            self._add_hierarchy_node(child, item)
        return item

    def _add_swatch(self, item, gate_name):
        button = QtWidgets.QPushButton()
        button.setFixedSize(self.SWATCH_SIZE, self.SWATCH_SIZE)
        self._style_swatch(button, gate_name)
        button.clicked.connect(lambda _checked=False, g=gate_name, b=button: self._on_swatch_clicked(g, b))
        self.tree.setItemWidget(item, 1, button)
        self._swatch_buttons[gate_name] = button

    def _style_swatch(self, button, gate_name):
        r, g, b, a = self.grid.gate_colours[gate_name]
        qcolor = QtGui.QColor.fromRgbF(r, g, b, a)
        button.setStyleSheet(f"background-color: {qcolor.name()}; border: 1px solid #888;")

    def _on_swatch_clicked(self, gate_name, button):
        r, g, b, a = self.grid.gate_colours[gate_name]
        initial = QtGui.QColor.fromRgbF(r, g, b, a)
        chosen = QtWidgets.QColorDialog.getColor(initial, self, f"Colour for '{gate_name}'")
        if chosen.isValid():
            rgba = (chosen.redF(), chosen.greenF(), chosen.blueF(), chosen.alphaF())
            self.grid.apply_gate_colour_change(gate_name, rgba)
            self._style_swatch(button, gate_name)

    def _on_reset(self):
        self.grid.reset_gate_colours()
        for gate_name, button in self._swatch_buttons.items():
            if gate_name in self.grid.gate_colours:
                self._style_swatch(button, gate_name)


# --------------------------------------------------------------------------
# Plot3DGridWidget - tile-grid bin-packing, ported from
# cytometry_grid_widget.py's CytometryGridWidget (pure QGridLayout
# arithmetic, no pyqtgraph-2D-vs-3D dependency either way), adapted for this
# plugin's own session-only plot_specs/tiles lists instead of a separate
# histograms array, and with no autosave/.kit persistence at all.
# --------------------------------------------------------------------------
class Plot3DGridWidget(QtWidgets.QScrollArea):
    def __init__(self, bus, controller, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.controller = controller
        self.data_for_cytometry_plots = None
        self.is_dark = False

        self.setWidgetResizable(True)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.container = QtWidgets.QWidget(parent=self)
        self.layout = QtWidgets.QGridLayout(self.container)
        self.layout.setSpacing(5)
        self.setWidget(self.container)

        self.debounce_timer = QtCore.QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.init_grid)

        self.n_columns = None
        self.tile_real_width = None
        self.occupied = []
        self.row = 0
        self.last_row = 0

        self.toolbar = None
        self.plot_specs = []
        self.tiles = []           # parallel list of Plot3DPlotWidget instances
        self._placeholder = None  # at most one NewPlot3DTile at a time
        self.selected_tile = None

        self.max_points = DEFAULT_MAX_POINTS
        self.default_show_grid = True
        self.default_plot_size = 2  # grid cells, both width and height - stays square
        self.gate_colours = {}  # gate_name -> (r,g,b,a) 0..1, shared across all tiles (request #3)

        # queued cross-tab notifications - see module docstring's
        # "core-code landmine" note. Flushed by PluginWidget once the user
        # navigates to a real cytometry-data tab.
        self.dirty_axis_transformed = set()
        self.dirty_axes_reset = set()
        # same landmine, different victim: gate_membership recalculation.
        # Flushed by PluginWidget once the user navigates to a tab where
        # data_for_cytometry_plots points at the unmixed dict again - see
        # UNMIXED_DATA_TAB_NAMES.
        self.needs_gate_recalc = False

    def set_toolbar(self, toolbar):
        self.toolbar = toolbar

    def set_context(self, data_for_cytometry_plots, is_dark):
        self.data_for_cytometry_plots = data_for_cytometry_plots
        self.is_dark = is_dark
        self.refresh_gate_colours()

    def mark_channel_transformed(self, channel):
        self.dirty_axis_transformed.add(channel)

    def mark_channels_reset(self, channels):
        self.dirty_axes_reset.update(channels)

    def set_max_points(self, value):
        self.max_points = value
        for tile in self.tiles:
            tile.set_max_points(value)

    def set_default_show_grid(self, checked):
        self.default_show_grid = checked

    def set_default_plot_size(self, value):
        self.default_plot_size = value

    # -- selection -------------------------------------------------------------
    def select_plot(self, widget):
        if widget != self.selected_tile:
            self.deselect_plot()
            widget.setFrameShape(QtWidgets.QFrame.Shape.Box)
            widget.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
            widget.setLineWidth(1)
            self.selected_tile = widget
            if self.toolbar is not None:
                self.toolbar.update_button_state(True)

    def deselect_plot(self):
        if self.selected_tile is not None:
            self.selected_tile.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            self.selected_tile = None
            if self.toolbar is not None:
                self.toolbar.update_button_state(False)

    # -- pop in/out -----------------------------------------------------
    def open_plot_in_modal(self, tile):
        self.debounce_timer.stop()
        self.debounce_timer.blockSignals(True)

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Pop-out 3D plot")
        dialog.setModal(True)
        dialog.setMinimumSize(800, 800)

        layout = QtWidgets.QVBoxLayout(dialog)
        tile.gl_view.in_modal = True
        layout.addWidget(tile)
        QtCore.QTimer.singleShot(0, tile.rebuild_gl_items)

        close_btn = QtWidgets.QPushButton("Pop back in")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec()

        tile.gl_view.in_modal = False
        w, h = self._tile_wh(tile)
        self.place_tile(tile, w, h)
        self.init_grid()
        QtCore.QTimer.singleShot(0, tile.rebuild_gl_items)
        self.debounce_timer.blockSignals(False)

    # -- add / delete / move ---------------------------------------------
    def show_new_plot_widget(self):
        if self._placeholder is not None:
            return
        self._placeholder = NewPlot3DTile(self, parent=self.container)
        self.place_tile(self._placeholder, 1, 1)
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum() + 1000)

    def add_new_plot(self, plot_spec):
        new_tile = Plot3DPlotWidget(
            self.bus, self, self.controller, self.data_for_cytometry_plots, plot_spec,
            max_points=self.max_points, is_dark=self.is_dark, parent=self.container,
        )
        self.plot_specs.append(plot_spec)
        self.tiles.append(new_tile)
        if self._placeholder is not None:
            self._placeholder.deleteLater()
            self._placeholder = None
        self.select_plot(new_tile)
        self.debounce_timer.start(300)

    def delete_current_plot(self):
        if self.selected_tile is None:
            return
        n = self.tiles.index(self.selected_tile)
        self.plot_specs.pop(n)
        tile = self.tiles.pop(n)
        tile.deleteLater()
        self.selected_tile = None
        if self.toolbar is not None:
            self.toolbar.update_button_state(False)
        self.debounce_timer.start(300)

    def move_tile(self, destination):
        if self.selected_tile is None:
            return
        n = self.tiles.index(self.selected_tile)
        N = len(self.tiles) - 1
        scrollbar = self.verticalScrollBar()
        if destination == 'start':
            m = 0
            scrollbar.setValue(scrollbar.minimum())
        elif destination == 'left':
            m = max(n - 1, 0)
        elif destination == 'right':
            m = min(n + 1, N)
        else:  # 'end'
            m = N
            scrollbar.setValue(scrollbar.maximum())

        if destination in ('left', 'right'):
            self.plot_specs[n], self.plot_specs[m] = self.plot_specs[m], self.plot_specs[n]
            self.tiles[n], self.tiles[m] = self.tiles[m], self.tiles[n]
        else:
            spec = self.plot_specs.pop(n)
            self.plot_specs.insert(m, spec)
            tile = self.tiles.pop(n)
            self.tiles.insert(m, tile)

        self.debounce_timer.start(300)

    # -- gate-list / data refresh ------------------------------------------
    def refresh_gate_options(self):
        self.refresh_gate_colours()
        for tile in self.tiles:
            tile.refresh_gate_options()

    def refresh_gate_colours(self):
        """Rebuild the shared gate->colour palette (request #3) from the
        current gating hierarchy: keeps any colour Oliver has customised for
        a gate that still exists, fills in a palette default for any new
        gate, and drops entries for gates that no longer exist."""
        gating = self.data_for_cytometry_plots.get('gating') if self.data_for_cytometry_plots else None
        gate_names = selectable_gate_names(gating)
        defaults = default_gate_colours(gate_names)
        self.gate_colours = {name: self.gate_colours.get(name, defaults[name]) for name in gate_names}

    def apply_gate_colour_change(self, gate_name, rgba):
        self.gate_colours[gate_name] = rgba
        self._recolour_gate_mode_tiles()

    def reset_gate_colours(self):
        gating = self.data_for_cytometry_plots.get('gating') if self.data_for_cytometry_plots else None
        self.gate_colours = default_gate_colours(selectable_gate_names(gating))
        self._recolour_gate_mode_tiles()

    def _recolour_gate_mode_tiles(self):
        for tile in self.tiles:
            if tile.plot.get('colour_mode', 'Density') != 'Density':
                tile.rebuild_colors()
                tile._refresh_scatter()

    def open_gate_colour_editor(self):
        dialog = GateColourEditorDialog(self, parent=self)
        dialog.exec()

    def refresh_after_data_change(self):
        for tile in self.tiles:
            tile.refresh_after_data_change()

    def refresh_for_axis_change(self, channels):
        for tile in self.tiles:
            if any(ch in tile.channels for ch in channels):
                tile.refresh_after_data_change()

    def clear_workspace(self):
        self.deselect_plot()
        if self._placeholder is not None:
            self._placeholder.deleteLater()
            self._placeholder = None
        for tile in self.tiles:
            tile.deleteLater()
        self.tiles = []
        self.plot_specs = []
        self.dirty_axis_transformed.clear()
        self.dirty_axes_reset.clear()
        self.row = 0
        self.last_row = 0
        self.occupied = []

    # -- bin-packing layout (ported from CytometryGridWidget) ------------------
    def _tile_wh(self, tile):
        if isinstance(tile, Plot3DPlotWidget):
            return min(tile.plot.get('width', 1), self.n_columns or 1), tile.plot.get('height', 1)
        return 1, 1

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        if event.oldSize().width() != event.size().width():
            self.debounce_timer.start(300)

    def init_grid(self):
        if not self.tiles and self._placeholder is None:
            return

        self.n_columns = max(self.width() // TILE_WIDTH_TARGET, 1)
        self.n_columns = min(max(self.n_columns, 1), 10)
        self.tile_real_width = (self.width() - 45) // self.n_columns
        for n in range(self.n_columns):
            self.layout.setColumnMinimumWidth(n, self.tile_real_width)

        self.occupied = []
        self.row = 0
        self.last_row = 0

        for tile in self.tiles:
            w, h = self._tile_wh(tile)
            self.place_tile(tile, w, h)

        if self._placeholder is not None:
            self.place_tile(self._placeholder, 1, 1)

    def fits(self, row, col, w, h):
        for r in range(row, row + h):
            for c in range(col, col + w):
                if c >= self.n_columns or (r, c) in self.occupied:
                    return False
        return True

    def occupy(self, row, col, w, h):
        for r in range(row, row + h):
            for c in range(col, col + w):
                self.occupied.append((r, c))

    def place_tile(self, tile, w, h):
        if self.tile_real_width is None:
            self.tile_real_width = max(self.width() // max(w, 1), 200)
        if self.n_columns is None:
            self.n_columns = max(self.width() // TILE_WIDTH_TARGET, 1)

        tile.setMinimumSize(self.tile_real_width * w, self.tile_real_width * h)

        placed = False
        while not placed:
            for col in range(self.n_columns):
                if not self.fits(self.row, col, w, h):
                    continue
                self.layout.addWidget(tile, self.row, col, h, w)
                self.occupy(self.row, col, w, h)
                self.last_row = max(self.last_row, self.row + h)
                placed = True
                break
            if not placed:
                self.row += 1
        self.set_last_row_stretch()

    def set_last_row_stretch(self):
        for row in range(self.last_row):
            self.layout.setRowStretch(row, 0)
        self.layout.setRowStretch(self.last_row, 1)


# --------------------------------------------------------------------------
# PluginWidget + plugin_name - the
# top-level tab. Required attributes per the Honeychrome plugin contract
# (data_processing_example_plugin_tab.py's docstring): plugin_name (str),
# PluginWidget(bus=..., controller=...).
# --------------------------------------------------------------------------
class PluginWidget(QtWidgets.QWidget):
    """
    Top-level 3D Plots plugin tab.

    Disabled (mirrors data_processing_example_plugin_tab.py's existing
    convention) until unmixing has been run and a sample with unmixed event
    data is loaded. The workspace is never persisted to the .kit file and
    starts empty every session  - it's rebuilt by the user via
    the toolbar's "Add Plot" button each time.

    See the module docstring's "core-code landmine" note for why
    axisTransformed/axesReset emissions are queued rather than fired
    directly from tile interactions.
    """

    def __init__(self, bus=None, controller=None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.controller = controller

        app = QtWidgets.QApplication.instance()
        palette = app.palette() if app is not None else QtGui.QPalette()
        base_color = palette.color(QtGui.QPalette.ColorRole.Base)
        self.is_dark = base_color.value() < 128

        self.label_disabled = QtWidgets.QLabel(
            f'{plugin_name}: unmixed data not available. Set up the spectral model first.'
        )

        self.grid = Plot3DGridWidget(self.bus, self.controller, parent=self)
        self.toolbar = Plot3DToolbar(self.grid, parent=self)
        self.grid.set_toolbar(self.toolbar)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label_disabled)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.grid)

        self.toolbar.setVisible(False)
        self.grid.setVisible(False)
        self._activated_once = False

        if self.bus is not None:
            self.bus.modeChangeRequested.connect(self._on_mode_changed)
            self.bus.loadSampleRequested.connect(self._on_load_sample_requested)
            self.bus.loadExpRequested.connect(self._on_load_exp_requested)
            self.bus.histsStatsRecalculated.connect(self._on_hists_stats_recalculated)
            self.bus.axisTransformed.connect(self._on_axis_transformed)
            self.bus.axesReset.connect(self._on_axes_reset)
            self.bus.changedGatingHierarchy.connect(self._on_gating_hierarchy_changed)
            self.bus.updateSourceChildGates.connect(self._on_gating_hierarchy_changed)
        else:
            logger.warning('PluginWidget (3D Plots): bus not connected')

    # -- lifecycle ------------------------------------------------
    def _on_mode_changed(self, mode):
        # deferred so controller.set_mode() (connected earlier, in view.py)
        # has definitely already run by the time this fires, regardless of
        # connection order between this plugin and the core view.
        QtCore.QTimer.singleShot(0, lambda: self._handle_mode_changed(mode))

    def _handle_mode_changed(self, mode):
        if mode == plugin_name:
            self._activate()
        elif mode in DATA_TAB_NAMES:
            self._flush_dirty_transforms()
            if mode in UNMIXED_DATA_TAB_NAMES:
                self._flush_gate_recalc()

    def _activate(self):
        unmixed = self.controller.data_for_cytometry_plots_unmixed
        unmixing_done = self.controller.experiment.process.get('unmixing_matrix') is not None
        has_event_data = unmixed.get('event_data') is not None

        if not (unmixing_done and has_event_data):
            self.label_disabled.setVisible(True)
            self.toolbar.setVisible(False)
            self.grid.setVisible(False)
            return

        self.label_disabled.setVisible(False)
        self.toolbar.setVisible(True)
        self.grid.setVisible(True)
        self.grid.set_context(unmixed, self.is_dark)
        self._check_gate_recalc_needed()

        if self._activated_once:
            self.grid.refresh_after_data_change()
        self._activated_once = True

    def _flush_dirty_transforms(self):
        # see module docstring's "core-code landmine" note - this is the
        # only place axisTransformed/axesReset are actually emitted.
        for channel in self.grid.dirty_axis_transformed:
            self.bus.axisTransformed.emit(channel)
        self.grid.dirty_axis_transformed.clear()

        if self.grid.dirty_axes_reset:
            self.bus.axesReset.emit(list(self.grid.dirty_axes_reset))
            self.grid.dirty_axes_reset.clear()

    def _check_gate_recalc_needed(self):
        # 'root' missing means calc_hists_and_stats() has never
        # successfully populated gate_membership for this dict at all -
        # the same landmine as axisTransformed/axesReset, just for gating.
        unmixed = self.controller.data_for_cytometry_plots_unmixed
        gate_membership = unmixed.get('gate_membership') or {}
        if 'root' not in gate_membership:
            self.grid.needs_gate_recalc = True

    def _flush_gate_recalc(self):
        # only called when UNMIXED_DATA_TAB_NAMES confirms
        # data_for_cytometry_plots is the unmixed dict again (see
        # _handle_mode_changed) - safe to call the controller's own method
        # directly here, same as _flush_dirty_transforms above.
        if self.grid.needs_gate_recalc:
            self.grid.needs_gate_recalc = False
            self.controller.calc_hists_and_stats()

    def _on_load_sample_requested(self, sample_path):
        QtCore.QTimer.singleShot(0, self._refresh_if_active)

    def _on_load_exp_requested(self, file_path):
        QtCore.QTimer.singleShot(0, self.grid.clear_workspace)

    def _refresh_if_active(self):
        if self.grid.isVisible():
            self._sync_unmixed_data_for_plugin_tab()
            self._check_gate_recalc_needed()
            self.grid.refresh_after_data_change()

    def _sync_unmixed_data_for_plugin_tab(self):
        # core-code landmine (see module docstring): controller.set_mode()'s
        # catch-all for plugin tabs leaves controller.data_for_cytometry_plots
        # None, so load_sample()'s own initialise_data_for_cytometry_plots()
        # call silently no-ops while this tab is active - event_data,
        # gate_membership and stats in data_for_cytometry_plots_unmixed never
        # get refreshed for the just-loaded sample. Same approach as
        # Plot3DPlotWidget.reset_axes_transforms(): briefly point the
        # controller at the real dict so its own public method does the
        # real work, then restore the None this tab needs.
        unmixed = self.controller.data_for_cytometry_plots_unmixed
        self.controller.data_for_cytometry_plots = unmixed
        self.controller.initialise_data_for_cytometry_plots()
        self.controller.data_for_cytometry_plots = None

    # -- data-change / cross-talk signals -------------------------------
    @QtCore.Slot(str, list)
    def _on_hists_stats_recalculated(self, mode, indices_plots_to_recalculate=None):
        if mode == 'unmixed' and self.grid.isVisible():
            self.grid.refresh_after_data_change()

    @QtCore.Slot(str)
    def _on_axis_transformed(self, channel):
        if self.grid.isVisible():
            self.grid.refresh_for_axis_change([channel])

    @QtCore.Slot(list)
    def _on_axes_reset(self, channels):
        if self.grid.isVisible():
            self.grid.refresh_for_axis_change(channels)

    def _on_gating_hierarchy_changed(self, *args):
        if self.grid.isVisible():
            self.grid.refresh_gate_options()


# --------------------------------------------------------------------------
# Standalone demo harness - loads a real experiment via Controller (same
# convention as nxn_grid.py / cytometry_grid_widget.py's __main__ blocks)
# and simulates switching to this plugin's tab. Needs the honeychrome
# package importable. Edit experiment_path below to point at a real,
# already-unmixed .kit experiment with at least one sample loaded.
# --------------------------------------------------------------------------
def main():
    import sys

    app = QtWidgets.QApplication(sys.argv)

    from honeychrome.controller import Controller
    from honeychrome.view_components.event_bus import EventBus

    bus = EventBus()
    kc = Controller()
    kc.bus = bus
    bus.modeChangeRequested.connect(kc.set_mode)

    base_directory = Path.home() / 'spectral_cytometry'
    experiment_name = base_directory / '20240620 Spectral Symposium-poor cell unmixed'
    experiment_path = experiment_name.with_suffix('.kit')
    kc.load_experiment(experiment_path)
    kc.set_mode('Unmixed Data')

    widget = PluginWidget(bus=bus, controller=kc)
    bus.modeChangeRequested.emit(plugin_name)

    window = QtWidgets.QMainWindow()
    window.setWindowTitle(f"{plugin_name} - standalone test harness")
    window.resize(1100, 850)
    window.setCentralWidget(widget)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

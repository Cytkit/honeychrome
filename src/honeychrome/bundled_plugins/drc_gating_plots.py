"""
drc_gating_plots.py — gating hierarchy plots of the Gating confirmation page
============================================================================
Companion module to ``dr_clustering_tab.py`` (filename intentionally does
NOT end in ``_tab.py``, so it is not picked up as a separate plugin tab).

Draws a gating strategy found by drc_gate_finder the way Unmixed Data draws
its plots: Honeychrome's ``CytometryPlotWidget`` on a private event bus,
square, with the app's density colour map and theme, on display transforms
whose axes can be dragged and switched (linear / logicle / log) exactly as
in Unmixed Data. Each step of the strategy is one plot of the population
left by the steps before it, with that step's gate drawn as an Unmixed Data
gate labelled with its % of the parent. The cluster's own events are drawn
over the density in the cluster's colour.

The plots are view only: gates cannot be moved, renamed or deleted, and the
channels and source gates are fixed by the strategy. Gate labels sit in
the upper-right corner of their gate and can be dragged elsewhere (double-
click one to put it back); where they were put is kept in
GatingPlotSet.label_positions, on the display scale.

Events are a weighted draw (with replacement) from the run's stored
per-cluster sample, so each cluster appears in proportion to its size in
the clustered data, and the gates' percentages are those of all clustered
cells to within sampling noise. Gate membership is decided on the run's own
transform scale, where the gates were found, so changing the display axes
moves the drawing, never which cells a gate holds.

``make_hierarchy_figure`` draws the same plots with matplotlib, for the
report and image export.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtWidgets import QGridLayout, QMenu, QWidget

import honeychrome.settings as hc_settings

import ag_report
import drc_gate_finder as gf
from transforms_panel import TransformSet, transform_from_params, transform_params

MODE = 'gating_confirmation'
SAMPLE_ID = 'gating_confirmation'
LABEL_PLACER = 'gating_confirmation_labels'
PLOT_EVENTS = 100_000
OVERLAY_POINTS = 20_000
OVERLAY_ALPHA = 170
GATE_COLOUR = '#00ff00'
GATE_LABEL_FILL = (0.0, 1.0, 0.0, 0.5)
HIST1D_FILL = (100 / 255, 100 / 255, 250 / 255, 150 / 255)
FIGURE_COLUMNS = 4
FIGURE_PLOT_INCHES = 3.0
FIGURE_STYLE = {
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 9, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'axes.edgecolor': 'black', 'axes.linewidth': 0.8,
    'axes.grid': False, 'axes.facecolor': 'white', 'figure.facecolor': 'white',
    'xtick.color': 'black', 'ytick.color': 'black', 'axes.labelcolor': 'black',
    'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.bottom': True, 'ytick.left': True,
}


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def run_transform(params: dict):
    """A Honeychrome Transform from a run's channel_transform_params entry
    ({'T', 'W', 'M', 'A', 'id', 'limits'})."""
    return transform_from_params({
        'id': int(params.get('id', 1)),
        'scale_t': params.get('T'), 'logicle_w': params.get('W'),
        'logicle_m': params.get('M'), 'logicle_a': params.get('A'),
        'limits': params.get('limits') or [0, 1],
    })


def display_transforms(controller, state, run: dict, saved: dict | None) -> TransformSet:
    """The axes of a run's plots: *saved* ({channel: transform parameters},
    the page's own adjustments) over Honeycluster's transforms (the
    Transforms tab), over the experiment's; channels known to none of them
    take the run's own transform."""
    base = TransformSet()
    base.load(getattr(state, 'saved_transforms', None) or {},
              getattr(controller, 'unmixed_transformations', None) or {})
    for ch, params in (run.get('transform_params') or {}).items():
        if ch not in base.objects and params:
            try:
                base.objects[ch] = run_transform(params)
            except Exception:
                continue
    out = TransformSet()
    out.load(saved or {}, base.objects)
    out.base = base.objects
    return out


def reset_channels(transform_set: TransformSet, channels) -> None:
    """Set *channels* of *transform_set* back to the transforms it was
    loaded over (Honeycluster's), in place."""
    base = getattr(transform_set, 'base', None) or {}
    for ch in channels:
        if ch in base and ch in transform_set.objects:
            transform_set.set_channel(ch, transform_params(base[ch]))


def to_display(tr, value: float | None) -> float | None:
    """A raw intensity on the display scale of *tr*; None stays None."""
    if value is None:
        return None
    return float(ag_report._apply(tr, np.array([float(value)]))[0])


# ---------------------------------------------------------------------------
# The hierarchy as Unmixed Data plot data
# ---------------------------------------------------------------------------

def step_pairs(f: dict, compromise: bool) -> list[tuple[str, str | None]]:
    """(x channel, y channel or None) of each plot: the strategy's steps, or
    the compromise's markers in pairs."""
    cuts = f['compromise'] if compromise else f['strategy']
    if not compromise and f.get('steps'):
        return [(s['x'], s['y'] or None) for s in f['steps']]
    chans = list(cuts)
    return [(chans[i], chans[i + 1] if i + 1 < len(chans) else None)
            for i in range(0, len(chans), 2)]


def _cut_label(label: str, lo, hi) -> str:
    if lo is not None and hi is not None:
        return f'{label} int'
    return f'{label}{"+" if lo is not None else "−"}'


def _antigen(label: str) -> str:
    """The antigen of an 'antigen - fluorochrome' label."""
    return label.split(' - ')[0].strip() or label


def gate_names(pairs: list, cuts: dict, labels: dict) -> list[str]:
    """A unique gate name per plot from its markers' antigens and cuts, as
    a person would name it: '+' above a cut, '−' below it, 'int' between
    two."""
    names, used = [], set()
    for pair in pairs:
        parts = [_cut_label(_antigen(labels.get(ch, ch)), *cuts[ch]) for ch in pair if ch]
        name = ' '.join(parts).replace('/', '-')
        unique, n = name, 2
        while unique in used or unique == 'root':
            unique = f'{name} ({n})'
            n += 1
        used.add(unique)
        names.append(unique)
    return names


class GatingPlotSet:
    """``data_for_cytometry_plots`` of one gating strategy.

    events     : drc_gating_confirmation.RunEvents (needs .raw).
    f          : a saved finding (finding_to_dict).
    clusters   : the target clusters, drawn over the density.
    transforms : {channel: Transform}, edited in place by the plots' axes.
    raw_of     : callable(channel, value on the run's scale) -> raw value.
    labels     : {channel: marker label}.
    """

    def __init__(self, events, f: dict, clusters: list, compromise: bool, transforms: dict,
                 raw_of, labels: dict, seed: int = 0):
        from flowkit import Dimension, GatingStrategy, gates

        self.transforms = transforms
        cuts = f['compromise'] if compromise else f['strategy']
        self.pairs = step_pairs(f, compromise)
        self.names = gate_names(self.pairs, cuts, labels)
        self.raw_cuts = {ch: (raw_of(ch, lo), raw_of(ch, hi)) for ch, (lo, hi) in cuts.items()}
        channels = list(events.channels)
        self.channels = channels

        rng = np.random.default_rng(seed)
        n = len(events.labels)
        size = int(min(PLOT_EVENTS, max(n, round(float(events.weights.sum())))))
        pick = rng.choice(n, size=size, replace=True, p=events.weights / events.weights.sum())
        raw = np.ascontiguousarray(events.raw[pick], dtype=np.float64)
        self.focus = np.isin(events.labels[pick], list(clusters))

        strategy = gf.strategy_from_dict(cuts, channels)
        X = events.X[pick]
        membership = {'root': np.ones(size, dtype=bool)}
        alive = membership['root']
        gating = GatingStrategy()
        gate_path = ('root',)
        plots = []
        for (cx, cy), name in zip(self.pairs, self.names):
            chans = [cx] + ([cy] if cy else [])
            alive = alive & strategy.inside(X, [channels.index(ch) for ch in chans])
            membership[name] = alive
            dims = [Dimension(ch, compensation_ref='uncompensated',
                              transformation_ref=ch if transforms[ch].xform else None,
                              range_min=0.0, range_max=1.0) for ch in chans]
            gating.add_gate(gates.RectangleGate(name, dimensions=dims), gate_path=gate_path)
            plot = {'type': 'hist2d' if cy else 'hist1d', 'channel_x': cx,
                    'source_gate': gate_path[-1], 'child_gates': [name]}
            if cy:
                plot['channel_y'] = cy
            plots.append(plot)
            gate_path = gate_path + (name,)

        self.data = {
            'pnn': channels,
            'pnn_labels': {ch: labels.get(ch, ch) for ch in channels},
            'fluoro_indices': [],
            'lookup_tables': {},
            'event_data': raw,
            'transformations': transforms,
            'statistics': {},
            'gating': gating,
            'sample_id': SAMPLE_ID,
            'plots': plots,
            'histograms': [],
            'gate_membership': membership,
            'cytometer_db_col': None,
            LABEL_PLACER: self,
        }
        self.label_positions: dict = {}
        self.overlay: list = []
        self.refresh()

    def gate_dimensions(self, name: str) -> dict:
        """{channel: (low, high)} of a gate on the display scale; an open
        side is the axis limit."""
        k = self.names.index(name)
        out = {}
        for ch in self.pairs[k]:
            if not ch:
                continue
            tr = self.transforms[ch]
            lo_raw, hi_raw = self.raw_cuts[ch]
            lim = [float(v) for v in tr.limits]
            lo, hi = to_display(tr, lo_raw), to_display(tr, hi_raw)
            out[ch] = (lim[0] if lo is None else lo, lim[1] if hi is None else hi)
        return out

    def refresh(self):
        """Gates, statistics, histograms and overlay on the current display
        transforms."""
        from honeychrome.controller_components.functions import calc_hists, calc_stats

        gating = self.data['gating']
        for name in self.names:
            gate = gating.get_gate(name)
            dims = self.gate_dimensions(name)
            for dim in gate.dimensions:
                dim.min, dim.max = dims[dim.id]
        self.data['statistics'] = calc_stats(self.data, initialise=True)
        self.data['histograms'] = calc_hists(self.data,
                                             density_cutoff=hc_settings.density_cutoff_retrieved)
        self.overlay = self._overlay()

    def _overlay(self, seed: int = 0) -> list:
        """Per plot, the cluster's events in its source population on the
        display scale: ('points', x, y) or ('counts', counts on the plot's
        bins); None without any."""
        from honeychrome.controller_components.functions import calc_hist1d

        data = self.data
        raw, pnn = data['event_data'], data['pnn']
        rng = np.random.default_rng(seed)
        out = []
        for plot in data['plots']:
            mask = data['gate_membership'][plot['source_gate']] & self.focus
            if not mask.any():
                out.append(None)
                continue
            tr_x = self.transforms[plot['channel_x']]
            if plot['type'] == 'hist1d':
                out.append(('counts', calc_hist1d(raw, mask, pnn.index(plot['channel_x']), tr_x)))
                continue
            rows = np.flatnonzero(mask)
            if len(rows) > OVERLAY_POINTS:
                rows = rng.choice(rows, OVERLAY_POINTS, replace=False)
            tr_y = self.transforms[plot['channel_y']]
            out.append(('points',
                        ag_report._apply(tr_x, raw[rows, pnn.index(plot['channel_x'])]),
                        ag_report._apply(tr_y, raw[rows, pnn.index(plot['channel_y'])])))
        return out

    def default_label_position(self, name: str) -> tuple[float, float]:
        """The upper-right corner of a gate on the display scale (for a
        histogram gate, its right edge at the height of the tallest bar),
        kept inside the axes."""
        k = self.names.index(name)
        plot = self.data['plots'][k]
        dims = self.gate_dimensions(name)
        tr_x = self.transforms[plot['channel_x']]
        x_lim = sorted(float(v) for v in tr_x.limits)
        x = min(max(dims[plot['channel_x']][1], x_lim[0]), x_lim[1])
        if plot['type'] == 'hist2d':
            y_lim = sorted(float(v) for v in self.transforms[plot['channel_y']].limits)
            y = min(max(dims[plot['channel_y']][1], y_lim[0]), y_lim[1])
        else:
            hist = self.data['histograms'][k]
            y = float(np.max(hist)) if hist is not None and np.size(hist) else 1.0
        return float(x), float(y)

    def label_position(self, name: str) -> tuple[float, float]:
        """Where a gate's label has its top-right corner: where it was
        dragged to, else the gate's upper-right corner."""
        placed = self.label_positions.get(name)
        if placed is not None:
            return float(placed[0]), float(placed[1])
        return self.default_label_position(name)

    def percent_of_parent(self, name: str) -> float:
        return 100.0 * float(self.data['statistics'].get(name, {}).get('p_gate_parent', np.nan))


# ---------------------------------------------------------------------------
# On screen: CytometryPlotWidget, view only
# ---------------------------------------------------------------------------

_PlotClass = None


def _plot_class():
    """``CytometryPlotWidget`` whose channels, source gate and gates are
    fixed; its axes can be dragged and their transform changed."""
    global _PlotClass
    if _PlotClass is not None:
        return _PlotClass
    from honeychrome.view_components.cytometry_plot_widget import (
        CytometryPlotWidget, copy_widget_to_clipboard,
    )

    class GatingConfirmationPlot(CytometryPlotWidget):
        def select_plot_on_parent_grid(self):
            pass

        def mouseDoubleClickEvent(self, event):
            event.accept()

        def set_up_plot(self, item=None, parent=None):
            pass

        def set_source_gate(self, n, parent):
            pass

        def new_plot_on_gate(self, source_gate):
            pass

        def configure_title(self):
            super().configure_title()
            self.plot_title.leftClickMenuItems = []

        def configure_axes(self):
            super().configure_axes()
            self.label_x.leftClickMenuItems = []
            self.label_y.leftClickMenuItems = []

        def right_click_menu(self, ev):
            menu = QMenu()
            menu.addAction('Fit Axes to Data', self.fit_axes_to_data)
            menu.addAction('Reset Axes', self.reset_axes_transforms)
            menu.addAction('Export Image', self.export_image)
            menu.addAction('Copy to Clipboard', lambda: copy_widget_to_clipboard(self))
            menu.exec(ev.screenPos().toPoint())

        def configure_rois(self, mode, index):
            super().configure_rois(mode, index)
            if mode == self.mode and index == self.n_in_plot_sequence:
                for roi in self.rois:
                    _freeze(roi)
                    self._place_label(roi)

        def _place_label(self, roi):
            """Put a gate's label at its position (top-right corner of the
            text) and let it be dragged; a double-click puts it back in the
            gate's upper-right corner."""
            label = getattr(roi, 'label', None) if roi is not None else None
            placer = self.data_for_cytometry_plots.get(LABEL_PLACER)
            if label is None or placer is None:
                return
            label.setAnchor((1, 0))
            label.setPos(*placer.label_position(label.gate_name))
            label.setFlag(label.GraphicsItemFlag.ItemIsMovable, True)
            label.setAcceptedMouseButtons(Qt.LeftButton)

            def reset(event, label=label):
                event.accept()
                placer.label_positions.pop(label.gate_name, None)
                label.setPos(*placer.default_label_position(label.gate_name))
                self._clamp_label(label)
                label.put_back = True
                if self.bus is not None:
                    self.bus.updateChildGateLabelOffset.emit(label.gate_name, (0.0, 0.0))

            label.mouseDoubleClickEvent = reset
            if not getattr(self, '_clamp_connected', False):
                self.vb.sigResized.connect(lambda *_a: self._clamp_labels())
                self._clamp_connected = True
            QTimer.singleShot(0, self, self._clamp_labels)

        def update_axes_stats_hist(self, mode, indices_plots_to_recalculate=None):
            super().update_axes_stats_hist(mode, indices_plots_to_recalculate)
            if mode == self.mode:
                QTimer.singleShot(0, self, self._clamp_labels)

        def _clamp_labels(self):
            for roi in self.rois:
                label = getattr(roi, 'label', None) if roi is not None else None
                if label is not None:
                    self._clamp_label(label)

        def _clamp_label(self, label):
            """Keep a label in its default place inside the plot: a label
            wider than the room left of its gate's corner moves right, one
            running below the plot moves up. Dragged labels stay put."""
            placer = self.data_for_cytometry_plots.get(LABEL_PLACER)
            if placer is None or label.gate_name in placer.label_positions or \
                    label.scene() is None:
                return
            label.setPos(*placer.default_label_position(label.gate_name))
            view = self.vb.sceneBoundingRect()
            box = label.sceneBoundingRect()
            if view.width() <= 0 or view.height() <= 0:
                return
            dx = max(0.0, view.left() - box.left())
            dy = min(0.0, view.bottom() - box.bottom())
            if dx == 0.0 and dy == 0.0:
                return
            p0 = self.vb.mapSceneToView(QPointF(0.0, 0.0))
            p1 = self.vb.mapSceneToView(QPointF(dx, dy))
            label.setPos(label.pos().x() + p1.x() - p0.x(), label.pos().y() + p1.y() - p0.y())

        def refresh_custom_rois(self, mode):
            pass

        def _restore(self, *args):
            QTimer.singleShot(0, lambda: self.configure_rois(self.mode, self.n_in_plot_sequence))

        update_polygon = update_rectangle = update_ellipse = update_range = update_quad = _restore
        remove_gate_and_roi = _restore

    _PlotClass = GatingConfirmationPlot
    return GatingConfirmationPlot


def _freeze(roi):
    """Stop a gate shape (and its handles, lines and label) taking mouse
    input; GatingConfirmationPlot then frees the label to be dragged."""
    if roi is None:
        return
    items = [roi] + [getattr(roi, a, None) for a in ('label', 'vx', 'vy', 'v1', 'v2')]
    seen = set()
    while items:
        item = items.pop()
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        if hasattr(item, 'translatable'):
            item.translatable = False
        if hasattr(item, 'setMovable'):
            item.setMovable(False)
        if hasattr(item, 'setAcceptedMouseButtons'):
            item.setAcceptedMouseButtons(Qt.NoButton)
        if hasattr(item, 'setAcceptHoverEvents'):
            item.setAcceptHoverEvents(False)
        if hasattr(item, 'childItems'):
            items.extend(item.childItems())


class GatingPlotRow(QWidget):
    """The plots of a GatingPlotSet, left to right (or wrapped after
    *columns*). Emits transformsEdited after an axis changes."""

    transformsEdited = Signal()
    labelsMoved = Signal()

    def __init__(self, columns: int | None = None, parent=None):
        super().__init__(parent)
        self.columns = columns
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(6)
        self.grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.plot_set: GatingPlotSet | None = None
        self.transform_set: TransformSet | None = None
        self.colour = '#d62728'
        self._bus = None
        self._plots: list = []
        self._overlay_items: list = []
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._recalculate)

    @staticmethod
    def plot_size() -> int:
        return int(getattr(hc_settings, 'cytometry_plot_width_target_retrieved', 220) or 220)

    def clear(self):
        for plot in self._plots:
            plot.setParent(None)
            plot.deleteLater()
        self._plots, self._overlay_items = [], []
        self._bus = None
        self.plot_set = None

    def show_set(self, plot_set: GatingPlotSet, colour: str, transform_set: TransformSet):
        """Draw *plot_set*, whose transforms are those of *transform_set*."""
        from honeychrome.view_components.event_bus import EventBus

        self.clear()
        self.plot_set, self.colour, self.transform_set = plot_set, colour, transform_set
        self._bus = EventBus()
        self._bus.axisTransformed.connect(lambda _ch: self._timer.start())
        self._bus.axesReset.connect(self._reset_axes)
        self._bus.updateChildGateLabelOffset.connect(self._label_moved)
        plot_class = _plot_class()
        size = self.plot_size()
        for i, plot in enumerate(plot_set.data['plots']):
            widget = plot_class(bus=self._bus, mode=MODE, n_in_plot_sequence=i, plot=plot,
                                data_for_cytometry_plots=plot_set.data)
            widget.setFixedSize(size, size)
            if self.columns:
                self.grid.addWidget(widget, i // self.columns, i % self.columns)
            else:
                self.grid.addWidget(widget, 0, i)
            self._plots.append(widget)
        self._bus.histsStatsRecalculated.emit(MODE, list(range(len(self._plots))))
        self._draw_overlay()

    def _label_moved(self, gate_name: str, _offset):
        """A label was dragged, or put back by a double-click: record where
        it now is."""
        if self.plot_set is None:
            return
        for widget in self._plots:
            for roi in widget.rois:
                label = getattr(roi, 'label', None)
                if label is None or label.gate_name != gate_name:
                    continue
                if getattr(label, 'put_back', False):
                    label.put_back = False
                    self.plot_set.label_positions.pop(gate_name, None)
                else:
                    self.plot_set.label_positions[gate_name] = (float(label.pos().x()),
                                                                float(label.pos().y()))
                self.labelsMoved.emit()
                return

    def redraw(self):
        """Redraw after the transforms changed elsewhere."""
        if self.plot_set is not None:
            self._recalculate(emit=False)

    def _draw_overlay(self):
        import pyqtgraph as pg
        from PySide6.QtGui import QColor

        for widget, item in zip(self._plots, self._overlay_items):
            if item is not None:
                widget.vb.removeItem(item)
        self._overlay_items = []
        c = QColor(self.colour)
        rgb = (c.red(), c.green(), c.blue())
        transforms = self.plot_set.transforms
        for widget, extra in zip(self._plots, self.plot_set.overlay):
            item = None
            if extra is not None and extra[0] == 'points':
                item = pg.ScatterPlotItem(extra[1], extra[2], size=2, pen=None,
                                          brush=pg.mkBrush(*rgb, OVERLAY_ALPHA))
            elif extra is not None:
                step = transforms[widget.plot['channel_x']].step_scale
                item = pg.PlotDataItem(step, extra[1], stepMode='center',
                                       pen=pg.mkPen(rgb, width=1.5))
            if item is not None:
                item.setZValue(30)
                widget.vb.addItem(item)
            self._overlay_items.append(item)

    def _recalculate(self, emit: bool = True):
        if self.plot_set is None or self._bus is None:
            return
        self.plot_set.refresh()
        n = len(self._plots)
        self._bus.histsStatsRecalculated.emit(MODE, list(range(n)))
        for i in range(n):
            self._bus.updateRois.emit(MODE, i)
        self._draw_overlay()
        if emit:
            self.transformsEdited.emit()

    def _reset_axes(self, channels: list):
        """Reset Axes: back to Honeycluster's transform of each channel."""
        if self.plot_set is None or self.transform_set is None:
            return
        reset_channels(self.transform_set, channels)
        self._recalculate()


# ---------------------------------------------------------------------------
# Report and image export: the same plots with matplotlib
# ---------------------------------------------------------------------------

def density_colourmap():
    """The app's density colour map (as CytometryPlotWidget builds it)."""
    import colorcet as cc
    from matplotlib.colors import LinearSegmentedColormap

    try:
        colours = cc.palette[hc_settings.colourmap_name_retrieved]
    except Exception:
        colours = cc.palette['bjy']
    lin = np.linspace(0.0, 1.0, len(colours))
    pos = 0.9 * lin ** 2 + 0.1 * lin
    return LinearSegmentedColormap.from_list('honeychrome_density', list(zip(pos, colours)))


def make_hierarchy_figure(plot_set: GatingPlotSet, colour: str, title: str = ''):
    """The plots of *plot_set* as a canvas-less matplotlib Figure: square
    plots on the display transforms, the Unmixed Data histograms, the
    cluster's events in *colour* and each gate with its % of the parent."""
    import matplotlib

    with matplotlib.rc_context(FIGURE_STYLE):
        return _hierarchy_figure(plot_set, colour, title)


def _hierarchy_figure(plot_set: GatingPlotSet, colour: str, title: str):
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    data = plot_set.data
    plots = data['plots']
    labels = data['pnn_labels']
    transforms = plot_set.transforms
    ncol = min(FIGURE_COLUMNS, max(1, len(plots)))
    nrow = int(np.ceil(len(plots) / ncol))
    size = FIGURE_PLOT_INCHES
    fig = Figure(figsize=(size * ncol + 0.4, size * nrow + (0.45 if title else 0.1)),
                 constrained_layout=True)
    cmap = density_colourmap()
    defaults = []
    for i, plot in enumerate(plots):
        ax = fig.add_subplot(nrow, ncol, i + 1)
        ax.set_box_aspect(1)
        tr_x = transforms[plot['channel_x']]
        x_lim = ag_report._limits(tr_x, np.array([]))
        hist = data['histograms'][i]
        extra = plot_set.overlay[i]
        if plot['type'] == 'hist2d':
            tr_y = transforms[plot['channel_y']]
            y_lim = ag_report._limits(tr_y, np.array([]))
            if hist is not None and np.size(hist):
                ax.imshow(np.ma.masked_less_equal(np.asarray(hist, dtype=float).T, 0),
                          origin='lower', extent=(x_lim[0], x_lim[1], y_lim[0], y_lim[1]),
                          aspect='auto', cmap=cmap, interpolation='nearest')
            if extra is not None:
                ax.scatter(extra[1], extra[2], s=0.4, c=colour, linewidths=0, alpha=0.7,
                           rasterized=True)
            ax.set_ylim(*y_lim)
            ax.set_ylabel(labels.get(plot['channel_y'], plot['channel_y']))
            ag_report._set_ticks(ax, 'y', tr_y)
        else:
            edges = np.asarray(tr_x.step_scale, dtype=float)
            if hist is not None and len(edges) == len(hist) + 1:
                ax.stairs(hist, edges, fill=True, color=HIST1D_FILL)
            if extra is not None and len(edges) == len(extra[1]) + 1:
                ax.stairs(extra[1], edges, color=colour, linewidth=1.2)
            ax.set_ylabel('Count')
            ax.set_ylim(bottom=0)
        ax.set_xlim(*x_lim)
        ax.set_xlabel(labels.get(plot['channel_x'], plot['channel_x']))
        ag_report._set_ticks(ax, 'x', tr_x)
        ax.grid(False)
        ax.set_title(plot['source_gate'], fontsize=10)

        name = plot['child_gates'][0]
        dims = plot_set.gate_dimensions(name)
        x0, x1 = dims[plot['channel_x']]
        if plot['type'] == 'hist2d':
            y0, y1 = dims[plot['channel_y']]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor=GATE_COLOUR, linewidth=1.8, zorder=5))
        else:
            for v in (x0, x1):
                ax.axvline(v, color=GATE_COLOUR, linewidth=1.8, zorder=5)
            ax.axvspan(x0, x1, color=GATE_COLOUR, alpha=0.08, zorder=1)
        lx, ly = plot_set.label_position(name)
        text = ax.text(lx, ly, f'{name}: {plot_set.percent_of_parent(name):.2f}%', fontsize=8,
                fontweight='bold', color='black', ha='right', va='top',
                zorder=6, clip_on=True,
                bbox=dict(boxstyle='square,pad=0.2', facecolor=GATE_LABEL_FILL, edgecolor='black',
                          linewidth=0.6))
        if name not in plot_set.label_positions:
            defaults.append((ax, text))
    if title:
        fig.suptitle(title, fontsize=11)
    _keep_inside(fig, defaults)
    return fig


def _keep_inside(fig, texts: list):
    """Move each (axes, text) label right or up so that it lies inside its
    axes, as the plots on screen do."""
    if not texts:
        return
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = fig.canvas if hasattr(fig.canvas, 'get_renderer') else FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    for ax, text in texts:
        box = text.get_bbox_patch().get_window_extent(renderer)
        area = ax.get_window_extent(renderer)
        dx = max(0.0, area.x0 - box.x0)
        dy = max(0.0, area.y0 - box.y0)
        if dx == 0.0 and dy == 0.0:
            continue
        x, y = ax.transData.transform(text.get_position())
        text.set_position(ax.transData.inverted().transform((x + dx, y + dy)))

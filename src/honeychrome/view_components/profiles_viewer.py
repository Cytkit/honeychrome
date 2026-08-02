from PySide6.QtCore import QPointF, QRectF, Qt, QRect, QSize, QPoint, Slot, QObject, QEvent
from PySide6.QtGui import QWheelEvent, QPen, QColor
from PySide6.QtWidgets import QFrame, QVBoxLayout, QLayout, QWidget, QHBoxLayout, QLabel, QSizePolicy, QCheckBox, QComboBox
import pyqtgraph as pg
import numpy as np
import logging
logger = logging.getLogger(__name__)

from honeychrome.settings import heading_style, line_colors
from honeychrome.view_components.cytometry_plot_components import (
    NoPanViewBox, ZoomAxis, TransparentGraphicsLayoutWidget,
)
from honeychrome.view_components.help_toggle_widget import WheelBlocker
from honeychrome.controller_components.cytometer_whitelist import (
    get_detector_laser_map, LASER_LABEL_COLORS,
)


# --------------------- Flow Layout -------------------------
# (Standard Qt FlowLayout implementation)
class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=6, spacing=6):
        super().__init__(parent)
        self.itemList = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

    def clear(self):
        for i in reversed(range(self.count())):
            item = self.takeAt(i)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def addItem(self, item):
        self.itemList.append(item)

    def count(self):
        return len(self.itemList)

    def itemAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self.doLayout(QRect(0, 0, width, 0), testOnly=True)
        return height

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self.doLayout(rect, testOnly=False)

    def sizeHint(self):
        return QSize(400, 200)

    def doLayout(self, rect, testOnly=False):
        x = rect.x()
        y = rect.y()
        lineHeight = 0

        for item in self.itemList:
            wid = item.widget()
            spaceX = self.spacing()
            spaceY = self.spacing()
            nextX = x + item.sizeHint().width() + spaceX

            if nextX - spaceX > rect.right():
                x = rect.x()
                y = y + lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0

            if not testOnly:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight - rect.y()


# --------------------- Legend Entry -------------------------
class LegendEntry(QWidget):
    """A single legend row: colored square + label."""
    def __init__(self, color, text):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(5)

        colorBox = QLabel()
        colorBox.setFixedSize(14, 14)
        colorBox.setStyleSheet(f"background-color: {color}; border:1px solid #444;")
        layout.addWidget(colorBox)

        nameLabel = QLabel(text)
        layout.addWidget(nameLabel)

        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)




class BottomAxisVerticalTickLabels(pg.AxisItem):
    def __init__(self, angle=90, **kwargs):
        self.angle = angle
        self._label_padding = 15
        self.orientation = 'bottom'
        self.tick_colors: dict = {}        # must be before super().__init__()
        super().__init__(self.orientation, **kwargs)

        # Give extra space by default to prevent clipping
        self.setStyle(tickTextOffset=30, tickLength=5)

    def setTicks(self, ticks):
        extra_offset = int(10 + abs(self.angle) * 0.4)
        self.setStyle(tickTextOffset=extra_offset)
        super().setTicks(ticks)
        self.updateGeometry()
        self.update()

    def generateDrawSpecs(self, p):
        # Temporarily expand boundingRect so pyqtgraph's edge-cull check
        # (br & rect != rect) never drops labels at the first/last detectors.
        # After rotation in drawPicture, labels extend upward into the axis
        # area, not sideways, so no horizontal clipping actually occurs.
        orig_boundingRect = self.boundingRect

        def _expanded_boundingRect():
            br = orig_boundingRect()
            return br.adjusted(-500, 0, 500, 0)

        self.boundingRect = _expanded_boundingRect
        try:
            result = super().generateDrawSpecs(p)
        finally:
            self.boundingRect = orig_boundingRect
        return result
    
    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        super().drawPicture(p, axisSpec, tickSpecs, [])

        tick_colors = getattr(self, 'tick_colors', {})
        default_pen = p.pen()

        p.save()
        try:
            for rect, flags, text in textSpecs:
                p.save()
                try:
                    color = tick_colors.get(text)
                    p.setPen(QPen(color) if color else default_pen)

                    if self.orientation == 'bottom' and self.angle == 90:
                        tick_anchor = QPointF(rect.center().x(), rect.top())
                        p.translate(tick_anchor)
                        p.rotate(-self.angle)
                        label_padding = -15
                        text_rect = QRectF(label_padding, -rect.height() / 2, rect.width(), rect.height())
                        p.drawText(text_rect, int(Qt.AlignLeft | Qt.AlignVCenter), text)
                    elif self.orientation == 'bottom' and self.angle:
                        # Any other angle (e.g. 45 deg): anchor at the tick and let the
                        # label trail away down-left, so labels never overlap each other.
                        tick_anchor = QPointF(rect.center().x(), rect.top())
                        p.translate(tick_anchor)
                        p.rotate(-self.angle)
                        text_rect = QRectF(-rect.width() - 4, -rect.height() / 2,
                                           rect.width(), rect.height())
                        p.drawText(text_rect, int(Qt.AlignRight | Qt.AlignVCenter), text)
                    else:
                        p.drawText(rect, int(flags), text)
                finally:
                    p.restore()
        finally:
            p.restore()

class TransparentPlotWidget(pg.PlotWidget):
    def wheelEvent(self, event: QWheelEvent):
        # We explicitly ignore the event.
        # This tells Qt: "I don't want this, give it to my parent."
        event.ignore()

class ProfilesViewer(QFrame):
    def __init__(self, bus, controller, pen_width=2, parent=None):
        super().__init__(parent)

        # connect
        self.bus = bus
        self.controller = controller
        self.pen_width = pen_width

        if self.bus:
            self.bus.showSelectedProfiles.connect(self.plot_profiles)
            self.bus.spectralControlAdded.connect(self.plot_latest_profile)
            self.bus.showSelectedProfiles.connect(lambda x: self.bus.spilloverSelectedCellChanged.emit(None, None))

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel("Profiles Viewer")
        self.layout.addWidget(self.title)
        self.title.setStyleSheet(heading_style)

        # ---- Flowing Legend ----
        self.legendContainer = QWidget()
        self.legendLayout = FlowLayout(self.legendContainer)
        self.layout.addWidget(self.legendContainer)

        # Create the PyQtGraph widget
        bottom_axis_vertical_tick_labels = BottomAxisVerticalTickLabels()
        self.plot_widget = TransparentPlotWidget(axisItems={'bottom': bottom_axis_vertical_tick_labels})
        self.layout.addWidget(self.plot_widget, stretch=1)
        vb = self.plot_widget.getViewBox()
        vb.setMenuEnabled(False)  # disable right-click menu

        # Configure the plot
        self.plot_widget.setLabel('left', 'Intensity')
        self.plot_widget.setLabel('bottom', 'All Fluorescence')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()

        # Store plot items for potential updates
        self.plot_items = {}

        # ---- Collapsible peak-channel histogram panel ----
        self._hist_toggle = QCheckBox('Show peak-channel event histograms')
        self._hist_toggle.setChecked(False)
        self._hist_toggle.setToolTip(
            'For each selected control, plot a 1-D histogram of its peak channel.\n'
            'Gate boundaries are marked. When cleaned data are available and "Use Cleaned"\n'
            'is ticked, the cleaned positive and negative pools are shown as filled overlays.'
        )

        self._hist_panel = QWidget()
        hist_panel_layout = QVBoxLayout(self._hist_panel)
        hist_panel_layout.setContentsMargins(0, 4, 0, 0)
        hist_panel_layout.setSpacing(0)

        # Control selector row (mirrors ScatterCleaningViewer pattern)
        hist_ctrl_row = QHBoxLayout()
        hist_ctrl_row.addWidget(QLabel('Control:'))
        self._hist_combo = QComboBox()
        self._hist_combo.setMinimumWidth(220)
        self._hist_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._hist_combo.installEventFilter(WheelBlocker(self._hist_combo))
        self._hist_combo.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        hist_ctrl_row.addWidget(self._hist_combo)
        hist_ctrl_row.addStretch()
        hist_panel_layout.addLayout(hist_ctrl_row)

        # Use the same graphics layout as CytometryPlotWidget so ZoomAxis works correctly
        self._hist_glw = TransparentGraphicsLayoutWidget()
        self._hist_glw.setMinimumHeight(200)
        self._hist_glw.setMaximumHeight(280)

        # Dummy viewbox just to satisfy ZoomAxis linkToView requirement
        self._hist_vb = NoPanViewBox()
        self._hist_axis_bottom = ZoomAxis('bottom', self._hist_vb)
        self._hist_axis_left = ZoomAxis('left', self._hist_vb)

        self._hist_glw.addItem(self._hist_axis_left,  row=0, col=0)
        self._hist_glw.addItem(self._hist_vb,          row=0, col=1)
        self._hist_glw.addItem(self._hist_axis_bottom, row=1, col=1)

        self._hist_axis_left.linkToView(self._hist_vb)
        self._hist_axis_bottom.linkToView(self._hist_vb)

        self._hist_vb.setMouseEnabled(x=False, y=False)
        self._hist_vb.enableAutoRange(axis=self._hist_vb.YAxis, enable=True)

        # Left axis — plain count, no transform
        self._hist_axis_left.setTicks(None)

        self._hist_wheel_blocker = WheelBlocker(self)
        self._hist_glw.viewport().installEventFilter(self._hist_wheel_blocker)
        hist_panel_layout.addWidget(self._hist_glw)
        self._hist_panel.setVisible(False)

        self._hist_toggle.toggled.connect(self._hist_panel.setVisible)
        self._hist_toggle.toggled.connect(self._on_hist_toggle)
        self._hist_combo.currentTextChanged.connect(self._refresh_histogram)
        if self.bus:
            # refresh the histogram in case gates were changed on the raw tab
            self._mode_change_handler = lambda mode: mode == "Spectral Process" and self._refresh_histogram()
            self.bus.modeChangeRequested.connect(self._mode_change_handler)

        if self.controller.experiment.process['profiles']:
            self.plot_profiles([])

    def set_hist_toggle_visible(self, visible: bool):
        """Show or hide the peak-channel histogram toggle (controlled by cleaning activation)."""
        self._hist_toggle.setVisible(visible)

    def _refresh_hist_combo(self):
        """Repopulate the control combo from profiles in spectral model order.
        Library controls are excluded — they have no FCS file to load events from."""
        spectral_model = self.controller.experiment.process.get('spectral_model', [])
        all_profile_keys = set(self.controller.experiment.process.get('profiles', {}).keys())
        labels = [
            c['label'] for c in spectral_model
            if c.get('label') in all_profile_keys
            and c.get('control_type') != 'Single Stained Spectral Control from Library'
        ]
        current = self._hist_combo.currentText()
        self._hist_combo.blockSignals(True)
        self._hist_combo.clear()
        self._hist_combo.addItems(labels)
        if current in labels:
            self._hist_combo.setCurrentText(current)
        self._hist_combo.blockSignals(False)

    def _on_hist_toggle(self, checked: bool):
        """Called when the histogram toggle checkbox changes state."""
        if not checked:
            return
        self._refresh_hist_combo()
        self._refresh_histogram()

    def _refresh_histogram(self, *_):
        """Redraw the peak-channel histogram for the control selected in the combo."""
        try:
            if not self._hist_toggle.isChecked():
                return
        except RuntimeError:
            return
        label = self._hist_combo.currentText()
        self._plot_peak_histograms([label] if label else [])

    def _plot_peak_histograms(self, labels: list):
        """
        Draw 1-D histogram of the peak fluorescence channel for the selected
        control, using the same logicle transform and ZoomAxis ticks as the
        Raw Data tab.  Only active when exactly one control is selected.
        """
        if len(labels) != 1:
            self._hist_vb.clear()
            return
        
        from honeychrome.controller_components.spectral_functions import get_raw_events
        from honeychrome.controller_components.functions import sample_from_fcs

        self._hist_vb.clear()

        # Remove any previously-added legend label widget
        old_legend = getattr(self, '_hist_legend_label', None)
        if old_legend is not None:
            try:
                old_legend.setParent(None)
                old_legend.deleteLater()
            except Exception:
                pass
        self._hist_legend_label = None

        if not labels:
            return

        spectral_model   = self.controller.experiment.process.get('spectral_model', [])
        cleaned_store    = self.controller.cleaned_events
        event_channels_pnn = self.controller.experiment.settings['raw']['event_channels_pnn']
        fluor_ch_ids     = self.controller.filtered_raw_fluorescence_channel_ids
        experiment_dir   = self.controller.experiment_dir
        all_samples      = self.controller.experiment.samples.get('all_samples', {})
        all_samples_rev  = {v: k for k, v in all_samples.items()}
        raw_gating       = self.controller.raw_gating
        transformations  = self.controller.raw_transformations  # dict[str -> Transform]

        legend_items: list[tuple[str, str]] = []

        for i, label in enumerate(labels):
            color    = line_colors[i % len(line_colors)]
            color_qt = pg.mkColor(color)

            control = next((c for c in spectral_model if c.get('label') == label), None)
            if control is None:
                continue

            peak_ch_name = control.get('gate_channel', '')
            if not peak_ch_name or peak_ch_name not in event_channels_pnn:
                continue
            try:
                peak_local_idx = fluor_ch_ids.index(event_channels_pnn.index(peak_ch_name))
            except (ValueError, IndexError):
                continue

            xform_obj = transformations.get(peak_ch_name) if transformations else None

            # ------------------------------------------------------------------
            # Helper: transform raw values and histogram them.
            # Matches Raw Data tab: uses step_scale as bin edges when a transform
            # is available (same bins, same resolution).  Normalises counts to
            # percent-of-peak so series with very different event counts overlay
            # cleanly.  Returns (edges, pct) where len(edges) == len(pct) + 1.
            # ------------------------------------------------------------------
            def _hist(vals: np.ndarray, _xf=xform_obj) -> tuple[np.ndarray, np.ndarray]:
                import honeychrome.settings as _settings
                n_bins = _settings.hist_bins_retrieved
                if _xf is not None and _xf.scale is not None:
                    count, _ = np.histogram(vals, bins=_xf.scale)
                    count = count.astype(float)
                    peak = count.max()
                    pct = count / peak * 100 if peak > 0 else count
                    return _xf.step_scale, pct
                else:
                    lo = float(np.nanpercentile(vals, 0.1))
                    hi = float(np.nanpercentile(vals, 99.9))
                    edges = np.linspace(lo, hi, n_bins + 2)
                    count, _ = np.histogram(vals, bins=edges)
                    count = count.astype(float)
                    peak = count.max()
                    pct = count / peak * 100 if peak > 0 else count
                    return edges, pct

            use_cleaned = control.get('use_cleaned') is not False
            cleaned     = cleaned_store.get(label) if use_cleaned else None

            from honeychrome.settings import INTERNAL_NEGATIVE_SENTINEL
            _neg_name = control.get('universal_negative_name')
            use_internal = not (
                _neg_name and _neg_name != INTERNAL_NEGATIVE_SENTINEL
            )

            rel_path = all_samples_rev.get(control.get('sample_name', ''))

            _COL_ALL_GRAY  = (160, 160, 160, 100)   # gray        — all single-stained events
            _COL_CLEAN_POS = (160,  50, 220, 200)   # purple      — cleaned & selected positive
            _COL_MATCHED   = ( 80, 160, 255, 200)   # bright blue — scatter-matched negative

            if cleaned is not None:
                pos_events = cleaned.get('spectral_sub')
                if pos_events is None or len(pos_events) == 0:
                    continue
                pos_peak_raw = pos_events[:, peak_local_idx]

                # Negative overlay: for external-negative controls load from the
                # unstained file; for internal/beads load the Neg gate from the
                # positive sample.
                neg_peak_raw = None
                if use_internal:
                    # Internal: load Neg gate from positive sample
                    if rel_path and raw_gating:
                        try:
                            sample = sample_from_fcs(str(experiment_dir / rel_path))
                            all_ev = get_raw_events(sample, fluor_ch_ids)
                            neg_gate_lbl = control.get('neg_gate_label') or f'Neg {label}'
                            if raw_gating.find_matching_gate_paths(neg_gate_lbl):
                                neg_mask = raw_gating.gate_sample(sample).get_gate_membership(neg_gate_lbl)
                                neg_ev = all_ev[neg_mask]
                                neg_peak_raw = neg_ev[:, peak_local_idx] if len(neg_ev) > 0 else None
                                logger.info(
                                    f'_plot_peak_histograms: "{label}" internal neg — '
                                    f'{len(neg_ev) if len(neg_ev) > 0 else 0} events from Neg gate'
                                )
                        except Exception as exc:
                            logger.warning(f'_plot_peak_histograms: internal neg load failed for "{label}": {exc}')
                else:
                    # External: load from the assigned unstained file
                    neg_name = control.get('universal_negative_name', '')
                    all_samples_rev_local = all_samples_rev  # already built above
                    neg_rel_path = all_samples_rev_local.get(neg_name)
                    if neg_rel_path:
                        try:
                            neg_sample = sample_from_fcs(str(experiment_dir / neg_rel_path))
                            neg_ev_all = get_raw_events(neg_sample, fluor_ch_ids)
                            neg_peak_raw = neg_ev_all[:, peak_local_idx] if len(neg_ev_all) > 0 else None
                            logger.info(
                                f'_plot_peak_histograms: "{label}" external neg "{neg_name}" — '
                                f'{len(neg_ev_all) if neg_ev_all is not None else 0} events'
                            )
                        except Exception as exc:
                            logger.warning(f'_plot_peak_histograms: external neg load failed for "{label}" ("{neg_name}"): {exc}')
                    else:
                        logger.warning(
                            f'_plot_peak_histograms: "{label}" external neg "{neg_name}" not found in all_samples'
                        )

                # All-events background
                if rel_path:
                    try:
                        sample = sample_from_fcs(str(experiment_dir / rel_path))
                        all_ev = get_raw_events(sample, fluor_ch_ids)
                        x_all, y_all = _hist(all_ev[:, peak_local_idx])
                        item = pg.PlotDataItem(stepMode="center", fillLevel=0,
                                            brush=(*_COL_ALL_GRAY[:3], 60),
                                            pen=pg.mkPen(_COL_ALL_GRAY, width=1))
                        item.setData(x_all, y_all)
                        self._hist_vb.addItem(item)
                        legend_items.append(('#a0a0a0', f'{label} — all events'))
                    except Exception:
                        pass

                # Cleaned positive — solid blue (Raw Data style)
                x_pos, y_pos = _hist(pos_peak_raw)
                item = pg.PlotDataItem(stepMode="center", fillLevel=0,
                                    brush=(*_COL_CLEAN_POS[:3], 120),
                                    pen=pg.mkPen(_COL_CLEAN_POS, width=2))
                item.setData(x_pos, y_pos)
                self._hist_vb.addItem(item)
                legend_items.append(('#a032dc', f'{label} — cleaned +ve'))

                # Matched negative — lighter blue, dashed outline
                if neg_peak_raw is not None:
                    x_neg, y_neg = _hist(neg_peak_raw)
                    item = pg.PlotDataItem(stepMode="center", fillLevel=0,
                                        brush=(*_COL_MATCHED[:3], 80),
                                        pen=pg.mkPen(_COL_MATCHED, width=1, style=Qt.DashLine))
                    item.setData(x_neg, y_neg)
                    self._hist_vb.addItem(item)
                    neg_legend = 'neg gate' if use_internal else 'matched −ve'
                    legend_items.append(('#50a0ff', f'{label} — {neg_legend} (dashed)'))

            else:
                # Standard gate-mean path
                if not rel_path:
                    logger.warning(f'_plot_peak_histograms: no rel_path for label "{label}" — skipping histogram')
                    continue
                try:
                    sample = sample_from_fcs(str(experiment_dir / rel_path))
                    all_ev = get_raw_events(sample, fluor_ch_ids)
                    x_all, y_all = _hist(all_ev[:, peak_local_idx])
                    item = pg.PlotDataItem(stepMode="center", fillLevel=0,
                                        brush=(*_COL_ALL_GRAY[:3], 150),
                                        pen=pg.mkPen(_COL_ALL_GRAY[:3], width=2))
                    item.setData(x_all, y_all)
                    self._hist_vb.addItem(item)
                    legend_items.append(('#a0a0a0', label))
                except Exception as exc:
                    logger.warning(f'_plot_peak_histograms: failed to load/histogram "{label}" from "{rel_path}": {exc}')
                    continue

            # Gate regions — drawn for BOTH cleaned and non-cleaned paths
            self._draw_hist_gates(label, control, xform_obj, raw_gating)

        # Apply transform axis and labels to the selected control's peak channel
        first_ctrl = next(
            (c for c in spectral_model if c.get('label') in labels), None)
        if first_ctrl and transformations:
            ch = first_ctrl.get('gate_channel', '')
            xf = transformations.get(ch)
            if xf is not None and xf.step_scale is not None:
                self._hist_axis_bottom.setTicks(xf.ticks())
                self._hist_axis_bottom.zoomZero = xf.zero
                self._hist_axis_bottom.limits   = xf.limits
                self._hist_vb.setXRange(xf.limits[0], xf.limits[1], padding=0)
            self._hist_axis_bottom.setLabel(ch)

        self._hist_axis_left.setLabel('% of peak')
        self._hist_vb.setYRange(0, 100, padding=0.02)
        self._hist_vb.enableAutoRange(axis=self._hist_vb.YAxis, enable=False)

        # Simple colour-coded text legend
        if legend_items:
            parts = [
                f'<span style="color:{hx};">&#9632;</span> {txt}'
                for hx, txt in legend_items
            ]
            lbl = QLabel('  '.join(parts))
            lbl.setTextFormat(Qt.RichText)
            lbl.setWordWrap(True)
            self._hist_panel.layout().addWidget(lbl)
            self._hist_legend_label = lbl

    def _draw_hist_gates(self, label: str, control: dict, xform_obj, raw_gating) -> None:
        """
        Draw movable Pos/Neg gate regions on the peak-channel histogram.
        Connects each region's sigRegionChangeFinished to write the new
        bounds back into raw_gating and emit bus.changedGatingHierarchy,
        so that the gate change propagates to the Raw Data tab and the
        full gating hierarchy exactly as a manual drag on the Raw Data tab
        would do.
        """
        from flowkit import Dimension

        gate_specs = [(control.get('gate_label', ''), pg.mkBrush(0, 200, 0, 40), pg.mkPen('g', width=2)), (f'Neg {label}', pg.mkBrush(100, 100, 255, 40), pg.mkPen('b', width=2)), ]

        for gate_lbl, region_brush, region_pen in gate_specs:
            if not gate_lbl:
                continue
            try:
                if not raw_gating.find_matching_gate_paths(gate_lbl):
                    continue
                gate_path = raw_gating.find_matching_gate_paths(gate_lbl)[0]
                gate_obj = raw_gating.get_gate(gate_lbl, gate_path=gate_path)
                dim = gate_obj.dimensions[0]
                # dim.min/max are already in display (transformed) space —
                # SpectralAutoGenerator stores them via
                # raw_gating.transformations[channel_x].apply(...) before adding the gate,
                # matching how configure_rois in CytometryPlotWidget reads them directly.
                lo_t, hi_t = float(dim.min), float(dim.max)

                region = pg.LinearRegionItem(values=(lo_t, hi_t), brush=region_brush, pen=region_pen, movable=True, )
                self._hist_vb.addItem(region)

                # Floating gate label that tracks the left edge of the region
                label_line = pg.InfiniteLine(pos=lo_t, angle=90, pen=pg.mkPen(None), label=gate_lbl, labelOpts={'position': 0.92, 'color': 'g', 'fill': pg.mkBrush(0, 0, 0, 120)}, )
                self._hist_vb.addItem(label_line)

                # Keep the floating label anchored at the left edge as the region moves
                region.sigRegionChanged.connect(lambda r=region, ll=label_line: ll.setPos(r.getRegion()[0]))

                # --- Write-back: update raw_gating and broadcast the change ---
                # _make_writeback captures region explicitly to avoid the closure
                # over the loop variable pitfall.
                # note: this does not update the Raw Data tab--that requires changes to the controller
                def _make_writeback(g_lbl, g_path, rg, rgn):
                    def _on_finished():
                        lo_d, hi_d = rgn.getRegion()
                        try:
                            # get range gate and set its new dimensions
                            g = rg.get_gate(g_lbl, gate_path=g_path)
                            ch = g.dimensions[0].id
                            g.dimensions[0] = Dimension(ch, compensation_ref='uncompensated', transformation_ref=ch, range_min=lo_d, range_max=hi_d, )

                            if self.bus is not None:
                                # signal recalculation of hists and stats for raw data
                                self.bus.changedGatingHierarchy.emit('raw', g_lbl)

                                # signal refresh of ROI - search for plots with a duplicate gate
                                for n, plot in enumerate(self.controller.data_for_cytometry_plots_raw['plots']):
                                    if g_lbl in plot['child_gates']:
                                        self.bus.updateRois.emit('raw', n)

                        except Exception as exc:
                            logger.warning(f'ProfilesViewer: failed to write back gate "{g_lbl}": {exc}')

                    return _on_finished

                region.sigRegionChangeFinished.connect(_make_writeback(gate_lbl, gate_path, raw_gating, region))

            except Exception:
                pass

    def show_context_menu(self, event):
        # Empty method to completely disable context menu
        pass

    @Slot()
    def plot_latest_profile(self):
        spectral_model = self.controller.experiment.process['spectral_model']
        if spectral_model:
            control = spectral_model[-1]['label']
            if control:
                if control in self.controller.experiment.process['profiles']:
                    self.plot_profiles([control], show_legend=False)

    @Slot(list)
    def plot_profiles(self, profile_list, show_legend=True):

        # Clear previous plots
        self.plot_widget.clear()
        self.plot_items.clear()
        self.legendLayout.clear()

        profiles = self.controller.experiment.process['profiles']

        x = list(range(len(self.controller.filtered_raw_fluorescence_channel_ids)))

        pnn = self.controller.experiment.settings['raw']['event_channels_pnn']
        fluoro_ids = self.controller.filtered_raw_fluorescence_channel_ids
        # plot max 80 detectors for legibility
        LABEL_DENSITY_TARGET = 80
        stride = max(1, round(len(fluoro_ids) / LABEL_DENSITY_TARGET))

        all_pairs = [(m + 0.5, pnn[n].removesuffix('-A')) for m, n in enumerate(fluoro_ids)]
        ticks = [
            [(m, label) for m, label in all_pairs if round(m - 0.5) % stride == 0],
            [],
        ]

        db_col = self.controller.experiment.settings['raw'].get('cytometer_db_col')
        if db_col:
            detector_laser_map = get_detector_laser_map(db_col)
            tick_colors = {
                label: LASER_LABEL_COLORS[laser]
                for _, label in all_pairs
                if (laser := detector_laser_map.get(label + '-A')) in LASER_LABEL_COLORS
            }
        else:
            tick_colors = {}

        axis = self.plot_widget.getAxis('bottom')
        axis.tick_colors = tick_colors
        axis.setTicks(ticks)


        # Plot each profile
        if not profile_list:
            # Derive order from spectral model, not profiles dict insertion order
            spectral_model = self.controller.experiment.process.get('spectral_model', [])
            profile_list = [
                c['label'] for c in spectral_model
                if c.get('label') and c['label'] in profiles
            ]
            # Fall back to dict order if spectral model is empty or unpopulated
            if not profile_list:
                profile_list = list(profiles.keys())

        profile_list = [p for p in profile_list if p]
        for i, profile_name in enumerate(profile_list):
            if profile_name in profiles:
                color = line_colors[i % len(line_colors)]
                pen = pg.mkPen(color=color, width=self.pen_width)
                plot_item = self.plot_widget.plot(
                    [m for m, _ in all_pairs], profiles[profile_name], pen=pen)
                self.plot_items[profile_name] = plot_item
                if show_legend:
                    entry = LegendEntry(color, profile_name)
                    self.legendLayout.addWidget(entry)

        n = len(fluoro_ids)
        self.plot_widget.setXRange(-0.5, n - 0.5, padding=0.02)
        self.plot_widget.setYRange(0, 1)  # Set custom y-axis range

        # Refresh histogram if it's visible
        if self._hist_toggle.isChecked():
            self._refresh_hist_combo()
            self._refresh_histogram()



if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication, QVBoxLayout
    import sys

    app = QApplication(sys.argv)

    from honeychrome.controller import Controller
    from pathlib import Path
    from event_bus import EventBus

    bus = EventBus()
    kc = Controller()
    base_directory = Path.home() / 'spectral_cytometry'
    experiment_name = base_directory / '20240620 Spectral Symposium-poor cell unmixed'
    experiment_path = experiment_name.with_suffix('.kit')
    kc.load_experiment(experiment_path) # note this loads first sample too and runs calculate all histograms and statistics

    plot_frame = ProfilesViewer(bus, kc)

    # Plot the profiles
    plot_frame.plot_profiles([])
    plot_frame.show()

    plot_frame.resize(950, 600)
    sys.exit(app.exec())
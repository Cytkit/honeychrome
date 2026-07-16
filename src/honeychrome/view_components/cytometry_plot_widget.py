import io
from copy import deepcopy
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRectF, Slot, Qt, QTimer, QEvent, QObject
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QVBoxLayout, QMenu, QFrame, QFileDialog, QApplication
import colorcet as cc
from flowkit import gates as Gates
from flowkit.exceptions import GateReferenceError
from networkx import NetworkXError

from honeychrome.controller_components.functions import define_quad_gates, define_range_gate, define_ellipse_gate, define_rectangle_gate, define_polygon_gate, get_set_or_initialise_label_offset, rename_label_offset
from honeychrome.controller_components.transform import transforms_menu_items
import honeychrome.settings as settings
from honeychrome.controller_components.cytometer_whitelist import get_detector_laser_map, LASER_LABEL_COLORS

import warnings

from honeychrome.view_components.cytometry_plot_components import InteractiveLabel, NoPanViewBox, ZoomAxis, TransparentGraphicsLayoutWidget
from honeychrome.view_components.regions_of_interest import RangeROI, EllipseROI, RectangleROI, QuadROI, PolygonROI, PolygonROIConstructor
import logging
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="t.core.qobject.connect: QObject::connect(QStyleHints, QStyleHints): unique connections require a pointer to member function of a QObject subclass")

from PySide6.QtCore import QMimeData, QByteArray, QBuffer, QIODevice

from PySide6.QtCore import QPoint
from PySide6.QtGui import QPixmap, Qt, QPainter
import pyqtgraph as pg



def get_widget_pixmap(widget, scale_factor=2):
    pm = QPixmap(widget.size() * scale_factor)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.scale(scale_factor, scale_factor)
    widget.render(painter, QPoint(0, 0))
    painter.end()
    return pm

def export_widget_png(cytometry_plot_widget, filename, scale_factor=2):
    filename_png = str(Path(filename).with_suffix('.png'))
    pm = get_widget_pixmap(cytometry_plot_widget, scale_factor)
    pm.save(filename_png)
    print(f"PNG exported to {filename_png}")

def copy_widget_to_clipboard(cytometry_plot_widget, prefer_compressed=True, scale_factor=2):
    """
    Copy QPixmap with multiple formats for maximum compatibility.

    Args:
        pixmap: QPixmap to copy
        prefer_compressed: If True, prefer PNG/JPEG formats
    """
    # --- Render the widget ---
    pm = get_widget_pixmap(cytometry_plot_widget)

    clipboard = QApplication.clipboard()
    mime_data = QMimeData()
    image = pm.toImage()

    if prefer_compressed:
        # Add PNG format (lossless compression)
        png_data = QByteArray()
        png_buffer = QBuffer(png_data)
        png_buffer.open(QIODevice.WriteOnly)
        image.save(png_buffer, "PNG", 50)
        mime_data.setData("image/png", png_data)

        # Add JPEG format (lossy but small)
        jpeg_data = QByteArray()
        jpeg_buffer = QBuffer(jpeg_data)
        jpeg_buffer.open(QIODevice.WriteOnly)
        image.save(jpeg_buffer, "JPEG", 75)
        mime_data.setData("image/jpeg", jpeg_data)

    # Always include the raw image data for compatibility
    # This is what setPixmap() does internally
    mime_data.setImageData(image)

    # Also include BMP format (Windows compatibility)
    bmp_data = QByteArray()
    bmp_buffer = QBuffer(bmp_data)
    bmp_buffer.open(QIODevice.WriteOnly)
    image.save(bmp_buffer, "BMP")
    mime_data.setData("image/bmp", bmp_data)

    clipboard.setMimeData(mime_data)


def pm_to_png_buffer(pm):
    image = pm.toImage()
    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly)
    image.save(buffer, "PNG")
    image_stream = io.BytesIO(buffer.data())
    return image_stream



class CytometryPlotWidget(QFrame):
    def __init__(self, bus=None, mode=None, n_in_plot_sequence=None, plot=None, data_for_cytometry_plots=None, parent=None):
        super().__init__(parent)
        # connect to data
        self.bus = bus
        self.mode = mode
        self.data_for_cytometry_plots = data_for_cytometry_plots
        self.plot = plot
        self.n_in_plot_sequence = n_in_plot_sequence

        if self.bus is not None:
            self.bus.updateSourceChildGates.connect(self.refresh_source_child_gates)
            self.bus.histsStatsRecalculated.connect(self.update_axes_stats_hist)
            self.bus.updateRois.connect(self.configure_rois)

        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # # Create control buttons layout... placeholder for buttons
        # control_layout = QtWidgets.QHBoxLayout()

        # Access the internal layout (QGraphicsGridLayout)
        self.graphics_widget = TransparentGraphicsLayoutWidget(parent=self)
        layout = self.graphics_widget.ci.layout

        # Set spacing between items (in pixels)
        layout.setHorizontalSpacing(0)
        layout.setVerticalSpacing(0)

        # Add plot title, ViewBox for plotting,
        self.plot_title = InteractiveLabel("Plot Title", parent_plot=self)
        self.plot_title.setParent(self)
        self.graphics_widget.addItem(self.plot_title, row=0, col=2)
        self.vb = NoPanViewBox()
        self.vb.setParent(self)
        self.graphics_widget.addItem(self.vb, row=1, col=2)

        # Y axis label, Y axis itself, X axis, X axis label, Link axes to viewbox
        self.label_y = InteractiveLabel("Y Axis", parent_plot=self, angle=-90)
        self.graphics_widget.addItem(self.label_y, row=1, col=0)
        self.axis_left = ZoomAxis('left', self.vb)
        self.graphics_widget.addItem(self.axis_left, row=1, col=1)
        self.axis_bottom = ZoomAxis('bottom', self.vb)
        self.graphics_widget.addItem(self.axis_bottom, row=2, col=2)
        self.label_x = InteractiveLabel("X Axis", parent_plot=self)
        self.graphics_widget.addItem(self.label_x, row=3, col=2)
        self.axis_left.linkToView(self.vb)
        self.axis_bottom.linkToView(self.vb)
        self.label_x.setParent(self.axis_bottom)
        self.label_y.setParent(self.axis_left)
        self.axis_bottom.setParent(self)
        self.axis_left.setParent(self)
        # connect applyZoom method
        self.axis_bottom.zoom_timer.timeout.connect(lambda: self.apply_zoom('x'))
        self.axis_left.zoom_timer.timeout.connect(lambda: self.apply_zoom('y'))

        # connect right click menu
        self.vb.raiseContextMenu = self.right_click_menu

        # initialise configuration
        self.pnn = self.data_for_cytometry_plots['pnn']
        self.pnn_labels = self.data_for_cytometry_plots.get('pnn_labels') or {}
        self.fluoro_indices = self.data_for_cytometry_plots['fluoro_indices']
        self.transformations = self.data_for_cytometry_plots['transformations']
        self.statistics = self.data_for_cytometry_plots['statistics']
        self.gating = self.data_for_cytometry_plots['gating']

        # initialise image and histogram curve
        # note: this has to be done in initialisation, otherwise new colourmap won't be loaded if changed in app config
        if settings.hist2dtype_retrieved == '2D Histogram':
            colors = cc.palette[settings.colourmap_name_retrieved]  # Get the colormap from Colorcet
            # cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), color=colors)  # Convert Colorcet colormap to PyQtGraph's format
            # cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)) ** 2, color=colors)  # Convert Colorcet colormap to PyQtGraph's format
            # cmap = pg.ColorMap(pos=(np.exp(np.linspace(0.0, 1.0, len(colors)))-1)/(np.exp(1)-1), color=colors)  # Convert Colorcet colormap to PyQtGraph's format
            cmap = pg.ColorMap(pos=0.9*np.linspace(0.0, 1.0, len(colors))**2 + 0.1*np.linspace(0.0, 1.0, len(colors)), color=colors)  # Convert Colorcet colormap to PyQtGraph's format
        else:
            colors = cc.palette['glasbey_bw_minc_20']  # categorical colormap
            cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), color=colors)  # Convert Colorcet colormap to PyQtGraph's format
        rgba_lut = cmap.getLookupTable(alpha=True)
        rgba_lut[0, 3] = 0  # Fully transparent for 0

        self.img = pg.ImageItem(parent=self)
        self.img.setLookupTable(rgba_lut)
        self.vb.addItem(self.img)
        self.hist = pg.PlotDataItem(stepMode='center', fillLevel=0, brush=(100, 100, 250, 150), parent=self)
        self.vb.addItem(self.hist)
        self.count = np.zeros(settings.hist_bins_retrieved+2)

        # Add widgets to main layout
        # main_layout.addLayout(control_layout) # placeholder for extra buttons
        main_layout.addWidget(self.graphics_widget)

        # set items for channel menus
        self.label_x.leftClickMenuFunction = self.set_up_plot
        self.label_y.leftClickMenuFunction = self.set_up_plot
        self.label_x.rightClickMenuFunction = self.set_axis_transform
        self.label_y.rightClickMenuFunction = self.set_axis_transform

        self.setFrameShape(QFrame.NoFrame)
        self.setLineWidth(1)
        self.selected = False

        # configure initial axes, labels, menus
        self.configure_axes()

        # configure rois associated with child gates
        # initialise dict of rois, gate_id:roi
        self.rois = []
        self.configure_rois(self.mode, self.n_in_plot_sequence)

        self._mouse_events_enabled = True

    def mousePressEvent(self, event: QMouseEvent):
        if self._mouse_events_enabled:
            self.select_plot_on_parent_grid()

    def select_plot_on_parent_grid(self):
        """Emit selection signal when clicked."""
        # if event.button() == Qt.MouseButton.LeftButton:
        grid = self.parent().parent().parent()
        if type(grid).__name__ == 'CytometryGridWidget':
            grid.select_plot(self)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self._mouse_events_enabled:
            if event.button() == Qt.LeftButton:
                grid = self.parent().parent().parent()
                if type(grid).__name__ == 'CytometryGridWidget':
                    self.parent().parent().parent().select_plot(self)
                    # self.parent().parent().parent().open_plot_in_modal(self)
                    # self.installEventFilter(DebugMouseFilter(self))
                    QTimer.singleShot(0, lambda: grid.open_plot_in_modal(self))

    def set_up_plot(self, item=None, parent=None):
        if item == 'All Fluorescence':
            self.plot['type'] = 'ribbon'
            self.plot.pop('channel_x', None)
            self.plot.pop('channel_y', None)
        elif item == 'Count':
            self.plot['type'] = 'hist1d'
            self.plot.pop('channel_y', None)
        else:
            n = item

            if parent == self.axis_bottom:
                channel = 'channel_x'
            else:
                channel = 'channel_y'

            self.plot[channel] = self.pnn[n]

            if 'channel_x' in self.plot and 'channel_y' in self.plot:
                self.plot['type'] = 'hist2d'
            elif self.plot['channel_x'] is not None:
                self.plot['type'] = 'hist1d'

        if self.bus is not None:
            self.bus.plotChangeRequested.emit(self.mode, self.n_in_plot_sequence)
        else:
            warnings.warn('Signals bus not connected')

        print(f'CytometryPlotWidget {self.mode} {self.n_in_plot_sequence}: set up plot {self.plot}')

    @Slot(str, list)
    def update_axes_stats_hist(self, mode, indices_plots_to_recalculate=None):
        if mode == self.mode:
            if indices_plots_to_recalculate is None or self.n_in_plot_sequence in indices_plots_to_recalculate:
                self.configure_axes()
                self.plot_histogram()
                # for roi in self.rois:
                #     roi.label.add_statistic_to_name()

                # print(f'CytometryPlotWidget {mode} {self.n_in_plot_sequence}: updated axes stats hist')

    def configure_axes(self):
        ######### example plots #########
        # [
        #     {'type': 'hist1d', 'channel_x': 'Time', 'source_gate': 'root', 'child_gates': []},
        #     {'type': 'hist2d', 'channel_x': 'FSC-A', 'channel_y': 'SSC-A', 'source_gate': 'root', 'child_gates': ['cells']},
        #     {'type': 'ribbon', 'source_gate': 'cells', 'child_gates': []},
        #     {'type': 'hist2d', 'channel_x': 'BUV805-A', 'channel_y': 'Spark UV 387-A', 'source_gate': 'cells',
        #      'child_gates': ['activated']},
        # ]

        #todo handle default (non-transformed) axes

        self.configure_title()
        self.axis_bottom.tick_colors = {}
        
        # Apply labels to axes, activate heatmap image or 1d histogram, set autorange if applicable
        if self.plot['type'] == 'ribbon':
            # set labels and view
            self.label_x.setText('All Fluorescence')
            self.label_y.setText('Intensity')
            self.img.setVisible(True)
            self.hist.setVisible(False)
            self.vb.enableAutoRange(axis=self.vb.XAxis, enable=True)

            # set menus
            self.label_x.leftClickMenuItems = self.pnn + ['All Fluorescence']
            self.label_x.leftItemSelected = len(self.pnn)
            self.label_x.rightItemSelected = 'default'
            self.label_y.leftClickMenuItems = ['Intensity']
            self.label_y.leftItemSelected = 0
            self.label_y.rightItemSelected = self.transformations['ribbon'].id
            self.label_x.rightClickMenuItems = []
            self.label_y.rightClickMenuItems = transforms_menu_items

            # set axes
            # set axes — with adaptive label thinning and laser colour-coding
            pnn = self.data_for_cytometry_plots['pnn']
            fluoro_indices = self.data_for_cytometry_plots['fluoro_indices']
            n_detectors = len(fluoro_indices)

            # Thinning: keep roughly LABEL_DENSITY_TARGET labels regardless of panel size.
            # At ≤80 detectors every label is shown; above that we skip proportionally.
            LABEL_DENSITY_TARGET = 80
            stride = max(1, round(n_detectors / LABEL_DENSITY_TARGET))

            all_pairs = [(m + 0.5, pnn[n].removesuffix('-A')) for m, n in enumerate(fluoro_indices)]
            ticks = [
                [(m, label) for m, label in all_pairs if round(m - 0.5) % stride == 0],
                [],
            ]

            # Laser colour-coding: look up which laser each detector belongs to.
            db_col = self.data_for_cytometry_plots.get('cytometer_db_col')
            if db_col:
                detector_laser_map = get_detector_laser_map(db_col)
                tick_colors = {
                    label: LASER_LABEL_COLORS[laser]
                    for _, label in all_pairs
                    if (laser := detector_laser_map.get(label + '-A')) in LASER_LABEL_COLORS
                }
            else:
                tick_colors = {}

            self.axis_bottom.tick_colors = tick_colors
            self.axis_bottom.setTicks(ticks, angle=90)
            self.axis_left.setTicks(self.transformations['ribbon'].ticks())
            self.axis_left.zoomZero = self.transformations['ribbon'].zero
            self.vb.setMouseEnabled(x=False, y=False)

            # set limits
            self.axis_bottom.limits = (-0.5, len(self.fluoro_indices) - 0.5)
            self.axis_left.limits = self.transformations['ribbon'].limits
            self.vb.setYRange(self.axis_left.limits[0], self.axis_left.limits[1], padding=0)

        elif self.plot['type'] == 'hist1d':
            # set labels and view
            self.label_x.setText(self.pnn_labels.get(self.plot['channel_x'], self.plot['channel_x']))
            self.label_y.setText('Count')
            self.img.setVisible(False)
            self.hist.setVisible(True)
            self.vb.enableAutoRange(axis=self.vb.YAxis, enable=True)

            # set menus
            self.label_x.leftClickMenuItems = [self.pnn_labels.get(ch, ch) for ch in self.pnn] + ['All Fluorescence']
            if self.transformations[self.plot['channel_x']].id == 'default':
                self.label_x.rightItemSelected = 'default'
                self.label_x.rightClickMenuItems = []
                self.vb.setMouseEnabled(x=True, y=False)
            else:
                self.label_x.rightItemSelected = self.transformations[self.plot['channel_x']].id
                self.label_x.rightClickMenuItems = transforms_menu_items
                self.vb.setMouseEnabled(x=False, y=False)
            self.label_y.leftClickMenuItems = [self.pnn_labels.get(ch, ch) for ch in self.pnn] + ['Count']
            self.label_y.leftItemSelected = len(self.pnn)
            self.label_y.rightItemSelected = 'default'
            self.label_y.rightClickMenuItems = []

            # set axes
            self.axis_bottom.setTicks(self.transformations[self.plot['channel_x']].ticks())
            self.axis_bottom.zoomZero = self.transformations[self.plot['channel_x']].zero
            self.axis_left.setTicks(None)

            # Record transformations in gating strategy
            self.gating.transformations[self.plot['channel_x']] = self.transformations[self.plot['channel_x']].xform

            # Set limits
            self.axis_bottom.limits = self.transformations[self.plot['channel_x']].limits
            self.vb.setXRange(self.axis_bottom.limits[0], self.axis_bottom.limits[1], padding=0)

        elif self.plot['type'] == 'hist2d':
            # set labels and view
            self.label_x.setText(self.pnn_labels.get(self.plot['channel_x'], self.plot['channel_x']))
            self.label_y.setText(self.pnn_labels.get(self.plot['channel_y'], self.plot['channel_y']))
            self.img.setVisible(True)
            self.hist.setVisible(False)

            # set menus
            self.label_x.leftClickMenuItems = [self.pnn_labels.get(ch, ch) for ch in self.pnn] + ['All Fluorescence']
            self.label_x.leftItemSelected = self.pnn.index(self.plot['channel_x'])
            self.label_x.rightItemSelected = self.transformations[self.plot['channel_x']].id
            self.label_y.leftClickMenuItems = [self.pnn_labels.get(ch, ch) for ch in self.pnn] + ['Count']
            self.label_y.leftItemSelected = self.pnn.index(self.plot['channel_y'])
            self.label_y.rightItemSelected = self.transformations[self.plot['channel_y']].id
            self.label_x.rightClickMenuItems = transforms_menu_items
            self.label_y.rightClickMenuItems = transforms_menu_items

            # set axes
            self.axis_bottom.setTicks(self.transformations[self.plot['channel_x']].ticks())
            self.axis_left.setTicks(self.transformations[self.plot['channel_y']].ticks())
            self.axis_bottom.zoomZero = self.transformations[self.plot['channel_x']].zero
            self.axis_left.zoomZero = self.transformations[self.plot['channel_y']].zero
            self.vb.setMouseEnabled(x=False, y=False)

            # Record transformations in gating strategy
            self.gating.transformations[self.plot['channel_x']] = self.transformations[self.plot['channel_x']].xform
            self.gating.transformations[self.plot['channel_y']] = self.transformations[self.plot['channel_y']].xform

            # Set limits
            self.axis_bottom.limits = self.transformations[self.plot['channel_x']].limits
            self.vb.setXRange(self.axis_bottom.limits[0], self.axis_bottom.limits[1], padding=0)
            self.axis_left.limits = self.transformations[self.plot['channel_y']].limits
            self.vb.setYRange(self.axis_left.limits[0], self.axis_left.limits[1], padding=0)

        else:
            warnings.warn(f'Plot not defined: {self.plot}')

        if len(self.plot['child_gates']):
            self.label_x.leftClickMenuItems = []
            self.label_y.leftClickMenuItems = []
            self.label_x.setToolTip("Channel fixed by child gates")
            self.label_y.setToolTip("Channel fixed by child gates")
        else:
            self.label_x.setToolTip("")
            self.label_x.setToolTip("")

        # print(f'CytometryPlotWidget {self.mode} {self.n_in_plot_sequence}: axes configured {self.plot}')

    @Slot(str, str)
    def refresh_source_child_gates(self, mode, new_gate_name):
        if mode == self.mode:
            # update all title menus
            # set source_gate if the previous source_gate has disappeared
            if new_gate_name != '' and self.plot['source_gate'] != 'root' and len(self.gating.find_matching_gate_paths(self.plot['source_gate'])) == 0:
                self.plot['source_gate'] = new_gate_name

            # if any previous child gate has disappeared, set to the new name
            for old_child_gate in deepcopy(self.plot['child_gates']):
                if len(self.gating.find_matching_gate_paths(old_child_gate)) == 0:
                    self.plot['child_gates'].remove(old_child_gate)
                    self.plot['child_gates'].append(new_gate_name)

                    rename_label_offset(self.plot, old_child_gate, new_gate_name)

            self.configure_title()

    def configure_title(self):
        # set title and items for source gate menu
        # this should include all gates (but not QuadrantGates)
        # but exclude the child gates and their descendants
        gate_ids = [g for g in self.gating.get_gate_ids() if self.gating._get_gate_node(g[0], g[1]).gate_type != 'QuadrantGate']
        descendant_set = [g[0] for g in gate_ids if len(set(self.plot['child_gates']) & set(list(g[1]) + [g[0]]))]
        gate_names = ['root'] + [g[0] for g in gate_ids if g[0] not in descendant_set]
        self.plot_title.leftClickMenuItems = gate_names
        self.plot_title.leftClickMenuFunction = self.set_source_gate

        if self.plot['source_gate'] not in gate_names:
            # source gate was removed; fall back to root
            logger.warning(f"CytometryPlotWidget configure_title: source_gate '{self.plot['source_gate']}' not in gate_names, reset to root")
            self.plot['source_gate'] = 'root'

        self.plot_title.setText(self.plot['source_gate'])
        self.plot_title.leftItemSelected = gate_names.index(self.plot['source_gate'])


    def set_axis_transform(self, n, parent):
        if self.plot['type'] == 'ribbon':
            channel = 'ribbon'
        else:
            if parent == self.axis_bottom:
                channel = self.plot['channel_x']
            else:
                channel = self.plot['channel_y']

        self.transformations[channel].set_transform(id=n)

        if self.bus is not None:
            self.bus.axisTransformed.emit(channel)
        else:
            warnings.warn('Signals bus not connected')

    def set_source_gate(self, n, parent):
        self.plot['source_gate'] = self.plot_title.leftClickMenuItems[n]
        self.configure_axes()
        print(f'CytometryPlotWidget {self.mode} {self.n_in_plot_sequence}: set source gate {self.plot['source_gate']}')

        for gate_name in self.plot['child_gates']:
            #todo seek some guidance on how to do this properly
            # gate = self.gating.get_gate(gate_name)
            node = self.gating._get_gate_node(gate_name)
            new_parent_node = self.gating._get_gate_node(self.plot['source_gate'])
            node.parent = new_parent_node

        self.bus.changedGatingHierarchy.emit(self.mode, 'root')

    def get_gate_path(self):
        # return source gate path to add gate to hierarchy
        if self.plot['source_gate'] == 'root':
            gate_path = ('root',)
        else:
            gate_path = self.gating.find_matching_gate_paths(self.plot['source_gate'])[0] + (self.plot['source_gate'],)
            # gate_path = tuple(gate_name for gate_name in gate_path if self.gating.get_gate(gate_name).gate_type != 'QuadrantGate')
            # gate_path = ('root',) + tuple(gate_name for n, gate_name in enumerate(gate_path[1:]) if self.gating._get_gate_node(gate_name, gate_path[:n+1]).gate_type != 'QuadrantGate')
        return gate_path

    def right_click_menu(self, ev):
        """Override the default context menu."""
        menu = QMenu()
        menu.addAction("Fit Axes to Data", self.fit_axes_to_data)
        menu.addAction("Reset Axes", self.reset_axes_transforms)
        menu.addAction("Delete Child Gates", self.remove_gate_and_roi) # remove all child gates
        menu.addAction("Export Image", self.export_image)
        menu.addAction("Copy to Clipboard", lambda : copy_widget_to_clipboard(self))
        menu.exec(ev.screenPos().toPoint())

    def export_image(self):
        base_directory = Path.home() / settings.experiments_folder
        path, _ = QFileDialog.getSaveFileName(self, "Export plot as image", str(base_directory), f"PNG Image File (*.png)")
        if path:
            export_widget_png(self, path)

    def fit_axes_to_data(self):
        if self.plot['type'] == 'ribbon':
            channels = ['ribbon']
            channel_max = np.percentile(self.data_for_cytometry_plots['event_data'][:,self.fluoro_indices], 99)
            self.transformations[channels[0]].scale_t = 2 * channel_max
            self.transformations[channels[0]].set_transform(limits=[0, 1])

        else:
            if self.plot['type']=='hist1d':
                channels = [self.plot['channel_x']]
            else: #hist2d
                channels = [self.plot['channel_x'], self.plot['channel_y']]

            for channel in channels:
                channel_min, channel_max = np.percentile(self.data_for_cytometry_plots['event_data'][:,self.pnn.index(channel)], [1, 99])
                if self.transformations[channel].id == 1:
                    self.transformations[channel].scale_t = 1.5 * channel_max
                    if channel_min < -10**self.transformations[channel].logicle_w:
                        logicle_a = min([
                            max([
                                np.log10(-channel_min) - self.transformations[channel].logicle_w,
                                0
                            ]),
                            0.5 * np.log10(self.transformations[channel].scale_t)
                        ])
                    else:
                        logicle_a = 0
                    self.transformations[channel].logicle_a = logicle_a
                    self.transformations[channel].set_transform(limits=[0, 1])

                    print([channel, channel_min, channel_max, self.transformations[channel].scale_t, logicle_a])


                elif self.transformations[channel].id == 0:
                    self.transformations[channel].scale_t = 1.5 * channel_max
                    linear_a = max([-channel_min * 2, 0])
                    self.transformations[channel].linear_a = linear_a
                    self.transformations[channel].set_transform(limits=[0, 1])

                    print([channel, channel_min, channel_max, self.transformations[channel].scale_t, linear_a])

        for channel in channels:
            if self.bus is not None:
                self.bus.axisTransformed.emit(channel)


    def reset_axes_transforms(self):
        # self.vb.autoRange() # not necessary - configured by configure_axes
        if self.plot['type'] == 'ribbon':
            channels = ['ribbon']
        else:
            if self.plot['type']=='hist1d':
                channels = [self.plot['channel_x']]
            else: #hist2d
                channels = [self.plot['channel_x'], self.plot['channel_y']]

        if self.bus is not None:
            self.bus.axesReset.emit(channels)
        else:
            warnings.warn('Signals bus not connected')

    def new_plot_on_gate(self, source_gate):
        plot = deepcopy(self.plot)
        plot['source_gate'] = source_gate
        plot['child_gates'] = []
        self.data_for_cytometry_plots['plots'].append(plot)
        self.bus.showNewPlot.emit(self.mode)

    @Slot(str, int)
    def configure_rois(self, mode, index):
        if mode == self.mode and index == self.n_in_plot_sequence:

            # first wipe out all rois but *not* their associated gates
            rois = self.rois.copy()
            for roi in rois:
                roi.request_remove(delete_gate=False)
            self.rois.clear()

            for gate_name in self.plot['child_gates']:
                try:
                    gate = self.gating.get_gate(gate_name)
                except Exception:
                    logger.warning(f'configure_rois: gate "{gate_name}" not found in gating strategy — skipping ROI.')
                    continue
                label_offset = get_set_or_initialise_label_offset(self.plot, gate_name)

                if gate.gate_type == 'PolygonGate':
                    vertices = gate.vertices
                    roi = PolygonROI(vertices, gate_name, self.gating, self.mode, self.vb, label_offset=label_offset)
                    roi.sigRegionChangeFinished.connect(lambda *args, r=roi: self.update_polygon(r))

                elif gate.gate_type == 'RectangleGate' and len(gate.dimensions)==2: # i.e. true rectangle
                    dim_x, dim_y = gate.dimensions
                    pos = [dim_x.min, dim_y.min]
                    size = [dim_x.max - dim_x.min, dim_y.max - dim_y.min]
                    roi = RectangleROI(pos, size, gate_name, self.gating, self.mode, self.vb, label_offset=label_offset)
                    roi.sigRegionChangeFinished.connect(lambda *args, r=roi: self.update_rectangle(r))

                elif gate.gate_type == 'RectangleGate' and len(gate.dimensions)==1: #i.e. range gate
                    dim_x = gate.dimensions[0]
                    roi = RangeROI(dim_x.min, dim_x.max, gate_name, self.gating, self.mode, self.vb, label_offset=label_offset)
                    roi.sigRangeChanged.connect(lambda *args, r=roi: self.update_range(r))

                elif gate.gate_type == 'EllipsoidGate':
                    coordinates = gate.coordinates
                    covariance_matrix = gate.covariance_matrix
                    distance_square = gate.distance_square

                    # Eigen decomposition
                    eigvals, eigvecs = np.linalg.eigh(covariance_matrix)

                    # Sort by descending eigenvalue
                    order = np.argsort(eigvals)[::-1]
                    eigvals = eigvals[order]
                    eigvecs = eigvecs[:, order]

                    # Axis lengths
                    w, h = eigvals

                    # Rotation angle (in radians → degrees)
                    theta = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
                    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    angle = np.rad2deg(theta)

                    # Reconstruct size and position
                    size = np.array([w, h])
                    pos = np.array(coordinates) - 0.5 * R @ size

                    # Consistency check
                    if not np.isclose(distance_square, w * h):
                        print("Warning: distance_square inconsistent with covariance")

                    roi = EllipseROI(pos, size, angle, gate_name, self.gating, self.mode, self.vb, label_offset=label_offset)
                    roi.sigRegionChangeFinished.connect(lambda *args, r=roi: self.update_ellipse(r))

                elif gate.gate_type == 'QuadrantGate':
                    quadrants = gate.quadrants
                    divider_double_positive = quadrants[list(quadrants)[0]]._divider_ranges
                    x = max([val if val else 0 for val in divider_double_positive['xdiv']])
                    y = max([val if val else 0 for val in divider_double_positive['ydiv']])
                    roi = QuadROI(x, y, gate_name, self.gating, self.mode, self.vb)
                    roi.sigPosChanged.connect(lambda *args, r=roi: self.update_quad(r))

                else:
                    warnings.warn('Wrong gate type')

                # set up the signal for deletion of the gate, but don't run it here
                roi.sigRemoveRequested.connect(self.remove_gate_and_roi)
                self.rois.append(roi)

    def initiate_polygon_roi(self):
        polygon_roi_constructor = PolygonROIConstructor(self)

    def create_polygon_gate(self, vertices):
        # set name
        name_increment = 1
        while f'P{name_increment}' in [g[0] for g in self.gating.get_gate_ids()]:
            name_increment += 1
        gate_name = f'P{name_increment}'

        # convert QPointF to list of pairs
        vertices = [(v.x(), v.y()) for v in vertices]

        # Create PolyLineROI
        roi = PolygonROI(vertices, gate_name, self.gating, self.mode, self.vb)
        roi.sigRemoveRequested.connect(self.remove_gate_and_roi)
        roi.sigRegionChangeFinished.connect(lambda: self.update_polygon(roi))
        self.rois.append(roi)

        vertices, dim_x, dim_y = define_polygon_gate(vertices, self.plot['channel_x'], self.plot['channel_y'], self.transformations)
        gate = Gates.PolygonGate(gate_name, [dim_x, dim_y], vertices, use_complement=False)
        self.gating.add_gate(gate, gate_path=self.get_gate_path())
        self.plot['child_gates'].append(gate_name)
        if self.bus is not None:
            self.bus.updateSourceChildGates.emit(self.mode, gate_name)
            self.bus.changedGatingHierarchy.emit(self.mode, gate_name)

    def update_polygon(self, roi):
        roi_state = roi.getState()
        origin = roi_state['pos']
        vertices = roi_state['points']

        # convert QPointF to list of pairs
        vertices = [(origin.x() + v.x(), origin.y() + v.y()) for v in vertices]
        vertices, dim_x, dim_y = define_polygon_gate(vertices, self.plot['channel_x'], self.plot['channel_y'], self.transformations)
        gate = self.gating.get_gate(roi.label.gate_name)
        gate.vertices = vertices

        if self.bus is not None:
            self.bus.changedGatingHierarchy.emit(self.mode, roi.label.gate_name)
        # print([gate, gate.vertices])

    def new_rectangle_gate(self):
        # set name
        name_increment = 1
        while f'R{name_increment}' in [g[0] for g in self.gating.get_gate_ids()]:
            name_increment += 1
        gate_name = f'R{name_increment}'

        # create ROI
        scale_x = self.transformations[self.plot['channel_x']].limits[1]
        scale_y = self.transformations[self.plot['channel_y']].limits[1]
        x0 = scale_x * (0.35 + 0.05 * name_increment)
        y0 = scale_y * (0.45 - 0.05 * name_increment)
        Dx = scale_x * 0.2
        Dy = scale_y * 0.2
        pos = [x0, y0]
        size = [Dx, Dy]

        roi = RectangleROI([x0, y0], [Dx, Dy], gate_name, self.gating, self.mode, self.vb)
        roi.sigRemoveRequested.connect(self.remove_gate_and_roi)
        roi.sigRegionChangeFinished.connect(lambda: self.update_rectangle(roi))
        self.rois.append(roi)

        dim_x, dim_y = define_rectangle_gate(pos, size, self.plot['channel_x'], self.plot['channel_y'], self.transformations)
        gate = Gates.RectangleGate(gate_name, dimensions=[dim_x, dim_y])
        self.gating.add_gate(gate, gate_path=self.get_gate_path())
        self.plot['child_gates'].append(gate_name)
        if self.bus is not None:
            self.bus.updateSourceChildGates.emit(self.mode, gate_name)
            self.bus.changedGatingHierarchy.emit(self.mode, gate_name)

    def update_rectangle(self, roi):
        roi_state = roi.getState()
        pos = roi_state['pos']
        size = roi_state['size']

        dim_x, dim_y = define_rectangle_gate(pos, size, self.plot['channel_x'], self.plot['channel_y'], self.transformations)
        gate = self.gating.get_gate(roi.label.gate_name)
        gate.dimensions = [dim_x, dim_y]

        if self.bus is not None:
            self.bus.changedGatingHierarchy.emit(self.mode, roi.label.gate_name)
        # print([gate, gate.dimensions, gate.dimensions[0].min, gate.dimensions[0].max, gate.dimensions[1].min, gate.dimensions[1].max])

    def new_ellipse_gate(self):
        # set name
        name_increment = 1
        while f'E{name_increment}' in [g[0] for g in self.gating.get_gate_ids()]:
            name_increment += 1
        gate_name = f'E{name_increment}'

        # create ROI
        scale_x = self.transformations[self.plot['channel_x']].limits[1]
        scale_y = self.transformations[self.plot['channel_y']].limits[1]
        x0 = scale_x * (0.3 + 0.05 * name_increment)
        y0 = scale_y * (0.4 - 0.05 * name_increment)
        Dx = scale_x * 0.3
        Dy = scale_y * 0.2
        pos = [x0, y0]
        size = [Dx, Dy]
        angle = 0

        roi = EllipseROI(pos, size, angle, gate_name, self.gating, self.mode, self.vb)
        roi.sigRemoveRequested.connect(self.remove_gate_and_roi)
        roi.sigRegionChangeFinished.connect(lambda: self.update_ellipse(roi))
        self.rois.append(roi)

        # define and add gate to gating
        dim_x, dim_y, coordinates, covariance_matrix, distance_square = define_ellipse_gate(pos, size, angle, self.plot['channel_x'], self.plot['channel_y'], self.transformations)
        gate = Gates.EllipsoidGate(gate_name, [dim_x, dim_y], coordinates, covariance_matrix, distance_square)
        self.gating.add_gate(gate, gate_path=self.get_gate_path())
        self.plot['child_gates'].append(gate_name)
        if self.bus is not None:
            self.bus.updateSourceChildGates.emit(self.mode, gate_name)
            self.bus.changedGatingHierarchy.emit(self.mode, gate_name)

    def update_ellipse(self, roi):
        roi_state = roi.getState()
        pos = roi_state['pos']
        size = roi_state['size']
        angle = roi_state['angle']

        dim_x, dim_y, coordinates, covariance_matrix, distance_square = define_ellipse_gate(pos, size, angle, self.plot['channel_x'], self.plot['channel_y'], self.transformations)

        gate = self.gating.get_gate(roi.label.gate_name)
        gate.coordinates = coordinates
        gate.covariance_matrix = covariance_matrix
        gate.distance_square = distance_square

        if self.bus is not None:
            self.bus.changedGatingHierarchy.emit(self.mode, roi.label.gate_name)
        # print([gate, gate.coordinates, gate.covariance_matrix, gate.distance_square])

    def new_range_gate(self):
        # set name
        name_increment = 1
        while f'Range{name_increment}' in [g[0] for g in self.gating.get_gate_ids()]:
            name_increment += 1
        gate_name = f'Range{name_increment}'

        # create ROI
        if self.data_for_cytometry_plots['transformations'][self.plot['channel_x']].id == 'default':
            x1, x2 = 0.35 * settings.default_ceiling, 0.65 * settings.default_ceiling
        else:
            x1, x2 = 0.35, 0.65

        roi = RangeROI(x1, x2, gate_name, self.gating, self.mode, self.vb)
        roi.sigRemoveRequested.connect(self.remove_gate_and_roi)
        #roi.sigRangeChanged.connect(lambda x1, x2: print(f"Gate moved to: x1={x1:.2f}, x2={x2:.2f}"))
        roi.sigRangeChanged.connect(lambda: self.update_range(roi))
        self.rois.append(roi)

        # define and add gate to gating
        dim_x = define_range_gate(x1, x2, self.plot['channel_x'], self.transformations)
        gate = Gates.RectangleGate(gate_name, dimensions=[dim_x])
        self.gating.add_gate(gate, gate_path=self.get_gate_path())
        self.plot['child_gates'].append(gate_name)
        if self.bus is not None:
            self.bus.updateSourceChildGates.emit(self.mode, gate_name)
            self.bus.changedGatingHierarchy.emit(self.mode, gate_name)

    def update_range(self, roi):
        x1 = roi.v1.value()
        x2 = roi.v2.value()
        dim_x = define_range_gate(x1, x2, self.plot['channel_x'], self.transformations)

        gate = self.gating.get_gate(roi.label.gate_name)
        gate.dimensions = [dim_x]

        if self.bus is not None:
            self.bus.changedGatingHierarchy.emit(self.mode, roi.label.gate_name)
        # print([gate, gate.dimensions, gate.dimensions[0].min, gate.dimensions[0].max])

    def new_quadrant_gate(self):
        # set name
        name_increment = 1
        while f'Q{name_increment}' in [g[0] for g in self.gating.get_gate_ids()]:
            name_increment += 1
        gate_name = f'Q{name_increment}'

        # create ROI
        x = 0.5
        y = 0.5

        roi = QuadROI(x, y, gate_name, self.gating, self.mode, self.vb)
        roi.sigRemoveRequested.connect(self.remove_gate_and_roi)
        roi.sigPosChanged.connect(lambda: self.update_quad(roi))
        self.rois.append(roi)

        # define and add gate to gating
        quad_divs, quadrants = define_quad_gates(x, y, self.plot['channel_x'], self.plot['channel_y'], self.transformations)
        gate = Gates.QuadrantGate(gate_name, dividers=quad_divs, quadrants=quadrants)
        self.gating.add_gate(gate, gate_path=self.get_gate_path())
        self.plot['child_gates'].append(gate_name)
        if self.bus is not None:
            self.bus.updateSourceChildGates.emit(self.mode, gate_name)
            self.bus.changedGatingHierarchy.emit(self.mode, gate_name)

    def update_quad(self, roi):
        x = roi.vx.value()
        y = roi.vy.value()
        quad_divs, quadrants = define_quad_gates(x, y, self.plot['channel_x'], self.plot['channel_y'], self.transformations)

        gate = self.gating.get_gate(roi.gate_name)
        gate.quadrants = {q.id: q for q in quadrants}
        gate.dimensions = quad_divs

        if self.bus is not None:
            self.bus.changedGatingHierarchy.emit(self.mode, roi.gate_name)
        # print([q[1]._divider_ranges for q in gate.quadrants.items()])

    def remove_gate_and_roi(self):
        """Remove ROI and associated gate"""
        if self.sender() is None: # i.e. called directly
            rois = self.rois.copy()  # copy list to make sure elements are not skipped
        else:
            if hasattr(self.sender(), 'text'):
                if self.sender().text() == 'Delete Child Gates':
                    rois = self.rois.copy() # copy list to make sure elements are not skipped
                else:
                    rois = []
            else:
                rois = [self.sender()]

        for roi in rois:
            gate_name = roi.label.gate_name
            try:
                parent_name = self.gating.get_parent_gate_id(gate_name)
                if parent_name is None:
                    parent_name = 'root'
                else:
                    parent_name = parent_name[0]

                try:
                    self.gating.remove_gate(gate_name, keep_children=True)
                    roi.request_remove(delete_gate=False) # just to delete the roi elements, don't clobber deletion of the gate
                    self.rois.remove(roi)
                    self.plot['child_gates'].remove(gate_name)

                    if self.bus is not None:
                        self.bus.updateSourceChildGates.emit(self.mode, parent_name)
                        self.bus.changedGatingHierarchy.emit(self.mode, parent_name)
                    self.configure_axes()
                    print(f'CytometryPlotWidget {self.mode} {self.n_in_plot_sequence}: removed gate {gate_name}')
                except (GateReferenceError, ValueError, NetworkXError) as e:
                    print(f'Already deleted or invalid gate: {e}')
                except RuntimeError:
                    pass

            except GateReferenceError as e:
                print(f'Already deleted: {e}')
            except ValueError as e:
                print(f'Already deleted: {e}')

    def apply_zoom(self, axis_name):
        if axis_name == 'x':
            axis = self.axis_bottom
        else: # axisname == 'y':
            axis = self.axis_left

        if axis._pending_delta == 0:
            return

        # Accumulate small changes
        threshold = 1  # pixels
        step = axis._pending_delta
        axis._pending_delta = 0
        zoom_rate = 1.04  # tune this

        if abs(step) < threshold:
            return

        if step > 0:
            factor = 1 / zoom_rate
        else:
            factor = zoom_rate

        if axis_name == 'x':
            vb_range_ind = 0
            if self.plot['type'] == 'hist2d' or self.plot['type'] == 'hist1d':
                channel = self.plot['channel_x']
            elif self.plot['type'] == 'ribbon':
                channel = None
            else:
                channel = None
            axis_vb_set_range = axis.vb.setXRange
            axis_vb_map_to_view = self.vb.mapToView(axis.initial_pos).x()
        else: # axis_name == 'y':
            vb_range_ind = 1
            if self.plot['type'] == 'hist2d':
                channel = self.plot['channel_y']
            elif self.plot['type'] == 'hist1d':
                channel = None
            elif self.plot['type'] == 'ribbon':
                channel = 'ribbon'
            else:
                channel = None
            axis_vb_set_range = axis.vb.setYRange
            axis_vb_map_to_view = self.vb.mapToView(axis.initial_pos).y()
            factor = 1/factor

        if channel is not None:
            min, max = axis.vb.viewRange()[vb_range_ind]
            if self.transformations[channel].id == 0 or self.transformations[channel].id == 2: # linear or log
                new_max = (max - axis.zoomZero) * factor + axis.zoomZero
                new_min = (min - axis.zoomZero) * factor + axis.zoomZero
                if new_max < axis.fullRange[1] * 1.01:
                    axis_vb_set_range(new_min, new_max, padding=0)
                axis.limits = (new_min, new_max)
                self.transformations[channel].set_transform(limits=axis.limits)
            elif self.transformations[channel].id == 1:
                # scale w in bottom half, limits in top half
                if axis_vb_map_to_view < 0.5 * max:
                    self.transformations[channel].logicle_w = self.transformations[channel].logicle_w / factor
                    self.transformations[channel].set_transform()
                else:
                    new_max = (max - axis.zoomZero) * factor + axis.zoomZero
                    new_min = (min - axis.zoomZero) * factor + axis.zoomZero
                    if new_max < axis.fullRange[1] * 1.01:
                        axis_vb_set_range(new_min, new_max, padding=0)
                    axis.limits = (new_min, new_max)
                    self.transformations[channel].set_transform(limits=axis.limits)

            axis.zoomZero = self.transformations[channel].zero
            axis.setTicks(self.transformations[channel].ticks())

        if self.bus is not None:
            self.bus.axisTransformed.emit(channel)
        else:
            warnings.warn('Signals bus not connected')

    def plot_histogram(self):
        histograms = self.data_for_cytometry_plots.get('histograms', [])
        if self.n_in_plot_sequence >= len(histograms):
            return
        if self.plot['type'] == 'ribbon':
            self.plot_ribbon_plot()
        elif self.plot['type'] == 'hist1d':
            self.plot_hist1d()
        elif self.plot['type'] == 'hist2d':
            self.plot_hist2d()

    def plot_ribbon_plot(self):
        heatmap = self.data_for_cytometry_plots['histograms'][self.n_in_plot_sequence]
        self.img.setImage(heatmap.T)

        # Set the position and scale of the image
        x0 = 0
        x1 = len(self.fluoro_indices)
        y0 = self.transformations['ribbon'].limits[0]
        y1 = self.transformations['ribbon'].limits[1]
        self.img.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))

    def plot_hist2d(self):
        heatmap = self.data_for_cytometry_plots['histograms'][self.n_in_plot_sequence]
        # Guard against a transient plot/histogram mismatch (e.g. mid template
        # switch, where this index briefly holds a histogram for a different
        # plot type/scale). Skip drawing; the next recalculation redraws.
        if getattr(heatmap, 'ndim', 0) != 2:
            return
        self.img.setImage(heatmap)

        # Set the position and scale of the image
        x0 = self.transformations[self.plot['channel_x']].limits[0]
        x1 = self.transformations[self.plot['channel_x']].limits[1]
        y0 = self.transformations[self.plot['channel_y']].limits[0]
        y1 = self.transformations[self.plot['channel_y']].limits[1]
        self.img.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))

    def plot_hist1d(self):
        count = self.data_for_cytometry_plots['histograms'][self.n_in_plot_sequence]
        step_scale = self.transformations[self.plot['channel_x']].step_scale
        # Guard against a transient mismatch (e.g. mid template switch) where the
        # histogram at this index was computed for a different plot/scale.
        # pyqtgraph step mode needs len(x) == len(y)+1; skip if inconsistent.
        if getattr(count, 'ndim', 0) != 1 or len(step_scale) != len(count) + 1:
            return
        self.hist.setData(step_scale, count)

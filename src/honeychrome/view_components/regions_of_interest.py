import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF, QObject, QLineF, QPoint, QTimer, Slot, QEvent
from PySide6.QtGui import QFont, QCursor
from PySide6.QtWidgets import QVBoxLayout, QMenu, QDialog, QLineEdit, QDialogButtonBox, QMessageBox, QApplication
import pyqtgraph as pg


import warnings
import logging

from honeychrome.settings import label_offset_default, roi_handle_size
from honeychrome.controller_components.functions import rename_label_offset

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="t.core.qobject.connect: QObject::connect(QStyleHints, QStyleHints): unique connections require a pointer to member function of a QObject subclass")

def clip_position(x, y):
    return (max(0.1, min(x, 0.9)), max(0.1, min(y, 0.9)))

class LabelEditDialog(QDialog):
    """Minimal dialog with just a QLineEdit for editing text."""
    def __init__(self, text="", existing_names=[], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Label")
        self.setModal(True)
        self.resize(250, 80)
        self.existing_names = existing_names
        self.old_name = text

        layout = QVBoxLayout(self)

        self.line_edit = QLineEdit(self)
        self.line_edit.setText(text)
        self.line_edit.selectAll()   # auto-select text
        self.line_edit.setFocus()    # auto-focus
        layout.addWidget(self.line_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal, self
        )
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.move(QCursor.pos()) # doesn't work on linux/wayland?


    def validate_and_accept(self):
        text = self.line_edit.text().strip()
        text = text.replace('/', '∕') # slash not allowed by flowkit
        if not text:
            QMessageBox.warning(self, "Error", "Input cannot be empty.")
        elif text == self.old_name:
            self.reject()
        elif text in self.existing_names:
            QMessageBox.warning(self, "Error", f'"{text}" already exists.')
        elif '/' in text:
            QMessageBox.warning(self, "Error", f'Character "/" not allowed in gate name.')
        else:
            self.line_edit.setText(text)
            self.accept()

    def getText(self):
        return self.line_edit.text()


class DraggableRoiLabel(pg.TextItem):
    """A text label that can be dragged and renamed via small dialog."""
    def __init__(self, parent_roi, gate_name, gating, mode, pos=(0, 0), anchor=(0, 0)):
        self.parent_roi = parent_roi
        self.gate_name = gate_name
        self.gating = gating
        self.mode = mode
        self.bus = self.parent_roi.vb.parent().bus
        self.data_for_cytometry_plots = self.parent_roi.vb.parent().data_for_cytometry_plots
        super().__init__(self.gate_name, anchor=anchor)
        self.setPos(*pos)
        self.setZValue(100)

        # Style: bold + bigger font
        font = QFont()
        # font.setStyleHint(QFont.StyleHint.SansSerif)
        font.setPointSize(12)
        font.setWeight(QFont.Weight.Bold)
        self.setFont(font)
        self.setColor("k")
        self.fill = pg.mkBrush(0, 255, 0, 128)
        self._apply_custom_fill()

        self.setFlag(self.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(self.GraphicsItemFlag.ItemIsSelectable, True)

        if self.bus is not None:
            self.bus.histsStatsRecalculated.connect(self.add_statistic_to_name)

    @property
    def current_sample_id(self):
        """The loaded sample, read live from the plot widget.

        ``data_for_cytometry_plots`` is *rebound* to a different dict when the
        mode changes, so the copy taken in ``__init__`` can go stale; ask the
        widget each time instead.
        """
        try:
            data = self.parent_roi.vb.parent().data_for_cytometry_plots
        except Exception:
            data = self.data_for_cytometry_plots
        return (data or {}).get('sample_id')

    def is_customised(self):
        """True if this gate is overridden for the loaded sample."""
        sample_id = self.current_sample_id
        if not sample_id:
            return False
        try:
            return bool(self.gating.is_custom_gate(sample_id, self.gate_name))
        except Exception as e:
            logger.debug(f'is_customised: {self.gate_name!r} for {sample_id} ({e})')
            return False

    def paint(self, p, *args):
        """Draw background behind text."""
        rect = self.boundingRect().adjusted(-2, -2, 2, 2)
        p.setBrush(self.fill)
        p.setPen(pg.mkPen("k"))
        p.drawRect(rect)
        super().paint(p, *args)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        roi_pos = self.parent_roi.pos()
        offset = (
            self.pos().x() - roi_pos.x(),
            self.pos().y() - roi_pos.y()
        )
        self.parent_roi.label_offset = offset

        if self.bus:
            self.bus.updateChildGateLabelOffset.emit(self.gate_name, offset)

    def mouseDoubleClickEvent(self, event):
        """Open small dialog for editing text."""
        gate_names = ['root'] + [g[0] for g in self.gating.get_gate_ids()]
        dlg = LabelEditDialog(self.gate_name, existing_names=gate_names, parent=self.parent_roi.vb.parent())
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_name = dlg.getText()
            if new_name.strip():
                self.setText(new_name)
                self.rename_gate(new_name)

    @Slot()
    def add_statistic_to_name(self):
        self._apply_custom_fill()
        if self.data_for_cytometry_plots['statistics']:
            if self.gate_name in self.data_for_cytometry_plots['statistics'].keys():
                self.setText(f'{self.gate_name}: {self.data_for_cytometry_plots['statistics'][self.gate_name]['p_gate_parent']*100:.2f}%')

    def _apply_custom_fill(self):
        """Tint the label AND the gate outline orange when this gate is
        customised for the current sample (a per-sample custom gate), else the
        default green (template)."""
        customised = self.is_customised()
        self.fill = pg.mkBrush(255, 165, 0, 160) if customised else pg.mkBrush(0, 255, 0, 128)
        self.update()

        # Recolour the gate outline itself, not just the label. Rectangle /
        # ellipse / polygon expose setPen; QuadROI (vx/vy) and RangeROI (v1/v2)
        # draw their edges as InfiniteLines, so recolour those too.
        pen = pg.mkPen((255, 165, 0) if customised else 'g', width=3)
        roi = self.parent_roi
        try:
            roi.setPen(pen)
        except Exception:
            pass
        for attr in ('vx', 'vy', 'v1', 'v2'):
            line = getattr(roi, attr, None)
            if line is not None:
                try:
                    line.setPen(pen)
                except Exception:
                    pass

    def move_label_with_roi(self):
        roi_pos = self.parent_roi.pos()
        new_label_pos = (roi_pos.x() + self.parent_roi.label_offset[0],
                         roi_pos.y() + self.parent_roi.label_offset[1])
        # print(f'new_label_pos{new_label_pos}')
        self.setPos(*new_label_pos)

    def rename_gate(self, new_name):
        if self.gating.find_matching_gate_paths(new_name):
            raise Exception(f"gate name {new_name} already exists")
        else:
            old_name = self.gate_name
            self.gating.rename_gate(old_name, new_name)
            self.gate_name = new_name
            # Update child_gates references in all plots
            for plot in self.data_for_cytometry_plots['plots']:
                if old_name in plot['child_gates']:
                    idx = plot['child_gates'].index(old_name)
                    plot['child_gates'][idx] = new_name
                    rename_label_offset(plot, old_name, new_name)
            # If this is a raw-mode rename, propagate to the spectral model's gate_label
            # so that generate() and Recalculate can still find the gate.
            mode = self.parent_roi.vb.parent().mode
            if mode == 'raw' and self.bus is not None:
                self.bus.rawGateRenamed.emit(old_name, new_name)
            if self.bus is not None:
                self.bus.updateSourceChildGates.emit(mode, new_name)
                self.bus.changedGatingHierarchy.emit(mode, new_name)

        # print(self.gating.get_gate_ids())

def _build_custom_gate_actions(menu, label):
    """Append the per-sample custom gate actions to ``menu``, mirroring the
    gating hierarchy tree's context menu, and return what was added.

    ``label`` is the ROI's :class:`DraggableRoiLabel`, which knows the gate name,
    the gating strategy, the mode and the event bus. With no sample loaded there
    is nothing to customise *for*, so the action is shown greyed out rather than
    hidden — otherwise the menu looks as though the feature is missing.
    """
    bus = getattr(label, 'bus', None)
    gate_name = getattr(label, 'gate_name', None)
    if bus is None or not gate_name or gate_name == 'root':
        return []

    mode = label.mode
    added = [menu.addSeparator()]
    if label.is_customised():
        # emit label.gate_name (not the captured name) so a rename still works
        added.append(menu.addAction(
            f"Revert '{gate_name}' to template",
            lambda: bus.revertGateRequested.emit(mode, label.gate_name)))
        added.append(menu.addAction(
            f"Adopt '{gate_name}' custom gate as template",
            lambda: bus.adoptGateRequested.emit(mode, label.gate_name)))
    else:
        action = menu.addAction(
            f"Customise '{gate_name}' for this sample",
            lambda: bus.customiseGateRequested.emit(mode, label.gate_name))
        action.setEnabled(bool(label.current_sample_id))
        added.append(action)
    return added


def exec_gate_menu(menu, label, screen_pos):
    """Show an ROI's context menu with fresh per-sample custom gate actions.

    Those actions are rebuilt on every right-click because whether the gate is
    customised for the current sample changes as the user works, and the menus
    themselves are built once in each ROI's ``__init__``.
    """
    for action in getattr(menu, '_custom_gate_actions', ()):
        menu.removeAction(action)
    menu._custom_gate_actions = _build_custom_gate_actions(menu, label)
    menu.exec(screen_pos)


class ContextMenuTargetItem(pg.TargetItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create context menu
        self.menu = QMenu()
        self.action_remove = self.menu.addAction("Delete Gate")
        self.gate_label = None  # set by the owning ROI once its label exists

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton:
            exec_gate_menu(self.menu, self.gate_label, ev.screenPos().toPoint())
            ev.accept()
        else:
            super().mouseClickEvent(ev)


class QuadROI(pg.ROI):
    sigPosChanged = Signal(float, float)  # min_x, max_x

    def __init__(self, x, y, gate_name, gating, mode, vb):
        pen = pg.mkPen('g', width=3)
        super().__init__((0,0), pen=pen)
        self.vx = pg.InfiniteLine(pos=x, angle=90, movable=True, pen=pen)
        self.vy = pg.InfiniteLine(pos=y, angle=0, movable=True, pen=pen)

        # Connect movement signals
        self.vx.sigPositionChangeFinished.connect(self._line_moved)
        self.vy.sigPositionChangeFinished.connect(self._line_moved)
        self.vb = vb
        self.vb.addItem(self.vx)
        self.vb.addItem(self.vy)
        self.addFreeHandle([0, 0])

        self.target = ContextMenuTargetItem(pos = (x, y), size = 20, pen = 'c', symbol = 's')
        self.vb.addItem(self.target)
        self.target.sigPositionChangeFinished.connect(self._target_moved)
        self.target.action_remove.triggered.connect(self.request_remove)

        # create labels
        self.gate_name = gate_name
        xlim, ylim = self.vb.viewRange()
        self.label = DraggableRoiLabel(self, gate_name, gating, mode, pos=(xlim[0], ylim[0]), anchor=(0, 1))
        self.vb.addItem(self.label)
        self.target.gate_label = self.label
        # self.label = [
        #     DraggableRoiLabel(self, gate_name +'++', gating, mode, pos=(xlim[1], ylim[1]), anchor=(1, 0)),
        #     DraggableRoiLabel(self, gate_name +'+-', gating, mode, pos=(xlim[1], ylim[0]), anchor=(1, 1)),
        #     DraggableRoiLabel(self, gate_name +'-+', gating, mode, pos=(xlim[0], ylim[1]), anchor=(0, 0)),
        #     DraggableRoiLabel(self, gate_name +'--', gating, mode, pos=(xlim[0], ylim[0]), anchor=(0, 1)),
        # ]
        # [self.vb.addItem(label) for label in self.label]

    def _line_moved(self, line):
        # Update region when a line moves
        x = self.vx.value()
        y = self.vy.value()
        self.target.setPos((x,y))
        self.sigPosChanged.emit(x, y)

    def _target_moved(self, target):
        # Update region when target moves
        pos = target.pos()
        x = pos.x()
        y = pos.y()
        self.vx.setPos(x)
        self.vy.setPos(y)
        self.sigPosChanged.emit(x, y)

    def pos(self):
        x = self.vx.value()
        y = self.vy.value()
        return QPointF(x,y)

    def request_remove(self, delete_gate=True):
        """Emit sigRemoveRequested (same as built-in ROI behavior)."""
        self.vb.removeItem(self.vx)
        self.vb.removeItem(self.vy)
        self.vb.removeItem(self.target)
        if delete_gate:
            self.sigRemoveRequested.emit(self)
        self.vb.removeItem(self.label)


class ContextMenuRangeRegion(pg.LinearRegionItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create context menu
        self.menu = QMenu()
        self.action_remove = self.menu.addAction("Delete Gate")
        self.gate_label = None  # set by the owning ROI once its label exists

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton:
            exec_gate_menu(self.menu, self.gate_label, ev.screenPos().toPoint())
            ev.accept()
        else:
            super().mouseClickEvent(ev)


class RangeROI(pg.ROI):
    sigRangeChanged = Signal(float, float)  # min_x, max_x

    def __init__(self, x1, x2, gate_name, gating, mode, vb, label_offset=None):
        pen = pg.mkPen('g', width=3)
        handle_size = roi_handle_size
        super().__init__((0,0), pen=pen, removable=True)
        self.v1 = pg.InfiniteLine(pos=x1, angle=90, movable=True, pen=pen)
        self.v2 = pg.InfiniteLine(pos=x2, angle=90, movable=True, pen=pen)

        # Connect movement signals
        self.v1.sigPositionChangeFinished.connect(self._line_moved)
        self.v2.sigPositionChangeFinished.connect(self._line_moved)
        self.region = ContextMenuRangeRegion(values=(x1, x2))
        self.vb = vb
        self.vb.addItem(self.region)
        self.vb.addItem(self.v1)
        self.vb.addItem(self.v2)
        self.region.sigRegionChangeFinished.connect(self._region_moved)
        # create label
        if label_offset:
            self.label_offset = label_offset
        else:
            self.label_offset = label_offset_default #'(x1 + label_offset_default[0], 1 + label_offset_default[1])
        self.label_pos = clip_position(x1+self.label_offset[0], self.label_offset[1])
        self.label = DraggableRoiLabel(self, gate_name, gating, mode, pos=self.label_pos, anchor=(0, 1))
        self.vb.addItem(self.label)
        self.region.gate_label = self.label
        self.region.sigRegionChanged.connect(self.label.move_label_with_roi)

        # Connect actions
        self.region.action_remove.triggered.connect(self.request_remove)

    def _line_moved(self, line):
        # Update region when a line moves
        min_x = min(self.v1.value(), self.v2.value())
        max_x = max(self.v1.value(), self.v2.value())
        self.region.setRegion((min_x, max_x))
        self.sigRangeChanged.emit(min_x, max_x)

    def _region_moved(self):
        # Update lines if the region is dragged
        min_x, max_x = self.region.getRegion()
        self.v1.setPos(min_x)
        self.v2.setPos(max_x)
        self.sigRangeChanged.emit(min_x, max_x)

    def setRange(self, x1, x2):
        self.v1.setPos(x1)
        self.v2.setPos(x2)
        self.region.setRegion((x1, x2))
        self.sigRangeChanged.emit(x1, x2)

    def pos(self):
        min_x, max_x = self.region.getRegion()
        return QPointF(min_x,1)

    def request_remove(self, delete_gate=True):
        """Emit sigRemoveRequested (same as built-in ROI behavior)."""
        self.vb.removeItem(self.region)
        self.vb.removeItem(self.v1)
        self.vb.removeItem(self.v2)
        if delete_gate:
            self.sigRemoveRequested.emit(self)
        self.vb.removeItem(self.label)


class PolygonROI(pg.PolyLineROI):
    def __init__(self, positions, gate_name, gating, mode, vb, label_offset=None):
        self.handleSize = roi_handle_size  # Set before super().__init__()
        super().__init__(
                positions = positions,
                closed = True,
                pen = pg.mkPen('g', width=3),
                movable = True,
                removable = True)
        self.vb = vb
        vertices = np.array(positions)
        xmin = vertices[:, 0].min()
        xmax = vertices[:, 0].max()
        ymin = vertices[:, 1].min()
        ymax = vertices[:, 1].max()
        # create label
        if label_offset:
            self.label_offset = label_offset
        else:
            self.label_offset = ((xmax-xmin + label_offset_default[0]) / 2, (ymax-ymin + label_offset_default[1]) / 2)
        self.label_pos = clip_position(self.pos().x() + self.label_offset[0], self.pos().y() + self.label_offset[1])
        self.label = DraggableRoiLabel(self, gate_name, gating, mode, pos=self.label_pos)
        self.sigRegionChanged.connect(self.label.move_label_with_roi)
        self.vb.addItem(self.label)

        # Create context menu
        self.menu = QMenu()
        self.action_new_plot_on_gate = self.menu.addAction("New Plot From This Gate", lambda : self.vb.parent().new_plot_on_gate(self.label.gate_name))
        self.action_delete = self.menu.addAction("Delete Gate", self.request_remove)
        self.vb.addItem(self)

        for segment in self.segments:
            segment.mouseClickEvent = lambda event, s=segment: self.segmentClickEvent(event, s)

    def addHandle(self, *args, **kwargs):
        self.handleSize = roi_handle_size
        return super().addHandle(*args, **kwargs)

    def mouseClickEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            # Show context menu at cursor position
            exec_gate_menu(self.menu, self.label, event.screenPos().toPoint())
            event.accept()
        else:
            # Keep normal ROI drag/resize behavior
            super().mouseClickEvent(event)

    def segmentClickEvent(self, event, segment):
        if event.button() == Qt.MouseButton.RightButton:
            event.accept()
            exec_gate_menu(self.menu, self.label, event.screenPos().toPoint())
        else:
            pg.LineSegmentROI.mouseClickEvent(segment, event)

    def request_remove(self, delete_gate=True):
        """Emit sigRemoveRequested (same as built-in ROI behavior)."""
        if delete_gate:
            self.sigRemoveRequested.emit(self)
        self.vb.removeItem(self.label)
        self.vb.removeItem(self)


class RectangleROI(pg.RectROI):

    def __init__(self, pos, size, gate_name, gating, mode, vb, label_offset=None):
        self.handleSize = roi_handle_size  # Set before super().__init__()
        super().__init__(pos, size, pen = pg.mkPen('g', width=3), movable = True, removable = True)
        self.vb = vb
        # create label
        if label_offset:
            self.label_offset = label_offset
        else:
            self.label_offset = (label_offset_default[0] + size[0] / 2, label_offset_default[1] + size[1] / 2)
        self.label_pos = clip_position(pos[0] + self.label_offset[0], pos[1] + self.label_offset[1])
        self.label = DraggableRoiLabel(self, gate_name, gating, mode, pos=self.label_pos)
        self.sigRegionChanged.connect(self.label.move_label_with_roi)
        self.vb.addItem(self.label)

        # Create context menu
        self.menu = QMenu()
        self.action_new_plot_on_gate = self.menu.addAction("New Plot From This Gate", lambda : self.vb.parent().new_plot_on_gate(self.label.gate_name))
        self.action_delete = self.menu.addAction("Delete Gate", self.request_remove)

        self.vb.addItem(self)

    def addHandle(self, *args, **kwargs):
        self.handleSize = roi_handle_size
        return super().addHandle(*args, **kwargs)

    def mouseClickEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            # Show context menu at cursor position
            exec_gate_menu(self.menu, self.label, event.screenPos().toPoint())
            event.accept()
        else:
            # Keep normal ROI drag/resize behavior
            super().mouseClickEvent(event)

    def request_remove(self, delete_gate=True):
        """Emit sigRemoveRequested (same as built-in ROI behavior)."""
        if delete_gate:
            self.sigRemoveRequested.emit(self)
        self.vb.removeItem(self.label)
        self.vb.removeItem(self)


class EllipseROI(pg.EllipseROI):
    def __init__(self, pos, size, angle, gate_name, gating, mode, vb, label_offset=None):
        self.handleSize = roi_handle_size  # Set before super().__init__()
        super().__init__(pos, size, angle=angle, pen=pg.mkPen('g', width=3), movable=True, removable=True)
        self.vb = vb

        # create label
        if label_offset:
            self.label_offset = label_offset
        else:
            self.label_offset = label_offset_default
        self.label_pos = clip_position(pos[0] + self.label_offset[0], pos[0] + self.label_offset[1])
        self.label = DraggableRoiLabel(self, gate_name, gating, mode, pos=self.label_pos)
        self.sigRegionChanged.connect(self.label.move_label_with_roi)
        self.vb.addItem(self.label)

        # Create context menu
        self.menu = QMenu()
        self.action_new_plot_on_gate = self.menu.addAction("New Plot From This Gate", lambda : self.vb.parent().new_plot_on_gate(self.label.gate_name))
        self.action_delete = self.menu.addAction("Delete Gate", self.request_remove)

        self.vb.addItem(self)


    def addHandle(self, *args, **kwargs):
        self.handleSize = roi_handle_size
        return super().addHandle(*args, **kwargs)

    def mouseClickEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            # Show context menu at cursor position
            exec_gate_menu(self.menu, self.label, event.screenPos().toPoint())
            event.accept()
        else:
            # Keep normal ROI drag/resize behavior
            super().mouseClickEvent(event)

    def request_remove(self, delete_gate=True):
        """Emit sigRemoveRequested (same as built-in ROI behavior)."""
        if delete_gate:
            self.sigRemoveRequested.emit(self)
        self.vb.removeItem(self.label)
        self.vb.removeItem(self)

class PolygonROIConstructor(QObject):
    def __init__(self, parent):
        super().__init__(parent)
        # Will hold our polygon ROI
        self.connector = None
        self.poly_roi = None
        self.vertex_plot = pg.ScatterPlotItem(pen='r', brush='g', size=roi_handle_size, symbol='s')
        self.parent().vb.addItem(self.vertex_plot)

        # Connect mouse events
        self.parent()._mouse_events_enabled = False
        self.vertices = []
        self.parent().vb.setCursor(Qt.CursorShape.CrossCursor)
        self.parent().vb.scene().sigMouseClicked.connect(self.drawing_polygon_handle_click)

    def drawing_polygon_handle_click(self, event):
        pos = self.parent().vb.mapSceneToView(event.scenePos())
        # print(pos)
        if event.button() == Qt.MouseButton.LeftButton:
            # Add vertex to current polygon
            self.vertices.append(pos)
            self.drawing_polygon_update_vertex_display()

            if event._double:
                if len(self.vertices) >= 4:
                    # Finish polygon
                    # remove last point which is duplicated
                    self.vertices = self.vertices[:-1]
                    # remove another point if it coincides with first point
                    length = QLineF(self.vertices[0], self.vertices[-1]).length()
                    if length < 0.1:
                        self.vertices = self.vertices[:-1]

                    # Remove temporary connector
                    self.parent()._mouse_events_enabled = True
                    self.parent().vb.scene().sigMouseClicked.disconnect(self.drawing_polygon_handle_click)
                    if self.connector is not None:
                        self.parent().vb.removeItem(self.connector)
                        self.parent().vb.removeItem(self.vertex_plot)

                    self.parent().create_polygon_gate(self.vertices)
                    self.parent().vb.setCursor(Qt.CursorShape.ArrowCursor)
                else:
                    self.parent().bus.warningMessage.emit("Need at least 3 vertices to create polygon")

    def drawing_polygon_update_vertex_display(self):
        """Show the vertices being added"""
        xs = [v.x() for v in self.vertices]
        ys = [v.y() for v in self.vertices]
        self.vertex_plot.setData(x=xs, y=ys)

        # Draw connecting lines
        if self.connector is not None:
            self.parent().vb.removeItem(self.connector)
        self.connector = pg.PlotCurveItem(
            x=xs, y=ys,
            pen=pg.mkPen('y', width=2, style=Qt.PenStyle.DashLine)
        )
        self.parent().vb.addItem(self.connector)

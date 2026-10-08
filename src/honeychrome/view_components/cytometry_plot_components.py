from PySide6.QtCore import Qt, QTimer, QPointF, QRectF, QObject, QEvent, QPoint
from PySide6.QtGui import QFont, QPen, QColor, QCursor, QWheelEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QMenu, QScrollArea
import pyqtgraph as pg

import logging
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", message="t.core.qobject.connect: QObject::connect(QStyleHints, QStyleHints): unique connections require a pointer to member function of a QObject subclass")

# pyqtgraph sizes a bottom axis for horizontal text, so rotated detector labels
# get clipped. Enough room for a 45-degree detector name.
ROTATED_LABEL_HEIGHT = 60
TICK_TEXT_OFFSET = 10          # gap between the axis line and the labels

class ZoomAxis(pg.AxisItem):
    def __init__(self, orientation, viewbox, angle=0, **kwargs):
        self.angle = angle
        self._label_padding = 15
        super().__init__(orientation, **kwargs)

        self.setAcceptHoverEvents(True)  # needed for hover detection
        self.vb = viewbox
        self.orientation = orientation
        self.initial_pos = None
        self._last_pos = None
        self._pending_delta = 0
        self.zoom_timer = QTimer()
        self.zoom_timer.setInterval(60)  # update rate in ms
        self.zoomZero = 0
        self.fullRange = (0, 1.1)
        self.limits = (0, 1)

        # Give extra space by default to prevent clipping
        self.setStyle(tickTextOffset=30, tickLength=5)

    def setTicks(self, ticks, angle=0):
        """Override setTicks to optionally update rotation angle."""
        if angle is not None:
            self.angle = angle
            self.setStyle(tickTextOffset=TICK_TEXT_OFFSET)
        super().setTicks(ticks)
        if self.orientation == 'bottom':
            self.setHeight(ROTATED_LABEL_HEIGHT if self.angle and ticks else None)
        self.updateGeometry()
        self.update()

    def _labels_are_rotated(self):
        """Only the explicitly set detector labels are drawn at an angle; the
        automatic numeric ticks of an empty plot stay horizontal."""
        return self.orientation == 'bottom' and self.angle and self._tickLevels

    def generateDrawSpecs(self, p):
        if not self._labels_are_rotated():
            return super().generateDrawSpecs(p)
        # Rotated labels extend past the axis rect, so widen it for pyqtgraph's
        # edge-cull check or the first and last detectors lose their label.
        orig_boundingRect = self.boundingRect
        self.boundingRect = lambda: orig_boundingRect().adjusted(-500, 0, 500, 0)
        try:
            return super().generateDrawSpecs(p)
        finally:
            self.boundingRect = orig_boundingRect

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        # Draw tick lines normally (skip text)
        super().drawPicture(p, axisSpec, tickSpecs, [])

        tick_colors = getattr(self, 'tick_colors', {})   # safe even if called before __init__ finishes
        default_pen = p.pen()
        rotated = self._labels_are_rotated()

        p.save()
        try:
            for rect, flags, text in textSpecs:
                p.save()
                try:
                    color = tick_colors.get(text)
                    p.setPen(QPen(color) if color else default_pen)

                    if rotated:
                        # Anchor at the tick and let the label trail away down-left,
                        # so labels never overlap each other.
                        p.translate(QPointF(rect.center().x(), rect.top()))
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


    def hoverEnterEvent(self, event):
        QApplication.setOverrideCursor(Qt.CursorShape.PointingHandCursor)

    def hoverLeaveEvent(self, event):
        QApplication.restoreOverrideCursor()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.initial_pos = ev.pos()
            self._last_pos = self.initial_pos
            self._pending_delta = 0
            self.zoom_timer.start()
            ev.accept()
        else:
            ev.ignore()

    def mouseMoveEvent(self, ev):
        if self._last_pos is None:
            ev.ignore()
            return

        delta = ev.pos() - self._last_pos
        self._last_pos = ev.pos()

        if self.orientation == 'bottom':
            self._pending_delta += delta.x()
        elif self.orientation == 'left':
            self._pending_delta += delta.y()

        ev.accept()

        if QApplication.overrideCursor() == Qt.CursorShape.PointingHandCursor:
            QApplication.setOverrideCursor(Qt.CursorShape.ClosedHandCursor)


    def mouseReleaseEvent(self, ev):
        self._last_pos = None
        self._pending_delta = 0
        self.zoom_timer.stop()
        ev.accept()

        QApplication.restoreOverrideCursor()

class TransparentGraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    def wheelEvent(self, ev):
        ev.ignore()

class NoPanViewBox(pg.ViewBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, enableMouse=True, **kwargs)
        self.setMouseMode(self.PanMode)
        self.setMouseEnabled(x=False, y=False)

    def mouseDragEvent(self, ev, axis=None):
        ev.ignore()  # 🔒 Disable all panning and dragging in plot area

    def wheelEvent(self, ev, axis=None):
        ev.ignore()  # 🔒 Disable mouse wheel zoom

class InteractiveLabel(pg.LabelItem):
    def __init__(self, text="", parent_plot=None, angle=0, **kwargs):
        self.angle = angle
        self._label_padding = 15
        self.tick_colors: dict = {}  
        super().__init__(text, angle=angle, **kwargs)
        self.setAcceptHoverEvents(True)
        self._default_font = self.item.font()
        self._hover_font = QFont(self._default_font)
        self._hover_font.setUnderline(True)
        self.leftClickMenuItems = []
        self.rightClickMenuItems = []
        self.leftItemSelected = None
        self.rightItemSelected = None
        self.leftClickMenuFunction = None
        self.rightClickMenuFunction = None
        self.parent_plot = parent_plot

    def hoverEnterEvent(self, event):
        self.item.setFont(self._hover_font)
        QApplication.setOverrideCursor(Qt.CursorShape.PointingHandCursor)

    def hoverLeaveEvent(self, event):
        self.item.setFont(self._default_font)
        QApplication.restoreOverrideCursor()

    def mousePressEvent(self, event):
        self.parent_plot.select_plot_on_parent_grid()
        if event.button() == Qt.MouseButton.RightButton:
            self.show_right_context_menu(event.screenPos())
        elif event.button() == Qt.MouseButton.LeftButton:
            self.selectable_menu_activates_function(event.screenPos())  # can choose to differentiate

    def selectable_menu_activates_function(self, pos):
        menu = QMenu()
        actions = {}

        for n, item in enumerate(self.leftClickMenuItems):
            if item == 'Count':
                menu.addSeparator()
                action = menu.addAction(item)
                actions[action] = item
            elif item == 'All Fluorescence':
                menu.addSeparator()
                action = menu.addAction(item)
                actions[action] = item
            else:
                action = menu.addAction(item)
                action.setCheckable(True)
                if n == self.leftItemSelected:
                    action.setChecked(True)
                actions[action] = n
        chosen_action = menu.exec(pos)

        if chosen_action:
            self.leftItemSelected = actions[chosen_action]
            func = self.leftClickMenuFunction
            func(self.leftItemSelected, self.parent())
            logger.info(f"{func} called with option {self.leftItemSelected} parent {self.parent()}")


    def show_right_context_menu(self, pos):
        menu = QMenu()
        actions = {}

        for n, item in enumerate(self.rightClickMenuItems):
            action = menu.addAction(item)
            action.setCheckable(True)
            if n == self.rightItemSelected:
                action.setChecked(True)
            actions[action] = n
        chosen_action = menu.exec(pos)

        if chosen_action:
            self.rightItemSelected = actions[chosen_action]
            func = self.rightClickMenuFunction
            func(self.rightItemSelected, self.parent())
            logger.info(f"{func} called with option {self.rightItemSelected} parent {self.parent()}")

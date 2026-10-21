"""
drc_workspace_canvas.py — page canvas for Honeycluster's Workspace tab
======================================================================
Companion module to ``dr_clustering_tab.py`` (not ``*_tab.py``, so the plugin
loader does not pick it up as a tab).

A QGraphicsScene whose unit is the millimetre. Pages (A4 by default) are
stacked vertically with a fixed gap. Everything on a page is a
``FigureItem``: an image of a matplotlib figure rendered by Agg at the
current zoom, so what is drawn on the page is what is exported.

- ``PlotItem``   a PlotCard's plot figure; resized from its edges (the plot
                 size, aspect ratio locked by default).
- ``LegendItem`` a PlotCard's separate legend, a child of its PlotItem (it
                 moves with the plot, and can be moved on its own); resizing
                 it reflows the entries into more or fewer columns.
- ``TextItem``   a text box or panel letter (``TextBox``); fixed font size,
                 sized to its text.
- ``SourceFigureItem`` a copy of a figure from another tab of the plugin
                 (``SourceFigure``: Cluster Annotation, Stats), resized with
                 its aspect ratio locked by default and updated on request.

Top-level items keep their layout relative to their page as
``{'page': int, 'x_mm': float, 'y_mm': float}``. Items can be grouped (a
group selects and moves as one, and scales from its frame's corner: plots
are re-rendered at the new size, text keeps its size). Moves, resizes,
group changes and scaling go through the scene's QUndoStack.

v1 simplifications: rendering is on the main thread; items resize from the
right edge, bottom edge and bottom-right corner only; snapping is to the
page grid (no alignment guides); text is edited in the inspector, not on
the page; page changes and plot/text settings are not undoable.
"""

from __future__ import annotations

import string

import numpy as np
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush, QColor, QCursor, QFont, QImage, QPainter, QPainterPath, QPen, QTransform,
    QUndoCommand, QUndoStack,
)
from PySide6.QtWidgets import (
    QApplication, QGraphicsItem, QGraphicsObject, QGraphicsRectItem,
    QGraphicsScene, QGraphicsView, QMenu,
)

MM_PER_IN = 25.4
PAGE_GAP_MM = 15.0
MIN_ITEM_MM = 20.0
MIN_LEGEND_MM = 3.0
PLACE_MARGIN_MM = 10.0
PLACE_GAP_MM = 5.0
PLACE_STEP_MM = 5.0
LEGEND_DOCK_GAP_MM = 2.0
DUPLICATE_OFFSET_MM = 10.0
LETTER_OFFSET_MM = 4.0
LETTER_ROW_MM = 10.0
TEXT_PAD_PT = 1.0
TEXT_STACK_GAP_MM = 2.0
SOURCE_FIGURE_MAX_FRACTION = 0.9     # of the page width, for a new source figure
SOURCE_FIGURE_IMAGE_DPI = 300        # copy as an image when a figure cannot be copied
RENDER_DELAY_MS = 120
LAYOUT_PASSES = 2
MIN_RENDER_DPI = 50.0
MAX_RENDER_DPI = 600.0
MAX_RENDER_PIXELS = 25_000_000
HANDLE_PX = 8.0
MIN_GROUP_SCALE = 0.2
MIN_ZOOM, MAX_ZOOM = 0.1, 8.0
ZOOM_STEP = 1.25
GRID_STEPS_MM = (0.5, 1.0, 2.0, 5.0)
DEFAULT_GRID_MM = 5.0
MAJOR_GRID_MM = 10.0
MIN_GRID_LINE_GAP_PX = 6.0
SELECTION_COLOUR = '#2f7de1'
GROUP_COLOUR = '#e07b1f'

# Page presets: (width, height) in mm, portrait. Column-width presets take a
# 247 mm default height, editable on the page.
PAGE_PRESETS = {
    'A4': (210.0, 297.0),
    'A5': (148.0, 210.0),
    'US Letter': (215.9, 279.4),
    '89 mm column': (89.0, 247.0),
    '120 mm (1.5 columns)': (120.0, 247.0),
    '183 mm (2 columns)': (183.0, 247.0),
    'Custom': None,
}
DEFAULT_PAGE = {'preset': 'A4', 'w_mm': 210.0, 'h_mm': 297.0}
DEFAULT_TEXT = {'text': 'Text', 'size_pt': 9.0, 'bold': False, 'italic': False,
                'colour': '#000000', 'align': 'left', 'kind': 'text', 'auto': False}
DEFAULT_LETTER = dict(DEFAULT_TEXT, text='A', size_pt=10.0, bold=True, kind='letter', auto=True)


# ----------------------------------------------------------------------
# Pure layout helpers (no widgets)
# ----------------------------------------------------------------------

def snap(value: float, step: float) -> float:
    """Nearest multiple of *step*."""
    return round(value / step) * step if step > 0 else value


def page_origins(pages: list[dict], gap: float = PAGE_GAP_MM) -> list[tuple[float, float]]:
    """Top-left scene position of each page (pages stacked downwards)."""
    origins, y = [], 0.0
    for p in pages:
        origins.append((0.0, y))
        y += float(p['h_mm']) + gap
    return origins


def page_index_at(pages: list[dict], y: float, gap: float = PAGE_GAP_MM) -> int:
    """Index of the page whose vertical band (page plus the gap below it)
    contains scene *y*; clamped to the first/last page."""
    origins = page_origins(pages, gap)
    for i, (_x0, y0) in enumerate(origins):
        if y < y0 + float(pages[i]['h_mm']) + gap:
            return i
    return len(pages) - 1


def _overlaps(a: tuple, b: tuple) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def free_spot(page_w: float, page_h: float, occupied: list[tuple], w: float, h: float,
              margin: float = PLACE_MARGIN_MM, gap: float = PLACE_GAP_MM,
              step: float = PLACE_STEP_MM):
    """First (x, y) in row-major order, on a *step* grid inside the page
    *margin*, where a w x h rectangle (plus *gap* clearance) overlaps none
    of *occupied* [(x, y, w, h), ...] (callers include the gap in those);
    None if nothing fits."""
    y = margin
    while y + h <= page_h - margin + 1e-9:
        x = margin
        while x + w <= page_w - margin + 1e-9:
            cand = (x, y, w + gap, h + gap)
            if not any(_overlaps(cand, o) for o in occupied):
                return x, y
            x += step
        y += step
    return None


def place_items(pages: list[dict], placed: list[dict], sizes: list[tuple[float, float]],
                start_page: int = 0) -> tuple[list[dict], list[dict]]:
    """Layouts for new items of *sizes* (w, h in mm), filling pages from
    *start_page* around the already *placed* layouts ({'page', 'x_mm',
    'y_mm', 'w_mm', 'h_mm'}). Adds pages (copies of the last page's size)
    when nothing fits; an item larger than a page goes at the margin of a
    page of its own. Returns (pages, layouts)."""
    pages = [dict(p) for p in pages] or [dict(DEFAULT_PAGE)]
    occupied = {i: [] for i in range(len(pages))}
    for lay in placed:
        occupied.setdefault(int(lay['page']), []).append(
            (lay['x_mm'], lay['y_mm'], lay['w_mm'] + PLACE_GAP_MM, lay['h_mm'] + PLACE_GAP_MM))
    layouts = []
    for w, h in sizes:
        page = min(max(start_page, 0), len(pages) - 1)
        while True:
            spot = free_spot(pages[page]['w_mm'], pages[page]['h_mm'],
                             occupied.get(page, []), w, h)
            if spot is None and not occupied.get(page):
                spot = (PLACE_MARGIN_MM, PLACE_MARGIN_MM)
            if spot is not None:
                break
            page += 1
            if page >= len(pages):
                pages.append(dict(pages[-1]))
                occupied[page] = []
        lay = {'page': page, 'x_mm': spot[0], 'y_mm': spot[1]}
        occupied.setdefault(page, []).append((spot[0], spot[1], w + PLACE_GAP_MM, h + PLACE_GAP_MM))
        layouts.append(lay)
    return pages, layouts


def grid_display_step(grid_mm: float, px_per_mm: float) -> float:
    """Spacing of the grid lines drawn: the snap step times the smallest of
    1, 2, 5, 10, 20, 50 … that puts lines at least MIN_GRID_LINE_GAP_PX apart
    on screen and keeps them in step with the MAJOR_GRID_MM lines (the
    spacing divides, or is a multiple of, MAJOR_GRID_MM). Every line drawn
    is a position items snap to."""
    def aligned(d):
        ratio = MAJOR_GRID_MM / d if d < MAJOR_GRID_MM else d / MAJOR_GRID_MM
        return abs(ratio - round(ratio)) < 1e-6
    decade = 1
    while True:
        for k in (1, 2, 5):
            d = grid_mm * k * decade
            if d * px_per_mm >= MIN_GRID_LINE_GAP_PX - 1e-9 and aligned(d):
                return d
        decade *= 10


def page_size_for(preset: str, orientation: str, custom: tuple[float, float]) -> tuple[float, float]:
    """(w, h) in mm for a preset and orientation; *custom* for 'Custom'."""
    size = PAGE_PRESETS.get(preset) or custom
    w, h = float(size[0]), float(size[1])
    if (orientation == 'Landscape' and w < h) or (orientation == 'Portrait' and w > h):
        w, h = h, w
    return w, h


def locked_size(w0: float, h0: float, w: float, h: float, edge: str) -> tuple[float, float]:
    """(w, h) keeping the w0:h0 ratio: the dragged edge decides (the corner
    follows whichever dimension changed more, relatively)."""
    ratio = w0 / h0
    if edge == 'right' or (edge == 'corner' and abs(w / w0 - 1) >= abs(h / h0 - 1)):
        return w, w / ratio
    return h * ratio, h


def letter_sequence(n: int, taken: set[str]) -> list[str]:
    """The first *n* upper-case letters (then AA, AB, ...) not in *taken*."""
    out, i = [], 0
    while len(out) < n:
        q, r = divmod(i, 26)
        label = (string.ascii_uppercase[q - 1] if q else '') + string.ascii_uppercase[r]
        if label not in taken:
            out.append(label)
        i += 1
    return out


def reading_order(positions: list[tuple[float, float]], row_mm: float = LETTER_ROW_MM) -> list[int]:
    """Indices of (x, y) positions in reading order: rows of *row_mm*
    (top to bottom), left to right within a row."""
    return sorted(range(len(positions)),
                  key=lambda i: (round(positions[i][1] / row_mm), positions[i][0]))


def align_moves(rects: dict, mode: str) -> dict:
    """{key: (dx, dy)} moving units with scene rects {key: (x, y, w, h)} to
    align their left / hcentre / right / top / vcentre / bottom edges, or to
    distribute them evenly ('hdistribute' / 'vdistribute': equal gaps
    between the first and last unit along that axis)."""
    if not rects:
        return {}
    xs = {k: r[0] for k, r in rects.items()}
    ys = {k: r[1] for k, r in rects.items()}
    rights = {k: r[0] + r[2] for k, r in rects.items()}
    bottoms = {k: r[1] + r[3] for k, r in rects.items()}
    if mode == 'left':
        t = min(xs.values())
        return {k: (t - xs[k], 0.0) for k in rects}
    if mode == 'right':
        t = max(rights.values())
        return {k: (t - rights[k], 0.0) for k in rects}
    if mode == 'hcentre':
        t = (min(xs.values()) + max(rights.values())) / 2
        return {k: (t - (xs[k] + rects[k][2] / 2), 0.0) for k in rects}
    if mode == 'top':
        t = min(ys.values())
        return {k: (0.0, t - ys[k]) for k in rects}
    if mode == 'bottom':
        t = max(bottoms.values())
        return {k: (0.0, t - bottoms[k]) for k in rects}
    if mode == 'vcentre':
        t = (min(ys.values()) + max(bottoms.values())) / 2
        return {k: (0.0, t - (ys[k] + rects[k][3] / 2)) for k in rects}
    axis = 0 if mode == 'hdistribute' else 1
    order = sorted(rects, key=lambda k: rects[k][axis])
    if len(order) < 3:
        return {k: (0.0, 0.0) for k in rects}
    start = rects[order[0]][axis]
    end = rects[order[-1]][axis] + rects[order[-1]][axis + 2]
    total = sum(rects[k][axis + 2] for k in order)
    gap = (end - start - total) / (len(order) - 1)
    moves, pos = {}, start
    for k in order:
        delta = pos - rects[k][axis]
        moves[k] = (delta, 0.0) if axis == 0 else (0.0, delta)
        pos += rects[k][axis + 2] + gap
    return moves


# ----------------------------------------------------------------------
# Undo
# ----------------------------------------------------------------------

class CallbackCommand(QUndoCommand):
    """An undo command from two callables. done=True: the change has
    already been made, so the first redo() (QUndoStack.push) does nothing."""

    def __init__(self, text: str, redo_fn, undo_fn, done: bool = False):
        super().__init__(text)
        self._redo_fn, self._undo_fn, self._skip = redo_fn, undo_fn, done

    def redo(self):
        if self._skip:
            self._skip = False
            return
        self._redo_fn()

    def undo(self):
        self._undo_fn()


# ----------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------

class PageItem(QGraphicsRectItem):
    """A white page with its number above it and, optionally, the snap grid:
    lines at the snap step (every n-th step when zoomed out, see
    grid_display_step), darker every MAJOR_GRID_MM."""

    def __init__(self, index: int, w_mm: float, h_mm: float):
        super().__init__(0.0, 0.0, w_mm, h_mm)
        self.index = index
        self.setZValue(-1e6)
        self.setBrush(QBrush(QColor('white')))
        pen = QPen(QColor('#9a9a9a'))
        pen.setCosmetic(True)
        self.setPen(pen)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        scene = self.scene()
        rect = self.rect()
        lod = option.levelOfDetailFromTransform(painter.worldTransform())
        if scene is not None and scene.show_grid:
            step = grid_display_step(scene.grid_mm, lod)
            minor, major = QPen(QColor('#e3e6ec')), QPen(QColor('#c8cdd6'))
            minor.setCosmetic(True)
            major.setCosmetic(True)
            for v in np.arange(step, max(rect.width(), rect.height()), step):
                on_major = abs(v / MAJOR_GRID_MM - round(v / MAJOR_GRID_MM)) < 1e-6
                painter.setPen(major if on_major else minor)
                if v < rect.width():
                    painter.drawLine(QPointF(v, 0.0), QPointF(v, rect.height()))
                if v < rect.height():
                    painter.drawLine(QPointF(0.0, v), QPointF(rect.width(), v))
            painter.setPen(self.pen())
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)
        font = QFont()
        font.setPixelSize(4)
        painter.setFont(font)
        painter.setPen(QColor('#707070'))
        painter.drawText(QPointF(0.0, -1.5), f"Page {self.index + 1}")


# ----------------------------------------------------------------------
# Figure items
# ----------------------------------------------------------------------

class FigureItem(QGraphicsObject):
    """
    An item showing a matplotlib figure at its true size (mm), re-rendered
    (debounced) when the figure or the zoom changes. Movable, selectable,
    snapped to the page grid; resized from the edges in RESIZE_EDGES.
    Right-click / double-click go to the figure first (legend picks of
    pick_owner()), then to the item menu / double-click action.
    """

    action_requested = Signal(str, object)   # (action, item)
    RESIZE_EDGES: tuple = ()

    def __init__(self, uid: str, source, parent=None):
        super().__init__(parent)
        self.uid = uid
        self.source = source
        self.layout_mm = {'page': 0, 'x_mm': 0.0, 'y_mm': 0.0}
        self.group_id: str | None = None
        self._w, self._h = 10.0, 10.0
        self._image: QImage | None = None
        self._image_stale = True
        self._render_pending = True
        self._resize_edge = None
        self._resize_start = None
        self._isolated = False       # double-clicked out of its group
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable
                      | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
                      | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(RENDER_DELAY_MS)
        self._render_timer.timeout.connect(self.render_now)

    # -- subclass hooks ----------------------------------------------------
    def figure(self):
        raise NotImplementedError

    def pick_owner(self):
        """Object with legend_pick_handled, or None (no figure picks)."""
        return None

    def aspect_locked(self) -> bool:
        return False

    def size_state(self):
        """What a resize changes, for undo."""
        return None

    def apply_size_state(self, state):
        pass

    def resize_to(self, w_mm: float, h_mm: float, edge: str):
        pass

    def menu_entries(self) -> list:
        return []

    def double_click_action(self) -> str | None:
        return None

    # -- geometry ------------------------------------------------------
    def figure_size_mm(self) -> tuple[float, float]:
        w_in, h_in = self.figure().get_size_inches()
        return float(w_in) * MM_PER_IN, float(h_in) * MM_PER_IN

    def size_mm(self) -> tuple[float, float]:
        return self._w, self._h

    def scene_rect_mm(self) -> QRectF:
        return self.mapRectToScene(QRectF(0.0, 0.0, self._w, self._h))

    def _handle_mm(self) -> float:
        scene = self.scene()
        px_per_mm = scene.px_per_mm if scene is not None else 4.0
        return HANDLE_PX / max(px_per_mm, 1e-6)

    def boundingRect(self) -> QRectF:
        m = self._handle_mm()
        return QRectF(-m, -m, self._w + 2 * m, self._h + 2 * m)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        m = self._handle_mm() if self.isSelected() and self.RESIZE_EDGES else 0.0
        path.addRect(QRectF(0.0, 0.0, self._w + m, self._h + m))
        return path

    def geometry_scale_changed(self):
        self.prepareGeometryChange()
        self.schedule_render()

    # -- rendering -----------------------------------------------------
    def on_figure_changed(self):
        w, h = self.figure_size_mm()
        if (w, h) != (self._w, self._h):
            self.prepareGeometryChange()
            self._w, self._h = w, h
            self.size_changed()
        self._image_stale = True
        self.schedule_render()

    def size_changed(self):
        """Called when the figure's size changed."""

    def schedule_render(self):
        # Started from paint(): Qt paints only items in view, so off-screen items are not rendered
        self._render_pending = True
        self.update()

    def render_dpi(self) -> float:
        scene = self.scene()
        px_per_mm = scene.px_per_mm if scene is not None else 4.0
        dpr = scene.device_pixel_ratio if scene is not None else 1.0
        dpi = px_per_mm * MM_PER_IN * dpr
        w_in, h_in = self._w / MM_PER_IN, self._h / MM_PER_IN
        dpi_cap = (MAX_RENDER_PIXELS / max(w_in * h_in, 1e-6)) ** 0.5
        return float(min(max(dpi, MIN_RENDER_DPI), MAX_RENDER_DPI, dpi_cap))

    def render_now(self):
        """Draw the figure with Agg at the zoom's dpi into self._image. After
        the figure has changed it is drawn twice: compressed layout only
        settles on its second pass (the x-axis title then sits ~6 pt
        higher), and a settled layout stays settled on later draws."""
        fig = self.figure()
        fig.set_dpi(self.render_dpi())
        canvas = fig.canvas
        for _ in range(LAYOUT_PASSES if self._image_stale else 1):
            canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())
        h_px, w_px = buf.shape[:2]
        self._image = QImage(buf.tobytes(), w_px, h_px, 4 * w_px,
                             QImage.Format.Format_RGBA8888).copy()
        self._image_stale = False
        self._render_pending = False
        self.update()

    def paint(self, painter, option, widget=None):
        rect = QRectF(0.0, 0.0, self._w, self._h)
        if self._render_pending and not self._render_timer.isActive():
            self._render_timer.start()
        if self._image is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(rect, self._image)
        else:
            painter.fillRect(rect, QColor('#f2f2f2'))
        if self.isSelected():
            pen = QPen(QColor(GROUP_COLOUR if self.group_id else SELECTION_COLOUR))
            pen.setCosmetic(True)
            pen.setWidthF(1.5)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)
            if self.RESIZE_EDGES and not self.group_id:
                painter.setBrush(QBrush(QColor(SELECTION_COLOUR)))
                m = self._handle_mm()
                for _edge, (cx, cy) in self._handles():
                    painter.drawRect(QRectF(cx - m / 2, cy - m / 2, m, m))

    def _handles(self):
        centres = {'right': (self._w, self._h / 2), 'bottom': (self._w / 2, self._h),
                   'corner': (self._w, self._h)}
        return [(e, centres[e]) for e in self.RESIZE_EDGES]

    def _edge_at(self, pos: QPointF):
        if not self.isSelected() or self.group_id:
            return None
        m = self._handle_mm()
        for edge, (cx, cy) in self._handles():
            if abs(pos.x() - cx) <= m and abs(pos.y() - cy) <= m:
                return edge
        return None

    # -- moving, snapping, resizing ------------------------------------
    def _snapping(self) -> bool:
        scene = self.scene()
        alt = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.AltModifier)
        return scene is not None and scene.snap_enabled and not alt and not scene.placing

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self._snapping():
            scene = self.scene()
            parent = self.parentItem()
            sp = parent.mapToScene(value) if parent is not None else value
            ox, oy = scene.page_origin(scene.page_index_at(sp.y() + self._h / 2))
            snapped = QPointF(ox + snap(sp.x() - ox, scene.grid_mm),
                              oy + snap(sp.y() - oy, scene.grid_mm))
            return parent.mapFromScene(snapped) if parent is not None else snapped
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self.prepareGeometryChange()
        return super().itemChange(change, value)

    def hoverMoveEvent(self, event):
        edge = self._edge_at(event.pos())
        cursors = {'right': Qt.SizeHorCursor, 'bottom': Qt.SizeVerCursor,
                   'corner': Qt.SizeFDiagCursor}
        self.setCursor(QCursor(cursors.get(edge, Qt.ArrowCursor)))
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            edge = self._edge_at(event.pos())
            if edge is not None:
                self._resize_edge = edge
                self._resize_start = (self._w, self._h, self.size_state())
                event.accept()
                return
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton and self.scene() is not None:
            if self.group_id and not event.modifiers() & (
                    Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier):
                self.scene().select_group(self.group_id)
            self.scene().begin_drag()

    def mouseMoveEvent(self, event):
        if self._resize_edge is None:
            super().mouseMoveEvent(event)
            return
        scene = self.scene()
        top_left = self.mapToScene(QPointF(0.0, 0.0))
        ox, oy = scene.page_origin(scene.page_index_at(top_left.y() + self._h / 2))
        sp = event.scenePos()
        right, bottom = sp.x() - ox, sp.y() - oy
        if self._snapping():
            right, bottom = snap(right, scene.grid_mm), snap(bottom, scene.grid_mm)
        w0, h0, _state = self._resize_start
        minimum = MIN_LEGEND_MM if isinstance(self, LegendItem) else MIN_ITEM_MM
        w, h = w0, h0
        if self._resize_edge in ('right', 'corner'):
            w = max(minimum, right - (top_left.x() - ox))
        if self._resize_edge in ('bottom', 'corner'):
            h = max(minimum, bottom - (top_left.y() - oy))
        if self.aspect_locked():
            w, h = locked_size(w0, h0, w, h, self._resize_edge)
        self.prepareGeometryChange()
        self._w, self._h = w, h
        self.update()

    def mouseReleaseEvent(self, event):
        if self._resize_edge is not None:
            edge, before = self._resize_edge, self._resize_start[2]
            self._resize_edge = None
            self.resize_to(self._w, self._h, edge)
            self.on_figure_changed()
            scene = self.scene()
            if scene is not None:
                scene.record_resize(self, before, self.size_state())
            event.accept()
            return
        super().mouseReleaseEvent(event)
        scene = self.scene()
        if event.button() == Qt.LeftButton and scene is not None:
            # A click (no drag) leaves only this item selected: keep the group
            if self.group_id and self.isSelected() and not self._isolated and not (
                    event.modifiers() & (Qt.KeyboardModifier.ShiftModifier
                                         | Qt.KeyboardModifier.ControlModifier)):
                scene.select_group(self.group_id)
            self._isolated = False
            scene.end_drag()

    # -- figure events / menu ------------------------------------------
    def dispatch_figure_click(self, pos: QPointF, button: int, dblclick: bool) -> bool:
        """Pass a click at item position *pos* (mm) to the figure as a
        matplotlib button press; True if pick_owner() handled it."""
        owner = self.pick_owner()
        if owner is None:
            return False
        from matplotlib.backend_bases import MouseEvent
        fig = self.figure()
        if self._image_stale or self._render_pending:
            self.render_now()
        dpi = fig.dpi
        x = pos.x() / MM_PER_IN * dpi
        y = (self._h - pos.y()) / MM_PER_IN * dpi
        owner.legend_pick_handled = False
        event = MouseEvent('button_press_event', fig.canvas, x, y, button=button,
                           dblclick=dblclick)
        fig.canvas.callbacks.process('button_press_event', event)
        return bool(owner.legend_pick_handled)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.dispatch_figure_click(event.pos(), 1, True):
                event.accept()
                return
            scene = self.scene()
            if self.group_id and scene is not None:
                scene.clearSelection()
                self.setSelected(True)
                self._isolated = True
                event.accept()
                return
            action = self.double_click_action()
            if action:
                self.action_requested.emit(action, self)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def legend_entry_at(self, pos: QPointF):
        """(kind, key, name) of the legend entry of pick_owner() at item
        position *pos*, or None."""
        owner = self.pick_owner()
        if owner is None:
            return None
        owner.legend_probe = True
        owner.legend_probe_hit = None
        try:
            self.dispatch_figure_click(pos, 1, False)
        finally:
            owner.legend_probe = False
        return owner.legend_probe_hit

    def contextMenuEvent(self, event):
        entry = self.legend_entry_at(event.pos())
        scene = self.scene()
        if not self.isSelected() and scene is not None:
            scene.clearSelection()
            self.setSelected(True)
            if self.group_id:
                scene.select_group(self.group_id)
        menu = QMenu()
        actions = {}
        entries = list(self.menu_entries())
        if entry is not None:
            kind, key, name = entry
            head = []
            if kind == 'cluster' or key != 'Unassigned':
                head.append(('entry_colour', f"Colour of “{name}”…"))
            if kind == 'cluster':
                head.append(('entry_rename', f"Rename “{name}”…"))
            entries = head + [(None, None)] + entries
        for key, text in entries:
            if key is None:
                menu.addSeparator()
            else:
                actions[menu.addAction(text)] = key
        chosen = menu.exec(event.screenPos())
        if chosen in actions:
            action = actions[chosen]
            if action in ('entry_colour', 'entry_rename'):
                self.pick_owner().legend_entry_action(action, entry[0], entry[1])
            else:
                self.action_requested.emit(action, self)
        event.accept()


_COMMON_MENU = [(None, None), ('group', "Group selection"), ('ungroup', "Ungroup"),
                (None, None), ('front', "Bring to front"), ('back', "Send to back")]


class PlotItem(FigureItem):
    """A PlotCard's plot figure. Resizing sets the card's plot size (aspect
    ratio kept when the card locks it). Owns the card's LegendItem when the
    legend is separate."""

    RESIZE_EDGES = ('right', 'bottom', 'corner')

    def __init__(self, card):
        super().__init__(card.plot_id, card)
        self.card = card
        self.legend_item: LegendItem | None = None
        card.figure_changed.connect(self.on_figure_changed)
        card.legend_changed.connect(self.on_legend_changed)
        self.on_figure_changed()
        self.on_legend_changed()

    def figure(self):
        return self.card.get_figure()

    def pick_owner(self):
        return self.card

    def aspect_locked(self) -> bool:
        return self.card.aspect_locked()

    def size_state(self):
        return self.card.plot_size_mm()

    def apply_size_state(self, state):
        self.card.set_plot_size_mm(*state)

    def resize_to(self, w_mm, h_mm, edge):
        self.card.set_plot_size_from_item(w_mm, h_mm, edge)

    def size_changed(self):
        if self.legend_item is not None:
            self.legend_item.place()

    def on_legend_changed(self):
        if self.card.legend_shown:
            if self.legend_item is None:
                self.legend_item = LegendItem(self)
                self.legend_item.action_requested.connect(self.action_requested)
            self.legend_item.on_figure_changed()
            self.legend_item.place()
        elif self.legend_item is not None:
            item, self.legend_item = self.legend_item, None
            if item.scene() is not None:
                item.scene().removeItem(item)
            item.deleteLater()

    def menu_entries(self):
        return ([('duplicate', "Duplicate"), (None, None),
                 ('copy_style', "Copy display settings"),
                 ('paste_style', "Paste display settings"), (None, None),
                 ('letter', "Add panel letter"), ('export_png', "Export PNG…")]
                + _COMMON_MENU + [(None, None), ('delete', "Remove plot")])


class LegendItem(FigureItem):
    """A PlotCard's separate legend, a child of its PlotItem. Docked beside
    the plot until moved; resizing reflows the entries into columns."""

    RESIZE_EDGES = ('right', 'bottom', 'corner')

    def __init__(self, plot_item: PlotItem):
        super().__init__('legend:' + plot_item.card.plot_id, plot_item.card, parent=plot_item)
        self.card = plot_item.card
        self.plot_item = plot_item
        self.on_figure_changed()

    def figure(self):
        return self.card.legend_figure()

    def pick_owner(self):
        return self.card

    def size_state(self):
        return self.card.legend_columns()

    def apply_size_state(self, state):
        self.card.set_legend_columns(state)

    def resize_to(self, w_mm, h_mm, edge):
        self.card.set_legend_box_mm(w_mm, h_mm, edge)

    def place(self):
        """Docked: just right of the plot, tops aligned; else the card's offset."""
        offset = self.card.legend_offset_mm
        if offset is None:
            offset = (self.plot_item.size_mm()[0] + LEGEND_DOCK_GAP_MM, 0.0)
        scene = self.scene()
        if scene is not None:
            placing, scene.placing = scene.placing, True
        self.setPos(QPointF(*offset))
        if scene is not None:
            scene.placing = placing

    def menu_entries(self):
        return [('dock_legend', "Put legend back beside the plot"),
                ('hide_legend', "Hide legend"), (None, None),
                ('export_png', "Export plot PNG…")]


class TextBox(QObject):
    """A text box or panel letter: its settings (spec) and a transparent
    matplotlib figure sized to the text."""

    changed = Signal()

    def __init__(self, uid: str, spec: dict | None = None):
        super().__init__()
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        self.uid = uid
        base = DEFAULT_LETTER if (spec or {}).get('kind') == 'letter' else DEFAULT_TEXT
        self.spec = dict(base, **(spec or {}))
        self._figure = Figure(figsize=(1.0, 0.3))
        FigureCanvasAgg(self._figure)
        self.redraw()

    def figure(self):
        return self._figure

    def set_spec(self, **changes):
        if all(self.spec.get(k) == v for k, v in changes.items()):
            return
        self.spec.update(changes)
        self.redraw()

    def redraw(self):
        fig = self._figure
        fig.clear()
        fig.patch.set_alpha(0.0)
        s = self.spec
        text = s['text'] if str(s['text']).strip() else ' '
        ha = {'left': 'left', 'centre': 'center', 'right': 'right'}[s['align']]
        t = fig.text(0.0, 1.0, text, ha='left', va='top', multialignment=ha,
                     fontsize=s['size_pt'], color=s['colour'],
                     fontweight='bold' if s['bold'] else 'normal',
                     fontstyle='italic' if s['italic'] else 'normal')
        renderer = fig.canvas.get_renderer()
        bb = t.get_window_extent(renderer)
        pad = TEXT_PAD_PT / 72.0
        w_in = bb.width / fig.dpi + 2 * pad
        h_in = bb.height / fig.dpi + 2 * pad
        fig.set_size_inches(max(w_in, 0.05), max(h_in, 0.05))
        t.set_position((pad / fig.get_figwidth(), 1.0 - pad / fig.get_figheight()))
        self.changed.emit()


class TextItem(FigureItem):
    """A TextBox on the page; sized by its text (font size fixed in pt)."""

    def __init__(self, box: TextBox):
        super().__init__(box.uid, box)
        self.box = box
        box.changed.connect(self.on_figure_changed)
        self.on_figure_changed()

    def figure(self):
        return self.box.figure()

    def double_click_action(self):
        return 'edit_text'

    def menu_entries(self):
        return [('edit_text', "Edit text"), ('duplicate', "Duplicate")] + _COMMON_MENU + [
            (None, None), ('delete', "Remove")]


def detached_copy(fig):
    """A copy of a matplotlib figure with its own Agg canvas, so it can be
    resized and drawn without touching the original (which may be on
    show in another tab). Plain functions stored on the figure (hover
    handlers) are left out. Returns (copy, as_image): if the figure cannot
    be copied, the copy is an image of it at SOURCE_FIGURE_IMAGE_DPI."""
    import io
    import pickle
    import types
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    held = {k: v for k, v in vars(fig).items() if isinstance(v, types.FunctionType)}
    for k in held:
        delattr(fig, k)
    try:
        copy = pickle.loads(pickle.dumps(fig))
        FigureCanvasAgg(copy)
        return copy, False
    except Exception:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=SOURCE_FIGURE_IMAGE_DPI)
        buf.seek(0)
        import matplotlib.image as mpimg
        image = mpimg.imread(buf)
        copy = Figure(figsize=tuple(fig.get_size_inches()))
        FigureCanvasAgg(copy)
        ax = copy.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.imshow(image, interpolation='antialiased')
        ax.set_axis_off()
        return copy, True
    finally:
        for k, v in held.items():
            setattr(fig, k, v)


class SourceFigure(QObject):
    """A figure from another tab, held on a page: spec {'tab', 'key',
    'label', 'w_mm', 'h_mm', 'aspect_locked'} and a detached copy of the
    figure (or a placeholder while the source is not available). Fonts stay
    in pt when it is resized."""

    changed = Signal()

    def __init__(self, uid: str, spec: dict):
        super().__init__()
        self.uid = uid
        self.spec = dict({'w_mm': None, 'h_mm': None, 'aspect_locked': True}, **spec)
        self.available = False
        self.as_image = False
        self.natural_mm: tuple[float, float] | None = None
        self._figure = None
        self.set_unavailable()

    def figure(self):
        return self._figure

    def _apply_size(self):
        w, h = self.spec['w_mm'], self.spec['h_mm']
        if w and h:
            self._figure.set_size_inches(w / MM_PER_IN, h / MM_PER_IN)

    def set_figure(self, fig, as_image: bool, max_w_mm: float):
        """Show *fig* (a detached copy). A first figure takes its own size,
        scaled down (aspect kept) to at most max_w_mm wide."""
        self._figure = fig
        self.available, self.as_image = True, as_image
        w_in, h_in = fig.get_size_inches()
        self.natural_mm = (float(w_in) * MM_PER_IN, float(h_in) * MM_PER_IN)
        if not (self.spec['w_mm'] and self.spec['h_mm']):
            w, h = self.natural_mm
            scale = min(1.0, max_w_mm / w)
            self.spec['w_mm'], self.spec['h_mm'] = w * scale, h * scale
        self._apply_size()
        self.changed.emit()

    def set_unavailable(self):
        """A placeholder saying where the figure comes from."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        self.available, self.as_image = False, False
        w = self.spec['w_mm'] or 80.0
        h = self.spec['h_mm'] or 50.0
        fig = Figure(figsize=(w / MM_PER_IN, h / MM_PER_IN))
        FigureCanvasAgg(fig)
        fig.patch.set_facecolor('#f4f4f4')
        fig.text(0.5, 0.5, f"{self.spec.get('label', '')}\nnot available — show it in the "
                 f"{self.spec.get('tab', '')} tab,\nthen right-click → Update from source",
                 ha='center', va='center', fontsize=7, color='#666666', wrap=True)
        self._figure = fig
        self.changed.emit()

    def set_size_mm(self, w_mm: float, h_mm: float):
        self.spec['w_mm'], self.spec['h_mm'] = float(w_mm), float(h_mm)
        self._apply_size()
        self.changed.emit()


class SourceFigureItem(FigureItem):
    """A SourceFigure on the page; resized like a plot (aspect ratio locked
    by default), fonts unchanged."""

    RESIZE_EDGES = ('right', 'bottom', 'corner')

    def __init__(self, source: SourceFigure):
        super().__init__(source.uid, source)
        self.figure_source = source
        source.changed.connect(self.on_figure_changed)
        self.on_figure_changed()

    def figure(self):
        return self.figure_source.figure()

    def aspect_locked(self) -> bool:
        return bool(self.figure_source.spec['aspect_locked'])

    def size_state(self):
        return (self.figure_source.spec['w_mm'], self.figure_source.spec['h_mm'])

    def apply_size_state(self, state):
        if state[0] and state[1]:
            self.figure_source.set_size_mm(*state)

    def resize_to(self, w_mm, h_mm, edge):
        self.figure_source.set_size_mm(w_mm, h_mm)

    def menu_entries(self):
        return ([('update_source', "Update from source"), ('original_size', "Original size"),
                 (None, None), ('duplicate', "Duplicate"), ('letter', "Add panel letter"),
                 ('export_png', "Export PNG…")]
                + _COMMON_MENU + [(None, None), ('delete', "Remove")])


class GroupFrame(QGraphicsObject):
    """Dashed frame around a selected group, with a corner handle that
    scales the group about its top-left corner."""

    def __init__(self):
        super().__init__()
        self.group_id: str | None = None
        self._rect = QRectF()
        self._scale = 1.0
        self._dragging = False
        self.setZValue(1e6)
        self.setVisible(False)
        self.setAcceptHoverEvents(True)

    def _handle_mm(self) -> float:
        scene = self.scene()
        return HANDLE_PX / max(scene.px_per_mm if scene is not None else 4.0, 1e-6)

    def show_for(self, group_id: str | None, rect: QRectF | None):
        self.prepareGeometryChange()
        self.group_id = group_id
        if group_id is None or rect is None:
            self.setVisible(False)
            return
        self._rect = QRectF(rect)
        self._scale = 1.0
        self.setVisible(True)
        self.update()

    def _corner(self) -> QPointF:
        r = self._rect
        return QPointF(r.left() + r.width() * self._scale, r.top() + r.height() * self._scale)

    def boundingRect(self) -> QRectF:
        m = self._handle_mm()
        r = self._rect
        big = max(self._scale, 1.0)
        return QRectF(r.left() - m, r.top() - m, r.width() * big + 2 * m, r.height() * big + 2 * m)

    def shape(self) -> QPainterPath:
        m = self._handle_mm()
        c = self._corner()
        path = QPainterPath()
        path.addRect(QRectF(c.x() - m, c.y() - m, 2 * m, 2 * m))
        return path

    def paint(self, painter, option, widget=None):
        pen = QPen(QColor(GROUP_COLOUR))
        pen.setCosmetic(True)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        r = self._rect
        painter.drawRect(QRectF(r.left(), r.top(), r.width() * self._scale,
                                r.height() * self._scale))
        m = self._handle_mm()
        c = self._corner()
        painter.setBrush(QBrush(QColor(GROUP_COLOUR)))
        painter.drawRect(QRectF(c.x() - m / 2, c.y() - m / 2, m, m))

    def hoverMoveEvent(self, event):
        self.setCursor(QCursor(Qt.SizeFDiagCursor))
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return
        r, p = self._rect, event.scenePos()
        sx = (p.x() - r.left()) / max(r.width(), 1e-6)
        sy = (p.y() - r.top()) / max(r.height(), 1e-6)
        self.prepareGeometryChange()
        self._scale = max(MIN_GROUP_SCALE, max(sx, sy))
        self.update()

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            scale, self._scale = self._scale, 1.0
            if abs(scale - 1.0) > 1e-3 and self.scene() is not None:
                self.scene().group_scale_requested.emit(self.group_id, scale)
            event.accept()
            return
        super().mouseReleaseEvent(event)


# ----------------------------------------------------------------------
# Scene
# ----------------------------------------------------------------------

class WorkspaceScene(QGraphicsScene):
    """Pages, figure items and the undo stack; scene unit = 1 mm."""

    layout_changed = Signal()
    current_page_changed = Signal(int)
    group_scale_requested = Signal(str, float)
    page_menu_requested = Signal(int, float, float, QPoint)   # page, x_mm, y_mm, screen pos

    def __init__(self, pages: list[dict], grid_mm: float = DEFAULT_GRID_MM, snap_enabled: bool = True,
                 show_grid: bool = True, parent=None):
        super().__init__(parent)
        self.pages: list[dict] = [dict(p) for p in pages] or [dict(DEFAULT_PAGE)]
        self.grid_mm = float(grid_mm)
        self.snap_enabled = bool(snap_enabled)
        self.show_grid = bool(show_grid)
        self.placing = False          # True while code sets positions (no snapping)
        self.px_per_mm = 4.0
        self.device_pixel_ratio = 1.0
        self.current_page = 0
        self.undo_stack = QUndoStack(self)
        self.last_click: tuple[int, float, float] | None = None   # (page, x_mm, y_mm)
        self._drag_start: dict | None = None
        self._page_items: list[PageItem] = []
        self.setBackgroundBrush(QBrush(QColor('#c9ccd2')))
        self.group_frame = GroupFrame()
        self.addItem(self.group_frame)
        self.selectionChanged.connect(self.update_group_frame)
        self._rebuild_pages()

    # -- lookups ---------------------------------------------------------
    def figure_items(self) -> list[FigureItem]:
        return [it for it in self.items() if isinstance(it, FigureItem)]

    def top_items(self) -> list[FigureItem]:
        return [it for it in self.figure_items() if it.parentItem() is None]

    def plot_items(self) -> list[PlotItem]:
        return [it for it in self.items() if isinstance(it, PlotItem)]

    def text_items(self) -> list[TextItem]:
        return [it for it in self.items() if isinstance(it, TextItem)]

    def source_items(self) -> list[SourceFigureItem]:
        return [it for it in self.items() if isinstance(it, SourceFigureItem)]

    def item_by_uid(self, uid: str):
        for it in self.figure_items():
            if it.uid == uid:
                return it
        return None

    def selected_figure_items(self) -> list[FigureItem]:
        return [it for it in self.selectedItems() if isinstance(it, FigureItem)]

    # -- pages -----------------------------------------------------------
    def page_origin(self, index: int) -> tuple[float, float]:
        return page_origins(self.pages)[min(max(index, 0), len(self.pages) - 1)]

    def page_index_at(self, y: float) -> int:
        return page_index_at(self.pages, y)

    def page_rect(self, index: int) -> QRectF:
        x0, y0 = self.page_origin(index)
        p = self.pages[index]
        return QRectF(x0, y0, p['w_mm'], p['h_mm'])

    def _rebuild_pages(self):
        for item in self._page_items:
            self.removeItem(item)
        self._page_items = []
        for i, (p, (x0, y0)) in enumerate(zip(self.pages, page_origins(self.pages))):
            item = PageItem(i, p['w_mm'], p['h_mm'])
            item.setPos(x0, y0)
            self.addItem(item)
            self._page_items.append(item)
        for it in self.top_items():
            self.position_item(it)
        widest = max(p['w_mm'] for p in self.pages)
        bottom = page_origins(self.pages)[-1][1] + self.pages[-1]['h_mm']
        self.setSceneRect(QRectF(-20.0, -20.0, widest + 40.0, bottom + 40.0))
        self.update_group_frame()

    def set_pages(self, pages: list[dict]):
        self.pages = [dict(p) for p in pages] or [dict(DEFAULT_PAGE)]
        self.current_page = min(self.current_page, len(self.pages) - 1)
        self._rebuild_pages()

    def set_page(self, index: int, page: dict):
        self.pages[index] = dict(page)
        self._rebuild_pages()
        self.layout_changed.emit()

    def add_page(self, after: int):
        self.pages.insert(after + 1, dict(self.pages[after]))
        for it in self.top_items():
            if it.layout_mm['page'] > after:
                it.layout_mm['page'] += 1
        self._rebuild_pages()
        self.layout_changed.emit()

    def remove_page(self, index: int):
        """Remove page *index* (its items must already be removed)."""
        if len(self.pages) <= 1:
            return
        del self.pages[index]
        for it in self.top_items():
            if it.layout_mm['page'] > index:
                it.layout_mm['page'] -= 1
        self.current_page = min(self.current_page, len(self.pages) - 1)
        self._rebuild_pages()
        self.layout_changed.emit()

    def set_grid(self, grid_mm: float, snap_enabled: bool, show_grid: bool):
        self.grid_mm, self.snap_enabled, self.show_grid = float(grid_mm), snap_enabled, show_grid
        for item in self._page_items:
            item.update()

    # -- positions -------------------------------------------------------
    def _set_pos_unsnapped(self, item: FigureItem, pos: QPointF):
        placing, self.placing = self.placing, True
        item.setPos(pos)
        self.placing = placing

    def position_item(self, item: FigureItem):
        """Place a top-level item from its layout_mm."""
        lay = item.layout_mm
        page = min(max(int(lay['page']), 0), len(self.pages) - 1)
        lay['page'] = page
        ox, oy = self.page_origin(page)
        self._set_pos_unsnapped(item, QPointF(ox + float(lay['x_mm']), oy + float(lay['y_mm'])))

    def update_layout(self, item: FigureItem):
        """Record a top-level item's page and position on it from its
        position; a legend's position is recorded as its card's offset."""
        if isinstance(item, LegendItem):
            item.card.legend_offset_mm = (round(item.pos().x(), 3), round(item.pos().y(), 3))
            return
        page = self.page_index_at(item.pos().y() + item.size_mm()[1] / 2)
        ox, oy = self.page_origin(page)
        item.layout_mm = {'page': page, 'x_mm': round(item.pos().x() - ox, 3),
                          'y_mm': round(item.pos().y() - oy, 3)}

    def positions(self, items) -> dict:
        return {it.uid: (it.pos().x(), it.pos().y()) for it in items}

    def set_positions(self, positions: dict):
        """Move items {uid: (x, y)} (parent coordinates) without snapping."""
        for uid, (x, y) in positions.items():
            item = self.item_by_uid(uid)
            if item is not None:
                self._set_pos_unsnapped(item, QPointF(x, y))
                self.update_layout(item)
        self.relabel_letters()
        self.update_group_frame()
        self.layout_changed.emit()

    def move_items(self, deltas: dict, text: str = "Move"):
        """Move items {uid: (dx, dy)} as one undoable step."""
        before, after = {}, {}
        for uid, (dx, dy) in deltas.items():
            item = self.item_by_uid(uid)
            if item is None or (abs(dx) < 1e-9 and abs(dy) < 1e-9):
                continue
            parent = item.parentItem()
            if isinstance(parent, FigureItem) and parent.uid in deltas:
                continue   # moves with its parent
            before[uid] = (item.pos().x(), item.pos().y())
            after[uid] = (item.pos().x() + dx, item.pos().y() + dy)
        if after:
            self.undo_stack.push(CallbackCommand(
                text, lambda: self.set_positions(after), lambda: self.set_positions(before)))

    def begin_drag(self):
        movable = [it for it in self.selected_figure_items()]
        self._drag_start = self.positions(movable)

    def end_drag(self):
        start, self._drag_start = self._drag_start, None
        if not start:
            return
        after = {}
        for uid in start:
            item = self.item_by_uid(uid)
            if item is not None:
                after[uid] = (item.pos().x(), item.pos().y())
        moved = {u: p for u, p in after.items() if p != start[u]}
        if not moved:
            return
        before = {u: start[u] for u in moved}
        for uid in moved:
            self.update_layout(self.item_by_uid(uid))
        self.relabel_letters()
        self.update_group_frame()
        self.layout_changed.emit()
        self.undo_stack.push(CallbackCommand(
            "Move", lambda: self.set_positions(moved), lambda: self.set_positions(before),
            done=True))

    def record_resize(self, item: FigureItem, before, after):
        if before == after:
            return
        uid = item.uid

        def apply(state):
            target = self.item_by_uid(uid)
            if target is not None:
                target.apply_size_state(state)

        self.undo_stack.push(CallbackCommand(
            "Resize", lambda: apply(after), lambda: apply(before), done=True))
        self.layout_changed.emit()

    # -- groups ----------------------------------------------------------
    def members(self, group_id: str) -> list[FigureItem]:
        return [it for it in self.top_items() if it.group_id == group_id]

    def select_group(self, group_id: str):
        for it in self.members(group_id):
            it.setSelected(True)

    def set_groups(self, groups: dict):
        """Set {uid: group_id or None} for top-level items."""
        for uid, gid in groups.items():
            item = self.item_by_uid(uid)
            if item is not None:
                item.group_id = gid
                item.update()
        self.update_group_frame()
        self.layout_changed.emit()

    def group_rect(self, group_id: str) -> QRectF:
        rect = QRectF()
        for it in self.members(group_id):
            rect = rect.united(it.scene_rect_mm())
            for child in it.childItems():
                if isinstance(child, FigureItem):
                    rect = rect.united(child.scene_rect_mm())
        return rect

    def selected_group(self) -> str | None:
        """The group whose members are exactly the selected top-level items."""
        selected = {it for it in self.selected_figure_items() if it.parentItem() is None}
        gids = {it.group_id for it in selected}
        if len(gids) != 1 or None in gids:
            return None
        gid = gids.pop()
        return gid if set(self.members(gid)) == selected else None

    def update_group_frame(self):
        gid = self.selected_group()
        self.group_frame.show_for(gid, self.group_rect(gid) if gid else None)

    # -- panel letters ---------------------------------------------------
    def relabel_letters(self):
        """Give automatic panel letters A, B, C ... in reading order on each
        page, skipping letters used by letters set by hand."""
        letters = [it for it in self.text_items() if it.box.spec['kind'] == 'letter']
        for page in range(len(self.pages)):
            on_page = [it for it in letters if it.layout_mm['page'] == page]
            taken = {it.box.spec['text'] for it in on_page if not it.box.spec['auto']}
            auto = [it for it in on_page if it.box.spec['auto']]
            order = reading_order([(it.layout_mm['x_mm'], it.layout_mm['y_mm']) for it in auto])
            for label, i in zip(letter_sequence(len(auto), taken), order):
                auto[i].box.set_spec(text=label)

    def next_letter(self, page: int, x_mm: float, y_mm: float) -> str:
        """The letter an automatic panel letter at (x, y) on *page* gets."""
        letters = [it for it in self.text_items()
                   if it.box.spec['kind'] == 'letter' and it.layout_mm['page'] == page]
        taken = {it.box.spec['text'] for it in letters if not it.box.spec['auto']}
        auto = [(it.layout_mm['x_mm'], it.layout_mm['y_mm'])
                for it in letters if it.box.spec['auto']] + [(x_mm, y_mm)]
        order = reading_order(auto)
        return letter_sequence(len(auto), taken)[order.index(len(auto) - 1)]

    # -- placement -------------------------------------------------------
    def placed_layouts(self, exclude=None) -> list[dict]:
        out = []
        for it in self.top_items():
            if it is exclude:
                continue
            rect = it.scene_rect_mm()
            for child in it.childItems():
                if isinstance(child, FigureItem):
                    rect = rect.united(child.scene_rect_mm())
            ox, oy = self.page_origin(it.layout_mm['page'])
            out.append(dict(it.layout_mm, x_mm=rect.x() - ox, y_mm=rect.y() - oy,
                            w_mm=rect.width(), h_mm=rect.height()))
        return out

    def _page_point(self, pos: QPointF):
        """(page, x_mm, y_mm) of a scene point on a page, else None."""
        page = self.page_index_at(pos.y())
        if not self.page_rect(page).contains(pos):
            return None
        ox, oy = self.page_origin(page)
        return page, pos.x() - ox, pos.y() - oy

    def take_last_click(self, height_mm: float):
        """The last clicked point on a page, or None; the next call gets the
        point just below an item of height_mm placed there."""
        point = self.last_click
        if point is not None:
            page, x, y = point
            self.last_click = (page, x, y + height_mm + TEXT_STACK_GAP_MM)
        return point

    def mousePressEvent(self, event):
        page = self.page_index_at(event.scenePos().y())
        if page != self.current_page:
            self.current_page = page
            self.current_page_changed.emit(page)
        point = self._page_point(event.scenePos())
        if point is not None:
            self.last_click = point
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        """Right-click on an empty part of a page: the page menu (add here)."""
        under = [it for it in self.items(event.scenePos())
                 if isinstance(it, (FigureItem, GroupFrame)) and it.isVisible()]
        point = self._page_point(event.scenePos())
        if under or point is None:
            super().contextMenuEvent(event)
            return
        self.last_click = point
        self.page_menu_requested.emit(point[0], point[1], point[2], event.screenPos())
        event.accept()


class WorkspaceView(QGraphicsView):
    """Zoomable view of a WorkspaceScene. 100% shows pages at their physical
    size on the current screen; Ctrl/⌘+wheel and trackpad pinch zoom under
    the cursor."""

    zoom_changed = Signal(float)

    def __init__(self, scene: WorkspaceScene, parent=None):
        super().__init__(scene, parent)
        self._zoom = 1.0
        self.setRenderHints(QPainter.RenderHint.Antialiasing
                            | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self._apply_zoom()

    def physical_px_per_mm(self) -> float:
        """Logical pixels per mm of the screen the view is on (from its
        reported physical size), falling back to the logical dpi."""
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            size = screen.physicalSize()
            if size.width() > 0:
                return screen.geometry().width() / size.width()
        return self.logicalDpiX() / MM_PER_IN

    def zoom(self) -> float:
        return self._zoom

    def set_zoom(self, zoom: float):
        self._zoom = float(min(max(zoom, MIN_ZOOM), MAX_ZOOM))
        self._apply_zoom()

    def _apply_zoom(self):
        scale = self._zoom * self.physical_px_per_mm()
        self.setTransform(QTransform.fromScale(scale, scale))
        scene = self.scene()
        scene.px_per_mm = scale
        scene.device_pixel_ratio = float(self.devicePixelRatioF())
        for it in scene.figure_items():
            it.geometry_scale_changed()
        scene.group_frame.prepareGeometryChange()
        self.zoom_changed.emit(self._zoom)

    def fit_page(self, index: int):
        scene = self.scene()
        ox, oy = scene.page_origin(index)
        p = scene.pages[index]
        vw, vh = self.viewport().width() - 20, self.viewport().height() - 20
        scale = min(vw / p['w_mm'], vh / p['h_mm'])
        self.set_zoom(scale / self.physical_px_per_mm())
        self.centerOn(ox + p['w_mm'] / 2, oy + p['h_mm'] / 2)

    def viewportEvent(self, event):
        if (event.type() == QEvent.Type.NativeGesture
                and event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture):
            self.set_zoom(self._zoom * (1.0 + event.value()))
            return True
        return super().viewportEvent(event)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            steps = event.angleDelta().y() / 120.0
            if steps:
                self.set_zoom(self._zoom * (ZOOM_STEP ** steps))
            event.accept()
            return
        super().wheelEvent(event)

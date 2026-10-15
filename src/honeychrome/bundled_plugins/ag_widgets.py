"""
ag_widgets.py — Qt helpers shared by the Gating Model Builder and Automated
Gating plugins
===========================================================================
  • add_concertina_page — one-section-at-a-time accordion (QToolBox) whose
                          pages scroll independently, as in the DR/Clustering
                          configuration tab.
  • SquareHolder        — keeps a child widget square inside any parent size.
  • zoom_axis_transform — applies a ZoomAxis drag to a Honeychrome Transform,
                          as the main window's plots do.
  • PopulationTree      — collapsible tree of the populations of a gating
                          hierarchy, with single or extended selection.
  • block_wheel         — stops a combo or spin box taking mouse-wheel
                          scrolling meant for the list or page around it.

The file name does not end in ``_tab.py``, so the plugin loader does not
treat it as a tab.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QFrame, QScrollArea, QSizePolicy, QToolBox,
    QTreeWidget, QTreeWidgetItem, QWidget,
)

__all__ = ['make_concertina', 'add_concertina_page', 'SquareHolder',
           'zoom_axis_transform', 'PopulationTree', 'block_wheel']


# ---------------------------------------------------------------------------
# Concertina
# ---------------------------------------------------------------------------

def make_concertina() -> QToolBox:
    """An accordion that keeps one section open at a time."""
    toolbox = QToolBox()
    toolbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return toolbox


def add_concertina_page(toolbox: QToolBox, page: QWidget, title: str,
                        tooltip: str = '') -> int:
    """Add *page* to *toolbox* inside its own scroll area; returns its index."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setWidget(page)
    index = toolbox.addItem(scroll, title)
    if tooltip:
        toolbox.setItemToolTip(index, tooltip)
    return index


# ---------------------------------------------------------------------------
# Square plot holder
# ---------------------------------------------------------------------------

class SquareHolder(QWidget):
    """Pins *child* to the largest square that fits, top-left aligned.

    A plot placed straight in a tall pane stretches to the pane's shape; the
    holder keeps density plots square whatever the pane's proportions.
    """

    def __init__(self, child: QWidget, minimum: int = 280, parent=None):
        super().__init__(parent)
        child.setParent(self)
        self._child = child
        self.setMinimumSize(minimum, minimum)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def resizeEvent(self, event):
        side = max(0, min(self.width(), self.height()))
        self._child.setGeometry(0, 0, side, side)
        super().resizeEvent(event)


# ---------------------------------------------------------------------------
# Axis rescaling
# ---------------------------------------------------------------------------

_ZOOM_RATE = 1.04


def zoom_axis_transform(axis, vb, axis_name: str, tr):
    """Apply the drag accumulated on a ZoomAxis to the Transform *tr*.

    Follows the main window's plots: on a logicle axis a drag over the lower
    half of the axis rescales the width basis W, a drag over the upper half
    zooms the displayed range; linear and log axes zoom the displayed range.
    Ticks and the axis zero are refreshed.

    axis_name : 'x' or 'y'.
    Returns None when no drag was pending, True when the transform itself
    changed (logicle W) so that boundaries stored in transformed units need
    converting, and False when only the displayed range moved.
    """
    step = axis._pending_delta
    if step == 0:
        return None
    axis._pending_delta = 0
    if abs(step) < 1:
        return None

    factor = (1 / _ZOOM_RATE) if step > 0 else _ZOOM_RATE
    if axis_name == 'y':
        factor = 1 / factor
        set_range = vb.setYRange
        vmin, vmax = vb.viewRange()[1]
        position = vb.mapToView(axis.initial_pos).y() if axis.initial_pos else 0.5
    else:
        set_range = vb.setXRange
        vmin, vmax = vb.viewRange()[0]
        position = vb.mapToView(axis.initial_pos).x() if axis.initial_pos else 0.5

    scale_changed = False
    if tr.id == 1 and position < 0.5 * vmax:
        tr.logicle_w = max(0.01, tr.logicle_w / factor)
        tr.set_transform()
        scale_changed = True
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
    return scale_changed


# ---------------------------------------------------------------------------
# Mouse wheel
# ---------------------------------------------------------------------------

def block_wheel(widget):
    """Make *widget* ignore the mouse wheel, so scrolling over it scrolls
    the table or page it sits in instead of changing its value."""
    widget.setFocusPolicy(Qt.StrongFocus)
    widget.wheelEvent = lambda event: event.ignore()
    return widget


# ---------------------------------------------------------------------------
# Population tree
# ---------------------------------------------------------------------------

_SEP = '\x1f'


def _encode(key) -> str:
    return f"{key[0] or ''}{_SEP}{key[1]}"


def _decode(value):
    """(gate, pop) from a stored key. Stored as a string: PySide6 hands
    tuples stored on items back as lists, which cannot be dictionary keys."""
    if value is None:
        return None
    gate, _, pop = str(value).partition(_SEP)
    return gate, pop


class PopulationTree(QTreeWidget):
    """Collapsible tree of populations.

    ``populationSelected(gate, pop)`` is emitted when the user picks a node;
    the root node has gate '' and pop 'root'. Programmatic selection with
    ``select_population`` does not emit. With *multi_select* several nodes
    can be selected (Ctrl/Shift-click); ``selected_populations`` lists them.
    ``expansionChanged`` is emitted when the user expands or collapses a
    node.
    """

    populationSelected = Signal(str, str)
    expansionChanged = Signal()

    _KEY_ROLE = Qt.UserRole

    def __init__(self, parent=None, multi_select: bool = False):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setRootIsDecorated(True)
        self.setUniformRowHeights(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection if multi_select
                              else QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._items: dict = {}
        self._depth: dict = {}
        self._parents: dict = {}
        self.itemSelectionChanged.connect(self._on_selection_changed)
        self.itemExpanded.connect(lambda _item: self.expansionChanged.emit())
        self.itemCollapsed.connect(lambda _item: self.expansionChanged.emit())

    def set_nodes(self, nodes: list[dict], expand_depth: int | None = None):
        """Fill the tree from ag_review.population_tree() nodes.

        Keeps the selection and the expanded state of nodes that survive
        the rebuild. New nodes start expanded, or, with *expand_depth*, only
        when they are fewer than *expand_depth* levels below the root (the
        root is level 0).
        """
        expanded = {key for key, item in self._items.items() if item.isExpanded()}
        known = set(self._items)
        selected = set(self.selected_populations())
        current = self.current_population()
        self.blockSignals(True)
        try:
            self.clear()
            self._items, self._depth, self._parents = {}, {}, {}
            for node in nodes:
                key = (node['gate'] or '', node['pop'])
                parent_key = node.get('parent')
                parent_key = (parent_key[0] or '', parent_key[1]) if parent_key else None
                parent_item = self._items.get(parent_key) if parent_key else None
                item = QTreeWidgetItem([node['label']])
                item.setData(0, self._KEY_ROLE, _encode(key))
                self._depth[key] = (self._depth.get(parent_key, -1) + 1) if parent_item else 0
                self._parents[key] = parent_key if parent_item else None
                if node.get('n_gates'):
                    item.setToolTip(0, f"{node['n_gates']} gate(s) drawn on this population")
                else:
                    item.setToolTip(0, "No gates below this population; selecting it shows "
                                       "the plot it is gated on.")
                if parent_item is None:
                    self.addTopLevelItem(item)
                else:
                    parent_item.addChild(item)
                self._items[key] = item
            for key, item in self._items.items():
                if key in known:
                    item.setExpanded(key in expanded)
                elif expand_depth is None:
                    item.setExpanded(True)
                else:
                    item.setExpanded(self._depth.get(key, 0) < expand_depth)
            for key in selected:
                if key in self._items:
                    self._items[key].setSelected(True)
            if current in self._items:
                self.setCurrentItem(self._items[current])
        finally:
            self.blockSignals(False)

    def _key(self, item):
        return _decode(item.data(0, self._KEY_ROLE)) if item is not None else None

    def current_population(self):
        """(gate, pop) of the current node (the one last clicked) when it is
        selected, else of the first selected node, or None."""
        item = self.currentItem()
        if item is not None and item.isSelected():
            return self._key(item)
        items = self.selectedItems()
        return self._key(items[0]) if items else None

    def selected_populations(self) -> list:
        """(gate, pop) of every selected node, in tree order."""
        chosen = {id(i) for i in self.selectedItems()}
        return [key for key, item in self._items.items() if id(item) in chosen]

    def population_at(self, pos):
        """(gate, pop) of the node at viewport position *pos*, or None."""
        return self._key(self.itemAt(pos))

    def depth(self, key) -> int:
        return self._depth.get((key[0] or '', key[1]), 0)

    def is_open(self, key) -> bool:
        """True when the node and every node above it are expanded, so the
        populations drawn on it are in view."""
        key = (key[0] or '', key[1])
        while key is not None:
            item = self._items.get(key)
            if item is None or not item.isExpanded():
                return False
            key = self._parents.get(key)
        return True

    def expand_population(self, gate, pop: str):
        """Expand a node and every node above it."""
        key = (gate or '', pop)
        while key is not None:
            item = self._items.get(key)
            if item is not None:
                item.setExpanded(True)
            key = self._parents.get(key)

    def select_population(self, gate, pop: str):
        """Select a node without emitting, expanding its ancestors."""
        item = self._items.get((gate or '', pop))
        self.blockSignals(True)
        try:
            self.clearSelection()
            if item is not None:
                item.setSelected(True)
                self.setCurrentItem(item)
                ancestor = item.parent()
                while ancestor is not None:
                    ancestor.setExpanded(True)
                    ancestor = ancestor.parent()
                self.scrollToItem(item)
        finally:
            self.blockSignals(False)

    def _on_selection_changed(self):
        key = self.current_population()
        if key is not None:
            self.populationSelected.emit(key[0], key[1])

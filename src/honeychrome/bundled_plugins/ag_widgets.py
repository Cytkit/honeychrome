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
                          hierarchy with single selection.

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
           'zoom_axis_transform', 'PopulationTree']


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
# Population tree
# ---------------------------------------------------------------------------

class PopulationTree(QTreeWidget):
    """Collapsible tree of populations with single selection.

    ``populationSelected(gate, pop)`` is emitted when the user picks a node;
    the root node has gate '' and pop 'root'. Programmatic selection with
    ``select_population`` does not emit.
    """

    populationSelected = Signal(str, str)

    _KEY_ROLE = Qt.UserRole

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setRootIsDecorated(True)
        self.setUniformRowHeights(True)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._items: dict = {}
        self.itemSelectionChanged.connect(self._on_selection_changed)

    def set_nodes(self, nodes: list[dict]):
        """Fill the tree from ag_review.population_tree() nodes.

        Keeps the current selection and the expanded state of nodes that
        survive the rebuild; new nodes start expanded.
        """
        expanded = {key for key, item in self._items.items() if item.isExpanded()}
        known = set(self._items)
        selected = self.current_population()
        self.blockSignals(True)
        try:
            self.clear()
            self._items = {}
            for node in nodes:
                key = (node['gate'] or '', node['pop'])
                parent_key = node.get('parent')
                parent_item = self._items.get(
                    (parent_key[0] or '', parent_key[1])) if parent_key else None
                item = QTreeWidgetItem([node['label']])
                item.setData(0, self._KEY_ROLE, key)
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
                item.setExpanded(key in expanded or key not in known)
            if selected in self._items:
                self._items[selected].setSelected(True)
        finally:
            self.blockSignals(False)

    def current_population(self):
        """(gate, pop) of the selected node, or None."""
        items = self.selectedItems()
        return items[0].data(0, self._KEY_ROLE) if items else None

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

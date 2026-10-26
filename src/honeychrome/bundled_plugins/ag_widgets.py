"""
ag_widgets.py — Qt helpers shared by the Gating Model Builder and Automated
Gating plugins
===========================================================================
  • add_concertina_page — one-section-at-a-time accordion (QToolBox) whose
                          pages scroll independently, as in the Honeycluster
                          configuration tab.
  • SquareHolder        — keeps a child widget square inside any parent size.
  • zoom_axis_transform — applies a ZoomAxis drag to a Honeychrome Transform,
                          as the main window's plots do.
  • PopulationTree      — collapsible tree of the populations of a gating
                          hierarchy, with single or extended selection, text
                          search and filtering.
  • block_wheel         — stops a combo or spin box taking mouse-wheel
                          scrolling meant for the list or page around it.
  • algorithm_combo     — algorithm picker whose entries are coloured as the
                          algorithms' thresholds are drawn on the plots.
  • FilterBar           — text search with a "show" filter, for long tables
                          and trees.
  • DecadeSpinBox       — spin box for small fractions that steps by factors
                          of ten and accepts any typed value.

The file name does not end in ``_tab.py``, so the plugin loader does not
treat it as a tab.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QAbstractScrollArea, QApplication, QComboBox, QDoubleSpinBox,
    QFrame, QHBoxLayout, QLabel, QLineEdit, QScrollArea, QSizePolicy, QToolBox,
    QTreeWidget, QTreeWidgetItem, QWidget,
)

import honeychrome.settings as settings

__all__ = ['make_concertina', 'add_concertina_page', 'SquareHolder',
           'zoom_axis_transform', 'PopulationTree', 'block_wheel',
           'algorithm_combo', 'colour_algorithm_items', 'set_combo_colour', 'FilterBar',
           'text_matches', 'DecadeSpinBox']


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
        w_step = settings.logicle_w_drag_step if factor < 1 else -settings.logicle_w_drag_step
        tr.logicle_w = min(tr.logicle_m / 2.0, max(0.0, tr.logicle_w + w_step))
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

class _WheelBlocker(QObject):
    """Event filter that keeps wheel events from the widgets it watches and
    hands them to the nearest scroll area above, so the page or table
    scrolls while the value under the pointer stays as it is."""

    def eventFilter(self, obj, event):
        if event.type() != QEvent.Type.Wheel:
            return False
        target = obj.parentWidget() if isinstance(obj, QWidget) else None
        while target is not None and not isinstance(target, QAbstractScrollArea):
            target = target.parentWidget()
        if target is not None:
            QApplication.sendEvent(target.viewport(), event)
        return True


_wheel_blocker = None


def block_wheel(widget):
    """Make *widget* ignore the mouse wheel, so scrolling over it scrolls
    the table or page it sits in instead of changing its value. Main
    thread only."""
    global _wheel_blocker
    if _wheel_blocker is None:
        _wheel_blocker = _WheelBlocker()
    widget.setFocusPolicy(Qt.StrongFocus)
    widget.installEventFilter(_wheel_blocker)
    return widget


class DecadeSpinBox(QDoubleSpinBox):
    """Spin box for a small positive fraction: the arrows multiply or divide
    the value by ten (0.0001, 0.001, 0.01, …) and any value in range can be
    typed. Shown without trailing zeros."""

    def __init__(self, minimum: float = 1e-6, maximum: float = 0.5, parent=None):
        super().__init__(parent)
        self.setDecimals(6)
        self.setRange(minimum, maximum)

    def stepBy(self, steps: int):
        value = float(self.value()) * (10.0 ** int(steps))
        self.setValue(min(max(value, self.minimum()), self.maximum()))

    def textFromValue(self, value: float) -> str:
        return f"{float(value):.6f}".rstrip('0').rstrip('.')


# ---------------------------------------------------------------------------
# Algorithm pickers
# ---------------------------------------------------------------------------

def _hex(rgb) -> str:
    return '#{:02x}{:02x}{:02x}'.format(*[int(v) for v in rgb[:3]])


def set_combo_colour(combo: QComboBox, rgb):
    """Show the combo's current text in *rgb* (None: the default colour)."""
    combo.setStyleSheet(f"QComboBox {{ color: {_hex(rgb)}; font-weight: bold; }}"
                        if rgb is not None else "")


def colour_algorithm_items(combo: QComboBox, colour_of):
    """Draw each entry of an algorithm combo (entries named by algorithm) in
    its colour, and the current entry's text likewise as it changes. Safe
    to call again after the entries are replaced."""
    for i in range(combo.count()):
        name = combo.itemData(i) or combo.itemText(i)
        combo.setItemData(i, QBrush(QColor(*colour_of(name)[:3])), Qt.ForegroundRole)

    def _recolour(_index=None):
        name = combo.currentData() or combo.currentText()
        set_combo_colour(combo, colour_of(name) if name else None)

    if not getattr(combo, '_algorithm_colours', False):
        combo.currentIndexChanged.connect(_recolour)
        combo._algorithm_colours = True
    _recolour()


def algorithm_combo(algorithms, current: str | None = None, colour_of=None) -> QComboBox:
    """Combo box of *algorithms*, each entry drawn in its colour.

    colour_of : callable, algorithm -> RGB tuple (e.g.
                ag_core.algorithm_colour). The current entry's colour
                follows the selection. The mouse wheel is blocked (see
                block_wheel).
    """
    combo = block_wheel(QComboBox())
    for name in algorithms:
        combo.addItem(name, name)
    if current in algorithms:
        combo.setCurrentIndex(list(algorithms).index(current))
    if colour_of is not None:
        colour_algorithm_items(combo, colour_of)
    return combo


# ---------------------------------------------------------------------------
# Search and filter bar
# ---------------------------------------------------------------------------

def text_matches(query: str, *texts) -> bool:
    """True when every word of *query* occurs (case-insensitively) in one of
    *texts*; an empty query matches everything."""
    words = [w for w in str(query or '').lower().split() if w]
    if not words:
        return True
    hay = ' '.join(str(t or '') for t in texts).lower()
    return all(w in hay for w in words)


class FilterBar(QWidget):
    """A search box and a "Show" filter for a list, table or tree.

    modes   : [(key, label)]; the first is the unfiltered view.
    changed : emitted with (search text, mode key) whenever either changes.
    """

    changed = Signal(str, str)

    def __init__(self, modes=None, placeholder: str = "Search…", vertical: bool = False,
                 parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QVBoxLayout
        outer = QVBoxLayout(self) if vertical else QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.search = QLineEdit()
        self.search.setPlaceholderText(placeholder)
        self.search.setClearButtonEnabled(True)
        outer.addWidget(self.search, stretch=1)
        self.combo = None
        if modes:
            row = QHBoxLayout()
            row.addWidget(QLabel("Show:"))
            self.combo = block_wheel(QComboBox())
            for key, label in modes:
                self.combo.addItem(label, key)
            row.addWidget(self.combo, stretch=1)
            outer.addLayout(row)
            self.combo.currentIndexChanged.connect(lambda _i: self._emit())
        self.search.textChanged.connect(lambda _t: self._emit())

    def text(self) -> str:
        return self.search.text().strip()

    def clear(self):
        """Empty the search and show everything."""
        self.blockSignals(True)
        self.search.clear()
        if self.combo is not None:
            self.combo.setCurrentIndex(0)
        self.blockSignals(False)
        self._emit()

    def mode(self) -> str:
        return self.combo.currentData() if self.combo is not None else ''

    def set_mode(self, key: str):
        if self.combo is not None:
            i = self.combo.findData(key)
            if i >= 0:
                self.combo.setCurrentIndex(i)

    def set_mode_label(self, key: str, label: str):
        """Change the text of one filter choice (e.g. to show a count)."""
        if self.combo is not None:
            i = self.combo.findData(key)
            if i >= 0:
                self.combo.setItemText(i, label)

    def is_active(self) -> bool:
        return bool(self.text()) or (self.combo is not None and self.combo.currentIndex() > 0)

    def _emit(self):
        self.changed.emit(self.text(), self.mode())


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
        self._labels: dict = {}
        self._tips: dict = {}
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
            self._labels, self._tips = {}, {}
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
                self._labels[key] = node['label']
                self._tips[key] = item.toolTip(0)
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

    def keys(self) -> list:
        """(gate, pop) of every node, parents before children."""
        return list(self._items)

    def parent_of(self, key):
        return self._parents.get((key[0] or '', key[1]))

    def set_filter(self, shown: set | None):
        """Show only the nodes in *shown* and the nodes above them (None
        shows every node). Nodes shown by the filter are expanded so they
        are in view."""
        self.blockSignals(True)
        try:
            if shown is None:
                for item in self._items.values():
                    item.setHidden(False)
                return
            visible = set()
            for key in shown:
                key = (key[0] or '', key[1])
                while key is not None and key not in visible:
                    visible.add(key)
                    key = self._parents.get(key)
            for key, item in self._items.items():
                item.setHidden(key not in visible)
            for key in visible:
                parent = self._parents.get(key)
                if parent is not None and parent in self._items:
                    self._items[parent].setExpanded(True)
        finally:
            self.blockSignals(False)
        self.expansionChanged.emit()

    def mark(self, marks: dict):
        """Colour nodes: {key: (RGB or None, tooltip suffix or '')}, or
        {key: (RGB or None, flag text, True)} to flag a node: its label then
        starts with ⚠ and its tooltip ends with the flag text. Nodes not in
        *marks* return to the default colour and label."""
        for key, item in self._items.items():
            rgb, tip, *rest = marks.get(key, (None, ''))
            flagged = bool(rest and rest[0])
            item.setForeground(0, QBrush(QColor(*rgb[:3])) if rgb is not None else QBrush())
            label = self._labels.get(key, item.text(0))
            item.setText(0, f"⚠ {label}" if flagged else label)
            base = self._tips.get(key, '')
            item.setToolTip(0, f"{base}\n{tip}" if flagged and tip else base)

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

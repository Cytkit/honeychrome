"""
drc_gate_tree.py — Checkable multi-select gate tree (§0.3 / Item 4)
=====================================================================
Companion module to ``dr_clustering_tab.py`` (filename intentionally does
NOT end in ``_tab.py``, so it is not picked up as a separate plugin tab).

Provides GateTreeWidget, used by both TransformTab (primary) and ConfigTab
(override) in place of the old single-select gate QComboBox.  Built from
the same nested-dict shape gating_hierarchy_widget.py's DictTreeModel
consumes (``controller.unmixed_gating.get_gate_hierarchy(output='dict')``),
but checkable instead of read-only, and exposing the checked set as a flat
``list[str]`` rather than a single current-text selection.

Checking a node cascades Checked down its whole subtree (tri-state
parents show partially-checked when only some descendants are checked).
``checked_names()`` only reports the TOPMOST fully-checked node in each
checked branch: a child gate's membership is always a subset of its
parent's in a FlowKit gating hierarchy, so including descendants of an
already-checked node would be redundant for
``apply_gates_union_by_lookup_table()``'s union-of-masks computation.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QAbstractItemModel, QModelIndex, Signal
from PySide6.QtWidgets import QTreeView, QWidget, QVBoxLayout


class _GateTreeItem:
    __slots__ = ('name', 'parent_item', 'child_items', 'check_state', 'explicit')

    def __init__(self, name: str, parent: '_GateTreeItem | None' = None):
        self.name = name
        self.parent_item = parent
        self.child_items: list['_GateTreeItem'] = []
        self.check_state = Qt.Unchecked
        self.explicit = False

    def append_child(self, item: '_GateTreeItem'):
        self.child_items.append(item)

    def child(self, row: int) -> '_GateTreeItem':
        return self.child_items[row]

    def child_count(self) -> int:
        return len(self.child_items)

    def row(self) -> int:
        if self.parent_item:
            return self.parent_item.child_items.index(self)
        return 0


class _CheckableGateTreeModel(QAbstractItemModel):
    """Tri-state checkable tree model built from a gating-hierarchy dict."""

    # Fired only from setData() — i.e. only on an actual checkbox click,
    # never from the programmatic set_hierarchy()/set_checked_names() paths.
    checkedChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.root_item = _GateTreeItem('Root')

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def set_hierarchy(self, hierarchy_dict: dict | None):
        """Rebuild the tree from a nested {'name', 'children'} dict."""
        self.beginResetModel()
        self.root_item = _GateTreeItem('Root')
        if hierarchy_dict:
            self._build(hierarchy_dict, self.root_item)
        self.endResetModel()

    def _build(self, node: dict, parent_item: _GateTreeItem):
        item = _GateTreeItem(node['name'], parent_item)
        parent_item.append_child(item)
        for child in node.get('children', []) or []:
            self._build(child, item)

    # ------------------------------------------------------------------
    # QAbstractItemModel
    # ------------------------------------------------------------------

    def columnCount(self, parent=QModelIndex()):
        return 1

    def rowCount(self, parent=QModelIndex()):
        parent_item = parent.internalPointer() if parent.isValid() else self.root_item
        return parent_item.child_count()

    def index(self, row, column, parent=QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        parent_item = parent.internalPointer() if parent.isValid() else self.root_item
        child_item = parent_item.child(row)
        if child_item:
            return self.createIndex(row, column, child_item)
        return QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()
        child_item = index.internalPointer()
        parent_item = child_item.parent_item
        if parent_item is None or parent_item is self.root_item:
            return QModelIndex()
        return self.createIndex(parent_item.row(), 0, parent_item)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        item = index.internalPointer()
        if role == Qt.DisplayRole:
            return item.name
        if role == Qt.CheckStateRole:
            return item.check_state
        return None

    def setData(self, index, value, role=Qt.CheckStateRole):
        if not index.isValid() or role != Qt.CheckStateRole:
            return False
        item = index.internalPointer()
        new_state = Qt.Checked if Qt.CheckState(value) == Qt.Checked else Qt.Unchecked
        touched: list[_GateTreeItem] = []
        self._set_subtree_state(item, new_state, touched)
        self._update_ancestors(item.parent_item, touched)
        for t in touched:
            idx = self.createIndex(t.row(), 0, t)
            self.dataChanged.emit(idx, idx, [Qt.CheckStateRole])
        self.checkedChanged.emit()
        return True

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return "Gate"
        return None

    # ------------------------------------------------------------------
    # Check-state propagation
    # ------------------------------------------------------------------

    def _set_subtree_state(self, item: _GateTreeItem, state, touched: list | None = None):
        """Set *item* and every descendant to *state* (convenience cascade).
        Marks the whole cascaded subtree 'explicit' so a later bottom-up
        recompute (_update_ancestors / _recompute_all) never second-guesses
        it — only nodes that were never a direct cascade target get their
        state inferred from their children."""
        item.check_state = state
        item.explicit = (state == Qt.Checked)
        if touched is not None:
            touched.append(item)
        for child in item.child_items:
            self._set_subtree_state(child, state, touched)

    def _update_ancestors(self, item: '_GateTreeItem | None', touched: list | None = None):
        """Recompute tri-state for *item* and every ancestor above it, from
        each node's children.  A node reached here was never itself the
        direct click target — it is only ever inferred Unchecked (every
        child unchecked) or PartiallyChecked (any child checked); it is
        NEVER promoted to fully Checked by inference, or checking two
        sibling leaves would silently select their whole parent branch
        (and cascade that promotion all the way to root).  Only a direct
        click on a node (via _set_subtree_state) can make it fully Checked.
        Any ancestor touched here also has its 'explicit' flag cleared,
        since a descendant just changed underneath it — it's no longer a
        wholly-selected branch even if it was before this click.  Leaves
        (no children) are left untouched — their state only ever comes
        from a direct assignment."""
        while item is not None and item is not self.root_item:
            item.explicit = False
            states = {c.check_state for c in item.child_items}
            item.check_state = (
                Qt.Unchecked if states <= {Qt.Unchecked} else Qt.PartiallyChecked
            )
            if touched is not None:
                touched.append(item)
            item = item.parent_item

    # ------------------------------------------------------------------
    # Checked-set <-> flat name list
    # ------------------------------------------------------------------

    def checked_names(self) -> list[str]:
        """Topmost fully-checked node name in each checked branch (see
        module docstring — descendants are redundant, so this does not
        descend into an already-Checked subtree)."""
        result: list[str] = []

        def _walk(item: _GateTreeItem):
            for child in item.child_items:
                if child.check_state == Qt.Checked:
                    result.append(child.name)
                elif child.check_state == Qt.PartiallyChecked:
                    _walk(child)

        _walk(self.root_item)
        return result

    def set_checked_names(self, names: list[str]):
        """
        Reset every checkbox, then check exactly the branches named in
        *names* (cascading each match's whole subtree Checked, same as an
        interactive click) and resync tri-state in one bottom-up pass.
        Never emits checkedChanged — reserved for actual checkbox clicks
        via setData() — so callers (restore / cross-tab sync) can re-apply
        freely without feedback loops.
        """
        names_set = set(names)
        touched: list[_GateTreeItem] = []
        self._reset_all(self.root_item, touched)
        self._apply_names(self.root_item, names_set)
        self._recompute_all(self.root_item)
        for t in touched:
            idx = self.createIndex(t.row(), 0, t)
            self.dataChanged.emit(idx, idx, [Qt.CheckStateRole])

    def _reset_all(self, item: _GateTreeItem, touched: list):
        if item is not self.root_item:
            item.check_state = Qt.Unchecked
            item.explicit = False
            touched.append(item)
        for child in item.child_items:
            self._reset_all(child, touched)

    def _apply_names(self, item: _GateTreeItem, names_set: set):
        """Top-down only — check matched branches, recurse into the rest.
        Tri-state is fixed up afterwards in one pass by _recompute_all()
        (see set_checked_names()), rather than repeatedly re-walking
        ancestors on every recursive return — O(n) instead of O(n*depth)."""
        for child in item.child_items:
            if child.name in names_set:
                self._set_subtree_state(child, Qt.Checked)
            else:
                self._apply_names(child, names_set)

    def _recompute_all(self, item: _GateTreeItem):
        """Bottom-up, single pass: fix every container node's tri-state from
        its (already-resolved) children.  A node marked 'explicit' was
        itself a direct match in _apply_names and cascaded top-down — its
        state (and its whole matched subtree's) is left exactly as cascaded.
        Any other container node is only ever inferred Unchecked (every
        child unchecked) or PartiallyChecked (any child checked) — never
        promoted to fully Checked, or two individually-matched sibling
        leaves would silently collapse into their shared parent gate.
        Leaves are left as directly set."""
        for child in item.child_items:
            self._recompute_all(child)
        if item.child_items and not item.explicit:
            states = {c.check_state for c in item.child_items}
            item.check_state = (
                Qt.Unchecked if states <= {Qt.Unchecked} else Qt.PartiallyChecked
            )


class GateTreeWidget(QWidget):
    """
    Checkable multi-select gate tree.  Wraps _CheckableGateTreeModel in a
    QTreeView with a shallow default expansion (two levels) so a hierarchy
    of hundreds of gates doesn't fully unfold on first display.
    """

    selectionChanged = Signal(list)   # list[str] of checked gate names

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = _CheckableGateTreeModel(self)
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.model)
        self.tree_view.setHeaderHidden(True)
        self.tree_view.header().setStretchLastSection(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tree_view)

        self.model.checkedChanged.connect(self._emit_selection)
    
    def _emit_selection(self):
        self.selectionChanged.emit(self.model.checked_names())

    def set_hierarchy(self, hierarchy_dict: dict | None):
        """Rebuild from a nested gating-hierarchy dict.  Does not preserve
        checked state — follow with set_checked_names(state.selected_gates),
        the shared authoritative list, to restore/sync selection."""
        self.model.set_hierarchy(hierarchy_dict)
        self.tree_view.expandToDepth(1)   # two levels, not expandAll()

    def checked_names(self) -> list[str]:
        return self.model.checked_names()

    def set_checked_names(self, names: list[str]):
        """Programmatic sync/restore — does not emit selectionChanged."""
        self.model.set_checked_names(names)
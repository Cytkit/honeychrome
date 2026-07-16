import json
from pathlib import Path

import numpy as np
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence, QClipboard
from PySide6.QtWidgets import (QApplication, QMainWindow, QTreeView, QWidget, QVBoxLayout, QHeaderView, QMenu,
                               QMessageBox, QComboBox, QHBoxLayout, QToolButton, QLabel, QInputDialog)
from PySide6.QtCore import Qt, QAbstractItemModel, QModelIndex, Slot, QMimeData

import logging
logger = logging.getLogger(__name__)

class TreeItem:
    def __init__(self, name, event_count, p_root, p_parent, event_conc, parent=None):
        self.name = name
        self.event_count = event_count
        self.event_conc = event_conc
        self.p_root = p_root
        self.p_parent = p_parent
        self.parent_item = parent
        self.child_items = []

    def append_child(self, item):
        self.child_items.append(item)

    def child(self, row):
        return self.child_items[row]

    def child_count(self):
        return len(self.child_items)

    def column_count(self):
        return 5

    def row(self):
        if self.parent_item:
            return self.parent_item.child_items.index(self)
        return 0

    def parent(self):
        return self.parent_item


class DictTreeModel(QAbstractItemModel):
    def __init__(self, gates=None, parent=None):
        super().__init__(parent)
        self.gating_hierarchy = None
        self.statistics = None
        self.root_item = TreeItem("Root", "", "", "", "")
        if gates:
            self.setup_model_data(gates, self.root_item)

    def setup_model_data(self, hierarchy_dict_node, parent_item):
        # e.g.
        # data = {'name': 'root', 'children': [
        #     {'gate_type': 'RectangleGate', 'custom_gates': {}, 'name': 'Cells', 'children': [
        #         {'gate_type': 'RectangleGate', 'custom_gates': {}, 'name': 'Singlets', 'children': [
        #              {'gate_type': 'RectangleGate', 'custom_gates': {}, 'name': 'Positive Unstained'},
        #              {'gate_type': 'RectangleGate', 'custom_gates': {}, 'name': 'Positive BUV805'},
        #              {'gate_type': 'RectangleGate', 'custom_gates': {}, 'name': 'Positive Super Bright 436'}
        #         ]}
        #     ]}
        # ]}
        #
        # statistics = {'root': {'n_events_gate': 55385, 'p_gate_total': 1.0, 'p_gate_parent': 1.0},
        #               'Cells': {'n_events_gate': 55385, 'p_gate_total': 1.0, 'p_gate_parent': 1.0},
        #               'Singlets': {'n_events_gate': 55385, 'p_gate_total': 1.0, 'p_gate_parent': 1.0},
        #               'Positive Unstained': {'n_events_gate': 75, 'p_gate_total': 0.0013541572627967862,
        #                                      'p_gate_parent': 0.0013541572627967862},
        #               'Positive BUV805': {'n_events_gate': 0, 'p_gate_total': 0.0, 'p_gate_parent': 0.0},
        #               'Positive Super Bright 436': {'n_events_gate': 0, 'p_gate_total': 0.0, 'p_gate_parent': 0.0}
        #               }
        if hierarchy_dict_node:
            name = hierarchy_dict_node['name']
            if name in self.statistics.keys():
                n_events_gate = f'{self.statistics[name]['n_events_gate']}'
                p_root = f'{self.statistics[name]['p_gate_total']*100:.2f}'
                p_parent = f'{self.statistics[name]['p_gate_parent']*100:.2f}'
                event_conc = f'{self.statistics[name]['event_conc']:.2f}' if not np.isnan(self.statistics[name]['event_conc']) else ''
            else:
                n_events_gate = ''
                p_root = ''
                p_parent = ''
                event_conc = ''

            item = TreeItem(name, n_events_gate, p_root, p_parent, event_conc, parent_item)
            parent_item.append_child(item)

            if 'children' in hierarchy_dict_node.keys():
                children = hierarchy_dict_node['children']
                for child in children:
                    self.setup_model_data(child, item)

    def update_statistics(self, new_statistics):
        """Update the statistics without rebuilding the tree"""
        self.statistics = new_statistics
        # print(json.dumps(new_statistics, indent=2))

        # Start recursive update from root
        self._update_tree_item_data(self.root_item)

        # # Notify the view that all data has changed todo for some reason the simple call to dataChanged doesn't work. layoutchanged works but it presumably slow. dataChanged for all items one by one works. Don't know why
        # self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1), [Qt.DisplayRole])
        # self.layoutChanged.emit()  # More reliable for structural changes

        # Emit dataChanged for all items in the tree
        self._emit_data_changed_for_tree(self.root_item)

    def _emit_data_changed_for_tree(self, item, parent_index=QModelIndex()):
        """Recursively emit dataChanged for all items in the tree"""
        if not item:
            return

        # Emit for this item if it's not the root
        if item != self.root_item:
            # Find this item's row
            if item.parent_item:
                row = item.parent_item.child_items.index(item)
                # Create indexes for all columns
                indexes = []
                for col in range(self.columnCount()):
                    index = self.createIndex(row, col, item)
                    if index.isValid():
                        indexes.append(index)

                # Emit dataChanged for this row
                if indexes:
                    self.dataChanged.emit(indexes[0], indexes[-1], [Qt.DisplayRole])

        # Process children
        for i, child in enumerate(item.child_items):
            child_parent_index = self.createIndex(i, 0, item) if item != self.root_item else QModelIndex()
            self._emit_data_changed_for_tree(child, child_parent_index)

    def _update_tree_item_data(self, item):
        """Recursively update TreeItem data based on new statistics"""
        if item.name in self.statistics:
            stats = self.statistics[item.name]

            # Update the TreeItem fields
            item.event_count = f"{stats['n_events_gate']}"
            item.p_root = f"{stats['p_gate_total'] * 100:.2f}"
            item.p_parent = f"{stats['p_gate_parent'] * 100:.2f}"
            item.event_conc = f"{stats['event_conc']:.2f}" if not np.isnan(stats['event_conc']) else ''

        # Recursively update children
        for child in item.child_items:
            self._update_tree_item_data(child)

    def columnCount(self, parent=QModelIndex()):
        return 5

    def rowCount(self, parent=QModelIndex()):
        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()
        return parent_item.child_count()

    def index(self, row, column, parent=QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()

        child_item = parent_item.child(row)
        if child_item:
            return self.createIndex(row, column, child_item)
        return QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()

        child_item = index.internalPointer()
        parent_item = child_item.parent()

        if parent_item == self.root_item or parent_item is None:
            return QModelIndex()

        return self.createIndex(parent_item.row(), 0, parent_item)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        item = index.internalPointer()
        col = index.column()

        if role == Qt.DisplayRole:
            if col == 0:
                return item.name
            elif col == 1:
                return item.event_count
            elif col == 2:
                return item.p_root
            elif col == 3:
                return item.p_parent
            elif col == 4:
                return item.event_conc
        elif role == Qt.TextAlignmentRole:
            if col != 0:
                return Qt.AlignRight | Qt.AlignVCenter

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return ["Gate Name", "Events", "%root", "%parent", "Conc [/uL]"][section]
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def set_dict(self, gating_hierarchy, statistics):
        self.statistics = statistics
        self.beginResetModel()
        self.root_item = TreeItem("Root", "", "", "", "")

        # print(['ROOT ITEM', self.root_item, gating_hierarchy])
        self.setup_model_data(gating_hierarchy, self.root_item)
        self.endResetModel()

    def copy_hierarchy(self, index):
        """Copy entire model data (including headers) to clipboard."""

        # Get headers
        headers = [self.headerData(c, Qt.Horizontal, Qt.DisplayRole) for c in range(self.columnCount())]

        # Get all data
        data_rows = []
        if index is None:
            item = self.root_item
        else:
            item = index.internalPointer()
        self._collect_all_data(item, data_rows)

        # Combine as tab-separated values (good for Excel / Office)
        lines = ["\t".join(headers)]
        for row in data_rows:
            lines.append("\t".join(row))

        text = "\n".join(lines)

        # Put to clipboard
        QApplication.clipboard().setText(text, QClipboard.Clipboard)

        # Support clipboard for X11 selection (optional)
        if QApplication.clipboard().supportsSelection():
            QApplication.clipboard().setText(text, QClipboard.Selection)

    def _collect_all_data(self, item, data_rows, parent_index=QModelIndex()):
        """Recursively collect data from all items in the tree"""
        # Skip the root item itself (it doesn't contain display data)
        if item != self.root_item:
            # Get data for this item
            row = []
            for c in range(self.columnCount()):
                # Create the index for this item and column
                if item.parent_item:
                    # Find this item's row position within its parent
                    row_position = item.parent_item.child_items.index(item)
                    # Create the index
                    index = self.createIndex(row_position, c, item)
                else:
                    # For top-level items
                    index = self.createIndex(0, c, item)

                text = self.data(index, Qt.DisplayRole)
                row.append("" if text is None else str(text))
            data_rows.append(row)

        # Recursively process all children
        for child_item in item.child_items:
            self._collect_all_data(child_item, data_rows)

class GatingHierarchyWidget(QWidget):
    def __init__(self, bus=None, mode=None, toolbar=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bus = bus
        self.mode = mode
        self.toolbar = toolbar
        self.data_for_cytometry_plots = None
        self.gating_hierarchy = None

        self.tree_view = QTreeView()
        self.model = DictTreeModel()
        self.tree_view.setModel(self.model)

        self.tree_view.setSelectionBehavior(QTreeView.SelectRows)
        # self.tree_view.setSelectionMode(QTreeView.ExtendedSelection)  # Multi-select
        # self.tree_view.setSelectionBehavior(QTreeView.SelectItems)
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.show_context_menu)

        # Template picker bar (per-sample gating). Choosing a template both shows
        # its gates and assigns the current sample to it; ＋ makes a new template.
        self.template_bar = self._build_template_bar()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.template_bar)
        layout.addWidget(self.tree_view)
        self.setLayout(layout)

        # Call header() on the tree_view instance
        header = self.tree_view.header()
        # Different resize modes for different columns
        header.setStretchLastSection(False)
        # Column 0: name
        self.tree_view.setColumnWidth(0, 250)
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        # Column 1: event count
        self.tree_view.setColumnWidth(1, 60)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        # Column 1: % root
        self.tree_view.setColumnWidth(2, 60)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        # Column 3: % parent
        self.tree_view.setColumnWidth(3, 60)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        # Column 4: event concentration
        self.tree_view.setColumnWidth(4, 60)
        header.setSectionResizeMode(4, QHeaderView.Stretch)

        if self.bus is not None:
            self.bus.changedGatingHierarchy.connect(self.update_hierarchy)
            self.bus.histsStatsRecalculated.connect(self.update_data)
            self.bus.templatesChanged.connect(self.populate_templates)

        # Install event filter on tree view
        self.tree_view.installEventFilter(self)

    def _build_template_bar(self):
        """A compact row above the tree: '<Raw|Unmixed> template: [combo ▼] [＋] [✎]'.

        The combo lists this scope's gating templates (raw and unmixed have
        independent lists); picking one selects it for the current sample. ＋
        creates a new (empty) template and selects it; ✎ renames the current one.
        """
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(4)

        scope_label = 'Unmixed' if self.mode == 'unmixed' else 'Raw'
        row.addWidget(QLabel(f"{scope_label} template:"))

        self.template_combo = QComboBox()
        self.template_combo.setToolTip(
            "Gating template for the current sample. Picking a template applies its "
            "gates to this sample; Time gates auto-scale to each sample."
        )
        self.template_combo.textActivated.connect(self._on_template_activated)
        row.addWidget(self.template_combo, 1)

        self.new_template_button = QToolButton()
        self.new_template_button.setText("＋")
        self.new_template_button.setToolTip("Create a new empty template (root only, no plots) and assign this sample to it")
        self.new_template_button.clicked.connect(self._on_new_template)
        row.addWidget(self.new_template_button)

        self.rename_template_button = QToolButton()
        self.rename_template_button.setText("✎")
        self.rename_template_button.setToolTip("Rename the current template")
        self.rename_template_button.clicked.connect(self._on_rename_template)
        row.addWidget(self.rename_template_button)

        return bar

    @Slot(str, list, str)
    def populate_templates(self, scope, names, active):
        """Refresh the template combo (controller -> UI) for this tree's scope.
        Ignores other scopes. Signals are blocked so this programmatic update
        never looks like a user selection."""
        if scope != self.mode:
            return
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        self.template_combo.addItems(names)
        if active in names:
            self.template_combo.setCurrentText(active)
        self.template_combo.blockSignals(False)

    def _on_template_activated(self, name):
        if self.bus is not None and name:
            self.bus.selectTemplateRequested.emit(self.mode, name)

    def _on_new_template(self):
        name, ok = QInputDialog.getText(self, "New template", "New empty template name:")
        name = (name or "").strip()
        if ok and name and self.bus is not None:
            self.bus.createTemplateRequested.emit(self.mode, name)

    def _on_rename_template(self):
        old = self.template_combo.currentText()
        if not old:
            return
        new, ok = QInputDialog.getText(self, "Rename template", "New name:", text=old)
        new = (new or "").strip()
        if ok and new and new != old and self.bus is not None:
            self.bus.renameTemplateRequested.emit(self.mode, old, new)

    def eventFilter(self, obj, event):
        if obj == self.tree_view and event.type() == event.Type.KeyPress:
            if event.matches(QKeySequence.Copy):
                self.model.copy_hierarchy(None)
                return True  # Event handled
        return super().eventFilter(obj, event)

    def set_toolbar(self, toolbar):
        self.toolbar = toolbar

    def toggle(self):
        sizes = self.parent().sizes()
        if sizes[1] > 0:  # Panel is visible
            sizes[1] = 0  # Collapse
            self.toolbar.action_toggle_gating_hierarchy.setChecked(False)
        else:  # Panel is collapsed
            width = sizes[0] + sizes[1]
            sizes[0] = 0.7 * width
            sizes[1] = 0.3 * width
            self.toolbar.action_toggle_gating_hierarchy.setChecked(True)
        self.parent().setSizes(sizes)

    def show_context_menu(self, position):
        # Get the index at the click position
        index = self.tree_view.indexAt(position)

        if index.isValid():
            # Item was clicked - get item data
            item_name = self.model.data(index.siblingAtColumn(0), Qt.DisplayRole)

            # Create context menu
            menu = QMenu(self)
            copy_action = menu.addAction(f"Copy hierarchy statistics from {item_name} downwards")
            copy_action.triggered.connect(lambda : self.model.copy_hierarchy(index))

        # todo add delete/rename, should be integrated with delete/rename menus on plots
        #
        # if index.isValid():
        #     # Item was clicked - get item data
        #     path = self.model.data(index.siblingAtColumn(2), Qt.DisplayRole)
        #     sample_name = str(Path(path).stem)
        #     # Add title to menu
        #     title_action = QAction(f"Actions for: {sample_name}", self)
        #     title_action.setEnabled(False)  # Make it non-clickable
        #     menu.addAction(title_action)
        #
        #     rename_action = QAction(f"Rename", self)
        #     rename_action.triggered.connect(lambda: self.rename_item(index))
        #     menu.addAction(rename_action)
        #
        #     delete_action = QAction(f"Delete", self)
        #     delete_action.triggered.connect(lambda: self.delete_item(index))
        #     menu.addAction(delete_action)
        #
        #
        # # Show the menu at cursor position

        menu.exec(self.tree_view.viewport().mapToGlobal(position))

    def rename_item(self, index):
        item_name = self.model.data(index.siblingAtColumn(0), Qt.DisplayRole)
        QMessageBox.information(self, "Rename", f"Renaming: {item_name}")

    def delete_item(self, index):
        item_name = self.model.data(index.siblingAtColumn(0), Qt.DisplayRole)
        reply = QMessageBox.question(self, "Delete", f"Are you sure you want to delete {item_name}?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            QMessageBox.information(self, "Deleted", f"Deleted: {item_name}")

    def init_data(self, data_for_cytometry_plots):
        self.data_for_cytometry_plots = data_for_cytometry_plots
        if self.data_for_cytometry_plots['gating']:
            self.update_hierarchy(mode=self.mode)

    @Slot(str, str)
    def update_hierarchy(self, mode=None, top_gate_id=None):
        if mode == self.mode:
            self.gating_hierarchy = self.data_for_cytometry_plots['gating'].get_gate_hierarchy(output='dict')
            self.model.set_dict(self.gating_hierarchy, self.data_for_cytometry_plots['statistics'])
            self.tree_view.expandAll()
            logger.info(f'GatingTree {self.mode}: hierarchy refreshed')

    @Slot(str)
    def update_data(self, mode):
        if mode == self.mode:
            if self.data_for_cytometry_plots is None:
                return
            self.model.update_statistics(self.data_for_cytometry_plots['statistics'])
            logger.info(f'GatingTree {self.mode} updated stats')


if __name__ == "__main__":
    import sys
    from honeychrome.controller import Controller

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.gating_tree = GatingHierarchyWidget(mode='raw')
            self.setCentralWidget(self.gating_tree)

            self.setWindowTitle("Gating Tree Viewer")
            self.resize(600, 800)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    kc = Controller()
    base_directory = Path.home() / 'spectral_cytometry'
    experiment_name = base_directory / '20240620 Spectral Symposium-poor cell unmixed'
    experiment_path = experiment_name.with_suffix('.kit')
    kc.load_experiment(experiment_path) # note this loads first sample too and runs calculate all histograms and statistics

    # note every time tab is changed (or every time plots or transforms changed), calculate all histograms and statistics
    kc.set_mode('Raw Data')
    kc.initialise_data_for_cytometry_plots()
    kc.calculate_lookup_tables()  # (re)create all lookup tabels

    window.gating_tree.init_data(kc.data_for_cytometry_plots)

    kc.load_sample(kc.experiment.samples['raw_samples'][0])
    window.gating_tree.update_data('raw')


    sys.exit(app.exec())




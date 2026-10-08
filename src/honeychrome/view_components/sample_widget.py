import shutil
from pathlib import Path

from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (QApplication, QMainWindow, QTreeView, QWidget, QVBoxLayout, QHeaderView, QToolBar, QMenu,
                               QMessageBox, QDialogButtonBox, QLineEdit, QDialog, QFileDialog)
from PySide6.QtCore import Qt, QAbstractItemModel, QModelIndex, Slot, QUrl, QThread
import csv
from pathvalidate import sanitize_filename

from honeychrome.controller import base_directory
from honeychrome.controller_components.exporter import ReportGenerator
from honeychrome.controller_components.functions import get_all_subfolders_recursive
from honeychrome.controller_components.unmixed_exporter import UnmixedExporter
from honeychrome.view_components.batch_export_samples_modal import BatchExportSamplesModal
from honeychrome.view_components.icon_loader import icon as load_icon, icon
from honeychrome.view_components.new_sample_modal import NewSampleModal

import logging
logger = logging.getLogger(__name__)

class SampleRenameDialog(QDialog):
    """Minimal dialog with just a QLineEdit for editing text."""
    def __init__(self, text="", existing_names=[], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Sample Name")
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

    def validate_and_accept(self):
        text = self.line_edit.text().strip()
        if not text:
            QMessageBox.warning(self, "Error", "Name cannot be empty.")
        elif text == self.old_name:
            self.reject()
        elif text in self.existing_names:
            QMessageBox.warning(self, "Error", f'"{text}" already exists.')
        else:
            self.accept()

    def getText(self):
        return self.line_edit.text()


class TreeItem:
    def __init__(self, icon, name, count, path, parent=None):
        self.icon = icon
        self.name = name
        self.path = path
        self.count = count
        self.parent_item = parent
        self.child_items = []

    def append_child(self, item):
        self.child_items.append(item)

    def child(self, row):
        return self.child_items[row]

    def child_count(self):
        return len(self.child_items)

    def column_count(self):
        return 3

    def row(self):
        if self.parent_item:
            return self.parent_item.child_items.index(self)
        return 0

    def parent(self):
        return self.parent_item

    def get_or_create_child(self, name):
        for c in self.child_items:
            if c.name == name:
                return c
        new_child = TreeItem("", name, "", "", parent=self)
        self.child_items.append(new_child)
        return new_child


class DictTreeModel(QAbstractItemModel):
    def __init__(self, full_data, parent=None):
        super().__init__(parent)
        self.full_data = full_data
        self.root_item = TreeItem(None, "All Samples", "", "")

    def _is_unstained(self, path: str) -> bool:
        """Return True if path is manually tagged OR name matches 'unstained' regex."""
        if path in self.full_data.get('unstained_samples', []):
            return True
        tube_name = self.full_data['all_samples'].get(path, '')
        import re
        return bool(re.search(r'unstained', tube_name, re.IGNORECASE)
                    or re.search(r'unstained', path, re.IGNORECASE))

    def setup_all_samples_from_dict(self, paths):
        title = 'All Samples'
        parent_item = TreeItem(None, title, "", self.root_item)
        self.root_item.append_child(parent_item)
        for path in paths:
            if path not in self.full_data['single_stain_controls']:
                title = self.full_data['all_samples'][path]
                nevents = self.full_data['all_sample_nevents'][path]
                if self._is_unstained(path):
                    icon = load_icon('droplet', colour='#4a90d9') if nevents > 0 else load_icon('droplet-empty')
                else:
                    icon = load_icon('droplet') if nevents > 0 else load_icon('droplet-empty')
                parts = path.replace("\\", "/").split("/")
                node = parent_item
                for part in parts[1:-1]:
                    if part.strip():
                        node = node.get_or_create_child(part)
                item = TreeItem(icon, title, nevents, path, node)
                node.append_child(item)

    def setup_single_stain_controls_from_list(self, data):
        title = 'Single Stain Controls'
        parent_item = TreeItem(None, title, "", self.root_item)
        self.root_item.append_child(parent_item)
        for path in data:
            nevents = self.full_data['all_sample_nevents'][path]
            if self._is_unstained(path):
                icon = load_icon('droplet', colour='#4a90d9') if nevents > 0 else load_icon('droplet-empty')
            else:
                icon = load_icon('droplet') if nevents > 0 else load_icon('droplet-empty')
            item = TreeItem(icon, str(Path(path).stem), nevents, path, parent_item)
            parent_item.append_child(item)

    def setup_model_data(self, data, parent_item):
        if isinstance(data, dict):
            for key, value in data.items():
                if key == 'single_stain_controls':
                    title = 'Single Stain Controls'
                elif key == 'all_samples':
                    title = 'All Samples'
                else:
                    title = value

                item = TreeItem(None, title, "", parent_item)
                parent_item.append_child(item)
                self.setup_model_data(value, item)

        elif isinstance(data, list):
            for path in data:
                nevents = self.full_data['all_sample_nevents'][path]
                if self._is_unstained(path):
                    icon = load_icon('droplet', colour='#4a90d9') if nevents > 0 else load_icon('droplet-empty')
                else:
                    icon = load_icon('droplet') if nevents > 0 else load_icon('droplet-empty')
                item = TreeItem(icon, str(Path(path).stem), nevents, path, parent_item)
                parent_item.append_child(item)
        else:
            # Single value, not dict or list
            item = TreeItem(None, str(data), "", "", parent_item)
            parent_item.append_child(item)

    def columnCount(self, parent=QModelIndex()):
        return 3

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
            if index.column() == 0:
                return item.name
            elif index.column() == 1:
                return item.count
            elif index.column() == 2:
                return item.path
        elif role == Qt.DecorationRole:
            # ONLY return icons for column 0 (Name column)
            if col == 0:
                return item.icon

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return ["Sample Name", "Events", "FCS Path"][section]
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def refresh_tree(self):
        self.beginResetModel()
        self.root_item = TreeItem(None, "Root", "", "")
        self.setup_single_stain_controls_from_list(self.full_data['single_stain_controls'])
        self.setup_all_samples_from_dict(self.full_data['all_samples'])
        self.endResetModel()

    def find_sample_row(self, sample_path, role=Qt.DisplayRole, parent=QModelIndex()):
        col = 2
        parent_row = parent.row()
        print(parent_row)
        rows = self.rowCount(parent)

        for row in range(rows):
            index = self.index(row, col, parent)

            # Check current item
            data = self.data(index, role)
            if data and sample_path == data:
                return parent_row + row

            # Recursively search children
            if self.hasChildren(index):
                row = self.find_sample_row(sample_path, role, index)
                return parent_row + row + 1

        return 0



    def find_index_iterative(self, column, value):
        """Iterative DFS search for first index whose data(column) == value."""
        stack = [QModelIndex()]  # start from root

        while stack:
            parent = stack.pop()
            row_count = self.rowCount(parent)

            # Iterate rows under current parent
            for row in range(row_count):
                idx = self.index(row, column, parent)

                if idx.data() == value:
                    return idx

                # Push children for later traversal (DFS)
                # Use column 0 for tree hierarchy
                # because children are structured by row, not column
                child_index = self.index(row, 0, parent)
                if self.rowCount(child_index) > 0:
                    stack.append(child_index)

        return QModelIndex()  # not found


class SampleWidget(QWidget):
    def __init__(self, bus=None, controller=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bus = bus
        self.controller = controller

        sample_toolbar = QToolBar("Sample Toolbar")
        sample_toolbar.setMovable(False)
        action_new_sample = QAction(load_icon('square-plus'), "New Sample", self)
        action_open_explorer = QAction(load_icon('folder-search'), "Open sample directory externally to create / modify folders", self)
        action_batch_add_samples = QAction(load_icon('file-import'), "Batch add samples from CSV", self)
        action_batch_add_samples.setToolTip('Batch add samples from CSV file \n\n'
                                            'CSV should have up to three columns: \n'
                                            '• (sample),\n'
                                            '• (group, sample),\n'
                                            '• or (category, group, sample).\n\n'
                                            'Note: Spectral Controls should be in two columns only,\n' 
                                            '• where group = \'single_stain_controls\'\n'
                                            '• and sample is in format \'Label (Cells)\' or \'Label (Beads)\'.')
        action_refresh = QAction(load_icon('refresh'), "Refresh sample view (rescan folder tree)", self)
        action_generate_report = QAction(load_icon('file-type-docx'), "Generate DOCX report on a selected sample \n(raw / unmixed / spectral process) according to settings", self)
        action_export_all = QAction(load_icon('file-export'), "Batch FCS export all unmixed", self)
        sample_toolbar.addAction(action_new_sample)
        sample_toolbar.addAction(action_batch_add_samples)
        sample_toolbar.addAction(action_open_explorer)
        sample_toolbar.addAction(action_refresh)
        sample_toolbar.addAction(action_generate_report)
        sample_toolbar.addAction(action_export_all)
        action_new_sample.triggered.connect(self.bus.showNewSampleWidget)
        action_batch_add_samples.triggered.connect(self.bus.batchAddSamples)
        action_open_explorer.triggered.connect(self.open_samples_folder)
        action_refresh.triggered.connect(self.refresh_sample_tree)
        action_generate_report.triggered.connect(self.generate_report)
        action_export_all.triggered.connect(self.show_export_modal)

        if not self.controller.experiment_compatible_with_acquisition:
            action_batch_add_samples.setVisible(False)

        # create tree view and select first sample
        self.tree_view = QTreeView()
        self.tree_view.setSelectionBehavior(QTreeView.SelectRows)
        self.tree_view.setSelectionMode(QTreeView.SingleSelection)
        self.model = DictTreeModel(self.controller.experiment.samples)
        self.tree_view.setModel(self.model)

        # Connect selection changed signal
        self.tree_view.selectionModel().selectionChanged.connect(self.open_item)
        # Enable context menu
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.show_context_menu)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(sample_toolbar)
        layout.addWidget(self.tree_view)
        self.setLayout(layout)

        # Call header() on the tree_view instance
        header = self.tree_view.header()
        # Different resize modes for different columns
        header.setStretchLastSection(False)
        # Column 0: name
        self.tree_view.setColumnWidth(0, 400)
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        # Column 1: count
        self.tree_view.setColumnWidth(1, 80)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        # Column 2: path
        header.setSectionResizeMode(2, QHeaderView.Stretch)

        self.bus.showNewSampleWidget.connect(self.show_new_sample_modal)
        self.bus.batchAddSamples.connect(self.batch_add_samples_csv)
        self.bus.sampleTreeUpdated.connect(self.refresh_sample_tree)
        self.bus.selectSample.connect(self.set_selection)
        self.bus.showExportModal.connect(self.show_export_modal)
        self.bus.batchExportRequested.connect(self.start_export_thread)
        self.bus.generateSampleReport.connect(self.generate_report)

        self.refresh_sample_tree()

        self.thread = None
        self.unmixed_exporter = None
        self.report_generator = None

        if not self.controller.experiment_compatible_with_acquisition:
            action_new_sample.setVisible(False)

    def show_new_sample_modal(self):
        path = str(self.controller.experiment_dir / self.controller.experiment.settings['raw']['raw_samples_subdirectory'])
        dialog = NewSampleModal(self, self.bus, path, self.controller.experiment_dir)
        dialog.open()

    def show_export_modal(self):
        raw = self.controller.experiment.settings['raw']['raw_samples_subdirectory']
        ssc = self.controller.experiment.settings['raw']['single_stain_controls_subdirectory']
        exp_dir = self.controller.experiment_dir

        raw_path = str(exp_dir / raw)
        raw_label = Path(raw).name  # e.g. "Raw"

        if Path(ssc).is_relative_to(raw):
            # Normal case: ssc is nested inside raw, walking raw covers everything.
            dialog = BatchExportSamplesModal(self, self.bus, raw_path, exp_dir, root_label=raw_label, extra_roots=None)
        else:
            # Symlink case: ssc lives outside raw as a top-level symlink.
            # Pass it as a second root so both appear in the modal with their
            # user-facing names from the experiment settings.
            ssc_path = str(exp_dir / ssc)
            ssc_label = Path(ssc).name  # e.g. "SSC" or "Link_to__..."
            dialog = BatchExportSamplesModal(self, self.bus, raw_path, exp_dir, root_label=raw_label, extra_roots=[(ssc_path, ssc_label)])

        dialog.open()

    @Slot(str, bool)
    def start_export_thread(self, folder, subsample):
        self.thread = QThread()
        self.unmixed_exporter = UnmixedExporter(folder, subsample, self.bus, self.controller)
        self.unmixed_exporter.moveToThread(self.thread)
        self.thread.started.connect(self.unmixed_exporter.run)
        self.unmixed_exporter.finished.connect(self.thread.quit)
        self.unmixed_exporter.finished.connect(self.unmixed_exporter.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def show_context_menu(self, position):
        # Get the index at the click position
        index = self.tree_view.indexAt(position)

        # Create context menu
        menu = QMenu(self)

        if index.isValid():
            # Item was clicked - get item data
            path = self.model.data(index.siblingAtColumn(2), Qt.DisplayRole)
            if isinstance(path, str):
                sample_name = str(Path(path).stem)
                menu.addSection(f"Actions for: {sample_name}")

                action_generate_report = QAction("Generate Sample Report", self)
                action_generate_report.triggered.connect(self.generate_report)
                menu.addAction(action_generate_report)

                rename_action = QAction("Rename", self)
                rename_action.triggered.connect(lambda: self.rename_item(path))
                menu.addAction(rename_action)

                delete_action = QAction("Delete", self)
                delete_action.triggered.connect(lambda: self.delete_item(path))
                menu.addAction(delete_action)

                # Add item-specific actions
                if path in self.model.full_data['all_samples']:
                    # Unstained toggle
                    unstained_list = self.controller.experiment.samples.get('unstained_samples', [])
                    is_unstained = path in unstained_list
                    mark_action = QAction("Unmark as Unstained" if is_unstained else "Mark as Unstained", self)
                    mark_action.setCheckable(True)
                    mark_action.setChecked(is_unstained)
                    mark_action.triggered.connect(lambda checked, p=path: self.toggle_unstained(p))
                    menu.addAction(mark_action)

                    source_folder = str(self.controller.experiment_dir / self.controller.experiment.settings['raw']['raw_samples_subdirectory'])
                    folders = get_all_subfolders_recursive(source_folder, self.controller.experiment_dir)
                    move_actions = []
                    menu.addSection('Move to:')
                    for folder in folders:
                        move_action = QAction(str(folder), self)
                        move_action.triggered.connect(lambda checked, f=folder: self.move_to_folder(path, str(f)))
                        menu.addAction(move_action)

        # Show the menu at cursor position
        menu.exec(self.tree_view.viewport().mapToGlobal(position))

    def open_item(self, selected):
        indexes = selected.indexes()
        if len(indexes) < 3:
            return
        datum = self.model.data(indexes[2])

        if isinstance(datum, str):
            if datum: # i.e. not empty
                path = datum
                self.bus.loadSampleRequested.emit(path)

    def move_to_folder(self, path, folder):
        sample_path = self.controller.experiment_dir / path
        print(sample_path, folder)
        shutil.move(str(sample_path), str(folder))
        self.refresh_sample_tree()

    def generate_report(self):
        # export() creates/renders real QWidgets (plots, heatmaps, NxN grid) to
        # build the report images, so it must run on the main GUI thread — do not
        # move ReportGenerator to a QThread here.
        self.report_generator = ReportGenerator(self.bus, self.controller)
        self.report_generator.export()

    def rename_item(self, path):
        sample_path = self.controller.experiment_dir / path
        sample_name = str(sample_path.stem)
        dlg = SampleRenameDialog(sample_name, existing_names=self.model.full_data['all_samples'].values(), parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # a path separator would move the file into a non-existent subfolder
            new_name = dlg.getText().replace('/', '_').replace('\\', '_')
            if new_name.strip():
                sample_path.rename(sample_path.parent / f'{new_name}.fcs')
                self.refresh_sample_tree()

    def delete_item(self, path):
        sample_path = self.controller.experiment_dir / path
        reply = QMessageBox.question(self, "Delete", f"Are you sure you want to delete {path}?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            sample_path.unlink()
            self.refresh_sample_tree()

    def toggle_unstained(self, path):
        unstained = self.controller.experiment.samples.setdefault('unstained_samples', [])
        if path in unstained:
            unstained.remove(path)
        else:
            unstained.append(path)
        self.bus.autoSaveRequested.emit()
        self.model.refresh_tree()
        self.tree_view.expandAll()
        if self.bus:
            self.bus.sampleTreeUpdated.emit()

    def refresh_sample_tree(self):
        self.controller.experiment.scan_sample_tree()
        self.model.refresh_tree()
        self.tree_view.expandAll()

    def open_samples_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.controller.experiment_dir))

    @Slot()
    def batch_add_samples_csv(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open List of Samples in CSV File", str(base_directory), "CSV file (*.csv)")
        if filepath:
            try:
                with open(filepath, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                if not rows:
                    return []

                # Get header if exists
                header = rows[0] if any(cell.lower() in ['category', 'group', 'sample', 'categories', 'groups', 'samples'] for cell in rows[0]) else None
                data = rows[1:] if header else rows
                num_cols = len(data[0]) if data else 0

                if num_cols == 3:
                    for row in data:
                        if row[1] == 'single_stain_controls':
                            row[0] = 'single_stain_controls'

                if num_cols in [1, 2, 3]:
                    data = [[sanitize_filename(cell) for cell in row] for row in data]
                    # note this raises valueerror if filename invalid
                    paths = ['/'.join(row) for row in data]
                    if len(paths) != len(set(paths)):
                        raise ValueError(f"Duplicate sample paths were found in list: {num_cols}")

                if num_cols == 1:
                    data = [row[0] for row in data]  # Just samples
                elif num_cols == 2:
                    data = [(row[0], row[1]) for row in data]  # (group, sample)
                elif num_cols == 3:
                    data = [(row[0], row[1], row[2]) for row in data]  # (category, group, sample)
                else:
                    raise ValueError(f"Unexpected number of columns: {num_cols}")

                self.controller.batch_new_samples(data, num_cols)

            except Exception as e:
                QMessageBox.warning(self, "Failed to read sample list",
                    f"{e}",
                    buttons=QMessageBox.StandardButton.Ok)

    @Slot(str)
    def set_selection(self, current_sample_path):
        index = self.model.find_index_iterative(2, current_sample_path)

        if index.isValid():
            self.tree_view.setCurrentIndex(index)
            self.tree_view.scrollTo(index)
        else:
            if current_sample_path:
                logger.info(f"SampleWidget: {current_sample_path} not found")
            else:
                logger.info('SampleWidget: no sample selected')


if __name__ == "__main__":
    import sys
    from honeychrome.controller import Controller
    from honeychrome.view_components.event_bus import EventBus

    bus = EventBus()
    bus.loadSampleRequested.connect(lambda path: print(path))

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()

            kc = Controller()
            base_directory = Path.home() / 'spectral_cytometry'
            experiment_name = base_directory / '20240620 Spectral Symposium-poor cell unmixed'
            experiment_path = experiment_name.with_suffix('.kit')
            kc.load_experiment(
                experiment_path)  # note this loads first sample too and runs calculate all histograms and statistics


            self.sample_tree = SampleWidget(bus=bus, parent=None, controller=kc)
            self.setCentralWidget(self.sample_tree)

            self.setWindowTitle("Sample Tree Viewer")
            self.resize(600, 800)

            # self.sample_tree.set_selection('Raw/Cell controls/Reference Group/A2 Spark UV 387 (Cells)_Cell controls.fcs')
            self.sample_tree.set_selection('Raw/Samples/AF controls/G4 Tg Spleen_Samples.fcs')


    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec())

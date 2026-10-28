from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, QLabel, QAbstractItemView, QFrame)
from PySide6.QtCore import Qt, Signal


class OrderedMultiSamplePicker(QFrame):
    '''
    class OrderedMultiSamplePicker

    A two column sample picker. Provides the following methods:
        set_items - populate the list
        get_ordered_list - retrieve the list of sample filenames

    Provides a signal to emit the current list whenever the selection is changed:
        changed
    '''

    # Signal emitted whenever the order or selection changes
    changed = Signal(list)

    def __init__(self, title="Samples Available", source_samples=None):
        super().__init__()
        self.source_samples = source_samples or []
        self.init_ui(title)

    def _create_styled_frame(self):
        """Helper to create a consistent border around layouts."""
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Raised)
        # Optional: Add a CSS-like border for more control
        frame.setStyleSheet("QListWidget { border: 1px solid #bdc3c7; border-radius: 4px; }")
        return frame

    def init_ui(self, title):
        layout = QHBoxLayout(self)

        # --- Left Side: Available ---
        self.avail_frame = self._create_styled_frame()
        left_layout = QVBoxLayout(self.avail_frame)
        left_layout.addWidget(QLabel(title))
        self.available_list = QListWidget()
        self.available_list.addItems(self.source_samples)
        self.available_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        left_layout.addWidget(self.available_list)
        layout.addWidget(self.avail_frame)

        # --- Center: Transfer Buttons ---
        btn_layout = QVBoxLayout()
        btn_layout.addStretch()
        self.add_btn = QPushButton("Add ➔")
        self.remove_btn = QPushButton("⬅ Remove")
        self.add_btn.clicked.connect(self.move_right)
        self.remove_btn.clicked.connect(self.move_left)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # --- Right Side: Selected & Ordered ---
        self.select_frame = self._create_styled_frame()
        right_layout = QVBoxLayout(self.select_frame)
        right_layout.addWidget(QLabel("Selected Samples"))
        self.selected_list = QListWidget()
        self.selected_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        right_layout.addWidget(self.selected_list)

        # Up/Down ordering buttons
        ord_btn_layout = QHBoxLayout()
        self.up_btn = QPushButton("↑ Move Up")
        self.down_btn = QPushButton("↓ Move Down")
        self.up_btn.clicked.connect(lambda: self.reorder(-1))
        self.down_btn.clicked.connect(lambda: self.reorder(1))
        ord_btn_layout.addWidget(self.up_btn)
        ord_btn_layout.addWidget(self.down_btn)
        right_layout.addLayout(ord_btn_layout)

        layout.addWidget(self.select_frame)

    def move_right(self):
        for item in self.available_list.selectedItems():
            self.selected_list.addItem(self.available_list.takeItem(self.available_list.row(item)))
        self.emit_changed()

    def move_left(self):
        for item in self.selected_list.selectedItems():
            self.available_list.addItem(self.selected_list.takeItem(self.selected_list.row(item)))
        self.emit_changed()

    def reorder(self, delta):
        curr_row = self.selected_list.currentRow()
        if curr_row == -1: return

        new_row = curr_row + delta
        if 0 <= new_row < self.selected_list.count():
            item = self.selected_list.takeItem(curr_row)
            self.selected_list.insertItem(new_row, item)
            self.selected_list.setCurrentRow(new_row)
            self.emit_changed()

    def emit_changed(self):
        # Returns the list of strings in the current UI order
        current_order = [self.selected_list.item(i).text() for i in range(self.selected_list.count())]
        self.changed.emit(current_order)

    def get_ordered_list(self):
        return [self.selected_list.item(i).text() for i in range(self.selected_list.count())]

    def clear(self):
        """Removes all items from both the available and selected lists."""
        self.available_list.clear()
        self.selected_list.clear()
        self.emit_changed()

    def set_items(self, source_samples, selected=None):
        """
        Resets the picker with a new set of strings.

        Args:
            items (list): The full pool of available strings.
            selected (list, optional): Strings that should start in the 'Selected' column.
        """
        self.clear()
        self.source_samples = source_samples
        selected = selected or []

        # Add to the 'Selected' list (maintaining the provided order)
        self.selected_list.addItems(selected)

        # Add everything else to the 'Available' list
        remaining = [i for i in source_samples if i not in selected]
        self.available_list.addItems(remaining)

        self.emit_changed()
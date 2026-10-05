"""
Cytkit State plugin (hardware monitor and settings)
"""
from PySide6.QtGui import QCursor, QIntValidator
from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QPushButton, QLabel, QTabWidget, QToolBox, QFormLayout, QComboBox, QCheckBox, QSpinBox, QHBoxLayout, QFrame, QTableWidget, QHeaderView, QLineEdit, QDoubleSpinBox
from PySide6.QtCore import Qt, Slot, Signal, QSize, QSettings
from PySide6.QtWidgets import QApplication

import logging

from honeychrome.instrument_driver_components.cytkit_components.alignment_camera import AlignmentCameraWidget
from honeychrome.instrument_driver_components.cytkit_components.cytkit_configuration import monitor_dictionary, dac_dictionary, number_of_dacs_pairs, registers_map
from honeychrome.main import configure_multiprocessing
from honeychrome.settings import heading_style, adc_channels
from honeychrome.view_components.event_bus import EventBus
from honeychrome.view_components.icon_loader import icon
from honeychrome import settings

q_settings = settings.q_settings

logger = logging.getLogger(__name__)

plugin_name = 'Cytkit Hardware'

class AdcChannelTable(QWidget):
    enable_changed = Signal(int, str, bool)
    invert_changed = Signal(int, str, bool)
    offset_changed = Signal(int, str, int)

    SETTINGS_GROUP = "adc_channels"

    def __init__(self, parent=None):
        super().__init__(parent)

        # guard so programmatic changes (load_state / reset) don't trigger saves
        self._loading = False

        # ---- table ----
        self.table = QTableWidget(len(adc_channels), 4)
        self.table.setHorizontalHeaderLabels(
            ["Enable", "Current Sample", "Invert", "Offset"]
        )
        self.table.verticalHeader().setVisible(True)
        self.table.setVerticalHeaderLabels(adc_channels)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        total_height = (self.table.verticalHeader().length() + self.table.horizontalHeader().height() + (self.table.frameWidth() * 2))
        self.table.setFixedHeight(total_height)

        # ---- per-row widgets ----
        self.enable_boxes: list[QCheckBox] = []
        self.sample_labels: list[QLabel] = []
        self.invert_boxes: list[QCheckBox] = []
        self.offset_spins: list[QSpinBox] = []

        for row, name in enumerate(adc_channels):
            enable = QCheckBox()
            enable.toggled.connect(
                lambda checked, r=row, n=name: self._on_enable(r, n, checked)
            )
            self.enable_boxes.append(enable)
            self.table.setCellWidget(row, 0, self._centered(enable))

            sample = QLabel("—")
            sample.setAlignment(Qt.AlignCenter)
            self.sample_labels.append(sample)
            self.table.setCellWidget(row, 1, sample)

            invert = QCheckBox()
            invert.toggled.connect(
                lambda checked, r=row, n=name: self._on_invert(r, n, checked)
            )
            self.invert_boxes.append(invert)
            self.table.setCellWidget(row, 2, self._centered(invert))

            offset = QSpinBox()
            offset.setRange(0, 65535)
            offset.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            offset.valueChanged.connect(
                lambda value, r=row, n=name: self._on_offset(r, n, value)
            )
            self.offset_spins.append(offset)
            self.table.setCellWidget(row, 3, offset)

        # ---- reset button ----
        self.reset_button = QPushButton("Reset to defaults")
        self.reset_button.clicked.connect(self.reset)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.reset_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        layout.addLayout(button_row)

        # load the saved settings
        self.load_state()

    # ---------- change handlers (auto-save) ----------

    def _on_enable(self, row: int, name: str, checked: bool) -> None:
        self._save_channel(name)
        self.enable_changed.emit(row, name, checked)

    def _on_invert(self, row: int, name: str, checked: bool) -> None:
        self._save_channel(name)
        self.invert_changed.emit(row, name, checked)

    def _on_offset(self, row: int, name: str, value: int) -> None:
        self._save_channel(name)
        self.offset_changed.emit(row, name, value)

    # ---------- helpers ----------

    @staticmethod
    def _centered(widget: QWidget) -> QWidget:
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.setAlignment(Qt.AlignCenter)
        h.addWidget(widget)
        return container

    def _row_for(self, name: str) -> int:
        return adc_channels.index(name)

    # ---------- persistence ----------

    def _save_channel(self, name: str) -> None:
        """Persist a single channel's state. Skipped while loading/resetting."""
        if self._loading:
            return
        row = self._row_for(name)
        s = q_settings
        s.beginGroup(self.SETTINGS_GROUP)
        s.beginGroup(name)
        s.setValue("enabled", self.enable_boxes[row].isChecked())
        s.setValue("inverted", self.invert_boxes[row].isChecked())
        s.setValue("offset", self.offset_spins[row].value())
        s.endGroup()
        s.endGroup()

    def save_state(self) -> None:
        """Write state for every channel (e.g. on app close)."""
        s = q_settings
        s.beginGroup(self.SETTINGS_GROUP)
        for row, name in enumerate(adc_channels):
            s.beginGroup(name)
            s.setValue("enabled", self.enable_boxes[row].isChecked())
            s.setValue("inverted", self.invert_boxes[row].isChecked())
            s.setValue("offset", self.offset_spins[row].value())
            s.endGroup()
        s.endGroup()
        s.sync()

    def load_state(self) -> None:
        """Read state from QSettings; programmatic changes are not re-saved."""
        for row, name in enumerate(adc_channels):
            enabled = settings.adc_enabled_retrieved[name]
            inverted = settings.adc_inverted_retrieved[name]
            offset = settings.adc_offset_retrieved[name]
            self._set_row_silently(row, enabled, inverted, offset)


    def _set_row_silently(
        self, row: int, enabled: bool, inverted: bool, offset: int
    ) -> None:
        widgets = (
            self.enable_boxes[row],
            self.invert_boxes[row],
            self.offset_spins[row],
        )
        for w in widgets:
            w.blockSignals(True)
        self.enable_boxes[row].setChecked(enabled)
        self.invert_boxes[row].setChecked(inverted)
        self.offset_spins[row].setValue(offset)
        for w in widgets:
            w.blockSignals(False)

    # ---------- reset ----------

    def reset(self) -> None:
        """Restore defaults (disabled, not inverted, offset 0) and persist."""
        self._loading = True
        try:
            for row, name in enumerate(adc_channels):
                self._set_row_silently(row, enabled=True, inverted=False if name=='SSC' else True, offset=0)
        finally:
            self._loading = False

        # persist the reset state and notify listeners
        self.save_state()
        for row, name in enumerate(adc_channels):
            self.enable_changed.emit(row, name, True)
            self.invert_changed.emit(row, name, False if name=='SSC' else True)
            self.offset_changed.emit(row, name, 0)

    # ---------- sample labels ----------

    def update_samples(self, samples) -> None:
        if isinstance(samples, dict):
            values = [samples.get(name, "—") for name in adc_channels]
        else:
            values = list(samples)
            if len(values) != len(adc_channels):
                raise ValueError(
                    f"expected {len(adc_channels)} samples, got {len(values)}"
                )
        for label, value in zip(self.sample_labels, values):
            label.setText("—" if value is None else str(value))

    def update_sample(self, row: int, value) -> None:
        self.sample_labels[row].setText("—" if value is None else str(value))

    # ---------- accessors ----------

    def is_enabled(self, row: int) -> bool:
        return self.enable_boxes[row].isChecked()

    def is_inverted(self, row: int) -> bool:
        return self.invert_boxes[row].isChecked()

    def offset(self, row: int) -> int:
        return self.offset_spins[row].value()

    def row_state(self, row: int) -> dict:
        return {
            "channel": adc_channels[row],
            "enabled": self.is_enabled(row),
            "inverted": self.is_inverted(row),
            "offset": self.offset(row),
            "sample": self.sample_labels[row].text(),
        }

    def all_states(self) -> list[dict]:
        return [self.row_state(r) for r in range(len(adc_channels))]

class TriggerChannelTable(QWidget):
    enable_changed = Signal(int, str, bool)
    mask_changed = Signal(int, str, bool)
    edge_changed = Signal(int, str, bool)
    level_changed = Signal(int, str, int)
    hyst_changed = Signal(int, str, int)
    delay_changed = Signal(int, str, int)
    h_off_changed = Signal(int, str, int)
    skew_changed = Signal(int, str, int)

    SETTINGS_GROUP = "trigger_channels"

    def __init__(self, parent=None):
        super().__init__(parent)

        # guard so programmatic changes (load_state / reset) don't trigger saves
        self._loading = False

        # ---- table ----
        self.table = QTableWidget(len(adc_channels), 9)
        self.table.setHorizontalHeaderLabels(
            ["Enable", "Mask", "Edge", "Level", "Hysteresis", "Delay", "Hold-off", "Skew", "Events"]
        )
        self.table.verticalHeader().setVisible(True)
        self.table.setVerticalHeaderLabels(adc_channels)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        total_height = (self.table.verticalHeader().length() + self.table.horizontalHeader().height() + (self.table.frameWidth() * 2))
        self.table.setFixedHeight(total_height)

        # ---- per-row widgets ----
        self.enable_boxes: list[QCheckBox] = []
        self.mask_boxes: list[QCheckBox] = []
        self.edge_boxes: list[QCheckBox] = []
        self.level_spins: list[QSpinBox] = []
        self.hyst_spins: list[QSpinBox] = []
        self.delay_spins: list[QSpinBox] = []
        self.h_off_spins: list[QSpinBox] = []
        self.skew_spins: list[QSpinBox] = []
        self.events_labels: list[QLabel] = []

        for row, name in enumerate(adc_channels):
            enable = QCheckBox()
            enable.toggled.connect(
                lambda checked, r=row, n=name: self._on_enable(r, n, checked)
            )
            self.enable_boxes.append(enable)
            self.table.setCellWidget(row, 0, self._centered(enable))

            mask = QCheckBox()
            mask.toggled.connect(
                lambda checked, r=row, n=name: self._on_mask(r, n, checked)
            )
            self.mask_boxes.append(mask)
            self.table.setCellWidget(row, 1, self._centered(mask))

            edge = QCheckBox()
            edge.toggled.connect(
                lambda checked, r=row, n=name: self._on_edge(r, n, checked)
            )
            self.edge_boxes.append(edge)
            self.table.setCellWidget(row, 2, self._centered(edge))

            level = QSpinBox()
            level.setRange(0, 16383)
            level.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            level.valueChanged.connect(
                lambda value, r=row, n=name: self._on_level(r, n, value)
            )
            self.level_spins.append(level)
            self.table.setCellWidget(row, 3, level)

            hyst = QSpinBox()
            hyst.setRange(0, 65535)
            hyst.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            hyst.valueChanged.connect(
                lambda value, r=row, n=name: self._on_hyst(r, n, value)
            )
            self.hyst_spins.append(hyst)
            self.table.setCellWidget(row, 4, hyst)

            delay = QSpinBox()
            delay.setRange(0, 65535)
            delay.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            delay.valueChanged.connect(
                lambda value, r=row, n=name: self._on_delay(r, n, value)
            )
            self.delay_spins.append(delay)
            self.table.setCellWidget(row, 5, delay)

            h_off = QSpinBox()
            h_off.setRange(0, 65535)
            h_off.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            h_off.valueChanged.connect(
                lambda value, r=row, n=name: self._on_h_off(r, n, value)
            )
            self.h_off_spins.append(h_off)
            self.table.setCellWidget(row, 6, h_off)

            skew = QSpinBox()
            skew.setRange(0, 65535)
            skew.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            skew.valueChanged.connect(
                lambda value, r=row, n=name: self._on_skew(r, n, value)
            )
            self.skew_spins.append(skew)
            self.table.setCellWidget(row, 7, skew)

            events = QLabel("—")
            events.setAlignment(Qt.AlignCenter)
            self.events_labels.append(events)
            self.table.setCellWidget(row, 8, events)

        # ---- reset button ----
        self.reset_button = QPushButton("Reset to defaults")
        self.reset_button.clicked.connect(self.reset)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.reset_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        layout.addLayout(button_row)

        # load the saved settings
        self.load_state()

    # ---------- change handlers (auto-save) ----------

    def _on_enable(self, row: int, name: str, checked: bool) -> None:
        self._save_channel(name)
        self.enable_changed.emit(row, name, checked)

    def _on_mask(self, row: int, name: str, checked: bool) -> None:
        self._save_channel(name)
        self.mask_changed.emit(row, name, checked)

    def _on_edge(self, row: int, name: str, checked: bool) -> None:
        self._save_channel(name)
        self.edge_changed.emit(row, name, checked)

    def _on_level(self, row: int, name: str, value: int) -> None:
        self._save_channel(name)
        self.level_changed.emit(row, name, value)

    def _on_hyst(self, row: int, name: str, value: int) -> None:
        self._save_channel(name)
        self.hyst_changed.emit(row, name, value)

    def _on_delay(self, row: int, name: str, value: int) -> None:
        self._save_channel(name)
        self.delay_changed.emit(row, name, value)

    def _on_h_off(self, row: int, name: str, value: int) -> None:
        self._save_channel(name)
        self.h_off_changed.emit(row, name, value)

    def _on_skew(self, row: int, name: str, value: int) -> None:
        self._save_channel(name)
        self.skew_changed.emit(row, name, value)



    # ---------- helpers ----------

    @staticmethod
    def _centered(widget: QWidget) -> QWidget:
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.setAlignment(Qt.AlignCenter)
        h.addWidget(widget)
        return container

    def _row_for(self, name: str) -> int:
        return adc_channels.index(name)

    # ---------- persistence ----------
        self.enable_boxes: list[QCheckBox] = []
        self.mask_boxes: list[QCheckBox] = []
        self.edge_boxes: list[QCheckBox] = []
        self.level_spins: list[QSpinBox] = []
        self.hyst_spins: list[QSpinBox] = []
        self.delay_spins: list[QSpinBox] = []
        self.h_off_spins: list[QSpinBox] = []
        self.skew_spins: list[QSpinBox] = []
        self.events_labels: list[QLabel] = []

    def _save_channel(self, name: str) -> None:
        """Persist a single channel's state. Skipped while loading/resetting."""
        if self._loading:
            return
        row = self._row_for(name)
        s = q_settings
        s.beginGroup(self.SETTINGS_GROUP)
        s.beginGroup(name)
        s.setValue("enabled", self.enable_boxes[row].isChecked())
        s.setValue("mask", self.mask_boxes[row].isChecked())
        s.setValue("edge", self.edge_boxes[row].isChecked())
        s.setValue("level", self.level_spins[row].value())
        s.setValue("hyst", self.hyst_spins[row].value())
        s.setValue("delay", self.delay_spins[row].value())
        s.setValue("h_off", self.h_off_spins[row].value())
        s.setValue("skew", self.skew_spins[row].value())
        s.endGroup()
        s.endGroup()

    def save_state(self) -> None:
        """Write state for every channel (e.g. on app close)."""
        s = q_settings
        s.beginGroup(self.SETTINGS_GROUP)
        for row, name in enumerate(adc_channels):
            s.beginGroup(name)
            s.setValue("enabled", self.enable_boxes[row].isChecked())
            s.setValue("mask", self.mask_boxes[row].isChecked())
            s.setValue("edge", self.edge_boxes[row].isChecked())
            s.setValue("level", self.level_spins[row].value())
            s.setValue("hyst", self.hyst_spins[row].value())
            s.setValue("delay", self.delay_spins[row].value())
            s.setValue("h_off", self.h_off_spins[row].value())
            s.setValue("skew", self.skew_spins[row].value())
            s.endGroup()
        s.endGroup()
        s.endGroup()
        s.sync()

    def load_state(self) -> None:
        """Read state from QSettings; programmatic changes are not re-saved."""

        for row, name in enumerate(adc_channels):
            enabled = settings.trigger_enabled_retrieved[name]
            mask = settings.trigger_mask_retrieved[name]
            edge = settings.trigger_edge_retrieved[name]
            level = settings.trigger_level_retrieved[name]
            hyst = settings.trigger_hyst_retrieved[name]
            delay = settings.trigger_delay_retrieved[name]
            h_off = settings.trigger_h_off_retrieved[name]
            skew = settings.trigger_skew_retrieved[name]

            self._set_row_silently(row, enabled, mask, edge, level, hyst, delay, h_off, skew)


    def _set_row_silently(
        self, row: int, enabled: bool, mask: bool, edge: bool, level: int, hyst: int, delay: int, h_off: int, skew: int
    ) -> None:
        widgets = (
            self.enable_boxes[row],
            self.mask_boxes[row],
            self.edge_boxes[row],
            self.level_spins[row],
            self.hyst_spins[row],
            self.delay_spins[row],
            self.h_off_spins[row],
            self.skew_spins[row],
        )
        for w in widgets:
            w.blockSignals(True)
        self.enable_boxes[row].setChecked(enabled)
        self.mask_boxes[row].setChecked(mask)
        self.edge_boxes[row].setChecked(edge)
        self.level_spins[row].setValue(level)
        self.hyst_spins[row].setValue(hyst)
        self.delay_spins[row].setValue(delay)
        self.h_off_spins[row].setValue(h_off)
        self.skew_spins[row].setValue(skew)
        for w in widgets:
            w.blockSignals(False)

    # ---------- reset ----------

    def reset(self) -> None:
        """Restore defaults (disabled, not inverted, offset 0) and persist."""
        self._loading = True
        try:
            for row, name in enumerate(adc_channels):
                self._set_row_silently(row, enabled=True if name=='FSC' else False, mask=True if name=='FSC' else False, edge=True if name=='FSC' else False, level=settings.threshold * settings.adc_scale_mv, hyst=settings.trigger_default_hysteresis, delay=0, h_off=0, skew=0)
        finally:
            self._loading = False

        # persist the reset state and notify listeners
        self.save_state()
        for row, name in enumerate(adc_channels):
            self.enable_changed.emit(row, name, True if name=='FSC' else False)
            self.mask_changed.emit(row, name, True if name=='FSC' else False)
            self.edge_changed.emit(row, name, True if name=='FSC' else False)
            self.level_changed.emit(row, name, settings.threshold * settings.adc_scale_mv)
            self.hyst_changed.emit(row, name, settings.trigger_default_hysteresis)
            self.delay_changed.emit(row, name, 0)
            self.h_off_changed.emit(row, name, 0)
            self.skew_changed.emit(row, name, 0)

    # ---------- events labels ----------

    def update_all_events(self, all_events) -> None:
        values = [all_events.get(name, "—") for name in adc_channels]
        for label, value in zip(self.events_labels, values):
            label.setText("—" if value is None else str(value))

    def update_events(self, row: int, value) -> None:
        self.events_labels[row].setText("—" if value is None else str(value))

    # ---------- accessors ----------

    def is_enabled(self, row: int) -> bool:
        return self.enable_boxes[row].isChecked()

    def is_mask(self, row: int) -> bool:
        return self.mask_boxes[row].isChecked()

    def is_edge(self, row: int) -> bool:
        return self.edge_boxes[row].isChecked()

    def level(self, row: int) -> int:
        return self.level_spins[row].value()

    def hyst(self, row: int) -> int:
        return self.hyst_spins[row].value()

    def delay(self, row: int) -> int:
        return self.delay_spins[row].value()

    def h_off(self, row: int) -> int:
        return self.h_off_spins[row].value()

    def skew(self, row: int) -> int:
        return self.skew_spins[row].value()

    def row_state(self, row: int) -> dict:
        return {
            "channel": adc_channels[row],
            "enabled": self.is_enabled(row),
            "mask": self.is_mask(row),
            "edge": self.is_edge(row),
            "level": self.level(row),
            "hyst": self.hyst(row),
            "delay": self.delay(row),
            "h_off": self.h_off(row),
            "skew": self.skew(row),
            "events": self.events_labels[row].text(),
        }

    def all_states(self) -> list[dict]:
        return [self.row_state(r) for r in range(len(adc_channels))]

from PySide6.QtCore import Qt, Signal, QSettings
from PySide6.QtWidgets import (
    QTableWidget, QCheckBox, QSpinBox, QLabel, QWidget,
    QHBoxLayout, QVBoxLayout, QPushButton, QHeaderView,
)


class CaptureChannelTable(QWidget):
    enable_changed = Signal(int, str, bool)
    pre_trigger_changed = Signal(int, str, int)
    post_trigger_changed = Signal(int, str, int)

    SETTINGS_GROUP = "capture_channels"

    # uint range for the trigger spinboxes — adjust to taste
    PRE_MIN, PRE_MAX = 0, 10000
    POST_MIN, POST_MAX = 0, 10000

    def __init__(self, parent=None):
        super().__init__(parent)

        self._loading = False   # suppress auto-save during load/reset

        # ---- table ----
        self.table = QTableWidget(len(adc_channels), 3)
        self.table.setHorizontalHeaderLabels(
            ["Enable", "Pre-trigger", "Post-trigger"]
        )
        self.table.verticalHeader().setVisible(True)
        self.table.setVerticalHeaderLabels(adc_channels)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        total_height = (self.table.verticalHeader().length() + self.table.horizontalHeader().height() + (self.table.frameWidth() * 2))
        self.table.setFixedHeight(total_height)

        # ---- per-row widgets ----
        self.enable_boxes: list[QCheckBox] = []
        self.pre_spins: list[QSpinBox] = []
        self.post_spins: list[QSpinBox] = []

        for row, name in enumerate(adc_channels):
            enable = QCheckBox()
            enable.toggled.connect(
                lambda checked, r=row, n=name: self._on_enable(r, n, checked)
            )
            self.enable_boxes.append(enable)
            self.table.setCellWidget(row, 0, self._centered(enable))

            pre = QSpinBox()
            pre.setRange(self.PRE_MIN, self.PRE_MAX)
            pre.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            pre.valueChanged.connect(
                lambda value, r=row, n=name: self._on_pre(r, n, value)
            )
            self.pre_spins.append(pre)
            self.table.setCellWidget(row, 1, pre)

            post = QSpinBox()
            post.setRange(self.POST_MIN, self.POST_MAX)
            post.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            post.valueChanged.connect(
                lambda value, r=row, n=name: self._on_post(r, n, value)
            )
            self.post_spins.append(post)
            self.table.setCellWidget(row, 2, post)

        # ---- reset button ----
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.reset_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        layout.addLayout(button_row)

        # load the saved settings
        self.load_state()

    # ---------- change handlers (auto-save) ----------

    def _on_enable(self, row: int, name: str, checked: bool) -> None:
        self._save_channel(name)
        self.enable_changed.emit(row, name, checked)

    def _on_pre(self, row: int, name: str, value: int) -> None:
        self._save_channel(name)
        self.pre_trigger_changed.emit(row, name, value)

    def _on_post(self, row: int, name: str, value: int) -> None:
        self._save_channel(name)
        self.post_trigger_changed.emit(row, name, value)

    # ---------- helpers ----------

    @staticmethod
    def _centered(widget: QWidget) -> QWidget:
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.setAlignment(Qt.AlignCenter)
        h.addWidget(widget)
        return container

    def _row_for(self, name: str) -> int:
        return adc_channels.index(name)

    # ---------- persistence ----------

    def _save_channel(self, name: str) -> None:
        if self._loading:
            return
        row = self._row_for(name)
        s = q_settings
        s.beginGroup(self.SETTINGS_GROUP)
        s.beginGroup(name)
        s.setValue("enabled", self.enable_boxes[row].isChecked())
        s.setValue("pre_trigger", self.pre_spins[row].value())
        s.setValue("post_trigger", self.post_spins[row].value())
        s.endGroup()
        s.endGroup()

    def save_state(self) -> None:
        s = q_settings
        s.beginGroup(self.SETTINGS_GROUP)
        for row, name in enumerate(adc_channels):
            s.beginGroup(name)
            s.setValue("enabled", self.enable_boxes[row].isChecked())
            s.setValue("pre_trigger", self.pre_spins[row].value())
            s.setValue("post_trigger", self.post_spins[row].value())
            s.endGroup()
        s.endGroup()
        s.sync()

    def load_state(self) -> None:
        for row, name in enumerate(adc_channels):
            enabled = settings.capture_enabled_retrieved[name]
            pre = settings.capture_pre_trigger_retrieved[name]
            post = settings.capture_post_trigger_retrieved[name]
            self._set_row_silently(row, enabled, pre, post)

    def _set_row_silently(
        self, row: int, enabled: bool, pre: int, post: int
    ) -> None:
        widgets = (
            self.enable_boxes[row],
            self.pre_spins[row],
            self.post_spins[row],
        )
        for w in widgets:
            w.blockSignals(True)
        self.enable_boxes[row].setChecked(enabled)
        self.pre_spins[row].setValue(pre)
        self.post_spins[row].setValue(post)
        for w in widgets:
            w.blockSignals(False)

    # ---------- reset ----------

    def reset(self) -> None:
        self._loading = True
        try:
            for row in range(len(adc_channels)):
                self._set_row_silently(
                    row,
                    settings.capture_enabled_default,
                    settings.capture_pre_trigger_default,
                    settings.capture_post_trigger_default,
                )
        finally:
            self._loading = False

        self.save_state()
        for row, name in enumerate(adc_channels):
            self.enable_changed.emit(row, name, settings.capture_enabled_default)
            self.pre_trigger_changed.emit(row, name, settings.capture_pre_trigger_default)
            self.post_trigger_changed.emit(row, name, settings.capture_post_trigger_default)

    # ---------- accessors ----------

    def is_enabled(self, row: int) -> bool:
        return self.enable_boxes[row].isChecked()

    def pre_trigger(self, row: int) -> int:
        return self.pre_spins[row].value()

    def post_trigger(self, row: int) -> int:
        return self.post_spins[row].value()

    def row_state(self, row: int) -> dict:
        return {
            "channel": adc_channels[row],
            "enabled": self.is_enabled(row),
            "pre_trigger": self.pre_trigger(row),
            "post_trigger": self.post_trigger(row),
        }

    def all_states(self) -> list[dict]:
        return [self.row_state(r) for r in range(len(adc_channels))]

class LabeledSpinBox(QWidget):
    def __init__(self, text, min=0, max=100, default=1, step=1, parent=None, label_right=False, double_spin=False):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # Removes default padding

        self.label = QLabel(text, self)
        if double_spin:
            self.spinbox = QDoubleSpinBox(self)
        else:
            self.spinbox = QSpinBox(self)

        self.spinbox.setRange(min, max)
        self.spinbox.setValue(default)
        self.spinbox.setSingleStep(step)

        if not label_right:
            layout.addWidget(self.label)
            layout.addWidget(self.spinbox)
        else:
            self.spinbox.setAlignment(Qt.AlignRight)  # Right justify
            layout.addWidget(self.spinbox)
            layout.addWidget(self.label)


class HLine(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)

def help_text(layout, text):
    label = QLabel(text)
    label.setStyleSheet('font-size: 12px; padding: 10px 30px 10px;')
    layout.addWidget(label)

class PluginWidget(QWidget):
    """
    Required arguments:
        bus: the signals to communicate with the rest of the honeychrome app
        controller: the honeychrome controller including all ephemeral data and the experiment model
    """

    getInstrumentState = Signal()
    setInstrumentState = Signal(dict)

    def __init__(self, bus=None, controller=None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.controller = controller

        # --- Create widget, scroll area and layouts to hold the plugin content ---

        # the content widget goes in a scroll widget, which goes in the PluginWidget
        content_widget = QWidget()
        main_layout = QVBoxLayout(content_widget)

        # make this widget scrollable and resizeable
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(content_widget)

        overall_layout = QVBoxLayout(self)
        overall_layout.addWidget(scroll)

        # --- Add GUI elements ---

        # Create tab widget
        toolbox = QTabWidget()
        toolbox.setTabPosition(QTabWidget.TabPosition.West)  # Tabs on left


        # Create tabs
        # connection: either "find and connect" button or connected text, version number and datetime stamp
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.update_connection_status_btn = QPushButton("Update Connection Status")
        self.update_connection_status_btn.clicked.connect(self.update_connection_status)
        layout.addWidget(self.update_connection_status_btn)
        self.connection_status_not_connected = QLabel('<span style="font-weight:bold; color:red">Cytkit not connected</span>')
        self.connection_status_not_connected.setTextFormat(Qt.RichText)
        layout.addWidget(self.connection_status_not_connected)
        self.connection_status_connected = QLabel('<span style="font-weight:bold; color:green">Cytkit connected</span>')
        self.connection_status_connected.setTextFormat(Qt.RichText)
        self.connection_status_connected.setVisible(False)
        layout.addWidget(self.connection_status_connected)
        self.version = QLabel()
        layout.addWidget(self.version)
        self.datetime = QLabel()
        layout.addWidget(self.datetime)
        self.initialised = QCheckBox("Initialised")
        self.initialised.setEnabled(False)
        layout.addWidget(self.initialised)

        self.version = QLabel()

        title = QLabel('Automation Settings')
        title.setStyleSheet(heading_style)
        layout.addWidget(title)

        self.pressure_set_point_spin = LabeledSpinBox(min=-40, max=0, step=1, default=settings.pressure_set_point_retrieved, text='Sheath pressure (vacuum) set point [Pa]')
        self.pressure_set_point_spin.spinbox.valueChanged.connect(self.set_pressure_set_point)
        layout.addWidget(self.pressure_set_point_spin)

        self.temperature_set_point_spin = LabeledSpinBox(min=0, max=50, step=1, default=settings.temperature_set_point_retrieved, text='Internal temperature set point [C]')
        self.temperature_set_point_spin.spinbox.valueChanged.connect(self.set_temperature_set_point)
        layout.addWidget(self.temperature_set_point_spin)

        self.sample_pump_priming_speed = LabeledSpinBox(min=0, max=65_535, step=100, default=settings.sample_pump_priming_speed_retrieved, text='Sample pump priming speed [steps/s]')
        self.sample_pump_priming_speed.spinbox.valueChanged.connect(lambda value: self.set_pump_controls({'sample_pump_priming_speed': value}))
        layout.addWidget(self.sample_pump_priming_speed)

        self.sample_pump_priming_time = LabeledSpinBox(min=0, max=60, step=1, default=settings.sample_pump_priming_time_retrieved, text='Sample pump priming time [s]', double_spin=True)
        self.sample_pump_priming_time.spinbox.valueChanged.connect(lambda value: self.set_pump_controls({'sample_pump_priming_time': value}))
        layout.addWidget(self.sample_pump_priming_time)

        self.sample_pump_unpriming_speed = LabeledSpinBox(min=0, max=65_535, step=100, default=settings.sample_pump_unpriming_speed_retrieved, text='Sample pump unpriming speed [steps/s]')
        self.sample_pump_unpriming_speed.spinbox.valueChanged.connect(lambda value: self.set_pump_controls({'sample_pump_unpriming_speed': value}))
        layout.addWidget(self.sample_pump_unpriming_speed)

        self.sample_pump_unpriming_time = LabeledSpinBox(min=0, max=60, step=1, default=settings.sample_pump_unpriming_time_retrieved, text='Sample pump unpriming time [s]', double_spin=True)
        self.sample_pump_unpriming_time.spinbox.valueChanged.connect(lambda value: self.set_pump_controls({'sample_pump_unpriming_time': value}))
        layout.addWidget(self.sample_pump_unpriming_time)

        self.sample_pump_acquisition_rate = LabeledSpinBox(min=0, max=65_535, step=100, default=settings.sample_pump_acquisition_rate_retrieved, text='Sample pump acquisition rate [uL/min]')
        self.sample_pump_acquisition_rate.spinbox.valueChanged.connect(lambda value: self.set_pump_controls({'sample_pump_acquisition_rate': value}))
        layout.addWidget(self.sample_pump_acquisition_rate)

        self.sample_pump_settle_time = LabeledSpinBox(min=0, max=60, step=0.1, default=settings.sample_pump_settle_time_retrieved, text='Sample pump settle time [s]', double_spin=True)
        self.sample_pump_settle_time.spinbox.valueChanged.connect(lambda value: self.set_pump_controls({'sample_pump_settle_time': value}))
        layout.addWidget(self.sample_pump_settle_time)

        self.sample_pump_flush_speed = LabeledSpinBox(min=0, max=65_535, step=100, default=settings.sample_pump_flush_speed_retrieved, text='Sample pump flush speed [steps/s]')
        self.sample_pump_flush_speed.spinbox.valueChanged.connect(lambda value: self.set_pump_controls({'sample_pump_flush_speed': value}))
        layout.addWidget(self.sample_pump_flush_speed)

        self.sample_pump_flush_time = LabeledSpinBox(min=0, max=600, step=1, default=settings.sample_pump_flush_time_retrieved, text='Sample pump flush time [s]', double_spin=True)
        self.sample_pump_flush_time.spinbox.valueChanged.connect(lambda value: self.set_pump_controls({'sample_pump_flush_time': value}))
        layout.addWidget(self.sample_pump_flush_time)

        self.sample_pump_backflush_speed = LabeledSpinBox(min=0, max=65_535, step=100, default=settings.sample_pump_backflush_speed_retrieved, text='Sample pump backflush speed [steps/s]')
        self.sample_pump_backflush_speed.spinbox.valueChanged.connect(lambda value: self.set_pump_controls({'sample_pump_backflush_speed': value}))
        layout.addWidget(self.sample_pump_backflush_speed)

        self.sample_pump_backflush_time = LabeledSpinBox(min=0, max=600, step=1, default=settings.sample_pump_backflush_time_retrieved, text='Sample pump backflush time [s]', double_spin=True)
        self.sample_pump_backflush_time.spinbox.valueChanged.connect(lambda value: self.set_pump_controls({'sample_pump_backflush_time': value}))
        layout.addWidget(self.sample_pump_backflush_time)


        help_text(layout, '🛈 Note that if Cytkit is initialised, automation will override the manual settings below')
        layout.addStretch()
        toolbox.addTab(tab, "Connection")

        # Light tab: laser and LED calibration, interlock status, disable interlocks with warning
        tab = QWidget()
        layout = QVBoxLayout(tab)
        title = QLabel('Laser')
        title.setStyleSheet(heading_style)
        layout.addWidget(title)
        self.laser_cb = QCheckBox("Laser enable")
        self.laser_cb.toggled.connect(lambda checked: self.set_laser_state({'laser_enable': checked}))
        layout.addWidget(self.laser_cb)

        self.interlock_status_label = QLabel("Interlock status: ⚠️ Disconnected")
        layout.addWidget(self.interlock_status_label)
        frame = QFrame()
        frame.setObjectName("warningFrame")  # Set a unique name
        frame.setStyleSheet('''
            QFrame#warningFrame {        
                border: 3px solid #ff4444;
                border-radius: 8px;
            }''')
        frame_layout = QVBoxLayout(frame)
        pixmap = icon('alert-triangle').pixmap(QSize(32, 32))
        icon_label = QLabel()
        icon_label.setPixmap(pixmap)
        frame_layout.addWidget(icon_label)
        self.interlock_enable_cb = QCheckBox("Interlock Enabled")
        self.interlock_enable_cb.toggled.connect(lambda checked: self.set_laser_state({'interlock_enabled': checked}))
        frame_layout.addWidget(self.interlock_enable_cb)
        frame_layout.addWidget(QLabel('Warning: if interlocks are disabled, laser can be on when the instrument cover is removed, thus exposing the beam. \nIt is recommended to follow laser safety training and carry out a risk assessment.'))
        self.interlock_force_cb = QCheckBox("Force interlock")
        self.interlock_force_cb.clicked.connect(lambda checked: self.set_laser_state({'interlock_forced': checked}))
        frame_layout.addWidget(self.interlock_force_cb)
        frame_layout.addWidget(QLabel('Force interlock: when checked, simulates opening the interlock switch to turn off the laser.'))
        layout.addWidget(frame)

        # LED calibration
        title = QLabel('LED flash calibration')
        title.setStyleSheet(heading_style)
        layout.addWidget(title)
        self.led_flash = QCheckBox("Enable LED flash")
        layout.addWidget(self.led_flash)
        help_text(layout, '🛈 The LED flasher is a standard signal used for testing fluorescence and side scatter sensitivity')

        layout.addStretch()
        toolbox.addTab(tab, "Light")




        # fluidics tab:
        # sample pump cb: enable, reverse, ramp,
        # sample pump spinbox: speed, rampSpC, rampCpC
        tab = QWidget()
        layout = QVBoxLayout(tab)
        title = QLabel('Sample Pump')
        title.setStyleSheet(heading_style)
        layout.addWidget(title)
        self.sample_pump_enable_cb = QCheckBox("Sample Pump Enable")
        self.sample_pump_enable_cb.toggled.connect(lambda checked: self.set_instrument_state({'sample_pump_state': {'enable': checked}}))
        layout.addWidget(self.sample_pump_enable_cb)
        self.sample_pump_reverse_cb = QCheckBox("Sample Pump Reverse")
        self.sample_pump_reverse_cb.toggled.connect(lambda checked: self.set_instrument_state({'sample_pump_state': {'reverse': checked}}))
        layout.addWidget(self.sample_pump_reverse_cb)
        self.sample_pump_ramp_cb = QCheckBox("Sample Pump Ramp")
        self.sample_pump_ramp_cb.toggled.connect(lambda checked: self.set_instrument_state({'sample_pump_state': {'ramp': checked}}))
        layout.addWidget(self.sample_pump_ramp_cb)
        # frequency of pump steps 0.1 Hz, i.e. 10_000 for 1 kHz - fpga can do range(65_535)
        self.sample_pump_speed_spinbox = LabeledSpinBox('Sample Pump Speed', 0, 65_535, 0, 1000)
        self.sample_pump_speed_spinbox.spinbox.valueChanged.connect(lambda value: self.set_instrument_state({'sample_pump_state': {'speed': value}}))
        layout.addWidget(self.sample_pump_speed_spinbox)
        help_text(layout, '🛈 Sample pump speed is the frequency of pump steps in units of 0.1 Hz')
        # steps per cycle - speed increments per cycle
        self.sample_pump_rampSpC_spinbox = LabeledSpinBox('Sample Pump Ramp SpC', 0, 100, settings.sample_pump_steps_per_cycle, 1)
        self.sample_pump_rampSpC_spinbox.spinbox.valueChanged.connect(lambda value: self.set_instrument_state({'sample_pump_state': {'steps_per_cycle': value}}))
        layout.addWidget(self.sample_pump_rampSpC_spinbox)
        help_text(layout, '🛈 Sample pump ramp SpC (speed increments per cycle) is the ramp step to make in units of 0.1 Hz when changing the pump speed')
        # clocks per cycle - how many clock cycles before increment ramp step
        # note 100 MHz FPGA clock
        self.sample_pump_rampCpC_spinbox = LabeledSpinBox('Sample Pump Ramp CpC', 0, 2_000_000_000, settings.sample_pump_clocks_per_cycle, 100)
        self.sample_pump_rampCpC_spinbox.spinbox.valueChanged.connect(lambda value: self.set_instrument_state({'sample_pump_state': {'clocks_per_cycle': value}}))
        layout.addWidget(self.sample_pump_rampCpC_spinbox)
        help_text(layout, '🛈 Sample pump ramp CpC (cycles per clock) is the number of FPGA clock cycles to count (at 2 GHz) before changing the sample pump speed by one step')


        # sheath pump cb: enable
        # sheath pump spinbox: duty, freq
        # layout.addStretch()
        # layout.addWidget(HLine())
        title = QLabel('Sheath Pump')
        title.setStyleSheet(heading_style)
        layout.addWidget(title)
        self.sheath_pump_enable_cb = QCheckBox("Sheath Pump Enable")
        self.sheath_pump_enable_cb.toggled.connect(lambda checked: self.set_instrument_state({'sheath_pump_state': {'enable': checked}}))
        layout.addWidget(self.sheath_pump_enable_cb)
        self.sheath_pump_duty_spinbox = LabeledSpinBox('Sheath Pump Duty', 0, 255, 127, 8)
        self.sheath_pump_duty_spinbox.spinbox.valueChanged.connect(lambda value: self.set_instrument_state({'sheath_pump_state': {'duty': value}}))
        layout.addWidget(self.sheath_pump_duty_spinbox)
        help_text(layout, '🛈 Sheath pump duty is a number in the range 0--255, where 0 is off, and 255 is on 100% of the time')
        self.sheath_pump_freq_spinbox = LabeledSpinBox('Sheath Pump Frequency', 0, 2_000_000_000, 1000, 100)
        self.sheath_pump_freq_spinbox.spinbox.valueChanged.connect(lambda value: self.set_instrument_state({'sheath_pump_state': {'freq': value}}))
        layout.addWidget(self.sheath_pump_freq_spinbox)
        help_text(layout, '🛈 Sheath pump frequency is the frequency of the duty cycle in Hz')

        # pressure measure, zero, value
        title = QLabel('Pressure Sensor')
        title.setStyleSheet(heading_style)
        layout.addWidget(title)
        self.pressure_value = QLabel('0 Pa')
        layout.addWidget(self.pressure_value)
        self.pressure_measure_btn = QPushButton('Measure Pressure')
        self.pressure_measure_btn.clicked.connect(lambda: self.get_instrument_state(['pressure']))
        layout.addWidget(self.pressure_measure_btn)
        self.pressure_zero_btn = QPushButton('Zero Pressure')
        self.pressure_zero_btn.clicked.connect(lambda: self.get_instrument_state(['zero_pressure']))
        layout.addWidget(self.pressure_zero_btn)

        layout.addStretch()
        toolbox.addTab(tab, "Fluidics")

        # DACs tab:
        # dac bias, dac ref x chanels
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel("DACs for each channel (DAC units 0..255)"))
        self.dac_table = QTableWidget(number_of_dacs_pairs, 2)
        self.dac_table.setHorizontalHeaderLabels(["Bias", "Ref"])
        for row in range(number_of_dacs_pairs):
            for col in range(2):
                dac_index = row + col * number_of_dacs_pairs
                dac_type = 'bias' if col == 0 else 'ref'
                spin = LabeledSpinBox(min=0, max=255, text=dac_dictionary[dac_index]['channel_name'], label_right=True)
                spin.spinbox.valueChanged.connect(lambda value: self.set_instrument_state({'dacs': {dac_type: {row: value}}}))
                self.dac_table.setCellWidget(row, col, spin)
                # value = dac_table.cellWidget(0, 1).value()
                # self.dac_table.cellWidget(0, 1).spinbox.setValue(value)

        self.dac_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.dac_table.verticalHeader().setVisible(False)  # Hide row numbers
        self.dac_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        total_height = (self.dac_table.verticalHeader().length() + self.dac_table.horizontalHeader().height() + (self.dac_table.frameWidth() * 2))
        self.dac_table.setFixedHeight(total_height)
        # dac_table.setVerticalHeaderLabels([dac_dictionary[n]['channel_name'] for n in range(number_of_dacs_pairs)])
        layout.addWidget(self.dac_table)

        layout.addStretch()
        toolbox.addTab(tab, "DACs")

        # ADCs tab:
        # ADC table by channel: ADC enable, current sample, invert, offset
        tab = QWidget()
        layout = QVBoxLayout(tab)
        title = QLabel('ADCs')
        title.setStyleSheet(heading_style)
        layout.addWidget(title)
        self.adc_table = AdcChannelTable()
        self.adc_table.enable_changed.connect(lambda channel_id, channel_name, value: self.set_instrument_state({'adcs': {'enable': {channel_id: value}}}))
        self.adc_table.invert_changed.connect(lambda channel_id, channel_name, value: self.set_instrument_state({'adcs': {'invert': {channel_id: value}}}))
        self.adc_table.offset_changed.connect(lambda channel_id, channel_name, value: self.set_instrument_state({'adcs': {'offset': {channel_id: value}}}))

        layout.addWidget(self.adc_table)
        layout.addStretch()
        toolbox.addTab(tab, "ADCs")


        # Triggers tab:
        # Triggers table by channel: enable, mask, edge, level, hyst, delay, h.off, skew, events
        # merge events number
        # clear all
        # force
        tab = QWidget()
        layout = QVBoxLayout(tab)
        title = QLabel('Triggers')
        title.setStyleSheet(heading_style)
        title.setStyleSheet(heading_style)
        layout.addWidget(title)
        self.trigger_table = TriggerChannelTable()
        self.trigger_table.enable_changed.connect(lambda channel_id, channel_name, value: self.set_instrument_state({'trigger': {'enable': {channel_id: value}}}))
        self.trigger_table.mask_changed.connect(lambda channel_id, channel_name, value: self.set_instrument_state({'trigger': {'mask': {channel_id: value}}}))
        self.trigger_table.edge_changed.connect(lambda channel_id, channel_name, value: self.set_instrument_state({'trigger': {'edge': {channel_id: value}}}))
        self.trigger_table.level_changed.connect(lambda channel_id, channel_name, value: self.set_instrument_state({'trigger': {'level': {channel_id: value}}}))
        self.trigger_table.hyst_changed.connect(lambda channel_id, channel_name, value: self.set_instrument_state({'trigger': {'hyst': {channel_id: value}}}))
        self.trigger_table.delay_changed.connect(lambda channel_id, channel_name, value: self.set_instrument_state({'trigger': {'delay': {channel_id: value}}}))
        self.trigger_table.h_off_changed.connect(lambda channel_id, channel_name, value: self.set_instrument_state({'trigger': {'h_off': {channel_id: value}}}))
        self.trigger_table.skew_changed.connect(lambda channel_id, channel_name, value: self.set_instrument_state({'trigger': {'skew': {channel_id: value}}}))
        layout.addWidget(self.trigger_table)

        form = QFormLayout()
        self.merged_count = QLabel('—')
        form.addRow('Merged:', self.merged_count)
        layout.addLayout(form)
        self.force_btn = QPushButton('Force trigger')
        self.force_btn.clicked.connect(lambda: self.set_instrument_state({'trigger': 'force'}))
        layout.addWidget(self.force_btn)
        self.clear_btn = QPushButton('Clear counts')
        self.clear_btn.clicked.connect(lambda: self.set_instrument_state({'trigger': 'clear'}))
        layout.addWidget(self.clear_btn)

        layout.addStretch()
        toolbox.addTab(tab, "Triggers")

        # Capture tab:
        # Capture table by channel: enable, pre-trigger, post-trigger
        # aggregator: enable cb, clear fifo btn, fifo level, fifo status
        # capture stream collector: enable, auto_reset, decode error level, auto reset error level
        tab = QWidget()
        layout = QVBoxLayout(tab)
        title = QLabel('Capture')
        title.setStyleSheet(heading_style)
        layout.addWidget(title)
        self.capture_table = CaptureChannelTable()
        self.capture_table.enable_changed.connect(lambda channel_id, channel_name, value: self.set_instrument_state({'capture': {'enable': {channel_id: value}}}))
        self.capture_table.pre_trigger_changed.connect(lambda channel_id, channel_name, value: self.set_instrument_state({'capture': {'pre_trigger': {channel_id: value}}}))
        self.capture_table.post_trigger_changed.connect(lambda channel_id, channel_name, value: self.set_instrument_state({'capture': {'post_trigger': {channel_id: value}}}))

        layout.addWidget(self.capture_table)
        form = QFormLayout()
        self.fifo_level = QLabel('—')
        form.addRow('FIFO level:', self.fifo_level)
        self.fifo_status = QLabel('—')
        form.addRow('FIFO status:', self.fifo_status)
        self.decode_errors = QLabel('—')
        form.addRow('Decode errors:', self.decode_errors)
        self.auto_resets = QLabel('—')
        form.addRow('Auto resets:', self.auto_resets)
        layout.addLayout(form)
        self.clear_fifo_btn = QPushButton('Clear FIFO')
        self.clear_fifo_btn.clicked.connect(lambda: self.set_instrument_state({'capture': 'clear_fifo'}))
        layout.addWidget(self.clear_fifo_btn)

        layout.addStretch()
        toolbox.addTab(tab, "Capture")


        # Monitoring tab:
        # label for each reading V, I
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Temp monitors
        title = QLabel('Temperatures')
        title.setStyleSheet(heading_style)
        layout.addWidget(title)
        self.read_temperatures = QPushButton('Read Temperatures')
        self.read_temperatures.clicked.connect(lambda: self.get_instrument_state(['temperatures']))
        layout.addWidget(self.read_temperatures)
        form = QFormLayout()
        self.temp_p_sensor_label = QLabel('None')
        form.addRow('Temperature (at pressure sensor)', self.temp_p_sensor_label)
        layout.addLayout(form)

        # VI monitors
        title = QLabel('VI Monitors')
        title.setStyleSheet(heading_style)
        layout.addWidget(title)
        self.read_monitors = QPushButton('Read VI Monitors')
        self.read_monitors.clicked.connect(lambda: self.get_instrument_state(['vi_monitors']))
        layout.addWidget(self.read_monitors)

        self.vi_table = QTableWidget(len(monitor_dictionary), 2)
        self.vi_table.setHorizontalHeaderLabels(["V [V]", "I [mA]"])
        for row in range(len(monitor_dictionary)):
            for col in range(2):
                self.vi_table.setCellWidget(row, col, QLabel())
                # value = self.vi_table.cellWidget(0, 1).value()
                # self.vi_table.cellWidget(0, 1).spinbox.setValue(value)

        self.vi_table.setVerticalHeaderLabels([monitor_dictionary[row]['name'] for row in range(len(monitor_dictionary))])
        self.vi_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        total_height = (self.vi_table.verticalHeader().length() + self.vi_table.horizontalHeader().height() + (self.vi_table.frameWidth() * 2))
        self.vi_table.setFixedHeight(total_height)
        self.vi_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.vi_table)

        # fan control: enable cb, duty spin, freq spin, tacho label
        title = QLabel('Cooling Fan')
        title.setStyleSheet(heading_style)
        layout.addWidget(title)
        self.fan_enable_cb = QCheckBox("Fan Enable")
        self.fan_enable_cb.toggled.connect(lambda checked: self.set_instrument_state({'fan_state': {'enable': checked}}))
        layout.addWidget(self.fan_enable_cb)
        self.fan_duty_spinbox = LabeledSpinBox('Fan Duty', 0, 255, 127, 8)
        self.fan_duty_spinbox.spinbox.valueChanged.connect(lambda value: self.set_instrument_state({'fan_state': {'duty': value}}))
        layout.addWidget(self.fan_duty_spinbox)
        help_text(layout, '🛈 Fan duty is a number in the range 0--255, where 0 is off, and 255 is on 100% of the time')
        self.fan_freq_spinbox = LabeledSpinBox('Fan Frequency', 0, 2_000_000_000, 1000, 100)
        self.fan_freq_spinbox.spinbox.valueChanged.connect(lambda value: self.set_instrument_state({'fan_state': {'freq': value}}))
        layout.addWidget(self.fan_freq_spinbox)
        help_text(layout, '🛈 fan frequency is the frequency of the duty cycle in Hz')
        self.fan_tacho_value = QLabel('0 rpm')
        layout.addWidget(self.fan_tacho_value)
        self.fan_tacho_btn = QPushButton('Measure Fan Speed')
        self.fan_tacho_btn.clicked.connect(lambda: self.get_instrument_state(['fan_state']))
        layout.addWidget(self.fan_tacho_btn)

        layout.addStretch()
        toolbox.addTab(tab, "Monitoring")

        # # Front panel display:
        # # file load dialog, upload button
        # tab = QWidget()
        # layout = QVBoxLayout(tab)
        # layout.addWidget(QLabel("Content for display_tab"))
        # layout.addStretch()
        # toolbox.addTab(tab, "Display")

        # Registers tab:
        # write: register field, data field
        # read: register field, data label
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.register_combo = QComboBox()
        self.register_combo.addItem('Select a register')
        self.register_combo.addItems(registers_map.keys())
        layout.addWidget(self.register_combo)
        self.register_value_lineedit = QLineEdit()
        validator = QIntValidator(0, 255, self.register_value_lineedit)
        self.register_value_lineedit.setValidator(validator)
        self.register_value_lineedit.setPlaceholderText("Value to set 0-255")
        layout.addWidget(self.register_value_lineedit)

        self.set_register_button = QPushButton('Set Register Value')
        self.set_register_button.clicked.connect(lambda: self.set_instrument_state({'register_setter':{'register': self.register_combo.currentText(), 'value': self.register_value_lineedit.text()}}))
        layout.addWidget(self.set_register_button)
        self.get_register_button = QPushButton('Get Register Value')
        self.get_register_button.clicked.connect(lambda: self.get_instrument_state({'register_getter':self.register_combo.currentText()}))
        layout.addWidget(self.get_register_button)
        self.register_value = QLabel('Value: None')
        layout.addWidget(self.register_value)
        layout.addStretch()
        toolbox.addTab(tab, "Registers")

        # Alignment camera tab:
        # usb webcam connection
        # exposure, gain, image
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.alignment_camera = AlignmentCameraWidget(parent=self)
        layout.addWidget(self.alignment_camera)
        layout.addStretch()
        toolbox.addTab(tab, "Alignment Camera")


        # Style the toolbox
        toolbox.setStyleSheet("""
            QToolBox::tab {
                text-decoration: none;
            }
            QToolBox::tab:selected {
                font-weight: bold;
            }
            QToolBox::tab:hover {
                text-decoration: underline;
            }
            QToolBox::tab:pressed {
                font-weight: bold;
            }
        """)
        # toolbox.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        main_layout.addWidget(toolbox)

        # Connect the signal to a slot
        toolbox.currentChanged.connect(self.on_tab_changed)
        self.toolbox = toolbox

        # update everything
        self.device_name = None
        # self.update_connection_status()

    def on_tab_changed(self, index):
        name = self.toolbox.tabText(index)
        match name:
            case "Connection":
                self.get_instrument_state(['version','datetime'])
                self.update_initialised()
            case "Light":
                self.get_instrument_state(['laser'])
            case "Fluidics":
                self.get_instrument_state(['pressure'])
            case "DACs":
                self.get_instrument_state(['dacs'])
            case "ADCs":
                self.get_instrument_state(['adcs'])
            case "Triggers":
                self.get_instrument_state(['trigger'])
            case "Capture":
                self.get_instrument_state(['capture'])
            case "Monitoring":
                self.get_instrument_state(['vi_monitors','temperatures','fan_tacho'])
            case "Registers":
                pass
            case "Alignment Camera":
                pass

    @Slot()
    def update_connection_status(self):
        if self.controller:
            self.controller.find_and_connect_instrument()
            self.device_name = self.controller.get_connected_device_name()
        if self.device_name == 'Cytkit':
            self.get_instrument_state([])
        else:
            self.connection_status_connected.setVisible(False)
            self.connection_status_not_connected.setVisible(True)

    @Slot(list)
    def get_instrument_state(self, parameters):
        if self.controller:
            self.controller.pipe_connection_instrument.send({'command': 'get_instrument_state', 'data': parameters})
            response = self.controller.pipe_connection_instrument.recv()
        else:
            response = {'message': {}}

        if 'check_connection' in response['message']:
            if response['message']['check_connection']:
                self.connection_status_connected.setVisible(True)
                self.connection_status_not_connected.setVisible(False)
            else:
                self.connection_status_connected.setVisible(False)
                self.connection_status_not_connected.setVisible(True)

        if 'read_id_data' in response['message']:
            self.version.setText(f'Firmware version: {response['message']['read_id_data']['version']}')
            self.datetime.setText(f'Firmware datestamp: {response['message']['read_id_data']['datetime']}')

        if 'laser' in response['message']:
            self.laser_cb.setChecked(response['message']['laser']['state'])
            self.interlock_enable_cb.setChecked(response['message']['laser']['interlock_mask'])
            if response['message']['laser']['interlock_state']:
                self.interlock_status_label.setText('Interlock status: 🔴 Open')
            else:
                self.interlock_status_label.setText('Interlock status: 🟢 Closed')
            self.interlock_force_cb.setChecked(response['message']['laser']['interlock_forced'])

        if 'pressure' in response['message']:
            self.pressure_value.setText(f'{response['message']['pressure']} Pa')

        if 'temperatures' in response['message']:
            if response['message']['temperatures']:
                self.temp_p_sensor_label.setText(f'{response['message']['temperatures']['temp_p_sensor']} C')

        if 'vi_monitors' in response['message']:
            if response['message']['vi_monitors']:
                for channel in monitor_dictionary:
                    for col, monitor_type in enumerate(['V', 'I']):
                        value = response['message']['vi_monitors']['V'][channel]
                        self.vi_table.cellWidget(channel, col).setText(f'{value:0.2f}')

        if 'fan_state' in response['message']:
            self.fan_enable_cb.setChecked(response['message']['fan_state']['enable'])
            self.fan_freq_spinbox.spinbox.setValue(response['message']['fan_state']['freq'])
            self.fan_duty_spinbox.spinbox.setValue(response['message']['fan_state']['duty'])
            self.fan_tacho_value.setText(f'{response['message']['fan_state']['tacho']} rpm')

        if 'sheath_pump_state' in response['message']:
            self.sheath_pump_enable_cb.setChecked(response['message']['sheath_pump_state']['enable'])
            self.sheath_pump_freq_spinbox.spinbox.setValue(response['message']['sheath_pump_state']['freq'])
            self.sheath_pump_duty_spinbox.spinbox.setValue(response['message']['sheath_pump_state']['duty'])

        if 'sample_pump_state' in response['message']:
            self.sample_pump_enable_cb.setChecked(response['message']['sample_pump_state']['enable'])
            self.sample_pump_reverse_cb.setChecked(response['message']['sample_pump_state']['reverse'])
            self.sample_pump_ramp_cb.setChecked(response['message']['sample_pump_state']['ramp'])
            self.sample_pump_speed_spinbox.spinbox.setValue(response['message']['sample_pump_state']['speed'])
            self.sample_pump_rampSpC_spinbox.spinbox.setValue(response['message']['sample_pump_state']['steps_per_cycle'])
            self.sample_pump_rampCpC_spinbox.spinbox.setValue(response['message']['sample_pump_state']['clocks_per_cycle'])

        if 'dacs' in response['message']:
            if response['message']['dacs']:
                if 'bias' in response['message']['dacs']:
                    for index in response['message']['dacs']['bias']:
                        value = response['message']['dacs']['bias'][index]
                        self.dac_table.cellWidget(index, 0).spinbox.setValue(value)
                if 'ref' in response['message']['dacs']:
                    for index in response['message']['dacs']['ref']:
                        value = response['message']['dacs']['bias'][index]
                        self.dac_table.cellWidget(index, 1).spinbox.setValue(value)

        if 'register_getter' in response['message']:
            value = response['message']['register_getter']
            self.register_value.setText(f'Value: {value}')

        if self.bus:
            self.bus.statusMessage.emit(f'{response['source']} {response['status']}: {response['message']}')

        logger.info(response)

    @Slot(dict)
    def set_laser_state(self, parameter_values):
        self.set_instrument_state(parameter_values)
        self.get_instrument_state(['laser'])

    @Slot(dict)
    def set_instrument_state(self, parameter_values):
        self.controller.pipe_connection_instrument.send({'command': 'set_instrument_state', 'data': parameter_values})
        response = self.controller.pipe_connection_instrument.recv()
        if self.bus:
            self.bus.statusMessage.emit(f'{response['source']} {response['status']}: {response['message']}')

        logger.info(response)

    @Slot(int)
    def set_pressure_set_point(self, set_point):
        q_settings.setValue("pressure_set_point", set_point)
        self.set_instrument_state({'pressure_set_point': set_point})

    @Slot(int)
    def set_temperature_set_point(self, set_point):
        self.set_instrument_state({'temperature_set_point': set_point})

    @Slot(dict)
    def set_pump_controls(self, dict_of_parameter_value):
        for parameter, value in dict_of_parameter_value.items():
            q_settings.setValue(parameter, value)
        self.set_instrument_state(dict_of_parameter_value)


    @Slot()
    def update_initialised(self):
        if self.controller:
            initialised = self.controller.is_instrument_initialised()
            self.initialised.setChecked(initialised)
            print(f'Initialisation status: {initialised}')


if __name__ == "__main__":
    import sys

    import multiprocessing as mp
    from multiprocessing import shared_memory, Lock
    import numpy as np
    import logging
    import logging.handlers

    configure_multiprocessing()

    '''
    define objects for communication between processes
    '''
    from honeychrome.settings import traces_cache_size, traces_cache_dtype
    from honeychrome.settings import max_events_in_cache, n_channels_per_event
    import honeychrome.settings as settings

    # Allocate shared memory block, plus head and tail indices
    traces_cache_shm = shared_memory.SharedMemory(create=True, size=np.zeros(traces_cache_size, dtype=traces_cache_dtype).nbytes)
    traces_cache_lock = Lock()
    index_head_traces_cache = mp.Value('i', 0)
    index_tail_traces_cache = mp.Value('i', 0)

    events_cache_shm = shared_memory.SharedMemory(create=True,
                                                  size=np.zeros((max_events_in_cache, n_channels_per_event),
                                                                dtype=np.int_).nbytes)
    events_cache_lock = Lock()
    index_head_events_cache = mp.Value('i', 0)
    index_tail_events_cache = mp.Value('i', 0)

    # oscilloscope traces
    oscilloscope_traces_queue = mp.Queue()
    # command pipes
    pipe_experiment_instrument_e, pipe_experiment_instrument_i = mp.Pipe()
    pipe_experiment_analyser_e, pipe_experiment_analyser_a = mp.Pipe()
    # logging queue
    logging_queue = mp.Queue()
    # Set up listener in the main process
    listener = logging.handlers.QueueListener(logging_queue, logging.StreamHandler(), respect_handler_level=True)
    listener.start()

    '''
    start instrument driver
    '''
    from honeychrome.instrument_communicator import Instrument

    instrument = Instrument(
        use_dummy_instrument=settings.use_dummy_instrument_retrieved,
        traces_cache_name=traces_cache_shm.name,
        traces_cache_lock=traces_cache_lock,
        index_head_traces_cache=index_head_traces_cache,
        index_tail_traces_cache=index_tail_traces_cache,
        pipe_connection=pipe_experiment_instrument_i,
        logging_queue=logging_queue
    )
    instrument.start()

    '''
    start trace analyser
    '''
    from honeychrome.trace_analyst import TraceAnalyser

    trace_analyser = TraceAnalyser(
        traces_cache_name=traces_cache_shm.name,
        traces_cache_lock=traces_cache_lock,
        index_head_traces_cache=index_head_traces_cache,
        index_tail_traces_cache=index_tail_traces_cache,
        events_cache_name=events_cache_shm.name,
        events_cache_lock=events_cache_lock,
        index_head_events_cache=index_head_events_cache,
        index_tail_events_cache=index_tail_events_cache,
        oscilloscope_traces_queue=oscilloscope_traces_queue,
        pipe_connection=pipe_experiment_analyser_a
    )
    trace_analyser.start()

    '''
    start controller
    '''
    from honeychrome.controller import Controller

    controller = Controller(
            events_cache_name=events_cache_shm.name,
            events_cache_lock=events_cache_lock,
            index_head_events_cache=index_head_events_cache,
            index_tail_events_cache=index_tail_events_cache,
            oscilloscope_traces_queue=oscilloscope_traces_queue,
            pipe_connection_instrument=pipe_experiment_instrument_e,
            pipe_connection_analyser=pipe_experiment_analyser_e)

    '''
    start application and view
    '''
    bus = EventBus()
    controller.bus = bus # connect signals coming from controller

    app = QApplication(sys.argv)
    window = PluginWidget(bus=bus, controller=controller)
    window.resize(1000, 1000)
    window.show()
    exit_code = app.exec()

    # end processes, free memory
    controller.quit_instrument_quit_analyser()
    trace_analyser.join()
    instrument.join()
    traces_cache_shm.close()
    events_cache_shm.close()
    traces_cache_shm.unlink()
    events_cache_shm.unlink()
    listener.stop()

    sys.exit(exit_code)

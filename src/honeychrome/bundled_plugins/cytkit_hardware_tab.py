"""
Cytkit State plugin (hardware monitor and settings)
"""
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QPushButton, QLabel, QTabWidget, QToolBox, QFormLayout, QComboBox, QCheckBox
from PySide6.QtCore import Qt, Slot, Signal

import logging
logger = logging.getLogger(__name__)

plugin_name = 'Cytkit Hardware'

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

        # --- Add some GUI elements to show functionality ---

        # Create tab widget
        toolbox = QToolBox()

        # Create tabs
        # connection: either "find and connect" button or connected text, version number and datetime stamp
        connection_tab = QWidget()

        # Light tab: laser and LED calibration, interlock status, disable interlocks with warning
        light_tab = QWidget()
        form = QFormLayout(connection_tab)
        form.setSpacing(10)
        self.laser_cb = QCheckBox("Laser control")
        form.addRow("Laser on / off", self.laser_cb)
        self.interlock_status = QLabel("Not connected")
        form.addRow("Interlock status", self.interlock_status)

        # fluidics tab:
        # sample pump enable, reverse, speed, rampSpC, rampCpC
        # sheath pump enable, duty, freq
        # pressure measure, zero, value, autoread
        fluidics_tab = QWidget()
        fluidics_tab_layout = QVBoxLayout()
        fluidics_tab_layout.addWidget(QLabel("Content for fluidics_tab"))
        fluidics_tab.setLayout(fluidics_tab_layout)

        # DACs tab:
        # dac bias, dac ref x chanels
        dacs_tab = QWidget()
        dacs_tab_layout = QVBoxLayout()
        dacs_tab_layout.addWidget(QLabel("Content for dacs_tab"))
        dacs_tab.setLayout(dacs_tab_layout)

        # ADCs tab:
        # checkboxes: enable x chanels, select all, select none
        # buttons: clear, capture
        # spinbox: capture samples
        # pg graph
        adcs_tab = QWidget()
        adcs_tab_layout = QVBoxLayout()
        adcs_tab_layout.addWidget(QLabel("Content for adcs_tab"))
        adcs_tab.setLayout(adcs_tab_layout)

        # VI Monitoring tab:
        # label for each reading
        monitoring_tab = QWidget()
        monitoring_tab_layout = QVBoxLayout()
        monitoring_tab_layout.addWidget(QLabel("Content for monitoring_tab"))
        monitoring_tab.setLayout(monitoring_tab_layout)

        # Temperatures tab:
        # fan control: enable cb, duty spin, freq spin, tacho label
        # label for each reading
        temperatures_tab = QWidget()
        temperatures_tab_layout = QVBoxLayout()
        temperatures_tab_layout.addWidget(QLabel("Content for temperatures_tab"))
        temperatures_tab.setLayout(temperatures_tab_layout)

        # Front panel display:
        # file load dialog, upload button
        display_tab = QWidget()
        display_tab_layout = QVBoxLayout()
        display_tab_layout.addWidget(QLabel("Content for display_tab"))
        display_tab.setLayout(display_tab_layout)

        # Registers tab:
        # write: register field, data field
        # read: register field, data label
        registers_tab = QWidget()
        registers_tab_layout = QVBoxLayout()
        registers_tab_layout.addWidget(QLabel("Content for registers_tab"))
        registers_tab.setLayout(registers_tab_layout)

        # Add tabs to toolbox
        toolbox.addItem(connection_tab, "Connection")
        toolbox.addItem(light_tab, "Light")
        toolbox.addItem(fluidics_tab, "Fluidics")
        toolbox.addItem(dacs_tab, "DACs")
        toolbox.addItem(adcs_tab, "ADCs")
        toolbox.addItem(monitoring_tab, "Monitoring")
        toolbox.addItem(temperatures_tab, "Temperatures")
        toolbox.addItem(display_tab, "Display")
        toolbox.addItem(registers_tab, "Registers")

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


    @Slot(list)
    def get_instrument_state(self, registers_to_read):
        self.controller.pipe_connection_instrument.send({'command': 'get_instrument_state', 'data': registers_to_read})
        response = self.controller.pipe_connection_instrument.recv()

        logger.info(response)

    @Slot(dict)
    def set_instrument_state(self, registers_and_values_to_write):
        self.controller.pipe_connection_instrument.send({'command': 'set_instrument_state', 'data': registers_and_values_to_write})
        response = self.controller.pipe_connection_instrument.recv()

        logger.info(response)


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication, QMainWindow
    app = QApplication([])
    window = PluginWidget()
    window.resize(400, 300)
    window.show()
    app.exec()
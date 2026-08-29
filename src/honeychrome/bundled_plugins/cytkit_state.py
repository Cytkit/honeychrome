"""
Cytkit State plugin (hardware monitor and settings)
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QPushButton, QLabel
from PySide6.QtCore import Qt, Slot, Signal

import logging
logger = logging.getLogger(__name__)

plugin_name = 'Cytkit State Plugin'

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
        self.label = QLabel('')
        self.label.setTextFormat(Qt.RichText)
        self.label.setWordWrap(True)

        self.refresh_button = QPushButton('Refresh')
        self.refresh_button.setToolTip('Runs a method "refresh"')
        self.refresh_button.clicked.connect(self.refresh)

        self.popup_button = QPushButton('Send signal to make popup')
        self.popup_button.setToolTip('Uses the signal bus to communicate with a function elsewhere in Honeychrome')
        self.popup_button.clicked.connect(lambda: self.bus.popupMessage.emit('Hello World!'))

        main_layout.addWidget(self.popup_button)
        main_layout.addWidget(self.refresh_button)
        main_layout.addWidget(self.label)

        self.refresh()

    def refresh(self):
        # put some data from the controller into the label
        import json

        self.label.setText(f'''
        <h1>Hello world!</h1>

        <p>Cytometry data can be accessed from the controller object (and the experiment object from controller.experiment):</p>

        <ul>
            <li> controller.experiment_dir: <pre>{self.controller.experiment_dir}</pre> </li>
            <li> controller.current_sample_path: <pre>{self.controller.current_sample_path}</pre> </li>
            <li> controller.expreriment.samples: <pre>{json.dumps(self.controller.experiment.samples, indent=2)}</pre> </li>
        </ul>
        ''')


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

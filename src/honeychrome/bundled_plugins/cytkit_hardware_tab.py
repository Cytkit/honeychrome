"""
Cytkit State plugin (hardware monitor and settings)
"""
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QPushButton, QLabel, QTabWidget, QToolBox, QFormLayout, QComboBox, QCheckBox, QSpinBox, QHBoxLayout, QFrame
from PySide6.QtCore import Qt, Slot, Signal, QSize

import logging

from honeychrome.controller import Controller
from honeychrome.instrument_driver_components.cykit_components.cytkit_configuration import monitor_dictionary
from honeychrome.settings import heading_style
from honeychrome.view_components.icon_loader import icon

logger = logging.getLogger(__name__)

plugin_name = 'Cytkit Hardware'


class LabeledSpinBox(QWidget):
    def __init__(self, text, min=0, max=100, default=1, step=1, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # Removes default padding

        self.label = QLabel(text, self)
        self.spinbox = QSpinBox(self)
        self.spinbox.setRange(min, max)
        self.spinbox.setValue(default)
        self.spinbox.setSingleStep(step)

        layout.addWidget(self.label)
        layout.addWidget(self.spinbox)

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
        toolbox = QToolBox()

        # Create tabs
        # connection: either "find and connect" button or connected text, version number and datetime stamp
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.update_connection_status_btn = QPushButton("Update Connection Status")
        self.update_connection_status_btn.clicked.connect(lambda: self.get_instrument_state(['check_id','version','datetime']))
        layout.addWidget(self.update_connection_status_btn)
        self.connection_status_not_connected = QLabel('<span style="font-weight:bold; color:red">Not connected</span>')
        self.connection_status_not_connected.setTextFormat(Qt.RichText)
        layout.addWidget(self.connection_status_not_connected)
        self.connection_status_connected = QLabel('<span style="font-weight:bold; color:green">Connected</span>')
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

        help_text(layout, '🛈 Note that if Cytkit is initialised, automation will override the settings below')
        layout.addStretch()
        toolbox.addItem(tab, "Connection")

        # Light tab: laser and LED calibration, interlock status, disable interlocks with warning
        tab = QWidget()
        layout = QVBoxLayout(tab)
        title = QLabel('Laser')
        title.setStyleSheet(heading_style)
        layout.addWidget(title)
        self.laser_cb = QCheckBox("Laser enable")
        self.laser_cb.toggled.connect(lambda checked: self.set_instrument_state({'laser_enable': checked}))
        layout.addWidget(self.laser_cb)
        self.interlock_status = QCheckBox("Interlocks closed")
        layout.addWidget(self.interlock_status)
        self.interlock_status.setEnabled(False)
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
        self.interlock_disable = QCheckBox("Disable Interlocks")
        frame_layout.addWidget(self.interlock_disable)
        frame_layout.addWidget(QLabel('Warning: if interlocks are disabled, laser can be on when the instrument cover is removed, thus exposing the beam. \nIt is recommended to follow laser safety training and carry out a risk assessment.'))
        layout.addWidget(frame)

        # LED calibration
        title = QLabel('LED flash calibration')
        title.setStyleSheet(heading_style)
        layout.addWidget(title)
        self.led_flash = QCheckBox("Enable LED flash")
        layout.addWidget(self.led_flash)
        help_text(layout, '🛈 The LED flasher is a standard signal used for testing fluorescence and side scatter sensitivity')

        layout.addStretch()
        toolbox.addItem(tab, "Light")




        # fluidics tab:
        # sample pump cb: enable, reverse, ramp,
        # sample pump spinbox: speed, rampSpC, rampCpC
        tab = QWidget()
        layout = QVBoxLayout(tab)
        title = QLabel('Sample Pump')
        title.setStyleSheet(heading_style)
        layout.addWidget(title)
        self.sample_pump_enable_cb = QCheckBox("Sample Pump Enable")
        layout.addWidget(self.sample_pump_enable_cb)
        self.sample_pump_reverse_cb = QCheckBox("Sample Pump Reverse")
        layout.addWidget(self.sample_pump_reverse_cb)
        self.sample_pump_ramp_cb = QCheckBox("Sample Pump Ramp") # put on by default
        layout.addWidget(self.sample_pump_ramp_cb)
        # frequency of pump steps 0.1 Hz, i.e. 10_000 for 1 kHz - fpga can do range(65_535)
        self.sample_pump_speed_spinbox = LabeledSpinBox('Sample Pump Speed', 0, 65_535, 0, 100)
        layout.addWidget(self.sample_pump_speed_spinbox)
        help_text(layout, '🛈 Sample pump speed is the frequency of pump steps in units of 0.1 Hz')
        # steps per cycle - speed increments per cycle
        self.sample_pump_rampSpC_spinbox = LabeledSpinBox('Sample Pump Ramp SpC', 0, 100, 1, 1)
        layout.addWidget(self.sample_pump_rampSpC_spinbox)
        help_text(layout, '🛈 Sample pump ramp SpC (speed increments per cycle) is the ramp step to make in units of 0.1 Hz when changing the pump speed')
        # cycles per clock - how many clock cycles before increment ramp step
        # note 100 MHz FPGA clock
        self.sample_pump_rampCpC_spinbox = LabeledSpinBox('Sample Pump Ramp CpC', 0, 2_000_000_000, 100_000, 1_000)
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
        layout.addWidget(self.sheath_pump_enable_cb)
        self.sheath_pump_duty_spinbox = LabeledSpinBox('Sheath Pump Duty', 0, 255, 127, 8)
        layout.addWidget(self.sheath_pump_duty_spinbox)
        help_text(layout, '🛈 Sheath pump duty is a number in the range 0--255, where 0 is off, and 255 is on 100% of the time')
        self.sheath_pump_freq_spinbox = LabeledSpinBox('Sheath Pump Frequency', 0, 2_000_000_000, 1000, 100)
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
        layout.addWidget(self.pressure_zero_btn)

        layout.addStretch()
        toolbox.addItem(tab, "Fluidics")

        # DACs tab:
        # dac bias, dac ref x chanels
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Content for dacs_tab"))
        layout.addStretch()
        toolbox.addItem(tab, "DACs")

        # ADCs tab:
        # checkboxes: enable x chanels, select all, select none
        # buttons: clear, capture
        # spinbox: capture samples
        # pg graph
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Content for adcs_tab"))
        layout.addStretch()
        toolbox.addItem(tab, "ADCs")

        # VI Monitoring tab:
        # label for each reading
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.read_monitors = QPushButton('Read Monitors')
        self.read_monitors.clicked.connect(lambda: self.get_instrument_state(['vi_monitors']))
        layout.addWidget(self.read_monitors)
        form = QFormLayout()
        self.monitor_labels = {}
        for channel in monitor_dictionary:
            self.monitor_labels[channel] = QLabel('None')
            form.addRow(monitor_dictionary[channel]['name'], self.monitor_labels[channel])
        layout.addLayout(form)

        title = QLabel('Cooling Fan')
        title.setStyleSheet(heading_style)
        layout.addWidget(title)
        self.fan_enable_cb = QCheckBox("Fan Enable")
        layout.addWidget(self.fan_enable_cb)
        self.fan_duty_spinbox = LabeledSpinBox('Fan Duty', 0, 255, 127, 8)
        layout.addWidget(self.fan_duty_spinbox)
        help_text(layout, '🛈 Fan duty is a number in the range 0--255, where 0 is off, and 255 is on 100% of the time')
        self.fan_freq_spinbox = LabeledSpinBox('Fan Frequency', 0, 2_000_000_000, 1000, 100)
        layout.addWidget(self.fan_freq_spinbox)
        help_text(layout, '🛈 fan frequency is the frequency of the duty cycle in Hz')
        self.fan_tacho_value = QLabel('0 rpm')
        layout.addWidget(self.fan_tacho_value)
        self.fan_tacho_btn = QPushButton('Measure Fan Speed')
        self.fan_tacho_btn.clicked.connect(lambda: self.get_instrument_state(['fan_tacho']))
        layout.addWidget(self.fan_tacho_btn)

        layout.addStretch()
        toolbox.addItem(tab, "Monitoring")

        # Temperatures tab:
        # fan control: enable cb, duty spin, freq spin, tacho label
        # label for each reading
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Content for temperatures_tab"))
        layout.addStretch()
        toolbox.addItem(tab, "Temperatures")

        # Front panel display:
        # file load dialog, upload button
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Content for display_tab"))
        layout.addStretch()
        toolbox.addItem(tab, "Display")

        # Registers tab:
        # write: register field, data field
        # read: register field, data label
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Content for registers_tab"))
        layout.addStretch()
        toolbox.addItem(tab, "Registers")

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

    def on_tab_changed(self, index):
        match index:
            case 0: # connection
                self.get_instrument_state(['version','datetime'])
                self.update_initialised()
            case 1: # light
                self.get_instrument_state(['laser_enable'])
            case 2: # fluidics
                self.get_instrument_state(['pressure'])
            case 3: # DACs
                pass
            case 4: # ADCs
                pass
            case 5: # monitoring
                pass
            case 6: # temperatures
                pass
            case 7: # display
                pass
            case 8: # registers
                pass

    @Slot(list)
    def get_instrument_state(self, parameters):
        if self.controller:
            self.controller.pipe_connection_instrument.send({'command': 'get_instrument_state', 'data': parameters})
            response = self.controller.pipe_connection_instrument.recv()
        else:
            response = {'message': {}}

        if 'check_id' in response['message']:
            if response['message']['check_id']:
                self.connection_status_connected.setVisible(True)
                self.connection_status_not_connected.setVisible(False)
            else:
                self.connection_status_connected.setVisible(False)
                self.connection_status_not_connected.setVisible(True)

        if 'version' in response['message']:
            self.version.setText(response['message']['version'])

        if 'datetime' in response['message']:
            self.datetime.setText(response['message']['datetime'])

        if 'pressure' in response['message']:
            self.pressure_value.setText(f'{response['message']['pressure']} Pa')

        if 'vi_monitors' in response['message']:
            for channel in monitor_dictionary:
                self.monitor_labels[channel].setText(f'{response['message'][channel]['V']} V, {response['message'][channel]['I']} mA')

        if 'fan_tacho' in response['message']:
            self.fan_tacho_value.setText(f'{response['message']['fan_tacho']} rpm')

        if self.bus:
            self.bus.statusMessage.emit(f'{response['source']} {response['status']}: {response['message']}')

        logger.info(response)

    @Slot(dict)
    def set_instrument_state(self, parameter_values):
        self.controller.pipe_connection_instrument.send({'command': 'set_instrument_state', 'data': parameter_values})
        response = self.controller.pipe_connection_instrument.recv()
        if self.bus:
            self.bus.statusMessage.emit(f'{response['source']} {response['status']}: {response['message']}')

        logger.info(response)

    @Slot()
    def update_initialised(self):
        if self.controller:
            if self.controller.is_instrument_initialised():
                self.initialised.setChecked(True)
            else:
                self.initialised.setChecked(False)


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication, QMainWindow
    app = QApplication([])
    window = PluginWidget()
    window.resize(1000, 1000)
    window.show()
    app.exec()
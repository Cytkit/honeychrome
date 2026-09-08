"""
Cytkit State plugin (hardware monitor and settings)
"""
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QPushButton, QLabel, QTabWidget, QToolBox, QFormLayout, QComboBox, QCheckBox, QSpinBox, QHBoxLayout, QFrame
from PySide6.QtCore import Qt, Slot, Signal, QSize

import logging

from honeychrome.controller import Controller
from honeychrome.instrument_driver_components.cykit_components.cytkit_configuration import monitor_dictionary
from honeychrome.main import configure_multiprocessing
from honeychrome.settings import heading_style
from honeychrome.view_components.event_bus import EventBus
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
        self.update_connection_status_btn.clicked.connect(lambda: self.get_instrument_state(['check_connection','read_id_data']))
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
        # clocks per cycle - how many clock cycles before increment ramp step
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
        form = QFormLayout()
        self.monitor_labels = {}
        for channel in monitor_dictionary:
            self.monitor_labels[channel] = QLabel('None')
            form.addRow(monitor_dictionary[channel]['name'], self.monitor_labels[channel])
        layout.addLayout(form)

        # fan control: enable cb, duty spin, freq spin, tacho label
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
                self.get_instrument_state(['vi_monitors','temperatures','fan_tacho'])
            case 6: # display
                pass
            case 7: # registers
                pass

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
            self.version.setText(response['message']['read_id_data']['version'])
            self.datetime.setText(response['message']['read_id_data']['datetime'])

        if 'pressure' in response['message']:
            self.pressure_value.setText(f'{response['message']['pressure']} Pa')

        if 'temperatures' in response['message']:
            self.temp_p_sensor_label.setText(f'{response['message']['temperatures']['temp_p_sensor']} C')

        if 'vi_monitors' in response['message']:
            for channel in monitor_dictionary:
                self.monitor_labels[channel].setText(f'{response['message']['vi_monitors']['V'][channel]} V, {response['message']['vi_monitors']['I'][channel]} mA')

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
            if 'bias' in response['message']['dacs']:
                for index in response['message']['dacs']['bias']:
                    self.dac_spinboxes[index].spinbox.setValue(response['message']['dacs']['bias'][index])
            if 'ref' in response['message']['dacs']:
                for index in response['message']['dacs']['ref']:
                    self.dac_spinboxes[index].spinbox.setValue(response['message']['dacs']['ref'][index])

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
    import sys

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    import multiprocessing as mp
    from multiprocessing import shared_memory, Lock
    import numpy as np

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

    app = QApplication([])
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

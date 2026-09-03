import time

import numpy as np

from honeychrome.instrument_driver_components.cykit_components.adcs import ADCs
from honeychrome.instrument_driver_components.cykit_components.cytkit_configuration import registers_map, monitor_dictionary
from honeychrome.instrument_driver_components.cykit_components.dacs import DACs
from honeychrome.instrument_driver_components.cykit_components.ft4222communicator import Ft4222Communicator
from honeychrome.instrument_driver_components.cykit_components.fan import Fan
from honeychrome.instrument_driver_components.cykit_components.i2c import I2C
from honeychrome.instrument_driver_components.cykit_components.id_data import IDData
from honeychrome.instrument_driver_components.cykit_components.laser import Laser
from honeychrome.instrument_driver_components.cykit_components.pressure import Pressure
from honeychrome.instrument_driver_components.cykit_components.sample_pump import SamplePump
from honeychrome.instrument_driver_components.cykit_components.sheath_pump import SheathPump
from honeychrome.instrument_driver_components.cykit_components.vi_monitor import VIMonitor


class CytkitDevice:
    """
    Device driver must provide the following methods:
        find_and_connect_to_device
            no arguments
            return status, message
        disconnect
            no arguments
            no return
        initialise
            no arguments
            return status, message
        start_acquisition
            no arguments
            return status, message
        stop_acquisition
            no arguments
            return status, message
        get_state
            argument: list of parameters to get, if list is empty gets everything
            return status, message (where message is dict of parameters)
        set_state
            argument: dict of parameters to set
            return status, message
        flush_sip
            no arguments
            return status, message
        backflush_sip
            no arguments
            return status, message
        read_out_traces
            no arguments
            returns blob of traces
    """
    def __init__(self):
        self.name = 'Cytkit'
        self.ft4222 = Ft4222Communicator()
        self.id_data = None
        self.fan = None
        self.laser = None
        self.pressure = None
        self.sample_pump = None
        self.sheath_pump = None
        self.i2c_bus_a = None
        self.i2c_bus_b = None
        self.vi_monitor = None
        self.dacs = None
        self.adcs = None

        self.initialised = False

    def find_and_connect_to_device(self):
        self.ft4222.find_and_connect()

        # if the connection above worked, initialise the hardware wrappers
        self.id_data = IDData(self.ft4222)
        self.fan = Fan(self.ft4222)
        self.laser = Laser(self.ft4222)
        self.pressure = Pressure(self.ft4222)
        self.sample_pump = SamplePump(self.ft4222)
        self.sheath_pump = SheathPump(self.ft4222)
        self.i2c_bus_a = I2C(self.ft4222, registers_map['I2CA_CTRL'], 'I2C Bus A')
        self.i2c_bus_b = I2C(self.ft4222, registers_map['I2CB_CTRL'], 'I2C Bus B')
        self.vi_monitor = VIMonitor(self.i2c_bus_a, self.i2c_bus_b)
        self.dacs = DACs(self.i2c_bus_a)
        self.adcs = ADCs(self.ft4222)

        print('[Cytkit driver] Connected')
        return  'OK', 'Connected to Cytkit'

    def disconnect(self):
        pass

    def initialise(self):
        # id_word = self.ft4222.register_read('ID_WORD')
        # print(id_word)

        if not self.initialised:
            self.laser.set_state(1)  # turn on laser
            self.initialised = True
            return 'OK', 'Cytkit initialised'
        else:
            self.laser.set_state(0)  # turn off laser
            self.initialised = False
            return 'OK', 'Cytkit on standby'


    def start_acquisition(self):
        return 'OK', 'Cytkit started acquisition'

    def stop_acquisition(self):
        return 'OK', 'Cytkit stopped acquisition'

    def set_state(self, dict_of_parameter_value):
        message = {}
        for parameter, value in dict_of_parameter_value.items():
            if parameter == 'laser_enable':
                self.laser.set_state(int(value))
                message['laser_enable'] = value

        return 'OK', message

    def get_state(self, list_of_parameters):
        message = {}
        if 'check_id' in list_of_parameters:
            value = self.id_data.check_id()
            message['check_id'] = value

        if 'pressure'in list_of_parameters:
            value = self.pressure.get_pressure('PRES_UNITS_PA', 1)
            message['pressure'] = value

        if 'temperatures' in list_of_parameters:
            message['temperatures'] = {}
            message['temperatures']['temp_p_sensor'] = self.pressure.get_temperature()

        if 'vi_monitors' in list_of_parameters:
            message['vi_monitors'] = {'V':{}, 'I':{}}
            for channel in monitor_dictionary:
                V = self.vi_monitor.read_voltage(channel)
                I = self.vi_monitor.read_current(channel)
                message['vi_monitors'][channel]['V'] = V
                message['vi_monitors'][channel]['I'] = I

        if 'fan_tacho'in list_of_parameters:
            value = self.fan.get_tacho()
            message['pressure'] = value

        return 'OK', message

    def flush_sip(self):
        return 'OK', 'Cytkit SIP flushed'

    def backflush_sip(self):
        return 'OK', 'Cytkit SIP backflushed'


    def read_out_traces(self):
        memory_head, memory_tail, n_events_in_memory = self.ft4222.get_memory_head_tail_n_events()
        blob_of_traces_as_array = self.ft4222.pop_from_memory(memory_head, memory_tail)
        return blob_of_traces_as_array


if __name__ == '__main__':

    cytkit_device = CytkitDevice()
    cytkit_device.find_and_connect_to_device()

    time.sleep(1)

    cytkit_device.start_acquisition()
    blob = cytkit_device.read_out_traces()

    print(blob.shape)

    cytkit_device.disconnect()


import time

import numpy as np

from honeychrome.instrument_driver_components.cykit_components.ft4222communicator import Ft4222Communicator
from honeychrome.instrument_driver_components.cykit_components.fan import Fan
from honeychrome.instrument_driver_components.cykit_components.laser import Laser
from honeychrome.instrument_driver_components.cykit_components.sample_pump import SamplePump
from honeychrome.instrument_driver_components.cykit_components.sheath_pump import SheathPump


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
        self.ft4222 = Ft4222Communicator()
        self.fan = None
        self.laser = None
        self.pressure = None
        self.sample_pump = None
        self.sheath_pump = None
        self.vi_monitor = None
        self.initialised = False

    def find_and_connect_to_device(self):
        self.ft4222.find_and_connect()

        # if the connection above worked, initialise the hardware wrappers
        self.fan = Fan(self.ft4222)
        self.laser = Laser(self.ft4222)
        self.pressure = Pressure(self.ft4222)
        self.sample_pump = SamplePump(self.ft4222)
        self.sheath_pump = SheathPump(self.ft4222)
        self.vi_monitor = VIMonitor(self.ft4222)

        # then configure the hardware
        self.initialise()

        print('[Cytkit driver] Connected')
        return {'source': '[Cytkit driver]', 'status': 'OK', 'message': 'Connected to Cytkit'}

    def disconnect(self):
        pass

    def initialise(self):
        id_word = self.ft4222.register_read('ID_WORD')
        print(id_word)

        self.laser.set_state(1) # turn on laser

        if not self.initialised:
            self.initialised = True
        else:
            self.initialised = False

        return 'OK', 'Cytkit initialised'

    def start_acquisition(self):
        return 'OK', 'Cytkit started acquisition'

    def stop_acquisition(self):
        return 'OK', 'Cytkit stopped acquisition'

    def set_state(self, dict_of_parameters):
        return 'OK', 'Cytkit state set'

    def get_state(self, list_of_parameters):
        return 'OK', {parameter:None for parameter in list_of_parameters}

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
    cytkit_device.connect_to_device()

    time.sleep(1)

    cytkit_device.start_acquisition()
    blob = cytkit_device.read_out_traces()

    print(blob.shape)

    cytkit_device.disconnect()


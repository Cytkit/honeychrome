import time

import numpy as np

from honeychrome.instrument_driver_components.cykit_components.ft4222communicator import Ft4222Communicator
from honeychrome.instrument_driver_components.cykit_components.fan import Fan
from honeychrome.instrument_driver_components.cykit_components.laser import Laser


class CytkitDevice:
    """
    Device driver must provide the following methods:
        connect_to_device
        disconnect
        start_acquisition
        stop_acquisition
        change_device_settings
        read_out_traces
    """
    def __init__(self):
        self.ft4222 = Ft4222Communicator()
        self.fan = None
        self.laser = None
        self.pressure = None
        self.sample_pump = None
        self.sheath_pump = None
        self.vi_monitor = None

    def connect_to_device(self):
        self.ft4222.find_and_connect()

        # if the connection above worked, initialise the hardware wrappers
        self.fan = Fan(self.ft4222)
        self.laser = Laser(self.ft4222)
        self.pressure = Pressure(self.ft4222)
        self.sample_pump = SamplePump(self.ft4222)
        self.sheath_pump = SheathPump(self.ft4222)
        self.vi_monitor = VIMonitor(self.ft4222)

        # then configure the hardware
        self._configure_instrument()

        print('[Cytkit driver] Connected')
        return {'source': '[Cytkit driver]', 'status': 'OK', 'message': 'Connected to Cytkit'}

    def disconnect(self):
        pass

    def start_acquisition(self):
        pass

    def stop_acquisition(self):
        pass

    def change_device_settings(self, settings):
        pass

    def read_out_traces(self):
        memory_head, memory_tail, n_events_in_memory = self.ft4222.get_memory_head_tail_n_events()
        blob_of_traces_as_array = self.ft4222.pop_from_memory(memory_head, memory_tail)
        return blob_of_traces_as_array

    def _configure_instrument(self):
        id_word = self.ft4222._register_read('ID_WORD')
        print(id_word)

        self.laser.set_state(1) # turn on laser


if __name__ == '__main__':

    cytkit_device = CytkitDevice()
    cytkit_device.connect_to_device()

    time.sleep(1)

    cytkit_device.start_acquisition()
    blob = cytkit_device.read_out_traces()

    print(blob.shape)

    cytkit_device.disconnect()


import logging
import random
import time
from pathlib import Path
from threading import Thread, Event, Lock

from flowkit import Sample
import numpy as np

from honeychrome import settings
from honeychrome.instrument_driver_components.cytkit_components.display import Display
from honeychrome.settings import adc_channels, magnitude_ceiling, traces_cache_dtype, n_channels_trace, n_time_points_in_event, transfer_target_repeat_time

fcs_file = Path(__file__).parent / 'data' / 'example_for_dummy_acquisition.fcs'
dummy_event_rate = 1000
trace_indices = np.arange(n_time_points_in_event)  # x positions

def gaussian_rows_areas(x_grid, areas, mu, sigma):
    """
    Create 2D array where each row is a Gaussian with specified area.

    Parameters:
    x_grid: 1D array of x positions
    areas: 1D array of areas
    mu: 1D array of mean position of Gaussian
    sigma: 1D array of standard deviation

    Returns:
    2D array of shape (len(areas), len(x_grid))
    """
    N, M = areas.shape
    array_of_traces = np.empty((N*M, len(x_grid)))
    for n in range(N):
        gaussian = np.exp(-0.5 * ((x_grid - mu[n]) / sigma[n]) ** 2) / (sigma[n] * np.sqrt(2 * np.pi))
        array_of_traces[n*M:(n+1)*M,:] = areas[n,:][:,None] * gaussian

    return array_of_traces

class EventRateCounter:
    def __init__(self):
        self.time_last = None
        self.event_rate = 0

    def update(self, number_of_new_events):
        time_now = time.perf_counter()
        if not self.time_last or time_now - self.time_last > 1.0:
            self.event_rate = 0
        elif self.event_rate == 0:
            interval = time_now - self.time_last
            self.event_rate = number_of_new_events/interval
        else:
            interval = time_now - self.time_last
            self.event_rate = self.event_rate * (1.0-interval) + number_of_new_events
        self.time_last = time_now

    def reset(self):
        self.time_last = None
        self.event_rate = 0

class SamplePumpSim:
    def __init__(self):
        self.enable = False
        self.reverse = False
        self.speed = 0

    def get_enable(self):
        return self.enable
    def get_reverse(self):
        return self.reverse
    def get_speed(self):
        return self.speed

class LaserSim:
    def __init__(self):
        self.enable = False

    def get_state(self):
        return self.enable

class SamplePumpFlowRateGetter(Thread):
    def __init__(self, parent, sample_pump):
        super().__init__(daemon=True)
        self.sample_pump = sample_pump
        self._stop_event = Event()
        self._lock = Lock()
        self.parent = parent
        self.flow_rate = 0

    def run(self):
        while not self._stop_event.is_set():
            enabled = self.sample_pump.get_enable()
            reverse = -1 if self.sample_pump.get_reverse() else 1
            speed = self.sample_pump.get_speed()
            steps_per_microlitre = self.parent.sample_pump_steps_per_microlitre
            self.flow_rate = enabled * reverse * speed / steps_per_microlitre
            time.sleep(0.5)

class LaserGetter(Thread):
    def __init__(self, laser):
        super().__init__(daemon=True)
        self.laser = laser
        self._stop_event = Event()
        self._lock = Lock()
        self.enabled = 0

    def run(self):
        while not self._stop_event.is_set():
            self.enabled = self.laser.get_state()
            time.sleep(0.5)

class RandomPressureGenerator(Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.pressure = 0

    def run(self):
        while True:
            self.pressure = random.normalvariate(-18, sigma=2)
            time.sleep(0.1)

class RandomTemperatureGenerator(Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.temperature = 0

    def run(self):
        while True:
            self.temperature = random.normalvariate(26, sigma=10)
            time.sleep(0.1)

class DummyDevice:
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
            set_gain
                argument: dict of parameters to set {channel: bias} in dac units 0..255
                return status, message
            set_sample_flow_rate
                argument: dict of parameters to set:
                    {'sample_flow_rate': (float)
                    'steps_per_microlitre': (float)}
                return status, message
            read_out_traces
                no arguments
                returns blob of traces
    """
    def __init__(self):
        self.name = 'Dummy'
        sample = Sample(fcs_file)
        self.events = sample.get_events('raw')

        fcs_file_channels = list(sample.channels['pnn'])
        area_channels = [s[:-2] for s in fcs_file_channels]
        self.channel_indices = [area_channels.index(channel) for channel in adc_channels]
        self.fsc_area_index = fcs_file_channels.index('FSC-A')
        self.fsc_height_index = fcs_file_channels.index('FSC-H')
        self.scale = magnitude_ceiling / int(sample.metadata['p2r'])
        self.initialised = False
        self.logger = logging.getLogger(__name__)

        self.sample_pump_acquisition_rate = settings.sample_pump_acquisition_rate_retrieved
        self.sample_pump_steps_per_microlitre = settings.steps_per_microlitre_retrieved

        self.event_rate_counter = EventRateCounter()
        self.sample_pump = SamplePumpSim()
        self.laser = LaserSim()
        self.sample_pump_flow_rate_getter = SamplePumpFlowRateGetter(self, self.sample_pump)
        self.sample_pump_flow_rate_getter.start()
        self.pressure_control_worker = RandomPressureGenerator()
        self.pressure_control_worker.start()
        self.temperature_control_worker = RandomTemperatureGenerator()
        self.temperature_control_worker.start()
        self.laser_getter = LaserGetter(self.laser)
        self.laser_getter.start()
        self.display = Display(transfer_object=self.event_rate_counter, sample_pump_object=self.sample_pump_flow_rate_getter, pressure_object=self.pressure_control_worker, temperature_object=self.temperature_control_worker, laser_object=self.laser_getter)
        self.display.start()

    def find_and_connect_to_device(self):
        return 'OK', 'Dummy device connected'

    def disconnect(self):
        if self.display.is_alive():
            self.display.join(timeout=2)

        return 'OK', 'Dummy device disconnected'

    def initialise(self):
        self.logger.info("Example initialisation message to log")
        if not self.initialised:
            self.initialised = True
            self.laser.enable = True
            self.display.action_message(["Initialised!", "Sheath on, laser on."])
            return 'OK', 'Dummy device initialised'
        else:
            self.initialised = False
            self.laser.enable = False
            self.display.action_message(["Stand by", "Sheath off, laser off."])
            return 'OK', 'Dummy device on standby'

    def start_acquisition(self):
        self.display.action_message("Acquiring...")
        self.sample_pump.enable = True
        self.sample_pump.reverse = False
        self.sample_pump.speed = int(self.sample_pump_acquisition_rate * self.sample_pump_steps_per_microlitre)
        return 'OK', 'Dummy device started acquisition'

    def stop_acquisition(self):
        self.display.action_message("Acquisition stopped.")
        self.sample_pump.enable = False
        self.sample_pump.reverse = False
        self.sample_pump.speed = 0
        self.event_rate_counter.reset()
        return 'OK', 'Dummy device stopped acquisition'

    def set_state(self, dict_of_parameter_value):
        print(dict_of_parameter_value)
        return 'OK', {parameter: value for parameter, value in dict_of_parameter_value.items()}

    def get_state(self, list_of_parameters):
        # return ('OK',
        #  {'dacs':
        #       {'bias': {0: 50, 1: 51, 2: 52, 3: 53, 4: 54, 5: 55, 6: 56, 7: 57, 8: 58, 9: 59, 10: 60, 11: 61, 12: 62, 13: 63, 14: 64, 15: 65},
        #        'ref': {0: 20, 1: 19, 2: 18, 3: 17, 4: 16, 5: 15, 6: 14, 7: 13, 8: 12, 9: 11, 10: 10, 11: 9, 12: 8, 13: 7, 14: 6, 15: 5}
        #        }
        #   }
        #  )
        print(list_of_parameters)
        return 'OK', {parameter:None for parameter in list_of_parameters}


    def flush_sip(self):
        self.display.action_message("Flushing SIP.")
        return 'OK', 'Dummy device doesn''t have a sip to flush'

    def backflush_sip(self):
        self.display.action_message("Backflushing SIP.")
        return 'OK', 'Dummy device doesn''t have a sip to backflush'

    def set_gain(self, dict_of_gains):
        return 'OK', 'Dummy device doesn''t have gains'

    def set_sample_flow_rate(self, data):
        if 'sample_flow_rate' in data:
            self.sample_pump_acquisition_rate = data['sample_flow_rate']
        if 'steps_per_microlitre' in data:
            self.sample_pump_steps_per_microlitre = data['steps_per_microlitre']
        self.sample_pump.speed = int(self.sample_pump_acquisition_rate * self.sample_pump_steps_per_microlitre)

        return 'OK', 'Dummy device doesn''t have a sample pump'

    def generate_traces(self, n):
        # reads events, returns a blob_np
        # note blob_np is 1d numpy array of n * n_channels_trace * n_time_points_in_event
        indices = np.random.choice(len(self.events), size=n, replace=False)
        areas_to_process = self.events[indices][:,self.channel_indices]
        fsc_area = self.events[indices, self.fsc_area_index]
        fsc_height = self.events[indices, self.fsc_height_index]
        widths = fsc_area / fsc_height / 3
        centres = [-np.random.randint(n_time_points_in_event//5) + n_time_points_in_event//2 for _ in range(len(widths))]
        traces = gaussian_rows_areas(trace_indices, areas_to_process, centres, widths)
        traces *= self.scale
        traces = traces.astype(traces_cache_dtype)
        traces = traces.reshape(-1)
        return traces, indices

    def read_out_traces(self):
        n = dummy_event_rate * transfer_target_repeat_time
        n_events_in_memory = np.random.poisson(n)
        self.event_rate_counter.update(n_events_in_memory)
        blob_of_traces_as_array, _ = self.generate_traces(n_events_in_memory)
        return blob_of_traces_as_array

if __name__ == '__main__':
    dummy_instrument = DummyDevice()
    blob_np, indices = dummy_instrument.generate_traces(5)
    print('blob_np:', blob_np)

    traces = blob_np.reshape(5, n_channels_trace, n_time_points_in_event)
    print('events:', dummy_instrument.events[0, dummy_instrument.channel_indices] * dummy_instrument.scale)
    print('sum of traces in each channel:', traces[0].sum(axis=1))

    from matplotlib import pyplot as plt
    print('FSC-A:', dummy_instrument.events[indices, dummy_instrument.fsc_area_index] * dummy_instrument.scale)
    print('FSC-H:', dummy_instrument.events[indices, dummy_instrument.fsc_height_index] * dummy_instrument.scale)
    plt.plot(traces[:,0,:].T)
    plt.show()

import logging
import time
from pathlib import Path
from flowkit import Sample
import numpy as np

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

    def find_and_connect_to_device(self):
        return 'OK', 'Dummy device connected'

    def disconnect(self):
        pass

    def initialise(self):
        self.logger.info("Example initialisation message to log")
        time.sleep(1)
        if not self.initialised:
            self.initialised = True
            return 'OK', 'Dummy device initialised'
        else:
            self.initialised = False
            return 'OK', 'Dummy device on standby'

    def start_acquisition(self):
        return 'OK', 'Dummy device started acquisition'

    def stop_acquisition(self):
        return 'OK', 'Dummy device stopped acquisition'

    def set_state(self, dict_of_parameter_value):
        return 'OK', {parameter: value for parameter, value in dict_of_parameter_value.items()}

    def get_state(self, list_of_parameters):
        return 'OK', {parameter:None for parameter in list_of_parameters}

    def flush_sip(self):
        return 'OK', 'Dummy device doesn''t have a sip to flush'

    def backflush_sip(self):
        return 'OK', 'Dummy device doesn''t have a sip to backflush'

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
        n_events_in_memory = np.random.randint(n)
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

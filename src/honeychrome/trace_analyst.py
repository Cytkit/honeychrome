'''
Trace Analyser:
-Listens for start event
-Consumes cached traces
-Calculates peak height, area, width (according to settings) and adds to events cache
-Copies latest trace with peak measurements
-Signals when new events chunk is ready
'''

import multiprocessing as mp
from multiprocessing import shared_memory, Lock
import threading
import numpy as np
import time

from honeychrome.settings import traces_cache_size, max_events_in_traces_cache, trace_n_points, n_channels_trace, adc_rate, threshold_retrieved, n_time_points_in_event, window_extension_length_pre, baseline_decay_rate, window_extension_length_post, timeout_length, deltaT, adc_scale_uv
from honeychrome.settings import max_events_in_cache, n_channels_per_event, event_channels_pnn, analyser_target_repeat_time, baseline_length
from honeychrome.settings import adc_channels, peak_detection_channel_retrieved, area_channels, height_channels, width_channels, scatter_channels, fluorescence_channels

verbosity_divider = int(1/analyser_target_repeat_time)
gap = 10 # gap between start of end of baseline and start of peak

default_baselines = np.zeros(n_channels_trace)

def peak_start_stop_baseline(tr, threshold):
    # initialise peak detector
    baseline = tr[gap:baseline_length].mean()
    countdown = 0
    n_start = 0
    in_peak = False

    for n in range(window_extension_length_pre, len(tr)):
        # if not in peak, update baselines and detect if there is a peak
        condition = tr[n] - baseline > threshold
        if not in_peak:
            if condition:
                in_peak = True
                n_start = n - window_extension_length_pre  # start of integration and end of blc
            else:
                baseline = baseline * (1 - baseline_decay_rate) + tr[n - window_extension_length_pre] * baseline_decay_rate

        if in_peak:
            # start countdown if signal dips below threshold
            if (not condition) and (countdown == 0) and (n > n_start + window_extension_length_pre):
                countdown = window_extension_length_post

            if countdown > 0:
                countdown -= 1
                if countdown == 0 or (n - n_start >= timeout_length + window_extension_length_pre) or (n == len(tr) - 1):
                    in_peak = False
                    break

    n_end = n  # end of integration and restart of blc

    # if no peak, then n_start=0 and n_end=end index of trace
    return n_start, n_end, baseline

def peak_measurements(tr, n_start, n_end, deltaT, area_indices, height_indices):
    # baselines = np.concatenate((tr[:,gap:n_start-gap], tr[:,n_end-gap:]), axis=1).mean(axis=1)
    baselines = tr[:,gap:n_start-gap].mean(axis=1)
    traces_truncated_baseline_subtracted = tr[:,n_start:n_end] - baselines.reshape(-1,1)
    heights = np.max(traces_truncated_baseline_subtracted[height_indices,:], axis=1)
    areas = np.sum(traces_truncated_baseline_subtracted[area_indices,:], axis=1)
    width = (n_end - n_start) * deltaT * 1000 # width in nanoseconds
    centre = (n_start + n_end) * 0.5
    peak = areas, heights, width, centre

    return peak, baselines

class TraceAnalyser(mp.Process):
    def __init__(self,
                 traces_cache_name=None,
                 traces_cache_lock=None,
                 index_head_traces_cache=None,
                 index_tail_traces_cache=None,
                 events_cache_name=None,
                 events_cache_lock=None,
                 index_head_events_cache=None,
                 index_tail_events_cache=None,
                 oscilloscope_traces_queue=None,
                 pipe_connection=None,
                 logging_queue=None):
        super().__init__()

        # Route standard logs through the queue
        import logging.handlers
        qh = logging.handlers.QueueHandler(logging_queue)
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(qh)

        # pipe connection
        self.pipe_connection = pipe_connection

        # Traces cache
        self.traces_cache = None
        self.traces_cache_name = traces_cache_name
        self.traces_cache_lock = traces_cache_lock
        self.max_events_in_traces_cache = max_events_in_traces_cache
        self.trace_n_points = trace_n_points
        self.n_channels_trace = n_channels_trace
        self.n_time_points_in_event = int(trace_n_points//n_channels_trace)

        # Events cache
        self.events_cache = None
        self.events_cache_name = events_cache_name
        self.events_cache_lock = events_cache_lock
        self.max_events_in_cache = max_events_in_cache
        self.index_head_traces_cache = index_head_traces_cache
        self.index_tail_traces_cache = index_tail_traces_cache
        self.index_head_events_cache = index_head_events_cache
        self.index_tail_events_cache = index_tail_events_cache

        self.indices_area_channels_in_traces = [(True if c in area_channels else False) for c in adc_channels]
        self.indices_height_channels_in_traces = [(True if c in height_channels else False) for c in adc_channels]
        self.indices_peak_detection_channel_in_traces = adc_channels.index(peak_detection_channel_retrieved)
        self.index_time_channel_in_events = event_channels_pnn.index('Time')
        self.index_event_id_in_events = event_channels_pnn.index('event_id')
        self.indices_area_channels_in_events = [event_channels_pnn.index(c + '-A') for c in area_channels]
        self.indices_height_channels_in_events = [event_channels_pnn.index(c + '-H') for c in height_channels]
        self.index_width_channel_in_events = [event_channels_pnn.index(c + '-W') for c in width_channels][0] # there is only one
        self.n_channels_per_event = n_channels_per_event
        self.threshold = threshold_retrieved * 1000 # uV

        # Oscilloscope traces queue
        self.oscilloscope_traces_queue = oscilloscope_traces_queue

        self.stop_analyser = None

    def run(self):
        # initialise the things that can't be pickled
        self.stop_analyser = threading.Event()

        # Attach to existing shared memory
        shm_traces = shared_memory.SharedMemory(name=self.traces_cache_name)
        with self.traces_cache_lock:
            self.traces_cache = np.ndarray((self.max_events_in_traces_cache * self.trace_n_points), dtype=np.uint16, buffer=shm_traces.buf)
        shm_events = shared_memory.SharedMemory(name=self.events_cache_name)
        with self.events_cache_lock:
            self.events_cache = np.ndarray((self.max_events_in_cache, self.n_channels_per_event), dtype=np.int64, buffer=shm_events.buf)

        # main loop waiting for commands from experiment control
        thread = None
        while True:
            try:
                incoming_from_experiment_control = self.pipe_connection.recv()
            except EOFError:
                self.logger.info("Pipe closed by experiment control; shutting down gracefully.")
                break

            if incoming_from_experiment_control['command'] == 'start':
                # reset events cache to zeros
                with self.events_cache_lock:
                    self.events_cache[:] = 0
                with self.index_head_events_cache.get_lock():
                    self.index_head_events_cache.value = 0
                with self.index_tail_events_cache.get_lock():
                    self.index_tail_events_cache.value = 0
                self.logger.info('[Trace Analyser] Events cache flushed!')
                if thread:
                    self.stop_analyser.set()
                    while thread.is_alive():
                        self.logger.info('[Trace Analyser] Waiting until previous analysis thread ends')
                        time.sleep(0.25)
                self.stop_analyser.clear()
                # create analysis thread
                thread = threading.Thread(
                    target=self.analyse,
                    daemon=True
                )
                thread.start()
                self.logger.info('[Trace Analyser] Started')
                response_to_experiment_control = {'status': 'OK', 'message': '[Trace Analyser] started'}

            elif incoming_from_experiment_control['command'] == 'stop':
                self.stop_analyser.set()
                if thread:
                    if thread.is_alive():
                        thread.join(0.1)
                        while thread.is_alive():
                            self.logger.info('[Trace Analyser] Waiting until analysis thread ends')
                            time.sleep(0.25)
                response_to_experiment_control = {'status':'OK', 'message':'[Trace Analyser] stopped'}

            elif incoming_from_experiment_control['command'] == 'configure':
                parameters = incoming_from_experiment_control['data']
                if 'height_channels' in parameters:
                    self.indices_height_channels_in_traces = [(True if c in parameters['height_channels'] else False) for c in adc_channels]
                    self.indices_height_channels_in_events = [event_channels_pnn.index(c + '-H') for c in height_channels]
                if 'peak_detection_channel' in parameters:
                    self.indices_peak_detection_channel_in_traces = adc_channels.index(parameters['peak_detection_channel'])
                    # todo: this also redirects the width measurement without changing the width channel in the event data
                if 'threshold' in parameters:
                    self.threshold = parameters['threshold'] * 1000 # uV

                self.logger.info('[Trace Analyser] Channel configuration set')
                response_to_experiment_control = {'status':'OK', 'message':'[Trace Analyser] channel configuration set'}

            elif incoming_from_experiment_control['command'] == 'quit':
                self.stop_analyser.set()
                if thread:
                    if thread.is_alive():
                        thread.join(0.1)
                        while thread.is_alive():
                            self.logger.info('[Trace Analyser] Waiting until analysis thread ends')
                            time.sleep(0.25)
                response_to_experiment_control = {'status':'OK', 'message':'[Trace Analyser] quit'}
                self.pipe_connection.send(response_to_experiment_control)
                break

            self.pipe_connection.send(response_to_experiment_control)

        shm_traces.close()
        shm_events.close()
        self.oscilloscope_traces_queue.cancel_join_thread()
        self.logger.info('[Trace Analyser] Quit')


    def analyse(self):
        start_time  = time.perf_counter()
        cycle_n = 0

        # stop if stop analyser event is set
        while not self.stop_analyser.is_set():
            cycle_start_time = time.perf_counter()
            cycle_n += 1

            '''
            input cached traces
            for all new events, calculate area, height, width as specified in channel_dict, add all channels to events array as specified
            '''
            with self.index_head_traces_cache.get_lock():
                traces_head = self.index_head_traces_cache.value
            with self.index_tail_traces_cache.get_lock():
                traces_tail = self.index_tail_traces_cache.value

            with self.index_head_events_cache.get_lock():
                events_head = self.index_head_events_cache.value # not used - events cache is not cycled
            with self.index_tail_events_cache.get_lock():
                events_tail = self.index_tail_events_cache.value

            # extract the blob as numpy array
            n_new_events = traces_tail - traces_head
            if traces_head < traces_tail:
                queue_begin_index = traces_head % self.max_events_in_traces_cache
                queue_end_index = traces_tail % self.max_events_in_traces_cache
                slice_begin = queue_begin_index * self.trace_n_points
                slice_end = queue_end_index * self.trace_n_points
                with self.traces_cache_lock:
                    if queue_end_index > queue_begin_index:
                        blob_np = self.traces_cache[slice_begin:slice_end]
                    else:
                        blob_np_back = self.traces_cache[slice_begin:]  # put get back of queue array
                        blob_np_front = self.traces_cache[:slice_end]  # get front of queue array
                        blob_np = np.concatenate((blob_np_back, blob_np_front))

                # scale to uV and convert to float
                traces_batch_scaled = blob_np.reshape(-1, self.n_channels_trace, self.n_time_points_in_event).astype(np.float32)
                traces_batch_scaled[traces_batch_scaled==65535] = np.nan
                traces_batch_scaled = traces_batch_scaled * adc_scale_uv  # scale in uV, float32

                # calculate area, height, width as defined in channel_dict and write to events_cache

                ### Simple calculation - only good if peaks already filtered and background-subtracted
                # areas = traces_batch_scaled[:, self.indices_area_channels_in_traces, :].sum(axis=2)
                # heights = traces_batch_scaled[:, self.indices_height_channels_in_traces, :].max(axis=2)
                # widths = self.calculate_width(traces_batch_scaled[:, self.indices_peak_detection_channel_in_traces, :])

                ### More sophisticated calculation - loop over traces and perform background substraction
                keep_mask = np.zeros(n_new_events, dtype=np.bool)
                areas = np.zeros((n_new_events, len(self.indices_area_channels_in_events)))
                heights = np.zeros((n_new_events, len(self.indices_height_channels_in_events)))
                widths = np.zeros(n_new_events)
                last_event_index = 0
                for n, traces in enumerate(traces_batch_scaled):
                    n_start, n_end, tr_baseline = peak_start_stop_baseline(traces[self.indices_peak_detection_channel_in_traces], self.threshold)
                    if n_start > 0:
                        peak, baselines = peak_measurements(traces, n_start, n_end, deltaT, self.indices_area_channels_in_traces, self.indices_height_channels_in_traces)
                        areas[n], heights[n], widths[n], centre = peak
                        keep_mask[n] = True
                        last_event_index = n

                # keep only the ones above threshold
                areas = areas[keep_mask]
                heights = heights[keep_mask]
                widths = widths[keep_mask]
                n_new_events = np.sum(keep_mask)
                times = np.ones(n_new_events, dtype=np.int64) * int((time.perf_counter() - start_time) * 1000)
                event_ids = np.array(range(events_tail, events_tail + n_new_events))

                # if there were no peaks detected but at least one trace, push the last one to the oscilloscope queue
                if last_event_index == 0:
                    if n > 0:
                        self.oscilloscope_traces_queue.put({'event_id': None, 'time': None,
                                                            'traces': traces_batch_scaled[last_event_index],
                                                            'n_start': n_start, 'n_end': n_end,
                                                            'peak': None, 'baselines': default_baselines})

                # if any above threshold, then push to events cache
                if last_event_index > 0:

                    self.oscilloscope_traces_queue.put({'event_id': event_ids[-1], 'time': times[-1],
                                                        'traces': traces_batch_scaled[last_event_index],
                                                        'n_start': n_start, 'n_end': n_end, 'peak': peak,
                                                        'baselines': baselines})
                    with self.events_cache_lock:
                        self.events_cache[events_tail:events_tail + n_new_events, self.indices_area_channels_in_events] = areas
                        self.events_cache[events_tail:events_tail + n_new_events, self.indices_height_channels_in_events] = heights
                        self.events_cache[events_tail:events_tail + n_new_events, self.index_width_channel_in_events] = widths
                        self.events_cache[events_tail:events_tail + n_new_events, self.index_time_channel_in_events] = times
                        self.events_cache[events_tail:events_tail + n_new_events, self.index_event_id_in_events] = event_ids

                        ### debug print latest events
                        #self.logger.info(self.events_cache[events_tail:events_tail + n_new_events])

                else:
                    if not cycle_n % verbosity_divider:
                        self.logger.info(f'[Trace Analyser] {len(keep_mask)} traces returned but none above threshold')

                # update head of traces cache and tail of events cache
                events_tail += n_new_events
                if not cycle_n % verbosity_divider:
                    self.logger.info(f'[Trace Analyser] analysed {n_new_events} events (traces cache old head:{traces_head}, new head and tail:{traces_tail}), (events cache head:{events_head}, tail:{events_tail})')
                traces_head = traces_tail

                with self.index_head_traces_cache.get_lock():
                    self.index_head_traces_cache.value = traces_head
                with self.index_tail_events_cache.get_lock():
                    self.index_tail_events_cache.value = events_tail


            else:
                if not cycle_n % verbosity_divider:
                    self.logger.info(f'[Trace Analyser] awaiting traces (traces cache head:{traces_head}, tail:{traces_tail})')

            # stop if events cache full
            if events_tail + n_new_events > self.max_events_in_cache:
                self.logger.warning("[Trace Analyser] Events cache is full, stopping trace analyser")
                break

            # Calculate elapsed time and sleep precisely
            elapsed = time.perf_counter() - cycle_start_time
            #print(f'[Trace Analyser] analysis elapsed time {elapsed}') # for debugging
            sleep_time = max(0, analyser_target_repeat_time - elapsed)
            time.sleep(sleep_time)

        self.stop_analyser.clear()
        self.logger.info('[Trace Analyser] Stopped')

    def calculate_width(self, traces):
        n_traces, n_time_points = traces.shape
        widths = np.zeros(n_traces)
        for n in range(n_traces):
            half_height = traces[n].max() * 0.5
            rising = np.argmax(traces[n] > half_height)
            falling = rising + np.argmin(traces[n][rising:] > half_height)
            widths[n] = falling - rising

        widths *= int(adc_rate * 1000) # nanoseconds
        return widths

if __name__ == '__main__':
    mp.set_start_method("spawn")

    # Allocate shared memory block, plus head and tail indices
    traces_cache_shm = shared_memory.SharedMemory(create=True, size=np.zeros(traces_cache_size, dtype=np.uint16).nbytes)
    traces_cache_lock = Lock()
    index_head_traces_cache = mp.Value('i', 0)
    index_tail_traces_cache = mp.Value('i', 0)

    events_cache_shm = shared_memory.SharedMemory(create=True, size=np.zeros((max_events_in_cache, n_channels_per_event), dtype=np.int64).nbytes)
    events_cache_lock = Lock()
    index_head_events_cache = mp.Value('i', 0)
    index_tail_events_cache = mp.Value('i', 0)

    # oscilloscope traces
    oscilloscope_traces_queue = mp.Queue()
    # command pipes
    pipe_experiment_instrument_e, pipe_experiment_instrument_i = mp.Pipe()
    pipe_experiment_analyser_e, pipe_experiment_analyser_a = mp.Pipe()

    '''
    Firstly, set up instrument
    '''
    # start instrument dummy
    from honeychrome.instrument_communicator import Instrument
    instrument = Instrument(
        use_dummy_instrument=True,
        traces_cache_name=traces_cache_shm.name,
        traces_cache_lock=traces_cache_lock,
        index_head_traces_cache=index_head_traces_cache,
        index_tail_traces_cache=index_tail_traces_cache,
        pipe_connection=pipe_experiment_instrument_i
    )
    instrument.start()

    '''
    Secondly, set up analyst
    '''
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
    Thirdly, send commands and read data
    
    connect instrument
    start instrument
    start analyser
    stop analyser
    set channels
    start analyser again
    stop instrument
    stop analyser
    quit analyser
    quit instrument
    '''
    # connect instrument
    pipe_experiment_instrument_e.send({'command':'connect'})
    response = pipe_experiment_instrument_e.recv()
    print(response)

    # start instrument
    pipe_experiment_instrument_e.send({'command':'start'})
    response = pipe_experiment_instrument_e.recv()
    print(response)

    # start analyser
    pipe_experiment_analyser_e.send({'command':'start'})
    response = pipe_experiment_analyser_e.recv()
    print(response)


    #wait for a bit
    time.sleep(1)

    from queue import Empty
    try:
        trace = oscilloscope_traces_queue.get_nowait()
        print(trace)
    except Empty:
        pass

    #stop instrument
    pipe_experiment_instrument_e.send({'command':'stop'})
    response = pipe_experiment_instrument_e.recv()
    print(response)

    #stop analyser
    pipe_experiment_analyser_e.send({'command':'stop'})
    response = pipe_experiment_analyser_e.recv()
    print(response)

    #quit analyser
    pipe_experiment_analyser_e.send({'command':'quit'})
    response = pipe_experiment_analyser_e.recv()
    print(response)

    #quit instrument
    pipe_experiment_instrument_e.send({'command':'quit'})
    response = pipe_experiment_instrument_e.recv()
    print(response)

    trace_analyser.join()
    instrument.join()

    # inspect event data output
    from pandas import DataFrame
    shm_events = shared_memory.SharedMemory(name=events_cache_shm.name)
    with events_cache_lock:
        events_cache = np.ndarray((max_events_in_cache, n_channels_per_event), dtype=np.int64, buffer=shm_events.buf)
    with index_head_events_cache.get_lock():
        events_head = index_head_events_cache.value  # not used - events cache is not cycled
    with index_tail_events_cache.get_lock():
        events_tail = index_tail_events_cache.value
    print([events_head, events_tail])
    with events_cache_lock:
        events_df = DataFrame(data=events_cache[events_head:events_tail], columns=event_channels_pnn)
        print(events_df.head(5))
        #events_df.to_csv('/home/ssr/Downloads/events.csv', index=False)

    traces_cache_shm.close()
    events_cache_shm.close()
    traces_cache_shm.unlink()
    events_cache_shm.unlink()
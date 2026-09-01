
'''
instrument communicator

Process to manage communications with instrument
Can use various instrument drivers (Cytkit, Picoscope, Dummy Instrument)
Method "run" is run as a process, listens for commands from controller: [connect, start, stop, set, quit]
Method "transfer" is started as thread when start command is received, repeadedly transfers traces to traces_cache

Example workflow:
    Connects to instrument
    Configures instrument
    Listens for start event
    Listens for stop event
    Applies cache condition
    Sets sample pump rate
    Sets sheath pump pressure
    Sets gains
    Sets laser

test instrument data generator

Shared memory: cache, head, tail, instrument settings
'''

import multiprocessing as mp
from multiprocessing import shared_memory, Lock
import threading
import numpy as np
import time
import warnings

from honeychrome.settings import devices_boot_order, traces_cache_size, traces_cache_dtype, max_events_in_traces_cache, trace_n_points, transfer_target_repeat_time

debug = False

class Instrument(mp.Process):
    def __init__(self, use_dummy_instrument=False,
                 traces_cache_name=None,
                 traces_cache_lock=None,
                 index_head_traces_cache=None, index_tail_traces_cache=None,
                 pipe_connection=None, logging_queue=None):
        super().__init__()

        # Route standard logs through the queue
        import logging.handlers
        qh = logging.handlers.QueueHandler(logging_queue)
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(qh)

        ### example
        # self.logger = logging.getLogger(__name__)
        # self.logger.info("************Worker logging message back to main process")

        # device will be a connected instrument or dummy if no instrument found
        self.device = None

        # dummy instrument
        self.dummy_memory_head = 0
        self.max_events_in_memory_per_pop = 500
        self.dummy_instrument = None
        if use_dummy_instrument:
            self.use_dummy_instrument = True
        else:
            self.use_dummy_instrument = False

        self.pipe_connection = pipe_connection
        self.index_head_traces_cache = index_head_traces_cache
        self.index_tail_traces_cache = index_tail_traces_cache
        self.traces_cache_name = traces_cache_name
        self.traces_cache = None
        self.traces_cache_lock = traces_cache_lock
        self.max_events_in_traces_cache = max_events_in_traces_cache
        self.trace_n_points = trace_n_points
        self.stop_transfer = None
        self.thread = None


    def run(self):
        self.find_and_connect_to_instrument()

        # initialise the things that can't be pickled
        self.stop_transfer = threading.Event()
        # initialise transfer thread
        self.thread = threading.Thread(
            target=self.transfer,
            daemon=True
        )

        # Attach to existing shared memory
        shm = shared_memory.SharedMemory(name=self.traces_cache_name)
        with self.traces_cache_lock:
            self.traces_cache = np.ndarray((self.max_events_in_traces_cache * self.trace_n_points), dtype=traces_cache_dtype, buffer=shm.buf)

        # main loop waiting for commands from experiment control
        while True:
            try:
                incoming_from_experiment_control = self.pipe_connection.recv()
            except EOFError:
                print("Pipe closed by experiment control; shutting down gracefully.")
                break

            response_to_experiment_control = None
            if incoming_from_experiment_control['command'] == 'find_and_connect':
                response_to_experiment_control = self.find_and_connect_to_instrument()
            elif incoming_from_experiment_control['command'] == 'initialise':
                response_to_experiment_control = self.initialise_instrument()
            elif incoming_from_experiment_control['command'] == 'start_acquisition':
                response_to_experiment_control = self.start_acquisition()
            elif incoming_from_experiment_control['command'] == 'stop_acquisition':
                response_to_experiment_control = self.stop_acquisition()
            elif incoming_from_experiment_control['command'] == 'flush_sip':
                response_to_experiment_control = self.flush_sip()
            elif incoming_from_experiment_control['command'] == 'backflush_sip':
                response_to_experiment_control = self.backflush_sip()
            elif incoming_from_experiment_control['command'] == 'set_instrument_state':
                response_to_experiment_control = self.set_instrument_state(incoming_from_experiment_control['data'])
            elif incoming_from_experiment_control['command'] == 'get_instrument_state':
                response_to_experiment_control = self.get_instrument_state(incoming_from_experiment_control['data'])
            elif incoming_from_experiment_control['command'] == 'quit':
                response_to_experiment_control = {'source':'[Instrument driver]', 'status':'OK', 'message':' Quitting'}
                self.pipe_connection.send(response_to_experiment_control)
                break

            self.pipe_connection.send(response_to_experiment_control)

        while self.thread.is_alive():
            print('[Instrument driver] Waiting until transfer thread ends')
            time.sleep(0.25)

        self.disconnect_instrument()
        shm.close()
        print('[Instrument driver] Quit')


    def find_and_connect_to_instrument(self):
        for device_name in devices_boot_order:
            try:
                if device_name == 'cytkit':
                    from honeychrome.instrument_driver_components.cytkit_driver import CytkitDevice
                    device = CytkitDevice()

                elif device_name == 'pico5000':
                    from honeychrome.instrument_driver_components.pico5000_driver import Pico5000_Device
                    device = Pico5000_Device()

                else: # device_name == 'dummy_device':
                    from honeychrome.instrument_driver_components.dummy_driver import DummyDevice
                    device = DummyDevice()

                status, message = device.find_and_connect_to_device()
                self.device = device
                print(f'[Instrument driver] {device_name} connected')
                return {'source': '[Instrument driver]', 'status': status, 'message': message}

            except Exception as e:
                print(f'[Instrument driver] {device_name} not connected: {e}')

        return {'source': '[Instrument driver]', 'status': 'OK', 'message': 'No device connected'}

    def disconnect_instrument(self):
        self.device.disconnect()

    def initialise_instrument(self):
        status, message = self.device.initialise()
        return {'source': '[Instrument driver]', 'status': status, 'message': message}

    def start_acquisition(self):
        status, message = self.device.start_acquisition()
        """Start transfer in a separate thread"""
        self.thread = threading.Thread(
            target=self.transfer,
            daemon=True
        )
        self.thread.start()
        print('[Instrument driver] Acquisition started')
        return {'source': '[Instrument driver]', 'status': status, 'message': message}

    def stop_acquisition(self):
        status, message = self.device.stop_acquisition()
        self.stop_transfer.set()
        return {'source': '[Instrument driver]', 'status': status, 'message': message}

    def set_instrument_state(self, data):
        status, message = self.device.set_state(data)
        return {'source': '[Instrument driver]', 'status': status, 'message': message}

    def get_instrument_state(self, data):
        status, message = self.device.get_state(data)
        return {'source': '[Instrument driver]', 'status': status, 'message': message}

    def flush_sip(self):
        status, message = self.device.flush_sip()
        return {'source': '[Instrument driver]', 'status': status, 'message': message}

    def backflush_sip(self):
        status, message = self.device.backflush_sip()
        return {'source': '[Instrument driver]', 'status': status, 'message': message}

    def transfer(self):
        while True:
            start_time = time.perf_counter()

            blob_of_traces_as_array = self.device.read_out_traces()
            self.push_to_traces_cache(blob_of_traces_as_array)

            if self.stop_transfer.is_set():
                self.stop_transfer.clear()
                break

            # Calculate elapsed time and sleep precisely
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0., transfer_target_repeat_time - elapsed)
            time.sleep(sleep_time)

    def push_to_traces_cache(self, blob_np):
        n_traces_from_memory = len(blob_np) // self.trace_n_points
        with self.index_tail_traces_cache.get_lock():
            tail = self.index_tail_traces_cache.value
        with self.index_head_traces_cache.get_lock():
            head = self.index_head_traces_cache.value

        cache_new_tail = tail + n_traces_from_memory
        if cache_new_tail <= head + self.max_events_in_traces_cache:

            queue_begin_index = tail % self.max_events_in_traces_cache
            queue_end_index = cache_new_tail % self.max_events_in_traces_cache
            slice_begin = queue_begin_index * self.trace_n_points
            slice_end = queue_end_index * self.trace_n_points
            with self.traces_cache_lock:
                if queue_end_index >= queue_begin_index:
                    try:
                        self.traces_cache[slice_begin:slice_end] = blob_np #todo fix: this crashes in long acquisition
                    except ValueError as e:
                        warnings.warn(str(e))
                else:
                    # note number of events that didn't fit is queue_end_index
                    switch_index = (self.max_events_in_traces_cache - queue_begin_index) * self.trace_n_points
                    self.traces_cache[slice_begin:] = blob_np[:switch_index]  # put first events into back of queue array
                    self.traces_cache[:slice_end] = blob_np[switch_index:]  # put last events into front of queue array

            with self.index_tail_traces_cache.get_lock():
                self.index_tail_traces_cache.value = cache_new_tail

            if debug == True:
                print(f'[Instrument driver] pushed data to traces cache (head:{head}, tail:{cache_new_tail})')
        else:
            warnings.warn("[Instrument driver] Traces cache is full, data dropped")
            pass



if __name__ == '__main__':
    import logging

    mp.set_start_method("spawn")

    # Allocate shared memory block, plus head and tail indices
    traces_cache_shm = shared_memory.SharedMemory(create=True, size=np.zeros(traces_cache_size, dtype=traces_cache_dtype).nbytes)
    traces_cache_lock = Lock()
    index_head_traces_cache = mp.Value('i', 0)
    index_tail_traces_cache = mp.Value('i', 0)

    pipe_experiment_instrument_e, pipe_experiment_instrument_i = mp.Pipe()
    # logging queue
    logging_queue = mp.Queue()
    # Set up listener in the main process
    listener = logging.handlers.QueueListener(logging_queue, logging.StreamHandler(), respect_handler_level=True)
    listener.start()

    # start instrument dummy
    instrument = Instrument(
        use_dummy_instrument=True,
        traces_cache_name=traces_cache_shm.name,
        traces_cache_lock=traces_cache_lock,
        index_head_traces_cache=index_head_traces_cache,
        index_tail_traces_cache=index_tail_traces_cache,
        pipe_connection=pipe_experiment_instrument_i,
        logging_queue=logging_queue
    )
    instrument.start()

    pipe_experiment_instrument_e.send({'command':'find_and_connect'})
    response = pipe_experiment_instrument_e.recv()
    print(response)

    pipe_experiment_instrument_e.send({'command':'star_acquisition'})
    response = pipe_experiment_instrument_e.recv()
    print(response)

    pipe_experiment_instrument_e.send({'command':'set_instrument_state', 'data':'TODO insert settings update here'})
    response = pipe_experiment_instrument_e.recv()
    print(response)

    #wait for a bit
    time.sleep(1)

    pipe_experiment_instrument_e.send({'command':'stop_acquisition'})
    response = pipe_experiment_instrument_e.recv()
    print(response)

    #wait for a bit
    time.sleep(3)

    pipe_experiment_instrument_e.send({'command':'quit'})
    response = pipe_experiment_instrument_e.recv()
    print(response)

    instrument.join()
    listener.stop()

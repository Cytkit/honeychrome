import multiprocessing as mp
from pathlib import Path
from deepdiff import DeepDiff

from honeychrome.experiment_model import ExperimentModel

mp.set_start_method("spawn", force=True)
from honeychrome.controller import Controller

'''
1.
open saved experiment
open a sample
view a sample
calculate statistics
save statistics

-Calculates all histograms
-Applies unmixing
-Applies compensation
-Applies gates
-Calculates masks
-Calculates stats
'''

def test_open_experiment_view_sample_add_gates_add_plots():
    kc = Controller()
    base_directory = Path.home() / 'spectral_cytometry'
    experiment_name = base_directory / '20240620 Spectral Symposium-poor cell unmixed'
    experiment_path = experiment_name.with_suffix('.kit')
    kc.load_experiment(experiment_path)

    # # Auto generate spectral model on loaded experiment, unmix, add plots
    kc.regenerate_spectral_model()
    kc.refresh_spectral_process()

    # note every time tab is changed (or every time plots or transforms changed), calculate all histograms and statistics
    kc.set_mode('Unmixed Data')
    kc.initialise_data_for_cytometry_plots()

    # add gates
    kc.create_or_update_gate(gate_name='activated', gate_type='rectangle', gate_path=('root', 'Cells', 'Singlets'), gate_data={'pos':[0.2, 0.2], 'size':[0.5, 0.5]}, channel_x='Spark UV 387', channel_y='SSC-A')
    kc.create_or_update_gate(gate_name='quaddy', gate_type='quad', gate_path=('root', 'Cells', 'Singlets'), gate_data={'pos':[0.2, 0.2], 'size':[0.5, 0.5]}, channel_x='Spark UV 387', channel_y='BUV805')
    kc.create_or_update_gate(gate_name='sub++', gate_type='ellipse', gate_path=('root', 'Cells', 'Singlets', 'quaddy', 'Spark UV 387+ BUV805+'), gate_data={'pos':[0.2, 0.2], 'size':[0.2, 0.2], 'angle':30}, channel_x='eFluor 450', channel_y='BUV805')

    kc.calculate_lookup_tables()  # (re)create all lookup tabels

    print('Test gating unmixed', kc.unmixed_gating.get_gate_ids())
    print('Test gating unmixed', kc.unmixed_gating.get_gate_hierarchy())

    # add plots
    kc.data_for_cytometry_plots['plots'] += [
        {'type': 'hist2d', 'channel_x': 'Spark UV 387', 'channel_y': 'SSC-A', 'source_gate': 'Singlets', 'child_gates': ['activated']},
        {'type': 'hist2d', 'channel_x': 'Spark UV 387', 'channel_y': 'BUV805', 'source_gate': 'Singlets', 'child_gates': ['quaddy']},
        {'type': 'hist2d', 'channel_x': 'eFluor 450', 'channel_y': 'BUV805', 'source_gate': 'Spark UV 387+ BUV805+', 'child_gates': ['sub++']}
    ]

    kc.set_mode('Raw Data')
    kc.load_sample(list(kc.experiment.samples['all_samples'])[0])

    print('Test statistics raw', kc.data_for_cytometry_plots_raw['statistics'])
    kc.set_mode('Unmixed Data')
    print('Test statistics unmixed', kc.data_for_cytometry_plots_unmixed['statistics'])

    # note every time tab is changed (or every time plots or transforms changed), calculate all histograms and statistics
    kc.set_mode('Raw Data')
    kc.new_sample('test', 'single_stain_controls')

    experiment_name = base_directory / '20240620 Spectral Symposium-poor cell unmixed'
    experiment_path = experiment_name.with_suffix('.kit')
    kc.save_experiment(experiment_path) # this saves as template (without samples)

    #1b. test whether loaded experiment is the same as current experiment
    test_experiment = ExperimentModel()
    test_experiment.load(experiment_path)
    test_experiment.scan_sample_tree() # this reloads samples

    diff = DeepDiff(kc.experiment,test_experiment,ignore_order=True, exclude_paths='root.progress_indicator')
    test_experiment.save()

    assert not diff


#
# '''
# 2.
# new experiment
# create sample
# receive live data
# update thread, calculate hists and stats
# save sample
#
# Creates live sample and carries out live analysis
# -Receives start acquisition signal
# -Initialises histograms
# -Listens for new chunk, updates histogram, updates stats
# -Runs spectral model process
# -Runs unmixing
# '''
#
#
# def test_new_experiment_acquire_live_data():
#     # Allocate shared memory block, plus head and tail indices
#     traces_cache_shm = shared_memory.SharedMemory(create=True, size=np.zeros(traces_cache_size, dtype=dtype).nbytes)
#     traces_cache_lock = Lock()
#     index_head_traces_cache = mp.Value('i', 0)
#     index_tail_traces_cache = mp.Value('i', 0)
#
#     events_cache_shm = shared_memory.SharedMemory(create=True,
#                                                   size=np.zeros((max_events_in_cache, n_channels_per_event),
#                                                                 dtype=np.int64).nbytes)
#     events_cache_lock = Lock()
#     index_head_events_cache = mp.Value('i', 0)
#     index_tail_events_cache = mp.Value('i', 0)
#
#     # oscilloscope traces
#     oscilloscope_traces_queue = mp.Queue()
#     # command pipes
#     pipe_experiment_instrument_e, pipe_experiment_instrument_i = mp.Pipe()
#     pipe_experiment_analyser_e, pipe_experiment_analyser_a = mp.Pipe()
#
#     '''
#     Firstly, set up experiment controller
#     '''
#     kc = Controller(
#             events_cache_name=events_cache_shm.name,
#             events_cache_lock=events_cache_lock,
#             index_head_events_cache=index_head_events_cache,
#             index_tail_events_cache=index_tail_events_cache,
#             oscilloscope_traces_queue=oscilloscope_traces_queue,
#             pipe_connection_instrument=pipe_experiment_instrument_e,
#             pipe_connection_analyser=pipe_experiment_analyser_e)
#
#     base_directory = Path.home() / experiments_folder
#     experiment_name = base_directory / 'Test experiment from new'
#     experiment_path = experiment_name.with_suffix('.kit')
#     kc.new_experiment(experiment_path) # this autosaves the experiment
#     kc.set_mode('Raw Data')
#
#     '''
#     Then, set up instrument
#     '''
#     # start instrument dummy
#     from honeychrome.instrument_driver import Instrument
#
#     instrument = Instrument(use_dummy_instrument=True, traces_cache_name=traces_cache_shm.name,
#         traces_cache_lock=traces_cache_lock, index_head_traces_cache=index_head_traces_cache,
#         index_tail_traces_cache=index_tail_traces_cache, pipe_connection=pipe_experiment_instrument_i)
#     instrument.start()
#
#     '''
#     Then, set up analyst
#     '''
#     from honeychrome.trace_analyst import TraceAnalyser
#
#     trace_analyser = TraceAnalyser(traces_cache_name=traces_cache_shm.name, traces_cache_lock=traces_cache_lock,
#         index_head_traces_cache=index_head_traces_cache, index_tail_traces_cache=index_tail_traces_cache,
#         events_cache_name=events_cache_shm.name, events_cache_lock=events_cache_lock,
#         index_head_events_cache=index_head_events_cache, index_tail_events_cache=index_tail_events_cache,
#         oscilloscope_traces_queue=oscilloscope_traces_queue,
#         pipe_connection=pipe_experiment_analyser_a)
#     trace_analyser.start()
#
#     '''
#     Then, send commands and read data
#
#     connect instrument
#     start instrument
#     start analyser
#     stop analyser
#     set channels
#     start analyser again
#     stop instrument
#     stop analyser
#     quit analyser
#     quit instrument
#     '''
#     # connect instrument
#     kc.connect_instrument()
#
#     # create blank sample
#     kc.new_sample('A0 Label (Cells)', 'single_stain_controls') # this autosaves the empty sample and experiment again
#
#     # acquire data
#     kc.start_acquisition()
#     # wait for a bit
#     time.sleep(2)
#     kc.stop_acquisition()
#
#     # end processes, free memory
#     kc.quit_instrument_quit_analyser()
#     trace_analyser.join()
#     instrument.join()
#     traces_cache_shm.close()
#     events_cache_shm.close()
#     traces_cache_shm.unlink()
#     events_cache_shm.unlink()
#
#     assert True
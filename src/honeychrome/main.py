'''
Honeychrome - open source cytometry data acquisition and analysis software
'''

# from honeychrome.tools import DepthFinder # debug module imports here

import os
import sys
import threading
from pathlib import Path
import platform
import shutil

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
import multiprocessing as mp
from multiprocessing import shared_memory, Lock
import numpy as np

from honeychrome import __version__
from honeychrome.settings import experiments_folder, send_debug_data
from honeychrome.settings import q_settings as q_settings_app_config

current_file_path = Path(__file__).resolve()

'''
Instrument Communicator:
-Connects to instrument
-Configures instrument
-Listens for start event
-Listens for stop event
'''
from honeychrome.instrument_communicator import Instrument

'''
Trace Analyser:
-Listens for start event
-Consumes cached traces
-Calculates peak height, area, width (according to settings) and adds to events cache 
-Copies latest trace with peak measurements
-Signals when new events chunk is ready
'''
from honeychrome.trace_analyst import TraceAnalyser


'''
Controller:
-Initialises Experiment Model
-Creates new experiment
-Loads saved experiment
-Loads sample
-Creates live sample and carries out live analysis
-Sends and receives signals to GUI
'''
from honeychrome.controller import Controller

'''
View:
-Creates GUI widgets
-Serves and updates data from controller
--sample list
--settings
--plots
--histograms
--gates
--spectral model
--instrument control
--oscilloscope
'''
from honeychrome.view import View, logo_icon

import logging
import warnings
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

def setup_logging(log_file):
    """Set up logging to both console and file, and capture all output."""

    # Create logger
    logger = logging.getLogger()
    # logger = logging.getLogger(__name__)  # Only logs from this file
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Silence specific libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    # logging.getLogger('PIL').setLevel(logging.WARNING)  # Often useful to block PIL too

    # Redirect warnings module to logging
    logging.captureWarnings(True)
    warnings.simplefilter("default")  # ensure warnings fire

    # Capture uncaught exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        logger.error("Uncaught exception",
                     exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = handle_exception

    return logger


def configure_multiprocessing():
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller executable
        mp.freeze_support()  # Required for Windows
        # On Unix systems, also need:
        if os.name == 'posix':
            mp.set_start_method('spawn', force=True)

def init_sentry():
    try:

        sentry_logging = LoggingIntegration(
            level=logging.INFO,  # Send INFO and above as breadcrumbs (context)
            event_level=logging.ERROR  # Actually send ERROR and above as alerts
        )
        # Keep it anonymous with send_default_pii=False (i.e. all personal data are scrubbed, e.g. usernames, file names and locations, etc)
        sentry_sdk.init(dsn="https://c99858ad29b86e3c551978e9b72c7724@o4511334133334016.ingest.de.sentry.io/4511334139887696",
                        shutdown_timeout=1,
                        integrations=[sentry_logging],
                        release=f"honeychrome@{__version__}",
                        send_default_pii=False)
        sentry_sdk.set_tag("os_name", platform.system())
        sentry_sdk.set_tag("os_release", platform.release())
        sentry_sdk.set_tag("os_version", platform.version())


        sentry_sdk.metrics.count("honeychrome.launch", 1,
            attributes={"version": __version__, "platform": sys.platform})
    except Exception:
        pass  # Sentry must never crash or stall the app


def main():
    configure_multiprocessing()

    # create experiments directory and plugins directory
    experiment_path = Path.home() / experiments_folder
    experiment_path.mkdir(parents=True, exist_ok=True)
    plugins_path = experiment_path / 'plugins'
    plugins_path.mkdir(parents=True, exist_ok=True)

    # copy plugin templates if they do not already exist in experiments/plugins folder
    meipass = getattr(sys, '_MEIPASS', None)
    if not meipass:
        template_plugins_path = current_file_path.parent / "plugin_templates"
        for template in template_plugins_path.glob("*_tab.py"):
            destination = plugins_path / template.name
            if not destination.exists():
                shutil.copy2(template, destination)

    logger = setup_logging(Path.home() / experiments_folder / 'honeychrome.log')

    # # Log messages at different levels
    # logger.debug("Debug message")  # Only to file
    # logger.info("Info message")  # To both console and file
    # logger.warning("Warning message")
    # logger.error("Error message")

    # send debug data if setting is on
    if q_settings_app_config.value("send_debug_data", send_debug_data, type=bool):
        threading.Thread(target=init_sentry, daemon=True).start()

        # # Test sentry
        # try:
        #     division_by_zero = 1 / 0
        # except Exception as e:
        #     # Manually send the caught exception to Sentry
        #     sentry_sdk.capture_exception(e)

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

    '''
    start instrument driver
    '''
    instrument = Instrument(
        use_dummy_instrument=settings.use_dummy_instrument_retrieved,
        traces_cache_name=traces_cache_shm.name,
        traces_cache_lock=traces_cache_lock,
        index_head_traces_cache=index_head_traces_cache,
        index_tail_traces_cache=index_tail_traces_cache,
        pipe_connection=pipe_experiment_instrument_i
    )
    instrument.start()

    '''
    start trace analyser
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
    start controller
    '''
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
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)

    if sys.platform == 'win32':
        import ctypes
        myappid = f'honeychrome.cytometry.v{__version__}'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    # Use Fusion style (works consistently across platforms)
    app.setStyle("Fusion")
    # Ensure consistent rounding of fractional scaling factors
    os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"

    # app.setWindowIcon(QIcon(str(Path(__file__).resolve().parent / 'view_components' / 'assets' / 'cytkit_web_logo.ico')))
    app.setWindowIcon(QIcon(logo_icon))
    app.setDesktopFileName("honeychrome")

    # # debug space usage by highlighting
    # app.setStyleSheet("""
    #     QMainWindow {
    #         background-color: Maroon;
    #     }
    #     QWidget {
    #             /* This background covers the Padding and Content area */
    #             background-color: Indigo;
    #     }
    #     QWidget {
    #         border: 1px solid green;
    #     }
    #     QFrame, QGroupBox {
    #         background-color: Goldenrod;
    #         border: 1px solid blue;
    #     }
    #     QScrollBar {
    #         background: DarkOliveGreen; border: 1px solid red;
    #     }
    # """)

    view = View(
        controller=controller
    )
    controller.bus = view.bus # connect signals coming from controller

    '''
    start QT application
    '''
    logger.info('***** Started Honeychrome *****')
    exit_code = app.exec()

    # end processes, free memory
    logger.info('***** Quitting Honeychrome *****')
    controller.quit_instrument_quit_analyser()
    trace_analyser.join()
    instrument.join()
    traces_cache_shm.close()
    events_cache_shm.close()
    traces_cache_shm.unlink()
    events_cache_shm.unlink()

    sys.exit(exit_code)

if __name__ == '__main__':
    main()
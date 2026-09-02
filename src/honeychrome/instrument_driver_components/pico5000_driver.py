import ctypes
import numpy as np
from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import assert_pico_ok, PICO_STATUS_LOOKUP

from honeychrome.settings import adc_channels, magnitude_ceiling, traces_cache_dtype, n_channels_trace, n_time_points_in_event, transfer_target_repeat_time, threshold, FSC_sense, adc_scale_mv, nearly_floor_uint16

index_channel_a = adc_channels.index('FSC')
index_channel_b = adc_channels.index('B1')
inject_signal = True

# note timebases for ps5000aRunBlock command
# 3→8 ns
# 4→16 ns
# 5→24 ns

# Ranges and coupling modes for ps5000aSetChannel command
# PS5000A_10MV    ±10    mV
# PS5000A_20MV    ±20    mV
# PS5000A_50MV    ±50    mV
# PS5000A_100MV    ±100    mV
# PS5000A_200MV    ±200    mV
# PS5000A_500MV    ±500    mV
# PS5000A_1V    ±1    V
# PS5000A_2V    ±2    V
# PS5000A_5V    ±5    V
# PS5000A_10V    ±10    V
# PS5000A_20V    ±20    V
# PS5000A_50V    ±50    V
# type = 0 (AC)
# type = 1 (DC)
# ps5000aSetChannel(handle, channel, enabled, type, range, analogOffset)

# --- PICOSCOPE CONFIG ---
SAMPLE_INTERVAL_NS = 8  # Timebase 3 (31.25 MS/s)
TIMEBASE = 3  # 8ns at 14-bit
MAX_SEGMENTS = int(1000 * transfer_target_repeat_time)  # Number of pulses to capture per "Burst"
resolution = ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_14BIT"]
auto_trig_timeout = 10 #ms
A_range = 1000
B_range = 1000
A_range_text = '1V'
B_range_text = '1V'
trigger_channel = 0 # 0=A, 1=B
deltaT = SAMPLE_INTERVAL_NS * 1e-9
analysis_decimation = 64 # e.g. 1024 x 8ns --> 8 us, 64 --> 0.5 us
deltaT_analysis = deltaT * analysis_decimation
PRE_TRIGGER = 8000 # e.g. 4000 x 8ns --> 32 us
# POST_TRIGGER = 2000 # e.g. 6000, or total window 8000 --> 64 us
# TOTAL_WINDOW = PRE_TRIGGER + POST_TRIGGER
TOTAL_WINDOW = int(analysis_decimation * n_time_points_in_event * 1.1)
POST_TRIGGER = TOTAL_WINDOW - PRE_TRIGGER

status = {}
chandle = ctypes.c_int16()

# inject test signal with AWG
def generate_test_signal():
    AWG_SIZE = 16384
    # WIDTHS_US = [5, 10, 20, 40]
    # AMPS_MV = [-50, -100, -200]
    WIDTHS_US = [10]
    AMPS_MV = [-150, -200]
    DC_LEVEL_MV = 200

    # Total time for the whole buffer to represent (e.g., 1.2 ms)
    TOTAL_PERIOD_US = 1200

    x = np.linspace(0, TOTAL_PERIOD_US, AWG_SIZE)
    buffer_full = np.zeros(AWG_SIZE)
    buffer_full = buffer_full + DC_LEVEL_MV / 200.0

    # Generate all 12 combinations
    current_pos_us = 50  # Start with a 50us offset
    spacing_us = TOTAL_PERIOD_US / (len(WIDTHS_US) + len(AMPS_MV) + 1)   # Space between peaks

    for w in WIDTHS_US:
        for a in AMPS_MV:
            # a is in mV. Normalize to 0.0 - 1.0 (Assume 2V AWG Range)
            norm_amp = a / 200.0  # Scaling relative to the 200mV max requested

            # Sigma for Gaussian: FWHM = 2.355 * sigma
            sigma = w / 2.355

            # Add peak to buffer
            peak = norm_amp * np.exp(-((x - current_pos_us) ** 2) / (2 * sigma ** 2))
            buffer_full += peak

            current_pos_us += spacing_us + w

    # Final scaling to 16-bit signed Int for the DAC
    # We scale so that 1.0 in our buffer equals the pkToPk voltage in SetSigGen
    awg_buffer = (np.clip(buffer_full, -1, 1) * 32767).astype(np.int16)

    # Convert frequency to Delta Phase (Helper logic)
    # Note: 5442B AWG update rate is 200 MHz
    awg_update_rate = 200_000_000
    frequency = 1.0 / (TOTAL_PERIOD_US * 1e-2)
    delta_phase = int((frequency * AWG_SIZE * 4294967296.0) / awg_update_rate)
    return awg_buffer, delta_phase


from scipy import signal
sos_butter = signal.butter(4, 1.25e6, fs=1e9 / SAMPLE_INTERVAL_NS, output='sos')
def filter_and_decimate(trace):
    trace = signal.sosfiltfilt(sos_butter, trace)
    trace = trace[::analysis_decimation]
    trace = trace[:n_time_points_in_event]

    return trace


class Pico5000_Device:
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
        self.buffer_list_a = []
        self.buffer_list_b = []
        self.name = 'Pico5000'

    def find_and_connect_to_device(self):
        # 1. Open and Power
        res = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, resolution)
        error_name = PICO_STATUS_LOOKUP.get(res, "UNKNOWN")

        if res == 0:
            print(f"Connected! Handle: {chandle.value}")
        elif "POWER" in error_name or res == 284:
            status["changePower"] = ps.ps5000aChangePowerSource(chandle, res)
            assert_pico_ok(status["changePower"])
            print("Power switched to USB.")
        elif res == 3:  # PICO_NOT_FOUND
            raise RuntimeError('No Picoscope found')
        else:
            assert_pico_ok(res)

        # 2. Setup Channels & Trigger and Memory
        # Split the scope's internal RAM into 1000 slots
        nMaxSamples = ctypes.c_int32()
        status["memorySegments"] = ps.ps5000aMemorySegments(chandle, MAX_SEGMENTS, ctypes.byref(nMaxSamples))
        assert_pico_ok(status["memorySegments"])

        # Set number of captures to perform per RunBlock call
        status["setNoCaptures"] = ps.ps5000aSetNoOfCaptures(chandle, MAX_SEGMENTS)
        assert_pico_ok(status["setNoCaptures"])

        range_A = ps.PS5000A_RANGE[f"PS5000A_{A_range_text}"]
        range_B = ps.PS5000A_RANGE[f"PS5000A_{B_range_text}"]
        ps.ps5000aSetChannel(chandle, 0, 1, 0, range_A, 0.0)
        ps.ps5000aSetChannel(chandle, 1, 1, 1, range_B, 0.0)

        # 0 = Channel A, 1 = Channel B
        # Bandwidth Limit: 0 = Full, 1 = 20 MHz
        status["bwA"] = ps.ps5000aSetBandwidthFilter(chandle, 0, 1)
        status["bwB"] = ps.ps5000aSetBandwidthFilter(chandle, 1, 1)
        assert_pico_ok(status["bwA"])
        assert_pico_ok(status["bwB"])

        self.max_adc = ctypes.c_int16()
        ps.ps5000aMaximumValue(chandle, ctypes.byref(self.max_adc))

        # AUTO TRIGGER: Ch A (0), 100mV Threshold, Rising (0), Auto-trigger (400ms)
        threshold_adc = int((threshold / A_range) * self.max_adc.value)
        if trigger_channel == 0:
            threshold_adc *= FSC_sense
        ps.ps5000aSetSimpleTrigger(chandle, 1, trigger_channel, threshold_adc, 1, 0, auto_trig_timeout)

        # Rapid Transfer Buffers
        # We create a list of buffers (one for each segment)
        # We use a list of ctypes arrays to hold the raw ADC data
        for i in range(MAX_SEGMENTS):
            # Create buffers for this specific segment
            buf_a = (ctypes.c_int16 * TOTAL_WINDOW)()
            buf_b = (ctypes.c_int16 * TOTAL_WINDOW)()

            # Tell the driver where to put data for segment 'i'
            # Channel A is 0, Channel B is 1
            ps.ps5000aSetDataBuffer(chandle, 0, ctypes.byref(buf_a), TOTAL_WINDOW, i, 0)
            ps.ps5000aSetDataBuffer(chandle, 1, ctypes.byref(buf_b), TOTAL_WINDOW, i, 0)

            self.buffer_list_a.append(buf_a)
            self.buffer_list_b.append(buf_b)

        self.initialised = False
        return 'OK', 'Picoscope 5000-series instrument connected'


    def disconnect(self):
        ps.ps5000aStop(chandle)
        ps.ps5000aCloseUnit(chandle)

    def initialise(self):
        # start arb sig generator
        if inject_signal:
            awg_buffer, delta_phase = generate_test_signal()
            status["setSigGen"] = ps.ps5000aSetSigGenArbitrary(chandle, 0,  # Offset (uV)
                400000,  # PkToPk (uV) -> 400mV (covers our 200mV peaks + headroom)
                delta_phase,  # startDeltaPhase
                delta_phase,  # stopDeltaPhase
                0, 0, awg_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)), len(awg_buffer), 0, 0, 0, 0, 0, 0, 0, 0)

        if not self.initialised:
            self.initialised = True
            return 'OK', 'Picoscope 5000-series initialised'
        else:
            self.initialised = False
            return 'OK', 'Picoscope 5000-series on standby'


    def start_acquisition(self):
        return 'OK', 'Picoscope 5000-series started acquisition'

    def stop_acquisition(self):
        return 'OK', 'Picoscope 5000-series stopped acquisition'

    def set_state(self, dict_of_parameters):
        return 'OK', 'Picoscope 5000-series parameters are set elsewhere'

    def get_state(self, list_of_parameters):
        return 'OK', {parameter:None for parameter in list_of_parameters}

    def flush_sip(self):
        return 'OK', 'Picoscope 5000-series doesn''t provide flush'

    def backflush_sip(self):
        return 'OK', 'Picoscope 5000-series doesn''t provide backflush'

    def read_out_traces(self):
        # Start capturing MAX_SEGMENTS
        ps.ps5000aRunBlock(chandle, PRE_TRIGGER, POST_TRIGGER, TIMEBASE, None, 0, None, None)

        ready = ctypes.c_int16(0)
        while ready.value == 0:
            ps.ps5000aIsReady(chandle, ctypes.byref(ready))

        # Bulk Transfer
        overflow = (ctypes.c_int16 * MAX_SEGMENTS)()
        ps.ps5000aGetValuesBulk(chandle, ctypes.byref(ctypes.c_int32(TOTAL_WINDOW)), 0, MAX_SEGMENTS - 1, 1, 0, ctypes.byref(overflow))

        # Get traces
        # Convert to Numpy Arrays (float32)
        # Using np.frombuffer is the fastest way to cast ctypes to numpy
        traces_a = np.array([np.frombuffer(b, dtype=np.int16) for b in self.buffer_list_a], dtype=np.float32) * A_range / self.max_adc.value * adc_scale_mv
        traces_b = np.array([np.frombuffer(b, dtype=np.int16) for b in self.buffer_list_b], dtype=np.float32) * B_range / self.max_adc.value * adc_scale_mv

        N = len(traces_a)
        traces = np.zeros((N, n_channels_trace, n_time_points_in_event), dtype=np.uint16)
        for n in range(N):
            traces[n, index_channel_a, :] = FSC_sense * filter_and_decimate(traces_a[n]) + nearly_floor_uint16
            traces[n, index_channel_b, :] = filter_and_decimate(traces_b[n]) + nearly_floor_uint16

        blob_of_traces_as_array = traces.reshape(-1)
        return blob_of_traces_as_array

if __name__ == '__main__':
    awg_buffer, delta_phase = generate_test_signal()

    pico_device = Pico5000_Device()
    pico_device.connect_to_device()
    pico_device.start_acquisition()
    blob = pico_device.read_out_traces()
    traces = blob.reshape((-1, n_channels_trace, n_time_points_in_event))

    pico_device.disconnect()

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(3)
    ax[0].plot(awg_buffer)
    ax[1].plot(traces[:,index_channel_a,:].T)
    ax[2].plot(traces[:,index_channel_b,:].T)
    plt.show()


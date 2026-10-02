'''
This is the default configuration for the instrument driver
'''

# hardware settings
pump_max = 255 # 0..255 sheath pump allowed PWM range in control loop
fan_max = 255 # 0..255 fan allowed PWM range in control loop
control_loop_interval = 2 # s
min_speed = 100 # guard against speed 0 --> speed 100 is 1e9/10 clocks, i.e. 1e7, or 0.1 s at 100 MHz
max_speed = 20000

# FGPA and communication settings
operation_write = b'\x01'
operation_read = b'\x02'
dummy_bytes = b'\x00\x00'
memory_start_address = 0
memory_end_address = 1_000_000

def lookup_address(register):
    if type(register) == str:
        return registers_map[register].to_bytes(2, byteorder='big')
    elif type(register) == int and 0 <= register <= 0xFFFF:
        return register.to_bytes(2, byteorder='big')
    else:
        raise TypeError

# registers_map = {
#     'RESERVED':	0x0000,   #
#     'ID_WORD':	0x0001,   #	ID
#     'VERSION_A':	0x0002,   #
#     'VERSION_B':	0x0003,   #
#     'VERSION_C':	0x0004,   #
#     'TIMESTAMP_A':	0x0005,   #
#     'TIMESTAMP_B':	0x0006,   #
#     'TIMESTAMP_C':	0x0007,   #
#     'TIMESTAMP_D':	0x0008,   #
#     'LASER':	    0x0010,   #	Laser
#     'SMPMP_CTRL':	0x0020,   #	Sample
#     'SMPMP_SPEED':	0x0021,   #	Pump
#     'SMPMP_SPC':	0x0022,   #
#     'SMPMP_CPC_L':	0x0023,   #
#     'SMPMP_CPC_H':	0x0024,   #
#     'SHPMP_CTRL':	0x0030,   #	Sheath
#     'SHPMP_DUTY':	0x0031,   #	Pump
#     'SHPMP_FREQ':	0x0032,   #
#     'FAN_CTRL':	0x0040,   #	Fan
#     'FAN_DUTY':	0x0041,   #
#     'FAN_FREQ':	0x0042,   #
#     'FAN_TACHO':	0x0043,   #
#     'INT_MASK':	0x0050,   #	Interlock
#     'INT_INV':	0x0051,   #
#     'INT_STATE':	0x0052,   #
#     'INT_MISC':	0x0053,   #
#     'PRES_CTRL':	0x0060,   #	Pressure
#     'PRES_TXFR_SIZE':	0x0061,   #	Sensor
#     'PRES_CS_WAIT':	0x0062,   #
#     'PRES_DATA_L':	0x0063,   #
#     'PRES_DATA_H':	0x0064,   #
#     'DISP_CTRL':	0x0070,   #	Display
#     'DISP_TXFR_SIZE':	0x0071,   #
#     'DISP_DATA_L':	0x0073,   #
#     'DISP_DATA_H':	0x0074,   #
#     'I2CA_CTRL':	0x0080,   #	I2C A
#     'I2CA_ADDRESS':	0x0081,   #
#     'I2CA_READ_SIZE':	0x0082,   #
#     'I2CA_DATA_IN':	0x0083,   #
#     'I2CA_DATA_OUT':	0x0084,   #
#     'I2CA_IN_LEVEL':	0x0085,   #
#     'I2CA_OUT_LEVEL':	0x0086,   #
#     'I2CA_STATE':	0x0087,   #
#     'I2CB_CTRL':	0x0090,   #	I2C B
#     'I2CB_ADDRESS':	0x0091,   #
#     'I2CB_READ_SIZE':	0x0092,   #
#     'I2CB_DATA_IN':	0x0093,   #
#     'I2CB_DATA_OUT':	0x0094,   #
#     'I2CB_IN_LEVEL':	0x0095,   #
#     'I2CB_OUT_LEVEL':	0x0096,   #
#     'I2CB_STATE':	0x0097,   #
#     'ADC_SELECT':	0x00A0,   #	ADC Source Selection
#     'ADC_ENABLE':	0x00B0,   #	HW ADC Controls
#     'ADC_VIRT_ENABLE':	0x00C0,   #	Virtual ADC Controls
# }

registers_map = {
    'RESERVED':	0x0000,	#	0 | 0x0000 | Not decoded, reads 0x0000.
    'ID_WORD':	0x0001,	#	ID | 51966 | 0xCAFE | Fixed identification word, always reads 0xCAFE. Use to confirm host to FPGA register access.
    'VERSION_A':	0x0002,	#	0 | 0x0000 | Firmware version major (15:8) and minor (7:0). Build dependent (BuildVersion.sv, generated from BuildVersion.template).
    'VERSION_B':	0x0003,	#	0 | 0x0000 | Firmware version revision. Build dependent.
    'VERSION_C':	0x0004,	#	0 | 0x0000 | Firmware build number. Build dependent.
    'TIMESTAMP_A':	0x0005,	#	0 | 0x0000 | Build timestamp: year, binary (e.g. 2026 = 0x07EA). Build dependent.
    'TIMESTAMP_B':	0x0006,	#	0 | 0x0000 | Build timestamp: month 1-12 (15:8) and day 1-31 (7:0), binary. Build dependent.
    'TIMESTAMP_C':	0x0007,	#	0 | 0x0000 | Build timestamp: hour 0-23 (15:8) and minute 0-59 (7:0), binary. Build dependent.
    'TIMESTAMP_D':	0x0008,	#	0 | 0x0000 | Build timestamp: second 0-59, binary. Build dependent.
    'LASER':	0x0010,	#	Laser | 0 | 0x0000 | ENABLE: 1 = enable laser driver (LASDRV_ENABLE). The output is gated by the laser interlock (see INT_MISC 0x0053) and forced off during reset.
    'SMPMP_CTRL':	0x0020,	#	Sample | 4608 | 0x1200 | ENABLE: 1 = run the stepper. DIRECTION: 1 = step sequence forwards, 0 = backwards. POWER: coil current 0 = Off, 1 = 1/3, 2 = 2/3, 3 = Full (drives PCTL_SMPL_I0x / I1x, active low). Coil current and phase outputs are switched off while ENABLE = 0 or the sample pump interlock is active. RAMP_ENABLE: 1 = step rate ramps towards SMPMP_SPEED at the rate set by SMPMP_SPC / SMPMP_CPC; 0 = SMPMP_SPEED applied immediately. While RAMP_ENABLE = 1 and ENABLE = 0 the ramped rate is held at 0, so the pump ramps up from rest when enabled. RAMP_COMPLETE (RO): 1 = ramped rate has reached SMPMP_SPEED.
    'SMPMP_SPEED':	0x0021,	#	Pump | 512 | 0x0200 | Target step rate in 0.1 Hz units. Default 0x0200 = 51.2 Hz. Do not write 0 (step period is calculated as 1e9 / SPEED clocks).
    'SMPMP_SPC':	0x0022,	#	1 | 0x0001 | Ramp step: change in SMPMP_SPEED units applied each ramp cycle. The final step is clamped so the ramp lands exactly on the target. 0 = ramp does not move.
    'SMPMP_CPC_L':	0x0023,	#	10000 | 0x2710 | Ramp cycle period in 100 MHz clocks, bits 15:0. Default 0x0000_2710 = 10000 clocks = 100 µs.
    'SMPMP_CPC_H':	0x0024,	#	0 | 0x0000 | Ramp cycle period in 100 MHz clocks, bits 31:16.
    'SHPMP_CTRL':	0x0030,	#	Sheath | 0 | 0x0000 | ENABLE: 1 = enable sheath pump PWM output (PCTL_SHEATH_PWM). Output is low when disabled or while the sheath pump interlock is active.
    'SHPMP_DUTY':	0x0031,	#	Pump | 128 | 0x0080 | PWM duty cycle = DUTY_CYCLE / 255 (0x00 = 0 %, 0xFF = 100 %). Default 0x80 = 50 %.
    'SHPMP_FREQ':	0x0032,	#	256 | 0x0100 | PWM frequency in Hz. Default 0x0100 = 256 Hz. Do not write 0.
    'FAN_CTRL':	0x0040,	#	Fan | 0 | 0x0000 | ENABLE: 1 = enable fan PWM. The FAN_PWM pin is active low (pin is high when disabled or while the fan interlock is active).
    'FAN_DUTY':	0x0041,	#	128 | 0x0080 | PWM duty cycle = DUTY_CYCLE / 255 (0x00 = 0 %, 0xFF = 100 %). Default 0x80 = 50 %.
    'FAN_FREQ':	0x0042,	#	256 | 0x0100 | PWM frequency in Hz. Default 0x0100 = 256 Hz. Do not write 0.
    'FAN_TACHO':	0x0043,	#	0 | 0x0000 | Fan speed in RPM (5 s measurement interval). Tacho measurement is not yet implemented in the FPGA, so this currently always reads 0.
    'INT_MASK':	0x0050,	#	Interlock | 1 | 0x0001 | Interlock input enable, bit n = input n (1 = input used). Only input 0 (INTERLOCK pin) is implemented.
    'INT_INV':	0x0051,	#	0 | 0x0000 | Interlock input polarity, bit n = input n (0 = active high, 1 = active low).
    'INT_STATE':	0x0052,	#	0 | 0x0000 | Current interlock input state after inversion and masking, bit n = input n (1 = active). Inputs pass through a 3-stage synchroniser.
    'INT_MISC':	0x0053,	#	15 | 0x000F | STOP_x: 1 = the interlock acts on that output (0 = Laser, 1 = Sample pump, 2 = Sheath pump, 3 = Fan). FORCE: 1 = force the interlock active on all selected outputs regardless of the inputs. An output is interlocked when its STOP bit is set and either FORCE = 1 or any INT_STATE bit is set.
    'PRES_CTRL':	0x0060,	#	Pressure | 0 | 0x0000 | START (SC): write 1 to run an SPI transfer: CS asserted, TRANSFER_SIZE bits clocked at 1.5 MHz, then CS held for PRES_CS_WAIT before release. COMPLETE (RO): 1 = transfer finished; cleared when the next transfer starts.
    'PRES_TXFR_SIZE':	0x0061,	#	Sensor | 8 | 0x0008 | Transfer length in bits, 1-32. Default 8.
    'PRES_CS_WAIT':	0x0062,	#	5 | 0x0005 | Delay from the last SPI bit to CS release, in units of 65536 clocks (655.36 µs). Default 5 = 3.28 ms.
    'PRES_DATA_L':	0x0063,	#	0 | 0x0000 | Write: transmit data bits 15:0. Read: received data bits 15:0 (not the written value). Data is right aligned (the TRANSFER_SIZE LSBs are used) and shifted MSB first.
    'PRES_DATA_H':	0x0064,	#	0 | 0x0000 | Write: transmit data bits 31:16. Read: received data bits 31:16. See PRES_DATA_L.
    'DISP_CTRL':	0x0070,	#	Display | 0 | 0x0000 | START (SC): write 1 to send TRANSFER_SIZE bits to the display over SPI at 1.5 MHz (FP_KEY[1] = nCS, FP_KEY[2] = SCLK, FP_KEY[3] = MOSI). COMPLETE (RO): 1 = transfer finished. DS_STATE: level driven onto FP_KEY[4].
    'DISP_TXFR_SIZE':	0x0071,	#	8 | 0x0008 | Transfer length in bits, 1-32. Default 8.
    'DISP_DATA_L':	0x0072,	#	0 | 0x0000 | Write: transmit data bits 15:0 (right aligned, shifted MSB first). Read: received data bits 15:0 (no MISO is connected, so this reads 0).
    'DISP_DATA_H':	0x0073,	#	0 | 0x0000 | Write: transmit data bits 31:16. Read: received data bits 31:16 (reads 0).
    'I2CA_CTRL':	0x0080,	#	I2C A | 0 | 0x0000 | I2C bus A, 100 kbit/s. START (SC): begin a transaction: all bytes queued in the Data In FIFO are written to I2CA_ADDRESS, then if READ_SIZE > 0 a repeated start reads READ_SIZE bytes into the Data Out FIFO (if the Data In FIFO is empty the transaction is read only). FLUSH (SC): empty both data FIFOs. DONE (RO): 0 = busy, 1 = done. STATUS (RO): 0 = OK, 1 = address NACK, 2 = write data NACK.
    'I2CA_ADDRESS':	0x0081,	#	0 | 0x0000 | 10BIT_MODE: 0 = 7-bit addressing (address in bits 6:0), 1 = 10-bit addressing (bits 9:0).
    'I2CA_READ_SIZE':	0x0082,	#	0 | 0x0000 | Number of bytes to read after the write phase. 0 = write only.
    'I2CA_DATA_IN':	0x0083,	#	0 | 0x0000 | Write: push one byte into the Data In (transmit) FIFO, 2048 bytes deep. Read: returns the last value written.
    'I2CA_DATA_OUT':	0x0084,	#	0 | 0x0000 | Read: pop the next received byte from the Data Out (receive) FIFO. Each read removes one byte.
    'I2CA_IN_LEVEL':	0x0085,	#	0 | 0x0000 | Number of bytes waiting in the Data In (transmit) FIFO.
    'I2CA_OUT_LEVEL':	0x0086,	#	0 | 0x0000 | Number of bytes available in the Data Out (receive) FIFO.
    'I2CA_STATE':	0x008F,	#	0 | 0x0000 | Debug only: current I2C master state machine state (0 = Idle ... 49 = Done).
    'I2CB_CTRL':	0x0090,	#	I2C B | 0 | 0x0000 | I2C bus B, 100 kbit/s. START (SC): begin a transaction: all bytes queued in the Data In FIFO are written to I2CB_ADDRESS, then if READ_SIZE > 0 a repeated start reads READ_SIZE bytes into the Data Out FIFO (if the Data In FIFO is empty the transaction is read only). FLUSH (SC): empty both data FIFOs. DONE (RO): 0 = busy, 1 = done. STATUS (RO): 0 = OK, 1 = address NACK, 2 = write data NACK.
    'I2CB_ADDRESS':	0x0091,	#	0 | 0x0000 | 10BIT_MODE: 0 = 7-bit addressing (address in bits 6:0), 1 = 10-bit addressing (bits 9:0).
    'I2CB_READ_SIZE':	0x0092,	#	0 | 0x0000 | Number of bytes to read after the write phase. 0 = write only.
    'I2CB_DATA_IN':	0x0093,	#	0 | 0x0000 | Write: push one byte into the Data In (transmit) FIFO, 2048 bytes deep. Read: returns the last value written.
    'I2CB_DATA_OUT':	0x0094,	#	0 | 0x0000 | Read: pop the next received byte from the Data Out (receive) FIFO. Each read removes one byte.
    'I2CB_IN_LEVEL':	0x0095,	#	0 | 0x0000 | Number of bytes waiting in the Data In (transmit) FIFO.
    'I2CB_OUT_LEVEL':	0x0096,	#	0 | 0x0000 | Number of bytes available in the Data Out (receive) FIFO.
    'I2CB_STATE':	0x009F,	#	0 | 0x0000 | Debug only: current I2C master state machine state (0 = Idle ... 49 = Done).
    'SYSCLK_CTRL':	0x00A0,	#	System | 0 | 0x0000 | ENABLE: 1 = system clock counts in 1 µs ticks. RESET (SC): clear the counter to 0. SNAPSHOT (SC): latch the counter into SYSCLK_SNAP_0..3. The live counter timestamps each capture record in Bulk Storage.
    'SYSCLK_SNAP_0':	0x00A1,	#	Clock | 0 | 0x0000 | System clock snapshot bits 15:0 (µs).
    'SYSCLK_SNAP_1':	0x00A2,	#	0 | 0x0000 | System clock snapshot bits 31:16.
    'SYSCLK_SNAP_2':	0x00A3,	#	0 | 0x0000 | System clock snapshot bits 47:32.
    'SYSCLK_SNAP_3':	0x00A4,	#	0 | 0x0000 | System clock snapshot bits 63:48.
    'ADC_SELECT':	0x0100,	#	ADC | 0 | 0x0000 | Sample source per channel, bit n = channel n: 0 = hardware ADC, 1 = virtual ADC.
    'ADC_CH0_VALUE':	0x0110,	#	ADC | 0 | 0x0000 | Most recent 14-bit sample for channel 0 (after source selection, before offset correction).
    'ADC_CH1_VALUE':	0x0111,	#	Values | 0 | 0x0000 | Most recent 14-bit sample for channel 1 (after source selection, before offset correction).
    'ADC_CH2_VALUE':	0x0112,	#	0 | 0x0000 | Most recent 14-bit sample for channel 2 (after source selection, before offset correction).
    'ADC_CH3_VALUE':	0x0113,	#	0 | 0x0000 | Most recent 14-bit sample for channel 3 (after source selection, before offset correction).
    'ADC_CH4_VALUE':	0x0114,	#	0 | 0x0000 | Most recent 14-bit sample for channel 4 (after source selection, before offset correction).
    'ADC_CH5_VALUE':	0x0115,	#	0 | 0x0000 | Most recent 14-bit sample for channel 5 (after source selection, before offset correction).
    'ADC_CH6_VALUE':	0x0116,	#	0 | 0x0000 | Most recent 14-bit sample for channel 6 (after source selection, before offset correction).
    'ADC_CH7_VALUE':	0x0117,	#	0 | 0x0000 | Most recent 14-bit sample for channel 7 (after source selection, before offset correction).
    'ADC_CH8_VALUE':	0x0118,	#	0 | 0x0000 | Most recent 14-bit sample for channel 8 (after source selection, before offset correction).
    'ADC_CH9_VALUE':	0x0119,	#	0 | 0x0000 | Most recent 14-bit sample for channel 9 (after source selection, before offset correction).
    'ADC_CH10_VALUE':	0x011A,	#	0 | 0x0000 | Most recent 14-bit sample for channel 10 (after source selection, before offset correction).
    'ADC_CH11_VALUE':	0x011B,	#	0 | 0x0000 | Most recent 14-bit sample for channel 11 (after source selection, before offset correction).
    'ADC_CH12_VALUE':	0x011C,	#	0 | 0x0000 | Most recent 14-bit sample for channel 12 (after source selection, before offset correction).
    'ADC_CH13_VALUE':	0x011D,	#	0 | 0x0000 | Most recent 14-bit sample for channel 13 (after source selection, before offset correction).
    'ADC_CH14_VALUE':	0x011E,	#	0 | 0x0000 | Most recent 14-bit sample for channel 14 (after source selection, before offset correction).
    'ADC_CH15_VALUE':	0x011F,	#	0 | 0x0000 | Most recent 14-bit sample for channel 15 (after source selection, before offset correction).
    'ADC_ENABLE':	0x0120,	#	HW | 0 | 0x0000 | Hardware ADC enable, bit n = ADC n (1 = acquiring, 2 MSPS nominal).
    'ADC_VIRT_TRIG':	0x0130,	#	Virtual | 0 | 0x0000 | Write 1 to bit n to trigger virtual ADC channel n (same as the TRIGGER bit in ADC_VIRTn_WAV).
    'ADC_VIRT0_WAV':	0x0140,	#	0 | 0x0000 | Virtual ADC channel 0. ENABLE: 1 = generator enabled. INVERT: 1 = invert the output. CONTINUOUS: 0 = one waveform per trigger, 1 = repeat continuously after the first trigger. TRIGGER (SC): start the waveform after TRIGGER_DELAY. WAVEFORM: 0 = sine, 1 = impulse, other values = no output.
    'ADC_VIRT0_OFFS':	0x0141,	#	16 | 0x0010 | Baseline level added to the waveform, in 14-bit ADC codes.
    'ADC_VIRT0_AMP':	0x0142,	#	14336 | 0x3800 | Waveform scale: output = ROM value x AMPLITUDE / 16384. Default 0x3800 = 0.875.
    'ADC_VIRT0_NOISE':	0x0143,	#	5 | 0x0005 | Peak-to-peak amplitude of the pseudo-random noise added to the waveform, in ADC codes (centred on the signal).
    'ADC_VIRT0_DELAY':	0x0144,	#	0 | 0x0000 | Delay from trigger to waveform start, in µs.
    'ADC_VIRT1_WAV':	0x0148,	#	0 | 0x0000 | Virtual ADC channel 1. ENABLE: 1 = generator enabled. INVERT: 1 = invert the output. CONTINUOUS: 0 = one waveform per trigger, 1 = repeat continuously after the first trigger. TRIGGER (SC): start the waveform after TRIGGER_DELAY. WAVEFORM: 0 = sine, 1 = impulse, other values = no output.
    'ADC_VIRT1_OFFS':	0x0149,	#	16 | 0x0010 | Baseline level added to the waveform, in 14-bit ADC codes.
    'ADC_VIRT1_AMP':	0x014A,	#	14336 | 0x3800 | Waveform scale: output = ROM value x AMPLITUDE / 16384. Default 0x3800 = 0.875.
    'ADC_VIRT1_NOISE':	0x014B,	#	5 | 0x0005 | Peak-to-peak amplitude of the pseudo-random noise added to the waveform, in ADC codes (centred on the signal).
    'ADC_VIRT1_DELAY':	0x014C,	#	0 | 0x0000 | Delay from trigger to waveform start, in µs.
    'ADC_VIRT2_WAV':	0x0150,	#	0 | 0x0000 | Virtual ADC channel 2. ENABLE: 1 = generator enabled. INVERT: 1 = invert the output. CONTINUOUS: 0 = one waveform per trigger, 1 = repeat continuously after the first trigger. TRIGGER (SC): start the waveform after TRIGGER_DELAY. WAVEFORM: 0 = sine, 1 = impulse, other values = no output.
    'ADC_VIRT2_OFFS':	0x0151,	#	16 | 0x0010 | Baseline level added to the waveform, in 14-bit ADC codes.
    'ADC_VIRT2_AMP':	0x0152,	#	14336 | 0x3800 | Waveform scale: output = ROM value x AMPLITUDE / 16384. Default 0x3800 = 0.875.
    'ADC_VIRT2_NOISE':	0x0153,	#	5 | 0x0005 | Peak-to-peak amplitude of the pseudo-random noise added to the waveform, in ADC codes (centred on the signal).
    'ADC_VIRT2_DELAY':	0x0154,	#	0 | 0x0000 | Delay from trigger to waveform start, in µs.
    'ADC_VIRT3_WAV':	0x0158,	#	0 | 0x0000 | Virtual ADC channel 3. ENABLE: 1 = generator enabled. INVERT: 1 = invert the output. CONTINUOUS: 0 = one waveform per trigger, 1 = repeat continuously after the first trigger. TRIGGER (SC): start the waveform after TRIGGER_DELAY. WAVEFORM: 0 = sine, 1 = impulse, other values = no output.
    'ADC_VIRT3_OFFS':	0x0159,	#	16 | 0x0010 | Baseline level added to the waveform, in 14-bit ADC codes.
    'ADC_VIRT3_AMP':	0x015A,	#	14336 | 0x3800 | Waveform scale: output = ROM value x AMPLITUDE / 16384. Default 0x3800 = 0.875.
    'ADC_VIRT3_NOISE':	0x015B,	#	5 | 0x0005 | Peak-to-peak amplitude of the pseudo-random noise added to the waveform, in ADC codes (centred on the signal).
    'ADC_VIRT3_DELAY':	0x015C,	#	0 | 0x0000 | Delay from trigger to waveform start, in µs.
    'ADC_VIRT4_WAV':	0x0160,	#	0 | 0x0000 | Virtual ADC channel 4. ENABLE: 1 = generator enabled. INVERT: 1 = invert the output. CONTINUOUS: 0 = one waveform per trigger, 1 = repeat continuously after the first trigger. TRIGGER (SC): start the waveform after TRIGGER_DELAY. WAVEFORM: 0 = sine, 1 = impulse, other values = no output.
    'ADC_VIRT4_OFFS':	0x0161,	#	16 | 0x0010 | Baseline level added to the waveform, in 14-bit ADC codes.
    'ADC_VIRT4_AMP':	0x0162,	#	14336 | 0x3800 | Waveform scale: output = ROM value x AMPLITUDE / 16384. Default 0x3800 = 0.875.
    'ADC_VIRT4_NOISE':	0x0163,	#	5 | 0x0005 | Peak-to-peak amplitude of the pseudo-random noise added to the waveform, in ADC codes (centred on the signal).
    'ADC_VIRT4_DELAY':	0x0164,	#	0 | 0x0000 | Delay from trigger to waveform start, in µs.
    'ADC_VIRT5_WAV':	0x0168,	#	0 | 0x0000 | Virtual ADC channel 5. ENABLE: 1 = generator enabled. INVERT: 1 = invert the output. CONTINUOUS: 0 = one waveform per trigger, 1 = repeat continuously after the first trigger. TRIGGER (SC): start the waveform after TRIGGER_DELAY. WAVEFORM: 0 = sine, 1 = impulse, other values = no output.
    'ADC_VIRT5_OFFS':	0x0169,	#	16 | 0x0010 | Baseline level added to the waveform, in 14-bit ADC codes.
    'ADC_VIRT5_AMP':	0x016A,	#	14336 | 0x3800 | Waveform scale: output = ROM value x AMPLITUDE / 16384. Default 0x3800 = 0.875.
    'ADC_VIRT5_NOISE':	0x016B,	#	5 | 0x0005 | Peak-to-peak amplitude of the pseudo-random noise added to the waveform, in ADC codes (centred on the signal).
    'ADC_VIRT5_DELAY':	0x016C,	#	0 | 0x0000 | Delay from trigger to waveform start, in µs.
    'ADC_VIRT6_WAV':	0x0170,	#	0 | 0x0000 | Virtual ADC channel 6. ENABLE: 1 = generator enabled. INVERT: 1 = invert the output. CONTINUOUS: 0 = one waveform per trigger, 1 = repeat continuously after the first trigger. TRIGGER (SC): start the waveform after TRIGGER_DELAY. WAVEFORM: 0 = sine, 1 = impulse, other values = no output.
    'ADC_VIRT6_OFFS':	0x0171,	#	16 | 0x0010 | Baseline level added to the waveform, in 14-bit ADC codes.
    'ADC_VIRT6_AMP':	0x0172,	#	14336 | 0x3800 | Waveform scale: output = ROM value x AMPLITUDE / 16384. Default 0x3800 = 0.875.
    'ADC_VIRT6_NOISE':	0x0173,	#	5 | 0x0005 | Peak-to-peak amplitude of the pseudo-random noise added to the waveform, in ADC codes (centred on the signal).
    'ADC_VIRT6_DELAY':	0x0174,	#	0 | 0x0000 | Delay from trigger to waveform start, in µs.
    'ADC_VIRT7_WAV':	0x0178,	#	0 | 0x0000 | Virtual ADC channel 7. ENABLE: 1 = generator enabled. INVERT: 1 = invert the output. CONTINUOUS: 0 = one waveform per trigger, 1 = repeat continuously after the first trigger. TRIGGER (SC): start the waveform after TRIGGER_DELAY. WAVEFORM: 0 = sine, 1 = impulse, other values = no output.
    'ADC_VIRT7_OFFS':	0x0179,	#	16 | 0x0010 | Baseline level added to the waveform, in 14-bit ADC codes.
    'ADC_VIRT7_AMP':	0x017A,	#	14336 | 0x3800 | Waveform scale: output = ROM value x AMPLITUDE / 16384. Default 0x3800 = 0.875.
    'ADC_VIRT7_NOISE':	0x017B,	#	5 | 0x0005 | Peak-to-peak amplitude of the pseudo-random noise added to the waveform, in ADC codes (centred on the signal).
    'ADC_VIRT7_DELAY':	0x017C,	#	0 | 0x0000 | Delay from trigger to waveform start, in µs.
    'ADC_VIRT8_WAV':	0x0180,	#	0 | 0x0000 | Virtual ADC channel 8. ENABLE: 1 = generator enabled. INVERT: 1 = invert the output. CONTINUOUS: 0 = one waveform per trigger, 1 = repeat continuously after the first trigger. TRIGGER (SC): start the waveform after TRIGGER_DELAY. WAVEFORM: 0 = sine, 1 = impulse, other values = no output.
    'ADC_VIRT8_OFFS':	0x0181,	#	16 | 0x0010 | Baseline level added to the waveform, in 14-bit ADC codes.
    'ADC_VIRT8_AMP':	0x0182,	#	14336 | 0x3800 | Waveform scale: output = ROM value x AMPLITUDE / 16384. Default 0x3800 = 0.875.
    'ADC_VIRT8_NOISE':	0x0183,	#	5 | 0x0005 | Peak-to-peak amplitude of the pseudo-random noise added to the waveform, in ADC codes (centred on the signal).
    'ADC_VIRT8_DELAY':	0x0184,	#	0 | 0x0000 | Delay from trigger to waveform start, in µs.
    'ADC_VIRT9_WAV':	0x0188,	#	0 | 0x0000 | Virtual ADC channel 9. ENABLE: 1 = generator enabled. INVERT: 1 = invert the output. CONTINUOUS: 0 = one waveform per trigger, 1 = repeat continuously after the first trigger. TRIGGER (SC): start the waveform after TRIGGER_DELAY. WAVEFORM: 0 = sine, 1 = impulse, other values = no output.
    'ADC_VIRT9_OFFS':	0x0189,	#	16 | 0x0010 | Baseline level added to the waveform, in 14-bit ADC codes.
    'ADC_VIRT9_AMP':	0x018A,	#	14336 | 0x3800 | Waveform scale: output = ROM value x AMPLITUDE / 16384. Default 0x3800 = 0.875.
    'ADC_VIRT9_NOISE':	0x018B,	#	5 | 0x0005 | Peak-to-peak amplitude of the pseudo-random noise added to the waveform, in ADC codes (centred on the signal).
    'ADC_VIRT9_DELAY':	0x018C,	#	0 | 0x0000 | Delay from trigger to waveform start, in µs.
    'ADC_VIRT10_WAV':	0x0190,	#	0 | 0x0000 | Virtual ADC channel 10. ENABLE: 1 = generator enabled. INVERT: 1 = invert the output. CONTINUOUS: 0 = one waveform per trigger, 1 = repeat continuously after the first trigger. TRIGGER (SC): start the waveform after TRIGGER_DELAY. WAVEFORM: 0 = sine, 1 = impulse, other values = no output.
    'ADC_VIRT10_OFFS':	0x0191,	#	16 | 0x0010 | Baseline level added to the waveform, in 14-bit ADC codes.
    'ADC_VIRT10_AMP':	0x0192,	#	14336 | 0x3800 | Waveform scale: output = ROM value x AMPLITUDE / 16384. Default 0x3800 = 0.875.
    'ADC_VIRT10_NOISE':	0x0193,	#	5 | 0x0005 | Peak-to-peak amplitude of the pseudo-random noise added to the waveform, in ADC codes (centred on the signal).
    'ADC_VIRT10_DELAY':	0x0194,	#	0 | 0x0000 | Delay from trigger to waveform start, in µs.
    'ADC_VIRT11_WAV':	0x0198,	#	0 | 0x0000 | Virtual ADC channel 11. ENABLE: 1 = generator enabled. INVERT: 1 = invert the output. CONTINUOUS: 0 = one waveform per trigger, 1 = repeat continuously after the first trigger. TRIGGER (SC): start the waveform after TRIGGER_DELAY. WAVEFORM: 0 = sine, 1 = impulse, other values = no output.
    'ADC_VIRT11_OFFS':	0x0199,	#	16 | 0x0010 | Baseline level added to the waveform, in 14-bit ADC codes.
    'ADC_VIRT11_AMP':	0x019A,	#	14336 | 0x3800 | Waveform scale: output = ROM value x AMPLITUDE / 16384. Default 0x3800 = 0.875.
    'ADC_VIRT11_NOISE':	0x019B,	#	5 | 0x0005 | Peak-to-peak amplitude of the pseudo-random noise added to the waveform, in ADC codes (centred on the signal).
    'ADC_VIRT11_DELAY':	0x019C,	#	0 | 0x0000 | Delay from trigger to waveform start, in µs.
    'ADC_VIRT12_WAV':	0x01A0,	#	0 | 0x0000 | Virtual ADC channel 12. ENABLE: 1 = generator enabled. INVERT: 1 = invert the output. CONTINUOUS: 0 = one waveform per trigger, 1 = repeat continuously after the first trigger. TRIGGER (SC): start the waveform after TRIGGER_DELAY. WAVEFORM: 0 = sine, 1 = impulse, other values = no output.
    'ADC_VIRT12_OFFS':	0x01A1,	#	16 | 0x0010 | Baseline level added to the waveform, in 14-bit ADC codes.
    'ADC_VIRT12_AMP':	0x01A2,	#	14336 | 0x3800 | Waveform scale: output = ROM value x AMPLITUDE / 16384. Default 0x3800 = 0.875.
    'ADC_VIRT12_NOISE':	0x01A3,	#	5 | 0x0005 | Peak-to-peak amplitude of the pseudo-random noise added to the waveform, in ADC codes (centred on the signal).
    'ADC_VIRT12_DELAY':	0x01A4,	#	0 | 0x0000 | Delay from trigger to waveform start, in µs.
    'ADC_VIRT13_WAV':	0x01A8,	#	0 | 0x0000 | Virtual ADC channel 13. ENABLE: 1 = generator enabled. INVERT: 1 = invert the output. CONTINUOUS: 0 = one waveform per trigger, 1 = repeat continuously after the first trigger. TRIGGER (SC): start the waveform after TRIGGER_DELAY. WAVEFORM: 0 = sine, 1 = impulse, other values = no output.
    'ADC_VIRT13_OFFS':	0x01A9,	#	16 | 0x0010 | Baseline level added to the waveform, in 14-bit ADC codes.
    'ADC_VIRT13_AMP':	0x01AA,	#	14336 | 0x3800 | Waveform scale: output = ROM value x AMPLITUDE / 16384. Default 0x3800 = 0.875.
    'ADC_VIRT13_NOISE':	0x01AB,	#	5 | 0x0005 | Peak-to-peak amplitude of the pseudo-random noise added to the waveform, in ADC codes (centred on the signal).
    'ADC_VIRT13_DELAY':	0x01AC,	#	0 | 0x0000 | Delay from trigger to waveform start, in µs.
    'ADC_VIRT14_WAV':	0x01B0,	#	0 | 0x0000 | Virtual ADC channel 14. ENABLE: 1 = generator enabled. INVERT: 1 = invert the output. CONTINUOUS: 0 = one waveform per trigger, 1 = repeat continuously after the first trigger. TRIGGER (SC): start the waveform after TRIGGER_DELAY. WAVEFORM: 0 = sine, 1 = impulse, other values = no output.
    'ADC_VIRT14_OFFS':	0x01B1,	#	16 | 0x0010 | Baseline level added to the waveform, in 14-bit ADC codes.
    'ADC_VIRT14_AMP':	0x01B2,	#	14336 | 0x3800 | Waveform scale: output = ROM value x AMPLITUDE / 16384. Default 0x3800 = 0.875.
    'ADC_VIRT14_NOISE':	0x01B3,	#	5 | 0x0005 | Peak-to-peak amplitude of the pseudo-random noise added to the waveform, in ADC codes (centred on the signal).
    'ADC_VIRT14_DELAY':	0x01B4,	#	0 | 0x0000 | Delay from trigger to waveform start, in µs.
    'ADC_VIRT15_WAV':	0x01B8,	#	0 | 0x0000 | Virtual ADC channel 15. ENABLE: 1 = generator enabled. INVERT: 1 = invert the output. CONTINUOUS: 0 = one waveform per trigger, 1 = repeat continuously after the first trigger. TRIGGER (SC): start the waveform after TRIGGER_DELAY. WAVEFORM: 0 = sine, 1 = impulse, other values = no output.
    'ADC_VIRT15_OFFS':	0x01B9,	#	16 | 0x0010 | Baseline level added to the waveform, in 14-bit ADC codes.
    'ADC_VIRT15_AMP':	0x01BA,	#	14336 | 0x3800 | Waveform scale: output = ROM value x AMPLITUDE / 16384. Default 0x3800 = 0.875.
    'ADC_VIRT15_NOISE':	0x01BB,	#	5 | 0x0005 | Peak-to-peak amplitude of the pseudo-random noise added to the waveform, in ADC codes (centred on the signal).
    'ADC_VIRT15_DELAY':	0x01BC,	#	0 | 0x0000 | Delay from trigger to waveform start, in µs.
    'TRG_MRG_CTRL':	0x0200,	#	Trigger | 0 | 0x0000 | ENABLE: 1 = merged trigger active. COUNT_RESET (SC): reload the trigger counter from TRG_MRG_CNT. FORCE (SC): generate one software trigger. EVENT_RESET (SC): clear TRG_MRG_EVT. The merged trigger (OR of the masked channel triggers) starts capture on all channels.
    'TRG_MRG_MASK':	0x0201,	#	Merge | 0 | 0x0000 | Channel trigger enable, bit n = channel n (1 = the channel trigger contributes to the merged trigger).
    'TRG_MRG_CNT_L':	0x0202,	#	0 | 0x0000 | Number of merged triggers to generate before stopping, bits 15:0. 0 = run continuously. The counter reloads while ENABLE = 0 or on COUNT_RESET.
    'TRG_MRG_CNT_H':	0x0203,	#	0 | 0x0000 | Number of merged triggers to generate, bits 31:16.
    'TRG_MRG_EVT_L':	0x0204,	#	0 | 0x0000 | Number of merged triggers generated since reset or EVENT_RESET, bits 15:0.
    'TRG_MRG_EVT_H':	0x0205,	#	0 | 0x0000 | Number of merged triggers generated, bits 31:16.
    'CH0_OFFS_CORR':	0x0300,	#	Channel 0 | 0 | 0x0000 | Channel 0 offset corrector. ENABLE: 1 = apply correction, 0 = raw samples pass through (INVERT is ignored). INVERT: 1 = sample = 0x3FFF - sample before the offset is applied. OFFSET_VALUE: subtracted from the sample; the result is clamped at 0.
    'CH0_TRIG_CTRL':	0x0301,	#	0 | 0x0000 | ENABLE: 1 = channel trigger detector active. EDGE: 0 = rising, 1 = falling. EVENT_RESET (SC): clear CHn_TRG_EVT.
    'CH0_TRIG_LEVEL':	0x0302,	#	8192 | 0x2000 | Trigger threshold in 14-bit ADC codes (after offset correction). Clamped internally to >= 5 + HYST (rising) or <= 0x3FFA - HYST (falling).
    'CH0_TRIG_HYST':	0x0303,	#	16 | 0x0010 | Trigger hysteresis in ADC codes. Rising: re-arms when the signal falls below LEVEL - HYST. Falling: re-arms when the signal rises above LEVEL + HYST.
    'CH0_TRIG_DELAY':	0x0304,	#	0 | 0x0000 | Delay from threshold crossing to trigger output, in µs. Currently disabled in the design (TBD - may be removed or re-enabled): reads 0, writes are ignored and the delay is fixed at 0.
    'CH0_TRIG_HOLD':	0x0305,	#	0 | 0x0000 | Hold-off after a trigger before the detector re-arms, in µs. Currently disabled in the design (TBD - to be re-enabled): reads 0, writes are ignored and the hold-off is fixed at 0.
    'CH0_TRG_EVT_L':	0x0306,	#	0 | 0x0000 | Number of triggers from this channel since reset or EVENT_RESET, bits 15:0.
    'CH0_TRG_EVT_H':	0x0307,	#	0 | 0x0000 | Number of triggers from this channel, bits 31:16.
    'CH0_CAPT_CTRL':	0x0308,	#	0 | 0x0000 | ENABLE: 1 = capture enabled: fill PRE samples, wait for the merged trigger (delayed by SKEW), capture POST samples, then present the record to Bulk Storage and re-arm. 0 = idle.
    'CH0_CAPT_PRE':	0x0309,	#	128 | 0x0080 | Number of samples captured before the trigger. PRE + POST must not exceed 1024 (capture FIFO depth). Default 128.
    'CH0_CAPT_POST':	0x030A,	#	768 | 0x0300 | Number of samples captured after the trigger. Default 768.
    'CH0_CAPT_SKEW':	0x030B,	#	0 | 0x0000 | Extra delay applied to the merged trigger for this channel before capture, in µs. Delay = TRIGGER_SKEW µs (0 = no delay). Maximum value 0xFFFE (65534 µs); do not write 0xFFFF - the channel will never capture.
    'CH1_OFFS_CORR':	0x0310,	#	Channel 1 | 0 | 0x0000 | Channel 1 offset corrector. ENABLE: 1 = apply correction, 0 = raw samples pass through (INVERT is ignored). INVERT: 1 = sample = 0x3FFF - sample before the offset is applied. OFFSET_VALUE: subtracted from the sample; the result is clamped at 0.
    'CH1_TRIG_CTRL':	0x0311,	#	0 | 0x0000 | ENABLE: 1 = channel trigger detector active. EDGE: 0 = rising, 1 = falling. EVENT_RESET (SC): clear CHn_TRG_EVT.
    'CH1_TRIG_LEVEL':	0x0312,	#	8192 | 0x2000 | Trigger threshold in 14-bit ADC codes (after offset correction). Clamped internally to >= 5 + HYST (rising) or <= 0x3FFA - HYST (falling).
    'CH1_TRIG_HYST':	0x0313,	#	16 | 0x0010 | Trigger hysteresis in ADC codes. Rising: re-arms when the signal falls below LEVEL - HYST. Falling: re-arms when the signal rises above LEVEL + HYST.
    'CH1_TRIG_DELAY':	0x0314,	#	0 | 0x0000 | Delay from threshold crossing to trigger output, in µs. Currently disabled in the design (TBD - may be removed or re-enabled): reads 0, writes are ignored and the delay is fixed at 0.
    'CH1_TRIG_HOLD':	0x0315,	#	0 | 0x0000 | Hold-off after a trigger before the detector re-arms, in µs. Currently disabled in the design (TBD - to be re-enabled): reads 0, writes are ignored and the hold-off is fixed at 0.
    'CH1_TRG_EVT_L':	0x0316,	#	0 | 0x0000 | Number of triggers from this channel since reset or EVENT_RESET, bits 15:0.
    'CH1_TRG_EVT_H':	0x0317,	#	0 | 0x0000 | Number of triggers from this channel, bits 31:16.
    'CH1_CAPT_CTRL':	0x0318,	#	0 | 0x0000 | ENABLE: 1 = capture enabled: fill PRE samples, wait for the merged trigger (delayed by SKEW), capture POST samples, then present the record to Bulk Storage and re-arm. 0 = idle.
    'CH1_CAPT_PRE':	0x0319,	#	128 | 0x0080 | Number of samples captured before the trigger. PRE + POST must not exceed 1024 (capture FIFO depth). Default 128.
    'CH1_CAPT_POST':	0x031A,	#	768 | 0x0300 | Number of samples captured after the trigger. Default 768.
    'CH1_CAPT_SKEW':	0x031B,	#	0 | 0x0000 | Extra delay applied to the merged trigger for this channel before capture, in µs. Delay = TRIGGER_SKEW µs (0 = no delay). Maximum value 0xFFFE (65534 µs); do not write 0xFFFF - the channel will never capture.
    'CH2_OFFS_CORR':	0x0320,	#	Channel 2 | 0 | 0x0000 | Channel 2 offset corrector. ENABLE: 1 = apply correction, 0 = raw samples pass through (INVERT is ignored). INVERT: 1 = sample = 0x3FFF - sample before the offset is applied. OFFSET_VALUE: subtracted from the sample; the result is clamped at 0.
    'CH2_TRIG_CTRL':	0x0321,	#	0 | 0x0000 | ENABLE: 1 = channel trigger detector active. EDGE: 0 = rising, 1 = falling. EVENT_RESET (SC): clear CHn_TRG_EVT.
    'CH2_TRIG_LEVEL':	0x0322,	#	8192 | 0x2000 | Trigger threshold in 14-bit ADC codes (after offset correction). Clamped internally to >= 5 + HYST (rising) or <= 0x3FFA - HYST (falling).
    'CH2_TRIG_HYST':	0x0323,	#	16 | 0x0010 | Trigger hysteresis in ADC codes. Rising: re-arms when the signal falls below LEVEL - HYST. Falling: re-arms when the signal rises above LEVEL + HYST.
    'CH2_TRIG_DELAY':	0x0324,	#	0 | 0x0000 | Delay from threshold crossing to trigger output, in µs. Currently disabled in the design (TBD - may be removed or re-enabled): reads 0, writes are ignored and the delay is fixed at 0.
    'CH2_TRIG_HOLD':	0x0325,	#	0 | 0x0000 | Hold-off after a trigger before the detector re-arms, in µs. Currently disabled in the design (TBD - to be re-enabled): reads 0, writes are ignored and the hold-off is fixed at 0.
    'CH2_TRG_EVT_L':	0x0326,	#	0 | 0x0000 | Number of triggers from this channel since reset or EVENT_RESET, bits 15:0.
    'CH2_TRG_EVT_H':	0x0327,	#	0 | 0x0000 | Number of triggers from this channel, bits 31:16.
    'CH2_CAPT_CTRL':	0x0328,	#	0 | 0x0000 | ENABLE: 1 = capture enabled: fill PRE samples, wait for the merged trigger (delayed by SKEW), capture POST samples, then present the record to Bulk Storage and re-arm. 0 = idle.
    'CH2_CAPT_PRE':	0x0329,	#	128 | 0x0080 | Number of samples captured before the trigger. PRE + POST must not exceed 1024 (capture FIFO depth). Default 128.
    'CH2_CAPT_POST':	0x032A,	#	768 | 0x0300 | Number of samples captured after the trigger. Default 768.
    'CH2_CAPT_SKEW':	0x032B,	#	0 | 0x0000 | Extra delay applied to the merged trigger for this channel before capture, in µs. Delay = TRIGGER_SKEW µs (0 = no delay). Maximum value 0xFFFE (65534 µs); do not write 0xFFFF - the channel will never capture.
    'CH3_OFFS_CORR':	0x0330,	#	Channel 3 | 0 | 0x0000 | Channel 3 offset corrector. ENABLE: 1 = apply correction, 0 = raw samples pass through (INVERT is ignored). INVERT: 1 = sample = 0x3FFF - sample before the offset is applied. OFFSET_VALUE: subtracted from the sample; the result is clamped at 0.
    'CH3_TRIG_CTRL':	0x0331,	#	0 | 0x0000 | ENABLE: 1 = channel trigger detector active. EDGE: 0 = rising, 1 = falling. EVENT_RESET (SC): clear CHn_TRG_EVT.
    'CH3_TRIG_LEVEL':	0x0332,	#	8192 | 0x2000 | Trigger threshold in 14-bit ADC codes (after offset correction). Clamped internally to >= 5 + HYST (rising) or <= 0x3FFA - HYST (falling).
    'CH3_TRIG_HYST':	0x0333,	#	16 | 0x0010 | Trigger hysteresis in ADC codes. Rising: re-arms when the signal falls below LEVEL - HYST. Falling: re-arms when the signal rises above LEVEL + HYST.
    'CH3_TRIG_DELAY':	0x0334,	#	0 | 0x0000 | Delay from threshold crossing to trigger output, in µs. Currently disabled in the design (TBD - may be removed or re-enabled): reads 0, writes are ignored and the delay is fixed at 0.
    'CH3_TRIG_HOLD':	0x0335,	#	0 | 0x0000 | Hold-off after a trigger before the detector re-arms, in µs. Currently disabled in the design (TBD - to be re-enabled): reads 0, writes are ignored and the hold-off is fixed at 0.
    'CH3_TRG_EVT_L':	0x0336,	#	0 | 0x0000 | Number of triggers from this channel since reset or EVENT_RESET, bits 15:0.
    'CH3_TRG_EVT_H':	0x0337,	#	0 | 0x0000 | Number of triggers from this channel, bits 31:16.
    'CH3_CAPT_CTRL':	0x0338,	#	0 | 0x0000 | ENABLE: 1 = capture enabled: fill PRE samples, wait for the merged trigger (delayed by SKEW), capture POST samples, then present the record to Bulk Storage and re-arm. 0 = idle.
    'CH3_CAPT_PRE':	0x0339,	#	128 | 0x0080 | Number of samples captured before the trigger. PRE + POST must not exceed 1024 (capture FIFO depth). Default 128.
    'CH3_CAPT_POST':	0x033A,	#	768 | 0x0300 | Number of samples captured after the trigger. Default 768.
    'CH3_CAPT_SKEW':	0x033B,	#	0 | 0x0000 | Extra delay applied to the merged trigger for this channel before capture, in µs. Delay = TRIGGER_SKEW µs (0 = no delay). Maximum value 0xFFFE (65534 µs); do not write 0xFFFF - the channel will never capture.
    'CH4_OFFS_CORR':	0x0340,	#	Channel 4 | 0 | 0x0000 | Channel 4 offset corrector. ENABLE: 1 = apply correction, 0 = raw samples pass through (INVERT is ignored). INVERT: 1 = sample = 0x3FFF - sample before the offset is applied. OFFSET_VALUE: subtracted from the sample; the result is clamped at 0.
    'CH4_TRIG_CTRL':	0x0341,	#	0 | 0x0000 | ENABLE: 1 = channel trigger detector active. EDGE: 0 = rising, 1 = falling. EVENT_RESET (SC): clear CHn_TRG_EVT.
    'CH4_TRIG_LEVEL':	0x0342,	#	8192 | 0x2000 | Trigger threshold in 14-bit ADC codes (after offset correction). Clamped internally to >= 5 + HYST (rising) or <= 0x3FFA - HYST (falling).
    'CH4_TRIG_HYST':	0x0343,	#	16 | 0x0010 | Trigger hysteresis in ADC codes. Rising: re-arms when the signal falls below LEVEL - HYST. Falling: re-arms when the signal rises above LEVEL + HYST.
    'CH4_TRIG_DELAY':	0x0344,	#	0 | 0x0000 | Delay from threshold crossing to trigger output, in µs. Currently disabled in the design (TBD - may be removed or re-enabled): reads 0, writes are ignored and the delay is fixed at 0.
    'CH4_TRIG_HOLD':	0x0345,	#	0 | 0x0000 | Hold-off after a trigger before the detector re-arms, in µs. Currently disabled in the design (TBD - to be re-enabled): reads 0, writes are ignored and the hold-off is fixed at 0.
    'CH4_TRG_EVT_L':	0x0346,	#	0 | 0x0000 | Number of triggers from this channel since reset or EVENT_RESET, bits 15:0.
    'CH4_TRG_EVT_H':	0x0347,	#	0 | 0x0000 | Number of triggers from this channel, bits 31:16.
    'CH4_CAPT_CTRL':	0x0348,	#	0 | 0x0000 | ENABLE: 1 = capture enabled: fill PRE samples, wait for the merged trigger (delayed by SKEW), capture POST samples, then present the record to Bulk Storage and re-arm. 0 = idle.
    'CH4_CAPT_PRE':	0x0349,	#	128 | 0x0080 | Number of samples captured before the trigger. PRE + POST must not exceed 1024 (capture FIFO depth). Default 128.
    'CH4_CAPT_POST':	0x034A,	#	768 | 0x0300 | Number of samples captured after the trigger. Default 768.
    'CH4_CAPT_SKEW':	0x034B,	#	0 | 0x0000 | Extra delay applied to the merged trigger for this channel before capture, in µs. Delay = TRIGGER_SKEW µs (0 = no delay). Maximum value 0xFFFE (65534 µs); do not write 0xFFFF - the channel will never capture.
    'CH5_OFFS_CORR':	0x0350,	#	Channel 5 | 0 | 0x0000 | Channel 5 offset corrector. ENABLE: 1 = apply correction, 0 = raw samples pass through (INVERT is ignored). INVERT: 1 = sample = 0x3FFF - sample before the offset is applied. OFFSET_VALUE: subtracted from the sample; the result is clamped at 0.
    'CH5_TRIG_CTRL':	0x0351,	#	0 | 0x0000 | ENABLE: 1 = channel trigger detector active. EDGE: 0 = rising, 1 = falling. EVENT_RESET (SC): clear CHn_TRG_EVT.
    'CH5_TRIG_LEVEL':	0x0352,	#	8192 | 0x2000 | Trigger threshold in 14-bit ADC codes (after offset correction). Clamped internally to >= 5 + HYST (rising) or <= 0x3FFA - HYST (falling).
    'CH5_TRIG_HYST':	0x0353,	#	16 | 0x0010 | Trigger hysteresis in ADC codes. Rising: re-arms when the signal falls below LEVEL - HYST. Falling: re-arms when the signal rises above LEVEL + HYST.
    'CH5_TRIG_DELAY':	0x0354,	#	0 | 0x0000 | Delay from threshold crossing to trigger output, in µs. Currently disabled in the design (TBD - may be removed or re-enabled): reads 0, writes are ignored and the delay is fixed at 0.
    'CH5_TRIG_HOLD':	0x0355,	#	0 | 0x0000 | Hold-off after a trigger before the detector re-arms, in µs. Currently disabled in the design (TBD - to be re-enabled): reads 0, writes are ignored and the hold-off is fixed at 0.
    'CH5_TRG_EVT_L':	0x0356,	#	0 | 0x0000 | Number of triggers from this channel since reset or EVENT_RESET, bits 15:0.
    'CH5_TRG_EVT_H':	0x0357,	#	0 | 0x0000 | Number of triggers from this channel, bits 31:16.
    'CH5_CAPT_CTRL':	0x0358,	#	0 | 0x0000 | ENABLE: 1 = capture enabled: fill PRE samples, wait for the merged trigger (delayed by SKEW), capture POST samples, then present the record to Bulk Storage and re-arm. 0 = idle.
    'CH5_CAPT_PRE':	0x0359,	#	128 | 0x0080 | Number of samples captured before the trigger. PRE + POST must not exceed 1024 (capture FIFO depth). Default 128.
    'CH5_CAPT_POST':	0x035A,	#	768 | 0x0300 | Number of samples captured after the trigger. Default 768.
    'CH5_CAPT_SKEW':	0x035B,	#	0 | 0x0000 | Extra delay applied to the merged trigger for this channel before capture, in µs. Delay = TRIGGER_SKEW µs (0 = no delay). Maximum value 0xFFFE (65534 µs); do not write 0xFFFF - the channel will never capture.
    'CH6_OFFS_CORR':	0x0360,	#	Channel 6 | 0 | 0x0000 | Channel 6 offset corrector. ENABLE: 1 = apply correction, 0 = raw samples pass through (INVERT is ignored). INVERT: 1 = sample = 0x3FFF - sample before the offset is applied. OFFSET_VALUE: subtracted from the sample; the result is clamped at 0.
    'CH6_TRIG_CTRL':	0x0361,	#	0 | 0x0000 | ENABLE: 1 = channel trigger detector active. EDGE: 0 = rising, 1 = falling. EVENT_RESET (SC): clear CHn_TRG_EVT.
    'CH6_TRIG_LEVEL':	0x0362,	#	8192 | 0x2000 | Trigger threshold in 14-bit ADC codes (after offset correction). Clamped internally to >= 5 + HYST (rising) or <= 0x3FFA - HYST (falling).
    'CH6_TRIG_HYST':	0x0363,	#	16 | 0x0010 | Trigger hysteresis in ADC codes. Rising: re-arms when the signal falls below LEVEL - HYST. Falling: re-arms when the signal rises above LEVEL + HYST.
    'CH6_TRIG_DELAY':	0x0364,	#	0 | 0x0000 | Delay from threshold crossing to trigger output, in µs. Currently disabled in the design (TBD - may be removed or re-enabled): reads 0, writes are ignored and the delay is fixed at 0.
    'CH6_TRIG_HOLD':	0x0365,	#	0 | 0x0000 | Hold-off after a trigger before the detector re-arms, in µs. Currently disabled in the design (TBD - to be re-enabled): reads 0, writes are ignored and the hold-off is fixed at 0.
    'CH6_TRG_EVT_L':	0x0366,	#	0 | 0x0000 | Number of triggers from this channel since reset or EVENT_RESET, bits 15:0.
    'CH6_TRG_EVT_H':	0x0367,	#	0 | 0x0000 | Number of triggers from this channel, bits 31:16.
    'CH6_CAPT_CTRL':	0x0368,	#	0 | 0x0000 | ENABLE: 1 = capture enabled: fill PRE samples, wait for the merged trigger (delayed by SKEW), capture POST samples, then present the record to Bulk Storage and re-arm. 0 = idle.
    'CH6_CAPT_PRE':	0x0369,	#	128 | 0x0080 | Number of samples captured before the trigger. PRE + POST must not exceed 1024 (capture FIFO depth). Default 128.
    'CH6_CAPT_POST':	0x036A,	#	768 | 0x0300 | Number of samples captured after the trigger. Default 768.
    'CH6_CAPT_SKEW':	0x036B,	#	0 | 0x0000 | Extra delay applied to the merged trigger for this channel before capture, in µs. Delay = TRIGGER_SKEW µs (0 = no delay). Maximum value 0xFFFE (65534 µs); do not write 0xFFFF - the channel will never capture.
    'CH7_OFFS_CORR':	0x0370,	#	Channel 7 | 0 | 0x0000 | Channel 7 offset corrector. ENABLE: 1 = apply correction, 0 = raw samples pass through (INVERT is ignored). INVERT: 1 = sample = 0x3FFF - sample before the offset is applied. OFFSET_VALUE: subtracted from the sample; the result is clamped at 0.
    'CH7_TRIG_CTRL':	0x0371,	#	0 | 0x0000 | ENABLE: 1 = channel trigger detector active. EDGE: 0 = rising, 1 = falling. EVENT_RESET (SC): clear CHn_TRG_EVT.
    'CH7_TRIG_LEVEL':	0x0372,	#	8192 | 0x2000 | Trigger threshold in 14-bit ADC codes (after offset correction). Clamped internally to >= 5 + HYST (rising) or <= 0x3FFA - HYST (falling).
    'CH7_TRIG_HYST':	0x0373,	#	16 | 0x0010 | Trigger hysteresis in ADC codes. Rising: re-arms when the signal falls below LEVEL - HYST. Falling: re-arms when the signal rises above LEVEL + HYST.
    'CH7_TRIG_DELAY':	0x0374,	#	0 | 0x0000 | Delay from threshold crossing to trigger output, in µs. Currently disabled in the design (TBD - may be removed or re-enabled): reads 0, writes are ignored and the delay is fixed at 0.
    'CH7_TRIG_HOLD':	0x0375,	#	0 | 0x0000 | Hold-off after a trigger before the detector re-arms, in µs. Currently disabled in the design (TBD - to be re-enabled): reads 0, writes are ignored and the hold-off is fixed at 0.
    'CH7_TRG_EVT_L':	0x0376,	#	0 | 0x0000 | Number of triggers from this channel since reset or EVENT_RESET, bits 15:0.
    'CH7_TRG_EVT_H':	0x0377,	#	0 | 0x0000 | Number of triggers from this channel, bits 31:16.
    'CH7_CAPT_CTRL':	0x0378,	#	0 | 0x0000 | ENABLE: 1 = capture enabled: fill PRE samples, wait for the merged trigger (delayed by SKEW), capture POST samples, then present the record to Bulk Storage and re-arm. 0 = idle.
    'CH7_CAPT_PRE':	0x0379,	#	128 | 0x0080 | Number of samples captured before the trigger. PRE + POST must not exceed 1024 (capture FIFO depth). Default 128.
    'CH7_CAPT_POST':	0x037A,	#	768 | 0x0300 | Number of samples captured after the trigger. Default 768.
    'CH7_CAPT_SKEW':	0x037B,	#	0 | 0x0000 | Extra delay applied to the merged trigger for this channel before capture, in µs. Delay = TRIGGER_SKEW µs (0 = no delay). Maximum value 0xFFFE (65534 µs); do not write 0xFFFF - the channel will never capture.
    'CH8_OFFS_CORR':	0x0380,	#	Channel 8 | 0 | 0x0000 | Channel 8 offset corrector. ENABLE: 1 = apply correction, 0 = raw samples pass through (INVERT is ignored). INVERT: 1 = sample = 0x3FFF - sample before the offset is applied. OFFSET_VALUE: subtracted from the sample; the result is clamped at 0.
    'CH8_TRIG_CTRL':	0x0381,	#	0 | 0x0000 | ENABLE: 1 = channel trigger detector active. EDGE: 0 = rising, 1 = falling. EVENT_RESET (SC): clear CHn_TRG_EVT.
    'CH8_TRIG_LEVEL':	0x0382,	#	8192 | 0x2000 | Trigger threshold in 14-bit ADC codes (after offset correction). Clamped internally to >= 5 + HYST (rising) or <= 0x3FFA - HYST (falling).
    'CH8_TRIG_HYST':	0x0383,	#	16 | 0x0010 | Trigger hysteresis in ADC codes. Rising: re-arms when the signal falls below LEVEL - HYST. Falling: re-arms when the signal rises above LEVEL + HYST.
    'CH8_TRIG_DELAY':	0x0384,	#	0 | 0x0000 | Delay from threshold crossing to trigger output, in µs. Currently disabled in the design (TBD - may be removed or re-enabled): reads 0, writes are ignored and the delay is fixed at 0.
    'CH8_TRIG_HOLD':	0x0385,	#	0 | 0x0000 | Hold-off after a trigger before the detector re-arms, in µs. Currently disabled in the design (TBD - to be re-enabled): reads 0, writes are ignored and the hold-off is fixed at 0.
    'CH8_TRG_EVT_L':	0x0386,	#	0 | 0x0000 | Number of triggers from this channel since reset or EVENT_RESET, bits 15:0.
    'CH8_TRG_EVT_H':	0x0387,	#	0 | 0x0000 | Number of triggers from this channel, bits 31:16.
    'CH8_CAPT_CTRL':	0x0388,	#	0 | 0x0000 | ENABLE: 1 = capture enabled: fill PRE samples, wait for the merged trigger (delayed by SKEW), capture POST samples, then present the record to Bulk Storage and re-arm. 0 = idle.
    'CH8_CAPT_PRE':	0x0389,	#	128 | 0x0080 | Number of samples captured before the trigger. PRE + POST must not exceed 1024 (capture FIFO depth). Default 128.
    'CH8_CAPT_POST':	0x038A,	#	768 | 0x0300 | Number of samples captured after the trigger. Default 768.
    'CH8_CAPT_SKEW':	0x038B,	#	0 | 0x0000 | Extra delay applied to the merged trigger for this channel before capture, in µs. Delay = TRIGGER_SKEW µs (0 = no delay). Maximum value 0xFFFE (65534 µs); do not write 0xFFFF - the channel will never capture.
    'CH9_OFFS_CORR':	0x0390,	#	Channel 9 | 0 | 0x0000 | Channel 9 offset corrector. ENABLE: 1 = apply correction, 0 = raw samples pass through (INVERT is ignored). INVERT: 1 = sample = 0x3FFF - sample before the offset is applied. OFFSET_VALUE: subtracted from the sample; the result is clamped at 0.
    'CH9_TRIG_CTRL':	0x0391,	#	0 | 0x0000 | ENABLE: 1 = channel trigger detector active. EDGE: 0 = rising, 1 = falling. EVENT_RESET (SC): clear CHn_TRG_EVT.
    'CH9_TRIG_LEVEL':	0x0392,	#	8192 | 0x2000 | Trigger threshold in 14-bit ADC codes (after offset correction). Clamped internally to >= 5 + HYST (rising) or <= 0x3FFA - HYST (falling).
    'CH9_TRIG_HYST':	0x0393,	#	16 | 0x0010 | Trigger hysteresis in ADC codes. Rising: re-arms when the signal falls below LEVEL - HYST. Falling: re-arms when the signal rises above LEVEL + HYST.
    'CH9_TRIG_DELAY':	0x0394,	#	0 | 0x0000 | Delay from threshold crossing to trigger output, in µs. Currently disabled in the design (TBD - may be removed or re-enabled): reads 0, writes are ignored and the delay is fixed at 0.
    'CH9_TRIG_HOLD':	0x0395,	#	0 | 0x0000 | Hold-off after a trigger before the detector re-arms, in µs. Currently disabled in the design (TBD - to be re-enabled): reads 0, writes are ignored and the hold-off is fixed at 0.
    'CH9_TRG_EVT_L':	0x0396,	#	0 | 0x0000 | Number of triggers from this channel since reset or EVENT_RESET, bits 15:0.
    'CH9_TRG_EVT_H':	0x0397,	#	0 | 0x0000 | Number of triggers from this channel, bits 31:16.
    'CH9_CAPT_CTRL':	0x0398,	#	0 | 0x0000 | ENABLE: 1 = capture enabled: fill PRE samples, wait for the merged trigger (delayed by SKEW), capture POST samples, then present the record to Bulk Storage and re-arm. 0 = idle.
    'CH9_CAPT_PRE':	0x0399,	#	128 | 0x0080 | Number of samples captured before the trigger. PRE + POST must not exceed 1024 (capture FIFO depth). Default 128.
    'CH9_CAPT_POST':	0x039A,	#	768 | 0x0300 | Number of samples captured after the trigger. Default 768.
    'CH9_CAPT_SKEW':	0x039B,	#	0 | 0x0000 | Extra delay applied to the merged trigger for this channel before capture, in µs. Delay = TRIGGER_SKEW µs (0 = no delay). Maximum value 0xFFFE (65534 µs); do not write 0xFFFF - the channel will never capture.
    'CH10_OFFS_CORR':	0x03A0,	#	Channel 10 | 0 | 0x0000 | Channel 10 offset corrector. ENABLE: 1 = apply correction, 0 = raw samples pass through (INVERT is ignored). INVERT: 1 = sample = 0x3FFF - sample before the offset is applied. OFFSET_VALUE: subtracted from the sample; the result is clamped at 0.
    'CH10_TRIG_CTRL':	0x03A1,	#	0 | 0x0000 | ENABLE: 1 = channel trigger detector active. EDGE: 0 = rising, 1 = falling. EVENT_RESET (SC): clear CHn_TRG_EVT.
    'CH10_TRIG_LEVEL':	0x03A2,	#	8192 | 0x2000 | Trigger threshold in 14-bit ADC codes (after offset correction). Clamped internally to >= 5 + HYST (rising) or <= 0x3FFA - HYST (falling).
    'CH10_TRIG_HYST':	0x03A3,	#	16 | 0x0010 | Trigger hysteresis in ADC codes. Rising: re-arms when the signal falls below LEVEL - HYST. Falling: re-arms when the signal rises above LEVEL + HYST.
    'CH10_TRIG_DELAY':	0x03A4,	#	0 | 0x0000 | Delay from threshold crossing to trigger output, in µs. Currently disabled in the design (TBD - may be removed or re-enabled): reads 0, writes are ignored and the delay is fixed at 0.
    'CH10_TRIG_HOLD':	0x03A5,	#	0 | 0x0000 | Hold-off after a trigger before the detector re-arms, in µs. Currently disabled in the design (TBD - to be re-enabled): reads 0, writes are ignored and the hold-off is fixed at 0.
    'CH10_TRG_EVT_L':	0x03A6,	#	0 | 0x0000 | Number of triggers from this channel since reset or EVENT_RESET, bits 15:0.
    'CH10_TRG_EVT_H':	0x03A7,	#	0 | 0x0000 | Number of triggers from this channel, bits 31:16.
    'CH10_CAPT_CTRL':	0x03A8,	#	0 | 0x0000 | ENABLE: 1 = capture enabled: fill PRE samples, wait for the merged trigger (delayed by SKEW), capture POST samples, then present the record to Bulk Storage and re-arm. 0 = idle.
    'CH10_CAPT_PRE':	0x03A9,	#	128 | 0x0080 | Number of samples captured before the trigger. PRE + POST must not exceed 1024 (capture FIFO depth). Default 128.
    'CH10_CAPT_POST':	0x03AA,	#	768 | 0x0300 | Number of samples captured after the trigger. Default 768.
    'CH10_CAPT_SKEW':	0x03AB,	#	0 | 0x0000 | Extra delay applied to the merged trigger for this channel before capture, in µs. Delay = TRIGGER_SKEW µs (0 = no delay). Maximum value 0xFFFE (65534 µs); do not write 0xFFFF - the channel will never capture.
    'CH11_OFFS_CORR':	0x03B0,	#	Channel 11 | 0 | 0x0000 | Channel 11 offset corrector. ENABLE: 1 = apply correction, 0 = raw samples pass through (INVERT is ignored). INVERT: 1 = sample = 0x3FFF - sample before the offset is applied. OFFSET_VALUE: subtracted from the sample; the result is clamped at 0.
    'CH11_TRIG_CTRL':	0x03B1,	#	0 | 0x0000 | ENABLE: 1 = channel trigger detector active. EDGE: 0 = rising, 1 = falling. EVENT_RESET (SC): clear CHn_TRG_EVT.
    'CH11_TRIG_LEVEL':	0x03B2,	#	8192 | 0x2000 | Trigger threshold in 14-bit ADC codes (after offset correction). Clamped internally to >= 5 + HYST (rising) or <= 0x3FFA - HYST (falling).
    'CH11_TRIG_HYST':	0x03B3,	#	16 | 0x0010 | Trigger hysteresis in ADC codes. Rising: re-arms when the signal falls below LEVEL - HYST. Falling: re-arms when the signal rises above LEVEL + HYST.
    'CH11_TRIG_DELAY':	0x03B4,	#	0 | 0x0000 | Delay from threshold crossing to trigger output, in µs. Currently disabled in the design (TBD - may be removed or re-enabled): reads 0, writes are ignored and the delay is fixed at 0.
    'CH11_TRIG_HOLD':	0x03B5,	#	0 | 0x0000 | Hold-off after a trigger before the detector re-arms, in µs. Currently disabled in the design (TBD - to be re-enabled): reads 0, writes are ignored and the hold-off is fixed at 0.
    'CH11_TRG_EVT_L':	0x03B6,	#	0 | 0x0000 | Number of triggers from this channel since reset or EVENT_RESET, bits 15:0.
    'CH11_TRG_EVT_H':	0x03B7,	#	0 | 0x0000 | Number of triggers from this channel, bits 31:16.
    'CH11_CAPT_CTRL':	0x03B8,	#	0 | 0x0000 | ENABLE: 1 = capture enabled: fill PRE samples, wait for the merged trigger (delayed by SKEW), capture POST samples, then present the record to Bulk Storage and re-arm. 0 = idle.
    'CH11_CAPT_PRE':	0x03B9,	#	128 | 0x0080 | Number of samples captured before the trigger. PRE + POST must not exceed 1024 (capture FIFO depth). Default 128.
    'CH11_CAPT_POST':	0x03BA,	#	768 | 0x0300 | Number of samples captured after the trigger. Default 768.
    'CH11_CAPT_SKEW':	0x03BB,	#	0 | 0x0000 | Extra delay applied to the merged trigger for this channel before capture, in µs. Delay = TRIGGER_SKEW µs (0 = no delay). Maximum value 0xFFFE (65534 µs); do not write 0xFFFF - the channel will never capture.
    'CH12_OFFS_CORR':	0x03C0,	#	Channel 12 | 0 | 0x0000 | Channel 12 offset corrector. ENABLE: 1 = apply correction, 0 = raw samples pass through (INVERT is ignored). INVERT: 1 = sample = 0x3FFF - sample before the offset is applied. OFFSET_VALUE: subtracted from the sample; the result is clamped at 0.
    'CH12_TRIG_CTRL':	0x03C1,	#	0 | 0x0000 | ENABLE: 1 = channel trigger detector active. EDGE: 0 = rising, 1 = falling. EVENT_RESET (SC): clear CHn_TRG_EVT.
    'CH12_TRIG_LEVEL':	0x03C2,	#	8192 | 0x2000 | Trigger threshold in 14-bit ADC codes (after offset correction). Clamped internally to >= 5 + HYST (rising) or <= 0x3FFA - HYST (falling).
    'CH12_TRIG_HYST':	0x03C3,	#	16 | 0x0010 | Trigger hysteresis in ADC codes. Rising: re-arms when the signal falls below LEVEL - HYST. Falling: re-arms when the signal rises above LEVEL + HYST.
    'CH12_TRIG_DELAY':	0x03C4,	#	0 | 0x0000 | Delay from threshold crossing to trigger output, in µs. Currently disabled in the design (TBD - may be removed or re-enabled): reads 0, writes are ignored and the delay is fixed at 0.
    'CH12_TRIG_HOLD':	0x03C5,	#	0 | 0x0000 | Hold-off after a trigger before the detector re-arms, in µs. Currently disabled in the design (TBD - to be re-enabled): reads 0, writes are ignored and the hold-off is fixed at 0.
    'CH12_TRG_EVT_L':	0x03C6,	#	0 | 0x0000 | Number of triggers from this channel since reset or EVENT_RESET, bits 15:0.
    'CH12_TRG_EVT_H':	0x03C7,	#	0 | 0x0000 | Number of triggers from this channel, bits 31:16.
    'CH12_CAPT_CTRL':	0x03C8,	#	0 | 0x0000 | ENABLE: 1 = capture enabled: fill PRE samples, wait for the merged trigger (delayed by SKEW), capture POST samples, then present the record to Bulk Storage and re-arm. 0 = idle.
    'CH12_CAPT_PRE':	0x03C9,	#	128 | 0x0080 | Number of samples captured before the trigger. PRE + POST must not exceed 1024 (capture FIFO depth). Default 128.
    'CH12_CAPT_POST':	0x03CA,	#	768 | 0x0300 | Number of samples captured after the trigger. Default 768.
    'CH12_CAPT_SKEW':	0x03CB,	#	0 | 0x0000 | Extra delay applied to the merged trigger for this channel before capture, in µs. Delay = TRIGGER_SKEW µs (0 = no delay). Maximum value 0xFFFE (65534 µs); do not write 0xFFFF - the channel will never capture.
    'CH13_OFFS_CORR':	0x03D0,	#	Channel 13 | 0 | 0x0000 | Channel 13 offset corrector. ENABLE: 1 = apply correction, 0 = raw samples pass through (INVERT is ignored). INVERT: 1 = sample = 0x3FFF - sample before the offset is applied. OFFSET_VALUE: subtracted from the sample; the result is clamped at 0.
    'CH13_TRIG_CTRL':	0x03D1,	#	0 | 0x0000 | ENABLE: 1 = channel trigger detector active. EDGE: 0 = rising, 1 = falling. EVENT_RESET (SC): clear CHn_TRG_EVT.
    'CH13_TRIG_LEVEL':	0x03D2,	#	8192 | 0x2000 | Trigger threshold in 14-bit ADC codes (after offset correction). Clamped internally to >= 5 + HYST (rising) or <= 0x3FFA - HYST (falling).
    'CH13_TRIG_HYST':	0x03D3,	#	16 | 0x0010 | Trigger hysteresis in ADC codes. Rising: re-arms when the signal falls below LEVEL - HYST. Falling: re-arms when the signal rises above LEVEL + HYST.
    'CH13_TRIG_DELAY':	0x03D4,	#	0 | 0x0000 | Delay from threshold crossing to trigger output, in µs. Currently disabled in the design (TBD - may be removed or re-enabled): reads 0, writes are ignored and the delay is fixed at 0.
    'CH13_TRIG_HOLD':	0x03D5,	#	0 | 0x0000 | Hold-off after a trigger before the detector re-arms, in µs. Currently disabled in the design (TBD - to be re-enabled): reads 0, writes are ignored and the hold-off is fixed at 0.
    'CH13_TRG_EVT_L':	0x03D6,	#	0 | 0x0000 | Number of triggers from this channel since reset or EVENT_RESET, bits 15:0.
    'CH13_TRG_EVT_H':	0x03D7,	#	0 | 0x0000 | Number of triggers from this channel, bits 31:16.
    'CH13_CAPT_CTRL':	0x03D8,	#	0 | 0x0000 | ENABLE: 1 = capture enabled: fill PRE samples, wait for the merged trigger (delayed by SKEW), capture POST samples, then present the record to Bulk Storage and re-arm. 0 = idle.
    'CH13_CAPT_PRE':	0x03D9,	#	128 | 0x0080 | Number of samples captured before the trigger. PRE + POST must not exceed 1024 (capture FIFO depth). Default 128.
    'CH13_CAPT_POST':	0x03DA,	#	768 | 0x0300 | Number of samples captured after the trigger. Default 768.
    'CH13_CAPT_SKEW':	0x03DB,	#	0 | 0x0000 | Extra delay applied to the merged trigger for this channel before capture, in µs. Delay = TRIGGER_SKEW µs (0 = no delay). Maximum value 0xFFFE (65534 µs); do not write 0xFFFF - the channel will never capture.
    'CH14_OFFS_CORR':	0x03E0,	#	Channel 14 | 0 | 0x0000 | Channel 14 offset corrector. ENABLE: 1 = apply correction, 0 = raw samples pass through (INVERT is ignored). INVERT: 1 = sample = 0x3FFF - sample before the offset is applied. OFFSET_VALUE: subtracted from the sample; the result is clamped at 0.
    'CH14_TRIG_CTRL':	0x03E1,	#	0 | 0x0000 | ENABLE: 1 = channel trigger detector active. EDGE: 0 = rising, 1 = falling. EVENT_RESET (SC): clear CHn_TRG_EVT.
    'CH14_TRIG_LEVEL':	0x03E2,	#	8192 | 0x2000 | Trigger threshold in 14-bit ADC codes (after offset correction). Clamped internally to >= 5 + HYST (rising) or <= 0x3FFA - HYST (falling).
    'CH14_TRIG_HYST':	0x03E3,	#	16 | 0x0010 | Trigger hysteresis in ADC codes. Rising: re-arms when the signal falls below LEVEL - HYST. Falling: re-arms when the signal rises above LEVEL + HYST.
    'CH14_TRIG_DELAY':	0x03E4,	#	0 | 0x0000 | Delay from threshold crossing to trigger output, in µs. Currently disabled in the design (TBD - may be removed or re-enabled): reads 0, writes are ignored and the delay is fixed at 0.
    'CH14_TRIG_HOLD':	0x03E5,	#	0 | 0x0000 | Hold-off after a trigger before the detector re-arms, in µs. Currently disabled in the design (TBD - to be re-enabled): reads 0, writes are ignored and the hold-off is fixed at 0.
    'CH14_TRG_EVT_L':	0x03E6,	#	0 | 0x0000 | Number of triggers from this channel since reset or EVENT_RESET, bits 15:0.
    'CH14_TRG_EVT_H':	0x03E7,	#	0 | 0x0000 | Number of triggers from this channel, bits 31:16.
    'CH14_CAPT_CTRL':	0x03E8,	#	0 | 0x0000 | ENABLE: 1 = capture enabled: fill PRE samples, wait for the merged trigger (delayed by SKEW), capture POST samples, then present the record to Bulk Storage and re-arm. 0 = idle.
    'CH14_CAPT_PRE':	0x03E9,	#	128 | 0x0080 | Number of samples captured before the trigger. PRE + POST must not exceed 1024 (capture FIFO depth). Default 128.
    'CH14_CAPT_POST':	0x03EA,	#	768 | 0x0300 | Number of samples captured after the trigger. Default 768.
    'CH14_CAPT_SKEW':	0x03EB,	#	0 | 0x0000 | Extra delay applied to the merged trigger for this channel before capture, in µs. Delay = TRIGGER_SKEW µs (0 = no delay). Maximum value 0xFFFE (65534 µs); do not write 0xFFFF - the channel will never capture.
    'CH15_OFFS_CORR':	0x03F0,	#	Channel 15 | 0 | 0x0000 | Channel 15 offset corrector. ENABLE: 1 = apply correction, 0 = raw samples pass through (INVERT is ignored). INVERT: 1 = sample = 0x3FFF - sample before the offset is applied. OFFSET_VALUE: subtracted from the sample; the result is clamped at 0.
    'CH15_TRIG_CTRL':	0x03F1,	#	0 | 0x0000 | ENABLE: 1 = channel trigger detector active. EDGE: 0 = rising, 1 = falling. EVENT_RESET (SC): clear CHn_TRG_EVT.
    'CH15_TRIG_LEVEL':	0x03F2,	#	8192 | 0x2000 | Trigger threshold in 14-bit ADC codes (after offset correction). Clamped internally to >= 5 + HYST (rising) or <= 0x3FFA - HYST (falling).
    'CH15_TRIG_HYST':	0x03F3,	#	16 | 0x0010 | Trigger hysteresis in ADC codes. Rising: re-arms when the signal falls below LEVEL - HYST. Falling: re-arms when the signal rises above LEVEL + HYST.
    'CH15_TRIG_DELAY':	0x03F4,	#	0 | 0x0000 | Delay from threshold crossing to trigger output, in µs. Currently disabled in the design (TBD - may be removed or re-enabled): reads 0, writes are ignored and the delay is fixed at 0.
    'CH15_TRIG_HOLD':	0x03F5,	#	0 | 0x0000 | Hold-off after a trigger before the detector re-arms, in µs. Currently disabled in the design (TBD - to be re-enabled): reads 0, writes are ignored and the hold-off is fixed at 0.
    'CH15_TRG_EVT_L':	0x03F6,	#	0 | 0x0000 | Number of triggers from this channel since reset or EVENT_RESET, bits 15:0.
    'CH15_TRG_EVT_H':	0x03F7,	#	0 | 0x0000 | Number of triggers from this channel, bits 31:16.
    'CH15_CAPT_CTRL':	0x03F8,	#	0 | 0x0000 | ENABLE: 1 = capture enabled: fill PRE samples, wait for the merged trigger (delayed by SKEW), capture POST samples, then present the record to Bulk Storage and re-arm. 0 = idle.
    'CH15_CAPT_PRE':	0x03F9,	#	128 | 0x0080 | Number of samples captured before the trigger. PRE + POST must not exceed 1024 (capture FIFO depth). Default 128.
    'CH15_CAPT_POST':	0x03FA,	#	768 | 0x0300 | Number of samples captured after the trigger. Default 768.
    'CH15_CAPT_SKEW':	0x03FB,	#	0 | 0x0000 | Extra delay applied to the merged trigger for this channel before capture, in µs. Delay = TRIGGER_SKEW µs (0 = no delay). Maximum value 0xFFFE (65534 µs); do not write 0xFFFF - the channel will never capture.
    'BULK_CTRL':	0x0400,	#	Bulk | 0 | 0x0000 | ENABLE: 1 = aggregator scans channels 0-15 and copies completed capture records into the bulk FIFO (65536 x 16-bit), which the host reads over QSPI (8-bit, high byte first). FIFO_RESET (SC): clear the bulk FIFO. FLOODED (RO): 1 = the bulk FIFO filled and aggregation stopped; clear by writing ENABLE = 0. Record format (16-bit words): 0xCAFE, 0xF00D, time 15:0, 31:16, 47:32, 63:48 (system clock, µs), channel number, sample count, then the samples (14-bit, zero extended).
    'BULK_LEVEL':	0x0401,	#	Storage | 0 | 0x0000 | Number of 16-bit words in the bulk FIFO (write side), 0-65535.
}

dac_dictionary = {
    0: {'address': 0x98, 'channel_number': 0, 'channel_name': 'SiPM Bias 0'},
    1: {'address': 0x98, 'channel_number': 1, 'channel_name': 'SiPM Bias 1'},
    2: {'address': 0x98, 'channel_number': 2, 'channel_name': 'SiPM Bias 2'},
    3: {'address': 0x98, 'channel_number': 3, 'channel_name': 'SiPM Bias 3'},
    4: {'address': 0x98, 'channel_number': 4, 'channel_name': 'SiPM Bias 4'},
    5: {'address': 0x98, 'channel_number': 5, 'channel_name': 'SiPM Bias 5'},
    6: {'address': 0x98, 'channel_number': 6, 'channel_name': 'SiPM Bias 6'},
    7: {'address': 0x98, 'channel_number': 7, 'channel_name': 'SiPM Bias 7'},
    8: {'address': 0x9A, 'channel_number': 0, 'channel_name': 'SiPM Bias 8'},
    9: {'address': 0x9A, 'channel_number': 1, 'channel_name': 'SiPM Bias 9'},
    10: {'address': 0x9A, 'channel_number': 2, 'channel_name': 'SiPM Bias 10'},
    11: {'address': 0x9A, 'channel_number': 3, 'channel_name': 'SiPM Bias 11'},
    12: {'address': 0x9A, 'channel_number': 4, 'channel_name': 'SiPM Bias 12'},
    13: {'address': 0x9A, 'channel_number': 5, 'channel_name': 'SiPM Bias 13'},
    14: {'address': 0x9A, 'channel_number': 6, 'channel_name': 'Spare 0'},
    15: {'address': 0x9A, 'channel_number': 7, 'channel_name': 'Spare 1'},
    16: {'address': 0x9C, 'channel_number': 0, 'channel_name': 'SiPM Ref 0'},
    17: {'address': 0x9C, 'channel_number': 1, 'channel_name': 'SiPM Ref 1'},
    18: {'address': 0x9C, 'channel_number': 2, 'channel_name': 'SiPM Ref 2'},
    19: {'address': 0x9C, 'channel_number': 3, 'channel_name': 'SiPM Ref 3'},
    20: {'address': 0x9C, 'channel_number': 4, 'channel_name': 'SiPM Ref 4'},
    21: {'address': 0x9C, 'channel_number': 5, 'channel_name': 'SiPM Ref 5'},
    22: {'address': 0x9C, 'channel_number': 6, 'channel_name': 'SiPM Ref 6'},
    23: {'address': 0x9C, 'channel_number': 7, 'channel_name': 'SiPM Ref 7'},
    24: {'address': 0x9E, 'channel_number': 0, 'channel_name': 'SiPM Ref 8'},
    25: {'address': 0x9E, 'channel_number': 1, 'channel_name': 'SiPM Ref 9'},
    26: {'address': 0x9E, 'channel_number': 2, 'channel_name': 'SiPM Ref 10'},
    27: {'address': 0x9E, 'channel_number': 3, 'channel_name': 'SiPM Ref 11'},
    28: {'address': 0x9E, 'channel_number': 4, 'channel_name': 'SiPM Ref 12'},
    29: {'address': 0x9E, 'channel_number': 5, 'channel_name': 'SiPM Ref 13'},
    30: {'address': 0x9E, 'channel_number': 6, 'channel_name': 'FSC Ref'},
    31: {'address': 0x9E, 'channel_number': 7, 'channel_name': 'SSC Ref'},
}
number_of_dacs_pairs = 16

adc_dictionary = {
    	0: {'name':'ADC_ID_SIPM_0'},
		1: {'name': 'ADC_ID_SIPM_1'},
		2: {'name': 'ADC_ID_SIPM_2'},
		3: {'name': 'ADC_ID_SIPM_3'},
		4: {'name': 'ADC_ID_SIPM_4'},
		5: {'name': 'ADC_ID_SIPM_5'},
		6: {'name': 'ADC_ID_SIPM_6'},
		7: {'name': 'ADC_ID_SIPM_7'},
		8: {'name': 'ADC_ID_SIPM_8'},
		9: {'name': 'ADC_ID_SIPM_9'},
		10: {'name': 'ADC_ID_SIPM_10'},
		11: {'name': 'ADC_ID_SIPM_11'},
		12: {'name': 'ADC_ID_SIPM_12'},
		13: {'name': 'ADC_ID_SIPM_13'},
		14: {'name': 'ADC_ID_FSC'},
		15: {'name': 'ADC_ID_SSC'},
}
for adc in adc_dictionary.keys():
    adc_dictionary[adc]['reg_base_dc_sample'] = 0x0110 + adc
    adc_dictionary[adc]['reg_base_real'] = 0x0120 + adc
    adc_dictionary[adc]['reg_base_virt'] = (0x030 + adc) << 4 #### todo collision here!!!
    adc_dictionary[adc]['offset_corrector'] = (0x030 + adc) << 4
    adc_dictionary[adc]['trigger'] = (0x030 + adc) << 4

monitor_dictionary = {
    0: {'name': 'MON_ID_P_36_0V', 'i2_c_address': 0x80, 'i2_c_bus':'A'},
    1: {'name': 'MON_ID_P_36_0V_BIAS', 'i2_c_address': 0x82, 'i2_c_bus':'A'},
    2: {'name': 'MON_ID_P_12_0V', 'i2_c_address': 0x84, 'i2_c_bus':'A'},
    3: {'name': 'MON_ID_P_12_0V_BIAS', 'i2_c_address': 0x86, 'i2_c_bus':'A'},
    4: {'name': 'MON_ID_P_5_0V', 'i2_c_address': 0x80, 'i2_c_bus':'B'},
    5: {'name': 'MON_ID_P_5_0V_BIAS', 'i2_c_address': 0x82, 'i2_c_bus':'B'},
    6: {'name': 'MON_ID_P_3_3V', 'i2_c_address': 0x84, 'i2_c_bus':'B'},
    7: {'name': 'MON_ID_P_1_8V', 'i2_c_address': 0x86, 'i2_c_bus':'B'},
}
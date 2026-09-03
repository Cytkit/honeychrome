'''
This is the default configuration for the instrument driver
'''

# FGPA and communication settings
operation_write = b'\x01'
operation_read = b'\x02'
dummy_bytes = b'\x00\x00'
memory_start_address = 0
memory_end_address = 1_000_000

def lookup_address(register):
    if type(register) == str:
        return registers_map[register].to_bytes(2, byteorder='big')
    elif type(register) == int and 0 <= register and 255 >= register:
        return register
    else:
        raise TypeError

registers_map = {
    'RESERVED':	0x0000,   #
    'ID_WORD':	0x0001,   #	ID
    'VERSION_A':	0x0002,   #
    'VERSION_B':	0x0003,   #
    'VERSION_C':	0x0004,   #
    'TIMESTAMP_A':	0x0005,   #
    'TIMESTAMP_B':	0x0006,   #
    'TIMESTAMP_C':	0x0007,   #
    'TIMESTAMP_D':	0x0008,   #
    'LASER':	    0x0010,   #	Laser
    'SMPMP_CTRL':	0x0020,   #	Sample
    'SMPMP_SPEED':	0x0021,   #	Pump
    'SMPMP_SPC':	0x0022,   #
    'SMPMP_CPC_L':	0x0023,   #
    'SMPMP_CPC_H':	0x0024,   #
    'SHPMP_CTRL':	0x0030,   #	Sheath
    'SHPMP_DUTY':	0x0031,   #	Pump
    'SHPMP_FREQ':	0x0032,   #
    'FAN_CTRL':	0x0040,   #	Fan
    'FAN_DUTY':	0x0041,   #
    'FAN_FREQ':	0x0042,   #
    'FAN_TACHO':	0x0043,   #
    'INT_MASK':	0x0050,   #	Interlock
    'INT_INV':	0x0051,   #
    'INT_STATE':	0x0052,   #
    'INT_MISC':	0x0053,   #
    'PRES_CTRL':	0x0060,   #	Pressure
    'PRES_TXFR_SIZE':	0x0061,   #	Sensor
    'PRES_CS_WAIT':	0x0062,   #
    'PRES_DATA_L':	0x0063,   #
    'PRES_DATA_H':	0x0064,   #
    'DISP_CTRL':	0x0070,   #	Display
    'DISP_TXFR_SIZE':	0x0071,   #
    'DISP_DATA_L':	0x0073,   #
    'DISP_DATA_H':	0x0074,   #
    'I2CA_CTRL':	0x0080,   #	I2C A
    'I2CA_ADDRESS':	0x0081,   #
    'I2CA_READ_SIZE':	0x0082,   #
    'I2CA_DATA_IN':	0x0083,   #
    'I2CA_DATA_OUT':	0x0084,   #
    'I2CA_IN_LEVEL':	0x0085,   #
    'I2CA_OUT_LEVEL':	0x0086,   #
    'I2CA_STATE':	0x0087,   #
    'I2CB_CTRL':	0x0090,   #	I2C B
    'I2CB_ADDRESS':	0x0091,   #
    'I2CB_READ_SIZE':	0x0092,   #
    'I2CB_DATA_IN':	0x0093,   #
    'I2CB_DATA_OUT':	0x0094,   #
    'I2CB_IN_LEVEL':	0x0095,   #
    'I2CB_OUT_LEVEL':	0x0096,   #
    'I2CB_STATE':	0x0097,   #
    'ADC_SELECT':	0x00A0,   #	ADC Source Selection
    'ADC_ENABLE':	0x00B0,   #	HW ADC Controls
    'ADC_VIRT_ENABLE':	0x00C0,   #	Virtual ADC Controls
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
index_dac_bias_zero = 0
index_dac_ref_zero = 15

adc_dictionary = {
    	0: {'name':'ADC_ID_SIPM_0', 'register_base': 0x1000},
		1: {'name': 'ADC_ID_SIPM_1', 'register_base': 0x1100},
		2: {'name': 'ADC_ID_SIPM_2', 'register_base': 0x1200},
		3: {'name': 'ADC_ID_SIPM_3', 'register_base': 0x1300},
		4: {'name': 'ADC_ID_SIPM_4', 'register_base': 0x1400},
		5: {'name': 'ADC_ID_SIPM_5', 'register_base': 0x1500},
		6: {'name': 'ADC_ID_SIPM_6', 'register_base': 0x1600},
		7: {'name': 'ADC_ID_SIPM_7', 'register_base': 0x1700},
		8: {'name': 'ADC_ID_SIPM_8', 'register_base': 0x1800},
		9: {'name': 'ADC_ID_SIPM_9', 'register_base': 0x1900},
		10: {'name': 'ADC_ID_SIPM_10', 'register_base': 0x1A00},
		11: {'name': 'ADC_ID_SIPM_11', 'register_base': 0x1B00},
		12: {'name': 'ADC_ID_SIPM_12', 'register_base': 0x1C00},
		13: {'name': 'ADC_ID_SIPM_13', 'register_base': 0x1D00},
		14: {'name': 'ADC_ID_FSC', 'register_base': 0x1E00},
		15: {'name': 'ADC_ID_SSC', 'register_base': 0x1F00},
		0xFF: {'name': 'ADC_ID_ALL', 'register_base': None},
}

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
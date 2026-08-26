'''
This is the default configuration for the instrument driver
'''

# FGPA and communication settings
operation_write = b'\x01'
operation_read = b'\x02'
dummy_bytes = b'\x00\x00'
memory_start_address = 0
memory_end_address = 1_000_000

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

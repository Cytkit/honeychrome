# sensor coefficients
coefficient_data = [
    0, # Unused
    32768, # 2 ^ 15
    131072, # 2 ^ 17
    128, # 2 ^ 7
    32, # 2 ^ 5
    128, # 2 ^ 7
    2097152, # 2 ^ 21
]

def convert_unit_to_psi(value, units):
    if units == 'PRES_UNITS_PA':
        return value / 6894.76

    elif units == 'PRES_UNITS_PSI':
        return value

    elif units == 'PRES_UNITS_BAR':
        return value / 0.0689476

    elif units == 'PRES_UNITS_MBAR':
        return value / 68.9476

    elif units == 'PRES_UNITS_TORR':
        return value / 51.7149

    elif units == 'PRES_UNITS_ATM':
        return value * 14.696

    elif units == 'PRES_UNITS_MICRONS':
        return value / 51714.9

    else:
        return 0

def convert_psi_to_unit(value, units):
    if units == 'PRES_UNITS_PA':
        return value * 6894.76

    elif units == 'PRES_UNITS_PSI':
        return value

    elif units == 'PRES_UNITS_BAR':
        return value * 0.0689476

    elif units == 'PRES_UNITS_MBAR':
        return value * 68.9476

    elif units == 'PRES_UNITS_TORR':
        return value * 51.7149

    elif units == 'PRES_UNITS_ATM':
        return value / 14.696

    elif units == 'PRES_UNITS_MICRONS':
        return value * 51714.9

    else:
        return 0


class Pressure:
    def __init__(self, ft4222_communicator):
        self.ft4222 = ft4222_communicator
        self.readback_offset = None
        self.calibration_data = None

    def connect(self):
        self.readback_offset = 0.0
        self._sensor_reset()
        self._sensor_load_coeff_data()
        self._sensor_load_cal_data()

    def disconnect(self):
        pass

    def get_pressure(self, units, num_averages):
        average_pressure = 0
        for count in range(num_averages):

            # Read the raw ADC pressure and temperature values
            raw_pressure_value = self._sensor_get_raw_pressure()
            raw_temperaure_value = self._sensor_get_raw_temperature()

            # Calculate the actual pressure
            dT = raw_temperaure_value - (self.calibration_data[5] * coefficient_data[5])
            offset = (self.calibration_data[2] * coefficient_data[2]) + ((self.calibration_data[4] * dT) / coefficient_data[4])
            sensitivity = (self.calibration_data[1] * coefficient_data[1]) + ((self.calibration_data[3] * dT) / coefficient_data[3])

            pressure = (raw_pressure_value * sensitivity) / (2097152 - offset) / 32768.0
            pressure /= 10000.0

            # Range check
            if pressure < -1.0 or pressure > 1.0:
                # Fault
                return

            average_pressure += pressure

        average_pressure /= num_averages

        # offset correction
        average_pressure -= self.readback_offset

        return convert_psi_to_unit(average_pressure, units)


    def get_temperature(self):

        # --- Get the measured device temperature( in C) ---

        raw_temperaure_value = self._sensor_get_raw_temperature()

        # Calculate the actual pressure
        dT = raw_temperaure_value - (self.calibration_data[5] * coefficient_data[5])
        temperature = 2000 + ((dT * self.calibration_data[6]) / coefficient_data[6])

        # Range check
        if temperature < -4000 or temperature > 12500:
            # Fault
            return 0

        return temperature / 100


    def set_offset(self, offset, units):
        self.readback_offset = convert_unit_to_psi(offset, units)

    def get_offset(self, units):
        return convert_psi_to_unit(self.readback_offset, units)

    def _sensor_reset(self):
        self._sensor_write_read(0x1E, 8, 5)

    def _sensor_PROM_read(self, address):
        command = 0xA0 + ((address & 0x07) << 1)
        command <<= 16

        return self._sensor_write_read(command, 24, 0) & 0xFFFF

    def _sensor_conversion_start(self, D1_D2, OSR):
        command = 0x40 + ((OSR & 0x07) << 1)
        if D1_D2:
            command |= 0x10

        self._sensor_write_read(command, 8, 15)

    def _sensor_ADC_read(self):
        pass

    def _sensor_get_raw_pressure(self):
        pass

    def _sensor_get_raw_temperature(self):
        pass

    def _sensor_load_cal_data(self):
        # Clear the calibration data
        self.calibration_data = []

        # Load elements 0-7 with the PROMvalues specific to the device
        for count in range(8):

            command = 0xA0 + ((count & 0x07) << 1)
            command <<= 16
            data = self._sensor_write_read(command, 24, 0) & 0xFFFF
            self.calibration_data.append(data)

        # Check the CRC
        CRC_calculated = self._CRC_calculate()

    def _CRC_calculate(self):
        n_rem = 0x00;
        crc_read = self.calibration_data[7] # save read CRC
        self.calibration_data[7] = 0xFF00 & self.calibration_data[7]

        for cnt in range(16):	# operation is performed on bytes
            # choose LSB or MSB
            if cnt % 2 == 1:
                n_rem ^= self.calibration_data[cnt>>1] & 0x00FF
            else:
                n_rem ^= (self.calibration_data[cnt>>1] >> 8) & 0xFFFF

            for n_bit in range(8, 0, -1):
                if n_rem & 0x8000:
                    n_rem = ((n_rem << 1) ^ 0x3000) & 0xFFFF
                else:
                    n_rem = (n_rem << 1) & 0xFFFF

        n_rem = 0x000F & (n_rem >> 12)	# final 4-bit reminder is CRC code
        self.calibration_data[7] = crc_read

        return n_rem ^ 0x00


    def _sensor_write_read(self, data_out, size, cs_delay):

        self.ft4222.register_write('PRES_TXFR_SIZE', size) # Set the transfer size in bits
        self.ft4222.register_2byte_write('PRES_DATA_L', 'PRES_DATA_H', data_out)
        self.ft4222.register_write('PRES_CS_WAIT', cs_delay) # Set the CSDelay in ~650us steps
        self.ft4222.register_write('PRES_CTRL', 1)	#Start the transfer

        while not (self.ft4222.register_read('PRES_CTRL') & 0x0010): # Wait for completion
            pass

        data_return = self.ft4222.register_2byte_read('PRES_DATA_L', 'PRES_DATA_H')
        return data_return

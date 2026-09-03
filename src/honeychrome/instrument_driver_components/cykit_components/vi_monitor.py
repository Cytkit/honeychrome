from honeychrome.instrument_driver_components.cykit_components.cytkit_configuration import monitor_dictionary


class VIMonitor:
    def __init__(self, I2CBusA, I2CBusB):
        self.I2CBusA = I2CBusA
        self.I2CBusB = I2CBusB

    def disconnect(self):
        pass

    def read_voltage(self, channel):
        return_value = self._read_raw_value(channel, 0x02)
        return return_value * 0.0016

    def read_current(self, channel):
        return_value = self._read_raw_value(channel, 0x01)
        return return_value * 0.00025

    def _read_raw_value(self, channel, command):
        i2_c_address = monitor_dictionary[channel]['i2_c_address']
        if monitor_dictionary[channel]['i2c_bus'] == 'A':
            bus = self.I2CBusA
        elif monitor_dictionary[channel]['i2c_bus'] == 'B':
            bus = self.I2CBusB
        else:
            bus = None

        # Set the command
        write_buffer = [command]
        read_buffer = [0, 0]

        # Send it
        if not bus.write_read(
                i2_c_address,
                write_buffer,
                1,
                read_buffer,
                2):
            return 0

        # Return the response
        ret_value = read_buffer[0]
        ret_value <<= 8
        ret_value |= read_buffer[1]
        return ret_value
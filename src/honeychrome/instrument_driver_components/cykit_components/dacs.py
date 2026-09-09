from honeychrome.instrument_driver_components.cykit_components.cytkit_configuration import dac_dictionary, number_of_dacs_pairs


class DACs:
    def __init__(self, i2c_bus):
        self.i2c_bus = i2c_bus

    def disconnect(self):
        for count in range(32):
            self.set_value(count, 0)

    def set_value(self, dac_id, value):
        self._dac_set(dac_dictionary[dac_id]['address'], dac_dictionary[dac_id]['channel_number'], value, dac_dictionary[dac_id]['channel_name'])

    def set_value_ref(self, index, value):
        if 0 <= index <= number_of_dacs_pairs:
            self.set_value(number_of_dacs_pairs + index, value)
        else:
            raise ValueError("Index out of range")

    def set_value_bias(self, index, value):
        if 0 <= index <= number_of_dacs_pairs:
            self.set_value(index, value)
        else:
            raise ValueError("Index out of range")

    def get_value(self, dac_id):
        return self._dac_get(dac_dictionary[dac_id]['address'], dac_dictionary[dac_id]['channel_number'], dac_dictionary[dac_id]['channel_name'])

    def get_value_ref(self, index):
        if 0 <= index <= number_of_dacs_pairs:
            return self.get_value(number_of_dacs_pairs + index)
        else:
            raise ValueError("Index out of range")

    def get_value_bias(self, index):
        if 0 <= index <= number_of_dacs_pairs:
            return self.get_value(index)
        else:
            raise ValueError("Index out of range")

    def _dac_set(self, i2c_address, channel, value, dac_name):
        # Limit checks
        if channel >= 8:
            return
        if value >= 1024:
            return

        # Construct the I2C packet
        write_buffer = [channel & 0xFF, (value >> 2) & 0xFF, (value << 6) & 0xFF]

        # Send it
        self.i2c_bus.write(i2c_address, write_buffer, 3)

    def _dac_get(self, i2c_address, channel, dac_name):
        # Limit checks
        if channel >= 8:
            return None

        write_buffer = [channel & 0xFF]
        read_buffer = [0] * 2

        if not self.i2c_bus.write_read(i2c_address, write_buffer, 1, read_buffer, 2):
            return None

        # Value convert
        ret_value = (read_buffer[0] << 2) & 0x03FC
        ret_value |= read_buffer[1] >> 6

        return ret_value
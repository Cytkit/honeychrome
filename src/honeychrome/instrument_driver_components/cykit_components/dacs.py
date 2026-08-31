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

class DACs:
    def __init__(self, i2c_bus):
        self.i2c_bus = i2c_bus

    def disconnect(self):
        for count in range(32):
            self.set_value(count, 0)

    def set_value(self, dac_id, value):
        self._dac_set(dac_dictionary[dac_id]['address'], dac_dictionary[dac_id]['channel_number'], value, dac_dictionary[dac_id]['channel_name'])

    def set_value_ref(self, index, value):
        self.set_value(index_dac_ref_zero + index, value)

    def set_value_bias(self, index, value):
        self.set_value(index_dac_bias_zero + index, value)

    def get_value(self, dac_id):
        self._dac_get(dac_dictionary[dac_id]['address'], dac_dictionary[dac_id]['channel_number'], dac_dictionary[dac_id]['channel_name'])

    def get_value_ref(self, index):
        self.get_value(index_dac_ref_zero + index)

    def get_value_bias(self, index):
        self.get_value(index_dac_bias_zero + index)

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
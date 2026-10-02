from honeychrome.instrument_driver_components.cytkit_components.cytkit_configuration import adc_dictionary


class OffsetCorrector:
    def __init__(self, ft4222_communicator):
        self.ft4222 = ft4222_communicator

    def set_enable(self, channel, enable):
        if enable:
            self.ft4222.register_bit_set(adc_dictionary[channel]['offset_corrector'], 14)
        else:
            self.ft4222.register_bit_clear(adc_dictionary[channel]['offset_corrector'], 14)

    def get_enable(self, channel):
        return self.ft4222.register_bit_get(adc_dictionary[channel]['offset_corrector'], 14)

    def set_invert(self, channel, invert):
        if invert:
            self.ft4222.register_bit_set(adc_dictionary[channel]['offset_corrector'], 15)
        else:
            self.ft4222.register_bit_clear(adc_dictionary[channel]['offset_corrector'], 15)

    def get_invert(self, channel):
        return self.ft4222.register_bit_get(adc_dictionary[channel]['offset_corrector'], 15)

    def set_level(self, channel, level):
        self.ft4222.register_read_modify_write(adc_dictionary[channel]['offset_corrector'], level, 0x3FFF)

    def get_level(self, channel):
        return self.ft4222.register_read(adc_dictionary[channel]['offset_corrector']) & 0x3FFF
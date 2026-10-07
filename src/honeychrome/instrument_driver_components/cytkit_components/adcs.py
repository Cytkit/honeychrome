from honeychrome.instrument_driver_components.cytkit_components.cytkit_configuration import adc_dictionary


class ADCs:
    def __init__(self, ft4222_communicator):
        self.ft4222 = ft4222_communicator

    def set_mode_select(self, channel, virtual):
        if virtual:
            self.ft4222.register_bit_set('ADC_SELECT', channel)
        else:
            self.ft4222.register_bit_clear('ADC_SELECT', channel)

    def get_mode_select(self, channel):
        return self.ft4222.register_bit_get('ADC_SELECT', channel)

    def real_set_enable(self, channel, enable):
        if enable:
            self.ft4222.register_bit_set('ADC_ENABLE', channel)
        else:
            self.ft4222.register_bit_clear('ADC_ENABLE', channel)

    def real_get_enable(self, channel):
        return self.ft4222.register_bit_get('ADC_ENABLE', channel)

    def real_read_value(self, channel):
        register_base_real = adc_dictionary[channel]['reg_base_dc_sample']
        return self.ft4222.register_read(register_base_real)

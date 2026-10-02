from honeychrome.instrument_driver_components.cytkit_components.cytkit_configuration import adc_dictionary


class Trigger:
    def __init__(self, ft4222_communicator):
        self.ft4222 = ft4222_communicator

    def merge_set_enable(self, channel, enable):
        if enable:
            self.ft4222.register_bit_set('TRG_MRG_CTRL', 0)
        else:
            self.ft4222.register_bit_clear('TRG_MRG_CTRL', 0)

    def merge_get_enable(self):
        return self.ft4222.register_bit_get('TRG_MRG_CTRL', 0)

    def merge_set_merge_mask(self, mask):
        self.ft4222.register_write('TRG_MRG_MASK', mask)

    def merge_get_merge_mask(self):
        return self.ft4222.register_read('TRG_MRG_MASK')

    def merge_set_trigger_count(self, max_triggers):
        self.ft4222.register_2reg_write('TRG_MRG_CNT_L', 'TRG_MRG_CNT_H', max_triggers)

    def merge_get_trigger_count(self):
        return self.ft4222.register_2reg_read('TRG_MRG_CNT_L', 'TRG_MRG_CNT_H')

    def merge_trigger_count_reset(self):
        self.ft4222.register_bit_set('TRG_MRG_CTRL', 4)

    def merge_force_trigger(self):
        self.ft4222.register_bit_set('TRG_MRG_CTRL', 8)

    def merge_get_event_count(self):
        return self.ft4222.register_2reg_read('TRG_MRG_EVT_L', 'TRG_MRG_EVT_H')

    def merge_reset_event_count(self):
        self.ft4222.register_bit_set('TRG_MRG_CTRL', 12)


    def channel_set_enable(self, channel, enable):
        if enable:
            self.ft4222.register_bit_set(adc_dictionary[channel]['trigger'] + 0x0001, 0)
        else:
            self.ft4222.register_bit_clear(adc_dictionary[channel]['trigger'] + 0x0001, 0)

    def channel_get_enable(self, channel):
        return self.ft4222.register_bit_get(adc_dictionary[channel]['trigger'] + 0x0001, 0)

    def channel_set_mask(self, channel, mask):
        if mask:
            self.ft4222.register_bit_set('TRG_MRG_MASK', channel)
        else:
            self.ft4222.register_bit_clear('TRG_MRG_MASK', channel)

    def channel_get_mask(self, channel):
        return self.ft4222.register_bit_get('TRG_MRG_MASK', channel)

    def channel_set_level(self, channel, level):
        self.ft4222.register_write(adc_dictionary[channel]['trigger'] + 0x0002, level)

    def channel_get_level(self, channel):
        return self.ft4222.register_read(adc_dictionary[channel]['trigger'] + 0x0002)

    def channel_set_hysteresis(self, channel, level):
        self.ft4222.register_write(adc_dictionary[channel]['trigger'] + 0x0003, level)

    def channel_get_hysteresis(self, channel):
        return self.ft4222.register_read(adc_dictionary[channel]['trigger'] + 0x0003)


    def channel_set_edge(self, channel, falling):
        if falling:
            self.ft4222.register_bit_set(adc_dictionary[channel]['trigger'] + 0x0001, 4)
        else:
            self.ft4222.register_bit_clear(adc_dictionary[channel]['trigger'] + 0x0001, 4)

    def channel_get_edge(self, channel):
        return self.ft4222.register_bit_get(adc_dictionary[channel]['trigger'] + 0x0001, 4)


    def channel_set_delay(self, channel, delay):
        self.ft4222.register_write(adc_dictionary[channel]['trigger'] + 0x0004, delay)

    def channel_get_delay(self, channel):
        return self.ft4222.register_read(adc_dictionary[channel]['trigger'] + 0x0004)

    def channel_set_holdoff(self, channel, delay):
        self.ft4222.register_write(adc_dictionary[channel]['trigger'] + 0x0005, delay)

    def channel_get_holdoff(self, channel):
        return self.ft4222.register_read(adc_dictionary[channel]['trigger'] + 0x0005)

    def channel_get_event_count(self, channel):
        reg_base = adc_dictionary[channel]['trigger']
        return self.ft4222.register_2reg_read(reg_base + 0x0006, reg_base + 0x0007)

    def channel_reset_event_count(self, channel):
        self.ft4222.register_bit_set(adc_dictionary[channel]['trigger'] + 0x0001, 8)


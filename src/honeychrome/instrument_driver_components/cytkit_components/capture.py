from honeychrome.instrument_driver_components.cytkit_components.cytkit_configuration import adc_dictionary

class Capture:
    def __init__(self, ft4222_communicator):
        self.ft4222 = ft4222_communicator


    def aggr_set_enable(self, enable):
        if enable:
            self.ft4222.register_bit_set('BULK_CTRL', 0)
        else:
            self.ft4222.register_bit_clear('BULK_CTRL', 0)

    def aggr_get_enable(self):
        return self.ft4222.register_bit_get('BULK_CTRL', 0)

    def aggr_get_flooded(self):
        return self.ft4222.register_bit_get('BULK_CTRL', 8)

    def aggr_fifo_clear(self):
        # must be disabled while clearing; hold the lock so the transfer thread can't read in between
        with self.ft4222.lock:
            enabled = self.aggr_get_enable()
            self.aggr_set_enable(False)
            self.ft4222.register_bit_set('BULK_CTRL', 4)
            self.aggr_set_enable(enabled)

    def aggr_get_fifo_level(self):
        return self.ft4222.register_read('BULK_LEVEL')

    def channel_set_enable(self, channel, enable):
        if enable:
            self.ft4222.register_bit_set(adc_dictionary[channel]['capture'] + 0x0008, 0)
        else:
            self.ft4222.register_bit_clear(adc_dictionary[channel]['capture'] + 0x0008, 0)

    def channel_get_enable(self, channel):
        return self.ft4222.register_bit_get(adc_dictionary[channel]['capture'] + 0x0008, 0)

    def channel_set_pre_trig_samples(self, channel, num_samples):
        self.ft4222.register_write(adc_dictionary[channel]['capture'] + 0x0009, num_samples)

    def channel_get_pre_trig_samples(self, channel):
        return self.ft4222.register_read(adc_dictionary[channel]['capture'] + 0x0009)

    def channel_set_post_trig_samples(self, channel, num_samples):
        self.ft4222.register_write(adc_dictionary[channel]['capture'] + 0x000A, num_samples)

    def channel_get_post_trig_samples(self, channel):
        return self.ft4222.register_read(adc_dictionary[channel]['capture'] + 0x000A)

    def channel_set_trigger_skew(self, channel, num_samples):
        self.ft4222.register_write(adc_dictionary[channel]['capture'] + 0x000B, num_samples)

    def channel_get_trigger_skew(self, channel):
        return self.ft4222.register_read(adc_dictionary[channel]['capture'] + 0x000B)

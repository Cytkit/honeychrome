from honeychrome.instrument_driver_components.cykit_components.cytkit_configuration import adc_dictionary


class ADCs:
    def __init__(self, ft4222_communicator):
        self.ft4222 = ft4222_communicator

    def capture_configure(self, channel, num_samples):
        register_base = adc_dictionary[channel]['register_base']
        # TBD

    def capture_start(self, channel):
        register_base = adc_dictionary[channel]['register_base']

        # Start the capture
        self.ft4222.register_write(register_base, 0x0001)

    def capture_flush(self, channel):
        register_base = adc_dictionary[channel]['register_base']

        # Start the capture
        self.ft4222.register_write(register_base, 0x0002)

    def get_capture_size(self, channel):
        register_base = adc_dictionary[channel]['register_base']

        # Get the FIFO level
        fifo_level = self.ft4222.register_read(register_base + 0x0002) & 0x3FFF

        return fifo_level

    def fetch_data(self, buffer, max_buffer_size, channel):
        register_base = adc_dictionary[channel]['register_base']

        # Buffer safety check
        if buffer is None:
            return 0

        # Get the FIFO level
        fifo_level = self.ft4222.register_read(register_base + 0x0002) & 0x3FFF

        # Limit check
        if fifo_level > max_buffer_size:
            fifo_level = max_buffer_size

        # Read the samples
        for count in range(fifo_level):
            buffer[count] = self.ft4222.register_read(register_base + 0x0001)

        # Flush any remaining
        self.ft4222.register_write(register_base + 0x0000, 0x0002)

        # Success
        return fifo_level

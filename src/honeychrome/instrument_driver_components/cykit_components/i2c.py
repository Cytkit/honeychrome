import time

from honeychrome.instrument_driver_components.cykit_components.cytkit_configuration import lookup_address


class I2C:
    def __init__(self, ft4222_communicator, base_register, bus_name):
        self.ft4222 = ft4222_communicator
        self.base_address = lookup_address(base_register)
        self.bus_name = bus_name

    def write(self, address, write_buffer, write_size):
        self._write_read(address, write_buffer, write_size, None, 0, 'Write')

    def read(self, address, read_buffer, read_size):
        self._write_read(address, None, 0, read_buffer, read_size, 'Read')

    def write_read(self, address, write_buffer, write_size, read_buffer, read_size):
        self._write_read(address, write_buffer, write_size, read_buffer, read_size, 'WriteRead')

    def _write_read(self, address, write_buffer, write_size, read_buffer, read_size, caller_name):
        if not self.ft4222.is_connected():
            return

        # Check if running when expected to be idle
	    status = self.ft4222.register_read(self.base_address + 0x0000)

        # Flush the FIFOs
        self.ft4222.register_write(self.base_address + 0x0000, 0x0002)

        # Verify the FIFO levels
        status = self.ft4222.register_read(self.base_address + 0x0005)
        status = self.ft4222.register_read(self.base_address + 0x0006)

        # Set the device address
        self.ft4222.register_write(self.base_address + 0x0001, address >> 1)
        # Load the write buffer
        if write_buffer:
            for count in range(write_size):
                self.ft4222.register_write(self.base_address + 0x0003, write_buffer[count])

        # Set the read size
        if read_buffer:
            self.ft4222.register_read(self.base_address + 0x0002, read_size)
        else:
            self.ft4222.register_write(self.base_address + 0x0002, 0)

        # Start the transfer
        self.ft4222.register_write(self.base_address + 0x0000, 0x0001)
        # Wait for completion
        count = 0
        while True:
            status = self.ft4222.register_read(self.base_address + 0x0000)
            time.sleep(1)
            count += 1
            if count > 100000:
                return False

            # Check FIFO level vs. ReadSize
            self.ft4222.register_read(self.base_address + 0x0006)

            status = self.ft4222.register_read(self.base_address + 0x0006)



	Status = DataLink->RegRead(BaseAddress + 0x0006);
    if (ReadSize != Status)
	{
		Log->Printf(MSGLOG_LVL_WARNING, MSGLOG_MASK_I2C, "I2C [Bus %s] %s: Out data FIFO level mis-match (Exp:%u / Act:%u).", BusName, CallerName, ReadSize, Status);
	}
    // Unload the read buffer
    for (Count=0; Count<ReadSize; Count++)
    {
        *ReadBuffer++ = DataLink->RegRead(BaseAddress + 0x0004);
    }

    // Return success
    return true;
    '''
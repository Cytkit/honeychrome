import time

from honeychrome.instrument_driver_components.cykit_components.cytkit_configuration import lookup_address


class I2C:
    def __init__(self, ft4222_communicator, bus_name):
        self.ft4222 = ft4222_communicator
        self.bus_name = bus_name
        if self.bus_name == 'I2C Bus A':
            reg_char = 'A'
        else:
            reg_char = 'B'

        self.reg_ctrl = f'I2C{reg_char}_CTRL'
        self.reg_address = f'I2C{reg_char}_ADDRESS'
        self.reg_read_size = f'I2C{reg_char}_READ_SIZE'
        self.reg_data_in = f'I2C{reg_char}_DATA_IN'
        self.reg_data_out = f'I2C{reg_char}_DATA_OUT'
        self.reg_in_level = f'I2C{reg_char}_IN_LEVEL'
        self.reg_out_level = f'I2C{reg_char}_OUT_LEVEL'


    def write(self, address, write_buffer, write_size):
        self._write_read(address, write_buffer, write_size, None, 0, 'Write')

    def read(self, address, read_buffer, read_size):
        self._write_read(address, None, 0, read_buffer, read_size, 'Read')

    def write_read(self, address, write_buffer, write_size, read_buffer, read_size):
        self._write_read(address, write_buffer, write_size, read_buffer, read_size, 'WriteRead')

    def _write_read(self, address, write_buffer, write_size, read_buffer, read_size, caller_name):
        if not self.ft4222.connected():
            return False

        # Check if running when expected to be idle
        status = self.ft4222.register_read(self.reg_ctrl)

        # Flush the FIFOs
        self.ft4222.register_write(self.reg_ctrl, 0x0002)

        # Verify the FIFO levels
        status = self.ft4222.register_read(self.reg_in_level)
        status = self.ft4222.register_read(self.reg_out_level)

        # Set the device address
        self.ft4222.register_write(self.reg_address, address >> 1)
        # Load the write buffer
        if write_buffer:
            for count in range(write_size):
                self.ft4222.register_write(self.reg_data_in, write_buffer[count])

        # Set the read size
        if read_buffer:
            self.ft4222.register_write(self.reg_read_size, read_size)
        else:
            self.ft4222.register_write(self.reg_read_size, 0)

        # Start the transfer
        self.ft4222.register_write(self.reg_ctrl, 0x0001)
        # Wait for completion
        count = 0
        while True:
            status = self.ft4222.register_read(self.reg_ctrl)
            if status & 0x0010:
                break

            time.sleep(1)
            count += 1
            if count > 100000:
                return False

        # Check FIFO level vs. ReadSize
        status = self.ft4222.register_read(self.reg_out_level)
        if read_size != status:
            print(f'Out data FIFO level {status} mis-match to read size{read_size}')

        # Unload the read buffer
        for count in range(read_size):
            read_buffer[count] = self.ft4222.register_read(self.reg_data_out)

        # Return success
        return True
import ft4222
from ft4222.SPI import Cpha, Cpol
from ft4222.SPIMaster import Mode, Clock, SlaveSelect
from bitarray import bitarray
import numpy as np

from honeychrome.settings import traces_cache_dtype
from honeychrome.instrument_driver_components.cytkit_configuration import operation_register, operation_memory, dummy_bytes, memory_start_address, memory_end_address, registers_map

empty_array = np.array([], dtype=np.uint16)

class CytkitDevice:
    """
    Device driver must provide the following methods:
        connect_to_device
        disconnect
        start_acquisition
        stop_acquisition
        change_device_settings
        read_out_traces
    """
    def __init__(self):
        self.devA = None
        self.devB = None

    def connect_to_device(self):
        num_devices = ft4222.createDeviceInfoList()
        for n in range(num_devices):
            device_info_detail = ft4222.getDeviceInfoDetail(n)

            if device_info_detail['description'] == b'FT4222 A':
                self.devA = ft4222.openByDescription('FT4222 A')

            if device_info_detail['description'] == b'FT4222 B':
                self.devB = ft4222.openByDescription('FT4222 B')

        if self.devA and self.devB:
            self.devA.spiMaster_Init(Mode.QUAD, Clock.DIV_2, Cpol.IDLE_LOW, Cpha.CLK_LEADING, SlaveSelect.SS0)  # for registers
            self.devB.spiMaster_Init(Mode.QUAD, Clock.DIV_2, Cpol.IDLE_LOW, Cpha.CLK_LEADING, SlaveSelect.SS0)  # for memory

            self._configure_instrument()
            print('[Cytkit driver] Connected')
            return {'source':'[Cytkit driver]', 'status':'OK', 'message':'Connected to Cytkit'}

        else:
            raise ConnectionError

    def disconnect(self):
        pass

    def start_acquisition(self):
        pass

    def stop_acquisition(self):
        pass

    def change_device_settings(self, settings):
        pass

    def _register_select(self):
        # call this every time to write to or read from registers
        self.devA.spiMaster_Init(Mode.QUAD, Clock.DIV_2, Cpol.IDLE_LOW, Cpha.CLK_LEADING, SlaveSelect.SS0) # for registers

    def _memory_select(self):
        # call this every time to read from memory
        self.devB.spiMaster_Init(Mode.QUAD, Clock.DIV_2, Cpol.IDLE_LOW, Cpha.CLK_LEADING, SlaveSelect.SS0) # for memory

    def _register_write(self, register_name, data_to_write):
        # self._register_select()
        byte_string = operation_register + registers_map[register_name].to_bytes(2) + dummy_bytes + data_to_write
        self.devA.spiMaster_MultiReadWrite(0, byte_string, 0)

    def _register_read(self, register_name, data_size):
        # self._register_select()
        byte_string = operation_memory + registers_map[register_name].to_bytes(2) + dummy_bytes
        return self.devA.spiMaster_MultiReadWrite(0, byte_string, data_size)

    def _memory_read(self, start_address, total_bytes, chunk_size=65535):
        self._memory_select()

        # read out block of memory in chunks
        data = bytearray(total_bytes)
        bytes_read = 0
        capture_address = start_address
        while bytes_read < total_bytes:
            # Calculate how many bytes to read in this chunk
            remaining = total_bytes - bytes_read
            current_chunk = min(chunk_size, remaining)
            capture_address += current_chunk

            # Write address and read the chunk
            byte_string = operation_memory + capture_address.to_bytes(3) + dummy_bytes
            chunk = self.devA.spiMaster_MultiReadWrite(0, byte_string, current_chunk)
            data.extend(chunk)
            bytes_read += len(chunk)

        return bytes(data)

    def _configure_instrument(self):
        fan_enable = True
        pump_sheath_enable = True
        pump_sample_enable = True
        laser_enable = True
        data_to_write = bitarray([False, False, False, False, fan_enable, pump_sheath_enable, pump_sample_enable, laser_enable]).tobytes()
        self._register_write('ENABLES', data_to_write)

    def _get_memory_head_tail_n_events(self):
        # self._register_select()

        byte_string_to_write = operation_register + registers_map['MEM_ADDR_L'].to_bytes(2) + dummy_bytes
        byte_string_output = self.devA.spiMaster_MultiReadWrite(0, byte_string_to_write, 4)
        memory_head = int.from_bytes(byte_string_output)

        byte_string_to_write = operation_register + registers_map['MEM_ADDR_H'].to_bytes(2) + dummy_bytes
        byte_string_output = self.devA.spiMaster_MultiReadWrite(0, byte_string_to_write, 4)
        memory_tail = int.from_bytes(byte_string_output)

        byte_string_to_write = operation_register + registers_map['MEM_ADDR_U'].to_bytes(2) + dummy_bytes
        byte_string_output = self.devA.spiMaster_MultiReadWrite(0, byte_string_to_write, 4)
        n_events_in_memory = int.from_bytes(byte_string_output)

        return memory_head, memory_tail, n_events_in_memory

    def _pop_from_memory(self, memory_head, memory_tail):
        """
        Read out memory starting at memory_head, keep going until memory_tail read, wrap if necessary
        return numpy array blob
        """
        if memory_tail > memory_head:
            blob_np = np.frombuffer(self._memory_read(memory_head, memory_tail - memory_head), dtype=traces_cache_dtype)
        elif memory_tail < memory_head:
            blob_np = np.concatenate((
                np.frombuffer(self._memory_read(memory_head, memory_end_address - memory_head), dtype=traces_cache_dtype),
                np.frombuffer(self._memory_read(memory_start_address, memory_tail), dtype=traces_cache_dtype)
            ))
        else:
            blob_np = empty_array

        return blob_np

    def read_out_traces(self):
        memory_head, memory_tail, n_events_in_memory = self._get_memory_head_tail_n_events()
        blob_of_traces_as_array = self._pop_from_memory(memory_head, memory_tail)
        return blob_of_traces_as_array



if __name__ == '__main__':

    cytkit_device = CytkitDevice()
    cytkit_device.connect_to_device()
    cytkit_device.start_acquisition()
    blob = cytkit_device.read_out_traces()

    print(blob.shape)

    cytkit_device.disconnect()


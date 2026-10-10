import re
import threading
import time

import numpy as np
import ft4222
from ft4222.SPI import Cpha, Cpol
from ft4222.SPIMaster import Mode, Clock, SlaveSelect

from honeychrome.instrument_driver_components.cytkit_components.cytkit_configuration import operation_write, operation_read, dummy_bytes, lookup_address
from honeychrome.settings import traces_cache_dtype

empty_array = np.array([], dtype=np.uint16)


class Ft4222Communicator:
    def __init__(self):
        self.devA = None
        self.devB = None
        self.buffer = 0
        self.bytes_read = 0
        # Serialises all device access between the command thread, the transfer thread and the worker threads.
        # Re-entrant so a read-modify-write (or a level read + data read) can hold it across several calls.
        self.lock = threading.RLock()

    def connected(self):
        return self.devA and self.devB

    def find_and_connect(self):
        num_devices = ft4222.createDeviceInfoList()
        for n in range(num_devices):
            device_info_detail = ft4222.getDeviceInfoDetail(n)

            if device_info_detail['description'] == b'FT4222 A':
                self.devA = ft4222.openByDescription('FT4222 A')

            if device_info_detail['description'] == b'FT4222 B':
                self.devB = ft4222.openByDescription('FT4222 B')

        if self.devA and self.devB:
            self._register_init()
            self._memory_init()

            return True

        else:
            raise ConnectionError(f'Connection did not succeed FT4222 A: {self.devA}, and B: {self.devB}')

    def _register_init(self):
        self.devA.spiMaster_Init(Mode.QUAD, Clock.DIV_4, Cpol.IDLE_LOW, Cpha.CLK_LEADING, SlaveSelect.SS0 | SlaveSelect.SS1 | SlaveSelect.SS2 | SlaveSelect.SS3) # for registers

    def _memory_init(self):
        self.devB.spiMaster_Init(Mode.QUAD, Clock.DIV_4, Cpol.IDLE_LOW, Cpha.CLK_LEADING, SlaveSelect.SS0 | SlaveSelect.SS1 | SlaveSelect.SS2 | SlaveSelect.SS3) # for memory

    def register_write(self, register, data_to_write):
        if type(data_to_write) != int:
            raise TypeError

        byte_string = operation_write + lookup_address(register) + dummy_bytes + data_to_write.to_bytes(2, byteorder='big')
        with self.lock:
            self.devA.spiMaster_MultiReadWrite(b'', byte_string, 0)

    def register_read(self, register):
        byte_string = operation_read + lookup_address(register)
        with self.lock:
            data_read = self.devA.spiMaster_MultiReadWrite(b'', byte_string, 4) # 2 bytes dummy, 2 bytes register
        return int.from_bytes(data_read[2:], byteorder='big', signed=False)

    def register_2reg_write(self, register_low, register_high, data_to_write):
        with self.lock:
            self.register_write(register_low, (data_to_write >> 0) & 0xFFFF)
            self.register_write(register_high, (data_to_write >> 16) & 0xFFFF)

    def register_2reg_read(self, register_low, register_high):
        with self.lock:
            value = self.register_read(register_high) << 16
            value |= self.register_read(register_low)
        return value

    def register_read_modify_write(self, register_name, value, mask):
        # hold the lock across the read and the write so the RMW is atomic
        with self.lock:
            working_value = self.register_read(register_name)
            working_value &= ~mask # sets masked bits to zero, keeps all other bits
            working_value |= (value & mask) # sets masked bits to value, keeps all other bits
            self.register_write(register_name, working_value)


    def register_bit_set(self, address, bit_pos):
        # Set the specified bit in a register
        if bit_pos >= 16:
            return
        self.register_read_modify_write(address, 1 << bit_pos, 1 << bit_pos)

    def register_bit_clear(self, address, bit_pos):
        # Clear the specified bit in a register
        if bit_pos >= 16:
            return
        self.register_read_modify_write(address, 0x0000, 1 << bit_pos)

    def register_bit_get(self, address, bit_pos):
        return bool(self.register_read(address) & (1 << bit_pos))


    def sample_read_buffer(self, total_bytes, chunk_size=65535):
        # read out block of memory in chunks
        data = bytearray()
        bytes_read = 0
        while bytes_read < total_bytes:
            # Calculate how many bytes to read in this chunk
            remaining = total_bytes - bytes_read
            current_chunk = min(chunk_size, remaining)

            # Write three bytes and read the current chunk
            chunk = self.devB.spiMaster_MultiReadWrite(b'', b'', current_chunk)
            data.extend(chunk)
            bytes_read += len(chunk)

        return bytes(data)

    def pop_from_memory(self):
        buffer = None
        # hold the lock across the level read and the data read, so the FIFO can't be cleared in between
        with self.lock:
            fifo_words = self.register_read('BULK_LEVEL')
            if fifo_words > 0:
                bytes_to_read = fifo_words * 2
                buffer = self.sample_read_buffer(bytes_to_read)

        return buffer

    # def memory_read(self, total_bytes, chunk_size=65535):
    #     # read out block of memory in chunks
    #     data = bytearray(total_bytes)
    #     bytes_read = 0
    #     while bytes_read < total_bytes:
    #         # Calculate how many bytes to read in this chunk
    #         remaining = total_bytes - bytes_read
    #         current_chunk = min(chunk_size, remaining)
    #
    #         # Write address and read the chunk
    #         chunk = self.devB.spiMaster_MultiReadWrite(b'', b'', current_chunk)
    #         data.extend(chunk)
    #         bytes_read += len(chunk)
    #
    #     return bytes(data)
    #
    #
    # def get_memory_head_tail_n_events(self):
    #
    #     byte_string_to_write = operation_write + registers_map['MEM_ADDR_L'].to_bytes(2) + dummy_bytes
    #     byte_string_output = self.devA.spiMaster_MultiReadWrite(0, byte_string_to_write, 4)
    #     memory_head = int.from_bytes(byte_string_output)
    #
    #     byte_string_to_write = operation_write + registers_map['MEM_ADDR_H'].to_bytes(2) + dummy_bytes
    #     byte_string_output = self.devA.spiMaster_MultiReadWrite(0, byte_string_to_write, 4)
    #     memory_tail = int.from_bytes(byte_string_output)
    #
    #     byte_string_to_write = operation_write + registers_map['MEM_ADDR_U'].to_bytes(2) + dummy_bytes
    #     byte_string_output = self.devA.spiMaster_MultiReadWrite(0, byte_string_to_write, 4)
    #     n_events_in_memory = int.from_bytes(byte_string_output)
    #
    #     return memory_head, memory_tail, n_events_in_memory
    #
    # def pop_from_memory(self, memory_head, memory_tail):
    #     """
    #     Read out memory starting at memory_head, keep going until memory_tail read, wrap if necessary
    #     return numpy array blob
    #     """
    #     if memory_tail > memory_head:
    #         blob_np = np.frombuffer(self.memory_read(memory_head, memory_tail - memory_head), dtype=traces_cache_dtype)
    #     elif memory_tail < memory_head:
    #         blob_np = np.concatenate((
    #             np.frombuffer(self.memory_read(memory_head, memory_end_address - memory_head), dtype=traces_cache_dtype),
    #             np.frombuffer(self.memory_read(memory_start_address, memory_tail), dtype=traces_cache_dtype)
    #         ))
    #     else:
    #         blob_np = empty_array
    #
    #     return blob_np

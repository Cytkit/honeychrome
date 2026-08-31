MON_ID_P_36_0V = 0
MON_ID_P_36_0V_BIAS = 1
MON_ID_P_12_0V = 2
MON_ID_P_12_0V_BIAS = 3
MON_ID_P_5_0V = 4
MON_ID_P_5_0V_BIAS = 5
MON_ID_P_3_3V = 6
MON_ID_P_1_8V = 7

class VIMonitor:
    def __init__(self, ft4222_communicator, I2CBusA, I2CBusB):
        self.ft4222_communicator = ft4222_communicator
        self.I2CBusA = I2CBusA
        self.I2CBusB = I2CBusB

    def disconnect(self):
        pass

    def read_voltage(self, channel):
        return_value = self._read_raw_value(channel, 0x02)
        return return_value * 0.0016

    def read_current(self, channel):
        return_value = self._read_raw_value(channel, 0x01)
        return return_value * 0.00025

    def _read_raw_value(self, channel, command):
        if channel == MON_ID_P_36_0V:
            bus = self.I2CBusA
            i2_c_address = 0x80
        elif channel == MON_ID_P_36_0V_BIAS:
            bus = self.I2CBusA
            i2_c_address = 0x82
        elif channel == MON_ID_P_12_0V:
            bus = self.I2CBusA
            i2_c_address = 0x84
        elif channel == MON_ID_P_12_0V_BIAS:
            bus = self.I2CBusA
            i2_c_address = 0x86
        elif channel == MON_ID_P_5_0V:
            bus = self.I2CBusB
            i2_c_address = 0x80
        elif channel == MON_ID_P_5_0V_BIAS:
            bus = self.I2CBusB
            i2_c_address = 0x82
        elif channel == MON_ID_P_3_3V:
            bus = self.I2CBusB
            i2_c_address = 0x84
        elif channel == MON_ID_P_1_8V:
            bus = self.I2CBusB
            i2_c_address = 0x86
        else:
            bus = None
            i2_c_address = None


        # Set the command
        write_buffer = [command]
        read_buffer = [0, 0]

        # Send it
        if not bus.write_read(
                i2_c_address,
                write_buffer,
                1,
                read_buffer,
                2):
            return 0

        # Return the response
        ret_value = read_buffer[0]
        ret_value <<= 8
        ret_value |= read_buffer[1]
        return ret_value
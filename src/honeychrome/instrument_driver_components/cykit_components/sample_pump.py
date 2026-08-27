class SamplePump:
    def __init__(self, ft4222_communicator):
        self.ft4222 = ft4222_communicator

    def disconnect(self):
        self.set_enable(False)

    def set_enable(self, enable):
        # enable is bit 0
        mask = 0x0001
        if enable:
            value = 0x0001
        else:
            value = 0x0000

        self.ft4222.register_read_modify_write('SMPMP_CTRL', value, mask)

    def get_enable(self):
        mask = 0x0001
        return self.ft4222.register_read('SMPMP_CTRL') & mask == 1


    def set_reverse(self, reverse):
        # direction is bit 4
        mask = 0x0010
        if reverse:
            value = 0x0010
        else:
            value = 0x0000

        self.ft4222.register_read_modify_write('SMPMP_CTRL', value, mask)

    def set_reverse(self):
        mask = 0x0010
        return self.ft4222.register_read('SMPMP_CTRL') & mask == 1

    def set_ramp(self, ramp):
        # direction is bit 4
        mask = 0x1000
        if ramp:
            value = 0x1000
        else:
            value = 0x0000

        self.ft4222.register_read_modify_write('SMPMP_CTRL', value, mask)

    def set_ramp(self):
        mask = 0x1000
        return self.ft4222.register_read('SMPMP_CTRL') & mask == 1


    def set_speed(self, speed):
        self.ft4222.register_write('SMPMP_SPEED', speed)

    def get_speed(self):
        return self.ft4222.register_read('SMPMP_SPEED')


    def set_steps_per_cycle(self, steps_per_cycle):
        self.ft4222.register_write('SMPMP_SPC', steps_per_cycle)

    def get_steps_per_cycle(self):
        return self.ft4222.register_read('SMPMP_SPC')

    def set_clocks_per_cycle(self, clocks_per_cycle):
        self.ft4222.register_2byte_write('SMPMP_CPC_L', 'SMPMP_CPC_H', clocks_per_cycle)

    def get_clocks_per_cycle(self):
        return self.ft4222.register_2byte_read('SMPMP_CPC_L', 'SMPMP_CPC_H')

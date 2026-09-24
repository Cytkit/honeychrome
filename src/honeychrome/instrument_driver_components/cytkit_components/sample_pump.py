import time


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

    def get_reverse(self):
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

    def get_ramp(self):
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

    def ramp_to(self, speed):
        reverse = speed < 0
        current_speed = self.get_speed()
        current_ramp = self.get_ramp()
        current_reverse = self.get_reverse()
        current_enable = self.get_enable()

        change_direction = current_reverse != reverse

        stage_speed = min(abs(speed), 10000)

        if change_direction or not current_enable:
            # disable first, then ramp from zero
            self.set_enable(False)
            self.set_ramp(False)
            self.set_speed(0)
            self.set_reverse(reverse)
            self.set_ramp(True)
            self.set_enable(True)
            self.set_speed(stage_speed)
        else:
            # not starting from zero, pump is currently enabled, and direction is not changing
            self.set_speed(stage_speed)

        # for n in range(10):
        #     print(self.get_speed(), bin(self.ft4222.register_read('SMPMP_CTRL')))
        #     time.sleep(0.0001)
        #
        # while abs(speed) != stage_speed:
        #     stage_speed = min(abs(speed), stage_speed + 10000)
        #     print(stage_speed)
        #     self.set_speed(stage_speed)
        #     time.sleep(1)
        #
        # time.sleep(1)
        # if speed == 0:
        #     self.set_enable(False)

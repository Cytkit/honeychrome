import time

from honeychrome.instrument_driver_components.cytkit_components.cytkit_configuration import min_speed, max_speed


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
        mask = 0b0000_0000_0001_0000
        if reverse:
            value = 0b0000_0000_0001_0000
        else:
            value = 0b0000_0000_0000_0000

        self.ft4222.register_read_modify_write('SMPMP_CTRL', value, mask)

    def get_reverse(self):
        mask = 0b0000_0000_0001_0000
        return self.ft4222.register_read('SMPMP_CTRL') & mask == 1

    def set_ramp(self, ramp):
        # ramp is bit 12
        mask = 0b0001_0000_0000_0000
        if ramp:
            value = 0b0001_0000_0000_0000
        else:
            value = 0b0000_0000_0000_0000

        self.ft4222.register_read_modify_write('SMPMP_CTRL', value, mask)

    def get_ramp(self):
        mask = 0b0001_0000_0000_0000
        return self.ft4222.register_read('SMPMP_CTRL') & mask == 1

    def get_ramp_complete(self):
        # ramp complete is bit 13
        mask = 0b0010_0000_0000_0000
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
        self.ft4222.register_2reg_write('SMPMP_CPC_L', 'SMPMP_CPC_H', clocks_per_cycle)

    def get_clocks_per_cycle(self):
        return self.ft4222.register_2reg_read('SMPMP_CPC_L', 'SMPMP_CPC_H')

    def ramp_to(self, speed):
        reverse = speed < 0
        # guard against speed 0
        if speed == 0:
            speed = min_speed
        current_speed = self.get_speed()
        current_ramp = self.get_ramp()
        current_reverse = self.get_reverse()
        current_enable = self.get_enable()

        change_direction = current_reverse != reverse

        stage_speed = min(abs(speed), max_speed)

        if change_direction or not current_enable:
            # stop then ramp
            self.stop()
            self.set_reverse(reverse)
            self.set_enable(True)
            self.set_ramp(True)
            self.set_speed(stage_speed)
        else:
            # not starting from zero, pump is currently enabled, and direction is not changing
            self.set_ramp(True)
            self.set_speed(stage_speed)

        time_start = time.perf_counter()
        while not self.get_ramp_complete() and not time.perf_counter() - time_start > 2:
            # print(self.get_speed(), bin(self.ft4222.register_read('SMPMP_CTRL')))
            time.sleep(0.001)

        # while abs(speed) != stage_speed:
        #     stage_speed = min(abs(speed), stage_speed + max_speed)
        #     print(stage_speed)
        #     self.set_speed(stage_speed)
        #     time.sleep(0.001)

        if speed == min_speed:
            self.stop()

    def stop(self):
        self.set_enable(False)
        self.set_ramp(False)
        self.set_speed(min_speed)

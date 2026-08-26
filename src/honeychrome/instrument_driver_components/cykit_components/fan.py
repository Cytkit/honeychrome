class Fan:
    def __init__(self, ft4222_communicator):
        self.ft4222 = ft4222_communicator

    def disconnect(self):
        self.set_enable(False)

    def set_enable(self, enable):
        self.ft4222.register_write('FAN_CTRL', 1)

    def get_enable(self):
        return self.ft4222.register_read('FAN_CTRL') == 1

    def set_pwm_frequency(self, frequency):
        self.ft4222.register_write('FAN_FREQ', frequency)

    def get_pwm_frequency(self):
        return self.ft4222.register_read('FAN_FREQ')

    def set_pwm_duty(self, duty):
        self.ft4222.register_write('FAN_DUTY', duty)

    def get_pwm_duty(self):
        return self.ft4222.register_read('FAN_DUTY')

    def get_tacho(self):
        return self.ft4222.register_read('FAN_TACHO')

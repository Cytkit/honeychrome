class Laser:
    def __init__(self, ft4222_communicator):
        self.ft4222 = ft4222_communicator

    def disconnect(self):
        self.set_state(0) # switch off

    def set_state(self, state):
        self.ft4222.register_write('LASER', int(state))

    def get_state(self):
        return self.ft4222.register_read('LASER') == 1 # boolean output

    def get_interlock_mask(self):
        return self.ft4222.register_read('INT_MASK') & 0x0001

    def set_interlock_mask(self, state):
        self.ft4222.register_write('INT_MASK', int(state))

    def get_interlock_inversion(self):
        return self.ft4222.register_read('INT_INV') & 0x0001

    def set_interlock_inversion(self, inversion):
        self.ft4222.register_write('INT_INV', int(inversion))

    def get_interlock_state(self):
        return self.ft4222.register_read('INT_STATE') & 0x0001

    def set_interlock_effects(self, force=False, stop_fan=False, stop_sheath=False, stop_sample=False, stop_laser=True):
        state = (force << 15) + (stop_fan << 3) + (stop_sheath << 2) + (stop_sample << 1) + (stop_laser<<0)
        self.ft4222.register_write('INT_MISC', state)

    def get_interlock_effects(self):
        state = self.ft4222.register_read('INT_MISC')
        force = state & 0b1000000000000000
        stop_fan = state & 0b0000000000001000
        stop_sheath = state & 0b0000000000000100
        stop_sample = state & 0b0000000000000010
        stop_laser = state & 0b0000000000000001
        return force, stop_fan, stop_sheath, stop_sample, stop_laser

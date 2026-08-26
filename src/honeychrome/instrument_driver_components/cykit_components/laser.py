class Laser:
    def __init__(self, ft4222_communicator):
        self.ft4222 = ft4222_communicator

    def disconnect(self):
        self.set_state(0) # switch off

    def set_state(self, state):
        self.ft4222.register_write('LASER', state)

    def get_state(self):
        return self.ft4222.register_read('LASER') == 1 # boolean output
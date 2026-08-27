class Pressure:
    def __init__(self, ft4222_communicator):
        self.ft4222 = ft4222_communicator

    def connect(self):
        self.readback_offset = 0.0
        self._sensor_reset()
        self._sensor_load_coeff_data()
        self._sensor_load_cal_data()

    def disconnect(self):
    def get_pressure(self):
    def get_temperature(self):
    def set_offset(self, offset, units):
    def get_offset(self, units):

    def _sensor_reset(self):
    def _sensor_load_coeff_data(self):
    def _sensor_load_cal_data(self):
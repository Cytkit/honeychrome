import time

from threading import Thread, Event, Lock

from honeychrome.instrument_driver_components.cykit_components.adcs import ADCs
from honeychrome.instrument_driver_components.cykit_components.cytkit_configuration import registers_map, monitor_dictionary, dac_dictionary, pump_max, control_loop_interval
from honeychrome.instrument_driver_components.cykit_components.dacs import DACs
from honeychrome.instrument_driver_components.cykit_components.ft4222communicator import Ft4222Communicator
from honeychrome.instrument_driver_components.cykit_components.fan import Fan
from honeychrome.instrument_driver_components.cykit_components.i2c import I2C
from honeychrome.instrument_driver_components.cykit_components.id_data import IDData
from honeychrome.instrument_driver_components.cykit_components.laser import Laser
from honeychrome.instrument_driver_components.cykit_components.pressure import Pressure
from honeychrome.instrument_driver_components.cykit_components.sample_pump import SamplePump
from honeychrome.instrument_driver_components.cykit_components.sheath_pump import SheathPump
from honeychrome.instrument_driver_components.cykit_components.vi_monitor import VIMonitor
from honeychrome.settings import pressure_set_point


class PID:
    '''
    pid = PID(kp=1.0, ki=0.1, kd=0.05, out_min=0, out_max=255)
    setpoint = 100
    current = 0

    for _ in range(100):
        error = setpoint - current
        output = pid.update(error, dt=0.1)
        current += output  # simple plant model
    '''
    def __init__(self, kp, ki, kd, out_min, out_max):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.integral = 0
        self.prev_error = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = (self.kp * error
                  + self.ki * self.integral
                  + self.kd * derivative)

        # Clamp output and prevent integral windup
        clamped = max(self.out_min, min(self.out_max, output))
        if clamped != output:
            self.integral -= error * dt  # undo integration

        self.prev_error = error
        return int(clamped)

class PressureControlWorker(Thread):
    '''
    w = Worker(0.5)
    w.start()
    time.sleep(3)
    w.stop()
    w.join()
    '''
    def __init__(self, pressure, sheath_pump, pressure_set_point, pump_max, interval):
        super().__init__(daemon=True)
        self.interval = interval
        self.value = 0
        self._stop_event = Event()
        self._lock = Lock()
        self.pressure = pressure
        self.sheath_pump = sheath_pump

        self.pid = PID(kp=1.0, ki=0.1, kd=0.05, out_min=0, out_max=pump_max)
        self.pressure_set_point = pressure_set_point
        self.current_pressure = self.pressure.get_pressure('PRES_UNITS_PA', 1)

    def run(self):
        while not self._stop_event.is_set():
            with self._lock:
                error = self.pressure_set_point - self.current_pressure
                output = self.pid.update(error, dt=self.interval)
                self.sheath_pump.set_pwm_duty(output)

            print(f"[PressureControlWorker] error={error} output={output}")
            self._stop_event.wait(self.interval)   # interruptible sleep

    def stop(self):
        self._stop_event.set()


class TemperatureControlWorker(Thread):
    '''
    w = Worker(0.5)
    w.start()
    time.sleep(3)
    w.stop()
    w.join()
    '''
    def __init__(self, temperature, fan, temperature_set_point, fan_max, interval):
        super().__init__(daemon=True)
        self.fan_max = fan_max
        self.interval = interval
        self.value = 0
        self._stop_event = Event()
        self._lock = Lock()
        self.temperature = temperature
        self.fan = fan

        self.pid = PID(kp=1.0, ki=0.1, kd=0.05, out_min=0, out_max=pump_max)
        self.temperature_set_point = temperature_set_point
        self.current_temperature = self.temperature.get_temperature()

    def run(self):
        while not self._stop_event.is_set():
            with self._lock:
                error = self.temperature_set_point - self.current_temperature
                output = self.pid.update(error, dt=self.interval)
                self.fan.set_pwm_duty(output)

            print(f"[TemperatureControlWorker] error={error} output={output}")
            self._stop_event.wait(self.interval)   # interruptible sleep

    def stop(self):
        self._stop_event.set()

class CytkitDevice:
    """
    Device driver must provide the following methods:
        find_and_connect_to_device
            no arguments
            return status, message
        disconnect
            no arguments
            no return
        initialise
            no arguments
            return status, message
        start_acquisition
            no arguments
            return status, message
        stop_acquisition
            no arguments
            return status, message
        get_state
            argument: list of parameters to get, if list is empty gets everything
            return status, message (where message is dict of parameters)
        set_state
            argument: dict of parameters to set
            return status, message
        flush_sip
            no arguments
            return status, message
        backflush_sip
            no arguments
            return status, message
        read_out_traces
            no arguments
            returns blob of traces
    """
    def __init__(self):
        self.name = 'Cytkit'
        self.ft4222 = Ft4222Communicator()
        self.id_data = None
        self.fan = None
        self.laser = None
        self.pressure = None
        self.sample_pump = None
        self.sheath_pump = None
        self.i2c_bus_a = None
        self.i2c_bus_b = None
        self.vi_monitor = None
        self.dacs = None
        self.adcs = None
        self.pressure_control_worker = None
        self.temperature_control_worker = None
        self.initialised = False

    def find_and_connect_to_device(self):
        self.ft4222.find_and_connect()

        # if the connection above worked, initialise the hardware wrappers
        self.id_data = IDData(self.ft4222)
        self.fan = Fan(self.ft4222)
        self.laser = Laser(self.ft4222)
        self.pressure = Pressure(self.ft4222)
        self.sample_pump = SamplePump(self.ft4222)
        self.sheath_pump = SheathPump(self.ft4222)
        self.i2c_bus_a = I2C(self.ft4222, 'I2C Bus A')
        self.i2c_bus_b = I2C(self.ft4222, 'I2C Bus B')
        self.vi_monitor = VIMonitor(self.i2c_bus_a, self.i2c_bus_b)
        self.dacs = DACs(self.i2c_bus_a)
        self.adcs = ADCs(self.ft4222)

        print('[Cytkit driver] Connected')

        # set initial settings
        self.sample_pump.set_ramp(True)
        self.pressure_control_worker = PressureControlWorker(self.pressure, self.sheath_pump, pressure_set_point, pump_max, control_loop_interval)
        self.temperature_control_worker = PressureControlWorker(self.pressure, self.sheath_pump, pressure_set_point, pump_max, control_loop_interval)
        self.temperature_control_worker.start()
        return  'OK', 'Connected to Cytkit'

    def disconnect(self):
        self.temperature_control_worker.stop()
        self.pressure_control_worker.stop()
        self.temperature_control_worker.join()
        self.pressure_control_worker.join()

    def initialise(self):
        # id_word = self.ft4222.register_read('ID_WORD')
        # print(id_word)

        if not self.initialised:
            self.laser.set_state(1)  # turn on laser
            self.pressure_control_worker.start()
            self.initialised = True
            return 'OK', 'Cytkit initialised'
        else:
            self.laser.set_state(0)  # turn off laser
            self.pressure_control_worker.stop()
            self.initialised = False
            return 'OK', 'Cytkit on standby'


    def start_acquisition(self):
        return 'OK', 'Cytkit started acquisition'

    def stop_acquisition(self):
        return 'OK', 'Cytkit stopped acquisition'

    def set_state(self, dict_of_parameter_value):
        message = {}
        for parameter, value in dict_of_parameter_value.items():
            if parameter == 'laser_enable':
                self.laser.set_state(value)
                message['laser_enable'] = value

            if parameter == 'fan_state':
                if 'freq' in value:
                    self.fan.set_pwm_frequency(value['freq'])
                if 'duty' in value:
                    self.fan.set_pwm_duty(value['duty'])
                if 'enable' in value:
                    self.fan.set_enable(value['enable'])
                message['fan_state'] = value

            if parameter == 'sheath_pump_state':
                if 'freq' in value:
                    self.sheath_pump.set_pwm_frequency(value['freq'])
                if 'duty' in value:
                    self.sheath_pump.set_pwm_duty(value['duty'])
                if 'enable' in value:
                    self.sheath_pump.set_enable(value['enable'])
                message['sheath_pump_state'] = value

            if parameter == 'sample_pump_state':
                if 'enable' in value:
                    self.sample_pump.set_enable(value['enable'])
                if 'reverse' in value:
                    self.sample_pump.set_reverse(value['reverse'])
                if 'ramp' in value:
                    self.sample_pump.set_ramp(value['ramp'])
                if 'speed' in value:
                    self.sample_pump.set_speed(value['speed'])
                if 'steps_per_cycle' in value:
                    self.sample_pump.set_steps_per_cycle(value['steps_per_cycle'])
                if 'clocks_per_cycle' in value:
                    self.sample_pump.set_clocks_per_cycle(value['clocks_per_cycle'])
                message['sample_pump_state'] = value

            if parameter == 'dacs':
                if 'bias' in value:
                    for index in value['bias']:
                        self.dacs.set_value_bias(index, value['bias'][index])
                if 'ref' in value:
                    for index in value['ref']:
                        self.dacs.set_value_ref(index, value['ref'][index])
                message['dacs'] = value

            if parameter == 'register_setter':
                try:
                    x = int(value['value'])
                except ValueError:
                    x = 0

                self.ft4222.register_write(value['register'], x)
                message['register_setter'] = {'register': value['register'], 'value': x}

        return 'OK', message

    def get_state(self, list_of_parameters):
        if not list_of_parameters: # if no list, get everything
            list_of_parameters = [
                'check_connection',
                'read_id_data',
                'pressure',
                'temperatures',
                'vi_monitors',
                'fan_state',
                'sheath_pump_state',
                'sample_pump_state',
                'dacs'
            ]

        message = {}
        if 'check_connection' in list_of_parameters:
            connected = self.id_data.check_connection()
            message['check_connection'] = connected

        if 'read_id_data' in list_of_parameters:
            version, datetime = self.id_data.read_id_data()
            message['read_id_data'] = {'version':version, 'datetime':datetime}

        if 'pressure'in list_of_parameters:
            value = self.pressure.get_pressure('PRES_UNITS_PA', 1)
            message['pressure'] = value

        if 'zero_pressure'in list_of_parameters:
            value = self.pressure.get_pressure('PRES_UNITS_PA', 10)
            self.pressure.set_offset(value, 'PRES_UNITS_PA')
            value = self.pressure.get_pressure('PRES_UNITS_PA', 1)
            message['pressure'] = value

        if 'temperatures' in list_of_parameters:
            message['temperatures'] = {}
            message['temperatures']['temp_p_sensor'] = self.pressure.get_temperature()

        if 'vi_monitors' in list_of_parameters:
            message['vi_monitors'] = {'V':{}, 'I':{}}
            for channel in monitor_dictionary:
                V = self.vi_monitor.read_voltage(channel)
                I = self.vi_monitor.read_current(channel)
                message['vi_monitors']['V'][channel] = V
                message['vi_monitors']['I'][channel] = I

        if 'fan_state'in list_of_parameters:
            message['fan_state'] = {}
            enable = self.fan.get_enable()
            message['fan_state']['enable'] = enable
            freq = self.fan.get_pwm_frequency()
            message['fan_state']['freq'] = freq
            duty = self.fan.get_pwm_duty()
            message['fan_state']['duty'] = duty
            tacho = self.fan.get_tacho()
            message['fan_state']['tacho'] = tacho

        if 'sheath_pump_state'in list_of_parameters:
            message['sheath_pump_state'] = {}
            enable = self.sheath_pump.get_enable()
            message['sheath_pump_state']['enable'] = enable
            freq = self.sheath_pump.get_pwm_frequency()
            message['sheath_pump_state']['freq'] = freq
            duty = self.sheath_pump.get_pwm_duty()
            message['sheath_pump_state']['duty'] = duty

        if 'sample_pump_state' in list_of_parameters:
            message['sample_pump_state'] = {}
            enable = self.sample_pump.get_enable()
            message['sample_pump_state']['enable'] = enable
            reverse = self.sample_pump.get_reverse()
            message['sample_pump_state']['reverse'] = reverse
            ramp = self.sample_pump.get_ramp()
            message['sample_pump_state']['ramp'] = ramp
            speed = self.sample_pump.get_speed()
            message['sample_pump_state']['speed'] = speed
            steps_per_cycle = self.sample_pump.get_steps_per_cycle()
            message['sample_pump_state']['steps_per_cycle'] = steps_per_cycle
            clocks_per_cycle = self.sample_pump.get_clocks_per_cycle()
            message['sample_pump_state']['clocks_per_cycle'] = clocks_per_cycle

        if 'dacs' in list_of_parameters:
            message['dacs'] = {'bias':{}, 'ref':{}}
            for index in range(16):
                message['dacs']['bias'][index] = self.dacs.get_value_bias(index)
                message['dacs']['ref'][index] = self.dacs.get_value_ref(index)

        if 'register_getter' in list_of_parameters:
            if type(list_of_parameters) is dict:
                if type(list_of_parameters['register_getter']) is int:
                    value = self.ft4222.register_read(message['register_getter'])
                    message['register_getter'] = value

        return 'OK', message

    def flush_sip(self):
        return 'OK', 'Cytkit SIP flushed'

    def backflush_sip(self):
        return 'OK', 'Cytkit SIP backflushed'


    def read_out_traces(self):
        memory_head, memory_tail, n_events_in_memory = self.ft4222.get_memory_head_tail_n_events()
        blob_of_traces_as_array = self.ft4222.pop_from_memory(memory_head, memory_tail)
        return blob_of_traces_as_array


if __name__ == '__main__':
    # test connection and id
    cytkit_device = CytkitDevice()
    print(cytkit_device.find_and_connect_to_device())
    print(cytkit_device.get_state(['check_connection']))

    print(cytkit_device.get_state(['read_id_data']))

    print('test laser')
    print(cytkit_device.set_state({'laser_enable' : True}))

    print('test pressure')
    print(cytkit_device.get_state(['pressure']))

    print('test temperatures')
    print(cytkit_device.get_state(['temperatures'])) # not yet working

    print('test vi monitors')
    print(cytkit_device.get_state(['vi_monitors'])) # not yet working

    print('test fan')
    print(cytkit_device.get_state(['fan_state']))
    print(cytkit_device.set_state({'fan_state': {'enable': True, 'freq': 100, 'duty': 128}}))
    print(cytkit_device.get_state(['fan_state']))

    print('test sheath pump')
    print(cytkit_device.set_state({'sheath_pump_state': {'enable': True, 'freq': 100, 'duty': 128}}))
    print(cytkit_device.get_state(['sheath_pump_state']))

    print('test sample pump')
    print(cytkit_device.set_state({'sample_pump_state': {'enable': True, 'reverse': False, 'ramp': True, 'speed': 6000, 'steps_per_cycle': 1, 'clocks_per_cycle': 100_000}}))
    print(cytkit_device.get_state(['sample_pump_state']))

    print('test dacs')
    print(cytkit_device.set_state({'dacs': {'bias': {index:50+index for index in range(16)}, 'ref': {index:20-index for index in range(16)}}}))
    print(cytkit_device.get_state(['dacs']))

    # test adcs

    time.sleep(2)
    print(cytkit_device.set_state({'laser_enable' : False}))
    print(cytkit_device.set_state({'sheath_pump_state': {'enable': False}}))
    print(cytkit_device.set_state({'sample_pump_state': {'enable': False}}))
    print(cytkit_device.set_state({'fan_state': {'enable': False}}))

    # # read traces
    # cytkit_device.start_acquisition()
    # blob = cytkit_device.read_out_traces()
    # print(blob.shape)

    cytkit_device.disconnect()

'''
Example output:

[Cytkit driver] Connected
('OK', 'Connected to Cytkit')
('OK', {'check_connection': True})
('OK', {'read_id_data': {'version': 152, 'datetime': datetime.datetime(2026, 9, 7, 21, 41, 53)}})
test laser
('OK', {'laser_enable': True})
test pressure
('OK', {'pressure': -44.38457712241033})
test temperatures
('OK', {'temperatures': {'temp_p_sensor': 25.0246448802948}})
test vi monitors
('OK', {'vi_monitors': {'V': {0: 31.556800000000003, 1: 31.553600000000003, 2: 11.942400000000001, 3: 11.939200000000001, 4: 5.0016, 5: 5.0, 6: 3.3104, 7: 1.8064}, 'I': {0: 16.206500000000002, 1: 16.37675, 2: 16.233, 3: 16.38175, 4: 16.0685, 5: 16.38325, 6: 16.26775, 7: 16.295}}})
test fan
('OK', {'fan_state': {'enable': False, 'freq': 100, 'duty': 128, 'tacho': 0}})
('OK', {'fan_state': {'enable': True, 'freq': 100, 'duty': 128}})
('OK', {'fan_state': {'enable': True, 'freq': 100, 'duty': 128, 'tacho': 0}})
test sheath pump
('OK', {'sheath_pump_state': {'enable': True, 'freq': 100, 'duty': 128}})
('OK', {'sheath_pump_state': {'enable': True, 'freq': 100, 'duty': 128}})
test sample pump
('OK', {'sample_pump_state': {'enable': True, 'reverse': False, 'ramp': True, 'speed': 6000, 'steps_per_cycle': 1, 'clocks_per_cycle': 100000}})
('OK', {'sample_pump_state': {'enable': True, 'reverse': False, 'ramp': False, 'speed': 6000, 'steps_per_cycle': 1, 'clocks_per_cycle': 100000}})
test dacs
('OK', {'dacs': {'bias': {0: 50, 1: 51, 2: 52, 3: 53, 4: 54, 5: 55, 6: 56, 7: 57, 8: 58, 9: 59, 10: 60, 11: 61, 12: 62, 13: 63, 14: 64, 15: 65}, 'ref': {0: 20, 1: 19, 2: 18, 3: 17, 4: 16, 5: 15, 6: 14, 7: 13, 8: 12, 9: 11, 10: 10, 11: 9, 12: 8, 13: 7, 14: 6, 15: 5}}})
('OK', {'dacs': {'bias': {0: 50, 1: 51, 2: 52, 3: 53, 4: 54, 5: 55, 6: 56, 7: 57, 8: 58, 9: 59, 10: 60, 11: 61, 12: 62, 13: 63, 14: 64, 15: 65}, 'ref': {0: 20, 1: 19, 2: 18, 3: 17, 4: 16, 5: 15, 6: 14, 7: 13, 8: 12, 9: 11, 10: 10, 11: 9, 12: 8, 13: 7, 14: 6, 15: 5}}})
('OK', {'laser_enable': False})
('OK', {'sheath_pump_state': {'enable': False}})
('OK', {'sample_pump_state': {'enable': False}})
('OK', {'fan_state': {'enable': False}})
'''
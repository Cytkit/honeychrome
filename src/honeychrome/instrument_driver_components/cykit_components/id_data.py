import datetime

class IDData:
    def __init__(self, ft4222_communicator):
        self.timestamp = None
        self.version_major = None
        self.version_minor = None
        self.version_revision = None
        self.version_build = None

        self.ft4222 = ft4222_communicator

    def check_id(self):
        if self.ft4222.register_read('ID_WORD') == 0xCAFE:
            return True
        return False

    def read_id_data(self):
        year = self.ft4222.read_reg('TIMESTAMP_A')
        value = self.ft4222.read_reg('TIMESTAMP_B')
        month = (value >> 8) & 0x00FF
        day = (value >> 0) & 0x00FF
        value = self.ft4222.read_reg('TIMESTAMP_C')
        hour = (value >> 8) & 0x00FF
        minute = (value >> 0) & 0x00FF

        value = self.ft4222.read_reg('TIMESTAMP_D')
        second = (value >> 0) & 0x00FF

        self.timestamp = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)

        value = self.ft4222.read_reg('VERSION_A')
        self.version_major = (value >> 8) & 0x00FF
        self.version_minor = (value >> 0) & 0x00FF

        value = self.ft4222.read_reg('VERSION_B')
        self.version_revision = (value >> 0) & 0x00FF

        self.version_build = self.ft4222.read_reg('VERSION_C')

from bluepy.btle import UUID, Peripheral, AssignedNumbers
import struct
from kalman import Kalman
import time
from madgwick import Madgwick
# Sensortag versions
AUTODETECT = "-"
SENSORTAG_V1 = "v1"
SENSORTAG_2650 = "CC2650"
macAddress = "54:6C:0E:53:33:31"

def _TI_UUID(val):
    return UUID("%08X-0451-4000-b000-000000000000" % (0xF0000000 + val))

class SensorBase:
    # Derived classes should set: svcUUID, ctrlUUID, dataUUID
    sensorOn = struct.pack("B", 0x01)
    sensorOff = struct.pack("B", 0x00)

    def __init__(self, periph):
        self.periph = periph
        self.service = None
        self.ctrl = None
        self.data = None

    def enable(self):
        if self.service is None:
            self.service = self.periph.getServiceByUUID(self.svcUUID)
        if self.ctrl is None:
            self.ctrl = self.service.getCharacteristics(self.ctrlUUID)[0]
        if self.data is None:
            self.data = self.service.getCharacteristics(self.dataUUID)[0]
        if self.sensorOn is not None:
            self.ctrl.write(self.sensorOn, withResponse=True)

    def read(self):
        return self.data.read()

    def disable(self):
        if self.ctrl is not None:
            self.ctrl.write(self.sensorOff)

    # Derived class should implement _formatData()


class AccelerometerSensor(SensorBase):
    svcUUID = _TI_UUID(0xAA10)
    dataUUID = _TI_UUID(0xAA11)
    ctrlUUID = _TI_UUID(0xAA12)

    def __init__(self, periph):
        SensorBase.__init__(self, periph)
        if periph.firmwareVersion.startswith("1.4 "):
            self.scale = 64.0
        else:
            self.scale = 16.0

    def read(self):
        '''Returns (x_accel, y_accel, z_accel) in units of g'''
        x_y_z = struct.unpack('bbb', self.data.read())
        return tuple([(val / self.scale) for val in x_y_z])


class MovementSensorMPU9250(SensorBase):
    svcUUID = _TI_UUID(0xAA80)
    dataUUID = _TI_UUID(0xAA81)
    ctrlUUID = _TI_UUID(0xAA82)
    sensorOn = None
    GYRO_XYZ = 7
    ACCEL_XYZ = 7 << 3
    MAG_XYZ = 1 << 6
    ACCEL_RANGE_2G = 0 << 8
    ACCEL_RANGE_4G = 1 << 8
    ACCEL_RANGE_8G = 2 << 8
    ACCEL_RANGE_16G = 3 << 8

    def __init__(self, periph):
        SensorBase.__init__(self, periph)
        self.ctrlBits = 0

    def enable(self, bits):
        SensorBase.enable(self)
        self.ctrlBits |= bits
        self.ctrl.write(struct.pack("<H", self.ctrlBits))

    def disable(self, bits):
        self.ctrlBits &= ~bits
        self.ctrl.write(struct.pack("<H", self.ctrlBits))

    def rawRead(self):
        dval = self.data.read()
        return struct.unpack("<hhhhhhhhh", dval)


class AccelerometerSensorMPU9250:
    def __init__(self, sensor_):
        self.sensor = sensor_
        self.bits = self.sensor.ACCEL_XYZ | self.sensor.ACCEL_RANGE_4G
        self.scale = 8.0 / 32768.0  # TODO: why not 4.0, as documented?

    def enable(self):
        self.sensor.enable(self.bits)

    def disable(self):
        self.sensor.disable(self.bits)

    def read(self):
        '''Returns (x_accel, y_accel, z_accel) in units of g'''
        rawVals = self.sensor.rawRead()[3:6]
        return tuple([v * self.scale for v in rawVals])


class MagnetometerSensor(SensorBase):
    svcUUID  = _TI_UUID(0xAA30)
    dataUUID = _TI_UUID(0xAA31)
    ctrlUUID = _TI_UUID(0xAA32)

    def __init__(self, periph):
        SensorBase.__init__(self, periph)

    def read(self):
        '''Returns (x, y, z) in uT units'''
        x_y_z = struct.unpack('<hhh', self.data.read())
        return tuple([ 1000.0 * (v/32768.0) for v in x_y_z ])
        # Revisit - some absolute calibration is needed

class MagnetometerSensorMPU9250:
    def __init__(self, sensor_):
        self.sensor = sensor_
        self.scale = 4912.0 / 32760
        # Reference: MPU-9250 register map v1.4

    def enable(self):
        self.sensor.enable(self.sensor.MAG_XYZ)

    def disable(self):
        self.sensor.disable(self.sensor.MAG_XYZ)

    def read(self):
        '''Returns (x_mag, y_mag, z_mag) in units of uT'''
        rawVals = self.sensor.rawRead()[6:9]
        return tuple([ v*self.scale for v in rawVals ])

class GyroscopeSensor(SensorBase):
    svcUUID = _TI_UUID(0xAA50)
    dataUUID = _TI_UUID(0xAA51)
    ctrlUUID = _TI_UUID(0xAA52)
    sensorOn = struct.pack("B", 0x07)

    def __init__(self, periph):
        SensorBase.__init__(self, periph)

    def read(self):
        '''Returns (x,y,z) rate in deg/sec'''
        x_y_z = struct.unpack('<hhh', self.data.read())
        return tuple([250.0 * (v / 32768.0) for v in x_y_z])


class GyroscopeSensorMPU9250:
    def __init__(self, sensor_):
        self.sensor = sensor_
        self.scale = 500.0 / 65536.0

    def enable(self):
        self.sensor.enable(self.sensor.GYRO_XYZ)

    def disable(self):
        self.sensor.disable(self.sensor.GYRO_XYZ)

    def read(self):
        '''Returns (x_gyro, y_gyro, z_gyro) in units of degrees/sec'''
        rawVals = self.sensor.rawRead()[0:3]
        return tuple([v * self.scale for v in rawVals])

class BatterySensor(SensorBase):
    svcUUID  = UUID("0000180f-0000-1000-8000-00805f9b34fb")
    dataUUID = UUID("00002a19-0000-1000-8000-00805f9b34fb")
    ctrlUUID = None
    sensorOn = None

    def __init__(self, periph):
       SensorBase.__init__(self, periph)

    def read(self):
        '''Returns the battery level in percent'''
        val = ord(self.data.read())
        return val

class SensorTag(Peripheral):
    def __init__(self, addr, version=AUTODETECT):
        Peripheral.__init__(self, addr)
        if version == AUTODETECT:
            svcs = self.discoverServices()
            if _TI_UUID(0xAA70) in svcs:
                version = SENSORTAG_2650
            else:
                version = SENSORTAG_V1

        fwVers = self.getCharacteristics(uuid=AssignedNumbers.firmwareRevisionString)
        if len(fwVers) >= 1:
            self.firmwareVersion = fwVers[0].read().decode("utf-8")
        else:
            self.firmwareVersion = u''

        if version == SENSORTAG_V1:
            self.accelerometer = AccelerometerSensor(self)
            self.magnetometer = MagnetometerSensor(self)
            self.gyroscope = GyroscopeSensor(self)
        elif version == SENSORTAG_2650:
            self._mpu9250 = MovementSensorMPU9250(self)
            self.accelerometer = AccelerometerSensorMPU9250(self._mpu9250)
            self.magnetometer = MagnetometerSensorMPU9250(self._mpu9250)
            self.gyroscope = GyroscopeSensorMPU9250(self._mpu9250)
            self.battery = BatterySensor(self)


def main():
    sensorfusion = Kalman()

    print('Connecting to sensortag...')
    tag = SensorTag(macAddress)
    print("connected.")

    tag.accelerometer.enable()
    tag.magnetometer.enable()
    tag.gyroscope.enable()

    time.sleep(1.0)  # Loading sensors

    prev_time = time.time()
    while True:
        ax, ay, az = tag.accelerometer.read()
        gx, gy, gz = tag.gyroscope.read()
        mx, my, mz = tag.magnetometer.read()

        curr_time = time.time()
        dt = curr_time - prev_time

        # print(gz, mx, my, mz)

        sensorfusion.computeAndUpdateRollPitchYaw(ax, ay, az, gx, gy, gz, mx, my, mz, dt)

        print("roll:{0} pitch:{1} yaw:{2} ".format(sensorfusion.roll, sensorfusion.pitch, sensorfusion.yaw))

        prev_time = curr_time

    print("Battery: ", tag.battery.read())


if __name__ == "__main__":
    main()


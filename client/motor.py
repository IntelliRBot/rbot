import RPi.GPIO as GPIO
import time

FORWARD = 1
STOP = -1
BACKWARD = 0

Motor1A = 12
Motor1B = 32

FREQ = 100
DC = 25

class Motor:
    def __init__(self):
        GPIO.setmode(GPIO.BOARD)

        GPIO.setup(Motor1A, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(Motor1B, GPIO.OUT, initial=GPIO.LOW)

        self.pwm1a = GPIO.PWM(Motor1A, FREQ)
        self.pwm1b = GPIO.PWM(Motor1B, FREQ)

    def _forward(self):
        print("Forward")
        self.pwm1a.start(DC)
        self.pwm1b.start(0)
        

    def _backward(self):
        print("Backward")
        self.pwm1a.start(0)
        self.pwm1b.start(DC)

    def _stop(self):
        print("Stop")
        self.pwm1a.start(0)
        self.pwm1b.start(0)

    def cleanup(self):
        print("Clean up")
        GPIO.cleanup()

    def set_direction(self, direction):
        if direction == FORWARD:
            self._forward()
        if direction == STOP:
            self._stop()
        if direction == BACKWARD:
            self._backward()


def main():
    motor = Motor()
    while True:
        motor.set_direction(1)
        time.sleep(3)

        motor.set_direction(-1)
        time.sleep(3)

        motor.set_direction(0)
        time.sleep(3)

        motor.set_direction(-1)
        time.sleep(3)
    motor.cleanup()

if __name__ == "__main__":
    main()

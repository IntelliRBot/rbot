import RPi.GPIO as GPIO

FORWARD = 1
STOP = 0
BACKWARD = -1

Motor1A = 13
Motor1B = 11
Motor2A = 16
Motor2B = 18


class Motor:
    def __init__(self):
        GPIO.setmode(GPIO.BOARD)

        GPIO.setup(Motor1A, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(Motor1B, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(Motor2A, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(Motor2B, GPIO.OUT, initial=GPIO.LOW)

    def _forward(self):
        print("Forward")
        GPIO.output(Motor1A, GPIO.HIGH)
        GPIO.output(Motor1B, GPIO.LOW)

        GPIO.output(Motor2A, GPIO.LOW)
        GPIO.output(Motor2B, GPIO.HIGH)

    def _backward(self):
        print("Backward")
        GPIO.output(Motor1A, GPIO.LOW)
        GPIO.output(Motor1B, GPIO.HIGH)

        GPIO.output(Motor2A, GPIO.HIGH)
        GPIO.output(Motor2B, GPIO.LOW)

    def _stop(self):
        print("Stop")
        GPIO.output(Motor1A, GPIO.LOW)
        GPIO.output(Motor1B, GPIO.LOW)

        GPIO.output(Motor2A, GPIO.LOW)
        GPIO.output(Motor2B, GPIO.LOW)

    def cleanup(self):
        print("Clean up")
        GPIO.cleanup()

    def set_direction(self, direction):
        if direction == DIRECTION.FORWARD:
            self._forward()
        if direction == DIRECTION.STOP:
            self._stop()
        if direction == DIRECTION.BACKWARD:
            self._backward()

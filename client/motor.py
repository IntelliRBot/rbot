import RPi.GPIO as GPIO
from time import sleep

class DIRECTION:
    FORWARD = 1
    STOP = 0
    BACKWARD = -1

class Motor:
    def __init__(self):
        GPIO.setmode(GPIO.BOARD)
        
        self.Motor1A = 13
        self.Motor1B = 11

        self.Motor2A = 16
        self.Motor2B = 18
 
        GPIO.setup(self.Motor1A, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.Motor1B, GPIO.OUT, initial=GPIO.LOW)

        GPIO.setup(self.Motor2A, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.Motor2B, GPIO.OUT, initial=GPIO.LOW)

    def _forward(self):
        print("Forward")
        GPIO.output(self.Motor1A,GPIO.HIGH)
        GPIO.output(self.Motor1B,GPIO.LOW)

        GPIO.output(self.Motor2A,GPIO.LOW)
        GPIO.output(self.Motor2B,GPIO.HIGH)

    def _backward(self):
        print("Backward")
        GPIO.output(self.Motor1A,GPIO.LOW)
        GPIO.output(self.Motor1B,GPIO.HIGH)

        GPIO.output(self.Motor2A,GPIO.HIGH)
        GPIO.output(self.Motor2B,GPIO.LOW)

    def _stop(self):
        print("Stop")
        GPIO.output(self.Motor1A,GPIO.LOW)
        GPIO.output(self.Motor1B,GPIO.LOW)

        GPIO.output(self.Motor2A,GPIO.LOW)
        GPIO.output(self.Motor2B,GPIO.LOW)

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

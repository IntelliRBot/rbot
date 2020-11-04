import RPi.GPIO as GPIO
from time import sleep
 
GPIO.setmode(GPIO.BOARD)
 
Motor1A = 13
Motor1B = 11
Motor2A = 16
Motor2B = 18
 
GPIO.setup(Motor1A,GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(Motor1B,GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(Motor2A,GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(Motor2B,GPIO.OUT, initial=GPIO.LOW)

try:
    print('Press CTRL+C to Interrupt')
    while True:
        print ("Forward")
        GPIO.output(Motor1A,GPIO.HIGH)
        GPIO.output(Motor1B,GPIO.LOW)
        GPIO.output(Motor2A,GPIO.LOW)
        GPIO.output(Motor2B,GPIO.HIGH)
        sleep(2)

        print ("Backward")
        GPIO.output(Motor1A,GPIO.LOW)
        GPIO.output(Motor1B,GPIO.HIGH)
        GPIO.output(Motor2A,GPIO.HIGH)
        GPIO.output(Motor2B,GPIO.LOW)
        sleep(2)

except KeyboardInterrupt:
    print('Interrupt')
    GPIO.cleanup()


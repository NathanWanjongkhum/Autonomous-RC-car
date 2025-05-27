import RPi.GPIO as GPIO
from time import sleep

def main(stop_event=None):
    GPIO.setwarnings(False)

    # Right Motor
    in1 = 24
    in2 = 23
    en_a = 25
    # Left Motor
    in3 = 17
    in4 = 27
    en_b = 5

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(in1,GPIO.OUT)
    GPIO.setup(in2,GPIO.OUT)
    GPIO.setup(en_a,GPIO.OUT)

    GPIO.setup(in3,GPIO.OUT)
    GPIO.setup(in4,GPIO.OUT)
    GPIO.setup(en_b,GPIO.OUT)

    q=GPIO.PWM(en_a,100)
    p=GPIO.PWM(en_b,100)
    p.start(75)
    q.start(75)

    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in4,GPIO.LOW)
    GPIO.output(in3,GPIO.LOW)

    try:
        while True:
            if stop_event and stop_event.is_set():
                break
            user_input = input()
            if user_input == 'w':
                GPIO.output(in1,GPIO.HIGH)
                GPIO.output(in2,GPIO.LOW)
                print("Forward")
            elif user_input == 's':
                GPIO.output(in1,GPIO.LOW)
                GPIO.output(in2,GPIO.HIGH)
                print('Back')
            elif user_input == 'd':
                GPIO.output(in4,GPIO.HIGH)
                GPIO.output(in3,GPIO.LOW)
                print('Right')
            elif user_input == 'a':
                GPIO.output(in4,GPIO.LOW)
                GPIO.output(in3,GPIO.HIGH)
                print('Left')
            elif user_input == 'stop':
                GPIO.output(in1,GPIO.LOW)
                GPIO.output(in2,GPIO.LOW)
                GPIO.output(in4,GPIO.LOW)
                GPIO.output(in3,GPIO.LOW)
                print('Stop')
            elif user_input == 'unturn':
                GPIO.output(in3,GPIO.LOW)
                GPIO.output(in4,GPIO.LOW)
            elif user_input == 'stopall':
                if stop_event:
                    stop_event.set()
                break
    except KeyboardInterrupt:
        GPIO.cleanup()
        print("GPIO Clean up")
    GPIO.cleanup()

if __name__ == '__main__':
    main()


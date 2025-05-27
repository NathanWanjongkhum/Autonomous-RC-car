import RPi.GPIO as GPIO
from time import sleep

def motor_control_loop(command_queue, stop_event):
    GPIO.setwarnings(False)

    # Right Motor
    in1 = 27
    in2 = 22
    en_a = 4
    # Left Motor
    in3 = 23
    in4 = 24
    en_b = 13

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
        while not stop_event.is_set():
            try:
                command = command_queue.get(timeout=0.1)
            except Exception:
                continue
            if command == 'w':
                GPIO.output(in1,GPIO.HIGH)
                GPIO.output(in2,GPIO.LOW)
                print("Forward")
            elif command == 's':
                GPIO.output(in1,GPIO.LOW)
                GPIO.output(in2,GPIO.HIGH)
                print('Back')
            elif command == 'd':
                GPIO.output(in4,GPIO.HIGH)
                GPIO.output(in3,GPIO.LOW)
                print('Right')
            elif command == 'a':
                GPIO.output(in4,GPIO.LOW)
                GPIO.output(in3,GPIO.HIGH)
                print('Left')
            elif command == 'stop':
                GPIO.output(in1,GPIO.LOW)
                GPIO.output(in2,GPIO.LOW)
                GPIO.output(in4,GPIO.LOW)
                GPIO.output(in3,GPIO.LOW)
                print('Stop')
            elif command == 'unturn':
                GPIO.output(in3,GPIO.LOW)
                GPIO.output(in4,GPIO.LOW)
            elif command == 'stopall':
                stop_event.set()
                break
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()
        print("GPIO Clean up")
        pass

if __name__ == '__main__':
    motor_control_loop()


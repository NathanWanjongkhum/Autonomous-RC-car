import threading
import queue
from camera.stereo_pi_integration import stereo_decision_loop
from control.motorcontrol import motor_control_loop

if __name__ == "__main__":
    stop_event = threading.Event()
    command_queue = queue.Queue()

    t1 = threading.Thread(target=motor_control_loop, args=(command_queue, stop_event))
    t2 = threading.Thread(target=stereo_decision_loop, args=(command_queue, stop_event))

    t1.start()
    t2.start()

    t1.join()
    t2.join()
import threading
import queue
from planning.decision import stereo_pathfinding_loop
from control.motorcontrol import motor_control_loop

if __name__ == "__main__":
    stop_event = threading.Event()
    command_queue = queue.Queue()

    # Path to your calibration file
    calib_file = "camera/stereo_calib.yaml"

    t1 = threading.Thread(target=motor_control_loop, args=(command_queue, stop_event))
    t2 = threading.Thread(target=stereo_pathfinding_loop, args=(command_queue, stop_event, calib_file))

    t1.start()
    t2.start()

    t1.join()
    t2.join()
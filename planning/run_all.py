import threading
import queue
from planning.decision import stereo_pathfinding_loop
from control.motorcontrol import motor_control_loop

"""
run_all.py

This script launches both the autonomous pathfinding (stereo vision + planning)
and the motor control loop in separate threads, allowing them to communicate
via a thread-safe queue.

- The pathfinding thread analyzes camera data, plans a path, and sends motor commands.
- The motor control thread receives commands from the queue and drives the motors.
- A stop_event is used to signal both threads to shut down cleanly.
"""

if __name__ == "__main__":
    # Event to signal both threads to stop
    stop_event = threading.Event()
    # Thread-safe queue for sending commands from pathfinding to motor control
    command_queue = queue.Queue()

    # Path to the stereo camera calibration file (update if needed)
    calib_file = "camera/stereo_calib.yaml"

    # Create and start the motor control thread
    t1 = threading.Thread(target=motor_control_loop, args=(command_queue, stop_event))
    # Create and start the pathfinding/planning thread
    t2 = threading.Thread(target=stereo_pathfinding_loop, args=(command_queue, stop_event, calib_file))

    t1.start()
    t2.start()

    # Wait for both threads to finish before exiting
    t1.join()
    t2.join()
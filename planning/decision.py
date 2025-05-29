from camera.stereo import StereoVisionPi
import time

def stereo_pathfinding_loop(command_queue, stop_event, calib_file, width=20, height=20, fps=2):
    """
    Main loop for autonomous pathfinding using stereo vision and occupancy grid mapping.

    Args:
        command_queue (queue.Queue): Queue to send motor commands to the motor controller.
        stop_event (threading.Event): Event to signal stopping the loop.
        calib_file (str): Path to stereo camera calibration file.
        width (int): Width of the occupancy grid.
        height (int): Height of the occupancy grid.
        fps (int): Frames per second for stereo vision capture.

    This function:
        1. Captures a stereo image and builds an occupancy grid.
        2. Sets start and goal positions for the robot.
        3. Runs a pathfinding and control algorithm to compute steering and velocity.
        4. Converts those into motor commands and sends them to the motor controller.
        5. Repeats until the stop event is set or a step limit is reached.
    """
    from test_phase2 import transition_to_phase2, Pose
    from maps.OccupancyGrid import OccupancyGrid

    print("[DEBUG] Starting stereo_pathfinding_loop")

    # 1. Build occupancy grid from stereo vision
    stereo = StereoVisionPi(calib_file, width, height, fps)
    try:
        print("[DEBUG] Capturing stereo for occupancy grid...")
        prob = stereo.grab_realtime()  # Get probability map from stereo vision
        print(f"[DEBUG] Stereo probability map shape: {prob.shape}, dtype: {prob.dtype}")
        occ_grid = (prob > 0.5).astype(int)  # Threshold to create binary occupancy grid
        print(f"[DEBUG] Occupancy grid created. Sum occupied: {occ_grid.sum()}")
        grid = OccupancyGrid(width=width, height=height, resolution=0.1)
        grid.binary_grid[:occ_grid.shape[0], :occ_grid.shape[1]] = occ_grid
        print("[DEBUG] OccupancyGrid updated.")

        # 2. Set start and goal poses for the robot
        start_pose = Pose(x=width * 0.1, y=height * 0.1, theta=0.0)
        goal_pose = Pose(x=width * 0.9, y=height * 0.9, theta=0.0)
        print(f"[DEBUG] Start pose: {start_pose}, Goal pose: {goal_pose}")

        # 3. Get the integrated controller (path planning + pure pursuit)
        class DummySim:
            def __init__(self, grid):
                self.grid = grid

        sim = DummySim(grid)
        print("[DEBUG] Calling transition_to_phase2...")
        controller = transition_to_phase2(sim, start_pose, goal_pose)
        if not controller:
            print("[ERROR] Path planning failed!")
            return
        print("[DEBUG] Controller initialized.")

        # 4. Real car execution loop: follow the planned path
        print("[DEBUG] Starting real car trajectory following...")
        current_pose = [start_pose.x, start_pose.y, start_pose.theta]
        dt = 0.1  # Time step in seconds

        for step in range(200):  # Limit steps for safety
            if stop_event.is_set():
                print("[DEBUG] Stop event set, breaking loop.")
                break

            # Compute steering and velocity from controller
            steering_angle, linear_velocity = controller.compute_steering(tuple(current_pose), time.time(), dt)
            print(f"[DEBUG] Step {step}: steering_angle={steering_angle:.3f}, linear_velocity={linear_velocity:.3f}")

            # Convert controller output to motor commands
            if abs(steering_angle) < 0.1 and linear_velocity > 0:
                command = 'w'      # Forward
            elif steering_angle > 0.1:
                command = 'a'      # Turn left
            elif steering_angle < -0.1:
                command = 'd'      # Turn right
            else:
                command = 'stop'   # Stop

            # Send command to motor controller
            command_queue.put(command)
            print(f"[DEBUG] Sent command: {command} (steering={steering_angle:.2f}, v={linear_velocity:.2f})")

            # TODO: Update current_pose with real odometry if available

            time.sleep(dt)  # Wait for next control step

    finally:
        print("[DEBUG] Releasing stereo resources.")
        stereo.release()
from camera.stereo import StereoVisionPi
import time

def stereo_pathfinding_loop(command_queue, stop_event, calib_file, width=20, height=20, fps=2):
    from test_phase2 import transition_to_phase2, Pose
    from maps.OccupancyGrid import OccupancyGrid

    # 1. Build occupancy grid from stereo vision
    stereo = StereoVisionPi(calib_file, width, height, fps)
    try:
        print("Capturing stereo for occupancy grid...")
        prob = stereo.grab_realtime()
        occ_grid = (prob > 0.5).astype(int)
        grid = OccupancyGrid(width=width, height=height, resolution=0.1)
        grid.binary_grid[:occ_grid.shape[0], :occ_grid.shape[1]] = occ_grid

        # 2. Set start and goal poses
        start_pose = Pose(x=width * 0.1, y=height * 0.1, theta=0.0)
        goal_pose = Pose(x=width * 0.9, y=height * 0.9, theta=0.0)

        # 3. Get the integrated controller (path + pure pursuit)
        class DummySim:
            def __init__(self, grid): self.grid = grid
        sim = DummySim(grid)
        controller = transition_to_phase2(sim, start_pose, goal_pose)
        if not controller:
            print("Path planning failed!")
            return

        # 4. Real car execution loop
        print("Starting real car trajectory following...")
        current_pose = [start_pose.x, start_pose.y, start_pose.theta]
        dt = 0.1
        for _ in range(200):  # Limit steps for safety
            if stop_event.is_set():
                break
            steering_angle, linear_velocity = controller.compute_steering(tuple(current_pose), time.time(), dt)
            # Convert to motor commands
            if abs(steering_angle) < 0.1 and linear_velocity > 0:
                command = 'w'
            elif steering_angle > 0.1:
                command = 'a'
            elif steering_angle < -0.1:
                command = 'd'
            else:
                command = 'stop'
            command_queue.put(command)
            print(f"Sent command: {command} (steering={steering_angle:.2f}, v={linear_velocity:.2f})")
            # TODO: Update current_pose with real odometry if available
            time.sleep(dt)
    finally:
        stereo.release()
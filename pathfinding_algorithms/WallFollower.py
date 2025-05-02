import numpy as np

class WallFollower:
    """
    Implements a simple wall following algorithm

    This class uses the occupancy grid to follow walls (boundaries)
    in the environment, which is useful for the exploration phase.
    """

    def __init__(self, occupancy_grid, side="right", distance=0.3):
        """
        Initialize the wall follower

        Parameters:
        occupancy_grid: OccupancyGrid instance
        side: Which side to follow the wall ('left' or 'right')
        distance: Desired distance from the wall
        """
        self.grid = occupancy_grid
        self.side = side  # 'left' or 'right'
        self.desired_distance = distance
        self.max_sensing_distance = 2.0  # Maximum distance to look for walls
        self.num_sensors = 8  # Number of sensor rays to use

    def sense_walls(self, x, y, theta):
        """
        Sense walls around the car using ray casting

        Parameters:
        x, y, theta: Current pose of the car

        Returns:
        ranges: List of measured distances to obstacles
        angles: List of angles (relative to car heading) of the measurements
        """
        ranges = []
        angles = []

        # Generate a set of rays around the car
        for i in range(self.num_sensors):
            angle = i * (2 *np.pi / self.num_sensors)
            angles.append(angle)

            # Cast ray until hitting an obstacle or max distance
            for d in np.arange(0, self.max_sensing_distance, self.grid.resolution):
                ray_x = x + d * np.cos(theta + angle)
                ray_y = y + d * np.sin(theta + angle)

                # Check if ray hit an obstacle
                if self.grid.is_occupied(ray_x, ray_y):
                    ranges.append(d)
                    break
                else:
                    # No obstacle found at max distance
                    ranges.append(self.max_sensing_distance)

        return ranges, angles

    def compute_steering(self, x, y, theta, velocity):
        """
        Compute steering command to follow walls

        Parameters:
        x, y, theta: Current pose of the car
        velocity: Current linear velocity

        Returns:
        angular_velocity: Desired angular velocity
        """
        # Sense walls
        ranges, angles = self.sense_walls(x, y, theta)

        # Find the closest wall sensor in front of the car
        front_idx = None
        min_front_dist = float("inf")

        for i, angle in enumerate(angles):
            # Consider only sensors in front (within ±90° of heading)
            if abs(angle - np.pi) < np.pi / 2:
                if ranges[i] < min_front_dist:
                    min_front_dist = ranges[i]
                    front_idx = i

        # Find the closest wall sensor on the preferred side
        side_idx = None
        min_side_dist = float("inf")

        for i, angle in enumerate(angles):
            if self.side == "right" and 0 < angle < np.pi:
                if ranges[i] < min_side_dist:
                    min_side_dist = ranges[i]
                    side_idx = i
            elif self.side == "left" and np.pi < angle < 2*np.pi:
                if ranges[i] < min_side_dist:
                    min_side_dist = ranges[i]
                    side_idx = i

        # If no walls detected, turn to find a wall
        if front_idx is None and side_idx is None:
            if self.side == "right":
                return -0.5  # Turn right to find a wall
            else:
                return 0.5  # Turn left to find a wall

        # If front wall is too close, avoid it
        if front_idx is not None and min_front_dist < 0.05:
            if self.side == "right":
                return 1.0  # Turn left to avoid collision
            else:
                return -1.0  # Turn right to avoid collision

        # Follow wall at desired distance
        if side_idx is not None:
            error = min_side_dist - self.desired_distance

            if self.side == "right":
                return -0.5 * error  # Negative feedback for right wall
            else:
                return 0.5 * error  # Positive feedback for left wall

        # Default: move forward
        return 0.0


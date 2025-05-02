import numpy as np

class PurePursuit:
    """
    Implements the Pure Pursuit controller for path following

    This controller calculates steering commands to follow a path by
    tracking a point ahead of the vehicle on the path.
    """

    def __init__(self, lookahead_distance=0.5):
        """
        Initialize the controller

        Parameters:
        lookahead_distance: Distance ahead of the car to look for path points
        """
        self.lookahead_distance = lookahead_distance
        self.path = []  # List of (x, y) points defining the path

    def set_path(self, path):
        """
        Set the path to follow

        Parameters:
        path: List of (x, y) points defining the path
        """
        self.path = path

    def get_target_point(self, x, y):
        """
        Find the target point on the path that is lookahead_distance away

        Parameters:
        x, y: Current position of the car

        Returns:
        target_x, target_y: Coordinates of the target point, or None if not found
        """
        # If path is empty, return None
        if not self.path:
            return None, None

        # Find the closest point on the path
        min_dist = float("inf")
        closest_idx = 0

        for i, (path_x, path_y) in enumerate(self.path):
            dist = np.sqrt((path_x - x) ** 2 + (path_y - y) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Search forward from the closest point to find a point at lookahead distance
        target_idx = closest_idx
        target_dist = min_dist

        while target_dist < self.lookahead_distance and target_idx + 1 < len(self.path):
            target_idx += 1
            target_dist = np.sqrt(
                (self.path[target_idx][0] - x) ** 2
                + (self.path[target_idx][1] - y) ** 2
            )

        # If we reached the end of the path, use the last point
        if target_idx == len(self.path) - 1 and target_dist < self.lookahead_distance:
            return self.path[-1]

        return self.path[target_idx]

    def compute_steering(self, x, y, theta, velocity):
        """
        Compute the steering command (angular velocity) to reach the target point

        Parameters:
        x, y: Current position
        theta: Current orientation (radians)
        velocity: Current linear velocity

        Returns:
        angular_velocity: Desired angular velocity
        """
        # Get target point
        target_x, target_y = self.get_target_point(x, y)

        # If no target point found, return zero steering
        if target_x is None:
            return 0.0

        # Calculate angle to target in vehicle frame
        target_angle = np.arctan2(target_y - y, target_x - x)

        # Calculate heading error
        alpha = target_angle - theta

        # Normalize to [-pi, pi]
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

        # Calculate curvature (1/R) = 2*sin(alpha) / lookahead_distance
        curvature = 2 * np.sin(alpha) / self.lookahead_distance

        # Angular velocity = curvature * velocity
        angular_velocity = curvature * velocity

        return angular_velocity

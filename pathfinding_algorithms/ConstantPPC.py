import numpy as np


class ConstantPurePursuitController:
    """
    Pure Pursuit controller that maintains constant angular velocity.

    For Ackermann steering cars, this controller:
    1. Maintains a constant angular velocity
    2. Adjusts linear velocity based on path curvature
    3. Handles both front and rear axle reference points
    4. Includes feedforward and PID control for improved tracking
    """

    def __init__(
        self,
        angular_velocity=0.5,
        base_lookahead=0.5,
        reference_point="rear",
        use_feedforward=True,
    ):
        # Controller parameters
        self.angular_velocity = angular_velocity
        self.base_lookahead = base_lookahead
        self.min_lookahead = 0.2
        self.max_lookahead = 1.5
        self.reference_point = reference_point  # 'front' or 'rear'
        self.use_feedforward = use_feedforward

        # Path data
        self.path = []
        self.path_curvature = []

        # PID controller gains
        self.kp = 1.0  # Proportional gain
        self.ki = 0.01  # Integral gain
        self.kd = 0.1  # Derivative gain

        # Error tracking
        self.error_history = []
        self.error_sum = 0.0
        self.last_error = 0.0

        # Performance tracking
        self.target_points = []

    def set_path(self, path):
        """Set the path to follow and calculate path curvature"""
        self.path = path
        self.error_history = []
        self.error_sum = 0.0
        self.last_error = 0.0

        # Calculate curvature at each path point
        self.path_curvature = self._calculate_path_curvature()

    def _calculate_path_curvature(self):
        """Calculate curvature at each point along the path"""
        if len(self.path) < 3:
            return [0.0] * len(self.path)

        curvature = [0.0] * len(self.path)

        # For each interior point, calculate curvature using three points
        for i in range(1, len(self.path) - 1):
            p1 = np.array(self.path[i - 1])
            p2 = np.array(self.path[i])
            p3 = np.array(self.path[i + 1])

            # Calculate vectors between points
            v1 = p2 - p1
            v2 = p3 - p2

            # Calculate cross product to determine turning direction
            cross_z = v1[0] * v2[1] - v1[1] * v2[0]

            # Calculate side lengths of the triangle
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p3 - p1)

            # Avoid division by zero
            if a * b * c < 1e-6:
                curvature[i] = 0.0
                continue

            # Area of triangle using Heron's formula
            s = (a + b + c) / 2
            area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))

            # Curvature = 4 * area / (a * b * c)
            # Sign determined by cross product (positive for left turn)
            curvature[i] = 4 * area / (a * b * c) * np.sign(cross_z)

        # Set endpoints to match neighbors
        curvature[0] = curvature[1]
        curvature[-1] = curvature[-2]

        return curvature

    def get_target_point(self, x, y):
        """Find the target point on the path at lookahead distance"""
        if not self.path:
            return None, None, 0

        # Find closest point on path
        min_dist = float("inf")
        closest_idx = 0

        for i, (path_x, path_y) in enumerate(self.path):
            dist = np.sqrt((path_x - x) ** 2 + (path_y - y) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Search forward for lookahead point
        target_idx = closest_idx
        target_dist = min_dist

        while target_dist < self.base_lookahead and target_idx + 1 < len(self.path):
            target_idx += 1
            target_dist = np.sqrt(
                (self.path[target_idx][0] - x) ** 2
                + (self.path[target_idx][1] - y) ** 2
            )

        # If we reached the end of path
        if target_idx == len(self.path) - 1 and target_dist < self.base_lookahead:
            return self.path[-1][0], self.path[-1][1], closest_idx

        return self.path[target_idx][0], self.path[target_idx][1], closest_idx

    def compute_steering(self, x, y, theta, wheelbase, dt=0.1):
        """
        Compute steering angle and velocity for constant angular velocity

        Parameters:
        x, y: Current position
        theta: Current orientation (radians)
        wheelbase: Distance between front and rear axles
        dt: Time step for derivative/integral calculations

        Returns:
        steering_angle: Desired steering angle (radians)
        linear_velocity: Desired linear velocity (m/s)
        """
        # Adjust reference point if using front axle
        if self.reference_point == "front":
            # Convert front to rear axle position for calculations
            rear_x = x - wheelbase * np.cos(theta)
            rear_y = y - wheelbase * np.sin(theta)
            calc_x, calc_y = rear_x, rear_y
        else:
            calc_x, calc_y = x, y

        # Get target point
        target_x, target_y, closest_idx = self.get_target_point(calc_x, calc_y)

        # If no target found, stop
        if target_x is None:
            return 0.0, 0.0

        # Calculate angle to target in vehicle frame
        target_angle = np.arctan2(target_y - calc_y, target_x - calc_x)

        # Calculate heading error
        heading_error = target_angle - theta

        # Normalize to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # Update error history for PID
        self.error_history.append(heading_error)
        if len(self.error_history) > 20:  # Limit history size
            self.error_history.pop(0)

        # Calculate curvature for Pure Pursuit
        curvature = 2 * np.sin(heading_error) / self.base_lookahead

        # For constant angular velocity:
        # angular_velocity = curvature * linear_velocity
        # Therefore: linear_velocity = angular_velocity / curvature

        # Handle straight segments (avoid division by zero)
        if abs(curvature) < 0.001:
            linear_velocity = 0.5  # Default velocity for straight segments
        else:
            # Compute linear velocity for constant angular velocity
            linear_velocity = self.angular_velocity / abs(curvature)

            # Cap velocity to reasonable limits
            linear_velocity = min(linear_velocity, 0.5)  # Max velocity
            linear_velocity = max(linear_velocity, 0.1)  # Min velocity

        # PID control for heading correction
        p_term = self.kp * heading_error

        # Integral term
        self.error_sum += heading_error * dt
        i_term = self.ki * self.error_sum
        # Anti-windup
        i_term = np.clip(i_term, -0.2, 0.2)

        # Derivative term
        if len(self.error_history) > 1:
            d_term = self.kd * (heading_error - self.last_error) / dt
        else:
            d_term = 0
        self.last_error = heading_error

        # Feedforward term based on path curvature
        ff_term = 0.0
        if self.use_feedforward and closest_idx < len(self.path_curvature):
            # Look ahead for upcoming curvature
            look_ahead = min(3, len(self.path_curvature) - closest_idx - 1)
            future_idx = closest_idx + look_ahead
            path_curvature = self.path_curvature[future_idx]
            ff_term = 0.2 * path_curvature * wheelbase  # Feedforward gain

        # Combine all control terms
        control_correction = p_term + i_term + d_term + ff_term

        # For Ackermann steering with constant angular velocity:
        # θ̇ = v * tan(φ) / L
        # Solving for φ: φ = arctan(θ̇ * L / v)
        base_steering = np.arctan2(self.angular_velocity * wheelbase, linear_velocity)

        # Apply sign from curvature direction
        base_steering *= np.sign(curvature)

        # Add PID correction to base steering
        steering_angle = base_steering + control_correction

        # Limit steering angle
        max_steering = np.radians(35)  # Maximum steering angle
        steering_angle = np.clip(steering_angle, -max_steering, max_steering)

        return steering_angle, linear_velocity

    def visualize_control(self, car_positions, target_points=None):
        """Visualize the car's trajectory and controller performance"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))

        # Plot reference path
        path_x = [p[0] for p in self.path]
        path_y = [p[1] for p in self.path]
        plt.plot(path_x, path_y, "b--", linewidth=2, label="Reference Path")

        # Plot car trajectory
        car_x = [p[0] for p in car_positions]
        car_y = [p[1] for p in car_positions]
        plt.plot(car_x, car_y, "r-", linewidth=2, label="Car Trajectory")

        # Plot target points if provided
        if target_points and len(target_points) > 0:
            target_x = [p[0] for p in target_points]
            target_y = [p[1] for p in target_points]
            plt.scatter(
                target_x, target_y, color="g", s=30, alpha=0.5, label="Target Points"
            )

        # Mark start and end
        plt.scatter(car_x[0], car_y[0], color="g", s=100, label="Start")
        plt.scatter(car_x[-1], car_y[-1], color="r", s=100, label="End")

        plt.title("Constant Angular Velocity Pure Pursuit Control")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

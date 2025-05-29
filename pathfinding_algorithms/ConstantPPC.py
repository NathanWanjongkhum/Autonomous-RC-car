import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import numpy.typing as npt
import matplotlib.axes


class ConstantPurePursuitController:
    """
    Pure Pursuit controller with discrete steering angles and constant angular velocity.

    For Ackermann steering cars, this controller:
    1. Maintains a constant angular velocity
    2. Adjusts linear velocity based on path curvature
    3. Handles both front and rear axle reference points
    4. Includes feedforward and PID control for improved tracking
    5. Uses discrete steering angles (left/neutral/right)
    6. Models physical actuator constraints (rate limits, inertia)
    """

    def __init__(
        self,
        angular_velocity: float = 0.5,
        base_lookahead: float = 0.8,
        reference_point: str = "rear",
        min_actuation_interval: float = 0.3,
        time_constant: float = 0.2,
        max_steering_angle: float = 35,
        # left_angle: float = 15,
        # right_angle: float = -15,
        max_integral_degrees: float = 10.0,  # Maximum integral correction in degrees
        feedforward_lookahead_points: int = 3,  # How many points to look ahead
        feedforward_lookahead_time: float = 0.5,  # Alternative: time-based lookahead
        hysteresis_threshold: float = 4.0,
        dead_zone_threshold: float = 0.1,
    ) -> None:
        # Controller parameters
        self.angular_velocity: float = angular_velocity
        self.base_lookahead: float = base_lookahead
        self.min_lookahead: float = 0.2
        self.max_lookahead: float = 1.5
        self.reference_point: str = reference_point  # 'front' or 'rear'
        self.max_steering_angle: float = np.radians(max_steering_angle)
        self.dead_zone_threshold = np.radians(dead_zone_threshold)

        # Proper integral limits based on steering angle
        self.max_integral_correction = np.radians(max_integral_degrees)

        # Alternative: Base on percentage of max steering
        # self.max_integral_correction = 0.3 * np.radians(35)  # 30% of max steering

        # Configurable feedforward lookahead
        self.feedforward_lookahead_points = feedforward_lookahead_points
        self.feedforward_lookahead_time = feedforward_lookahead_time

        # Path data
        self.path: List[Tuple[float, float]] = []
        self.path_curvature: List[float] = []

        # PID controller gains
        self.kp: float = 1.0  # Proportional gain
        self.ki: float = 0.01  # Integral gain
        self.kd: float = 0.1  # Derivative gain

        # Error tracking
        self.error_history: List[float] = []
        self.error_sum: float = 0.0
        self.last_error: float = 0.0

        # Performance tracking
        self.target_points: List[Tuple[float, float]] = []

        # Discrete steering state parameters
        # self.steering_states: Dict[str, float] = {
        #     "left": left_angle,  # Left steering angle
        #     "neutral": 0.0,  # Neutral (center) steering
        #     "right": right_angle,  # Right steering angle
        # }

        self.steering_states: Dict[str, float] = {
            "left": self.max_steering_angle,  # Left steering angle
            "neutral": 0.0,  # Neutral (center) steering
            "right": -self.max_steering_angle,  # Right steering angle
        }

        # Hysteresis thresholds to prevent oscillations
        self.threshold_to_left: float = np.radians(hysteresis_threshold)
        self.threshold_to_right: float = -np.radians(hysteresis_threshold)
        self.threshold_to_neutral: float = np.radians(hysteresis_threshold) * 0.8
        # 20% dead zone

        # Actuation rate limiting
        self.current_state: str = "neutral"  # Current steering state
        self.last_actuation_time: float = time.time()  # Time of last actuation
        self.min_actuation_interval: float = (
            min_actuation_interval  # Minimum time between actuations
        )

        # Physical actuator model
        self.time_constant: float = time_constant  # Time constant for actuator response
        self.current_angle: float = 0.0  # Current actual steering angle
        self.last_update_time: float = time.time()  # Time of last physics update

        # Additional tracking for visualization and debugging
        self.desired_angles: List[float] = []  # Continuous angles before discretization
        self.actual_angles: List[float] = []  # Actual angles after physical model
        self.steering_states_history: List[str] = []  # History of steering states

    def set_path(self, path: List[Tuple[float, float]]) -> None:
        """Set the path to follow and calculate path curvature"""
        self.path = path
        self.error_history = []
        self.error_sum = 0.0
        self.last_error = 0.0
        self.desired_angles = []
        self.actual_angles = []
        self.steering_states_history = []
        self.target_points = []

        # Calculate curvature at each path point
        self.path_curvature = self._calculate_path_curvature()

    def _calculate_path_curvature(self) -> List[float]:
        """Calculate curvature at each point along the path"""
        if len(self.path) < 3:
            return [0.0] * len(self.path)

        curvature: List[float] = [0.0] * len(self.path)

        # For each interior point, calculate curvature using three points
        for i in range(1, len(self.path) - 1):
            p1: npt.NDArray[np.float64] = np.array(self.path[i - 1])
            p2: npt.NDArray[np.float64] = np.array(self.path[i])
            p3: npt.NDArray[np.float64] = np.array(self.path[i + 1])

            # Calculate vectors between points
            v1: npt.NDArray[np.float64] = p2 - p1
            v2: npt.NDArray[np.float64] = p3 - p2

            # Calculate side lengths of the triangle
            a: float = np.linalg.norm(p2 - p1)
            b: float = np.linalg.norm(p3 - p2)
            c: float = np.linalg.norm(p3 - p1)

            # Avoid division by zero
            if a * b * c < 1e-6:
                curvature[i] = 0.0
                continue

            # Calculate angle between vectors
            dot_product: float = np.dot(v1, v2)
            mag_v1: float = np.linalg.norm(v1)
            mag_v2: float = np.linalg.norm(v2)

            if mag_v1 * mag_v2 == 0:
                curvature[i] = 0.0
                continue

            cos_angle: float = dot_product / (mag_v1 * mag_v2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure in valid range
            angle: float = np.arccos(cos_angle)

            # Calculate sign of curvature (left or right turn)
            # np.cross for 2D vectors returns a scalar
            cross_z_scalar: float = np.cross(v1, v2)
            sign: int = 1 if cross_z_scalar >= 0 else -1

            # Curvature is inversely proportional to radius
            segment_length: float = (mag_v1 + mag_v2) / 2
            if segment_length > 0:
                # Apply sign to distinguish left vs right turns
                curvature[i] = sign * angle / segment_length
            else:
                curvature[i] = 0.0

        # Set endpoints to match neighbors
        if len(curvature) > 1:
            curvature[0] = curvature[1]
        if len(curvature) > 1:
            curvature[-1] = curvature[-2]

        return curvature

    def get_target_point(
        self, x: float, y: float
    ) -> Tuple[Optional[float], Optional[float], int]:
        """Find the target point on the path at lookahead distance"""
        if not self.path:
            return None, None, 0

        # Find closest point on path
        min_dist: float = float("inf")
        closest_idx: int = 0

        for i, (path_x, path_y) in enumerate(self.path):
            dist: float = np.sqrt((path_x - x) ** 2 + (path_y - y) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Search forward for lookahead point
        target_idx: int = closest_idx
        target_dist: float = min_dist

        while target_dist < self.base_lookahead and target_idx + 1 < len(self.path):
            target_idx += 1
            target_dist = np.sqrt(
                (self.path[target_idx][0] - x) ** 2
                + (self.path[target_idx][1] - y) ** 2
            )

        # If we reached the end of path
        if target_idx == len(self.path) - 1 and target_dist < self.base_lookahead:
            return self.path[-1][0], self.path[-1][1], closest_idx

        # Store target point for visualization
        if target_idx < len(self.path):
            self.target_points.append(
                (self.path[target_idx][0], self.path[target_idx][1])
            )

        return self.path[target_idx][0], self.path[target_idx][1], closest_idx

    def determine_target_state(self, desired_angle: float) -> str:
        """
        Determine the target steering state based on desired angle and hysteresis

        Parameters:
        desired_angle: The calculated ideal steering angle

        Returns:
        state: The target steering state ("left", "neutral", or "right")
        """
        # Apply hysteresis logic to prevent oscillations
        if desired_angle > self.threshold_to_left:
            return "left"
        elif desired_angle < self.threshold_to_right:
            return "right"
        elif abs(desired_angle) < self.threshold_to_neutral:
            return "neutral"
        else:
            # If in the hysteresis band, maintain current state
            return self.current_state

    def update_steering_state(self, target_state: str, current_time: float) -> bool:
        """
        Update steering state respecting actuation rate limits

        Parameters:
        target_state: The desired steering state
        current_time: Current system time

        Returns:
        bool: True if state changed, False otherwise
        """
        # Check if enough time has passed since the last actuation
        time_since_last: float = current_time - self.last_actuation_time

        if (
            target_state != self.current_state
            and time_since_last >= self.min_actuation_interval
        ):
            self.current_state = target_state
            self.last_actuation_time = current_time
            return True

        return False

    def update_physical_model(self, target_angle: float, current_time: float) -> float:
        """
        Update physical model of the actuator

        Parameters:
        target_angle: The target steering angle
        current_time: Current system time

        Returns:
        float: The current actual steering angle
        """
        # Calculate time delta since last update
        dt: float = current_time - self.last_update_time
        self.last_update_time = current_time

        # First-order actuator model: dθ/dt = (θ_target - θ)/τ
        # Discretized: θ_new = θ + (θ_target - θ) * (dt/τ)
        if dt > 0:  # Avoid division by zero
            self.current_angle += (target_angle - self.current_angle) * (
                dt / self.time_constant
            )

        return self.current_angle

    def calculate_time_based_lookahead(
        self, closest_idx: int, lookahead_time: float = 0.5
    ):
        """
        Look ahead by time, not points

        Example: At 0.5 m/s with 0.5s lookahead = look 0.25m ahead
        At 1.0 m/s with 0.5s lookahead = look 0.5m ahead
        """
        if not hasattr(self, "path_resolution"):
            return 3  # Fallback to fixed points

        lookahead_distance = self.angular_velocity * lookahead_time
        look_ahead_points = int(lookahead_distance / self.path_resolution)

        return min(look_ahead_points, len(self.path_curvature) - closest_idx - 1)

    def calculate_distance_based_lookahead(
        self, closest_idx: int, lookahead_distance: float = 0.3
    ):
        """
        Look ahead by fixed distance regardless of path resolution

        Example: Always look 30cm ahead
        """
        if not hasattr(self, "path_resolution"):
            return 3  # Fallback

        look_ahead_points = int(lookahead_distance / self.path_resolution)
        return min(look_ahead_points, len(self.path_curvature) - closest_idx - 1)

    def calculate_adaptive_lookahead(self, closest_idx: int):
        """
        Look further ahead on straights, closer on curves

        Reasoning: Gentle curves need early preparation, sharp curves need immediate response
        """
        if closest_idx >= len(self.path_curvature):
            return 3

        current_curvature = abs(self.path_curvature[closest_idx])

        if current_curvature < 0.1:  # Straight or gentle curve
            base_lookahead = 5  # Look far ahead
        elif current_curvature < 0.5:  # Moderate curve
            base_lookahead = 3  # Standard lookahead
        else:  # Sharp curve
            base_lookahead = 1  # Look close ahead

        return min(base_lookahead, len(self.path_curvature) - closest_idx - 1)

    def calculate_weighted_curvature(self, closest_idx: int, max_lookahead: int = 5):
        """
        Instead of single point, use weighted average of upcoming curvature

        Gives smoother feedforward control
        """
        if closest_idx >= len(self.path_curvature):
            return 0.0

        total_weighted_curvature = 0.0
        total_weight = 0.0

        for i in range(
            1, min(max_lookahead + 1, len(self.path_curvature) - closest_idx)
        ):
            weight = 1.0 / i  # Closer points have higher weight
            curvature = self.path_curvature[closest_idx + i]

            total_weighted_curvature += weight * curvature
            total_weight += weight

        return total_weighted_curvature / total_weight if total_weight > 0 else 0.0

    def compute_steering(
        self, x: float, y: float, theta: float, wheelbase: float, dt: float = 0.1
    ) -> Tuple[float, float]:
        """
        Discrete RC car controller with fixed velocity and discrete steering

        Process:
        1. Use Pure Pursuit + PID to calculate desired steering angle
        2. Map to discrete steering state (left/neutral/right)
        3. Return actual steering angle + fixed velocity
        """

        # ========== PURE PURSUIT CONTROL ==========

        # Adjust reference point if using front axle
        calc_x, calc_y = (
            (x, y)
            if self.reference_point == "rear"
            else (x - wheelbase * np.cos(theta), y - wheelbase * np.sin(theta))
        )

        # Get target point
        target_x, target_y, closest_idx = self.get_target_point(calc_x, calc_y)
        if target_x is None or target_y is None:
            return 0.0, self.angular_velocity

        # Pure Pursuit: Calculate desired steering angle
        target_angle = np.arctan2(target_y - calc_y, target_x - calc_x)
        heading_error = target_angle - theta
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        # Apply dead zone to prevent oscillations on straights
        if abs(heading_error) < self.dead_zone_threshold:
            heading_error = 0.0

        # Update error history for PID
        self.error_history.append(heading_error)

        # Pure Pursuit base steering calculation
        lookahead_distance = self.calculate_adaptive_lookahead(closest_idx)

        if lookahead_distance > 0.01:
            base_steering = np.arctan2(
                2 * wheelbase * np.sin(heading_error), lookahead_distance
            )
        else:
            base_steering = 0.0

        # ========== PID CONTROL ==========

        # PID terms
        p_term = self.kp * heading_error

        # Integral term with anti-windup
        self.error_sum += heading_error * dt
        i_term = self.ki * self.error_sum
        i_term = np.clip(i_term, -0.5, 0.5)

        # Derivative term
        d_term = 0.0
        if len(self.error_history) > 1 and dt > 0:
            if abs(heading_error) > np.radians(0.5):
                d_term = self.kd * (heading_error - self.last_error) / dt

        self.last_error = heading_error

        # Feedforward term based on path curvature
        ff_term = 0.0
        if closest_idx < len(self.path_curvature):
            look_ahead = min(
                self.feedforward_lookahead_points,
                len(self.path_curvature) - closest_idx - 1,
            )

            future_idx = closest_idx + look_ahead
            path_curvature = self.path_curvature[future_idx]
            if abs(path_curvature) > 0.001:
                ff_term = 0.2 * path_curvature * wheelbase

            # Method 2: Time-based lookahead (more sophisticated)
            # Calculate how many path points represent the lookahead time
            # This accounts for varying path resolution and vehicle speed
            # if hasattr(self, "path_resolution"):  # meters per point
            #     lookahead_distance = (
            #         self.angular_velocity * self.feedforward_lookahead_time
            #     )
            #     look_ahead = min(
            #         int(lookahead_distance / self.path_resolution),
            #         len(self.path_curvature) - closest_idx - 1,
            #     )
            # else:
            #     look_ahead = self.feedforward_lookahead_points  # Fallback

            future_idx = closest_idx + look_ahead
            path_curvature = self.path_curvature[future_idx]

            if abs(path_curvature) > 0.001:
                ff_term = 0.2 * path_curvature * wheelbase

        # Combine all control terms
        control_correction = p_term + i_term + d_term + ff_term

        # Calculate desired steering angle
        desired_angle = base_steering + control_correction

        # Limit to physical steering constraints
        desired_angle = np.clip(
            desired_angle, -self.max_steering_angle, self.max_steering_angle
        )

        # ========== DISCRETE STEERING STATE SELECTION ==========

        # Apply your existing discrete steering logic
        current_time = time.time()
        target_state = self.determine_target_state(desired_angle)
        self.update_steering_state(target_state, current_time)

        # Get the actual steering angle from your discrete states
        target_angle_discrete = self.steering_states[self.current_state]
        actual_steering_angle = self.update_physical_model(
            target_angle_discrete, current_time
        )

        # Store for visualization/debugging
        self.desired_angles.append(desired_angle)
        self.actual_angles.append(actual_steering_angle)
        self.steering_states_history.append(self.current_state)

        return actual_steering_angle, self.angular_velocity

    def visualize_control(
        self,
        car_positions: List[Tuple[float, float, float, float]],
        time_steps: Optional[List[float]] = None,
    ) -> None:
        """Visualize the car's trajectory and controller performance"""
        # Create a figure with 4 subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))  # type: ignore

        # Plot 1: Path Following
        ax: matplotlib.axes.Axes = axs[0, 0]  # type: ignore

        # Plot reference path
        path_x: List[float] = [p[0] for p in self.path]
        path_y: List[float] = [p[1] for p in self.path]
        ax.plot(path_x, path_y, "b--", linewidth=2, label="Reference Path")

        # Plot car trajectory
        car_x: List[float] = [p[0] for p in car_positions]
        car_y: List[float] = [p[1] for p in car_positions]
        ax.plot(car_x, car_y, "r-", linewidth=2, label="Actual Trajectory")

        # Plot target points if recorded
        if self.target_points:
            target_x: List[float] = [p[0] for p in self.target_points]
            target_y: List[float] = [p[1] for p in self.target_points]
            ax.scatter(
                target_x,
                target_y,
                color="g",
                s=30,
                alpha=0.5,
                label="Target Points",
            )

        # Mark start and end
        if car_x and car_y:
            ax.scatter(car_x[0], car_y[0], color="g", s=100, label="Start")
            ax.scatter(car_x[-1], car_y[-1], color="r", s=100, label="End")

        ax.set_title("Path Following")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend()
        ax.grid(True)
        ax.set_aspect("equal")

        # Plot 2: Velocity Profile
        ax: matplotlib.axes.Axes = axs[0, 1]  # type: ignore

        if time_steps is not None:
            ax.plot(
                time_steps,
                [v_val for _, _, _, v_val in car_positions],
                "b-",
                linewidth=2,
            )
        else:
            ax.plot(
                range(len(car_positions)),
                [v_val for _, _, _, v_val in car_positions],
                "b-",
                linewidth=2,
            )

        ax.set_title("Velocity Profile")
        ax.set_xlabel("Step")
        ax.set_ylabel("Velocity (m/s)")
        ax.grid(True)

        # Plot 3: Steering Commands (Angular Velocity)
        ax: matplotlib.axes.Axes = axs[1, 0]  # type: ignore

        # Plot desired vs actual steering angles
        if self.desired_angles:
            if time_steps is not None:
                ax.plot(
                    time_steps[: len(self.desired_angles)],
                    [np.degrees(a) for a in self.desired_angles],
                    "g-",
                    linewidth=1,
                    alpha=0.7,
                    label="Desired Angle",
                )
                ax.plot(
                    time_steps[: len(self.actual_angles)],
                    [np.degrees(a) for a in self.actual_angles],
                    "r-",
                    linewidth=2,
                    label="Actual Angle",
                )
            else:
                ax.plot(
                    [np.degrees(a) for a in self.desired_angles],
                    "g-",
                    linewidth=1,
                    alpha=0.7,
                    label="Desired Angle",
                )
                ax.plot(
                    [np.degrees(a) for a in self.actual_angles],
                    "r-",
                    linewidth=2,
                    label="Actual Angle",
                )

            # Add horizontal lines for the discrete states
            ax.axhline(
                y=np.degrees(self.steering_states["left"]),
                color="k",
                linestyle="--",
                alpha=0.5,
            )
            ax.axhline(
                y=np.degrees(self.steering_states["neutral"]),
                color="k",
                linestyle="--",
                alpha=0.5,
            )
            ax.axhline(
                y=np.degrees(self.steering_states["right"]),
                color="k",
                linestyle="--",
                alpha=0.5,
            )

        ax.set_title("Steering Commands (Angular Velocity)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Steering Angle (degrees)")
        ax.legend()
        ax.grid(True)

        # Plot 4: Heading
        ax = axs[1, 1]  # type: ignore

        # Plot heading over time
        headings: List[float] = [p[2] for p in car_positions]
        if time_steps is not None:
            ax.plot(time_steps, [np.degrees(h) for h in headings], "g-", linewidth=2)
        else:
            ax.plot(
                range(len(headings)),
                [np.degrees(h) for h in headings],
                "g-",
                linewidth=2,
            )

        ax.set_title("Heading")
        ax.set_xlabel("Step")
        ax.set_ylabel("Heading (degrees)")
        ax.grid(True)

        plt.tight_layout()
        plt.show()  # type: ignore

    def get_steering_metrics(self) -> Dict[str, str | float]:
        """
        Returns current controller metrics for debugging and tuning

        Returns:
        dict: Dictionary of current controller metrics
        """
        return {
            "current_state": self.current_state,
            "current_angle": self.current_angle,
            "target_angle": self.steering_states[self.current_state],
            "time_since_last_actuation": time.time() - self.last_actuation_time,
            "desired_angle": (self.desired_angles[-1] if self.desired_angles else 0.0),
            "heading_error": self.error_history[-1] if self.error_history else 0.0,
            "p_term": self.kp * (self.error_history[-1] if self.error_history else 0.0),
            "i_term": self.ki * self.error_sum,
        }

    # def compute_steering(
    #     self, x: float, y: float, theta: float, wheelbase: float, dt: float = 0.1
    # ) -> Tuple[float, float]:
    #     """
    #     Compute steering for constant angular velocity with discrete angles

    #     Parameters:
    #     x, y: Current position
    #     theta: Current orientation (radians)
    #     wheelbase: Distance between front and rear axles
    #     dt: Time step for derivative/integral calculations

    #     Returns:
    #     steering_angle: Actual steering angle (radians)
    #     linear_velocity: Desired linear velocity (m/s)
    #     """
    #     # Adjust reference point if using front axle
    #     calc_x: float
    #     calc_y: float
    #     if self.reference_point == "front":
    #         # Convert front to rear axle position for calculations
    #         rear_x: float = x - wheelbase * np.cos(theta)
    #         rear_y: float = y - wheelbase * np.sin(theta)
    #         calc_x, calc_y = rear_x, rear_y
    #     else:
    #         calc_x, calc_y = x, y

    #     # Get target point
    #     target_x, target_y, closest_idx = self.get_target_point(calc_x, calc_y)

    #     # If no target found, maintain speed
    #     if target_x is None or target_y is None:
    #         return 0.0, self.angular_velocity

    #     # Calculate angle to target in vehicle frame
    #     target_angle: float = np.arctan2(target_y - calc_y, target_x - calc_x)

    #     # Calculate heading error
    #     heading_error: float = target_angle - theta

    #     # Normalize to [-pi, pi]
    #     heading_error = float(np.arctan2(np.sin(heading_error), np.cos(heading_error)))

    #     # Apply a dead zone to heading error to prevent oscillations on straight paths
    #     dead_zone_threshold: float = np.radians(0.2)  # e.g., 0.2 degrees
    #     if abs(heading_error) < dead_zone_threshold:
    #         heading_error = 0.0

    #     # Update error history for PID
    #     self.error_history.append(heading_error)
    #     # if len(self.error_history) > 20:  # Limit history size
    #     #     self.error_history.pop(0)

    #     # Calculate curvature for Pure Pursuit
    #     curvature: float = 2 * np.sin(heading_error) / self.base_lookahead

    #     # For constant angular velocity:
    #     # angular_velocity = curvature * linear_velocity
    #     # Therefore: linear_velocity = angular_velocity / curvature

    #     # Handle straight segments (avoid division by zero)
    #     linear_velocity = self.angular_velocity / abs(curvature)

    #     # Clip linear velocity to maximum allowed
    #     # linear_velocity = np.clip(linear_velocity, 0.0, self.max_linear_velocity)

    #     # PID control for heading correction
    #     p_term: float = self.kp * heading_error

    #     # Integral term
    #     self.error_sum += heading_error * dt
    #     i_term: float = self.ki * self.error_sum
    #     # Anti-windup
    #     i_term = np.clip(i_term, -0.5, 0.5)

    #     # Derivative term
    #     d_term: float = 0.0
    #     if len(self.error_history) > 1:
    #         if dt > 0:
    #             # Only apply derivative term if heading error is significant
    #             if abs(heading_error) > np.radians(0.5):  # Threshold of 0.5 degrees
    #                 d_term = self.kd * (heading_error - self.last_error) / dt

    #     self.last_error = heading_error

    #     # Feedforward term based on path curvature
    #     ff_term: float = 0.0
    #     if closest_idx < len(self.path_curvature):
    #         # Look ahead for upcoming curvature
    #         look_ahead: int = min(3, len(self.path_curvature) - closest_idx - 1)
    #         future_idx: int = closest_idx + look_ahead
    #         path_curvature: float = self.path_curvature[future_idx]

    #         # Explicitly zero out feedforward term if path is straight
    #         if (
    #             abs(path_curvature) < 0.001
    #         ):  # Use the same threshold as for linear_velocity and base_steering
    #             ff_term = 0.0
    #         else:
    #             ff_term = 0.2 * path_curvature * wheelbase  # Feedforward gain

    #     # Combine all control terms
    #     control_correction: float = p_term + i_term + d_term + ff_term

    #     # For Ackermann steering with constant angular velocity:
    #     # θ̇ = v * tan(φ) / L
    #     # Solving for φ: φ = arctan(θ̇ * L / v)
    #     base_steering: float
    #     if abs(curvature) < 0.001:  # Use the same threshold as for linear_velocity
    #         base_steering = 0.0
    #     else:
    #         base_steering = np.arctan2(
    #             self.angular_velocity * wheelbase, linear_velocity
    #         )
    #         # Apply sign from curvature direction
    #         base_steering *= np.sign(curvature)

    #     # Add PID correction to base steering
    #     desired_angle: float = base_steering + control_correction

    #     # Limit steering angle
    #     max_steering: float = np.radians(35)  # Maximum steering angle
    #     desired_angle = np.clip(desired_angle, -max_steering, max_steering)

    #     # Store the desired angle for visualization/debugging
    #     self.desired_angles.append(desired_angle)

    #     # ---- Discrete steering logic ----
    #     # Get current time for rate limiting
    #     current_time: float = time.time()

    #     # Determine target discrete state
    #     target_state: str = self.determine_target_state(desired_angle)

    #     # Update steering state (respecting rate limits)
    #     self.update_steering_state(target_state, current_time)

    #     # Get target angle for the current state
    #     target_angle_discrete: float = self.steering_states[self.current_state]

    #     # Update physical model to get actual steering angle
    #     actual_steering_angle: float = self.update_physical_model(
    #         target_angle_discrete, current_time
    #     )

    #     # Store actual angle and state for tracking
    #     self.actual_angles.append(actual_steering_angle)
    #     self.steering_states_history.append(self.current_state)

    #     return actual_steering_angle, linear_velocity


###################################
# Start conservative, increase until you see oscillation
# max_integral_degrees = 5.0  # Start here
# Test on tight curves - if car consistently undershoots, increase
# If you see overshoot/oscillation after tight curves, decrease

# Alternative approach: Base on your discrete steering angles
# If your discrete states are ±15°, maybe limit integral to ±5°
# max_integral_correction = 0.3 * np.radians(15)  # 30% of discrete step size
###################################
# Method 1: Based on your car's response time
# How long does it take your servo to move from neutral to full deflection?
# servo_response_time = 0.2  # seconds
# lookahead_time = 2 * servo_response_time  # Look ahead 2x response time

# Method 2: Based on path resolution
# If your path points are 5cm apart and you want to look 20cm ahead:
# lookahead_distance = 0.20  # meters
# lookahead_points = int(lookahead_distance / path_resolution)

# Method 3: Start with distance-based, tune empirically
# Test on S-curves: too small = jerky, too large = cuts corners
###################################

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
        base_lookahead: float = 0.5,
        reference_point: str = "rear",
        use_feedforward: bool = True,
        min_actuation_interval: float = 0.3,
        time_constant: float = 0.2,
        left_angle: float = 15,
        right_angle: float = -15,
    ) -> None:
        # Controller parameters
        self.angular_velocity: float = angular_velocity
        self.base_lookahead: float = base_lookahead
        self.min_lookahead: float = 0.2
        self.max_lookahead: float = 1.5
        self.reference_point: str = reference_point  # 'front' or 'rear'
        self.use_feedforward: bool = use_feedforward

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
        self.steering_states: Dict[str, float] = {
            "left": np.radians(left_angle),  # Left steering angle
            "neutral": 0.0,  # Neutral (center) steering
            "right": np.radians(right_angle),  # Right steering angle
        }

        # Hysteresis thresholds to prevent oscillation
        self.threshold_to_left: float = np.radians(10)  # Threshold to switch to left
        self.threshold_to_right: float = np.radians(-10)  # Threshold to switch to right
        self.threshold_to_neutral: float = np.radians(5)  # Threshold band for neutral

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

    def compute_steering(
        self, x: float, y: float, theta: float, wheelbase: float, dt: float = 0.1
    ) -> Tuple[float, float]:
        """
        Compute steering for constant angular velocity with discrete angles

        Parameters:
        x, y: Current position
        theta: Current orientation (radians)
        wheelbase: Distance between front and rear axles
        dt: Time step for derivative/integral calculations

        Returns:
        steering_angle: Actual steering angle (radians)
        linear_velocity: Desired linear velocity (m/s)
        """
        # Adjust reference point if using front axle
        calc_x: float
        calc_y: float
        if self.reference_point == "front":
            # Convert front to rear axle position for calculations
            rear_x: float = x - wheelbase * np.cos(theta)
            rear_y: float = y - wheelbase * np.sin(theta)
            calc_x, calc_y = rear_x, rear_y
        else:
            calc_x, calc_y = x, y

        # Get target point
        target_x, target_y, closest_idx = self.get_target_point(calc_x, calc_y)

        # If no target found, maintain speed
        if target_x is None or target_y is None:
            return 0.0, self.angular_velocity

        # Calculate angle to target in vehicle frame
        target_angle: float = np.arctan2(target_y - calc_y, target_x - calc_x)

        # Calculate heading error
        heading_error: float = target_angle - theta

        # Normalize to [-pi, pi]
        heading_error = float(np.arctan2(np.sin(heading_error), np.cos(heading_error)))

        # Update error history for PID
        self.error_history.append(heading_error)
        # if len(self.error_history) > 20:  # Limit history size
        #     self.error_history.pop(0)

        # Calculate curvature for Pure Pursuit
        curvature: float = 2 * np.sin(heading_error) / self.base_lookahead

        # For constant angular velocity:
        # angular_velocity = curvature * linear_velocity
        # Therefore: linear_velocity = angular_velocity / curvature

        # Handle straight segments (avoid division by zero)
        linear_velocity: float
        if abs(curvature) < 0.001:
            linear_velocity = 0.5  # Default velocity for straight segments
        else:
            # Compute linear velocity for constant angular velocity
            linear_velocity = self.angular_velocity / abs(curvature)

        # PID control for heading correction
        p_term: float = self.kp * heading_error

        # Integral term
        self.error_sum += heading_error * dt
        i_term: float = self.ki * self.error_sum
        # Anti-windup
        i_term = np.clip(i_term, -0.5, 0.5)

        # Derivative term
        d_term: float = 0.0
        if len(self.error_history) > 1:
            # n_samples = min(5, len(self.error_history))
            # recent_errors = list(self.error_history)[-n_samples:]
            if dt > 0:
                # derivative = (recent_errors[-1] - recent_errors[0]) / (n_samples * dt)
                d_term = self.kd * (heading_error - self.last_error) / dt

        self.last_error = heading_error

        # Feedforward term based on path curvature
        ff_term: float = 0.0
        if self.use_feedforward and closest_idx < len(self.path_curvature):
            # Look ahead for upcoming curvature
            look_ahead: int = min(3, len(self.path_curvature) - closest_idx - 1)
            future_idx: int = closest_idx + look_ahead
            path_curvature: float = self.path_curvature[future_idx]
            ff_term = 0.2 * path_curvature * wheelbase  # Feedforward gain

        # Combine all control terms
        control_correction: float = p_term + i_term + d_term + ff_term

        # For Ackermann steering with constant angular velocity:
        # θ̇ = v * tan(φ) / L
        # Solving for φ: φ = arctan(θ̇ * L / v)
        base_steering: float = np.arctan2(
            self.angular_velocity * wheelbase, linear_velocity
        )

        # Apply sign from curvature direction
        base_steering *= np.sign(curvature)

        # Add PID correction to base steering
        desired_angle: float = base_steering + control_correction

        # Limit steering angle
        max_steering: float = np.radians(35)  # Maximum steering angle
        desired_angle = np.clip(desired_angle, -max_steering, max_steering)

        # Store the desired angle for visualization/debugging
        self.desired_angles.append(desired_angle)

        # ---- Discrete steering logic ----
        # Get current time for rate limiting
        current_time: float = time.time()

        # Determine target discrete state
        target_state: str = self.determine_target_state(desired_angle)

        # Update steering state (respecting rate limits)
        self.update_steering_state(target_state, current_time)

        # Get target angle for the current state
        target_angle_discrete: float = self.steering_states[self.current_state]

        # Update physical model to get actual steering angle
        actual_steering_angle: float = self.update_physical_model(
            target_angle_discrete, current_time
        )

        # Store actual angle and state for tracking
        self.actual_angles.append(actual_steering_angle)
        self.steering_states_history.append(self.current_state)

        # Calculate angular velocity from actual steering angle for Ackermann
        # if abs(linear_velocity) > 1e-5:  # Only calculate if moving
        #     self.angular_velocity = (
        #         linear_velocity * np.tan(actual_steering_angle) / wheelbase
        #     )
        # else:
        #     self.angular_velocity = 0.0

        return actual_steering_angle, linear_velocity

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
                target_x, target_y, color="g", s=30, alpha=0.5, label="Target Points"
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
            "desired_angle": self.desired_angles[-1] if self.desired_angles else 0.0,
            "heading_error": self.error_history[-1] if self.error_history else 0.0,
            "p_term": self.kp * (self.error_history[-1] if self.error_history else 0.0),
            "i_term": self.ki * self.error_sum,
        }

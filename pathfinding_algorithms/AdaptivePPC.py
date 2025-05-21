import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class AdaptivePurePursuitController:
    """
    Implements an enhanced Pure Pursuit controller with cascaded PID control,
    gain scheduling, and feedforward control for path following.

    This controller improves on the basic Pure Pursuit algorithm by:
    1. Using cascaded PIDs for both path following and velocity control
    2. Implementing gain scheduling based on velocity
    3. Adding feedforward terms based on path curvature
    4. Adapting parameters based on observed performance
    5. Dynamically adjusting lookahead distance based on speed and curvature
    6. Supporting different modes for exploration and racing phases
    """

    def __init__(self, base_lookahead=0.5, history_size=100, mode="exploration"):
        """
        Initialize the controller

        Parameters:
        base_lookahead: Base lookahead distance when stationary (meters)
        history_size: Size of the history buffer for parameter adaptation
        mode: Either "exploration" or "racing" to set initial parameters
        """
        # Path parameters
        self.path = []  # List of (x, y) points defining the path
        self.base_lookahead = base_lookahead
        self.min_lookahead = 0.2  # Minimum lookahead distance (meters)
        self.max_lookahead = 1.5  # Maximum lookahead distance (meters)
        self.curvature_factor = (
            2.0  # Factor to adjust lookahead based on path curvature
        )

        # PID parameters for steering control
        self.kp_steering = 1.0  # Proportional gain
        self.ki_steering = 0.0  # Integral gain
        self.kd_steering = 0.1  # Derivative gain

        # PID parameters for velocity control
        self.kp_velocity = 1.0  # Proportional gain
        self.ki_velocity = 0.1  # Integral gain
        self.kd_velocity = 0.05  # Derivative gain

        # Feedforward parameters
        self.use_feedforward = True
        self.feedforward_gain = 1.0  # Gain for curvature-based feedforward
        self.velocity_curvature_factor = 3.0  # Speed reduction in curves
        self.anticipation_distance = 5  # Look-ahead points for velocity planning

        # Gain scheduling parameters for exploration
        self.exploration_gains = {
            "slow": {
                "kp": 1.5,
                "ki": 0.05,
                "kd": 0.2,
                "lookahead_factor": 0.7,
                "ff_gain": 0.8,
            },
            "medium": {
                "kp": 1.2,
                "ki": 0.02,
                "kd": 0.25,
                "lookahead_factor": 0.8,
                "ff_gain": 0.9,
            },
            "fast": {
                "kp": 1.0,
                "ki": 0.01,
                "kd": 0.3,
                "lookahead_factor": 0.9,
                "ff_gain": 1.0,
            },
        }

        # Gain scheduling parameters for racing
        self.racing_gains = {
            "slow": {
                "kp": 1.2,
                "ki": 0.1,
                "kd": 0.05,
                "lookahead_factor": 0.8,
                "ff_gain": 1.0,
            },
            "medium": {
                "kp": 1.0,
                "ki": 0.05,
                "kd": 0.1,
                "lookahead_factor": 1.0,
                "ff_gain": 1.1,
            },
            "fast": {
                "kp": 0.8,
                "ki": 0.01,
                "kd": 0.2,
                "lookahead_factor": 1.3,
                "ff_gain": 1.2,
            },
        }

        # Set active gain schedule based on mode
        self.mode = mode
        self.gain_schedule = (
            self.exploration_gains if mode == "exploration" else self.racing_gains
        )

        # Error history for integral and derivative terms
        self.steering_error_history = deque(maxlen=history_size)
        self.velocity_error_history = deque(maxlen=history_size)

        # Performance history for parameter adaptation
        self.path_error_history = deque(maxlen=history_size)
        self.control_effort_history = deque(maxlen=history_size)
        self.feedforward_terms = deque(maxlen=history_size)

        # Path progress tracking
        self.current_segment = 0
        self.path_completion = 0.0  # 0.0 to 1.0 representing progress along path

        # Performance metrics
        self.cross_track_error = 0.0
        self.heading_error = 0.0

        # Debug and visualization
        self.debug_mode = False
        self.fig = None
        self.axes = None

        # Set initial mode-specific parameters
        self.set_mode(mode)

    def set_mode(self, mode):
        """
        Set controller mode between exploration and racing

        Parameters:
        mode: Either "exploration" or "racing"
        """
        self.mode = mode
        if mode == "exploration":
            self.gain_schedule = self.exploration_gains
            # Exploration-specific settings
            self.velocity_curvature_factor = 4.0  # More conservative speed in curves
            self.anticipation_distance = 3  # Shorter anticipation for exploration
            print(f"Controller set to EXPLORATION mode with increased safety margins")
        else:  # Racing mode
            self.gain_schedule = self.racing_gains
            # Racing-specific settings
            self.velocity_curvature_factor = 3.0  # More aggressive speed in curves
            self.anticipation_distance = 5  # Longer anticipation for racing
            print(
                f"Controller set to RACING mode with optimized performance parameters"
            )

        # Reset error history when changing modes
        self.steering_error_history.clear()
        self.velocity_error_history.clear()

        # Keep performance history for adaptation between phases
        # but separate performance metrics for clearer reporting
        return self.gain_schedule

    def set_path(self, path):
        """
        Set the path to follow

        Parameters:
        path: List of (x, y) points defining the path
        """
        if not path:
            print("Warning: Empty path provided!")
            return

        self.path = path
        self.current_segment = 0
        self.path_completion = 0.0

        # Calculate path curvature at each point (for lookahead adjustment)
        self.path_curvature = self._calculate_path_curvature()
        self.path_length = self._calculate_path_length()

        # Reset error history when path changes
        self.steering_error_history.clear()
        self.velocity_error_history.clear()

        # Keep recording performance history for adaptation
        print(f"New path set: {len(path)} points, {self.path_length:.2f}m total length")

    def _calculate_path_length(self):
        """
        Calculate the total length of the path

        Returns:
        Total path length in meters
        """
        length = 0.0
        for i in range(1, len(self.path)):
            p1 = self.path[i - 1]
            p2 = self.path[i]
            length += np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        return length

    def _calculate_path_curvature(self):
        """
        Calculate curvature at each point of the path

        Returns:
        List of curvature values for each path segment
        """
        if len(self.path) < 3:
            return [0.0] * len(self.path)

        curvature = [0.0] * len(self.path)

        # Calculate curvature using three consecutive points
        for i in range(1, len(self.path) - 1):
            p1 = np.array(self.path[i - 1])
            p2 = np.array(self.path[i])
            p3 = np.array(self.path[i + 1])

            # Calculate vectors
            v1 = p2 - p1
            v2 = p3 - p2

            # Calculate angle between vectors
            dot_product = np.dot(v1, v2)
            mag_v1 = np.linalg.norm(v1)
            mag_v2 = np.linalg.norm(v2)

            if mag_v1 * mag_v2 == 0:
                curvature[i] = 0.0
                continue

            cos_angle = dot_product / (mag_v1 * mag_v2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure in valid range
            angle = np.arccos(cos_angle)

            # Calculate sign of curvature (left or right turn)
            cross_z = np.cross(v1, v2)
            if isinstance(cross_z, np.ndarray):
                # Handle case where cross returns a vector
                cross_z = cross_z if cross_z.size > 0 else 0
            sign = 1 if cross_z >= 0 else -1

            # Curvature is inversely proportional to radius
            segment_length = (mag_v1 + mag_v2) / 2
            if segment_length > 0:
                # Apply sign to distinguish left vs right turns
                curvature[i] = sign * angle / segment_length
            else:
                curvature[i] = 0.0

        # Set first and last points to their neighbors
        curvature[0] = curvature[1]
        curvature[-1] = curvature[-2]

        return curvature

    def _get_adjusted_lookahead(self, velocity, path_idx):
        """
        Dynamically adjust lookahead distance based on velocity and path curvature

        Parameters:
        velocity: Current linear velocity (m/s)
        path_idx: Current index in the path for curvature lookup

        Returns:
        Adjusted lookahead distance (m)
        """
        # Velocity-based adjustment
        speed_factor = abs(velocity) + 0.5  # Add small constant to avoid zero

        # Get appropriate gain schedule based on velocity
        if abs(velocity) < 0.3:
            schedule = self.gain_schedule["slow"]
        elif abs(velocity) < 0.7:
            schedule = self.gain_schedule["medium"]
        else:
            schedule = self.gain_schedule["fast"]

        # Apply lookahead factor from gain schedule
        lookahead_factor = schedule["lookahead_factor"]

        # Curvature-based adjustment (reduce lookahead in curves)
        curvature_idx = min(path_idx, len(self.path_curvature) - 1)
        curvature = abs(self.path_curvature[curvature_idx]) if curvature_idx >= 0 else 0

        # Look ahead for upcoming curvature change
        curvature_change = 0
        if path_idx < len(self.path_curvature) - 1:
            next_curvature = abs(self.path_curvature[path_idx + 1])
            curvature_change = abs(next_curvature - curvature)

        # Reduce lookahead more for high curvature and approaching curve transitions
        curvature_adjustment = 1.0 / (
            1.0 + self.curvature_factor * curvature + 2.0 * curvature_change
        )

        # Calculate final lookahead distance
        lookahead = (
            self.base_lookahead * speed_factor * lookahead_factor * curvature_adjustment
        )

        # Ensure within bounds
        return np.clip(lookahead, self.min_lookahead, self.max_lookahead)

    def get_target_point(self, x, y, velocity):
        """
        Find the target point on the path at adaptive lookahead distance

        Parameters:
        x, y: Current position of the car
        velocity: Current velocity for lookahead adjustment

        Returns:
        target_x, target_y: Coordinates of the target point, or None if not found
        closest_idx: Index of the closest point on the path
        """
        # If path is empty, return None
        if not self.path:
            return None, None, 0

        # Find the closest point on the path
        min_dist = float("inf")
        closest_idx = 0

        for i, (path_x, path_y) in enumerate(self.path):
            dist = np.sqrt((path_x - x) ** 2 + (path_y - y) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Update path completion metric
        self.path_completion = closest_idx / max(1, len(self.path) - 1)

        # Calculate lookahead distance based on velocity and curvature
        lookahead = self._get_adjusted_lookahead(velocity, closest_idx)

        # Search forward from the closest point to find a point at lookahead distance
        target_idx = closest_idx
        target_dist = min_dist

        # Check if we should wrap around for closed paths
        is_closed_path = False
        if len(self.path) > 2:
            start_point = self.path[0]
            end_point = self.path[-1]
            end_dist = np.sqrt(
                (end_point[0] - start_point[0]) ** 2
                + (end_point[1] - start_point[1]) ** 2
            )
            is_closed_path = (
                end_dist < 0.5
            )  # If start and end are close, consider it closed

        # Set max search range based on path type
        max_search = len(self.path) if is_closed_path else len(self.path) - 1

        # Search for target point
        search_count = 0
        while target_dist < lookahead and search_count < max_search:
            target_idx = (target_idx + 1) % len(
                self.path
            )  # Wrap around for closed paths
            target_dist = np.sqrt(
                (self.path[target_idx][0] - x) ** 2
                + (self.path[target_idx][1] - y) ** 2
            )
            search_count += 1

            # For non-closed paths, stop at the end
            if not is_closed_path and target_idx == len(self.path) - 1:
                break

        # If we reached the end of the path, use the last point
        if (
            target_idx == len(self.path) - 1
            and target_dist < lookahead
            and not is_closed_path
        ):
            return self.path[-1][0], self.path[-1][1], closest_idx

        # Calculate cross-track error for diagnostic purposes
        self.cross_track_error = min_dist

        return self.path[target_idx][0], self.path[target_idx][1], closest_idx

    def _update_pid_gains(self, velocity):
        """
        Update PID gains based on velocity using gain scheduling

        Parameters:
        velocity: Current velocity (m/s)

        Returns:
        Dictionary of updated gain values
        """
        # Select appropriate gain schedule based on velocity
        if abs(velocity) < 0.3:
            schedule = self.gain_schedule["slow"]
        elif abs(velocity) < 0.7:
            schedule = self.gain_schedule["medium"]
        else:
            schedule = self.gain_schedule["fast"]

        # Scale gains based on specific velocity within range
        speed_factor = min(1.0, 0.5 + 0.5 * (abs(velocity) / 1.0))

        # Scale different gains differently based on speed
        # - Reduce proportional gain at higher speeds
        # - Reduce integral gain at higher speeds (less steady-state correction needed)
        # - Increase derivative gain at higher speeds (more damping)
        kp = schedule["kp"] * (1.1 - 0.3 * speed_factor)
        ki = schedule["ki"] * (1.2 - 0.4 * speed_factor)
        kd = schedule["kd"] * (0.8 + 0.4 * speed_factor)

        # Get feedforward gain from schedule
        ff_gain = schedule.get("ff_gain", 1.0)

        return {"kp": kp, "ki": ki, "kd": kd, "ff_gain": ff_gain}

    def compute_steering(self, x, y, theta, velocity, dt):
        """
        Compute steering and velocity commands using cascaded PID control
        with feedforward terms

        Parameters:
        x, y: Current position
        theta: Current orientation (radians)
        velocity: Current linear velocity
        dt: Time step for integral and derivative calculations

        Returns:
        angular_velocity: Desired angular velocity (steering command)
        linear_velocity: Desired linear velocity
        """
        # Get target point using adaptive lookahead
        target_x, target_y, closest_idx = self.get_target_point(x, y, velocity)

        # If no target point found, maintain current state
        if target_x is None:
            return 0.0, velocity

        # Calculate angle to target in vehicle frame
        target_angle = np.arctan2(target_y - y, target_x - x)

        # Calculate heading error (difference between current heading and target direction)
        heading_error = target_angle - theta

        # Normalize to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # Store for diagnostics
        self.heading_error = heading_error

        # Get scheduled PID gains based on velocity
        gains = self._update_pid_gains(velocity)

        # Add error to history
        self.steering_error_history.append(heading_error)

        # Calculate PID terms for steering
        p_term = gains["kp"] * heading_error

        # Integral term with anti-windup
        i_term = 0
        if len(self.steering_error_history) > 1:
            integral = sum(self.steering_error_history) * dt
            i_term = gains["ki"] * integral
            # Anti-windup: limit integral term
            i_term = np.clip(i_term, -0.5, 0.5)

        # Derivative term (use only last few samples to reduce noise sensitivity)
        d_term = 0
        if len(self.steering_error_history) > 2:
            n_samples = min(5, len(self.steering_error_history))
            recent_errors = list(self.steering_error_history)[-n_samples:]
            if dt > 0:
                derivative = (recent_errors[-1] - recent_errors[0]) / (n_samples * dt)
                d_term = gains["kd"] * derivative

        # Feedforward term based on path curvature
        ff_term = 0
        if self.use_feedforward and closest_idx < len(self.path_curvature):
            # Current curvature
            curvature = self.path_curvature[closest_idx]

            # Basic curvature-based feedforward: ω = v * κ
            # The linear relationship between speed, curvature, and angular velocity
            ff_term = velocity * curvature * gains["ff_gain"]

            # Store feedforward term for analysis
            self.feedforward_terms.append(ff_term)

        # Combine feedforward and PID terms for angular velocity command
        angular_velocity = ff_term + p_term + i_term + d_term

        # Store control effort and path error for adaptation
        self.control_effort_history.append(abs(angular_velocity))
        self.path_error_history.append(abs(heading_error))

        # Velocity control based on path curvature and tracking error
        # Look ahead for upcoming curvature to anticipate curves
        max_curvature = 0
        for i in range(
            min(self.anticipation_distance, len(self.path_curvature) - closest_idx)
        ):
            idx = closest_idx + i
            if idx < len(self.path_curvature):
                max_curvature = max(max_curvature, abs(self.path_curvature[idx]))

        # Speed reduction factor based on curvature and tracking error
        # Higher values of velocity_curvature_factor make speed more sensitive to curves
        reduction = 1.0 / (
            1.0
            + self.velocity_curvature_factor * max_curvature
            + 2.0 * abs(heading_error)
        )

        # Set base velocity depending on mode
        if self.mode == "exploration":
            base_velocity = 0.4  # Conservative speed for exploration
        else:
            base_velocity = 0.8  # Higher speed for racing

        target_velocity = base_velocity * reduction

        # Simple proportional control for velocity
        # More complex velocity control could be implemented here
        velocity_error = target_velocity - velocity
        self.velocity_error_history.append(velocity_error)

        linear_velocity = velocity + 0.5 * velocity_error  # Simple P control

        return angular_velocity, linear_velocity

    def adapt_parameters(self):
        """
        Adapt controller parameters based on observed performance

        This should be called between runs to update parameters based on
        collected performance data.

        Returns:
        Dictionary of adapted parameters
        """
        if len(self.path_error_history) < 10 or len(self.control_effort_history) < 10:
            print("Not enough performance data to adapt parameters")
            return None

        # Calculate performance metrics
        mean_path_error = np.mean(self.path_error_history)
        max_path_error = np.max(self.path_error_history)
        mean_control_effort = np.mean(self.control_effort_history)

        # Calculate feedforward effectiveness (if used)
        ff_correlation = 0
        if self.use_feedforward and len(self.feedforward_terms) > 10:
            # Correlation between feedforward terms and control effort
            # Higher correlation means feedforward is working well
            ff_array = np.array(list(self.feedforward_terms))
            effort_array = np.array(list(self.control_effort_history))
            if len(ff_array) > len(effort_array):
                ff_array = ff_array[: len(effort_array)]
            elif len(effort_array) > len(ff_array):
                effort_array = effort_array[: len(ff_array)]

            # Calculate correlation if arrays are not constant
            if np.std(ff_array) > 0 and np.std(effort_array) > 0:
                ff_correlation = np.corrcoef(ff_array, effort_array)[0, 1]

        # Print performance report
        print(f"\nPerformance metrics ({self.mode} mode):")
        print(f"  Mean path error: {mean_path_error:.4f} rad")
        print(f"  Max path error: {max_path_error:.4f} rad")
        print(f"  Mean control effort: {mean_control_effort:.4f}")
        if self.use_feedforward:
            print(f"  Feedforward correlation: {ff_correlation:.4f}")

        # Adjust base parameters based on performance

        # If errors are large but control effort is low, increase gains
        if mean_path_error > 0.1 and mean_control_effort < 0.3:
            adjustment_factor = 1.2
            print("Increasing control gains for better tracking")
        # If errors are small but control effort is high, decrease gains
        elif mean_path_error < 0.05 and mean_control_effort > 0.5:
            adjustment_factor = 0.9
            print("Decreasing control gains for smoother control")
        else:
            adjustment_factor = 1.0
            print("Keeping current gain values")

        # Apply adjustments to gain schedules
        for speed in self.gain_schedule:
            self.gain_schedule[speed]["kp"] *= adjustment_factor

        # Adjust lookahead parameters based on max error
        if max_path_error > 0.2:
            # If large errors occurred, reduce lookahead to be more responsive
            self.base_lookahead *= 0.9
            print(
                f"Reducing base lookahead to {self.base_lookahead:.2f} for better responsiveness"
            )
        elif max_path_error < 0.1:
            # If tracking was good, can increase lookahead for smoother paths
            self.base_lookahead *= 1.1
            print(
                f"Increasing base lookahead to {self.base_lookahead:.2f} for smoother paths"
            )

        # Adjust feedforward gain based on correlation
        if self.use_feedforward:
            if ff_correlation > 0.7:
                # High correlation means feedforward is effective, can increase its influence
                for speed in self.gain_schedule:
                    self.gain_schedule[speed]["ff_gain"] *= 1.1
                print(f"Increasing feedforward influence for better predictive control")
            elif ff_correlation < 0.3:
                # Low correlation means feedforward may be ineffective or counter-productive
                for speed in self.gain_schedule:
                    self.gain_schedule[speed]["ff_gain"] *= 0.9
                print(f"Reducing feedforward influence due to poor correlation")

        # Ensure lookahead stays within reasonable bounds
        self.base_lookahead = np.clip(self.base_lookahead, 0.3, 1.0)

        # Return the adapted parameters
        return {
            "base_lookahead": self.base_lookahead,
            "gain_schedules": self.gain_schedule,
            "mode": self.mode,
            "metrics": {
                "mean_path_error": mean_path_error,
                "max_path_error": max_path_error,
                "mean_control_effort": mean_control_effort,
                "feedforward_correlation": (
                    ff_correlation if self.use_feedforward else 0
                ),
            },
        }

    def plot_performance(self):
        """
        Plot controller performance metrics

        This can be used between runs to visualize performance and tune parameters
        """
        if not self.path_error_history or not self.control_effort_history:
            print("No performance data to plot")
            return

        # Create a new figure
        plt.figure(figsize=(14, 10))

        # Create subplot layout
        n_plots = 4 if self.use_feedforward and self.feedforward_terms else 3

        # Plot 1: Path errors
        plt.subplot(n_plots, 1, 1)
        plt.plot(list(self.path_error_history), "b-", label="Path Error")
        plt.axhline(
            y=np.mean(self.path_error_history),
            color="r",
            linestyle="--",
            label=f"Mean: {np.mean(self.path_error_history):.4f}",
        )
        plt.title(f"Path Tracking Error ({self.mode} mode)")
        plt.ylabel("Error (rad)")
        plt.grid(True)
        plt.legend()

        # Plot 2: Control effort
        plt.subplot(n_plots, 1, 2)
        plt.plot(list(self.control_effort_history), "g-", label="Control Effort")
        plt.axhline(
            y=np.mean(self.control_effort_history),
            color="r",
            linestyle="--",
            label=f"Mean: {np.mean(self.control_effort_history):.4f}",
        )
        plt.title("Control Effort (Angular Velocity Commands)")
        plt.ylabel("Effort")
        plt.grid(True)
        plt.legend()

        # Plot 3: Heading error vs target
        if (
            hasattr(self, "current_theta_history")
            and len(self.current_theta_history) > 0
        ):
            plt.subplot(n_plots, 1, 3)
            plt.plot(self.current_theta_history, "b-", label="Current Heading")
            plt.plot(self.desired_heading_history, "g-", label="Desired Heading")
            plt.plot(self.goal_angle_history, "r-", label="Goal Angle")
            plt.title("Heading Tracking")
            plt.ylabel("Angle (rad)")
            plt.grid(True)
            plt.legend()
        else:
            plt.subplot(n_plots, 1, 3)
            plt.plot(list(self.steering_error_history), "m-", label="Steering Errors")
            plt.title("Steering Error History")
            plt.ylabel("Error (rad)")
            plt.grid(True)
            plt.legend()

        # Plot 4: Feedforward terms (if enabled)
        if self.use_feedforward and self.feedforward_terms:
            plt.subplot(n_plots, 1, 4)
            plt.plot(list(self.feedforward_terms), "c-", label="Feedforward Term")

            # Plot control effort on same axis for comparison
            if len(self.control_effort_history) > 0:
                # Match lengths for comparison
                ff_data = list(self.feedforward_terms)
                ce_data = list(self.control_effort_history)
                min_len = min(len(ff_data), len(ce_data))
                plt.plot(ce_data[:min_len], "g--", alpha=0.7, label="Control Effort")

                # Calculate correlation if possible
                if min_len > 5:
                    ff_array = np.array(ff_data[:min_len])
                    ce_array = np.array(ce_data[:min_len])
                    if np.std(ff_array) > 0 and np.std(ce_array) > 0:
                        corr = np.corrcoef(ff_array, ce_array)[0, 1]
                        plt.title(f"Feedforward Contribution (Correlation: {corr:.4f})")
                    else:
                        plt.title("Feedforward Contribution")
                else:
                    plt.title("Feedforward Contribution")
            else:
                plt.title("Feedforward Contribution")

            plt.xlabel("Sample")
            plt.ylabel("Magnitude")
            plt.grid(True)
            plt.legend()
        else:
            # If feedforward not used or no data, add xlabel to last plot
            plt.subplot(n_plots, 1, 3)
            plt.xlabel("Sample")

        plt.tight_layout()
        plt.show()

    def plot_path_tracking(self, car_positions, path):
        """
        Plot the car's trajectory against the reference path

        Parameters:
        car_positions: List of (x, y) car positions
        path: List of (x, y) points defining the reference path
        """
        plt.figure(figsize=(10, 8))

        # Plot reference path
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_x, path_y, "b--", linewidth=2, label="Reference Path")

        # Plot actual trajectory
        pos_x = [p[0] for p in car_positions]
        pos_y = [p[1] for p in car_positions]
        plt.plot(pos_x, pos_y, "r-", linewidth=2, label="Actual Trajectory")

        # Add markers for start and end
        plt.plot(path_x[0], path_y[0], "go", markersize=10, label="Start")
        plt.plot(path_x[-1], path_y[-1], "ro", markersize=10, label="End")

        # Add arrows to show direction along path
        for i in range(0, len(path), max(1, len(path) // 20)):  # Add ~20 arrows
            plt.arrow(
                path_x[i],
                path_y[i],
                0.1 * (path_x[min(i + 1, len(path) - 1)] - path_x[i]),
                0.1 * (path_y[min(i + 1, len(path) - 1)] - path_y[i]),
                head_width=0.05,
                head_length=0.1,
                fc="b",
                ec="b",
                alpha=0.5,
            )

        plt.title(f"Path Tracking Performance ({self.mode} mode)")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# Example usage:
"""
from robots.AckermannSteeringCar import AckermannSteeringCar

# Create car
car = AckermannSteeringCar(
    x=0,
    y=0,
    theta=0,
    wheelbase=0.25,
    max_velocity=0.5,
    max_steering_angle=np.radians(35)
)

# Create controller
controller = AdaptivePurePursuitController(base_lookahead=0.5)

# Create simulation
sim = AdaptiveControlSimulation(car, controller)

# Generate test path
path = sim.generate_test_path('oval')

# Run simulation
results = sim.run_simulation(path)

# Plot results
sim.plot_results(results, path)

# Show controller performance analysis
controller.plot_performance()

# Run a second simulation with adapted parameters
results2 = sim.run_simulation(path)
sim.plot_results(results2, path)
"""

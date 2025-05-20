import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class AdaptivePurePursuitController:
    """
    Implements an enhanced Pure Pursuit controller with cascaded PID control
    and gain scheduling for path following.

    This controller improves on the basic Pure Pursuit algorithm by:
    1. Using cascaded PIDs for both path following and velocity control
    2. Implementing gain scheduling based on velocity
    3. Adapting parameters based on observed performance
    4. Dynamically adjusting lookahead distance based on speed and curvature
    """

    def __init__(self, base_lookahead=0.5, history_size=100):
        """
        Initialize the controller

        Parameters:
        base_lookahead: Base lookahead distance when stationary (meters)
        history_size: Size of the history buffer for parameter adaptation
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

        # Gain scheduling parameters
        self.gain_schedule = {
            "slow": {"kp": 1.2, "ki": 0.1, "kd": 0.05, "lookahead_factor": 0.8},
            "medium": {"kp": 1.0, "ki": 0.05, "kd": 0.1, "lookahead_factor": 1.0},
            "fast": {"kp": 0.8, "ki": 0.01, "kd": 0.2, "lookahead_factor": 1.3},
        }

        # Error history for integral and derivative terms
        self.steering_error_history = deque(maxlen=history_size)
        self.velocity_error_history = deque(maxlen=history_size)

        # Performance history for parameter adaptation
        self.path_error_history = deque(maxlen=history_size)
        self.control_effort_history = deque(maxlen=history_size)

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

    def set_path(self, path):
        """
        Set the path to follow

        Parameters:
        path: List of (x, y) points defining the path
        """
        self.path = path
        self.current_segment = 0
        self.path_completion = 0.0

        # Calculate path curvature at each point (for lookahead adjustment)
        self.path_curvature = self._calculate_path_curvature()

        # Reset error history when path changes
        self.steering_error_history.clear()
        self.velocity_error_history.clear()
        self.path_error_history.clear()
        self.control_effort_history.clear()

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

            # Curvature is inversely proportional to radius
            segment_length = (mag_v1 + mag_v2) / 2
            if segment_length > 0:
                curvature[i] = angle / segment_length
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
        curvature = self.path_curvature[curvature_idx] if curvature_idx >= 0 else 0
        curvature_adjustment = 1.0 / (1.0 + self.curvature_factor * abs(curvature))

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

        # Update path completion metric
        self.path_completion = closest_idx / max(1, len(self.path) - 1)

        # Calculate lookahead distance based on velocity and curvature
        lookahead = self._get_adjusted_lookahead(velocity, closest_idx)

        # Search forward from the closest point to find a point at lookahead distance
        target_idx = closest_idx
        target_dist = min_dist

        while target_dist < lookahead and target_idx + 1 < len(self.path):
            target_idx += 1
            target_dist = np.sqrt(
                (self.path[target_idx][0] - x) ** 2
                + (self.path[target_idx][1] - y) ** 2
            )

        # If we reached the end of the path, use the last point
        if target_idx == len(self.path) - 1 and target_dist < lookahead:
            return self.path[-1]

        # Calculate cross-track error for diagnostic purposes
        self.cross_track_error = min_dist

        return self.path[target_idx]

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

        return {"kp": kp, "ki": ki, "kd": kd}

    def compute_control(self, x, y, theta, velocity, dt):
        """
        Compute steering and velocity commands using cascaded PID control

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
        target_x, target_y = self.get_target_point(x, y, velocity)

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

        # Combine PID terms for angular velocity command
        angular_velocity = p_term + i_term + d_term

        # Store control effort for adaptation
        self.control_effort_history.append(abs(angular_velocity))
        self.path_error_history.append(abs(heading_error))

        # Velocity control based on path curvature and tracking error
        # Reduce speed in curves and when tracking error is high
        path_idx = int(self.path_completion * (len(self.path) - 1))
        if 0 <= path_idx < len(self.path_curvature):
            curvature = abs(self.path_curvature[path_idx])
            # Speed reduction factor based on curvature and tracking error
            reduction = 1.0 / (1.0 + 3.0 * curvature + 2.0 * abs(heading_error))
            target_velocity = 1.0 * reduction  # Base velocity * reduction factor
        else:
            target_velocity = 1.0  # Default base velocity

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
            return

        # Calculate performance metrics
        mean_path_error = np.mean(self.path_error_history)
        max_path_error = np.max(self.path_error_history)
        mean_control_effort = np.mean(self.control_effort_history)

        print(f"Performance metrics:")
        print(f"  Mean path error: {mean_path_error:.4f} rad")
        print(f"  Max path error: {max_path_error:.4f} rad")
        print(f"  Mean control effort: {mean_control_effort:.4f}")

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

        # Ensure lookahead stays within reasonable bounds
        self.base_lookahead = np.clip(self.base_lookahead, 0.3, 1.0)

        # Return the adapted parameters
        return {
            "base_lookahead": self.base_lookahead,
            "gain_schedules": self.gain_schedule,
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
        plt.figure(figsize=(12, 8))

        # Plot 1: Path errors
        plt.subplot(2, 1, 1)
        plt.plot(list(self.path_error_history), "b-", label="Path Error")
        plt.axhline(
            y=np.mean(self.path_error_history),
            color="r",
            linestyle="--",
            label=f"Mean: {np.mean(self.path_error_history):.4f}",
        )
        plt.title("Path Tracking Error")
        plt.ylabel("Error (rad)")
        plt.grid(True)
        plt.legend()

        # Plot 2: Control effort
        plt.subplot(2, 1, 2)
        plt.plot(list(self.control_effort_history), "g-", label="Control Effort")
        plt.axhline(
            y=np.mean(self.control_effort_history),
            color="r",
            linestyle="--",
            label=f"Mean: {np.mean(self.control_effort_history):.4f}",
        )
        plt.title("Control Effort (Angular Velocity Commands)")
        plt.xlabel("Sample")
        plt.ylabel("Effort")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()


class AdaptiveControlSimulation:
    """
    Simulation for testing the Adaptive Pure Pursuit Controller
    """

    def __init__(self, car, controller):
        self.car = car
        self.controller = controller
        self.dt = 0.1  # Simulation time step (seconds)
        self.max_steps = 500  # Maximum simulation steps

    def generate_test_path(self, path_type="oval"):
        """Generate a test path for the controller"""
        if path_type == "oval":
            t = np.linspace(0, 2 * np.pi, 100)
            x = 5 + 3 * np.cos(t)
            y = 5 + 2 * np.sin(t)
            path = list(zip(x, y))

        elif path_type == "figure8":
            t = np.linspace(0, 2 * np.pi, 100)
            x = 5 + 3 * np.sin(t)
            y = 5 + 2 * np.sin(2 * t)
            path = list(zip(x, y))

        elif path_type == "complex":
            path = []
            # Straight segment
            for i in range(20):
                path.append((i * 0.2, 5.0))

            # Curve
            for i in range(30):
                angle = i * np.pi / 30
                path.append((4.0 + 2.0 * np.cos(angle), 5.0 + 2.0 * np.sin(angle)))

            # Another straight segment
            for i in range(20):
                path.append((4.0 - i * 0.2, 7.0))

        else:
            # Simple line
            path = [(i, 5.0) for i in range(10)]

        return path

    def run_simulation(self, path):
        """
        Run a simulation of the car following the specified path

        Parameters:
        path: List of (x, y) points defining the path to follow

        Returns:
        Dictionary of simulation results
        """
        # Set path in controller
        self.controller.set_path(path)

        # Initialize results storage
        positions = []
        headings = []
        velocities = []
        steering_commands = []
        target_points = []

        # Reset car to start of path
        self.car.x = path[0][0]
        self.car.y = path[0][1]
        self.car.theta = 0.0  # Initial heading (adjust as needed)
        self.car.v = 0.0  # Initial velocity

        # Simulation loop
        for step in range(self.max_steps):
            # Store current state
            positions.append((self.car.x, self.car.y))
            headings.append(self.car.theta)
            velocities.append(self.car.v)

            # Get control commands from controller
            angular_velocity, linear_velocity = self.controller.compute_control(
                self.car.x, self.car.y, self.car.theta, self.car.v, self.dt
            )

            # Store commands
            steering_commands.append(angular_velocity)

            # Store target point
            target = self.controller.get_target_point(
                self.car.x, self.car.y, self.car.v
            )
            target_points.append(target)

            # Set car controls
            steering_angle = np.arctan(
                angular_velocity * self.car.wheelbase / max(0.1, abs(linear_velocity))
            )
            self.car.set_control_inputs(linear_velocity, steering_angle)

            # Update car state
            self.car.update_state(self.dt)

            # Check if we've reached the end of the path
            distance_to_goal = np.hypot(
                self.car.x - path[-1][0], self.car.y - path[-1][1]
            )

            if distance_to_goal < 0.2:
                print(f"Goal reached after {step+1} steps")
                break

        # Adapt controller parameters based on performance
        self.controller.adapt_parameters()

        # Return simulation results
        return {
            "positions": positions,
            "headings": headings,
            "velocities": velocities,
            "steering_commands": steering_commands,
            "target_points": target_points,
            "steps": step + 1,
        }

    def plot_results(self, results, path):
        """
        Plot simulation results

        Parameters:
        results: Dictionary of simulation results from run_simulation
        path: The path that was followed
        """
        plt.figure(figsize=(15, 10))

        # Plot 1: Path following
        plt.subplot(2, 2, 1)
        # Plot the reference path
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_x, path_y, "b--", linewidth=2, label="Reference Path")

        # Plot the actual trajectory
        pos_x = [p[0] for p in results["positions"]]
        pos_y = [p[1] for p in results["positions"]]
        plt.plot(pos_x, pos_y, "r-", linewidth=2, label="Actual Trajectory")

        # Plot target points
        if results["target_points"][0] is not None:
            target_x = [
                p[0]
                for p in results["target_points"]
                if p is not None and p[0] is not None
            ]
            target_y = [
                p[1]
                for p in results["target_points"]
                if p is not None and p[1] is not None
            ]
            plt.scatter(target_x, target_y, c="g", s=20, label="Target Points")

        plt.title("Path Following")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()

        # Plot 2: Velocity profile
        plt.subplot(2, 2, 2)
        plt.plot(results["velocities"], "b-", linewidth=2)
        plt.title("Velocity Profile")
        plt.xlabel("Step")
        plt.ylabel("Velocity (m/s)")
        plt.grid(True)

        # Plot 3: Steering commands
        plt.subplot(2, 2, 3)
        plt.plot(results["steering_commands"], "r-", linewidth=2)
        plt.title("Steering Commands (Angular Velocity)")
        plt.xlabel("Step")
        plt.ylabel("Angular Velocity (rad/s)")
        plt.grid(True)

        # Plot 4: Heading
        plt.subplot(2, 2, 4)
        plt.plot([np.degrees(h) for h in results["headings"]], "g-", linewidth=2)
        plt.title("Heading")
        plt.xlabel("Step")
        plt.ylabel("Heading (degrees)")
        plt.grid(True)

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

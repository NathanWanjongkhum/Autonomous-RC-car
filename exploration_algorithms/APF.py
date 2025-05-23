import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import time
from matplotlib.colors import ListedColormap


class APF:
    """
    Implements the Artifical Potential Field (APF) algorithm on a grid map
    This class uses the track boundaries to find a path to a dynamic goal on the frontier
    """

    MIN_FRONTIER_SIZE = 10

    def __init__(self, car, grid):
        # robot configuration (q)
        self.car = car

        # obstacles (q_o)
        self.grid = grid

        # Data collection for plotting
        self.time_steps = []
        self.heading_error_history = []
        self.angular_velocity_history = []
        self.goal_angle_history = []
        self.desired_heading_history = []
        self.current_theta_history = []
        self.step_counter = 0

        # Setup plotting
        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 12))
        self.setup_plots()

        # goal configuration (q_g)
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.q_min = 0.1
        self.q_max = 2

        # Distance threshold (Q_g)
        self.q_safe = 0.2
        # Parabolic attractive constant (epsilon_q)
        self.epsilon_q = 0.5
        # Conic repulsive constant (epsilon_c)
        self.epsilon_c = 0.5
        # Parabolic-Conic repulsive constant (epsilon_d)
        self.epsilon_d = self.epsilon_q / self.epsilon_c
        # Repulsive constant (epsilon_r)
        self.epsilon_r = 0.5

        # Discrete steering state parameters (from ConstantPurePursuitController)
        self.steering_states = {
            "left": np.radians(15),  # Left steering angle
            "neutral": 0.0,  # Neutral (center) steering
            "right": np.radians(-15),  # Right steering angle
        }

        # Hysteresis thresholds to prevent oscillation
        self.threshold_to_left = np.radians(10)  # Threshold to switch to left
        self.threshold_to_right = np.radians(-10)  # Threshold to switch to right
        self.threshold_to_neutral = np.radians(5)  # Threshold band for neutral

        # Actuation rate limiting
        self.current_state = "neutral"  # Current steering state
        self.last_actuation_time = time.time()  # Time of last actuation
        self.min_actuation_interval = 0.3  # Minimum time between actuations

        # Physical actuator model
        self.time_constant = 0.2  # Time constant for actuator response
        self.current_angle = 0.0  # Current actual steering angle
        self.last_update_time = time.time()  # Time of last physics update

        # Additional tracking for visualization and debugging
        self.desired_angles = []  # Continuous angles before discretization
        self.actual_angles = []  # Actual angles after physical model
        self.steering_states_history = []  # History of steering states

    def setup_plots(self):
        """Initialize the plot structure"""
        # First subplot: Heading angles
        self.axs[0].set_title("Heading Angles")
        self.axs[0].set_ylabel("Angle (radians)")
        self.axs[0].grid(True)

        # Second subplot: Heading error
        self.axs[1].set_title("Heading Error")
        self.axs[1].set_ylabel("Error (radians)")
        self.axs[1].grid(True)

        # Third subplot: Steering Angle
        self.axs[2].set_title("Steering Angle")
        self.axs[2].set_xlabel("Time Step")
        self.axs[2].set_ylabel("Steering Angle (rad)")
        self.axs[2].grid(True)

        # Initialize empty lines
        (self.goal_line,) = self.axs[0].plot([], [], "r-", label="Goal Angle")
        (self.desired_line,) = self.axs[0].plot([], [], "g-", label="Desired Heading")
        (self.current_line,) = self.axs[0].plot([], [], "b-", label="Current Heading")
        (self.error_line,) = self.axs[1].plot([], [], "k-", label="Heading Error")
        (self.actual_angle_line,) = self.axs[2].plot(
            [], [], "m-", label="Actual Steering Angle"
        )

        self.axs[0].legend()
        self.axs[2].legend()

        plt.tight_layout()
        plt.ion()  # Interactive mode on

    def update_plot(self):
        """Update the plot with the latest data"""
        # Update lines
        self.goal_line.set_data(self.time_steps, self.goal_angle_history)
        self.desired_line.set_data(self.time_steps, self.desired_heading_history)
        self.current_line.set_data(self.time_steps, self.current_theta_history)
        self.error_line.set_data(self.time_steps, self.heading_error_history)
        self.actual_angle_line.set_data(self.time_steps, self.actual_angles)

        # Draw the updated figure
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def save_plots(self, filename="apf_performance.png"):
        """Save the current plots to a file"""
        plt.figure(self.fig.number)
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Plots saved to {filename}")

    def attractive_force(self, target=None):
        """
        U_attr(q,q_g)
        Calculate attractive force toward the target (nearest frontier point)
        Attractive potential function: Depeonds on robot and goal configurations.
        """
        # If no target is provided, use the default goal
        if target is None:
            target_x, target_y = self.goal_x, self.goal_y
        else:
            target_x, target_y = target[0], target[1]

        # Calculate the distance between the car and the goal
        # d_g = sqrt((x - x_g)^2 + (y - y_g)^2)
        dx = target_x - self.car.x
        dy = target_y - self.car.y
        distance = np.hypot(dx, dy)

        # Initialize force vector
        force = np.zeros(2)

        # U_attr(q, q_q) = {parabolic(), conic()}
        if distance < self.q_safe:
            # Quadratic potential force
            # x = epsilon_q * d_g * (x - x_g)
            # y = epsilon_q * d_g * (y - y_g)

            force[0] = -self.epsilon_q * distance * dx
            force[1] = -self.epsilon_q * distance * dy

        elif distance >= self.q_safe:
            # Conic potential force
            # x = (epsilon_q * epsilon_q * (x-x_g)) / d_g
            # y = (epsilon_q * epsilon_q * (y-y_g)) / d_g
            force[0] = -(self.q_safe * self.epsilon_q * dx) / distance
            force[1] = -(self.q_safe * self.epsilon_q * dy) / distance
        return force

    def get_obstacle_positions(self):
        """Extract obstacle positions from the occupancy grid"""
        obstacles = np.where(self.grid.grid == 1)
        return list(zip(obstacles[1], obstacles[0]))

    def repulsive_force(self):
        """
        Repulsive potential function: Depeonds on the obstacle and robot configuration. Avoids the robot to get clsoe to the obstacle.
        """
        total_force = np.zeros(2)

        # Get obstacle positions from occupancy grid
        obstacle_positions = self.get_obstacle_positions()

        for obstacle_pos in obstacle_positions:
            # Calculate the distance between the car and the obstacle
            dx = self.car.x - obstacle_pos[0]
            dy = self.car.y - obstacle_pos[1]

            distance = np.hypot(dx, dy)

            # Apply repulsive force if within range of influence
            if (
                distance < self.q_max and distance > 0
            ):  # Ensure distance is not zero to avoid division by zero
                # Calculate the repulsive force based on the gradient of the potential field
                # F_rep = epsilon_r * (1/distance - 1/Q_max) * (1/distance^2) * unit_vector_away_from_obstacle
                # unit_vector_away_from_obstacle = (dx/distance, dy/distance)
                force_magnitude = (
                    self.epsilon_r * (1 / distance - 1 / self.q_max) * (1 / distance**2)
                )
                total_force[0] += force_magnitude * (dx / distance)
                total_force[1] += force_magnitude * (dy / distance)

        return total_force

    def detect_frontiers(self):
        """Detect frontiers between explored and unexplored areas using probability thresholds"""
        # Define probability thresholds
        free_threshold = 0.4  # Cells with p < 0.4 are considered free
        occ_threshold = 0.6  # Cells with p > 0.6 are considered occupied
        # Values between are considered unknown/uncertain

        # Create masks based on probability thresholds
        free_space = (self.grid.grid < free_threshold).astype(np.uint8)
        obstacle = (self.grid.grid > occ_threshold).astype(np.uint8)
        unknown = (~(free_space | obstacle)).astype(np.uint8)

        # Combined grid for visualization
        combined_grid = np.zeros_like(self.grid.grid)
        combined_grid[free_space == 1] = 1  # Free space
        combined_grid[unknown == 1] = 2  # Unknown
        combined_grid[obstacle == 1] = 3  # Obstacle

        # Frontier detection - need kernel for dilation
        kernel = np.ones((3, 3), np.uint8)
        kernel[1, 1] = 0  # Exclude center pixel

        # Dilate free space
        free_dilated = cv2.dilate(free_space, kernel, iterations=1)

        # Frontier cells: where dilated free space meets unknown
        frontier_cells = free_dilated & unknown

        # Labeled frontiers with centroids
        num_labels, labels = cv2.connectedComponents(frontier_cells)

        # Find centroids
        frontiers = []
        for label in range(1, num_labels):
            points = np.where(labels == label)
            if len(points[0]) > APF.MIN_FRONTIER_SIZE:
                centroid = (np.mean(points[1]), np.mean(points[0]))
                frontiers.append(centroid)

        return frontiers

    def detect_frontiers_debug(self):
        """Detect frontiers between explored and unexplored areas using probability thresholds"""
        # Setup figure with subplots
        fig, axs = plt.subplots(3, 3, figsize=(12, 10))

        # Define probability thresholds
        free_threshold = 0.4  # Cells with p < 0.4 are considered free
        occ_threshold = 0.6  # Cells with p > 0.6 are considered occupied
        # Values between are considered unknown/uncertain

        # Create masks based on probability thresholds
        free_space = (self.grid.grid < free_threshold).astype(np.uint8)
        obstacle = (self.grid.grid > occ_threshold).astype(np.uint8)
        unknown = (~(free_space | obstacle)).astype(
            np.uint8
        )  # Everything else is unknown

        # First row: Display binary masks
        axs[0, 0].imshow(free_space, cmap="binary")
        axs[0, 0].set_title("Free Space (p < 0.4)")
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])

        axs[0, 1].imshow(unknown, cmap="binary")
        axs[0, 1].set_title("Unknown (0.4 <= p <= 0.6)")
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])

        axs[0, 2].imshow(obstacle, cmap="binary")
        axs[0, 2].set_title("Obstacle (p > 0.6)")
        axs[0, 2].set_xticks([])
        axs[0, 2].set_yticks([])

        # Second row, left: Original probability grid as heatmap
        probability_img = axs[1, 0].imshow(
            self.grid.grid, cmap="plasma", vmin=0, vmax=1
        )
        axs[1, 0].set_title("Probability Grid")
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        fig.colorbar(
            probability_img,
            ax=axs[1, 0],
            orientation="vertical",
            fraction=0.046,
            pad=0.04,
        )

        # Combined grid for visualization
        combined_grid = np.zeros_like(self.grid.grid)
        combined_grid[free_space == 1] = 1  # Free space
        combined_grid[unknown == 1] = 2  # Unknown
        combined_grid[obstacle == 1] = 3  # Obstacle

        cmap = ListedColormap(["black", "white", "gray", "red"])
        axs[1, 1].imshow(combined_grid, cmap=cmap)
        axs[1, 1].set_title("Thresholded Grid")
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])

        # Frontier detection - need kernel for dilation
        kernel = np.ones((3, 3), np.uint8)
        kernel[1, 1] = 0  # Exclude center pixel

        # Dilate free space
        free_dilated = cv2.dilate(free_space, kernel, iterations=1)
        axs[1, 2].imshow(free_dilated, cmap="binary")
        axs[1, 2].set_title("Dilated Free Space")
        axs[1, 2].set_xticks([])
        axs[1, 2].set_yticks([])

        # Frontier cells: where dilated free space meets unknown
        frontier_cells = free_dilated & unknown
        axs[2, 0].imshow(frontier_cells, cmap="binary")
        axs[2, 0].set_title("Frontier Cells")
        axs[2, 0].set_xticks([])
        axs[2, 0].set_yticks([])

        # Labeled frontiers with centroids
        num_labels, labels = cv2.connectedComponents(frontier_cells)

        # Random colormap for the labels
        label_cmap = plt.colormaps["coolwarm"].resampled(10)

        # Show labeled regions
        axs[2, 1].imshow(labels, cmap=label_cmap)
        axs[2, 1].set_title(f"Labeled Frontiers ({num_labels-1} regions)")
        axs[2, 1].set_xticks([])
        axs[2, 1].set_yticks([])

        # Find and plot centroids
        frontiers = []
        for label in range(1, num_labels):
            points = np.where(labels == label)
            if len(points[0]) > APF.MIN_FRONTIER_SIZE:
                centroid = (np.mean(points[1]), np.mean(points[0]))
                frontiers.append(centroid)
                axs[2, 1].plot(centroid[0], centroid[1], "wo", markersize=8)
                axs[2, 1].plot(centroid[0], centroid[1], "ko", markersize=6)

        # Overlay frontiers on original grid
        axs[2, 2].imshow(self.grid.grid, cmap="plasma", vmin=0, vmax=1)
        axs[2, 2].set_title("Frontiers on Probability Grid")
        axs[2, 2].set_xticks([])
        axs[2, 2].set_yticks([])

        # Mark frontiers on the probability grid
        for centroid in frontiers:
            axs[2, 2].plot(centroid[0], centroid[1], "wo", markersize=8)
            axs[2, 2].plot(centroid[0], centroid[1], "ko", markersize=6)

        plt.tight_layout()
        plt.show()

        return frontiers

    def determine_target_state(self, desired_angle):
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

    def update_steering_state(self, target_state, current_time):
        """
        Update steering state respecting actuation rate limits

        Parameters:
        target_state: The desired steering state
        current_time: Current system time

        Returns:
        bool: True if state changed, False otherwise
        """
        # Check if enough time has passed since the last actuation
        time_since_last = current_time - self.last_actuation_time

        if (
            target_state != self.current_state
            and time_since_last >= self.min_actuation_interval
        ):
            self.current_state = target_state
            self.last_actuation_time = current_time
            return True

        return False

    def update_physical_model(self, target_angle, current_time):
        """
        Update physical model of the actuator

        Parameters:
        target_angle: The target steering angle
        current_time: Current system time

        Returns:
        float: The current actual steering angle
        """
        # Calculate time delta since last update
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # First-order actuator model: dθ/dt = (θ_target - θ)/τ
        # Discretized: θ_new = θ + (θ_target - θ) * (dt/τ)
        if dt > 0:  # Avoid division by zero
            self.current_angle += (target_angle - self.current_angle) * (
                dt / self.time_constant
            )

        return self.current_angle

    def calculate_forces(self):
        """
        The negative gradients locally points toward the configuration with a lower potential.
        The robot feels a force opposite to the gradient.
        """

        # Get attractive force toward nearest frontier
        frontiers = self.detect_frontiers()

        # gradient U_attr(q,q_g)
        if frontiers:
            # Calculate distances to all frontiers
            distances = [
                np.hypot(self.car.x - f[0], self.car.y - f[1]) for f in frontiers
            ]
            # Find index of minimum distance
            min_idx = np.argmin(distances)
            # Get the corresponding frontier
            nearest_frontier = frontiers[min_idx]

            f_attr = self.attractive_force(nearest_frontier)
        else:
            # If no frontiers found, use default goal
            f_attr = self.attractive_force()

        f_rep = self.repulsive_force()  # sum(gradient U_rep(q,q_o^i))
        return f_attr, f_rep

    def compute_steering(self, dt):
        # Get car's current state
        theta = self.car.theta

        # Find frontiers and select the best one
        frontiers = self.detect_frontiers()
        if frontiers:
            # Find closest frontier
            distances = [
                np.hypot(self.car.x - f[0], self.car.y - f[1]) for f in frontiers
            ]
            nearest_frontier = frontiers[np.argmin(distances)]

            # Set this as the goal
            self.goal_x, self.goal_y = nearest_frontier

            # Calculate angle to the goal (in world frame)
            goal_angle = np.arctan2(self.goal_y - self.car.y, self.goal_x - self.car.x)

            # Calculate attractive and repulsive forces
            f_attr, f_rep = self.calculate_forces()

            # Combine forces to get resultant direction (both forces are in world frame)
            F = f_attr + f_rep

            # Desired heading from force vector
            desired_heading = np.arctan2(F[1], F[0])

            # Calculate heading error
            heading_error = desired_heading - theta
            # Normalize to [-pi, pi]
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

            # Calculate desired steering angle (proportional control for now, can be expanded to PID)
            # A simple proportional gain to convert heading error to a desired steering angle
            k_p_steering = 1.0  # This gain will need tuning
            desired_angle = k_p_steering * heading_error

            # Limit desired angle to a reasonable range (e.g., max steering angle of the car)
            max_steering_angle = np.radians(35)  # Example max steering angle
            desired_angle = np.clip(
                desired_angle, -max_steering_angle, max_steering_angle
            )

            # Store the desired angle for visualization/debugging
            self.desired_angles.append(desired_angle)

            # ---- Discrete steering logic ----
            # Get current time for rate limiting
            current_time = time.time()

            # Determine target discrete state
            target_state = self.determine_target_state(desired_angle)

            # Update steering state (respecting rate limits)
            self.update_steering_state(target_state, current_time)

            # Get target angle for the current state
            target_angle = self.steering_states[self.current_state]

            # Update physical model to get actual steering angle
            actual_steering_angle = self.update_physical_model(
                target_angle, current_time
            )

            # Store actual angle and state for tracking
            self.actual_angles.append(actual_steering_angle)
            self.steering_states_history.append(self.current_state)

            # Record data for plotting
            self.time_steps.append(self.step_counter)
            self.heading_error_history.append(heading_error)
            self.goal_angle_history.append(goal_angle)
            self.desired_heading_history.append(desired_heading)
            self.current_theta_history.append(theta)
            self.step_counter += 1

            # Update the plot every N steps to avoid slowing down simulation
            if self.step_counter % 5 == 0:
                self.update_plot()

            # For exploration, assume a constant forward velocity
            forward_velocity = 0.2  # m/s

            return forward_velocity, actual_steering_angle
        else:
            # No frontiers found, maintain current direction and stop
            return 0.0, 0.0


## Example of using the APF class
# car = Car(x=0, y=0, theta=0)  # Define car with required attributes
# grid = build_occupancy_grid()  # Create grid from stereovision
# apf = APF(car, grid)

## Main control loop
# while not exploration_complete():
#     left_speed, right_speed = apf.compute_steering()
#     car.set_wheel_speeds(left_speed, right_speed)
#     update_car_position()
#     update_occupancy_grid()

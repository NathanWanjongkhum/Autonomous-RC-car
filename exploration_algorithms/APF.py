import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
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

        # goal configuration (q_g)
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.q_min = 0
        self.q_max = 2

        # Distance threshold (Q_g)
        self.q_safe = 0.1
        # Parabolic attractive constant (epsilon_q)
        self.epsilon_q = 0.1
        # Conic repulsive constant (epsilon_c)
        self.epsilon_c = 0.1
        # Parabolic-Conic repulsive constant (epsilon_d)
        self.epsilon_d = self.epsilon_q / self.epsilon_c
        # Repulsive constant (epsilon_r)
        self.epsilon_r = 0.1

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
            if self.q_min < distance < self.q_max:
                # x = e_r * l_x,i / d^3
                total_force[0] += (self.epsilon_r * dx) / (distance**3)
                # y = e_r * l_y,i / d^3
                total_force[1] += (self.epsilon_r * dy) / (distance**3)
            
        return total_force

    def detect_frontiers(self):
        """Detect frontiers between explored and unexplored areas using probability thresholds"""
        # Define probability thresholds
        free_threshold = 0.4      # Cells with p < 0.4 are considered free
        occ_threshold = 0.6       # Cells with p > 0.6 are considered occupied
        # Values between are considered unknown/uncertain
        
        # Create masks based on probability thresholds
        free_space = (self.grid.grid < free_threshold).astype(np.uint8)
        obstacle = (self.grid.grid > occ_threshold).astype(np.uint8)
        unknown = (~(free_space | obstacle)).astype(np.uint8)  # Everything else is unknown
        
        # Combined grid for visualization
        combined_grid = np.zeros_like(self.grid.grid)
        combined_grid[free_space == 1] = 1  # Free space
        combined_grid[unknown == 1] = 2     # Unknown
        combined_grid[obstacle == 1] = 3    # Obstacle

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
        free_threshold = 0.4      # Cells with p < 0.4 are considered free
        occ_threshold = 0.6       # Cells with p > 0.6 are considered occupied
        # Values between are considered unknown/uncertain
        
        # Create masks based on probability thresholds
        free_space = (self.grid.grid < free_threshold).astype(np.uint8)
        obstacle = (self.grid.grid > occ_threshold).astype(np.uint8)
        unknown = (~(free_space | obstacle)).astype(np.uint8)  # Everything else is unknown
        
        # First row: Display binary masks
        axs[0, 0].imshow(free_space, cmap='binary')
        axs[0, 0].set_title('Free Space (p < 0.4)')
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])

        axs[0, 1].imshow(unknown, cmap='binary')
        axs[0, 1].set_title('Unknown (0.4 <= p <= 0.6)')
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])

        axs[0, 2].imshow(obstacle, cmap='binary')
        axs[0, 2].set_title('Obstacle (p > 0.6)')
        axs[0, 2].set_xticks([])
        axs[0, 2].set_yticks([])

        # Second row, left: Original probability grid as heatmap
        probability_img = axs[1, 0].imshow(self.grid.grid, cmap='plasma', vmin=0, vmax=1)
        axs[1, 0].set_title('Probability Grid')
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        fig.colorbar(probability_img, ax=axs[1, 0], orientation='vertical', fraction=0.046, pad=0.04)
        
        # Combined grid for visualization
        combined_grid = np.zeros_like(self.grid.grid)
        combined_grid[free_space == 1] = 1  # Free space
        combined_grid[unknown == 1] = 2     # Unknown
        combined_grid[obstacle == 1] = 3    # Obstacle
        
        cmap = ListedColormap(['black', 'white', 'gray', 'red'])
        axs[1, 1].imshow(combined_grid, cmap=cmap)
        axs[1, 1].set_title('Thresholded Grid')
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])
        
        # Frontier detection - need kernel for dilation
        kernel = np.ones((3, 3), np.uint8)
        kernel[1, 1] = 0  # Exclude center pixel
        
        # Dilate free space
        free_dilated = cv2.dilate(free_space, kernel, iterations=1)
        axs[1, 2].imshow(free_dilated, cmap='binary')
        axs[1, 2].set_title('Dilated Free Space')
        axs[1, 2].set_xticks([])
        axs[1, 2].set_yticks([])
        
        # Frontier cells: where dilated free space meets unknown
        frontier_cells = free_dilated & unknown
        axs[2, 0].imshow(frontier_cells, cmap='binary')
        axs[2, 0].set_title('Frontier Cells')
        axs[2, 0].set_xticks([])
        axs[2, 0].set_yticks([])
        
        # Labeled frontiers with centroids
        num_labels, labels = cv2.connectedComponents(frontier_cells)
        
        # Random colormap for the labels
        label_cmap = plt.colormaps['coolwarm'].resampled(10)
        
        # Show labeled regions
        axs[2, 1].imshow(labels, cmap=label_cmap)
        axs[2, 1].set_title(f'Labeled Frontiers ({num_labels-1} regions)')
        axs[2, 1].set_xticks([])
        axs[2, 1].set_yticks([])
        
        # Find and plot centroids
        frontiers = []
        for label in range(1, num_labels):
            points = np.where(labels == label)
            if len(points[0]) > APF.MIN_FRONTIER_SIZE:
                centroid = (np.mean(points[1]), np.mean(points[0]))
                frontiers.append(centroid)
                axs[2, 1].plot(centroid[0], centroid[1], 'wo', markersize=8)
                axs[2, 1].plot(centroid[0], centroid[1], 'ko', markersize=6)
        
        # Overlay frontiers on original grid
        axs[2, 2].imshow(self.grid.grid, cmap='plasma', vmin=0, vmax=1)
        axs[2, 2].set_title('Frontiers on Probability Grid')
        axs[2, 2].set_xticks([])
        axs[2, 2].set_yticks([])
        
        # Mark frontiers on the probability grid
        for centroid in frontiers:
            axs[2, 2].plot(centroid[0], centroid[1], 'wo', markersize=8)
            axs[2, 2].plot(centroid[0], centroid[1], 'ko', markersize=6)
        
        plt.tight_layout()
        plt.show()
        
        return frontiers

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

    def compute_steering(self):
        # world angle
        theta = self.car.theta
        # Wheel base
        b = self.car.wheel_distance
        # Vehicle speed
        velocity = self.car.v
        # Radius of the wheels
        r = self.car.wheel_radius
        # Set e to 0 for standard differential drive
        e = 0
        # Create rotation matrix from world to robot frame
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        
        # Calculate forces
        f_attr, f_rep = self.calculate_forces()
        F = f_attr + rotation_matrix @ f_rep
        
        # Compute wheel velocities
        eta = velocity * F
        
        # Solve for Jacobian matrix
        if e == 0:
            jacobian_matrix = (
                np.array(
                    [
                        [math.cos(theta), math.sin(theta)],
                        [math.sin(theta), math.cos(theta)],
                    ]
                )
                / r
            )
        else:
            factor = b / e
            jacobian_matrix = (
                np.array([
                    [
                        math.cos(theta) + factor * math.sin(theta),
                        math.sin(theta) - factor * math.cos(theta),
                    ],
                    [
                        math.cos(theta) - factor * math.sin(theta),
                        math.sin(theta) + factor * math.cos(theta),
                    ],
                ]) / r
            )
            
        # Solve for omegas
        u = jacobian_matrix @ eta
        
        # Convert to Python scalers
        v_l = float(u[0])
        v_r = float(u[1])

        return (v_l, v_r)

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
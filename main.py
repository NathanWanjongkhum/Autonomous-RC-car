import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
import math

class DifferentialDriveCar:
    """
    Models a differential drive car with rectangular dimensions

    This class simulates the kinematics of a car with two powered wheels
    in the back and two smaller wheels for stability, as described in the
    requirements.
    """

    def __init__(
        self,
        x=0,
        y=0,
        theta=0,
        wheel_distance=0.2,
        wheel_radius=0.05,
        max_velocity=0.5,
        max_angular_velocity=1.0,
    ):
        # State variables
        self.x = x  # Position x (meters)
        self.y = y  # Position y (meters)
        self.theta = theta  # Orientation (radians)

        # Physical parameters
        self.wheel_distance = wheel_distance  # Distance between wheels (meters)
        self.wheel_radius = wheel_radius  # Wheel radius (meters)
        self.max_velocity = max_velocity  # Maximum linear velocity (m/s)
        self.max_angular_velocity = max_angular_velocity  # Maximum angular velocity (rad/s)

        # Control variables
        self.v_l = 0  # Left wheel velocity (m/s)
        self.v_r = 0  # Right wheel velocity (m/s)
        self.v = 0  # Linear velocity (m/s)
        self.omega = 0  # Angular velocity (rad/s)

        # Car dimensions (rectangular shape)
        self.length = 0.3  # Car length (meters)
        self.width = 0.2  # Car width (meters)

    def update_state(self, dt):
        """
        Update car state based on wheel velocities over time dt

        Implements the differential drive kinematics equations
        to update position and orientation

        Parameters:
        dt: Time step in seconds
        """
        # Calculate linear and angular velocity from wheel velocities
        self.v = (self.v_r + self.v_l) / 2
        self.omega = (self.v_r - self.v_l) / self.wheel_distance

        # Update state using basic differential drive kinematics
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.omega * dt

        # Normalize theta to (-pi, pi) to prevent growing continuously
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

    def set_wheel_velocities(self, v_l, v_r):
        """
        Set the wheel velocities directly, respecting maximum limits

        Parameters:
        v_l: Left wheel velocity (m/s)
        v_r: Right wheel velocity (m/s)
        """
        self.v_l = np.clip(v_l, -self.max_velocity, self.max_velocity)
        self.v_r = np.clip(v_r, -self.max_velocity, self.max_velocity)

    def set_velocities(self, v, omega):
        """
        Set the linear and angular velocities, and calculate corresponding wheel velocities

        Handles the conversion from desired robot velocities to individual wheel velocities
        while respecting maximum limits

        Parameters:
        v: Desired linear velocity (m/s)
        omega: Desired angular velocity (rad/s)
        """
        # Apply limits to requested velocities
        v = np.clip(v, -self.max_velocity, self.max_velocity)
        omega = np.clip(omega, -self.max_angular_velocity, self.max_angular_velocity)

        # Calculate wheel velocities from robot velocities
        # For differential drive: v_l = v - (omega * L/2), v_r = v + (omega * L/2)
        # where L is the distance between wheels
        self.v_l = v - (omega * self.wheel_distance / 2)
        self.v_r = v + (omega * self.wheel_distance / 2)

        # Ensure wheel velocities don't exceed limits
        # If they would, scale both proportionally
        scale = max(1, max(abs(self.v_l), abs(self.v_r)) / self.max_velocity)
        self.v_l /= scale
        self.v_r /= scale

        # Update the overall velocity and angular velocity
        self.v = v
        self.omega = omega

    def get_corners(self):
        """
        Get the four corners of the car for collision detection and visualization

        Returns a list of (x,y) coordinates for the four corners of the car,
        taking into account its position, orientation, length, and width
        """
        # Calculate the four corners of the car based on position, orientation, length, and width
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        # Calculate the corners relative to the car's position and orientation
        # Using rotation matrix to transform from car frame to world frame
        front_right = (
            self.x + self.length / 2 * cos_theta - self.width / 2 * sin_theta,
            self.y + self.length / 2 * sin_theta + self.width / 2 * cos_theta,
        )

        front_left = (
            self.x + self.length / 2 * cos_theta + self.width / 2 * sin_theta,
            self.y + self.length / 2 * sin_theta - self.width / 2 * cos_theta,
        )

        rear_left = (
            self.x - self.length / 2 * cos_theta + self.width / 2 * sin_theta,
            self.y - self.length / 2 * sin_theta - self.width / 2 * cos_theta,
        )

        rear_right = (
            self.x - self.length / 2 * cos_theta - self.width / 2 * sin_theta,
            self.y - self.length / 2 * sin_theta + self.width / 2 * cos_theta,
        )

        return [front_right, front_left, rear_left, rear_right]


class OccupancyGrid:
    """
    Represents the environment as a grid of occupancy probabilities

    This class implements the occupancy grid mapping algorithm described in the
    implementation plan, which is ideal for stereovision-based probability maps.
    """

    def __init__(self, width, height, resolution):
        """
        Initialize an occupancy grid map

        Parameters:
        width: Width of the environment in meters
        height: Height of the environment in meters
        resolution: Size of each grid cell in meters (smaller = more detailed)
        """
        self.width = width
        self.height = height
        self.resolution = resolution

        # Calculate grid dimensions in cells
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)

        # Initialize the grid with unknown probability (0.5)
        # 0 = free, 1 = occupied, 0.5 = unknown
        self.grid = np.ones((self.grid_height, self.grid_width)) * 0.5

        # Binary grid (for path planning) - 0 = free, 1 = occupied
        # self.binary_grid = np.ones((self.grid_height, self.grid_width), dtype=bool) # Initialize as occupied
        self.binary_grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)  # Initialize as free
        
        # Log-odds representation for Bayesian updates
        # Log-odds representation makes probabilistic updates more efficient
        # log-odds = log(p/(1-p))
        self.log_odds_grid = np.zeros((self.grid_height, self.grid_width))

        # Occupancy threshold for binary grid
        self.occupancy_threshold = 0.6

        # Free space threshold
        self.free_threshold = 0.4

        # Constants for log-odds update
        # These control how quickly cells are updated when observed as occupied/free
        self.lo_occ = np.log(0.65 / 0.35)  # log-odds for occupied measurement
        self.lo_free = np.log(0.35 / 0.65)  # log-odds for free measurement

    def world_to_grid(self, x, y):
        """
        Convert world coordinates (meters) to grid cell indices

        Parameters:
        x, y: World coordinates

        Returns:
        grid_x, grid_y: Grid cell indices
        """
        grid_x = math.ceil(x / self.resolution)
        grid_y = math.ceil(y / self.resolution)

        # Ensure within grid bounds
        grid_x = np.clip(grid_x, 0, self.grid_width - 1)
        grid_y = np.clip(grid_y, 0, self.grid_height - 1)

        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        """
        Convert grid cell indices to world coordinates (cell center)

        Parameters:
        grid_x, grid_y: Grid cell indices

        Returns:
        x, y: World coordinates of cell center
        """
        x = (grid_x + 0.5) * self.resolution
        y = (grid_y + 0.5) * self.resolution
        return x, y

    def is_occupied(self, x, y):
        """
        Check if a world position is occupied

        Parameters:
        x, y: World coordinates

        Returns:
        bool: True if occupied, False otherwise
        """
        grid_x, grid_y = self.world_to_grid(x, y)
        return self.binary_grid[grid_y, grid_x]

    def update_cell(self, x, y, occupied):
        """
        Update a single cell with an observation using log-odds

        Parameters:
        x, y: World coordinates
        occupied: Boolean indicating if cell is observed as occupied
        """
        grid_x, grid_y = self.world_to_grid(x, y)

        # Update log-odds
        # This is the Bayesian update in log-odds form
        if occupied:
            self.log_odds_grid[grid_y, grid_x] += self.lo_occ
        else:
            self.log_odds_grid[grid_y, grid_x] += self.lo_free

        # Convert back to probability
        # p = 1 - 1/(1 + exp(log_odds))
        p = 1 - (1 / (1 + np.exp(self.log_odds_grid[grid_y, grid_x])))
        self.grid[grid_y, grid_x] = p

        # Update binary grid
        self.binary_grid[grid_y, grid_x] = p > self.occupancy_threshold

    def update_from_lidar(
        self, sensor_x, sensor_y, sensor_theta, ranges, angles, max_range
    ):
        """
        Update the grid from lidar-like scan

        This method updates the occupancy grid using a range-sensor model,
        marking cells along rays as free until the detected obstacle or max range

        Parameters:
        sensor_x, sensor_y, sensor_theta: Pose of the sensor
        ranges: List of measured ranges
        angles: List of angles for the ranges (relative to sensor_theta)
        max_range: Maximum measurement range
        """
        for i, (r, angle) in enumerate(zip(ranges, angles)):
            # Calculate absolute angle in world frame
            abs_angle = sensor_theta + angle

            # Trace the ray
            if r >= max_range:
                # No obstacle detected, mark cells along ray as free
                for d in np.arange(0, max_range, self.resolution):
                    ray_x = sensor_x + d * np.cos(abs_angle)
                    ray_y = sensor_y + d * np.sin(abs_angle)
                    self.update_cell(ray_x, ray_y, False)
            else:
                # Obstacle detected at range r
                # Mark cells before r as free
                for d in np.arange(0, r, self.resolution):
                    ray_x = sensor_x + d * np.cos(abs_angle)
                    ray_y = sensor_y + d * np.sin(abs_angle)
                    self.update_cell(ray_x, ray_y, False)

                # Mark cell at r as occupied
                ray_x = sensor_x + r * np.cos(abs_angle)
                ray_y = sensor_y + r * np.sin(abs_angle)
                self.update_cell(ray_x, ray_y, True)

    def update_from_stereovision(self, sensor_x, sensor_y, sensor_theta, vision_data):
        """
        Update the grid from stereovision data

        This method integrates probability data from a stereovision system
        into the occupancy grid

        Parameters:
        sensor_x, sensor_y, sensor_theta: Pose of the sensor
        vision_data: 2D array of probability values (0-1) from stereovision
        """
        # Get dimensions of vision data
        v_height, v_width = vision_data.shape

        # Calculate the physical dimensions of the vision field
        field_width = v_width * self.resolution
        field_height = v_height * self.resolution

        # Calculate bottom-left corner of vision field in world frame
        corner_x = (
            sensor_x
            + self.resolution * np.cos(sensor_theta)
            - field_width / 2 * np.sin(sensor_theta)
        )
        corner_y = (
            sensor_y
            + self.resolution * np.sin(sensor_theta)
            + field_width / 2 * np.cos(sensor_theta)
        )

        # Update each cell in the vision field
        for i in range(v_height):
            for j in range(v_width):
                # Transform from vision frame to world frame
                world_x = (
                    corner_x
                    + j * self.resolution * np.sin(sensor_theta)
                    + i * self.resolution * np.cos(sensor_theta)
                )
                world_y = (
                    corner_y
                    + j * self.resolution * np.cos(sensor_theta)
                    - i * self.resolution * np.sin(sensor_theta)
                )

                # Update cell with probability
                grid_x, grid_y = self.world_to_grid(world_x, world_y)

                # Check if within grid bounds
                if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                    # Use inverse logit update based on vision probability
                    prob = vision_data[i, j]

                    # Only update if the probability is informative (not around 0.5)
                    if abs(prob - 0.5) > 0.1:
                        # Convert probability to log-odds update
                        update = np.log(prob / (1 - prob))
                        self.log_odds_grid[grid_y, grid_x] += update

                        # Update probability grid
                        self.grid[grid_y, grid_x] = 1 - (
                            1 / (1 + np.exp(self.log_odds_grid[grid_y, grid_x]))
                        )

                        # Update binary grid
                        self.binary_grid[grid_y, grid_x] = (
                            self.grid[grid_y, grid_x] > self.occupancy_threshold
                        )

    def process_map(self):
        """
        Process the occupancy grid to clean up noise

        This method applies filtering to the probability grid and updates the binary grid.
        It's useful to run after mapping to get a cleaner map for path planning.
        """
        # Apply Gaussian filter to smooth the probability grid
        smoothed_grid = gaussian_filter(self.grid, sigma=1.0)

        # Threshold to get binary grid
        self.binary_grid = smoothed_grid > self.occupancy_threshold

        # Update log-odds and probability grids for consistency
        self.log_odds_grid = np.log(smoothed_grid / (1 - smoothed_grid + 1e-10))
        self.grid = smoothed_grid


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
            angle = i * (2 * np.pi / self.num_sensors)
            angles.append(angle)

            # Cast ray until hitting an obstacle or max distance
            for d in np.arange(0.1, self.max_sensing_distance, self.grid.resolution):
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
            if self.side == "right" and abs(angle - np.pi / 2) < np.pi / 6:
                if ranges[i] < min_side_dist:
                    min_side_dist = ranges[i]
                    side_idx = i
            elif self.side == "left" and abs(angle - np.pi / 2) < np.pi / 6:
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
        if front_idx is not None and min_front_dist < 0.5:
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


class AStarPlanner:
    """
    Implements the A* path planning algorithm on a grid map

    This class finds the shortest path from a start position to a goal
    position in an occupancy grid.
    """

    def __init__(self, occupancy_grid):
        """
        Initialize the A* planner

        Parameters:
        occupancy_grid: OccupancyGrid instance
        """
        self.grid = occupancy_grid

    def heuristic(self, a, b):
        """
        Calculate the heuristic distance between two grid cells

        Parameters:
        a, b: Grid cell indices (x, y)

        Returns:
        float: Euclidean distance
        """
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def get_neighbors(self, node):
        """
        Get valid neighboring cells of a grid cell

        Parameters:
        node: Grid cell indices (x, y)

        Returns:
        list: List of valid neighboring cells
        """
        x, y = node
        neighbors = []

        # Define 8-connected grid (diagonal and orthogonal neighbors)
        directions = [
            # Orthogonal
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),  
            # Diagonal
            (1, 1),
            (1, -1),
            (-1, -1),
            (-1, 1),  
        ]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check if within grid bounds
            if 0 <= nx < self.grid.grid_width and 0 <= ny < self.grid.grid_height:
                # Check if cell is free (not occupied)
                if not self.grid.binary_grid[ny, nx]:
                    neighbors.append((nx, ny))

        return neighbors

    def plan_path(self, start_x, start_y, goal_x, goal_y):
        """
        Find the shortest path from start to goal

        Parameters:
        start_x, start_y: Start position in world coordinates
        goal_x, goal_y: Goal position in world coordinates

        Returns:
        path: List of (x, y) points defining the path in world coordinates
            or None if no path found
        """
        # Convert to grid coordinates
        start_grid = self.grid.world_to_grid(start_x, start_y)
        goal_grid = self.grid.world_to_grid(goal_x, goal_y)

        # Check if start or goal is in obstacle
        if (
            self.grid.binary_grid[start_grid[1], start_grid[0]]
            or self.grid.binary_grid[goal_grid[1], goal_grid[0]]
        ):
            print("Start or goal is in obstacle!")
            return None

        # Initialize the open and closed sets
        open_set = {start_grid}
        closed_set = set()

        # Dictionary to store the best path to each node
        came_from = {}

        # Dictionary to store the cost to reach each node from the start
        g_score = {start_grid: 0}

        # Dictionary to store the estimated total cost from start to goal through each node
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        while open_set:
            # Find the node in open_set with the lowest f_score
            current = min(open_set, key=lambda node: f_score.get(node, float("inf")))

            # If goal reached, reconstruct and return the path
            if current == goal_grid:
                path = []
                while current in came_from:
                    # Convert back to world coordinates
                    world_x, world_y = self.grid.grid_to_world(current[0], current[1])
                    path.append((world_x, world_y))
                    current = came_from[current]

                # Add start position
                world_x, world_y = self.grid.grid_to_world(start_grid[0], start_grid[1])
                path.append((world_x, world_y))

                # Reverse to get path from start to goal
                path.reverse()
                return path

            # Move current from open to closed set
            open_set.remove(current)
            closed_set.add(current)

            # Check each neighbor
            for neighbor in self.get_neighbors(current):
                # Skip if already evaluated
                if neighbor in closed_set:
                    continue

                # Calculate tentative g_score
                tentative_g_score = g_score[current] + self.heuristic(current, neighbor)

                # If neighbor not in open set, add it
                if neighbor not in open_set:
                    open_set.add(neighbor)
                # If this path to neighbor is worse, skip
                elif tentative_g_score >= g_score.get(neighbor, float("inf")):
                    continue

                # This path is the best so far, record it
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.heuristic(
                    neighbor, goal_grid
                )

        # No path found
        print("No path found!")
        return None


class Simulation:
    """
    Simulation environment for the car and track

    This class simulates the car's movement in the environment and
    handles visualization and sensor simulation.
    """

    def __init__(self, width=10.0, height=10.0, resolution=0.05):
        """
        Initialize the simulation

        Parameters:
        width, height: Size of the environment in meters
        resolution: Grid resolution in meters
        """
        # Create the occupancy grid
        self.grid = OccupancyGrid(width, height, resolution)

        # Create the car
        self.car = DifferentialDriveCar(x=2.0, y=2.0, theta=0.0)

        # Create path planners and controllers
        self.wall_follower = WallFollower(self.grid)
        self.path_planner = AStarPlanner(self.grid)
        self.pure_pursuit = PurePursuit()

        # Simulation parameters
        self.dt = 0.1  # Time step in seconds
        self.max_steps = 1000  # Maximum simulation steps
        self.current_step = 0

        # Path and mapping
        self.path = []
        self.visited_positions = []

        # Visualization
        self.fig = None
        self.ax = None
        self.car_patch = None

    def create_track(self, track_type="simple"):
        """
        Create a track in the environment

        Parameters:
        track_type: Type of track to create ('simple', 'oval', 'complex')
        """
        # Clear the binary grid before creating track
        self.grid.binary_grid = np.zeros((self.grid.grid_height, self.grid.grid_width), dtype=bool)
        # Make walls thicker for better detection
        wall_thickness = 1  # Cells
        wall_offset = self.grid.resolution / wall_thickness
    
        if track_type == "simple":
            # Set parameters for inner track
            inner_margin_percent = 0.3  # 30% margin from the outer walls
            inner_width = self.grid.width * (1 - 2 * inner_margin_percent)
            inner_height = self.grid.height * (1 - 2 * inner_margin_percent)
            inner_start_x = self.grid.width * inner_margin_percent
            inner_start_y = self.grid.height * inner_margin_percent
            
            # Outer walls - use your existing code
            # Bottom wall
            for x in np.arange(0.0, self.grid.width, self.grid.resolution):
                for y in np.arange(-wall_offset, wall_offset, self.grid.resolution):
                    self.grid.update_cell(x, y, True)
                    
            # Top wall
            for x in np.arange(0.0, self.grid.width, self.grid.resolution):
                for y in np.arange(self.grid.height-wall_offset, self.grid.height+wall_offset, self.grid.resolution):
                    self.grid.update_cell(x, y, True)
                    
            # Left wall
            for y in np.arange(0.0, self.grid.height, self.grid.resolution):
                for x in np.arange(-wall_offset, wall_offset, self.grid.resolution):
                    self.grid.update_cell(x, y, True)
                    
            # Right wall
            for y in np.arange(0.0, self.grid.height, self.grid.resolution):
                for x in np.arange(self.grid.width-wall_offset, self.grid.width+wall_offset, self.grid.resolution):
                    self.grid.update_cell(x, y, True)
            
            # Inner walls
            # Bottom inner wall
            for x in np.arange(inner_start_x, inner_start_x + inner_width, self.grid.resolution):
                for y in np.arange(inner_start_y - wall_offset, inner_start_y + wall_offset, self.grid.resolution):
                    self.grid.update_cell(x, y, True)
                    
            # Top inner wall
            for x in np.arange(inner_start_x, inner_start_x + inner_width, self.grid.resolution):
                for y in np.arange(inner_start_y + inner_height - wall_offset, 
                                inner_start_y + inner_height + wall_offset, self.grid.resolution):
                    self.grid.update_cell(x, y, True)
                    
            # Left inner wall
            for y in np.arange(inner_start_y, inner_start_y + inner_height, self.grid.resolution):
                for x in np.arange(inner_start_x - wall_offset, inner_start_x + wall_offset, self.grid.resolution):
                    self.grid.update_cell(x, y, True)
                    
            # Right inner wall
            for y in np.arange(inner_start_y, inner_start_y + inner_height, self.grid.resolution):
                for x in np.arange(inner_start_x + inner_width - wall_offset, 
                                inner_start_x + inner_width + wall_offset, self.grid.resolution):
                    self.grid.update_cell(x, y, True)
            
        elif track_type == "oval":
            # Oval track
            center_x, center_y = 5.0, 5.0
            a, b = 4.0, 3.0  # Semi-major and semi-minor axes

            # Outer wall
            for theta in np.linspace(0, 2 * np.pi, 100):
                x = center_x + a * np.cos(theta)
                y = center_y + b * np.sin(theta)
                self.grid.update_cell(x, y, True)

            # Inner wall
            a, b = 2.0, 1.5  # Smaller semi-major and semi-minor axes
            for theta in np.linspace(0, 2 * np.pi, 100):
                x = center_x + a * np.cos(theta)
                y = center_y + b * np.sin(theta)
                self.grid.update_cell(x, y, True)

        elif track_type == "complex":
            # More complex track with turns
            # Outer walls
            for x in np.arange(1.0, 9.0, self.grid.resolution):
                self.grid.update_cell(x, 1.0, True)
                self.grid.update_cell(x, 9.0, True)

            for y in np.arange(1.0, 9.0, self.grid.resolution):
                self.grid.update_cell(1.0, y, True)
                self.grid.update_cell(9.0, y, True)

            # Inner obstacles
            for x in np.arange(3.0, 5.0, self.grid.resolution):
                for y in np.arange(3.0, 5.0, self.grid.resolution):
                    self.grid.update_cell(x, y, True)

            for x in np.arange(6.0, 8.0, self.grid.resolution):
                for y in np.arange(6.0, 8.0, self.grid.resolution):
                    self.grid.update_cell(x, y, True)

    def sense_environment(self):
        """
        Simulate sensing the environment with car sensors

        This method simulates stereo vision and updates the occupancy grid accordingly.
        """
        # Simulate lidar-like sensing
        # Generate angles and ranges
        num_rays = 20
        # angles = 180
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        max_range = 2.0

        # Cast rays and detect obstacles
        ranges = []
        for angle in angles:
            # Cast ray until hitting an obstacle
            for d in np.arange(0.1, max_range, self.grid.resolution):
                ray_x = self.car.x + d * np.cos(self.car.theta + angle)
                ray_y = self.car.y + d * np.sin(self.car.theta + angle)

                # Check if ray is outside the environment
                if (
                    ray_x < 0
                    or ray_x >= self.grid.width
                    or ray_y < 0
                    or ray_y >= self.grid.height
                ):
                    ranges.append(d)
                    break

                # Check ground truth (simplified for simulation)
                grid_x, grid_y = self.grid.world_to_grid(ray_x, ray_y)
                if (
                    grid_x < 0
                    or grid_x >= self.grid.grid_width
                    or grid_y < 0
                    or grid_y >= self.grid.grid_height
                ):
                    continue

                if self.grid.binary_grid[grid_y, grid_x]:
                    ranges.append(d)
                    break
            else:
                # No obstacle found at max range
                ranges.append(max_range)

        # Update grid with sensed data
        self.grid.update_from_lidar(
            self.car.x, self.car.y, self.car.theta, ranges, angles, max_range
        )

        # Record visited position
        self.visited_positions.append((self.car.x, self.car.y))

    def phase1_exploration(self, use_wall_following=True):
        """
        Perform Phase 1 exploration of the environment

        This method controls the car to explore the environment and build
        a map using the wall following algorithm.

        Parameters:
        use_wall_following: Whether to use wall following or just predefined path
        """
        # Use wall following for exploration
        if use_wall_following:
            # Get steering command from wall follower
            angular_velocity = self.wall_follower.compute_steering(
                self.car.x, self.car.y, self.car.theta, 0.2
            )

            # Set car velocities (constant forward speed, varying steering)
            self.car.set_velocities(0.2, angular_velocity)
        else:
            # For testing, just move forward
            self.car.set_velocities(0.2, 0.0)

        # Update car state
        self.car.update_state(self.dt)

        # Sense environment and update map
        self.sense_environment()

        # Increment step counter
        self.current_step += 1

    def plan_optimal_path(self, start_x, start_y, goal_x, goal_y):
        """
        Plan an optimal path from start to goal

        This method would be used after Phase 1 exploration to plan
        the optimal path for Phase 2.

        Parameters:
        start_x, start_y: Start position
        goal_x, goal_y: Goal position

        Returns:
        path: List of (x, y) points defining the path or None if no path found
        """
        # Process the map to clean up noise
        self.grid.process_map()

        # Plan path using A*
        path = self.path_planner.plan_path(start_x, start_y, goal_x, goal_y)

        if path:
            # Set the path for pure pursuit
            self.pure_pursuit.set_path(path)
            self.path = path
            return path

        return None

    def phase2_execution(self):
        """
        Perform Phase 2 execution of the planned path

        This method controls the car to follow the planned path
        using the pure pursuit controller.
        """
        # Check if we have a path
        if not self.path:
            print("No path to follow!")
            return

        # Get steering command from pure pursuit
        angular_velocity = self.pure_pursuit.compute_steering(
            self.car.x, self.car.y, self.car.theta, 0.5
        )

        # Set car velocities (faster than exploration)
        self.car.set_velocities(0.5, angular_velocity)

        # Update car state
        self.car.update_state(self.dt)

        # Increment step counter
        self.current_step += 1

    def initialize_visualization(self):
        """
        Initialize the matplotlib figure for visualization
        """
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(0, self.grid.width)
        self.ax.set_ylim(0, self.grid.height)
        self.ax.set_aspect("equal")
        self.ax.set_title("Car Navigation Simulation")

        # Create car patch
        corners = self.car.get_corners()
        xy = np.array(corners + [corners[0]])  # Close the polygon
        self.car_patch = patches.Polygon(xy, closed=True, color="blue", fill=True)
        self.ax.add_patch(self.car_patch)

        # Create occupancy grid visualization
        grid_img = np.zeros((self.grid.grid_height, self.grid.grid_width, 4))
        grid_img[self.grid.binary_grid, 3] = 1.0  # Set alpha for occupied cells
        grid_img[self.grid.binary_grid, 0] = 1.0  # Set red for occupied cells

        self.ax.imshow(
            grid_img,
            extent=(0, self.grid.width, 0, self.grid.height),
            origin="lower",
            interpolation="nearest",
        )

        # Plot initial position
        self.ax.plot(self.car.x, self.car.y, "go", markersize=8)

        return self.fig, self.ax

    def update_visualization(self):
        """
        Update the visualization with current state
        """
        if self.fig is None:
            self.initialize_visualization()

        # Update car patch
        corners = self.car.get_corners()
        xy = np.array(corners + [corners[0]])  # Close the polygon
        self.car_patch.set_xy(xy)

        # Update occupancy grid visualization
        grid_img = np.zeros((self.grid.grid_height, self.grid.grid_width, 4))

        # Set probability visualization (grayscale for unknown areas)
        grid_img[:, :, 0] = 0.8 * (1 - self.grid.grid)  # Red channel
        grid_img[:, :, 1] = 0.8 * (1 - self.grid.grid)  # Green channel
        grid_img[:, :, 2] = 0.8 * (1 - self.grid.grid)  # Blue channel
        grid_img[:, :, 3] = 0.5  # Semi-transparent for probabilities

        # Overlay binary grid (solid red for occupied, solid green for known free)
        grid_img[self.grid.binary_grid, 0] = 1.0  # Red for occupied
        grid_img[self.grid.binary_grid, 1:3] = 0.0  # No green/blue for occupied
        grid_img[self.grid.binary_grid, 3] = 1.0  # Fully opaque for occupied

        self.ax.images[0].set_data(grid_img)

        # Plot path
        if self.path:
            path_x = [p[0] for p in self.path]
            path_y = [p[1] for p in self.path]
            if not hasattr(self, "path_line"):
                (self.path_line,) = self.ax.plot(path_x, path_y, "g-", linewidth=2)
            else:
                self.path_line.set_data(path_x, path_y)

        # Plot visited positions
        visited_x = [p[0] for p in self.visited_positions]
        visited_y = [p[1] for p in self.visited_positions]
        if not hasattr(self, "visited_line"):
            (self.visited_line,) = self.ax.plot(
                visited_x, visited_y, "y.", markersize=1
            )
        else:
            self.visited_line.set_data(visited_x, visited_y)

        self.fig.canvas.draw()
        plt.pause(0.01)

    def run_simulation(self, mode="exploration", num_steps=None):
        """
        Run the simulation

        Parameters:
        mode: 'exploration' for Phase 1, 'execution' for Phase 2
        num_steps: Number of steps to run, or None for max_steps
        """
        if num_steps is None:
            num_steps = self.max_steps

        # Initialize visualization
        self.initialize_visualization()

        # Reset step counter
        self.current_step = 0

        print(f"Running simulation for {num_steps} steps...")
        # Run simulation loop
        for step in range(num_steps):
            print(f"Step {step}")
            if mode == "exploration":
                self.phase1_exploration()
            elif mode == "execution":
                self.phase2_execution()

            # Update visualization every few steps
            if step % 5 == 0:
                self.update_visualization()

            # Check termination conditions
            if self.current_step >= self.max_steps:
                break

        print("Simulation complete!")

        # Final update
        self.update_visualization()
        print("Displaying final state. Close the plot window to continue.")
        plt.show()


def main():
    # Create simulation
    sim = Simulation(width=10.0, height=10.0, resolution=0.05)

    # Create track
    sim.create_track(track_type="simple")

    # Phase 1: Exploration
    print("Running Phase 1: Exploration...")
    sim.run_simulation(mode="exploration", num_steps=500)

    # Process map and plan optimal path
    start_x, start_y = sim.car.x, sim.car.y
    goal_x, goal_y = 1.5, 1.5

    print("Planning optimal path...")
    path = sim.plan_optimal_path(start_x, start_y, goal_x, goal_y)

    if path:
        print("Optimal path found!")

        # Phase 2: Execution
        print("Running Phase 2: Execution...")
        # Reset car position for demonstration
        sim.car.x, sim.car.y, sim.car.theta = start_x, start_y, 0.0
        sim.run_simulation(mode="execution", num_steps=200)
    else:
        print("Failed to find a path!")


if __name__ == "__main__":
    main()

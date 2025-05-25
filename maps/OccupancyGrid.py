import numpy as np
import math
from scipy.ndimage import gaussian_filter


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
        self.binary_grid = np.zeros(
            (self.grid_height, self.grid_width), dtype=bool
        )  # Initialize as free

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

    def _discretize_state(self, x, y):
        """
        Convert world coordinates (meters) to grid cell indices

        Parameters:
        x, y: World coordinates

        Returns:
        grid_x, grid_y: Grid cell indices
        """
        grid_x = math.floor(x / self.resolution)
        grid_y = math.floor(y / self.resolution)

        # Ensure within grid bounds
        grid_x = np.clip(grid_x, 0, self.grid_width - 1)
        grid_y = np.clip(grid_y, 0, self.grid_height - 1)

        return grid_x, grid_y

    def _continuous_state(self, grid_x, grid_y):
        """
        Convert grid cell indices to world coordinates

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
        grid_x, grid_y = self._discretize_state(x, y)
        return self.binary_grid[grid_y, grid_x]

    def update_cell(self, x, y, occupied):
        """
        Update a single cell with an observation using log-odds

        Parameters:
        x, y: World coordinates
        occupied: Boolean indicating if cell is observed as occupied
        """
        grid_x, grid_y = self._discretize_state(x, y)

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
                grid_x, grid_y = self._discretize_state(world_x, world_y)

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

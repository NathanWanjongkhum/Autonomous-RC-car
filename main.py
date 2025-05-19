import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

from maps.OccupancyGrid import OccupancyGrid
from robots.DifferentialDriveCar import DifferentialDriveCar

from exploration_algorithms.APF import APF

from pathfinding_algorithms.PurePursuit import PurePursuit
from pathfinding_algorithms.WallFollower import WallFollower
from pathfinding_algorithms.AStar import AStarPlanner


class Simulation:
    """
    Simulation environment for the car and track

    This class simulates the car's movement in the environment and
    handles visualization and sensor simulation.
    """

    def __init__(self, car: DifferentialDriveCar, grid: OccupancyGrid):
        """
        Initialize the simulation

        Parameters:
        width, height: Size of the environment in meters
        resolution: Grid resolution in meters
        """
        self.car = car
        self.grid = grid
        self.explorer_algorithm = explorer_algorithm
        self.pure_pursuit = pure_pursuit

        # Simulation parameters
        self.dt = 0.1  # Time step in seconds
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
        self.grid.binary_grid = np.zeros(
            (self.grid.grid_height, self.grid.grid_width), dtype=bool
        )
        # Make walls thicker for better detection
        wall_thickness = 1  # Cells
        wall_offset = self.grid.resolution / wall_thickness

        if track_type == "simple":
            # Set parameters for inner track
            inner_margin_percent = 0.1  # 30% margin from the outer walls
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
                for y in np.arange(
                    self.grid.height - wall_offset,
                    self.grid.height + wall_offset,
                    self.grid.resolution,
                ):
                    self.grid.update_cell(x, y, True)

            # Left wall
            for y in np.arange(0.0, self.grid.height, self.grid.resolution):
                for x in np.arange(-wall_offset, wall_offset, self.grid.resolution):
                    self.grid.update_cell(x, y, True)

            # Right wall
            for y in np.arange(0.0, self.grid.height, self.grid.resolution):
                for x in np.arange(
                    self.grid.width - wall_offset,
                    self.grid.width + wall_offset,
                    self.grid.resolution,
                ):
                    self.grid.update_cell(x, y, True)

            # Inner walls
            # Bottom inner wall
            for x in np.arange(
                inner_start_x, inner_start_x + inner_width, self.grid.resolution
            ):
                for y in np.arange(
                    inner_start_y - wall_offset,
                    inner_start_y + wall_offset,
                    self.grid.resolution,
                ):
                    self.grid.update_cell(x, y, True)

            # Top inner wall
            for x in np.arange(
                inner_start_x, inner_start_x + inner_width, self.grid.resolution
            ):
                for y in np.arange(
                    inner_start_y + inner_height - wall_offset,
                    inner_start_y + inner_height + wall_offset,
                    self.grid.resolution,
                ):
                    self.grid.update_cell(x, y, True)

            # Left inner wall
            for y in np.arange(
                inner_start_y, inner_start_y + inner_height, self.grid.resolution
            ):
                for x in np.arange(
                    inner_start_x - wall_offset,
                    inner_start_x + wall_offset,
                    self.grid.resolution,
                ):
                    self.grid.update_cell(x, y, True)

            # Right inner wall
            for y in np.arange(
                inner_start_y, inner_start_y + inner_height, self.grid.resolution
            ):
                for x in np.arange(
                    inner_start_x + inner_width - wall_offset,
                    inner_start_x + inner_width + wall_offset,
                    self.grid.resolution,
                ):
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
                    or ray_x >= self.grid.grid_width
                    or ray_y < 0
                    or ray_y >= self.grid.grid_height
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

    def run_exploration(self, max_steps=250):
        """
        Run the exploration phase of the simulation

        Parameters:
        max_steps: Maximum number of steps to run (default: 250)
        """
        if self.explorer_algorithm is None:
            print("No exploration algorithm specified. Skipping...")
            return

        # Initialize visualization
        self.initialize_visualization()

        print(f"Running simulation for {max_steps} steps...")
        # Run simulation loop
        for step in range(max_steps):
            # Compute steering command
            av = self.explorer_algorithm.compute_steering()

            self.car.set_velocities(0.2, av)

            # Update car state
            self.car.update_state(self.dt)

            # Sense environment and update map
            self.sense_environment()

            # Update visualization every few steps
            if step % 5 == 0:
                self.update_visualization()

        print("Simulation complete!")
        self.explorer_algorithm.save_plots()

        # Final update
        self.update_visualization()
        print("Displaying final state. Close the plot window to continue.")
        plt.show()


def main():
    car = DifferentialDriveCar(x=2.0, y=2.0, theta=0.0)
    grid = OccupancyGrid(width=10.0, height=10.0, resolution=0.05)
    apf = APF(car, grid)
    pure_pursuit = PurePursuit()
    # pid = PIDController(kp=0.5, ki=0.0, kd=0.0)
    # pathfinder_algorithm = KKT(grid, pid)

    # sim = Simulation(car=car, grid=grid, explorer_algorithm=None, pure_pursuit=pure_pursuit)
    sim = Simulation(
        car=car, grid=grid, explorer_algorithm=apf, pure_pursuit=pure_pursuit
    )

    # Create track
    sim.create_track(track_type="simple")

    #### Debugging ####
    # sim.initialize_visualization()
    # sim.update_visualization()
    # plt.show()

    # Phase 1: Exploration
    print("Running Phase 1: Exploration...")
    sim.run_exploration(max_steps=250)

    # # Process map and plan optimal path
    # start_x, start_y = sim.car.x, sim.car.y
    # goal_x, goal_y = 1.5, 1.5

    # print("Planning optimal path...")
    # path = sim.plan_optimal_path(start_x, start_y, goal_x, goal_y)

    # if path:
    #     print("Optimal path found!")

    #     # Phase 2: Execution
    #     print("Running Phase 2: Execution...")
    #     # Reset car position for demonstration
    #     sim.car.x, sim.car.y, sim.car.theta = start_x, start_y, 0.0
    #     sim.run_simulation(mode="execution", num_steps=200)
    # else:
    #     print("Failed to find a path!")


if __name__ == "__main__":
    main()

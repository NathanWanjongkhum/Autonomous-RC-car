import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PID import AdaptiveControlSimulation, AdaptivePurePursuitController
from maps.OccupancyGrid import OccupancyGrid

from robots.AckermannSteeringCar import AckermannSteeringCar

from exploration_algorithms.APF import APF


class Simulation:
    """
    Simulation environment for the car and track

    This class simulates the car's movement in the environment and
    handles visualization and sensor simulation.
    """

    def __init__(self, car=None, grid=None, width=10.0, height=10.0, resolution=0.05):
        """
        Initialize the simulation

        Parameters:
        car: An instance of AckermannSteeringCar (will create a default one if None)
        grid: An instance of OccupancyGrid (will create a default one if None)
        width, height: Size of the environment in meters (used if grid is None)
        resolution: Grid resolution in meters (used if grid is None)
        """
        # Create or use the provided occupancy grid
        if grid is None:
            self.grid = OccupancyGrid(width, height, resolution)
        else:
            self.grid = grid

        # Create or use the provided car
        if car is None:
            self.car = AckermannSteeringCar(x=2.0, y=2.0, theta=0.0)
        else:
            self.car = car

        # Create the APF controller for exploration
        self.explorer = APF(self.car, self.grid)

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
        self.wheel_patches = []
        self.steering_lines = []

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
            inner_margin_percent = 0.1  # 10% margin from the outer walls
            inner_width = self.grid.width * (1 - 2 * inner_margin_percent)
            inner_height = self.grid.height * (1 - 2 * inner_margin_percent)
            inner_start_x = self.grid.width * inner_margin_percent
            inner_start_y = self.grid.height * inner_margin_percent

            # Outer walls
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
        angles = np.linspace(-np.pi / 2, np.pi / 2, num_rays)  # Front-facing 180° arc
        max_range = 2.0

        # Cast rays and detect obstacles
        ranges = []
        for angle in angles:
            # Cast ray until hitting an obstacle
            for d in np.arange(0.1, max_range, self.grid.resolution):
                # Calculate ray endpoint in world coordinates
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

    def phase1_exploration(self):
        """
        Perform Phase 1 exploration of the environment using APF controller
        """
        # Get control inputs from APF controller
        velocity, steering_angle = self.explorer.compute_steering(self.dt)

        # Set car control inputs
        self.car.set_control_inputs(velocity, steering_angle)

        # Update car state
        self.car.update_state(self.dt)

        # Sense environment and update map
        self.sense_environment()

        # Increment step counter
        self.current_step += 1

        # Return the car's action for logging or analysis
        return velocity, steering_angle

    def initialize_visualization(self):
        """
        Initialize the matplotlib figure for visualization
        """
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(0, self.grid.width)
        self.ax.set_ylim(0, self.grid.height)
        self.ax.set_aspect("equal")
        self.ax.set_title("Car Navigation Simulation (Ackermann Steering)")

        # Create car patch
        corners = self.car.get_corners()
        xy = np.array(corners + [corners[0]])  # Close the polygon
        self.car_patch = patches.Polygon(xy, closed=True, color="blue", fill=False)
        self.ax.add_patch(self.car_patch)

        # Create wheel visualizations
        self.wheel_patches = []

        # Get wheel positions and steering angles
        steering_coords = self.car.visualize_steering()
        self.steering_lines = []

        # Front wheels (with steering)
        for x1, y1, x2, y2 in steering_coords:
            # Draw a line representing the wheel direction
            (line,) = self.ax.plot([x1, x2], [y1, y2], "r-", linewidth=2)
            self.steering_lines.append(line)

            # Draw the wheel itself as a small rectangle
            wheel_length = self.car.wheel_radius * 2
            wheel_width = wheel_length / 2

            # Calculate wheel angle
            wheel_angle = np.arctan2(y2 - y1, x2 - x1)

            # Create wheel patch
            wheel_patch = patches.Rectangle(
                (
                    x1 - self.car.wheel_width / 2,
                    y1 - self.car.wheel_width / 2,
                ),  # xy position (centered on wheel)
                self.car.wheel_radius * 2,  # width
                self.car.wheel_width,  # height
                angle=np.degrees(wheel_angle),  # angle in degrees
                color="black",
                fill=True,
            )
            self.ax.add_patch(wheel_patch)
            self.wheel_patches.append(wheel_patch)

        # Rear wheels (fixed orientation)
        cos_theta = np.cos(self.car.theta)
        sin_theta = np.sin(self.car.theta)

        # Rear axle center
        rear_center_x = self.car.x - (self.car.length / 2) * cos_theta
        rear_center_y = self.car.y - (self.car.length / 2) * sin_theta

        # Calculate rear wheel positions (left and right)
        rear_wheel_positions = [
            # Left rear wheel - include the wheel offset
            (
                rear_center_x
                + (self.car.width / 2 + self.car.wheel_offset) * -sin_theta,
                rear_center_y
                + (self.car.width / 2 + self.car.wheel_offset) * cos_theta,
            ),
            # Right rear wheel - include the wheel offset
            (
                rear_center_x
                + (self.car.width / 2 + self.car.wheel_offset) * sin_theta,
                rear_center_y
                + (self.car.width / 2 + self.car.wheel_offset) * -cos_theta,
            ),
        ]

        # Draw rear wheels (parallel to car orientation)
        wheel_length = self.car.wheel_radius * 2
        wheel_width = wheel_length / 2

        for x, y in rear_wheel_positions:
            wheel_patch = patches.Rectangle(
                (
                    x - self.car.wheel_width / 2,
                    y - self.car.wheel_width / 2,
                ),  # xy position
                self.car.wheel_radius * 2,  # width
                self.car.wheel_width,  # height
                angle=np.degrees(self.car.theta),  # angle in degrees
                color="black",
                fill=True,
            )
            self.ax.add_patch(wheel_patch)
            self.wheel_patches.append(wheel_patch)

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

        # Remove old wheel patches
        for patch in self.wheel_patches:
            patch.remove()
        self.wheel_patches = []

        # Update steering lines
        steering_coords = self.car.visualize_steering()
        for i, (x1, y1, x2, y2) in enumerate(steering_coords):
            self.steering_lines[i].set_data([x1, x2], [y1, y2])

        # Re-create front wheel patches with updated positions and angles
        for x1, y1, x2, y2 in steering_coords:
            # Calculate wheel angle
            wheel_angle = np.arctan2(y2 - y1, x2 - x1)

            # Create wheel patch
            wheel_length = self.car.wheel_radius * 2
            wheel_width = wheel_length / 2

            wheel_patch = patches.Rectangle(
                (x1 - wheel_width / 2, y1 - wheel_width / 2),
                wheel_length,
                wheel_width,
                angle=np.degrees(wheel_angle),
                color="black",
                fill=True,
            )
            self.ax.add_patch(wheel_patch)
            self.wheel_patches.append(wheel_patch)

        # Update rear wheel positions and patches
        cos_theta = np.cos(self.car.theta)
        sin_theta = np.sin(self.car.theta)

        # Rear axle center
        rear_center_x = self.car.x - (self.car.length / 2) * cos_theta
        rear_center_y = self.car.y - (self.car.length / 2) * sin_theta

        # Calculate rear wheel positions
        rear_wheel_positions = [
            # Left rear wheel - include the wheel offset
            (
                rear_center_x
                + (self.car.width / 2 + self.car.wheel_offset) * -sin_theta,
                rear_center_y
                + (self.car.width / 2 + self.car.wheel_offset) * cos_theta,
            ),
            # Right rear wheel - include the wheel offset
            (
                rear_center_x
                + (self.car.width / 2 + self.car.wheel_offset) * sin_theta,
                rear_center_y
                + (self.car.width / 2 + self.car.wheel_offset) * -cos_theta,
            ),
        ]

        # Draw rear wheels
        wheel_length = self.car.wheel_radius * 2
        wheel_width = wheel_length / 2

        for x, y in rear_wheel_positions:
            wheel_patch = patches.Rectangle(
                (x - wheel_width / 2, y - wheel_width / 2),
                wheel_length,
                wheel_width,
                angle=np.degrees(self.car.theta),
                color="black",
                fill=True,
            )
            self.ax.add_patch(wheel_patch)
            self.wheel_patches.append(wheel_patch)

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

    def run_simulation(self, mode="exploration", num_steps=None, pause_interval=None):
        """
        Run the simulation

        Parameters:
        mode: 'exploration' for Phase 1
        num_steps: Number of steps to run, or None for max_steps
        pause_interval: If specified, pause after this many steps and wait for user input
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
                velocity, steering_angle = self.phase1_exploration()
                print(
                    f"  Velocity: {velocity:.2f}, Steering: {np.degrees(steering_angle):.2f}°"
                )

            # Update visualization every few steps or at pause intervals
            if step % 5 == 0 or (pause_interval and step % pause_interval == 0):
                self.update_visualization()

                # If we have a pause interval and we're at a pause step, wait for user input
                if pause_interval and step % pause_interval == 0:
                    plt.draw()
                    input(f"Step {step} - Press Enter to continue...")
                else:
                    plt.pause(0.01)  # Small pause for visualization

            # Check termination conditions
            if self.current_step >= self.max_steps:
                break

        print("Simulation complete!")

        # Final update
        self.update_visualization()
        print("Displaying final state. Close the plot window to continue.")
        plt.show(block=True)  # Wait until user closes the window


def main():
    """
    Main function to run the simulation
    """
    # Create a car with specified parameters
    car = AckermannSteeringCar(
        x=2.0,  # Starting x position
        y=2.0,  # Starting y position
        theta=0.0,  # Starting orientation (radians)
        wheelbase=0.25,  # Distance between front and rear axles
        wheel_radius=0.05,  # Radius of wheels
        wheel_width=0.04,  # Width of wheels
        wheel_offset=0.05,  # How far wheels extend beyond car body
        max_velocity=0.5,  # Maximum velocity (m/s)
        max_steering_angle=np.radians(35),  # Maximum steering angle (radians)
        max_angular_velocity=1.0,  # Maximum angular velocity (rad/s)
        length=0.3,  # Car length (meters)
        width=0.2,  # Car width (meters)
    )

    # Create an occupancy grid
    grid = OccupancyGrid(width=4.0, height=4.0, resolution=0.05)

    # Create the simulation with our car and grid
    sim = Simulation(car=car, grid=grid)

    # Create a simple track
    sim.create_track(track_type="simple")

    # For debugging - initialize and show the visualization
    sim.initialize_visualization()
    sim.update_visualization()
    input("Press Enter to continue to the next step...")

    # Test car movement
    # Move the car forward and turn the wheels
    car.set_control_inputs(v=0.0, steering_angle=np.radians(20))
    sim.update_visualization()
    input("Press Enter to continue to the next step...")

    # Move the car forward with steering
    car.update_state(dt=0.5)  # Update for 0.5 seconds
    sim.update_visualization()
    input("Press Enter to continue to the next step...")

    # Uncomment to run the actual simulation with pauses
    # print("Starting simulation...")
    # sim.run_simulation(mode="exploration", num_steps=200, pause_interval=20)  # Pause every 20 steps


def debug_main():
    # Create a car with specified parameters
    car = AckermannSteeringCar(
        x=2.0,  # Starting x position
        y=2.0,  # Starting y position
        theta=0.0,  # Starting orientation (radians)
        wheelbase=0.25,  # Distance between front and rear axles
        wheel_radius=0.05,  # Radius of wheels
        wheel_width=0.04,  # Width of wheels
        wheel_offset=0.05,  # How far wheels extend beyond car body
        max_velocity=0.5,  # Maximum velocity (m/s)
        max_steering_angle=np.radians(35),  # Maximum steering angle (radians)
        max_angular_velocity=1.0,  # Maximum angular velocity (rad/s)
        length=0.3,  # Car length (meters)
        width=0.2,  # Car width (meters)
    )

    # Create controller
    controller = AdaptivePurePursuitController(base_lookahead=0.4, mode="exploration")

    # Create simulation
    sim = AdaptiveControlSimulation(car, controller)

    # Generate test path
    path = sim.generate_test_path("zigzag")

    for _ in range(1):
        # Run simulation
        results = sim.run_simulation(path)

    # Plot results
    sim.plot_results(results, path)

    # Show controller performance analysis
    controller.plot_performance()


if __name__ == "__main__":
    # main()
    debug_main()

from dataclasses import dataclass
from maps.OccupancyGrid import OccupancyGrid
import matplotlib.pyplot as plt
from path_optimizing_algorithms.LatticeMotionPlanner import (
    phase2_discrete_planning,
)
from DrawableElements import VisualizationManager, MarkerDrawer
import numpy as np


@dataclass
class Pose:
    """A simple pose class to represent a position and orientation in 2D space."""

    x: float
    y: float
    theta: float


def generate_grid(
    width: int, height: int, grid_type: str = "empty"
) -> tuple[OccupancyGrid, Pose, Pose]:
    # Generate an empty grid with resolution that balances accuracy and performance
    grid = OccupancyGrid(
        width=width,
        height=height,
        resolution=0.15,  # Balance between accuracy and computational efficiency
    )

    match grid_type:
        case "empty":
            # All cells are free (initialized to 0 in OccupancyGrid)
            pass
        case "corridor":
            # Outer walls
            for i in range(width):
                grid.binary_grid[i, 0] = 1  # Bottom wall
                grid.binary_grid[i, height - 1] = 1  # Top wall
            for i in range(height):
                grid.binary_grid[0, i] = 1  # Left wall
                grid.binary_grid[width - 1, i] = 1  # Right wall

            # Clear a horizontal path in the middle
            corridor_start_y = int(height / 2) - 1
            corridor_end_y = int(height / 2) + 1
            for x in range(1, width - 1):
                for y in range(corridor_start_y, corridor_end_y + 1):
                    grid.binary_grid[x, y] = 0  # Clear inner area
        case "obstacles":
            # Outer walls
            for i in range(width):
                grid.binary_grid[i, 0] = 1  # Bottom wall
                grid.binary_grid[i, height - 1] = 1  # Top wall
            for i in range(height):
                grid.binary_grid[0, i] = 1  # Left wall
                grid.binary_grid[width - 1, i] = 1  # Right wall

            # Add some obstacles that allow a path, defined in world coordinates
            # A few blocks in the top-left (relative to world dimensions)
            ox1, oy1 = grid.world_to_grid(width * 0.2, height * 0.2)
            grid.binary_grid[oy1, ox1] = 1
            grid.binary_grid[oy1 + 1, ox1] = 1
            grid.binary_grid[oy1, ox1 + 1] = 1
            grid.binary_grid[oy1 + 1, ox1 + 1] = 1

            # A few blocks in the bottom-right (relative to world dimensions)
            ox2, oy2 = grid.world_to_grid(width * 0.8, height * 0.8)
            grid.binary_grid[oy2, ox2] = 1
            grid.binary_grid[oy2 + 1, ox2] = 1
            grid.binary_grid[oy2, ox2 + 1] = 1
            grid.binary_grid[oy2 + 1, ox2 + 1] = 1

            # A long obstacle in the middle, leaving space above and below
            # Define start and end world coordinates for the obstacle
            obs_start_x_world = width * 0.3
            obs_end_x_world = width * 0.7
            obs_y_world = height / 2

            # Convert to grid coordinates
            obs_start_x_grid, obs_y_grid = grid.world_to_grid(
                obs_start_x_world, obs_y_world
            )
            obs_end_x_grid, _ = grid.world_to_grid(obs_end_x_world, obs_y_world)

            for i in range(obs_start_x_grid, obs_end_x_grid + 1):
                grid.binary_grid[obs_y_grid, i] = 1
                grid.binary_grid[obs_y_grid + 1, i] = 1

    # Position start and goal to allow smooth path planning around obstacles
    start_x = 3.0
    start_y = height / 2 - 3.0  # Further from center to avoid middle obstacle
    goal_x = width - 3.0
    goal_y = height / 2 + 3.0  # Further from center on opposite side

    return (
        grid,
        Pose(
            start_x, start_y, np.pi / 6
        ),  # Start with slight angle for better initial motion
        Pose(
            goal_x, goal_y, -np.pi / 6
        ),  # End with opposite angle for smoother approach
    )


def test_phase2() -> None:
    # Phase 2: Racing Line Optimization
    # Assuming we have a fully mapped environment and want to optimize the path for racing
    # The real car is constrained by discrete steering angles, fixed angular velocity, and discrete commands
    # Making this problem a motion planning problem, instead of a trajectory optimization problem
    # The DiscreteLatticeMotionPlanner discretizes the continuous space making this a graph search problem
    # So we can use A* search on a lattice of possible states

    # Initialize the occupancy grid
    grid_width = 20
    grid_height = 20
    occupancy_grid, start_pose, goal_pose = generate_grid(
        grid_width, grid_height, "obstacles"
    )

    # After Phase 1 exploration
    occupancy_grid.process_map()  # Clean up noise - Commented out to see raw obstacles

    # Visualize the occupancy grid
    # Create a dummy car for VisualizationManager, as it requires one
    # We don't need to simulate car movement here, just visualize the grid
    class DummyCar:
        def __init__(self, x, y, theta):
            self.x = x
            self.y = y
            self.theta = theta
            self.length = 0.3  # Example value
            self.width = 0.15  # Example value
            self.wheel_width = 0.05
            self.wheel_radius = 0.1
            self.wheel_offset = 0.02

        def get_corners(self):
            # Dummy implementation for visualization
            half_length = self.length / 2
            half_width = self.width / 2
            corners = [
                (-half_length, -half_width),
                (-half_length, half_width),
                (half_length, half_width),
                (half_length, -half_width),
            ]
            # Rotate and translate
            rotated_corners = []
            cos_theta = np.cos(self.theta)
            sin_theta = np.sin(self.theta)
            for x, y in corners:
                rotated_x = x * cos_theta - y * sin_theta + self.x
                rotated_y = x * sin_theta + y * cos_theta + self.y
                rotated_corners.append((rotated_x, rotated_y))
            return rotated_corners

        def visualize_steering(self):
            return []  # Not needed for grid visualization

    dummy_car = DummyCar(start_pose.x, start_pose.y, start_pose.theta)
    vis_manager = VisualizationManager(dummy_car, occupancy_grid)
    vis_manager.enable_element("car_body", False)
    vis_manager.enable_element("wheels", False)
    vis_manager.enable_element("steering_lines", False)
    vis_manager.enable_element("reference_path", False)
    vis_manager.enable_element("trajectory", False)

    # Add start and goal markers
    vis_manager.add_element(
        "start_marker",
        MarkerDrawer(
            start_pose.x, start_pose.y, color="green", marker="o", label="Start"
        ),
    )
    vis_manager.add_element(
        "goal_marker",
        MarkerDrawer(goal_pose.x, goal_pose.y, color="red", marker="x", label="Goal"),
    )

    vis_manager.initialize_all()
    vis_manager.show_legend()
    vis_manager.update_all()

    # Optimize for racing
    # Configure planner with optimized parameters
    planner_instance = phase2_discrete_planning(
        occupancy_grid,
        (start_pose.x, start_pose.y, start_pose.theta),
        (goal_pose.x, goal_pose.y, goal_pose.theta),
        primitive_duration=0.3,  # Short duration for precise control
        steering_angles=30,  # Moderate steering angle for balance of maneuverability and smoothness
        angular_velocity=0.8,  # Increased angular velocity for better turning
        wheelbase=0.3,  # Slightly larger wheelbase for stability
    )

    if planner_instance and planner_instance.command_sequence:
        # The phase2_discrete_planning function already visualizes the path
        # So we just need to keep the plot open
        print("Path found and visualized by LatticeMotionPlanner.")
    else:
        print("No path found by LatticeMotionPlanner.")
        if planner_instance:
            print("Visualizing explored states...")
            planner_instance.visualize_explored_states()

    print("Displaying occupancy grid and markers. Close the plot to continue.")
    plt.show()  # Keep the plot open until closed by user

    # print(
    #     "Command Sequence:",
    #     planner_instance.command_sequence if planner_instance else None,
    # )


if __name__ == "__main__":
    test_phase2()

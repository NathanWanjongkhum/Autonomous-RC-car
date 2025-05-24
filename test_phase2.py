from dataclasses import dataclass
from maps.OccupancyGrid import OccupancyGrid
import matplotlib.pyplot as plt
from path_optimizing_algorithms.LatticeMotionPlanner import (
    phase2_discrete_planning,
)
from DrawableElements import VisualizationManager, MarkerDrawer


@dataclass
class Pose:
    """A simple pose class to represent a position and orientation in 2D space."""

    x: float
    y: float
    theta: float


def generate_grid(
    width: int, height: int, grid_type: str = "corridor"
) -> tuple[OccupancyGrid, Pose, Pose]:
    # Generate a corridor
    grid = OccupancyGrid(
        width=width,
        height=height,
        resolution=0.1,
    )

    match grid_type:
        case "corridor":
            # Generate a corridor
            # Generate a straight horizontal corridor
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

    return (
        grid,
        Pose(1.0, int(height / 2), 0.0),  # Start in the middle of the left side
        Pose(width - 1.0, int(height / 2), 0.0),  # Goal in the middle of the right side
    )


def test_phase2() -> None:
    # Phase 2: Racing Line Optimization
    # Assuming we have a fully mapped environment and want to optimize the path for racing
    # The real car is constrained by discrete steering angles, fixed angular velocity, and discrete commands
    # Making this problem a motion planning problem, instead of a trajectory optimization problem
    # The DiscreteLatticeMotionPlanner discretizes the continuous space making this a graph search problem
    # So we can use A* search on a lattice of possible states

    # Initialize the occupancy grid
    grid_width = 10
    grid_height = 10
    occupancy_grid, start_pose, goal_pose = generate_grid(
        grid_width, grid_height, "corridor"
    )

    # After Phase 1 exploration
    occupancy_grid.process_map()  # Clean up noise

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
    command_sequence = phase2_discrete_planning(
        occupancy_grid,
        (start_pose.x, start_pose.y, start_pose.theta),
        (goal_pose.x, goal_pose.y, goal_pose.theta),
        primitive_duration=0.5,  # Increased duration to help find path in larger grid
    )

    if command_sequence:
        # The phase2_discrete_planning function already visualizes the path
        # So we just need to keep the plot open
        print("Path found and visualized by LatticeMotionPlanner.")
    else:
        print("No path found by LatticeMotionPlanner.")

    print("Displaying occupancy grid and markers. Close the plot to continue.")
    plt.show()  # Keep the plot open until closed by user

    print("Command Sequence:", command_sequence)


if __name__ == "__main__":
    test_phase2()

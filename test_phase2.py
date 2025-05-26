from dataclasses import dataclass
from maps.OccupancyGrid import OccupancyGrid
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

mplstyle.use(["dark_background", "fast"])


from path_optimizing_algorithms.LatticeMotionPlanner import (
    apply_motion_primitive,
    phase2_discrete_planning,
)
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
        resolution=0.1,  # Balance between accuracy and computational efficiency
    )

    adjusted_width = grid.binary_grid.shape[0]
    adjusted_height = grid.binary_grid.shape[1]

    match grid_type:
        case "empty":
            # All cells are free (initialized to 0 in OccupancyGrid)
            pass
        case "corridor":
            # Outer walls
            for i in range(adjusted_width):
                grid.binary_grid[i, 0] = 1  # Bottom wall
                grid.binary_grid[i, adjusted_height - 1] = 1  # Top wall
            for i in range(adjusted_height):
                grid.binary_grid[0, i] = 1  # Left wall
                grid.binary_grid[adjusted_width - 1, i] = 1  # Right wall

            # Clear a horizontal path in the middle
            corridor_start_y = int(adjusted_height / 2) - 1
            corridor_end_y = int(adjusted_height / 2) + 1
            for x in range(1, adjusted_width - 1):
                for y in range(corridor_start_y, corridor_end_y + 1):
                    grid.binary_grid[x, y] = 0  # Clear inner area
        case "obstacles":
            # Outer walls
            for i in range(adjusted_width):
                grid.binary_grid[i, 0] = 1  # Bottom wall
                grid.binary_grid[i, adjusted_height - 1] = 1  # Top wall
            for i in range(adjusted_height):
                grid.binary_grid[0, i] = 1  # Left wall
                grid.binary_grid[adjusted_width - 1, i] = 1  # Right wall

            # Add some obstacles that allow a path, defined in world coordinates
            # A few blocks in the top-left (relative to world dimensions)
            ox1, oy1 = grid._discretize_state(
                adjusted_width * 0.2, adjusted_height * 0.2
            )
            grid.binary_grid[oy1, ox1] = 1
            grid.binary_grid[oy1 + 1, ox1] = 1
            grid.binary_grid[oy1, ox1 + 1] = 1
            grid.binary_grid[oy1 + 1, ox1 + 1] = 1

            # A few blocks in the bottom-right (relative to world dimensions)
            ox2, oy2 = grid._discretize_state(
                adjusted_width * 0.8, adjusted_height * 0.8
            )
            grid.binary_grid[oy2, ox2] = 1
            grid.binary_grid[oy2 + 1, ox2] = 1
            grid.binary_grid[oy2, ox2 + 1] = 1
            grid.binary_grid[oy2 + 1, ox2 + 1] = 1

            # A long obstacle in the middle, leaving space above and below
            # Define start and end world coordinates for the obstacle
            obs_start_x_world = adjusted_width * 0.3
            obs_end_x_world = adjusted_width * 0.7
            obs_y_world = adjusted_height / 2

            # Convert to grid coordinates
            obs_start_x_grid, obs_y_grid = grid._discretize_state(
                obs_start_x_world, obs_y_world
            )
            obs_end_x_grid, _ = grid._discretize_state(obs_end_x_world, obs_y_world)

            for i in range(obs_start_x_grid, obs_end_x_grid + 1):
                grid.binary_grid[obs_y_grid, i] = 1
                grid.binary_grid[obs_y_grid + 1, i] = 1

    # Position start and goal to allow path planning around obstacles
    start_x = width * 0.1
    start_y = height * 0.1  # Further from center to avoid middle obstacle
    goal_x = width * 0.5  # Further from center on opposite side
    goal_y = height * 0.91  # Further from center on opposite side

    return (
        grid,
        Pose(start_x, start_y, np.pi / 6),
        Pose(goal_x, goal_y, -np.pi / 6),
    )


def test_phase2() -> None:
    # Phase 2: Racing Line Optimization
    # Assuming we have a fully mapped environment and want to optimize the path for racing
    # The real car is constrained by discrete steering angles, fixed angular velocity, and discrete commands
    # Making this problem a motion planning problem, instead of a trajectory optimization problem
    # The DiscreteLatticeMotionPlanner discretizes the continuous space making this a graph search problem
    # So we can use A* search on a lattice of possible states

    # Initialize the occupancy grid
    grid_width = 5
    grid_height = 5
    occupancy_grid, start_pose, goal_pose = generate_grid(
        grid_width, grid_height, "corridor"
    )

    # After Phase 1 exploration
    # occupancy_grid.process_map()  # Clean up noise

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    print(
        f"Occupancy grid width: {occupancy_grid.width}, height: {occupancy_grid.height}"
    )
    print(
        f"Occupancy grid grid_width: {occupancy_grid.grid_width}, grid_height: {occupancy_grid.grid_height}"
    )

    # Optimize for racing
    # Configure planner with optimized parameters
    planner = phase2_discrete_planning(
        occupancy_grid,
        start_pose,
        goal_pose,
        ax=ax,
        primitive_duration=0.5,  # Short duration for precise control
        steering_angles=30,  # Moderate steering angle for balance of maneuverability and smoothness
        angular_velocity=0.8,  # Increased angular velocity for better turning
        wheelbase=0.3,  # Slightly larger wheelbase for stability
    )

    if planner and planner.command_sequence:
        # Extract trajectory from command sequence
        trajectory = []
        current_x, current_y, current_theta = (
            start_pose.x,
            start_pose.y,
            start_pose.theta,
        )

        for primitive in planner.command_sequence:
            # Transform each point in the primitive's trajectory to world coordinates
            for x_local, y_local, theta_local in primitive.trajectory:
                # Transform from primitive's local frame to world frame
                x_world, y_world, _ = apply_motion_primitive(
                    (current_x, current_y, current_theta),
                    (x_local, y_local, theta_local),
                )

                trajectory.append((x_world, y_world))

            # Update current position using the primitive's end displacement
            current_x, current_y, current_theta = apply_motion_primitive(
                (current_x, current_y, current_theta), primitive.end_displacement
            )

            # Update trajectory in VisualizationManager
            # trajectory_drawer = vis_manager.get_element("trajectory")
            # if trajectory_drawer and isinstance(trajectory_drawer, TrajectoryDrawer):
            # print(f"Trajectory: {trajectory}")
            # print(f"TrajectoryDrawer: {trajectory_drawer}")
            # trajectory_drawer.set_positions(trajectory)

        planner.visualize_explored_states(ax=ax)
        plt.show()
        print("Displaying occupancy grid and markers. Close the plot to continue.")

        # print("Path found and visualized by LatticeMotionPlanner.")

    else:
        print("No path found by LatticeMotionPlanner.")
        print("Visualizing explored states...")
        # planner_instance.visualize_explored_states(ax=ax)


if __name__ == "__main__":
    test_phase2()

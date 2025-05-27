from dataclasses import dataclass
import math
import time
from Phase2Controller import Phase2Controller
from Simulation import Simulation
from maps.OccupancyGrid import OccupancyGrid
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

from pathfinding_algorithms.ConstantPPC import ConstantPurePursuitController
from robots.AckermannSteeringCar import AckermannSteeringCar

mplstyle.use(["dark_background", "fast"])


from path_optimizing_algorithms.LatticeMotionPlanner import (
    DiscreteLatticeMotionPlanner,
    apply_motion_primitive,
    SteeringCommand,
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
        case "aligned corridor":
            # Vertical wall
            for x in list(range(adjusted_width // 4 - 1, adjusted_width // 4 + 1)):
                for y in list(range(adjusted_height - 1)):
                    grid.binary_grid[x, y] = True

            # Gap in the middle of the wall
            for x in list(range(adjusted_width // 4 - 1, adjusted_width // 4 + 1)):
                for y in list(
                    range(
                        math.floor(adjusted_height / 4), math.floor(adjusted_height / 2)
                    )
                ):
                    grid.binary_grid[x, y] = False

            # Vertical wall
            for x in list(range(adjusted_width // 2 - 1, adjusted_width // 2 + 1)):
                for y in list(range(adjusted_height - 1)):
                    grid.binary_grid[x, y] = True

            # Gap in the middle of the wall
            for x in list(range(adjusted_width // 2 - 1, adjusted_width // 2 + 1)):
                for y in list(
                    range(math.floor(3 * adjusted_height / 4), adjusted_height - 1)
                ):
                    grid.binary_grid[x, y] = False
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
    goal_x = width * 0.9  # Further from center on opposite side
    goal_y = height * 0.9  # Further from center on opposite side

    return (
        grid,
        Pose(start_x, start_y, np.pi / 6),
        Pose(goal_x, goal_y, np.pi / 6),
    )


def test_phase2() -> None:
    # Phase 2: Racing Line Optimization
    # Assuming we have a fully mapped environment and want to optimize the path for racing
    # The real car is constrained by discrete steering angles, fixed angular velocity, and discrete commands
    # Making this problem a motion planning problem, instead of a trajectory optimization problem
    # The DiscreteLatticeMotionPlanner discretizes the continuous space making this a graph search problem
    # So we can use A* search on a lattice of possible states

    # Initialize the occupancy grid
    print("=== TESTING PHASE 2 ===")
    # print("=== PHASE 2: RACING LINE OPTIMIZATION ===")
    grid_width = 20
    grid_height = 20
    grid, start_pose, goal_pose = generate_grid(
        grid_width, grid_height, "aligned corridor"
    )

    car = AckermannSteeringCar(x=0.5, y=0.5, theta=0.0)

    sim = Simulation(car=car, grid=grid, start_pose=start_pose, goal_pose=goal_pose)

    # Intermediate Phase: Planning
    integrated_controller = transition_to_phase2(sim, start_pose, goal_pose)

    if integrated_controller:
        # Phase 2: Execution
        execute_phase2(sim, integrated_controller)  # Commented out by user
        pass
    else:
        print("Phase 2 planning failed!")


def transition_to_phase2(
    simulation, start_pose: Pose | None, goal_pose: Pose | None
) -> Phase2Controller:
    """
    Transition from Phase 1 exploration to Phase 2 execution

    This function handles the intermediate processing phase where we:
    1. Process the explored map
    2. Plan the optimal trajectory
    3. Set up the integrated controller
    """
    print("=== INTERMEDIATE PHASE: PROCESSING AND PLANNING ===")

    start_time = time.time()

    # Step 1: Process the occupancy grid from exploration
    print("Processing occupancy grid...")
    # simulation.grid.process_map()  # Clean up noise from exploration

    # Step 2: Define start and goal for Phase 2
    # In a real competition, this might be the same start/finish line
    if start_pose is None:
        start_pose = start_pose

    # For demonstration, let's set a goal position
    # In practice, this might be the finish line or a lap completion point
    if goal_pose is None:
        goal_pose = goal_pose

    print(f"Planning from {start_pose} to {goal_pose}")
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Step 3: Create and configure the lattice planner
    lattice_planner = DiscreteLatticeMotionPlanner(
        occupancy_grid=simulation.grid,
        angular_velocity=0.8,  # Faster for Phase 2
        steering_angle_left=20,  # Degrees
        steering_angle_right=-20,
        wheelbase=0.25,
        primitive_duration=0.4,  # Shorter for more responsive control
        num_angle_discretizations=64,  # Higher resolution for smoother paths
    )

    # Set the minimum progress threshold
    lattice_planner.min_progress_threshold = 0.01

    # Step 4: Plan the optimal trajectory
    print("Planning optimal trajectory...")
    command_sequence = lattice_planner.plan_discrete_path(
        start_pose.x,
        start_pose.y,
        start_pose.theta,
        goal_pose.x,
        goal_pose.y,
        goal_pose.theta,
        timeout=2500.0,  # Allow more time for complex paths
    )

    # Test the calculate_goal_progress method
    primitive = lattice_planner.motion_primitives[0][SteeringCommand.NEUTRAL]
    progress = lattice_planner.calculate_goal_progress(primitive)
    print(f"Goal progress for neutral primitive: {progress}")

    if not command_sequence:
        print("ERROR: Could not find path for Phase 2!")
        return None

    if command_sequence:
        # Extract trajectory from command sequence
        trajectory = []
        current_x, current_y, current_theta = (
            start_pose.x,
            start_pose.y,
            start_pose.theta,
        )

        for primitive in command_sequence:
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

        lattice_planner.visualize_command_sequence(
            start_x=start_pose.x,
            start_y=start_pose.y,
            start_theta=start_pose.theta,
            goal_x=goal_pose.x,
            goal_y=goal_pose.y,
            goal_theta=goal_pose.theta,
            ax=ax,
        )

        lattice_planner.visualize_explored_states(ax=ax)

        # Save tuning parameters to a file
        tuning_params = {
            "angular_velocity": lattice_planner.angular_velocity,
            "steering_angle_left": lattice_planner.steering_angles[
                SteeringCommand.LEFT
            ],
            "steering_angle_right": lattice_planner.steering_angles[
                SteeringCommand.RIGHT
            ],
            "wheelbase": lattice_planner.wheelbase,
            "primitive_duration": lattice_planner.primitive_duration,
            "num_angle_discretizations": lattice_planner.num_angles,
        }
        filename = "saved_simulations/tuning_params.txt"
        with open(filename, "w") as f:
            for key, value in tuning_params.items():
                f.write(f"{key}: {value}\n")

            f.write(f"\nPath planning completed in {time.time() - start_time:.3f}s\n")
            f.write(f"States explored: {lattice_planner.nodes_explored}\n")
            f.write(f"Found optimal path with {len(command_sequence)} commands\n")
            f.write(
                f"Simulated time: {len(command_sequence) * lattice_planner.primitive_duration:.3f}s\n"
            )

        # Save the plot to a file
        plot_filename = "saved_simulations/trajectory_plot.png"
        fig.savefig(plot_filename)

        plt.show(block=True)

        print("Displaying occupancy grid and markers. Close the plot to continue.")
    else:
        print("No path found by LatticeMotionPlanner.")
        print("Visualizing explored states...")

    # Step 5: Create the pure pursuit controller
    pure_pursuit = ConstantPurePursuitController(
        angular_velocity=0.8,  # Match lattice planner
        base_lookahead=0.6,  # Larger lookahead for higher speeds
        hysteresis_threshold=5.0,  # Tighter for precision
        dead_zone_threshold=2.0,  # Smaller dead zone
        max_integral_degrees=15.0,  # More aggressive correction
        feedforward_lookahead_points=7,  # Look further ahead
    )

    # Step 6: Create the integrated controller
    integrated_controller = Phase2Controller(
        lattice_planner=lattice_planner,
        pure_pursuit=pure_pursuit,
        max_path_deviation=0.8,  # Tighter tolerance for racing
        replan_interval=2.0,  # Faster replanning
        emergency_deviation=1.0,  # Emergency threshold
    )

    # Step 7: Set the planned trajectory
    integrated_controller.set_planned_trajectory(
        command_sequence, (start_pose.x, start_pose.y, start_pose.theta)
    )

    print("=== READY FOR PHASE 2 EXECUTION ===")
    return integrated_controller


def execute_phase2(simulation, integrated_controller):
    """
    Execute Phase 2 with the integrated controller
    """
    print("=== PHASE 2: HIGH-SPEED EXECUTION ===")

    # Reset car to starting position
    start_x, start_y = integrated_controller.reference_trajectory[0]
    simulation.car.x = start_x
    simulation.car.y = start_y
    simulation.car.theta = 0.0
    simulation.car.v = 0.0

    # Initialize visualization if needed
    if not simulation.fig:
        simulation.initialize_visualization()

    dt = 0.05  # Faster control loop for Phase 2
    max_steps = 5000
    step = 0

    while step < max_steps:
        current_time = time.time()
        current_pose = (simulation.car.x, simulation.car.y, simulation.car.theta)

        # Get control commands from integrated controller
        steering_angle, linear_velocity = integrated_controller.compute_steering(
            current_pose, current_time, dt
        )

        if steering_angle == 0.0 and linear_velocity == 0.0:
            break  # Stop if no valid control command is returned

        # Apply control commands
        simulation.car.set_control_inputs(linear_velocity, steering_angle)
        simulation.car.update_state(dt)

        # Update visualization every few steps
        if step % 10 == 0:
            simulation.update_visualization()

            # Print performance metrics
            metrics = integrated_controller.get_performance_metrics()
            print(
                f"Step {step}: Mode={metrics.get('current_mode', 'unknown')}, "
                f"Deviation={metrics.get('mean_deviation', 0):.3f}m, "
                f"Progress={metrics.get('primitive_progress', 0)*100:.1f}%"
            )

        # Emergency stop check
        if metrics.get("max_deviation", 0) > integrated_controller.emergency_deviation:
            print("Emergency stop triggered!")
            break

        step += 1
        time.sleep(dt)  # Maintain real-time execution

    # Show final performance
    integrated_controller.visualize_performance()
    final_metrics = integrated_controller.get_performance_metrics()

    print("=== PHASE 2 PERFORMANCE SUMMARY ===")
    for key, value in final_metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    test_phase2()

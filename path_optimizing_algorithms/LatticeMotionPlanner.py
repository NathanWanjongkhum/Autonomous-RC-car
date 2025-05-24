import numpy as np
import heapq
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time
from enum import Enum


class SteeringCommand(Enum):
    """Discrete steering commands available to the robot"""

    LEFT = "left"
    NEUTRAL = "neutral"
    RIGHT = "right"


@dataclass
class DiscreteMotionPrimitive:
    """
    Represents a motion primitive for discrete control
    Each primitive is a fixed-duration execution of a steering command
    """

    steering_command: SteeringCommand
    duration: float  # How long to execute this command
    trajectory: List[Tuple[float, float, float]]  # Resulting trajectory
    end_displacement: Tuple[float, float, float]  # (dx, dy, dtheta) from start
    linear_velocity: float  # Resulting linear velocity for this primitive
    cost: float  # Cost of executing this primitive


class DiscreteLatticeMotionPlanner:
    """
    Motion planner for robot with discrete steering angles and fixed angular velocity.
    Outputs a sequence of discrete commands that can be directly executed.
    """

    def __init__(
        self,
        occupancy_grid,
        angular_velocity: float = 0.5,
        steering_angle_left: float = 35,
        steering_angle_right: float = -35,
        wheelbase: float = 0.25,
        primitive_duration: float = 0.5,
        num_angle_discretizations: int = 32,  # Increased for smoother paths
    ):
        """
        Initialize the discrete motion planner

        Parameters:
        occupancy_grid: OccupancyGrid instance
        angular_velocity: Fixed angular velocity when turning (rad/s)
        steering_angle_left/right: Fixed steering angles for left/right
        wheelbase: Distance between axles (m)
        primitive_duration: Duration of each motion primitive (s)
        num_angle_discretizations: Number of discrete heading angles
        """
        self.grid = occupancy_grid
        self.angular_velocity = angular_velocity
        self.wheelbase = wheelbase
        self.primitive_duration = primitive_duration
        self.num_angles = num_angle_discretizations

        # Discrete steering angles
        self.steering_angles = {
            SteeringCommand.LEFT: np.radians(steering_angle_left),
            SteeringCommand.NEUTRAL: 0.0,
            SteeringCommand.RIGHT: np.radians(steering_angle_right),
        }

        # Calculate linear velocities for each steering command
        # For fixed angular velocity: v = ω * R, where R = L/tan(φ)
        self.linear_velocities = {}
        for cmd, angle in self.steering_angles.items():
            if abs(angle) > 1e-6:
                # Turning radius
                radius = self.wheelbase / np.tan(abs(angle))
                self.linear_velocities[cmd] = self.angular_velocity * radius
            else:
                # Straight motion - use a reasonable forward speed
                self.linear_velocities[cmd] = (
                    0.5  # m/s (tune this based on your hardware)
                )

        # Resolution for discretization
        self.xy_resolution = occupancy_grid.resolution
        self.angle_resolution = 2 * np.pi / num_angle_discretizations

        # Precomputed motion primitives
        self.motion_primitives = {}
        self._generate_motion_primitives()

        self.explored_states = []  # To store states visited during search

    def _generate_motion_primitives(self):
        """
        Generate motion primitives for each discrete angle and steering command.
        These represent the exact motions the robot will execute.
        """
        print("Generating discrete motion primitives...")

        dt = 0.05  # Integration timestep
        num_steps = int(self.primitive_duration / dt)

        # For each discrete starting angle
        for angle_idx in range(self.num_angles):
            self.motion_primitives[angle_idx] = {}
            start_angle = angle_idx * self.angle_resolution

            # For each steering command
            for steering_cmd in SteeringCommand:
                steering_angle = self.steering_angles[steering_cmd]
                linear_velocity = self.linear_velocities[steering_cmd]

                # Simulate the motion
                trajectory = []
                x, y, theta = 0.0, 0.0, start_angle

                for _ in range(num_steps):
                    trajectory.append((x, y, theta))

                    # Update using bicycle kinematics
                    x += linear_velocity * np.cos(theta) * dt
                    y += linear_velocity * np.sin(theta) * dt
                    theta += (
                        (linear_velocity / self.wheelbase) * np.tan(steering_angle) * dt
                    )

                # Final displacement
                end_displacement = (x, y, theta - start_angle)

                # Cost calculation
                distance = np.sqrt(x**2 + y**2)
                if steering_cmd == SteeringCommand.NEUTRAL:
                    cost = distance  # Prefer straight motion
                else:
                    cost = distance * 1.5  # Penalize turns slightly

                # Create primitive
                primitive = DiscreteMotionPrimitive(
                    steering_command=steering_cmd,
                    duration=self.primitive_duration,
                    trajectory=trajectory,
                    end_displacement=end_displacement,
                    linear_velocity=linear_velocity,
                    cost=cost,
                )

                self.motion_primitives[angle_idx][steering_cmd] = primitive

        print(
            f"Generated {len(self.motion_primitives) * len(SteeringCommand)} motion primitives"
        )

    def plan_discrete_path(
        self,
        start_x: float,
        start_y: float,
        start_theta: float,
        goal_x: float,
        goal_y: float,
        goal_theta: float,
        timeout: float = 120.0,  # Increased timeout for larger grids
    ) -> Optional[List[DiscreteMotionPrimitive]]:
        """
        Plan a sequence of discrete motion primitives from start to goal.

        Returns:
        List of DiscreteMotionPrimitive objects representing the command sequence,
        or None if no path found
        """
        start_time = time.time()

        # Discretize start and goal
        start_discrete = self._discretize_state(start_x, start_y, start_theta)
        goal_discrete = self._discretize_state(goal_x, goal_y, goal_theta)

        # A* search setup
        # State: (x_idx, y_idx, theta_idx)
        # Node: (f_score, g_score, state, parent_state, primitive_used)
        open_set = []
        heapq.heappush(open_set, (0, 0, start_discrete, None, None))

        closed_set = set()
        came_from = {}  # state -> (parent_state, primitive)
        g_score = {start_discrete: 0}

        # Goal tolerance - increased for better path finding
        goal_xy_tolerance = 5  # grid cells
        goal_theta_tolerance = 4  # angle indices

        while open_set and (time.time() - start_time) < timeout:
            # Get the node with lowest f_score
            _, current_g, current_state, parent_state, primitive_used = heapq.heappop(
                open_set
            )

            # Store explored state for visualization
            self.explored_states.append(self._continuous_state(*current_state))

            # Check if pose is close enough to goal
            dx = abs(current_state[0] - goal_discrete[0])
            dy = abs(current_state[1] - goal_discrete[1])
            dtheta = min(
                abs(current_state[2] - goal_discrete[2]),
                self.num_angles - abs(current_state[2] - goal_discrete[2]),
            )

            # If close enough, reconstruct command sequence and return
            if (
                dx <= goal_xy_tolerance
                and dy <= goal_xy_tolerance
                and dtheta <= goal_theta_tolerance
            ):
                return self._reconstruct_command_sequence(current_state, came_from)

            # If current state is already closed, skip
            if current_state in closed_set:
                continue

            # Mark current state as closed
            closed_set.add(current_state)
            # Always store the parent and primitive that got us to this state
            if parent_state is not None and primitive_used is not None:
                came_from[current_state] = (parent_state, primitive_used)
                # print(
                #     f"Adding to path: {current_state} from {parent_state} using {primitive_used.steering_command.value}"
                # )

            # Get continuous position
            current_x, current_y, current_theta = self._continuous_state(*current_state)

            # Try each motion primitive
            current_angle_idx = current_state[2]
            for steering_cmd, primitive in self.motion_primitives[
                current_angle_idx
            ].items():
                # Check if trajectory is collision-free
                if not self._is_primitive_collision_free(
                    current_x, current_y, current_theta, primitive
                ):
                    continue

                # Calculate next state
                dx, dy, dtheta = primitive.end_displacement
                next_x = current_x + dx
                next_y = current_y + dy
                next_theta = (current_state[2] * self.angle_resolution + dtheta) % (
                    2 * np.pi
                )

                next_state = self._discretize_state(next_x, next_y, next_theta)

                if next_state in closed_set:
                    continue

                # Calculate cost
                tentative_g = current_g + primitive.cost

                # Add switching penalty if changing steering direction
                if (
                    parent_state is not None
                    and primitive_used is not None
                    and primitive_used.steering_command != primitive.steering_command
                ):
                    tentative_g += 0.5  # Small penalty for switching

                # Update if better path found
                if next_state not in g_score or tentative_g < g_score[next_state]:
                    g_score[next_state] = tentative_g
                    h_score = self._heuristic(next_x, next_y, goal_x, goal_y)
                    f_score = tentative_g + h_score

                    # Store this transition immediately
                    came_from[next_state] = (current_state, primitive)
                    # print(
                    #     f"Found better path to {next_state} from {current_state} using {primitive.steering_command.value}"
                    # )

                    heapq.heappush(
                        open_set,
                        (f_score, tentative_g, next_state, current_state, primitive),
                    )

        print("No path found within timeout")
        return None

    def _discretize_state(
        self, x: float, y: float, theta: float
    ) -> Tuple[int, int, int]:
        """Convert continuous state to discrete indices"""
        x_idx = int(round(x / self.xy_resolution))
        y_idx = int(round(y / self.xy_resolution))

        # Normalize theta to [0, 2π)
        theta_norm = (theta + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi)
        theta_norm = (theta_norm + 2 * np.pi) % (2 * np.pi)  # Then to [0, 2pi)

        # convert to index
        theta_idx = int(round(theta_norm / self.angle_resolution)) % self.num_angles

        # print(f"Discretize: ({x:.2f}, {y:.2f}, {np.degrees(theta):.2f} deg) -> ({x_idx}, {y_idx}, {theta_idx})")
        return x_idx, y_idx, theta_idx

    def _continuous_state(
        self, x_idx: int, y_idx: int, theta_idx: int
    ) -> Tuple[float, float, float]:
        """Convert discrete indices to continuous state"""
        x = x_idx * self.xy_resolution
        y = y_idx * self.xy_resolution
        theta = theta_idx * self.angle_resolution
        return x, y, theta

    def _is_primitive_collision_free(
        self,
        start_x: float,
        start_y: float,
        start_theta: float,
        primitive: DiscreteMotionPrimitive,
    ) -> bool:
        """Check if executing this primitive from start position is collision-free"""
        # Vehicle dimensions and safety margin (in meters)
        vehicle_length = 0.3
        vehicle_width = 0.15
        safety_margin = 0.2
        total_margin = safety_margin + max(vehicle_length, vehicle_width) / 2

        # Sample points along trajectory more densely
        dt = 0.05  # 50ms sampling
        t = 0
        while t <= primitive.duration:
            # Get state at this time
            idx = int(t / 0.05)  # Match dt from _generate_motion_primitives
            if idx >= len(primitive.trajectory):
                break
            x_local, y_local, _ = primitive.trajectory[idx]

            # Transform to world coordinates
            cos_theta = np.cos(start_theta)
            sin_theta = np.sin(start_theta)
            x_world = start_x + x_local * cos_theta - y_local * sin_theta
            y_world = start_y + x_local * sin_theta + y_local * cos_theta

            # Check bounds with total margin
            if (
                x_world < total_margin
                or x_world >= (self.grid.width - total_margin)
                or y_world < total_margin
                or y_world >= (self.grid.height - total_margin)
            ):
                return False

            # Convert to grid coordinates
            grid_x, grid_y = self.grid.world_to_grid(x_world, y_world)
            margin_cells = int(total_margin / self.grid.resolution)

            # Check a rectangular region around the point
            for dy in range(-margin_cells, margin_cells + 1):
                for dx in range(-margin_cells, margin_cells + 1):
                    check_x = grid_x + dx
                    check_y = grid_y + dy

                    if (
                        check_x < 0
                        or check_x >= self.grid.grid_width
                        or check_y < 0
                        or check_y >= self.grid.grid_height
                    ):
                        continue

                    # Weight diagonal cells less to approximate circular check
                    if (
                        dx * dx + dy * dy <= margin_cells * margin_cells
                        and self.grid.binary_grid[check_y, check_x]
                    ):
                        return False

            t += dt

        return True

    def _heuristic(self, x: float, y: float, goal_x: float, goal_y: float) -> float:
        """Combined Euclidean and Manhattan distance heuristic"""
        dx = abs(goal_x - x)
        dy = abs(goal_y - y)
        # Combine both metrics for better path finding
        euclidean = np.sqrt(dx * dx + dy * dy)
        manhattan = dx + dy
        return 0.6 * euclidean + 0.4 * manhattan

    def _reconstruct_command_sequence(
        self, final_state: Tuple[int, int, int], came_from: Dict
    ) -> List[DiscreteMotionPrimitive]:
        """Reconstruct the sequence of motion primitives"""
        sequence = []
        current = final_state

        # Debug info
        print(f"Reconstructing path from state {final_state}")
        print(f"Number of states in came_from: {len(came_from)}")

        while current in came_from:
            parent_state, primitive = came_from[current]
            # print(f"Current state: {current}, Parent state: {parent_state}")

            if primitive is not None:
                sequence.append(primitive)
                # print(f"Added primitive: {primitive.steering_command.value}")
            # else:
            # print("Warning: No primitive for this state transition")

            if parent_state == current:
                # print("Warning: Parent state equals current state")
                break

            current = parent_state

        if not sequence:
            print("Warning: No primitives found in path reconstruction")
            # Try to find any valid primitive that gets us closer to the goal
            for state, (parent, primitive) in came_from.items():
                if primitive is not None:
                    sequence.append(primitive)
                    break

        sequence.reverse()
        return sequence

    def execute_command_sequence(
        self, car, command_sequence: List[DiscreteMotionPrimitive]
    ):
        """
        Execute the planned command sequence on the car.
        This shows how to convert primitives to actual car commands.
        """
        for i, primitive in enumerate(command_sequence):
            print(
                f"Step {i+1}: {primitive.steering_command.value} for {primitive.duration}s"
            )

            # Set steering angle based on command
            if primitive.steering_command == SteeringCommand.LEFT:
                steering_angle = self.steering_angles[SteeringCommand.LEFT]
            elif primitive.steering_command == SteeringCommand.RIGHT:
                steering_angle = self.steering_angles[SteeringCommand.RIGHT]
            else:
                steering_angle = 0.0

            # Set velocities
            car.set_control_inputs(primitive.linear_velocity, steering_angle)

            # In real implementation, would execute for primitive.duration seconds
            # For simulation, update for multiple timesteps
            num_steps = int(primitive.duration / 0.1)  # 0.1s simulation timestep
            for _ in range(num_steps):
                car.update_state(0.1)

    def visualize_command_sequence(
        self,
        start_x: float,
        start_y: float,
        start_theta: float,
        command_sequence: List[DiscreteMotionPrimitive],
    ):
        """Visualize the planned command sequence"""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Colors for different commands
        colors = {
            SteeringCommand.LEFT: "red",
            SteeringCommand.NEUTRAL: "green",
            SteeringCommand.RIGHT: "blue",
        }

        # Draw the path
        current_x, current_y, current_theta = start_x, start_y, start_theta

        for i, primitive in enumerate(command_sequence):
            # Get trajectory in world coordinates
            trajectory_world = []
            cos_theta = np.cos(current_theta)
            sin_theta = np.sin(current_theta)

            for x_local, y_local, theta_local in primitive.trajectory:
                x_world = current_x + x_local * cos_theta - y_local * sin_theta
                y_world = current_y + x_local * sin_theta + y_local * cos_theta
                trajectory_world.append((x_world, y_world))

            # Plot this segment
            x_coords = [p[0] for p in trajectory_world]
            y_coords = [p[1] for p in trajectory_world]
            ax.plot(
                x_coords,
                y_coords,
                color=colors[primitive.steering_command],
                linewidth=3,
                label=f"{primitive.steering_command.value}" if i == 0 else "",
            )

            # Update current position
            dx, dy, dtheta = primitive.end_displacement
            current_x += dx * cos_theta - dy * sin_theta
            current_y += dx * sin_theta + dy * cos_theta
            current_theta += dtheta

        # Draw occupancy grid
        if hasattr(self.grid, "binary_grid"):
            ax.imshow(
                self.grid.binary_grid.T,
                origin="lower",
                extent=[0, self.grid.width, 0, self.grid.height],
                cmap="gray_r",
                alpha=0.3,
            )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Discrete Command Sequence Visualization")
        ax.legend()
        ax.grid(True)
        ax.axis("equal")
        plt.show()

    def visualize_explored_states(self):
        """Visualize the states explored by the planner."""
        if not self.explored_states:
            print("No states were explored to visualize.")
            return

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Draw occupancy grid
        if hasattr(self.grid, "binary_grid"):
            ax.imshow(
                self.grid.binary_grid.T,
                origin="lower",
                extent=[0, self.grid.width, 0, self.grid.height],
                cmap="gray_r",
                alpha=0.3,
            )

        # Plot explored states
        explored_x = [s[0] for s in self.explored_states]
        explored_y = [s[1] for s in self.explored_states]
        ax.plot(
            explored_x,
            explored_y,
            "o",
            color="purple",
            markersize=2,
            alpha=0.5,
            label="Explored States",
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Explored States Visualization")
        ax.legend()
        ax.grid(True)
        ax.axis("equal")
        plt.show()


# Integration with your existing system
def phase2_discrete_planning(
    occupancy_grid,
    start_pose,
    goal_pose,
    steering_angles=35,
    angular_velocity=0.5,
    wheelbase=0.25,
    primitive_duration=0.5,  # Increased duration for better path finding
) -> List[DiscreteMotionPrimitive]:
    """
    Plan the optimal discrete command sequence for Phase 2

    Parameters:
    occupancy_grid: Processed occupancy grid from Phase 1
    start_pose: (x, y, theta) starting position
    goal_pose: (x, y, theta) goal position

    Returns:
    command_sequence: List of discrete motion primitives to execute
    """
    # Create the discrete planner
    planner = DiscreteLatticeMotionPlanner(
        occupancy_grid=occupancy_grid,
        angular_velocity=angular_velocity,
        steering_angle_left=steering_angles,
        steering_angle_right=-steering_angles,
        wheelbase=wheelbase,
        primitive_duration=primitive_duration,
        num_angle_discretizations=32,  # Increased angle discretization for smoother paths
    )

    # Plan the command sequence
    command_sequence = planner.plan_discrete_path(
        start_pose[0],
        start_pose[1],
        start_pose[2],
        goal_pose[0],
        goal_pose[1],
        goal_pose[2],
    )

    planner.command_sequence = (
        command_sequence  # Store command sequence as an attribute
    )

    if planner.command_sequence:
        print(f"Found path with {len(planner.command_sequence)} commands")

        # Visualize the planned sequence
        # planner.visualize_command_sequence(
        #     start_pose[0], start_pose[1], start_pose[2], planner.command_sequence
        # )

        # Print command summary
        # print("\nCommand sequence:")
        # for i, cmd in enumerate(planner.command_sequence):
        #     print(f"  {i+1}. {cmd.steering_command.value} for {cmd.duration}s")
    else:
        print("No path found!")

    return planner

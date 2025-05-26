import math
import numpy as np
import heapq
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time
from enum import Enum
import matplotlib.pyplot as plt


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
    cost: float  # Cost of executing this primitive


@dataclass
class ContinuousState:
    """Represents an exact continuous state with full precision"""

    x: float
    y: float
    theta: float

    def __hash__(self):
        # For use in sets/dicts - based on discrete representation
        return hash((int(self.x * 1000), int(self.y * 1000), int(self.theta * 1000)))


@dataclass
class DiscreteState:
    """Represents an exact continuous state with full precision"""

    x: int
    y: int
    theta: int

    def __hash__(self):
        # For use in sets/dicts - based on discrete representation
        return hash((self.x * 1000, self.y * 1000, self.theta * 1000))


@dataclass
class SearchNode:
    """Enhanced A* search node that maintains trajectory connectivity"""

    discrete_pose: DiscreteState  # For A* indexing
    continuous_pose: ContinuousState  # Exact continuous position

    trajectory_segment: List[ContinuousState]  # Connecting trajectory
    parent_node: Optional["SearchNode"]  # Parent node reference

    primitive_used: Optional[
        "DiscreteMotionPrimitive"
    ]  # Primitive that led to this node

    g_score: float
    f_score: float


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
        wheelradius: float = 1,
        reference_point: str = "rear",
        primitive_duration: float = 0.5,
        num_angle_discretizations: int = 64,  # Increased for smoother paths
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
        self.wheelradius = wheelradius
        self.reference_point = reference_point
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

        # Basic wheel speed to linear speed
        self.linear_velocity = self.angular_velocity * self.wheelradius

        # Resolution for discretization
        self.resolution = occupancy_grid.resolution
        self.angle_resolution = 2 * np.pi / num_angle_discretizations

        # Goal tolerance - tightened for precise goal reaching
        self.goal_tolerance = 1  # grid cells - much tighter tolerance
        self.goal_theta_tolerance = 2  # angle indices - tighter tolerance

        # Precomputed motion primitives
        self.motion_primitives = {}
        self._generate_motion_primitives()

        self.explored_states: List[ContinuousState] = (
            []
        )  # To store states visited during search

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

                # Simulate the motion
                trajectory = []
                x, y, theta = 0.0, 0.0, start_angle

                trajectory.append((x, y, theta))
                for _ in range(num_steps):
                    # Update using bicycle kinematics
                    x += self.linear_velocity * np.cos(theta) * dt
                    y += self.linear_velocity * np.sin(theta) * dt
                    # Calculate angular velocity (same as car)
                    if abs(self.linear_velocity) > 1e-5:
                        omega = (
                            self.linear_velocity
                            * np.tan(steering_angle)
                            / self.wheelbase
                        )
                    else:
                        omega = 0

                    theta += omega * dt

                    # Normalize theta to prevent accumulation (same as car)
                    theta = np.arctan2(np.sin(theta), np.cos(theta))

                    trajectory.append((x, y, theta))

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
        timeout: float = 300.0,  # Increased timeout for larger grids
    ) -> Optional[List[DiscreteMotionPrimitive]]:
        """
        Plan a sequence of discrete motion primitives from start to goal.

        Returns:
        List of DiscreteMotionPrimitive objects representing the command sequence,
        or None if no path found
        """
        print("=== CONNECTIVITY-PRESERVING A* SEARCH ===")
        print(f"Start: ({start_x:.3f}, {start_y:.3f}, {start_theta:.3f})")
        print(f"Goal:  ({goal_x:.3f}, {goal_y:.3f}, {goal_theta:.3f})")

        start_time = time.time()

        # Discretize start and goal
        start_continuous = ContinuousState(start_x, start_y, start_theta)
        start_discrete = self._discretize_state(start_x, start_y, start_theta)
        goal_discrete = self._discretize_state(goal_x, goal_y, goal_theta)

        print(f"Start discrete state: {start_discrete}")
        print(f"Goal discrete state:  {goal_discrete}")

        # Create start node
        start_node = SearchNode(
            discrete_pose=start_discrete,
            continuous_pose=start_continuous,
            trajectory_segment=[],  # No trajectory to get to start
            parent_node=None,
            primitive_used=None,
            g_score=0.0,
            f_score=self._heuristic(start_x, start_y, goal_x, goal_y),
        )

        # A* search setup
        # State: (x_idx, y_idx, theta_idx)
        # Node: (f_score, g_score, state, parent_state, primitive_used)
        nodes_explored = 0
        open_set = []
        heapq.heappush(open_set, (start_node.f_score, id(start_node), start_node))

        self.search_nodes = {start_discrete: start_node}
        closed_set = set()

        while open_set and (time.time() - start_time) < timeout:
            # Get the node with lowest f_score
            _, _, current_node = heapq.heappop(open_set)
            nodes_explored += 1
            if nodes_explored % 100 == 0:
                print(f"Explored {nodes_explored} nodes...")

            # Store explored state for visualization
            self.explored_states.append(current_node.continuous_pose)

            print(
                f"Exploring node {nodes_explored}: discrete={current_node.discrete_pose}, "
                f"exact=({current_node.continuous_pose.x:.3f}, {current_node.continuous_pose.y:.3f}, "
                f"{current_node.continuous_pose.theta:.3f})"
            )

            # Check if pose is close enough to goal
            dx = abs(current_node.discrete_pose[0] - goal_discrete[0])
            dy = abs(current_node.discrete_pose[1] - goal_discrete[1])
            dtheta = min(
                abs(current_node.discrete_pose[2] - goal_discrete[2]),
                self.num_angles - abs(current_node.discrete_pose[2] - goal_discrete[2]),
            )

            # If close enough, reconstruct command sequence and return
            if (
                dx <= self.goal_tolerance
                and dy <= self.goal_tolerance
                and dtheta <= self.goal_theta_tolerance
            ):
                print(f"GOAL REACHED! Nodes explored: {nodes_explored}")
                return self._reconstruct_connectivity_preserving_path(current_node)

            # If current state is already closed, skip
            if current_node.discrete_pose in closed_set:
                continue

            # Mark current state as closed
            closed_set.add(current_node.discrete_pose)

            # Always store the parent and primitive that got us to this state

            # Try each motion primitive
            current_angle_idx = current_node.discrete_pose[2]
            for steering_cmd, primitive in self.motion_primitives[
                current_angle_idx
            ].items():
                # Check if trajectory is collision-free
                if not self._is_primitive_collision_free(
                    current_node.continuous_pose.x,
                    current_node.continuous_pose.y,
                    current_node.continuous_pose.theta,
                    primitive,
                ):
                    continue

                # Calculate next state
                next_x, next_y, next_theta = apply_motion_primitive(
                    current_node.continuous_pose, primitive.end_displacement
                )

                # Create exact next pose
                next_continuous_state = ContinuousState(next_x, next_y, next_theta)

                next_discrete_state = self._discretize_state(next_x, next_y, next_theta)

                if next_discrete_state in closed_set:
                    continue

                # Calculate cost
                tentative_g = current_node.g_score + primitive.cost

                # Add switching penalty if changing steering direction
                if (
                    current_node.primitive_used is not None
                    and current_node.primitive_used.steering_command
                    != primitive.steering_command
                ):
                    tentative_g += 0.5  # Small penalty for switching

                # Update if better path found
                if (
                    next_discrete_state not in self.search_nodes
                    or tentative_g < self.search_nodes[next_discrete_state].g_score
                ):
                    # Generate trajectory segment in world coordinates
                    world_trajectory = []
                    for x_local, y_local, theta_local in primitive.trajectory:
                        x_world, y_world, theta_world = apply_motion_primitive(
                            current_node.continuous_pose,
                            (x_local, y_local, theta_local),
                        )
                        world_trajectory.append((x_world, y_world, theta_world))

                    # Create new search node
                    h_score = self._heuristic(next_x, next_y, goal_x, goal_y)
                    f_score = tentative_g + h_score

                    next_node = SearchNode(
                        discrete_pose=next_discrete_state,
                        continuous_pose=next_continuous_state,
                        trajectory_segment=world_trajectory,
                        parent_node=current_node,
                        primitive_used=primitive,
                        g_score=tentative_g,
                        f_score=f_score,
                    )

                    # Store node and add to open set
                    self.search_nodes[next_discrete_state] = next_node
                    heapq.heappush(open_set, (f_score, id(next_node), next_node))

        print("No path found within timeout")
        return None

    def _reconstruct_connectivity_preserving_path(
        self, goal_node: SearchNode
    ) -> List[DiscreteMotionPrimitive]:
        """
        Reconstruct the sequence of motion primitives with guaranteed connectivity
        """
        print("=== RECONSTRUCTING CONNECTIVITY-PRESERVING PATH ===")

        sequence = []
        trajectory_segments = []
        current_node = goal_node
        path_nodes = []

        # Trace back to start
        while current_node.parent_node is not None:
            path_nodes.append(current_node)
            current_node = current_node.parent_node

        path_nodes.reverse()

        print(f"Path has {len(path_nodes)} segments")

        # Build sequence and verify connectivity
        for i, node in enumerate(path_nodes):
            if node.primitive_used is not None:
                sequence.append(node.primitive_used)
                trajectory_segments.append(node.trajectory_segment)

                print(
                    f"Segment {i+1}: {node.primitive_used.steering_command.value} "
                    f"-> exact pose ({node.continuous_pose.x:.3f}, {node.continuous_pose.y:.3f}, "
                    f"{node.continuous_pose.theta:.3f})"
                )

        # Verify complete connectivity
        self._verify_path_connectivity(trajectory_segments)

        return sequence

    def _verify_path_connectivity(
        self, trajectory_segments: List[List[Tuple[float, float, float]]]
    ):
        """
        Verify that the path maintains complete connectivity between segments
        """
        print("=== VERIFYING PATH CONNECTIVITY ===")

        if not trajectory_segments:
            print("No trajectory segments to verify")
            return

        total_disconnections = 0
        max_disconnect_distance = 0.0

        for i in range(len(trajectory_segments) - 1):
            # End of current segment
            current_end = trajectory_segments[i][-1]
            # Start of next segment
            next_start = trajectory_segments[i + 1][0]

            # Calculate discontinuity
            dx = next_start[0] - current_end[0]
            dy = next_start[1] - current_end[1]
            dtheta = next_start[2] - current_end[2]

            distance = np.sqrt(dx**2 + dy**2)

            if distance > 1e-6:  # Tolerance for floating point precision
                total_disconnections += 1
                max_disconnect_distance = max(max_disconnect_distance, distance)
                print(
                    f"DISCONNECTION at segment {i}->{i+1}: "
                    f"distance={distance:.6f}m, dtheta={dtheta:.6f}rad"
                )
            else:
                print(f"Segment {i}->{i+1}: CONNECTED (distance={distance:.8f}m)")

        if total_disconnections == 0:
            print("✓ PERFECT CONNECTIVITY: All segments perfectly connected!")
            self.connectivity_verified = True
        else:
            print(
                f"✗ CONNECTIVITY ISSUES: {total_disconnections} disconnections, "
                f"max distance: {max_disconnect_distance:.6f}m"
            )
            self.connectivity_verified = False

    def _discretize_state(self, x: float, y: float, theta: float) -> DiscreteState:
        """Convert continuous state to discrete indices"""
        x_idx: int = math.floor(x / self.resolution)
        y_idx: int = math.floor(y / self.resolution)

        # Normalize theta to [0, 2π)
        theta_norm = (theta + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-π, π)
        theta_norm = (theta_norm + 2 * np.pi) % (2 * np.pi)  # Then to [0, 2π)

        # convert to index
        theta_idx: int = (
            math.floor(theta_norm / self.angle_resolution) % self.num_angles
        )

        return x_idx, y_idx, theta_idx

    def _continuous_state(
        self, x_idx: int, y_idx: int, theta_idx: int
    ) -> tuple[float, float, float]:
        """Convert discrete indices to continuous state"""
        x = x_idx * self.resolution
        y = y_idx * self.resolution
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
            x_local, y_local, theta_local = primitive.trajectory[idx]

            # Transform to world coordinates
            x_world, y_world, _ = apply_motion_primitive(
                (start_x, start_y, start_theta),
                (x_local, y_local, theta_local),  # No change in theta
            )

            # Check bounds with total margin
            if (
                x_world < total_margin
                or x_world >= (self.grid.width - total_margin)
                or y_world < total_margin
                or y_world >= (self.grid.height - total_margin)
            ):
                return False

            # Convert to grid coordinates
            grid_x, grid_y = self.grid._discretize_state(x_world, y_world)
            margin_cells = int(total_margin / self.grid.resolution)

            # Check a rectangular region around the point
            check_x = int(grid_x)
            check_y = int(grid_y)

            if (
                check_x < 0
                or check_x >= self.grid.grid_width
                or check_y < 0
                or check_y >= self.grid.grid_height
            ):
                continue

            if self.grid.binary_grid[check_y, check_x]:
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
        self,
        current_node: SearchNode,
    ) -> List[DiscreteMotionPrimitive]:
        """Reconstruct the sequence of motion primitives"""
        sequence = []
        current = current_node.discrete_pose

        # Debug info
        print(f"Reconstructing path from state {current}")

        while current_node.parent_node is not None:
            parent_node = current_node.parent_node
            primitive = parent_node.primitive_used
            print(f"Current state: {parent_node.discrete_pose} ")

            if primitive is not None:
                sequence.append(primitive)
                # print(f"Added primitive: {primitive.steering_command.value}")
            # else:
            # print("Warning: No primitive for this state transition")

            if parent_node.discrete_pose == current:
                print("Warning: Parent state equals current state")
                break

            current = parent_node.discrete_pose

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
            steering_angle = self.steering_angles[primitive.steering_command]

            # Set velocities
            car.set_control_inputs(self.linear_velocity, steering_angle)

            # In real implementation, would execute for primitive.duration seconds
            # For simulation, update for multiple timesteps
            num_steps = int(primitive.duration / 0.05)  # 0.05s simulation timestep
            for _ in range(num_steps):
                car.update_state(0.05)

    def visualize_command_sequence(
        self,
        start_x: float,
        start_y: float,
        start_theta: float,
        ax=None,
    ):
        """Visualize the planned command sequence"""
        # Colors for different commands
        colors = {
            SteeringCommand.LEFT: "red",
            SteeringCommand.NEUTRAL: "green",
            SteeringCommand.RIGHT: "blue",
        }

        labels = {
            SteeringCommand.LEFT: "Left",
            SteeringCommand.NEUTRAL: "Neutral",
            SteeringCommand.RIGHT: "Right",
        }

        # Draw the path
        current_x, current_y, current_theta = start_x, start_y, start_theta

        # Add start marker
        if ax is not None:
            ax.plot(
                start_x,
                start_y,
                "ro",
                markersize=8,
                label="Start",
            )
        else:
            plt.plot(
                start_x,
                start_y,
                "ro",
                markersize=8,
                label="Start",
            )

        # Store all trajectory points for continuous line
        all_trajectory_x = []
        all_trajectory_y = []

        for primitive in self.command_sequence:
            # Get trajectory in world coordinates
            for x_local, y_local, theta_local in primitive.trajectory:
                # Transform from primitive's local frame to world frame
                x_world, y_world, _ = apply_motion_primitive(
                    (current_x, current_y, current_theta),
                    (x_local, y_local, theta_local),  # No change in theta
                )

                all_trajectory_x.append(x_world)
                all_trajectory_y.append(y_world)

            # Update current position
            current_x, current_y, current_theta = apply_motion_primitive(
                (current_x, current_y, current_theta), primitive.end_displacement
            )

        # Plot the continuous trajectory line
        if all_trajectory_x and all_trajectory_y:
            if ax is not None:
                ax.plot(
                    all_trajectory_x,
                    all_trajectory_y,
                    color="green",
                    linewidth=3,
                    label="Trajectory",
                )
            else:
                plt.plot(
                    all_trajectory_x,
                    all_trajectory_y,
                    color="green",
                    linewidth=3,
                    label="Trajectory",
                )

        # Add goal marker
        if ax is not None:
            marker_transform = self.goal_tolerance * self.resolution * 2

            # Approximate the goal zone with a circle
            circle = plt.Circle(
                (self.goal_pose[0], self.goal_pose[1]),
                radius=marker_transform,
                color="blue",
                fill=True,
                alpha=0.3,
                label="Goal Area",
            )

            if hasattr(self, "goal_pose"):
                ax.add_patch(circle)

            # Add end marker at the actual final position
            ax.plot(
                current_x,
                current_y,
                "rx",
                markersize=8,
                label="End",
            )
        else:
            if hasattr(self, "goal_pose"):
                plt.add_patch(circle)

        # Draw occupancy grid
        if hasattr(self.grid, "binary_grid"):
            if ax is not None:
                ax.imshow(
                    self.grid.binary_grid.T,
                    origin="lower",
                    extent=[0, self.grid.width, 0, self.grid.height],
                    cmap="gray_r",
                    alpha=0.3,
                )
            else:
                plt.imshow(
                    self.grid.binary_grid.T,
                    origin="lower",
                    extent=[0, self.grid.width, 0, self.grid.height],
                    cmap="gray_r",
                    alpha=0.3,
                )

        if ax is not None:
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title("Discrete Command Sequence Visualization")
            ax.legend()
            ax.grid(True)
            ax.axis("equal")
        else:
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.title("Discrete Command Sequence Visualization")
            plt.legend()
            plt.grid(True)
            plt.axis("equal")
            # plt.show() # Remove plt.show()

    def visualize_explored_states(self, ax):
        """Visualize the states explored by the planner."""
        if not self.explored_states:
            print("No states were explored to visualize.")
            return

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
        explored_x = [s.x for s in self.explored_states]
        explored_y = [s.y for s in self.explored_states]
        ax.plot(
            explored_x,
            explored_y,
            "o",
            color="purple",
            markersize=2,
            alpha=0.2,
            label="Explored States",
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Explored States Visualization")
        ax.legend()
        ax.grid(True)
        ax.axis("equal")


def apply_motion_primitive(
    current_pose: ContinuousState,
    motion_primitive: ContinuousState,
) -> ContinuousState:
    if isinstance(current_pose, tuple):
        x, y, theta = current_pose
    else:
        x, y, theta = current_pose.x, current_pose.y, current_pose.theta

    if isinstance(motion_primitive, tuple):
        dx, dy, dtheta = motion_primitive
    else:
        dx, dy, dtheta = motion_primitive.x, motion_primitive.y, motion_primitive.theta

    # Transform motion from vehicle frame to world frame
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    # Vehicle frame to world frame transformation matrix
    world_dx = dx * cos_theta - dy * sin_theta
    world_dy = dx * sin_theta + dy * cos_theta

    # Apply motion in world coordinates
    new_x = x + world_dx
    new_y = y + world_dy
    new_theta = theta + dtheta

    # Normalize theta
    new_theta = math.atan2(math.sin(new_theta), math.cos(new_theta))

    return (new_x, new_y, new_theta)


# Integration with your existing system
def phase2_discrete_planning(
    occupancy_grid,
    start_pose,
    goal_pose,
    ax=None,
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
        start_pose.x,
        start_pose.y,
        start_pose.theta,
        goal_pose.x,
        goal_pose.y,
        goal_pose.theta,
    )

    planner.command_sequence = (
        command_sequence  # Store command sequence as an attribute
    )

    # Store goal pose for visualization
    planner.goal_pose = (goal_pose.x, goal_pose.y, goal_pose.theta)

    if planner.command_sequence:
        print(f"Found path with {len(planner.command_sequence)} commands")
    else:
        print("No path found!")
        return None

    # Visualize the planned sequence
    if ax is not None:
        planner.visualize_command_sequence(
            start_pose.x,
            start_pose.y,
            start_pose.theta,
            ax,
        )

    return planner

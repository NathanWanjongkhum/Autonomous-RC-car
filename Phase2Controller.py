import numpy as np
from typing import List, Tuple, Optional, Dict
from enum import Enum
from dataclasses import dataclass

# Import your existing controllers
from pathfinding_algorithms.ConstantPPC import ConstantPurePursuitController
from path_optimizing_algorithms.LatticeMotionPlanner import (
    DiscreteLatticeMotionPlanner,
    DiscreteMotionPrimitive,
    SteeringCommand,
)


class ControlMode(Enum):
    """Different control modes for Phase 2 execution"""

    LATTICE_FOLLOWING = "lattice_following"  # Following discrete command sequence
    PURE_PURSUIT = "pure_pursuit"  # Pure pursuit path tracking
    ERROR_CORRECTION = "error_correction"  # Correcting deviations
    REPLANNING = "replanning"  # Re-planning locally


@dataclass
class ControlState:
    """Current state of the integrated controller"""

    mode: ControlMode
    current_primitive_index: int
    path_deviation: float
    last_replan_time: float
    emergency_stop: bool = False


class Phase2Controller:
    """
    Integrated controller that combines LatticeMotionPlanner and PurePursuitController
    for Phase 2 execution with real-time error correction.

    Architecture:
    1. High-level: Discrete command sequence from LatticeMotionPlanner
    2. Mid-level: Trajectory generation and deviation monitoring
    3. Low-level: PurePursuitController for smooth tracking and corrections
    """

    def __init__(
        self,
        lattice_planner: DiscreteLatticeMotionPlanner,
        pure_pursuit: ConstantPurePursuitController,
        max_path_deviation: float = 0.3,  # Max allowed deviation before correction
        replan_interval: float = 2.0,  # Minimum time between replanning
        emergency_deviation: float = 0.8,  # Deviation threshold for emergency stop
    ):
        """
        Initialize the integrated controller

        Parameters:
        lattice_planner: Pre-configured lattice motion planner
        pure_pursuit: Pre-configured pure pursuit controller
        max_path_deviation: Maximum allowed deviation from planned path (meters)
        replan_interval: Minimum time between local replanning (seconds)
        emergency_deviation: Deviation threshold for emergency procedures (meters)
        """
        self.lattice_planner = lattice_planner
        self.pure_pursuit = pure_pursuit
        self.max_path_deviation = max_path_deviation
        self.replan_interval = replan_interval
        self.emergency_deviation = emergency_deviation

        # Control state
        self.state = ControlState(
            mode=ControlMode.LATTICE_FOLLOWING,
            current_primitive_index=0,
            path_deviation=0.0,
            last_replan_time=0.0,
        )

        # Planned trajectory data
        self.command_sequence: List[DiscreteMotionPrimitive] = []
        self.reference_trajectory: List[Tuple[float, float]] = []
        self.trajectory_timestamps: List[float] = []

        # Execution tracking
        self.execution_start_time: float = 0.0
        self.current_target_primitive: Optional[DiscreteMotionPrimitive] = None

        # Performance monitoring
        self.deviation_history: List[float] = []
        self.control_mode_history: List[ControlMode] = []

    def set_planned_trajectory(
        self,
        command_sequence: List[DiscreteMotionPrimitive],
        start_pose: Tuple[float, float, float],
    ):
        """
        Set the planned trajectory from the lattice planner

        Parameters:
        command_sequence: Sequence of discrete motion primitives
        start_pose: Starting (x, y, theta) pose
        """
        self.command_sequence = command_sequence
        self.state.current_primitive_index = 0

        # Convert discrete command sequence to continuous reference trajectory
        self.reference_trajectory = self._extract_reference_trajectory(
            command_sequence, start_pose
        )

        # Set the reference trajectory for pure pursuit
        self.pure_pursuit.set_path(self.reference_trajectory)

        print(f"Set planned trajectory with {len(command_sequence)} primitives")
        print(f"Generated reference path with {len(self.reference_trajectory)} points")

    def _extract_reference_trajectory(
        self,
        command_sequence: List[DiscreteMotionPrimitive],
        start_pose: Tuple[float, float, float],
    ) -> List[Tuple[float, float]]:
        """
        Extract a continuous reference trajectory from discrete command sequence

        This creates a smooth path that the pure pursuit controller can follow
        while respecting the high-level plan from the lattice planner.
        """
        trajectory = []
        current_x, current_y, current_theta = start_pose

        # Add starting point
        trajectory.append((current_x, current_y))

        for primitive in command_sequence:
            # Extract points along each primitive's trajectory
            for x_local, y_local, theta_local in primitive.trajectory:
                # Transform from primitive's local frame to world frame
                cos_theta = np.cos(current_theta)
                sin_theta = np.sin(current_theta)

                # Apply transformation
                x_world = current_x + x_local * cos_theta - y_local * sin_theta
                y_world = current_y + x_local * sin_theta + y_local * cos_theta

                trajectory.append((x_world, y_world))

            # Update current pose using primitive's end displacement
            dx, dy, dtheta = primitive.end_displacement
            current_x += dx * np.cos(current_theta) - dy * np.sin(current_theta)
            current_y += dx * np.sin(current_theta) + dy * np.cos(current_theta)
            current_theta += dtheta

        return trajectory

    def compute_steering(
        self,
        current_pose: Tuple[float, float, float],
        current_time: float,
        dt: float = 0.1,
    ) -> Tuple[float, float]:
        """
        Main control loop - computes steering and velocity commands

        Parameters:
        current_pose: Current (x, y, theta) of the vehicle
        current_time: Current system time
        dt: Control timestep

        Returns:
        (steering_angle, linear_velocity): Control commands
        """
        x, y, theta = current_pose

        # Initialize execution timer
        if self.execution_start_time == 0:
            self.execution_start_time = current_time

        # Calculate current deviation from reference trajectory
        self.state.path_deviation = self._calculate_path_deviation(x, y)
        self.deviation_history.append(self.state.path_deviation)

        # Determine control mode based on current state
        self._update_control_mode(current_time)

        # Execute control based on current mode
        if self.state.mode == ControlMode.LATTICE_FOLLOWING:
            return self._execute_lattice_following(current_pose, current_time, dt)

        elif self.state.mode == ControlMode.PURE_PURSUIT:
            return self._execute_pure_pursuit(current_pose, current_time, dt)

        elif self.state.mode == ControlMode.ERROR_CORRECTION:
            return self._execute_error_correction(current_pose, current_time, dt)

        elif self.state.mode == ControlMode.REPLANNING:
            return self._execute_replanning(current_pose, current_time, dt)
        else:
            # Emergency stop
            print("Emergency stop triggered!")
            return 0.0, 0.0

    def _calculate_path_deviation(self, x: float, y: float) -> float:
        """Calculate perpendicular distance to reference trajectory"""
        if not self.reference_trajectory:
            return 0.0

        min_distance = float("inf")
        for traj_x, traj_y in self.reference_trajectory:
            distance = np.sqrt((x - traj_x) ** 2 + (y - traj_y) ** 2)
            min_distance = min(min_distance, distance)

        return min_distance

    def _update_control_mode(self, current_time: float):
        """Update control mode based on current conditions"""
        # Check for emergency conditions
        if self.state.path_deviation > self.emergency_deviation:
            self.state.mode = ControlMode.REPLANNING
            self.state.emergency_stop = True
            return

        # Check if we need error correction
        if self.state.path_deviation > self.max_path_deviation:
            if self.state.mode != ControlMode.ERROR_CORRECTION:
                print(
                    f"Switching to error correction mode (deviation: {self.state.path_deviation:.3f}m)"
                )
            self.state.mode = ControlMode.ERROR_CORRECTION
            return

        # Check if we should replan
        if (
            current_time - self.state.last_replan_time > self.replan_interval
            and self.state.path_deviation > self.max_path_deviation * 0.7
        ):
            self.state.mode = ControlMode.REPLANNING
            return

        # Default to appropriate tracking mode
        if self.state.path_deviation < self.max_path_deviation * 0.5:
            # Low deviation - can follow planned trajectory closely
            self.state.mode = ControlMode.LATTICE_FOLLOWING
        else:
            # Moderate deviation - use pure pursuit for smoother tracking
            self.state.mode = ControlMode.PURE_PURSUIT

        # Store mode history
        self.control_mode_history.append(self.state.mode)

    def _execute_lattice_following(
        self, current_pose: Tuple[float, float, float], current_time: float, dt: float
    ) -> Tuple[float, float]:
        """
        Execute the planned discrete command sequence

        When deviation is low, we can follow the optimal discrete commands
        from the lattice planner more closely.
        """
        # Get current primitive based on execution time
        execution_time = current_time - self.execution_start_time

        # Find which primitive we should be executing
        cumulative_time = 0.0
        current_primitive = None

        for i, primitive in enumerate(self.command_sequence):
            if execution_time < cumulative_time + primitive.duration:
                current_primitive = primitive
                self.state.current_primitive_index = i
                break
            cumulative_time += primitive.duration

        if current_primitive is None:
            # Reached end of command sequence
            print("Reached end of command sequence!")
            return 0.0, 0.0

        # Execute the discrete command with some pure pursuit smoothing
        discrete_steering = self.lattice_planner.steering_angles[
            current_primitive.steering_command
        ]
        discrete_velocity = self.lattice_planner.linear_velocity

        # Apply minor corrections using pure pursuit
        pp_steering, pp_velocity = self.pure_pursuit.compute_steering(
            current_pose[0],
            current_pose[1],
            current_pose[2],
            self.lattice_planner.wheelbase,
            dt,
        )

        # Blend discrete command with pure pursuit correction
        # Higher weight on discrete command when deviation is low
        blend_factor = max(
            0.1, 1.0 - (self.state.path_deviation / self.max_path_deviation)
        )

        final_steering = (
            blend_factor * discrete_steering + (1 - blend_factor) * pp_steering
        )
        final_velocity = discrete_velocity  # Keep planned velocity

        return final_steering, final_velocity

    def _execute_pure_pursuit(
        self, current_pose: Tuple[float, float, float], current_time: float, dt: float
    ) -> Tuple[float, float]:
        """
        Execute pure pursuit path following

        When deviation is moderate, rely more heavily on pure pursuit
        for smooth trajectory tracking.
        """
        return self.pure_pursuit.compute_steering(
            current_pose[0],
            current_pose[1],
            current_pose[2],
            self.lattice_planner.wheelbase,
            dt,
        )

    def _execute_error_correction(
        self, current_pose: Tuple[float, float, float], current_time: float, dt: float
    ) -> Tuple[float, float]:
        """
        Execute error correction maneuvers

        When deviation is high, prioritize getting back to the reference trajectory.
        """
        # Use pure pursuit with increased aggressiveness
        steering, velocity = self.pure_pursuit.compute_steering(
            current_pose[0],
            current_pose[1],
            current_pose[2],
            self.lattice_planner.wheelbase,
            dt,
        )

        # Reduce velocity for better control during correction
        correction_velocity = velocity * 0.7

        # Increase steering response for faster correction
        correction_steering = steering * 1.2

        # Limit steering to physical constraints
        max_steering = self.lattice_planner.steering_angles[SteeringCommand.LEFT]
        correction_steering = np.clip(correction_steering, -max_steering, max_steering)

        return correction_steering, correction_velocity

    def _execute_replanning(
        self, current_pose: Tuple[float, float, float], current_time: float, dt: float
    ) -> Tuple[float, float]:
        """
        Execute local replanning when large deviations occur

        This could trigger a local replan or emergency maneuver.
        """
        self.state.last_replan_time = current_time

        if self.state.emergency_stop:
            # Emergency stop procedure
            print("Emergency stop triggered due to large deviation!")
            return 0.0, 0.0

        # For now, fall back to aggressive pure pursuit
        # In a full implementation, this would trigger local replanning
        steering, velocity = self._execute_error_correction(
            current_pose, current_time, dt
        )

        # Transition back to normal modes once deviation decreases
        if self.state.path_deviation < self.max_path_deviation:
            self.state.mode = ControlMode.PURE_PURSUIT
            self.state.emergency_stop = False

        return steering, velocity

    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for analysis"""
        if not self.deviation_history:
            return {}

        return {
            "mean_deviation": np.mean(self.deviation_history),
            "max_deviation": np.max(self.deviation_history),
            "deviation_std": np.std(self.deviation_history),
            "current_mode": self.state.mode.value,
            "primitive_progress": (
                self.state.current_primitive_index / max(1, len(self.command_sequence))
            ),
            "emergency_stops": sum(
                1
                for mode in self.control_mode_history
                if mode == ControlMode.REPLANNING
            ),
        }

    def visualize_performance(self):
        """Visualize controller performance"""
        import matplotlib.pyplot as plt

        if not self.deviation_history:
            print("No performance data to visualize")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot deviation over time
        ax1.plot(self.deviation_history, "b-", linewidth=2)
        ax1.axhline(
            y=self.max_path_deviation,
            color="r",
            linestyle="--",
            label=f"Max Deviation ({self.max_path_deviation}m)",
        )
        ax1.axhline(
            y=self.emergency_deviation,
            color="r",
            linestyle="-",
            label=f"Emergency Threshold ({self.emergency_deviation}m)",
        )
        ax1.set_title("Path Deviation Over Time")
        ax1.set_ylabel("Deviation (m)")
        ax1.legend()
        ax1.grid(True)

        # Plot control modes over time
        mode_values = [mode.value for mode in self.control_mode_history]
        unique_modes = list(set(mode_values))
        mode_numbers = [unique_modes.index(mode) for mode in mode_values]

        ax2.plot(mode_numbers, "g-", linewidth=2)
        ax2.set_title("Control Mode Over Time")
        ax2.set_ylabel("Control Mode")
        ax2.set_xlabel("Time Step")
        ax2.set_yticks(range(len(unique_modes)))
        ax2.set_yticklabels(unique_modes)
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


# Example usage:
"""
# Phase 2 setup
lattice_planner = DiscreteLatticeMotionPlanner(occupancy_grid, ...)
pure_pursuit = ConstantPurePursuitController(...)

# Create integrated controller
controller = IntegratedPhase2Controller(
    lattice_planner=lattice_planner,
    pure_pursuit=pure_pursuit,
    max_path_deviation=0.3,
    replan_interval=2.0
)

# Set planned trajectory from offline planning
controller.set_planned_trajectory(command_sequence, start_pose)

# Main execution loop
while not finished:
    current_pose = get_car_pose()
    current_time = time.time()
    
    steering, velocity = controller.compute_steering(current_pose, current_time)
    
    car.set_control_inputs(velocity, steering)
    car.update_state(dt)
    
    # Monitor performance
    metrics = controller.get_performance_metrics()
    if metrics['max_deviation'] > emergency_threshold:
        # Handle emergency...
        pass
"""

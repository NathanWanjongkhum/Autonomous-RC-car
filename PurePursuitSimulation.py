import numpy as np

import matplotlib

# matplotlib.use("Agg")  # Use a non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

mplstyle.use("fast")
mplstyle.use(["dark_background", "fast"])

import time
from typing import List, Tuple, Dict, Optional, Any

from DrawableElements import *


class SimulationState:
    """
    Encapsulates and manages the state history of the simulation.

    This class efficiently tracks the complete history of simulation states,
    providing structured access to position, velocity, steering, and other
    data for analysis, debugging, and visualization.
    """

    def __init__(self):
        # Core state history
        self.positions: List[Tuple[float, float]] = []  # (x, y) positions
        self.headings: List[float] = []  # Orientation (theta)
        self.velocities: List[float] = []  # Linear velocities
        self.steering_angles: List[float] = []  # Steering angles

        # Controller-specific state
        self.target_points: List[Optional[Tuple[float, float]]] = (
            []
        )  # Target points from controller
        self.lookahead_distances: List[float] = []  # Lookahead distances used

        # Timing information
        self.timestamps: List[float] = []  # Time at each state
        self.step_durations: List[float] = []  # Duration of each simulation step

        # Additional debug information
        self.control_errors: List[float] = []  # Control errors at each step

        # Path information
        self.path: List[Tuple[float, float]] = []  # Reference path

    def record_state(
        self, car, controller, timestamp: float, step_duration: Optional[float] = None
    ):
        """
        Record current state of the simulation

        Parameters:
        car: The car model object with current state
        controller: The controller object
        timestamp: Current simulation time
        step_duration: Duration of the current simulation step (if available)
        """
        self.positions.append((car.x, car.y))
        self.headings.append(car.theta)
        self.velocities.append(car.v)
        self.steering_angles.append(car.steering_angle)

        # Get target point if available
        target = controller.get_target_point(car.x, car.y)
        if target and len(target) >= 2:
            self.target_points.append((target[0], target[1]))
        else:
            self.target_points.append(None)

        # Timing
        self.timestamps.append(timestamp)
        if step_duration is not None:
            self.step_durations.append(step_duration)

        # Optional: additional controller metrics
        if hasattr(controller, "get_steering_metrics"):
            metrics = controller.get_steering_metrics()
            if "heading_error" in metrics:
                self.control_errors.append(metrics["heading_error"])

    def set_path(self, path: List[Tuple[float, float]]):
        """Set the reference path"""
        self.path = path.copy()

    def clear(self):
        """Clear all state history"""
        self.positions.clear()
        self.headings.clear()
        self.velocities.clear()
        self.steering_angles.clear()
        self.target_points.clear()
        self.lookahead_distances.clear()
        self.timestamps.clear()
        self.step_durations.clear()
        self.control_errors.clear()
        # Note: We don't clear the path as it's a reference

    def get_metrics(self) -> Dict[str, Any]:
        """
        Calculate and return performance metrics

        Returns:
        dict: Dictionary of performance metrics
        """
        metrics = {}

        if not self.positions or not self.path:
            return metrics

        # Calculate path following error
        if self.path and len(self.positions) > 0:
            # Mean distance to closest path point
            path_errors = []
            for pos in self.positions:
                min_dist = float("inf")
                for path_point in self.path:
                    dist = np.hypot(pos[0] - path_point[0], pos[1] - path_point[1])
                    min_dist = min(min_dist, dist)
                path_errors.append(min_dist)

            metrics["mean_path_error"] = np.mean(path_errors)
            metrics["max_path_error"] = np.max(path_errors)
            metrics["path_error_std"] = np.std(path_errors)

        # Time metrics
        if self.timestamps:
            metrics["total_time"] = self.timestamps[-1] - self.timestamps[0]
            metrics["steps"] = len(self.timestamps)

        # Velocity metrics
        if self.velocities:
            metrics["mean_velocity"] = np.mean(self.velocities)
            metrics["max_velocity"] = np.max(self.velocities)

        # Steering metrics
        if self.steering_angles:
            metrics["mean_steering"] = np.mean(np.abs(self.steering_angles))
            metrics["max_steering"] = np.max(np.abs(self.steering_angles))

        return metrics

    def get_state_at_index(self, index: int) -> Dict[str, Any]:
        """
        Get complete state at a specific index

        Parameters:
        index: Index of the state to retrieve

        Returns:
        dict: Complete state information at the requested index
        """
        if index < 0 or index >= len(self.positions):
            return {}

        state = {
            "position": self.positions[index],
            "heading": self.headings[index],
            "velocity": self.velocities[index],
            "steering_angle": self.steering_angles[index],
            "timestamp": self.timestamps[index],
        }

        if index < len(self.target_points) and self.target_points[index]:
            state["target_point"] = self.target_points[index]

        if index < len(self.control_errors):
            state["control_error"] = self.control_errors[index]

        return state


class SimulationConfig:
    """
    Configuration for the Pure Pursuit Simulation

    This class encapsulates all configurable parameters for the simulation,
    allowing for easy adjustment and customization.
    """

    def __init__(
        self,
        dt: float = 0.1,  # Simulation time step (seconds)
        max_steps: int = 1000,  # Maximum simulation steps
        debug_mode: bool = False,  # Enable debug mode
        record_history: bool = True,  # Record state history
        visualize: bool = True,  # Enable visualization
        real_time_factor: float = 1.0,  # Real-time factor (1.0 = real-time)
    ):
        self.dt = dt
        self.max_steps = max_steps
        self.debug_mode = debug_mode
        self.record_history = record_history
        self.visualize = visualize
        self.real_time_factor = real_time_factor

        # Visualization configuration
        self.visualization_config = {
            "show_car_body": True,
            "show_wheels": True,
            "show_steering_lines": True,
            "show_path": True,
            "show_trajectory": True,
            "show_target_points": True,
            "show_occupancy_grid": True,
            "figsize": (10, 8),
        }

        # Stopping criteria
        self.stopping_criteria = {
            "goal_distance": 0.2,  # Distance to goal to consider reached
            "timeout": 60.0,  # Simulation timeout in seconds
            "max_path_error": 2.0,  # Maximum allowable path error
        }


class PurePursuitSimulation:
    """
    Enhanced simulation for testing the Pure Pursuit Controllers

    This refactored class provides improved state tracking, visualization,
    and debugging capabilities for Pure Pursuit controller testing.

    Assumes:
        - We have found a path to follow
        - We have defined a car model
        - We have defined a pure pursuit controller
    """

    def __init__(self, car, controller, config: Optional[SimulationConfig] = None):
        """
        Initialize the simulation

        Parameters:
        car: The car model object
        controller: The controller object
        config: Optional configuration object (default: creates a default config)
        """
        self.car = car
        self.controller = controller
        self.config = config or SimulationConfig()

        # State management
        self.state_history = SimulationState()
        self.current_time = 0.0
        self.step_count = 0
        self.simulation_complete = False
        self.completion_reason = ""

        # Path and mapping
        self.path = []

        # Visualization
        self.viz = None
        if self.config.visualize:
            self.setup_visualization()

    def setup_visualization(self):
        """
        Initialize and configure the visualization system
        """
        # Create visualization manager
        self.viz = VisualizationManager(
            self.car, figsize=self.config.visualization_config["figsize"]
        )

        # Configure visualization elements based on config
        self.viz.enable_element(
            "car_body", self.config.visualization_config["show_car_body"]
        )
        self.viz.enable_element(
            "wheels", self.config.visualization_config["show_wheels"]
        )
        self.viz.enable_element(
            "steering_lines", self.config.visualization_config["show_steering_lines"]
        )

        # Add custom elements
        self.viz.add_element(
            "goal_marker",
            MarkerDrawer(0, 0, color="red", marker="x", markersize=10, label="Goal"),
        )

        # Add multiple path types
        self.viz.add_element(
            "planned_path",
            PathDrawer(color="green", linestyle="-", linewidth=3, label="Planned Path"),
        )

        self.viz.add_element(
            "actual_path",
            PathDrawer(color="orange", linestyle=":", linewidth=2, label="Actual Path"),
        )

    def initialize_visualization(self):
        """
        Initialize the visualization system and return figure and axis objects

        Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
        """
        if not self.viz:
            return None, None

        self.viz.initialize_all()
        self.viz.show_legend()
        return self.viz.fig, self.viz.ax

    def set_reference_path(self, path: List[Tuple[float, float]]):
        """
        Set the reference path for the simulation and controller

        Parameters:
        path: List of (x, y) points defining the path
        """
        self.path = path.copy()
        self.controller.set_path(path)
        self.state_history.set_path(path)

        if self.viz:
            self.viz.get_element("reference_path").set_path(path)

            # Set goal marker at the end of the path
            if path:
                self.set_goal_position(path[-1][0], path[-1][1])

    def set_goal_position(self, x: float, y: float):
        """
        Set the goal marker position

        Parameters:
        x, y: Coordinates of the goal
        """
        if self.viz:
            goal_marker = self.viz.get_element("goal_marker")
            goal_marker.set_position(x, y)
            goal_marker.update(self.viz.ax)

    def update_trajectory(self, x: float, y: float):
        """
        Add a new position to the trajectory visualization

        Parameters:
        x, y: New position to add
        """
        if self.viz:
            self.viz.get_element("trajectory").add_position(x, y)

    def reset_simulation(self):
        """
        Reset the simulation to initial state
        """
        # Reset simulation state
        self.current_time = 0.0
        self.step_count = 0
        self.simulation_complete = False
        self.completion_reason = ""

        # Clear state history
        self.state_history.clear()

        # Reset car to start of path if path exists
        if self.path:
            self.car.x = self.path[0][0]
            self.car.y = self.path[0][1]
            self.car.theta = 0.0  # Initial heading (can be customized)
            self.car.v = 0.0  # Initial velocity

        # Reset visualization
        if self.viz:
            # Clear trajectory
            self.viz.get_element("trajectory").set_positions([])

            # Reset other visualization elements as needed
            self.viz.update_all()

    def run_simulation_step(self) -> bool:
        """
        Run a single simulation step

        Returns:
        bool: True if simulation is complete, False otherwise
        """
        # Record start time if in debug mode
        step_start_time = time.time() if self.config.debug_mode else None

        # Get control commands from controller
        steering_angle, linear_velocity = self.controller.compute_steering(
            x=self.car.x,
            y=self.car.y,
            theta=self.car.theta,
            wheelbase=self.car.wheelbase,
            dt=self.config.dt,
        )

        # Set car controls
        self.car.set_control_inputs(linear_velocity, steering_angle)

        # Update car state
        self.car.update_state(self.config.dt)

        # Update simulation time
        self.current_time += self.config.dt
        self.step_count += 1

        # Record state if enabled
        if self.config.record_history:
            step_duration = time.time() - step_start_time if step_start_time else None
            self.state_history.record_state(
                self.car, self.controller, self.current_time, step_duration
            )

        # Update trajectory visualization
        self.update_trajectory(self.car.x, self.car.y)

        # Update visualization if enabled
        if self.viz:
            self.viz.update_all()

        # Check stopping criteria
        if self.path:
            # Check if we've reached the end of the path
            distance_to_goal = np.hypot(
                self.car.x - self.path[-1][0], self.car.y - self.path[-1][1]
            )

            if distance_to_goal < self.config.stopping_criteria["goal_distance"]:
                self.simulation_complete = True
                self.completion_reason = f"Goal reached after {self.step_count} steps, time {self.current_time:.2f}s"
                if self.config.debug_mode:
                    print(self.completion_reason)
                return True

        # Check timeout
        if self.current_time > self.config.stopping_criteria["timeout"]:
            self.simulation_complete = True
            self.completion_reason = f"Simulation timeout after {self.step_count} steps, time {self.current_time:.2f}s"
            if self.config.debug_mode:
                print(self.completion_reason)
            return True

        # Check if we've exceeded the maximum number of steps
        if self.step_count >= self.config.max_steps:
            self.simulation_complete = True
            self.completion_reason = f"Maximum steps ({self.config.max_steps}) reached, time {self.current_time:.2f}s"
            if self.config.debug_mode:
                print(self.completion_reason)
            return True

        return False

    def run_simulation(
        self,
        path: Optional[List[Tuple[float, float]]] = None,
        num_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete simulation

        Parameters:
        path: Optional path to follow (if None, uses previously set path)
        num_steps: Optional maximum number of steps (overrides config.max_steps)

        Returns:
        dict: Simulation results
        """

        # Set path if provided
        if path:
            path = path.copy()
            self.set_reference_path(path)

            self.car.x = path[0][0]
            self.car.y = path[0][1]

        # Use provided num_steps if specified, otherwise use config
        if num_steps is not None:
            max_steps = num_steps
        else:
            max_steps = self.config.max_steps

        if self.config.visualize:
            # Reset simulation
            self.reset_simulation()

            # Initialize visualization
            self.initialize_visualization()

        # Run simulation steps until completion or max steps
        for _ in range(max_steps):
            # Run a single step
            completed = self.run_simulation_step()

            # Optional pause for visualization
            if self.viz and self.config.real_time_factor > 0:
                # Minimum time
                plt.pause(0.01)

            # Stop if simulation is complete
            if completed:
                print("Successful Completion")
                break

        # Return simulation results
        return self.get_results()

    def get_results(self) -> Dict[str, Any]:
        """
        Get simulation results

        Returns:
        dict: Simulation results including state history and metrics
        """
        results = {
            "positions": self.state_history.positions,
            "headings": self.state_history.headings,
            "velocities": self.state_history.velocities,
            "steering_angles": self.state_history.steering_angles,
            "target_points": self.state_history.target_points,
            "timestamps": self.state_history.timestamps,
            "steps": self.step_count,
            "completion_reason": self.completion_reason,
            "completed": self.simulation_complete,
            "metrics": self.state_history.get_metrics(),
        }

        return results

    def plot_results(
        self,
        results: Optional[Dict[str, Any]] = None,
        show_metrics: bool = True,
        save_path: Optional[str] = None,
    ):
        """
        Plot simulation results

        Parameters:
        results: Optional results dictionary (if None, uses current results)
        show_metrics: Whether to show metrics in the plot
        save_path: Optional path to save the figure
        """
        if results is None:
            results = self.get_results()

        plt.figure(figsize=(15, 10))

        # Plot 1: Path following
        plt.subplot(2, 2, 1)
        # Plot the reference path
        path_x = [p[0] for p in self.path]
        path_y = [p[1] for p in self.path]
        plt.plot(path_x, path_y, "b--", linewidth=2, label="Reference Path")

        # Plot the actual trajectory
        pos_x = [p[0] for p in results["positions"]]
        pos_y = [p[1] for p in results["positions"]]
        plt.plot(pos_x, pos_y, "r-", linewidth=2, label="Actual Trajectory")

        # Plot target points
        if results["target_points"] and results["target_points"][0] is not None:
            target_x = [
                p[0] for p in results["target_points"] if p is not None and len(p) >= 2
            ]
            target_y = [
                p[1] for p in results["target_points"] if p is not None and len(p) >= 2
            ]
            plt.scatter(target_x, target_y, c="g", s=20, label="Target Points")

        plt.title("Path Following")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()

        # Plot 2: Velocity profile
        plt.subplot(2, 2, 2)
        plt.plot(results["timestamps"], results["velocities"], "b-", linewidth=2)
        plt.title("Velocity Profile")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.grid(True)

        # Plot 3: Steering commands
        plt.subplot(2, 2, 3)
        plt.plot(results["timestamps"], results["steering_angles"], "r-", linewidth=2)
        plt.title("Steering Angle")
        plt.xlabel("Time (s)")
        plt.ylabel("Steering Angle (rad)")
        plt.grid(True)

        # Plot 4: Heading
        plt.subplot(2, 2, 4)
        plt.plot(
            results["timestamps"],
            [np.degrees(h) for h in results["headings"]],
            "g-",
            linewidth=2,
        )
        plt.title("Heading")
        plt.xlabel("Time (s)")
        plt.ylabel("Heading (degrees)")
        plt.grid(True)

        # Add metrics as text if requested
        if show_metrics and "metrics" in results:
            metrics = results["metrics"]
            metrics_text = f"Steps: {results['steps']}\n"
            metrics_text += f"Completion: {results['completion_reason']}\n"

            if "mean_path_error" in metrics:
                metrics_text += f"Mean Path Error: {metrics['mean_path_error']:.3f}m\n"

            if "total_time" in metrics:
                metrics_text += f"Total Time: {metrics['total_time']:.2f}s\n"

            if "mean_velocity" in metrics:
                metrics_text += f"Mean Velocity: {metrics['mean_velocity']:.2f}m/s\n"

            plt.figtext(
                0.5,
                0.01,
                metrics_text,
                ha="center",
                fontsize=10,
                bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5},
            )

        plt.tight_layout()

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def export_data(self, filename: str):
        """
        Export simulation data to a file

        Parameters:
        filename: Name of the file to export to
        """
        results = self.get_results()

        # Determine file type from extension
        ext = filename.split(".")[-1].lower()

        if ext == "csv":
            # Export to CSV
            import csv

            with open(filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                # Write header
                header = ["time", "x", "y", "theta", "velocity", "steering_angle"]
                if results["target_points"] and results["target_points"][0] is not None:
                    header.extend(["target_x", "target_y"])
                writer.writerow(header)

                # Write data
                for i in range(len(results["timestamps"])):
                    row = [
                        results["timestamps"][i],
                        results["positions"][i][0],
                        results["positions"][i][1],
                        results["headings"][i],
                        results["velocities"][i],
                        results["steering_angles"][i],
                    ]

                    if (
                        results["target_points"]
                        and i < len(results["target_points"])
                        and results["target_points"][i] is not None
                    ):
                        row.extend(
                            [
                                results["target_points"][i][0],
                                results["target_points"][i][1],
                            ]
                        )
                    else:
                        row.extend([None, None])

                    writer.writerow(row)

        elif ext in ["npy", "npz"]:
            # Export to NumPy format
            np.savez(
                filename,
                timestamps=np.array(results["timestamps"]),
                positions=np.array(results["positions"]),
                headings=np.array(results["headings"]),
                velocities=np.array(results["velocities"]),
                steering_angles=np.array(results["steering_angles"]),
                target_points=np.array(
                    [
                        p if p is not None else (np.nan, np.nan)
                        for p in results["target_points"]
                    ]
                ),
                path=np.array(self.path),
            )
        else:
            raise ValueError(f"Unsupported file format: {ext}")

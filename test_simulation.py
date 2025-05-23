from typing import List, Tuple
import numpy as np
import numpy.typing as npt
from PurePursuitSimulation import (
    PurePursuitSimulation,
    SimulationConfig,
    SimulationResults,
)
from robots.AckermannSteeringCar import AckermannSteeringCar
from pathfinding_algorithms.ConstantPPC import ConstantPurePursuitController
import pandas as pd


def generate_path(path_type: str = "oval") -> List[Tuple[float, float]]:
    """
    Generate a test path for the controller

    Parameters:
    path_type: Type of path to generate (oval, figure8, complex, zigzag)

    Returns:
    list: List of (x, y) points defining the path
    """
    path: List[Tuple[float, float]]
    if path_type == "oval":
        t: npt.NDArray[np.float64] = np.linspace(0, 2 * np.pi, 100, dtype=np.float64)
        x: npt.NDArray[np.float64] = 5 + 3 * np.cos(t)
        y: npt.NDArray[np.float64] = 5 + 2 * np.sin(t)
        path = list(zip(x, y))

    elif path_type == "figure8":
        t = np.linspace(0, 2 * np.pi, 100, dtype=np.float64)
        x = 5 + 3 * np.sin(t)
        y = 5 + 2 * np.sin(2 * t)
        path = list(zip(x, y))

    elif path_type == "complex":
        path = []
        # Straight segment
        for i in range(20):
            path.append(i * 0.2, 5.0)

        # Curve
        for i in range(30):
            angle: float = i * np.pi / 30
            path.append(
                (
                    4.0 + 2.0 * np.cos(angle),
                    5.0 + 2.0 * np.sin(angle),
                )
            )

        # Another straight segment
        for i in range(20):
            path.append((float(4.0 - i * 0.2), 7.0))
    elif path_type == "zigzag":
        # Zigzag path that moves across the environment
        start_x, start_y = 1.0, 1.0
        path = [(start_x, start_y)]

        # Create a zigzag pattern
        for i in range(10):
            # Forward segment
            next_x_val: float = start_x + (i + 1) * 0.8
            next_y_val: float = start_y + (i % 2) * 1.5  # Alternate up and down

            # Add points along the segment
            last_x, last_y = path[-1]
            steps: int = 15
            for j in range(1, steps + 1):
                x_val: float = float(last_x + (next_x_val - last_x) * j / steps)
                y_val: float = float(last_y + (next_y_val - last_y) * j / steps)
                path.append((x_val, y_val))
    else:
        # Default straight line
        path = []
        for i in range(20):
            path.append((float(i * 0.2), 5.0))

    # Remove last few points to ensure we have a clear goal
    return path[:-10]


def run_simulation(
    path: List[Tuple[float, float]], visualize: bool = True, debug_mode: bool = False
) -> SimulationResults:
    # Create the car model
    car: AckermannSteeringCar = AckermannSteeringCar(
        wheelbase=0.25, max_angular_velocity=0.5, max_steering_angle=np.radians(35)
    )

    # Create the controller
    controller: ConstantPurePursuitController = ConstantPurePursuitController(
        angular_velocity=0.5,
        hysteresis_threshold=10.0,
        dead_zone_threshold=5.0,
        base_lookahead=0.5,
        max_integral_degrees=10.0,
        feedforward_lookahead_points=5,
    )

    # Create simulation configuration
    config: SimulationConfig = SimulationConfig(
        dt=0.1,
        max_steps=5000,
        debug_mode=debug_mode,
        visualize=visualize,
    )

    # Create the simulation
    simulation: PurePursuitSimulation = PurePursuitSimulation(car, controller, config)

    # Run the simulation
    results: SimulationResults = simulation.run_simulation(path)

    # Plot the results with metrics
    simulation.plot_results(results, show_metrics=True)

    return results


if __name__ == "__main__":
    # Generate a test path
    path: List[Tuple[float, float]] = generate_path("figure8")

    # Run the simulation
    print("Running simulation with visualization...")
    results_with_visualization: SimulationResults = run_simulation(path, visualize=True)

    # Print performance metrics
    # df: pd.DataFrame = pd.DataFrame([results_with_visualization["metrics"]])
    # print(f"Performance metrics: {df}")

    # Run the simulation without visualization
    # print("Running simulation without visualization...")
    # results_without_visualization: SimulationResults = run_simulation(
    #     path, visualize=False
    # )

    # df = pd.DataFrame([results_without_visualization["metrics"]])
    # print(f"Performance metrics: {df}")

    # Compare results
    # print("Comparing results...")
    # for key, value in results_with_visualization["metrics"].items():
    #     print(f"{key}: {value}")
    #     print(f"{key}: {results_without_visualization['metrics'][key]}")

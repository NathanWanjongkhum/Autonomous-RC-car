import numpy as np
import matplotlib.pyplot as plt
from PurePursuitSimulation import PurePursuitSimulation, SimulationConfig
from robots.AckermannSteeringCar import AckermannSteeringCar
from pathfinding_algorithms.ConstantPPC import ConstantPurePursuitController
import pandas as pd


def main():
    # Create the car model
    car = AckermannSteeringCar(
        wheelbase=0.25, max_velocity=0.5, max_steering_angle=np.radians(35)
    )

    # Create the controller
    controller = ConstantPurePursuitController(angular_velocity=0.5, base_lookahead=0.5)

    # Create simulation configuration
    config = SimulationConfig(
        dt=0.1,
        max_steps=5000,
        debug_mode=True,
        real_time_factor=4.0,  # Run faster than real-time
    )

    # Create the simulation
    simulation = PurePursuitSimulation(car, controller, config)

    # Generate a test path
    path = simulation.generate_test_path("figure8")

    # Run the simulation
    print("Running simulation...")
    results = simulation.run_simulation(path)

    # Plot the results with metrics
    simulation.plot_results(results, show_metrics=True)

    print(f"Simulation completed: {results['completion_reason']}")
    df = pd.DataFrame(results["metrics"])
    print(f"Performance metrics: {df}")


if __name__ == "__main__":
    main()

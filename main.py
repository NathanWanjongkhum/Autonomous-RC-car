from Simulation import Simulation
from maps.OccupancyGrid import OccupancyGrid
from robots.AckermannSteeringCar import AckermannSteeringCar

from exploration_algorithms.APF import APF
from test_phase2 import execute_phase2, transition_to_phase2


def main():
    """Enhanced main function with integrated Phase 2 controller"""

    # Phase 1: Exploration (using your existing code)
    car = AckermannSteeringCar(x=0.5, y=0.5, theta=0.0)
    grid = OccupancyGrid(width=10.0, height=10.0, resolution=0.05)

    sim = Simulation(car=car, grid=grid)
    sim.create_track(track_type="simple")

    print("Running Phase 1: Exploration...")
    sim.run_simulation(mode="exploration", num_steps=300)

    # Intermediate Phase: Planning
    integrated_controller = transition_to_phase2(sim)

    if integrated_controller:
        # Phase 2: Execution
        execute_phase2(sim, integrated_controller)
    else:
        print("Phase 2 planning failed!")


if __name__ == "__main__":
    main()

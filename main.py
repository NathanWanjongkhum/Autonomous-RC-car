import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

from Simulation import Simulation
from maps.OccupancyGrid import OccupancyGrid
from robots.DifferentialDriveCar import DifferentialDriveCar

from exploration_algorithms.APF import APF

from pathfinding_algorithms.PurePursuit import PurePursuit
from pathfinding_algorithms.WallFollower import WallFollower
from pathfinding_algorithms.AStar import AStarPlanner


def main():
    # Create Car
    car = DifferentialDriveCar(x=2.0, y=2.0, theta=0.0)
    # Create Occupancy Grid
    grid = OccupancyGrid(width=10.0, height=10.0, resolution=0.05)
    # Create exploration algorithm
    apf = APF(car, grid)
    # Create pure pursuit controller
    pure_pursuit = PurePursuit(kp=0.5, ki=0.0, kd=0.0)

    # Create simulation
    sim = Simulation(
        car=car, grid=grid, explorer_algorithm=apf, pure_pursuit=pure_pursuit
    )

    # Create track
    sim.create_track(track_type="simple")

    #### Debugging ####
    # sim.initialize_visualization()
    # sim.update_visualization()
    # plt.show()

    # Phase 1: Exploration
    print("Running Phase 1: Exploration...")
    sim.run_exploration(max_steps=250)

    # # Process map and plan optimal path
    # start_x, start_y = sim.car.x, sim.car.y
    # goal_x, goal_y = 1.5, 1.5

    # print("Planning optimal path...")
    # path = sim.plan_optimal_path(start_x, start_y, goal_x, goal_y)

    # if path:
    #     print("Optimal path found!")

    #     # Phase 2: Execution
    #     print("Running Phase 2: Execution...")
    #     # Reset car position for demonstration
    #     sim.car.x, sim.car.y, sim.car.theta = start_x, start_y, 0.0
    #     sim.run_simulation(mode="execution", num_steps=200)
    # else:
    #     print("Failed to find a path!")


if __name__ == "__main__":
    main()

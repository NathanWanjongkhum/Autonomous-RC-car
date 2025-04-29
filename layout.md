# Car Dynamics System - Data Structure Schema

This schema documents the main classes, their relationships, and key data structures
in the car navigation and dynamics simulation system.

## CLASS OVERVIEW

DifferentialDriveCar:
Represents the physical car with differential drive mechanics.

    State Variables:
        x: float                 # Position x (meters)
        y: float                 # Position y (meters)
        theta: float             # Orientation (radians)

    Physical Parameters:
        wheel_distance: float    # Distance between wheels (meters)
        wheel_radius: float      # Wheel radius (meters)
        max_velocity: float      # Maximum linear velocity (m/s)
        max_angular_velocity     # Maximum angular velocity (rad/s)
        length: float            # Car length (meters)
        width: float             # Car width (meters)

    Control Variables:
        v_l: float               # Left wheel velocity (m/s)
        v_r: float               # Right wheel velocity (m/s)
        v: float                 # Linear velocity (m/s)
        omega: float             # Angular velocity (rad/s)

    Main Methods:
        update_state(dt)         # Updates position and orientation based on velocities
        set_wheel_velocities()   # Sets individual wheel velocities directly
        set_velocities()         # Sets linear and angular velocities
        get_corners()            # Returns car corner positions for visualization/collision

OccupancyGrid:
Represents the environment as a grid for mapping and navigation.

    Grid Representations:
        grid: np.array           # Probabilistic occupancy (0-1 values)
        binary_grid: np.array    # Binary occupancy (True=occupied, False=free)
        log_odds_grid: np.array  # Log-odds representation for Bayesian updates

    Parameters:
        width: float             # Physical width of environment (meters)
        height: float            # Physical height of environment (meters)
        resolution: float        # Size of each cell (meters)
        grid_width: int          # Width in cells
        grid_height: int         # Height in cells
        occupancy_threshold      # Threshold for considering a cell occupied

    Main Methods:
        world_to_grid()          # Converts world coordinates to grid indices
        grid_to_world()          # Converts grid indices to world coordinates
        is_occupied()            # Checks if a world position is occupied
        update_cell()            # Updates a single cell with an observation
        update_from_lidar()      # Updates grid from range sensor data
        update_from_stereovision()  # Updates grid from vision probability data
        process_map()            # Filters and cleans the map

PurePursuit:
Controller for following a predefined path.

    Parameters:
        lookahead_distance: float  # Distance to look ahead on path
        path: list                 # List of (x,y) waypoints

    Main Methods:
        set_path()               # Sets the path to follow
        get_target_point()       # Finds target point on path
        compute_steering()       # Calculates steering command to follow path

WallFollower:
Controller for following walls or boundaries during exploration.

    Parameters:
        grid: OccupancyGrid      # Reference to environment grid
        side: str                # Which side to follow ('left' or 'right')
        desired_distance: float  # Desired distance from wall

    Main Methods:
        sense_walls()            # Senses walls around the car using ray casting
        compute_steering()       # Calculates steering command to follow walls

AStarPlanner:
Path planning algorithm to find optimal paths through the environment.

    Parameters:
        grid: OccupancyGrid      # Reference to environment grid

    Main Methods:
        heuristic()              # Calculates heuristic distance between cells
        get_neighbors()          # Gets valid neighboring cells
        plan_path()              # Finds shortest path from start to goal

Simulation:
Main simulation environment that integrates all components.

    Components:
        grid: OccupancyGrid      # Environment representation
        car: DifferentialDriveCar  # Car being simulated
        wall_follower: WallFollower  # Wall following controller
        path_planner: AStarPlanner   # Path planning algorithm
        pure_pursuit: PurePursuit    # Path following controller

    Simulation Data:
        dt: float                # Time step in seconds
        max_steps: int           # Maximum simulation steps
        path: list               # Planned path
        visited_positions: list  # Record of positions visited

    Visualization:
        fig, ax                  # Matplotlib figure and axes
        car_patch                # Visual representation of car

    Main Methods:
        create_track()           # Creates a track in the environment
        sense_environment()      # Simulates sensing with car sensors
        phase1_exploration()     # Performs Phase 1 exploration
        plan_optimal_path()      # Plans optimal path after exploration
        phase2_execution()       # Executes planned path in Phase 2
        initialize_visualization()  # Sets up visualization
        update_visualization()   # Updates visualization with current state
        run_simulation()         # Runs the complete simulation

## DATA FLOW OVERVIEW

1. Simulation creates an OccupancyGrid and DifferentialDriveCar
2. Phase 1 (Exploration):
   - WallFollower computes steering commands based on sensed environment
   - Car updates its state based on these commands
   - Environment is sensed and OccupancyGrid is updated
3. Intermediate Phase:
   - AStarPlanner computes optimal path through the processed OccupancyGrid
   - Path is stored for Phase 2
4. Phase 2 (Execution):
   - PurePursuit computes steering commands to follow planned path
   - Car updates its state based on these commands
   - Real-time corrections handle deviations

## MAIN ALGORITHMS

- Occupancy Grid Mapping: Probabilistic mapping using Bayesian updates
- Wall Following: Simple reactive control for exploration
- A\* Path Planning: Graph search algorithm for finding optimal paths
- Pure Pursuit: Path tracking algorithm for following planned paths

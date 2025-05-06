import itertools
import numpy as np

class AStarPlanner:
    """
    Implements the A* path planning algorithm on a grid map

    This class finds the shortest path from a start position to a goal
    position in an occupancy grid.
    """

    def __init__(self, occupancy_grid):
        """
        Initialize the A* planner

        Parameters:
        occupancy_grid: OccupancyGrid instance
        """
        self.grid = occupancy_grid

    def heuristic(self, a, b):
        """
        Calculate the heuristic distance between two grid cells

        Parameters:
        a, b: Grid cell indices (x, y)

        Returns:
        float: Euclidean distance
        """
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def get_neighbors(self, node):
        """
        Get valid neighboring cells of a grid cell

        Parameters:
        node: Grid cell indices (x, y)

        Returns:
        list: List of valid neighboring cells
        """
        x, y = node
        neighbors = []

        # Define 8-connected grid (diagonal and orthogonal neighbors)
        directions = np.array(list(itertools.product([-1, 0, 1], repeat=2)))

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check if within grid bounds
            if 0 <= nx < self.grid.grid_width and 0 <= ny < self.grid.grid_height:
                # Check if cell is free (not occupied)
                if not self.grid.binary_grid[ny, nx]:
                    neighbors.append((nx, ny))

        return neighbors

    def plan_path(self, start_x, start_y, goal_x, goal_y):
        """
        Find the shortest path from start to goal

        Parameters:
        start_x, start_y: Start position in world coordinates
        goal_x, goal_y: Goal position in world coordinates

        Returns:
        path: List of (x, y) points defining the path in world coordinates
            or None if no path found
        """
        # Convert to grid coordinates
        start_grid = self.grid.world_to_grid(start_x, start_y)
        goal_grid = self.grid.world_to_grid(goal_x, goal_y)

        # Check if start or goal is in obstacle
        if (
            self.grid.binary_grid[start_grid[1], start_grid[0]]
            or self.grid.binary_grid[goal_grid[1], goal_grid[0]]
        ):
            print("Start or goal is in obstacle!")
            return None

        # Initialize the open and closed sets
        open_set = {start_grid}
        closed_set = set()

        # Dictionary to store the best path to each node
        came_from = {}

        # Dictionary to store the cost to reach each node from the start
        g_score = {start_grid: 0}

        # Dictionary to store the estimated total cost from start to goal through each node
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        while open_set:
            # Find the node in open_set with the lowest f_score
            current = min(open_set, key=lambda node: f_score.get(node, float("inf")))

            # If goal reached, reconstruct and return the path
            if current == goal_grid:
                path = []
                while current in came_from:
                    # Convert back to world coordinates
                    world_x, world_y = self.grid.grid_to_world(current[0], current[1])
                    path.append((world_x, world_y))
                    current = came_from[current]

                # Add start position
                world_x, world_y = self.grid.grid_to_world(start_grid[0], start_grid[1])
                path.append((world_x, world_y))

                # Reverse to get path from start to goal
                path.reverse()
                return path

            # Move current from open to closed set
            open_set.remove(current)
            closed_set.add(current)

            # Check each neighbor
            for neighbor in self.get_neighbors(current):
                # Skip if already evaluated
                if neighbor in closed_set:
                    continue

                # Calculate tentative g_score
                tentative_g_score = g_score[current] + self.heuristic(current, neighbor)

                # If neighbor not in open set, add it
                if neighbor not in open_set:
                    open_set.add(neighbor)
                # If this path to neighbor is worse, skip
                elif tentative_g_score >= g_score.get(neighbor, float("inf")):
                    continue

                # This path is the best so far, record it
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.heuristic(
                    neighbor, goal_grid
                )

        # No path found
        print("No path found!")
        return None

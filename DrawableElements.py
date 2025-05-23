import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from abc import ABC, abstractmethod


class DrawableElement(ABC):
    """Base class for drawable elements"""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.artists = []  # Store matplotlib artists for this element

    @abstractmethod
    def create(self, ax):
        """Create the visual element and return matplotlib artists"""
        pass

    @abstractmethod
    def update(self, ax):
        """Update the visual element"""
        pass

    def remove(self):
        """Remove all artists from the plot"""
        for artist in self.artists:
            if artist in artist.axes.get_children():
                artist.remove()
        self.artists.clear()

    def set_enabled(self, enabled):
        """Enable or disable this element"""
        self.enabled = enabled
        for artist in self.artists:
            artist.set_visible(enabled)


class CarBodyDrawer(DrawableElement):
    """Draws the car body as a polygon"""

    def __init__(self, car, color="blue", fill=False, enabled=True):
        super().__init__(enabled)
        self.car = car
        self.color = color
        self.fill = fill
        self.car_patch = None

    def create(self, ax):
        corners = self.car.get_corners()
        xy = np.array(corners + [corners[0]])  # Close the polygon
        self.car_patch = patches.Polygon(
            xy, closed=True, color=self.color, fill=self.fill
        )
        ax.add_patch(self.car_patch)
        self.artists.append(self.car_patch)
        return self.car_patch

    def update(self, ax):
        if not self.enabled or self.car_patch is None:
            return

        corners = self.car.get_corners()
        xy = np.array(corners + [corners[0]])  # Close the polygon
        self.car_patch.set_xy(xy)


class WheelDrawer(DrawableElement):
    """Draws car wheels"""

    def __init__(self, car, color="black", enabled=True):
        super().__init__(enabled)
        self.car = car
        self.color = color
        self.front_wheel_patches = []
        self.rear_wheel_patches = []

    def create(self, ax):
        if not self.enabled:
            return []

        self._create_front_wheels(ax)
        self._create_rear_wheels(ax)
        return self.artists

    def _create_front_wheels(self, ax):
        steering_coords = self.car.visualize_steering()

        for x1, y1, x2, y2 in steering_coords:
            # Calculate wheel angle
            wheel_angle = np.arctan2(y2 - y1, x2 - x1)

            # Create wheel patch
            wheel_patch = patches.Rectangle(
                (x1 - self.car.wheel_width / 2, y1 - self.car.wheel_width / 2),
                self.car.wheel_radius * 2,
                self.car.wheel_width,
                angle=np.degrees(wheel_angle),
                color=self.color,
                fill=True,
            )
            ax.add_patch(wheel_patch)
            self.front_wheel_patches.append(wheel_patch)
            self.artists.append(wheel_patch)

    def _create_rear_wheels(self, ax):
        cos_theta = np.cos(self.car.theta)
        sin_theta = np.sin(self.car.theta)

        # Rear axle center
        rear_center_x = self.car.x - (self.car.length / 2) * cos_theta
        rear_center_y = self.car.y - (self.car.length / 2) * sin_theta

        # Calculate rear wheel positions
        rear_wheel_positions = [
            (
                rear_center_x
                + (self.car.width / 2 + self.car.wheel_offset) * -sin_theta,
                rear_center_y
                + (self.car.width / 2 + self.car.wheel_offset) * cos_theta,
            ),
            (
                rear_center_x
                + (self.car.width / 2 + self.car.wheel_offset) * sin_theta,
                rear_center_y
                + (self.car.width / 2 + self.car.wheel_offset) * -cos_theta,
            ),
        ]

        for x, y in rear_wheel_positions:
            wheel_patch = patches.Rectangle(
                (x - self.car.wheel_width / 2, y - self.car.wheel_width / 2),
                self.car.wheel_radius * 2,
                self.car.wheel_width,
                angle=np.degrees(self.car.theta),
                color=self.color,
                fill=True,
            )
            ax.add_patch(wheel_patch)
            self.rear_wheel_patches.append(wheel_patch)
            self.artists.append(wheel_patch)

    def update(self, ax):
        if not self.enabled:
            return

        # Remove old wheel patches
        for patch in self.front_wheel_patches + self.rear_wheel_patches:
            patch.remove()

        self.front_wheel_patches.clear()
        self.rear_wheel_patches.clear()
        self.artists.clear()

        # Recreate wheels with updated positions
        self._create_front_wheels(ax)
        self._create_rear_wheels(ax)


class SteeringLinesDrawer(DrawableElement):
    """Draws steering direction lines"""

    def __init__(self, car, color="red", linewidth=2, enabled=True):
        super().__init__(enabled)
        self.car = car
        self.color = color
        self.linewidth = linewidth
        self.steering_lines = []

    def create(self, ax):
        if not self.enabled:
            return []

        steering_coords = self.car.visualize_steering()

        for x1, y1, x2, y2 in steering_coords:
            (line,) = ax.plot(
                [x1, x2], [y1, y2], color=self.color, linewidth=self.linewidth
            )
            self.steering_lines.append(line)
            self.artists.append(line)

        return self.artists

    def update(self, ax):
        if not self.enabled:
            return

        steering_coords = self.car.visualize_steering()

        for i, (x1, y1, x2, y2) in enumerate(steering_coords):
            if i < len(self.steering_lines):
                self.steering_lines[i].set_data([x1, x2], [y1, y2])


class OccupancyGridDrawer(DrawableElement):
    """Draws the occupancy grid"""

    def __init__(self, grid, enabled=True):
        super().__init__(enabled)
        self.grid = grid
        self.grid_image = None

    def create(self, ax):
        if not self.enabled:
            return None

        grid_img = self._create_grid_image()
        self.grid_image = ax.imshow(
            grid_img,
            extent=(0, self.grid.width, 0, self.grid.height),
            origin="lower",
            interpolation="nearest",
        )
        self.artists.append(self.grid_image)
        return self.grid_image

    def _create_grid_image(self):
        grid_img = np.zeros((self.grid.grid_height, self.grid.grid_width, 4))

        # Set probability visualization (grayscale for unknown areas)
        grid_img[:, :, 0] = 0.8 * (1 - self.grid.grid)  # Red channel
        grid_img[:, :, 1] = 0.8 * (1 - self.grid.grid)  # Green channel
        grid_img[:, :, 2] = 0.8 * (1 - self.grid.grid)  # Blue channel
        grid_img[:, :, 3] = 0.5  # Semi-transparent for probabilities

        # Overlay binary grid (solid red for occupied, solid green for known free)
        grid_img[self.grid.binary_grid, 0] = 1.0  # Red for occupied
        grid_img[self.grid.binary_grid, 1:3] = 0.0  # No green/blue for occupied
        grid_img[self.grid.binary_grid, 3] = 1.0  # Fully opaque for occupied

        return grid_img

    def update(self, ax):
        if not self.enabled or self.grid_image is None:
            return

        grid_img = self._create_grid_image()
        self.grid_image.set_data(grid_img)


class PathDrawer(DrawableElement):
    """Draws a path as a line"""

    def __init__(
        self,
        path=None,
        color="blue",
        linestyle="--",
        linewidth=2,
        label="Reference Path",
        enabled=True,
    ):
        super().__init__(enabled)
        self.path = path or []
        self.color = color
        self.linestyle = linestyle
        self.linewidth = linewidth
        self.label = label
        self.path_line = None

    def set_path(self, path):
        """Update the path to be drawn"""
        self.path = path

    def create(self, ax):
        if not self.enabled or not self.path:
            return None

        path_x = [p[0] for p in self.path]
        path_y = [p[1] for p in self.path]

        (self.path_line,) = ax.plot(
            path_x,
            path_y,
            color=self.color,
            linestyle=self.linestyle,
            linewidth=self.linewidth,
            label=self.label,
        )
        self.artists.append(self.path_line)
        return self.path_line

    def update(self, ax):
        if not self.enabled or self.path_line is None or not self.path:
            return

        path_x = [p[0] for p in self.path]
        path_y = [p[1] for p in self.path]
        self.path_line.set_data(path_x, path_y)


class TrajectoryDrawer(DrawableElement):
    """Draws visited positions as a trajectory"""

    def __init__(
        self,
        positions=None,
        color="yellow",
        marker=".",
        markersize=1,
        label="Trajectory",
        enabled=True,
    ):
        super().__init__(enabled)
        self.positions = positions or []
        self.color = color
        self.marker = marker
        self.markersize = markersize
        self.label = label
        self.trajectory_line = None

    def add_position(self, x, y):
        """Add a new position to the trajectory"""
        self.positions.append((x, y))

    def set_positions(self, positions):
        """Set the entire trajectory"""
        self.positions = positions

    def create(self, ax):
        if not self.enabled:
            return None

        if self.positions:
            visited_x = [p[0] for p in self.positions]
            visited_y = [p[1] for p in self.positions]
        else:
            visited_x, visited_y = [], []

        (self.trajectory_line,) = ax.plot(
            visited_x,
            visited_y,
            color=self.color,
            marker=self.marker,
            markersize=self.markersize,
            linestyle="",
            label=self.label,
        )
        self.artists.append(self.trajectory_line)
        return self.trajectory_line

    def update(self, ax):
        if not self.enabled or self.trajectory_line is None:
            return

        if self.positions:
            visited_x = [p[0] for p in self.positions]
            visited_y = [p[1] for p in self.positions]
            self.trajectory_line.set_data(visited_x, visited_y)


class MarkerDrawer(DrawableElement):
    """Draws individual markers (like start/goal positions)"""

    def __init__(
        self,
        x=0,
        y=0,
        color="green",
        marker="o",
        markersize=8,
        label="Marker",
        enabled=True,
    ):
        super().__init__(enabled)
        self.x = x
        self.y = y
        self.color = color
        self.marker = marker
        self.markersize = markersize
        self.label = label
        self.marker_plot = None

    def set_position(self, x, y):
        """Update marker position"""
        self.x = x
        self.y = y

    def create(self, ax):
        if not self.enabled:
            return None

        (self.marker_plot,) = ax.plot(
            self.x,
            self.y,
            color=self.color,
            marker=self.marker,
            markersize=self.markersize,
            linestyle="",
            label=self.label,
        )
        self.artists.append(self.marker_plot)
        return self.marker_plot

    def update(self, ax):
        if not self.enabled or self.marker_plot is None:
            return

        self.marker_plot.set_data([self.x], [self.y])


class VisualizationManager:
    """Manages all drawable elements and provides a unified interface"""

    def __init__(self, car, grid=None, figsize=(10, 8)):
        self.car = car
        self.grid = grid
        self.figsize = figsize

        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=figsize)

        # Dictionary to store all drawable elements
        self.elements = {}

        # Initialize default elements
        self._initialize_default_elements()

        # Setup axis
        self._setup_axis()

    def _initialize_default_elements(self):
        """Initialize default drawable elements"""
        # Car elements
        self.elements["car_body"] = CarBodyDrawer(self.car, color="white")
        self.elements["wheels"] = WheelDrawer(
            self.car,
            color="white",
        )
        self.elements["steering_lines"] = SteeringLinesDrawer(self.car)

        # Environment elements
        if self.grid:
            self.elements["occupancy_grid"] = OccupancyGridDrawer(self.grid)

        # Path and trajectory elements
        self.elements["reference_path"] = PathDrawer(
            color="blue", linestyle="--", label="Reference Path"
        )
        self.elements["trajectory"] = TrajectoryDrawer(
            color="yellow", marker=".", markersize=1, label="Trajectory"
        )

        # Markers
        self.elements["start_marker"] = MarkerDrawer(
            self.car.x,
            self.car.y,
            color="green",
            marker="o",
            markersize=8,
            label="Start",
        )

    def _setup_axis(self):
        """Setup the axis properties"""
        if self.grid:
            self.ax.set_xlim(0, self.grid.width)
            self.ax.set_ylim(0, self.grid.height)
        else:
            self.ax.set_xlim(-5, 5)
            self.ax.set_ylim(-5, 5)

        self.ax.set_aspect("equal")
        self.ax.set_title("Car Navigation Simulation")
        self.ax.grid(True, alpha=0.3)

    def add_element(self, name, element):
        """Add a custom drawable element"""
        self.elements[name] = element

    def remove_element(self, name):
        """Remove an element"""
        if name in self.elements:
            self.elements[name].remove()
            del self.elements[name]

    def enable_element(self, name, enabled=True):
        """Enable or disable an element"""
        if name in self.elements:
            self.elements[name].set_enabled(enabled)

    def get_element(self, name):
        """Get a specific element"""
        return self.elements.get(name)

    def initialize_all(self):
        """Initialize all enabled elements"""
        for element in self.elements.values():
            if element.enabled:
                element.create(self.ax)

    def update_all(self):
        """Update all enabled elements"""
        for element in self.elements.values():
            if element.enabled:
                element.update(self.ax)

        self.fig.canvas.draw()

    def show_legend(self):
        """Show legend for labeled elements"""
        self.ax.legend()

    def save_figure(self, filename):
        """Save the current figure"""
        self.fig.savefig(filename, dpi=300, bbox_inches="tight")

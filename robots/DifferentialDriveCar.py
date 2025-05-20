import numpy as np


class DifferentialDriveCar:
    """
    Models a differential drive car with rectangular dimensions

    This class simulates the kinematics of a car with two powered wheels
    in the back and two smaller wheels for stability, as described in the
    requirements.
    """

    def __init__(
        self,
        x=0,
        y=0,
        theta=0,
        wheel_distance=0.2,
        wheel_radius=0.05,
        max_velocity=0.5,
        max_angular_velocity=1.0,
    ):
        # State variables
        self.x = x  # Position x (meters)
        self.y = y  # Position y (meters)
        self.theta = theta  # Orientation (radians)

        # Physical parameters
        self.wheel_distance = wheel_distance  # Distance between wheels (meters)
        self.wheel_radius = wheel_radius  # Wheel radius (meters)
        self.max_velocity = max_velocity  # Maximum linear velocity (m/s)
        self.max_angular_velocity = (
            max_angular_velocity  # Maximum angular velocity (rad/s)
        )

        # Control variables
        self.v = 0  # Linear velocity (m/s)
        self.omega = 0  # Angular velocity (rad/s)

        # Car dimensions (rectangular shape)
        self.length = 0.3  # Car length (meters)
        self.width = 0.2  # Car width (meters)

    def update_state(self, dt):
        """
        Update car state based on wheel velocities over time dt

        Implements the differential drive kinematics equations
        to update position and orientation

        Parameters:
        dt: Time step in seconds
        """
        # Calculate linear and angular velocity from wheel velocities
        # self.v = (self.v_r + self.v_l) / 2
        # self.omega = (self.v_r - self.v_l) / self.wheel_distance

        # Update state using basic differential drive kinematics
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.omega * dt
        
        # Normalize theta to (-pi, pi) to prevent growing continuously
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

    def set_velocities(self, v, omega):
        """
        Set the linear and angular velocities, and calculate corresponding wheel velocities

        Handles the conversion from desired robot velocities to individual wheel velocities
        while respecting maximum limits

        Parameters:
        v: Desired linear velocity (m/s)
        omega: Desired angular velocity (rad/s)
        """
        # Apply limits to requested velocities
        self.v = np.clip(v, -self.max_velocity, self.max_velocity)
        self.omega = np.clip(omega, -self.max_angular_velocity, self.max_angular_velocity)

    def get_corners(self):
        """
        Get the four corners of the car for collision detection and visualization

        Returns a list of (x,y) coordinates for the four corners of the car,
        taking into account its position, orientation, length, and width
        """
        # Calculate the four corners of the car based on position, orientation, length, and width
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        # Calculate the corners relative to the car's position and orientation
        # Using rotation matrix to transform from car frame to world frame
        front_right = (
            self.x + self.length / 2 * cos_theta - self.width / 2 * sin_theta,
            self.y + self.length / 2 * sin_theta + self.width / 2 * cos_theta,
        )

        front_left = (
            self.x + self.length / 2 * cos_theta + self.width / 2 * sin_theta,
            self.y + self.length / 2 * sin_theta - self.width / 2 * cos_theta,
        )

        rear_left = (
            self.x - self.length / 2 * cos_theta + self.width / 2 * sin_theta,
            self.y - self.length / 2 * sin_theta - self.width / 2 * cos_theta,
        )

        rear_right = (
            self.x - self.length / 2 * cos_theta - self.width / 2 * sin_theta,
            self.y - self.length / 2 * sin_theta + self.width / 2 * cos_theta,
        )

        return [front_right, front_left, rear_left, rear_right]

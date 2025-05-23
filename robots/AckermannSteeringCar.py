import numpy as np
from typing import List, Tuple


class AckermannSteeringCar:
    """
    Models a car with Ackermann steering (front-wheel steering)

    This class simulates the kinematics of a car with front-wheel steering,
    two powered wheels in the back, and two steerable wheels in the front.
    Unlike differential drive, this model steers by physically turning the wheels.

    Key Kinematic Equations:
    - ẋ = v * cos(θ)
    - ẏ = v * sin(θ)
    - θ̇ = v * tan(φ) / L

    Where:
    - v is the linear velocity
    - θ is the orientation
    - φ is the steering angle
    - L is the wheelbase (distance between front and rear axles)
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        wheelbase: float = 0.25,  # Distance between front and rear axles (meters)
        wheel_radius: float = 0.05,
        wheel_width: float = 0.04,  # Width of wheel (meters)
        wheel_offset: float = 0.03,  # How far wheels extend beyond the body width (meters)
        max_velocity: float = 0.5,
        max_steering_angle: float = np.radians(
            35
        ),  # Max steering angle in radians (~35 degrees)
        max_angular_velocity: float = 1.0,
        length: float = 0.3,  # Car length (meters)
        width: float = 0.2,  # Car width (meters)
    ) -> None:
        # State variables
        self.x: float = x  # Position x (meters)
        self.y: float = y  # Position y (meters)
        self.theta: float = theta  # Orientation (radians)
        self.steering_angle: float = 0.0  # Front wheel steering angle (radians)

        # Physical parameters
        self.wheelbase: float = (
            wheelbase  # Distance between front and rear axles (meters)
        )
        self.wheel_radius: float = wheel_radius  # Wheel radius (meters)
        self.wheel_width: float = wheel_width  # Width of wheel (meters)
        self.wheel_offset: float = (
            wheel_offset  # How far wheels extend beyond the body (meters)
        )
        self.max_velocity: float = max_velocity  # Maximum linear velocity (m/s)
        self.max_steering_angle: float = (
            max_steering_angle  # Maximum steering angle (rad)
        )
        self.max_angular_velocity: float = (
            max_angular_velocity  # Maximum angular velocity (rad/s)
        )

        # Control variables
        self.v: float = 0.0  # Linear velocity (m/s)
        self.omega: float = (
            0.0  # Angular velocity (rad/s) - derived from steering angle and velocity
        )

        # Car dimensions (rectangular shape)
        self.length: float = length  # Car length (meters)
        self.width: float = width  # Car width (meters)

    def update_state(self, dt: float):
        """
        Update car state based on velocity and steering angle over time dt

        Implements the Ackermann steering kinematics equations to update
        position and orientation.

        Parameters:
        dt: Time step in seconds
        """
        # Calculate angular velocity from steering angle (bicycle model)
        # θ̇ = v * tan(φ) / L
        if abs(self.v) > 1e-5:  # Only update angular velocity if car is moving
            self.omega = self.v * np.tan(self.steering_angle) / self.wheelbase
        else:
            self.omega = 0

        # Update state using Ackermann kinematics
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.omega * dt

        # Normalize theta to (-pi, pi) to prevent growing continuously
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

    def set_control_inputs(self, v: float, steering_angle: float):
        """
        Set the linear velocity and steering angle

        Handles respecting the physical limits of the vehicle

        Parameters:
        v: Desired linear velocity (m/s)
        steering_angle: Desired steering angle (radians)
        """
        # Apply limits to requested control inputs
        self.v = np.clip(v, -self.max_velocity, self.max_velocity)

        # Adjust steering angle limits based on speed for stability
        # At higher speeds, reduce maximum allowed steering angle
        speed_factor = min(1.0, 0.5 + 0.5 * (1 - abs(self.v) / self.max_velocity))
        effective_max_steering = self.max_steering_angle * speed_factor

        # Apply limits to steering angle
        self.steering_angle = np.clip(
            steering_angle, -effective_max_steering, effective_max_steering
        )

        # Calculate resulting angular velocity (for compatibility with algorithms
        # that might still expect it as a control input)
        if abs(self.v) > 1e-5:
            self.omega = self.v * np.tan(self.steering_angle) / self.wheelbase
        else:
            self.omega = 0

    def get_corners(self) -> List[Tuple[float, float]]:
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

    def visualize_steering(self) -> List[Tuple[float, float, float, float]]:
        """
        Get coordinates to visualize the steering angle

        Returns:
        A list of tuples, where each tuple contains four float values:
        (x_start, y_start, x_end, y_end). These represent the start and end
        coordinates of the lines that visualize the front wheels with their
        steering angle.
        """
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        # Front axle center - position at the front edge of the car
        front_center_x = self.x + (self.length / 2) * cos_theta
        front_center_y = self.y + (self.length / 2) * sin_theta

        # Calculate wheel positions (centers of front wheels)
        # Position the wheels at the sides of the car PLUS the offset
        left_wheel_x = (
            front_center_x + (self.width / 2 + self.wheel_offset) * -sin_theta
        )
        left_wheel_y = front_center_y + (self.width / 2 + self.wheel_offset) * cos_theta

        right_wheel_x = (
            front_center_x + (self.width / 2 + self.wheel_offset) * sin_theta
        )
        right_wheel_y = (
            front_center_y + (self.width / 2 + self.wheel_offset) * -cos_theta
        )

        # Calculate the direction of the wheels
        wheel_angle = self.theta + self.steering_angle
        wheel_cos = np.cos(wheel_angle)
        wheel_sin = np.sin(wheel_angle)

        # Calculate wheel endpoints (for visualization)
        wheel_extension = self.wheel_radius * 2
        left_wheel_end_x = left_wheel_x + wheel_extension * wheel_cos
        left_wheel_end_y = left_wheel_y + wheel_extension * wheel_sin

        right_wheel_end_x = right_wheel_x + wheel_extension * wheel_cos
        right_wheel_end_y = right_wheel_y + wheel_extension * wheel_sin

        # Return visualization coordinates for the left and right wheels
        return [
            (left_wheel_x, left_wheel_y, left_wheel_end_x, left_wheel_end_y),
            (right_wheel_x, right_wheel_y, right_wheel_end_x, right_wheel_end_y),
        ]

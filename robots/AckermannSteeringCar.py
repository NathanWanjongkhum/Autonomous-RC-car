from dataclasses import dataclass
import numpy as np
from typing import List, Tuple


@dataclass
class DiscretePose:
    """Represents an exact continuous state with full precision"""

    x: int
    y: int
    theta: int

    def __hash__(self):
        # For use in sets/dicts - based on discrete representation
        return hash((self.x * 1000, self.y * 1000, self.theta * 1000))


@dataclass
class ContinuousPose:
    """Represents an exact continuous state with full precision"""

    x: float
    y: float
    theta: float

    def __hash__(self):
        # For use in sets/dicts - based on discrete representation
        return hash((int(self.x * 1000), int(self.y * 1000), int(self.theta * 1000)))


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
        start_pose: ContinuousPose,
        wheelbase: float = 0.25,
        wheel_radius: float = 0.05,
        wheel_width: float = 0.04,
        wheel_offset: float = 0.03,
        max_steering_angle: float = np.radians(35),
        max_angular_velocity: float = 1.0,
        length: float = 0.3,
        width: float = 0.2,
        reference_point: str = "rear",
    ) -> None:
        # Physical parameters
        self.length: float = length  # Car length (meters)
        self.width: float = width  # Car width (meters)
        self.wheelbase: float = (
            wheelbase  # Distance between front and rear axles (meters)
        )
        self.wheel_radius: float = wheel_radius  # Wheel radius (meters)
        self.wheel_width: float = wheel_width  # Width of wheel (meters)
        self.wheel_offset: float = (
            wheel_offset  # How far wheels extend beyond the body (meters)
        )
        # Physical constraints
        self.max_steering_angle: float = (
            max_steering_angle  # Maximum steering angle (rad)
        )
        self.max_angular_velocity: float = (
            max_angular_velocity  # Maximum angular velocity (rad/s)
        )
        # State variables
        self.pose = start_pose  # (x, y, theta)
        self.steering_angle: float = 0.0  # Front wheel steering angle (radians)
        self.v: float = 0.0  # Linear velocity (m/s)
        self.omega: float = (
            0.0  # Angular velocity (rad/s) - derived from steering angle and velocity
        )

    def update_state(self, dt: float):
        """
        Update car state based on velocity and steering angle over time dt

        Implements the Ackermann steering kinematics equations to update
        position and orientation.

        Parameters:
        dt: Time step in seconds
        """
        # Calculate angular velocity from steering angle (bicycle model) θ̇ = v * tan(φ) / L
        if abs(self.v) > 1e-5:  # Only update angular velocity if car is moving
            self.omega = self.v * np.tan(self.steering_angle) / self.wheelbase
        else:
            self.omega = 0

        # Update state using Ackermann kinematics
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta = self.get_principal_value(self.theta + self.omega * dt)

    @staticmethod
    def get_principal_value(angle: float) -> float:
        """
        Normalize theta to [-π, π] to prevent growing continuously
        """
        return np.arctan2(np.sin(angle), np.cos(angle))

    def set_control_inputs(self, v: float, steering_angle: float):
        """
        Set the linear velocity and steering angle

        Handles respecting the physical limits of the vehicle

        Parameters:
        v: Desired linear velocity (m/s)
        steering_angle: Desired steering angle (radians)
        """
        self.v = v
        self.steering_angle = steering_angle

    def get_corners(self) -> List[Tuple[float, float]]:
        """
        Get the four corners of the car for collision detection and visualization

        Returns a list of (x,y) coordinates for the four corners of the car,
        taking into account its position, orientation, length, and width
        """
        x, y, theta = self.pose.x, self.pose.y, self.pose.theta
        width, length = self.width, self.length

        # Calculate the four corners of the car based on position, orientation, length, and width
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Calculate the corners relative to the car's position and orientation
        # Using rotation matrix to transform from car frame to world frame
        front_right = (
            x + length / 2 * cos_theta - width / 2 * sin_theta,
            y + length / 2 * sin_theta + width / 2 * cos_theta,
        )

        front_left = (
            x + length / 2 * cos_theta + width / 2 * sin_theta,
            y + length / 2 * sin_theta - width / 2 * cos_theta,
        )

        rear_left = (
            x - length / 2 * cos_theta + width / 2 * sin_theta,
            y - length / 2 * sin_theta - width / 2 * cos_theta,
        )

        rear_right = (
            x - length / 2 * cos_theta - width / 2 * sin_theta,
            y - length / 2 * sin_theta + width / 2 * cos_theta,
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
        x, y, theta = self.pose.x, self.pose.y, self.pose.theta
        width, length = self.width, self.length
        wheel_offset = self.wheel_offset

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Front axle center - position at the front edge of the car
        front_center_x = x + (length / 2) * cos_theta
        front_center_y = y + (length / 2) * sin_theta

        # Calculate wheel positions (centers of front wheels)
        # Position the wheels at the sides of the car PLUS the offset
        left_wheel_x = front_center_x + (width / 2 + wheel_offset) * -sin_theta
        left_wheel_y = front_center_y + (width / 2 + wheel_offset) * cos_theta

        right_wheel_x = front_center_x + (width / 2 + wheel_offset) * sin_theta
        right_wheel_y = front_center_y + (width / 2 + wheel_offset) * -cos_theta

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

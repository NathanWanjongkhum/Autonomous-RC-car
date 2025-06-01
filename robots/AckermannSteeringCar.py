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
        wheel_base: float = 0.25,
        wheel_radius: float = 1,
        wheel_width: float = 0.04,
        wheel_offset: float = 0.03,
        max_wheel_steering_angle: float = np.radians(35),
        wheel_angular_velocity: float = 1.0,
        length: float = 0.3,
        width: float = 0.2,
        reference_point: str = "rear",
    ) -> None:
        # Physical parameters
        self.length: float = length  # Car length (meters)
        self.width: float = width  # Car width (meters)
        self.wheel_base: float = wheel_base  # Distance between front and rear axles (meters)
        self.wheel_radius: float = wheel_radius  # Wheel radius (meters)
        self.wheel_width: float = wheel_width  # Width of wheel (meters)
        self.wheel_offset: float = wheel_offset  # How far wheels extend beyond the body (meters)
        self.reference_point = reference_point
        # Physical constraints
        self.max_wheel_steering_angle: float = (
            max_wheel_steering_angle  # Maximum steering angle (rad)
        )
        self.wheel_angular_velocity: float = wheel_angular_velocity  # angular velocity (rad/s)
        # State variables
        self.pose = start_pose  # (x, y, theta)
        self.wheel_steering_angle: float = 0.0  # Front wheel steering angle (radians)
        self.linear_velocity: float = 0.0  # Linear velocity (m/s) (v = wheel_ω * radius)
        self.car_angular_velocity = 0.0  # Angular velocity (rad/s) (ω = v*tan(φ)/L)

    def update_state(self, dt: float):
        """
        Update car state based on velocity and steering angle over time dt

        Implements the Ackermann steering kinematics equations to update
        position and orientation.

        Parameters:
        dt: Time step in seconds
        """
        # Calculate angular velocity from steering angle (bicycle model) θ̇ = v * tan(φ) / L
        if abs(self.linear_velocity) > 1e-5:
            self.car_angular_velocity = (
                self.linear_velocity * np.tan(self.wheel_steering_angle) / self.wheel_base
            )
        else:
            # Only update angular velocity if car is moving
            self.car_angular_velocity = 0

        # Update state using Ackermann kinematics
        self.x += self.linear_velocity * np.cos(self.theta) * dt
        self.y += self.linear_velocity * np.sin(self.theta) * dt
        self.theta = get_principal_value(self.theta + self.car_angular_velocity * dt)

    def set_control_inputs(self, v: float, steering_angle: float):
        """
        Set the linear velocity and steering angle

        Handles respecting the physical limits of the vehicle

        Parameters:
        v: Desired linear velocity (m/s)
        steering_angle: Desired steering angle (radians)
        """
        self.linear_velocity = v
        self.wheel_steering_angle = steering_angle

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
        wheel_angle = self.theta + self.wheel_steering_angle
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

    def report_parameters(self):
        """
        Report car parameters
        """
        return {
            "Type": type(self).__name__,
            "wheel_base": self.wheel_base,
            "wheel_radius": self.wheel_radius,
            "wheel_width": self.wheel_width,
            "wheel_offset": self.wheel_offset,
            "max_wheel_steering_angle": self.max_wheel_steering_angle,
            "wheel_angular_velocity": self.wheel_angular_velocity,
            "length": self.length,
            "width": self.width,
            "reference_point": self.reference_point,
        }

    def __copy__(self):
        """
        Create a copy of the car
        """
        return AckermannSteeringCar(
            self.pose,
            wheel_base=self.wheel_base,
            wheel_radius=self.wheel_radius,
            wheel_width=self.wheel_width,
            wheel_offset=self.wheel_offset,
            max_wheel_steering_angle=self.max_wheel_steering_angle,
            wheel_angular_velocity=self.wheel_angular_velocity,
            length=self.length,
            width=self.width,
            reference_point=self.reference_point,
        )


@staticmethod
def get_principal_value(angle: float) -> float:
    """
    Normalize theta to [-π, π] to prevent growing continuously
    """
    return np.arctan2(np.sin(angle), np.cos(angle))

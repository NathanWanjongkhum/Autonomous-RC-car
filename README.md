# Autonomous RC car

# Computer Vision

For the car to be able to drive itself it will need to be able to somehow detect its environment. To solve this problem we will use computer vision. Specifically we will use stereo vision to detect the car and the environment.

## Camera Calibration

1. Get a calibration board (Black and white squares of known size)
2. Capture images of the board at different angles and distances
3. Use the calibration board to find the chessboard corners
4. Use the chessboard corners to find the camera intrinsics
5. Use the camera intrinsics to calibrate the camera

## Stereo Vision Calibration

1. After calibrating each camera, relate the two cameras
2. Apply stero calibration
3. Apply stero-rectification transformation

## Stero Vision

1. Setup camera parameters
2. Calculate the depth map

# Race logic

Now that we know where it is in relation to the world we can start to think about how to drive it.

## Obstacle Avoidance

The car will need to avoid obstacles.

## Path Optimization

We want calculate the optimal path to take.

## Acceleration / Braking

We want to be able to race as quckly as possible.

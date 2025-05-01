#!/usr/bin/env python3
"""
dummy_stereo.py

A simple script to test two cameras in unison by computing and displaying
a raw disparity map without calibration/rectification.
"""
import cv2
import numpy as np

def main():
    # Open left and right cameras (indexes may vary)
    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(1)

    # Check if cameras opened
    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Error: Cannot open one or both cameras")
        return

    # Set resolution (optional)
    width, height, fps = 640, 480, 15
    for cap in (cap_left, cap_right):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

    # Create StereoBM matcher
    num_disparities = 16 * 6  # must be divisible by 16
    block_size = 15
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)

    print("Press 'q' to quit.")

    while True:
        # Capture frames
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()
        if not ret_l or not ret_r:
            print("Failed to grab frames")
            break

        # Convert to grayscale
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        # Compute disparity (raw)
        disp = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

        # Normalize for display
        disp_display = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
        disp_display = np.uint8(disp_display)

        # Show images
        cv2.imshow('Left', frame_l)
        cv2.imshow('Right', frame_r)
        cv2.imshow('Disparity', disp_display)

        # Break on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

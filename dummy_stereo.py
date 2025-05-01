#!/usr/bin/env python3
"""
dummy_stereo.py

A simple script to test two cameras in unison by computing and displaying
an unrectified, raw disparity map without calibration. Captures from:
 - Pi Camera (CSI) via Picamera2
 - USB camera via OpenCV
Press 'q' to quit.
"""

from picamera2 import Picamera2
import cv2
import numpy as np


def main():
    # Initialize Pi Camera (left)
    picam2 = Picamera2()
    # Configure for BGR preview at 640x480
    preview_config = picam2.create_preview_configuration(
        main={"format": "BGR888", "size": (640, 480)}
    )
    picam2.configure(preview_config)
    picam2.start()

    # Initialize USB Camera (right)
    cap_right = cv2.VideoCapture(1)
    if not cap_right.isOpened():
        print("Error: Cannot open USB camera")
        return

    # Set USB camera resolution and FPS
    width, height, fps = 640, 480, 15
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap_right.set(cv2.CAP_PROP_FPS, fps)

    # Create StereoBM matcher (raw, uncalibrated)
    num_disparities = 16 * 6  # must be divisible by 16
    block_size = 15
    stereo = cv2.StereoBM_create(
        numDisparities=num_disparities,
        blockSize=block_size
    )

    print("Press 'q' to quit.")

    try:
        while True:
            # Capture left frame (Pi Camera)
            frame_l = picam2.capture_array()

            # Capture right frame (USB Camera)
            ret_r, frame_r = cap_right.read()
            if not ret_r:
                print("Failed to grab right frame")
                break

            # Convert to grayscale
            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

            # Compute disparity (raw)
            disp = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

            # Normalize for display
            disp_display = cv2.normalize(
                disp, None, 0, 255, cv2.NORM_MINMAX
            )
            disp_display = np.uint8(disp_display)

            # Show images
            cv2.imshow('Left (Pi Camera)', frame_l)
            cv2.imshow('Right (USB Camera)', frame_r)
            cv2.imshow('Disparity', disp_display)

            # Break on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Cleanup
        cap_right.release()
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
```

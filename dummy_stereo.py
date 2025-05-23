#!/usr/bin/env python3
"""
dummy_stereo.py

A simple script to test two Pi Camera Module 2s in unison by computing and displaying
an unrectified, raw disparity map without calibration. Press 'q' to quit.
"""

from picamera2 import Picamera2
import cv2
import numpy as np
import time

def main(stop_event=None):
    # Lower resolution and framerate for stability
    preview_config = {
        "main": {"format": "BGR888", "size": (320, 240)},
        "controls": {"FrameDurationLimits": (100000, 100000)}  # 10 FPS (1e6/10)
    }

    # Initialize both Pi Cameras
    picam_left = Picamera2(0)   # Camera on CSI0
    picam_right = Picamera2(1)  # Camera on CSI1

    picam_left.configure(picam_left.create_preview_configuration(**preview_config))
    picam_right.configure(picam_right.create_preview_configuration(**preview_config))

    picam_left.start()
    time.sleep(0.2)
    picam_right.start()
    time.sleep(0.2)

    # Create StereoBM matcher (raw, uncalibrated)
    num_disparities = 16 * 4  # must be divisible by 16
    block_size = 15
    stereo = cv2.StereoBM_create(
        numDisparities=num_disparities,
        blockSize=block_size
    )

    print("Press 'q' to quit.")

    try:
        while True:
            if stop_event and stop_event.is_set():
                break
            # Capture left and right frames
            frame_l = picam_left.capture_array()
            frame_r = picam_right.capture_array()

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
            cv2.imshow('Left (Pi Camera 0)', frame_l)
            cv2.imshow('Right (Pi Camera 1)', frame_r)
            cv2.imshow('Disparity', disp_display)

            # Break on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if stop_event:
                    stop_event.set()
                break
    finally:
        # Cleanup
        picam_left.stop()
        picam_right.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

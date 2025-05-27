#!/usr/bin/env python3
"""
stereo_pi_integration.py

Unified stereo vision script for Pi Camera Module 2s, supporting:
- Raw or calibrated (rectified) stereo
- Normalized disparity/probability map output
- Optional display
"""

from picamera2 import Picamera2
import cv2
import numpy as np
import time
import argparse
import yaml
import os

# Set this variable to True to show GUI, False to run headless
show_gui = False

class StereoPiVision:
    def __init__(self, width=320, height=240, fps=5, calib_file=None, num_disparities=64, block_size=15):
        self.width = width
        self.height = height
        self.fps = fps
        self.period = 1.0 / fps
        self.last_t = time.time() - self.period
        self.calib_loaded = False

        # Camera config
        preview_config = {
            "main": {"format": "BGR888", "size": (width, height)},
            "controls": {"FrameDurationLimits": (int(1e6/fps), int(1e6/fps))}
        }
        self.picam_left = Picamera2(0)
        self.picam_right = Picamera2(1)
        self.picam_left.configure(self.picam_left.create_preview_configuration(**preview_config))
        self.picam_right.configure(self.picam_right.create_preview_configuration(**preview_config))
        self.picam_left.start()
        time.sleep(0.2)
        self.picam_right.start()
        time.sleep(0.2)

        # Stereo matcher
        self.matcher = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)

        # Calibration (optional)
        if calib_file:
            with open(calib_file, 'r') as f:
                cal = yaml.safe_load(f)
            self.M1, self.d1 = np.array(cal['M1']), np.array(cal['d1'])
            self.M2, self.d2 = np.array(cal['M2']), np.array(cal['d2'])
            self.R, self.T   = np.array(cal['R']), np.array(cal['T'])
            R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
                self.M1, self.d1, self.M2, self.d2, (width, height), self.R, self.T,
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
            )
            self.map1x, self.map1y = cv2.initUndistortRectifyMap(
                self.M1, self.d1, R1, P1, (width, height), cv2.CV_32FC1
            )
            self.map2x, self.map2y = cv2.initUndistortRectifyMap(
                self.M2, self.d2, R2, P2, (width, height), cv2.CV_32FC1
            )
            self.calib_loaded = True

    def grab_disparity(self, normalize=True):
        # Maintain target FPS
        now = time.time()
        dt = now - self.last_t
        if dt < self.period:
            time.sleep(self.period - dt)
        self.last_t = time.time()

        # Capture
        frame_l = self.picam_left.capture_array()
        frame_r = self.picam_right.capture_array()

        # Rectify if calibration loaded
        if self.calib_loaded:
            frame_l = cv2.remap(frame_l, self.map1x, self.map1y, cv2.INTER_LINEAR)
            frame_r = cv2.remap(frame_r, self.map2x, self.map2y, cv2.INTER_LINEAR)

        # Grayscale
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        # Disparity
        disp = self.matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0

        if normalize:
            dmin, dmax = np.nanmin(disp), np.nanmax(disp)
            disp_norm = np.clip((disp - dmin) / (dmax - dmin + 1e-6), 0, 1)
            return disp_norm
        else:
            return disp

    def stop(self):
        self.picam_left.stop()
        self.picam_right.stop()

def main():
    global show_gui
    parser = argparse.ArgumentParser(description="Pi Camera Stereo Vision Integration")
    parser.add_argument("--calib", help="stereo_calib.yaml (optional)")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--disp-show", action="store_true", help="Display disparity/probability map")
    parser.add_argument("--headless", action="store_true", help="Run without GUI windows")
    parser.add_argument("--out", help="Save .npy output (optional)")
    args = parser.parse_args()

    # Override show_gui based on command-line arguments
    if args.headless:
        show_gui = False
    elif args.disp_show:
        show_gui = True

    sv = StereoPiVision(
        width=args.width,
        height=args.height,
        fps=args.fps,
        calib_file=args.calib
    )

    try:
        while True:
            prob = sv.grab_disparity(normalize=True)
            if args.out:
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                np.save(args.out, prob)
                print(f"Saved probability map → {args.out}")
            if show_gui:
                cv2.imshow("probability map", prob)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Probability map shape:", prob.shape)
                # Optionally print summary stats instead of the whole array
                print("min:", np.min(prob), "max:", np.max(prob), "mean:", np.mean(prob))
                time.sleep(0.5)
    finally:
        sv.stop()
        if show_gui:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
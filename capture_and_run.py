#!/usr/bin/env python3
"""
capture_and_run.py

Capture a stereo pair from two cameras, save to disk, then run stereo_vision.py
to compute a probability map.
"""

import cv2
import subprocess
import argparse
import time
import os
import sys

def capture_image(cam_index, filename, width, height):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {cam_index}", file=sys.stderr)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Warm up camera
    time.sleep(1.0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Error: Failed to capture image from camera {cam_index}", file=sys.stderr)
        sys.exit(1)
    cv2.imwrite(filename, frame)


def main():
    parser = argparse.ArgumentParser(
        description="Capture stereo images and run stereo_vision.py"
    )
    parser.add_argument("--calib", required=True, help="Path to stereo_calib.yaml")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--left-index", type=int, default=0, help="Left camera index")
    parser.add_argument("--right-index", type=int, default=1, help="Right camera index")
    parser.add_argument("--out", default="maps/prob.npy", help="Output .npy path")
    parser.add_argument("--disp-show", action="store_true", help="Display the probability map")
    parser.add_argument("--mode", choices=["files","realtime"], default="files", help="Mode for stereo_vision.py")
    args = parser.parse_args()

    left_img = "left.jpg"
    right_img = "right.jpg"
    # Ensure output directory exists
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    print("Capturing left image...")
    capture_image(args.left_index, left_img, args.width, args.height)
    print("Capturing right image...")
    capture_image(args.right_index, right_img, args.width, args.height)

    cmd = [
        sys.executable, "stereo_vision.py",
        "--calib", args.calib,
        "--mode", args.mode,
        "--left", left_img,
        "--right", right_img,
        "--out", args.out,
    ]
    if args.disp_show:
        cmd.append("--disp-show")

    print("Running stereo_vision.py...")
    subprocess.run(cmd, check=True)
    print(f"Probability map saved to {args.out}")

if __name__ == "__main__":
    main()

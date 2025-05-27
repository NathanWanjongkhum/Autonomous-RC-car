#!/usr/bin/env python3
"""
calibrate_stereo.py

Capture chessboard images and compute stereo calibration parameters.
Outputs a YAML file with M1, d1, M2, d2, R, T, E, F.
"""
import cv2
import numpy as np
import glob
import yaml
import argparse
import os

def find_image_points(image_paths, pattern_size):
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    # Optionally, multiply by square_size if you want real-world units

    objpoints = []  # 3d in real world space
    imgpoints = []  # 2d image points
    img_shape = None

    for fname in image_paths:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, pattern_size,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            # refine as needed
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                        (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))
            objpoints.append(objp)
            imgpoints.append(corners2)
        else:
            print(f"Warning: Chessboard not found in {fname}")
    return objpoints, imgpoints, img_shape


def main():
    parser = argparse.ArgumentParser(description="Stereo camera calibration")
    parser.add_argument("--left_dir",  required=True, help="Directory with left images (*.jpg)")
    parser.add_argument("--right_dir", required=True, help="Directory with right images (*.jpg)")
    parser.add_argument("--pattern_cols", type=int, default=9, help="Number of chessboard columns")
    parser.add_argument("--pattern_rows", type=int, default=6, help="Number of chessboard rows")
    parser.add_argument("--square_size", type=float, default=1.0, help="Size of one square (arbitrary units)")
    parser.add_argument("--output", required=True, help="Output YAML filename")
    args = parser.parse_args()

    # Gather image paths
    left_images  = sorted(glob.glob(os.path.join(args.left_dir,  '*.jpg')))
    right_images = sorted(glob.glob(os.path.join(args.right_dir, '*.jpg')))
    if len(left_images) != len(right_images):
        print("Error: Unequal number of left/right images.")
        return

    pattern_size = (args.pattern_cols, args.pattern_rows)
    # Find image points
    objpoints, imgpoints_l, img_shape = find_image_points(left_images, pattern_size)
    _,        imgpoints_r, _         = find_image_points(right_images, pattern_size)

    # Calibrate individual cameras
    ret_l, M1, d1, r1, t1 = cv2.calibrateCamera(
        objpoints, imgpoints_l, img_shape, None, None)
    ret_r, M2, d2, r2, t2 = cv2.calibrateCamera(
        objpoints, imgpoints_r, img_shape, None, None)

    # Stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TermCriteria_MAX_ITER + cv2.TermCriteria_EPS, 100, 1e-5)
    ret_s, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        M1, d1, M2, d2, img_shape,
        criteria=criteria, flags=flags)

    # Save to YAML
    data = {
        'M1': M1.tolist(), 'd1': d1.tolist(),
        'M2': M2.tolist(), 'd2': d2.tolist(),
        'R':  R.tolist(),  'T':  T.tolist(),
        'E':  E.tolist(),  'F':  F.tolist()
    }
    with open(args.output, 'w') as f:
        yaml.dump(data, f)
    print(f"Stereo calibration saved to {args.output}")

if __name__ == '__main__':
    main()

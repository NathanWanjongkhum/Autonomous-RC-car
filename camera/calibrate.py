import cv2
import numpy as np
import glob
import yaml
import os

# To capture from pi cameras:
#python3 calibrate.py --capture --left_dir ./left --right_dir ./right --num_pairs 20
#To calibrate using captured images:
#python3 calibrate.py --left_dir ./left --right_dir ./right --pattern_cols 9 --pattern_rows 6 --output stereo_calib.yaml


def find_image_points(image_paths, pattern_size):
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    img_shape = None

    for fname in image_paths:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret:
            corners2 = cv2.cornerSubPix(
                gray, corners, (11,11), (-1,-1),
                (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
            )
            objpoints.append(objp)
            imgpoints.append(corners2)
        else:
            print(f"Warning: Chessboard not found in {fname}")
    return objpoints, imgpoints, img_shape

def capture_stereo_images(left_dir, right_dir, width=640, height=480, num_pairs=20):
    from picamera2 import Picamera2
    import time

    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    picam_left = Picamera2(0)
    picam_right = Picamera2(1)
    config = {
        "main": {"format": "BGR888", "size": (width, height)}
    }
    picam_left.configure(picam_left.create_preview_configuration(**config))
    picam_right.configure(picam_right.create_preview_configuration(**config))
    picam_left.start()
    picam_right.start()
    time.sleep(0.5)

    print("Press SPACE to capture a stereo pair, or ESC to finish.")
    count = 0
    while count < num_pairs:
        frame_l = picam_left.capture_array()
        frame_r = picam_right.capture_array()
        cv2.imshow("Left", frame_l)
        cv2.imshow("Right", frame_r)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            left_path = os.path.join(left_dir, f"left_{count:02d}.jpg")
            right_path = os.path.join(right_dir, f"right_{count:02d}.jpg")
            cv2.imwrite(left_path, frame_l)
            cv2.imwrite(right_path, frame_r)
            print(f"Saved pair {count}: {left_path}, {right_path}")
            count += 1
        time.sleep(0.1)
    picam_left.stop()
    picam_right.stop()
    cv2.destroyAllWindows()

def calibrate_stereo(
    left_dir, right_dir, pattern_cols, pattern_rows, square_size, output_yaml
):
    left_images  = sorted(glob.glob(os.path.join(left_dir,  '*.jpg')))
    right_images = sorted(glob.glob(os.path.join(right_dir, '*.jpg')))
    if len(left_images) != len(right_images):
        raise ValueError("Unequal number of left/right images.")

    pattern_size = (pattern_cols, pattern_rows)
    objpoints, imgpoints_l, img_shape = find_image_points(left_images, pattern_size)
    _,        imgpoints_r, _         = find_image_points(right_images, pattern_size)

    # Calibrate individual cameras
    ret_l, M1, d1, r1, t1 = cv2.calibrateCamera(objpoints, imgpoints_l, img_shape, None, None)
    ret_r, M2, d2, r2, t2 = cv2.calibrateCamera(objpoints, imgpoints_r, img_shape, None, None)

    # Stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TermCriteria_MAX_ITER + cv2.TermCriteria_EPS, 100, 1e-5)
    ret_s, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        M1, d1, M2, d2, img_shape,
        criteria=criteria, flags=flags
    )

    # Save to YAML
    data = {
        'M1': M1.tolist(), 'd1': d1.tolist(),
        'M2': M2.tolist(), 'd2': d2.tolist(),
        'R':  R.tolist(),  'T':  T.tolist(),
        'E':  E.tolist(),  'F':  F.tolist()
    }
    with open(output_yaml, 'w') as f:
        yaml.dump(data, f)
    print(f"Stereo calibration saved to {output_yaml}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Stereo camera calibration with Pi Cameras")
    parser.add_argument("--capture", action="store_true", help="Capture images from Pi Cameras")
    parser.add_argument("--left_dir",  required=True, help="Directory for left images (*.jpg)")
    parser.add_argument("--right_dir", required=True, help="Directory for right images (*.jpg)")
    parser.add_argument("--pattern_cols", type=int, default=9, help="Number of chessboard columns")
    parser.add_argument("--pattern_rows", type=int, default=6, help="Number of chessboard rows")
    parser.add_argument("--square_size", type=float, default=1.0, help="Size of one square (arbitrary units)")
    parser.add_argument("--output", required=False, help="Output YAML filename")
    parser.add_argument("--num_pairs", type=int, default=20, help="Number of stereo pairs to capture")
    args = parser.parse_args()

    if args.capture:
        capture_stereo_images(args.left_dir, args.right_dir, num_pairs=args.num_pairs)
    elif args.output:
        calibrate_stereo(
            args.left_dir, args.right_dir,
            args.pattern_cols, args.pattern_rows,
            args.square_size, args.output
        )
    else:
        print("Specify --capture to capture images or --output to calibrate.")
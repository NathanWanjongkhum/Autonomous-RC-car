#!/usr/bin/env python3
# stereo_vision.py
import cv2
import numpy as np
import time
import yaml
import argparse
import os

class StereoVision:
    def __init__(
        self,
        calib_file: str,
        width: int,
        height: int,
        fps: float,
        num_disparities: int = 64,
        block_size: int = 15,
        cam_left_index: int = 0,
        cam_right_index: int = 1,
    ):
        # Load calibration
        with open(calib_file, 'r') as f:
            cal = yaml.safe_load(f)
        self.M1, self.d1 = np.array(cal['M1']), np.array(cal['d1'])
        self.M2, self.d2 = np.array(cal['M2']), np.array(cal['d2'])
        self.R, self.T   = np.array(cal['R']), np.array(cal['T'])

        # Precompute rectification maps
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
            self.M1, self.d1,
            self.M2, self.d2,
            (width, height),
            self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.M1, self.d1, R1, P1, (width, height), cv2.CV_32FC1
        )
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.M2, self.d2, R2, P2, (width, height), cv2.CV_32FC1
        )

        # Set up stereo matcher
        self.matcher = cv2.StereoBM_create(
            numDisparities=num_disparities,
            blockSize=block_size
        )

        # For real-time capture
        self.fps    = fps
        self.period = 1.0 / fps
        self.last_t = time.time() - self.period

        self.cap_l = cv2.VideoCapture(cam_left_index)
        self.cap_r = cv2.VideoCapture(cam_right_index)
        for cap in (self.cap_l, self.cap_r):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS,         fps)

    def compute_from_images(self, img1, img2):
        """Given two BGR images, produce a normalized probability map."""
        # rectify
        im1r = cv2.remap(img1, self.map1x, self.map1y, cv2.INTER_LINEAR)
        im2r = cv2.remap(img2, self.map2x, self.map2y, cv2.INTER_LINEAR)
        # gray
        g1 = cv2.cvtColor(im1r, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(im2r, cv2.COLOR_BGR2GRAY)
        # disparity
        disp = self.matcher.compute(g1, g2).astype(np.float32) / 16.0
        # normalize → [0,1]
        dmin, dmax = disp.min(), disp.max()
        disp_norm = np.clip((disp - dmin) / (dmax - dmin + 1e-6), 0, 1)
        return disp_norm

    def grab_realtime(self):
        """Grabs a stereo pair from cameras at target FPS and returns prob map."""
        now = time.time()
        dt  = now - self.last_t
        if dt < self.period:
            time.sleep(self.period - dt)
        self.last_t = time.time()

        ret1, img1 = self.cap_l.read()
        ret2, img2 = self.cap_r.read()
        if not (ret1 and ret2):
            raise RuntimeError("Camera frame grab failed")
        return self.compute_from_images(img1, img2)

    def release(self):
        self.cap_l.release()
        self.cap_r.release()


def main():
    p = argparse.ArgumentParser(
        description="Compute stereo disparity → occupancy‐probability map"
    )
    p.add_argument("--calib",   required=True, help="stereo_calib.yaml")
    p.add_argument("--width",   type=int, default=640)
    p.add_argument("--height",  type=int, default=480)
    p.add_argument("--fps",     type=float, default=5.0)
    p.add_argument("--mode",    choices=("realtime","files"), default="realtime")
    p.add_argument("--left",    help="left image (for files mode)")
    p.add_argument("--right",   help="right image (for files mode)")
    p.add_argument("--out",     help=".npy output path (optional)")
    p.add_argument("--disp-show", action="store_true", help="display map")
    args = p.parse_args()

    sv = StereoVision(
        calib_file=args.calib,
        width=args.width,
        height=args.height,
        fps=args.fps
    )

    try:
        if args.mode == "files":
            if not (args.left and args.right):
                p.error("files mode requires --left and --right")
            im1 = cv2.imread(args.left)
            im2 = cv2.imread(args.right)
            prob = sv.compute_from_images(im1, im2)

        else:  # realtime
            prob = sv.grab_realtime()

        # save if requested
        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            np.save(args.out, prob)
            print(f"Saved probability map → {args.out}")

        # display
        if args.disp_show:
            cv2.imshow("probability map", prob)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(prob)
    finally:
        sv.release()


if __name__ == "__main__":
    main()
import cv2
import numpy as np
import yaml
import time
from picamera2 import Picamera2


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
        img1_rect = cv2.remap(img1, self.map1x, self.map1y, cv2.INTER_LINEAR)
        img2_rect = cv2.remap(img2, self.map2x, self.map2y, cv2.INTER_LINEAR)
        gray1 = cv2.cvtColor(img1_rect, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_rect, cv2.COLOR_BGR2GRAY)
        disp = self.matcher.compute(gray1, gray2).astype(np.float32) / 16.0
        dmin, dmax = np.nanmin(disp), np.nanmax(disp)
        prob = np.clip((disp - dmin) / (dmax - dmin + 1e-6), 0, 1)
        return prob

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

# This is a class for stereo vision using Raspberry Pi cameras.
# To use:
# from camera.stereo import StereoVisionPi
# stereo = StereoVisionPi('stereo_calib.yaml', width=320, height=240, fps=10)
# prob_map = stereo.grab_realtime()

class StereoVisionPi:
    def __init__(self, calib_file: str, width: int, height: int, fps: float):
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
            numDisparities=16,
            blockSize=15
        )

        # For real-time capture
        self.fps    = fps
        self.period = 1.0 / fps
        self.last_t = time.time() - self.period

        self.picam_left = Picamera2(0)
        self.picam_right = Picamera2(1)
        # configure and start both cameras as shown in previous examples

    def compute_from_images(self, img1, img2):
        """Given two BGR images, produce a normalized probability map."""
        img1_rect = cv2.remap(img1, self.map1x, self.map1y, cv2.INTER_LINEAR)
        img2_rect = cv2.remap(img2, self.map2x, self.map2y, cv2.INTER_LINEAR)
        gray1 = cv2.cvtColor(img1_rect, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_rect, cv2.COLOR_BGR2GRAY)
        disp = self.matcher.compute(gray1, gray2).astype(np.float32) / 16.0
        dmin, dmax = np.nanmin(disp), np.nanmax(disp)
        prob = np.clip((disp - dmin) / (dmax - dmin + 1e-6), 0, 1)
        return prob

    def grab_realtime(self):
        frame_l = self.picam_left.capture_array()
        frame_r = self.picam_right.capture_array()
        return self.compute_from_images(frame_l, frame_r)

    def release(self):
        self.picam_left.close()
        self.picam_right.close()
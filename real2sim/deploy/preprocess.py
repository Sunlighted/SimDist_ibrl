"""Real ZED frame -> the exact image the policy saw in sim.

Shared by the colour-correction fit (cluster) and the live policy server (robot
host), so the two can never drift apart.

Geometry (must match env/peg_insertion.py's camera definitions):
  1. undistort with the camera's FACTORY K, D, keeping newK = K. publish_zed_uvc.py
     streams RAW (unrectified, distorted) frames; the sim cameras are ideal
     pinholes with the factory focal length, so undistort-with-newK=K is the
     frame that matches them. Verified on the real GELLO demos: the undistorted
     frame lines up with the splat replay to ~15-30 px / 448 on both cameras
     (scripts/compare_real_sim_demos.py, viz/real_vs_sim/replay_*.png).
  2. square crop centred on the PRINCIPAL POINT (not the image centre), side =
     2 * the shortest distance from (cx, cy) to an edge. MuJoCo renders a
     centred pinhole, so the crop has to put cx, cy in the middle.
  3. resize to the policy resolution with INTER_AREA.

Colour: the splats were trained on PHONE photos, so a ZED frame differs in
white balance, exposure and saturation. `apply_color` maps ZED RGB -> splat RGB
with a per-camera 3x4 affine fitted by real2sim/deploy/fit_color_correction.py
(stored in color_correction.json next to this file).
"""
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_COLOR_JSON = os.path.join(_HERE, "color_correction.json")

# ROS camera name -> sim camera name the policy was trained on
SIM_CAMERA = {"scene": "zedview", "wrist": "robot0_zedm_wrist"}


def decode_jpeg_rgb(buf, reduce=1):
    """reduce=2 decodes at half resolution inside libjpeg (~4x cheaper). The final
    policy image is 224 px from a ~703 px crop, so half-res (~351 px) loses nothing."""
    import cv2
    flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4}[reduce]
    im = cv2.imdecode(np.frombuffer(bytes(buf), np.uint8), flag)
    if im is None:
        raise ValueError("could not decode JPEG")
    return im[..., ::-1]                                          # BGR -> RGB


class Rectifier:
    """Precomputes the undistortion maps once per (K, D, size)."""

    def __init__(self, K, D, width, height, out_size, reduce=1):
        """K, D, width, height are the camera's FULL-resolution calibration; with
        reduce=r the maps are built for the 1/r-scaled frame decode_jpeg_rgb(.., r)
        returns (distortion coefficients are resolution-independent; K scales)."""
        import cv2
        self.K = np.asarray(K, np.float64).reshape(3, 3).copy()
        self.K[:2] /= reduce
        self.D = np.asarray(D, np.float64).ravel()
        self.w, self.h, self.S = int(width) // reduce, int(height) // reduce, int(out_size)
        self.reduce = reduce
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K, self.D, None, self.K, (self.w, self.h), cv2.CV_16SC2)
        cx, cy = self.K[0, 2], self.K[1, 2]
        half = int(np.floor(min(cx, self.w - cx, cy, self.h - cy)))
        self.x0, self.y0 = int(round(cx - half)), int(round(cy - half))
        self.side = 2 * half

    def __call__(self, rgb, undistort=True):
        import cv2
        if rgb.shape[1] != self.w or rgb.shape[0] != self.h:
            raise ValueError(f"frame {rgb.shape[1]}x{rgb.shape[0]} != calibrated "
                             f"{self.w}x{self.h} (wrong ZED mode?)")
        im = cv2.remap(rgb, self.map1, self.map2, cv2.INTER_LINEAR) if undistort else rgb
        im = im[self.y0:self.y0 + self.side, self.x0:self.x0 + self.side]
        return cv2.resize(im, (self.S, self.S), interpolation=cv2.INTER_AREA)


def load_color(path=DEFAULT_COLOR_JSON):
    """-> {ros_camera_name: 3x4 float32 matrix} or {} if no file."""
    if not path or not os.path.exists(path):
        return {}
    d = json.load(open(path))
    return {k: np.asarray(v["M"], np.float32) for k, v in d["cameras"].items()}


def apply_color(rgb_u8, M):
    """ZED RGB (uint8 HxWx3) -> splat-like RGB via out = M[:, :3] @ rgb + M[:, 3]."""
    if M is None:
        return rgb_u8
    x = rgb_u8.reshape(-1, 3).astype(np.float32) / 255.0
    y = x @ M[:, :3].T + M[:, 3]
    return (np.clip(y, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).reshape(rgb_u8.shape)

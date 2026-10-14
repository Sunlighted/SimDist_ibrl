"""Real ZED frame -> the exact image the policy saw in sim.

Shared by the colour-correction fit (cluster) and the live policy server (robot
host), so the two can never drift apart.

Geometry (must match the sim camera the policy was trained on):
  1. undistort with the camera's K, D, keeping newK = K. publish_zed_uvc.py
     streams RAW (unrectified, distorted) frames; the sim cameras are ideal
     pinholes, so undistort-with-newK=K is the frame that matches them.
     K, D = the camera_info FACTORY values unless the cell gives the camera a
     `real_intrinsics` file (env/sim2real.py RealCameraConfig): the back-of-hand
     ZED-M's factory D is wrong (calib_20261010/README.md), see `camera_intrinsics`.
  2. square crop centred on the PRINCIPAL POINT (not the image centre), side =
     2 * fy * tan(fovy_sim / 2): exactly the field of view of the sim camera.
     MuJoCo renders a centred pinhole, so the crop has to put cx, cy in the middle.
     ⚠️ Until 2026-10-11 the side was "the largest square about (cx, cy)". That
     equals the sim fovy only for PegInsertion_D0 (whose fovy 67.010 / 48.977 were
     DERIVED from that square); on the Threading cell (fovy 50) it fed the policy a
     67 deg view, 1.42x too wide. The fovy is now a required argument.
  3. resize to the policy resolution with INTER_AREA.
  Steps 1-2 are ONE remap into the crop (sub-pixel exact). scripts/convert_real_teleop.py
  (the offline stage-2 converter) calls this same class; scripts/test_preprocess_crop.py
  checks that projected 3D points land on the sim pinhole's pixels.

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

# ROS camera name -> sim camera name the policy was trained on (PegInsertion_D0).
SIM_CAMERA = {"scene": "zedview", "wrist": "robot0_zedm_wrist"}
# ... and on the RealCell tasks (Threading_RealCell etc., env/sim2real.py cameras).
SIM_CAMERA_REALCELL = {"scene": "agentview", "wrist": "robot0_eye_in_hand"}


def sim_camera_map(rl_cameras):
    """ROS camera -> sim camera for a policy trained on `rl_cameras`."""
    rl = set(rl_cameras)
    for m in (SIM_CAMERA, SIM_CAMERA_REALCELL):
        if set(m.values()) <= rl:
            return dict(m)
    raise ValueError(f"no known ROS<->sim camera mapping covers {sorted(rl)}")


def crop_side(fy, fovy_deg):
    """Square side (px) whose vertical FOV through a pinhole of focal fy is fovy."""
    return int(round(2.0 * float(fy) * np.tan(np.radians(float(fovy_deg)) / 2.0)))


def peg_fovy():
    """sim camera -> fovy of PegInsertion_D0 (env/peg_insertion.py is the source)."""
    from env import peg_insertion as pi
    return {"zedview": float(pi.ZEDVIEW_FOVY), "robot0_zedm_wrist": float(pi.ZEDM_WRIST_FOVY)}


def camera_intrinsics(cell_camera, K, D, width, height):
    """-> (K, D) the real frames must be undistorted with.

    `cell_camera` is the env.sim2real.RealCameraConfig of the sim camera (or None);
    K, D, width, height are what the camera itself reports (camera_info / the
    recorder's attrs). A `real_intrinsics` file on the cell camera wins: it is a
    re-calibration of exactly this camera, and the factory values it replaces are
    known to be wrong."""
    K = np.asarray(K, np.float64).reshape(3, 3)
    D = np.asarray(D, np.float64).ravel()
    path = getattr(cell_camera, "real_intrinsics", "") if cell_camera is not None else ""
    if not path:
        return K, D
    j = json.load(open(path))
    if tuple(j["size"]) != (int(width), int(height)):
        raise ValueError(f"{path} was calibrated at {j['size']}, the camera streams "
                         f"{width}x{height} (wrong ZED mode?)")
    return np.asarray(j["K"], np.float64).reshape(3, 3), np.asarray(j["D"], np.float64).ravel()


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

    def __init__(self, K, D, width, height, out_size, fovy_deg, reduce=1):
        """K, D, width, height are the camera's FULL-resolution calibration; with
        reduce=r the maps are built for the 1/r-scaled frame decode_jpeg_rgb(.., r)
        returns (distortion coefficients are resolution-independent; K scales).
        fovy_deg = the SIM camera's vertical fovy (cell yaml / task constants)."""
        import cv2
        self.K = np.asarray(K, np.float64).reshape(3, 3).copy()
        if reduce != 1:                     # pixel-centre convention: u' = (u + .5)/r - .5
            self.K[:2, :2] /= reduce
            self.K[:2, 2] = (self.K[:2, 2] + 0.5) / reduce - 0.5
        self.D = np.asarray(D, np.float64).ravel()
        self.w, self.h, self.S = int(width) // reduce, int(height) // reduce, int(out_size)
        self.reduce, self.fovy = reduce, float(fovy_deg)
        # Undistort STRAIGHT INTO the crop: the output camera is the sim pinhole
        # itself -- principal point at the crop centre, focal giving exactly fovy.
        # An integer crop of a full-frame undistort instead leaves round(cx) - cx
        # (up to 0.5 px, 0.2 px @ 224) of constant offset and a 0.1% scale error.
        self.side = crop_side(self.K[1, 1], self.fovy)
        f = (self.side / 2.0) / np.tan(np.radians(self.fovy) / 2.0)
        c = (self.side - 1) / 2.0                     # pixel-index coords of the centre
        self.newK = np.array([[f, 0.0, c], [0.0, f, c], [0.0, 0.0, 1.0]])
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K, self.D, None, self.newK, (self.side, self.side), cv2.CV_32FC1)
        # Outside the frame remap fills black, as convert_real_teleop's padding did.
        # Never happens on the cells we have (fovy below the largest square).
        self.padded = bool(min(self.K[0, 2], self.w - 1 - self.K[0, 2],
                               self.K[1, 2], self.h - 1 - self.K[1, 2]) < c)

    def __call__(self, rgb, undistort=True):
        """undistort=False: same crop on an already-rectified (pinhole K) frame."""
        import cv2
        if rgb.shape[1] != self.w or rgb.shape[0] != self.h:
            raise ValueError(f"frame {rgb.shape[1]}x{rgb.shape[0]} != calibrated "
                             f"{self.w}x{self.h} (wrong ZED mode?)")
        if undistort:
            im = cv2.remap(rgb, self.map1, self.map2, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            A = self.newK @ np.linalg.inv(self.K)
            im = cv2.warpAffine(rgb, A[:2], (self.side, self.side), flags=cv2.INTER_LINEAR)
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

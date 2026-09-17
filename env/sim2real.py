"""Sim-to-real CELL configuration: make the sim scene match a MEASURED real cell.

This is the visual/geometric counterpart to `env/domain_shift.py`. That module
randomises PHYSICS (friction / mass / solref / solimp / gripper kp / object
geometry); nothing in this repo ever touched the RENDERED scene, so a pixel
policy trained here has only ever seen one table texture, one lighting rig, one
camera pose and zero shadows (docs/simdist.md 44: the reference recipe uses four
augmentations, we shipped none).

Two DISTINCT jobs, deliberately kept in one config but separate fields:

  1. MATCH  -- quantities you MEASURED on the real robot (table height, camera
     extrinsics/intrinsics, table colour). These are point estimates; sim should
     reproduce them exactly. Randomising a measured quantity wastes capacity.
  2. RANDOMISE -- quantities that are uncontrolled or drift (lighting, exposure,
     background, small mounting errors). These get a range AROUND the matched
     nominal.

The task's tolerance window is ~1.8 mm (findings.md 27/39), so geometry belongs
in group 1: at that scale a randomisation range wide enough to "cover" a
mis-measured table height is also wide enough to destroy the signal.

--------------------------------------------------------------------------- why
EVERY field defaults to the value that makes this module a NO-OP, and `enabled`
gates the whole thing, so every existing state run, every existing pixel ckpt and
every replay generated so far is byte-for-byte unaffected. Verify with
scripts/test_sim2real.py, which asserts tensor-equality of the touched model
arrays when enabled=0. (Iron law: a hook that exists but is never called costs a
multi-day run and reports nothing -- 44's augment_fn was exactly that.)

------------------------------------------------------------------- conventions
* Camera poses are given in the frame named by `frame`:
    "base" -- the robot base frame (robot0_base body). This is what a hand-eye /
              ChArUco calibration chain reports for a statically mounted camera.
    "ee"   -- the end-effector frame, for a wrist camera. IMPORTANT: "EE frame"
              is ambiguous across stacks. Here it means robosuite's
              `gripper0_grip_site`. Franka's O_T_EE depends on F_T_EE/NE_T_EE and
              is generally NOT the same point; convert before filling this in, or
              the wrist extrinsic will be wrong by the TCP offset.
* `quat` is (w, x, y, z) in the OPENCV optical convention: +x right, +y DOWN,
  +z FORWARD along the optical axis. MuJoCo cameras use +y UP and look down -z;
  the conversion is applied internally (R_mujoco = R_opencv @ diag(1,-1,-1)).
  Getting this wrong flips the image vertically and is not otherwise detectable.
* `look_at` is a FALLBACK for when a calibration only reports the optical AXIS
  (2 DOF) and not the roll about it. It synthesises the missing DOF from `up`.
  A camera rolled 10 degrees produces an image rolled 10 degrees, which no amount
  of training fixes -- so treat look_at as a placeholder and replace it with the
  measured `quat` before generating any dataset.
* `fovy` is the VERTICAL field of view in degrees, matching MuJoCo. If you have
  intrinsics instead, set `fy`/`res` and it is derived as
  fovy = 2*atan(res / (2*fy)). Renders are square so fx must equal fy; if the
  real stream is 16:9 and you crop to square, the crop must preserve the vertical
  FOV or the pixel->geometry mapping is globally wrong.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# --------------------------------------------------------------------- helpers
def mat2quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> (w,x,y,z). Shepperd's method (branch on the largest
    diagonal term) so it stays conditioned for all rotations; the naive
    w = sqrt(1+trace)/2 form loses precision near 180 degrees."""
    m = np.asarray(R, dtype=np.float64)
    t = np.trace(m)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                         (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                         (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                         0.25 * s, (m[1, 2] + m[2, 1]) / s])
    s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                     (m[1, 2] + m[2, 1]) / s, 0.25 * s])


def quat_wxyz2mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    assert n > 1e-8, f"degenerate quaternion {q}"
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def lookat_R_opencv(fwd, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Optical axis + an up HINT -> OpenCV rotation (columns = x_right, y_down,
    z_forward). Only 2 of the 3 DOF come from `fwd`; `up` invents the third."""
    z = np.asarray(fwd, dtype=np.float64)
    z = z / np.linalg.norm(z)
    u = np.asarray(up, dtype=np.float64)
    # x = z X up, NOT up X z. OpenCV's +y points DOWN, so the right-handed frame
    # (x, y=z*x, z) only comes out with the image upright for this order; the
    # flipped order yields a 180-degree roll -- a picture that is upside down and
    # mirrored, which renders perfectly happily and is easy to stare past.
    # Guarded by test_sim2real.py::test_lookat_matches_known_camera, which checks
    # against sim's own agentview (a camera whose correct quat is in the XML).
    x = np.cross(z, u)
    nx = np.linalg.norm(x)
    assert nx > 1e-6, "look_at direction is parallel to `up`; pick another up"
    x = x / nx
    return np.stack([x, np.cross(z, x), z], axis=1)


OPENCV_TO_MUJOCO = np.diag([1.0, -1.0, -1.0])


# --------------------------------------------------------------------- configs
@dataclass
class RealCameraConfig:
    """One physically-measured camera. Default = every field inert."""
    name: str = ""                  # sim camera name, e.g. "agentview"
    frame: str = "base"             # "base" (statically mounted) | "ee" (wrist)
    pos: tuple = (0.0, 0.0, 0.0)
    # Orientation, OpenCV optical convention (w,x,y,z). PREFERRED.
    quat: Optional[tuple] = None
    # Fallback when only the optical axis is known -- see the module docstring.
    # Expressed in the SAME frame as `pos`. Either a direction (`look_dir`) or a
    # point to aim at (`look_at`); the point form is easier to read off a layout.
    look_dir: Optional[tuple] = None
    look_at: Optional[tuple] = None
    up: tuple = (0.0, 0.0, 1.0)
    # Intrinsics. Set EITHER fovy (degrees, vertical) OR fy+res.
    fovy: float = 0.0               # 0 = leave the sim value alone
    fy: float = 0.0                 # pixels, from calibration
    res: int = 0                    # image height the fy was calibrated at

    def resolved_fovy(self) -> float:
        """0 => do not touch the camera's fovy."""
        if self.fy > 0:
            assert self.res > 0, f"camera {self.name!r}: fy set but res is 0"
            return float(np.degrees(2.0 * np.arctan(self.res / (2.0 * self.fy))))
        return float(self.fovy)

    def rotation_opencv(self) -> Optional[np.ndarray]:
        """R such that a point p_cam = R.T @ (p_frame - pos). None = leave as-is."""
        n_given = sum(x is not None for x in (self.quat, self.look_dir, self.look_at))
        assert n_given <= 1, (
            f"camera {self.name!r}: give at most one of quat / look_dir / look_at")
        if self.quat is not None:
            return quat_wxyz2mat(self.quat)
        if self.look_dir is not None:
            return lookat_R_opencv(self.look_dir, self.up)
        if self.look_at is not None:
            d = np.asarray(self.look_at, dtype=np.float64) - np.asarray(self.pos)
            return lookat_R_opencv(d, self.up)
        return None

    def is_inert(self) -> bool:
        return (not self.name) or (
            self.rotation_opencv() is None
            and self.resolved_fovy() == 0.0
            and tuple(self.pos) == (0.0, 0.0, 0.0))


@dataclass
class Sim2RealConfig:
    # Master gate. 0 => this module does nothing at all (byte-for-byte legacy).
    enabled: int = 0

    # ---------------------------------------------------- 1. MEASURED GEOMETRY
    # Table top height expressed RELATIVE TO THE ROBOT BASE ORIGIN, which is the
    # frame a real calibration reports. None = leave the sim table where it is.
    # NOTE the sim default is NEGATIVE: robosuite mounts the Panda on a 0.912 m
    # column while the table top is at 0.800, i.e. the top sits 112 mm BELOW the
    # base origin. A robot bolted to the bench usually has it ABOVE. Getting the
    # sign wrong moves the whole task by >0.1 m.
    table_above_base: Optional[float] = None
    # Visual-only enlargement of the table, as HALF-sizes (x, y) in metres.
    # (0,0) = leave alone. This writes `table_visual`'s geom_size, which is
    # contype=conaffinity=0, so the collision box and -- critically -- the robot
    # base placement are untouched. Do NOT enlarge via `table_full_size`:
    # threading.py:244 computes the base xpos as a FUNCTION of table_full_size[0],
    # so that route silently walks the robot backwards out of reach.
    # The z half-size is never written, so the table TOP height cannot move here.
    table_visual_half: tuple = (0.0, 0.0)
    # Flat diffuse colour for the table top, 0-1 RGB. None = keep the ceramic
    # texture. Setting it also clears geom_matid so the texture stops overriding.
    table_rgb: Optional[tuple] = None
    # Flat diffuse colour for the TASK OBJECTS (needle + tripod), 0-1 RGB.
    # None = keep the wood texture the MJCF ships with. Real parts are usually
    # 3D-printed in one colour, and the wood grain is a texture the policy can
    # latch onto that simply does not exist on the bench.
    # ⚠️ Object colour is not cosmetic here: it sets the contrast against the
    # table, which is what the whole pixel arm has to resolve at ~1.8 mm. A dark
    # part on a white bench is a much easier problem than a white part on a white
    # bench -- so this must match the real parts, not look pretty.
    object_rgb: Optional[tuple] = None
    # Geom-name prefixes treated as "the objects". MimicGen names every geom of an
    # object with the object's name as prefix, so this catches visual AND
    # collision geoms; recolouring a collision geom is harmless (group 0 is not
    # rendered) and keeps the two in sync if the grouping ever changes.
    object_geom_prefixes: tuple = ("needle_obj", "tripod_obj")

    # Rigid (dx, dy) translation of the WHOLE reset distribution, in the table
    # frame, metres. (0,0) = leave the task's own bounds alone.
    #
    # This is a CELL-LAYOUT quantity, not a task change: it says where on the
    # bench the parts get placed. It moves every object by the same vector, so
    # their relative geometry -- and therefore the task -- is untouched.
    # ⚠️ Shift objects INDIVIDUALLY at your peril: moving only the tripod toward
    # the camera runs it into the needle's sampling box and robosuite throws
    # RandomizationError, and the gain is nil anyway because closing the distance
    # degrades the viewing incidence by about as much as it helps.
    #
    # Measured on the real cell (camera at base (0.60,0.48,0.78), ring opening is
    # 20 mm, hole pixels at fovy 70 / fovy 45):
    #     dy +0.00  d=0.850  4.2x3.6 / 7.2x6.0  incidence 42.1
    #     dy +0.20  d=0.714  5.2x4.2            incidence 53.0
    #     dy +0.30  d=0.659  5.5x4.7 / 9.4x8.0  incidence 59.9   <-- optimum
    #     dy +0.40  d=0.615  5.0x5.7 / 8.4x9.6  incidence 68.0   (past the peak)
    # Reach was verified by driving the OSC to each object: residual <1 mm at
    # every dy tested, on the LIFTED table.
    workspace_offset: tuple = (0.0, 0.0)

    # ----------------------------------------------------- 2. MEASURED CAMERAS
    cameras: tuple = ()              # tuple[RealCameraConfig, ...]

    # ------------------------------------------------------- 3. RANDOMISATION
    # All ranges are AROUND the matched nominal above; 0 = that axis is off.
    dr_table_rgb_tint: float = 0.0      # per-channel +-, on the 0-1 colour
    dr_table_rgb_gain: tuple = (1.0, 1.0)   # multiplicative albedo range
    dr_object_rgb_tint: float = 0.0     # per-channel +-, on object_rgb
    dr_object_rgb_gain: tuple = (1.0, 1.0)
    dr_background_gray: tuple = (0.0, 0.0)  # walls+floor flat gray range; (0,0)=off
    dr_light_diffuse: tuple = (1.0, 1.0)    # multiplier on the nominal diffuse
    dr_light_ambient: tuple = (0.0, 0.0)    # ABSOLUTE ambient range
    dr_light_specular: tuple = (1.0, 1.0)   # multiplier on the nominal specular
    dr_light_dir: float = 0.0           # +- per-axis perturbation of the light dir
    dr_light_pos: float = 0.0           # +- per-axis, metres
    dr_base_height: float = 0.0         # +- metres on the robot base z
    # Camera mounting error, applied ON TOP of the measured extrinsics. This is
    # the residual a calibration cannot remove; your PARK solve quoted
    # 1.0 deg / 11.4 mm, which is the right order for these.
    dr_cam_pos: float = 0.0             # +- metres, per axis
    dr_cam_rot_deg: float = 0.0         # +- degrees, per axis
    dr_cam_fovy_deg: float = 0.0        # +- degrees on the vertical FOV

    def any_dr(self) -> bool:
        return bool(self.enabled) and any([
            self.dr_table_rgb_tint, self.dr_table_rgb_gain != (1.0, 1.0),
            self.dr_object_rgb_tint, self.dr_object_rgb_gain != (1.0, 1.0),
            self.dr_background_gray != (0.0, 0.0),
            self.dr_light_diffuse != (1.0, 1.0), self.dr_light_ambient != (0.0, 0.0),
            self.dr_light_specular != (1.0, 1.0), self.dr_light_dir, self.dr_light_pos,
            self.dr_base_height, self.dr_cam_pos, self.dr_cam_rot_deg,
            self.dr_cam_fovy_deg])


# ---------------------------------------------------------------- application
_WALL_GEOMS = ("wall_leftcorner_visual", "wall_rightcorner_visual",
               "wall_left_visual", "wall_right_visual", "wall_rear_visual",
               "wall_front_visual", "floor")


def table_offset_for(cfg: Sim2RealConfig, base_world_z: float) -> Optional[np.ndarray]:
    """The `table_offset` a MimicGen env must be built with so the top lands at
    `table_above_base` metres above the base origin. Must be applied BEFORE
    reset(): robosuite runs hard_reset=True, so `_load_model` rebuilds the arena
    AND the placement initializer (whose `reference` IS table_offset) from this
    value every episode -- which is exactly why objects follow the table for
    free. Returns None when the height is not being matched."""
    if not cfg.enabled or cfg.table_above_base is None:
        return None
    return np.array([0.0, 0.0, float(base_world_z) + float(cfg.table_above_base)])


def apply_workspace_offset(env, cfg: Sim2RealConfig) -> None:
    """Translate the whole reset distribution by cfg.workspace_offset.

    Wraps `_get_initial_placement_bounds` rather than restating the bounds, so it
    composes with whatever the task (D0/D1) or an InitDistConfig already set, and
    stays correct if those change. Like InitDistConfig this patches the instance
    once: robosuite re-reads the bounds on every hard reset, so it persists.

    Call BEFORE the first reset. Idempotent -- re-applying does not stack, which
    matters because a wrapper may be constructed more than once per process.
    """
    if not cfg.enabled or tuple(cfg.workspace_offset) == (0.0, 0.0):
        return
    if getattr(env, "_s2r_offset_applied", False):
        return
    inner = env._get_initial_placement_bounds
    dx, dy = (float(v) for v in cfg.workspace_offset)

    def shifted():
        out = {}
        for name, b in inner().items():
            b = dict(b)
            b["x"] = (b["x"][0] + dx, b["x"][1] + dx)
            b["y"] = (b["y"][0] + dy, b["y"][1] + dy)
            out[name] = b
        return out

    env._get_initial_placement_bounds = shifted
    env._s2r_offset_applied = True


def _rand(rng, lo_hi):
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    return lo if lo == hi else rng.uniform(lo, hi)


def _small_rot(rng, deg):
    """Random rotation with each axis angle ~ U(-deg, deg), applied as an
    intrinsic XYZ composition. Small-angle so order barely matters, but it is
    fixed anyway to keep a seeded run repeatable."""
    a, b, c = np.radians(rng.uniform(-deg, deg, 3))
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def apply_sim2real(env, cfg: Sim2RealConfig, rng=None, nominal: Optional[dict] = None):
    """Apply the cell config to a FRESHLY RESET robosuite env, in place.

    Call AFTER env.reset() (hard_reset rebuilds the model, so anything written
    before is discarded) and BEFORE pulling the observation -- the frames handed
    back by reset() were rendered with the old scene.

    `nominal` is the dict returned by `snapshot_visual_nominal`; it is required
    whenever a multiplicative DR range is used, so gains compose from the ORIGINAL
    values rather than from the previous episode's already-scaled ones (the same
    trap domain_shift.py solves with restore_nominal)."""
    if not cfg.enabled:
        return
    sim = env.sim if hasattr(env, "sim") else env.env.sim
    m = sim.model
    rng = rng if rng is not None else np.random

    # ---- table visual: size, then colour ---------------------------------
    gid = _geom_id(m, "table_visual")
    if gid is not None:
        hx, hy = cfg.table_visual_half
        if hx > 0 and hy > 0:
            sz = m.geom_size[gid].copy()
            m.geom_size[gid] = np.array([float(hx), float(hy), sz[2]])  # z untouched
        if cfg.table_rgb is not None:
            rgb = np.asarray(cfg.table_rgb, dtype=np.float64)
            if cfg.dr_table_rgb_tint:
                rgb = rgb + rng.uniform(-cfg.dr_table_rgb_tint, cfg.dr_table_rgb_tint, 3)
            if cfg.dr_table_rgb_gain != (1.0, 1.0):
                rgb = rgb * _rand(rng, cfg.dr_table_rgb_gain)
            m.geom_matid[gid] = -1        # drop the ceramic texture
            m.geom_rgba[gid] = np.array([*np.clip(rgb, 0.0, 1.0), 1.0])

    # ---- task objects ------------------------------------------------------
    if cfg.object_rgb is not None:
        base_rgb = np.asarray(cfg.object_rgb, dtype=np.float64)
        for gi in range(m.ngeom):
            nm = _geom_name(m, gi)
            if not nm or not nm.startswith(tuple(cfg.object_geom_prefixes)):
                continue
            rgb = base_rgb
            if cfg.dr_object_rgb_tint:
                rgb = rgb + rng.uniform(-cfg.dr_object_rgb_tint,
                                        cfg.dr_object_rgb_tint, 3)
            if cfg.dr_object_rgb_gain != (1.0, 1.0):
                rgb = rgb * _rand(rng, cfg.dr_object_rgb_gain)
            m.geom_matid[gi] = -1
            m.geom_rgba[gi] = np.array([*np.clip(rgb, 0.0, 1.0),
                                        float(m.geom_rgba[gi][3])])

    # ---- background -------------------------------------------------------
    if cfg.dr_background_gray != (0.0, 0.0):
        for nm in _WALL_GEOMS:
            g = _geom_id(m, nm)
            if g is None:
                continue
            v = _rand(rng, cfg.dr_background_gray)
            m.geom_matid[g] = -1
            m.geom_rgba[g] = np.array([v, v, v, 1.0])

    # ---- lighting ---------------------------------------------------------
    if nominal is not None and any([cfg.dr_light_diffuse != (1.0, 1.0),
                                    cfg.dr_light_ambient != (0.0, 0.0),
                                    cfg.dr_light_specular != (1.0, 1.0),
                                    cfg.dr_light_dir, cfg.dr_light_pos]):
        for li in range(m.nlight):
            if cfg.dr_light_diffuse != (1.0, 1.0):
                m.light_diffuse[li] = np.clip(
                    nominal["light_diffuse"][li] * _rand(rng, cfg.dr_light_diffuse), 0, 1)
            if cfg.dr_light_ambient != (0.0, 0.0):
                m.light_ambient[li] = np.clip(_rand(rng, cfg.dr_light_ambient), 0, 1)
            if cfg.dr_light_specular != (1.0, 1.0):
                m.light_specular[li] = np.clip(
                    nominal["light_specular"][li] * _rand(rng, cfg.dr_light_specular), 0, 1)
            if cfg.dr_light_dir:
                d = nominal["light_dir"][li] + rng.uniform(-cfg.dr_light_dir,
                                                           cfg.dr_light_dir, 3)
                n = np.linalg.norm(d)
                if n > 1e-8:
                    m.light_dir[li] = d / n
            if cfg.dr_light_pos:
                m.light_pos[li] = nominal["light_pos"][li] + rng.uniform(
                    -cfg.dr_light_pos, cfg.dr_light_pos, 3)

    # ---- robot base height -------------------------------------------------
    # Physics-affecting (it moves the whole kinematic chain), so keep the range
    # small and treat it as mounting tolerance, not as a way to paper over a
    # mis-measured table height.
    if cfg.dr_base_height and nominal is not None:
        bid = _body_id(m, "robot0_base")
        if bid is not None:
            m.body_pos[bid] = nominal["base_pos"] + np.array(
                [0.0, 0.0, rng.uniform(-cfg.dr_base_height, cfg.dr_base_height)])

    # ---- cameras -----------------------------------------------------------
    for cam in cfg.cameras:
        if cam.is_inert():
            continue
        _apply_camera(env, m, sim, cam, cfg, rng)

    sim.forward()


def _apply_camera(env, m, sim, cam: RealCameraConfig, cfg: Sim2RealConfig, rng):
    cid = _cam_id(m, cam.name)
    if cid is None:
        return                      # state-only runs build no cameras
    d = sim.data
    R_cv = cam.rotation_opencv()
    p = np.asarray(cam.pos, dtype=np.float64)

    if cfg.dr_cam_pos:
        p = p + rng.uniform(-cfg.dr_cam_pos, cfg.dr_cam_pos, 3)
    if R_cv is not None and cfg.dr_cam_rot_deg:
        R_cv = R_cv @ _small_rot(rng, cfg.dr_cam_rot_deg)

    if cam.frame == "base":
        bid = _body_id(m, "robot0_base")
        assert bid is not None, "sim2real: robot0_base not found"
        fp, fR = d.body_xpos[bid].copy(), d.body_xmat[bid].reshape(3, 3).copy()
    elif cam.frame == "ee":
        sid = _site_id(m, "gripper0_grip_site")
        assert sid is not None, "sim2real: gripper0_grip_site not found"
        fp, fR = d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()
    else:
        raise ValueError(f"camera {cam.name!r}: unknown frame {cam.frame!r}")

    pos_w = fp + fR @ p
    # MuJoCo stores a camera pose RELATIVE TO ITS PARENT BODY. A statically
    # mounted camera hangs off worldbody (parent = world, so world == parent),
    # but a wrist camera hangs off the hand, and writing world coordinates there
    # would put it in a different place every step.
    pid = int(m.cam_bodyid[cid])
    if pid == 0:
        m.cam_pos[cid] = pos_w
        if R_cv is not None:
            m.cam_quat[cid] = mat2quat_wxyz(fR @ R_cv @ OPENCV_TO_MUJOCO)
    else:
        pp, pR = d.body_xpos[pid].copy(), d.body_xmat[pid].reshape(3, 3).copy()
        m.cam_pos[cid] = pR.T @ (pos_w - pp)
        if R_cv is not None:
            m.cam_quat[cid] = mat2quat_wxyz(pR.T @ (fR @ R_cv @ OPENCV_TO_MUJOCO))

    fovy = cam.resolved_fovy()
    if fovy > 0:
        if cfg.dr_cam_fovy_deg:
            fovy += rng.uniform(-cfg.dr_cam_fovy_deg, cfg.dr_cam_fovy_deg)
        m.cam_fovy[cid] = float(fovy)


def snapshot_visual_nominal(env) -> dict:
    """Capture the values every multiplicative DR range composes from. Taken once
    per rebuilt model; hard_reset means the model is fresh each episode, so this
    is cheap insurance against a gain compounding across episodes."""
    sim = env.sim if hasattr(env, "sim") else env.env.sim
    m = sim.model
    bid = _body_id(m, "robot0_base")
    return {
        "light_diffuse": m.light_diffuse.copy(),
        "light_ambient": m.light_ambient.copy(),
        "light_specular": m.light_specular.copy(),
        "light_dir": m.light_dir.copy(),
        "light_pos": m.light_pos.copy(),
        "base_pos": m.body_pos[bid].copy() if bid is not None else None,
    }


# MuJoCo's name2id raises on a miss in some bindings and returns -1 in others;
# wrap once so callers can just test for None.
def _lookup(fn, name):
    try:
        i = fn(name)
    except Exception:
        return None
    return None if i is None or i < 0 else int(i)


def _geom_id(m, name):
    return _lookup(m.geom_name2id, name)


def _geom_name(m, gid):
    try:
        return m.geom_id2name(gid)
    except Exception:
        return None


def _body_id(m, name):
    return _lookup(m.body_name2id, name)


def _site_id(m, name):
    return _lookup(m.site_name2id, name)


def _cam_id(m, name):
    return _lookup(m.camera_name2id, name)


# ------------------------------------------------------------------- yaml I/O
def load_sim2real(path) -> Sim2RealConfig:
    """Read a cell YAML into a Sim2RealConfig.

    Hand-rolled rather than routed through pyrallis because `cameras` is a
    variable-length list of nested dataclasses, which pyrallis resolves by eager
    type-hint introspection and cannot decode. Unknown keys RAISE: a silently
    ignored typo in a cell file would be baked into a multi-day dataset before
    anyone noticed."""
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cams = []
    for i, c in enumerate(raw.pop("cameras", []) or []):
        unknown = set(c) - set(RealCameraConfig.__dataclass_fields__)
        assert not unknown, f"{path}: cameras[{i}] has unknown keys {sorted(unknown)}"
        cams.append(RealCameraConfig(**{
            k: (tuple(v) if isinstance(v, list) else v) for k, v in c.items()}))

    unknown = set(raw) - set(Sim2RealConfig.__dataclass_fields__)
    assert not unknown, f"{path}: unknown keys {sorted(unknown)}"
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items()}
    cfg = Sim2RealConfig(cameras=tuple(cams), **kw)
    for c in cfg.cameras:
        c.rotation_opencv()        # validate now, not mid-dataset-generation
    return cfg

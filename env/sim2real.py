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

import os
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


def _quat_mul_xyzw(a, b) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def eef_in_base(sim, pos_w, quat_xyzw):
    """World-frame eef pose -> robot0_base frame of the CURRENT model state.

    The one implementation of prop_frame="base": the wrapper (training / eval),
    the deploy server and the real-demo converter all call this, so the prop on
    the robot is the same function of q as in the sim. Quaternion product rather
    than a matrix round trip, so the sign convention of the input is kept (a
    mat2quat would be free to flip it)."""
    m, d = sim.model, sim.data
    bid = m.body_name2id("robot0_base")
    R = np.asarray(d.body_xmat[bid]).reshape(3, 3)
    pos = R.T @ (np.asarray(pos_w, np.float64) - d.body_xpos[bid])
    w, x, y, z = (float(v) for v in d.body_xquat[bid])
    quat = _quat_mul_xyzw((-x, -y, -z, w), np.asarray(quat_xyzw, np.float64))
    return pos, quat


def apply_prop_frame(sim, cfg, obs) -> None:
    """In place on a robosuite obs dict: rewrite the eef keys into cfg.prop_frame."""
    if not (cfg is not None and cfg.enabled and cfg.prop_frame == "base"):
        return
    pos, quat = eef_in_base(sim, obs["robot0_eef_pos"], obs["robot0_eef_quat"])
    obs["robot0_eef_pos"] = pos.astype(np.asarray(obs["robot0_eef_pos"]).dtype)
    obs["robot0_eef_quat"] = quat.astype(np.asarray(obs["robot0_eef_quat"]).dtype)


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
    frame: str = "base"             # base | ee (grip site) | hand (sim hand body)
    randomize: bool = True          # disable for a fixed, post-restoration camera
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
    # REAL-side intrinsics: JSON {"K": 3x3, "D": [...], "size": [w, h]} that the
    # real frames of THIS camera must be undistorted with, instead of the camera's
    # factory camera_info. "" = factory. Relative paths resolve against the cell
    # yaml. Read by the real side only (real2sim/deploy/preprocess.py
    # camera_intrinsics: policy_server + scripts/convert_real_teleop.py); the sim
    # never looks at it. The 2026-10-10 back-of-hand ZED-M needs it (factory D wrong).
    real_intrinsics: str = ""
    # Per-camera DR widths. None = use the cell-wide dr_cam_pos / dr_cam_rot_deg /
    # dr_cam_fovy_deg (the old behaviour). Only read when randomize is true.
    # ⚠️ They OVERRIDE the cell-wide knobs: zeroing dr_cam_* no longer switches a camera with its own
    # widths off (a test did exactly that and failed) -- set randomize=False on the camera instead.
    # Exists for the 2026-10-14 wrist camera: its hand-eye TRANSLATION is ~1 cm
    # uncertain (calib_20261014/README.md) while the scene camera is validated to
    # 1-2 px, so the two need different widths.
    # dr_pos / dr_rot_deg may also be a 3-tuple, one width per axis (pos: the camera's `frame`, rot: the camera's
    # own axes, x/y = tilt, z = roll about the optical axis): the measured wrist error is strongly directional
    # (calib_20261014/README.md), and an isotropic box wide enough for its worst axis shakes every other axis too.
    dr_pos: Optional[object] = None
    dr_rot_deg: Optional[object] = None
    dr_fovy_deg: Optional[float] = None

    def dr_widths(self, cfg) -> tuple:
        """(pos m, rot deg, fovy deg) DR widths for this camera; zeros if fixed. pos / rot are a float
        or a 3-tuple."""
        if not self.randomize:
            return 0.0, 0.0, 0.0
        def pick(own, glob):
            v = glob if own is None else own
            return tuple(float(x) for x in v) if isinstance(v, (tuple, list)) else float(v)
        return (pick(self.dr_pos, cfg.dr_cam_pos), pick(self.dr_rot_deg, cfg.dr_cam_rot_deg),
                pick(self.dr_fovy_deg, cfg.dr_cam_fovy_deg))

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
    # MEASURED ARENA (collision table + robot base placement). None = leave the
    # task's own arena alone. Both are measured by the real2sim pipeline
    # (docs/real2sim.md §2e) and live as constants in env/peg_insertion.py; set
    # `arena_from_real2sim: 1` in a cell YAML to pull them (and table_above_base)
    # from there instead of restating the numbers -- one source, so this cell and
    # PegInsertion_D0 cannot drift apart.
    #   table_full_size: full (x, y, z) of the table, metres -- COLLISION and
    #       visual alike (unlike table_visual_half). Real work surface: 0.739 x
    #       1.240, long axis along the robot's y.
    #   base_xy: robot base (x, y) relative to the table centre. REQUIRED
    #       whenever table_full_size is set: robosuite derives the base x from
    #       table_full_size[0] (threading.py:244), so resizing the table alone
    #       would silently walk the robot.
    arena_from_real2sim: int = 0
    table_full_size: Optional[tuple] = None
    # Extra height of the WORK SURFACE over table_above_base, metres: e.g. a board
    # lying on the bench mat that real2sim measured (arena_from_real2sim takes the
    # MAT height, shared with PegInsertion_D0, which has no board). Moves the
    # collision table, the visual table and the reset distribution (table_offset)
    # up together; the robot and its cameras stay. Read it through
    # work_surface_above_base(), never add it by hand. 0 = legacy.
    # Threading 2026-10-10 cell: the white board top is ~9 mm above the mat.
    table_top_dz: float = 0.0
    base_xy: Optional[tuple] = None
    # OBJECT LAYOUT on this bench, in BASE frame metres / degrees. None = keep the
    # task's own reset distribution (shifted by workspace_offset, the legacy path).
    # Set here rather than in each RL config because "where the parts sit on this
    # bench" is a property of the cell: the demo generator (Threading_RealCell)
    # and the RL/SimDist envs must not disagree.
    # 2026-09-22: the ring moved to the camera side. Measured on the real cell,
    # with the ring on the far side the camera's line of sight to it is blocked by
    # the table edge (first ray hit = table_collision) and the 20 mm opening spans
    # 4.2 px; swapping the two objects makes the ring visible, 0.73 m away,
    # 5.2 px, at a 71 deg incidence instead of 52.
    layout_needle_x: Optional[tuple] = None      # (lo, hi) in base frame
    layout_needle_y: Optional[tuple] = None
    layout_needle_z_rot_deg: Optional[tuple] = None
    layout_tripod_x: Optional[tuple] = None
    layout_tripod_y: Optional[tuple] = None
    layout_tripod_z_rot_deg: Optional[tuple] = None
    # GRIPPER model, as robosuite's `gripper_types` (e.g. "Robotiq85Gripper").
    # None = the task's default (the Panda hand). CONSTRUCTION-time, unlike every
    # other field here: PixelRobosuite passes it to robosuite.make. The real cell
    # carries a Robotiq 2F-85 since 2026-10-10.
    # ⚠️ Not a visual swap: Robotiq85 has 6 gripper joints vs the Panda's 2, so
    # robot0_gripper_qpos (prop, state) grows 2 -> 6, and the grip site / TCP
    # depth moves. Every expert, demo set and stage-1 built on the Panda hand is
    # incompatible with a cell that sets this.
    gripper_types: Optional[str] = None
    # CONTROL RATE of the real cell (Hz), as robosuite's `control_freq`. None = the
    # task's default (robosuite 20). CONSTRUCTION-time like gripper_types: the
    # wrapper passes it to robosuite.make. 2026-10-15: the Robotiq cell runs at
    # 10 Hz (calib_20261015/README.md §2; HD720@30 fps = 3 frames/step). One
    # action = 1/control_freq s of OSC, so the world model's p(s'|s,a), every
    # expert and every dataset are tied to it (CLAUDE.md rule #37): data at
    # another rate does not transfer. policy_server refuses a client at another hz.
    control_freq: Optional[float] = None
    # Mount of the gripper on robot0_right_hand: yaw about the hand z axis (deg)
    # and an offset along it (m), applied to the gripper's root body. 0/0 = the
    # robosuite mount (byte-for-byte legacy). robosuite bolts the gripper to
    # robot0_right_hand, which sits 45.03 deg about z from the FR3 flange (iron law
    # #27); the real Robotiq is bolted square to the FLANGE through a coupling.
    # Fitted 2026-10-11 to the run_1010_2327 wrist photos (finger-mask IoU 0.196 at
    # 0/0 -> 0.86 at 45.03 deg / +7 mm). Moves grip_site too, i.e. the TCP.
    gripper_mount_yaw_deg: float = 0.0
    gripper_mount_z: float = 0.0
    # Robot reset joint configuration (7 arm joints, rad). None = robosuite's
    # default, which was designed for a table 112 mm BELOW the base: on the
    # measured cell (table 20.7 mm ABOVE) it leaves the grip site only 78 mm over
    # the table, and sigma=0.1 joint noise puts the fingers INTO the table on
    # ~5% of resets. The joint-space init noise (hand_init_perturb_joint_sigma)
    # is centred on this value, so it also sets the centre of the init
    # distribution -- it must equal the real robot's home pose.
    init_qpos: Optional[tuple] = None
    # Visual-only enlargement of the table, as HALF-sizes (x, y) in metres.
    # (0,0) = leave alone. This writes `table_visual`'s geom_size, which is
    # contype=conaffinity=0, so the collision box and -- critically -- the robot
    # base placement are untouched. Do NOT enlarge via `table_full_size`:
    # threading.py:244 computes the base xpos as a FUNCTION of table_full_size[0],
    # so that route silently walks the robot backwards out of reach.
    # The z half-size is never written, so the table TOP height cannot move here.
    table_visual_half: tuple = (0.0, 0.0)
    # (dx, dy) of the VISUAL table relative to the collision table's centre, m.
    # The real cell's white board is not centred on the measured work surface:
    # the surface (0.739 x 1.240, which RANSAC fitted over the mats) is mostly
    # green mat and wood, and only a ~0.62 x 0.66 m white board sits on it,
    # offset by (+0.036, +0.076) from the surface centre (back-projected from the
    # real scene camera). Rendering white over the whole surface would put white
    # where the real cell has a green mat.
    table_visual_offset: tuple = (0.0, 0.0)
    # ⚠️ Until 2026-10-11 table_visual_offset NEVER REACHED THE RENDER. The geom
    # compiles with geom_sameframe=1 (it sits at its body's origin), so MuJoCo's
    # kinematics copies the body pose and ignores geom_pos/geom_quat: every frame
    # ever rendered from a cell with a non-zero offset shows the board centred on
    # the COLLISION table (base (0.489, 0.034)), not where the yaml says.
    # The fix is opt-in so those datasets keep matching their configs:
    #   0 = legacy (offset written to geom_pos but inert; yaw must be 0)
    #   1 = clear geom_sameframe, so table_visual_offset/_yaw_deg really move it.
    table_visual_apply_pose: int = 0
    # Yaw of the VISUAL table about the table-frame z axis, degrees (about its own
    # centre); needs table_visual_apply_pose=1. The white board is placed by hand
    # and is not square to the robot base (2026-10-11 photo fit: 0.76 deg); with an
    # axis-aligned box the edge misfit is ~2 px at 1280x720. Visual only: the
    # collision table stays axis-aligned.
    table_visual_yaw_deg: float = 0.0
    # The arena's four steel table legs. They were invisible while the visual
    # table was a 5 m plane; with the table cut down to the real white board they
    # show up as two grey pillars that have no counterpart in the real cell (an
    # optical breadboard on a bench). 1 = make them fully transparent.
    hide_table_legs: int = 0
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

    # ---- MATCHED LIGHTING ---------------------------------------------------
    # The arena ships ONE directional light from the far right with
    # castshadow=False, i.e. no shadows at all. The real cell is lit from above
    # the scene camera, so its shadows fall AWAY from that camera; a policy that
    # has only ever seen shadowless renders has no way to interpret them.
    # Positions are in the BASE frame, like the cameras.
    light_pos: Optional[tuple] = None        # None = leave the arena light alone
    light_target: Optional[tuple] = None     # point it at this (base frame)
    light_directional: Optional[int] = None  # 0 = point light (position matters)
    light_castshadow: Optional[int] = None   # 1 = cast shadows
    # The real fixture is a LED STRIP, not a bulb: a line above the scene camera
    # running along the robot's fore-aft axis (base +x). MuJoCo has no area or
    # line light, so it is approximated by `light_line_n` point lights spread
    # over `light_line_len` metres, each at 1/n intensity. The overlapping
    # penumbrae are the point of the exercise -- one bulb gives a hard, single
    # shadow that a strip never produces. n=1 (default) = the single point light.
    light_line_n: int = 1
    light_line_len: float = 0.0              # metres, along base +x
    # TOTAL diffuse summed over the strip (each light gets 1/n of it). None
    # leaves whatever the arena/XML set. Fitted, not guessed: with the board's
    # albedo pinned by the ZED, this is what puts the rendered board median on
    # the real 216.
    light_diffuse: Optional[float] = None
    # How many of the strip's lights cast shadows. This is the UMBRA-DEPTH dial:
    # a point shadowed from k of n lights still receives (n-k)/n of the light, so
    # k sets how dark the shadow core is without touching its shape. None = all.
    # Measured at light_pos z=3.0, board spread p90-p10 (real = 52):
    #   k=3 -> 35 | k=4 -> 41 | k=5 -> 57 | k=6 -> 61 | k=8 -> 106
    light_shadow_n: Optional[int] = None

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
    dr_light_pos: float = 0.0           # +- per-axis, metres. ⚠️ With a strip
    # (light_line_n>1) this offset is drawn ONCE and applied to every light, so
    # the rig translates as one object. Drawing it per light would scatter the
    # line into a cloud and randomise the very thing the strip exists to model.
    dr_light_shadow_n: int = 0          # +- on light_shadow_n, per episode. The
    # real cell's shadows are much weaker than one hard rig produces and vary
    # with whatever else is on in the room, so the umbra depth is randomised too.
    # Photo backgrounds. "" = off. One or more directories ("+"- or ","-separated)
    # of jpg/png; one image is painted onto the WALL texture (and, with
    # dr_background_floor, the floor) per episode. The real cell has a poster and
    # furniture behind the table that cannot be removed, and a flat grey wall
    # (dr_background_gray) does not cover that: measured on the first real
    # episodes, the whole background region is structured content the policy has
    # never seen. Photos are the cheap stand-in.
    # ⚠️ This changes the RENDERED frames, so it only reaches an offline stage-1
    # arm through a re-render of the source replay (scripts/rerender_simdist_data.py).
    dr_background_image: str = ""
    dr_background_floor: int = 1        # also repaint the floor texture
    dr_base_height: float = 0.0         # +- metres on the robot base z
    # 1 = cameras with frame "base" follow the RANDOMISED base pose (the camera is
    # calibrated against the base, so base<->table error moves the camera with the
    # robot: dr_base_height then IS a table-height DR in the image too).
    # 0 = legacy: they were placed off the base pose from BEFORE this episode's
    # base DR (body_xpos not yet recomputed), i.e. fixed in the world/table frame
    # while the robot moved -- the base DR never reached the scene image and
    # instead jittered camera<->robot by up to ~20 mm. Before 2026-10-11 every
    # cell ran with 0.
    cam_follow_base_dr: int = 0
    # Residual base<->table misregistration (docs/infra.md §14a R1-R3). All of
    # them perturb the ROBOT BASE pose, never the table: what matters physically
    # is the relative pose, and tilting the table would unseat objects already
    # placed on it (gaps/penetration of up to size*tilt at the edges).
    dr_base_yaw_deg: float = 0.0        # +- deg about world z      (R1: yaw snap 1.2 deg)
    dr_base_tilt_deg: float = 0.0       # +- deg about world x AND y (R2: table tilt 0.99 deg)
    dr_base_xy: float = 0.0             # +- metres per axis in x,y (R3: two solves ~7 mm apart)
    # Frame of robot0_eef_pos / robot0_eef_quat in the wrapper's obs (state[:9] AND
    # prop -- every trainer slices prop out of state, so both change together).
    # "world" = robosuite's own (legacy). "base" = the robot0_base frame of THIS
    # episode (2026-10-15): on the robot the prop is FK(q) in the base frame with
    # no calibration in it, while a world-frame prop carries the base<->table
    # registration -- which the base DR above randomises in the sim but nobody
    # knows on the robot. In "base" the base DR reaches the policy only through
    # the images, as on the robot. object-state stays WORLD (its *_to_eef terms
    # are gripper-frame and do not care). Changes the obs contract: experts /
    # data / stage-1 built with "world" do not transfer.
    prop_frame: str = "world"
    # EXECUTION LAYER of the real cell (env/exec_layer.py, 2026-10-17): the deploy chain between an action and
    # the arm -- reference-arm OSC, TCP speed / lead caps, 100 Hz ramp, joint impedance with the real gains --
    # plus the real arm's fitted armature / stiction / damping. A dict of env/exec_layer.py DEFAULTS keys
    # (+ enabled). None = off: robosuite's OSC moves the arm itself (legacy, byte-for-byte). Installed by the
    # task class (Threading_RealCellRobotiq), so MimicGen / bare robosuite.make get it too. Changes what an
    # action does: experts / data / stage-1 made without it do not transfer.
    exec_layer: Optional[dict] = None
    # GRIPPER CONTACT (2026-10-18): solref (timeconst, dampratio) of every gripper collision geom. None = robosuite's
    # (Robotiq fingers / tips 0.01 0.25: stiff and underdamped -- it squeezes the needle handle out: the scripted grasp
    # grid lifted 1/12 in the sim vs 12/12 on the robot). (0.02, 1.0) = critically damped: 12/12 trial-by-trial
    # agreement with the real grid (calib_20261015 README "10-18 robot visit"). Applied by the task class.
    gripper_contact_solref: Optional[tuple] = None
    # PHYSICS DR of the real arm / gripper (2026-10-19, TRAIN_PLAN.md): per episode, applied by the WRAPPER (so every
    # "dr_* off" copy -- deploy ReferenceArm, replays, twins -- stays deterministic; bare robosuite.make / MimicGen
    # generation run the nominal values). Relative: each arm joint's armature / frictionloss x U(1-w, 1+w), from the
    # fitted values the task class installed (exec_layer). dr_gripper_dampratio: (lo, hi) for the gripper collision
    # solref dampratio (the grasp grid only showed that 1.0 works, not where it stops working).
    dr_arm_armature: float = 0.0
    dr_arm_frictionloss: float = 0.0
    dr_gripper_dampratio: Optional[tuple] = None

    # Placement tolerance of a peg that is ARENA geometry rather than a sampled
    # object (Square's peg1/peg2 live in pegs_arena.xml at fixed coordinates, so
    # the task's own reset sampler never touches them and every episode sees the
    # peg in exactly one spot). Applied per reset as a body pose offset.
    # ⚠️ For a STATE policy this perturbation is UNOBSERVABLE: Square's 23-dim
    # obs carries SquareNut pose + proprio only, no peg. So it is noise the
    # policy must absorb, not information it can use, and the usable range is
    # bounded by the nut/peg clearance (45.4 mm hole vs 32 mm peg => 6.7 mm per
    # side when the yaw is aligned, ~0 at 45 deg of yaw error). A PIXEL policy
    # does see it. Measure before picking a value -- docs/infra.md §15k.
    dr_peg_xy: float = 0.0              # +- metres per axis in x,y
    dr_peg_yaw_deg: float = 0.0         # +- deg about world z
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
            self.dr_light_shadow_n,
            self.dr_base_height, self.dr_cam_pos, self.dr_cam_rot_deg,
            self.dr_cam_fovy_deg, self.any_base_dr(), self.any_peg_dr(), self.any_physics_dr(),
            bool(self.dr_background_image),
            any(any(np.any(w) for w in c.dr_widths(self)) for c in self.cameras)])

    def work_surface_above_base(self) -> Optional[float]:
        """Height of the surface the objects sit on, above the base origin."""
        if self.table_above_base is None:
            return None
        return float(self.table_above_base) + float(self.table_top_dz)

    def any_base_dr(self) -> bool:
        """True when the robot base pose is randomised (the OSC must re-latch)."""
        return bool(self.enabled) and any([
            self.dr_base_height, self.dr_base_yaw_deg, self.dr_base_tilt_deg,
            self.dr_base_xy])

    def any_physics_dr(self) -> bool:
        return bool(self.enabled) and any([self.dr_arm_armature, self.dr_arm_frictionloss,
                                           self.dr_gripper_dampratio is not None])

    def any_peg_dr(self) -> bool:
        """True when an arena peg is randomised. Deliberately NOT part of
        any_base_dr: that one means the ROBOT moved and the OSC must re-latch,
        which a peg never causes."""
        return bool(self.enabled) and any([self.dr_peg_xy, self.dr_peg_yaw_deg])


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
    return np.array([0.0, 0.0, float(base_world_z) + cfg.work_surface_above_base()])


def apply_cell_arena(env, cfg: Sim2RealConfig) -> None:
    """Install the measured table size + robot base xy on a MimicGen table env.

    Wraps the instance's `_load_model`, because hard_reset=True rebuilds the
    arena and the robot model every episode: the table size is set before the
    original runs, the base xy is overwritten after it (the task's own
    `set_base_xpos(f(table_full_size[0]))` runs inside). The robot's root body
    element is shared with the merged task model, and the XML is only generated
    later in `_initialize_sim`, so the post-hoc write lands. Takes effect on the
    NEXT reset. Idempotent, like apply_workspace_offset.
    """
    if not cfg.enabled or cfg.table_full_size is None:
        return
    assert cfg.base_xy is not None, (
        "sim2real: table_full_size without base_xy would move the robot "
        "(the base x is derived from table_full_size[0])")
    if getattr(env, "_s2r_arena_applied", False):
        return
    size = tuple(float(v) for v in cfg.table_full_size)
    bx, by = (float(v) for v in cfg.base_xy)
    inner = env._load_model

    def load_model():
        env.table_full_size = size
        inner()
        env.robots[0].robot_model.set_base_xpos((bx, by, 0.0))

    env._load_model = load_model
    env._s2r_arena_applied = True



# Shadow-map settings used whenever the strip is on. Module-level so a render
# experiment can sweep them without a config field: the frustum has to cover a
# light 1.6 m up (default 0.6 clips into visible rectangles on the table) and
# the resolution has to keep the penumbra from looking like stair steps.
SHADOW_SCALE = 1.2
SHADOW_SIZE = 8192


def _line_light_offsets(cfg: Sim2RealConfig) -> Optional[np.ndarray]:
    """Base-frame offsets of the strip's point samples, or None if n<=1."""
    n = int(getattr(cfg, "light_line_n", 1) or 1)
    if n <= 1:
        return None
    L = float(getattr(cfg, "light_line_len", 0.0) or 0.0)
    t = np.linspace(-0.5, 0.5, n) * L
    return np.stack([t, np.zeros(n), np.zeros(n)], axis=1)


def apply_line_lights(env, cfg: Sim2RealConfig) -> None:
    """Turn the arena's single light into a strip of `light_line_n` lights.

    Has to happen on the MJCF, before `_initialize_sim` compiles it -- the light
    COUNT is structural, unlike every other lighting knob. Wraps `_load_model`
    for the same reason `apply_cell_arena` does (hard_reset rebuilds the arena
    every episode). Each copy is dimmed by 1/n so the total flux is unchanged,
    which also means the captured nominal (and therefore every dr_light_*
    multiplier) keeps its meaning. Idempotent; takes effect on the next reset.

    ⚠️ Changes `m.nlight` ⇒ changes the length of `capture_model_state`'s vector
    ⇒ pixel data recorded with a different `light_line_n` cannot be re-rendered.
    """
    n = int(getattr(cfg, "light_line_n", 1) or 1)
    if not cfg.enabled or n <= 1 or getattr(env, "_s2r_lights_applied", False):
        return
    assert n <= 8, (
        f"light_line_n={n}: MuJoCo renders at most 8 lights, the extra ones "
        "are silently dropped (and the 1/n dimming would then be wrong)")
    inner = env._load_model

    def load_model():
        inner()
        wb = env.model.worldbody
        lights = wb.findall("light")
        if not lights:
            return
        proto = lights[0]
        # MuJoCo's own <light> defaults, needed because the arena writes neither:
        # an absent attribute is NOT zero, so "scale what is written" silently
        # left diffuse at 0.7 per light and made the strip n x too bright.
        MJ_DEFAULT = {"diffuse": "0.7 0.7 0.7", "specular": "0.3 0.3 0.3"}
        for li in lights:
            for attr, dflt in MJ_DEFAULT.items():
                v = li.get(attr) or dflt
                li.set(attr, " ".join(f"{float(x) / n:.6g}" for x in v.split()))
        import copy as _copy
        for _ in range(n - 1):
            wb.append(_copy.deepcopy(proto))
        # Each light gets its own shadow map, and the default frustum
        # (shadowscale 0.6) is too small for a light 1.6 m up: the maps clip
        # into visible RECTANGLES on the table. Widen it and raise the
        # resolution, or the strip looks like a rendering bug rather than a
        # soft shadow.
        import xml.etree.ElementTree as _ET
        vis = env.model.root.find("visual")
        if vis is None:
            vis = _ET.SubElement(env.model.root, "visual")
        mp = vis.find("map")
        if mp is None:
            mp = _ET.SubElement(vis, "map")
        mp.set("shadowscale", str(SHADOW_SCALE))
        ql = vis.find("quality")
        if ql is None:
            ql = _ET.SubElement(vis, "quality")
        ql.set("shadowsize", str(SHADOW_SIZE))

    env._load_model = load_model
    env._s2r_lights_applied = True

def apply_init_qpos(env, cfg: Sim2RealConfig) -> None:
    """Set the robot's reset joint configuration via robosuite's own
    `initial_qpos` robot config. NOT `env.robots[0].init_qpos`: `_load_robots`
    RE-INSTANTIATES every Robot from `robot_configs` on each hard reset, so an
    attribute written on the instance is silently gone one reset later (this
    exact bug shipped once -- the gate now checks the reset qpos). robosuite's
    joint init noise is added on top, so this moves the centre of the init
    distribution. Takes effect on the next reset."""
    if not cfg.enabled or cfg.init_qpos is None:
        return
    q = np.asarray(cfg.init_qpos, dtype=np.float64)
    n = len(env.robots[0].init_qpos)
    assert q.shape == (n,), f"sim2real.init_qpos has {q.shape[0]} joints, robot has {n}"
    env.robot_configs[0]["initial_qpos"] = q


def cell_layout_bounds(cfg: Sim2RealConfig, table_centre_xy) -> Optional[dict]:
    """The cell's object layout as MimicGen-convention bounds (TABLE frame).

    The yaml states it in BASE frame because that is what a tape measure on the
    bench gives; the env wants it relative to the table centre."""
    if cfg.layout_needle_x is None and cfg.layout_tripod_x is None:
        return None
    cx, cy = float(table_centre_xy[0]), float(table_centre_xy[1])

    def rng(v, c):
        return None if v is None else (float(v[0]) - c, float(v[1]) - c)

    def rot(v):
        return None if v is None else (np.radians(float(v[0])), np.radians(float(v[1])))

    return {"needle": {"x": rng(cfg.layout_needle_x, cx), "y": rng(cfg.layout_needle_y, cy),
                       "z_rot": rot(cfg.layout_needle_z_rot_deg)},
            "tripod": {"x": rng(cfg.layout_tripod_x, cx), "y": rng(cfg.layout_tripod_y, cy),
                       "z_rot": rot(cfg.layout_tripod_z_rot_deg)}}


def apply_cell_layout(env, cfg: Sim2RealConfig) -> bool:
    """Override the env's reset distribution with the cell's layout. Idempotent.

    Wraps `_get_initial_placement_bounds` like apply_workspace_offset does, so it
    survives robosuite's per-reset rebuild and composes with the task's own
    `reference` (the table offset). Returns True if anything was installed.
    ⚠️ Takes precedence over workspace_offset for the fields it sets -- having
    both shift the same object would double-count."""
    if not cfg.enabled or getattr(env, "_s2r_layout_applied", False):
        return False
    if cfg.layout_needle_x is None and cfg.layout_tripod_x is None:
        return False          # no layout stated -> nothing to install, and the
                              # base_xy requirement below does not apply
    # Table centre in BASE frame = -base_xy, taken from the CONFIG. Reading it off
    # the env instead is an ordering trap: this hook runs before the arena patch,
    # so env.table_offset / the base body still hold robosuite's defaults and the
    # centre comes out (0.56, 0) instead of (0.489, 0.034) -- the layout then
    # lands ~7 cm off with no error (measured: needle x fell back onto D0's range).
    assert cfg.base_xy is not None, "cell layout needs base_xy (arena_from_real2sim)"
    tc = -np.asarray(cfg.base_xy, dtype=np.float64)
    bounds = cell_layout_bounds(cfg, tc)
    if bounds is None:
        return False
    inner = env._get_initial_placement_bounds

    def overridden():
        out = {}
        for name, b in inner().items():
            b = dict(b)
            spec = bounds.get(name)
            if spec is not None:
                for k in ("x", "y", "z_rot"):
                    if spec[k] is not None:
                        b[k] = spec[k]
            out[name] = b
        return out

    env._get_initial_placement_bounds = overridden
    env._s2r_layout_applied = True
    return True


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
    """Random rotation with each axis angle ~ U(-deg, deg) (deg: scalar or per-axis), applied as an
    intrinsic XYZ composition. Small-angle so order barely matters, but it is
    fixed anyway to keep a seeded run repeatable."""
    a, b, c = np.radians(rng.uniform(-deg, deg, 3))
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def apply_sim2real(env, cfg: Sim2RealConfig, rng=None, nominal: Optional[dict] = None,
                   bg_index: Optional[int] = None):
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
        if tuple(cfg.table_visual_offset) != (0.0, 0.0):
            gp = m.geom_pos[gid].copy()
            m.geom_pos[gid] = np.array([float(cfg.table_visual_offset[0]),
                                        float(cfg.table_visual_offset[1]), gp[2]])
        if cfg.table_visual_yaw_deg:
            assert cfg.table_visual_apply_pose, (
                "table_visual_yaw_deg is inert without table_visual_apply_pose=1")
            h = np.radians(float(cfg.table_visual_yaw_deg)) / 2
            m.geom_quat[gid] = np.array([np.cos(h), 0.0, 0.0, np.sin(h)])
        if cfg.table_visual_apply_pose:
            m.geom_sameframe[gid] = 0     # else geom_pos/geom_quat are ignored
        if cfg.table_rgb is not None:
            rgb = np.asarray(cfg.table_rgb, dtype=np.float64)
            if cfg.dr_table_rgb_tint:
                rgb = rgb + rng.uniform(-cfg.dr_table_rgb_tint, cfg.dr_table_rgb_tint, 3)
            if cfg.dr_table_rgb_gain != (1.0, 1.0):
                rgb = rgb * _rand(rng, cfg.dr_table_rgb_gain)
            m.geom_matid[gid] = -1        # drop the ceramic texture
            m.geom_rgba[gid] = np.array([*np.clip(rgb, 0.0, 1.0), 1.0])

    if cfg.hide_table_legs:
        for gi in range(m.ngeom):
            nm = _geom_name(m, gi) or ""
            if nm.startswith("table_leg") and nm.endswith("_visual"):
                rgba = m.geom_rgba[gi].copy(); rgba[3] = 0.0
                m.geom_rgba[gi] = rgba

    if cfg.gripper_mount_yaw_deg or cfg.gripper_mount_z:
        hid = _body_id(m, "robot0_right_hand")
        roots = [b for b in range(m.nbody) if int(m.body_parentid[b]) == hid
                 and (m.body_id2name(b) or "").startswith("gripper0_")]
        assert len(roots) == 1, f"sim2real: gripper root under robot0_right_hand: {roots}"
        h = np.radians(float(cfg.gripper_mount_yaw_deg)) / 2
        # absolute, not incremental: the root sits at the hand origin with identity
        # quat in every robosuite gripper, so this stays idempotent across resets
        m.body_pos[roots[0]] = np.array([0.0, 0.0, float(cfg.gripper_mount_z)])
        m.body_quat[roots[0]] = np.array([np.cos(h), 0.0, 0.0, np.sin(h)])

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
    # ⚠️ gray and photo are two ways to repaint the SAME surfaces, and they used
    # to silently fight: gray sets geom_matid = -1 (drop the material so the flat
    # rgba shows), but `_bg_texids` resolves the photo's target THROUGH the
    # material and skips any geom with matid < 0. With both knobs on -- which is
    # what the shipped cell asks for -- the photo therefore had no target at all
    # and 100k episodes were collected with flat gray backgrounds, no error, no
    # warning. Fix: when a photo set is configured, the photo owns these geoms
    # and the material must be left intact.
    _photo_owns_bg = bool(cfg.dr_background_image)
    if cfg.dr_background_gray != (0.0, 0.0) and not _photo_owns_bg:
        for nm in _WALL_GEOMS:
            g = _geom_id(m, nm)
            if g is None:
                continue
            v = _rand(rng, cfg.dr_background_gray)
            m.geom_matid[g] = -1
            m.geom_rgba[g] = np.array([v, v, v, 1.0])

    # ---- matched lighting (before the DR jitter, which composes on top) ----
    _lit_pos = _lit_dir = None
    # per-light matched pose, so the DR jitter composes on EACH strip sample
    # rather than on whichever one the loop happened to end on.
    _matched_pos: dict = {}
    _matched_dir: dict = {}
    _matched_diffuse: dict = {}
    if cfg.light_pos is not None or cfg.light_target is not None or \
            cfg.light_directional is not None or cfg.light_castshadow is not None:
        bid = _body_id(m, "robot0_base")
        bp = sim.data.body_xpos[bid].copy() if bid is not None else np.zeros(3)
        bR = sim.data.body_xmat[bid].reshape(3, 3).copy() if bid is not None else np.eye(3)
        _line_off = _line_light_offsets(cfg)
        for li in range(m.nlight):
            if cfg.light_directional is not None:
                m.light_directional[li] = int(cfg.light_directional)
            if cfg.light_castshadow is not None:
                m.light_castshadow[li] = int(cfg.light_castshadow)
            if cfg.light_diffuse is not None:
                m.light_diffuse[li] = float(cfg.light_diffuse) / max(1, m.nlight)
                _matched_diffuse[li] = m.light_diffuse[li].copy()
            if cfg.light_pos is not None:
                _p = np.asarray(cfg.light_pos, dtype=np.float64)
                if _line_off is not None and li < len(_line_off):
                    _p = _p + _line_off[li]
                _lit_pos = bp + bR @ _p
                m.light_pos[li] = _lit_pos
                _matched_pos[li] = _lit_pos
            if cfg.light_target is not None:
                src = _lit_pos if _lit_pos is not None else np.asarray(m.light_pos[li])
                d = (bp + bR @ np.asarray(cfg.light_target, dtype=np.float64)) - src
                n = np.linalg.norm(d)
                if n > 1e-8:
                    _lit_dir = d / n
                    m.light_dir[li] = _lit_dir
                    _matched_dir[li] = _lit_dir

    # ---- how many strip lights cast a shadow (umbra depth) -----------------
    if cfg.light_shadow_n is not None:
        k = int(cfg.light_shadow_n)
        if cfg.dr_light_shadow_n:
            k += int(rng.integers(-cfg.dr_light_shadow_n, cfg.dr_light_shadow_n + 1)
                     if hasattr(rng, "integers") else
                     rng.randint(-cfg.dr_light_shadow_n, cfg.dr_light_shadow_n + 1))
        k = int(np.clip(k, 0, m.nlight))
        m.light_castshadow[:] = 0
        if k > 0:
            # spread the casters along the strip, so the penumbra stays even
            m.light_castshadow[np.linspace(0, m.nlight - 1, k).astype(int)] = 1

    # ---- lighting ---------------------------------------------------------
    if nominal is not None and any([cfg.dr_light_diffuse != (1.0, 1.0),
                                    cfg.dr_light_ambient != (0.0, 0.0),
                                    cfg.dr_light_specular != (1.0, 1.0),
                                    cfg.dr_light_dir, cfg.dr_light_pos]):
        # one draw for the whole rig -- see dr_light_pos
        _rig_dpos = (rng.uniform(-cfg.dr_light_pos, cfg.dr_light_pos, 3)
                     if cfg.dr_light_pos else np.zeros(3))
        for li in range(m.nlight):
            if cfg.dr_light_diffuse != (1.0, 1.0):
                # compose on the MATCHED value, not on the arena snapshot: the
                # snapshot is taken before the matched lighting is written, so
                # using it silently discards `light_diffuse` (the fitted value
                # had literally no effect on any DR sample until this was fixed).
                _b = _matched_diffuse.get(li, nominal["light_diffuse"][li])
                m.light_diffuse[li] = np.clip(_b * _rand(rng, cfg.dr_light_diffuse), 0, 1)
            if cfg.dr_light_ambient != (0.0, 0.0):
                # ABSOLUTE range, but it is a TOTAL: every light contributes its
                # own ambient, so with an n-light strip the draw must be split
                # n ways or the scene gets n x the intended fill (the strip made
                # [0.02, 0.12] behave like [0.16, 0.96] and washed the board out).
                m.light_ambient[li] = np.clip(_rand(rng, cfg.dr_light_ambient), 0, 1) / max(1, m.nlight)
            if cfg.dr_light_specular != (1.0, 1.0):
                m.light_specular[li] = np.clip(
                    nominal["light_specular"][li] * _rand(rng, cfg.dr_light_specular), 0, 1)
            if cfg.dr_light_dir:
                _b = _matched_dir.get(li, nominal["light_dir"][li])
                d = _b + rng.uniform(-cfg.dr_light_dir, cfg.dr_light_dir, 3)
                n = np.linalg.norm(d)
                if n > 1e-8:
                    m.light_dir[li] = d / n
            if cfg.dr_light_pos:
                _b = _matched_pos.get(li, nominal["light_pos"][li])
                m.light_pos[li] = _b + _rig_dpos

    # ---- robot base height -------------------------------------------------
    # Physics-affecting (it moves the whole kinematic chain), so keep the range
    # small and treat it as mounting tolerance, not as a way to paper over a
    # mis-measured table height.
    if cfg.any_base_dr() and nominal is not None:
        bid = _body_id(m, "robot0_base")
        if bid is not None:
            # Draw order: height first, exactly as before, so a config that only
            # sets dr_base_height reproduces its old per-seed samples.
            dz = rng.uniform(-cfg.dr_base_height, cfg.dr_base_height) \
                if cfg.dr_base_height else 0.0
            dxy = rng.uniform(-cfg.dr_base_xy, cfg.dr_base_xy, 2) \
                if cfg.dr_base_xy else np.zeros(2)
            m.body_pos[bid] = nominal["base_pos"] + np.array([dxy[0], dxy[1], dz])
            if cfg.dr_base_yaw_deg or cfg.dr_base_tilt_deg:
                yaw = np.radians(rng.uniform(-cfg.dr_base_yaw_deg, cfg.dr_base_yaw_deg)) \
                    if cfg.dr_base_yaw_deg else 0.0
                rx, ry = (np.radians(rng.uniform(-cfg.dr_base_tilt_deg,
                                                  cfg.dr_base_tilt_deg, 2))
                          if cfg.dr_base_tilt_deg else (0.0, 0.0))
                cz, sz, cx, sx, cy, sy = (np.cos(yaw), np.sin(yaw), np.cos(rx),
                                          np.sin(rx), np.cos(ry), np.sin(ry))
                Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
                Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
                Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
                # delta applied in the WORLD frame about the base origin
                m.body_quat[bid] = mat2quat_wxyz(
                    Rz @ Ry @ Rx @ quat_wxyz2mat(nominal["base_quat"]))

    # ---- photo background --------------------------------------------------
    if cfg.dr_background_image and nominal is not None and "geom_matid" in nominal:
        # A RE-RENDER restores model_state from an episode that was collected
        # while the gray knob was still stripping geom_matid to -1, so the broken
        # value comes straight back and _bg_texids finds no target again. Put the
        # material back for the background geoms before resolving them. Harmless
        # on the live path (the values are already the nominal ones).
        for _nm in tuple(_WALL_GEOMS) + ("floor",):
            _g = _geom_id(m, _nm)
            if _g is not None and int(m.geom_matid[_g]) < 0 <= int(nominal["geom_matid"][_g]):
                m.geom_matid[_g] = nominal["geom_matid"][_g]
    if cfg.dr_background_image:
        # bg_index makes the draw REPRODUCIBLE: which photo an episode got is
        # recorded as an INDEX (background_files() is sorted), so the dataset can
        # be rebuilt frame-identically without carrying 2.3 MB of pixels.
        # ⚠️ The photo lives in `tex_rgb`, which is NOT in MODEL_STATE_FIELDS, so
        # restore_model_state does NOT bring it back -- recording the index here
        # is the only thing that makes a re-render reproduce the background.
        # Collection left this unrecorded until 2026-09-23 and the replay gate
        # failed with max |diff| 212 the moment photo backgrounds were turned on.
        if bg_index is None:
            _nbg = len(background_files(cfg.dr_background_image))
            bg_index = int(rng.integers(_nbg) if hasattr(rng, "integers")
                           else rng.randint(_nbg))
        _applied = apply_background_image(env, cfg, rng, index=bg_index)
        # None => no offscreen context (state-only env), i.e. no photo was
        # painted; do not claim one.
        setattr(env, "_s2r_bg_index", bg_index if _applied is not None else None)
        # Never let this fail quietly again: if the targets could not be
        # resolved, the run would train on a background that is not the one the
        # config asks for, and nothing downstream would ever notice.
        assert _applied is not None or getattr(sim, "_render_context_offscreen", None) is None, (
            "dr_background_image is set but no wall/floor texture could be resolved "
            "(_bg_texids returned nothing). Something stripped geom_matid before "
            "this ran -- the photo would be silently ignored.")

    # ---- arena pegs (Square) ----------------------------------------------
    # Placement tolerance for a peg the task's sampler never touches. Composed
    # from `nominal`, like every other DR here, so repeated resets on one model
    # cannot accumulate. body_pos/body_quat are in MODEL_STATE_FIELDS, so the
    # draw is captured by capture_model_state and a pixel re-render reproduces
    # it exactly (the replay contract, docs/infra.md §14e).
    if cfg.any_peg_dr() and nominal is not None:
        for name, (p0, q0) in (nominal.get("peg_pose") or {}).items():
            bid = _body_id(m, name)
            if bid is None:
                continue
            if cfg.dr_peg_xy:
                dxy = rng.uniform(-cfg.dr_peg_xy, cfg.dr_peg_xy, 2)
                m.body_pos[bid] = p0 + np.array([dxy[0], dxy[1], 0.0])
            if cfg.dr_peg_yaw_deg:
                yaw = np.radians(rng.uniform(-cfg.dr_peg_yaw_deg, cfg.dr_peg_yaw_deg))
                cz, sz = np.cos(yaw), np.sin(yaw)
                Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
                m.body_quat[bid] = mat2quat_wxyz(Rz @ quat_wxyz2mat(q0))

    # ---- cameras -----------------------------------------------------------
    if cfg.cam_follow_base_dr and cfg.any_base_dr():
        sim.forward()                 # body_xpos of the base as moved above
    for cam in cfg.cameras:
        if cam.is_inert():
            continue
        _apply_camera(env, m, sim, cam, cfg, rng)

    sim.forward()


_BG_FILES: dict = {}


def background_files(spec: str) -> list:
    """Sorted image list for a "+"/","-separated set of directories (cached).

    Sorted, so an index into this list is a STABLE identifier for "which photo"
    -- that is what the re-render pass records per episode instead of the 2.3 MB
    of pixels (the replay contract cannot carry the texture itself)."""
    key = str(spec)
    if key not in _BG_FILES:
        exts = (".jpg", ".jpeg", ".png")
        files = []
        for d in key.replace("+", ",").split(","):
            d = d.strip()
            if not d:
                continue
            for root, _dirs, names in os.walk(d):
                files += [os.path.join(root, n) for n in sorted(names)
                          if n.lower().endswith(exts)]
        files = sorted(files)
        assert files, f"dr_background_image: no images found under {spec!r}"
        _BG_FILES[key] = files
    return _BG_FILES[key]


def _bg_texids(m):
    """(wall_texid, floor_texid); either may be None. Resolved through the geoms
    rather than hard-coded ids, which differ per task/arena."""
    wall = floor = None
    for g in range(m.ngeom):
        nm = _geom_name(m, g) or ""
        mid = int(m.geom_matid[g])
        if mid < 0:
            continue
        tid = int(m.mat_texid[mid])
        if tid < 0:
            continue
        if nm.startswith("wall_") and wall is None:
            wall = (tid, mid)
        elif nm == "floor" and floor is None:
            floor = (tid, mid)
    return wall, floor


def apply_background_image(env, cfg: Sim2RealConfig, rng=None, index=None):
    """Paint a photo onto the wall (and floor) texture. Returns the file used.

    Writes straight into `model.tex_rgb` and re-uploads that texture to the GPU;
    `mat_texrepeat` is forced to 1 so the photo shows once instead of tiling.
    A state-only env has no render context -- the write is then a harmless no-op.
    """
    if not cfg.enabled or not cfg.dr_background_image:
        return None
    sim = env.sim if hasattr(env, "sim") else env.env.sim
    m = sim.model
    ctx = getattr(sim, "_render_context_offscreen", None)
    if ctx is None:
        # No offscreen context => nothing is ever rendered from this env (a
        # state-only RL run, or the bare MimicGen class). Decoding a photo and
        # rewriting tex_rgb would be pure cost, and it is what made a state-only
        # expert run pay for a randomisation it cannot see. Verified that this
        # cannot silently skip a PIXEL episode: PixelRobosuite has the context
        # already AT CONSTRUCTION, i.e. before the first reset() ever calls this.
        return None
    from PIL import Image
    files = background_files(cfg.dr_background_image)
    rng = rng if rng is not None else np.random
    idx = int(index) if index is not None else int(rng.integers(len(files))
                                                   if hasattr(rng, "integers")
                                                   else rng.randint(len(files)))
    path = files[idx % len(files)]
    img = Image.open(path).convert("RGB")
    tex = getattr(m, "tex_rgb", None)
    if tex is None:                       # mujoco >= 3 renamed it
        tex = m.tex_data
    targets = [t for t in _bg_texids(m) if t is not None]
    if not cfg.dr_background_floor:
        targets = targets[:1]
    for tid, mid in targets:
        w, h = int(m.tex_width[tid]), int(m.tex_height[tid])
        adr = int(m.tex_adr[tid])
        tex[adr:adr + 3 * w * h] = np.asarray(img.resize((w, h)), dtype=np.uint8).ravel()
        m.mat_texrepeat[mid] = [1.0, 1.0]
        if ctx is not None:
            # NOT ctx.upload_texture(): robosuite passes its own MjModel wrapper
            # to mjr_uploadTexture, which only accepts the raw mujoco.MjModel.
            import mujoco
            ctx.gl_ctx.make_current()
            mujoco.mjr_uploadTexture(getattr(m, "_model", m), ctx.con, tid)
    return path


def apply_cameras(env, cfg: Sim2RealConfig, rng=None) -> int:
    """Apply ONLY the cell's camera extrinsics/intrinsics to an already-built sim.

    Needed after `restore_model_state`: that vector carries cam_pos/cam_quat/
    cam_fovy as they were WHEN THE EPISODE WAS COLLECTED, so restoring it silently
    puts back the OLD camera even when the cell has since been re-calibrated --
    a re-render with a new extrinsic would quietly produce the old view.
    Returns the number of cameras written."""
    if not cfg.enabled or not cfg.cameras:
        return 0
    sim = env.sim if hasattr(env, "sim") else env.env.sim
    m = sim.model
    n = 0
    for cam in cfg.cameras:
        if cam.is_inert():
            continue
        _apply_camera(env, m, sim, cam, cfg, rng if rng is not None else np.random)
        n += 1
    sim.forward()
    return n


def _apply_camera(env, m, sim, cam: RealCameraConfig, cfg: Sim2RealConfig, rng):
    cid = _cam_id(m, cam.name)
    if cid is None:
        return                      # state-only runs build no cameras
    d = sim.data
    R_cv = cam.rotation_opencv()
    p = np.asarray(cam.pos, dtype=np.float64)

    dr_pos, dr_rot, dr_fovy = cam.dr_widths(cfg)
    if np.any(dr_pos):
        w = np.asarray(dr_pos, dtype=np.float64)        # scalar or per-axis
        p = p + rng.uniform(-w, w, 3)
    if R_cv is not None and np.any(dr_rot):
        R_cv = R_cv @ _small_rot(rng, np.asarray(dr_rot, dtype=np.float64))

    if cam.frame == "base":
        bid = _body_id(m, "robot0_base")
        assert bid is not None, "sim2real: robot0_base not found"
        fp, fR = d.body_xpos[bid].copy(), d.body_xmat[bid].reshape(3, 3).copy()
    elif cam.frame == "hand":
        bid = _body_id(m, "robot0_right_hand")
        assert bid is not None, "sim2real: robot0_right_hand not found"
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
        if dr_fovy:
            fovy += rng.uniform(-dr_fovy, dr_fovy)
        m.cam_fovy[cid] = float(fovy)


# ---------------------------------------------------- render-replay contract
# Every model field this module (or domain_shift) can write per episode. Storing
# the RESULT -- not the RNG seed -- is deliberate: a seed only replays if the
# draw ORDER never changes, and it already changed once (dr_base_height is drawn
# first so old height-only configs keep reproducing). With the fields themselves
# on disk, re-rendering is independent of the sampling code entirely.
# Rendering a frame is a pure function of (these fields, qpos) => an offline pass
# can reproduce any camera, byte for byte. Gate: scripts/test_rerender.py.
MODEL_STATE_FIELDS = (
    "body_pos", "body_quat", "geom_rgba", "geom_matid", "geom_size",
    "light_diffuse", "light_ambient", "light_specular", "light_dir", "light_pos",
    # dr_light_shadow_n picks WHICH lights cast shadows each episode. Left out
    # until 2026-09-23, when photo backgrounds and the strip light went in and
    # the replay gate failed at max |diff| 118 with everything else restored:
    # the re-render drew its own shadow set. A field that is randomised per
    # episode and not listed here is silently unreproducible.
    "light_castshadow",
    "cam_pos", "cam_quat", "cam_fovy",
)
# ⚠️ Adding a field changes the model_state VECTOR LENGTH. restore_model_state
# asserts the length, and every dataset stamps `model_state_layout`, so old and
# new data cannot be mixed -- change it only when no dataset depends on it.


def _model_of(env):
    sim = env.sim if hasattr(env, "sim") else env.env.sim
    return sim.model


def model_state_layout(env) -> list:
    """[(field, shape), ...] for the flat vector; constant for a given task+cell,
    so it is stored once per dataset rather than per episode."""
    m = _model_of(env)
    return [(f, tuple(np.asarray(getattr(m, f)).shape)) for f in MODEL_STATE_FIELDS]


def capture_model_state(env) -> np.ndarray:
    """Flat float64 snapshot of every per-episode-mutable model field."""
    m = _model_of(env)
    return np.concatenate([
        np.asarray(getattr(m, f), dtype=np.float64).ravel() for f in MODEL_STATE_FIELDS])


def restore_model_state(env, vec) -> tuple:
    """Write a `capture_model_state` vector back. Restores the episode's DR draw
    (base pose, lights, colours, camera) without re-running any sampling.

    Returns the tuple of fields it could NOT restore because the vector predates
    them. Callers must handle a non-empty result: those fields keep whatever the
    previous episode left, which is a silent cross-episode dependency."""
    m = _model_of(env)
    vec = np.asarray(vec, dtype=np.float64)
    sizes = [(f, int(np.prod(np.asarray(getattr(m, f)).shape)) or 1)
             for f in MODEL_STATE_FIELDS]
    want = sum(n for _, n in sizes)
    n_have = int(vec.shape[0])
    _skipped = ()
    if n_have != want:
        # A dataset recorded before a field JOINED MODEL_STATE_FIELDS is short by
        # that field's width -- and the field may sit in the MIDDLE of the layout
        # (light_castshadow went in between light_pos and cam_pos), so a plain
        # prefix restore would misalign everything after it. Drop the known
        # late additions, newest first, until the widths agree. The naive loop
        # used to die inside reshape with a message naming the wrong field.
        trial = list(sizes)
        for f in _MODEL_STATE_ADDED_LATE:
            if sum(n for _, n in trial) == n_have:
                break
            trial = [(g, n) for g, n in trial if g != f]
        assert sum(n for _, n in trial) == n_have, (
            f"model state length {n_have} matches neither this build's layout "
            f"({want}) nor any prefix of it with the known late additions "
            f"{list(_MODEL_STATE_ADDED_LATE)} removed. Restoring it would "
            f"misalign the fields, so it is refused.")
        dropped = [f for f, _ in sizes if f not in {g for g, _ in trial}]
        if dropped and not getattr(restore_model_state, "_warned", False):
            print(f"[sim2real] model_state is {n_have} long, this build wants "
                  f"{want}: {dropped} joined MODEL_STATE_FIELDS after this "
                  f"dataset was recorded, so they are NOT reproducible from it "
                  f"and keep whatever the current model has.", flush=True)
            restore_model_state._warned = True
        sizes = trial
        _skipped = tuple(dropped)
    else:
        _skipped = ()
    i = 0
    for f, n in sizes:
        cur = np.asarray(getattr(m, f))
        chunk = vec[i:i + n].reshape(cur.shape)
        i += n
        if np.issubdtype(cur.dtype, np.integer):
            chunk = np.rint(chunk).astype(cur.dtype)
        getattr(m, f)[:] = chunk
    return _skipped


# Fields appended to MODEL_STATE_FIELDS after data had already been recorded,
# NEWEST FIRST. restore_model_state drops them in this order to read an older
# vector. Add to the front whenever the tuple above grows.
_MODEL_STATE_ADDED_LATE = ("light_castshadow",)


PEG_BODIES = ("peg1", "peg2")


def snapshot_visual_nominal(env) -> dict:
    """Capture the values every multiplicative DR range composes from. Taken once
    per rebuilt model; hard_reset means the model is fresh each episode, so this
    is cheap insurance against a gain compounding across episodes."""
    sim = env.sim if hasattr(env, "sim") else env.env.sim
    m = sim.model
    bid = _body_id(m, "robot0_base")
    return {
        # geom_matid is here for ONE reason: dr_background_gray used to strip it
        # to -1, and a re-render restores that -1 out of model_state, after which
        # the photo background can no longer resolve its target. Keeping the
        # pristine value lets apply_sim2real put the material back.
        "geom_matid": m.geom_matid.copy(),
        "light_diffuse": m.light_diffuse.copy(),
        "light_ambient": m.light_ambient.copy(),
        "light_specular": m.light_specular.copy(),
        "light_dir": m.light_dir.copy(),
        "light_pos": m.light_pos.copy(),
        "base_pos": m.body_pos[bid].copy() if bid is not None else None,
        "base_quat": m.body_quat[bid].copy() if bid is not None else None,
        # Arena pegs (Square). Absent on Threading -> empty dict -> inert.
        "peg_pose": {n: (m.body_pos[i].copy(), m.body_quat[i].copy())
                     for n in PEG_BODIES
                     for i in [_body_id(m, n)] if i is not None},
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
def real2sim_arena() -> dict:
    """The measured arena, read from the real2sim task's constants (the single
    source of truth, docs/real2sim.md §2e). Imported lazily: env.peg_insertion
    pulls in mimicgen."""
    from env import peg_insertion as pi
    return {
        "table_above_base": float(pi.BASE_BELOW_SURFACE),  # surface ABOVE base
        "table_full_size": tuple(float(v) for v in pi.MEASURED_TABLE_SIZE),
        "base_xy": tuple(float(v) for v in pi.MEASURED_BASE_XY),
    }


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

    # `inherit: <relative path>` -- one cell file built on another. The DROID
    # cell is one physical bench: the camera solve, the table colour and the DR
    # widths are properties of the room, not of the task, so a per-task file
    # that restated them would silently go stale the next time the cameras are
    # calibrated (exactly the class of drift docs/infra.md §14b is about).
    # Child keys win; `cameras` is replaced wholesale, not merged per camera,
    # because a half-overridden extrinsic is never what anyone means.
    parent = raw.pop("inherit", None)
    if parent is not None:
        import os as _os
        ppath = parent if _os.path.isabs(parent) else _os.path.join(
            _os.path.dirname(_os.path.abspath(path)), parent)
        with open(ppath) as f:
            base = yaml.safe_load(f) or {}
        assert "inherit" not in base, f"{ppath}: inherit chains are not supported"
        base.update(raw)
        raw = base

    cams = []
    for i, c in enumerate(raw.pop("cameras", []) or []):
        unknown = set(c) - set(RealCameraConfig.__dataclass_fields__)
        assert not unknown, f"{path}: cameras[{i}] has unknown keys {sorted(unknown)}"
        if c.get("real_intrinsics") and not os.path.isabs(c["real_intrinsics"]):
            c = dict(c, real_intrinsics=os.path.join(
                os.path.dirname(os.path.abspath(path)), c["real_intrinsics"]))
        if c.get("real_intrinsics"):
            assert os.path.exists(c["real_intrinsics"]), (
                f"{path}: cameras[{i}].real_intrinsics {c['real_intrinsics']} not found")
        cams.append(RealCameraConfig(**{
            k: (tuple(v) if isinstance(v, list) else v) for k, v in c.items()}))

    unknown = set(raw) - set(Sim2RealConfig.__dataclass_fields__)
    assert not unknown, f"{path}: unknown keys {sorted(unknown)}"
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items()}
    if kw.get("arena_from_real2sim"):
        restated = {"table_above_base", "table_full_size", "base_xy"} & set(kw)
        assert not restated, (
            f"{path}: arena_from_real2sim=1 takes {sorted(restated)} from "
            "env/peg_insertion.py; do not restate them here")
        kw.update(real2sim_arena())
    cfg = Sim2RealConfig(cameras=tuple(cams), **kw)
    for c in cfg.cameras:
        c.rotation_opencv()        # validate now, not mid-dataset-generation
    assert cfg.prop_frame in ("world", "base"), f"{path}: prop_frame {cfg.prop_frame!r}"
    if cfg.exec_layer:
        from env.exec_layer import exec_layer_config
        exec_layer_config(cfg.exec_layer)          # validate keys now
    assert cfg.control_freq is None or 1 <= float(cfg.control_freq) <= 500, (
        f"{path}: control_freq {cfg.control_freq!r}")
    return cfg


def apply_physics_dr(env, cfg: Sim2RealConfig, rng=None):
    """Per-episode arm / gripper physics DR (see the dr_arm_* fields). Call after the reset, like apply_sim2real.
    -> dict of what was drawn (for logs), or None."""
    if not cfg.any_physics_dr():
        return None
    rng = np.random if rng is None else rng
    m = env.sim.model
    rob = env.robots[0]
    jv = np.asarray(rob._ref_joint_vel_indexes)
    fitted = getattr(env, "exec_arm_params", None)       # set by Threading_RealCellRobotiq after the exec layer
    arm0 = fitted[0] if fitted is not None else m.dof_armature[jv].copy()
    fr0 = fitted[1] if fitted is not None else m.dof_frictionloss[jv].copy()
    out = {}
    if cfg.dr_arm_armature:
        k = rng.uniform(1 - cfg.dr_arm_armature, 1 + cfg.dr_arm_armature, len(jv))
        m.dof_armature[jv] = arm0 * k
        out["armature_scale"] = np.round(k, 3).tolist()
    if cfg.dr_arm_frictionloss:
        k = rng.uniform(1 - cfg.dr_arm_frictionloss, 1 + cfg.dr_arm_frictionloss, len(jv))
        m.dof_frictionloss[jv] = fr0 * k
        out["frictionloss_scale"] = np.round(k, 3).tolist()
    if cfg.dr_gripper_dampratio is not None:
        dr = float(rng.uniform(*cfg.dr_gripper_dampratio))
        for g in range(m.ngeom):
            n = m.geom_id2name(g) or ""
            if n.startswith("gripper0") and "collision" in n:
                m.geom_solref[g][1] = dr
        out["gripper_dampratio"] = round(dr, 3)
    return out

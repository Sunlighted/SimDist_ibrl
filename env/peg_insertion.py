"""Real2sim square peg-in-hole task (docs/real2sim.md Phase B).

Geometry comes from the scanned real board/peg (real2sim/scan_pipeline/peg_task,
2026-09-15): target = the CORNER rectangular pocket, 35.6 x 23.8 mm through-hole,
board 41 mm thick, peg 33.0 x 21.2 x 122.9 mm => ~1.3 mm/side clearance on both
axes -- AT the 1.3 mm floor of findings §39 (STL-sliced; caliper-check the real
parts). ⚠️ Before 2026-09-23 the pocket was the OVAL's bounding box (37.9 x 26.1,
2.4 mm/side): every earlier run trained on the wrong hole.

Design decisions (docs/real2sim.md §3/§5):
- Collision = box primitives only (4 walls + 1 peg box): exact collision and
  fast physics (physics is now the throughput bottleneck, real2sim.md §1).
- The peg STARTS ON THE TABLE, standing upright (project-owner decision
  2026-09-15, overriding the earlier pre-grasped design): the task is the full
  grasp -> transport -> insert sequence, like the other MimicGen tasks. The
  upright rest pose is statically stable (tip-over angle ~9.8 deg) and matches
  how the operator will stage the real peg.
- The board is a STATIC fixture (no joint) at a fixed pose, matching the real
  cell where the board is clamped to the table.

Importing this module registers PegInsertion / PegInsertion_D0 with robosuite
(EnvMeta auto-registration), same mechanism as env/threading_variants.py.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string, find_elements
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)

from mimicgen.envs.robosuite.single_arm_env_mg import SingleArmEnv_MG


# ---- measured geometry (all meters; source: the real scans in
# real2sim/scan_pipeline/peg_task/meshes/). The PHYSICAL model is the scanned
# STL itself (visual mesh + its convex-decomposition hulls for collision);
# these constants only parameterize the SUCCESS CHECK and the reset sampler.
HOLE_HALF_X = 0.01778  # corner pocket opening 35.56 mm (was the oval bbox, 37.86)
HOLE_HALF_Y = 0.011885  # corner pocket opening 23.77 mm (was the oval bbox, 26.07)
BOARD_HALF_Z = 0.02049  # board 41 mm thick
PEG_HALF = (0.016524, 0.010622, 0.061471)  # peg 33.0 x 21.2 x 122.9 mm (bbox)
# => clearance: x (35.56-33.05)/2 = 1.26 mm/side, y (23.77-21.24)/2 = 1.26 mm/side

# ---- MEASURED cell layout (real2sim/scan_pipeline/cell_layout.json, from the
# scan5 splat + scan_run2 grounding; see docs/real2sim.md §2e) --------------
# Work surface = the mats on the optical breadboard: 1.240 x 0.739 m, with the
# robot base flange 20.7 mm BELOW it (robosuite's stock arena puts the base
# 112 mm ABOVE the table -- a 133 mm error that made sim and real impossible to
# align at both the base and the work surface).
# Axes are swapped vs the measurement frame (the table's long axis runs along
# the robot's y), and the measured 91.2 deg base yaw is snapped to 90 deg:
# a 1.2 deg residual, i.e. ~13 mm at the far table corner.
MEASURED_TABLE_SIZE = (0.739, 1.240, 0.03)
BASE_BELOW_SURFACE = 0.0207       # base flange -> work surface, measured
MOUNT_BASE_Z = 0.912              # robosuite mount height (fixed by the mesh)
MEASURED_TABLE_OFFSET = (0.0, 0.0, MOUNT_BASE_Z + BASE_BELOW_SURFACE)
MEASURED_BASE_XY = (-0.489, -0.034)   # base wrt work-surface centre

# Franka's own "ready" configuration, which is what the real cell resets to.
# robosuite's Panda default is wrong for this task in the MEASURED arena: it puts
# the grip site only 78 mm above the table top, i.e. BELOW the top of the 123 mm
# peg it has to grasp (1.011 m vs the peg tip at 1.056 m), so the episode begins
# with the gripper under the object. Ready starts 473 mm up with the approach axis
# exactly vertical ([0,0,-1] vs 7.3 deg off), and matching the real robot's reset
# means sim and real episodes start from the same configuration.
# The REAL external ZED2i, expressed in sim world coordinates
# (real2sim/scan_pipeline/zed_sim_camera.json): T_base_board (6.1 mm / 0.11 deg)
# composed with T_board_zed (correct-K PnP, 1.33 px), plus the measured base xpos.
# This is the viewpoint the deployed system actually sees, so it is the camera the
# pixel stage should consume -- robosuite's `agentview` clips the board out of
# frame in 100% of resets, while this one sees both objects whole.
# MuJoCo camera convention: looks down -z with +y up (the PnP pose is OpenCV, so
# the rotation carries a diag(1,-1,-1) flip).
ZEDVIEW_POS = (-0.3536, 0.4803, 1.6911)
ZEDVIEW_QUAT = (-0.47888, -0.13004, 0.36614, 0.78721)   # wxyz
# ★ hd720 FACTORY intrinsics (unrectified) -- what publish_zed_uvc.py streams --
# NOT the scan_run2 SDK-rectified HD2K values this used to carry. The ZED's modes
# are NOT rescales of one another: hd1080 fx is 2.00x hd720, not 1.50x, so the FOV
# genuinely differs per mode and must come from that mode's own K.
# zed2i hd720: fx 532.445 fy 532.470 cx 622.555 cy 352.755
#   -> principal-point square crop 705 px -> fovy 67.010
# Assumes the publisher undistorts to a pinhole with newK = factory K, D = 0.
# Verified: that square crop is 100% free of undistortion black border, so
# getOptimalNewCameraMatrix(alpha=0) buys nothing (it would give 68.134).
ZEDVIEW_FOVY = 67.010

# The REAL wrist ZED Mini, expressed in robosuite's `robot0_right_hand` frame.
#
# ★ WHY THIS IS NOT JUST T_ee_zedm (docs/real2sim.md §2d-z): polymetis reports the
# FLANGE, and robosuite's right_hand frame is that flange rotated +45.03 deg about
# z -- constant to 0.000 deg over all 14 scan poses. §2d-y verified only POSITION
# (0.5 mm), which is blind to rotation about the frame origin. So the hand-frame
# pose is Rz(45.03) @ T_ee_zedm, NOT T_ee_zedm. Skipping the Rz swings the
# camera 45 deg around the flange and the gripper covers 23.8% of the frame
# instead of 4.6% -- see viz/wrist_handeye/.
#
# This REPLACES robosuite's stock `eye_in_hand` for sim2real work: that camera is
# at pos (0.05, 0, 0) with fovy 75, against the real Mini's (-29.4, -77.2, 15.5) mm
# and 45.0 deg. Training on eye_in_hand teaches a viewpoint the robot cannot
# produce. (The 77 mm offset is why one finger sits in the frame corner.)
#
# Derivation, since the sign is easy to get backwards (I did, first pass):
# frame_check.py measures R_hand^T @ R_ee = Rz(+45.03), so T_sw_ee = T_sw_hand @ Rz,
# and a camera parented to right_hand needs T_hand_cam = Rz @ T_ee_zedm.
# inv(Rz) @ T_ee_zedm puts the gripper across 29% of the frame -- verified wrong
# against the real finger mask, which is the regression test below.
#
# Quat is MuJoCo convention (-z fwd, +y up); T_ee_zedm is OpenCV, hence the
# diag(1,-1,-1) flip baked into the value below.
ZEDM_WRIST_POS = (0.077238, -0.029357, 0.015518)
ZEDM_WRIST_QUAT = (0.092666, 0.671783, 0.723946, 0.126582)   # wxyz
# zedm hd720: fx 771.808 fy 771.705 cx 652.745 cy 351.798 -> crop 703 px
#   -> fovy 48.977   (alpha=0 alternative: 49.521)
# ⚠️ T_ee_zedm was solved on SDK-RECTIFIED images, which also carry a small
# stereo-rectification rotation that undistorting a single eye does NOT. At the
# wrist's ~0.3 m working distance that is ~5 mm per degree -- larger than the
# 2.2 mm hand-eye LOO -- so measure it from the stereo conf, do not assume.
ZEDM_WRIST_FOVY = 48.977

# Both FOVs are the VERTICAL fov of a square crop centred on the camera's
# PRINCIPAL POINT, not on the image centre. MuJoCo renders a centred pinhole and
# cannot represent an off-centre cx/cy (the Mini's cx is 40.6 px off), so cropping
# about (cx, cy) is what makes the sim camera exactly right instead of ~24 px off
# at 1280 width. The deployed pipeline MUST use the same crop:
#   zed2i: 1240x1240 at (1083.7, 621.6)    zedm: 1223x1223 at (1144.6, 611.6)
# then resize to rl_image_size. A crop about the image centre instead silently
# reintroduces the principal-point error into every training frame.

# LIGHTING: the real cell is lit by a STRIP fixture above the setup, running
# along the LEFT edge of the table from the robot's perspective (owner, measured
# setup). The robot faces +x, so its left is +y, i.e. the strip runs along x at
# y ~ +0.62 (the table's +y edge, half of MEASURED_TABLE_SIZE[1]=1.240).
#
# Why this matters and is not cosmetic: robosuite's default is ONE directional
# light pointing almost straight down (dir z = -0.96). The peg is a tall thin
# box, so its VERTICAL faces get N.L ~ 0.27 and render mid-grey despite an
# albedo of 0.95 -- which is exactly the "MuJoCo peg looks dark" artefact. A
# side-running strip lights those faces the way the real one does.
#
# MuJoCo has no area lights, so the strip is N point lights in a row. Default
# attenuation is (1,0,0) = no distance falloff, so STRIP_Z sets the DIRECTION of
# the lighting but not its intensity -- the height is therefore the least
# critical number here.
STRIP_N = 5
STRIP_Y = 0.62                    # table +y edge = robot's left
STRIP_Z = 2.30                    # ceiling fixture; ~1.37 m above the work surface
STRIP_X = (-0.37, 0.37)           # spans the table's x extent (0.739 full)
STRIP_DIFFUSE = 0.55              # per-lamp; N lamps sum to roughly the old 0.7
STRIP_AMBIENT = 0.22              # room bounce -- a real room is not ambient-0

FRANKA_READY_QPOS = np.array(
    [0.0, -np.pi / 4.0, 0.0, -3.0 * np.pi / 4.0, 0.0, np.pi / 2.0, np.pi / 4.0]
)

_SCAN_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "real2sim", "scan_pipeline", "peg_task",
)


class ScannedBoardObject(MujocoXMLObject):
    """The scanned shape-sorter board, used AS-IS: visual = board.stl, collision
    = its convex-decomposition hulls (board_hull_*.stl -- machine decomposition
    of the same mesh, all cavities included, no hand-fitted primitives).

    Object frame: origin at the TARGET POCKET center (x, y) and slab mid-plane
    (z) -- board_scan.xml offsets every geom by the pocket center measured in
    the mesh frame -- so a body-frame point with |x|<HOLE_HALF_X,
    |y|<HOLE_HALF_Y, z<BOARD_HALF_Z is inside the pocket column.
    Static fixture: joints=None (real board is clamped).
    """

    hole_half_x = HOLE_HALF_X
    hole_half_y = HOLE_HALF_Y
    half_z = BOARD_HALF_Z

    def __init__(self, name):
        super().__init__(
            os.path.join(_SCAN_DIR, "board_scan.xml"),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class ScannedPegObject(MujocoXMLObject):
    """The scanned peg, used AS-IS: the mesh is both the visual and the
    collision geom (it is convex up to print/scan rounding, so MuJoCo's convex
    hull IS the part). Mesh frame origin = bbox center, so PEG_HALF applies."""

    def __init__(self, name):
        super().__init__(
            os.path.join(_SCAN_DIR, "peg_scan.xml"),
            name=name,
            joints="default",
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class PegInsertion(SingleArmEnv_MG):
    """Square(ish) peg-in-hole with a pre-grasped peg and a static board.

    Success: the peg's lower tip is inside the hole column and at least
    `insert_depth` below the board's top surface.
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        board_pos_noise=0.0,
        board_yaw_noise=0.0,
        cam_pos_noise=0.0,
        cam_rot_noise=0.0,
        light_noise=0.0,
        # ★ 2026-09-23 defaults: the old ones (0.015 / None / None / None) let a
        # policy score by wedging a TILTED peg ~8 mm into the soft default-contact
        # walls just past the mouth (IBRL 0.86 "success", 0.14 without
        # penetration). Measured with scripts/verify_peg_ibrl.py: under these
        # values the scripted expert is 0.95 (n=60, <=0.43 mm, <=0.7 deg) and that
        # same hacking policy drops to 0.04. Pass the old values to reproduce
        # pre-fix runs.
        insert_depth=0.030,          # of the 41 mm through-hole (11 mm above the table)
        success_max_tilt=5.0,        # deg; the expert stays <= 2 deg
        board_solref=(0.004, 1.0),   # 0.004 = 2x timestep, MuJoCo's stiffest stable value
        board_solimp=(0.99, 0.999, 0.001),
        hole_clearance=None,         # m per side; None = the scanned pocket (1.26 mm/side)
        initialization_noise="default",
        table_full_size=MEASURED_TABLE_SIZE,
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array(MEASURED_TABLE_OFFSET)

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs

        # board pose on the table (hole center in table-frame xy). (0.0538,
        # -0.1538) = the old (0, -0.15) + the oval->corner pocket shift, so the
        # PHYSICAL board stays exactly where it was (and where the real one is).
        self.board_xy = (0.0538, -0.1538)
        # BOARD RESET JITTER. The real board is placed by hand and will not land
        # in the same spot twice, so a fixed board teaches the policy a constant
        # it cannot rely on in the cell. Default 0.0 keeps every existing run
        # byte-identical. Applied in _load_model, which hard_reset=True re-runs
        # every reset -- so this is a no-op if anyone sets hard_reset=False.
        # The board pose IS observable (object-state carries board_pos/quat and
        # board_to_robot0_eef_*), so this does not make the task partially
        # observed for the state-based expert.
        self.board_pos_noise = float(board_pos_noise)   # metres, uniform +-
        # CAMERA EXTRINSIC DR. The external ZED is re-mounted between sessions and
        # the wrist bracket can shift; §4 B calls this mandatory. Position is
        # sampled uniformly IN A BALL (not per-axis, which would bias corners).
        # Applied to the MuJoCo camera so the splat composite and the MuJoCo
        # render stay in agreement -- jittering only the splat renderer's own
        # cam_pos_sigma would desync the two halves of every frame.
        self.cam_pos_noise = float(cam_pos_noise)       # metres, ball radius
        self.cam_rot_noise = float(cam_rot_noise)       # degrees, per-axis
        # LIGHTING DR around the measured strip (STRIP_* above).
        self.light_noise = float(light_noise)           # 0 = off, 1 = nominal band
        self.board_yaw_noise = float(board_yaw_noise)   # degrees, uniform +-
        # success = tip this far below the board top surface
        self.insert_depth = float(insert_depth)
        # success also requires the peg's long axis within this many degrees of
        # the board normal (None = no tilt check). A peg wedged in at an angle
        # can have its tip inside the pocket column without being inserted.
        self.success_max_tilt = None if success_max_tilt is None else float(success_max_tilt)
        # CONTACT STIFFNESS of the 4 pocket walls (None = MuJoCo default soft
        # contact, solref 0.02 / solimp 0.9-0.95, which lets a pushed peg sink
        # several mm -- up to tens of mm -- into a wall). Applied to the walls
        # only, with a large solmix so the peg-wall PAIR takes the wall's values
        # while peg-finger contacts (the grasp) are untouched.
        self.board_solref = None if board_solref is None else tuple(float(x) for x in board_solref)
        self.board_solimp = None if board_solimp is None else tuple(float(x) for x in board_solimp)
        # POCKET CLEARANCE per side (sim-only diagnostic knob; the real pocket is
        # fixed). None = the scanned geometry. Moves the 4 collision walls' INNER
        # faces (outer faces fixed) and the success-check pocket size together;
        # the visual mesh is NOT changed, so rendered frames still show the scanned hole.
        self.hole_clearance = None if hole_clearance is None else float(hole_clearance)

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    # ---------------------------------------------------------------- model
    def reward(self, action=None):
        reward = 1.0 if self._check_success() else 0.0
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward

    def _jitter_pose(self, pos, quat_wxyz):
        """pos + uniform-in-BALL offset; quat + small random rotation.

        Ball, not per-axis uniform: a per-axis cube biases toward the corners and
        the radius would be sqrt(3) larger on the diagonals than advertised.
        """
        pos = np.asarray(pos, dtype=float).copy()
        if self.cam_pos_noise > 0:
            v = np.random.normal(size=3)
            v /= max(np.linalg.norm(v), 1e-12)
            r = self.cam_pos_noise * np.random.uniform() ** (1.0 / 3.0)
            pos = pos + v * r
        q = np.asarray(quat_wxyz, dtype=float).copy()
        if self.cam_rot_noise > 0:
            d = np.radians(np.random.uniform(-self.cam_rot_noise, self.cam_rot_noise, 3))
            h = d / 2.0
            dq = np.array([1.0, h[0], h[1], h[2]])       # small-angle quaternion
            dq /= np.linalg.norm(dq)
            w0, x0, y0, z0 = q; w1, x1, y1, z1 = dq
            q = np.array([w0*w1 - x0*x1 - y0*y1 - z0*z1,
                          w0*x1 + x0*w1 + y0*z1 - z0*y1,
                          w0*y1 - x0*z1 + y0*w1 + z0*x1,
                          w0*z1 + x0*y1 - y0*x1 + z0*w1])
            q /= np.linalg.norm(q)
        return pos, q

    def _add_strip_light(self, mujoco_arena):
        """Replace the arena's default overhead light with the real cell's strip.

        See the STRIP_* block above. Removes robosuite's single near-vertical
        directional light first -- leaving it in would double-light the scene and
        keep the top faces blown out relative to the sides.
        """
        wb = mujoco_arena.worldbody
        for lt in list(wb.findall("light")):
            wb.remove(lt)
        x0, x1 = STRIP_X
        n = self.light_noise
        # one draw PER RESET shared by the whole strip: a real fixture moves and
        # dims as a unit, so per-lamp independent noise would be the wrong model.
        dy = np.random.uniform(-0.08, 0.08) * n
        dz = np.random.uniform(-0.15, 0.15) * n
        gain = 1.0 + np.random.uniform(-0.25, 0.25) * n
        amb = 1.0 + np.random.uniform(-0.35, 0.35) * n
        ddir = np.random.uniform(-0.10, 0.10, 2) * n
        for i in range(STRIP_N):
            x = x0 + (x1 - x0) * (i / max(STRIP_N - 1, 1))
            ET.SubElement(wb, "light", attrib={
                "name": f"striplight{i}",
                "pos": f"{x} {STRIP_Y + dy} {STRIP_Z + dz}",
                "dir": f"{ddir[0]} {-0.35 + ddir[1]} -0.94",
                "directional": "false",
                "diffuse": " ".join([str(STRIP_DIFFUSE * gain)] * 3),
                "specular": "0.12 0.12 0.12",
                # one lamp carries the room-bounce ambient; N copies would stack
                "ambient": " ".join([str(STRIP_AMBIENT * amb if i == 0 else 0.0)] * 3),
                "castshadow": "true" if i == STRIP_N // 2 else "false",
            })

    def _add_zedm_wrist_camera(self):
        """Attach a camera at the calibrated wrist ZED Mini pose to the hand body.

        `mujoco_arena.set_camera` only makes world-fixed cameras, so a wrist camera
        must be a child of `right_hand` in the robot's own MJCF. Idempotent:
        `_load_model` re-runs on every hard reset and the model is rebuilt, but
        `robot_model` is not, so a second append would duplicate the camera name
        and MuJoCo would refuse to compile ("repeated name in camera array").
        """
        model = self.robots[0].robot_model
        hand = find_elements(
            root=model.root, tags="body", attribs={"name": model.correct_naming("right_hand")}
        )
        if hand is None:                      # non-Panda arm: nothing to attach to
            return
        _wp, _wq = self._jitter_pose(ZEDM_WRIST_POS, ZEDM_WRIST_QUAT)
        name = model.correct_naming("zedm_wrist")
        if find_elements(root=model.root, tags="camera", attribs={"name": name}) is not None:
            return                            # already added on an earlier reset
        ET.SubElement(
            hand,
            "camera",
            attrib={
                "mode": "fixed",
                "name": name,
                "pos": array_to_string(_wp),
                "quat": array_to_string(_wq),
                "fovy": str(ZEDM_WRIST_FOVY),
            },
        )

    def _load_model(self):
        super()._load_model()

        # MEASURED base placement (real2sim.md §2e), not robosuite's default
        # table-relative offset: the real arm is bolted to the breadboard the
        # mats sit on, 20.7 mm BELOW the work surface, 49 cm behind its centre.
        # z stays 0 here -- the mount supplies the 0.912 m base height, and the
        # base-to-surface relation is carried by MEASURED_TABLE_OFFSET[2].
        self.robots[0].robot_model.set_base_xpos(MEASURED_BASE_XY + (0.0,))

        # Reset to Franka's READY pose, not robosuite's Panda default (see the
        # FRANKA_READY_QPOS note). Set here rather than at construction because
        # Robot.load_model() fills init_qpos from the robot model, and _load_model
        # re-runs on every hard reset, so the override has to re-apply each time.
        # hand_init_perturb_joint_sigma noise is added on top of this by
        # Robot.reset(), so the reset distribution is centred on ready.
        self.robots[0].init_qpos = FRANKA_READY_QPOS.copy()

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])
        self._add_agentview_full_camera(mujoco_arena)
        # a camera at the real external ZED's pose (see ZEDVIEW_* above)
        _zp, _zq = self._jitter_pose(ZEDVIEW_POS, ZEDVIEW_QUAT)
        mujoco_arena.set_camera(
            camera_name="zedview",
            pos=_zp,
            quat=_zq,
            camera_attribs={"fovy": str(ZEDVIEW_FOVY)},
        )
        # a camera at the real WRIST ZED Mini's pose (see ZEDM_WRIST_* above).
        # Arena cameras are world-fixed, so this one has to be parented to the
        # hand body instead -- appended to the robot's MJCF before compilation.
        self._add_zedm_wrist_camera()
        self._add_strip_light(mujoco_arena)

        self.board = ScannedBoardObject(name="board_obj")
        # static fixture: no joint, so the pose is baked into the model here
        # (hard_reset=True rebuilds the model every reset, so this stays exact)
        dxy = np.zeros(2)
        if self.board_pos_noise > 0:
            dxy = np.random.uniform(-self.board_pos_noise, self.board_pos_noise, 2)
        board_pos = self.table_offset + np.array(
            [self.board_xy[0] + dxy[0], self.board_xy[1] + dxy[1], self.board.half_z]
        )
        self.board.get_obj().set("pos", array_to_string(board_pos))
        if self.board_yaw_noise > 0:
            yaw = np.radians(np.random.uniform(-self.board_yaw_noise, self.board_yaw_noise))
            self.board.get_obj().set(
                "quat", array_to_string(np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]))
            )

        if self.board_solref is not None or self.board_solimp is not None:
            for g in self.board.get_obj().iter("geom"):
                if "wall_" not in (g.get("name") or "") or g.get("contype") == "0":
                    continue
                if self.board_solref is not None:
                    g.set("solref", array_to_string(np.array(self.board_solref)))
                if self.board_solimp is not None:
                    g.set("solimp", array_to_string(np.array(self.board_solimp)))
                g.set("solmix", "1000")

        if self.hole_clearance is not None:
            self._set_pocket_clearance(self.hole_clearance)

        self.peg = ScannedPegObject(name="peg_obj")

        self._get_placement_initializer()

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.board, self.peg],
        )

    def _get_initial_placement_bounds(self):
        """MimicGen-convention reset bounds (peg only; the board is baked into
        the model). Upright peg (identity quat = long axis vertical), z_rot
        +-30 deg: the rectangular peg's yaw matters for both grasp and hole
        entry, so the window is deliberately narrower than Square's 360 -- widen
        via init_distribution overrides once a policy clears this one."""
        return dict(
            peg=dict(
                x=(-0.10, 0.10),
                y=(0.10, 0.20),
                z_rot=(-np.pi / 6.0, np.pi / 6.0),
                reference=self.table_offset,
            ),
        )

    def _get_placement_initializer(self):
        bounds = self._get_initial_placement_bounds()
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="PegSampler",
                mujoco_objects=self.peg,
                x_range=bounds["peg"]["x"],
                y_range=bounds["peg"]["y"],
                rotation=bounds["peg"]["z_rot"],
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["peg"]["reference"],
                z_offset=0.001,
            )
        )

    def _setup_references(self):
        super()._setup_references()
        self.obj_body_id = dict(
            peg=self.sim.model.body_name2id(self.peg.root_body),
            board=self.sim.model.body_name2id(self.board.root_body),
        )

    # ---------------------------------------------------------------- reset
    def _reset_internal(self):
        super()._reset_internal()
        # Board is baked into the model; only the peg is sampled.
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

    def finger_travel_axis(self):
        """World-frame axis the gripper fingers slide along, computed from the
        model. NOT the grip-site y: for the Panda they differ by 90 deg (that
        assumption once clamped the fingers onto the peg's 33 mm faces with
        6 mm of interpenetration). Used by the scripted expert for yaw align."""
        sim = self.sim
        gripper = self.robots[0].gripper
        act0 = sim.model.actuator_name2id(gripper.actuators[0])
        jid0 = int(sim.model.actuator_trnid[act0][0])
        body0 = int(sim.model.jnt_bodyid[jid0])
        return (
            np.array(sim.data.body_xmat[body0]).reshape(3, 3)
            @ np.array(sim.model.jnt_axis[jid0], dtype=float)
        )

    def _set_pocket_clearance(self, c):
        """Widen (c > scanned) or narrow the pocket to `c` m per side on both axes."""
        ex = c - (HOLE_HALF_X - PEG_HALF[0])
        ey = c - (HOLE_HALF_Y - PEG_HALF[1])
        walls = {}
        for g in self.board.get_obj().iter("geom"):
            name = g.get("name") or ""
            for side in ("wall_left", "wall_right", "wall_front", "wall_back"):
                if name.endswith(side):
                    walls[side] = g
        assert len(walls) == 4, f"expected 4 pocket walls, found {sorted(walls)}"
        for side, g in walls.items():
            pos = np.array([float(v) for v in g.get("pos").split()])
            size = np.array([float(v) for v in g.get("size").split()])
            ax, sign = {"wall_left": (0, -1), "wall_right": (0, +1),
                        "wall_front": (1, -1), "wall_back": (1, +1)}[side]
            e = ex if ax == 0 else ey
            # inner face moves out by e, outer face stays: centre shifts e/2, half-size shrinks e/2
            pos[ax] += sign * e / 2
            size[ax] -= e / 2
            assert size[ax] > 0.001, f"{side} would vanish at clearance {c}"
            if ax == 1:
                size[0] += ex          # front/back span the pocket's x width: close the corners
            g.set("pos", array_to_string(pos))
            g.set("size", array_to_string(size))
        self.board.hole_half_x = HOLE_HALF_X + ex
        self.board.hole_half_y = HOLE_HALF_Y + ey

    # -------------------------------------------------------------- success
    def _peg_tip_pos(self):
        """World position of the peg end currently pointing down."""
        pid = self.obj_body_id["peg"]
        p = np.array(self.sim.data.body_xpos[pid])
        R = np.array(self.sim.data.body_xmat[pid]).reshape(3, 3)
        half_axis = R[:, 2] * PEG_HALF[2]
        a, b = p + half_axis, p - half_axis
        return a if a[2] < b[2] else b

    def _check_success(self):
        """Purely geometric: peg tip inside the hole column, `insert_depth`
        below the board top surface (portable to the real cell, real2sim.md
        Phase B)."""
        tip = self._peg_tip_pos()
        bid = self.obj_body_id["board"]
        b_pos = np.array(self.sim.data.body_xpos[bid])
        b_R = np.array(self.sim.data.body_xmat[bid]).reshape(3, 3)
        d = b_R.T @ (tip - b_pos)
        ok = bool(
            abs(d[0]) < self.board.hole_half_x
            and abs(d[1]) < self.board.hole_half_y
            and d[2] < self.board.half_z - self.insert_depth
        )
        if ok and self.success_max_tilt is not None:
            R = np.array(self.sim.data.body_xmat[self.obj_body_id["peg"]]).reshape(3, 3)
            cos = abs(float(R[:, 2] @ b_R[:, 2]))
            ok = np.degrees(np.arccos(min(1.0, cos))) <= self.success_max_tilt
        return ok

    # ---------------------------------------------------------- observables
    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            @sensor(modality=modality)
            def world_pose_in_gripper(obs_cache):
                return (
                    T.pose_inv(
                        T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"]))
                    )
                    if f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache
                    else np.eye(4)
                )

            sensors = [world_pose_in_gripper]
            names = ["world_pose_in_gripper"]
            actives = [False]

            for obj_name in self.obj_body_id:
                obj_sensors, obj_sensor_names = self._create_obj_sensors(
                    obj_name=obj_name, modality=modality
                )
                sensors += obj_sensors
                names += obj_sensor_names
                actives += [True] * len(obj_sensors)

            for name, s, active in zip(names, sensors, actives):
                observables[name] = Observable(
                    name=name, sensor=s, sampling_rate=self.control_freq, active=active
                )

        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(
                self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw"
            )

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            if any(
                name not in obs_cache
                for name in [f"{obj_name}_pos", f"{obj_name}_quat", "world_pose_in_gripper"]
            ):
                return np.zeros(3)
            obj_pose = T.pose2mat((obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"]))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return (
                obs_cache[f"{obj_name}_to_{pf}eef_quat"]
                if f"{obj_name}_to_{pf}eef_quat" in obs_cache
                else np.zeros(4)
            )

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [
            f"{obj_name}_pos",
            f"{obj_name}_quat",
            f"{obj_name}_to_{pf}eef_pos",
            f"{obj_name}_to_{pf}eef_quat",
        ]
        return sensors, names

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.peg)


class PegInsertion_D0(PegInsertion):
    """Canonical name, matching the MimicGen *_D0 convention used everywhere
    else in this repo (cfgs, STATE_KEYS, resolve_bindings)."""

    pass

"""Real2sim square peg-in-hole task (docs/real2sim.md Phase B).

Geometry comes from the scanned real board/peg (real2sim/scan_pipeline/peg_task,
2026-09-15): hole 37.9 x 26.1 mm, board 41 mm thick, peg 33.0 x 21.2 x 122.9 mm
=> ~2.4 mm/side clearance on BOTH axes (above the 1.3 mm floor of findings §39,
near the ~3 mm working point of Threading).

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

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string
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
HOLE_HALF_X = 0.01893  # target pocket opening 37.86 mm
HOLE_HALF_Y = 0.013035  # target pocket opening 26.07 mm
BOARD_HALF_Z = 0.02049  # board 41 mm thick
PEG_HALF = (0.016524, 0.010622, 0.061471)  # peg 33.0 x 21.2 x 122.9 mm (bbox)
# => clearance: x (37.86-33.05)/2 = 2.41 mm/side, y (26.07-21.24)/2 = 2.41 mm/side

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
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
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
        self.table_offset = np.array((0, 0, 0.8))

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs

        # board pose on the table (hole center in table-frame xy)
        self.board_xy = (0.0, -0.15)
        # success = tip this far below the board top surface
        self.insert_depth = 0.015

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

    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])
        self._add_agentview_full_camera(mujoco_arena)

        self.board = ScannedBoardObject(name="board_obj")
        # static fixture: no joint, so the pose is baked into the model here
        # (hard_reset=True rebuilds the model every reset, so this stays exact)
        board_pos = self.table_offset + np.array(
            [self.board_xy[0], self.board_xy[1], self.board.half_z]
        )
        self.board.get_obj().set("pos", array_to_string(board_pos))

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
        return bool(
            abs(d[0]) < self.board.hole_half_x
            and abs(d[1]) < self.board.hole_half_y
            and d[2] < self.board.half_z - self.insert_depth
        )

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

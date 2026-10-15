"""Custom Threading reset-distribution variants for MimicGen data generation.

Importing this module registers the env(s) with robosuite (robosuite's EnvMeta
metaclass auto-registers any MujocoEnv subclass on definition) WITHOUT editing
the vendored mimicgen submodule. MimicGen's generate_dataset.py can then target
`Threading_Enlarged` (pass --task_name Threading_Enlarged, or set it in the
config) as long as this module is imported in the generating process.

Threading_Enlarged = the DATA-GENERATION reset distribution (deliberately a
superset of the RL "enlarged" env so the generated BC prior over-covers it):
  * needle box x[-0.25, 0.2] y[-0.05, 0.28]  (vs D1's x[-0.2,0.05] y[0.15,0.25];
    vs the RL enlarged env's x[-0.2,0.1] y[-0.05,0.25] — this is larger on both
    axes so demos cover beyond what the RL policy is trained on).
  * tripod + rotations = MimicGen D1.
Same objects/state space as D0/D1 (37-dim), so the generated demos convert and
train exactly like the D1 dataset.
"""

import numpy as np
from mimicgen.envs.robosuite.threading import Threading_D1


class Threading_Enlarged(Threading_D1):
    """D1 tripod/rotations + an enlarged needle placement box (data gen)."""

    def _get_initial_placement_bounds(self):
        return dict(
            needle=dict(
                x=(-0.25, 0.2),
                y=(-0.05, 0.28),
                # D1 near-full needle rotation
                z_rot=(-7.0 * np.pi / 6.0, np.pi / 6.0),
                reference=self.table_offset,
            ),
            tripod=dict(
                x=(-0.1, 0.15),
                y=(-0.2, -0.1),
                z_rot=(np.pi / 6.0, 5.0 * np.pi / 6.0),
                reference=self.table_offset,
            ),
        )


# ---------------------------------------------------------------------------
# REAL-CELL variant: the D0 task laid out on the MEASURED bench geometry.
# ---------------------------------------------------------------------------
import os as _os

from mimicgen.envs.robosuite.threading import Threading_D0

# THREADING_REAL_CELL_YAML lets a diagnostic run (e.g. a layout sweep) point the
# registered class at a candidate cell file WITHOUT editing the live yaml that
# running jobs re-read on every reset. Unset => byte-identical to before.
REAL_CELL_YAML = _os.environ.get("THREADING_REAL_CELL_YAML") or _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "release/cfgs/sim2real/threading_real_cell.yaml")


class Threading_RealCell(Threading_D0):
    """Threading_D0 with the table height and workspace placement read from the
    sim2real cell YAML -- the SAME file train_rl.py's `sim2real_cell` consumes, so
    the demos, the BC prior and the RL env cannot drift apart.

    Why a registered subclass instead of the PixelRobosuite `sim2real` hook:
    MimicGen (generation) and robomimic (tools/dataset_states_to_obs.py) build
    envs from the env_name stored in the hdf5, never through our wrapper. A
    geometry that only exists inside the wrapper would be silently absent from
    both -- the converted demos would be replayed on the OLD table.

    Only the MATCHED geometry is applied here (table height, workspace offset).
    Randomisation (dr_*) stays in the wrapper at RL time: demos are generated at
    the nominal cell, which is what a BC prior should imitate.

    Same objects, same 37-dim state layout as D0, so the output converts and
    trains exactly like the core D0 dataset.
    """

    def _cell(self):
        from env.sim2real import load_sim2real
        cfg = load_sim2real(REAL_CELL_YAML)
        assert cfg.enabled and cfg.table_above_base is not None, (
            f"{REAL_CELL_YAML}: Threading_RealCell needs enabled=1 and table_above_base")
        return cfg

    def _load_model(self):
        cfg = self._cell()
        # Robot base world z comes from the mount (0.912 for Panda on its default
        # column) and does NOT depend on table_offset -- verified: the base stays
        # at z=0.912 after the lift. So the table can be placed relative to it
        # before the arena is built.
        base_z = 0.912
        self.table_offset = [0.0, 0.0, base_z + cfg.work_surface_above_base()]
        import numpy as np
        self.table_offset = np.array(self.table_offset)
        self._cell_offset = tuple(float(v) for v in cfg.workspace_offset)
        self._assumed_base_z = base_z
        # Measured arena (real2sim, docs/real2sim.md §2e): collision table size
        # and base xy. The size must be set BEFORE super() builds the arena; the
        # base xy AFTER, because super() derives it from table_full_size[0].
        if cfg.table_full_size is not None:
            assert cfg.base_xy is not None, "table_full_size needs base_xy"
            self.table_full_size = tuple(float(v) for v in cfg.table_full_size)
        # Everything the wrapper's sim2real hook would install is already baked
        # in here; mark it so a wrapper built on this env does not stack a
        # second workspace offset / arena patch on top.
        self._s2r_offset_applied = True
        self._s2r_arena_applied = True
        if cfg.init_qpos is not None:
            # Via robot_configs, NOT robots[0].init_qpos: super()._load_model()
            # re-instantiates the Robot from robot_configs on every hard reset.
            # robosuite adds the joint init noise around this value.
            self.robot_configs[0]["initial_qpos"] = np.array(cfg.init_qpos, dtype=np.float64)
        super()._load_model()
        if cfg.base_xy is not None:
            bx, by = (float(v) for v in cfg.base_xy)
            self.robots[0].robot_model.set_base_xpos((bx, by, 0.0))

    def _reset_internal(self):
        super()._reset_internal()
        # Guard the base-height assumption in _load_model against the BUILT sim:
        # if the mount ever changes, the table would sit at the wrong height
        # relative to the robot with no other symptom.
        self.sim.forward()
        got = float(self.sim.data.body_xpos[self.sim.model.body_name2id("robot0_base")][2])
        assert abs(got - self._assumed_base_z) < 1e-6, (
            f"robot base z is {got}, Threading_RealCell assumes {self._assumed_base_z}")

    def _get_initial_placement_bounds(self):
        b = super()._get_initial_placement_bounds()
        # the cell's explicit layout wins over the rigid workspace offset
        from env.sim2real import cell_layout_bounds
        cfg = self._cell()
        # table centre in BASE frame = -base_xy (the cell states the base wrt the table)
        tc = -np.asarray(cfg.base_xy if cfg.base_xy is not None else (-0.489, -0.034))
        spec = cell_layout_bounds(cfg, tc)
        if spec is not None:
            for name, s_ in spec.items():
                if name in b:
                    b[name] = dict(b[name])
                    for k in ("x", "y", "z_rot"):
                        if s_[k] is not None:
                            b[name][k] = s_[k]
            return b
        dx, dy = getattr(self, "_cell_offset", (0.0, 0.0))
        for k in b:
            b[k] = dict(b[k])
            b[k]["x"] = (b[k]["x"][0] + dx, b[k]["x"][1] + dx)
            b[k]["y"] = (b[k]["y"][0] + dy, b[k]["y"][1] + dy)
        return b


# ---------------------------------------------------------------------------
# ROBOTIQ real cell (2026-10-11; cell = the 2026-10-14 calibration since 10-15) + the Robotiq 2F-85
# the real arm now carries. A separate registered class -- not a knob on
# Threading_RealCell -- because the gripper changes the MODEL: a dataset's
# env_name must say which hand it was made with, and MimicGen / robomimic build
# envs from that name without ever seeing our wrapper, so the gripper, its mount
# and the work-surface height have to live in the class itself.
# ---------------------------------------------------------------------------
REAL_CELL_ROBOTIQ_YAML = _os.environ.get("THREADING_ROBOTIQ_CELL_YAML") or _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "real2sim/zed_capture/calib_20261014/threading_real_cell_20261014.yaml")
ROBOTIQ_DRIVER_JOINTS = ("gripper0_finger_joint", "gripper0_right_outer_knuckle_joint")
# Real Robotiq knuckle (/gripper/joint_states robotiq_85_left_knuckle_joint, rad) -> sim driver
# joints. In-contact anchor 0.440 real <-> 0.499 sim on the 40 mm handle, clipped at the sim
# stall (calib_20261014/README.md "Correction to the Robotiq obs map"; the older 0.83 is wrong).
ROBOTIQ_REAL_TO_SIM = 1.134
ROBOTIQ_SIM_STALL = 0.556


def robotiq_real_to_sim(knuckle_real) -> np.ndarray:
    """One real knuckle angle -> the 2 sim driver-joint values robot0_gripper_qpos carries."""
    v = float(np.clip(ROBOTIQ_REAL_TO_SIM * float(knuckle_real), 0.0, ROBOTIQ_SIM_STALL))
    return np.array([v, v])


def _robotiq_cell():
    from env.sim2real import load_sim2real
    cfg = load_sim2real(REAL_CELL_ROBOTIQ_YAML)
    assert cfg.enabled and cfg.table_above_base is not None, REAL_CELL_ROBOTIQ_YAML
    assert cfg.gripper_types == "Robotiq85Gripper", (
        f"{REAL_CELL_ROBOTIQ_YAML}: Threading_RealCellRobotiq needs gripper_types: Robotiq85Gripper")
    return cfg


class Threading_RealCellRobotiq(Threading_RealCell):
    """Threading_RealCell on the 2026-10-10 cell with the Robotiq 2F-85.

    On top of Threading_RealCell (table height incl. table_top_dz, arena, layout,
    init_qpos -- all read from REAL_CELL_ROBOTIQ_YAML):
      * gripper_types forced to the cell's Robotiq85Gripper;
      * the gripper mount (gripper_mount_yaw_deg / _z: square to the FR3 flange,
        45.03 deg from robot0_right_hand) written into the MJCF, so it is in the
        model xml every dataset stores;
      * robot0_gripper_qpos / _qvel = the two ACTUATED knuckle joints only
        (ROBOTIQ_DRIVER_JOINTS), not all six: the four passive joints are tied to
        them by equality constraints, and 2 dims keep the 37-dim state / 9-dim
        prop contract (PROP_DIM) every trainer slices by. Range 0 (open) .. 0.8
        (closed) rad -- NOT the Panda's +-finger metres; the real side must map
        the Robotiq's position register onto it.
    """

    def __init__(self, *args, gripper_types="default", **kwargs):
        want = _robotiq_cell().gripper_types
        got = gripper_types[0] if isinstance(gripper_types, (list, tuple)) else gripper_types
        assert got in ("default", None, want), (
            f"Threading_RealCellRobotiq is the {want} cell; got gripper_types={gripper_types!r}")
        super().__init__(*args, gripper_types=want, **kwargs)

    def _cell(self):
        return _robotiq_cell()

    def _load_model(self):
        super()._load_model()
        cfg = self._cell()
        root = self.robots[0].robot_model.worldbody.find(
            ".//body[@name='gripper0_robotiq_85_adapter_link']")
        assert root is not None, "Robotiq adapter link not found in the robot model"
        h = np.radians(float(cfg.gripper_mount_yaw_deg)) / 2
        root.set("pos", f"0 0 {float(cfg.gripper_mount_z)!r}")
        root.set("quat", f"{np.cos(h)!r} 0 0 {np.sin(h)!r}")

    def _reset_internal(self):
        super()._reset_internal()
        # Start OPEN, like the real arm's home. robosuite resets the Robotiq to
        # current_action 0 = ctrl 0.4 rad, i.e. HALF-closed (drivers read 0.28 rad
        # after one step), which the real gripper never is at the start.
        rob = self.robots[0]
        rob.gripper.current_action = -np.ones(rob.gripper.dof)
        self.sim.data.qpos[rob._ref_gripper_joint_pos_indexes] = 0.0
        self.sim.data.qvel[rob._ref_gripper_joint_vel_indexes] = 0.0
        self.sim.forward()

    def _setup_observables(self):
        from robosuite.utils.observables import Observable, sensor
        obs = super()._setup_observables()
        modality = "robot0_proprio"

        def _addr(kind, name):
            m = self.sim.model
            return m.get_joint_qpos_addr(name) if kind == "q" else m.get_joint_qvel_addr(name)

        @sensor(modality=modality)
        def gripper_qpos(obs_cache):
            return np.array([self.sim.data.qpos[_addr("q", n)] for n in ROBOTIQ_DRIVER_JOINTS])

        @sensor(modality=modality)
        def gripper_qvel(obs_cache):
            return np.array([self.sim.data.qvel[_addr("v", n)] for n in ROBOTIQ_DRIVER_JOINTS])

        for name, s in (("robot0_gripper_qpos", gripper_qpos), ("robot0_gripper_qvel", gripper_qvel)):
            assert name in obs, name
            obs[name] = Observable(name=name, sensor=s, sampling_rate=self.control_freq)
        return obs

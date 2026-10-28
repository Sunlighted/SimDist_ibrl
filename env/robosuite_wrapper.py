from typing import Optional
import types
from collections import defaultdict, deque

import torch
import robosuite
from robosuite import load_controller_config
import numpy as np
from common_utils import ibrl_utils as utils
import common_utils

# Importing mimicgen registers its environments (Threading_D0, Stack_D0, ...)
# with robosuite's global env registry on import. Wrapped so the original
# robosuite tasks still work if mimicgen isn't installed.
try:
    import mimicgen  # noqa: F401
except Exception:  # pragma: no cover - mimicgen optional
    pass

# Registers ToolHangInsert / ToolHangInsertHold (stage-1-only ToolHang) with the
# same robosuite registry. Local module, no optional dependency.
import env.tool_hang_variants  # noqa: F401,E402

# Registers PegInsertion / PegInsertion_D0 (real2sim square peg-in-hole,
# docs/real2sim.md Phase B). Depends on mimicgen (SingleArmEnv_MG base).
import env.peg_insertion  # noqa: F401,E402

from env.sim2real import (
    Sim2RealConfig,
    apply_cell_arena,
    apply_physics_dr,
    apply_cell_layout,
    apply_init_qpos,
    apply_line_lights,
    apply_prop_frame,
    apply_sim2real,
    apply_workspace_offset,
    snapshot_visual_nominal,
    table_offset_for,
)
from env.domain_shift import (
    DomainShiftConfig,
    InitDistConfig,
    DRSampler,
    resolve_threading_bindings,
    resolve_bindings,
    snapshot_nominal,
    restore_nominal,
    apply_contact_shift,
    apply_hole_offset,
    apply_ring_shift,
    apply_needle_shift,
    apply_tripod_trap_shift,
    apply_square_peg_shift,
    apply_coffee_shift,
    apply_three_piece_shift,
    mask_hole_offset_from_obs,
    apply_init_distribution,
)
from env.strict_success import StrictSuccessConfig, SuccessValidator


# all avail views:
# 'frontview', 'birdview', --> too far for this task
# 'agentview', 'robot0_robotview', --> same
# 'sideview', 'robot0_eye_in_hand'
GOOD_CAMERAS = {
    "Lift": ["agentview", "sideview", "robot0_eye_in_hand"],
    "PickPlaceCan": ["agentview", "robot0_eye_in_hand"],
    "NutAssemblySquare": ["agentview", "robot0_eye_in_hand"],
}
DEFAULT_CAMERA = "agentview"


DEFAULT_STATE_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
STATE_KEYS = {
    "Lift": DEFAULT_STATE_KEYS,
    "PickPlaceCan": DEFAULT_STATE_KEYS,
    "NutAssemblySquare": DEFAULT_STATE_KEYS,
    "TwoArmTransport": [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot1_eef_pos",
        "robot1_eef_quat",
        "robot1_gripper_qpos",
        "object",
    ],
    "ToolHang": [
        "object",  # (389, 44)
        "robot0_eef_pos",  # (389, 3)
        "robot0_eef_quat",  # (389, 4)
        "robot0_gripper_qpos",  # (389, 2)
        # "robot0_gripper_qvel",  # (389, 2)
        # "robot0_eef_vel_ang",  # (389, 3)
        # "robot0_eef_vel_lin",  # (389, 3)
        # "robot0_joint_pos", # (389, 7)
        # "robot0_joint_pos_cos",  # (389, 7)
        # "robot0_joint_pos_sin",  # (389, 7)
        # "robot0_joint_vel",  # (389, 7)
    ],
}
# ToolHangInsert / ToolHangInsertHold are `ToolHang` with a different
# `_check_success` only -- identical objects, obs keys and state layout.
for _t in ["ToolHangInsert", "ToolHangInsertHold"]:
    STATE_KEYS[_t] = STATE_KEYS["ToolHang"]

STATE_SHAPE = {
    "Lift": (19,),
    "PickPlaceCan": (23,),
    "NutAssemblySquare": (23,),
    "TwoArmTransport": (59,),
    "ToolHang": (53,),
    "ToolHangInsert": (53,),
    "ToolHangInsertHold": (53,),
}

# MimicGen D0 tasks all use the default state keys (eef_pos[3] + eef_quat[4] +
# gripper_qpos[2] + object-state[N]). Their state dim depends on the task's
# object-state size; Threading_D0 is precomputed (9 prop + 28 object = 37) and
# the rest are inferred at construction via _infer_state_dim() so we don't have
# to hardcode every variant. (robosuite_task_zoo tasks HammerCleanup_D0 /
# Kitchen_D0 are intentionally not installed -- see CLAUDE.md.)
MIMICGEN_D0_TASKS = [
    "Threading_D0",
    "Stack_D0",
    "StackThree_D0",
    "Square_D0",
    "Coffee_D0",
    "CoffeePreparation_D0",
    "ThreePieceAssembly_D0",
    "PickPlace_D0",
    "NutAssembly_D0",
    "MugCleanup_D0",
]
for _t in MIMICGEN_D0_TASKS:
    STATE_KEYS[_t] = DEFAULT_STATE_KEYS
# Threading-family variants (D1, Enlarged, RealCell) share Threading_D0's spec.
for _t in ["Threading_D1", "Threading_D2", "Threading_Enlarged", "Threading_RealCell",
           "Threading_RealCellRobotiq"]:
    STATE_KEYS[_t] = DEFAULT_STATE_KEYS
# real2sim peg insertion (env/peg_insertion.py): 9 prop + 28 object (peg/board
# pos+quat, absolute and eef-relative) = 37, same layout as Threading_D0.
STATE_KEYS["PegInsertion_D0"] = DEFAULT_STATE_KEYS
STATE_SHAPE["PegInsertion_D0"] = (37,)
GOOD_CAMERAS["PegInsertion_D0"] = ["agentview", "robot0_eye_in_hand"]
STATE_SHAPE["Threading_D0"] = (37,)
STATE_SHAPE["Threading_D1"] = (37,)
STATE_SHAPE["Threading_D2"] = (37,)
STATE_SHAPE["Threading_Enlarged"] = (37,)
STATE_SHAPE["Threading_RealCell"] = (37,)
# Robotiq cell: gripper_qpos is cut to the 2 driver joints in the class => still 37
STATE_SHAPE["Threading_RealCellRobotiq"] = (37,)
# exec_obs: the execution layer's hidden reference-arm state (SimRefController.hidden_state, 20) + the last action (7),
# appended as the LAST state key so state[:37] is unchanged (a prior trained without it reads that prefix)
EXEC_OBS_KEY = "exec_ref"
EXEC_OBS_DIM = 27
EXEC_OBS_SCALE = np.array([10.0] * 13 + [1.0] * 7)     # m / rad deltas of ~1e-2 -> ~1e-1; v_ref (rad/s) as is
# Register the custom Threading variants with robosuite in EVERY process that
# builds this wrapper. A dataset converted from Threading_RealCell demos stores
# that env_name, and train_bc's in-training eval (including the mp eval worker
# processes) rebuilds the env from it -- without this import those processes
# would fail to find the class.
import env.threading_variants  # noqa: E402,F401
# Same for the Square real-cell variant. It is NOT just a reset distribution:
# Square's pegs are arena geometry at fixed world coordinates, so only this
# class puts them on the measured table (see env/square_variants.py).
for _t in ["Square_RealCell", "Square_RealCell_RotW1"]:
    STATE_KEYS[_t] = DEFAULT_STATE_KEYS
    GOOD_CAMERAS[_t] = ["agentview", "robot0_eye_in_hand"]
import env.square_variants  # noqa: E402,F401

PROP_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
PROP_DIM = 9


def prop_from_state(states, exec_obs: bool = False):
    """What every trainer feeds the encoder as `prop`, sliced from the stored state: state[..., :PROP_DIM], plus
    (exec_obs) the execution-layer block, which is the LAST EXEC_OBS_DIM dims of the state. Same order as the env's
    rl_obs["prop"]. Works on numpy arrays and torch tensors."""
    if not exec_obs:
        return states[..., :PROP_DIM]
    cat = np.concatenate if isinstance(states, np.ndarray) else torch.cat
    return cat([states[..., :PROP_DIM], states[..., -EXEC_OBS_DIM:]], -1)


class PixelRobosuite:
    def __init__(
        self,
        env_name,
        robots,
        episode_length,
        *,
        reward_shaping=False,
        image_size=224,
        rl_image_size=96,
        device="cuda",
        camera_names=[DEFAULT_CAMERA],
        rl_cameras=["agentview"],
        env_reward_scale=1.0,
        end_on_success=True,
        use_state=False,
        obs_stack=1,
        state_stack=1,
        prop_stack=1,
        cond_action=0,
        flip_image=True,  # only false if using with eval_with_init_state
        ctrl_delta=True,
        record_sim_state: bool = False,
        # PHOTOREAL obs: when set, camera images are replaced by 3DGS composites
        # (splat scene + splat board; MuJoCo arm + peg). See real2sim/splat_obs.py
        # for what is splatted and why. Default None = byte-identical MuJoCo path.
        splat_obs: Optional[dict] = None,
        # extra kwargs forwarded verbatim to robosuite.make (e.g. the peg env's
        # board_pos_noise / board_yaw_noise). Empty = unchanged.
        env_kwargs: Optional[dict] = None,
        # ---- domain-shift / strict-success / init-perturb extensions ----
        domain_shift: Optional[DomainShiftConfig] = None,
        # SIM-TO-REAL cell config (env/sim2real.py): matched table height/colour,
        # measured camera extrinsics+intrinsics, and the VISUAL randomisation
        # around them. Orthogonal to `domain_shift`, which randomises PHYSICS.
        # Default None => Sim2RealConfig(enabled=0) => this is a no-op.
        sim2real: Optional[Sim2RealConfig] = None,
        strict_success: Optional[StrictSuccessConfig] = None,
        hand_init_perturb: Optional[tuple] = None,
        hand_init_perturb_joint_sigma: Optional[float] = None,
        init_distribution: Optional[InitDistConfig] = None,
        state_only_no_render: bool = False,
        stage_reward: float = 0.0,
        # PIXEL arm: also emit rl_obs["state"] when use_state=False. The pixel
        # world model consumes images, but the frozen PRIVILEGED STATE teacher
        # reads the 37-dim state to produce V_src labels, and the sequence buffer
        # stores it alongside the frames. Costs 37 floats/step.
        extra_state: bool = False,
        # the execution layer's hidden state in obs["state"] (EXEC_OBS_*); needs a cell with exec_layer
        exec_obs: bool = False,
    ):
        assert isinstance(camera_names, list)
        self.camera_names = camera_names
        self.ctrl_config = load_controller_config(default_controller="OSC_POSE")
        self.ctrl_config["control_delta"] = ctrl_delta
        self.record_sim_state = record_sim_state

        # When True (state-based eval only), skip ALL camera rendering: build the
        # robosuite env with no offscreen GL context + no camera obs, and skip the
        # camera loop in _extract_images. The policy only consumes rl_obs["state"]
        # so the rendered image is pure waste; turning it off removes the per-step
        # 224x224 render (the dominant single-step cost) AND the EGL context that
        # makes multi-process eval deadlock. Default False = byte-for-byte the old
        # rendering path. Incompatible with video recording (no high_res_images).
        self.state_only_no_render = bool(state_only_no_render)
        self.extra_state = bool(extra_state) and not use_state
        if self.state_only_no_render:
            assert use_state, "state_only_no_render requires use_state=True"

        # ---- resolve the two mutually-exclusive hand-init regimes ----
        # hand_init_perturb (3-tuple): EE-space box; we disable robosuite's
        #   joint noise and IK to a sampled EE offset at each reset.
        # hand_init_perturb_joint_sigma (float): rescale robosuite's built-in
        #   joint-space gaussian (sigma=0.02 is the demo-collection default).
        # both None: robosuite's default (sigma=0.02 gaussian on joint qpos).
        assert not (
            hand_init_perturb is not None and hand_init_perturb_joint_sigma is not None
        ), "hand_init_perturb and hand_init_perturb_joint_sigma are mutually exclusive"
        self.hand_init_perturb = (
            tuple(float(x) for x in hand_init_perturb)
            if hand_init_perturb is not None
            else None
        )
        self.hand_init_perturb_joint_sigma = hand_init_perturb_joint_sigma
        # Only IK-perturb when an EE box with a non-zero half-extent is given.
        # An all-zero tuple means "fully deterministic init" (noise off, no IK).
        self._ik_perturb_active = self.hand_init_perturb is not None and any(
            abs(x) > 0 for x in self.hand_init_perturb
        )
        if self.hand_init_perturb is not None:
            # disable robosuite's joint noise; we set the pose ourselves (IK or
            # deterministic baseline).
            init_noise = {"magnitude": 0.0, "type": "gaussian"}
        elif self.hand_init_perturb_joint_sigma is not None:
            init_noise = {
                "magnitude": float(self.hand_init_perturb_joint_sigma),
                "type": "gaussian",
            }
        else:
            init_noise = "default"

        _render = not self.state_only_no_render
        # set before robosuite.make: anything that runs during construction and
        # reaches _extract_images must find these attributes present.
        self._splat_cfg = splat_obs
        self._splat = None

        env_kwargs = dict(env_kwargs or {})
        if sim2real is not None and sim2real.enabled and sim2real.gripper_types:
            # construction-time cell field (env/sim2real.py): the real cell's gripper
            assert env_kwargs.get("gripper_types", sim2real.gripper_types) == sim2real.gripper_types, \
                "env_kwargs.gripper_types disagrees with the sim2real cell"
            env_kwargs["gripper_types"] = sim2real.gripper_types
        if sim2real is not None and sim2real.enabled and sim2real.control_freq is not None:
            # construction-time cell field: the real cell's control rate (env/sim2real.py)
            assert float(env_kwargs.get("control_freq", sim2real.control_freq)) == float(sim2real.control_freq), \
                "env_kwargs.control_freq disagrees with the sim2real cell"
            env_kwargs["control_freq"] = sim2real.control_freq
        if env_name == "Threading_RealCellRobotiq" and sim2real is not None and sim2real.enabled:
            import env.threading_variants as _tv     # the class's own cell fields (exec_layer, table, ...) = this cell
            _tv._ROBOTIQ_CELL_OVERRIDE = sim2real
        self.env = robosuite.make(
            env_name=env_name,
            robots=robots,
            controller_configs=self.ctrl_config,
            has_offscreen_renderer=_render,
            use_camera_obs=_render,
            reward_shaping=reward_shaping,
            camera_names=self.camera_names,
            camera_heights=image_size,
            camera_widths=image_size,
            horizon=episode_length,
            initialization_noise=init_noise,
            **env_kwargs,
        )
        self.env_name = env_name
        self.rl_cameras = rl_cameras if isinstance(rl_cameras, list) else [rl_cameras]
        self.image_size = image_size
        self.rl_image_size = rl_image_size or image_size
        self.env_reward_scale = env_reward_scale
        self.end_on_success = end_on_success
        self.use_state = use_state
        self.state_keys = STATE_KEYS[env_name]
        self.exec_obs = bool(exec_obs)
        self.prop_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        if self.exec_obs:
            # pixel runs too (extra_state): it is proprioception on the robot as well (policy_server's ReferenceArm),
            # so it goes into prop, after the 9 dims (= prop_from_state on the stored state)
            assert (use_state or self.extra_state) and state_stack == 1 and prop_stack == 1, \
                "exec_obs: a state vector, no state / prop stacking"
            assert getattr(self.env, "exec_layer", None) is not None, "exec_obs needs a cell with an execution layer"
            self.state_keys = list(self.state_keys) + [EXEC_OBS_KEY]
            self.prop_keys = self.prop_keys + [EXEC_OBS_KEY]
        self._last_action = np.zeros(len(self.env.action_spec[0]))
        self.flip_image = flip_image

        self.resize_transform = None
        if self.rl_image_size != self.image_size:
            self.resize_transform = utils.get_rescale_transform(self.rl_image_size)

        self.action_dim: int = len(self.env.action_spec[0])
        self._observation_shape: tuple[int, ...] = (3 * obs_stack, rl_image_size, rl_image_size)
        if env_name in STATE_SHAPE:
            state_base = STATE_SHAPE[env_name][0]
        else:
            state_base = self._infer_state_dim()
        if self.exec_obs and env_name in STATE_SHAPE:
            state_base += EXEC_OBS_DIM
        self._state_shape: tuple[int] = (state_base * state_stack,)
        self.prop_shape: tuple[int] = ((PROP_DIM + (EXEC_OBS_DIM if self.exec_obs else 0)) * prop_stack,)
        self.device = device

        self.time_step = 0
        # Optional ONE-SHOT bonus for finishing an intermediate subtask, for
        # tasks whose env exposes _get_partial_task_metrics() (currently only
        # ThreePieceAssembly). 0.0 = off => byte-for-byte upstream behavior.
        #
        # WHY: TPA's reward fires only when BOTH pieces are assembled, so
        # "piece_1 placed" carries no return signal at all. Forensics on the
        # plain-IBRL run (300k steps, 30/30 eval points at 0.00): piece_1 gets
        # placed in 4/12 rollouts but piece_2 is never touched (max disp 0.002 m,
        # z pinned at the 0.880 table height), and cumulative successes freeze
        # at 207 from 230k on while Q drifts back up 0.25 -> 0.36.
        #
        # ONE-SHOT is load-bearing: a per-step bonus would let the agent park in
        # stage 1 and farm it. success/terminal are untouched, so eval stays
        # comparable with every earlier run.
        self.stage_reward = float(stage_reward)
        self._stage1_fired = False
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.terminal = True

        self.obs_stack = obs_stack
        self.state_stack = state_stack
        self.prop_stack = prop_stack
        self.cond_action = cond_action
        self.past_obses = defaultdict(list)
        self.past_actions = deque(maxlen=self.cond_action)

        # ---- optional init-distribution override (Threading D1/Enlarged,
        # Square nut z_rot window) ----
        # Task-dispatched; no-op when disabled. Persists across hard resets
        # (Threading re-reads the patched bounds; Square keeps the mutated sampler).
        self.init_distribution = init_distribution
        if self.init_distribution is not None and getattr(
            self.init_distribution, "enabled", False
        ):
            apply_init_distribution(self.env, self.init_distribution)

        # ---- sim2real cell ----
        self.sim2real = sim2real if sim2real is not None else Sim2RealConfig()
        self.last_bg_index: Optional[int] = None
        self._s2r_active = bool(self.sim2real.enabled)
        self._inhand_pending, self.last_inhand_dr = False, None
        self._grip_queue, self.last_gripper_timing = None, None
        self._s2r_nominal: Optional[dict] = None
        if self._s2r_active:
            # The table height must be installed BEFORE the next reset:
            # hard_reset=True means `_load_model` rebuilds the arena AND the
            # placement initializer from `env.table_offset` every episode, which
            # is precisely why the objects follow the table for free. Writing the
            # geom after reset would move the table out from under them.
            bid = self.env.sim.model.body_name2id("robot0_base")
            base_z = float(self.env.sim.data.body_xpos[bid][2])
            off = table_offset_for(self.sim2real, base_z)
            # Placement bounds are re-read on every hard reset, so this is a
            # one-shot patch like apply_init_distribution's. It must land before
            # the first reset AND after any InitDistConfig override, so that the
            # offset composes on top of whatever bounds are in force.
            if not apply_cell_layout(self.env, self.sim2real):
                apply_workspace_offset(self.env, self.sim2real)
            apply_cell_arena(self.env, self.sim2real)
            apply_line_lights(self.env, self.sim2real)
            apply_init_qpos(self.env, self.sim2real)
            # needle CoM: part of the MODEL (an <inertial> the task class adds on every hard reset), so it too must be
            # in place before the first reset
            self.env._needle_com_y = getattr(self.sim2real, "needle_com_y", None)
            if off is not None:
                assert hasattr(self.env, "table_offset"), (
                    f"{env_name} has no `table_offset`; sim2real.table_above_base "
                    "only works on the MimicGen table envs")
                print(f"[sim2real] table_offset {self.env.table_offset} -> {off} "
                      f"(top {self.sim2real.work_surface_above_base():+.4f} m vs base origin, "
                      f"table_top_dz {self.sim2real.table_top_dz:+.4f})")
                self.env.table_offset = off

        # ---- domain shift + strict success ----
        self.domain_shift = domain_shift if domain_shift is not None else DomainShiftConfig()
        self.strict_success = (
            strict_success if strict_success is not None else StrictSuccessConfig()
        )
        self._shift_active = self.domain_shift.mode != "none"
        self._strict_active = bool(self.strict_success.enabled)
        # Bindings/nominal are needed by the shift functions AND the strict
        # validator. Resolved (Threading-only) only when one of them is active,
        # so non-Threading tasks with no shift behave exactly like upstream.
        self._needs_bindings = self._shift_active or self._strict_active
        self._bindings: Optional[dict] = None
        self._nominal: Optional[dict] = None
        self._dr_sampler: Optional[DRSampler] = None
        self._success_validator: Optional[SuccessValidator] = None
        if self._needs_bindings:
            self._bindings = resolve_bindings(self.env)
            self._nominal = snapshot_nominal(self.env)
            if self.domain_shift.mode == "dr":
                self._dr_sampler = DRSampler(self.domain_shift)
        if self._strict_active:
            self._success_validator = SuccessValidator(self.strict_success)

        # ---- IK-perturb bindings (only when the EE-box regime is active) ----
        self._ik_baseline_eef: Optional[np.ndarray] = None
        if self._ik_perturb_active:
            gp = self.env.robots[0].gripper.naming_prefix
            self._grip_site = gp + "grip_site"
            self._arm_qpos_idx = np.asarray(self.env.robots[0]._ref_joint_pos_indexes)
            self._arm_dof_idx = np.asarray(self.env.robots[0]._ref_joint_vel_indexes)
            self._ik_max_iter = 150
            self._ik_damping = 0.05

    def _infer_state_dim(self) -> int:
        """Sum the per-key sizes of self.state_keys from a fresh observation.

        Used for MimicGen tasks not in the precomputed STATE_SHAPE table.
        """
        obs = self.env.observation_spec()
        total = 0
        for key in self.state_keys:
            env_key = "object-state" if key == "object" else key
            total += int(np.asarray(obs[env_key]).size)
        return total

    @property
    def state_shape(self):
        """The 37-dim state layout, regardless of use_state. Needed so a pixel run
        can build a PRIVILEGED STATE teacher (train_rl.load_frozen_agent)."""
        return self._state_shape

    @property
    def observation_shape(self):
        if self.use_state:
            return self._state_shape
        else:
            return self._observation_shape

    def _apply_domain_shift(self) -> None:
        """Re-apply the domain shift after an env.reset().

        robosuite uses hard_reset=True, so the sim is rebuilt from MJCF on every
        reset and any prior mutation is wiped. We re-resolve bindings against the
        fresh sim, restore nominal (so scales compose from nominal, not the last
        shifted values), apply the fixed/DR shift, and forward() so data catches
        up. No-op when mode == "none".
        """
        if self.domain_shift.mode == "none":
            return
        # hard_reset rebuilt the sim -> re-resolve ids.
        self._bindings = resolve_bindings(self.env)
        restore_nominal(self.env, self._nominal)
        cfg = self.domain_shift
        if self.domain_shift.mode == "dr":
            self._dr_sampler.sample_and_apply(self.env, self._bindings)
        else:  # "fixed"
            # Common contact shift (nut/needle mass+friction, gripper kp) —
            # consumes object_body_ids/object_geom_ids/gripper_act_ids, present
            # in both task families.
            apply_contact_shift(
                self.env,
                self._bindings,
                friction_scale=cfg.friction_scale,
                mass_scale=cfg.mass_scale,
                solref_scale=cfg.solref_scale,
                solimp_scale=cfg.solimp_scale,
                com_offset=cfg.com_offset,
                gripper_kp_scale=cfg.gripper_kp_scale,
            )
            # Task-specific geometry shift, branched on binding contents.
            if "ring_geom_ids" in self._bindings:  # Threading
                apply_hole_offset(self.env, self._bindings, cfg.hole_offset)
                apply_ring_shift(
                    self.env,
                    self._bindings,
                    ring_size_scale=cfg.ring_size_scale,
                    ring_friction_scale=cfg.ring_friction_scale,
                )
                apply_needle_shift(
                    self.env,
                    self._bindings,
                    needle_size_scale=cfg.needle_size_scale,
                )
                apply_tripod_trap_shift(
                    self.env,
                    self._bindings,
                    tripod_mass_scale=cfg.tripod_mass_scale,
                    tripod_com_height_scale=cfg.tripod_com_height_scale,
                )
            elif "peg_geom_ids" in self._bindings:  # Square
                apply_square_peg_shift(
                    self.env,
                    self._bindings,
                    peg_size_scale=cfg.peg_size_scale,
                    peg_friction_scale=cfg.peg_friction_scale,
                    nut_size_scale=cfg.nut_size_scale,
                )
            elif "pod_holder_geom_ids" in self._bindings:  # Coffee
                apply_coffee_shift(
                    self.env,
                    self._bindings,
                    pod_holder_size_scale=cfg.pod_holder_size_scale,
                    pod_holder_friction_scale=cfg.pod_holder_friction_scale,
                    lid_damping_scale=cfg.lid_damping_scale,
                    lid_frictionloss=cfg.lid_frictionloss,
                )
            elif "base_wall_geom_ids" in self._bindings:  # ThreePieceAssembly
                apply_three_piece_shift(
                    self.env,
                    self._bindings,
                    base_hole_size_scale=cfg.base_hole_size_scale,
                    base_friction_scale=cfg.base_friction_scale,
                )
        self.env.sim.forward()

    def _ik_to_perturbed_pos(self) -> None:
        """Damped-least-squares 3-DOF positional IK to a sampled EE offset.

        Snapshots the nominal EE position lazily on the first reset, then at
        every reset samples an offset from uniform([-rx,rx]x[-ry,ry]x[-rz,rz])
        and drives the 7 arm joints there. Orientation drifts as a null-space
        side effect (small for <= ~2 cm offsets). Gripper/object placements are
        untouched.
        """
        env = self.env
        sim = env.sim
        site = self._grip_site
        sim.forward()
        if self._ik_baseline_eef is None:
            self._ik_baseline_eef = sim.data.get_site_xpos(site).copy()
        rx, ry, rz = self.hand_init_perturb
        offset = np.array(
            [
                np.random.uniform(-rx, rx),
                np.random.uniform(-ry, ry),
                np.random.uniform(-rz, rz),
            ]
        )
        target = self._ik_baseline_eef + offset
        qpos_idx = self._arm_qpos_idx
        dof_idx = self._arm_dof_idx
        for _ in range(self._ik_max_iter):
            sim.forward()
            cur = sim.data.get_site_xpos(site)
            err = target - cur
            if np.linalg.norm(err) < 1e-4:
                break
            Jp = sim.data.get_site_jacp(site)[:, dof_idx]  # (3, 7)
            JJt = Jp @ Jp.T + (self._ik_damping ** 2) * np.eye(3)
            dq = Jp.T @ np.linalg.solve(JJt, err)
            sim.data.qpos[qpos_idx] += dq
        sim.forward()
        # Re-sync the controller so it holds the new pose instead of driving the
        # arm back toward the reset goal on the first action.
        try:
            controller = env.robots[0].controller
            controller.update_initial_joints(sim.data.qpos[qpos_idx].copy())
            controller.reset_goal()
        except Exception:
            pass

    def _extract_images(self, obs):
        # assert self.frame_stack == 1, "frame stack not supported"

        # Blind the policy to the hole offset (sgft "obs-blind" design). No-op
        # unless mode == "fixed", masking is enabled, and the offset is non-zero.
        if (
            self.domain_shift.mode == "fixed"
            and self.domain_shift.mask_hole_offset_in_obs
            and self._bindings is not None
            and "obj_state_slices" in self._bindings  # Threading-only; Square has none
        ):
            mask_hole_offset_from_obs(obs, self._bindings, self.domain_shift.hole_offset)

        # PHOTOREAL: overwrite the raw MuJoCo frames before anything downstream
        # touches them, so resize / stacking / storage are all unchanged. Built
        # lazily: the renderer needs a CUDA context and the env's compiled model,
        # neither of which exists at __init__ time in a forked worker.
        if self._splat_cfg is not None:
            if self._splat is None:
                from real2sim.splat_obs import SplatCompositor
                self._splat = SplatCompositor(self.env.sim.model, self.env.sim.data,
                                              **self._splat_cfg)
            for cam in self.rl_cameras:
                key = f"{cam}_image"
                if key in obs:
                    h, w = obs[key].shape[:2]
                    obs[key] = self._splat.composite(self.env.sim, cam, h, w)

        # prop_frame (env/sim2real.py): before BOTH state and prop are assembled,
        # since every trainer slices prop = state[:PROP_DIM]. No-op for "world".
        if self._s2r_active:
            apply_prop_frame(self.env.sim, self.sim2real, obs)
        if self.exec_obs:
            obs[EXEC_OBS_KEY] = np.concatenate([self.env.exec_layer.hidden_state() * EXEC_OBS_SCALE,
                                                self._last_action])

        high_res_images = {}
        rl_obs = {}

        if self.use_state or self.extra_state:
            states = []
            for key in self.state_keys:
                if key == "object":
                    key = "object-state"
                states.append(obs[key])
            state = torch.from_numpy(np.concatenate(states).astype(np.float32))
            # first append, then concat
            self.past_obses["state"].append(state)
            rl_obs["state"] = utils.concat_obs(
                len(self.past_obses["state"]) - 1, self.past_obses["state"], self.state_stack
            ).to(self.device)

        props = []
        for key in self.prop_keys:
            props.append(obs[key])
        prop = torch.from_numpy(np.concatenate(props).astype(np.float32))
        # first append, then concat
        self.past_obses["prop"].append(prop)
        rl_obs["prop"] = utils.concat_obs(
            len(self.past_obses["prop"]) - 1, self.past_obses["prop"], self.prop_stack
        ).to(self.device)

        camera_names = [] if self.state_only_no_render else self.camera_names
        for camera_name in camera_names:
            image_key = f"{camera_name}_image"
            image_obs = obs[image_key]
            if self.flip_image:
                image_obs = image_obs[::-1]
            image_obs = torch.from_numpy(image_obs.copy()).permute([2, 0, 1])

            # keep the high-res version for rendering
            high_res_images[camera_name] = image_obs
            if camera_name not in self.rl_cameras:
                continue

            rl_image_obs = image_obs
            if self.resize_transform is not None:
                # set the device here because transform is 5x faster on GPU
                rl_image_obs = self.resize_transform(rl_image_obs.to(self.device))
            # first append, then concat
            self.past_obses[camera_name].append(rl_image_obs)
            rl_obs[camera_name] = utils.concat_obs(
                len(self.past_obses[camera_name]) - 1,
                self.past_obses[camera_name],
                self.obs_stack,
            )

        if self.record_sim_state:
            sim_state = self.env.sim.get_state().flatten()
            rl_obs["sim_state"] = torch.from_numpy(sim_state)
            for key in DEFAULT_STATE_KEYS:
                env_key = "object-state" if key == "object" else key
                rl_obs[key] = torch.from_numpy(obs[env_key])
            if self.exec_obs:
                rl_obs[EXEC_OBS_KEY] = torch.from_numpy(obs[EXEC_OBS_KEY].astype(np.float32))

        return rl_obs, high_res_images

    # ------------------------------------------------------------ initial-state capture / replay
    def _object_free_joints(self):
        m = self.env.sim.model
        out = []
        for j in range(m.njnt):
            if m.jnt_type[j] == 0:                       # free joint
                name = m.joint_id2name(j) or ""
                if not name.startswith("robot"):
                    out.append(name)
        return out

    def get_init_state(self) -> dict:
        """This episode's initial configuration, ROBOT-BASE relative (base DR changes the base every episode):
        each free object's position + rotation in the base frame, and the arm joints. For failure replay
        (train_rl.py fail_replay_*)."""
        m, d = self.env.sim.model, self.env.sim.data
        bid = m.body_name2id(self.env.robots[0].robot_model.root_body)
        b, Rb = np.array(d.body_xpos[bid]), np.array(d.body_xmat[bid]).reshape(3, 3)
        objs = {}
        for jn in self._object_free_joints():
            a = m.get_joint_qpos_addr(jn)[0]
            p = np.array(d.qpos[a:a + 3])
            from robosuite.utils.transform_utils import quat2mat
            w, x, y, z = d.qpos[a + 3:a + 7]
            R = quat2mat(np.array([x, y, z, w]))
            objs[jn] = (Rb.T @ (p - b)).tolist(), (Rb.T @ R).tolist()
        q = np.array(d.qpos[self.env.robots[0]._ref_joint_pos_indexes]).tolist()
        return {"objects": objs, "q": q}

    def ring_centre(self) -> Optional[np.ndarray]:
        """Threading: world position of the ring centre (mean of the ring geoms); None for other tasks. Used by the
        mid-episode reset (train_rl.py mid_reset_*) to find when the needle first pushes the ring."""
        ids = (self._bindings or {}).get("ring_geom_ids") if self._needs_bindings else None
        if not ids:
            return None
        return np.mean(self.env.sim.data.geom_xpos[ids], axis=0)

    def teleport(self, spec: dict):
        """Write a Threading OmniReset state (env/omni_reset.py) into the freshly reset sim: needle pose (world), arm
        joints, the 6 Robotiq joints; all of their velocities zero; then re-latch the controller (= the exec layer's
        reference arm) on the new joints, exactly as _apply_init_state does. Returns the fresh obs."""
        m, d = self.env.sim.model, self.env.sim.data
        rob = self.env.robots[0]
        nb = m.body_name2id("needle_obj_root")
        j = m.body_jntadr[nb]
        a, v = m.jnt_qposadr[j], m.jnt_dofadr[j]
        from robosuite.utils.transform_utils import mat2quat
        x, y, z, w = mat2quat(np.asarray(spec["needle_R"]))
        d.qpos[a:a + 3] = spec["needle_p"]
        d.qpos[a + 3:a + 7] = [w, x, y, z]
        d.qvel[v:v + 6] = 0.0
        d.qpos[rob._ref_joint_pos_indexes] = spec["arm_q"]
        d.qvel[rob._ref_joint_vel_indexes] = 0.0
        d.qpos[rob._ref_gripper_joint_pos_indexes] = spec["grip_q"]
        d.qvel[rob._ref_gripper_joint_vel_indexes] = 0.0
        # robosuite's Robotiq command is incremental (current_action moves 0.01 per control call): without this the
        # fingers first open from the reset's command and drop the needle
        rob.gripper.current_action = np.array(spec["grip_action"], copy=True)
        self.env.sim.forward()
        rob.controller.update(force=True)
        rob.controller.reset_goal()
        obs = self.env._get_observations(force_update=True)
        rl_obs, high_res_images = self._extract_images(obs)
        return rl_obs, high_res_images

    def _apply_init_state(self, st: dict):
        """Write a get_init_state() dict into the freshly reset (and DR'd) sim: objects at their base-relative
        pose under THIS episode's base (z kept: they rest on this episode's table), arm at the stored joints."""
        from robosuite.utils.transform_utils import mat2quat
        m, d = self.env.sim.model, self.env.sim.data
        bid = m.body_name2id(self.env.robots[0].robot_model.root_body)
        b, Rb = np.array(d.body_xpos[bid]), np.array(d.body_xmat[bid]).reshape(3, 3)
        for jn, (p_rel, R_rel) in st["objects"].items():
            a = m.get_joint_qpos_addr(jn)[0]
            v = m.get_joint_qvel_addr(jn)[0]
            p = b + Rb @ np.asarray(p_rel)
            R = Rb @ np.asarray(R_rel)
            x, y, z, w = mat2quat(R)
            d.qpos[a:a + 3] = [p[0], p[1], d.qpos[a + 2]]
            d.qpos[a + 3:a + 7] = [w, x, y, z]
            d.qvel[v:v + 6] = 0.0
        rob = self.env.robots[0]
        d.qpos[rob._ref_joint_pos_indexes] = st["q"]
        d.qvel[rob._ref_joint_vel_indexes] = 0.0
        self.env.sim.forward()
        rob.controller.update(force=True)
        rob.controller.reset_goal()

    def reset(self, init_state: Optional[dict] = None) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """init_state: a get_init_state() dict to start from (failure replay); None = the env's own sampling."""
        self.time_step = 0
        self.table_impact = False
        self.contact_health: dict = {}
        self._stage1_fired = False
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.terminal = False
        self.past_obses.clear()
        self.past_actions.clear()
        self._last_action = np.zeros_like(self._last_action)
        for _ in range(self.cond_action):
            self.past_actions.append(torch.zeros(self.action_dim))

        obs = self.env.reset()

        # Re-resolve bindings against the freshly-rebuilt sim (hard_reset).
        if self._needs_bindings:
            self._bindings = resolve_bindings(self.env)

        mutated = False
        if self._shift_active:
            self._apply_domain_shift()  # restores nominal + applies + forward()
            mutated = True
        if self._s2r_active:
            # Snapshot first: multiplicative DR ranges compose from the ORIGINAL
            # values, never from the previous episode's already-scaled ones (the
            # trap restore_nominal exists to avoid on the physics side).
            self._s2r_nominal = snapshot_visual_nominal(self.env)
            apply_sim2real(self.env, self.sim2real, nominal=self._s2r_nominal)
            self.last_physics_dr = apply_physics_dr(self.env, self.sim2real)
            self._needle_reset()
            self._gripper_timing_reset()
            # Which background photo this episode drew. The collector stores it
            # per episode so a re-render can repaint the same one (the photo is
            # in tex_rgb, which model_state does not carry).
            self.last_bg_index = getattr(self.env, "_s2r_bg_index", None)
            if self.sim2real.any_base_dr():
                # Moving the base translates the whole chain, so the OSC
                # controller must re-latch or it drives toward the pre-move goal.
                ctrl = self.env.robots[0].controller
                ctrl.update_initial_joints(
                    self.env.sim.data.qpos[
                        self.env.robots[0]._ref_joint_pos_indexes].copy())
                ctrl.reset_goal()
            mutated = True
        if self._ik_perturb_active:
            self._ik_to_perturbed_pos()  # forward()
            mutated = True
        if init_state is not None:
            self._apply_init_state(init_state)
            mutated = True
        # The obs returned by env.reset() is stale after we mutate the sim; pull
        # a fresh observation so the policy sees the shifted/perturbed start.
        if mutated:
            obs = self.env._get_observations(force_update=True)

        # Snapshot the ring center AFTER the shift forward so the displacement
        # gate measures drift from the true (possibly shifted) start pose.
        if self._success_validator is not None:
            self._success_validator.reset(self.env, self._bindings)

        rl_obs, high_res_images = self._extract_images(obs)

        if self.cond_action > 0:
            past_action = torch.from_numpy(np.stack(self.past_actions)).to(self.device)
            rl_obs["past_action"] = past_action

        return rl_obs, high_res_images

    def _needle_reset(self):
        """cell needle_com_y / inhand_dr_* (Threading): the needle's centre of mass (the model is rebuilt on a hard
        reset, so every episode), and this episode's in-hand re-placement draw (np.random, only when enabled, so the
        RNG stream of the other cells is unchanged)."""
        c = self.sim2real
        self._inhand_pending, self.last_inhand_dr = False, None
        if c.needle_com_y is None and not c.inhand_dr_prob:
            return
        m = self.env.sim.model
        nb = m.body_name2id("needle_obj_root")
        if c.needle_com_y is not None:
            assert abs(m.body_ipos[nb][1] - c.needle_com_y) < 1e-9, (
                f"needle CoM {m.body_ipos[nb][1]} != cell needle_com_y {c.needle_com_y}: the task class must install it")
        if c.inhand_dr_prob and np.random.uniform() < c.inhand_dr_prob:
            self._inhand_draw = dict(
                along=np.random.uniform(*c.inhand_dr_along_mm) / 1000, depth=np.random.uniform(*c.inhand_dr_depth_mm) / 1000,
                pitch=np.radians(np.random.uniform(*c.inhand_dr_pitch_deg)),
                yaw=np.radians(np.random.uniform(*c.inhand_dr_yaw_deg)))
            self._inhand_pending = True
            self._inhand_z0 = float(self.env.sim.data.body_xpos[nb][2])

    def _gripper_timing_reset(self):
        """cell gripper_speed / dr_gripper_speed / gripper_cmd_delay (Sim2RealConfig): this episode's Robotiq closing
        speed (robosuite's per-substep increment, patched on the gripper instance: the model is rebuilt on a hard
        reset) and command delay. np.random only when the DR fields are set, so the other cells' RNG stream is
        unchanged."""
        c = self.sim2real
        self._grip_queue, self.last_gripper_timing = None, None
        dr_speed, delay = getattr(c, "dr_gripper_speed", None), tuple(getattr(c, "gripper_cmd_delay", (0, 0)))
        speed = getattr(c, "gripper_speed", None)
        if speed is None and dr_speed is None and delay == (0, 0):
            return
        if dr_speed is not None:
            speed = float(np.random.uniform(*dr_speed))
        k = int(np.random.randint(delay[0], delay[1] + 1)) if delay != (0, 0) else 0
        if speed is not None:
            g = self.env.robots[0].gripper
            assert g.dof == 1, "gripper_speed: 1-DoF gripper only"

            def format_action(self_g, action, _s=speed):
                self_g.current_action = np.clip(self_g.current_action + _s * np.sign(action), -1.0, 1.0)
                return self_g.current_action
            g.format_action = types.MethodType(format_action, g)
        if k > 0:
            self._grip_queue = deque([-1.0] * k)   # the gripper starts open
        self.last_gripper_timing = dict(speed=speed, delay=k)

    def _maybe_inhand_dr(self, action) -> bool:
        """Re-place the needle in the gripper once, when it is first lifted in a closed gripper (see Sim2RealConfig
        inhand_dr_*). Keeps the needle's quaternion sign (the state carries the raw quaternion: q and -q, the same
        rotation, are different inputs) and zeroes its velocity. -> True when it moved the needle."""
        from scipy.spatial.transform import Rotation
        sim = self.env.sim
        m, d = sim.model, sim.data
        nb = m.body_name2id("needle_obj_root")
        if action[6] <= 0 or d.body_xpos[nb][2] - self._inhand_z0 < self.sim2real.inhand_dr_lift_m:
            return False
        self._inhand_pending = False
        sim.forward()                     # after mj_step the x* poses are one substep behind qpos
        gs = m.site_name2id("gripper0_grip_site")
        Rg, pg = d.site_xmat[gs].reshape(3, 3).copy(), d.site_xpos[gs].copy()
        s = 1.0 if (Rg.T @ d.body_xmat[nb].reshape(3, 3))[1, 1] >= 0 else -1.0   # body +y (tail) along grip +-y
        w = self._inhand_draw
        R0 = np.diag([-s, s, -1.0])       # body z up (grip z points down the fingers)
        R_in = Rotation.from_rotvec([0, 0, w["yaw"]]).as_matrix() @ Rotation.from_rotvec([-s * w["pitch"], 0, 0]).as_matrix() @ R0
        h = np.array([0.0, s * w["along"], w["depth"]])          # handle centre in the grip frame
        p_in = h - R_in @ np.array([0.0, 0.06, 0.0])               # needle root (the handle is at body +0.06 y)
        jid = m.joint_name2id("needle_obj_joint0")
        qa, va = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
        q = Rotation.from_matrix(Rg @ R_in).as_quat()
        q = np.array([q[3], q[0], q[1], q[2]])
        if q @ d.qpos[qa + 3:qa + 7] < 0:
            q = -q
        d.qpos[qa:qa + 3] = pg + Rg @ p_in
        d.qpos[qa + 3:qa + 7] = q
        d.qvel[va:va + 6] = 0.0
        sim.forward()
        self.last_inhand_dr = dict(t=self.time_step, along_mm=round(w["along"] * 1000, 2), depth_mm=round(w["depth"] * 1000, 2),
                                   pitch_deg=round(float(np.degrees(w["pitch"])), 2), yaw_deg=round(float(np.degrees(w["yaw"])), 2))
        return True

    def _contact_probe(self) -> dict:
        """Threading cell contact health, current state: summed gripper <-> table normal force [N] and the deepest
        interpenetration [m] of the needle / tripod / table / gripper pairs that the 10-21 / 10-22 fixes target."""
        import mujoco
        sim = self.env.sim
        m, d = sim.model, sim.data
        key = id(m._model)
        if getattr(self, "_probe_ids", (None,))[0] != key:      # the model is rebuilt on a hard reset
            def cls(g):
                n = m.geom_id2name(g) or ""
                if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                    return None
                if n.startswith("gripper0"):
                    return "grip"
                if n.startswith("needle_obj"):
                    return "needle"
                if n.startswith("tripod_obj"):
                    return "tripod"
                return "table" if n == "table_collision" else None
            self._probe_ids = (key, [cls(g) for g in range(m.ngeom)])
        kinds = self._probe_ids[1]
        out = dict(table_force=0.0, pen_needle_tripod=0.0, pen_tripod_table=0.0, pen_needle_table=0.0,
                   pen_grip_needle=0.0)
        fr = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {kinds[c.geom1], kinds[c.geom2]}
            if pair == {"grip", "table"}:
                mujoco.mj_contactForce(m._model, d._data, i, fr)
                out["table_force"] += abs(fr[0])
            elif len(pair) == 2 and None not in pair and c.dist < 0:
                k = "pen_" + "_".join(sorted(pair, key=["grip", "needle", "tripod", "table"].index))
                if k in out:
                    out[k] = max(out[k], -float(c.dist))
        return out

    def _table_impact_n(self) -> float:
        """summed normal force [N] of the current gripper <-> table contacts."""
        return self._contact_probe()["table_force"]

    def step(self, actions: torch.Tensor) -> tuple[dict, float, bool, bool, dict]:
        """
        all inputs and outputs are tensors
        """
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        num_action = actions.size(0)

        rl_obs = {}
        # record the action in original format from model
        if self.cond_action > 0:
            for i in range(actions.size(0)):
                self.past_actions.append(actions[i])
            past_action = torch.stack(list(self.past_actions)).to(self.device)
            rl_obs["past_action"] = past_action

        actions = actions.numpy()

        reward = 0
        success = False
        terminal = False
        high_res_images = {}
        for i in range(num_action):
            self.time_step += 1
            if self._grip_queue is not None:   # cell gripper_cmd_delay: the gripper executes the command of k steps ago
                act = np.array(actions[i], copy=True)
                self._grip_queue.append(float(act[6]))
                act[6] = self._grip_queue.popleft()
            else:
                act = actions[i]
            obs, step_reward, terminal, _ = self.env.step(act)
            self._last_action = np.clip(np.asarray(actions[i], np.float64), -1.0, 1.0)
            if self._inhand_pending and self._maybe_inhand_dr(act):
                obs = self.env._get_observations(force_update=True)
            # NOTE: extract images every step for potential obs stacking
            # this is not efficient
            curr_rl_obs, curr_high_res_images = self._extract_images(obs)

            if i == num_action - 1:
                rl_obs.update(curr_rl_obs)
                high_res_images.update(curr_high_res_images)

            # Strict-success gating: replace the raw, hackable sparse reward with
            # the validated signal (dwell + ring-displacement + optional speed),
            # so reward AND success agree on the honest objective. When disabled,
            # this is byte-for-byte the upstream behavior.
            if self._success_validator is not None:
                raw_success = step_reward == 1
                frame_success = self._success_validator.step(
                    self.env, self._bindings, raw_success
                )
                step_reward = 1.0 if frame_success else 0.0
            else:
                frame_success = step_reward == 1

            # table impact (sim2real cell): a hard gripper <-> table hit ends the episode as a failure; the same
            # probe keeps the per-episode contact-health maxima (self.contact_health)
            if self._s2r_active and getattr(self.sim2real, "table_impact_force", None) is not None:
                probe = self._contact_probe()
                for k, v in probe.items():
                    self.contact_health[k] = max(self.contact_health.get(k, 0.0), v)
                if probe["table_force"] > self.sim2real.table_impact_force:
                    step_reward, frame_success, terminal = 0.0, False, True
                    self.table_impact = True

            if self.stage_reward > 0.0 and not self._stage1_fired:
                metrics_fn = getattr(self.env, "_get_partial_task_metrics", None)
                if metrics_fn is not None and metrics_fn().get("first_piece_assembled"):
                    self._stage1_fired = True
                    step_reward += self.stage_reward

            reward += step_reward
            self.episode_reward += step_reward

            if frame_success:
                success = True
                if self.end_on_success:
                    terminal = True

            if terminal:
                break

        reward = reward * self.env_reward_scale
        self.terminal = terminal
        return rl_obs, reward, terminal, success, high_res_images


if __name__ == "__main__":
    from torchvision.utils import save_image

    env = PixelRobosuite("Lift", "Panda", 200, image_size=256, camera_names=GOOD_CAMERAS["Lift"])
    x = env.reset()[0][GOOD_CAMERAS["Lift"][0]].float() / 255
    print(x.dtype)
    print(x.shape)
    save_image(x, "test_env.png")

from typing import Optional
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
STATE_SHAPE = {
    "Lift": (19,),
    "PickPlaceCan": (23,),
    "NutAssemblySquare": (23,),
    "TwoArmTransport": (59,),
    "ToolHang": (53,),
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
# Threading-family variants (D1, Enlarged, ...) share Threading_D0's spec.
for _t in ["Threading_D1", "Threading_D2", "Threading_Enlarged"]:
    STATE_KEYS[_t] = DEFAULT_STATE_KEYS
STATE_SHAPE["Threading_D0"] = (37,)
STATE_SHAPE["Threading_D1"] = (37,)
STATE_SHAPE["Threading_D2"] = (37,)
STATE_SHAPE["Threading_Enlarged"] = (37,)

PROP_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
PROP_DIM = 9


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
        # ---- domain-shift / strict-success / init-perturb extensions ----
        domain_shift: Optional[DomainShiftConfig] = None,
        strict_success: Optional[StrictSuccessConfig] = None,
        hand_init_perturb: Optional[tuple] = None,
        hand_init_perturb_joint_sigma: Optional[float] = None,
        init_distribution: Optional[InitDistConfig] = None,
        state_only_no_render: bool = False,
        stage_reward: float = 0.0,
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
        )
        self.env_name = env_name
        self.rl_cameras = rl_cameras if isinstance(rl_cameras, list) else [rl_cameras]
        self.image_size = image_size
        self.rl_image_size = rl_image_size or image_size
        self.env_reward_scale = env_reward_scale
        self.end_on_success = end_on_success
        self.use_state = use_state
        self.state_keys = STATE_KEYS[env_name]
        self.prop_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
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
        self._state_shape: tuple[int] = (state_base * state_stack,)
        self.prop_shape: tuple[int] = (PROP_DIM * prop_stack,)
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

        high_res_images = {}
        rl_obs = {}

        if self.use_state:
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

        return rl_obs, high_res_images

    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.time_step = 0
        self._stage1_fired = False
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.terminal = False
        self.past_obses.clear()
        self.past_actions.clear()
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
        if self._ik_perturb_active:
            self._ik_to_perturbed_pos()  # forward()
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
            obs, step_reward, terminal, _ = self.env.step(actions[i])
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

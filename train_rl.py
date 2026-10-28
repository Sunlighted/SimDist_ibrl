import os
import time
import sys
from dataclasses import dataclass, field
import yaml
import copy
from typing import Optional
import pyrallis
import torch
import numpy as np
from tqdm import tqdm

import common_utils
from common_utils import ibrl_utils as utils
from evaluate import run_eval, run_eval_mp
from env.robosuite_wrapper import PixelRobosuite, EXEC_OBS_DIM, EXEC_OBS_KEY
from env.domain_shift import DomainShiftConfig, InitDistConfig
from env.strict_success import StrictSuccessConfig
from rl.q_agent import QAgent, QAgentConfig
from rl import replay
import train_bc


class FrozenAgentBCPolicy(torch.nn.Module):
    """Adapt a frozen source-domain QAgent to the BC-policy interface IBRL's
    `_act_ibrl` expects (`.act(obs, eval_mode=, cpu=) -> [B, action_dim]`).

    The wrapped agent is held in a list so it is NOT registered as a submodule
    (it must not land in the target agent's state_dict). `act` always runs the
    source's deterministic RL policy and strips any caller-supplied `feat` so
    the source encodes pixel obs with its own encoder.
    """

    def __init__(self, agent: QAgent):
        super().__init__()
        self._agent = [agent]
        self.training = False

    @property
    def agent(self) -> QAgent:
        return self._agent[0]

    def train(self, mode: bool = True):
        # Frozen regardless of `mode` (mirrors how add_bc_policy treats priors).
        self.training = False
        return self

    @torch.no_grad()
    def act(self, obs, *, eval_mode: bool = True, stddev: float = 0.0, cpu: bool = True):
        src_obs = {k: v for k, v in obs.items() if k != "feat"}
        return self.agent.act(src_obs, eval_mode=True, stddev=0.0, cpu=cpu)


class StatePrefixPolicy(torch.nn.Module):
    """A BC prior trained without exec_obs, in a run with it: the reference-arm state is appended LAST to
    obs["state"] (env/robosuite_wrapper.py EXEC_OBS_*), so the prior reads the unchanged prefix state[..., :dim]."""

    def __init__(self, policy: torch.nn.Module, dim: int):
        super().__init__()
        self.policy = policy
        self.dim = dim

    def _obs(self, obs):
        return dict(obs, state=obs["state"][..., : self.dim])

    def act(self, obs, **kwargs):
        return self.policy.act(self._obs(obs), **kwargs)

    def act_chunk(self, obs, **kwargs):
        return self.policy.act_chunk(self._obs(obs), **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.policy, name)


def load_frozen_agent(weight_file: str, env: PixelRobosuite, cfg: "MainConfig",
                      device: str, *, privileged_state: bool = False) -> QAgent:
    """Rebuild a source-domain QAgent from its own cfg.yaml + weights, frozen.

    Reuses `env`'s shapes (source and target are the same task) but the source's
    OWN q_agent architecture config. SGFT needs the source agent to consume the
    TARGET run's observations (both the V_src shaping and the BC prior run on
    the target obs), so the observation space — `use_state` and the stacking —
    MUST match between source and target, else the source net's input dim won't
    line up with the target obs (and load_state_dict would fail cryptically).

    privileged_state=True (PIXEL arm): the frozen agent is a PRIVILEGED STATE
    TEACHER. It does NOT consume this run's observations -- it reads the env's
    state vector (rl_obs["state"], which PixelRobosuite emits alongside the
    frames), so `use_state`/`obs_stack` are deliberately allowed to differ and
    only the STATE layout has to match. This is what lets a pixel world model be
    distilled against a state expert's V_src without training a pixel expert.
    """

    src_cfg_path = os.path.join(os.path.dirname(weight_file), "cfg.yaml")
    src_cfg = pyrallis.load(MainConfig, open(src_cfg_path, "r"))  # type: ignore

    if privileged_state:
        assert src_cfg.use_state, (
            "privileged_state=True needs a STATE-based source ckpt "
            f"(got use_state={src_cfg.use_state} in {src_cfg_path})")
        assert not cfg.use_state, (
            "privileged_state=True is only for a pixel run (use_state=0)")
        # obs space differs ON PURPOSE; only the STATE layout must line up.
        # `prop_stack` is deliberately NOT checked here: this branch already
        # asserts the source is state-based (use_state=1), and a state QAgent
        # provably never reads obs["prop"] -- every prop access in rl/q_agent.py
        # is behind `if not self.use_state`. train_simdist's only source-agent
        # call in the offline arm is value({"state": ...}), which passes no prop
        # at all. Checking it blocked the PIXEL history arm (prop_stack=K feeds
        # the world model's own encoder, not the frozen teacher) for no reason.
        # state_stack DOES matter and is still enforced.
        attrs = ("state_stack",)
    else:
        attrs = ("use_state", "state_stack", "obs_stack", "prop_stack")
    mismatches = [
        (a, getattr(src_cfg, a), getattr(cfg, a))
        for a in attrs
        if getattr(src_cfg, a) != getattr(cfg, a)
    ]
    # exec_obs changes the state layout (37 -> 64): the env must be built with the source's value
    if bool(getattr(src_cfg, "exec_obs", 0)) != bool(getattr(cfg, "exec_obs", 0)):
        mismatches.append(("exec_obs", src_cfg.exec_obs, getattr(cfg, "exec_obs", 0)))
    if mismatches:
        detail = ", ".join(f"{a}: source={s} vs run={t}" for a, s, t in mismatches)
        raise ValueError(
            f"SGFT source/target observation-space mismatch ({detail}). The "
            f"frozen source agent must consume THIS run's observations (V_src "
            f"shaping + IBRL BC prior), so the stacking must match. Fix: set "
            f"this run to the source's values (e.g. --state_stack "
            f"{src_cfg.state_stack} --obs_stack {src_cfg.obs_stack} "
            f"--prop_stack {src_cfg.prop_stack}), or retrain the source with "
            f"this run's config. (source ckpt: {weight_file})"
        )

    agent = QAgent(
        True if privileged_state else cfg.use_state,
        # privileged: the teacher is a STATE net, so hand it the state shape even
        # though env.observation_shape is (C,H,W) for this run.
        env.state_shape if privileged_state else env.observation_shape,
        env.prop_shape,
        env.action_dim,
        cfg.rl_camera,
        src_cfg.q_agent,
    )
    agent.load_state_dict(torch.load(weight_file))
    # Use the source's own RL policy (not its ibrl hybrid) as the prior/value.
    agent.cfg.act_method = "rl"
    agent.train(False)
    for p in agent.parameters():
        p.requires_grad_(False)
    return agent.to(device)


@torch.no_grad()
def _pe_warmstart_first_layer(agent, pretrained: dict, obs_dim: int, action_dim: int):
    """Copy a NON-PE pretrained agent's first layer into a PE-enabled agent so the
    PE agent starts byte-identical to the warm-started baseline. Requires the PE
    module to use include_raw + normalize=0 (so PE(s)[:, :obs_dim] == raw state).
    Layout of the PE'd first-layer input: [raw(obs_dim), sin(obs_dim*L), cos(...)]
    then the critic appends [action(action_dim)]. We copy raw+action blocks from
    the pretrained weights and zero the sin/cos blocks."""
    pe_dim = agent.critic.pe.out_dim  # obs_dim*(1+2L)
    assert agent.critic.pe.include_raw and not agent.critic.pe.normalize, (
        "PE warm-start copy needs include_raw=1 and normalize=0")

    # --- critic (_MultiLinear: weights [num_q, in, out], biases [1,num_q,1]) ---
    for name, mod in (("critic", agent.critic), ("critic_target", agent.critic_target)):
        w_new = mod.net[0].weights          # [num_q, pe_dim+action_dim, out]
        w_old = pretrained[f"{name}.net.0.weights"]  # [num_q, obs_dim+action_dim, out]
        w_new.zero_()
        w_new[:, :obs_dim, :].copy_(w_old[:, :obs_dim, :])            # raw block
        w_new[:, pe_dim:pe_dim + action_dim, :].copy_(w_old[:, obs_dim:, :])  # action
        mod.net[0].biases.copy_(pretrained[f"{name}.net.0.biases"])

    # --- actor (Linear: weight [out, in], bias [out]) ---
    for name, mod in (("actor", agent.actor), ("actor_target", agent.actor_target)):
        w_new = mod.net[0].weight           # [out, pe_dim]
        w_old = pretrained[f"{name}.net.0.weight"]  # [out, obs_dim]
        w_new.zero_()
        w_new[:, :obs_dim].copy_(w_old)
        mod.net[0].bias.copy_(pretrained[f"{name}.net.0.bias"])
    print(f"[warm-start] PE first-layer copy: raw+action blocks from pretrained, "
          f"sin/cos zeroed (agent starts == non-PE baseline)")


def attach_ibrl_prior(agent: QAgent, weight_file: str, env: PixelRobosuite, device: str) -> str:
    """Re-attach the IBRL prior that `weight_file`'s agent was TRAINED with.

    `load_frozen_agent` deliberately drops the prior and pins act_method="rl",
    because SGFT only ever wanted V_src under the bare RL actor. But an agent
    trained with act_method="ibrl" EXECUTES the max-Q pick over {RL actor, prior},
    so `V(s) = min_k Q_k(s, pi_rl(s))` describes a policy it never runs — and the
    two Vs are not a monotone rescaling of each other (the prior lifts V only where
    it wins), so they induce DIFFERENT rankings. Anything that consumes the value as
    a ranking signal (value distillation, MPPI terminal value) must therefore attach
    the prior and call `agent.value(obs, use_ibrl=True)`.

    Two prior flavours exist in this repo:
      bc_policy         -> a trained BC head            (e.g. threading_ibrl_source_seed1)
      sgft_source_agent -> a frozen RL agent as prior   (e.g. ..._rlprior_seed1)

    Does NOT touch act_method, so the agent's own `act()` (used for data collection)
    keeps its previous behavior. Returns a description, or "" if the ckpt declares
    no prior (caller should then fall back to the bare-actor value).
    """
    import train_bc

    src_cfg = pyrallis.load(  # type: ignore
        MainConfig, open(os.path.join(os.path.dirname(weight_file), "cfg.yaml"), "r")
    )
    if src_cfg.bc_policy:
        bc_policy, _, _ = train_bc.load_model(src_cfg.bc_policy, device)
        if src_cfg.exec_obs:        # a prior trained without exec_obs reads the state prefix
            bc_policy = StatePrefixPolicy(bc_policy, env.state_shape[0] - EXEC_OBS_DIM)
        agent.add_bc_policy(copy.deepcopy(bc_policy))
        return f"BC head <- {src_cfg.bc_policy}" + (" (state prefix, exec_obs)" if src_cfg.exec_obs else "")
    if src_cfg.sgft_source_agent:
        prior_agent = load_frozen_agent(src_cfg.sgft_source_agent, env, src_cfg, device)
        agent.add_bc_policy(FrozenAgentBCPolicy(prior_agent))
        return f"frozen RL agent <- {src_cfg.sgft_source_agent}"
    return ""


@dataclass
class MainConfig(common_utils.RunConfig):
    seed: int = 1
    # env
    task_name: str = "Lift"
    episode_length: int = 200
    end_on_success: int = 1
    # render image in higher resolution for recording or using pretrained models
    image_size: int = 224
    rl_image_size: int = 96
    rl_camera: str = "robot0_eye_in_hand"
    obs_stack: int = 1
    prop_stack: int = 1
    state_stack: int = 1
    # agent
    use_state: int = 0
    q_agent: QAgentConfig = field(default_factory=lambda: QAgentConfig())
    stddev_max: float = 1.0
    stddev_min: float = 0.1
    stddev_step: int = 500000
    nstep: int = 3
    discount: float = 0.99
    replay_buffer_size: int = 500
    batch_size: int = 256
    num_critic_update: int = 1
    update_freq: int = 2
    bc_policy: str = ""
    # rl with preload data
    mix_rl_rate: float = 1  # 1: only use rl, <1, mix in some bc data
    preload_num_data: int = 0
    preload_datapath: str = ""
    freeze_bc_replay: int = 1
    # pretrain rl policy with bc and finetune
    pretrain_only: int = 1
    pretrain_num_epoch: int = 0
    pretrain_epoch_len: int = 10000
    load_pretrained_agent: str = ""
    load_policy_only: int = 1
    # Mirror image of load_policy_only: keep the pretrained CRITIC and throw the
    # pretrained actor away, so the RL actor still cold-starts and the ONLY thing
    # transferred from `load_pretrained_agent` is Q. That makes an arm that
    # differs from a plain IBRL run in exactly one field.
    # Mutually exclusive with load_policy_only (asserted below).
    load_critic_only: int = 0
    # Skip robosuite's offscreen renderer entirely for state-based runs. The
    # policy only ever reads obs["state"], so the 224x224 render is pure waste --
    # and building the EGL context is what breaks on MIG-partitioned GPUs, where
    # CUDA_VISIBLE_DEVICES is a "MIG-<uuid>" string and robosuite's EGL device
    # selection does int() on it:
    #     ValueError: invalid literal for int() with base 10: 'MIG-ca42af...'
    # (hit on Compute Canada's fir). It also removes the EGL context that makes
    # multi-process eval deadlock. Requires use_state=1; asserted below.
    # Default 0 keeps the old rendering path byte-for-byte.
    state_only_no_render: int = 0
    add_bc_loss: int = 0
    # IBRL prior-preference margin (safe policy improvement; see §26 CLAUDE.md +
    # train_ibrl_simdist). The RL actor's Q is penalized by `margin` in the max-Q
    # selection, so max-Q keeps deferring to the BC prior until the RL actor
    # genuinely beats the prior by > margin. margin(t) anneals linearly from
    # prior_margin_init to a PERMANENT floor prior_margin_final over
    # prior_margin_steps env steps:
    #   margin(t) = max(prior_margin_final, prior_margin_init*(1 - t/prior_margin_steps))
    # prior_margin_init = 0.0 (default) => plain IBRL, byte-for-byte unchanged.
    # Fixes the cold-start stall when the prior isn't strong enough for the
    # critic to self-calibrate (max-Q locks onto a still-bad RL actor).
    # One-shot intermediate-subtask bonus, passed to PixelRobosuite. Only the
    # TRAIN env gets it: eval must keep the original sparse success signal so
    # score/score stays comparable with every earlier run. 0.0 = off.
    stage_reward: float = 0.0
    prior_margin_init: float = 0.0
    prior_margin_steps: int = 100000
    prior_margin_final: float = 0.0
    # While the margin is above freeze_actor_margin_thresh, FREEZE the RL actor
    # (critic-only updates). §26 identified this as the necessary half of the
    # permanent-margin recipe: without it the free RL actor drifts on OOD-Q
    # during the prior-deferral phase and can collapse. 0 = off (default).
    freeze_actor_during_margin: int = 0
    freeze_actor_margin_thresh: float = 0.15
    # others
    env_reward_scale: float = 1
    num_warm_up_episode: int = 50
    # Count the transitions collected during warm_up() toward global_step, so
    # num_train_step is a TOTAL target-interaction budget rather than "budget
    # after an unbilled warm-up". Without this a model-free arm silently gets
    # num_warm_up_episode extra episodes of target interaction that the Dyna arms
    # (whose max_transitions counts every collected transition) are charged for.
    # The warm-up rollouts are real target steps; the preloaded demos are not
    # counted here because the Dyna arms do not count their preload either.
    # 0 restores the old behaviour for reproducing pre-2026-09-06 runs.
    count_warmup_transitions: int = 1
    num_eval_episode: int = 10
    save_per_success: int = -1
    mp_eval: int = 0  # eval with multiprocess
    num_train_step: int = 200000
    log_per_step: int = 5000
    # log
    save_dir: str = "exps/rl/run1"
    use_wb: int = 0
    # >0: also dump the agent state_dict every N train steps to
    # save_dir/snapshot_{step}.pt (alongside cfg.yaml, so load_model works on
    # each snapshot). Used to harvest a quality-spectrum of policy checkpoints
    # {random -> expert} for SimDist diverse data generation (方案A / §17).
    snapshot_period: int = 0
    # If set, dump the ENTIRE RL replay buffer (every collected transition,
    # grouped per-episode) to this hdf5 path at the end of training. Empty
    # = off (default; back-compat). Used to harvest a full source-domain
    # replay dataset. NOTE: to avoid FIFO eviction of early episodes, make
    # `replay_buffer_size` (in EPISODES) larger than the number of episodes
    # the run will produce.
    save_replay_buffer: str = ""
    # domain shift (Threading_D0 only for now)
    train_domain_shift: DomainShiftConfig = field(default_factory=DomainShiftConfig)
    eval_domain_shift: DomainShiftConfig = field(default_factory=DomainShiftConfig)
    # Reset placement distribution override (Threading-only). Enlarges the
    # needle/tripod reset ranges (e.g. D1 + wider needle) for recovery training.
    # Disabled by default → upstream Threading_D0 bounds. Train/eval split so you
    # can train on a wide dist but eval on the nominal one if desired.
    train_init_dist: InitDistConfig = field(default_factory=InitDistConfig)
    eval_init_dist: InitDistConfig = field(default_factory=InitDistConfig)
    # Strict-success gate (Threading-only). Applied to BOTH train and eval so
    # the reward signal and the eval metric agree on the honest objective.
    # Disabled by default; enable in YAML with `strict_success.enabled: true`.
    strict_success: StrictSuccessConfig = field(default_factory=StrictSuccessConfig)
    # SIM-TO-REAL cell (env/sim2real.py): path to a cell YAML such as
    # release/cfgs/sim2real/threading_real_cell.yaml. Applied to BOTH the train and
    # the eval env -- it describes the physical bench (table height, where parts
    # are laid out, cameras), not a source/target domain, so the two must agree.
    # A path rather than an inline block because the cell has a variable-length
    # list of nested camera dataclasses, which pyrallis cannot decode.
    # "" = off (byte-for-byte legacy). The file's sha256 is logged at startup:
    # the YAML is edited in place as calibration improves, so the path alone does
    # not pin what a run actually trained on.
    sim2real_cell: str = ""
    # comma list of Sim2RealConfig field overrides on top of the cell, e.g. "layout_scale=0.25". Applied to the train
    # AND eval envs (and so to load_model / eval.py, which rebuild the Workspace from this cfg). "" = the cell as is.
    sim2real_overrides: str = ""
    # FAILURE REPLAY resets (Mulligan, arXiv 2610.05882, adapted to sim RL; calib_20261015/IBRL_EXPERIMENTS.md):
    # with probability fail_replay_prob a training episode starts from a stored FAILED initial state (object poses +
    # arm joints; the DR of the episode is drawn fresh). New failures enter a FIFO buffer of fail_replay_cap; a replayed
    # state leaves it when it succeeds ("retired") or after fail_replay_max_tries failed replays. 0 = off (unchanged).
    fail_replay_prob: float = 0.0
    fail_replay_cap: int = 200
    # EXECUTION-LAYER STATE in the obs (2026-10-25): the cell's exec layer makes an action a delta on a hidden reference
    # arm; with it off IBRL reaches 0.90 (325-500k) vs 0.57 with it on. 1 = append the reference-arm state + last action
    # (27) to obs["state"] for actor and critic; the BC prior keeps reading state[:37]. Preload demos need obs/exec_ref
    # (scripts/add_exec_obs_to_demos.py). 0 = off (unchanged).
    exec_obs: int = 0
    fail_replay_max_tries: int = 3
    # MID-EPISODE RESET to just before the needle first pushes the ring (Threading; 2026-10-23, failure anatomy in
    # calib_20261015/IBRL_EXPERIMENTS.md): at first contact successes and failures look identical (angle 4.8 vs 4.7
    # deg, 0.6 mm/step); the outcome is decided in the next ~20 steps. With probability mid_reset_prob a training
    # episode starts from a stored state mid_reset_back_{min,max} steps before an earlier episode's first ring push
    # (ring centre moved > mid_reset_push_mm vs step 5). The state is rebuilt EXACTLY by re-seeding the reset and
    # replaying the recorded actions (DR draws and exec-layer internals are not in sim_state; replay is verified
    # bit-exact against the stored sim_state, mismatches are counted and fall back to a normal reset). Failed fresh
    # episodes always enter the FIFO buffer, successful ones with prob mid_reset_success_frac (so the critic also sees
    # the recoveries). Entries retire / drop like fail replay. The remaining time budget is the original one
    # (horizon - t0). 0 = off (unchanged; the reset path is then byte-for-byte the old one).
    mid_reset_prob: float = 0.0
    mid_reset_back_min: int = 5
    mid_reset_back_max: int = 15
    mid_reset_push_mm: float = 10.0
    mid_reset_success_frac: float = 0.25
    mid_reset_cap: int = 200
    mid_reset_max_tries: int = 3
    # OMNIRESET-style reset pool (UWLab omnireset MultiResetManager, adapted to Threading; 2026-10-23): a
    # policy-independent pool of start states, one per reset type, sampled with these probabilities, e.g.
    # "fresh=0.25,grasped=0.25,preinsert=0.25,partial=0.25". fresh = the env's own reset (fail replay still applies);
    # grasped = needle in hand anywhere between lift and the pre-insertion pose; preinsert = needle ~6 cm before the
    # ring, being aligned; partial = needle partially through the ring (insert phase, before raw success). States
    # come from the scripted expert (scripts/threading_robotiq_expert.py; used ONLY to generate start states, never
    # as the prior) stopped at a random step of the phase, then omni_perturb_steps random actions (UWLab perturbs
    # the grasp pose); kept only if stable/valid (needle lifted, ring not moved, no deep penetration, not succeeded).
    # Rebuilt by seed + action replay exactly like mid_reset. omni_pool_size entries per type, generated at startup
    # and saved to <save_dir>/omni_pool.pt. "" = off.
    omni_reset_probs: str = ""
    # ONLINE DAGGER (2026-10-25): after a fresh / fail-replay training episode ends, dagger_k takeover steps t0 ~ U[0, T)
    # are drawn; the episode's state at t0 is rebuilt bit-exactly (seed + action replay, as mid_reset) and the scripted
    # expert takes over from there (ThreadingExpert.takeover, scripts/dagger_threading_robotiq.py offline). Successful
    # takeovers without a table impact go into the demo buffer (bc_replay: the mix_rl_rate demo half of every batch and,
    # with add_bc_loss, the actor's BC term) and, with dagger_to_rl, the RL replay. Successful episodes are labelled
    # with prob dagger_success_frac, failed ones always. The expert is a labeller only, never acts in the training
    # episodes or eval. Takeover steps are NOT counted in global_step (logged as dagger/steps_total). 0 = off.
    dagger_k: int = 0
    dagger_success_frac: float = 0.3
    dagger_to_rl: int = 1
    dagger_expert_steps: int = 350
    dagger_bc_capacity: int = 0          # bc_replay capacity in episodes; 0 = replay_buffer_size
    omni_pool_size: int = 200
    omni_perturb_steps: int = 8
    # faithful UWLab variant (env/omni_reset.py): types resting_grasped / anywhere_grasped / partial_grasped, states
    # written into the sim (needle + IK arm + dataset grasp) instead of reached by the expert; the grasp dataset =
    # omni_num_grasps in-hand transforms of the scripted expert's grasps
    omni_num_grasps: int = 40
    # PegInsertion board reset jitter (env/peg_insertion.py board_pos_noise [m,
    # uniform +- per axis] / board_yaw_noise [deg, uniform +-]), applied to BOTH
    # train and eval env. 0/0 = env_kwargs None = byte-for-byte legacy.
    board_pos_noise: float = 0.0
    board_yaw_noise: float = 0.0
    # PegInsertion pocket clearance per side (m; env hole_clearance). 0 = scanned 1.26 mm.
    hole_clearance: float = 0.0
    # SGFT (Simulation-Guided Fine-Tuning, Yin et al. ICLR 2025) on top of IBRL.
    # Path to a frozen source-domain QAgent checkpoint. When set, that agent is
    # reused for BOTH roles:
    #   (1) IBRL BC base — its (deterministic RL) policy becomes the prior the
    #       target actor competes with in max-Q (so set q_agent.act_method=ibrl
    #       and bootstrap_method=ibrl). Replaces a separately-trained bc_policy
    #       (the two are mutually exclusive).
    #   (2) SGFT value source — V_src(s)=Q_src(s,pi_src(s)) drives the critic's
    #       potential-based reward shaping when q_agent.sgft_shaping=1.
    # Only the target-domain training is implemented here; the source agent is
    # assumed already trained (normal train_rl on the source / no-shift env).
    sgft_source_agent: str = ""
    # Use the source agent's IBRL HYBRID (max-Q over its RL actor + its own
    # prior) as this run's prior, instead of its bare RL head. load_frozen_agent
    # pins act_method="rl" for SGFT's V_src semantics, which silently weakens the
    # prior for bootstrapping runs. Default 0 keeps every existing run bit-identical.
    source_agent_use_hybrid: int = 0
    # Robot arm init-state perturbation. Same three regimes as BC
    # (see env/robosuite_wrapper.py):
    #   None (default)   → robosuite's σ=0.02 gaussian on joint qpos
    #                      (≈22 mm EE std, matches MimicGen demo distribution)
    #   (0.0, 0.0, 0.0)  → fully deterministic init
    #   (rx, ry, rz) >0  → IK an EE offset from uniform half-range box
    # train_hand_init_perturb drives the RL exploration env (used to roll out
    # transitions into the replay buffer); eval_hand_init_perturb drives the
    # eval env used for in-training score logging.
    train_hand_init_perturb: Optional[tuple] = None
    eval_hand_init_perturb: Optional[tuple] = None
    # Joint-space gaussian σ (rad), mutually exclusive with the EE-IK
    # variants above. Use this to scale up the demo-style init noise
    # (robosuite default = 0.02). Perturbs both pos AND ori consistently.
    train_hand_init_perturb_joint_sigma: Optional[float] = None
    eval_hand_init_perturb_joint_sigma: Optional[float] = None

    def __post_init__(self):
        self.rl_cameras = self.rl_camera.split("+")

        if self.bc_policy in ["none", "None"]:
            self.bc_policy = ""

        if self.bc_policy:
            print(f"Using BC policy {self.bc_policy}")
            os.path.exists(self.bc_policy)

        if self.sgft_source_agent in ["none", "None"]:
            self.sgft_source_agent = ""

        if self.save_replay_buffer in ["none", "None"]:
            self.save_replay_buffer = ""

        # A bc_policy is consumed ONLY by _act_ibrl (act_method "ibrl"/"ibrl_soft")
        # and value(use_ibrl=True). With act_method="rl" the prior is loaded,
        # printed, and never consulted -- the run is plain RL wearing an IBRL name,
        # and prior_margin becomes dead code. That failure is silent and has now
        # cost five debugging sessions (CLAUDE.md rule 8), most recently two arms
        # differing only in prior_margin coming out byte-identical. Fail loudly.
        if self.bc_policy and self.q_agent.act_method == "rl":
            raise ValueError(
                "bc_policy is set but q_agent.act_method='rl' (the QAgentConfig "
                "default), so the prior would never enter the max-Q argmax and "
                "prior_margin would do nothing.\n"
                "Pass --q_agent.act_method ibrl (or ibrl_soft) for IBRL, or drop "
                "--bc_policy if plain RL is what you want."
            )

        assert not (self.load_policy_only and self.load_critic_only), (
            "load_policy_only and load_critic_only are mutually exclusive: the "
            "first keeps only the pretrained actor, the second only the "
            "pretrained critic. Set load_policy_only=0 to warm-start Q alone."
        )
        if self.load_critic_only and not self.load_pretrained_agent:
            raise ValueError(
                "load_critic_only=1 with an empty load_pretrained_agent -- there "
                "is no critic to warm-start from."
            )

        assert not (self.bc_policy and self.sgft_source_agent), (
            "bc_policy and sgft_source_agent are mutually exclusive — IBRL uses "
            "exactly one BC prior. Use sgft_source_agent to make the source RL "
            "agent the prior."
        )

        if self.pretrain_num_epoch > 0:
            assert self.preload_num_data > 0

        self.stddev_min = min(self.stddev_max, self.stddev_min)

        if self.state_only_no_render:
            assert self.use_state, "state_only_no_render requires use_state=1"
            assert self.save_per_success <= 0, \
                "state_only_no_render cannot record videos (no images rendered)"

        if self.preload_datapath:
            self.num_warm_up_episode += self.preload_num_data

        if self.task_name == "TwoArmTransport":
            self.robots: list[str] = ["Panda", "Panda"]
        else:
            self.robots: list[str] = ["Panda"]

    @property
    def bc_cameras(self) -> list[str]:
        if not self.bc_policy:
            return []

        bc_cfg_path = os.path.join(os.path.dirname(self.bc_policy), f"cfg.yaml")
        bc_cfg = pyrallis.load(train_bc.MainConfig, open(bc_cfg_path, "r"))  # type: ignore
        return bc_cfg.dataset.rl_cameras

    @property
    def stddev_schedule(self):
        return f"linear({self.stddev_max},{self.stddev_min},{self.stddev_step})"


class Workspace:
    def __init__(self, cfg: MainConfig, from_main=True):
        self.work_dir = cfg.save_dir
        print(f"workspace: {self.work_dir}")

        if from_main:
            common_utils.set_all_seeds(cfg.seed)
            sys.stdout = common_utils.Logger(cfg.log_path, print_to_stdout=True)

            pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # type: ignore
            print(common_utils.wrap_ruler("config"))
            with open(cfg.cfg_path, "r") as f:
                print(f.read(), end="")
            print(common_utils.wrap_ruler(""))

        self.cfg = cfg
        self.cfg_dict = yaml.safe_load(open(cfg.cfg_path, "r"))

        self.global_step = 0
        self.global_episode = 0
        self.train_step = 0
        self._setup_env()

        print(self.train_env.observation_shape)
        self.agent = QAgent(
            self.cfg.use_state,
            self.train_env.observation_shape,
            self.train_env.prop_shape,
            self.train_env.action_dim,
            self.cfg.rl_camera,
            cfg.q_agent,
        )

        if not from_main:
            return

        if cfg.load_pretrained_agent and cfg.load_pretrained_agent != "None":
            print(f"loading loading pretrained agent from {cfg.load_pretrained_agent}")
            critic_states = copy.deepcopy(self.agent.critic.state_dict())
            actor_states = copy.deepcopy(self.agent.actor.state_dict())
            pretrained = torch.load(cfg.load_pretrained_agent)
            # Shape-tolerant warm-start: load every param whose shape matches the
            # current agent, skip the rest. Identical to the old strict load when
            # architectures match; when Fourier (PE) encoding widens the state
            # input, the first layer (+ pe buffers) can't be transferred from a
            # non-PE ckpt, so they re-init while the rest of the nets warm-start.
            cur = self.agent.state_dict()
            compatible = {k: v for k, v in pretrained.items()
                          if k in cur and cur[k].shape == v.shape}
            skipped = [k for k in pretrained if k not in compatible]
            self.agent.load_state_dict(compatible, strict=False)
            if skipped:
                print(f"[warm-start] skipped {len(skipped)} shape-mismatched keys "
                      f"(PE input widening): e.g. {skipped[:4]}")
                # The repair below assumes the ONLY reason for a mismatch is
                # Fourier/PE widening the state input. When PE is off, a mismatch
                # means the architectures simply differ -- and silently continuing
                # would leave a randomly-initialised critic behind a warm-started
                # actor. Say which knobs to align instead of dying on
                # `'NoneType' object has no attribute 'out_dim'` three frames in.
                if getattr(self.agent.critic, "pe", None) is None:
                    raise ValueError(
                        f"load_pretrained_agent architecture mismatch: "
                        f"{len(skipped)} keys have different shapes and PE is off, "
                        f"so this is not the PE-widening case. e.g. {skipped[:4]}\n"
                        f"Align q_agent.state_critic / state_actor with the "
                        f"checkpoint's cfg.yaml (this project's IBRL runs are "
                        f"critic num_q=5 hidden_dim=1024 layer_norm=1, actor "
                        f"hidden_dim=1024 dropout=0.0; train_rl.py's dataclass "
                        f"defaults are 10x512 / 512 and do NOT match)."
                    )
                # Fourier (PE) widens the state input, so the first layer couldn't
                # transfer above (random -> broken critic feeding warm deep layers
                # -> cold-start deadlock). With include_raw + normalize=0 the PE
                # output's first obs_dim columns ARE the raw state, so copy the
                # pretrained first layer into the raw+action blocks and zero the
                # sin/cos blocks => the PE agent starts BYTE-IDENTICAL to the
                # warm-started baseline; PE features grow from zero contribution.
                _pe_warmstart_first_layer(
                    self.agent, pretrained,
                    self.train_env.observation_shape[0], self.train_env.action_dim,
                )
            if cfg.load_policy_only:
                # avoid overwriting critic
                self.agent.critic.load_state_dict(critic_states)
                self.agent.critic_target.load_state_dict(critic_states)
            if cfg.load_critic_only:
                # The reverse: undo the actor half of the load, so only Q carries
                # over. critic_target keeps the pretrained weights it just got
                # (load_state_dict above wrote both critic and critic_target).
                self.agent.actor.load_state_dict(actor_states)
                self.agent.actor_target.load_state_dict(actor_states)
                print("[warm-start] load_critic_only=1 -> critic + critic_target "
                      "from the pretrained agent, actor re-initialised")

        self.ref_agent = copy.deepcopy(self.agent)
        # override to always use RL even when self.agent is ibrl
        self.ref_agent.cfg.act_method = "rl"

        # set up bc related stuff
        self.bc_policy: Optional[torch.nn.Module] = None
        if cfg.bc_policy:
            bc_policy, _, bc_env_params = train_bc.load_model(cfg.bc_policy, "cuda")
            assert bc_env_params["obs_stack"] == self.eval_env_params["obs_stack"]
            if cfg.exec_obs:
                bc_policy = StatePrefixPolicy(bc_policy, self.train_env.observation_shape[0] - EXEC_OBS_DIM)

            self.agent.add_bc_policy(copy.deepcopy(bc_policy))
            self.bc_policy = bc_policy

        # SGFT: the source-domain agent serves as BOTH the IBRL BC prior and the
        # value source for potential-based reward shaping.
        self.sgft_source_agent: Optional[QAgent] = None
        if cfg.sgft_source_agent:
            print(f"loading SGFT source agent from {cfg.sgft_source_agent}")
            source_agent = load_frozen_agent(
                cfg.sgft_source_agent, self.train_env, cfg, "cuda"
            )
            if cfg.source_agent_use_hybrid:
                # load_frozen_agent pins act_method="rl", so the prior would be the
                # source's BARE RL head. An agent trained with act_method="ibrl"
                # EXECUTES the max-Q pick over {RL actor, its own prior} -- that
                # hybrid is what scores 99.2% here, the bare head is a policy it
                # never runs. Re-attach the source's own prior and let it act as
                # the hybrid, so the target run competes against the real expert.
                desc = attach_ibrl_prior(
                    source_agent, cfg.sgft_source_agent, self.train_env, "cuda")
                if not desc:
                    raise ValueError(
                        f"source_agent_use_hybrid=1 but {cfg.sgft_source_agent} "
                        f"declares no prior in its cfg.yaml (nothing to hybridise)")
                source_agent.cfg.act_method = "ibrl"
                source_agent.prior_margin = 0.0
                print(f"[SGFT] prior = source HYBRID (max-Q over RL actor + {desc})")
            # (1) BC prior for IBRL max-Q.
            self.agent.add_bc_policy(FrozenAgentBCPolicy(source_agent))
            self.bc_policy = self.agent.bc_policies[-1]
            # (2) value source for shaping (only acts when q_agent.sgft_shaping).
            self.agent.set_sgft_value_source(source_agent)
            self.sgft_source_agent = source_agent
            if self.agent.cfg.act_method == "rl":
                print("[SGFT] warning: q_agent.act_method='rl' — the source BC "
                      "prior won't be used in action selection; set 'ibrl' to "
                      "enable the IBRL max-Q hybrid.")

        self._setup_replay()

    def _board_env_kwargs(self):
        kw = {}
        if self.cfg.board_pos_noise or self.cfg.board_yaw_noise:
            print(f"[board jitter] pos +-{self.cfg.board_pos_noise} m, yaw +-{self.cfg.board_yaw_noise} deg")
            kw.update(board_pos_noise=self.cfg.board_pos_noise,
                      board_yaw_noise=self.cfg.board_yaw_noise)
        if self.cfg.hole_clearance:
            print(f"[pocket] clearance {self.cfg.hole_clearance * 1000:.2f} mm/side")
            kw.update(hole_clearance=self.cfg.hole_clearance)
        return kw or None

    def _setup_env(self):
        self.rl_cameras: list[str] = list(set(self.cfg.rl_cameras + self.cfg.bc_cameras))
        if self.cfg.use_state:
            self.rl_cameras = []
        print(f"rl_cameras: {self.rl_cameras}")

        if self.cfg.save_per_success > 0:
            for cam in ["agentview", "robot0_eye_in_hand"]:
                if cam not in self.rl_cameras:
                    print(f"Adding {cam} to recording camera because {self.cfg.save_per_success=}")
                    self.rl_cameras.append(cam)

        self.obs_stack = self.cfg.obs_stack
        self.prop_stack = self.cfg.prop_stack

        s2r = None
        if self.cfg.sim2real_cell:
            import hashlib
            from env.sim2real import load_sim2real
            s2r = load_sim2real(self.cfg.sim2real_cell)
            if self.cfg.sim2real_overrides:
                import dataclasses as _dc
                kv = {}
                for item in self.cfg.sim2real_overrides.split(","):
                    k, v = item.split("=")
                    k = k.strip()
                    assert k in {f.name for f in _dc.fields(s2r)}, f"sim2real_overrides: unknown field {k}"
                    kv[k] = float(v)
                s2r = _dc.replace(s2r, **kv)
                print(f"[sim2real] overrides {kv}")
            with open(self.cfg.sim2real_cell, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            print(f"[sim2real] cell={self.cfg.sim2real_cell} sha256={sha} "
                  f"enabled={s2r.enabled} table_above_base={s2r.table_above_base} "
                  f"workspace_offset={s2r.workspace_offset} any_dr={s2r.any_dr()}")

        self.train_env = PixelRobosuite(
            env_name=self.cfg.task_name,
            robots=self.cfg.robots,
            episode_length=self.cfg.episode_length,
            reward_shaping=False,
            image_size=self.cfg.image_size,
            rl_image_size=self.cfg.rl_image_size,
            camera_names=self.rl_cameras,
            rl_cameras=self.rl_cameras,
            env_reward_scale=self.cfg.env_reward_scale,
            end_on_success=bool(self.cfg.end_on_success),
            use_state=bool(self.cfg.use_state),
            obs_stack=self.obs_stack,
            state_stack=self.cfg.state_stack,
            prop_stack=self.prop_stack,
            record_sim_state=bool(self.cfg.save_per_success > 0),
            domain_shift=self.cfg.train_domain_shift,
            strict_success=self.cfg.strict_success,
            hand_init_perturb=self.cfg.train_hand_init_perturb,
            hand_init_perturb_joint_sigma=self.cfg.train_hand_init_perturb_joint_sigma,
            init_distribution=self.cfg.train_init_dist,
            state_only_no_render=bool(self.cfg.state_only_no_render),
            stage_reward=self.cfg.stage_reward,
            sim2real=s2r,
            exec_obs=bool(self.cfg.exec_obs),
            env_kwargs=self._board_env_kwargs(),
        )
        self.eval_env_params = dict(
            env_name=self.cfg.task_name,
            robots=self.cfg.robots,
            episode_length=self.cfg.episode_length,
            reward_shaping=False,
            image_size=self.cfg.image_size,
            rl_image_size=self.cfg.rl_image_size,
            camera_names=self.rl_cameras,
            rl_cameras=self.rl_cameras,
            use_state=self.cfg.use_state,
            obs_stack=self.obs_stack,
            state_stack=self.cfg.state_stack,
            prop_stack=self.prop_stack,
            domain_shift=self.cfg.eval_domain_shift,
            strict_success=self.cfg.strict_success,
            hand_init_perturb=self.cfg.eval_hand_init_perturb,
            hand_init_perturb_joint_sigma=self.cfg.eval_hand_init_perturb_joint_sigma,
            init_distribution=self.cfg.eval_init_dist,
            state_only_no_render=bool(self.cfg.state_only_no_render),
            sim2real=s2r,
            env_kwargs=self._board_env_kwargs(),
            exec_obs=bool(self.cfg.exec_obs),
        )
        self.eval_env = PixelRobosuite(**self.eval_env_params)  # type: ignore

    def _build_omni_pool(self, types: list, seeded_reset) -> dict:
        """omni_reset_probs: omni_pool_size start states per type from the scripted expert stopped at a random step
        of the type's phase + a random perturbation, validated like UWLab's check_reset_state_success. Each entry =
        (reset seed, actions, end sim state) and is replayed bit-exact by the training loop."""
        path = os.path.join(self.work_dir, "omni_pool.pt")
        if os.path.exists(path):
            pool = torch.load(path, weights_only=False)
            print(f"[omni] loaded {path}: " + ", ".join(f"{k} {len(v)}" for k, v in pool.items()))
            return pool
        from scripts.threading_robotiq_expert import ThreadingExpert
        env, rng = self.train_env, np.random.RandomState(self.cfg.seed + 7919)
        pool, t_start = {}, time.time()
        tele = [t for t in types if t.endswith("_grasped")]
        if tele:                                             # faithful UWLab OmniReset: env/omni_reset.py
            from env.omni_reset import OmniResetGenerator
            gen = OmniResetGenerator(env, seeded_reset, rng)
            print(f"[omni] grasp dataset: {gen.collect_grasps(self.cfg.omni_num_grasps)} grasps "
                  f"({time.time() - t_start:.0f} s)", flush=True)
            for typ in tele:
                pool[typ], tries, why = [], 0, {}
                while len(pool[typ]) < self.cfg.omni_pool_size and tries < 20 * self.cfg.omni_pool_size:
                    tries += 1
                    seed = int(rng.randint(2**31 - 1))
                    seeded_reset(seed, None)
                    spec = gen.draw(typ)
                    if spec is None:
                        k = gen.why.split(" ")[0]
                        why[k] = why.get(k, 0) + 1
                        continue
                    seeded_reset(seed, None)                 # the clean pass = what training replays
                    env.teleport(spec)
                    acts = gen.settle(typ)
                    if acts is None:
                        why["settle"] = why.get("settle", 0) + 1
                        continue
                    pool[typ].append(dict(seed=seed, init_state=None, spec=spec, actions=acts, t0=len(acts),
                                          state=env.env.sim.get_state().flatten().copy()))
                print(f"[omni] {typ}: {len(pool[typ])} states from {tries} resets, rejected {why} "
                      f"({time.time() - t_start:.0f} s)", flush=True)
            types = [t for t in types if t not in tele]
        # phase(s) to stop in, per-step stop hazard, perturbation amplitude, max perturbation steps
        spec = {"grasped": (("lift", "coarse"), 1 / 30, 0.5, self.cfg.omni_perturb_steps),
                "preinsert": (("fine",), 1 / 12, 0.3, self.cfg.omni_perturb_steps),
                "partial": (("insert",), 1 / 10, 0.1, min(3, self.cfg.omni_perturb_steps))}
        nb = env.env.sim.model.body_name2id("needle_obj_root")
        for typ in types:
            phases, hazard, amp, kmax = spec[typ]
            pool[typ], tries = [], 0
            while len(pool[typ]) < self.cfg.omni_pool_size and tries < 6 * self.cfg.omni_pool_size:
                tries += 1
                seed = int(rng.randint(2**31 - 1))
                seeded_reset(seed, None)
                e = env.env
                ex = ThreadingExpert(e)
                z0 = e.sim.data.xpos[nb][2]
                acts, ring5, stopped, term = [], None, False, False
                for t in range(self.cfg.episode_length - 40):
                    a, info = ex.act()
                    in_phase = info["phase"] in phases and not (info["phase"] == "lift" and info["k"] < 5)
                    if typ == "partial" and in_phase:
                        in_phase = ex._needle_errors(0.0)[1] < 0           # tip already past the ring centre
                    if in_phase and rng.rand() < hazard:
                        stopped = True
                        break
                    if info["phase"] in ("hold", "done"):
                        break
                    a = torch.tensor(a, dtype=torch.float32)
                    _, _, term, _, _ = env.step(a)
                    acts.append(a)
                    if t == 5:
                        ring5 = env.ring_centre().copy()
                    if term:
                        break
                if not stopped or term or ring5 is None:
                    continue
                for _ in range(rng.randint(0, kmax + 1)):            # UWLab perturbs the sampled grasp pose
                    a = torch.tensor(np.r_[rng.uniform(-amp, amp, 6), 1.0], dtype=torch.float32)
                    _, _, term, _, _ = env.step(a)
                    acts.append(a)
                    if term:
                        break
                if term:
                    continue
                # the expert's ref() reads the exec layer's shadow arm and leaves a 1e-13 trace in the dynamics, so a
                # plain replay is not bit-exact with this pass: the stored state (and its validation) comes from a
                # clean replay of the actions, which is what the training loop reproduces
                seeded_reset(seed, None)
                e = env.env
                for t, a in enumerate(acts):
                    _, _, term, _, _ = env.step(a)
                    if t == 5:
                        ring5 = env.ring_centre().copy()
                    if term:
                        break
                if term:
                    continue
                probe = env._contact_probe()
                ok = (e.sim.data.xpos[nb][2] - z0 > 0.02 and not e._check_success()
                      and np.linalg.norm(env.ring_centre() - ring5) < 0.015
                      and probe["pen_needle_tripod"] < 0.002 and probe["pen_grip_needle"] < 0.003)
                if ok:
                    pool[typ].append(dict(seed=seed, init_state=None, actions=acts, t0=len(acts),
                                          state=e.sim.get_state().flatten().copy()))
            t0s = [x["t0"] for x in pool[typ]]
            print(f"[omni] {typ}: {len(pool[typ])} states from {tries} expert runs, t0 median "
                  f"{np.median(t0s) if t0s else float('nan'):.0f} ({time.time() - t_start:.0f} s)", flush=True)
        torch.save(pool, path)
        return pool

    def _setup_replay(self):
        use_bc = False
        if self.cfg.mix_rl_rate < 1:
            use_bc = True
        if self.cfg.save_per_success > 0:
            use_bc = True
        if self.cfg.pretrain_num_epoch > 0 or self.cfg.add_bc_loss:
            assert self.cfg.preload_num_data
            use_bc = True

        self.replay = replay.ReplayBuffer(
            self.cfg.nstep,
            self.cfg.discount,
            frame_stack=1,
            max_episode_length=self.cfg.episode_length,
            replay_size=self.cfg.replay_buffer_size,
            use_bc=use_bc,
            save_per_success=self.cfg.save_per_success,
            save_dir=self.cfg.save_dir,
            bc_replay_size=self.cfg.dagger_bc_capacity,
        )

        if self.cfg.preload_num_data:
            replay.add_demos_to_replay(
                self.replay,
                self.cfg.preload_datapath,
                num_data=self.cfg.preload_num_data,
                rl_cameras=self.rl_cameras,
                use_state=self.cfg.use_state,
                obs_stack=self.obs_stack,
                state_stack=self.cfg.state_stack,
                prop_stack=self.prop_stack,
                reward_scale=self.cfg.env_reward_scale,
                record_sim_state=bool(self.cfg.save_per_success > 0),
                extra_state_keys=(EXEC_OBS_KEY,) if self.cfg.exec_obs else (),
                extra_prop_keys=(EXEC_OBS_KEY,) if self.cfg.exec_obs else (),
            )
        if self.cfg.freeze_bc_replay:
            assert self.cfg.save_per_success <= 0, "cannot save a non-growing replay"
            self.replay.freeze_bc_replay = True

    def eval(self, seed, policy) -> float:
        random_state = np.random.get_state()

        if self.cfg.mp_eval:
            scores: list[float] = run_eval_mp(
                env_params=self.eval_env_params,
                agent=policy,
                num_proc=10,
                num_game=self.cfg.num_eval_episode,
                seed=seed,
                verbose=False,
            )
        else:
            scores: list[float] = run_eval(
                env_params=self.eval_env_params,
                agent=policy,
                num_game=self.cfg.num_eval_episode,
                seed=seed,
                record_dir=None,
                verbose=False,
            )

        np.random.set_state(random_state)
        return float(np.mean(scores))  # type: ignore

    def warm_up(self):
        # warm up stage, fill the replay with some episodes
        # it can either be human demos, or generated by the bc, or purely random
        obs, _ = self.train_env.reset()
        self.replay.new_episode(obs)
        total_reward = 0
        num_episode = 0
        num_transition = 0
        target = self.cfg.num_warm_up_episode
        pbar = tqdm(total=target, initial=self.replay.size(),
                    desc="warm-up", dynamic_ncols=True)
        while True:
            if self.bc_policy is not None:
                # we have a BC policy
                with torch.no_grad(), utils.eval_mode(self.bc_policy):
                    action = self.bc_policy.act(obs, eval_mode=True)
            elif self.cfg.load_pretrained_agent or self.cfg.pretrain_num_epoch > 0:
                # the policy has been pretrained/initialized
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(obs, eval_mode=True)
            else:
                action = torch.zeros(self.train_env.action_dim)
                action = action.uniform_(-1.0, 1.0)

            obs, reward, terminal, success, image_obs = self.train_env.step(action)
            reply = {"action": action}
            self.replay.add(obs, reply, reward, terminal, success, image_obs)
            num_transition += 1

            if terminal:
                num_episode += 1
                total_reward += self.train_env.episode_reward
                pbar.n = min(self.replay.size(), target)
                pbar.set_postfix(succ=self.replay.num_success, ep=num_episode)
                pbar.refresh()
                if self.replay.size() < self.cfg.num_warm_up_episode:
                    self.replay.new_episode(obs)
                    obs, _ = self.train_env.reset()
                else:
                    break

        pbar.close()
        print(f"Warm up done. #episode: {self.replay.size()}")
        print(f"#episode from warmup: {num_episode}, #reward: {total_reward}")
        print(f"#transition from warmup: {num_transition}")
        if self.cfg.count_warmup_transitions:
            # Bill the warm-up against the interaction budget. This also shifts
            # every global_step-driven schedule (stddev, prior_margin) forward by
            # the same amount, which is the point: they should be functions of
            # target interaction spent, not of gradient steps taken.
            self.global_step += num_transition
            print(f"[budget] warm-up billed to global_step -> {self.global_step}"
                  f"/{self.cfg.num_train_step}")

    def _maybe_fit_fourier_stats(self):
        """If the state critic/actor use Fourier (PE) encoding, fit per-dim
        z-score stats over a large sample of replay states and push them into
        the agent's PE modules. No-op otherwise."""
        qa = self.cfg.q_agent
        if not (getattr(qa.state_critic, "fourier_l", 0) > 0
                or getattr(qa.state_actor, "fourier_l", 0) > 0):
            return
        n = min(8192, max(1, self.replay.size()))
        batch = self.replay.sample(n, "cuda:0")
        states = batch.obs["state"].float()
        mean = states.mean(0)
        std = states.std(0)
        num = self.agent.set_state_fourier_stats(mean, std)
        print(f"[fourier-PE] fit z-score on {n} replay states -> {num} PE modules "
              f"(|mean| avg={mean.abs().mean():.3f}, std avg={std.mean():.3f})")

    def train(self):
        stat = common_utils.MultiCounter(
            self.work_dir,
            bool(self.cfg.use_wb),
            wb_exp_name=self.cfg.wb_exp,
            wb_run_name=self.cfg.wb_run,
            wb_group_name=self.cfg.wb_group,
            config=self.cfg_dict,
        )
        self.agent.set_stats(stat)
        saver = common_utils.TopkSaver(save_dir=self.work_dir, topk=1)

        if self.replay.num_episode < self.cfg.num_warm_up_episode:
            self.warm_up()

        # Fit z-score stats for the optional Fourier (PE) state input, on the
        # warm-up + demo transitions now in the replay. No-op when PE is off.
        self._maybe_fit_fourier_stats()

        stopwatch = common_utils.Stopwatch()
        # failure replay (Mulligan-style; fail_replay_prob = 0 -> exactly the old reset path)
        fail_buf: list[dict] = []
        cur_replay = None                                    # the buffer entry this episode replays, if any
        fr_stats = {"replayed": 0, "retired": 0, "dropped": 0}

        # mid-episode reset (mid_reset_prob = 0 -> none of this runs)
        mid_on = self.cfg.mid_reset_prob > 0
        if mid_on:
            assert self.train_env.ring_centre() is not None, "mid_reset_* needs the Threading ring (strict success on)"
        dagger_on = self.cfg.dagger_k > 0
        rec_on = mid_on or dagger_on                         # record seed + actions of fresh episodes
        if dagger_on:
            assert self.replay.bc_replay is not None, "dagger_k needs a demo buffer (mix_rl_rate < 1 or add_bc_loss)"
            from scripts.threading_robotiq_expert import ThreadingExpert
        dagger_stats = {"tried": 0, "ok": 0, "added": 0, "mismatch": 0, "impact": 0, "steps": 0}
        mid_buf: list[dict] = []
        cur_mid = None                                       # the mid_buf entry this episode starts from, if any
        mid_stats = {"replayed": 0, "retired": 0, "dropped": 0, "mismatch": 0}
        # this episode's record: reset seed + init_state, actions, flattened sim states, ring centres
        ep_rec: dict = {}

        def _seeded_reset(seed, init_state):
            rng = np.random.get_state()                      # keep the training RNG stream independent of the seed
            np.random.seed(seed)
            out = self.train_env.reset(init_state=init_state)
            np.random.set_state(rng)
            return out

        def _start_record(seed, init_state):
            ep_rec.clear()
            ep_rec.update(seed=seed, init_state=init_state, t0=0, actions=[],
                          states=[self.train_env.env.sim.get_state().flatten().copy()],
                          ring=[self.train_env.ring_centre().copy()])

        def _replay_entry(e):
            """re-seed the reset and replay e's actions; None if the end state is not bit-exact e['state']"""
            obs, img = _seeded_reset(e["seed"], e["init_state"])
            if e.get("spec") is not None:                    # faithful OmniReset: written state + hold steps
                obs, img = self.train_env.teleport(e["spec"])
            term = False
            for a in e["actions"]:
                obs, _, term, _, img = self.train_env.step(a)
                if term:
                    break
            if term or not np.array_equal(self.train_env.env.sim.get_state().flatten(), e["state"]):
                return None                                  # never expected (replay verified bit-exact 10-23)
            return obs, img

        def _mid_reset():
            nonlocal cur_mid
            e = mid_buf[np.random.randint(len(mid_buf))]
            out = _replay_entry(e)
            if out is None:
                mid_stats["mismatch"] += 1
                mid_buf.remove(e)
                return None
            cur_mid = e
            mid_stats["replayed"] += 1
            return out

        # OmniReset-style pool (omni_reset_probs = "" -> none of this runs)
        omni_types, omni_p, omni_pool = [], None, {}
        cur_omni = None                                      # reset type of this episode when omni is on
        omni_stats = {"mismatch": 0}
        if self.cfg.omni_reset_probs:
            for item in self.cfg.omni_reset_probs.split(","):
                k, v = item.split("=")
                assert k in ("fresh", "grasped", "preinsert", "partial", "resting_grasped", "anywhere_grasped",
                             "partial_grasped"), f"omni_reset_probs: unknown type {k}"
                omni_types.append(k)
                omni_p = np.r_[omni_p if omni_p is not None else [], float(v)]
            omni_p = omni_p / omni_p.sum()
            omni_pool = self._build_omni_pool([t for t in omni_types if t != "fresh"], _seeded_reset)

        def _omni_reset():
            """None -> fresh reset; else (obs, img) from a pool state of the drawn type"""
            nonlocal cur_omni
            cur_omni = omni_types[np.random.choice(len(omni_types), p=omni_p)]
            if cur_omni == "fresh" or not omni_pool.get(cur_omni):
                cur_omni = "fresh"
                return None
            pool = omni_pool[cur_omni]
            e = pool[np.random.randint(len(pool))]
            out = _replay_entry(e)
            if out is None:
                omni_stats["mismatch"] += 1
                pool.remove(e)
                cur_omni = "fresh"
            return out

        def reset_train_env():
            nonlocal cur_replay, cur_mid, cur_omni
            cur_replay, cur_mid, cur_omni = None, None, None
            if hasattr(self.agent, "reset_prior"):
                self.agent.reset_prior()          # prior_chunk > 1: no chunk carries over an episode boundary
            if omni_types:
                out = _omni_reset()
                if out is not None:
                    ep_rec.clear()
                    return out
            if not rec_on and not omni_types:
                if self.cfg.fail_replay_prob > 0 and fail_buf and np.random.rand() < self.cfg.fail_replay_prob:
                    cur_replay = fail_buf[np.random.randint(len(fail_buf))]
                    fr_stats["replayed"] += 1
                    return self.train_env.reset(init_state=cur_replay["state"])
                return self.train_env.reset()
            if mid_buf and np.random.rand() < self.cfg.mid_reset_prob:
                out = _mid_reset()
                if out is not None:
                    ep_rec.clear()                           # mid-started episodes do not feed the buffer
                    return out
            init_state = None
            if self.cfg.fail_replay_prob > 0 and fail_buf and np.random.rand() < self.cfg.fail_replay_prob:
                cur_replay = fail_buf[np.random.randint(len(fail_buf))]
                fr_stats["replayed"] += 1
                init_state = cur_replay["state"]
            seed = int(np.random.randint(2**31 - 1))
            out = _seeded_reset(seed, init_state)
            if rec_on:
                _start_record(seed, init_state)
            return out

        def note_mid_end(success):
            if not mid_on:
                return
            if cur_mid is not None:
                if success:
                    mid_buf.remove(cur_mid); mid_stats["retired"] += 1
                else:
                    cur_mid["tries"] += 1
                    if cur_mid["tries"] >= self.cfg.mid_reset_max_tries:
                        mid_buf.remove(cur_mid); mid_stats["dropped"] += 1
                return
            if not ep_rec or (success and np.random.rand() >= self.cfg.mid_reset_success_frac):
                return
            ring = np.array(ep_rec["ring"])
            if len(ring) <= 6:
                return
            disp = np.linalg.norm(ring - ring[5], axis=1)    # vs step 5: the tripod settles ~5 mm after reset
            pushed = np.nonzero(disp > self.cfg.mid_reset_push_mm * 1e-3)[0]
            if not len(pushed):
                return
            t0 = int(pushed[0]) - np.random.randint(self.cfg.mid_reset_back_min, self.cfg.mid_reset_back_max + 1)
            if t0 < 6:
                return
            mid_buf.append(dict(seed=ep_rec["seed"], init_state=ep_rec["init_state"],
                                actions=ep_rec["actions"][:t0], state=ep_rec["states"][t0], t0=t0,
                                from_success=bool(success), tries=0))
            if len(mid_buf) > self.cfg.mid_reset_cap:
                mid_buf.pop(0)

        def dagger_takeover(success):
            """label states of the episode that just ended with the scripted expert's takeover (see dagger_k)"""
            if not dagger_on or not ep_rec or (success and np.random.rand() >= self.cfg.dagger_success_frac):
                return
            T = len(ep_rec["actions"])
            for t0 in sorted(set(int(t) for t in np.random.randint(0, T, size=self.cfg.dagger_k))):
                e = dict(seed=ep_rec["seed"], init_state=ep_rec["init_state"], actions=ep_rec["actions"][:t0],
                         state=ep_rec["states"][t0])
                out = _replay_entry(e)
                dagger_stats["tried"] += 1
                if out is None:
                    dagger_stats["mismatch"] += 1
                    continue
                obs_ = out[0]
                ex = ThreadingExpert(self.train_env.env)
                ex.takeover()
                obses, acts, rews, terms, ok = [obs_], [], [], [], False
                for _ in range(self.cfg.dagger_expert_steps):
                    a = torch.from_numpy(ex.act()[0].astype(np.float32))
                    obs_, r, term, ok, _ = self.train_env.step(a)
                    obses.append(obs_); acts.append(a); rews.append(r); terms.append(term)
                    if term:
                        break
                dagger_stats["steps"] += len(acts)
                if not ok:
                    continue
                dagger_stats["ok"] += 1
                if getattr(self.train_env, "table_impact", False):
                    dagger_stats["impact"] += 1
                    continue
                self.replay.add_external_episode(obses, acts, rews, terms, to_rl=bool(self.cfg.dagger_to_rl),
                                                 to_bc=True)
                dagger_stats["added"] += 1

        def note_episode_end(success, init_state):
            if self.cfg.fail_replay_prob <= 0:
                return
            if cur_replay is not None:
                if success:
                    fail_buf.remove(cur_replay); fr_stats["retired"] += 1
                else:
                    cur_replay["tries"] += 1
                    if cur_replay["tries"] >= self.cfg.fail_replay_max_tries:
                        fail_buf.remove(cur_replay); fr_stats["dropped"] += 1
            elif not success:
                fail_buf.append({"state": init_state, "tries": 0})
                if len(fail_buf) > self.cfg.fail_replay_cap:
                    fail_buf.pop(0)

        obs, _ = reset_train_env()
        ep_init = self.train_env.get_init_state() if self.cfg.fail_replay_prob > 0 else None
        self.replay.new_episode(obs)
        pbar = tqdm(total=self.cfg.num_train_step, initial=self.global_step,
                    desc="RL train", dynamic_ncols=True, mininterval=1.0)
        recent_train_scores: list[float] = []
        while self.global_step < self.cfg.num_train_step:
            ### prior-preference margin schedule (safe policy improvement) ###
            if self.cfg.prior_margin_init > 0:
                frac = max(0.0, 1.0 - self.global_step / max(self.cfg.prior_margin_steps, 1))
                self.agent.prior_margin = max(
                    self.cfg.prior_margin_final, self.cfg.prior_margin_init * frac
                )

            ### act ###
            with stopwatch.time("act"), torch.no_grad(), utils.eval_mode(self.agent):
                stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
                action = self.agent.act(obs, eval_mode=False, stddev=stddev)
                stat["data/stddev"].append(stddev)

            ### env.step ###
            with stopwatch.time("env step"):
                obs, reward, terminal, success, image_obs = self.train_env.step(action)
                if ep_rec:                                   # mid_reset: record this fresh episode for replay
                    ep_rec["actions"].append(action.detach().cpu().reshape(-1).clone())
                    ep_rec["states"].append(self.train_env.env.sim.get_state().flatten().copy())
                    ep_rec["ring"].append(self.train_env.ring_centre().copy())

            with stopwatch.time("add"):
                assert isinstance(terminal, bool)
                reply = {"action": action}
                self.replay.add(obs, reply, reward, terminal, success, image_obs)
                self.global_step += 1
                pbar.update(1)

            if terminal:
                with stopwatch.time("reset"):
                    self.global_episode += 1
                    stat["score/train_score"].append(float(success))
                    stat["data/episode_len"].append(self.train_env.time_step)
                    # contact health (sim2real cell with table_impact_force): impact rate + per-episode max depths [mm]
                    health = getattr(self.train_env, "contact_health", None)
                    if health:
                        stat["safety/table_impact"].append(float(self.train_env.table_impact))
                        stat["safety/table_force_max_N"].append(health.get("table_force", 0.0))
                        for k in ("pen_needle_tripod", "pen_tripod_table", "pen_needle_table", "pen_grip_needle"):
                            stat[f"safety/{k}_mm"].append(1e3 * health.get(k, 0.0))

                    # Rolling train-score window for pbar postfix (last 20 eps).
                    recent_train_scores.append(float(success))
                    if len(recent_train_scores) > 20:
                        recent_train_scores.pop(0)
                    pbar.set_postfix(
                        ep=self.global_episode,
                        succ=f"{np.mean(recent_train_scores):.2f}",
                        replay=self.replay.size(),
                    )

                    started_mid = cur_mid is not None or (cur_omni not in (None, "fresh"))
                    if not started_mid:                      # a mid-started episode has no initial state to replay
                        note_episode_end(bool(success), ep_init)
                    if omni_types:
                        stat[f"omni/success_{cur_omni}"].append(float(success))
                        stat["omni/mismatch_total"].append(omni_stats["mismatch"])
                    note_mid_end(bool(success))
                    if dagger_on:
                        dagger_takeover(bool(success))
                        for k, v in dagger_stats.items():
                            stat[f"dagger/{k}_total"].append(v)
                    if mid_on:
                        stat["mid_reset/buffer"].append(len(mid_buf))
                        stat["mid_reset/is_mid"].append(float(cur_mid is not None))
                        if cur_mid is not None:
                            stat["mid_reset/t0"].append(cur_mid["t0"])
                            stat["mid_reset/mid_success"].append(float(success))
                        for k, v in mid_stats.items():
                            stat[f"mid_reset/{k}_total"].append(v)
                    if self.cfg.fail_replay_prob > 0:
                        stat["fail_replay/buffer"].append(len(fail_buf))
                        stat["fail_replay/replayed_total"].append(fr_stats["replayed"])
                        stat["fail_replay/retired_total"].append(fr_stats["retired"])
                        stat["fail_replay/dropped_total"].append(fr_stats["dropped"])
                        stat["fail_replay/is_replay"].append(float(cur_replay is not None))

                    # reset env
                    obs, _ = reset_train_env()
                    ep_init = self.train_env.get_init_state() if self.cfg.fail_replay_prob > 0 else None
                    self.replay.new_episode(obs)

            ### logging ###
            if self.global_step % self.cfg.log_per_step == 0:
                self.log_and_save(stopwatch, stat, saver)

            ### periodic quality-spectrum snapshot (for SimDist diverse data-gen) ###
            if self.cfg.snapshot_period > 0 and self.global_step % self.cfg.snapshot_period == 0:
                snap = os.path.join(self.work_dir, f"snapshot_{self.global_step}.pt")
                torch.save(self.agent.state_dict(), snap)
                print(f"[snapshot] saved {snap}")

            ### train ###
            if self.global_step % self.cfg.update_freq == 0:
                with stopwatch.time("train"):
                    self.rl_train(stat)
                    self.train_step += 1
        pbar.close()

        # Dump the ENTIRE replay buffer (all collected transitions, grouped
        # per-episode) once training finishes. `replay_buffer_size` must be
        # large enough that no early episode was evicted (FIFO) — see the
        # field doc on MainConfig.save_replay_buffer.
        if self.cfg.save_replay_buffer:
            num_ep = self.replay.size()
            print(
                f"[replay-dump] saving full replay buffer: {num_ep} episodes "
                f"(capacity={self.cfg.replay_buffer_size}, "
                f"global_step={self.global_step}) -> {self.cfg.save_replay_buffer}"
            )
            if num_ep >= self.cfg.replay_buffer_size:
                print(
                    "[replay-dump] WARNING: #episodes >= replay_buffer_size — the "
                    "buffer was full, so the EARLIEST episodes were evicted (FIFO) "
                    "and are NOT in this dump. Re-run with a larger "
                    "--replay_buffer_size to capture all transitions."
                )
            out_dir = os.path.dirname(os.path.abspath(self.cfg.save_replay_buffer))
            os.makedirs(out_dir, exist_ok=True)
            self.replay.save_replay(self.cfg.save_replay_buffer)
            print("[replay-dump] done")

    def log_and_save(
        self,
        stopwatch: common_utils.Stopwatch,
        stat: common_utils.MultiCounter,
        saver: common_utils.TopkSaver,
    ):
        elapsed_time = stopwatch.elapsed_time_since_reset
        stat["other/speed"].append(self.cfg.log_per_step / elapsed_time)
        stat["other/elapsed_time"].append(elapsed_time)
        stat["other/episode"].append(self.global_episode)
        stat["other/step"].append(self.global_step)
        stat["other/train_step"].append(self.train_step)
        stat["other/replay"].append(self.replay.size())
        stat["score/num_success"].append(self.replay.num_success)

        if self.replay.bc_replay is not None:
            stat["data/bc_replay_size"].append(self.replay.size(bc=True))

        with stopwatch.time("eval"):
            eval_seed = (self.global_step // self.cfg.log_per_step) * self.cfg.num_eval_episode
            stat["eval/seed"].append(eval_seed)
            eval_score = self.eval(seed=eval_seed, policy=self.agent)
            stat["score/score"].append(eval_score)

            original_act_method = self.agent.cfg.act_method
            # if self.agent.cfg.act_method != "rl":
            #     with self.agent.override_act_method("rl"):
            #         rl_score = self.eval(seed=eval_seed, policy=self.agent)
            #         stat["score/score_rl"].append(rl_score)
            #         stat["score_diff/hybrid-rl"].append(eval_score - rl_score)

            if self.agent.cfg.act_method == "ibrl_soft":
                with self.agent.override_act_method("ibrl"):
                    greedy_score = self.eval(seed=eval_seed, policy=self.agent)
                    stat["score/greedy_score"].append(greedy_score)
                    stat["score_diff/greedy-soft"].append(greedy_score - eval_score)
            assert self.agent.cfg.act_method == original_act_method

        saved = saver.save(self.agent.state_dict(), eval_score, save_latest=True)
        # //1000 keeps this run on the same wandb x-axis as the dyna arms, which
        # call summary() once per 1000-transition round: 100k transitions -> 100.
        # Without it, log_per_step=5000 would put 300k at step 60 against dyna's 300.
        stat.summary(self.global_step, reset=True,
                     wandb_step=self.global_step // 1000)
        print(f"saved?: {saved}")
        stopwatch.summary(reset=True)
        print("total time:", common_utils.sec2str(stopwatch.total_time))
        print(common_utils.get_mem_usage())

    def rl_train(self, stat: common_utils.MultiCounter):
        stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
        for i in range(self.cfg.num_critic_update):
            if self.cfg.mix_rl_rate < 1:
                rl_bsize = int(self.cfg.batch_size * self.cfg.mix_rl_rate)
                bc_bsize = self.cfg.batch_size - rl_bsize
                batch = self.replay.sample_rl_bc(rl_bsize, bc_bsize, "cuda:0")
            else:
                batch = self.replay.sample(self.cfg.batch_size, "cuda:0")

            # in RED-Q, only update actor once
            update_actor = i == self.cfg.num_critic_update - 1

            # freeze the RL actor while the prior-preference margin is high (§26)
            if (update_actor and self.cfg.freeze_actor_during_margin
                    and self.agent.prior_margin > self.cfg.freeze_actor_margin_thresh):
                update_actor = False

            bc_batch = None
            if update_actor and self.cfg.add_bc_loss:
                bc_batch = self.replay.sample_bc(self.cfg.batch_size, "cuda:0")

            metrics = self.agent.update(batch, stddev, update_actor, bc_batch, self.ref_agent)

            stat.append(metrics)
            stat["data/discount"].append(batch.bootstrap.mean().item())

    def pretrain_policy(self):
        stat = common_utils.MultiCounter(
            self.work_dir,
            bool(self.cfg.use_wb),
            wb_exp_name=self.cfg.wb_exp,
            wb_run_name=self.cfg.wb_run,
            wb_group_name=self.cfg.wb_group,
            config=self.cfg_dict,
        )
        saver = common_utils.TopkSaver(save_dir=self.work_dir, topk=1)

        for epoch in tqdm(range(self.cfg.pretrain_num_epoch),
                          desc="pretrain", dynamic_ncols=True):
            for _ in range(self.cfg.pretrain_epoch_len):
                batch = self.replay.sample_bc(self.cfg.batch_size, "cuda")
                metrics = self.agent.pretrain_actor_with_bc(batch)

                for k, v in metrics.items():
                    stat[k].append(v)

            eval_seed = epoch * self.cfg.pretrain_epoch_len
            score = self.eval(eval_seed, policy=self.agent)
            stat["pretrain/score"].append(score)

            stat.summary(epoch, reset=True)
            saved = saver.save(self.agent.state_dict(), score, save_latest=True)
            print(f"saved?: {saved}")
            print(common_utils.get_mem_usage())


def load_model(weight_file, device):
    cfg_path = os.path.join(os.path.dirname(weight_file), f"cfg.yaml")
    print(common_utils.wrap_ruler("config of loaded agent"))
    with open(cfg_path, "r") as f:
        print(f.read(), end="")
    print(common_utils.wrap_ruler(""))

    cfg = pyrallis.load(MainConfig, open(cfg_path, "r"))  # type: ignore
    cfg.preload_num_data = 0  # override this to avoid loading data
    workplace = Workspace(cfg, from_main=False)

    eval_env = workplace.eval_env
    eval_env_params = workplace.eval_env_params
    agent = workplace.agent
    state_dict = torch.load(weight_file)
    agent.load_state_dict(state_dict)

    if cfg.bc_policy and cfg.exec_obs:
        # the prior's own env (no exec_obs) gives its input size; it reads the state prefix
        bc_policy, _, _ = train_bc.load_model(cfg.bc_policy, device)
        agent.add_bc_policy(StatePrefixPolicy(bc_policy, eval_env.observation_shape[0] - EXEC_OBS_DIM))
    elif cfg.bc_policy:
        bc_policy = train_bc._load_model(cfg.bc_policy, eval_env, device)
        agent.add_bc_policy(bc_policy)

    if cfg.sgft_source_agent:
        # Re-add the source agent as the IBRL BC prior (needed for act_method=
        # ibrl at eval). The value source is training-only, so not needed here.
        source_agent = load_frozen_agent(cfg.sgft_source_agent, eval_env, cfg, device)
        if cfg.source_agent_use_hybrid:
            # Mirror training: the prior this run competed against was the source's
            # ibrl HYBRID, so eval must rebuild the same thing or the loaded agent
            # silently faces a weaker prior than it trained with.
            desc = attach_ibrl_prior(source_agent, cfg.sgft_source_agent, eval_env, device)
            assert desc, (
                f"source_agent_use_hybrid=1 but {cfg.sgft_source_agent} declares "
                f"no prior in its cfg.yaml")
            source_agent.cfg.act_method = "ibrl"
            source_agent.prior_margin = 0.0
        agent.add_bc_policy(FrozenAgentBCPolicy(source_agent))

    # prior_margin is a runtime attribute set by the training loop, not part of
    # the state_dict, so a reloaded agent would silently act with margin 0 --
    # a different argmax than the one it trained and was evaluated with.
    # Restore the permanent floor (= the margin in force at the end of training).
    # 0.0 for every run without a margin => unchanged.
    if cfg.prior_margin_init > 0:
        agent.prior_margin = float(cfg.prior_margin_final)
        print(f"[load_model] restored prior_margin = {agent.prior_margin}")

    agent = agent.to(device)
    return agent, eval_env, eval_env_params


def main():
    cfg = pyrallis.parse(config_class=MainConfig)  # type: ignore
    workspace = Workspace(cfg)
    if cfg.pretrain_num_epoch > 0:
        print("Pretraining")
        workspace.pretrain_policy()
        if not cfg.pretrain_only:
            print("RL finetuning")
            workspace.train()
    else:
        workspace.train()

    if cfg.use_wb:
        wandb.finish()

    # The EGL/GPU MuJoCo render contexts + rela's non-daemon replay prefetch
    # threads block a normal interpreter shutdown, so the process hangs after
    # training finishes. All real work + the wandb flush are done -> flush stdio
    # and hard-exit, skipping the stuck teardown. (Replaces the old `assert False`
    # that crashed out for the same reason.)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    import wandb
    from rich.traceback import install

    install()
    os.environ.setdefault("MUJOCO_GL", "egl")   # setdefault, not "=": an outer
    # MUJOCO_GL=osmesa must survive. On MIG-partitioned GPUs (Compute Canada fir)
    # `import mujoco` with MUJOCO_GL=egl dies at IMPORT time -- "your EGL driver
    # does not support the PLATFORM_DEVICE extension" -- so those runs must pass
    # MUJOCO_GL=osmesa from outside, together with --state_only_no_render 1 so no
    # GL context is ever actually created.
    torch.backends.cudnn.allow_tf32 = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    main()

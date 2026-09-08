import os
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
from env.robosuite_wrapper import PixelRobosuite
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


def load_frozen_agent(weight_file: str, env: PixelRobosuite, cfg: "MainConfig", device: str) -> QAgent:
    """Rebuild a source-domain QAgent from its own cfg.yaml + weights, frozen.

    Reuses `env`'s shapes (source and target are the same task) but the source's
    OWN q_agent architecture config. SGFT needs the source agent to consume the
    TARGET run's observations (both the V_src shaping and the BC prior run on
    the target obs), so the observation space — `use_state` and the stacking —
    MUST match between source and target, else the source net's input dim won't
    line up with the target obs (and load_state_dict would fail cryptically).
    """
    src_cfg_path = os.path.join(os.path.dirname(weight_file), "cfg.yaml")
    src_cfg = pyrallis.load(MainConfig, open(src_cfg_path, "r"))  # type: ignore

    mismatches = [
        (a, getattr(src_cfg, a), getattr(cfg, a))
        for a in ("use_state", "state_stack", "obs_stack", "prop_stack")
        if getattr(src_cfg, a) != getattr(cfg, a)
    ]
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
        cfg.use_state,
        env.observation_shape,
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
        agent.add_bc_policy(copy.deepcopy(bc_policy))
        return f"BC head <- {src_cfg.bc_policy}"
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
        )
        self.eval_env = PixelRobosuite(**self.eval_env_params)  # type: ignore

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
        obs, _ = self.train_env.reset()
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

                    # Rolling train-score window for pbar postfix (last 20 eps).
                    recent_train_scores.append(float(success))
                    if len(recent_train_scores) > 20:
                        recent_train_scores.pop(0)
                    pbar.set_postfix(
                        ep=self.global_episode,
                        succ=f"{np.mean(recent_train_scores):.2f}",
                        replay=self.replay.size(),
                    )

                    # reset env
                    obs, _ = self.train_env.reset()
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

    if cfg.bc_policy:
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

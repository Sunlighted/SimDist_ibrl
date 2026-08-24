"""SimDist TARGET adaptation (SimDist paper's deploy-time adaptation, on top of a
stage-1-distilled world model).

Loads a stage-1 SimDist checkpoint (e.g.
`exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1/latest.pt`) and runs the two
adaptation phases the paper prescribes, both keeping the ENCODER (and everything
not named below) FROZEN:

  Stage 2  — TARGET behavior-clone of the POLICY head only.
             20 TARGET demos (release/data/mimicgen/threading_d0_target_dynamics),
             flat MSE BC on (s,a); only `model.policy` trains. ~`bc_steps` steps.

  Stage 3  — TARGET online adaptation of the DYNAMICS head only.
             Roll the SimDist MPPI planner in the TARGET env, push transitions into
             a (large) sequence buffer; every `adapt_interval` (1000) collected
             transitions, train ONLY `model.dynamics` for `dyn_update_steps`
             gradient steps (encoder/reward/value/policy all frozen), then EVAL
             `num_eval_episode` (500) episodes on TARGET and log to wandb. Repeat
             until `max_transitions` (20000) transitions have been collected
             (i.e. 20 logged points), keyed by #transitions.

Execution + eval use MPPI (`use_planner` is forced to 1 here regardless of the
stage-1 cfg, which trained/evaluated with the policy head directly).

Run:
  python train_simdist_adapt.py \
      --stage1_ckpt exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1/latest.pt \
      --save_dir exps/mimicgen/threading_simdist_adapt_seed1
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import yaml
import pyrallis
import torch
import numpy as np
from tqdm import tqdm

import common_utils
from common_utils import ibrl_utils as utils
from evaluate import run_eval, run_eval_mp
from rl.sequence_replay import SequenceReplayBuffer, load_demos_to_sequence_buffer
from train_simdist import MainConfig, Workspace


@dataclass
class AdaptConfig(common_utils.RunConfig):
    # the stage-1 checkpoint to adapt (its sibling cfg.yaml rebuilds the agent/env)
    stage1_ckpt: str = "exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1/latest.pt"
    save_dir: str = "exps/mimicgen/threading_simdist_adapt_seed1"
    seed: int = 1
    # stage 2: TARGET policy BC
    bc_steps: int = 1000             # ~"1000 epochs" of policy-only BC
    bc_lr: float = 3e-4
    bc_batch_size: int = 256
    # OVERRIDE the stage-2 demo set (else inherit the stage-1 cfg's
    # target_demo_path / target_num_data). Use to BC the policy on a LARGER target
    # demo set (e.g. the 500-demo expert set) so stage-2 yields a STRONG policy
    # (~0.875) before dynamics adaptation — tests whether a good proposal lets the
    # dynamics adapt to a higher MPPI ceiling than the weak 20-demo policy (~0.40).
    target_demo_path: Optional[str] = None
    target_num_data: Optional[int] = None
    # SOURCE ANCHOR on stage-2 (0.0 = OFF => legacy path byte-identical). Adds
    # `bc_anchor_coef * MSE(pi_theta(s_src), pi_src(s_src))` on SOURCE states, with
    # pi_src's actions captured BEFORE stage-2 begins. Makes the policy's distance
    # from pi_src an explicit knob (the frozen stage-1 value head is a distilled
    # V^{pi_src}; a policy that drifts far from pi_src may invalidate the terminal
    # value MPPI ranks with). Unlike shrinking `bc_steps`, this moves LESS in a
    # DIRECTED way: the demo term still fully fits the target actions where the
    # demos live, the anchor only pins states the demos never cover.
    bc_anchor_coef: float = 0.0
    bc_anchor_path: str = "release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5"
    bc_anchor_episodes: int = 2000   # source episodes to sample the anchor states from
    bc_anchor_states: int = 40000    # anchor pool size
    bc_anchor_batch: int = 256       # anchor minibatch per BC step
    # ---- FORM of the pi_src proximity constraint (all default 0 => legacy path) ----
    # The stage-2 anchor above is EXACTLY a KL penalty: modelling the deterministic
    # policy head as N(pi(s), sigma^2 I) gives KL = ||pi_theta-pi_src||^2/(2 sigma^2),
    # i.e. the MSE up to a constant absorbed by bc_anchor_coef. These three knobs
    # sweep the FORM of that same constraint at MATCHED D_SRC (the achieved distance),
    # so form is not confounded with strength:
    #   F1 bc_anchor_coef            KL penalty, correct (Fisher) metric   [existing]
    #   F2 bc_weight_decay           NOT a KL: pulls theta->0, not ->pi_src  (control)
    #   F3 bc_l2init_coef            KL with the Fisher metric replaced by I
    #   F4 bc_anchor_eps             KL as a per-state HARD trust region (needs F1 on)
    bc_anchor_eps: float = 0.0       # >0 => hinge relu(||da||-eps)^2 instead of MSE
    #   F5 bc_anchor_sat             KL's mirror on the g(d) axis: SATURATING
    #      (= Hellinger^2, 1-exp(-d^2/tau)) -- caps the penalty on large deviations
    #      so the demo term can win outright where it needs to. Tests whether any
    #      form reaches a (D_SRC, bc_loss) frontier strictly inside F1's.
    bc_anchor_sat: float = 0.0       # >0 => saturating; takes precedence over eps
    bc_l2init_coef: float = 0.0      # >0 => + coef * ||theta - theta_src||^2
    bc_weight_decay: float = 0.0     # >0 => Adam weight_decay on the policy head
    # Build the source pool + snapshot pi_src even when the anchor is OFF, purely to
    # report D_SRC (mean |pi_theta(s_src) - pi_src(s_src)|, the metric docs/simdist.md
    # §42 tabulates) at the end of stage-2. Needed to calibrate F2/F3 to F1's D_SRC.
    bc_measure_dsrc: int = 0
    # stage 3: TARGET dynamics adaptation
    adapt_interval: int = 1000       # collect this many TARGET transitions per round
    max_transitions: int = 20000     # stop after this many collected (=> #rounds)
    dyn_update_steps: int = 1000     # dynamics gradient steps per round (UTD knob = A)
    dyn_lr: float = 3e-4
    # Skip dynamics training until the buffer holds at least this many
    # transitions. Default 0 = old behavior. WHY: dyn_update_steps is a FIXED
    # 1000 grad steps x batch 256 = 256k samples per round, so round 1 does 256
    # EPOCHS over its 1000-transition buffer (round 5: 51, round 10: 26, round
    # 100: 2.6) at stage-1's lr -- and every adapt arm dips hard exactly in that
    # 5k-25k window before recovering. Same disease as stage-2's 157-epoch
    # unregularised BC that bc_anchor fixed (+35 pts).
    dyn_start_transitions: int = 0
    # (B) DEMO-INTO-DYNAMICS: mix this fraction of each dynamics-training batch from
    # the TARGET demos (demo_replay) instead of only the MPPI adapt buffer. The demos
    # are the cleanest, best-covered contact-phase (s,a,s') available and are otherwise
    # used ONLY for stage-2 policy BC — feeding them into the dynamics consistency loss
    # fixes the success-path contact dynamics up-front (free, no extra interaction).
    # 0.0 = OFF (dynamics trains on adapt buffer only; legacy). mse path only.
    dyn_demo_frac: float = 0.0
    # ---- SOURCE anchor for the DYNAMICS (mirror of bc_anchor_*, default OFF) ----
    # stage-3 refits the dynamics on whatever narrow on-policy slice the current
    # planner visits. A meta-analysis over 134 SimDist adapt runs found
    # corr(round-0 score, adapt gain) = -0.72 with every arm collapsing toward one
    # ~0.7 attractor no matter where it started -- i.e. arms that begin ABOVE it
    # get dragged DOWN. This pins next(z,a) to its stage-1 value on SOURCE (z,a)
    # the target rollouts never cover, leaving the target slice free to move.
    # Targets are snapshotted from the stage-1 dynamics BEFORE adapt starts.
    dyn_anchor_coef: float = 0.0
    dyn_anchor_path: str = "release/data/mimicgen/coffee_d0_source_diverse_v1/dataset.hdf5"
    dyn_anchor_episodes: int = 2000
    dyn_anchor_states: int = 40000
    dyn_anchor_batch: int = 256
    # (C) EXPLORATION BONUS at data collection: >0 sets the COLLECT-time planner's
    # dynamics_disagree_coef to -collect_disagree_coef, i.e. ret += coef*disagree, so
    # the MPPI rollout SEEKS the ensemble's most-uncertain regions (Plan2Explore/MAX
    # intrinsic reward). Fills dynamics-coverage holes faster, task-agnostically. Eval
    # always uses coef=0 (pure exploit). 0.0 = OFF. Needs num_dynamics_heads>1.
    collect_disagree_coef: float = 0.0
    # BALANCE SAMPLING: bias the dynamics-training sampler toward RECENTLY-collected
    # target rollouts. 0 (default) = plain uniform-over-(episode,start). >0 ramps the
    # per-episode weight by exp(recency_alpha * chronological_rank), so the newest
    # episode is exp(recency_alpha)x more likely than the oldest. Rationale: as the
    # dynamics head adapts, the freshest MPPI rollouts reflect the current (better)
    # planner, so upweighting them focuses the update on the current data distribution
    # while still keeping older data in the mix.
    recency_alpha: float = 0.0
    # SELF-DISTILL the policy on the planner's SUCCESSFUL target rollouts: each
    # round, after adapting dynamics, BC the policy head on (s, a_executed) from
    # the SUCCESS episodes in the adapt buffer (DAgger / planner->policy loop).
    # Makes the policy self-improve from few demos toward the BC ceiling AND
    # improves MPPI's proposal (virtuous loop). NOTE: clones SUCCESSES only, so it
    # cannot learn the hard configs that never succeed -> expected to reach ~the
    # BC ceiling, not the RL ~100% (which learns from failures via the value).
    selfdistill: int = 0
    selfdistill_steps: int = 500     # policy-BC gradient steps per round
    # FIXES for the death-spiral seen at 500 steps / no anchor / early-start:
    #  * warmup: don't self-distill until dynamics has stabilized MPPI (else we
    #    clone the FEW low-quality successes from a disrupted early MPPI).
    #  * demo_frac: anchor each self-distill batch to the clean stage-2 demos so
    #    the policy can't drift off them into overfit-to-a-handful-of-rollouts.
    selfdistill_warmup_transitions: int = 5000
    selfdistill_demo_frac: float = 0.5
    eval_policy_only: int = 1        # also log policy-head-only success each round
    needle_scale: float = 1.0        # widen the TARGET needle init box by this factor
                                     # (1.0=off); applied to BOTH collect + eval env.
    # Only eval every N rounds. Dynamics still trains EVERY round (UTD unchanged);
    # this only thins the (expensive) rollout eval. The final round always evals.
    eval_every_rounds: int = 1
    # Dump the stage-3 TARGET rollout buffer to <save_dir>/rollouts.npz in the
    # `scripts/verify_iql_critic.py` episode format, so the SAME transitions that
    # adapted the dynamics can be reused for offline IQL pretraining (no extra
    # target interaction). 0 = off.
    dump_rollouts: int = 0
    # also adapt the VALUE head on Monte-Carlo return-to-go from the TARGET
    # rollouts (incl. FAILURES=0): the frozen SOURCE value is flat on target so
    # MPPI's gamma^H*V(z_H) can't rank candidates. Unlike self-distill (successes
    # only), the value learns from failures too -> the RL-like "avoid bad states"
    # signal. When on, Stage-3 trains dynamics + value.
    adapt_value: int = 0
    # dynamics adapt loss: "mse"(latent 一致性, 原样) | "vaml"(只惩罚影响 value 评估的误差,
    # ||V(next(z,a))-sg V(encode(s'))||^2)。只在 adapt_value=0 时生效。用于"只改 loss"对照。
    dyn_loss: str = "mse"
    value_coef: float = 1.0
    adapt_batch_size: int = 256
    adapt_buffer_max_episode: int = 100000   # "large enough"
    # eval
    num_eval_episode: int = 500
    mp_eval: int = 1
    num_eval_proc: int = 10
    use_wb: int = 1
    # Override the TARGET eval env's hand-init joint-space gaussian sigma. None =>
    # inherit the stage-1 cfg (typically null => robosuite default sigma=0.02,
    # MimicGen demo-collection noise). Set e.g. 0.1 to match the project's other
    # final evals (recovery / weighted BC) for apples-to-apples comparison.
    eval_hand_init_perturb_joint_sigma: Optional[float] = None
    # Comma-separated PESSIMISM coefficients to sweep at every MPPI eval, e.g.
    # "0,0.01,0.05,0.1". Empty (default) => single eval at the planner's current
    # `dynamics_disagree_coef`, byte-identical to the old behaviour.
    # Each coef is logged separately (`score/score_c<coef>`); `score/score` keeps
    # the FIRST coef in the list so it stays comparable to a no-sweep run, and
    # `score/score_best` records the max.
    # WARNING: `score_best` is a max over len(coefs) noisy evals -> it carries a
    # winner's-curse bias of roughly +sigma_eval (~+0.05 at 100 games, 4 coefs).
    # A 1-head (plain) run cannot do the same sweep, so comparing its curve to
    # `score_best` is NOT apples-to-apples; compare to the matching fixed-coef
    # curve instead. (CLAUDE.md 38: this exact mechanism inflated a post-hoc
    # grouping by 2.4x.)
    eval_disagree_coefs: str = ""
    # Skip robosuite's offscreen renderer (state-based runs only). The stage-1
    # cfg.yaml on disk predates this field, so pyrallis fills the default 0 --
    # hence the explicit override onto base_cfg below. Required on MIG-partitioned
    # GPUs: CUDA_VISIBLE_DEVICES is a "MIG-<uuid>" string there and robosuite's
    # EGL device selection does int() on it. Default 0 = unchanged.
    state_only_no_render: int = 0
    # Override the MPPI planner to the ORIGINAL SimDist recipe (broad search:
    # low policy mixture + large init_std + more iterations) instead of the
    # ibrl-tuned "trust-the-proposal" defaults (mixture 0.8 / init_std 0.15).
    # See /home/Chenyang/simdist/config/control/mppi.yaml.
    original_mppi: int = 1
    # Comma-separated planner overrides applied AFTER original_mppi, e.g.
    # "init_std=0.15,base_policy_std=0.0". Same syntax as evaluate/eval.py's
    # --planner_set. Empty = untouched, so existing runs stay bit-identical.
    # Motivation: apply_original_recipe sets init_std=1.0 AND base_policy_std=0.1,
    # both measured HARMFUL on the Square rotw1 world model (source MPPI:
    # base_policy_std 0->0.1 cost 0.935->0.610; init_std 0.15->0.5 cost
    # 0.935->0.855). Whether that survives stage-3 dynamics adapt is exactly what
    # a paired arm with this flag set answers.
    planner_set: str = ""
    # ANCHOR POLICY: which policy seeds the MPPI base candidates. "" = the world
    # model's own (stage-2 BC'd) policy head, queried IN LATENT via
    # policy_from_latent (shared_encoder=1, no decode). A path loads a frozen
    # QAgent and installs it as planner.proposal_policy.
    # !! That switch also forces _policy_rollout onto the DECODE bridge
    # (z -> decode -> state -> policy), which the default path never touches and
    # which §30 measured at 53 mm round-trip error on a latent world model. So an
    # external anchor pays a decode tax the default does not, and the comparison
    # is confounded unless the control below is run too:
    #   --anchor_policy self_decode  = the SAME policy head, forced through decode.
    # (self_decode - default) is the tax; (external - self_decode) is the anchor.
    anchor_policy: str = ""

    def __post_init__(self):
        if self.stage1_ckpt in ["none", "None"]:
            self.stage1_ckpt = ""


class AdaptWorkspace:
    def __init__(self, cfg: AdaptConfig):
        self.cfg = cfg
        common_utils.set_all_seeds(cfg.seed)
        os.makedirs(cfg.save_dir, exist_ok=True)
        sys.stdout = common_utils.Logger(cfg.log_path, print_to_stdout=True)

        # --- rebuild the stage-1 agent + envs from the stage-1 cfg.yaml ---
        stage1_dir = os.path.dirname(cfg.stage1_ckpt)
        stage1_cfg_path = os.path.join(stage1_dir, "cfg.yaml")
        self.base_cfg: MainConfig = pyrallis.load(MainConfig, open(stage1_cfg_path, "r"))  # type: ignore
        # FORCE MPPI for adaptation rollout + eval (stage-1 may have used the policy head)
        self.base_cfg.simdist.use_planner = 1
        # ORIGINAL SimDist MPPI recipe (broad search), defined once in
        # rl.simdist_planner.apply_original_recipe: 100 noised-base + 250 random
        # candidates, init_std 1.0, min_std 0.05, 3 iters, temp 0.4, elites 64,
        # base_policy_std 0.1. The ibrl defaults were retuned to "trust the BC
        # proposal" (mixture 0.8 / init_std 0.15), which caps performance at the
        # frozen BC policy; this recipe explores widely and refines against the
        # value. NOTE two unavoidable deviations:
        #   * warm_start kept OFF: our mp/batched eval shares ONE planner instance
        #     across envs, so a per-instance _prev_mean would cross-contaminate
        #     (the original runs a single env). Per-env warm-start needs more
        #     plumbing; the broad init_std re-optimizes each step anyway.
        #   * horizon kept at the stage-1 value (3): the dynamics head was distilled
        #     for H=3 rollouts; planning deeper would query an untrained dynamics.
        if cfg.original_mppi:
            from rl.simdist_planner import apply_original_recipe
            p = apply_original_recipe(self.base_cfg.simdist.planner)
            _nb = p.num_base_override if p.num_base_override >= 0 else int(p.num_samples * p.mixture_coef)
            print("[adapt] using ORIGINAL SimDist MPPI recipe "
                  f"(base={_nb}+rand={p.num_samples - _nb}, init_std={p.init_std}, "
                  f"min_std={p.min_std}, iters={p.iterations}, temp={p.temperature}, "
                  f"elites={p.num_elites}, base_policy_std={p.base_policy_std}, "
                  f"horizon={p.horizon} [kept])")
        if cfg.planner_set:
            _p = self.base_cfg.simdist.planner
            for kv in cfg.planner_set.split(","):
                k, v = kv.split("=")
                k = k.strip()
                old = getattr(_p, k)
                setattr(_p, k, type(old)(v))
                print(f"[adapt] planner.{k}: {old} -> {getattr(_p, k)}")

        # optional override of the TARGET eval hand-init sigma (else inherit stage-1)
        if cfg.eval_hand_init_perturb_joint_sigma is not None:
            self.base_cfg.eval_hand_init_perturb_joint_sigma = \
                cfg.eval_hand_init_perturb_joint_sigma
            print(f"[adapt] override eval hand_init_perturb_joint_sigma = "
                  f"{cfg.eval_hand_init_perturb_joint_sigma}")
        if cfg.state_only_no_render:
            self.base_cfg.state_only_no_render = 1
            print("[adapt] state_only_no_render = 1 (no offscreen GL context)")
        self.ws = Workspace(self.base_cfg, from_main=False)
        self.agent = self.ws.agent
        print(f"loading stage-1 ckpt: {cfg.stage1_ckpt}")
        self.agent.load_state_dict(torch.load(cfg.stage1_ckpt))
        self.agent = self.agent.to("cuda")

        self.eval_env = self.ws.eval_env                 # TARGET env (eval_domain_shift)
        self.eval_env_params = self.ws.eval_env_params

        if cfg.anchor_policy:
            # ANCHOR = who seeds the MPPI base candidates. Mirrors
            # train_dynamics_adapt.py's --proposal_agent (a frozen LATENT SimDist
            # agent driving the planner through `policy_action`), except only the
            # POLICY is taken -- value/reward stay this run's own, so the single
            # variable is the anchor.
            if cfg.anchor_policy == "self_decode":
                _src = self.agent.model
                _desc = "OWN post-BC policy head, forced through decode (decode-tax control)"
            else:
                # SOURCE anchor: a stage-1 ckpt's policy head, i.e. the SOURCE-
                # distilled policy BEFORE stage-2 target BC overwrote it.
                import yaml as _yaml, dataclasses as _dc
                from rl.world_model import WorldModelConfig as _WMC
                from rl.simdist_agent import SimDistAgent as _SDA, SimDistConfig as _SDC
                from rl.simdist_planner import PlannerConfig as _PC
                _wm = _yaml.safe_load(open(os.path.join(
                    os.path.dirname(cfg.anchor_policy), "cfg.yaml")))["simdist"]["world_model"]
                _f = {x.name for x in _dc.fields(_WMC)}
                _wmc = _WMC(**{k: v for k, v in _wm.items() if k in _f})
                _lat = _SDA(True, self.eval_env.observation_shape, self.eval_env.prop_shape,
                            self.eval_env.action_dim, self.base_cfg.rl_camera,
                            _SDC(world_model=_wmc,
                                 planner=_PC(horizon=_wmc.horizon), use_planner=0),
                            device="cuda")
                _lat.load_state_dict(torch.load(cfg.anchor_policy, map_location="cuda",
                                                weights_only=False), strict=False)
                _lat.eval()
                for _p in _lat.parameters():
                    _p.requires_grad_(False)
                _src = _lat.model
                _desc = f"SOURCE stage-1 policy head <- {cfg.anchor_policy}"
            # NOTE: setting proposal_policy ALSO switches _policy_rollout off the
            # latent shortcut (policy_from_latent) onto the DECODE bridge, which
            # SS30 measured at 53 mm round-trip on a latent world model. Hence the
            # self_decode control: (self_decode - default) is the decode tax,
            # (source anchor - self_decode) is the anchor effect proper.
            self.agent.planner.proposal_policy = (
                lambda st, _m=_src: _m.policy_action(st))
            print(f"[adapt] anchor = {_desc}")


        # target+wide: widen the TARGET needle init box (D0 tripod stays fixed).
        # Applied to BOTH the collection env (self.eval_env) and the mp-eval params.
        if cfg.needle_scale != 1.0:
            from env.domain_shift import InitDistConfig
            from env.robosuite_wrapper import PixelRobosuite
            PI = 3.141592653589793
            s = cfg.needle_scale
            ncx, ncy, nhx, nhy = -0.125, 0.2, 0.075 * s, 0.05 * s
            idc = InitDistConfig(
                enabled=True,
                needle_x=(ncx - nhx, ncx + nhx), needle_y=(ncy - nhy, ncy + nhy),
                needle_z_rot=(-2.0 * PI / 3.0, -PI / 3.0),
                tripod_x=(0.0, 0.0), tripod_y=(-0.15, -0.15),
                tripod_z_rot=(PI / 2.0, PI / 2.0))
            self.eval_env_params = dict(self.eval_env_params)
            self.eval_env_params["init_distribution"] = idc
            self.eval_env = PixelRobosuite(**self.eval_env_params)  # type: ignore
            self.ws.eval_env = self.eval_env
            print(f"[adapt] TARGET+WIDE: needle box scaled {s}x -> "
                  f"x=({ncx-nhx:.3f},{ncx+nhx:.3f}) y=({ncy-nhy:.3f},{ncy+nhy:.3f}) "
                  f"(tripod fixed, z_rot D0)")

        self.H = self.base_cfg.simdist.world_model.horizon

        # optional override of the stage-2 demo set (else inherit the stage-1 cfg)
        if cfg.target_demo_path:
            self.base_cfg.target_demo_path = cfg.target_demo_path
            print(f"[adapt] override target_demo_path = {cfg.target_demo_path}")
        if cfg.target_num_data is not None:
            self.base_cfg.target_num_data = cfg.target_num_data
            print(f"[adapt] override target_num_data = {cfg.target_num_data}")

        # --- TARGET demos (stage-2 BC) ---
        self.demo_replay = SequenceReplayBuffer(max_episodes=10000, device="cpu")
        load_demos_to_sequence_buffer(
            self.demo_replay, self.base_cfg.target_demo_path,
            num_data=self.base_cfg.target_num_data,
            reward_scale=self.base_cfg.env_reward_scale,
        )
        assert self.demo_replay.can_sample(self.H), "no TARGET demo >= horizon"

        # --- TARGET rollout buffer (stage-3 dynamics adaptation) ---
        self.adapt_replay = SequenceReplayBuffer(
            max_episodes=cfg.adapt_buffer_max_episode, device="cpu")
        self._obs = None  # persistent rollout episode state (see _collect)
        self._total_transitions = 0  # for the self-distill warmup gate

        # dump the resolved adapt cfg
        self.cfg_dict = {**cfg.__dict__}
        pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # type: ignore
        self.stat = common_utils.MultiCounter(
            cfg.save_dir, bool(cfg.use_wb), wb_exp_name=cfg.wb_exp,
            wb_run_name=cfg.wb_run, wb_group_name=cfg.wb_group, config=self.cfg_dict,
        )
        self.saver = common_utils.TopkSaver(save_dir=cfg.save_dir, topk=1)

    # ----------------------------------------------------------------- eval
    def eval(self, seed: int, use_planner: Optional[int] = None) -> float:
        """use_planner=None -> current agent setting; 0 -> policy head only (fast);
        1 -> MPPI. Restores the agent's flag afterwards."""
        prev = self.agent.cfg.use_planner
        if use_planner is not None:
            self.agent.cfg.use_planner = use_planner
        random_state = np.random.get_state()
        try:
            if self.cfg.mp_eval:
                scores = run_eval_mp(
                    env_params=self.eval_env_params, agent=self.agent,
                    num_proc=self.cfg.num_eval_proc, num_game=self.cfg.num_eval_episode,
                    seed=seed, verbose=False)
            else:
                scores = run_eval(
                    env_params=self.eval_env_params, agent=self.agent,
                    num_game=self.cfg.num_eval_episode, seed=seed,
                    record_dir=None, verbose=False)
        finally:
            np.random.set_state(random_state)
            self.agent.cfg.use_planner = prev
        return float(np.mean(scores))

    # -------------------------------------------------- stage 2: policy BC
    def _build_source_anchor(self):
        """Sample SOURCE states and snapshot pi_src's actions on them BEFORE stage-2
        touches the policy head. Returns (states, frozen pi_src actions) or (None,
        None) when the anchor is off. Called at the top of `bc_finetune`, where
        `self.agent.model.policy` is still exactly the stage-1 (pi_src) head."""
        if self.cfg.bc_anchor_coef <= 0 and not self.cfg.bc_measure_dsrc:
            return None, None
        import h5py
        rng = np.random.RandomState(self.cfg.seed)
        with h5py.File(self.cfg.bc_anchor_path, "r") as f:
            keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            pick = rng.choice(len(keys), min(self.cfg.bc_anchor_episodes, len(keys)),
                              replace=False)
            S = np.concatenate([np.asarray(f[f"data/{keys[i]}"]["obs"]["state"])
                                for i in pick])
        if len(S) > self.cfg.bc_anchor_states:
            S = S[rng.choice(len(S), self.cfg.bc_anchor_states, replace=False)]
        pool = torch.as_tensor(S, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            tgt = torch.cat([self.agent.model.policy_action(pool[i:i + 8192])
                             for i in range(0, len(pool), 8192)])
        print(f"[anchor] coef={self.cfg.bc_anchor_coef} | {len(pick)} source episodes "
              f"-> {len(pool)} states | pi_src targets frozen "
              f"(|a| mean {float(tgt.abs().mean()):.4f})")
        return pool, tgt

    def _build_dyn_anchor(self):
        """Snapshot (z, a) -> next_stage1(z, a) on SOURCE data BEFORE stage-3 moves
        the dynamics. Encoder is frozen for the whole adapt, so z is stable and we
        can cache it. Returns (z, a, target) or (None, None, None) when off."""
        if self.cfg.dyn_anchor_coef <= 0:
            return None, None, None
        import h5py
        rng = np.random.RandomState(self.cfg.seed)
        S, A = [], []
        with h5py.File(self.cfg.dyn_anchor_path, "r") as f:
            keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            pick = rng.choice(len(keys), min(self.cfg.dyn_anchor_episodes, len(keys)),
                              replace=False)
            for i in pick:
                g = f[f"data/{keys[i]}"]
                st = np.asarray(g["obs"]["state"], dtype=np.float32)
                ac = np.asarray(g["actions"], dtype=np.float32)
                n = min(len(st), len(ac))
                S.append(st[:n]); A.append(ac[:n])
        S = np.concatenate(S); A = np.concatenate(A)
        if len(S) > self.cfg.dyn_anchor_states:
            j = rng.choice(len(S), self.cfg.dyn_anchor_states, replace=False)
            S, A = S[j], A[j]
        s_t = torch.as_tensor(S, device="cuda")
        a_t = torch.as_tensor(A, device="cuda")
        with torch.no_grad():
            z = torch.cat([self.agent.model.encode(s_t[i:i + 8192])
                           for i in range(0, len(s_t), 8192)])
            tgt = torch.cat([self.agent.model.next(z[i:i + 8192], a_t[i:i + 8192])
                             for i in range(0, len(z), 8192)])
        print(f"[dyn-anchor] coef={self.cfg.dyn_anchor_coef} | {len(pick)} source "
              f"episodes -> {len(z)} (z,a) pairs | stage-1 next() targets frozen")
        return z, a_t, tgt

    def bc_finetune(self):
        print(common_utils.wrap_ruler("stage 2: TARGET policy-only BC (encoder frozen)"))
        self.agent.train()
        anchor_pool, anchor_tgt = self._build_source_anchor()
        pbar = tqdm(total=self.cfg.bc_steps, desc="policy BC", dynamic_ncols=True,
                    mininterval=1.0)
        for step in range(self.cfg.bc_steps):
            states, actions, _ = self.demo_replay.sample(
                self.cfg.bc_batch_size, self.H, "cuda")
            a_s = a_a = None
            if anchor_pool is not None and self.cfg.bc_anchor_coef > 0:
                j = torch.randint(len(anchor_pool), (self.cfg.bc_anchor_batch,),
                                  device=anchor_pool.device)
                a_s, a_a = anchor_pool[j], anchor_tgt[j]
            metrics = self.agent.update_bc_policy(
                states, actions, lr=self.cfg.bc_lr,
                anchor_states=a_s, anchor_actions=a_a,
                anchor_coef=self.cfg.bc_anchor_coef,
                anchor_eps=self.cfg.bc_anchor_eps,
                anchor_sat=self.cfg.bc_anchor_sat,
                l2init_coef=self.cfg.bc_l2init_coef,
                weight_decay=self.cfg.bc_weight_decay)
            self.stat.append(metrics)
            pbar.update(1)
            if (step + 1) % max(self.cfg.bc_steps // 10, 1) == 0:
                pbar.set_postfix(loss=f"{metrics['bc/policy_loss']:.4f}")
        pbar.close()
        print(f"stage-2 done: final policy BC loss = {metrics['bc/policy_loss']:.5f}")
        if anchor_pool is not None:
            # D_SRC = mean elementwise |pi_theta(s_src) - pi_src(s_src)| over the
            # SOURCE pool (verified against the §42 lambda sweep: 0.2390/0.0522/
            # 0.0246/0.0182 for lambda=0/0.1/1/3 vs the tabulated 0.2354/0.0517/
            # 0.0248/0.0179). This is the axis F1..F4 are matched on.
            with torch.no_grad():
                cur = torch.cat([self.agent.model.policy_action(anchor_pool[i:i + 8192])
                                 for i in range(0, len(anchor_pool), 8192)])
                d = cur - anchor_tgt
                d_src = float(d.abs().mean())
                d_l2 = float(d.norm(dim=-1).mean())
            print(f"[D_SRC] {d_src:.4f}  (L2mean {d_l2:.4f})  "
                  f"bc_loss {metrics['bc/policy_loss']:.5f}  "
                  f"| anchor={self.cfg.bc_anchor_coef} eps={self.cfg.bc_anchor_eps} "
                  f"l2init={self.cfg.bc_l2init_coef} wd={self.cfg.bc_weight_decay}")
            self.stat.append({"bc/D_SRC": d_src})

    # ------------------------------------ stage 3: TARGET dynamics adaptation
    def _collect(self, n_transitions: int):
        """Roll the MPPI planner in the TARGET env, pushing transitions into the
        adapt buffer until `n_transitions` NEW transitions have been added. The
        rollout episode state (`self._obs`) persists across rounds so partial
        episodes spanning a round boundary are not discarded."""
        if self._obs is None:  # first call: start the first episode
            self._obs, _ = self.eval_env.reset()
            self.adapt_replay.new_episode(self._obs["state"])
        collected = 0
        pbar = tqdm(total=n_transitions, desc="collect (target MPPI)",
                    dynamic_ncols=True, mininterval=1.0, leave=False)
        # (C) exploration bonus: negative planner coef => MPPI SEEKS high-disagreement
        # (model-uncertain) regions during collection. Restored to the eval value
        # (0 => pure exploit) afterwards so eval is unaffected.
        prev_coef = self.agent.planner.cfg.dynamics_disagree_coef
        if self.cfg.collect_disagree_coef > 0:
            self.agent.planner.cfg.dynamics_disagree_coef = -self.cfg.collect_disagree_coef
        try:
            with utils.eval_mode(self.agent):
                while collected < n_transitions:
                    # MPPI action (use_planner forced on); deterministic planning
                    action = self.agent.act(self._obs, eval_mode=True, stddev=0.0, cpu=True)
                    self._obs, reward, terminal, success, _ = self.eval_env.step(action)
                    self.adapt_replay.add(self._obs["state"], action, reward, terminal, success)
                    collected += 1
                    pbar.update(1)
                    if terminal:
                        self._obs, _ = self.eval_env.reset()
                        self.adapt_replay.new_episode(self._obs["state"])
        finally:
            self.agent.planner.cfg.dynamics_disagree_coef = prev_coef
        pbar.close()

    def _train_dynamics(self):
        if self._total_transitions < self.cfg.dyn_start_transitions:
            print(f"[adapt] buffer {self._total_transitions} < "
                  f"dyn_start_transitions={self.cfg.dyn_start_transitions}; "
                  "skipping dynamics update this round")
            return
        if not self.adapt_replay.can_sample(self.H):
            print(f"[adapt] buffer has no episode >= horizon={self.H} yet; "
                  "skipping dynamics update this round")
            return
        self.agent.train()
        for _ in range(self.cfg.dyn_update_steps):
            if self.cfg.adapt_value:
                s, a, _, vt = self.adapt_replay.sample(
                    self.cfg.adapt_batch_size, self.H, "cuda",
                    return_value=True, discount=self.base_cfg.discount,
                    recency_alpha=self.cfg.recency_alpha)
                self.stat.append(self.agent.update_dynamics_value(
                    s, a, vt, lr=self.cfg.dyn_lr, value_coef=self.cfg.value_coef))
            else:
                B = self.cfg.adapt_batch_size
                states, actions, _ = self.adapt_replay.sample(
                    B, self.H, "cuda", recency_alpha=self.cfg.recency_alpha)
                if self.cfg.dyn_loss == "vaml":
                    self.stat.append(self.agent.update_dynamics_vaml(
                        states, actions, lr=self.cfg.dyn_lr))
                else:
                    # (B) mix in TARGET demos: cleanest contact-phase (s,a,s') so the
                    # dynamics consistency loss learns the success-path contact
                    # transitions up-front (free, no extra target interaction).
                    if self.cfg.dyn_demo_frac > 0 and self.demo_replay.can_sample(self.H):
                        nd = int(B * self.cfg.dyn_demo_frac)
                        if nd > 0:
                            ds, da, _ = self.demo_replay.sample(nd, self.H, "cuda")
                            states = torch.cat([states[:B - nd], ds], 0)
                            actions = torch.cat([actions[:B - nd], da], 0)
                    az = aa = at = None
                    if self._dyn_anchor[0] is not None:
                        k = torch.randint(len(self._dyn_anchor[0]),
                                          (self.cfg.dyn_anchor_batch,),
                                          device=self._dyn_anchor[0].device)
                        az, aa, at = (self._dyn_anchor[0][k], self._dyn_anchor[1][k],
                                      self._dyn_anchor[2][k])
                    self.stat.append(self.agent.update_dynamics_only(
                        states, actions, lr=self.cfg.dyn_lr,
                        anchor_z=az, anchor_a=aa, anchor_tgt=at,
                        anchor_coef=self.cfg.dyn_anchor_coef))
        # self-distill the policy on the planner's SUCCESSFUL rollouts, ANCHORED to
        # the clean stage-2 demos (demo_frac of each batch) and only AFTER dynamics
        # has warmed up (else we clone a handful of disrupted-MPPI successes).
        if self.cfg.selfdistill and self._total_transitions >= self.cfg.selfdistill_warmup_transitions:
            n_succ = self.adapt_replay.num_success_sampleable(self.H)
            self.stat["selfdistill/success_episodes"].append(n_succ)
            if n_succ > 0:
                nd = int(self.cfg.adapt_batch_size * self.cfg.selfdistill_demo_frac)
                ns = self.cfg.adapt_batch_size - nd
                for _ in range(self.cfg.selfdistill_steps):
                    ps, pa = [], []
                    if nd > 0:
                        ds, da, _ = self.demo_replay.sample(nd, self.H, "cuda")
                        ps.append(ds); pa.append(da)
                    if ns > 0:
                        ss, sa, _ = self.adapt_replay.sample(
                            ns, self.H, "cuda", success_only=True)
                        ps.append(ss); pa.append(sa)
                    s = torch.cat(ps, 0); a = torch.cat(pa, 0)
                    m = self.agent.update_bc_policy(s, a, lr=self.cfg.bc_lr)
                    self.stat["selfdistill/policy_loss"].append(m["bc/policy_loss"])

    def _eval_mppi(self, seed: int):
        """MPPI eval, optionally sweeping the pessimism coef. Returns
        (primary, best, {coef: score}); primary == the first coef in the list."""
        coefs = [c for c in self.cfg.eval_disagree_coefs.split(",") if c.strip()]
        if not coefs:
            v = self.eval(seed=seed, use_planner=1)
            return v, v, {}
        prev = self.agent.planner.cfg.dynamics_disagree_coef
        out = {}
        try:
            for c in coefs:
                cf = float(c)
                self.agent.planner.cfg.dynamics_disagree_coef = cf
                out[cf] = self.eval(seed=seed, use_planner=1)
        finally:
            self.agent.planner.cfg.dynamics_disagree_coef = prev
        first = out[float(coefs[0])]
        return first, max(out.values()), out

    def adapt(self):
        self._dyn_anchor = self._build_dyn_anchor()
        print(common_utils.wrap_ruler(
            "stage 3: TARGET dynamics-only adaptation via MPPI rollouts"))
        n_rounds = self.cfg.max_transitions // self.cfg.adapt_interval
        total_transitions = 0

        # round 0: baseline (BC-finetuned policy + stage-1 dynamics) under MPPI
        base_score, base_best, base_all = self._eval_mppi(seed=0)
        self.stat["adapt/transitions"].append(0)
        self.stat["score/score"].append(base_score)
        for _c, _v in base_all.items():
            self.stat[f"score/score_c{_c:g}"].append(_v)
        if base_all:
            self.stat["score/score_best"].append(base_best)
        pol0 = None
        if self.cfg.eval_policy_only:
            pol0 = self.eval(seed=0, use_planner=0)
            self.stat["score/policy_only"].append(pol0)
        self.saver.save(self.agent.state_dict(), base_score, save_latest=True)
        p0s = f"  policy_only={pol0:.3f}" if pol0 is not None else ""
        print(f"[adapt @ 0 transitions] target_score={base_score:.3f}{p0s} (baseline)")
        self.stat.summary(0, reset=True)

        for r in range(1, n_rounds + 1):
            self._collect(self.cfg.adapt_interval)
            total_transitions += self.cfg.adapt_interval
            self._total_transitions = total_transitions
            self._train_dynamics()

            if r % self.cfg.eval_every_rounds != 0 and r != n_rounds:
                self.saver.save(self.agent.state_dict(), None, save_latest=True)
                print(f"[adapt @ {total_transitions} transitions] (eval skipped)")
                self.stat.summary(total_transitions, reset=True)
                continue

            score, score_best, score_all = self._eval_mppi(
                seed=r * self.cfg.num_eval_episode)
            for _c, _v in score_all.items():
                self.stat[f"score/score_c{_c:g}"].append(_v)
            if score_all:
                self.stat["score/score_best"].append(score_best)
            pol = None
            if self.cfg.eval_policy_only:
                pol = self.eval(seed=r * self.cfg.num_eval_episode, use_planner=0)
                self.stat["score/policy_only"].append(pol)
            self.stat["adapt/transitions"].append(total_transitions)
            self.stat["score/score"].append(score)
            self.stat["other/buffer_episodes"].append(self.adapt_replay.size())
            self.stat["other/buffer_transitions"].append(self.adapt_replay.num_transitions())
            saved = self.saver.save(self.agent.state_dict(), score, save_latest=True)
            ps = f"  policy_only={pol:.3f}" if pol is not None else ""
            print(f"[adapt @ {total_transitions} transitions] "
                  f"target_score={score:.3f}{ps}  saved?={saved}")
            self.stat.summary(total_transitions, reset=True)

        if self.cfg.dump_rollouts:
            self._dump_rollouts()

    def _dump_rollouts(self):
        """Export the stage-3 TARGET rollout buffer in the episode format
        `scripts/verify_iql_critic.py` / the residual+IBRL trainers consume.

        The buffer's actions are the MPPI actions of the dynamics model AS IT WAS
        at collection time, i.e. a NON-STATIONARY policy -- they are NOT the final
        base's actions. Consumers that need `base_action` must re-plan it with the
        final base; `action == base_action` only holds for a frozen-base rollout.
        """
        path = os.path.join(self.cfg.save_dir, "rollouts.npz")
        episodes = []
        for ep in self.adapt_replay.episodes:
            episodes.append(dict(
                states=ep["state"].numpy().astype(np.float32),
                actions=ep["action"].numpy().astype(np.float32),
                rewards=ep["reward"].numpy().astype(np.float32),
                success=bool(ep["success"])))
        n_tr = sum(e["actions"].shape[0] for e in episodes)
        n_succ = sum(int(e["success"]) for e in episodes)
        np.savez(path, episodes=np.array(episodes, dtype=object))
        print(f"[dump] {len(episodes)} episodes, {n_tr} transitions, "
              f"{n_succ} success ({n_succ/max(1,len(episodes)):.2%}) -> {path}")


def main():
    cfg = pyrallis.parse(config_class=AdaptConfig)  # type: ignore
    ws = AdaptWorkspace(cfg)
    if cfg.bc_steps > 0:
        ws.bc_finetune()
    else:
        print("[adapt] bc_steps=0 -> SKIPPING stage-2 policy BC (dynamics-only ablation)")
    ws.adapt()
    if cfg.use_wb:
        import wandb
        wandb.finish()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    from rich.traceback import install
    install()
    os.environ.setdefault("MUJOCO_GL", "egl")   # setdefault, not "=": an outer
    # MUJOCO_GL=osmesa must survive. On MIG-partitioned GPUs (Compute Canada fir)
    # `import mujoco` with MUJOCO_GL=egl dies at IMPORT time -- "your EGL driver
    # does not support the PLATFORM_DEVICE extension" -- so those runs must pass
    # MUJOCO_GL=osmesa from outside, together with --state_only_no_render 1 so no
    # GL context is ever actually created.
    torch.backends.cudnn.allow_tf32 = True  # type: ignore
    torch.backends.cudnn.benchmark = True   # type: ignore
    main()

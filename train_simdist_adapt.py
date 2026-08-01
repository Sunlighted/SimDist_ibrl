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
    # stage 3: TARGET dynamics adaptation
    adapt_interval: int = 1000       # collect this many TARGET transitions per round
    max_transitions: int = 20000     # stop after this many collected (=> #rounds)
    dyn_update_steps: int = 1000     # dynamics gradient steps per round (UTD knob = A)
    dyn_lr: float = 3e-4
    # (B) DEMO-INTO-DYNAMICS: mix this fraction of each dynamics-training batch from
    # the TARGET demos (demo_replay) instead of only the MPPI adapt buffer. The demos
    # are the cleanest, best-covered contact-phase (s,a,s') available and are otherwise
    # used ONLY for stage-2 policy BC — feeding them into the dynamics consistency loss
    # fixes the success-path contact dynamics up-front (free, no extra interaction).
    # 0.0 = OFF (dynamics trains on adapt buffer only; legacy). mse path only.
    dyn_demo_frac: float = 0.0
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
    # Override the MPPI planner to the ORIGINAL SimDist recipe (broad search:
    # low policy mixture + large init_std + more iterations) instead of the
    # ibrl-tuned "trust-the-proposal" defaults (mixture 0.8 / init_std 0.15).
    # See /home/Chenyang/simdist/config/control/mppi.yaml.
    original_mppi: int = 1

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
        # optional override of the TARGET eval hand-init sigma (else inherit stage-1)
        if cfg.eval_hand_init_perturb_joint_sigma is not None:
            self.base_cfg.eval_hand_init_perturb_joint_sigma = \
                cfg.eval_hand_init_perturb_joint_sigma
            print(f"[adapt] override eval hand_init_perturb_joint_sigma = "
                  f"{cfg.eval_hand_init_perturb_joint_sigma}")
        self.ws = Workspace(self.base_cfg, from_main=False)
        self.agent = self.ws.agent
        print(f"loading stage-1 ckpt: {cfg.stage1_ckpt}")
        self.agent.load_state_dict(torch.load(cfg.stage1_ckpt))
        self.agent = self.agent.to("cuda")

        self.eval_env = self.ws.eval_env                 # TARGET env (eval_domain_shift)
        self.eval_env_params = self.ws.eval_env_params

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
    def bc_finetune(self):
        print(common_utils.wrap_ruler("stage 2: TARGET policy-only BC (encoder frozen)"))
        self.agent.train()
        pbar = tqdm(total=self.cfg.bc_steps, desc="policy BC", dynamic_ncols=True,
                    mininterval=1.0)
        for step in range(self.cfg.bc_steps):
            states, actions, _ = self.demo_replay.sample(
                self.cfg.bc_batch_size, self.H, "cuda")
            metrics = self.agent.update_bc_policy(states, actions, lr=self.cfg.bc_lr)
            self.stat.append(metrics)
            pbar.update(1)
            if (step + 1) % max(self.cfg.bc_steps // 10, 1) == 0:
                pbar.set_postfix(loss=f"{metrics['bc/policy_loss']:.4f}")
        pbar.close()
        print(f"stage-2 done: final policy BC loss = {metrics['bc/policy_loss']:.5f}")

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
                    self.stat.append(self.agent.update_dynamics_only(
                        states, actions, lr=self.cfg.dyn_lr))
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

    def adapt(self):
        print(common_utils.wrap_ruler(
            "stage 3: TARGET dynamics-only adaptation via MPPI rollouts"))
        n_rounds = self.cfg.max_transitions // self.cfg.adapt_interval
        total_transitions = 0

        # round 0: baseline (BC-finetuned policy + stage-1 dynamics) under MPPI
        base_score = self.eval(seed=0, use_planner=1)
        self.stat["adapt/transitions"].append(0)
        self.stat["score/score"].append(base_score)
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

            score = self.eval(seed=r * self.cfg.num_eval_episode, use_planner=1)
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
    os.environ["MUJOCO_GL"] = "egl"
    torch.backends.cudnn.allow_tf32 = True  # type: ignore
    torch.backends.cudnn.benchmark = True   # type: ignore
    main()

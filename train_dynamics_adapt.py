"""TARGET adaptation for the RAW-STATE dynamics-only SimDist (CLAUDE.md §17).

This is the target-transfer counterpart of `eval_simdist_mpc.py`: instead of only
sanity-checking MPPI-on-source-dynamics, it ADAPTS to the TARGET domain with the
SAME two-phase logic as the latent `train_simdist_adapt.py` (§24), but for the
raw-state / dynamics-only world model (the one trained by `train_dynamics.py`,
where ONLY the dynamics head is meaningful and value/reward/policy are borrowed):

  Stage 2  — TARGET policy BC (the "base policy fine-tuned on the 20 target demos").
             The raw-state world model DOES have a policy MLP head
             (`policy_action(s) = policy(znorm(s))`, znorm is the parameter-free
             z-score "encoder"); we behavior-clone ONLY that head on the 20 TARGET
             demos via `update_bc_policy`. It becomes the MPPI candidate proposal.

  Stage 3  — TARGET dynamics-only adaptation. Roll the MPPI planner in the TARGET
             env, push transitions into a sequence buffer; every `adapt_interval`
             collected transitions, train ONLY `model.dynamics` (multi-step latent
             consistency, frozen z-score encoder) for `dyn_update_steps` steps, then
             EVAL `num_eval_episode` on TARGET and log to wandb, keyed by
             #transitions — identical bookkeeping to `train_simdist_adapt.py`.

MPPI planning sources (the "value/reward/policy all from the originally trained"):
  value    : FROZEN source critic  V_src(s) = min-ensemble Q(s, pi_src(s)).
  policy   : the world model's OWN (stage-2 BC'd) policy head = target-adapted proposal.
  reward   : off (rank by terminal V_src only; the heuristic reward is a proxy, §17).
  dynamics : the ONLY thing that adapts (stage 3).

The planner + source-value closure live in the MAIN process; multi-process eval
only ships env_params to workers (the agent runs in main), so `mp_eval=1` works
even with the external source-value closure.

Run:
  python train_dynamics_adapt.py \
      --dyn_ckpt exps/mimicgen/threading_dyn_diverse_v2_seed1/best.pt \
      --source_agent exps/mimicgen/threading_ibrl_source_seed1/model0.pt \
      --save_dir exps/mimicgen/threading_dyn_adapt_seed1
"""

import json
import os
import sys
import dataclasses
from dataclasses import dataclass

import yaml
from typing import Optional

import numpy as np
import pyrallis
import torch

import common_utils
from common_utils import ibrl_utils as utils
from evaluate import run_eval, run_eval_mp
from env.robosuite_wrapper import PixelRobosuite
from env.domain_shift import DomainShiftConfig
from env.strict_success import StrictSuccessConfig
from rl.world_model import WorldModelConfig
from rl.simdist_planner import PlannerConfig
from rl.simdist_agent import SimDistConfig, SimDistAgent
from rl.sequence_replay import (SequenceReplayBuffer, load_demos_to_sequence_buffer,
                                load_source_replay_to_sequence_buffer)
import train_simdist
import train_rl

# §12 dynamics-only TARGET (matches eval_simdist_mpc.py / the other target evals).
TARGET_SHIFT = dict(
    mode="fixed", friction_scale=0.5, mass_scale=3.0, solref_scale=1.0,
    solimp_scale=1.0, gripper_kp_scale=0.12, hole_offset=[0.0, 0.0, 0.0],
    ring_size_scale=0.85, ring_friction_scale=2.0, mask_hole_offset_in_obs=False,
)


@dataclass
class DynAdaptConfig(common_utils.RunConfig):
    # the raw-state dynamics ckpt (train_dynamics.py) + its sibling dyn_cfg.json
    dyn_ckpt: str = "exps/mimicgen/threading_dyn_diverse_v2_seed1/best.pt"
    source_agent: str = "exps/mimicgen/threading_ibrl_source_seed1/model0.pt"
    save_dir: str = "exps/mimicgen/threading_dyn_adapt_seed1"
    seed: int = 1
    # ---- symmetric-adapt control (2026-07-17, §30): make the raw dynamics adapt
    # on the SAME planning sources the latent arm uses, so the eval_dyn_arm_compare
    # head-to-head isolates REPRESENTATION rather than on-policy-vs-off-policy.
    # When set to a LATENT SimDist adapt ckpt: value_fn + MPPI proposal both come
    # from ITS value + policy heads (replacing source.value and the BC'd raw policy
    # head), and stage-2 policy BC is SKIPPED (the raw model's own policy head is
    # unused for planning). "" (default) = OFF => byte-identical to existing runs.
    proposal_agent: str = ""
    proposal_stage1_cfg: str = ""  # arch for proposal_agent (default: read its cfg.yaml)
    # TARGET demos (stage-2 policy BC)
    target_demo_path: str = "release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5"
    target_num_data: int = -1        # -1 = all (the set is 20)
    # stage 2: TARGET policy BC
    bc_steps: int = 1000
    bc_lr: float = 3e-4
    bc_batch_size: int = 256
    # stage 3: TARGET dynamics-only adaptation
    adapt_interval: int = 1000       # collect this many TARGET transitions per round
    max_transitions: int = 40000     # stop after this many collected (=> #rounds)
    dyn_update_steps: int = 1000     # dynamics gradient steps per round
    dyn_lr: float = 3e-4
    adapt_batch_size: int = 256
    adapt_buffer_max_episode: int = 100000
    # ---- anti-forgetting: mix the ORIGINAL source data into the adapt batches ----
    # Diagnosed 2026-07-17 (scripts/diag_raw_dyn_adapt.py): the vanilla adapt does
    # 20 rounds x 1000 steps x batch 256 = 5.1M samples drawn from a ~20k-transition
    # on-policy buffer (=256 epochs), against a source model pretrained on
    # diverse_v2's 100k trajectories. Held-out TARGET prediction DEGRADES 2-7x
    # (free-space needle MAE 1.69 -> 8.4mm) — catastrophic forgetting of the broad
    # prior. It still raised the MPPI score (0.538 -> 0.708, the model specialises
    # to its own on-policy distribution, which is what MPPI scores on), but the raw
    # planner never beat the bare proposal (-0.02 at best) while the latent planner
    # does (+0.097). This tests whether retaining the broad prior lets it.
    # source_mix_ratio = fraction of each dynamics batch drawn from the SOURCE set.
    # 0.0 (default) = OFF => byte-identical to every existing run.
    source_mix_ratio: float = 0.0
    source_replay_path: str = "release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5"
    source_replay_num_data: int = 20000   # trajectories to load (-1 = all 100k = 4.1GB)
    source_buffer_max_episode: int = 25000
    # MPPI planner (VALIDATED on this dynamics: eval_simdist_mpc.py got 0.96@source
    # with these; the latent "original SimDist recipe" was tuned for the latent WM,
    # so we use the raw-state-validated params instead).
    num_samples: int = 512
    num_elites: int = 64
    iterations: int = 6
    mixture_coef: float = 0.8
    temperature: float = 2.0
    init_std: float = 0.15
    use_reward: int = 0              # 0 = rank by terminal V_src only (cleanest)
    reward_scale: float = 0.3
    strict_success: int = 1
    # eval
    num_eval_episode: int = 500
    mp_eval: int = 1
    num_eval_proc: int = 10
    eval_policy_only: int = 1        # also log the BC'd policy head's raw success
    eval_hand_init_perturb_joint_sigma: float = 0.1  # match §24 / other final evals
    use_wb: int = 1


class DynAdaptWorkspace:
    def __init__(self, cfg: DynAdaptConfig):
        self.cfg = cfg
        common_utils.set_all_seeds(cfg.seed)
        os.makedirs(cfg.save_dir, exist_ok=True)
        sys.stdout = common_utils.Logger(cfg.log_path, print_to_stdout=True)
        dev = "cuda"

        # ---- dynamics cfg (rebuilds the raw-state world model) ----
        dcfg = json.load(open(os.path.join(os.path.dirname(cfg.dyn_ckpt), "dyn_cfg.json")))
        self.H = dcfg["horizon"]

        # ---- TARGET env (eval + rollout) ----
        self.eval_env_params = dict(
            env_name="Threading_D0", robots=["Panda"], episode_length=400,
            reward_shaping=False, image_size=224, rl_image_size=96,
            camera_names=[], rl_cameras=[], use_state=True,
            obs_stack=1, state_stack=1, prop_stack=1,
            strict_success=StrictSuccessConfig(
                enabled=bool(cfg.strict_success), dwell_steps=3, max_tripod_disp=0.04),
            domain_shift=DomainShiftConfig(**TARGET_SHIFT),
            hand_init_perturb_joint_sigma=(
                cfg.eval_hand_init_perturb_joint_sigma
                if cfg.eval_hand_init_perturb_joint_sigma > 0 else None),
        )
        self.eval_env = PixelRobosuite(**self.eval_env_params)  # type: ignore

        # ---- raw-state world model (only dynamics is meaningful) ----
        sd_cfg = SimDistConfig(
            world_model=WorldModelConfig(
                encoder_type="identity", mlp_hidden_dim=dcfg["hidden_dim"],
                num_layer=dcfg["num_layer"], layer_norm=dcfg["layer_norm"],
                horizon=self.H, activation=dcfg.get("activation", "relu")),
            planner=PlannerConfig(
                horizon=self.H, num_samples=cfg.num_samples, num_elites=cfg.num_elites,
                iterations=cfg.iterations, mixture_coef=cfg.mixture_coef,
                temperature=cfg.temperature, init_std=cfg.init_std,
                use_reward=bool(cfg.use_reward), discount=0.99),
            use_planner=1,
        )
        self.agent = SimDistAgent(
            True, self.eval_env.observation_shape, self.eval_env.prop_shape,
            self.eval_env.action_dim, "robot0_eye_in_hand", sd_cfg, device=dev)
        self.agent.model.load_state_dict(torch.load(cfg.dyn_ckpt, map_location=dev))
        self.agent = self.agent.to(dev)
        print(f"loaded dynamics ckpt: {cfg.dyn_ckpt} (H={self.H})")

        # ---- frozen source agent: value (+ NOT proposal: proposal = BC'd head) ----
        sim_cfg = train_simdist.MainConfig()  # use_state=1 / state_stack=1 loader cfg
        self.source = train_rl.load_frozen_agent(cfg.source_agent, self.eval_env, sim_cfg, dev)

        @torch.no_grad()
        def reward_fn(state, action):              # [N,obs],[N,act] -> [N]
            # BODY needle/ring positions (state[9:12] / [23:26]); planning proxy only.
            dist = (state[:, 9:12] - state[:, 23:26]).norm(dim=-1)
            return -cfg.reward_scale * dist

        # ---- planning sources ----
        self._proposal_agent = None
        if cfg.proposal_agent:
            # SYMMETRIC adapt: value + proposal from a frozen LATENT SimDist agent,
            # so the raw dynamics adapts on the same MPPI-source distribution the
            # latent arm plans in (isolates representation in the head-to-head).
            self._proposal_agent = self._load_latent_agent(dev)
            lat = self._proposal_agent.model

            @torch.no_grad()
            def value_fn(state):                   # SHARED latent value head
                return lat.predict_value(lat.encode(state))

            @torch.no_grad()
            def proposal_policy(state):            # SHARED latent policy head
                return lat.policy_action(state)

            self.agent.set_planning_sources(
                value_fn=value_fn,
                reward_fn=reward_fn if cfg.use_reward else None,
                proposal_policy=proposal_policy,
            )
            print(f"[symmetric] value + proposal from LATENT agent {cfg.proposal_agent} "
                  f"(stage-2 BC skipped)")
        else:
            @torch.no_grad()
            def value_fn(state):                   # [N,obs] -> [N]
                return self.source.value({"state": state})

            # value from source; proposal LEFT as the model's own (BC'd) policy head.
            self.agent.set_planning_sources(
                value_fn=value_fn,
                reward_fn=reward_fn if cfg.use_reward else None,
                proposal_policy=None,
            )

        # ---- TARGET demos (stage-2 policy BC) ----
        self.demo_replay = SequenceReplayBuffer(max_episodes=10000, device="cpu")
        load_demos_to_sequence_buffer(
            self.demo_replay, cfg.target_demo_path, num_data=cfg.target_num_data)
        assert self.demo_replay.can_sample(self.H), "no TARGET demo >= horizon"

        # ---- TARGET rollout buffer (stage-3 dynamics adaptation) ----
        self.adapt_replay = SequenceReplayBuffer(
            max_episodes=cfg.adapt_buffer_max_episode, device="cpu")
        self._obs = None  # persistent rollout episode state

        # ---- SOURCE replay (anti-forgetting mix; off unless source_mix_ratio>0) ----
        self.source_replay = None
        if cfg.source_mix_ratio > 0:
            assert 0 < cfg.source_mix_ratio < 1, "source_mix_ratio must be in (0,1)"
            self.source_replay = SequenceReplayBuffer(
                max_episodes=cfg.source_buffer_max_episode, device="cpu")
            load_source_replay_to_sequence_buffer(
                self.source_replay, cfg.source_replay_path,
                num_data=cfg.source_replay_num_data)
            assert self.source_replay.can_sample(self.H), "no SOURCE traj >= horizon"
            print(f"[anti-forget] source replay: {self.source_replay.size()} episodes, "
                  f"{self.source_replay.num_transitions()} transitions; "
                  f"mix_ratio={cfg.source_mix_ratio}")

        pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # type: ignore
        self.stat = common_utils.MultiCounter(
            cfg.save_dir, bool(cfg.use_wb), wb_exp_name=cfg.wb_exp,
            wb_run_name=cfg.wb_run, wb_group_name=cfg.wb_group, config={**cfg.__dict__})
        self.saver = common_utils.TopkSaver(save_dir=cfg.save_dir, topk=1)

    # ------------------------------------------------ symmetric proposal agent
    def _load_latent_agent(self, dev):
        """Load a frozen LATENT SimDist agent whose value + policy heads drive the
        raw dynamics' MPPI collection (symmetric-adapt control)."""
        cfg = self.cfg
        stage1 = cfg.proposal_stage1_cfg
        if not stage1:
            adapt_cfg = yaml.safe_load(
                open(os.path.join(os.path.dirname(cfg.proposal_agent), "cfg.yaml")))
            stage1 = os.path.join(os.path.dirname(adapt_cfg["stage1_ckpt"]), "cfg.yaml")
        wm = yaml.safe_load(open(stage1))["simdist"]["world_model"]
        fields = {f.name for f in dataclasses.fields(WorldModelConfig)}
        wm_cfg = WorldModelConfig(**{k: v for k, v in wm.items() if k in fields})
        assert wm_cfg.encoder_type == "latent", "proposal_agent must be a latent arm"
        lat = SimDistAgent(
            True, self.eval_env.observation_shape, self.eval_env.prop_shape,
            self.eval_env.action_dim, "robot0_eye_in_hand",
            SimDistConfig(world_model=wm_cfg,
                          planner=PlannerConfig(horizon=wm_cfg.horizon), use_planner=0),
            device=dev)
        sd = torch.load(cfg.proposal_agent, map_location=dev, weights_only=False)
        miss, unexp = lat.load_state_dict(sd, strict=False)
        lat.eval()
        for p in lat.parameters():
            p.requires_grad_(False)
        assert wm_cfg.horizon == self.H, (
            f"proposal latent H={wm_cfg.horizon} != raw dyn H={self.H}")
        print(f"[symmetric] loaded latent proposal agent {cfg.proposal_agent} "
              f"(missing={len(miss)} unexpected={len(unexp)})")
        return lat

    # ----------------------------------------------------------------- eval
    def eval(self, seed: int, use_planner: int) -> float:
        prev = self.agent.cfg.use_planner
        self.agent.cfg.use_planner = use_planner
        rng = np.random.get_state()
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
            np.random.set_state(rng)
            self.agent.cfg.use_planner = prev
        return float(np.mean(scores))

    # -------------------------------------------------- stage 2: policy BC
    def bc_finetune(self):
        print(common_utils.wrap_ruler("stage 2: TARGET policy-only BC (znorm encoder frozen)"))
        self.agent.train()
        from tqdm import tqdm
        pbar = tqdm(total=self.cfg.bc_steps, desc="policy BC", dynamic_ncols=True, mininterval=1.0)
        metrics = {}
        for step in range(self.cfg.bc_steps):
            states, actions, _ = self.demo_replay.sample(self.cfg.bc_batch_size, self.H, "cuda")
            metrics = self.agent.update_bc_policy(states, actions, lr=self.cfg.bc_lr)
            self.stat.append(metrics)
            pbar.update(1)
            if (step + 1) % max(self.cfg.bc_steps // 10, 1) == 0:
                pbar.set_postfix(loss=f"{metrics['bc/policy_loss']:.4f}")
        pbar.close()
        print(f"stage-2 done: final policy BC loss = {metrics.get('bc/policy_loss', float('nan')):.5f}")

    # ------------------------------------ stage 3: TARGET dynamics adaptation
    def _collect(self, n_transitions: int):
        if self._obs is None:
            self._obs, _ = self.eval_env.reset()
            self.adapt_replay.new_episode(self._obs["state"])
        collected = 0
        from tqdm import tqdm
        pbar = tqdm(total=n_transitions, desc="collect (target MPPI)",
                    dynamic_ncols=True, mininterval=1.0, leave=False)
        with utils.eval_mode(self.agent):
            while collected < n_transitions:
                action = self.agent.act(self._obs, eval_mode=True, stddev=0.0, cpu=True)
                self._obs, reward, terminal, success, _ = self.eval_env.step(action)
                self.adapt_replay.add(self._obs["state"], action, reward, terminal, success)
                collected += 1
                pbar.update(1)
                if terminal:
                    self._obs, _ = self.eval_env.reset()
                    self.adapt_replay.new_episode(self._obs["state"])
        pbar.close()

    def _train_dynamics(self):
        if not self.adapt_replay.can_sample(self.H):
            print(f"[adapt] buffer has no episode >= horizon={self.H} yet; skipping")
            return
        self.agent.train()
        B = self.cfg.adapt_batch_size
        n_src = int(round(B * self.cfg.source_mix_ratio)) if self.source_replay else 0
        n_tgt = B - n_src
        for _ in range(self.cfg.dyn_update_steps):
            states, actions, _ = self.adapt_replay.sample(n_tgt, self.H, "cuda")
            if n_src > 0:
                # anti-forgetting: keep the broad SOURCE prior alive in every batch.
                # The dynamics gap means source segments pull toward the WRONG
                # transition — but only on the shared (s,a) support, and the point is
                # to retain generic coverage (free-space motion etc.) that 256 epochs
                # over a narrow on-policy buffer otherwise destroys.
                s_src, a_src, _ = self.source_replay.sample(n_src, self.H, "cuda")
                states = torch.cat([states, s_src], dim=0)
                actions = torch.cat([actions, a_src], dim=0)
            self.stat.append(self.agent.update_dynamics_only(states, actions, lr=self.cfg.dyn_lr))

    def adapt(self):
        print(common_utils.wrap_ruler(
            "stage 3: TARGET dynamics-only adaptation via MPPI rollouts"))
        n_rounds = self.cfg.max_transitions // self.cfg.adapt_interval
        total = 0

        # round 0: baseline (BC-finetuned policy + source dynamics) under MPPI
        base_score = self.eval(seed=0, use_planner=1)
        self.stat["adapt/transitions"].append(0)
        self.stat["score/score"].append(base_score)
        pol0 = None
        if self.cfg.eval_policy_only:
            pol0 = self.eval(seed=0, use_planner=0)
            self.stat["score/policy_only"].append(pol0)
        self.saver.save(self.agent.state_dict(), base_score, save_latest=True)
        p0 = f"  policy_only={pol0:.3f}" if pol0 is not None else ""
        print(f"[adapt @ 0 transitions] target_score={base_score:.3f}{p0} (baseline)")
        self.stat.summary(0, reset=True)

        for r in range(1, n_rounds + 1):
            self._collect(self.cfg.adapt_interval)
            total += self.cfg.adapt_interval
            self._train_dynamics()
            score = self.eval(seed=r * self.cfg.num_eval_episode, use_planner=1)
            pol = None
            if self.cfg.eval_policy_only:
                pol = self.eval(seed=r * self.cfg.num_eval_episode, use_planner=0)
                self.stat["score/policy_only"].append(pol)
            self.stat["adapt/transitions"].append(total)
            self.stat["score/score"].append(score)
            self.stat["other/buffer_episodes"].append(self.adapt_replay.size())
            self.stat["other/buffer_transitions"].append(self.adapt_replay.num_transitions())
            saved = self.saver.save(self.agent.state_dict(), score, save_latest=True)
            ps = f"  policy_only={pol:.3f}" if pol is not None else ""
            print(f"[adapt @ {total} transitions] target_score={score:.3f}{ps}  saved?={saved}")
            self.stat.summary(total, reset=True)


def main():
    cfg = pyrallis.parse(config_class=DynAdaptConfig)  # type: ignore
    ws = DynAdaptWorkspace(cfg)
    if cfg.proposal_agent:
        print("[symmetric] proposal from external latent agent -> "
              "SKIPPING stage-2 policy BC (raw policy head unused for planning)")
    elif cfg.bc_steps > 0:
        ws.bc_finetune()
    else:
        print("[adapt] bc_steps=0 -> SKIPPING stage-2 policy BC")
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

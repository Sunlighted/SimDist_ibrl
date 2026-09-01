"""Stage-3 alternative: DYNA instead of planning (the planning-free baseline).

`train_simdist_adapt.py` deploys the world model as an ONLINE planner (MPPI
re-plans from the true state every step). This script keeps everything else the
same but uses the model OFFLINE -- to manufacture training data for the policy,
which then acts on its own. Same stage-1 world model, same stage-2 BC policy head,
same distilled reward/value, same real-data budget, same dynamics schedule.

Per round (1000 collected transitions, matching the planning arms):
    1. collect  -- roll the POLICY (no planner) in the TARGET env
    2. dynamics -- same multi-step latent-consistency update as the planning arms
    3. imagine  -- branch `rollout_length` steps from real states
    4. learn    -- TD on Q(z,a) + DPG on the policy, over a real/synthetic mix

Usage (matches the 3-arm sweep's conventions; wandb project/run/group come from
`--save_dir`, see common_utils/py/run_cfg.py):

    python train_simdist_dyna.py \
        --stage1_ckpt exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1/latest.pt \
        --target_demo_path release/data/mimicgen/threading_d0_sat_400000_s1/dataset.hdf5 \
        --save_dir exps/simdist_threading_3arm/dyna_h5_seed1 --seed 1 \
        --max_transitions 100000 --eval_every_rounds 5 --num_eval_episode 100 \
        --eval_hand_init_perturb_joint_sigma 0.1
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyrallis
import torch
from tqdm import tqdm

import common_utils
from common_utils import ibrl_utils as utils
from rl.simdist_dyna import DynaConfig, DynaTrainer, FlatBuffer
from train_simdist_adapt import AdaptConfig, AdaptWorkspace


@dataclass
class DynaMainConfig(AdaptConfig):
    save_dir: str = "exps/simdist_threading_3arm/dyna_h5_seed1"
    dyna: DynaConfig = field(default_factory=DynaConfig)


class DynaWorkspace(AdaptWorkspace):
    """AdaptWorkspace with the planner swapped out for imagination + actor-critic.

    Reuses the parent's env/agent construction, stage-2 BC, dynamics update, eval
    and checkpointing verbatim; only collection (policy instead of MPPI) and the
    per-round learning step differ.
    """

    def __init__(self, cfg: DynaMainConfig):
        super().__init__(cfg)
        self.cfg: DynaMainConfig = cfg
        # The policy ACTS -- no planning anywhere in this arm (the parent forces
        # use_planner=1 for the adapt arms).
        self.agent.cfg.use_planner = 0

        obs_dim = self.eval_env.observation_shape[0]
        action_dim = self.eval_env.action_dim
        d = cfg.dyna
        # Both buffers hold LATENTS now (see rl/simdist_dyna.FlatBuffer): the
        # encoder is frozen for all of stage 3, so a real state encoded once at
        # insert stays valid, and imagined transitions never leave latent space.
        with torch.no_grad():
            z_dim = self.agent.model.encode(
                torch.zeros(1, obs_dim, device="cuda")).shape[-1]
        # Sized to the run's own budget: a latent row is z_dim*2 floats, so the
        # old hard-coded 1M would be ~4 GB at z=512. Collection adds exactly
        # max_transitions rows over the run.
        real_cap = max(cfg.max_transitions + cfg.adapt_interval, 10_000)
        self.real_buf = FlatBuffer(real_cap, z_dim, action_dim, "cuda")
        self.synth_buf = FlatBuffer(d.synth_buffer_size, z_dim, action_dim, "cuda")
        self.dyna = DynaTrainer(self.agent, d, obs_dim, action_dim, "cuda")

    # ---------------------------------------------------------------- eval
    def _eval_mppi(self, seed: int):
        """Score the POLICY. Dyna has no planner -- there is no MPPI eval here.

        The name is the parent's; `adapt()` calls it at every eval point. The
        parent's version plans (MPPI is the deliverable for those arms) and its
        no-coef branch hard-codes `use_planner=1`, which OVERWRITES the flag this
        workspace sets in `__init__` for the duration of the eval and restores it
        after. So `use_planner=0` only ever governed `_collect`, and every
        `score/score` this arm logged before 2026-08-31 was the score of "current
        policy used as an MPPI proposal" -- a quantity Dyna neither claims nor
        deploys. Measured gap on the fixed smoke run: policy 0.25 / 0.35 against
        MPPI 0.65 / 0.75 at the same two eval points.

        Corroboration that the old numbers were the planner's: `dyna_h5` and
        `dynens_h5` logged the SAME round-0 (0.274), possible only if both were
        "same stage-2 policy + same stage-1 dynamics + same MPPI"; and the
        `policy_lr=0` cell, whose policy is frozen bit-for-bit, still climbed
        0.23 -> 0.4+, which only the adapting dynamics head can explain.
        """
        pol = self.eval(seed=seed, use_planner=0)
        return pol, pol, {}

    # ------------------------------------------------------------- collect
    def _collect(self, n_transitions: int):
        """Roll the POLICY (deterministic) in the TARGET env.

        Overrides the parent's MPPI collection. Transitions go to BOTH buffers:
        the sequence buffer feeds the dynamics update (it needs contiguous
        H-step segments) and the flat buffer feeds the actor-critic.
        `done = success` -- Threading ends the episode on success, so a timeout
        is NOT an absorbing terminal and must keep bootstrapping.
        """
        if self._obs is None:
            self._obs, _ = self.eval_env.reset()
            self.adapt_replay.new_episode(self._obs["state"])
        collected = 0
        pbar = tqdm(total=n_transitions, desc="collect (target policy)",
                    dynamic_ncols=True, mininterval=1.0, leave=False)
        S, A, R, S2, D = [], [], [], [], []
        with utils.eval_mode(self.agent):
            while collected < n_transitions:
                s_prev = self._obs["state"].clone()
                # eval_mode=False + stddev>0 => the policy head adds Gaussian noise
                # and clamps to [-1,1] (rl/simdist_agent.py act()). Without it every
                # real transition sits at exactly a=pi(s) and the critic has no
                # action coverage to shape its gradient (see DynaConfig.collect_std).
                _std = self.cfg.dyna.collect_std
                action = self.agent.act(self._obs, eval_mode=(_std <= 0),
                                        stddev=_std, cpu=True)
                self._obs, reward, terminal, success, _ = self.eval_env.step(action)
                self.adapt_replay.add(self._obs["state"], action, reward, terminal, success)
                S.append(s_prev); A.append(action.float())
                R.append(float(reward)); S2.append(self._obs["state"].clone())
                D.append(1.0 if success else 0.0)
                collected += 1
                pbar.update(1)
                if terminal:
                    self._obs, _ = self.eval_env.reset()
                    self.adapt_replay.new_episode(self._obs["state"])
        pbar.close()
        dev = "cuda"
        with torch.no_grad():
            z = self.agent.model.encode(torch.stack(S).to(dev).float())
            z2 = self.agent.model.encode(torch.stack(S2).to(dev).float())
        self.real_buf.add_batch(
            z, torch.stack(A).to(dev).float(),
            torch.tensor(R, device=dev).float(), z2,
            torch.tensor(D, device=dev).float())

    # --------------------------------------------------------------- learn
    def _train_actor_critic(self):
        if len(self.real_buf) < self.cfg.dyna.batch_size:
            return
        self.agent.train()
        d = self.cfg.dyna
        z0, _, _, _, _ = self.real_buf.sample(d.rollout_batch)
        with torch.no_grad():
            self.dyna.imagine(z0, self.synth_buf)
        H = self.H
        for _ in range(d.q_updates):
            demo = None
            if d.bc_coef > 0:
                st, ac, _ = self.demo_replay.sample(d.batch_size, H, "cuda")
                demo = (st[:, 0], ac[:, 0])
            self.stat.append(self.dyna.update(self.real_buf, self.synth_buf, demo))
        self.stat["dyna/synth_buffer"].append(len(self.synth_buf))

    def _train_dynamics(self):
        """Same dynamics update as the planning arms, then the actor-critic step."""
        super()._train_dynamics()
        self._train_actor_critic()

    # ---------------------------------------------------------------- init
    def dyna_init(self):
        """Initialise Q to the source value's implied Q, on stage-2 demo states."""
        H = self.H

        def sample_states(n):
            st, ac, _ = self.demo_replay.sample(n, H, "cuda")
            return st[:, 0], ac[:, 0]

        self.dyna.init_from_source(sample_states)


def main():
    cfg = pyrallis.parse(config_class=DynaMainConfig)  # type: ignore
    ws = DynaWorkspace(cfg)
    if cfg.bc_steps > 0:
        ws.bc_finetune()
    else:
        print("[dyna] bc_steps=0 -> SKIPPING stage-2 policy BC")
    ws.dyna_init()
    ws.adapt()
    if cfg.use_wb:
        import wandb
        wandb.finish()
    os._exit(0)


if __name__ == "__main__":
    main()

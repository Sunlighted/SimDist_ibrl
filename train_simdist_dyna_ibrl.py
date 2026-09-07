"""Stage-3 alternative: IBRL in the world model's latent space, fed by Dyna.

Positioning against the two arms it borrows from:

    train_simdist_adapt.py   the model is a PLANNER (MPPI, online, re-planned from
                             the true state every step). Deployment needs the model.
    train_simdist_dyna.py    the model is a DATA GENERATOR for a policy that then
                             acts alone -- and DPG trains the very head that acts,
                             so a corrupted actor IS the deployed policy.
    this file                the model is a data generator, but the acting rule is
                             IBRL: argmax_a Q(z,a) over {RL actor, FROZEN prior}.
                             A corrupted actor stops winning the argmax and the
                             prior carries the episode. Deployment needs no planner.

Why this combination and not either alone (both numbers measured on this target):
  * model-free IBRL gets to 0.98 but needs ~1M transitions; at the matched 100k
    budget it is 0.082 and flat 0 for its first 60k. Right learning rule, no data.
  * plain Dyna at 100k destroys its own policy, because nothing floors the DPG
    step. Data, wrong deployment rule.
  * and the data really is unreliable: one-step latent error on 3000 TARGET
    on-policy transitions is 2.4x the one-step motion it has to explain, with a
    sup of 65-87 (scripts/diag_horizon_onestep.py). Flat across H=3/5/7/10, so it
    is not a distillation-horizon artefact and an H=1 model would not fix it. The
    prior floor is therefore not optional -- it is the reason this can work at all.

Per round (1000 collected transitions, matching every other stage-3 arm):
    1. collect   -- roll the IBRL hybrid in the TARGET env (no planner)
    2. dynamics  -- the same multi-step latent-consistency update as the planning arms
    3. imagine   -- branch `rollout_length` latent steps under the hybrid rule
    4. learn     -- TD on Q(z,a) + DPG on the RL actor, over a real/synthetic mix

Usage:
    python train_simdist_dyna_ibrl.py \
        --stage1_ckpt exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1/latest.pt \
        --target_demo_path release/data/mimicgen/threading_d0_sat_160000_s1/dataset.hdf5 \
        --target_num_data 20 --save_dir exps/simdist_threading_3arm/dynaibrl_seed1 \
        --seed 1 --max_transitions 100000 --eval_every_rounds 5 \
        --num_eval_episode 100 --eval_hand_init_perturb_joint_sigma 0.1
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from dataclasses import dataclass, field

import numpy as np
import pyrallis
import torch
from tqdm import tqdm

import common_utils
from common_utils import ibrl_utils as utils
from rl.simdist_dyna import FlatBuffer
from rl.simdist_dyna_ibrl import DynaIBRLConfig, DynaIBRLTrainer
from train_simdist_adapt import AdaptConfig, AdaptWorkspace


@dataclass
class DynaIBRLMainConfig(AdaptConfig):
    save_dir: str = "exps/simdist_threading_3arm/dynaibrl_seed1"
    dyna: DynaIBRLConfig = field(default_factory=DynaIBRLConfig)
    actor_init_steps: int = 500      # regression of the fresh actor onto the prior


class DynaIBRLWorkspace(AdaptWorkspace):
    """AdaptWorkspace with the planner replaced by imagination + latent IBRL."""

    def __init__(self, cfg: DynaIBRLMainConfig):
        super().__init__(cfg)
        self.cfg: DynaIBRLMainConfig = cfg
        self.agent.cfg.use_planner = 0          # no planning anywhere in this arm

        obs_dim = self.eval_env.observation_shape[0]
        action_dim = self.eval_env.action_dim
        d = cfg.dyna
        with torch.no_grad():
            z_dim = self.agent.model.encode(
                torch.zeros(1, obs_dim, device="cuda")).shape[-1]
        # Both buffers hold LATENTS: the encoder is frozen for all of stage 3, so a
        # real state encoded once at insert stays valid, and imagined states never
        # leave latent space (no decode round trip).
        real_cap = max(cfg.max_transitions + cfg.adapt_interval, 10_000)
        self.real_buf = FlatBuffer(real_cap, z_dim, action_dim, "cuda")
        self.synth_buf = FlatBuffer(d.synth_buffer_size, z_dim, action_dim, "cuda")
        # Demo-only view of the preloaded target demos, for `dyna.demo_mix_rate`.
        # Allocated lazily in _preload_demos (its size is #demo transitions, not
        # known here). None => the knob is inert, which is also what happens when
        # preload_demos=0.
        self.demo_buf = None
        self.dyna = DynaIBRLTrainer(self.agent, d, obs_dim, action_dim, "cuda")
        # Make the trained heads part of the agent so `saver.save(agent.state_dict())`
        # captures them. Without this the checkpoint holds only the world model --
        # the FROZEN prior and the adapted dynamics -- while the RL actor and critic,
        # the only things stage 3 actually trains, are silently dropped. That is what
        # made the first margin sweep un-analysable after the fact: its 0.775 could
        # not be attributed to the actor because the actor no longer existed.
        # nn.Module attribute assignment registers them as submodules, so they land
        # under `dyna_actor.` / `dyna_q.`. Loading a PRE-fix checkpoint into a
        # post-fix agent therefore needs strict=False.
        self.agent.dyna_actor = self.dyna.actor
        self.agent.dyna_q = self.dyna.q

    # ---------------------------------------------------------------- act
    @torch.no_grad()
    def _hybrid_action(self, state: torch.Tensor, *, eval_mode: bool, stddev: float):
        """IBRL action for a single obs or a [B, obs] stack."""
        single = state.dim() == 1
        s = state.view(1, -1) if single else state
        z = self.agent.model.encode(s.to("cuda").float())
        a, frac = self.dyna.act_latent(z, eval_mode=eval_mode, stddev=stddev,
                                       channel="eval" if eval_mode else "act")
        return (a[0] if single else a), float(frac)

    # --------------------------------------------------------------- eval
    def _eval_mppi(self, seed: int):
        """Score the IBRL hybrid. There is no planner in this arm, so no MPPI eval.

        The parent's version hard-codes `use_planner=1`, which would silently
        report the planner's score instead -- the defect that made every
        pre-2026-08-31 `dyna_*` number a planner score.
        """
        v = self._eval_hybrid(seed=seed)
        return v, v, {}

    def _eval_hybrid(self, seed: int) -> float:
        """mp eval with the hybrid rule patched into the agent's `act`."""
        agent = self.agent
        orig_act = agent.act

        def act(obs, *, eval_mode=False, stddev=0.0, cpu=True):
            a, _ = self._hybrid_action(obs["state"], eval_mode=eval_mode, stddev=stddev)
            return a.cpu() if cpu else a

        agent.act = act                              # type: ignore[method-assign]
        try:
            return self.eval(seed=seed, use_planner=0)
        finally:
            agent.act = orig_act                     # type: ignore[method-assign]

    # ------------------------------------------------------------- collect
    def _collect(self, n_transitions: int):
        """Roll the IBRL hybrid in the TARGET env.

        Exploration noise is on by default (`dyna.collect_std`): with a
        deterministic collector every real transition sits at exactly one action
        per state, leaving the critic's action-dimension shape to free
        extrapolation -- the direction DPG ascends.
        """
        self._update_margin()
        if self._obs is None:
            self._obs, _ = self.eval_env.reset()
            self.adapt_replay.new_episode(self._obs["state"])
        std = self.cfg.dyna.collect_std
        collected = 0
        pbar = tqdm(total=n_transitions, desc="collect (target IBRL hybrid)",
                    dynamic_ncols=True, mininterval=1.0, leave=False)
        S, A, R, S2, D, F = [], [], [], [], [], []
        with utils.eval_mode(self.agent):
            while collected < n_transitions:
                s_prev = self._obs["state"].clone()
                action, frac = self._hybrid_action(
                    s_prev, eval_mode=(std <= 0), stddev=std)
                action = action.cpu()
                self._obs, reward, terminal, success, _ = self.eval_env.step(action)
                self.adapt_replay.add(self._obs["state"], action, reward, terminal, success)
                S.append(s_prev); A.append(action.float())
                R.append(float(reward)); S2.append(self._obs["state"].clone())
                D.append(1.0 if success else 0.0); F.append(frac)
                collected += 1
                pbar.update(1)
                if terminal:
                    self._obs, _ = self.eval_env.reset()
                    self.adapt_replay.new_episode(self._obs["state"])
        pbar.close()
        with torch.no_grad():
            z = self.agent.model.encode(torch.stack(S).cuda().float())
            z2 = self.agent.model.encode(torch.stack(S2).cuda().float())
        self.real_buf.add_batch(
            z, torch.stack(A).cuda().float(),
            torch.tensor(R, device="cuda").float(), z2,
            torch.tensor(D, device="cuda").float())
        # How often the RL actor beats the prior while ACTING. 0 => the arm is
        # running the stage-2 policy and nothing has been learned yet; a rise
        # without a rise in score means the critic is overrating the actor.
        self.stat["dyna/collect_rl_frac"].append(float(np.mean(F)))

    def _update_margin(self):
        """Apply the linear margin schedule, if one is configured.

        Called at the top of every round so collection, imagination and the IBRL
        bootstrap all see the same margin within a round.
        """
        d = self.cfg.dyna
        if d.prior_margin_init < 0:
            return
        frac = max(0.0, 1.0 - self._total_transitions / max(d.prior_margin_steps, 1))
        d.prior_margin = max(d.prior_margin_final, d.prior_margin_init * frac)
        self.stat["dyna/prior_margin"].append(d.prior_margin)

    # --------------------------------------------------------------- learn
    def _train_actor_critic(self):
        if len(self.real_buf) < self.cfg.dyna.batch_size:
            return
        self.agent.train()
        d = self.cfg.dyna
        if d.real_ratio < 1.0:
            # At real_ratio=1.0 `update` draws n_synth=0, so imagining would be
            # pure waste (rollout_batch model forward passes per round). Skipping
            # it makes the model-off ablation cheap as well as clean.
            z0, _, _, _, _ = self.real_buf.sample(d.rollout_batch)
            with torch.no_grad():
                self.dyna.imagine(z0, self.synth_buf)
        # Keep actor steps pinned to the env-step budget, not to the UTD: raising
        # q_updates must buy the CRITIC more data, not multiply the DPG steps.
        if d.actor_updates_per_round > 0:
            d.policy_update_freq = max(1, d.q_updates // d.actor_updates_per_round)
        for _ in range(d.q_updates):
            self.stat.append(
                self.dyna.update(self.real_buf, self.synth_buf, self.demo_buf))
        self.stat["dyna/synth_buffer"].append(len(self.synth_buf))

    def _train_dynamics(self):
        """Same dynamics update as the planning arms, then the actor-critic step."""
        super()._train_dynamics()
        self._train_actor_critic()

    # ---------------------------------------------------------------- init
    def dyna_init(self):
        """Freeze the prior, seed the actor from it, warm-start Q on the demos."""
        H = self.H

        def sample_za(n):
            st, ac, _ = self.demo_replay.sample(n, H, "cuda")
            with torch.no_grad():
                return self.agent.model.encode(st[:, 0]), ac[:, 0]

        def sample_z(n):
            return sample_za(n)[0]

        self.dyna.freeze_prior()
        self.dyna.init_actor_from_prior(sample_z, steps=self.cfg.actor_init_steps)
        self.dyna.init_q_from_source(sample_za)
        # Set the schedule's t=0 value BEFORE adapt() runs. The parent evaluates
        # round 0 before the first `_collect`, so hooking the anneal into _collect
        # alone left that eval running at the raw `prior_margin` default of 0.0 --
        # i.e. at the one setting measured to destroy the policy outright. It cost
        # dynaibrl_m005anneal_seed1 its floor reading (round-0 came out 0.12, the
        # margin-0 value, instead of the ~0.6 its schedule should have given).
        self._update_margin()
        if self.cfg.dyna.preload_demos:
            self._preload_demos()

    def _preload_demos(self):
        """Put the 20 target demos into the TD replay, as encoded latents.

        The model-free IBRL baseline preloads exactly these (preload_num_data 20)
        and samples them at mix_rl_rate 0.5, which is how its critic keeps seeing
        r=1 through the 60k in which its own rollouts return nothing. Our critic
        had no such channel: `real_buf` was fed only by `_collect`, so a policy
        that dips to zero takes the reward signal with it.

        Threading is sparse 0/1 with end_on_success, so `done` is the reward: the
        terminal transition of a successful demo is the only one with r=1, and a
        truncated (unsuccessful) episode must keep bootstrapping.
        """
        eps = self.demo_replay.episodes
        S, A, R, S2, D = [], [], [], [], []
        for ep in eps:
            st, ac, rw = ep["state"], ep["action"], ep["reward"]
            S.append(st[:-1]); S2.append(st[1:]); A.append(ac); R.append(rw)
            D.append((rw > 0.5).float())
        S = torch.cat(S).cuda().float(); S2 = torch.cat(S2).cuda().float()
        A = torch.cat(A).cuda().float()
        R = torch.cat(R).cuda().float(); D = torch.cat(D).cuda().float()
        with torch.no_grad():
            z, z2 = self.agent.model.encode(S), self.agent.model.encode(S2)
        self.real_buf.add_batch(z, A, R, z2, D)
        # Separate demo-only buffer so `dyna.demo_mix_rate` can hold the demo
        # share of each batch FIXED. Kept in ADDITION to the real_buf copy above
        # so that demo_mix_rate=0 is exactly the already-measured arm.
        self.demo_buf = FlatBuffer(z.shape[0], z.shape[-1], A.shape[-1], "cuda")
        self.demo_buf.add_batch(z, A, R, z2, D)
        print(f"[dyna-ibrl] preloaded {len(eps)} target demos into the TD replay: "
              f"{len(S)} transitions, {int(D.sum())} terminal (r=1)")


def main():
    cfg = pyrallis.parse(config_class=DynaIBRLMainConfig)  # type: ignore
    ws = DynaIBRLWorkspace(cfg)
    if cfg.bc_steps > 0:
        ws.bc_finetune()              # stage 2: this is what makes the prior a PRIOR
    else:
        print("[dyna-ibrl] bc_steps=0 -> SKIPPING stage-2 policy BC")
    ws.dyna_init()                    # order matters: freeze AFTER stage 2
    ws.adapt()
    if cfg.use_wb:
        import wandb
        wandb.finish()
    os._exit(0)


if __name__ == "__main__":
    main()

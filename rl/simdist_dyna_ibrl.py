"""IBRL in the world model's latent space, with Dyna data augmentation.

Motivation
----------
Two measured facts frame this:
  * model-free IBRL on this target reaches 0.98, but needs ~1M transitions; at the
    matched 100k budget it scores 0.082 and is flat 0 for its first 60k
    (docs/on-policy-adaptation.md). It has the right learning rule and no data.
  * plain Dyna at 100k destroys its own policy. The diagnosed mechanism is that
    DPG ascends `grad_a Q` wherever the critic has no data, and the SAME head that
    DPG corrupts is also the policy that acts.

IBRL fixes exactly that second failure by construction: it acts with
`argmax_a Q(z,a)` over {RL actor, FROZEN BC prior}, so a corrupted actor is
floored by the prior instead of replacing it. Dyna supplies the data IBRL lacks.

What the distilled model already provides -- nothing here is trained from scratch:
  prior   the stage-1 policy head, stage-2 finetuned on the 20 target demos, then
          FROZEN. This is the load-bearing difference from `simdist_dyna.py`,
          where DPG trains the acting policy itself and destroys it.
  actor   a fresh head INITIALISED FROM THE PRIOR, so step 0 is already the BC
          solution rather than noise, and every later step is a departure the
          critic has to justify.
  critic  Q(z,a) warm-started onto the Q the distilled heads already imply,
          r_hat(z,a) + gamma*V_hat(next(z,a)), exactly as simdist_dyna does.
          `q_init_from_value=0` re-initialises instead (ablation).
  dynamics the generator for the synthetic half of each batch.

Everything is latent-native (`shared_encoder=1`): the encoder is frozen for all of
stage 3, so a real state is encoded once at insert and imagined states never leave
latent space. simdist_dyna.py used to decode and re-encode, corrupting the stored
next-state by 0.98x-3.94x the one-step motion.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.world_model import mlp


@dataclass
class DynaIBRLConfig:
    # --- imagination (identical accounting to rl/simdist_dyna.py) ---
    rollout_length: int = 1
    # Sized against `q_updates` so the synthetic half of each batch is neither
    # starved nor stale: q_updates x batch_size x (1-real_ratio) synthetic DRAWS
    # per round, over a buffer holding 5 rounds, at MBPO's ~6x reuse.
    #   5000 x 256 x 0.5 = 640,000 draws / (20,000 x 5) = 6.4
    rollout_batch: int = 20_000
    synth_buffer_size: int = 100_000
    explore_std: float = 0.1
    real_ratio: float = 0.5
    collect_std: float = 0.1
    # Seed the TD replay with the 20 target demos, matching what the model-free
    # IBRL baseline does (`preload_datapath` + `preload_num_data 20`, with
    # `mix_rl_rate 0.5`). WITHOUT this the critic's only source of r=1 is the
    # agent's own successes, so once the policy dips to 0 the reward signal is
    # gone and there is nothing to recover from -- which is exactly how IBRL
    # survives its own 60k of flat-zero online returns. Leaving it off would also
    # make the comparison unfair in the baseline's favour.
    preload_demos: int = 1
    # --- IBRL ---
    # Permanent prior-preference floor: the RL actor wins the argmax only when it
    # beats the prior by more than this. 0 = vanilla IBRL. A positive value turns
    # IBRL into safe policy improvement (CLAUDE.md: 0.85 floor -> 0.94 @55k on this
    # task with an MPPI prior).
    prior_margin: float = 0.0
    # Linear anneal, same three-parameter form as train_rl.py (so the two sides
    # stay cross-referenceable):
    #     margin(t) = max(final, init * (1 - t / steps))
    # init < 0 (default) => no schedule, `prior_margin` stays constant.
    # Motivation from the constant-margin sweep (single seed, 75-100k steady):
    #   margin 0    -> 0.000  the argmax is decided by critic noise and a ~5.8%
    #                         action deviation is enough to kill the episode
    #   margin 0.05 -> 0.775  best; the actor clears the bar occasionally (frac 0.08)
    #   margin 0.1  -> 0.632  bar too high, the actor essentially never acts
    # i.e. early on the margin has to be high enough to survive an uncalibrated
    # critic, and later low enough to let a good actor through -- which is exactly
    # what a schedule buys, and why train_tdmpc2.py calls the anneal a cold-start
    # necessity (0.00 -> 0.70 there).
    prior_margin_init: float = -1.0
    prior_margin_steps: int = 50_000
    prior_margin_final: float = 0.01
    # Fraction of critic updates whose TARGET action comes from the IBRL argmax
    # rather than the actor alone (IBRL's `bootstrap_method=ibrl`).
    ibrl_bootstrap: int = 1
    # --- optimisation ---
    batch_size: int = 256
    q_lr: float = 3e-4
    actor_lr: float = 1e-5
    # Update-to-data ratio -- the whole sample-efficiency claim lives here.
    # The model-free IBRL baseline runs update_freq=2 with num_critic_update=5,
    # i.e. 1000/2*5 = 2500 critic updates per 1000 env steps. `simdist_dyna.py`
    # used q_updates=1000 "to match the planning arms' dyn_update_steps" -- but
    # that is the DYNAMICS head's step count, not an actor-critic UTD, so the Dyna
    # arm was running at 0.4x the baseline's UTD. Beating IBRL from below its own
    # UTD would prove nothing; the point of imagined data is that it BUYS a higher
    # one (MBPO runs 20-40 updates per env step).
    #   1000 = the old, under-powered setting
    #   2500 = EXACTLY the model-free IBRL baseline's UTD (the clean control:
    #          same UTD, does the extra imagined data help?)
    #   5000 = 2x the baseline, the regime model-based methods claim
    q_updates: int = 5000
    # Actor updates per round = q_updates // this. Pinned by `actor_updates_per_round`
    # below rather than set directly, so raising the UTD does not silently multiply
    # the DPG steps -- DPG is what destroyed the policy in the plain Dyna arm
    # (0.40 -> 0.000 policy-only, 3/3 seeds), and IBRL's own RED-Q updates the
    # actor once per 5 critic steps for the same reason.
    policy_update_freq: int = 10
    # The actor is stepped this many times per 1000-step round regardless of
    # q_updates -- matching model-free IBRL exactly (1000/update_freq = 500).
    actor_updates_per_round: int = 500
    tau: float = 0.01
    discount: float = 0.99
    num_q: int = 2
    clamp_q: int = 1
    q_init_steps: int = 500
    q_init_from_value: int = 1
    # --- SGFT potential-based reward shaping, default OFF ---
    # r' = r + coef * (gamma*V_hat(z') - V_hat(z)) with V_hat the FROZEN distilled
    # value head. Potential-based, so the optimal policy is unchanged; it only
    # redistributes credit along the trajectory. The motivation is the same one
    # that made it worth trying in plain Dyna: a sparse 0/1 reward with gamma=0.99
    # gives the TD backup almost nothing to propagate over a 400-step episode.
    #
    # It shifts the Q function to Q'(z,a) = Q(z,a) - coef*V(z), so both the clamp
    # bounds and the Q initialisation shift with it -- a naive [0,1] clamp would
    # silently truncate the shaped Q.
    #
    # Prior evidence is a null: on plain Dyna, sgft_coef=1.0 scored 0.139 against
    # 0.124 (n=10 each, +0.015, ns) -- and that was under the MPPI-proposal metric
    # on the pre-fix implementation, so it says little about this arm.
    sgft_coef: float = 0.0
    target_policy_noise: float = 0.1
    target_noise_clip: float = 0.3


class QHead(nn.Module):
    """Twin Q(z, a) over the world model's dynamics latent."""

    def __init__(self, z_dim, action_dim, hidden, num_layer, num_q=2):
        super().__init__()
        self.nets = nn.ModuleList([
            mlp(z_dim + action_dim, hidden, 1, num_layer, True, act_out="none")
            for _ in range(num_q)
        ])

    def forward(self, z, a):
        za = torch.cat([z, a], dim=-1)
        return torch.stack([net(za).squeeze(-1) for net in self.nets], dim=0)

    def min_q(self, z, a):
        return self.forward(z, a).min(0).values

    def mean_q(self, z, a):
        return self.forward(z, a).mean(0)


class LatentActor(nn.Module):
    """Deterministic actor on the dynamics latent, same shape as the policy head."""

    def __init__(self, z_dim, action_dim, hidden, num_layer):
        super().__init__()
        self.net = mlp(z_dim, hidden, action_dim, num_layer, True, act_out="tanh")

    def forward(self, z):
        return self.net(z)


class DynaIBRLTrainer:
    """Owns the Q head, the RL actor, and the frozen prior.

    The world model belongs to the agent; only `policy` (the prior) is read from
    it here, never written -- that is what keeps the prior a floor.
    """

    def __init__(self, agent, cfg: DynaIBRLConfig, obs_dim, action_dim, device="cuda"):
        self.agent = agent
        self.cfg = cfg
        self.device = device
        wm = agent.model
        wm_cfg = wm.cfg
        with torch.no_grad():
            z_dim = wm.encode(torch.zeros(1, obs_dim, device=device)).shape[-1]
        h, nl = wm_cfg.mlp_hidden_dim, wm_cfg.num_layer

        self.q = QHead(z_dim, action_dim, h, nl, cfg.num_q).to(device)
        self.q_target = QHead(z_dim, action_dim, h, nl, cfg.num_q).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad_(False)

        self.actor = LatentActor(z_dim, action_dim, h, nl).to(device)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.q_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self._step = 0

    # ------------------------------------------------------------- prior
    @torch.no_grad()
    def prior(self, z: torch.Tensor) -> torch.Tensor:
        """The BC prior: the stage-2 policy head, never updated after stage 2."""
        return self.agent.model.policy_from_latent(z)

    def freeze_prior(self, log=print):
        """Take the prior out of every optimiser's reach.

        `simdist_dyna.py` puts `model.policy_parameters()` in an optimiser and lets
        DPG rewrite them; the acting policy and the thing being corrupted are then
        the same object. Here the prior is the floor, so it must not move: stage 3
        trains only `self.actor` and `self.q`, and the dynamics head via the
        parent's own optimiser.
        """
        n = 0
        for p in self.agent.model.policy_parameters():
            p.requires_grad_(False)
            n += p.numel()
        log(f"[dyna-ibrl] prior frozen ({n:,} params); stage 3 trains actor + Q only")

    def init_actor_from_prior(self, sample_z, steps=500, log=print):
        """Regress the fresh actor onto the prior, so step 0 IS the BC solution.

        Cheaper and more faithful than copying weights: the prior may sit behind a
        shared encoder or a different head shape, and this works for every variant.
        """
        opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        losses = []
        for _ in range(steps):
            z = sample_z(self.cfg.batch_size)
            with torch.no_grad():
                tgt = self.prior(z)
            loss = F.mse_loss(self.actor(z), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        log(f"[dyna-ibrl] actor initialised from prior: "
            f"mse {losses[0]:.5f} -> {sum(losses[-20:])/20:.5f}")

    # ------------------------------------------------------------ Q init
    @torch.no_grad()
    def _source_q(self, z, a):
        """The Q the distilled heads already imply: r_hat + gamma*V_hat(next)."""
        wm = self.agent.model
        q = wm.predict_reward(z, a) + self.cfg.discount * wm.predict_value(wm.next(z, a))
        if self.cfg.clamp_q:
            q = q.clamp(0.0, 1.0)
        if self.cfg.sgft_coef > 0:
            # Match the shift the shaped Bellman operator induces, so step 0 is a
            # fixed point of the SHAPED backup rather than the unshaped one.
            q = q - self.cfg.sgft_coef * wm.predict_value(z)
        return q

    def init_q_from_source(self, sample_za, log=print):
        if not self.cfg.q_init_from_value:
            log("[dyna-ibrl] q_init_from_value=0 -> Q left at random init")
            return
        losses = []
        for _ in range(self.cfg.q_init_steps):
            z, a = sample_za(self.cfg.batch_size)
            with torch.no_grad():
                tgt = self._source_q(z, a)
            loss = F.mse_loss(self.q(z, a), tgt.unsqueeze(0).expand(self.cfg.num_q, -1))
            self.q_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.q_opt.step()
            losses.append(float(loss))
        self.q_target.load_state_dict(self.q.state_dict())
        log(f"[dyna-ibrl] Q initialised from source value: "
            f"mse {losses[0]:.5f} -> {sum(losses[-20:])/20:.5f}")

    # ------------------------------------------------------- IBRL action
    @torch.no_grad()
    def act_latent(self, z, *, eval_mode: bool, stddev: float = 0.0,
                   use_target: bool = False) -> torch.Tensor:
        """IBRL: argmax_a Q(z, a) over {RL actor, frozen prior}.

        Selected AND scored with the same critic (`q_target` when building a
        bootstrap, `q` otherwise) -- mixing the two breaks V_hybrid >= V_rl, which
        rl/q_agent.py documents as measured (it held for only ~60% of states).
        """
        cfg = self.cfg
        a_rl = self.actor(z)
        if not eval_mode and stddev > 0:
            a_rl = (a_rl + torch.randn_like(a_rl) * stddev).clamp(-1, 1)
        a_bc = self.prior(z)
        critic = self.q_target if use_target else self.q
        q_rl = critic.min_q(z, a_rl)
        q_bc = critic.min_q(z, a_bc)
        if cfg.prior_margin != 0.0:
            q_rl = q_rl - cfg.prior_margin      # actor must WIN by the margin
        pick_rl = (q_rl >= q_bc).unsqueeze(-1)
        return torch.where(pick_rl, a_rl, a_bc), pick_rl.float().mean()

    # --------------------------------------------------------- imagination
    @torch.no_grad()
    def imagine(self, start_z: torch.Tensor, synth):
        """Branch imagined steps from real latents, acting with the IBRL rule.

        Acting with the hybrid (not the actor alone) matters: the synthetic data
        should follow the distribution the agent will actually visit, and that
        distribution is the argmax over {actor, prior}. Imagining under the actor
        alone would hand the critic data from a policy that never runs.
        """
        z = start_z
        for _ in range(self.cfg.rollout_length):
            a, _ = self.act_latent(z, eval_mode=False, stddev=self.cfg.explore_std)
            r = self.agent.model.predict_reward(z, a).clamp(0.0, 1.0)
            z2 = self.agent.model.next(z, a)
            d = (r > 0.5).float()          # sparse 0/1 + end_on_success => r IS done
            synth.add_batch(z, a, r, z2, d)
            z = z2
            if d.mean() > 0.99:
                break

    # -------------------------------------------------------------- update
    def update(self, real, synth) -> dict:
        """One critic update (+ delayed actor update) on a real/synthetic mix."""
        cfg = self.cfg
        n_real = int(round(cfg.batch_size * cfg.real_ratio))
        n_synth = cfg.batch_size - n_real
        parts = [real.sample(n_real)]
        if n_synth > 0 and len(synth) > 0:
            parts.append(synth.sample(n_synth))
        z, a, r, z2, d = [torch.cat(x, 0) for x in zip(*parts)]

        with torch.no_grad():
            if cfg.ibrl_bootstrap:
                # IBRL bootstrap: the target action is the hybrid argmax, so the
                # critic learns the value of the policy that will actually act.
                a2, frac = self.act_latent(z2, eval_mode=False,
                                           stddev=cfg.target_policy_noise,
                                           use_target=True)
            else:
                a2 = self.actor(z2)
                frac = torch.zeros(())
                if cfg.target_policy_noise > 0:
                    eps = (torch.randn_like(a2) * cfg.target_policy_noise
                           ).clamp(-cfg.target_noise_clip, cfg.target_noise_clip)
                    a2 = (a2 + eps).clamp(-1, 1)
            r_eff, lo = r, 0.0
            if cfg.sgft_coef > 0:
                # Potential Phi(z) = V_hat(z). On a terminal the next potential
                # drops out (the episode ends), matching the (1-d) bootstrap.
                wm = self.agent.model
                v, v2 = wm.predict_value(z), wm.predict_value(z2)
                r_eff = r + cfg.sgft_coef * (cfg.discount * (1.0 - d) * v2 - v)
                lo = -cfg.sgft_coef            # true Q' = Q - coef*V lives in [-coef, 1]
            y = r_eff + cfg.discount * (1.0 - d) * self.q_target.min_q(z2, a2)
            if cfg.clamp_q:
                y = y.clamp(lo, 1.0)

        q_all = self.q(z, a)
        q_loss = F.mse_loss(q_all, y.unsqueeze(0).expand(cfg.num_q, -1))
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()

        m = {"dyna/q_loss": float(q_loss), "dyna/q_mean": float(q_all.mean()),
             "dyna/td_target": float(y.mean()), "dyna/bootstrap_rl_frac": float(frac),
             "dyna/synth_frac": n_synth / cfg.batch_size}

        self._step += 1
        if self._step % cfg.policy_update_freq == 0:
            # DPG on the RL actor only. If it degenerates, the IBRL argmax stops
            # picking it and the prior carries the episode -- the failure is
            # bounded, which is the whole point of doing this inside IBRL.
            pi = self.actor(z.detach())
            actor_loss = -self.q.mean_q(z.detach(), pi).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            m["dyna/actor_loss"] = float(actor_loss)
            with torch.no_grad():
                for p, tp in zip(self.q.parameters(), self.q_target.parameters()):
                    tp.data.lerp_(p.data, cfg.tau)
        return m

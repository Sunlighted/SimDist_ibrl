"""Dyna-style actor-critic on top of a SimDist world model (planning-free baseline).

Contrast with the planning arms (`train_simdist_adapt.py`): there the world model
is used ONLINE, re-planned from the true state every step (MPPI), so model error
only has H steps to accumulate before being reset by reality. Here the model is
used OFFLINE to MANUFACTURE training data for a policy that then executes on its
own. That is the Dyna/MBPO trade: cheaper at deploy time, but the policy can
exploit model error instead of being corrected by it every step.

What is shared with the planning arms (so the comparison is about planning vs
Dyna, not about assets): the same stage-1 world model, the same stage-2 policy
head (BC on the 20 target demos), the same distilled reward/value heads, the same
real-data budget, and the same dynamics-adaptation schedule.

Q head
------
SimDist distils V(z) and r(z,a), not Q(z,a); Dyna needs a Q to take the policy
gradient through. So a Q(z,a) head is added and INITIALISED to the Q that the
source-distilled heads already imply:

    Q_0(z,a)  ==  r_hat(z,a) + gamma * V_hat(next(z,a))

`init_from_source` regresses the fresh head onto exactly that, so at step 0 the
critic IS the source value (as requested), and everything after is TD.
`r_hat`/`V_hat` stay FROZEN: they are the anchor the initialisation is defined
against, and CLAUDE.md 33 measured that refitting the value head offline is what
destroys this system (its discriminative signal is only 0.0068).

Q is clamped to [0, 1]
----------------------
Threading's reward is sparse 0/1 with `end_on_success`, so the true return is in
[0,1] and clamping the TD target is UNBIASED. Without it this codebase has
repeatedly seen Q diverge negative (CLAUDE.md 16/20/22).

Hyper-parameters that are NOT free choices here
-----------------------------------------------
`rollout_length = 1`  -- CLAUDE.md 30 pinned this with three independent
    measurements: the mm ruler says H=3 already exceeds the 1.8 mm tolerance
    window, and the imagined-latent contamination of a Dyna backup is 0.03-0.06
    at H=1 vs 0.30 (latent) / 0.17 (raw) at H=5.
`real_ratio = 0.5`    -- CLAUDE.md 23 chose this for this task: with sparse 0/1
    plus `end_on_success` the positive class is rare, and synthetic data biases Q
    low; half real data keeps the reward signal alive.

Known prior result (this is a replication, not a fresh idea)
-----------------------------------------------------------
A latent Dyna variant on this same world model failed in 2026-07: four variants
sat at 0.66-0.87 and never improved, and dynaK5 showed the diagnostic signature
q_mean +21% while TRUE success fell 76.0% -> 71.5% -- the policy pushed imagined
rollouts into model-preferred regions rather than truly-successful ones. What is
different now: a 5-head dynamics ensemble (which CLAUDE.md 31 showed cuts rollout
error a lot) and the policy head itself being trained rather than a residual.
`q_mean` is logged every round precisely so that signature can be checked again.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.world_model import mlp


@dataclass
class DynaConfig:
    # --- imagination ---
    rollout_length: int = 1          # pinned; see module docstring
    rollout_batch: int = 400         # imagined transitions generated per round
    synth_buffer_size: int = 100_000
    explore_std: float = 0.1         # action noise when imagining (policy is deterministic)
    # --- optimisation ---
    real_ratio: float = 0.5          # fraction of each training batch drawn from REAL data
    batch_size: int = 256
    q_lr: float = 3e-4
    policy_lr: float = 1e-5          # << q_lr on purpose: the policy STARTS at a
                                     # good BC solution, so a large DPG step walks
                                     # it straight into the model's preferred
                                     # region (smoke test: 0.30 -> 0.00 in one
                                     # round at 3e-4, while q_mean stayed 0.67).
                                     # Same lesson as the residual arms
                                     # (CLAUDE.md 20/25 use actor_lr 1e-6..1e-5).
    bc_coef: float = 0.1             # BC pull toward the 20 demos, so the policy
                                     # cannot drift off demo support while chasing
                                     # an imagined Q. 0 = pure DPG.
    q_updates: int = 1000            # per round (matches the planning arms' dyn_update_steps)
    policy_update_freq: int = 2      # TD3-style delayed actor
    tau: float = 0.01                # Q target soft-update
    discount: float = 0.99
    num_q: int = 2                   # twin Q, min for the TD target
    q_init_steps: int = 500          # regression onto r_hat + gamma*V_hat(next)
    clamp_q: int = 1                 # clamp the TD target to [0,1] (sparse 0/1 + end_on_success)
    # --- SGFT potential-based reward shaping (CLAUDE.md 10), default OFF ---
    # r' = r + sgft_coef * (gamma*V_hat(z') - V_hat(z)), with V_hat the FROZEN
    # source-distilled value head. Potential-based => the optimal policy is
    # unchanged; it only redistributes credit along the trajectory. Motivation
    # here: the plain Dyna arm's q_mean decayed 0.57 -> 0.13 because a sparse 0/1
    # reward gives the TD backup almost nothing to propagate; this makes it dense.
    # Shifts the Q function to Q'(z,a) = Q(z,a) - sgft_coef*V(z), so BOTH the
    # clamp bounds and the Q initialisation are shifted accordingly (below) --
    # a naive [0,1] clamp would silently truncate the shaped Q.
    sgft_coef: float = 0.0


class QHead(nn.Module):
    """Twin Q(z, a) over the world model's dynamics latent."""

    def __init__(self, z_dim: int, action_dim: int, hidden: int, num_layer: int,
                 num_q: int = 2):
        super().__init__()
        self.nets = nn.ModuleList([
            mlp(z_dim + action_dim, hidden, 1, num_layer, True, act_out="none")
            for _ in range(num_q)
        ])

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """[num_q, B] Q values."""
        za = torch.cat([z, a], dim=-1)
        return torch.stack([net(za).squeeze(-1) for net in self.nets], dim=0)

    def min_q(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.forward(z, a).min(0).values

    def mean_q(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.forward(z, a).mean(0)


class FlatBuffer:
    """Minimal (s, a, r, s', done) ring buffer on GPU.

    Needed because `SequenceReplayBuffer` yields contiguous H-step segments, not
    the arbitrary (possibly model-generated) single transitions Dyna produces.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device="cuda"):
        self.cap = capacity
        self.s = torch.zeros(capacity, obs_dim, device=device)
        self.a = torch.zeros(capacity, action_dim, device=device)
        self.r = torch.zeros(capacity, device=device)
        self.s2 = torch.zeros(capacity, obs_dim, device=device)
        self.d = torch.zeros(capacity, device=device)
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.cap if self.full else self.idx

    def add_batch(self, s, a, r, s2, d):
        n = s.shape[0]
        assert n <= self.cap
        end = self.idx + n
        if end <= self.cap:
            sl = slice(self.idx, end)
            self.s[sl], self.a[sl], self.r[sl], self.s2[sl], self.d[sl] = s, a, r, s2, d
        else:                                    # wrap
            k = self.cap - self.idx
            self.add_batch(s[:k], a[:k], r[:k], s2[:k], d[:k])
            self.add_batch(s[k:], a[k:], r[k:], s2[k:], d[k:])
            return
        self.idx = end % self.cap
        self.full = self.full or end >= self.cap

    def sample(self, n: int):
        i = torch.randint(0, len(self), (n,), device=self.s.device)
        return self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i]


class DynaTrainer:
    """Owns the Q head + policy optimiser; the world model belongs to the agent."""

    def __init__(self, agent, cfg: DynaConfig, obs_dim: int, action_dim: int,
                 device: str = "cuda"):
        self.agent = agent
        self.cfg = cfg
        self.device = device
        wm_cfg = agent.model.cfg
        z_dim = agent.model.encode(torch.zeros(1, obs_dim, device=device)).shape[-1]
        self.q = QHead(z_dim, action_dim, wm_cfg.mlp_hidden_dim, wm_cfg.num_layer,
                       cfg.num_q).to(device)
        self.q_target = QHead(z_dim, action_dim, wm_cfg.mlp_hidden_dim, wm_cfg.num_layer,
                              cfg.num_q).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.q_lr)
        self.pi_opt = torch.optim.Adam(agent.model.policy_parameters(), lr=cfg.policy_lr)
        self._step = 0

    # -------------------------------------------------------------- init
    @torch.no_grad()
    def _source_q(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """The Q the SOURCE-distilled heads already imply: r_hat + gamma*V_hat(next)."""
        r = self.agent.model.predict_reward(z, a)
        z2 = self.agent.model.next(z, a)
        v = self.agent.model.predict_value(z2)
        q = r + self.cfg.discount * v
        if self.cfg.clamp_q:
            q = q.clamp(0.0, 1.0)
        # Match the shift the shaped Bellman operator induces, so step 0 is a
        # fixed point of the SHAPED backup rather than of the unshaped one.
        if self.cfg.sgft_coef > 0:
            q = q - self.cfg.sgft_coef * self.agent.model.predict_value(z)
        return q

    def init_from_source(self, sample_states, log=print):
        """Regress the fresh Q head onto `_source_q` so step 0 == the source value."""
        losses = []
        for i in range(self.cfg.q_init_steps):
            s, a = sample_states(self.cfg.batch_size)
            with torch.no_grad():
                z = self.agent.model.encode(s)
                tgt = self._source_q(z, a)
            loss = F.mse_loss(self.q(z, a), tgt.unsqueeze(0).expand(self.cfg.num_q, -1))
            self.q_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.q_opt.step()
            losses.append(float(loss))
        self.q_target.load_state_dict(self.q.state_dict())
        log(f"[dyna] Q initialised from source value: "
            f"mse {losses[0]:.5f} -> {sum(losses[-20:])/20:.5f}")

    # --------------------------------------------------------- imagination
    @torch.no_grad()
    def imagine(self, start_states: torch.Tensor, synth: FlatBuffer):
        """Branch `rollout_length` imagined steps from real states.

        The stored transition is in STATE space (decode(z)) so real and synthetic
        data live in one buffer and the Q head re-encodes both the same way.
        Reward doubles as the terminal signal: Threading's reward is 0/1 and the
        episode ends exactly on success, so `done = r_hat > 0.5` is the model's
        own success prediction -- no separate done head is needed (nor available).
        """
        s = start_states
        for _ in range(self.cfg.rollout_length):
            z = self.agent.model.encode(s)
            a = self.agent.model.policy_action(s)
            if self.cfg.explore_std > 0:
                a = (a + torch.randn_like(a) * self.cfg.explore_std).clamp(-1, 1)
            r = self.agent.model.predict_reward(z, a).clamp(0.0, 1.0)
            z2 = self.agent.model.next(z, a)
            s2 = self.agent.model.decode(z2)
            d = (r > 0.5).float()
            synth.add_batch(s, a, r, s2, d)
            s = s2
            if d.mean() > 0.99:      # everything terminated; nothing left to roll
                break

    # ------------------------------------------------------------- update
    def update(self, real: FlatBuffer, synth: FlatBuffer, demo=None) -> dict:
        """One Q update (+ delayed policy update) on a real/synthetic mix."""
        cfg = self.cfg
        n_real = int(round(cfg.batch_size * cfg.real_ratio))
        n_synth = cfg.batch_size - n_real
        parts = [real.sample(n_real)]
        if n_synth > 0 and len(synth) > 0:
            parts.append(synth.sample(n_synth))
        s, a, r, s2, d = [torch.cat(x, 0) for x in zip(*parts)]

        z = self.agent.model.encode(s)
        with torch.no_grad():
            z2 = self.agent.model.encode(s2)
            a2 = self.agent.model.policy_action(s2)
            r_eff, lo = r, 0.0
            if cfg.sgft_coef > 0:
                # Potential Phi(z) = V_hat(z); on a terminal the next potential
                # drops out (the episode ends), matching the (1-d) bootstrap.
                v, v2 = self.agent.model.predict_value(z), self.agent.model.predict_value(z2)
                r_eff = r + cfg.sgft_coef * (cfg.discount * (1.0 - d) * v2 - v)
                lo = -cfg.sgft_coef          # true Q' = Q - coef*V in [-coef, 1]
            y = r_eff + cfg.discount * (1.0 - d) * self.q_target.min_q(z2, a2)
            if cfg.clamp_q:
                y = y.clamp(lo, 1.0)
        q_all = self.q(z, a)
        q_loss = F.mse_loss(q_all, y.unsqueeze(0).expand(cfg.num_q, -1))
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()

        metrics = {"dyna/q_loss": float(q_loss), "dyna/q_mean": float(q_all.mean()),
                   "dyna/td_target": float(y.mean()), "dyna/synth_frac": n_synth / cfg.batch_size}

        self._step += 1
        if self._step % cfg.policy_update_freq == 0:
            # DPG: maximise Q(z, pi(s)). z is detached -- the policy must improve
            # the ACTION, not reshape the latent (the encoder is frozen anyway).
            pi = self.agent.model.policy_action(s)
            pi_loss = -self.q.mean_q(z.detach(), pi).mean()
            if cfg.bc_coef > 0 and demo is not None:
                ds, da = demo
                bc = F.mse_loss(self.agent.model.policy_action(ds), da)
                pi_loss = pi_loss + cfg.bc_coef * bc
                metrics["dyna/bc_loss"] = float(bc)
            self.pi_opt.zero_grad(set_to_none=True)
            pi_loss.backward()
            self.pi_opt.step()
            metrics["dyna/pi_loss"] = float(pi_loss)
            with torch.no_grad():
                for p, tp in zip(self.q.parameters(), self.q_target.parameters()):
                    tp.data.lerp_(p.data, cfg.tau)
        return metrics

"""TD-MPC2 (online model-based RL) with an IBRL prior — CLAUDE.md §28.

The SimDist baseline (§17) *distills* a world model from a frozen expert's data
and plans with MPPI. This module is the other half of that comparison: the SAME
world-model architecture + the SAME MPPI planning mechanism, but trained ONLINE
by TD-MPC2's own objective (latent consistency + reward + TD-learned Q, with a
policy prior trained by max-Q). Faithful port of `~/tdmpc2/tdmpc2/tdmpc2.py` +
`common/world_model.py` to this repo's state-based, single-task setting.

The one addition — and the reason it can work at all on Threading — is IBRL.
Threading's reward is sparse (0/1 + end_on_success); a from-scratch TD-MPC2 never
stumbles into a success, so reward/Q stay at 0 and MPPI has nothing to rank. We
therefore attach a FROZEN prior policy `pi_prior` (a BC head, or a frozen RL
agent) and let the LEARNED Q decide which policy acts, exactly as IBRL does:

    a = argmax_{a in {a_plan, a_prior}} Q(z, a)

with three touch points:
  1. ACT      — the MPPI action competes with the prior's action under Q(z0, .).
  2. PLAN     — `num_prior_trajs` of the MPPI candidates are SEEDED with the
                prior's action at t=0 (then rolled with pi), so the planner can
                also discover it inside its own return criterion.
  3. BOOTSTRAP— the TD target picks max-Q over {pi(z'), pi_prior(s')} instead of
                pi(z') alone (IBRL's bootstrap_method="ibrl").

The prior is queried on the RAW state (which we always have: at act time it's the
env obs, at update time it's `states[:, t+1]` in the batch), so we never need the
decode-bridge that §27 found to be a large error source.

Deviations from the reference impl, all deliberate:
  * no multitask / no pixels / no torch.compile / no CudaGraphs (state-only, and
    the repo's eval framework pickles nothing but obs).
  * the planner is STATELESS (no `_prev_mean` warm start across control steps) so
    `act()` is safe in the multi-process eval path, and it supports a BATCHED
    plan (mp eval hands us [B, obs] and wants [B, act] back).
  * `clamp_q_target` [0,1]: with a sparse 0/1 reward + end_on_success the true Q
    is in [0,1], so clamping the Bellman target is UNBIASED and severs the
    negative-divergence loop this repo has hit repeatedly (§16/§20).
  * two-hot bins default to vmin/vmax = -2/+2 (not the reference's -10/+10):
    the return here lives in [0,1], and symlog([-10,10]) with 101 bins wastes
    almost all of its resolution outside the range we care about.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.world_model import SimNorm, two_hot_inv, soft_ce


# --------------------------------------------------------------------- config
@dataclass
class TDMPC2Config:
    # architecture (TD-MPC2 defaults, state-based)
    latent_dim: int = 512
    mlp_dim: int = 512
    num_layer: int = 2
    simnorm_dim: int = 8
    num_q: int = 5
    dropout: float = 0.01
    # discrete (two-hot) reward + Q heads. The task return is in [0,1] (sparse 0/1
    # + end_on_success), so the reference's [-10,10] symlog range is ~all wasted.
    num_bins: int = 101
    vmin: float = -2.0
    vmax: float = 2.0
    # policy prior (the world model's own actor)
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    entropy_coef: float = 1e-4
    # planning (MPPI / CEM)
    mpc: int = 1                  # 0 => act with the pi head (+IBRL select) only
    horizon: int = 3
    iterations: int = 6
    num_samples: int = 512
    num_elites: int = 64
    num_pi_trajs: int = 24
    min_std: float = 0.05
    max_std: float = 2.0
    temperature: float = 0.5
    # optimization
    lr: float = 3e-4
    enc_lr_scale: float = 0.3
    tau: float = 0.01
    grad_clip: float = 20.0
    consistency_coef: float = 20.0
    reward_coef: float = 0.1
    value_coef: float = 0.1
    rho: float = 0.5
    discount: float = 0.99
    # ---------------------------------------------------------------- IBRL
    # 1 = the learned Q arbitrates between the world model's action and the frozen
    # prior's action (at act time AND in the TD bootstrap). 0 = vanilla TD-MPC2
    # (the prior, if attached, is then only used for warm-up data collection).
    ibrl: int = 1
    # MPPI candidates seeded from the prior (first action = prior(s) + noise, the
    # rest rolled with pi). 0 = don't seed the planner with the prior.
    num_prior_trajs: int = 24
    prior_traj_std: float = 0.1
    # OPTIONAL BC anchor on the policy head — OFF by default, to match vanilla IBRL.
    # (IBRL's QAgentConfig also carries `bc_loss_coef: 0.1`, but it is DEAD in the
    # threading_ibrl_* runs: `add_bc_loss: 0` there, so `bc_batch` is None and the
    # rft actor loss that would use it is never called. Its actor is pure DPG.)
    # Vanilla IBRL's ONLY cold-start mechanism is the demo-mixed batch
    # (train_tdmpc2.MainConfig.mix_rl_rate). Set >0 to add the anchor as an ablation.
    bc_loss_coef: float = 0.0
    # Permanent prior preference: the RL/plan action is only taken when it beats
    # the prior by > prior_margin under Q. >0 turns IBRL into a *safe policy
    # improvement* over the prior (project-ibrl-perm-margin); 0 = plain IBRL
    # (what `threading_ibrl_*` actually runs — pure greedy max-Q from step 0).
    prior_margin: float = 0.0
    # sparse 0/1 reward + end_on_success => true Q in [0,1] => clamping the
    # Bellman target is unbiased and kills negative divergence.
    clamp_q_target: int = 1
    q_min: float = 0.0
    q_max: float = 1.0


# ------------------------------------------------------------------- modules
def _mlp(in_dim, hidden, out_dim, num_layer, dropout=0.0, act_out=None):
    """TD-MPC2's building block: [Linear+LayerNorm+Mish] * num_layer -> Linear.
    `dropout` is applied after the FIRST hidden layer only (reference: Q heads).
    `act_out` (e.g. SimNorm) is applied to the output when given."""
    dims = [in_dim] + [hidden] * num_layer
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.LayerNorm(dims[i + 1]))
        layers.append(nn.Mish(inplace=False))
        if dropout > 0 and i == 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(dims[-1], out_dim))
    if act_out is not None:
        layers.append(act_out)
    return nn.Sequential(*layers)


def _weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class TDMPC2WorldModel(nn.Module):
    """encoder / dynamics / reward / pi / Q-ensemble (+ EMA target Qs)."""

    def __init__(self, obs_dim: int, action_dim: int, cfg: TDMPC2Config):
        super().__init__()
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        z, h, nl = cfg.latent_dim, cfg.mlp_dim, cfg.num_layer
        assert z % cfg.simnorm_dim == 0, "latent_dim must be divisible by simnorm_dim"
        simnorm = SimNorm(cfg.simnorm_dim)

        self.encoder = _mlp(obs_dim, h, z, nl, act_out=simnorm)
        self.dynamics = _mlp(z + action_dim, h, z, nl, act_out=SimNorm(cfg.simnorm_dim))
        self.reward = _mlp(z + action_dim, h, cfg.num_bins, nl)
        self.pi = _mlp(z, h, 2 * action_dim, nl)
        self.Qs = nn.ModuleList([
            _mlp(z + action_dim, h, cfg.num_bins, nl, dropout=cfg.dropout)
            for _ in range(cfg.num_q)
        ])
        self.apply(_weight_init)
        # zero-init the reward / Q output layers (reference: init.zero_)
        nn.init.zeros_(self.reward[-1].weight)
        for q in self.Qs:
            nn.init.zeros_(q[-1].weight)

        self.Qs_target = nn.ModuleList([
            _mlp(z + action_dim, h, cfg.num_bins, nl, dropout=cfg.dropout)
            for _ in range(cfg.num_q)
        ])
        self.Qs_target.load_state_dict(self.Qs.state_dict())
        for p in self.Qs_target.parameters():
            p.requires_grad_(False)

        self.register_buffer("log_std_min", torch.tensor(float(cfg.log_std_min)))
        self.register_buffer("log_std_dif",
                             torch.tensor(float(cfg.log_std_max - cfg.log_std_min)))

    # ------------------------------------------------------------- forwards
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)

    def next(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.dynamics(torch.cat([z, a], dim=-1))

    def reward_logits(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.reward(torch.cat([z, a], dim=-1))

    def predict_reward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        return two_hot_inv(self.reward_logits(z, a), c.num_bins, c.vmin, c.vmax)

    def pi_dist(self, z: torch.Tensor):
        """Squashed-Gaussian policy prior. Returns (action, mean, entropy)."""
        mean, log_std = self.pi(z).chunk(2, dim=-1)
        log_std = self.log_std_min + 0.5 * self.log_std_dif * (torch.tanh(log_std) + 1)
        eps = torch.randn_like(mean)
        # gaussian logprob of the pre-squash sample, then the tanh correction
        log_prob = (-0.5 * eps.pow(2) - log_std - 0.9189385175704956).sum(-1, keepdim=True)
        pre = mean + eps * log_std.exp()
        action = torch.tanh(pre)
        log_prob = log_prob - torch.log(F.relu(1 - action.pow(2)) + 1e-6).sum(-1, keepdim=True)
        return action, torch.tanh(mean), -log_prob

    def Q(self, z: torch.Tensor, a: torch.Tensor, *, target: bool = False,
          return_type: str = "min") -> torch.Tensor:
        """return_type:
             'all'      -> raw logits [num_q, ..., num_bins] (for the CE loss)
             'min'/'avg'-> scalar over 2 RANDOM heads (TD-MPC2's subsampling)
             'min_all'  -> scalar, min over ALL heads (deterministic + pessimistic;
                           used for the IBRL argmax so the choice isn't noisy)
        """
        c = self.cfg
        x = torch.cat([z, a], dim=-1)
        nets = self.Qs_target if target else self.Qs
        out = torch.stack([net(x) for net in nets])  # [num_q, ..., num_bins]
        if return_type == "all":
            return out
        if return_type == "min_all":
            return two_hot_inv(out, c.num_bins, c.vmin, c.vmax).min(0).values
        idx = torch.randperm(c.num_q, device=out.device)[:2]
        q = two_hot_inv(out[idx], c.num_bins, c.vmin, c.vmax)  # [2, ...]
        return q.min(0).values if return_type == "min" else q.mean(0)

    def soft_update_target_Q(self, tau: float):
        with torch.no_grad():
            for p, pt in zip(self.Qs.parameters(), self.Qs_target.parameters()):
                pt.data.lerp_(p.data, tau)

    def train(self, mode: bool = True):
        super().train(mode)
        self.Qs_target.train(False)  # target nets always in eval (dropout off)
        return self

    def model_parameters(self):
        """Everything the model optimizer trains (encoder/dynamics/reward/Qs)."""
        return (list(self.encoder.parameters()) + list(self.dynamics.parameters())
                + list(self.reward.parameters()) + list(self.Qs.parameters()))


class RunningScale(nn.Module):
    """TD-MPC2's running trimmed (5-95 percentile) scale for the pi loss."""

    def __init__(self, tau: float):
        super().__init__()
        self.tau = tau
        self.register_buffer("value", torch.ones(1))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.detach().flatten()
        if x.numel() < 2:
            return
        lo = torch.quantile(x, 0.05)
        hi = torch.quantile(x, 0.95)
        self.value.data.lerp_(torch.clamp(hi - lo, min=1.0).view(1), self.tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.value


# --------------------------------------------------------------------- agent
class TDMPC2Agent(nn.Module):
    def __init__(self, use_state, obs_shape, prop_shape, action_dim, rl_camera,
                 cfg: TDMPC2Config, device: str = "cuda"):
        super().__init__()
        assert use_state, "TDMPC2Agent is state-only"
        self.cfg = cfg
        self.device = device
        self.obs_dim = int(obs_shape[0])
        self.action_dim = int(action_dim)

        self.model = TDMPC2WorldModel(self.obs_dim, self.action_dim, cfg)
        self.scale = RunningScale(cfg.tau)
        self.to(device)

        enc_ids = {id(p) for p in self.model.encoder.parameters()}
        other = [p for p in self.model.model_parameters() if id(p) not in enc_ids]
        self.model_opt = torch.optim.Adam([
            {"params": list(self.model.encoder.parameters()), "lr": cfg.lr * cfg.enc_lr_scale},
            {"params": other, "lr": cfg.lr},
        ])
        self.pi_opt = torch.optim.Adam(self.model.pi.parameters(), lr=cfg.lr, eps=1e-5)

        # FROZEN IBRL prior, held in a LIST so it is NOT a submodule (stays out of
        # state_dict; re-attached by path at eval, like every other prior here).
        self._prior: list = []
        # fraction of the last act() batch where Q preferred the RL/plan action
        self._last_take_rl: float = 1.0

    # ------------------------------------------------------------- IBRL prior
    def attach_prior(self, prior_policy):
        """`prior_policy.act(obs_dict, eval_mode=, cpu=) -> [B, act]` — the same
        interface IBRL's `QAgent.add_bc_policy` expects, so any BC head (MLP /
        flow / chunked) or `train_rl.FrozenAgentBCPolicy` (a frozen QAgent) works."""
        prior_policy.eval()
        for p in prior_policy.parameters():
            p.requires_grad_(False)
        self._prior = [prior_policy]

    @property
    def prior(self):
        return self._prior[0] if self._prior else None

    @property
    def has_prior(self) -> bool:
        return bool(self._prior) and bool(self.cfg.ibrl)

    @torch.no_grad()
    def prior_action(self, state: torch.Tensor) -> torch.Tensor:
        """state[B, obs] -> prior action [B, act], on-device, clamped.

        Deliberately calls `act(obs, eval_mode=True, cpu=False)` and nothing else:
        that is the lowest common denominator across the prior flavours here (BC
        heads take no `stddev`; FrozenAgentBCPolicy defaults it)."""
        a = self.prior.act({"state": state}, eval_mode=True, cpu=False)
        if isinstance(a, tuple):
            a = a[0]
        if a.dim() == 3:      # a chunked BC head returns [B, T, act]; take the first
            a = a[:, 0]
        if a.dim() == 1:      # some heads squeeze a single-row batch
            a = a.view(1, -1)
        return a.to(state.device).float().clamp(-1, 1)

    def _ibrl_pick(self, z, a_rl, a_prior, *, target: bool = False):
        """IBRL max-Q: take a_rl only where it beats a_prior by > prior_margin."""
        q_rl = self.model.Q(z, a_rl, target=target, return_type="min_all")
        q_pr = self.model.Q(z, a_prior, target=target, return_type="min_all")
        take_rl = (q_rl > q_pr + self.cfg.prior_margin).unsqueeze(-1)  # [B,1]
        action = torch.where(take_rl, a_rl, a_prior)
        return action, take_rl.float().mean().item()

    # -------------------------------------------------------------------- act
    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], *, eval_mode: bool = False,
            stddev: float = 0.0, cpu: bool = True) -> torch.Tensor:
        state = obs["state"].to(self.device).float()
        single = state.dim() == 1
        s = state.view(1, -1) if single else state          # [B, obs]

        z = self.model.encode(s)
        if self.cfg.mpc:
            action = self._plan(z, s, eval_mode=eval_mode)
        else:
            a_samp, a_mean, _ = self.model.pi_dist(z)
            action = a_mean if eval_mode else a_samp

        if self.has_prior:
            a_prior = self.prior_action(s)
            action, self._last_take_rl = self._ibrl_pick(z, action, a_prior)

        if not eval_mode and stddev > 0:
            action = (action + torch.randn_like(action) * stddev).clamp(-1, 1)
        action = action.clamp(-1, 1)
        action = action[0] if single else action
        return action.cpu() if cpu else action

    # ------------------------------------------------------------------ plan
    @torch.no_grad()
    def _pi_trajs(self, z: torch.Tensor, num: int) -> torch.Tensor:
        """Roll the pi head H steps in latent for `num` copies per env.
        z: [B, z] -> [B, H, num, act]."""
        H, B, zd = self.cfg.horizon, z.shape[0], z.shape[1]
        zz = z[:, None, :].expand(B, num, zd).reshape(B * num, zd)
        acts = []
        for _ in range(H):
            a, _, _ = self.model.pi_dist(zz)
            acts.append(a.view(B, num, -1))
            zz = self.model.next(zz, a)
        return torch.stack(acts, dim=1)

    @torch.no_grad()
    def _prior_trajs(self, z: torch.Tensor, s: torch.Tensor, num: int) -> torch.Tensor:
        """IBRL candidate seeding: first action = prior(s) (+ noise, so the `num`
        copies are not identical), then roll pi in latent. z:[B,z], s:[B,obs]
        -> [B, H, num, act]."""
        H, B, zd = self.cfg.horizon, z.shape[0], z.shape[1]
        a0 = self.prior_action(s)[:, None, :].expand(B, num, self.action_dim)
        a0 = (a0 + torch.randn_like(a0) * self.cfg.prior_traj_std).clamp(-1, 1)
        acts = [a0]
        zz = self.model.next(z[:, None, :].expand(B, num, zd).reshape(B * num, zd),
                             a0.reshape(B * num, -1))
        for _ in range(H - 1):
            a, _, _ = self.model.pi_dist(zz)
            acts.append(a.view(B, num, -1))
            zz = self.model.next(zz, a)
        return torch.stack(acts, dim=1)

    @torch.no_grad()
    def _estimate_value(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """z:[B,z], actions:[B,H,N,act] -> return estimate [B,N].
        G = sum_t gamma^t r_hat_t + gamma^H Q(z_H, pi(z_H))  (IBRL: max-Q over
        {pi, prior} is NOT applied here — the prior is not queryable at an
        imagined latent without a decode bridge, which §27 showed is a large
        error source. The prior instead enters planning via `_prior_trajs`.)"""
        B, H, N, _ = actions.shape
        zd = z.shape[1]
        zz = z[:, None, :].expand(B, N, zd).reshape(B * N, zd)
        G = torch.zeros(B * N, device=z.device)
        disc = 1.0
        for t in range(H):
            a = actions[:, t].reshape(B * N, -1)
            G = G + disc * self.model.predict_reward(zz, a)
            zz = self.model.next(zz, a)
            disc *= self.cfg.discount
        a_end, _, _ = self.model.pi_dist(zz)
        G = G + disc * self.model.Q(zz, a_end, return_type="avg")
        return G.view(B, N)

    @torch.no_grad()
    def _plan(self, z: torch.Tensor, s: torch.Tensor, eval_mode: bool) -> torch.Tensor:
        """MPPI/CEM over the latent world model. z:[B,z], s:[B,obs] -> [B,act].
        Batched (mp eval sends B envs at once); stateless (no warm-start mean)."""
        cfg = self.cfg
        H, B, A = cfg.horizon, z.shape[0], self.action_dim

        seeded = []
        if cfg.num_pi_trajs > 0:
            seeded.append(self._pi_trajs(z, cfg.num_pi_trajs))
        n_prior = cfg.num_prior_trajs if (self._prior and cfg.num_prior_trajs > 0) else 0
        if n_prior > 0:
            seeded.append(self._prior_trajs(z, s, n_prior))
        seed_actions = torch.cat(seeded, dim=2) if seeded else None  # [B,H,n_seed,act]
        n_seed = seed_actions.shape[2] if seed_actions is not None else 0
        n_rand = max(cfg.num_samples - n_seed, 1)

        mean = torch.zeros(B, H, A, device=z.device)
        std = torch.full((B, H, A), cfg.max_std, device=z.device)
        score = None
        elite_acts = None
        for _ in range(cfg.iterations):
            noise = torch.randn(B, H, n_rand, A, device=z.device)
            rand_actions = (mean[:, :, None, :] + std[:, :, None, :] * noise).clamp(-1, 1)
            actions = (torch.cat([rand_actions, seed_actions], dim=2)
                       if seed_actions is not None else rand_actions)   # [B,H,N,act]

            value = self._estimate_value(z, actions).nan_to_num(0)      # [B,N]
            k = min(cfg.num_elites, value.shape[1])
            elite_val, elite_idx = torch.topk(value, k, dim=1)          # [B,K]
            idx = elite_idx[:, None, :, None].expand(B, H, k, A)
            elite_acts = torch.gather(actions, 2, idx)                  # [B,H,K,act]

            score = torch.exp(cfg.temperature
                              * (elite_val - elite_val.max(dim=1, keepdim=True).values))
            score = score / (score.sum(dim=1, keepdim=True) + 1e-9)     # [B,K]
            sc = score[:, None, :, None]
            mean = (sc * elite_acts).sum(2)
            std = torch.sqrt((sc * (elite_acts - mean[:, :, None, :]) ** 2).sum(2))
            std = std.clamp(cfg.min_std, cfg.max_std)

        # sample one elite ~ score (TD-MPC2 gumbel-samples; a plain multinomial over
        # the same weights is the same distribution)
        pick = torch.multinomial(score, 1)                              # [B,1]
        a = torch.gather(elite_acts[:, 0], 1,
                         pick[:, :, None].expand(B, 1, A)).squeeze(1)   # [B,act]
        if not eval_mode:
            a = a + std[:, 0] * torch.randn_like(a)
        return a.clamp(-1, 1)

    # ---------------------------------------------------------------- update
    @torch.no_grad()
    def _td_target(self, next_z, next_state, reward, done):
        """IBRL bootstrap: max-Q over {pi(z'), prior(s')} under the TARGET Qs."""
        a, _, _ = self.model.pi_dist(next_z)
        take_rl = 1.0
        if self.has_prior:
            a, take_rl = self._ibrl_pick(next_z, a, self.prior_action(next_state),
                                         target=True)
        q = self.model.Q(next_z, a, target=True, return_type="min")
        y = reward + self.cfg.discount * (1 - done) * q
        if self.cfg.clamp_q_target:
            y = y.clamp(self.cfg.q_min, self.cfg.q_max)
        return y, take_rl

    def update(self, states, actions, rewards, bc_states=None, bc_actions=None
               ) -> dict[str, float]:
        """states[B,H+1,obs], actions[B,H,act], rewards[B,H].

        `done` is derived from the reward (sparse 0/1 + end_on_success => r=1 IS
        the terminal step). Timeouts are NOT terminal (we keep bootstrapping),
        which is the correct treatment for a time-limit truncation.

        bc_states[N,obs] / bc_actions[N,act] (optional): flat DEMO (s,a) pairs for
        the policy head's BC anchor (IBRL's `bc_loss_coef`).
        """
        cfg = self.cfg
        B, Hp1, _ = states.shape
        H = Hp1 - 1
        self.model.train()

        done = (rewards > 0.5).float()                       # [B,H]

        with torch.no_grad():
            next_states = states[:, 1:].reshape(B * H, -1)
            next_z = self.model.encode(next_states)          # [B*H, z]
            td_targets, take_rl = self._td_target(
                next_z, next_states, rewards.reshape(-1), done.reshape(-1))
            td_targets = td_targets.view(B, H)
            next_z = next_z.view(B, H, -1)

        # ---- latent rollout ----
        z = self.model.encode(states[:, 0])
        zs = [z]
        cons_l = 0.0
        for t in range(H):
            z = self.model.next(z, actions[:, t])
            cons_l = cons_l + F.mse_loss(z, next_z[:, t]) * (cfg.rho ** t)
            zs.append(z)
        zs = torch.stack(zs, dim=1)                          # [B, H+1, z]

        # ---- reward + Q (two-hot cross entropy, per-step rho-weighted) ----
        rew_l = 0.0
        val_l = 0.0
        nb, vmin, vmax = cfg.num_bins, cfg.vmin, cfg.vmax
        for t in range(H):
            w = cfg.rho ** t
            z_t, a_t = zs[:, t], actions[:, t]
            rew_l = rew_l + w * soft_ce(
                self.model.reward_logits(z_t, a_t), rewards[:, t], nb, vmin, vmax).mean()
            q_logits = self.model.Q(z_t, a_t, return_type="all")   # [num_q, B, nb]
            for qi in range(cfg.num_q):
                val_l = val_l + w * soft_ce(
                    q_logits[qi], td_targets[:, t], nb, vmin, vmax).mean()

        cons_l = cons_l / H
        rew_l = rew_l / H
        val_l = val_l / (H * cfg.num_q)
        loss = (cfg.consistency_coef * cons_l + cfg.reward_coef * rew_l
                + cfg.value_coef * val_l)

        self.model_opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(self.model.model_parameters(), cfg.grad_clip)
        self.model_opt.step()

        # ---- policy prior: maximize Q over the (detached) rolled latents ----
        pi_info = self._update_pi(zs.detach(), bc_states, bc_actions)

        self.model.soft_update_target_Q(cfg.tau)
        self.model.eval()

        with torch.no_grad():
            q_pred = self.model.Q(zs[:, 0].detach(), actions[:, 0], return_type="min_all")
        out = {
            "tdmpc2/loss": float(loss.item()),
            "tdmpc2/consistency": float(cons_l.item()),
            "tdmpc2/reward_loss": float(rew_l.item()),
            "tdmpc2/value_loss": float(val_l.item()),
            "tdmpc2/grad_norm": float(gnorm),
            "tdmpc2/q_mean": float(q_pred.mean()),
            "tdmpc2/td_target": float(td_targets.mean()),
            "tdmpc2/batch_reward": float(rewards.mean()),
            "ibrl/bootstrap_take_rl": float(take_rl),
        }
        out.update(pi_info)
        return out

    def _update_pi(self, zs, bc_states=None, bc_actions=None) -> dict[str, float]:
        cfg = self.cfg
        a, _, entropy = self.model.pi_dist(zs)                 # [B,H+1,act]
        qs = self.model.Q(zs, a, return_type="avg")            # [B,H+1]
        self.scale.update(qs[:, 0])
        qs = self.scale(qs)
        rho = torch.pow(cfg.rho, torch.arange(zs.shape[1], device=zs.device))
        pi_loss = (-(cfg.entropy_coef * entropy.squeeze(-1) + qs).mean(dim=0) * rho).mean()

        # IBRL's BC anchor: keep the policy head near the demo actions so it is not
        # pure (initially garbage) DPG. The latent is detached, so this trains only
        # the pi head, never the encoder.
        bc_loss = torch.zeros((), device=zs.device)
        if bc_actions is not None and cfg.bc_loss_coef > 0:
            with torch.no_grad():
                z_bc = self.model.encode(bc_states)
            _, mu, _ = self.model.pi_dist(z_bc)                # deterministic mean
            bc_loss = F.mse_loss(mu, bc_actions)
            pi_loss = pi_loss + cfg.bc_loss_coef * bc_loss

        self.pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        gnorm = nn.utils.clip_grad_norm_(self.model.pi.parameters(), cfg.grad_clip)
        self.pi_opt.step()
        # the pi loss also deposited grads on the Q heads (they are on the graph);
        # drop them so they can't leak into the NEXT model step.
        self.model_opt.zero_grad(set_to_none=True)
        return {
            "tdmpc2/pi_loss": float(pi_loss.item()),
            "tdmpc2/pi_bc_loss": float(bc_loss.item()),
            "tdmpc2/pi_grad_norm": float(gnorm),
            "tdmpc2/pi_entropy": float(entropy.mean()),
            "tdmpc2/pi_scale": float(self.scale.value.item()),
        }

    # ------------------------------------------------------------ eval hooks
    # `train()` / `eval()` are nn.Module's: they recurse into `self.model`, whose
    # own `train()` override keeps the target Qs in eval (dropout off) always.
    # The frozen prior lives in a list, so it is never touched by either.

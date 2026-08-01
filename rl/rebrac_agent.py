"""ReBRAC (Tarasov et al., NeurIPS 2023) ported onto ibrl — offline TD3+BC.

Faithful PyTorch port of the JAX reference (`/home/Chenyang/ReBRAC`,
`src/algorithms/rebrac.py`) with ONE deliberate deviation for a clean head-to-head
against MLP-BC: **the actor IS the MLP-BC actor network** (`bc.bc_policy.StateBcPolicy`,
state -> MLP -> Tanh). So ReBRAC and the `threading_d0_state_mlp_bc_target` baseline
share the identical actor architecture (num_layer/hidden_dim/dropout/layer_norm);
the only difference is the training objective — pure MSE BC vs. TD3+BC.

ReBRAC = TD3+BC with the four "minimalist" knobs from the paper:
  1. DECOUPLED actor / critic BC penalties (`actor_bc_coef` / `critic_bc_coef`).
     The critic penalty rides on the TD target: next_q -= critic_bc_coef * ||a'_pi - a'_data||^2.
  2. Deep LayerNorm critic (default 3x256 + LayerNorm).
  3. Q-normalization in the actor loss: lambda = sg(1 / |Q|.mean()) so the BC term
     and the Q term are on a comparable scale.
  4. Large batch + long training (set in the trainer / YAML).
Plus standard TD3: target policy smoothing (policy_noise/noise_clip), twin-Q min,
delayed actor + target updates (policy_freq).

We do NOT port the calibration branch's MC-return lower bound (`maximum(..., mc_returns)`)
— that is the separate Cal-QL-style addition on the `calibration` git branch, not core
ReBRAC. Target = r + (1-done) * gamma * (min_q - critic_bc_coef * bc_penalty).

State normalization (z-score) is part of ReBRAC's recipe; it lives in registered
buffers (identity until `set_normalization()`), applied in both `update` and `act`,
so the saved ckpt self-normalizes at eval with no dataset needed. `normalize_states=0`
turns it into the apples-to-apples (raw-state) ablation matching MLP-BC's input.
"""

from dataclasses import dataclass, field
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from bc.bc_policy import StateBcPolicy, StateBcPolicyConfig


def _build_q_mlp(in_dim, hidden_dim, n_layer, layer_norm):
    dims = [in_dim] + [hidden_dim for _ in range(n_layer)]
    layers: list[nn.Module] = []
    for i in range(n_layer):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(dims[i + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-1], 1))
    return nn.Sequential(*layers)


class EnsembleQ(nn.Module):
    """N independent Q-nets over concat(state, action). Mirrors ReBRAC's
    `EnsembleCritic` (default 2 critics, LayerNorm, 3 hidden)."""

    def __init__(self, obs_dim, action_dim, hidden_dim, n_layer, layer_norm, num_critics=2):
        super().__init__()
        self.num_critics = num_critics
        self.qs = nn.ModuleList([
            _build_q_mlp(obs_dim + action_dim, hidden_dim, n_layer, layer_norm)
            for _ in range(num_critics)
        ])

    def forward(self, state, action) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        # [num_critics, B]
        return torch.stack([q(x).squeeze(-1) for q in self.qs], dim=0)


@dataclass
class ReBRACConfig:
    # actor = MLP-BC network (StateBcPolicy). Mirror the baseline YAML's state_policy.
    actor: StateBcPolicyConfig = field(default_factory=lambda: StateBcPolicyConfig(
        num_layer=3, hidden_dim=1024, dropout=0.0, layer_norm=0))
    # critic (ReBRAC default: 3x256 + LayerNorm, twin Q)
    critic_hidden_dim: int = 256
    critic_n_layer: int = 3
    critic_layer_norm: int = 1
    num_critics: int = 2
    # optim
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    # TD3+BC
    gamma: float = 0.99
    tau: float = 5e-3
    actor_bc_coef: float = 1.0      # beta in the actor loss (BC pull on pi)
    critic_bc_coef: float = 1.0     # beta in the TD target (BC pull on a')
    policy_noise: float = 0.2       # target policy smoothing std (in [-1,1] action space)
    noise_clip: float = 0.5
    policy_freq: int = 2            # delayed actor + target update cadence
    normalize_q: int = 1            # lambda = sg(1/|Q|.mean()) in the actor loss
    # state z-score (ReBRAC recipe). 0 = raw state (matches MLP-BC input exactly).
    normalize_states: int = 1
    # Per-sample gap weighting (for the weighted-BC comparison; see CLAUDE.md §19).
    # When the batch carries `weights` (target demos = 1, source = exp(-gap/tau)),
    # the BC terms (actor BC penalty + critic BC penalty) are ALWAYS weighted.
    # `weight_td` (default ON) ALSO weights the critic TD/Bellman loss by the same
    # per-sample weight (the §16 gated "trust the backup less where source
    # dynamics is wrong"); set 0 to weight only the BC terms.
    weight_td: int = 1


class ReBRACAgent(nn.Module):
    def __init__(self, use_state, obs_shape, prop_shape, action_dim, rl_camera,
                 cfg: ReBRACConfig, device: str = "cuda"):
        super().__init__()
        assert use_state, "ReBRACAgent is state-only"
        assert len(obs_shape) == 1, "state obs must be 1-D"
        self.cfg = cfg
        self.device = device
        obs_dim = int(obs_shape[0])
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # actor = MLP-BC network (identical to threading_d0_state_mlp_bc_target)
        self.actor = StateBcPolicy(obs_shape, action_dim, cfg.actor)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = EnsembleQ(
            obs_dim, action_dim, cfg.critic_hidden_dim, cfg.critic_n_layer,
            bool(cfg.critic_layer_norm), cfg.num_critics,
        )
        self.critic_target = copy.deepcopy(self.critic)

        # state normalization buffers (identity until set_normalization)
        self.register_buffer("norm_state_mean", torch.zeros(obs_dim))
        self.register_buffer("norm_state_std", torch.ones(obs_dim))

        for p in self.actor_target.parameters():
            p.requires_grad_(False)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self._update_count = 0

    # ------------------------------------------------------------ normalization
    def set_normalization(self, state_mean=None, state_std=None):
        if not self.cfg.normalize_states:
            return
        if state_mean is not None:
            self.norm_state_mean.copy_(torch.as_tensor(
                state_mean, dtype=torch.float32, device=self.device).flatten())
        if state_std is not None:
            std = torch.as_tensor(state_std, dtype=torch.float32, device=self.device).flatten()
            self.norm_state_std.copy_(std.clamp_min(1e-6))

    def _norm(self, state: torch.Tensor) -> torch.Tensor:
        if not self.cfg.normalize_states:
            return state
        return (state - self.norm_state_mean) / self.norm_state_std

    def _actor_forward(self, actor, state_norm):
        """state_norm already normalized -> tanh action (StateBcPolicy ends in Tanh)."""
        return actor({"state": state_norm})

    # --------------------------------------------------------------------- act
    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], *, eval_mode: bool = True,
            stddev: float = 0.0, cpu: bool = True) -> torch.Tensor:
        state = obs["state"].to(self.device).float()
        single = state.dim() == 1
        s = state.view(1, -1) if single else state
        action = self._actor_forward(self.actor, self._norm(s))
        if not eval_mode and stddev > 0:
            action = (action + torch.randn_like(action) * stddev).clamp(-1, 1)
        action = action[0] if single else action
        return action.cpu() if cpu else action

    # ------------------------------------------------------------------- update
    def _update_critic(self, batch) -> dict[str, float]:
        cfg = self.cfg
        s = self._norm(batch["states"])
        s_next = self._norm(batch["next_states"])
        a = batch["actions"]
        w = batch.get("weights")  # [B] per-sample gap weight; None => unweighted

        with torch.no_grad():
            next_a = self._actor_forward(self.actor_target, s_next)
            noise = (torch.randn_like(next_a) * cfg.policy_noise).clamp(
                -cfg.noise_clip, cfg.noise_clip)
            next_a = (next_a + noise).clamp(-1, 1)

            bc_penalty = ((next_a - batch["next_actions"]) ** 2).sum(-1)  # [B]
            # critic BC penalty (on a') — ALWAYS gap-weighted per-sample.
            bc_pen_w = bc_penalty if w is None else w * bc_penalty
            next_q = self.critic_target(s_next, next_a).min(0).values       # [B]
            next_q = next_q - cfg.critic_bc_coef * bc_pen_w
            target_q = batch["rewards"] + (1.0 - batch["dones"]) * cfg.gamma * next_q  # [B]

        q = self.critic(s, a)  # [num_critics, B]
        sq = (q - target_q.unsqueeze(0)) ** 2  # [num_critics, B]
        if w is not None and cfg.weight_td:
            # weighted-mean over batch (trust the Bellman backup less where the
            # source dynamics is unreliable), summed over the critic ensemble.
            critic_loss = (sq * w.unsqueeze(0)).sum(1).div(w.sum().clamp_min(1e-6)).sum(0)
        else:
            critic_loss = sq.mean(1).sum(0)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        return {
            "rebrac/critic_loss": critic_loss.item(),
            "rebrac/q_mean": q.mean().item(),
            "rebrac/target_q": target_q.mean().item(),
            "rebrac/critic_bc_penalty": bc_penalty.mean().item(),
        }

    def _update_actor(self, batch) -> dict[str, float]:
        cfg = self.cfg
        s = self._norm(batch["states"])
        a_data = batch["actions"]

        w = batch.get("weights")  # [B] per-sample gap weight; None => unweighted
        pi = self._actor_forward(self.actor, s)
        bc_penalty = ((pi - a_data) ** 2).sum(-1)  # [B]
        q = self.critic(s, pi).min(0).values        # [B]
        lmbda = 1.0
        if cfg.normalize_q:
            lmbda = (1.0 / q.abs().mean().detach()).clamp_max(1e6)
        # BC term = gap-weighted mean (matches train_weighted_bc.weighted_loss:
        # sum(w*e)/sum(w)); the RL Q term is a plain mean over the batch.
        if w is None:
            bc_term = bc_penalty.mean()
        else:
            bc_term = (w * bc_penalty).sum() / w.sum().clamp_min(1e-6)
        actor_loss = cfg.actor_bc_coef * bc_term - lmbda * q.mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        return {
            "rebrac/actor_loss": actor_loss.item(),
            "rebrac/actor_bc_penalty": bc_penalty.mean().item(),
            "rebrac/lambda": float(lmbda),
            "rebrac/q_for_actor": q.mean().item(),
        }

    def _soft_update(self):
        tau = self.cfg.tau
        with torch.no_grad():
            for src, tgt in [(self.actor, self.actor_target), (self.critic, self.critic_target)]:
                for p, pt in zip(src.parameters(), tgt.parameters()):
                    pt.mul_(1 - tau).add_(p, alpha=tau)

    def update(self, batch) -> dict[str, float]:
        """One ReBRAC step. `batch` = dict of cuda tensors with keys
        states/actions/rewards/next_states/next_actions/dones (see ReBRACOfflineDataset)."""
        metrics = self._update_critic(batch)
        # delayed actor + target update (TD3, policy_freq)
        if self._update_count % self.cfg.policy_freq == 0:
            metrics.update(self._update_actor(batch))
            self._soft_update()
        self._update_count += 1
        return metrics

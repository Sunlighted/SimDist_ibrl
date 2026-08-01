"""Faithful ("original") residual off-policy RL — a clean ResFiT port (RLPD-style).

Unlike `rl/residual_agent.py` (the sim2sim, GT-gap-weighted variant), this is the
straight ResFiT recipe (Ankile et al. 2025, ~/residual-offpolicy-rl) with the four
RLPD ingredients the gap-weighted version dropped — exactly the differences ①②③⑤
flagged in CLAUDE.md's residual-vs-ResFiT diff:

  ① Every RL minibatch mixes ONLINE + OFFLINE demo transitions (RLPD symmetric
     sampling; the demo half anchors Q on real returns). Handled by the trainer:
     it samples both pools and concatenates; the agent consumes the mixed batch.
  ② Deep **LayerNorm critic ensemble** with **REDQ**-style min over a RANDOM
     subset of `num_min_critics` heads (not the full pessimistic min).
  ③ base+noise exploration warmup + critic-only warmup (handled by the trainer).
  ⑤ Deterministic residual in the actor's Q-max, **TD3 target-policy smoothing**
     on the next residual, and **gradient clipping**.

NO dynamics-gap weighting — this is the in-domain ResFiT baseline (train + eval in
the SAME domain), used to validate the implementation: in-domain residual RL on a
frozen BC base should clearly beat the base BC.

Residual mechanism (identical to ResFiT / rl/residual_agent.py):
    a = clip( a_base(s) + action_scale * tanh(res(s, a_base)),  -1, 1 )
the residual actor's last layer is zero-init so delta=0 at step 0 (policy == base);
the critic learns Q(s, a) on the FULL action, the residual actor maximises Q.
State-only. The frozen base policy is fed RAW state (it was BC-trained on raw
state); the residual actor + critic use z-scored state.
"""

from dataclasses import dataclass
import copy

import torch
import torch.nn as nn

from rl.rebrac_agent import EnsembleQ


class ResidualActor(nn.Module):
    """delta(s, a_base) in [-action_scale, +action_scale]; zero-init last layer."""

    def __init__(self, obs_dim, action_dim, hidden, n_layer, layer_norm, action_scale):
        super().__init__()
        dims = [obs_dim + action_dim] + [hidden for _ in range(n_layer)]
        layers: list[nn.Module] = []
        for i in range(n_layer):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], action_dim)
        # zero-init => residual == 0 at step 0 => the policy IS the base policy.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.action_scale = action_scale

    def forward(self, state_norm: torch.Tensor, base_a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state_norm, base_a], dim=-1)
        return self.action_scale * torch.tanh(self.head(self.trunk(x)))


@dataclass
class ResidualRLConfig:
    base_policy_path: str = ""
    action_scale: float = 0.1
    # residual actor MLP
    actor_hidden_dim: int = 1024
    actor_num_layer: int = 3
    actor_layer_norm: int = 1
    # critic ensemble — REDQ: many heads, min over a random subset of num_min_critics
    critic_hidden_dim: int = 256
    critic_num_layer: int = 3
    critic_layer_norm: int = 1
    num_critics: int = 10
    num_min_critics: int = 2
    # optim (ResFiT defaults: tiny actor lr, normal critic lr)
    actor_lr: float = 1e-6
    critic_lr: float = 1e-4
    gamma: float = 0.99
    tau: float = 5e-3
    policy_freq: int = 2            # delayed actor + target update cadence (TD3)
    # TD3 target-policy smoothing on the NEXT residual (in residual space)
    policy_noise: float = 0.05
    noise_clip: float = 0.5
    # optional L2 penalty on residual magnitude in the actor loss (keep near base)
    residual_l2_coef: float = 0.0
    # clamp the Bellman target to [q_min,q_max]; with reward in {0,1}+end_on_success
    # the true Q is in [0,1], so clamping is UNBIASED (ResFiT clip_q_target_to_reward_range)
    clamp_q_target: int = 1
    q_target_min: float = 0.0
    q_target_max: float = 1.0
    grad_clip: float = 10.0        # 0 = off
    normalize_states: int = 1      # z-score state for residual actor + critic
    # Feature space for the residual actor + critic (the base ALWAYS gets raw state).
    #   0 (default) = z-scored raw state — the 37-d state is compact + complete.
    #   1           = the FROZEN SimDist encoder's latent z = encode(s).
    # Only motivated if the imagination/rollouts also live in latent; note encode()
    # is a deterministic function of s, so by the data-processing inequality it can
    # only LOSE information (and SimNorm actively discards). Measured offline
    # (scripts/diag_latent_qres_sensitivity.py, same data/recipe/eval, single seed):
    # latent IQL critic Spearman(V,-steps) +0.605 / Pearson(Q,g^steps) +0.495 vs
    # raw's +0.782 / +0.655 — i.e. latent looked WORSE as a value feature space.
    # This flag exists to test that end-to-end on the real residual loop.
    latent_features: int = 0
    # Latent width when latent_features=1 (SimDist WorldModel latent = 512). Must be
    # set at construction: the trainer builds the nets BEFORE attaching the base, so
    # it cannot be inferred from the encoder. Verified against the base on attach.
    latent_dim: int = 0
    # ResFiT progressive clipping: ramp the residual bound linearly from 0 to
    # action_scale over this many env steps, so the residual is forced ~0 early
    # (policy == base ⇒ eval starts at the base's score, no early dip) and only
    # grows as the critic calibrates. 0 = off (full action_scale from step 0).
    progressive_clipping_steps: int = 0
    # Actor Q-aggregation in the policy-improvement step (line 20):
    #   "min"  = min over a random subset of `num_min_critics` heads (pessimistic;
    #            resists OOD Q-overestimation — REQUIRED on precise-insertion tasks
    #            with a strong base, else the residual leaves base support and the
    #            policy collapses [observed 2026-07-07: ensemble-mean + scale 0.2
    #            collapsed to 0.23 mid-training]).
    #   "mean" = ensemble mean over ALL heads (ResFiT Algorithm-1 default /
    #            policy_gradient_type="ensemble_mean"; less pessimistic).
    # The critic TARGET always uses the min-subset (line 16) regardless.
    actor_q_agg: str = "min"
    # SGFT potential-based reward shaping on the critic TD target (§10): replace
    # the n-step reward r with r + coef*(gamma^n*V_src(s_{t+n}) - V_src(s_t)),
    # V_src = a frozen source QAgent's value(). Potential-based ⇒ preserves the
    # optimal policy (only densifies credit assignment). Default OFF (byte-
    # identical residual run). Requires the trainer to attach the value source
    # via attach_value_source(); the shift-aware clamp tracks the shaped range
    # Q'(x)=Q(x)-coef*V_src(x) per-state (else clamping the mostly-negative Q' to
    # [0,1] pins the critic at the floor — the exact bug that killed the
    # standalone SGFT-RLPD run).
    sgft_shaping: int = 0
    sgft_coef: float = 1.0
    # Phase-gated residual scale (§30 scale矛盾的解药): modulate the residual
    # magnitude by the needle-ring distance so the correction is SMALL in free
    # space (base already good; a big residual there leaves base support → OOD-Q
    # collapse, observed at fixed scale 0.2) and LARGE near contact (base fails
    # 75% at align/insert → the residual needs room). effective residual =
    # m(s)·action_scale·tanh(...), m = min + (max-min)·sigmoid((mid-d)/width),
    # d = ‖needle-ring‖ from the RAW state (Threading: state[9:12] vs [23:26];
    # measured d≈0.43 free → 0.094 contact). Default OFF (m≡1 ⇒ byte-identical).
    phase_gate: int = 0
    phase_gate_min: float = 0.5      # free-space multiplier (shrink residual)
    phase_gate_max: float = 1.5      # contact multiplier (grow residual; <2.0 to dodge §30 collapse)
    phase_gate_mid: float = 0.2      # needle-ring distance at the transition midpoint
    phase_gate_width: float = 0.06   # transition sharpness
    phase_needle_lo: int = 9         # raw-state needle-pos slice [lo, lo+3)
    phase_ring_lo: int = 23          # raw-state ring-pos slice [lo, lo+3)


class ResidualRLAgent(nn.Module):
    def __init__(self, obs_shape, action_dim, cfg: ResidualRLConfig, device: str = "cuda"):
        super().__init__()
        assert len(obs_shape) == 1, "state obs must be 1-D"
        self.cfg = cfg
        self.device = device
        obs_dim = int(obs_shape[0])
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # actor/critic input width: raw state, or the frozen encoder's latent
        if cfg.latent_features:
            assert cfg.latent_dim > 0, (
                "residual.latent_features=1 requires residual.latent_dim (the SimDist "
                "WorldModel latent width, 512) — the nets are built before the base "
                "is attached, so it cannot be inferred")
            feat_dim = int(cfg.latent_dim)
        else:
            feat_dim = obs_dim
        self.feat_dim = feat_dim

        self.actor = ResidualActor(
            feat_dim, action_dim, cfg.actor_hidden_dim, cfg.actor_num_layer,
            bool(cfg.actor_layer_norm), cfg.action_scale,
        )
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = EnsembleQ(
            feat_dim, action_dim, cfg.critic_hidden_dim, cfg.critic_num_layer,
            bool(cfg.critic_layer_norm), cfg.num_critics,
        )
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.actor_target.parameters():
            p.requires_grad_(False)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # state z-score buffers (identity until set_normalization)
        self.register_buffer("norm_state_mean", torch.zeros(obs_dim))
        self.register_buffer("norm_state_std", torch.ones(obs_dim))

        # frozen base policy, list-held => NOT in state_dict (re-attached at eval)
        self._base_policy: list[nn.Module] = []
        # frozen state->latent encoder (latent_features=1 only), list-held for the
        # same reason: it belongs to the frozen base, not to this agent's weights.
        self._encoder: list = []

        self.to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self._update_count = 0

    # -------------------------------------------------------------- featurizer
    def attach_encoder(self, fn):
        """Frozen state->latent map (e.g. the base SimDist WorldModel's `encode`).
        Required when latent_features=1; list-held so it stays out of state_dict."""
        self._encoder = [fn]

    # ------------------------------------------------------------ normalization
    def set_normalization(self, state_mean=None, state_std=None):
        # latent features are already bounded (SimNorm) and are not z-scored, so
        # the raw-state stats are meaningless here — no-op keeps the two arms'
        # call sites identical (the trainer calls this unconditionally).
        if self.cfg.latent_features:
            return
        if not self.cfg.normalize_states:
            return
        if state_mean is not None:
            self.norm_state_mean.copy_(torch.as_tensor(
                state_mean, dtype=torch.float32, device=self.device).flatten())
        if state_std is not None:
            std = torch.as_tensor(state_std, dtype=torch.float32, device=self.device).flatten()
            self.norm_state_std.copy_(std.clamp_min(1e-6))

    def _norm(self, state: torch.Tensor) -> torch.Tensor:
        """RAW state -> the feature the residual actor/critic consume. The single
        hook for the feature space: every actor/critic call site goes through it."""
        if self.cfg.latent_features:
            assert self._encoder, "latent_features=1 but no encoder attached"
            # frozen encoder => compute under no_grad; the returned feature is a
            # constant w.r.t. the actor/critic losses (no graph through the encoder).
            with torch.no_grad():
                return self._encoder[0](state)
        if not self.cfg.normalize_states:
            return state
        return (state - self.norm_state_mean) / self.norm_state_std

    # -------------------------------------------------------------------- base
    def attach_base_policy(self, module: nn.Module):
        module = module.to(self.device)
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)
        self._base_policy = [module]

    @torch.no_grad()
    def _base_action(self, state_raw: torch.Tensor) -> torch.Tensor:
        """Deterministic base action a_base(s) in [-1,1] from RAW state (the base
        BC policy was trained on raw, un-normalized state)."""
        assert self._base_policy, "base policy not attached"
        a = self._base_policy[0]({"state": state_raw})
        if isinstance(a, (tuple, list)):
            a = a[0]
        if a.dim() == 3:  # a chunk policy returns [B,T,A]; take the first step
            a = a[:, 0]
        return a.clamp(-1.0, 1.0)

    def _min_q(self, q_all: torch.Tensor) -> torch.Tensor:
        """REDQ: min over a RANDOM subset of num_min_critics heads. q_all [K,B]."""
        k = q_all.shape[0]
        n = min(self.cfg.num_min_critics, k)
        if n >= k:
            return q_all.min(0).values
        idx = torch.randperm(k, device=q_all.device)[:n]
        return q_all.index_select(0, idx).min(0).values

    # --------------------------------------------------------------------- act
    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], *, eval_mode: bool = False,
            stddev: float = 0.0, cpu: bool = True) -> torch.Tensor:
        state = obs["state"]
        single = state.dim() == 1
        s_raw = (state.view(1, -1) if single else state).to(self.device).float()
        base_a = self._base_action(s_raw)
        res = self.actor(self._norm(s_raw), base_a)
        if (not eval_mode) and stddev > 0:
            res = (res + torch.randn_like(res) * stddev).clamp(
                -self.cfg.action_scale, self.cfg.action_scale)
        action = (base_a + res).clamp(-1.0, 1.0)
        action = action[0] if single else action
        return action.cpu() if cpu else action

    # ------------------------------------------------------------------ update
    def _clamp_q(self, q: torch.Tensor) -> torch.Tensor:
        if self.cfg.clamp_q_target:
            return q.clamp(self.cfg.q_target_min, self.cfg.q_target_max)
        return q

    def update(self, batch, update_actor: bool) -> dict[str, float]:
        """One TD3 step on a MIXED (online + offline-demo) batch. `batch` keys:
        states / actions(full executed) / rewards / next_states / dones (cuda)."""
        cfg = self.cfg
        s_raw = batch["states"]
        s2_raw = batch["next_states"]
        s = self._norm(s_raw)
        s2 = self._norm(s2_raw)
        a = batch["actions"]
        r = batch["rewards"]
        done = batch["dones"]
        metrics: dict[str, float] = {}

        # ---- critic: TD3 target with target-policy smoothing on the residual ----
        with torch.no_grad():
            base_a2 = self._base_action(s2_raw)
            res2 = self.actor_target(s2, base_a2)
            noise = (torch.randn_like(res2) * cfg.policy_noise).clamp(
                -cfg.noise_clip, cfg.noise_clip)
            res2 = (res2 + noise).clamp(-cfg.action_scale, cfg.action_scale)
            next_action = (base_a2 + res2).clamp(-1.0, 1.0)
            q_next = self._min_q(self.critic_target(s2, next_action))  # [B]
            y = r + cfg.gamma * (1.0 - done) * q_next
            y = self._clamp_q(y)

        q = self.critic(s, a)  # [K, B]
        critic_loss = ((q - y.unsqueeze(0)) ** 2).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
        self.critic_opt.step()

        metrics["residual/critic_loss"] = critic_loss.item()
        metrics["residual/q_mean"] = q.mean().item()
        metrics["residual/td_target"] = y.mean().item()
        metrics["data/batch_R"] = r.mean().item()

        # ---- delayed residual actor: maximise Q(s, clip(base + delta)) ----
        if update_actor and (self._update_count % cfg.policy_freq == 0):
            base_a = self._base_action(s_raw)
            res = self.actor(s, base_a)  # deterministic (no exploration noise)
            a_pi = (base_a + res).clamp(-1.0, 1.0)
            q_pi = self._min_q(self.critic(s, a_pi))
            actor_loss = -q_pi.mean()
            if cfg.residual_l2_coef > 0:
                l2 = (res ** 2).sum(-1).mean()
                actor_loss = actor_loss + cfg.residual_l2_coef * l2
                metrics["residual/delta_l2"] = l2.item()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
            self.actor_opt.step()
            self._soft_update()
            metrics["residual/actor_loss"] = actor_loss.item()
            metrics["residual/delta_abs"] = res.abs().mean().item()

        self._update_count += 1
        return metrics

    def _soft_update(self):
        tau = self.cfg.tau
        with torch.no_grad():
            for src, tgt in [(self.actor, self.actor_target), (self.critic, self.critic_target)]:
                for p, pt in zip(src.parameters(), tgt.parameters()):
                    pt.mul_(1 - tau).add_(p, alpha=tau)

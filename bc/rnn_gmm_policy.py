"""State-based BC-RNN with GMM head — MimicGen-style baseline for ibrl.

Mirrors robomimic's BC_RNN_GMM (algo/bc.py + models/policy_nets.py
RNNGMMActorNetwork) along the dimensions that actually matter for behavior:
  - LSTM backbone unrolling pred_horizon steps
  - K-mode Gaussian Mixture head per timestep with `low_noise_eval`
  - tanh-squashed means → actions roughly in [-1, 1]

Two deliberate simplifications vs robomimic:
  1. **Open-loop unroll**: the LSTM is fed the SAME encoded obs at every of
     its T input steps (robomimic's `rnn.open_loop=True` regime). This lets
     us emit a full action chunk from a single observation, which fits the
     existing chunked-eval infrastructure (evaluate/multi_process_eval.py)
     with zero changes. The trade-off vs closed-loop RNN is small for the
     short chunks (T<=16) used here.
  2. **No goal conditioning** — ibrl's BC datasets are not goal-conditioned.

Loss:    log_prob of action_chunk under per-timestep GMM, summed over T,
         negated and averaged over batch.
Inference: sample (eval picks the dominant mode via low_noise scales),
           return the first Ta actions of the predicted chunk.

Public surface matches StateFlowPolicy where compatible:
    loss(batch)                  -> scalar
    act(obs, eval_mode=True)     -> (B, D) or (D,)  — back-compat
    act_chunk(obs, eval_mode=True)  -> (B, Ta, D) or (Ta, D)
    update_ema()                 — call after every optim.step()
    apply_ema() / restore_from_ema()
"""

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D


@dataclass
class StateRnnGmmPolicyConfig:
    # Obs encoder before LSTM (matches robomimic actor_layer_dims=[] —
    # i.e. the LSTM gets the raw encoded obs without an MLP between).
    # Setting num_mlp_layer > 0 adds MLP layers between obs and LSTM.
    obs_mlp_hidden: int = 0       # 0 = no obs MLP; LSTM consumes raw state
    num_mlp_layer: int = 0
    # LSTM
    rnn_hidden_dim: int = 400     # robomimic default for low_dim
    rnn_num_layers: int = 2
    # GMM
    num_modes: int = 5
    min_std: float = 0.01
    std_activation: str = "softplus"  # softplus | exp
    low_noise_eval: int = 1            # if 1, scales clamped to 1e-4 at eval
    use_tanh: int = 1                  # tanh-squash means (low_noise=1 effectively
                                       # makes actions ≈ tanh(means))
    # Horizons (must match dataset.action_horizon)
    pred_horizon: int = 10             # T — chunk length predicted/trained
    action_horizon: int = 8            # Ta — chunk length executed per re-prediction
    # EMA — disabled by default (robomimic BC-RNN-GMM doesn't use it).
    # Keep the hooks so train_bc.py's `if hasattr(policy, 'update_ema')`
    # branch is satisfied either way.
    use_ema: int = 0
    ema_decay: float = 0.9999


class StateRnnGmmPolicy(nn.Module):
    def __init__(self, obs_shape, action_dim: int, cfg: StateRnnGmmPolicyConfig):
        super().__init__()
        assert len(obs_shape) == 1, f"StateRnnGmmPolicy expects flat state, got {obs_shape}"
        assert cfg.action_horizon <= cfg.pred_horizon
        assert cfg.std_activation in ("softplus", "exp")
        self.cfg = cfg
        self.state_dim = obs_shape[0]
        self.action_dim = action_dim
        self.K = cfg.num_modes
        self.Tp = cfg.pred_horizon
        self.Ta = cfg.action_horizon

        # ---- obs encoder ----
        if cfg.num_mlp_layer == 0:
            self.obs_enc = nn.Identity()
            obs_feat_dim = self.state_dim
        else:
            layers, prev = [], self.state_dim
            for _ in range(cfg.num_mlp_layer):
                layers += [nn.Linear(prev, cfg.obs_mlp_hidden), nn.ReLU()]
                prev = cfg.obs_mlp_hidden
            self.obs_enc = nn.Sequential(*layers)
            obs_feat_dim = cfg.obs_mlp_hidden

        # ---- LSTM (open-loop: same obs_feat fed at every of T steps) ----
        self.lstm = nn.LSTM(
            input_size=obs_feat_dim,
            hidden_size=cfg.rnn_hidden_dim,
            num_layers=cfg.rnn_num_layers,
            batch_first=True,
        )

        # ---- GMM head: per-timestep linear to (mean, scale, logits) ----
        # output sizes: K*D mean + K*D scale + K logits
        self.head = nn.Linear(
            cfg.rnn_hidden_dim, self.K * self.action_dim * 2 + self.K
        )

        # ---- EMA shadow (lazy init; mirrors flow_policy convention) ----
        self._ema_initialized = False
        self._ema_state: "dict[str, torch.Tensor]" = {}
        self._ema_backup: "dict[str, torch.Tensor]" = {}

    # =================================================================
    # core forward → GMM distribution over T timesteps
    # =================================================================

    def _gmm_dist(self, state: torch.Tensor) -> D.MixtureSameFamily:
        """Forward state through obs_enc + LSTM + GMM head.

        state: (B, state_dim) — single obs (open-loop, broadcast over Tp).
        Returns a torch Distribution with batch_shape (B, Tp), event_shape (D,).
        """
        B = state.shape[0]
        obs_feat = self.obs_enc(state)                              # (B, H_in)
        # repeat obs_feat over T timesteps; LSTM unrolls Tp steps with the
        # same input each step — open-loop "memorize trajectory shape"
        seq_in = obs_feat.unsqueeze(1).expand(B, self.Tp, -1).contiguous()
        h, _ = self.lstm(seq_in)                                    # (B, Tp, H_rnn)
        raw = self.head(h)                                          # (B, Tp, K*2D+K)
        means_flat, scales_flat, logits = torch.split(
            raw,
            [self.K * self.action_dim, self.K * self.action_dim, self.K],
            dim=-1,
        )
        means = means_flat.view(B, self.Tp, self.K, self.action_dim)
        scales = scales_flat.view(B, self.Tp, self.K, self.action_dim)

        if self.cfg.use_tanh:
            means = torch.tanh(means)

        if self.cfg.low_noise_eval and (not self.training):
            scales = torch.ones_like(means) * 1e-4
        elif self.cfg.std_activation == "softplus":
            scales = F.softplus(scales) + self.cfg.min_std
        else:  # exp
            scales = torch.exp(scales) + self.cfg.min_std

        component = D.Independent(D.Normal(means, scales), 1)        # (B, Tp, K), event D
        mixture = D.Categorical(logits=logits)                       # (B, Tp, K)
        return D.MixtureSameFamily(mixture, component)               # (B, Tp), event D

    def loss(self, batch):
        """NLL of action_chunk under per-timestep GMM."""
        state = batch.obs["state"]                                   # (B, S)
        # DatasetConfig.action_horizon == 1 stores a plain "action" (bc/dataset.py
        # only builds "action_chunk" when H > 1). A Tp=1 policy is the single-step
        # prior IBRL actually queries, so accept that layout too.
        a = batch.action.get("action_chunk")
        if a is None:
            a = batch.action["action"].unsqueeze(1)                  # (B, 1, D)
        assert a.shape[1] == self.Tp, (
            f"action_chunk Tp={a.shape[1]} doesn't match policy pred_horizon={self.Tp} "
            "— check DatasetConfig.action_horizon"
        )
        assert a.shape[2] == self.action_dim

        dist = self._gmm_dist(state)
        log_probs = dist.log_prob(a)                                 # (B, Tp)
        # Match flow_policy convention: sum over time, mean over batch
        return -log_probs.sum(dim=-1).mean()

    # =================================================================
    # eval-facing inference
    # =================================================================

    @torch.no_grad()
    def _sample_full_chunk(self, state: torch.Tensor) -> torch.Tensor:
        """Sample a full chunk (B, Tp, D) — picks GMM mode under low_noise_eval."""
        dist = self._gmm_dist(state)
        return dist.sample()                                         # (B, Tp, D)

    def act_chunk(self, obs, *, eval_mode=True):
        """Return (B, Ta, D) or (Ta, D) — first Ta actions of the chunk."""
        assert eval_mode
        assert not self.training
        state = obs["state"]
        unsqueezed = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            unsqueezed = True
        chunk = self._sample_full_chunk(state)[:, : self.Ta].contiguous()
        if unsqueezed:
            chunk = chunk.squeeze(0)
        return chunk.cpu()

    def act(self, obs, *, eval_mode=True, cpu=True):
        """Back-compat single-action path; returns first action of fresh chunk."""
        assert eval_mode
        assert not self.training
        state = obs["state"]
        unsqueezed = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            unsqueezed = True
        chunk = self._sample_full_chunk(state)
        action = chunk[:, 0]
        if unsqueezed:
            action = action.squeeze(0)
        if cpu:
            action = action.cpu()
        return action

    # =================================================================
    # EMA hooks (identical layout to StateFlowPolicy)
    # =================================================================

    def _init_ema(self) -> None:
        for name, p in self.named_parameters():
            self._ema_state[name] = p.data.detach().clone()
        self._ema_initialized = True

    @torch.no_grad()
    def update_ema(self) -> None:
        if not self.cfg.use_ema:
            return
        if not self._ema_initialized:
            self._init_ema()
            return
        d = self.cfg.ema_decay
        for name, p in self.named_parameters():
            self._ema_state[name].mul_(d).add_(p.data, alpha=1.0 - d)

    @torch.no_grad()
    def apply_ema(self) -> bool:
        if not self.cfg.use_ema or not self._ema_initialized:
            return False
        for name, p in self.named_parameters():
            self._ema_backup[name] = p.data.detach().clone()
            p.data.copy_(self._ema_state[name])
        return True

    @torch.no_grad()
    def restore_from_ema(self) -> None:
        if not self._ema_backup:
            return
        for name, p in self.named_parameters():
            if name in self._ema_backup:
                p.data.copy_(self._ema_backup[name])
        self._ema_backup.clear()

    # =================================================================
    # state_dict / load_state_dict: persist EMA shadow alongside params
    # =================================================================
    _EMA_PREFIX = "_ema_shadow__"

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        sd = super().state_dict(
            *args, destination=destination, prefix=prefix, keep_vars=keep_vars,
        )
        if self._ema_initialized:
            for name, t in self._ema_state.items():
                sd[f"{prefix}{self._EMA_PREFIX}{name}"] = t.detach().clone()
        return sd

    def load_state_dict(self, state_dict, strict=True):
        state_dict = dict(state_dict)
        ema_keys = [k for k in state_dict if self._EMA_PREFIX in k]
        new_ema: "dict[str, torch.Tensor]" = {}
        for k in ema_keys:
            param_name = k.split(self._EMA_PREFIX, 1)[1]
            new_ema[param_name] = state_dict.pop(k).clone()
        ret = super().load_state_dict(state_dict, strict=strict)
        if new_ema:
            device = next(self.parameters()).device
            self._ema_state = {n: t.to(device) for n, t in new_ema.items()}
            self._ema_initialized = True
            self._ema_backup = {}
        return ret

"""Deterministic MLP behavior-cloning policy that predicts an ACTION CHUNK.

This is the chunked counterpart of `StateBcPolicy` (single-step MLP regression):
from one observation it regresses the next `pred_horizon` (Tp) actions at once
and executes the first `action_horizon` (Ta) per re-prediction — same chunking /
open-loop execution convention as the flow / diffusion / rnn heads, but with a
plain deterministic MLP and a direct MSE loss (NO generative sampling, NO GMM).

Lets us isolate the effect of *action chunking* alone, holding the model class
fixed at "simple deterministic MLP": StateBcPolicy (Ta=1, closed-loop) vs this
(Ta>1, open-loop chunk).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from common_utils import ibrl_utils as utils


def _build_fc(in_dim, hidden_dim, out_dim, num_layer, layer_norm, dropout):
    dims = [in_dim] + [hidden_dim for _ in range(num_layer)]
    layers: list[nn.Module] = []
    for i in range(num_layer):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if layer_norm == 1:
            layers.append(nn.LayerNorm(dims[i + 1]))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-1], out_dim))
    layers.append(nn.Tanh())
    return nn.Sequential(*layers)


@dataclass
class StateChunkMlpPolicyConfig:
    num_layer: int = 3
    hidden_dim: int = 1024
    dropout: float = 0.0
    layer_norm: int = 0
    pred_horizon: int = 16     # Tp — chunk length predicted/trained; must match dataset.action_horizon
    action_horizon: int = 8    # Ta — chunk length executed per re-prediction; Ta <= Tp


class StateChunkMlpPolicy(nn.Module):
    def __init__(self, obs_shape, action_dim, cfg: StateChunkMlpPolicyConfig):
        super().__init__()
        assert len(obs_shape) == 1
        assert cfg.action_horizon <= cfg.pred_horizon
        self.cfg = cfg
        self.action_dim = action_dim
        self.Tp = cfg.pred_horizon
        self.Ta = cfg.action_horizon
        self.net = _build_fc(
            obs_shape[0], cfg.hidden_dim, action_dim * cfg.pred_horizon,
            cfg.num_layer, cfg.layer_norm, cfg.dropout,
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        flat = self.net(obs["state"])                       # (B, Tp*D)
        return flat.view(-1, self.Tp, self.action_dim)      # (B, Tp, D)

    def loss(self, batch):
        pred = self.forward(batch.obs)                      # (B, Tp, D)
        target = batch.action["action_chunk"]               # (B, Tp, D)
        assert target.shape[1] == self.Tp, (
            f"action_chunk Tp={target.shape[1]} != pred_horizon={self.Tp} "
            "— check DatasetConfig.action_horizon"
        )
        return nn.functional.mse_loss(pred, target)         # mean reduction (matches flow/diffusion)

    @torch.no_grad()
    def act_chunk(self, obs, *, eval_mode=True):
        """Return (B, Ta, D) (or (Ta, D) for a 1-env obs) on CPU."""
        assert eval_mode and not self.training
        state = obs["state"]
        unsqueezed = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            unsqueezed = True
        chunk = self.net(state).view(-1, self.Tp, self.action_dim)
        chunk_exec = chunk[:, : self.Ta].contiguous()
        if unsqueezed:
            chunk_exec = chunk_exec.squeeze(0)
        return chunk_exec.cpu()

    @torch.no_grad()
    def act(self, obs, *, eval_mode=True, cpu=True):
        """Back-compat single action (first of the fresh chunk)."""
        assert eval_mode and not self.training
        state = obs["state"]
        unsqueezed = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            unsqueezed = True
        action = self.net(state).view(-1, self.Tp, self.action_dim)[:, 0]
        if unsqueezed:
            action = action.squeeze(0)
        if cpu:
            action = action.cpu()
        return action

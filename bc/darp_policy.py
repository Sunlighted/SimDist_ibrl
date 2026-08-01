"""DARP: retrieval-conditioned BC policy (port of /home/Chenyang/darp).

For a query state s_q, retrieve the K nearest neighbors from a fixed POOL of
(state, action) transitions (KNN in z-scored state space), and for each
neighbor j feed

    [ s_j_norm , a_j , (s_j_norm - s_q_norm) ]            # neighbor + state-diff

through a shared MLP -> â_j, then output the MEAN over the K neighbors. The
third block (neighbor_obs - query_obs) is DARP's "delta_state" feature: in the
standard (non-mixed) setting the delta_state dataset == the obs, so the model
sees how far each retrieved neighbor is from the query and can correct that
neighbor's action accordingly. (This is a faithful port of darp_wrapper.py's
forward with lookback=1 "simple KNN" retrieval, which the DARP README/paper
highlights as already strong; lookback>1 cumulative-distance retrieval would
need per-episode query history at inference and is asserted off here.)

The number of neighbors averaged is `num_neighbors = floor(k * final_neighbors_ratio)`
(at lookback=1 the cumulative distance == the single-step distance, so this is
simply the `num_neighbors` nearest). The retrieval POOL is set after dataset
construction via `set_pool(states, actions)` — for the gap-weighted / co-train
runs it is the SOURCE transitions (the large pool to retrieve from); for a
target-only run it is the target transitions.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class StateDARPPolicyConfig:
    # retrieval (DARP README hyperparameters)
    k: int = 100                       # candidate neighbors (KNN)
    final_neighbors_ratio: float = 0.5  # keep floor(k*ratio) closest -> num_neighbors
    lookback: int = 1                  # only 1 supported (no inference history needed)
    decay_rate: float = 0.0            # unused at lookback=1
    # per-neighbor MLP
    num_layer: int = 3
    hidden_dim: int = 1024
    dropout: float = 0.0
    layer_norm: int = 0


def _build_fc(in_dim, hidden_dim, out_dim, num_layer, layer_norm, dropout):
    dims = [in_dim] + [hidden_dim] * num_layer
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


class StateDARPPolicy(nn.Module):
    def __init__(self, obs_shape, action_dim, cfg: StateDARPPolicyConfig):
        super().__init__()
        assert len(obs_shape) == 1
        assert cfg.lookback == 1, (
            "StateDARPPolicy only supports lookback=1 (simple KNN); lookback>1 "
            "needs per-episode query history at inference (not wired into the "
            "stateless multi-process eval).")
        self.cfg = cfg
        self.obs_dim = obs_shape[0]
        self.action_dim = action_dim
        self.num_neighbors = max(1, math.floor(cfg.k * cfg.final_neighbors_ratio))
        self.candidates = cfg.k
        # per-neighbor MLP input = [neighbor_obs(d), neighbor_act(da), diff(d)]
        in_dim = 2 * self.obs_dim + action_dim
        self.net = _build_fc(in_dim, cfg.hidden_dim, action_dim,
                             cfg.num_layer, cfg.layer_norm, cfg.dropout)
        # retrieval pool (registered as buffers so they persist in the ckpt,
        # move with .to(), and ride along to multi-process eval workers).
        self.register_buffer("pool_obs", torch.zeros(1, self.obs_dim))      # raw
        self.register_buffer("pool_act", torch.zeros(1, action_dim))
        self.register_buffer("pool_obs_norm", torch.zeros(1, self.obs_dim))  # z-scored
        self.register_buffer("obs_mean", torch.zeros(self.obs_dim))
        self.register_buffer("obs_std", torch.ones(self.obs_dim))
        self._pool_set = False

    def set_pool(self, states: torch.Tensor, actions: torch.Tensor):
        """Install the retrieval pool (N transitions). states (N,d), actions (N,da)."""
        states = states.float()
        actions = actions.float()
        assert states.shape[1] == self.obs_dim and actions.shape[1] == self.action_dim
        mean = states.mean(0)
        std = states.std(0).clamp_min(1e-6)
        dev = self.obs_mean.device
        self.obs_mean = mean.to(dev)
        self.obs_std = std.to(dev)
        self.pool_obs = states.to(dev)
        self.pool_act = actions.to(dev)
        self.pool_obs_norm = ((states.to(dev) - self.obs_mean) / self.obs_std)
        self.num_neighbors = min(self.num_neighbors, self.pool_obs.shape[0])
        self._pool_set = True
        print(f"[DARP] pool set: N={self.pool_obs.shape[0]} transitions, "
              f"k={self.candidates}, num_neighbors={self.num_neighbors}")

    def _retrieve_and_predict(self, query_obs: torch.Tensor, exclude_self: bool) -> torch.Tensor:
        """query_obs (B, d) -> predicted action (B, da)."""
        assert self._pool_set, "call set_pool() before using StateDARPPolicy"
        B = query_obs.shape[0]
        qn = (query_obs - self.obs_mean) / self.obs_std            # (B, d)
        # squared euclidean distance to every pool entry (z-scored space)
        # dist2 = |q|^2 - 2 q.p + |p|^2 ; |q|^2 const per row -> ignore for topk
        pn = self.pool_obs_norm                                    # (N, d)
        dist2 = (qn * qn).sum(1, keepdim=True) - 2.0 * (qn @ pn.t()) + (pn * pn).sum(1)
        if exclude_self:
            dist2 = dist2.masked_fill(dist2 <= 1e-12, float("inf"))  # drop self / exact dups
        M = self.num_neighbors
        _, idx = torch.topk(dist2, k=M, largest=False, dim=1)       # (B, M)
        nb_obs_norm = pn[idx]                                       # (B, M, d)
        nb_act = self.pool_act[idx]                                 # (B, M, da)
        diff = nb_obs_norm - qn.unsqueeze(1)                        # (B, M, d)
        inp = torch.cat([nb_obs_norm, nb_act, diff], dim=-1)        # (B, M, 2d+da)
        a = self.net(inp.reshape(B * M, -1)).reshape(B, M, self.action_dim)
        return a.mean(dim=1)                                        # (B, da)

    def load_state_dict(self, state_dict, strict=True):
        # The pool buffers are (1, d) at __init__ but (N, d) in a trained ckpt;
        # resize them to the saved shapes before loading so standalone eval
        # gets the persisted retrieval pool (no dataset needed at load time).
        for name in ("pool_obs", "pool_act", "pool_obs_norm"):
            if name in state_dict:
                saved = state_dict[name]
                if getattr(self, name).shape != saved.shape:
                    setattr(self, name, saved.clone().to(getattr(self, name).device))
        self.num_neighbors = min(self.num_neighbors, self.pool_obs.shape[0])
        self._pool_set = True
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, obs: dict) -> torch.Tensor:
        # exclude self when training (query may be a pool member); not at eval.
        return self._retrieve_and_predict(obs["state"], exclude_self=self.training)

    def loss(self, batch):
        pred = self.forward(batch.obs)
        action = batch.action["action"]
        loss = nn.functional.mse_loss(pred, action, reduction="none")
        return loss.sum(1).mean(0)

    @torch.no_grad()
    def act(self, obs: dict, *, eval_mode=True, cpu=True):
        assert eval_mode and not self.training
        state = obs["state"]
        unsqueezed = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            unsqueezed = True
        action = self._retrieve_and_predict(state, exclude_self=False)
        if unsqueezed:
            action = action.squeeze(0)
        return action.cpu() if cpu else action

    @torch.no_grad()
    def act_chunk(self, obs, *, eval_mode=True):
        """Length-1 chunk for the chunked-eval framework (Ta=1)."""
        assert eval_mode and not self.training
        state = obs["state"]
        unsqueezed = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            unsqueezed = True
        action = self._retrieve_and_predict(state, exclude_self=False)  # (B, da)
        chunk = action.unsqueeze(1)                                     # (B, 1, da)
        if unsqueezed:
            chunk = chunk.squeeze(0)
        return chunk.cpu()

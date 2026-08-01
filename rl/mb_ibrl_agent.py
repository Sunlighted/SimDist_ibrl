"""Model-Based IBRL (MBPO-on-IBRL) — §23.

A Dyna-style model-based extension of the base IBRL `QAgent` for faster
convergence / better single-domain sample efficiency. The IBRL critic/actor/
BC-prior/`_act_ibrl` max-Q machinery is REUSED unchanged (this module subclasses
`QAgent`); we only add:

  * `DynamicsEnsemble` — a probabilistic state-space ensemble that predicts
    (s, a) -> (Δs as a Gaussian, reward, done). Trained on the agent's own real
    replay (MBPO).
  * `FlatReplayBuffer` — a tiny batched GPU ring buffer of single-step
    (s, a, r, s', bootstrap) used for both the real transitions the model is fit
    on AND the synthetic transitions the model generates.
  * `MBIBRLAgent.generate_rollouts` — short synthetic rollouts branched from
    real states, actions chosen by the SAME IBRL policy (`self.act`), next
    states/rewards from the ensemble. The synthetic transitions feed the IBRL
    critic/actor update at a high update-to-data ratio.

State-only (use_state=1, state_stack=1). The model predicts a NORMALIZED
next-state delta (z-scored by replay stats) so the heterogeneous 37-dim state
scales don't dominate the NLL.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.q_agent import QAgent, QAgentConfig
from rl.replay import Batch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class DynamicsEnsembleConfig:
    ensemble_size: int = 5
    num_elites: int = 5
    hidden_dim: int = 256
    num_layer: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-5
    logvar_min: float = -10.0
    logvar_max: float = 0.5
    # sparse-success tasks have very few r>0 transitions; upweight them so the
    # reward head doesn't collapse to predicting 0 everywhere.
    reward_pos_weight: float = 10.0
    predict_done: int = 1
    grad_clip: float = 0.0


# ---------------------------------------------------------------------------
# Ensemble linear layer: E parallel models, weight [E, in, out].
# ---------------------------------------------------------------------------
class EnsembleLinear(nn.Module):
    def __init__(self, ensemble_size: int, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, 1, out_dim))
        for e in range(ensemble_size):
            nn.init.trunc_normal_(self.weight[e], std=1.0 / (2.0 * in_dim ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [E, B, in]
        return torch.bmm(x, self.weight) + self.bias  # [E, B, out]


# ---------------------------------------------------------------------------
# Probabilistic state-space dynamics ensemble (PETS / MBPO style).
# ---------------------------------------------------------------------------
class DynamicsEnsemble(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, cfg: DynamicsEnsembleConfig, device="cuda"):
        super().__init__()
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        E = cfg.ensemble_size

        in_dim = obs_dim + action_dim
        # output = Δs_mean[obs] + Δs_logvar[obs] + reward[1] + done_logit[1]
        out_dim = 2 * obs_dim + 2
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(cfg.num_layer):
            layers.append(EnsembleLinear(E, dim, cfg.hidden_dim))
            layers.append(nn.SiLU())
            dim = cfg.hidden_dim
        self.trunk = nn.ModuleList(layers)
        self.head = EnsembleLinear(E, dim, out_dim)

        # learnable per-dim logvar bounds (PETS) — registered as params so they
        # are trained toward the data scale within the soft-clamp.
        self.max_logvar = nn.Parameter(torch.full((1, 1, obs_dim), cfg.logvar_max))
        self.min_logvar = nn.Parameter(torch.full((1, 1, obs_dim), cfg.logvar_min))

        # state normalization (identity until set_normalization). Buffers so they
        # persist in state_dict and move with .to().
        self.register_buffer("norm_state_mean", torch.zeros(obs_dim))
        self.register_buffer("norm_state_std", torch.ones(obs_dim))
        # which members to sample from at predict time (top-`num_elites`).
        self.register_buffer("elites", torch.arange(min(cfg.num_elites, E)))

        self.opt = torch.optim.AdamW(self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.to(device)

    # --- normalization ---
    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        self.norm_state_mean.copy_(mean.to(self.device).reshape(-1))
        self.norm_state_std.copy_(std.to(self.device).reshape(-1).clamp_min(1e-6))

    def _norm(self, s: torch.Tensor) -> torch.Tensor:
        return (s - self.norm_state_mean) / self.norm_state_std

    # --- forward ---
    def _net(self, inp_EB: torch.Tensor):
        """inp_EB: [E, B, obs+act] -> split outputs (all [E, B, ...])."""
        x = inp_EB
        for layer in self.trunk:
            x = layer(x)
        out = self.head(x)
        d = self.obs_dim
        delta_mean = out[..., :d]
        logvar = out[..., d:2 * d]
        reward = out[..., 2 * d]
        done_logit = out[..., 2 * d + 1]
        # soft-clamp logvar to learnable bounds (PETS).
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return delta_mean, logvar, reward, done_logit

    def _make_input(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """s,a: [B, .] (shared across members) -> [E, B, obs+act]."""
        inp = torch.cat([self._norm(s), a], dim=-1)  # [B, in]
        return inp.unsqueeze(0).expand(self.cfg.ensemble_size, -1, -1)

    # --- training ---
    def train_step(self, s, a, r, s2, done) -> dict:
        """One grad step. Each member sees an independent bootstrap resample of
        the batch for ensemble diversity. Δs is learned in normalized space."""
        cfg = self.cfg
        E = cfg.ensemble_size
        B = s.shape[0]
        # per-member bootstrap indices [E, B]
        idx = torch.randint(0, B, (E, B), device=self.device)
        s_e = s[idx]            # [E, B, obs]
        a_e = a[idx]            # [E, B, act]
        s2_e = s2[idx]
        r_e = r[idx]            # [E, B]
        done_e = done[idx].float()

        inp = torch.cat([self._norm(s_e), a_e], dim=-1)  # [E, B, in]
        delta_mean, logvar, reward_pred, done_logit = self._net(inp)

        # normalized-delta target
        target_delta = (s2_e - s_e) / self.norm_state_std
        inv_var = torch.exp(-logvar)
        nll = 0.5 * (inv_var * (target_delta - delta_mean) ** 2 + logvar)
        dyn_loss = nll.sum(-1).mean()  # sum over obs dim, mean over E,B
        # logvar bound regularizer (PETS): keep the soft bounds tight.
        bound_loss = 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())

        # reward: weighted MSE (upweight rare r>0).
        rw = 1.0 + cfg.reward_pos_weight * (r_e > 0).float()
        reward_loss = (rw * (reward_pred - r_e) ** 2).mean()

        loss = dyn_loss + bound_loss + reward_loss
        done_loss_val = 0.0
        if cfg.predict_done:
            done_loss = F.binary_cross_entropy_with_logits(done_logit, done_e)
            loss = loss + done_loss
            done_loss_val = done_loss.item()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.parameters(), cfg.grad_clip)
        self.opt.step()

        return {
            "model/loss": loss.item(),
            "model/dyn_nll": dyn_loss.item(),
            "model/reward_loss": reward_loss.item(),
            "model/done_loss": done_loss_val,
            "model/logvar_mean": logvar.mean().item(),
        }

    # --- prediction ---
    @torch.no_grad()
    def predict(self, s: torch.Tensor, a: torch.Tensor, sample: bool = True):
        """Returns (next_state[B,obs], reward[B], done[B] bool). For each input a
        random elite member is selected; Δs is sampled from its Gaussian."""
        inp = self._make_input(s, a)  # [E, B, in]
        delta_mean, logvar, reward_pred, done_logit = self._net(inp)
        B = s.shape[0]
        # pick a random elite member per sample
        elite = self.elites
        midx = elite[torch.randint(0, elite.numel(), (B,), device=self.device)]  # [B]
        ar = torch.arange(B, device=self.device)
        dm = delta_mean[midx, ar]      # [B, obs]
        lv = logvar[midx, ar]          # [B, obs]
        rp = reward_pred[midx, ar]     # [B]
        dl = done_logit[midx, ar]      # [B]

        if sample:
            std = torch.exp(0.5 * lv)
            delta = dm + std * torch.randn_like(dm)
        else:
            delta = dm
        next_state = s + delta * self.norm_state_std  # denormalize the delta
        done = torch.sigmoid(dl) > 0.5 if self.cfg.predict_done else torch.zeros(B, dtype=torch.bool, device=self.device)
        return next_state, rp, done


# ---------------------------------------------------------------------------
# Flat batched GPU ring buffer of single-step (s, a, r, s', bootstrap).
# Used for both the real transitions the model is fit on and the synthetic
# transitions it generates. `sample` returns a state-only `Batch` (compatible
# with QAgent's state-path update and with Batch.merge_batches).
# ---------------------------------------------------------------------------
class FlatReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device="cuda"):
        self.capacity = capacity
        self.device = device
        self.state = torch.zeros(capacity, obs_dim, device=device)
        self.action = torch.zeros(capacity, action_dim, device=device)
        self.reward = torch.zeros(capacity, device=device)
        self.next_state = torch.zeros(capacity, obs_dim, device=device)
        self.bootstrap = torch.zeros(capacity, device=device)
        self.ptr = 0
        self.full = False
        self.num_added = 0

    def add_batch(self, s, a, r, s2, boot):
        """All inputs [B, ...] tensors (r, boot are [B])."""
        s = s.to(self.device, torch.float32)
        a = a.to(self.device, torch.float32)
        r = r.to(self.device, torch.float32).reshape(-1)
        s2 = s2.to(self.device, torch.float32)
        boot = boot.to(self.device, torch.float32).reshape(-1)
        B = s.shape[0]
        if B == 0:
            return
        idx = (torch.arange(B, device=self.device) + self.ptr) % self.capacity
        self.state[idx] = s
        self.action[idx] = a
        self.reward[idx] = r
        self.next_state[idx] = s2
        self.bootstrap[idx] = boot
        self.ptr = (self.ptr + B) % self.capacity
        self.num_added += B
        if self.num_added >= self.capacity:
            self.full = True

    def size(self) -> int:
        return self.capacity if self.full else self.ptr

    def _idx(self, batchsize):
        return torch.randint(0, self.size(), (batchsize,), device=self.device)

    def sample(self, batchsize, device=None) -> Batch:
        idx = self._idx(batchsize)
        obs = {"state": self.state[idx]}
        next_obs = {"state": self.next_state[idx]}
        action = {"action": self.action[idx]}
        reward = self.reward[idx]
        bootstrap = self.bootstrap[idx]
        if device is not None and device != self.device:
            obs = {k: v.to(device) for k, v in obs.items()}
            next_obs = {k: v.to(device) for k, v in next_obs.items()}
            action = {k: v.to(device) for k, v in action.items()}
            reward = reward.to(device)
            bootstrap = bootstrap.to(device)
        return Batch(obs, next_obs, action, reward, bootstrap)

    def sample_states(self, batchsize) -> torch.Tensor:
        return self.state[self._idx(batchsize)]

    def state_stats(self):
        n = self.size()
        s = self.state[:n]
        return s.mean(0), s.std(0)


# ---------------------------------------------------------------------------
# Agent: IBRL + dynamics ensemble. critic/actor/IBRL/value are all inherited
# from QAgent unchanged; we only add the ensemble + synthetic rollout.
# ---------------------------------------------------------------------------
class MBIBRLAgent(QAgent):
    def __init__(
        self,
        use_state,
        obs_shape,
        prop_shape,
        action_dim,
        rl_camera: str,
        cfg: QAgentConfig,
        dyn_cfg: DynamicsEnsembleConfig,
    ):
        assert use_state, "MBIBRLAgent is state-only (use_state=1)"
        assert len(obs_shape) == 1, "state_stack must be 1 (model predicts a flat next state)"
        super().__init__(use_state, obs_shape, prop_shape, action_dim, rl_camera, cfg)
        self.dyn_cfg = dyn_cfg
        self.dynamics = DynamicsEnsemble(obs_shape[0], action_dim, dyn_cfg, device=cfg.device)

    def set_dynamics_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        self.dynamics.set_normalization(mean, std)

    def train_dynamics(self, batch: Batch) -> dict:
        s = batch.obs["state"]
        a = batch.action["action"]
        s2 = batch.next_obs["state"]
        r = batch.reward
        # bootstrap == 0 marks an absorbing/terminal transition (done).
        done = (batch.bootstrap <= 0).float()
        return self.dynamics.train_step(s, a, r, s2, done)

    @torch.no_grad()
    def generate_rollouts(self, start_states: torch.Tensor, horizon: int, stddev: float, discount: float):
        """Branch short synthetic rollouts from real start states. Actions come
        from the SAME IBRL policy (self.act, sampled w/ exploration noise); next
        states/rewards/dones from the ensemble. Trajectories that the model
        predicts to be `done` stop early. Returns flat tensors of all collected
        transitions on device, or None if empty.

        MUST be called within `utils.eval_mode(self)` (self.act asserts not
        self.training)."""
        device = self.cfg.device
        s = start_states.to(device, torch.float32)
        B = s.shape[0]
        alive = torch.ones(B, dtype=torch.bool, device=device)
        out_s, out_a, out_r, out_s2, out_boot = [], [], [], [], []
        # _act_ibrl's stats-logging path assumes single-env train collection
        # (asserts bsize==1); bypass it for the batched model rollout.
        saved_stats = self.stats
        self.stats = None
        try:
            for _ in range(horizon):
                if alive.sum() == 0:
                    break
                obs = {"state": s}
                a = self.act(obs, eval_mode=False, stddev=stddev, cpu=False)  # [B, act]
                s2, r, done = self.dynamics.predict(s, a, sample=True)
                boot = discount * (1.0 - done.float())
                m = alive  # transitions valid where the traj was alive at step start
                out_s.append(s[m])
                out_a.append(a[m])
                out_r.append(r[m])
                out_s2.append(s2[m])
                out_boot.append(boot[m])
                s = s2
                alive = alive & (~done)
        finally:
            self.stats = saved_stats
        if not out_s:
            return None
        return {
            "state": torch.cat(out_s, 0),
            "action": torch.cat(out_a, 0),
            "reward": torch.cat(out_r, 0),
            "next_state": torch.cat(out_s2, 0),
            "bootstrap": torch.cat(out_boot, 0),
        }

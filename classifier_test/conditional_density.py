"""Conditional density models p(Δs' | s, a, c) for the dynamics-gap detector.

c in {source=0, target=1, ∅}; ∅ = domain-invariant (CFG null condition).
Δs' = s' - s is the modeled quantity (zero-centered, better-scaled than s').

Both heads expose the SAME api:
  - loss(s, a, ds, c_onehot)        -> scalar training loss (NLL or ε-MSE)
  - dgi(s, a, sp, n_t, n_noise)     -> per-row |err(c=target) - err(c=source)|
  - set_norm(...) / normalize helpers (buffers, persisted in ckpt)

The (s, a) are ALWAYS pure context (never noised); only Δs' is modeled, so the
(s,a)-marginal "behavior signature" confound never enters the score.

CFG null is encoded as the all-zero domain vector (matches bc/flow_policy.py).
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def mlp(sizes, act=nn.SiLU, last_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if last_act is not None:
        layers.append(last_act())
    return nn.Sequential(*layers)


class _NormMixin(nn.Module):
    """Holds normalization buffers + the domain-onehot/CFG helper."""

    def _init_norm(self, state_dim, action_dim, ds_dim, num_domains=2):
        self.state_dim, self.action_dim, self.ds_dim = state_dim, action_dim, ds_dim
        self.num_domains = num_domains
        for n, dim in [("s", state_dim), ("a", action_dim), ("ds", ds_dim)]:
            self.register_buffer(f"{n}_mean", torch.zeros(dim))
            self.register_buffer(f"{n}_std", torch.ones(dim))

    def set_norm(self, **kw):
        for n in ("s", "a", "ds"):
            if f"{n}_mean" in kw:
                getattr(self, f"{n}_mean").copy_(torch.as_tensor(kw[f"{n}_mean"], dtype=torch.float32))
                getattr(self, f"{n}_std").copy_(torch.as_tensor(kw[f"{n}_std"], dtype=torch.float32).clamp_min(1e-6))

    def nz(self, x, which):
        return (x - getattr(self, f"{which}_mean")) / getattr(self, f"{which}_std")

    def context(self, s, a, c_onehot):
        """Concat normalized (s, a) with the domain one-hot (∅ = zeros)."""
        return torch.cat([self.nz(s, "s"), self.nz(a, "a"), c_onehot], dim=-1)

    def onehot(self, c, B, device):
        """c: int (0/1) or None(=∅) or LongTensor(B,). Returns (B, num_domains)."""
        if c is None:
            return torch.zeros(B, self.num_domains, device=device)
        if isinstance(c, int):
            oh = torch.zeros(B, self.num_domains, device=device)
            oh[:, c] = 1.0
            return oh
        return F.one_hot(c.long(), self.num_domains).float().to(device)


# --------------------------------------------------------------------------- #
# Phase A: conditional Gaussian (diagonal) over Δs'
# --------------------------------------------------------------------------- #
class CondGaussian(_NormMixin):
    def __init__(self, state_dim, action_dim, ds_dim, hidden=512, num_domains=2):
        super().__init__()
        self._init_norm(state_dim, action_dim, ds_dim, num_domains)
        cdim = state_dim + action_dim + num_domains
        self.body = mlp([cdim, hidden, hidden])
        self.mu = nn.Linear(hidden, ds_dim)
        self.logvar = nn.Linear(hidden, ds_dim)
        self.logvar_clamp = (-8.0, 4.0)

    def forward(self, s, a, c_onehot):
        h = self.body(self.context(s, a, c_onehot))
        return self.mu(h), self.logvar(h).clamp(*self.logvar_clamp)

    def loss(self, s, a, ds, c_onehot):
        mu, logvar = self.forward(s, a, c_onehot)
        dsn = self.nz(ds, "ds")
        # Gaussian NLL (per-dim), mean over dims+batch
        return 0.5 * (logvar + (dsn - mu) ** 2 / logvar.exp()).mean()

    @torch.no_grad()
    def _mean(self, s, a, c):
        """Predicted E[Δs' | s, a, c] in normalized space."""
        oh = self.onehot(c, s.size(0), s.device)
        mu, _ = self.forward(s, a, oh)
        return mu

    @torch.no_grad()
    def dgi(self, s, a, sp, n_t=1, n_noise=1):
        # DGI = || E[Δs'|tgt] - E[Δs'|src] ||  (matches |E(τ0|d=1)-E(τ0|d=0)|).
        # sp is unused for the Gaussian (mean does not depend on the observed s').
        return (self._mean(s, a, 1) - self._mean(s, a, 0)).norm(dim=-1)


# --------------------------------------------------------------------------- #
# Residual-parameterized Gaussian (sim-base + zero-init target residual)
# --------------------------------------------------------------------------- #
class CondGaussianResidual(_NormMixin):
    """μ(s,a,c) = μ_base(s,a)  +  1(c==target) · Δμ_domain(s,a).

    - base (shared by c=0/sim AND ∅) learns the general source dynamics from the
      huge replay; its mean μ_base is the "sim-first" prediction.
    - Δμ_domain (the target residual) has a ZERO-INIT last layer + stronger weight
      decay, so on OOD (no target gradient) it stays ≈0 -> μ(c=1)→μ(c=0) → DGI→0
      *by construction*. This is exactly the "graceful degradation to ∅" fix.
    - DGI(s,a) = ||Δμ_domain(s,a)|| analytically (no observed s' needed).
    logvar is base-only (aleatoric scale shared across domains).
    """

    def __init__(self, state_dim, action_dim, ds_dim, hidden=512, num_domains=2):
        super().__init__()
        self._init_norm(state_dim, action_dim, ds_dim, num_domains)
        cin = state_dim + action_dim
        self.base = mlp([cin, hidden, hidden])
        self.mu_base = nn.Linear(hidden, ds_dim)
        self.logvar = nn.Linear(hidden, ds_dim)
        self.logvar_clamp = (-8.0, 4.0)
        # target residual branch (zero-init last layer)
        self.res = mlp([cin, hidden, hidden])
        self.dmu = nn.Linear(hidden, ds_dim)
        nn.init.zeros_(self.dmu.weight); nn.init.zeros_(self.dmu.bias)

    def residual_params(self):
        return list(self.res.parameters()) + list(self.dmu.parameters())

    def base_params(self):
        rp = set(id(p) for p in self.residual_params())
        return [p for p in self.parameters() if id(p) not in rp]

    def _base(self, s, a):
        h = self.base(torch.cat([self.nz(s, "s"), self.nz(a, "a")], -1))
        return self.mu_base(h), self.logvar(h).clamp(*self.logvar_clamp)

    def _delta(self, s, a):
        return self.dmu(self.res(torch.cat([self.nz(s, "s"), self.nz(a, "a")], -1)))

    def mu_for(self, s, a, is_target):
        """is_target: (B,) float/bool 1 for c=1 target, 0 for c=0/∅."""
        mb, lv = self._base(s, a)
        dm = self._delta(s, a)
        w = is_target.float().unsqueeze(-1) if torch.is_tensor(is_target) else float(is_target)
        return mb + w * dm, lv

    def energy(self, s, a, sp, is_target):
        """Reconstruction energy E(Δs'|s,a,c) = ||Δs'_norm - μ_c||^2 (per row).
        Stable surrogate for the denoising energy (no variance division)."""
        dsn = self.nz(sp - s, "ds")
        mu, _ = self.mu_for(s, a, is_target)
        return ((dsn - mu) ** 2).sum(-1)

    def nll(self, s, a, sp, is_target):
        dsn = self.nz(sp - s, "ds")
        mu, lv = self.mu_for(s, a, is_target)
        return 0.5 * (lv + (dsn - mu) ** 2 / lv.exp()).sum(-1)

    def loss(self, s, a, ds, is_target):
        mu, lv = self.mu_for(s, a, is_target)
        dsn = self.nz(ds, "ds")
        return 0.5 * (lv + (dsn - mu) ** 2 / lv.exp()).mean()

    @torch.no_grad()
    def dgi(self, s, a, sp=None, n_t=1, n_noise=1):
        # DGI = ||μ(c=1) - μ(c=0)|| = ||Δμ_domain(s,a)|| (zero on OOD by design).
        return self._delta(s, a).norm(dim=-1)


# --------------------------------------------------------------------------- #
# Phase B: conditional diffusion (ε-prediction) over Δs' + CFG
# --------------------------------------------------------------------------- #
class _SinEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        ang = t.float()[:, None] * freqs[None]
        return torch.cat([ang.sin(), ang.cos()], dim=-1)


class CondDiffusion(_NormMixin):
    def __init__(self, state_dim, action_dim, ds_dim, hidden=512, num_domains=2,
                 num_train_timesteps=100, t_emb=64):
        super().__init__()
        self._init_norm(state_dim, action_dim, ds_dim, num_domains)
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        self.sched = DDPMScheduler(num_train_timesteps=num_train_timesteps,
                                   beta_schedule="squaredcos_cap_v2",
                                   prediction_type="epsilon")
        self.T = num_train_timesteps
        self.temb = _SinEmb(t_emb)
        cdim = state_dim + action_dim + num_domains + t_emb + ds_dim
        self.net = mlp([cdim, hidden, hidden, hidden])
        self.out = nn.Linear(hidden, ds_dim)

    def eps(self, x_t, t, s, a, c_onehot):
        h = torch.cat([x_t, self.temb(t), self.context(s, a, c_onehot)], dim=-1)
        return self.out(self.net(h))

    def loss(self, s, a, ds, c_onehot):
        dsn = self.nz(ds, "ds")
        B = dsn.size(0)
        t = torch.randint(0, self.T, (B,), device=dsn.device)
        noise = torch.randn_like(dsn)
        x_t = self.sched.add_noise(dsn, noise, t)
        pred = self.eps(x_t, t, s, a, c_onehot)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def _x0(self, s, a, c, packed):
        """Predicted clean Δs' (x0) under condition c, from the observed Δs'
        noised at the given timesteps, averaged over draws. This is the
        denoising expectation E(τ0|c) in the DGI definition.

        x0_hat = (x_t - sqrt(1-α_t)·ε_θ) / sqrt(α_t).  Noised from the SAME
        anchor (the observed s'_src/s'_tgt is the same input) with SHARED noise
        across c so the difference is a fair, calibrated comparison.

        packed = (x_ts, ts, sqrt_ac, sqrt_1mac) lists over draws.
        """
        B = s.size(0)
        oh = self.onehot(c, B, s.device)
        x_ts, tt, sa, s1 = packed
        x0s = []
        for x_t, t, sqrt_ac, sqrt_1mac in zip(x_ts, tt, sa, s1):
            pred = self.eps(x_t, t, s, a, oh)
            x0s.append((x_t - sqrt_1mac * pred) / sqrt_ac)
        return torch.stack(x0s, 0).mean(0)  # (B, ds_dim)

    @torch.no_grad()
    def dgi(self, s, a, sp, n_t=8, n_noise=1):
        # Noise the OBSERVED Δs' at shared (t, noise) for both conditions, denoise
        # to x0 under c=1 and c=0, DGI = || x0(tgt) - x0(src) ||.
        dsn = self.nz(sp - s, "ds")
        ac = self.sched.alphas_cumprod.to(s.device)
        B = dsn.size(0)
        x_ts, tt, sa, s1 = [], [], [], []
        for _ in range(n_t * n_noise):
            ti = int(torch.randint(0, self.T, (1,)).item())
            t = torch.full((B,), ti, device=s.device, dtype=torch.long)
            noise = torch.randn_like(dsn)
            x_ts.append(self.sched.add_noise(dsn, noise, t))
            tt.append(t)
            sa.append(ac[ti].sqrt()); s1.append((1 - ac[ti]).sqrt())
        packed = (x_ts, tt, sa, s1)
        xt = self._x0(s, a, 1, packed)
        xs = self._x0(s, a, 0, packed)
        return (xt - xs).norm(dim=-1)


def build_model(head, state_dim, action_dim, ds_dim, **kw):
    if head == "gaussian":
        return CondGaussian(state_dim, action_dim, ds_dim, **kw)
    if head == "gaussian_res":
        return CondGaussianResidual(state_dim, action_dim, ds_dim, **kw)
    if head == "diffusion":
        return CondDiffusion(state_dim, action_dim, ds_dim, **kw)
    raise ValueError(f"unknown head {head}")

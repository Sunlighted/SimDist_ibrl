"""NeRF-style positional / Fourier encoding of a (z-scored) state vector.

Used as an OPTIONAL input transform for state-based nets (IBRL critic/actor,
SimDist encoder). `num_freq == 0` => the module is never instantiated by its
callers (they keep `self.pe = None`), so state_dict / behavior are byte-for-byte
unchanged and old checkpoints load strictly. See CLAUDE.md discussion on PE input
for the finer Threading task.

Encoding (per input dim x, after optional z-score):
    gamma(x) = [ (x), sin(2^0*pi*x), cos(2^0*pi*x), ..., sin(2^{L-1}*pi*x), cos(2^{L-1}*pi*x) ]
so out_dim = in_dim * ((1 if include_raw else 0) + 2*L).

z-score buffers (mean/std) are IDENTITY until `set_stats()` is called; they align
the input to unit scale so the fixed 2^k*pi frequencies are meaningful, and they
persist in the ckpt so eval self-normalizes.
"""
import math
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_freq: int = 6,
        include_raw: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        assert num_freq > 0, "callers must skip constructing this module when num_freq==0"
        self.in_dim = in_dim
        self.num_freq = num_freq
        self.include_raw = bool(include_raw)
        self.normalize = bool(normalize)

        self.register_buffer("mean", torch.zeros(in_dim))
        self.register_buffer("std", torch.ones(in_dim))
        freqs = (2.0 ** torch.arange(num_freq, dtype=torch.float32)) * math.pi
        self.register_buffer("freqs", freqs)

        per = (1 if self.include_raw else 0) + 2 * num_freq
        self.out_dim = in_dim * per

    @torch.no_grad()
    def set_stats(self, mean, std):
        m = torch.as_tensor(mean, dtype=torch.float32, device=self.mean.device).reshape(-1)
        s = torch.as_tensor(std, dtype=torch.float32, device=self.std.device).reshape(-1)
        assert m.numel() == self.in_dim and s.numel() == self.in_dim
        self.mean.copy_(m)
        self.std.copy_(s.clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_dim] -> [..., out_dim]
        if self.normalize:
            x = (x - self.mean) / self.std
        xf = x.unsqueeze(-1) * self.freqs            # [..., in_dim, num_freq]
        sin = torch.sin(xf).flatten(-2)              # [..., in_dim*num_freq]
        cos = torch.cos(xf).flatten(-2)
        if self.include_raw:
            return torch.cat([x, sin, cos], dim=-1)
        return torch.cat([sin, cos], dim=-1)

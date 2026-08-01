"""Transformer backbone for Diffusion Policy (ported from real-stanford/diffusion_policy).

Source: https://github.com/real-stanford/diffusion_policy
   - diffusion_policy/model/diffusion/transformer_for_diffusion.py
   - diffusion_policy/model/diffusion/positional_embedding.py (SinusoidalPosEmb)

What's preserved (faithful port):
  - GPT-style encoder-decoder layout: noisy-action tokens (Tp) attend to a
    short conditioning sequence (1 time token + To obs tokens) via cross-
    attention.
  - Optional causal self-attention on the action axis (`causal_attn=True`)
    and a matching diagonal memory mask so action token `t` only sees
    cond tokens up to position `t+1`.
  - Pre-norm (`norm_first=True`) transformer blocks for stability.
  - Cond encoder = small per-token MLP when `n_cond_layers=0`, else a stack
    of TransformerEncoderLayers (To+1 is tiny so MLP is usually enough).
  - GPT-style init (Normal(0, 0.02) for Linear/Embedding/pos_emb, zeros for
    LayerNorm bias) and the `get_optim_groups` helper that separates
    decayed vs non-decayed param sets (matches diffusion_policy paper's
    AdamW recipe).

What's removed for the ibrl port:
  - Inpainting / `obs_as_cond=False` branch — not used in our setting where
    we always condition on obs.
  - `time_as_cond=False` (BERT-style encoder-only) branch — likewise unused
    and complicates the API. We always treat time as a cond token.
  - `ModuleAttrMixin` (`_dummy_variable`-based device probing) — we don't
    need it; the policy that owns this module already has a device.
"""

from typing import Optional, Tuple, Union
import math

import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):
    """Diffusion-step sinusoidal embedding (verbatim from diffusion_policy)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TransformerForDiffusion(nn.Module):
    """Transformer noise-prediction network for diffusion policy.

    Inputs / outputs:
        sample:   (B, Tp, input_dim)  — noisy action chunk
        timestep: (B,) or scalar int   — diffusion step (will be broadcast)
        cond:     (B, To, cond_dim)    — per-step obs conditioning
        returns:  (B, Tp, output_dim)  — predicted noise (or sample, per scheduler)

    Default hyperparameters mirror the lowdim config from diffusion_policy
    (n_layer=8, n_head=4, n_emb=256, p_drop_attn=0.3, causal_attn=True).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,            # Tp
        cond_dim: int,           # per-step obs feature dim
        n_obs_steps: int = 1,    # To
        n_layer: int = 8,
        n_head: int = 4,
        n_emb: int = 256,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.3,
        causal_attn: bool = True,
        n_cond_layers: int = 0,
    ) -> None:
        super().__init__()
        assert cond_dim > 0, "TransformerForDiffusion requires cond_dim > 0 (obs conditioning)"

        T = horizon
        T_cond = 1 + n_obs_steps  # one time token + To obs tokens

        # ---- input embedding stem (noisy actions) ----
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # ---- cond encoder (time + obs tokens → memory) ----
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))

        if n_cond_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=n_emb, nhead=n_head, dim_feedforward=4 * n_emb,
                dropout=p_drop_attn, activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_cond_layers)
        else:
            # Per-token MLP "encoder" — T_cond is tiny (=1+To) so this is plenty.
            self.encoder = nn.Sequential(
                nn.Linear(n_emb, 4 * n_emb),
                nn.Mish(),
                nn.Linear(4 * n_emb, n_emb),
            )

        # ---- decoder (cross-attention from actions to memory) ----
        dec_layer = nn.TransformerDecoderLayer(
            d_model=n_emb, nhead=n_head, dim_feedforward=4 * n_emb,
            dropout=p_drop_attn, activation="gelu",
            batch_first=True, norm_first=True,  # critical for stability
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layer)

        # ---- attention masks (causal on tgt; diagonal-shifted on memory) ----
        if causal_attn:
            # tgt mask: upper-triangular set to -inf (additive form for PyTorch).
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )
            self.register_buffer("mask", mask)

            # memory mask: action token t may attend to cond tokens s where s-1 <= t.
            # The "-1" accounts for the time token at position 0.
            t_idx, s_idx = torch.meshgrid(
                torch.arange(T), torch.arange(T_cond), indexing="ij"
            )
            mem = t_idx >= (s_idx - 1)
            mem = (
                mem.float()
                .masked_fill(mem == 0, float("-inf"))
                .masked_fill(mem == 1, float(0.0))
            )
            self.register_buffer("memory_mask", mem)
        else:
            self.mask = None
            self.memory_mask = None

        # ---- output head ----
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # ---- constants ----
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps

        # ---- init (GPT-style) ----
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.GELU,
            nn.Sequential,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            for name in ("in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"):
                w = getattr(module, name, None)
                if w is not None:
                    nn.init.normal_(w, mean=0.0, std=0.02)
            for name in ("in_proj_bias", "bias_k", "bias_v"):
                b = getattr(module, name, None)
                if b is not None:
                    nn.init.zeros_(b)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            # Unknown leaf module — be loud so we notice if a new layer was added.
            raise RuntimeError(f"Unaccounted module {module}")

    # =================================================================
    # AdamW param groups (GPT-style decay vs no-decay split)
    # =================================================================

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """Return AdamW param groups separating decayed weights from biases /
        norms / positional embeddings (which should not be decayed).

        Matches diffusion_policy's transformer recipe. The caller can pass
        these straight to `torch.optim.AdamW`.
        """
        decay, no_decay = set(), set()
        whitelist = (nn.Linear, nn.MultiheadAttention)
        blacklist = (nn.LayerNorm, nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias") or pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    no_decay.add(fpn)
        no_decay.add("pos_emb")
        no_decay.add("cond_pos_emb")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter = decay & no_decay
        assert not inter, f"params in both decay/no_decay: {inter}"
        leftover = param_dict.keys() - decay - no_decay
        assert not leftover, f"params not assigned to any group: {leftover}"

        return [
            {"params": [param_dict[n] for n in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[n] for n in sorted(no_decay)], "weight_decay": 0.0},
        ]

    # =================================================================
    # forward
    # =================================================================

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        sample:   (B, T, input_dim)
        timestep: (B,) or scalar
        cond:     (B, To, cond_dim)
        returns:  (B, T, output_dim)
        """
        # ---- 1. time token ----
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)  # (B, 1, n_emb)

        # ---- 2. encoder over [time, obs] tokens ----
        cond_obs = self.cond_obs_emb(cond)                  # (B, To, n_emb)
        cond_tokens = torch.cat([time_emb, cond_obs], dim=1)  # (B, 1+To, n_emb)
        tc = cond_tokens.shape[1]
        x = self.drop(cond_tokens + self.cond_pos_emb[:, :tc])
        memory = self.encoder(x)                            # (B, 1+To, n_emb)

        # ---- 3. decoder over noisy-action tokens ----
        input_emb = self.input_emb(sample)                  # (B, T, n_emb)
        t = input_emb.shape[1]
        x = self.drop(input_emb + self.pos_emb[:, :t])
        x = self.decoder(
            tgt=x, memory=memory,
            tgt_mask=self.mask, memory_mask=self.memory_mask,
        )

        # ---- 4. head ----
        x = self.ln_f(x)
        return self.head(x)

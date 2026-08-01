"""Multi-step (window-level) counterfactual dynamics discriminator.

Generalizes the single-step H2O discriminator (discriminator.py) to a k-step
window. For a start state s0 and an action SEQUENCE a_{0..k-1}:
  d_sa(s0, a_seq)                  -> {target=0, source=1}; SHARED across the
                                      counterfactual pair (same start + actions)
                                      so it collapses to 0.5.
  d_sas(s0, a_seq, win_states)     -> advantage; win_states = the k resulting
                                      states, the only thing that differs between
                                      the source k-step rollout and the target
                                      demo segment. So it must key on the
                                      COMPOUNDED dynamics divergence.
log_dynamics_ratio isolates the multi-step dynamics gap (the (s0, a_seq) marginal
cancels). Inputs are z-scored per block (buffers persisted in the ckpt).
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from discriminator import ConcatDiscriminator


class MultiStepDiscriminator(nn.Module):
    def __init__(self, state_dim, action_dim, k, hidden=256, dropout=True):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.k = k
        self.a_seq = action_dim * k          # flattened action sequence
        self.w_seq = state_dim * k           # flattened window states
        # d_sa over (s0, a_seq); d_sas over (s0, a_seq, win_states)
        self.d_sa = ConcatDiscriminator(state_dim + self.a_seq, hidden, 2, dropout=dropout)
        self.d_sas = ConcatDiscriminator(state_dim + self.a_seq + self.w_seq, hidden, 2, dropout=dropout)
        for n, dim in [("s", state_dim), ("a", self.a_seq), ("w", self.w_seq)]:
            self.register_buffer(f"{n}_mean", torch.zeros(dim))
            self.register_buffer(f"{n}_std", torch.ones(dim))

    def set_norm(self, s_mean, s_std, a_mean, a_std, w_mean, w_std):
        for n, m, s in [("s", s_mean, s_std), ("a", a_mean, a_std), ("w", w_mean, w_std)]:
            getattr(self, f"{n}_mean").copy_(torch.as_tensor(m, dtype=torch.float32))
            getattr(self, f"{n}_std").copy_(torch.as_tensor(s, dtype=torch.float32).clamp_min(1e-6))

    def _nz(self, x, which):
        return (x - getattr(self, f"{which}_mean")) / getattr(self, f"{which}_std")

    def prob_sa(self, s0, aseq):
        return F.softmax(self.d_sa(self._nz(s0, "s"), self._nz(aseq, "a")), dim=1)

    def prob_sas(self, s0, aseq, win):
        s0n, an, wn = self._nz(s0, "s"), self._nz(aseq, "a"), self._nz(win, "w")
        sa_logits = self.d_sa(s0n, an)
        adv = self.d_sas(s0n, an, wn)
        return F.softmax(sa_logits, dim=1), F.softmax(adv + sa_logits, dim=1)

    def log_dynamics_ratio(self, s0, aseq, win):
        sa_prob, sas_prob = self.prob_sas(s0, aseq, win)
        eps = 1e-8
        return (torch.log(sas_prob[:, 0] + eps) - torch.log(sas_prob[:, 1] + eps)
                - torch.log(sa_prob[:, 0] + eps) + torch.log(sa_prob[:, 1] + eps))

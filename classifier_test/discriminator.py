"""Two-discriminator dynamics-gap classifier, ported from H2O.

H2O (Niu et al. 2022) trains two discriminators to separate source ("sim")
from target ("real") data and recovers the dynamics-density ratio
    p_target(s'|s,a) / p_source(s'|s,a)
by *dividing out* the state-action marginal:

    d_sa(s, a)         -> logits over {target=0, source=1}
    d_sas(s, a, s')    -> *advantage* logits; the joint classifier uses
                          softmax(adv_logits + sa_logits)

so that d_sas only has to model the residual that the transition s' adds on
top of (s, a). The dynamics log-ratio is then

    log r = [log p_sas(target) - log p_sas(source)]
          - [log p_sa(target)  - log p_sa(source)]

which cancels the (s, a) marginal mismatch and isolates the dynamics gap.

Class convention (matches H2O): index 0 = target ("real"), 1 = source ("sim").
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Discriminator(nn.Module):
    def __init__(self, num_input, num_hidden, num_output=2, dropout=False):
        super().__init__()
        self.fc1 = nn.Linear(num_input, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_output)
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, x):
        if self.dropout:
            x = F.relu(self.dropout_layer(self.fc1(x)))
            x = F.relu(self.dropout_layer(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        # H2O squashes the logits to [-2, 2] to keep the softmax ratio bounded.
        return 2 * torch.tanh(self.fc3(x))


class ConcatDiscriminator(Discriminator):
    """Concatenate the inputs along the feature dim, then MLP."""

    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs):
        return super().forward(torch.cat(inputs, dim=self.dim))


class DynamicsDiscriminator(nn.Module):
    """Bundles d_sa + d_sas and exposes classification / ratio utilities.

    Normalization (mean/std over the training transitions) is stored as
    buffers so a reloaded checkpoint reproduces train-time preprocessing.
    """

    def __init__(self, state_dim, action_dim, hidden=256, dropout=False):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_sa = ConcatDiscriminator(state_dim + action_dim, hidden, 2, dropout=dropout)
        self.d_sas = ConcatDiscriminator(2 * state_dim + action_dim, hidden, 2, dropout=dropout)

        self.register_buffer("s_mean", torch.zeros(state_dim))
        self.register_buffer("s_std", torch.ones(state_dim))
        self.register_buffer("a_mean", torch.zeros(action_dim))
        self.register_buffer("a_std", torch.ones(action_dim))

    # ---- preprocessing -------------------------------------------------
    def set_norm(self, s_mean, s_std, a_mean, a_std):
        self.s_mean.copy_(torch.as_tensor(s_mean, dtype=torch.float32))
        self.s_std.copy_(torch.as_tensor(s_std, dtype=torch.float32).clamp_min(1e-6))
        self.a_mean.copy_(torch.as_tensor(a_mean, dtype=torch.float32))
        self.a_std.copy_(torch.as_tensor(a_std, dtype=torch.float32).clamp_min(1e-6))

    def norm_s(self, s):
        return (s - self.s_mean) / self.s_std

    def norm_a(self, a):
        return (a - self.a_mean) / self.a_std

    # ---- logits --------------------------------------------------------
    def logits(self, s, a, ns):
        """Return (sa_logits, sas_logits) on normalized inputs.

        sas_logits = adv_logits + sa_logits, matching H2O's parameterization.
        """
        s, a, ns = self.norm_s(s), self.norm_a(a), self.norm_s(ns)
        sa_logits = self.d_sa(s, a)
        adv_logits = self.d_sas(s, a, ns)
        return sa_logits, adv_logits + sa_logits

    # ---- probabilities / predictions -----------------------------------
    def prob_sa(self, s, a):
        s, a = self.norm_s(s), self.norm_a(a)
        return F.softmax(self.d_sa(s, a), dim=1)

    def prob_sas(self, s, a, ns):
        sa_logits, sas_logits = self.logits(s, a, ns)
        return F.softmax(sa_logits, dim=1), F.softmax(sas_logits, dim=1)

    def log_dynamics_ratio(self, s, a, ns):
        """log p_target(s'|s,a) / p_source(s'|s,a) (the H2O real/sim ratio)."""
        sa_prob, sas_prob = self.prob_sas(s, a, ns)
        eps = 1e-8
        log_ratio = (
            torch.log(sas_prob[:, 0] + eps)
            - torch.log(sas_prob[:, 1] + eps)
            - torch.log(sa_prob[:, 0] + eps)
            + torch.log(sa_prob[:, 1] + eps)
        )
        return log_ratio

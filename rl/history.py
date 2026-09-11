"""Canonical proprio+action HISTORY layout, shared by training and acting.

The reference SimDist implementation feeds the world model a HISTORY -- proprio
observations and past actions -- alongside only the most recent image
(docs/simdist.md §44; the released quadruped code interleaves them as
`hist_enc[:, 0::2] = proprio, [:, 1::2] = actions` with `history_length: 25`).
Our port had none: `obs_stack/state_stack/prop_stack` were all 1.

This module exists because the history is assembled in TWO places -- from the
replay buffer at training time, and from the live env at act/eval time -- and if
those two layouts disagree the model trains on one convention and is evaluated on
another. That failure is silent: both paths produce a correctly-shaped vector and
the loss still falls. So the layout is defined ONCE, here, and both callers go
through it. `scripts/test_history_layout.py` asserts they agree.

LAYOUT (newest first, matching common_utils.concat_obs which `prop_stack` uses):
    [ prop_t, prop_{t-1}, ..., prop_{t-K+1},  a_{t-1}, a_{t-2}, ..., a_{t-K} ]
    dim = K*prop_dim + K*act_dim
Out-of-episode indices are ZERO-padded, which is what `concat_obs` does at an
episode start and what the wrapper's `past_actions` deque is initialised to.
Note actions are offset by one: a_{t-1} is the action that LED to prop_t.
"""

import torch


def history_dim(k: int, prop_dim: int, act_dim: int) -> int:
    return k * prop_dim + k * act_dim


def build_batch(prop: torch.Tensor, actions: torch.Tensor,
                starts, horizon: int, k: int) -> torch.Tensor:
    """Training-side builder.

    prop:    [T+1, prop_dim] for ONE episode
    actions: [T,   act_dim]
    starts:  int start index of the sampled segment
    returns: [horizon+1, k*prop_dim + k*act_dim]
    """
    tp, pd = prop.shape
    ad = actions.shape[-1]
    out = torch.zeros(horizon + 1, history_dim(k, pd, ad), dtype=torch.float32)
    for t in range(horizon + 1):
        cur = starts + t
        for j in range(k):                      # prop_t, prop_{t-1}, ...
            idx = cur - j
            if 0 <= idx < tp:
                out[t, j * pd:(j + 1) * pd] = prop[idx]
        base = k * pd
        for j in range(k):                      # a_{t-1}, a_{t-2}, ...
            idx = cur - 1 - j
            if 0 <= idx < actions.shape[0]:
                out[t, base + j * ad:base + (j + 1) * ad] = actions[idx]
    return out


def build_step(prop_stacked: torch.Tensor, past_action: torch.Tensor,
               k: int, prop_dim: int, act_dim: int) -> torch.Tensor:
    """Act-side builder, from what the env wrapper already provides.

    prop_stacked: obs["prop"] with `prop_stack=k` -- concat_obs already returns
                  NEWEST FIRST, [prop_t, prop_{t-1}, ...], zero-padded. Used as is.
    past_action:  obs["past_action"], the wrapper's deque as [oldest ... newest]
                  = [a_{t-k}, ..., a_{t-1}]. ⚠️ OPPOSITE order to the layout, so it
                  is REVERSED here. Getting this wrong is silent.
    returns: [k*prop_dim + k*act_dim] (1-D) or [B, ...] if inputs are batched.
    """
    p = prop_stacked.reshape(-1) if prop_stacked.dim() == 1 else prop_stacked
    assert p.shape[-1] == k * prop_dim, (
        f"prop_stacked is {p.shape[-1]}, expected k*prop_dim={k * prop_dim}: "
        f"the env must be built with prop_stack={k}")
    pa = past_action
    if pa.dim() == 2 and pa.shape[0] == k:          # [k, act_dim] from the wrapper
        pa = pa.flip(0).reshape(-1)                 # -> newest first
    elif pa.dim() == 3:                             # [B, k, act_dim]
        pa = pa.flip(1).reshape(pa.shape[0], -1)
    assert pa.shape[-1] == k * act_dim, (
        f"past_action is {pa.shape[-1]}, expected k*act_dim={k * act_dim}: "
        f"the env must be built with cond_action={k}")
    return torch.cat([p.to(pa.dtype), pa], dim=-1)

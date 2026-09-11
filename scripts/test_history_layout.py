"""The TRAINING history and the ACT history must be byte-identical.

They are built by different code from different sources -- the replay buffer at
train time, the live env wrapper at act time -- and a mismatch is SILENT: both
produce a correctly-shaped vector, the loss still falls, and the policy is simply
evaluated on a different convention than it trained on. Two conventions differ in
particular:
  * `concat_obs` (prop_stack) returns NEWEST FIRST;
  * the wrapper's `past_actions` deque is OLDEST FIRST.
So `rl/history.py::build_step` reverses the actions. This test proves it.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rl.history import build_batch, build_step, history_dim

K, PD, AD, T = 4, 9, 7, 12
torch.manual_seed(0)
prop = torch.arange(( T + 1) * PD, dtype=torch.float32).reshape(T + 1, PD)
acts = (torch.arange(T * AD, dtype=torch.float32).reshape(T, AD) + 1000.0)

def concat_obs_ref(seq, t, k):
    """common_utils.concat_obs: [seq[t], seq[t-1], ...], ZERO-pad below 0."""
    parts = []
    for off in range(k):
        i = t - off
        parts.append(seq[i] if i >= 0 else torch.zeros_like(seq[0]))
    return torch.cat(parts, dim=0)

fails = 0
for t in range(0, T):
    # ---- TRAIN side: one-row segment starting at t
    tr = build_batch(prop, acts, starts=t, horizon=0, k=K)[0]
    # ---- ACT side: what the env would hand us at the same timestep
    prop_stacked = concat_obs_ref(prop, t, K)                 # newest first
    past = torch.stack([acts[t - 1 - j] if 0 <= t - 1 - j < T
                        else torch.zeros(AD) for j in range(K)])   # newest first
    past_oldest_first = past.flip(0)                          # the deque's order
    ac = build_step(prop_stacked, past_oldest_first, K, PD, AD)
    if not torch.equal(tr, ac):
        fails += 1
        d = (tr - ac).abs()
        print(f"  t={t}: MISMATCH, max|diff|={d.max():.1f} "
              f"first bad idx={int(d.argmax())}")
    assert tr.shape == (history_dim(K, PD, AD),), tr.shape

print(f"train-vs-act layout: {T - fails}/{T} timesteps identical")
assert fails == 0, f"{fails} timesteps disagree -- the two paths have DRIFTED"

# episode-start zero padding
z = build_batch(prop, acts, starts=0, horizon=0, k=K)[0]
assert torch.equal(z[K * PD:], torch.zeros(K * AD)), "t=0 must have NO action history"
assert torch.equal(z[:PD], prop[0]), "t=0 newest prop must be prop[0]"
assert torch.equal(z[PD:K * PD], torch.zeros((K - 1) * PD)), "t=0 older prop must be zero"
print("episode-start zero padding: OK")

# a batched segment keeps per-timestep alignment
seg = build_batch(prop, acts, starts=5, horizon=3, k=K)
assert seg.shape == (4, history_dim(K, PD, AD))
for i in range(4):
    assert torch.equal(seg[i, :PD], prop[5 + i]), f"row {i} newest prop misaligned"
    assert torch.equal(seg[i, K * PD:K * PD + AD], acts[5 + i - 1]), f"row {i} a_(t-1) misaligned"
print("batched segment alignment: OK")
print("\nHISTORY LAYOUT CHECKS PASSED")

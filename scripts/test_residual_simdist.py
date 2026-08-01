"""Smoke test for ResidualSimDistAgent (rl/residual_simdist_agent.py) + the rela
n-step / stored-base_action path used by train_residual_simdist.py. No robosuite
env / no MPPI planner — the SimDist base is a tiny stub exposing .act(). Run:

  source set_env.sh && python scripts/test_residual_simdist.py
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import torch.nn as nn

from rl import replay
from rl.residual_rl_agent import ResidualRLConfig
from rl.residual_simdist_agent import ResidualSimDistAgent

OBS, ACT = 8, 4
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class StubBase(nn.Module):
    """Stand-in for the adapted SimDistAgent: a deterministic 'MPPI' base whose
    action is a fixed linear map of the state, exposing the .act() interface the
    residual agent queries."""

    def __init__(self):
        super().__init__()
        self.register_buffer("W", torch.randn(OBS, ACT) * 0.1)

    @torch.no_grad()
    def act(self, obs, *, eval_mode=False, stddev=0.0, cpu=True):
        s = obs["state"].float()
        single = s.dim() == 1
        s2 = s.view(1, -1) if single else s
        a = torch.tanh(s2 @ self.W.to(s2.device))
        a = a[0] if single else a
        return a.cpu() if cpu else a


def make_agent():
    cfg = ResidualRLConfig(action_scale=0.1, critic_hidden_dim=64, critic_num_layer=2,
                           num_critics=6, num_min_critics=2, normalize_states=1)
    agent = ResidualSimDistAgent((OBS,), ACT, cfg, device=DEV)
    agent.attach_base_policy(StubBase())
    agent.set_normalization(torch.zeros(OBS), torch.ones(OBS))
    return agent


def push_episode(buf, T, base, success):
    states = np.random.randn(T + 1, OBS).astype(np.float32)
    ba = base.act({"state": torch.from_numpy(states)}, cpu=True).numpy()  # [T+1, act]
    actions = np.clip(np.random.randn(T, ACT).astype(np.float32), -1, 1)
    rewards = np.zeros(T, dtype=np.float32)
    if success:
        rewards[-1] = 1.0
    buf.new_episode({"state": torch.from_numpy(states[0]), "base_action": torch.from_numpy(ba[0])})
    for t in range(T):
        terminal = (t == T - 1)
        nxt = {"state": torch.from_numpy(states[min(t + 1, T - 1)]),
               "base_action": torch.from_numpy(ba[min(t + 1, T - 1)])}
        buf.add(nxt, {"action": torch.from_numpy(actions[t])},
                float(rewards[t]), terminal, bool(success and terminal), image_obs={})


def main():
    torch.manual_seed(0); np.random.seed(0)
    agent = make_agent()
    base = agent._base_policy[0]

    # 1) _base_action shape + range
    s = torch.randn(5, OBS, device=DEV)
    ba = agent._base_action(s)
    assert ba.shape == (5, ACT) and ba.abs().max() <= 1.0
    print("[ok] _base_action shape/range", tuple(ba.shape))

    # 2) act == base at zero-init (residual last layer is zero -> delta==0)
    obs_single = {"state": torch.randn(OBS, device=DEV)}
    a_single = agent.act(obs_single, eval_mode=True)
    base_single = base.act({"state": obs_single["state"]}, cpu=True)
    assert a_single.shape == (ACT,)
    assert torch.allclose(a_single, base_single.clamp(-1, 1), atol=1e-5), "act != base at zero-init"
    obs_batch = {"state": torch.randn(6, OBS, device=DEV)}
    a_batch = agent.act(obs_batch, eval_mode=True)
    assert a_batch.shape == (6, ACT) and a_batch.abs().max() <= 1.0
    print("[ok] act single/batched; act == base at zero-init")

    # 2b) act reuses a precomputed base_action if present in obs
    pre = torch.zeros(OBS, device=DEV)
    a_reuse = agent.act({"state": pre, "base_action": torch.full((ACT,), 0.5, device=DEV)},
                        eval_mode=True)
    assert torch.allclose(a_reuse, torch.full((ACT,), 0.5), atol=1e-5), "did not reuse base_action"
    print("[ok] act reuses stored base_action")

    # 3) two rela buffers (nstep=5), push episodes, mixed merge + update
    online = replay.ReplayBuffer(5, 0.99, 1, 60, 5000, use_bc=False)
    offline = replay.ReplayBuffer(5, 0.99, 1, 60, 5000, use_bc=False)
    for _ in range(6):
        push_episode(online, T=25, base=base, success=False)
        push_episode(offline, T=20, base=base, success=True)
    on = online.sample(128, DEV)
    off = offline.sample(128, DEV)
    batch = replay.Batch.merge_batches(on, off)
    assert "base_action" in batch.obs and "base_action" in batch.next_obs
    assert batch.obs["state"].shape[0] == 256
    print("[ok] rela n-step buffers + merged mix carries base_action in obs/next_obs")

    # snapshot params to confirm they move
    c0 = [p.detach().clone() for p in agent.critic.parameters()]
    a0 = [p.detach().clone() for p in agent.actor.parameters()]
    m = None
    for i in range(8):
        m = agent.update(batch, update_actor=True)
    assert np.isfinite(m["residual/critic_loss"]) and np.isfinite(m.get("residual/actor_loss", 0.0))
    moved_c = any(not torch.equal(p, q) for p, q in zip(agent.critic.parameters(), c0))
    moved_a = any(not torch.equal(p, q) for p, q in zip(agent.actor.parameters(), a0))
    assert moved_c, "critic did not move"
    assert moved_a, "actor did not move"
    print(f"[ok] update: critic_loss={m['residual/critic_loss']:.4f} "
          f"actor_loss={m.get('residual/actor_loss', float('nan')):.4f} "
          f"delta_abs={m.get('residual/delta_abs', 0.0):.4f} (critic+actor moved)")

    # 4) state_dict excludes the base; round-trips norm buffers
    sd = agent.state_dict()
    assert not any(k.startswith("_base_policy") for k in sd), "base leaked into state_dict"
    assert "norm_state_mean" in sd and "norm_state_std" in sd
    agent.set_normalization(torch.arange(OBS).float(), torch.arange(1, OBS + 1).float())
    sd2 = agent.state_dict()
    agent2 = make_agent()
    agent2.load_state_dict(sd2)
    assert torch.allclose(agent2.norm_state_mean.cpu(), torch.arange(OBS).float())
    print("[ok] state_dict excludes base + round-trips norm buffers")

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
    # rela ReplayBuffer prefetch threads block a clean interpreter shutdown; all
    # work + prints are done -> hard-exit (mirrors the trainers).
    import sys
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)

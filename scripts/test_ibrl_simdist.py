"""Smoke test for train_ibrl_simdist.py: StoredActionBCPolicy prior + a QAgent
(act_method=ibrl) consuming a rela batch whose obs carries a stored `base_action`.
No robosuite env / no MPPI planner — the SimDist base is a tiny .act() stub. Run:

  source set_env.sh && python scripts/test_ibrl_simdist.py
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import torch.nn as nn

from common_utils import ibrl_utils as utils
from rl import replay
from rl.q_agent import QAgent, QAgentConfig
from train_ibrl_simdist import StoredActionBCPolicy

OBS, ACT = 8, 4
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class StubBase(nn.Module):
    """Deterministic 'MPPI' base: fixed linear map, exposing the .act() interface."""

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
    cfg = QAgentConfig()
    cfg.act_method = "ibrl"
    cfg.bootstrap_method = "ibrl"
    cfg.use_prop = 0
    cfg.device = DEV
    cfg.state_critic.num_q = 4
    cfg.state_critic.num_k = 2
    cfg.state_critic.hidden_dim = 64
    cfg.state_critic.num_layer = 2
    cfg.state_actor.hidden_dim = 64
    cfg.state_actor.num_layer = 2
    agent = QAgent(True, (OBS,), (1,), ACT, "cam", cfg)
    base = StubBase().to(DEV)
    agent.add_bc_policy(StoredActionBCPolicy(base))
    return agent, base


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
    agent, base = make_agent()
    prior = agent.bc_policies[0]

    # 1) StoredActionBCPolicy reads obs["base_action"] when present, else MPPI base
    s = torch.randn(5, OBS, device=DEV)
    stored = torch.rand(5, ACT, device=DEV) * 2 - 1
    a_read = prior.act({"state": s, "base_action": stored}, cpu=False)
    assert torch.allclose(a_read, stored.clamp(-1, 1), atol=1e-6), "did not read stored base_action"
    a_fallback = prior.act({"state": s}, cpu=False)
    assert a_fallback.shape == (5, ACT) and a_fallback.abs().max() <= 1.0
    assert not prior.training, "prior should be frozen"
    print("[ok] StoredActionBCPolicy: reads stored base_action + MPPI fallback + frozen")

    # 2) QAgent(ibrl).act with a stored base_action in obs (single + batched)
    with utils.eval_mode(agent):
        a1 = agent.act({"state": torch.randn(OBS, device=DEV),
                        "base_action": torch.zeros(ACT, device=DEV)}, eval_mode=True)
        assert a1.shape == (ACT,)
        ab = agent.act({"state": torch.randn(6, OBS, device=DEV),
                        "base_action": torch.zeros(6, ACT, device=DEV)}, eval_mode=True)
        assert ab.shape == (6, ACT)
    print("[ok] QAgent(ibrl).act single/batched with stored base_action")

    # 3) rela n-step buffer with base_action in obs; update moves critic+actor
    buf = replay.ReplayBuffer(3, 0.99, 1, 60, 5000, use_bc=True)
    for _ in range(8):
        push_episode(buf, T=22, base=base, success=(np.random.rand() < 0.5))
    batch = buf.sample(128, DEV)
    assert "base_action" in batch.obs and "base_action" in batch.next_obs
    c0 = [p.detach().clone() for p in agent.critic.parameters()]
    a0 = [p.detach().clone() for p in agent.actor.parameters()]
    bc_batch = buf.sample_bc(128, DEV) if buf.num_success > 0 else None
    m = None
    for _ in range(6):
        m = agent.update(batch, 0.1, update_actor=True, bc_batch=bc_batch, ref_agent=agent)
    assert np.isfinite(m["train/critic_loss"]) and np.isfinite(m["train/actor_loss"])
    moved_c = any(not torch.equal(p, q) for p, q in zip(agent.critic.parameters(), c0))
    moved_a = any(not torch.equal(p, q) for p, q in zip(agent.actor.parameters(), a0))
    assert moved_c and moved_a, "critic/actor did not move"
    print(f"[ok] ibrl update on stored-prior batch: critic_loss={m['train/critic_loss']:.4f} "
          f"actor_loss={m['train/actor_loss']:.4f} bootstrap_bc={m.get('actor/bootstrap_bc', float('nan'))} "
          f"(critic+actor moved)")

    # 4) state_dict excludes the base (prior is list-held in bc_policies)
    sd = agent.state_dict()
    assert not any("bc_polic" in k or "_base" in k for k in sd), "prior/base leaked into state_dict"
    print("[ok] state_dict excludes the SimDist prior/base")

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)

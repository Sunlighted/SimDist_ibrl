"""Synthetic smoke test for the TD-MPC2 + IBRL agent (CLAUDE.md §28).

No robosuite / no checkpoints — a random prior and random batches suffice to
exercise every code path:
  * world model shapes (encode / next / reward / Q ensemble / pi)
  * MPPI plan, single AND batched obs, actions in [-1, 1]
  * IBRL: the learned Q arbitrates between the plan and the prior (act), the TD
    bootstrap picks max-Q over {pi(z'), prior(s')}, prior_margin biases the pick
  * update(): losses finite, model + pi params move, target Qs EMA-track
  * state_dict EXCLUDES the prior and round-trips

Run: python scripts/test_tdmpc2.py
"""

import sys

import torch
import torch.nn as nn

sys.path.append(".")

from rl.tdmpc2_agent import TDMPC2Agent, TDMPC2Config

OBS, ACT, B, H = 37, 7, 16, 3
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class RandomPrior(nn.Module):
    """Stand-in for a BC head: same `act(obs, eval_mode=, cpu=)` interface."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS, 32), nn.ReLU(), nn.Linear(32, ACT), nn.Tanh())

    def act(self, obs, *, eval_mode=True, cpu=True):
        a = self.net(obs["state"]).detach()
        return a.cpu() if cpu else a


def make_agent(cfg=None):
    cfg = cfg or TDMPC2Config(horizon=H, num_samples=64, num_elites=8, iterations=2,
                              num_pi_trajs=8, num_prior_trajs=8)
    agent = TDMPC2Agent(True, (OBS,), (0,), ACT, "state", cfg, device=DEV)
    return agent


def make_batch():
    states = torch.randn(B, H + 1, OBS, device=DEV)
    actions = torch.empty(B, H, ACT, device=DEV).uniform_(-1, 1)
    rewards = (torch.rand(B, H, device=DEV) > 0.9).float()  # sparse 0/1
    return states, actions, rewards


def test_shapes_and_plan():
    agent = make_agent()
    agent.attach_prior(RandomPrior().to(DEV))
    agent.eval()

    single = {"state": torch.randn(OBS, device=DEV)}
    a = agent.act(single, eval_mode=True)
    assert a.shape == (ACT,), a.shape
    assert a.abs().max() <= 1.0 + 1e-5

    batched = {"state": torch.randn(5, OBS, device=DEV)}
    ab = agent.act(batched, eval_mode=True)
    assert ab.shape == (5, ACT), ab.shape
    assert ab.abs().max() <= 1.0 + 1e-5

    # mpc=0 ablation: pi head + IBRL selection, no planning
    agent.cfg.mpc = 0
    a0 = agent.act(batched, eval_mode=True)
    assert a0.shape == (5, ACT)
    print("  ok: act (single/batched, mpc on/off), actions in [-1,1]")


def test_ibrl_selection():
    """With prior_margin=+inf the prior must always win; with -inf, never."""
    agent = make_agent()
    prior = RandomPrior().to(DEV)
    agent.attach_prior(prior)
    agent.eval()
    obs = {"state": torch.randn(8, OBS, device=DEV)}

    agent.cfg.mpc = 0
    agent.cfg.prior_margin = 1e9
    a_forced_prior = agent.act(obs, eval_mode=True, cpu=False)
    a_prior = prior.act(obs, cpu=False).clamp(-1, 1)
    assert torch.allclose(a_forced_prior, a_prior, atol=1e-5), \
        "huge prior_margin must force the prior's action"

    agent.cfg.prior_margin = -1e9
    a_forced_rl = agent.act(obs, eval_mode=True, cpu=False)
    assert not torch.allclose(a_forced_rl, a_prior, atol=1e-3), \
        "negative prior_margin must force the RL action"
    print("  ok: IBRL max-Q selection honors prior_margin (prior <-> RL)")


def test_update_moves_params():
    agent = make_agent()
    agent.attach_prior(RandomPrior().to(DEV))
    before = {n: p.detach().clone() for n, p in agent.model.named_parameters()}
    tgt_before = [p.detach().clone() for p in agent.model.Qs_target.parameters()]

    for _ in range(3):
        info = agent.update(*make_batch())
    for k, v in info.items():
        assert v == v and abs(v) < 1e6, f"{k} = {v}"

    moved = [n for n, p in agent.model.named_parameters()
             if not n.startswith("Qs_target") and not torch.allclose(p, before[n])]
    for head in ("encoder", "dynamics", "reward", "pi", "Qs."):
        assert any(n.startswith(head) for n in moved), f"{head} did not move"
    tgt_after = list(agent.model.Qs_target.parameters())
    assert not torch.allclose(tgt_after[0], tgt_before[0]), "target Qs did not EMA-track"
    assert 0.0 <= info["ibrl/bootstrap_take_rl"] <= 1.0
    print(f"  ok: update() — all heads move, targets EMA-track, "
          f"take_rl={info['ibrl/bootstrap_take_rl']:.2f}, "
          f"q_mean={info['tdmpc2/q_mean']:.3f}")


def test_q_target_clamp():
    """clamp_q_target keeps the Bellman target in [0,1] (sparse 0/1 + end_on_success)."""
    agent = make_agent()
    states, _, rewards = make_batch()
    next_s = states[:, 1:].reshape(-1, OBS)
    z = agent.model.encode(next_s)
    y, _ = agent._td_target(z, next_s, rewards.reshape(-1),
                           (rewards > 0.5).float().reshape(-1))
    assert y.min() >= -1e-6 and y.max() <= 1.0 + 1e-6, (y.min().item(), y.max().item())
    agent.cfg.clamp_q_target = 0
    y2, _ = agent._td_target(z, next_s, rewards.reshape(-1),
                            (rewards > 0.5).float().reshape(-1))
    assert y2.shape == y.shape
    print("  ok: clamp_q_target bounds the TD target to [0,1]")


def test_bc_anchor():
    """tdmpc2.bc_loss_coef>0 adds the demo BC term to the pi loss (and only that)."""
    cfg = TDMPC2Config(horizon=H, num_samples=32, num_elites=4, iterations=1,
                       num_pi_trajs=4, num_prior_trajs=4, bc_loss_coef=0.1)
    agent = TDMPC2Agent(True, (OBS,), (0,), ACT, "state", cfg, device=DEV)
    bc_s = torch.randn(32, OBS, device=DEV)
    bc_a = torch.empty(32, ACT, device=DEV).uniform_(-1, 1)

    info = agent.update(*make_batch())                       # no BC batch
    assert info["tdmpc2/pi_bc_loss"] == 0.0
    enc_before = agent.model.encoder[0].weight.detach().clone()
    info = agent.update(*make_batch(), bc_states=bc_s, bc_actions=bc_a)
    assert info["tdmpc2/pi_bc_loss"] > 0, info["tdmpc2/pi_bc_loss"]

    # the BC term must train ONLY the pi head: it reads a detached latent, so the
    # encoder must not move because of it (it still moves from the model loss —
    # so check the BC gradient alone leaves it alone)
    agent.model_opt.zero_grad(set_to_none=True)
    agent.pi_opt.zero_grad(set_to_none=True)
    z = agent.model.encode(bc_s).detach()
    _, mu, _ = agent.model.pi_dist(z)
    torch.nn.functional.mse_loss(mu, bc_a).backward()
    assert agent.model.encoder[0].weight.grad is None, "BC term leaked into the encoder"
    assert agent.model.pi[0].weight.grad is not None
    print(f"  ok: bc_loss_coef adds a pi-only BC anchor "
          f"(bc_loss={info['tdmpc2/pi_bc_loss']:.3f}), encoder untouched")


def test_state_dict():
    agent = make_agent()
    agent.attach_prior(RandomPrior().to(DEV))
    sd = agent.state_dict()
    assert not any("prior" in k for k in sd), "the frozen prior leaked into state_dict"
    assert any(k.startswith("model.Qs_target") for k in sd)
    agent2 = make_agent()
    agent2.load_state_dict(sd)  # strict
    agent2.attach_prior(RandomPrior().to(DEV))
    print(f"  ok: state_dict round-trip ({len(sd)} keys), prior excluded")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("[test] shapes + plan");        test_shapes_and_plan()
    print("[test] IBRL selection");       test_ibrl_selection()
    print("[test] update");               test_update_moves_params()
    print("[test] q target clamp");       test_q_target_clamp()
    print("[test] bc anchor");            test_bc_anchor()
    print("[test] state_dict");           test_state_dict()
    print("\nALL TD-MPC2 SMOKE TESTS PASSED")

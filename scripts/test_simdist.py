"""Smoke test for the SimDist baseline (no env / no MuJoCo).

Checks:
  * SequenceReplayBuffer: online add + offline add_episode + (H+1)-segment sample;
  * WorldModel forward helpers (encode/next/reward/value/pi) shapes;
  * SimDistAgent.update_distill: all losses finite, trainable params move,
    value_target EMA-tracks (changes slowly), value_target NOT in optimizer;
  * SimDistAgent.update_finetune: only dynamics+policy params move; encoder /
    reward / value params unchanged;
  * MppiPlanner.plan on a random world model returns action in [-1,1];
  * agent.act handles single AND batched obs (mp-eval contract);
  * state_dict round-trip.

Run:  python scripts/test_simdist.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import torch

from rl.world_model import WorldModelConfig, WorldModel
from rl.simdist_planner import PlannerConfig
from rl.simdist_agent import SimDistConfig, SimDistAgent
from rl.sequence_replay import SequenceReplayBuffer

OBS_DIM, ACT_DIM = 37, 7
DEV = "cuda" if torch.cuda.is_available() else "cpu"
H = 5


def _cfg():
    return SimDistConfig(
        world_model=WorldModelConfig(latent_dim=128, mlp_hidden_dim=128, num_layer=2,
                                     horizon=H),
        planner=PlannerConfig(horizon=H, num_samples=64, num_elites=8, iterations=2),
        finetune_steps=10,
    )


def test_sequence_replay():
    buf = SequenceReplayBuffer(max_episodes=100)
    # online: two episodes of length 20 and 8
    for T in (20, 8):
        buf.new_episode(torch.randn(OBS_DIM))
        for t in range(T):
            buf.add(torch.randn(OBS_DIM), torch.rand(ACT_DIM) * 2 - 1,
                    reward=float(t == T - 1), terminal=(t == T - 1), success=(t == T - 1))
    # offline add_episode
    buf.add_episode(torch.randn(15 + 1, OBS_DIM), torch.rand(15, ACT_DIM) * 2 - 1,
                    torch.zeros(15), success=False)
    assert buf.size() == 3, buf.size()
    assert buf.can_sample(H)
    s, a, r = buf.sample(32, H, DEV)
    assert s.shape == (32, H + 1, OBS_DIM), s.shape
    assert a.shape == (32, H, ACT_DIM), a.shape
    assert r.shape == (32, H), r.shape
    # episode of length 8 >= H=5 contributes; a too-short one (<H) would be skipped
    print("OK  sequence_replay: add/add_episode/sample shapes", s.shape, a.shape, r.shape)


def test_world_model():
    wm = WorldModel(OBS_DIM, ACT_DIM, WorldModelConfig(latent_dim=128, mlp_hidden_dim=128)).to(DEV)
    z = wm.encode(torch.randn(16, OBS_DIM, device=DEV))
    a = torch.rand(16, ACT_DIM, device=DEV) * 2 - 1
    assert z.shape == (16, 128)
    assert wm.next(z, a).shape == (16, 128)
    assert wm.predict_reward(z, a).shape == (16,)
    assert wm.predict_value(z).shape == (16,)
    assert wm.predict_value(z, target=True).shape == (16,)
    s = torch.randn(16, OBS_DIM, device=DEV)
    pi = wm.policy_action(s)  # own-encoder state->action policy
    assert pi.shape == (16, ACT_DIM) and float(pi.abs().max()) <= 1.0 + 1e-5
    assert wm.decode(z).shape == (16, OBS_DIM)  # latent -> state (for MPPI proposal)
    # value_target excluded from trainable params
    n_train = len(wm.trainable_parameters())
    n_all = len(list(wm.parameters()))
    assert n_train < n_all, (n_train, n_all)
    print("OK  world_model: head shapes + value_target excluded from trainable")


def _rand_segment(B=64):
    s = torch.randn(B, H + 1, OBS_DIM, device=DEV)
    a = (torch.rand(B, H, ACT_DIM, device=DEV) * 2 - 1)
    r = (torch.rand(B, H, device=DEV) > 0.8).float()  # sparse-ish 0/1
    return s, a, r


def test_distill():
    agent = SimDistAgent(1, (OBS_DIM,), (9,), ACT_DIM, "cam", _cfg(), device=DEV)
    before = {n: p.detach().clone() for n, p in agent.model.named_parameters()}
    vt_before = copy.deepcopy(agent.model.value_target.state_dict())
    losses = None
    for _ in range(5):
        losses = agent.update_distill(*_rand_segment())
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values()), losses
    # encoder (trainable) moved
    enc_moved = any(
        not torch.equal(before[n], p) for n, p in agent.model.named_parameters()
        if n.startswith("encoder.")
    )
    assert enc_moved, "encoder should move during distill"
    # value_target changed only slightly (EMA), not a hard copy
    vt_after = agent.model.value_target.state_dict()
    moved = any(not torch.equal(vt_before[k], vt_after[k]) for k in vt_before)
    assert moved, "value_target should EMA-track"
    print("OK  distill: losses finite, encoder moves, value_target EMA-tracks", {k: round(v, 4) for k, v in losses.items()})


def test_finetune_freezes_heads():
    agent = SimDistAgent(1, (OBS_DIM,), (9,), ACT_DIM, "cam", _cfg(), device=DEV)
    snap = {n: p.detach().clone() for n, p in agent.model.named_parameters()}
    for _ in range(5):
        m = agent.update_finetune(*_rand_segment())
    assert all(torch.isfinite(torch.tensor(v)) for v in m.values()), m
    for n, p in agent.model.named_parameters():
        moved = not torch.equal(snap[n], p)
        if n.startswith(("dynamics.", "policy.", "enc_pi.")):  # dynamics head + policy branch
            assert moved, f"{n} should move during finetune"
        elif n.startswith(("encoder.", "reward.", "value.", "value_target.", "decoder.")):
            assert not moved, f"{n} should be FROZEN during finetune"
    print("OK  finetune: dynamics+policy(enc_pi) move; encoder/reward/value/decoder frozen")


def test_planner_and_act():
    agent = SimDistAgent(1, (OBS_DIM,), (9,), ACT_DIM, "cam", _cfg(), device=DEV)
    # single obs
    a1 = agent.act({"state": torch.randn(OBS_DIM, device=DEV)}, eval_mode=True)
    assert a1.shape == (ACT_DIM,) and float(a1.abs().max()) <= 1.0 + 1e-5, a1.shape
    # batched obs (mp-eval sends [B, obs_dim])
    aB = agent.act({"state": torch.randn(4, OBS_DIM, device=DEV)}, eval_mode=True)
    assert aB.shape == (4, ACT_DIM), aB.shape
    # exploration noise path
    a2 = agent.act({"state": torch.randn(OBS_DIM, device=DEV)}, eval_mode=False, stddev=0.3)
    assert a2.shape == (ACT_DIM,)
    print("OK  planner/act: single -> [act], batched -> [B,act], in [-1,1]")


def test_state_dict_roundtrip():
    agent = SimDistAgent(1, (OBS_DIM,), (9,), ACT_DIM, "cam", _cfg(), device=DEV)
    sd = agent.state_dict()
    agent2 = SimDistAgent(1, (OBS_DIM,), (9,), ACT_DIM, "cam", _cfg(), device=DEV)
    agent2.load_state_dict(sd)
    s = torch.randn(OBS_DIM, device=DEV)
    # same weights -> identical encode
    z1 = agent.model.encode(s.view(1, -1))
    z2 = agent2.model.encode(s.view(1, -1))
    assert torch.allclose(z1, z2), "state_dict round-trip mismatch"
    print("OK  state_dict round-trip")


# ----------------------------------------------------------- RAW-STATE arm
def _cfg_raw():
    return SimDistConfig(
        world_model=WorldModelConfig(encoder_type="identity", mlp_hidden_dim=128,
                                     num_layer=2, horizon=H),
        planner=PlannerConfig(horizon=H, num_samples=64, num_elites=8, iterations=2),
        finetune_steps=10,
    )


def test_raw_world_model():
    wm = WorldModel(OBS_DIM, ACT_DIM, WorldModelConfig(encoder_type="identity",
                                                       mlp_hidden_dim=128)).to(DEV)
    assert wm.raw_state and wm.encoder is None and wm.decoder is None and wm.enc_pi is None
    # the "latent" z is z-scored raw state: dim == obs_dim
    mean = torch.randn(OBS_DIM, device=DEV)
    std = torch.rand(OBS_DIM, device=DEV) + 0.5
    wm.set_normalization(mean, std)
    s = torch.randn(8, OBS_DIM, device=DEV)
    z = wm.encode(s)
    assert z.shape == (8, OBS_DIM), z.shape
    assert torch.allclose(z, (s - mean) / std, atol=1e-5), "encode != z-score"
    # decode is the EXACT analytic inverse of encode
    assert torch.allclose(wm.decode(z), s, atol=1e-4), "decode is not encode's inverse"
    # next is residual: z' = z + delta (shape obs_dim, generally != z)
    a = torch.rand(8, ACT_DIM, device=DEV) * 2 - 1
    zn = wm.next(z, a)
    assert zn.shape == (8, OBS_DIM) and not torch.allclose(zn, z)
    assert wm.predict_reward(z, a).shape == (8,)
    assert wm.predict_value(z).shape == (8,)
    pi = wm.policy_action(s)
    assert pi.shape == (8, ACT_DIM) and float(pi.abs().max()) <= 1.0 + 1e-5
    # no SimNorm module; norm buffers present and in state_dict
    assert wm.simnorm is None
    assert "norm_state_mean" in wm.state_dict() and "norm_state_std" in wm.state_dict()
    # value_target excluded from trainable
    assert len(wm.trainable_parameters()) < len(list(wm.parameters()))
    print("OK  raw_world_model: identity encode/decode inverse, residual next, no SimNorm/enc_pi/decoder")


def test_raw_distill():
    agent = SimDistAgent(1, (OBS_DIM,), (9,), ACT_DIM, "cam", _cfg_raw(), device=DEV)
    agent.set_normalization(torch.randn(OBS_DIM), torch.rand(OBS_DIM) + 0.5)
    before = {n: p.detach().clone() for n, p in agent.model.named_parameters()}
    losses = None
    for _ in range(5):
        losses = agent.update_distill(*_rand_segment())
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values()), losses
    assert losses["distill/decoder"] == 0.0, "raw-state has no decoder loss"
    # dynamics + policy + reward + value all move
    for prefix in ("dynamics.", "policy.", "reward.", "value."):
        moved = any(not torch.equal(before[n], p) for n, p in agent.model.named_parameters()
                    if n.startswith(prefix) and not n.startswith("value_target."))
        assert moved, f"{prefix} should move during raw distill"
    print("OK  raw_distill: losses finite, decoder loss==0, heads move", {k: round(v, 4) for k, v in losses.items()})


def test_raw_finetune():
    agent = SimDistAgent(1, (OBS_DIM,), (9,), ACT_DIM, "cam", _cfg_raw(), device=DEV)
    snap = {n: p.detach().clone() for n, p in agent.model.named_parameters()}
    for _ in range(5):
        m = agent.update_finetune(*_rand_segment())
    assert all(torch.isfinite(torch.tensor(v)) for v in m.values()), m
    for n, p in agent.model.named_parameters():
        moved = not torch.equal(snap[n], p)
        if n.startswith(("dynamics.", "policy.")):
            assert moved, f"{n} should move during raw finetune"
        elif n.startswith(("reward.", "value.", "value_target.")):
            assert not moved, f"{n} should be FROZEN during raw finetune"
    print("OK  raw_finetune: dynamics+policy move; reward/value frozen")


def test_raw_act_and_roundtrip():
    agent = SimDistAgent(1, (OBS_DIM,), (9,), ACT_DIM, "cam", _cfg_raw(), device=DEV)
    agent.set_normalization(torch.randn(OBS_DIM), torch.rand(OBS_DIM) + 0.5)
    a1 = agent.act({"state": torch.randn(OBS_DIM, device=DEV)}, eval_mode=True)
    assert a1.shape == (ACT_DIM,) and float(a1.abs().max()) <= 1.0 + 1e-5
    aB = agent.act({"state": torch.randn(4, OBS_DIM, device=DEV)}, eval_mode=True)
    assert aB.shape == (4, ACT_DIM), aB.shape
    # state_dict round-trip preserves norm buffers (eval self-normalizes)
    agent2 = SimDistAgent(1, (OBS_DIM,), (9,), ACT_DIM, "cam", _cfg_raw(), device=DEV)
    agent2.load_state_dict(agent.state_dict())
    assert torch.allclose(agent.model.norm_state_mean, agent2.model.norm_state_mean)
    assert torch.allclose(agent.model.norm_state_std, agent2.model.norm_state_std)
    s = torch.randn(OBS_DIM, device=DEV).view(1, -1)
    assert torch.allclose(agent.model.encode(s), agent2.model.encode(s))
    print("OK  raw act: single/batched + state_dict round-trip preserves norm buffers")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_sequence_replay()
    test_world_model()
    test_distill()
    test_finetune_freezes_heads()
    test_planner_and_act()
    test_state_dict_roundtrip()
    print("--- raw-state arm ---")
    test_raw_world_model()
    test_raw_distill()
    test_raw_finetune()
    test_raw_act_and_roundtrip()
    print("\nAll SimDist smoke tests passed (latent + raw-state).")

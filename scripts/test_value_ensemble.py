"""Smoke test for the SimDist value ENSEMBLE (CLAUDE.md §29).

The load-bearing property is BACK-COMPAT: num_value_heads=1 (default) must leave
the state_dict and every number byte-for-byte identical to the legacy single-head
model, so old ckpts load strictly and existing runs are unaffected.
"""
import sys

import torch

sys.path.insert(0, "/home/Chenyang/ibrl")

from rl.world_model import WorldModel, WorldModelConfig  # noqa: E402
from rl.simdist_agent import SimDistAgent, SimDistConfig  # noqa: E402
from rl.simdist_planner import PlannerConfig  # noqa: E402

OBS, ACT, B, H = 37, 7, 16, 5


def wm_cfg(**kw):
    d = dict(encoder_type="latent", latent_dim=64, mlp_hidden_dim=64, num_layer=2,
             discrete_heads=1, num_bins=51, shared_encoder=1, horizon=H)
    d.update(kw)
    return WorldModelConfig(**d)


def test_backcompat_state_dict():
    torch.manual_seed(0)
    legacy = WorldModel(OBS, ACT, wm_cfg())                      # default N=1
    torch.manual_seed(0)
    ens = WorldModel(OBS, ACT, wm_cfg(num_value_heads=1))
    k_legacy, k_ens = set(legacy.state_dict()), set(ens.state_dict())
    assert k_legacy == k_ens, f"N=1 state_dict differs: {k_legacy ^ k_ens}"
    assert not any("value_extra" in k for k in k_ens), "N=1 must add NO params"
    # a legacy ckpt must load strictly into an N=1 model
    ens.load_state_dict(legacy.state_dict())
    z = torch.randn(B, 64)
    assert torch.equal(legacy.predict_value(z), ens.predict_value(z)), "N=1 value differs"
    assert ens.num_value_heads() == 1
    print("[ok] N=1: state_dict identical, legacy ckpt loads strictly, value bit-identical")


def test_ensemble_shapes_and_reduce():
    torch.manual_seed(0)
    m = WorldModel(OBS, ACT, wm_cfg(num_value_heads=10, value_reduce="min"))
    assert m.num_value_heads() == 10
    assert len(m.value_extra) == 9 and len(m.value_target_extra) == 9
    z = torch.randn(B, 64)
    allv = m.predict_value_all(z)
    assert allv.shape == (10, B), allv.shape
    vmin, vmean = m.predict_value(z, reduce="min"), m.predict_value(z, reduce="mean")
    assert vmin.shape == (B,) and torch.all(vmin <= vmean + 1e-6), "min must be <= mean"
    assert torch.allclose(m.predict_value(z), vmin), "cfg.value_reduce=min not honored"
    # EMA targets excluded from stage-1 trainables
    names = {n for n, _ in m.named_parameters()}
    tp = {id(p) for p in m.trainable_parameters()}
    for n, p in m.named_parameters():
        if n.startswith("value_target"):
            assert id(p) not in tp, f"EMA target {n} must not be trainable"
    assert any(n.startswith("value_extra") for n in names)
    print("[ok] N=10: shapes, min<=mean, cfg.value_reduce honored, EMA targets excluded")


def _agent(**wm):
    cfg = SimDistConfig(world_model=wm_cfg(**wm),
                        planner=PlannerConfig(horizon=H, num_samples=8, num_elites=2,
                                              iterations=1))
    return SimDistAgent(True, (OBS,), (9,), ACT, "rl", cfg, device="cpu")


def test_distill_trains_all_heads():
    torch.manual_seed(0)
    ag = _agent(num_value_heads=5, value_reduce="min", value_bagging=0.5)
    states = torch.randn(B, H + 1, OBS)
    actions = torch.randn(B, H, ACT).clamp(-1, 1)
    rewards = torch.zeros(B, H)
    vt = torch.rand(B, H + 1)
    before = [p.clone() for hd in ag.model.value_heads() for p in hd.parameters()]
    out = ag.update_distill(states, actions, rewards, value_targets=vt)
    after = [p for hd in ag.model.value_heads() for p in hd.parameters()]
    moved = sum(1 for b, a in zip(before, after) if not torch.equal(b, a))
    assert moved == len(before), f"only {moved}/{len(before)} value params moved"
    assert "distill/value_disagree" in out and out["distill/value_disagree"] >= 0
    for k in ("distill/loss", "distill/value"):
        assert torch.isfinite(torch.tensor(out[k])), f"{k} not finite"
    print(f"[ok] N=5 distill: ALL heads train, disagree={out['distill/value_disagree']:.4f}")


def test_planner_uses_pessimistic_value():
    torch.manual_seed(0)
    ag = _agent(num_value_heads=6, value_reduce="min")
    state = torch.randn(OBS)
    a = ag.planner.plan(state, eval_mode=True)
    assert a.shape == (ACT,) and torch.all(a.abs() <= 1.0 + 1e-6)
    full = ag.planner.plan(state, eval_mode=True, return_full=True)
    assert full.shape == (H, ACT), full.shape
    print("[ok] planner plans through the min-ensemble value (single + return_full)")


def test_random_priors():
    """Priors must be in the state_dict, NEVER trained, and must create real
    disagreement (the whole point — §29's collapse fix)."""
    torch.manual_seed(0)
    m0 = WorldModel(OBS, ACT, wm_cfg(num_value_heads=10))                    # no prior
    torch.manual_seed(0)
    m1 = WorldModel(OBS, ACT, wm_cfg(num_value_heads=10, value_prior_scale=1.0))
    assert len(m0.value_prior) == 0, "prior_scale=0 must add NO params"
    assert len(m1.value_prior) == 10, "one prior net per head"
    assert any(k.startswith("value_prior.") for k in m1.state_dict()), \
        "priors must be in state_dict (eval must reproduce the ensemble)"
    tp = {id(p) for p in m1.trainable_parameters()}
    for n, p in m1.named_parameters():
        if n.startswith("value_prior."):
            assert id(p) not in tp, f"prior {n} must NOT be trainable"
    z = torch.randn(B, 64)
    d0 = m0.predict_value_all(z).std(0).mean()
    d1 = m1.predict_value_all(z).std(0).mean()
    assert d1 > d0, f"priors must increase disagreement ({d1:.4f} vs {d0:.4f})"
    print(f"[ok] random priors: in state_dict, not trainable, disagree {d0:.4f} -> {d1:.4f}")


def test_priors_frozen_through_distill():
    torch.manual_seed(0)
    ag = _agent(num_value_heads=5, value_reduce="min", value_prior_scale=1.0)
    before = [p.clone() for p in ag.model.value_prior.parameters()]
    heads_before = [p.clone() for hd in ag.model.value_heads() for p in hd.parameters()]
    out = ag.update_distill(torch.randn(B, H + 1, OBS),
                            torch.randn(B, H, ACT).clamp(-1, 1),
                            torch.zeros(B, H), value_targets=torch.rand(B, H + 1))
    for b, a in zip(before, ag.model.value_prior.parameters()):
        assert torch.equal(b, a), "prior moved during distill — must stay FIXED"
    moved = sum(1 for b, a in zip(heads_before,
                                  [p for hd in ag.model.value_heads() for p in hd.parameters()])
                if not torch.equal(b, a))
    assert moved == len(heads_before), "heads must still train with priors on"
    assert out["distill/value_disagree"] > 0
    print(f"[ok] distill w/ priors: priors FROZEN, heads train, "
          f"disagree={out['distill/value_disagree']:.4f}")


def test_single_head_agent_unchanged():
    torch.manual_seed(0)
    a1 = _agent()
    torch.manual_seed(0)
    a2 = _agent(num_value_heads=1)
    a1.model.load_state_dict(a2.model.state_dict())
    s = torch.randn(B, OBS)
    z = a1.model.encode(s)
    assert torch.equal(a1.model.predict_value(z), a2.model.predict_value(z))
    print("[ok] N=1 agent path bit-identical to legacy")


if __name__ == "__main__":
    test_backcompat_state_dict()
    test_ensemble_shapes_and_reduce()
    test_distill_trains_all_heads()
    test_planner_uses_pessimistic_value()
    test_random_priors()
    test_priors_frozen_through_distill()
    test_single_head_agent_unchanged()
    print("\nALL PASS")

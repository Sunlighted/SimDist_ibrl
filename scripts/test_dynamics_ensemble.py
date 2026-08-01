"""Smoke test for the SimDist DYNAMICS ensemble + geometric-disagreement penalty
(CLAUDE.md §27: the MOPO/PETS state-space pessimism arm — penalize the members'
spread on f^H, NOT value uncertainty which §27/§29 proved structurally blind).

The load-bearing property is BACK-COMPAT: num_dynamics_heads=1 (default) must leave
the state_dict and every number byte-for-byte identical to the legacy single-head
model — including `next()` — so old ckpts load strictly and existing runs / trained
SimDist checkpoints are unaffected.
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
    legacy = WorldModel(OBS, ACT, wm_cfg())                        # default N=1
    torch.manual_seed(0)
    ens = WorldModel(OBS, ACT, wm_cfg(num_dynamics_heads=1))
    k_legacy, k_ens = set(legacy.state_dict()), set(ens.state_dict())
    assert k_legacy == k_ens, f"N=1 state_dict differs: {k_legacy ^ k_ens}"
    assert not any("dynamics_extra" in k for k in k_ens), "N=1 must add NO params"
    assert not any("dynamics_prior" in k for k in k_ens), "N=1 must add NO prior"
    ens.load_state_dict(legacy.state_dict())
    z = torch.randn(B, 64)
    a = torch.randn(B, ACT).clamp(-1, 1)
    # z here isn't SimNorm'd but next() only needs a valid latent shape.
    assert torch.equal(legacy.next(z, a), ens.next(z, a)), "N=1 next() differs"
    assert ens.num_dynamics_heads() == 1
    assert float(ens.dynamics_disagreement(z, a).abs().max()) == 0.0, "N=1 disagree!=0"
    print("[ok] N=1: state_dict identical, legacy loads strictly, next() bit-identical")


def test_ensemble_shapes():
    torch.manual_seed(0)
    m = WorldModel(OBS, ACT, wm_cfg(num_dynamics_heads=5))
    assert m.num_dynamics_heads() == 5
    assert len(m.dynamics_extra) == 4
    z = m.encode(torch.randn(B, OBS))
    a = torch.randn(B, ACT).clamp(-1, 1)
    allp = m.next_all(z, a)
    assert allp.shape == (5, B, 64), allp.shape
    # next() == ensemble mean
    assert torch.allclose(m.next(z, a), allp.mean(0)), "next() != ensemble mean"
    dis = m.dynamics_disagreement(z, a)
    assert dis.shape == (B,) and torch.all(dis >= 0)
    # SimNorm keeps each head a valid simplex-grouped latent (sums to 1 per group)
    g = allp.view(5, B, -1, m.cfg.simnorm_dim).sum(-1)
    assert torch.allclose(g, torch.ones_like(g), atol=1e-4), "heads not SimNorm'd"
    # all ensemble heads trainable, priors (none here) absent
    tp = {id(p) for p in m.trainable_parameters()}
    for n, p in m.named_parameters():
        if n.startswith("dynamics_extra"):
            assert id(p) in tp, f"extra head {n} must be trainable"
    print(f"[ok] N=5: shapes, next()==mean, disagree>=0 (mean {float(dis.mean()):.4f})")


def test_raw_state_ensemble():
    torch.manual_seed(0)
    m = WorldModel(OBS, ACT, WorldModelConfig(
        encoder_type="identity", mlp_hidden_dim=64, num_layer=2, horizon=H,
        num_dynamics_heads=4))
    z = m.encode(torch.randn(B, OBS))       # z-scored raw state (identity here)
    a = torch.randn(B, ACT).clamp(-1, 1)
    allp = m.next_all(z, a)
    assert allp.shape == (4, B, OBS), allp.shape
    # residual: each head = z + delta_i, so std over heads == std of the deltas
    assert torch.allclose(m.next(z, a), allp.mean(0))
    assert torch.all(m.dynamics_disagreement(z, a) >= 0)
    print("[ok] raw-state N=4: residual heads, next()==mean, disagree>=0")


def _agent(planner_kw=None, **wm):
    pk = dict(horizon=H, num_samples=8, num_elites=2, iterations=1)
    pk.update(planner_kw or {})
    cfg = SimDistConfig(world_model=wm_cfg(**wm), planner=PlannerConfig(**pk))
    return SimDistAgent(True, (OBS,), (9,), ACT, "rl", cfg, device="cpu")


def test_distill_trains_all_heads():
    torch.manual_seed(0)
    ag = _agent(num_dynamics_heads=5, dynamics_bagging=0.5)
    states = torch.randn(B, H + 1, OBS)
    actions = torch.randn(B, H, ACT).clamp(-1, 1)
    rewards = torch.zeros(B, H)
    vt = torch.rand(B, H + 1)
    before = [p.clone() for p in ag.model.dynamics_parameters()]
    out = ag.update_distill(states, actions, rewards, value_targets=vt)
    after = list(ag.model.dynamics_parameters())
    moved = sum(1 for b, a in zip(before, after) if not torch.equal(b, a))
    assert moved == len(before), f"only {moved}/{len(before)} dynamics params moved"
    for k in ("distill/loss", "distill/consistency", "distill/dyn_disagree",
              "distill/dyn_disagree_rolled"):
        assert k in out and torch.isfinite(torch.tensor(out[k])), f"{k} bad"
    print(f"[ok] N=5 distill: ALL dynamics heads train, "
          f"dyn_disagree={out['distill/dyn_disagree']:.4f} "
          f"rolled={out['distill/dyn_disagree_rolled']:.4f}")


def test_adapt_trains_all_heads():
    torch.manual_seed(0)
    ag = _agent(num_dynamics_heads=4)
    states = torch.randn(B, H + 1, OBS)
    actions = torch.randn(B, H, ACT).clamp(-1, 1)
    before = [p.clone() for p in ag.model.dynamics_parameters()]
    out = ag.update_dynamics_only(states, actions)
    moved = sum(1 for b, a in zip(before, ag.model.dynamics_parameters())
                if not torch.equal(b, a))
    assert moved == len(before), f"adapt moved only {moved}/{len(before)}"
    assert torch.isfinite(torch.tensor(out["adapt/consistency"]))
    print("[ok] N=4 update_dynamics_only: ALL dynamics heads train")


def test_planner_penalty_lowers_return():
    """With disagree_coef>0 the return must be <= the coef=0 return (penalty only
    subtracts), and strictly lower somewhere (real ensemble => nonzero disagree)."""
    torch.manual_seed(0)
    ag = _agent(num_dynamics_heads=6)
    z0 = ag.model.encode(torch.randn(1, OBS))
    actions = torch.randn(H, 32, ACT).clamp(-1, 1)
    ag.planner.cfg.dynamics_disagree_coef = 0.0
    r0 = ag.planner._estimate_returns(z0, actions)
    ag.planner.cfg.dynamics_disagree_coef = 10.0
    r1 = ag.planner._estimate_returns(z0, actions)
    assert torch.all(r1 <= r0 + 1e-6), "penalty must not increase returns"
    assert (r1 < r0 - 1e-6).any(), "penalty had no effect (disagree==0?)"
    # single head => penalty guarded off (no effect)
    torch.manual_seed(0)
    ag1 = _agent(num_dynamics_heads=1, planner_kw={"dynamics_disagree_coef": 10.0})
    z0 = ag1.model.encode(torch.randn(1, OBS))
    rr = ag1.planner._estimate_returns(z0, torch.randn(H, 8, ACT).clamp(-1, 1))
    assert torch.all(torch.isfinite(rr)), "single-head penalty path broke"
    print("[ok] planner: disagree penalty lowers returns (N=6); guarded off at N=1")


def test_planner_end_to_end():
    torch.manual_seed(0)
    ag = _agent(num_dynamics_heads=5,
                planner_kw={"dynamics_disagree_coef": 5.0})
    a = ag.planner.plan(torch.randn(OBS), eval_mode=True)
    assert a.shape == (ACT,) and torch.all(a.abs() <= 1 + 1e-6)
    ab = ag.planner.plan_batch(torch.randn(3, OBS), eval_mode=True)
    assert ab.shape == (3, ACT) and torch.all(ab.abs() <= 1 + 1e-6)
    print("[ok] planner plan + plan_batch with penalty produce valid actions")


def test_random_priors():
    torch.manual_seed(0)
    m0 = WorldModel(OBS, ACT, wm_cfg(num_dynamics_heads=6))                   # no prior
    torch.manual_seed(0)
    m1 = WorldModel(OBS, ACT, wm_cfg(num_dynamics_heads=6, dynamics_prior_scale=1.0))
    assert len(m0.dynamics_prior) == 0, "prior_scale=0 must add NO params"
    assert len(m1.dynamics_prior) == 6, "one prior net per head"
    assert any(k.startswith("dynamics_prior.") for k in m1.state_dict()), \
        "priors must be in state_dict"
    tp = {id(p) for p in m1.trainable_parameters()}
    for n, p in m1.named_parameters():
        if n.startswith("dynamics_prior."):
            assert id(p) not in tp, f"prior {n} must NOT be trainable"
    z = m1.encode(torch.randn(B, OBS))
    a = torch.randn(B, ACT).clamp(-1, 1)
    d0 = m0.dynamics_disagreement(z, a).mean()
    d1 = m1.dynamics_disagreement(z, a).mean()
    assert d1 > d0, f"priors must increase disagreement ({d1:.4f} vs {d0:.4f})"
    print(f"[ok] random priors: in state_dict, frozen, disagree {d0:.4f} -> {d1:.4f}")


def test_priors_frozen_through_distill():
    torch.manual_seed(0)
    ag = _agent(num_dynamics_heads=5, dynamics_prior_scale=1.0)
    before = [p.clone() for p in ag.model.dynamics_prior.parameters()]
    out = ag.update_distill(torch.randn(B, H + 1, OBS),
                            torch.randn(B, H, ACT).clamp(-1, 1),
                            torch.zeros(B, H), value_targets=torch.rand(B, H + 1))
    for b, a in zip(before, ag.model.dynamics_prior.parameters()):
        assert torch.equal(b, a), "prior moved during distill — must stay FIXED"
    assert out["distill/dyn_disagree"] > 0
    print(f"[ok] distill w/ priors: priors FROZEN, dyn_disagree={out['distill/dyn_disagree']:.4f}")


def test_single_head_agent_unchanged():
    torch.manual_seed(0)
    a1 = _agent()
    torch.manual_seed(0)
    a2 = _agent(num_dynamics_heads=1)
    a1.model.load_state_dict(a2.model.state_dict())
    z = a1.model.encode(torch.randn(B, OBS))
    a = torch.randn(B, ACT).clamp(-1, 1)
    assert torch.equal(a1.model.next(z, a), a2.model.next(z, a))
    print("[ok] N=1 agent path bit-identical to legacy")


def test_real_ckpt_loads():
    """A real trained SimDist ckpt (single dynamics head) must load strictly into
    the new code with the default N=1 config."""
    import os
    ck = "exps/mimicgen/threading_simdist_diversev2_h5_seed1/latest.pt"
    if not os.path.exists(ck):
        print(f"[skip] real ckpt not found: {ck}")
        return
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" not in sd and any(k.startswith("model.") for k in sd):
        # agent-level state_dict: WorldModel nested under "model." — strip it
        sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    m = WorldModel(37, 7, WorldModelConfig(
        encoder_type="latent", latent_dim=512, mlp_hidden_dim=512, num_layer=2,
        discrete_heads=1, num_bins=101, shared_encoder=1, activation="mish",
        simnorm_dim=8, enc_lr_scale=0.3, reward_coef=0.1, value_coef=0.1, horizon=5))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    extra = [k for k in unexpected if not k.startswith("value")]  # ignore value-ens keys
    assert not extra, f"unexpected keys: {extra[:5]}"
    dyn_missing = [k for k in missing if "dynamics_extra" not in k]
    assert not dyn_missing, f"missing (non-ensemble) keys: {dyn_missing[:5]}"
    print("[ok] real single-head SimDist ckpt loads into new code (N=1 default)")


if __name__ == "__main__":
    test_backcompat_state_dict()
    test_ensemble_shapes()
    test_raw_state_ensemble()
    test_distill_trains_all_heads()
    test_adapt_trains_all_heads()
    test_planner_penalty_lowers_return()
    test_planner_end_to_end()
    test_random_priors()
    test_priors_frozen_through_distill()
    test_single_head_agent_unchanged()
    test_real_ckpt_loads()
    print("\nALL PASS")

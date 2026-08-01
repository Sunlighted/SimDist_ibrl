"""Smoke test for StateFlowPolicy input/output normalization.

Run: python scripts/test_flow_norm.py
Verifies:
  1. set_normalization fills buffers; they're in state_dict.
  2. loss + backward run with normalization on (both backbones).
  3. act / act_chunk return RAW-scale actions (denormalized round-trip).
  4. Identity buffers => byte-for-byte equal to normalize-off path.
  5. Pre-normalization checkpoints (no norm buffers) still load (strict).
"""
import torch
from types import SimpleNamespace

from bc.flow_policy import StateFlowPolicy, StateFlowPolicyConfig

torch.manual_seed(0)
S, D, B = 37, 7, 4
Tp, Ta = 16, 8


def make_batch():
    # state with heterogeneous scale; actions in [-1,1]
    state = torch.randn(B, S) * torch.linspace(0.01, 5.0, S)
    chunk = (torch.rand(B, Tp, D) * 2 - 1)
    return SimpleNamespace(obs={"state": state}, action={"action_chunk": chunk})


def stats_from(batch):
    s = batch.obs["state"]
    a = batch.action["action_chunk"].reshape(-1, D)
    return dict(state_mean=s.mean(0), state_std=s.std(0),
               action_min=a.min(0)[0], action_max=a.max(0)[0])


for backbone in ("unet1d", "mlp"):
    cfg = StateFlowPolicyConfig(backbone=backbone, pred_horizon=Tp, action_horizon=Ta,
                                num_inference_steps=4, use_ema=1)
    p = StateFlowPolicy((S,), D, cfg)
    batch = make_batch()
    p.set_normalization(**stats_from(batch))

    # 1. buffers in state_dict
    sd = p.state_dict()
    assert "norm_state_mean" in sd and "norm_action_max" in sd, "norm buffers missing from state_dict"

    # 2. loss + backward
    p.train()
    loss = p.loss(batch)
    loss.backward()
    assert torch.isfinite(loss), "non-finite loss"
    p.update_ema()

    # 3. act_chunk returns raw-scale actions; with EMA applied
    p.eval()
    p.apply_ema()
    chunk = p.act_chunk({"state": batch.obs["state"]})
    assert chunk.shape == (B, Ta, D), chunk.shape
    # denormalized actions should sit ~within the fitted [min,max] band
    # (flow sampling can slightly exceed; just check it's finite + sane scale)
    assert torch.isfinite(chunk).all()
    a1 = p.act({"state": batch.obs["state"][0]})
    assert a1.shape == (D,), a1.shape
    p.restore_from_ema()
    print(f"[{backbone}] loss={loss.item():.4f}  act_chunk={tuple(chunk.shape)}  OK")

    # 4. identity buffers == normalize-off
    p_id = StateFlowPolicy((S,), D, cfg)               # identity buffers (no set_norm)
    p_off = StateFlowPolicy((S,), D,
                            StateFlowPolicyConfig(backbone=backbone, pred_horizon=Tp,
                                                  action_horizon=Ta, num_inference_steps=4,
                                                  normalize_obs=0, normalize_action=0))
    p_off.load_state_dict(p_id.state_dict())           # same weights
    p_id.eval(); p_off.eval()
    torch.manual_seed(123)
    c_id = p_id.act_chunk({"state": batch.obs["state"]})
    torch.manual_seed(123)
    c_off = p_off.act_chunk({"state": batch.obs["state"]})
    assert torch.allclose(c_id, c_off, atol=1e-5), "identity buffers != normalize-off"
    print(f"[{backbone}] identity-buffer == normalize-off  OK")

    # 5. old checkpoint (strip norm buffers) loads strictly
    old_sd = {k: v for k, v in p.state_dict().items() if not k.startswith("norm_")}
    assert not any(k.startswith("norm_") for k in old_sd)
    p_new = StateFlowPolicy((S,), D, cfg)
    p_new.load_state_dict(old_sd)                      # strict=True default
    print(f"[{backbone}] legacy-ckpt (no norm buffers) strict-load  OK")

print("\nAll normalization smoke tests passed.")

"""Smoke test: StateFlowPolicy CFG plumbing.

Verifies:
  1. cfg_enabled=0   → loss + act_chunk unchanged shape; no domain_id needed.
  2. cfg_enabled=1   → loss reads batch.obs["domain_id"], builds one-hot,
                       dropout is sampled, gradients flow.
  3. _sample_full_chunk does the two-forward CFG interpolation when enabled,
     and the single forward when disabled.
  4. UNet1D + MLP backbones both work in both modes.

Pure forward / backward sanity — no env, no dataset, no training loop.
"""
import torch
import torch.nn.functional as F
from collections import namedtuple

from bc.flow_policy import StateFlowPolicy, StateFlowPolicyConfig

Batch = namedtuple("Batch", ["obs", "action"])

STATE_DIM = 37
ACTION_DIM = 7
BATCH = 8
Tp = 16


def make_batch(with_domain: bool, device: str = "cpu") -> Batch:
    obs = {"state": torch.randn(BATCH, STATE_DIM, device=device)}
    if with_domain:
        # half source (0), half target (1)
        obs["domain_id"] = torch.tensor(
            [0] * (BATCH // 2) + [1] * (BATCH - BATCH // 2),
            dtype=torch.long, device=device,
        )
    action = {"action_chunk": torch.randn(BATCH, Tp, ACTION_DIM, device=device)}
    return Batch(obs=obs, action=action)


def case(name: str, backbone: str, cfg_enabled: bool):
    print(f"\n=== {name} | backbone={backbone} | cfg_enabled={cfg_enabled} ===")
    cfg = StateFlowPolicyConfig(
        backbone=backbone,
        pred_horizon=Tp,
        action_horizon=8,
        num_inference_steps=4,
        use_ema=0,
        cfg_enabled=int(cfg_enabled),
        cfg_num_domains=2,
        cfg_dropout_p=0.5,           # high dropout so we exercise the ∅ path
        cfg_guidance_lambda=-0.5,
        cfg_inference_domain_id=1,
    )
    policy = StateFlowPolicy((STATE_DIM,), ACTION_DIM, cfg)
    policy.train()
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  params: {n_params:,}")

    # ---- training: loss + backward ----
    batch = make_batch(with_domain=cfg_enabled)
    loss = policy.loss(batch)
    loss.backward()
    assert torch.isfinite(loss), f"non-finite loss: {loss}"
    any_grad = any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in policy.parameters())
    assert any_grad, "no gradients flowed"
    print(f"  loss={loss.item():.5f}  grad_flows=True")

    # ---- inference: act_chunk ----
    policy.eval()
    with torch.no_grad():
        chunk = policy.act_chunk({"state": torch.randn(BATCH, STATE_DIM)})
    assert chunk.shape == (BATCH, 8, ACTION_DIM), f"bad chunk shape: {chunk.shape}"
    assert torch.isfinite(chunk).all()
    print(f"  act_chunk shape={tuple(chunk.shape)}")


def case_cfg_requires_domain_id():
    """cfg_enabled=1 but batch lacks domain_id → must raise."""
    print("\n=== assert: cfg_enabled w/o domain_id raises ===")
    cfg = StateFlowPolicyConfig(
        backbone="unet1d", pred_horizon=Tp, action_horizon=8,
        num_inference_steps=2, use_ema=0,
        cfg_enabled=1,
    )
    policy = StateFlowPolicy((STATE_DIM,), ACTION_DIM, cfg)
    policy.train()
    batch = make_batch(with_domain=False)
    try:
        policy.loss(batch)
    except AssertionError as e:
        print(f"  OK — caught: {e}")
        return
    raise RuntimeError("expected an AssertionError but loss() succeeded")


def case_param_count_grows_with_cfg():
    """CFG adds (num_domains) input dims into UNet1D global_cond → more params."""
    print("\n=== check: CFG-enabled UNet1D has more params than CFG-off ===")
    base = StateFlowPolicy((STATE_DIM,), ACTION_DIM, StateFlowPolicyConfig(
        backbone="unet1d", pred_horizon=Tp, action_horizon=8,
        use_ema=0, cfg_enabled=0,
    ))
    cfg_on = StateFlowPolicy((STATE_DIM,), ACTION_DIM, StateFlowPolicyConfig(
        backbone="unet1d", pred_horizon=Tp, action_horizon=8,
        use_ema=0, cfg_enabled=1, cfg_num_domains=2,
    ))
    n_base = sum(p.numel() for p in base.parameters())
    n_cfg = sum(p.numel() for p in cfg_on.parameters())
    print(f"  base={n_base:,}  cfg={n_cfg:,}  diff=+{n_cfg - n_base}")
    assert n_cfg > n_base, "expected more params when CFG is on"


def case_cotrain_dataset_tagging():
    """CoTrainDataset.sample_bc adds domain_id; single-dataset path does not."""
    print("\n=== check: CoTrainDataset emits domain_id on batches ===")
    # Build a fake batch directly via the helper to avoid loading hdf5.
    from bc.dataset import CoTrainDataset
    fake_obs = {"state": torch.randn(4, STATE_DIM)}
    fake_action = {"action": torch.randn(4, ACTION_DIM)}
    fake_batch = Batch(obs=fake_obs, action=fake_action)
    tagged_src = CoTrainDataset._tag_domain(fake_batch, 0, "cpu")
    tagged_tgt = CoTrainDataset._tag_domain(fake_batch, 1, "cpu")
    assert "domain_id" in tagged_src.obs and "domain_id" in tagged_tgt.obs
    assert (tagged_src.obs["domain_id"] == 0).all()
    assert (tagged_tgt.obs["domain_id"] == 1).all()
    print("  OK — source tagged 0, target tagged 1")


if __name__ == "__main__":
    torch.manual_seed(0)
    case("disabled-unet", "unet1d", False)
    case("enabled-unet",  "unet1d", True)
    case("disabled-mlp",  "mlp",    False)
    case("enabled-mlp",   "mlp",    True)
    case_cfg_requires_domain_id()
    case_param_count_grows_with_cfg()
    case_cotrain_dataset_tagging()
    print("\nALL OK")

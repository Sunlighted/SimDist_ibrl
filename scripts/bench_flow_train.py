"""Micro-benchmark flow BC training throughput: fp32 vs bf16-autocast vs
torch.compile, across batch sizes. No env build (build_env=False), no eval.
Run on a free GPU: CUDA_VISIBLE_DEVICES=2 python scripts/bench_flow_train.py
"""
import time, torch
from dataclasses import replace
from bc.dataset import RobomimicDataset, DatasetConfig
from bc.flow_policy import StateFlowPolicy, StateFlowPolicyConfig

DEV = "cuda:0"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

dcfg = DatasetConfig(
    path="/home/Chenyang/ibrl/release/data/mimicgen/threading_d0_source_sigma0.1/dataset.hdf5",
    use_state=1, action_horizon=16, num_data=200, eval_episode_len=400, val_ratio=0.0,
)
ds = RobomimicDataset(dcfg, build_env=False)
fcfg = StateFlowPolicyConfig(backbone="unet1d", unet_down_dims=[64,128,256],
    diffusion_step_embed_dim=128, kernel_size=5, n_groups=8, obs_feat_dim=128,
    obs_mlp_layers=[256], time_scale=1000.0, pred_horizon=16, action_horizon=8,
    num_inference_steps=10, use_ema=1, ema_decay=0.9999)
print("obs", ds.obs_shape, "act", ds.action_dim)


def build():
    p = StateFlowPolicy(ds.obs_shape, ds.action_dim, fcfg).to(DEV)
    if hasattr(p, "set_normalization"):
        try: p.set_normalization(**ds.get_normalization_stats())
        except Exception as e: print("norm skip", e)
    return p


def bench(tag, bs, amp=False, compile_=False, n=120, warmup=30):
    p = build()
    if compile_:
        p = torch.compile(p)
    opt = torch.optim.AdamW(p.parameters(), lr=1e-4, weight_decay=1e-6)
    batches = [ds.sample_bc(bs, DEV) for _ in range(8)]
    def step(i):
        b = batches[i % len(batches)]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss = p.loss(b)
        opt.zero_grad(); loss.backward(); opt.step()
        if hasattr(p, "update_ema"): p.update_ema()
        return loss
    for i in range(warmup): step(i)
    torch.cuda.synchronize(); t0 = time.time()
    for i in range(n): step(i)
    torch.cuda.synchronize(); dt = time.time() - t0
    sps = n / dt
    print(f"{tag:28s} bs={bs:4d}  {sps:7.1f} st/s  ({1000*dt/n:5.1f} ms/step)")
    del p, opt, batches; torch.cuda.empty_cache()
    return sps


print("="*60)
bench("fp32           ", 256)
bench("bf16-autocast  ", 256, amp=True)
bench("fp32           ", 1024)
bench("bf16-autocast  ", 1024, amp=True)
try:
    bench("compile fp32   ", 256, compile_=True)
    bench("compile bf16   ", 256, amp=True, compile_=True)
    bench("compile bf16   ", 1024, amp=True, compile_=True)
except Exception as e:
    print("compile failed:", repr(e)[:200])

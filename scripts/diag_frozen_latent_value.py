"""Is the FROZEN encoder's latent even ABLE to represent a per-state value?

The head-ablation (threading_simdist_ablate_val_seed1) trained ONLY the value
head on a frozen encoder and it (a) plateaued at CE~0.45, (b) collapsed to
predicting the mean (pred_mean==target_mean==0.357), (c) DROPPED MPPI 0.74->0.38.
This probe isolates the cause with NO env / NO MPPI: freeze the encoder, encode a
big pool of states, then OVERFIT a fresh value head z->value on those latents and
ask how well it can fit the per-state target — globally and in the near-success
"last mile". If it cannot fit even by overfitting, the frozen latent lacks the
information (an encoder bottleneck), and no head-only training can recover it.

Run source vs target back-to-back for the control:
  python scripts/diag_frozen_latent_value.py \
      --wm_ckpt exps/mimicgen/threading_simdist_adapt_seed1/latest.pt \
      --wm_cfg_dir exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1 \
      --target_data release/data/mimicgen/threading_d0_target_diverse_v2/dataset.hdf5 \
      --source_data release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5 \
      --num_data 3000 --gpu 0
"""
import argparse, os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, pyrallis, h5py


def load_states_values(path, num_data):
    f = h5py.File(path, "r"); g = f["data"]
    ids = sorted((k for k in g.keys() if k.startswith("demo_")),
                 key=lambda k: int(k.split("_")[1]))[:num_data]
    S, V = [], []
    for k in ids:
        s = np.array(g[k]["obs"]["state"], np.float32); v = np.array(g[k]["values"], np.float32)
        n = min(len(s), len(v)); S.append(s[:n]); V.append(v[:n])
        # near-success = last 10 steps of each trajectory
    near = np.concatenate([np.arange(len(x))[-10:] + off
                           for x, off in zip(V, np.cumsum([0] + [len(x) for x in V])[:-1])])
    f.close()
    return (torch.from_numpy(np.concatenate(S)), torch.from_numpy(np.concatenate(V)), near)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def encode_all(model, states, dev, bs=8192):
    zs = []
    for i in range(0, len(states), bs):
        zs.append(model.encode(states[i:i+bs].to(dev)).cpu())
    return torch.cat(zs)


def probe(tag, model, path, num_data, dev, steps=8000):
    S, V, near = load_states_values(path, num_data)
    Z = encode_all(model, S, dev).to(dev)          # FROZEN latent
    Vt = V.to(dev)
    zdim = Z.shape[1]
    # a GENEROUS fresh value head (deeper/wider than the real one) — we WANT it to
    # overfit; if even this can't fit, the latent is the bottleneck, not the head.
    head = nn.Sequential(nn.Linear(zdim, 1024), nn.LayerNorm(1024), nn.Mish(),
                         nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.Mish(),
                         nn.Linear(1024, 1)).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=3e-4)
    N = len(Z)
    for it in range(steps):
        idx = torch.randint(0, N, (4096,), device=dev)
        pred = head(Z[idx]).squeeze(-1)
        loss = F.mse_loss(pred, Vt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = head(Z).squeeze(-1).cpu().numpy()
    v = V.numpy()
    # variance explained (best case the frozen latent allows)
    r2 = 1.0 - np.mean((pred - v) ** 2) / (np.var(v) + 1e-9)
    sp_all = spearman(pred, v)
    sp_near = spearman(pred[near], v[near])
    # also: how much does the FROZEN latent itself vary across near-success states,
    # relative to how much the TARGET value varies there? (collapse signature)
    zc = Z.cpu().numpy()
    near_z_std = float(np.mean(np.std(zc[near], axis=0)))
    all_z_std = float(np.mean(np.std(zc, axis=0)))
    near_v_std = float(np.std(v[near])); all_v_std = float(np.std(v))
    print(f"\n[{tag}]  n={N}  (overfit a 2x1024 head on the FROZEN latent)")
    print(f"  fit R^2 (train=eval, overfit) : {r2:+.3f}   "
          f"({'latent CARRIES the value' if r2 > 0.8 else 'latent CANNOT represent per-state value (bottleneck)'})")
    print(f"  Spearman(pred, target) global : {sp_all:+.3f}")
    print(f"  Spearman(pred, target) near-succ: {sp_near:+.3f}   <-- the last mile")
    print(f"  latent std  near/all          : {near_z_std:.4f} / {all_z_std:.4f}  "
          f"(ratio {near_z_std/max(all_z_std,1e-9):.2f})")
    print(f"  target-V std near/all         : {near_v_std:.4f} / {all_v_std:.4f}  "
          f"(ratio {near_v_std/max(all_v_std,1e-9):.2f})")
    return dict(r2=r2, sp_all=sp_all, sp_near=sp_near)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm_ckpt", required=True)
    ap.add_argument("--wm_cfg_dir", required=True)
    ap.add_argument("--target_data", required=True)
    ap.add_argument("--source_data", required=True)
    ap.add_argument("--num_data", type=int, default=3000)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from train_simdist import MainConfig
    from rl.world_model import WorldModel

    dev = "cuda"
    base_cfg: MainConfig = pyrallis.load(MainConfig, open(os.path.join(args.wm_cfg_dir, "cfg.yaml")))  # type: ignore
    wm = base_cfg.simdist.world_model
    # obs/action dim from a data file
    f = h5py.File(args.target_data, "r"); k0 = sorted(f["data"].keys())[0]
    obs_dim = f["data"][k0]["obs"]["state"].shape[1]
    act_dim = f["data"][k0]["actions"].shape[1]; f.close()
    model = WorldModel(obs_dim, act_dim, wm).to(dev)
    sd = torch.load(args.wm_ckpt)
    msd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(msd); model.eval()
    print("Loaded FROZEN world model; encoder is the thing under test.")
    print(f"  encoder_type={wm.encoder_type} latent_dim={wm.latent_dim} "
          f"shared_encoder={wm.shared_encoder} discrete_heads={wm.discrete_heads}")

    rt = probe("TARGET latent", model, args.target_data, args.num_data, dev)
    rs = probe("SOURCE latent", model, args.source_data, args.num_data, dev)
    print("\n" + "=" * 70)
    print("VERDICT")
    print(f"  overfit R^2   source {rs['r2']:+.3f}  vs  target {rt['r2']:+.3f}")
    print(f"  near-succ Sp  source {rs['sp_near']:+.3f}  vs  target {rt['sp_near']:+.3f}")
    if rt["r2"] < 0.8 and rs["r2"] > rt["r2"] + 0.1:
        print("  => The FROZEN encoder's latent does NOT carry the target per-state value")
        print("     (esp. near success). No head-only value training can recover it;")
        print("     the encoder must be UNFROZEN / co-adapted, or MPPI must score with")
        print("     an EXTERNAL value on decoded states. Dynamics head-only works because")
        print("     its target IS the frozen encoder's own next output (representable by")
        print("     construction); value's target is EXTERNAL and was not preserved.")
    import sys; sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()

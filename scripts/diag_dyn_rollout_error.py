"""Cheap go/no-go for the Dyna-augmentation idea (CLAUDE.md §17/§23 + the
residual-speedup discussion): does a next-STATE dynamics model — trained on the
target-domain on-policy distribution the residual sees — predict accurately
enough, ESPECIALLY in the contact/insertion phase, for its imagined transitions
to be usable as TD-backup data for the residual critic?

Zero new env interaction. Reuses the already-dumped target rollouts
(threading_simdist_adapt_30k_seed1/rollouts.npz — 128 target MPPI-base episodes,
episode-contiguous). Measures HELD-OUT multi-step open-loop rollout error (feed
the REAL executed actions), bucketed by task phase, reported as needle-position
mm-MAE against the two physical yardsticks §27 established:
  * the target tolerance window = 1.8 mm/side (ring_size 0.85);
  * the per-step needle displacement (~3.7 mm) — the motion the model must track.

Two models:
  A) the EXISTING adapted SimDist latent world model (encode -> next x H -> decode;
     the literal "SimDist imagined data" — decode error is included because the
     residual critic eats raw state);
  B) a FRESH raw-state DynamicsEnsemble (rl/mb_ibrl_agent.py — the model MB-IBRL /
     Dyna would actually use), trained here on the train split.

Phase split is DATA-DRIVEN: the obs needle/ring are BODY positions (NOT the
geom_xpos the success check uses, §17), so their distance never reaches the 12 mm
success radius; we bucket by tertiles of the pooled body needle-ring distance and
cross-check with a steps-to-terminal proxy.

Run:
  python scripts/diag_dyn_rollout_error.py \
      --rollouts exps/mimicgen/threading_simdist_adapt_30k_seed1/rollouts.npz \
      --simdist_ckpt exps/mimicgen/threading_simdist_adapt_30k_seed1/model0.pt \
      --stage1_cfg exps/mimicgen/threading_simdist_diversev2_h5_seed1/cfg.yaml \
      --horizons 1,3,5 --model_b_steps 20000
"""
import argparse
import dataclasses
import sys

import numpy as np
import torch
import yaml

from rl.mb_ibrl_agent import DynamicsEnsemble, DynamicsEnsembleConfig
from rl.world_model import WorldModel, WorldModelConfig

NEEDLE = slice(9, 12)
RING = slice(23, 26)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_episodes(path):
    d = np.load(path, allow_pickle=True)
    eps = list(d["episodes"])
    out = []
    for e in eps:
        st = np.asarray(e["states"], dtype=np.float32)   # [T+1, 37]
        ac = np.asarray(e["actions"], dtype=np.float32)  # [T, 7]
        rw = np.asarray(e["rewards"], dtype=np.float32)  # [T]
        succ = bool(e["success"])
        if st.shape[0] >= 2 and ac.shape[0] == st.shape[0] - 1:
            out.append(dict(states=st, actions=ac, rewards=rw, success=succ))
    return out


def flatten(eps):
    S, A, R, S2, D = [], [], [], [], []
    for e in eps:
        st, ac, rw = e["states"], e["actions"], e["rewards"]
        T = ac.shape[0]
        S.append(st[:-1]); A.append(ac); S2.append(st[1:]); R.append(rw)
        done = np.zeros(T, dtype=np.float32)
        if e["success"]:
            done[-1] = 1.0
        D.append(done)
    return (np.concatenate(S), np.concatenate(A), np.concatenate(R),
            np.concatenate(S2), np.concatenate(D))


def build_model_a(simdist_ckpt, stage1_cfg, obs_dim, action_dim, device):
    with open(stage1_cfg) as f:
        cfg_all = yaml.safe_load(f)
    wm = cfg_all["simdist"]["world_model"]
    fields = {f.name for f in dataclasses.fields(WorldModelConfig)}
    wm_cfg = WorldModelConfig(**{k: v for k, v in wm.items() if k in fields})
    model = WorldModel(obs_dim, action_dim, wm_cfg).to(device)
    sd = torch.load(simdist_ckpt, map_location=device, weights_only=False)
    stripped = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    log(f"[A] WorldModel({wm_cfg.encoder_type}, H_train={wm_cfg.horizon}) "
        f"loaded; missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model


@torch.no_grad()
def rollout_a(model, s0, actions, H):
    # s0 [B,obs], actions [B,H,act] -> predicted state after H steps [B,obs]
    z = model.encode(s0)
    for k in range(H):
        z = model.next(z, actions[:, k])
    return model.decode(z)


def train_model_b(train_flat, mean, std, obs_dim, action_dim, device, steps, horizon):
    cfg = DynamicsEnsembleConfig()
    ens = DynamicsEnsemble(obs_dim, action_dim, cfg, device=device)
    ens.set_normalization(mean, std)
    S, A, R, S2, D = [torch.as_tensor(x, device=device) for x in train_flat]
    n = S.shape[0]
    bs = 256
    for it in range(steps):
        idx = torch.randint(0, n, (bs,), device=device)
        info = ens.train_step(S[idx], A[idx], R[idx], S2[idx], D[idx])
        if (it + 1) % max(1, steps // 5) == 0:
            log(f"[B] step {it+1}/{steps} nll={info.get('model/dyn_nll', float('nan')):.4f}")
    ens.eval()
    return ens


@torch.no_grad()
def rollout_b(ens, s0, actions, H):
    s = s0
    for k in range(H):
        s, _, _ = ens.predict(s, actions[:, k], sample=False)
    return s


def collect_starts(val_eps, H):
    # returns s0[N,obs], acts[N,H,act], sH[N,obs], dist0[N]
    S0, AC, SH = [], [], []
    for e in val_eps:
        st, ac = e["states"], e["actions"]
        T = ac.shape[0]
        for t in range(0, T - H + 1):
            S0.append(st[t]); AC.append(ac[t:t + H]); SH.append(st[t + H])
    S0 = np.asarray(S0); AC = np.asarray(AC); SH = np.asarray(SH)
    dist0 = np.linalg.norm(S0[:, NEEDLE] - S0[:, RING], axis=1)
    return S0, AC, SH, dist0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", default="exps/mimicgen/threading_simdist_adapt_30k_seed1/rollouts.npz")
    p.add_argument("--simdist_ckpt", default="exps/mimicgen/threading_simdist_adapt_30k_seed1/model0.pt")
    p.add_argument("--stage1_cfg", default="exps/mimicgen/threading_simdist_diversev2_h5_seed1/cfg.yaml")
    p.add_argument("--horizons", default="1,3,5")
    p.add_argument("--val_frac", type=float, default=0.22)
    p.add_argument("--model_b_steps", type=int, default=20000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    horizons = [int(x) for x in args.horizons.split(",")]
    device = args.device

    eps = load_episodes(args.rollouts)
    log(f"loaded {len(eps)} episodes; success rate "
        f"{np.mean([e['success'] for e in eps]):.2f}")
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(eps))
    n_val = max(1, int(round(len(eps) * args.val_frac)))
    val_eps = [eps[i] for i in idx[:n_val]]
    train_eps = [eps[i] for i in idx[n_val:]]
    log(f"split: {len(train_eps)} train / {len(val_eps)} val episodes")

    train_flat = flatten(train_eps)
    obs_dim, action_dim = train_flat[0].shape[1], train_flat[1].shape[1]
    mean = torch.as_tensor(train_flat[0].mean(0), device=device)
    std = torch.as_tensor(train_flat[0].std(0), device=device)

    model_a = build_model_a(args.simdist_ckpt, args.stage1_cfg, obs_dim, action_dim, device)
    model_b = train_model_b(train_flat, mean, std, obs_dim, action_dim, device,
                            args.model_b_steps, max(horizons))

    # data-driven phase buckets from the pooled val start distances
    _, _, _, all_dist = collect_starts(val_eps, min(horizons))
    q33, q66 = np.quantile(all_dist, [0.33, 0.66])
    buckets = [("insertion", -np.inf, q33), ("approach", q33, q66), ("free", q66, np.inf)]
    log(f"phase buckets by needle-ring BODY dist tertiles: "
        f"insertion<{q33:.3f}  approach {q33:.3f}-{q66:.3f}  free>{q66:.3f}")

    print("\n=== held-out multi-step needle-position MAE (mm) ===")
    print(f"{'H':>2} {'phase':>10} {'n':>6} {'noop':>7} {'moved':>7} "
          f"{'A(SimDist)':>11} {'B(Ensemble)':>12}")
    for H in horizons:
        S0, AC, SH, dist0 = collect_starts(val_eps, H)
        s0 = torch.as_tensor(S0, device=device)
        ac = torch.as_tensor(AC, device=device)
        sH = torch.as_tensor(SH, device=device)
        pa = rollout_a(model_a, s0, ac, H)
        pb = rollout_b(model_b, s0, ac, H)

        def mm(pred, mask):
            e = torch.norm((pred[:, NEEDLE] - sH[:, NEEDLE]), dim=1) * 1000.0
            return e[mask].mean().item() if mask.any() else float("nan")

        for name, lo, hi in buckets:
            m = torch.as_tensor((dist0 >= lo) & (dist0 < hi), device=device)
            n = int(m.sum())
            noop = (torch.norm(s0[:, NEEDLE] - sH[:, NEEDLE], dim=1) * 1000.0)  # = actual move
            moved = noop[m].mean().item() if m.any() else float("nan")
            print(f"{H:>2} {name:>10} {n:>6} {moved:>7.2f} {moved:>7.2f} "
                  f"{mm(pa, m):>11.2f} {mm(pb, m):>12.2f}")
        print("-" * 62)
    print("yardsticks:  tolerance window = 1.8 mm/side ;  per-step needle move ~3.7 mm")
    print("'noop'/'moved' = actual needle displacement over H steps (the signal to beat).")


if __name__ == "__main__":
    main()

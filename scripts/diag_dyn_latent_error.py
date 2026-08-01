"""Follow-up to diag_dyn_rollout_error: if the imagined data is consumed by a
LATENT-space value/critic (no decode to raw state), the decoder error is
irrelevant — only (1) latent consistency and (2) VALUE error matter. The latter
is what actually corrupts a Dyna backup y = gamma*V(z_hat'), and §27#5 showed
latent geometry and value can decouple, so measure value error DIRECTLY.

Model = the adapted SimDist latent world model (encode / next / value).
Metrics per phase per H (held-out target rollouts, real executed actions):
  latent  : ||next^H(encode(s_t),a) - encode(s_{t+H})||   vs the per-step latent move
  value   : |V(z_hat_H) - V(encode(s_{t+H}))|              vs the state-spread std[V]
"""
import argparse
import dataclasses
import sys

import numpy as np
import torch
import yaml

from rl.world_model import WorldModel, WorldModelConfig

NEEDLE = slice(9, 12)
RING = slice(23, 26)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_eps(path):
    d = np.load(path, allow_pickle=True)
    out = []
    for e in d["episodes"]:
        st = np.asarray(e["states"], dtype=np.float32)
        ac = np.asarray(e["actions"], dtype=np.float32)
        if st.shape[0] >= 2 and ac.shape[0] == st.shape[0] - 1:
            out.append((st, ac))
    return out


def build_model(ckpt, cfg_path, device):
    wm = yaml.safe_load(open(cfg_path))["simdist"]["world_model"]
    fl = {f.name for f in dataclasses.fields(WorldModelConfig)}
    cfg = WorldModelConfig(**{k: v for k, v in wm.items() if k in fl})
    m = WorldModel(37, 7, cfg).to(device).eval()
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")}, strict=False)
    return m


@torch.no_grad()
def value(m, z):
    v = m.predict_value(z)
    return v.reshape(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", default="exps/mimicgen/threading_simdist_adapt_30k_seed1/rollouts.npz")
    p.add_argument("--ckpt", default="exps/mimicgen/threading_simdist_adapt_30k_seed1/model0.pt")
    p.add_argument("--cfg", default="exps/mimicgen/threading_simdist_diversev2_h5_seed1/cfg.yaml")
    p.add_argument("--horizons", default="1,3,5")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    Hs = [int(x) for x in args.horizons.split(",")]
    dev = args.device

    eps = load_eps(args.rollouts)
    m = build_model(args.ckpt, args.cfg, dev)

    # reference scales from all start states
    allS = np.concatenate([st[:-1] for st, _ in eps]).astype("float32")
    with torch.no_grad():
        allZ = m.encode(torch.as_tensor(allS, device=dev))
        allV = value(m, allZ)
    v_std = allV.std().item()
    z_norm = allZ.norm(dim=1).mean().item()
    log(f"latent dim={allZ.shape[1]}  mean|z|={z_norm:.3f}   "
        f"V: mean={allV.mean().item():.3f} std={v_std:.3f} "
        f"range=[{allV.min().item():.3f},{allV.max().item():.3f}]")

    # collect (s0, acts[H], sH) then bucket by needle-ring body dist tertiles
    def starts(H):
        S0, AC, SH = [], [], []
        for st, ac in eps:
            T = ac.shape[0]
            for t in range(0, T - H + 1):
                S0.append(st[t]); AC.append(ac[t:t + H]); SH.append(st[t + H])
        return np.asarray(S0), np.asarray(AC), np.asarray(SH)

    S0, _, _ = starts(min(Hs))
    d0 = np.linalg.norm(S0[:, NEEDLE] - S0[:, RING], axis=1)
    q33, q66 = np.quantile(d0, [0.33, 0.66])
    buckets = [("insertion", -np.inf, q33), ("approach", q33, q66), ("free", q66, np.inf)]
    log(f"buckets: insertion<{q33:.3f}  approach {q33:.3f}-{q66:.3f}  free>{q66:.3f}")

    print(f"\nreference: mean|z|={z_norm:.2f}   std_states[V]={v_std:.3f}")
    print(f"{'H':>2} {'phase':>10} {'n':>6} | {'lat_move':>8} {'lat_err':>8} {'err/move':>8} "
          f"| {'|dV|':>7} {'|dV|/stdV':>9}")
    for H in Hs:
        S0, AC, SH = starts(H)
        s0 = torch.as_tensor(S0, device=dev)
        ac = torch.as_tensor(AC, device=dev)
        sH = torch.as_tensor(SH, device=dev)
        d0 = np.linalg.norm(S0[:, NEEDLE] - S0[:, RING], axis=1)
        with torch.no_grad():
            z = m.encode(s0)
            z0 = z.clone()
            for k in range(H):
                z = m.next(z, ac[:, k])
            zH_real = m.encode(sH)
            lat_err = (z - zH_real).norm(dim=1)
            lat_move = (zH_real - z0).norm(dim=1)
            dv = (value(m, z) - value(m, zH_real)).abs()
        for name, lo, hi in buckets:
            mk = (d0 >= lo) & (d0 < hi)
            mk_t = torch.as_tensor(mk, device=dev)
            n = int(mk.sum())
            lm = lat_move[mk_t].mean().item()
            le = lat_err[mk_t].mean().item()
            dvm = dv[mk_t].mean().item()
            print(f"{H:>2} {name:>10} {n:>6} | {lm:>8.3f} {le:>8.3f} {le/max(lm,1e-6):>8.2f} "
                  f"| {dvm:>7.4f} {dvm/max(v_std,1e-6):>9.2f}")
        print("-" * 74)
    print("lat_err/move > 1  => latent dynamics worse than 'no-op' at that horizon")
    print("|dV|/stdV        => value-target corruption relative to the value signal spread")


if __name__ == "__main__":
    main()

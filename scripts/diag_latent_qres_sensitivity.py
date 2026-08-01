"""Path-L gate: does the residual's OWN critic, built on the FROZEN SimDist
latent, tolerate the latent dynamics' prediction error?

Context (see CLAUDE.md §27#5 + the Dyna discussion): diag_dyn_latent_error showed
the latent dynamics is geometrically BAD (1-step error ~2x the real latent move)
yet SimDist's own value head barely notices (|dV|/std[V] = 0.02-0.03) — the error
hides in THAT head's null-space, by construction. The open risk: a DIFFERENT
value/critic trained on the same latent may read exactly the poorly-predicted
directions, so its Dyna backup y = gamma*V(z_hat') would be corrupted.

This script closes that gap AND builds the Path-L component itself:
  1. freeze SimDist's encoder, featurize every state z = encode(s);
  2. train an IQL critic Q(z,a) + V(z) offline on the target rollouts (same
     recipe as scripts/verify_iql_critic.py, latent features instead of raw state);
  3. HEALTH gate (§26): Spearman(V, -steps_to_go) + Pearson(Q, gamma^steps).
     If the latent critic cannot even fit, Path L dies here;
  4. THE test: z_hat' = next^H(encode(s_t), a) vs z'_real = encode(s_{t+H}) ->
     |V(z_hat') - V(z'_real)| and |Q(z_hat',a') - Q(z'_real,a')|, bucketed by
     phase, relative to the state-spread std. Compare against the SimDist-V
     baseline (0.02-0.03): similar => green light for latent Dyna.

Run:
  python scripts/diag_latent_qres_sensitivity.py --iql_steps 50000
"""
import argparse
import dataclasses
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml

from rl.residual_rl_agent import EnsembleQ
from rl.world_model import WorldModel, WorldModelConfig

NEEDLE = slice(9, 12)
RING = slice(23, 26)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class ValueMLP(nn.Module):
    def __init__(self, obs_dim, hidden, n_layer, layer_norm):
        super().__init__()
        dims = [obs_dim] + [hidden] * n_layer
        layers = []
        for i in range(n_layer):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, s):
        return self.net(s).squeeze(-1)


def spearman(x, y):
    def rank(v):
        o = np.argsort(v); r = np.empty_like(o, dtype=np.float64)
        r[o] = np.arange(len(v)); return r
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx, ry = rank(np.asarray(x)), rank(np.asarray(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def load_eps(path):
    d = np.load(path, allow_pickle=True)
    out = []
    for e in d["episodes"]:
        st = np.asarray(e["states"], dtype=np.float32)
        ac = np.asarray(e["actions"], dtype=np.float32)
        rw = np.asarray(e["rewards"], dtype=np.float32)
        if st.shape[0] >= 2 and ac.shape[0] == st.shape[0] - 1:
            out.append(dict(states=st, actions=ac, rewards=rw, success=bool(e["success"])))
    return out


def build_flat(eps, device):
    S, A, R, S2, D, STG = [], [], [], [], [], []
    for e in eps:
        st, ac, rw = e["states"], e["actions"], e["rewards"]
        T = len(ac)
        for t in range(T):
            S.append(st[t]); A.append(ac[t]); R.append(rw[t]); S2.append(st[t + 1])
            D.append(1.0 if (t == T - 1 and e["success"]) else 0.0)
            STG.append((T - 1 - t) if e["success"] else -1)
    to = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=device)
    return dict(s=to(S), a=to(A), r=to(R), s2=to(S2), done=to(D),
                stg=np.asarray(STG, dtype=np.int64))


def build_model(ckpt, cfg_path, device):
    wm = yaml.safe_load(open(cfg_path))["simdist"]["world_model"]
    fl = {f.name for f in dataclasses.fields(WorldModelConfig)}
    cfg = WorldModelConfig(**{k: v for k, v in wm.items() if k in fl})
    m = WorldModel(37, 7, cfg).to(device).eval()
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")}, strict=False)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def encode_chunked(m, s, bs=8192):
    return torch.cat([m.encode(s[i:i + bs]) for i in range(0, s.shape[0], bs)], 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", default="exps/mimicgen/threading_simdist_adapt_30k_seed1/rollouts.npz")
    p.add_argument("--ckpt", default="exps/mimicgen/threading_simdist_adapt_30k_seed1/model0.pt")
    p.add_argument("--cfg", default="exps/mimicgen/threading_simdist_diversev2_h5_seed1/cfg.yaml")
    p.add_argument("--horizons", default="1,3,5")
    p.add_argument("--iql_steps", type=int, default=50000)
    p.add_argument("--expectile", type=float, default=0.7)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--iql_lr", type=float, default=3e-4)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--num_layer", type=int, default=3)
    p.add_argument("--num_critics", type=int, default=2)
    p.add_argument("--ens_steps", type=int, default=15000,
                   help="raw DynamicsEnsemble train steps (--features raw only)")
    p.add_argument("--val_frac", type=float, default=0.22)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save", default="")
    p.add_argument("--features", default="latent", choices=["latent", "raw"],
                   help="CONTROL: 'raw' trains the SAME IQL on z-scored raw state "
                        "instead of the frozen SimDist latent, on the SAME data and "
                        "the SAME eval — the only way to compare the two feature "
                        "spaces apples-to-apples (the §26 numbers were measured on "
                        "clean expert demos, NOT on these rollouts, so they are not "
                        "a valid baseline for the latent health gate).")
    args = p.parse_args()
    Hs = [int(x) for x in args.horizons.split(",")]
    dev = args.device
    torch.manual_seed(args.seed)

    eps = load_eps(args.rollouts)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(eps))
    n_val = max(1, int(round(len(eps) * args.val_frac)))
    val_eps = [eps[i] for i in idx[:n_val]]
    train_eps = [eps[i] for i in idx[n_val:]]
    log(f"{len(train_eps)} train / {len(val_eps)} val episodes")

    m = build_model(args.ckpt, args.cfg, dev)
    tr = build_flat(train_eps, dev)
    va = build_flat(val_eps, dev)

    if args.features == "latent":
        featurize = lambda s: encode_chunked(m, s)
    else:
        # CONTROL: z-scored raw state (what the residual critic normally eats),
        # stats fit on the TRAIN split only.
        r_mean = tr["s"].mean(0); r_std = tr["s"].std(0).clamp_min(1e-6)
        featurize = lambda s: (s - r_mean) / r_std

    zs = featurize(tr["s"]); zs2 = featurize(tr["s2"])
    latent_dim = zs.shape[1]
    log(f"features={args.features}  dim={latent_dim}  train transitions={zs.shape[0]}")

    critic = EnsembleQ(latent_dim, 7, args.hidden, args.num_layer, True, args.num_critics).to(dev)
    import copy
    critic_target = copy.deepcopy(critic)
    for q in critic_target.parameters():
        q.requires_grad_(False)
    vf = ValueMLP(latent_dim, args.hidden, args.num_layer, True).to(dev)
    q_opt = torch.optim.Adam(critic.parameters(), lr=args.iql_lr)
    v_opt = torch.optim.Adam(vf.parameters(), lr=args.iql_lr)

    n = zs.shape[0]
    for step in range(args.iql_steps):
        i = torch.randint(0, n, (args.batch_size,), device=dev)
        z, a, r, z2, done = zs[i], tr["a"][i], tr["r"][i], zs2[i], tr["done"][i]
        with torch.no_grad():
            q_sa = critic_target(z, a).min(0).values
        v = vf(z)
        diff = q_sa - v
        w = torch.where(diff > 0, args.expectile, 1.0 - args.expectile)
        v_loss = (w * diff ** 2).mean()
        v_opt.zero_grad(set_to_none=True); v_loss.backward(); v_opt.step()
        with torch.no_grad():
            y = (r + args.gamma * (1.0 - done) * vf(z2)).clamp(0.0, 1.0)
        q = critic(z, a)
        q_loss = ((q - y.unsqueeze(0)) ** 2).mean()
        q_opt.zero_grad(set_to_none=True); q_loss.backward(); q_opt.step()
        with torch.no_grad():
            for pp, pt in zip(critic.parameters(), critic_target.parameters()):
                pt.mul_(1 - args.tau).add_(pp, alpha=args.tau)
        if (step + 1) % max(1, args.iql_steps // 8) == 0:
            log(f"  IQL {step+1}/{args.iql_steps} v_loss={v_loss.item():.4f} "
                f"q_loss={q_loss.item():.5f} q={q.mean().item():.3f} v={v.mean().item():.3f}")

    critic.eval(); vf.eval()

    # ---------------- [health] gate on VAL (successful episodes only) ---------
    zv = featurize(va["s"])
    with torch.no_grad():
        v_val = vf(zv).cpu().numpy()
        q_val = critic(zv, va["a"]).min(0).values.cpu().numpy()
    stg = va["stg"]
    ok = stg >= 0
    sp_global = spearman(v_val[ok], -stg[ok])
    r_cal = pearson(q_val[ok], args.gamma ** stg[ok])
    # per-episode spearman
    per, i = [], 0
    for e in val_eps:
        T = len(e["actions"])
        if e["success"]:
            per.append(spearman(v_val[i:i + T], -stg[i:i + T]))
        i += T
    per = [x for x in per if not np.isnan(x)]
    print("\n[health] latent IQL critic on held-out episodes")
    print(f"  Spearman(V, -steps_to_go): global {sp_global:+.3f}   "
          f"per-episode mean {np.mean(per):+.3f}  (gate: >0.3)")
    print(f"  Pearson(Q, gamma^steps)  : {r_cal:+.3f}  (gate: >0.6)")

    v_std = float(v_val.std()); q_std = float(q_val.std())
    print(f"  spread: std_states[V]={v_std:.3f}  std_states[Q]={q_std:.3f}")

    # ---------------- [the test] sensitivity to the imagined next state -------
    # Path L: imagined via the SimDist latent dynamics (encode -> next^H).
    # Path B: imagined via a raw-state DynamicsEnsemble (trained here), then
    #         featurized the same way the critic sees it. Same |dV|/std metric
    #         for both => the apples-to-apples decision number.
    ens = None
    if args.features == "raw":
        from rl.mb_ibrl_agent import DynamicsEnsemble, DynamicsEnsembleConfig
        ens = DynamicsEnsemble(37, 7, DynamicsEnsembleConfig(), device=dev)
        ens.set_normalization(tr["s"].mean(0), tr["s"].std(0).clamp_min(1e-6))
        log(f"training raw DynamicsEnsemble for {args.ens_steps} steps ...")
        for it in range(args.ens_steps):
            i = torch.randint(0, tr["s"].shape[0], (256,), device=dev)
            info = ens.train_step(tr["s"][i], tr["a"][i], tr["r"][i],
                                  tr["s2"][i], tr["done"][i])
            if (it + 1) % max(1, args.ens_steps // 4) == 0:
                log(f"  ens {it+1}/{args.ens_steps} nll={info.get('model/dyn_nll', float('nan')):.4f}")
        ens.eval()

    def starts(H):
        S0, AC, SH, ANEXT = [], [], [], []
        for e in val_eps:
            st, ac = e["states"], e["actions"]
            T = ac.shape[0]
            for t in range(0, T - H):  # need a_{t+H} to exist
                S0.append(st[t]); AC.append(ac[t:t + H]); SH.append(st[t + H]); ANEXT.append(ac[t + H])
        return (np.asarray(S0), np.asarray(AC), np.asarray(SH), np.asarray(ANEXT))

    S0, _, _, _ = starts(min(Hs))
    d0 = np.linalg.norm(S0[:, NEEDLE] - S0[:, RING], axis=1)
    q33, q66 = np.quantile(d0, [0.33, 0.66])
    buckets = [("insertion", -np.inf, q33), ("approach", q33, q66), ("free", q66, np.inf)]

    print(f"\n[test] corruption of the Dyna backup by the imagined latent")
    print(f"       (SimDist-V baseline for reference: |dV|/std = 0.02-0.03 @H=1)")
    print(f"{'H':>2} {'phase':>10} {'n':>6} | {'|dV_res|':>9} {'/std[V]':>8} "
          f"| {'|dQ_res|':>9} {'/std[Q]':>8}")
    for H in Hs:
        S0, AC, SH, AN = starts(H)
        s0 = torch.as_tensor(S0, device=dev); ac = torch.as_tensor(AC, device=dev)
        sH = torch.as_tensor(SH, device=dev); an = torch.as_tensor(AN, device=dev)
        d0 = np.linalg.norm(S0[:, NEEDLE] - S0[:, RING], axis=1)
        with torch.no_grad():
            if args.features == "latent":
                z = m.encode(s0)
                for k in range(H):
                    z = m.next(z, ac[:, k])
            else:
                s = s0
                for k in range(H):
                    s, _, _ = ens.predict(s, ac[:, k], sample=False)
                z = featurize(s)
            z_real = featurize(sH)
            dV = (vf(z) - vf(z_real)).abs()
            dQ = (critic(z, an).min(0).values - critic(z_real, an).min(0).values).abs()
        for name, lo, hi in buckets:
            mk = torch.as_tensor((d0 >= lo) & (d0 < hi), device=dev)
            n_ = int(mk.sum())
            dv_ = dV[mk].mean().item(); dq_ = dQ[mk].mean().item()
            print(f"{H:>2} {name:>10} {n_:>6} | {dv_:>9.4f} {dv_/max(v_std,1e-6):>8.2f} "
                  f"| {dq_:>9.4f} {dq_/max(q_std,1e-6):>8.2f}")
        print("-" * 62)
    print("verdict: /std ~= SimDist-V's 0.02-0.03 => latent Dyna backup is safe;")
    print("         /std >> that (e.g. >0.3) => the residual critic reads the")
    print("         badly-predicted latent directions => Path L needs rework.")

    if args.save:
        torch.save({"critic": critic.state_dict(), "vf": vf.state_dict(),
                    "latent_dim": latent_dim, "args": vars(args)}, args.save)
        log(f"saved latent critic -> {args.save}")


if __name__ == "__main__":
    main()

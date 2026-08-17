"""Reproduce the LOST sigma=0.02 demo set (adapt 0.952) as closely as possible.

That set's collecting checkpoint was overwritten by TopkSaver (CLAUDE.md 35 /
[[feedback-ckpt-provenance-topksaver]]): it came from step ~220000 of
`threading_ibrl_newtgt_seed1`, which no longer exists. What CAN be reproduced is
the DISTRIBUTION: `threading_ibrl_newtgt_snap_seed1` is the same recipe re-run
with `snapshot_period=20000`, so `snapshot_220000.pt` is the same *training
stage* of the same *recipe* (a different network, but statistically matched --
eval 0.96 vs the original's 0.98).

Given a large pool collected from that snapshot, pick the N demos that best
reproduce the original's per-episode distribution -- not just its means. We
match on (len, contact_frac, |da|) by greedy quantile alignment: sort the
original's episodes by each feature, then for each original episode pick the
unused pool episode closest to it in z-scored feature space. That reproduces the
original's *spread* (its len std is 14.4 while a raw pool sample is ~25) rather
than only its centre.

If the resulting adapt lands near 0.952, the original result was driven by the
data's reproducible properties, not by a one-off lucky checkpoint.

Usage:
    python scripts/match_original_profile.py --pool threading_d0_repro952_pool \
        --out threading_d0_repro952_n20 --n 20
"""
import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import h5py
import numpy as np

ORIG = "release/data/mimicgen/threading_d0_target_dynamics"
NEEDLE = slice(9, 12)
TRIPOD = slice(23, 26)
EEF = slice(0, 3)


def feats(path, n=None):
    with h5py.File(os.path.join(_REPO, path, "dataset.hdf5"), "r") as f:
        g = f["data"]
        keys = sorted(g.keys(), key=lambda k: int(k.split("_")[1]))
        if n:
            keys = keys[:n]
        S = [np.asarray(g[k]["obs"]["state"]) for k in keys]
        A = [np.asarray(g[k]["actions"]) for k in keys]
    L = np.array([len(a) for a in A], dtype=float)
    cf = np.array([
        float((np.linalg.norm(s[:len(a), NEEDLE] - s[:len(a), TRIPOD], axis=1)
               < np.linalg.norm(s[len(a)-1, NEEDLE] - s[len(a)-1, TRIPOD]) + 0.025).sum()) / len(a)
        for s, a in zip(S, A)])
    jk = np.array([float(np.abs(np.diff(a, axis=0)).mean()) for a in A])
    e0 = np.stack([s[0, EEF] for s in S])
    return dict(keys=keys, X=np.stack([L, cf, jk], 1), L=L, cf=cf, jk=jk, e0=e0)


def describe(tag, f, idx=None):
    s = slice(None) if idx is None else idx
    L, cf, jk, e0 = f["L"][s], f["cf"][s], f["jk"][s], f["e0"][s]
    r = np.linalg.norm(e0 - e0.mean(0), axis=1)
    print(f"  {tag:<26} n={len(np.atleast_1d(L)):>3} | len {L.mean():6.1f}±{L.std():5.1f} "
          f"[{L.min():.0f},{L.max():.0f}] | contact_frac {cf.mean():.3f}±{cf.std():.3f} | "
          f"|da| {jk.mean():.4f} | eef-init {r.mean()*1000:5.1f}mm | Σsteps {int(L.sum())}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    o = feats(ORIG, 20)
    pool = feats(f"release/data/mimicgen/{args.pool}")
    print("target / pool")
    describe("ORIGINAL (lost ckpt)", o)
    describe(f"POOL {args.pool}", pool)

    # z-score on the pool so the metric is scale-free, then greedy assignment
    mu, sd = pool["X"].mean(0), pool["X"].std(0) + 1e-9
    Z, T = (pool["X"] - mu) / sd, (o["X"] - mu) / sd
    # match the original's HARDEST episodes first (largest |z|) -- those are the
    # tails a greedy pass would otherwise fail to cover
    order = np.argsort(-np.linalg.norm(T, axis=1))
    # n may exceed the original's episode count: tile the target list so each
    # original episode is matched ceil(n/len) times. That reproduces the
    # original's distribution SHAPE at any n, instead of silently capping at len.
    reps = int(np.ceil(args.n / len(order)))
    order = np.concatenate([order] * reps)[: args.n]
    used, pick = set(), []
    for t in order:
        d = np.linalg.norm(Z - T[t], axis=1)
        d[list(used)] = np.inf
        j = int(np.argmin(d)); used.add(j); pick.append(j)
    idx = np.array(sorted(pick))
    print("\nselected")
    describe(f"MATCHED n={args.n}", pool, idx)

    if args.dry_run:
        return
    out = os.path.join(_REPO, "release/data/mimicgen", args.out)
    os.makedirs(out, exist_ok=True)
    src = h5py.File(os.path.join(_REPO, "release/data/mimicgen", args.pool, "dataset.hdf5"), "r")
    with h5py.File(os.path.join(out, "dataset.hdf5"), "w") as hf:
        gg = hf.create_group("data")
        for i, j in enumerate(idx):
            gs = src[f"data/{pool['keys'][j]}"]
            gd = gg.create_group(f"demo_{i}")
            for k in gs["obs"].keys():
                gd.create_dataset(f"obs/{k}", data=gs[f"obs/{k}"][:])
            for k in gs.keys():
                if k != "obs":
                    gd.create_dataset(k, data=gs[k][:])
            for ak, av in gs.attrs.items():
                gd.attrs[ak] = av
            gd.attrs["src_demo"] = pool["keys"][j]
        gg.attrs["num_demos"] = len(idx)
    src.close()
    cfg = json.load(open(os.path.join(_REPO, "release/data/mimicgen", args.pool, "env_cfg.json")))
    cfg["composition"] = {"num_demos_total": len(idx), "selected_from": args.pool,
                          "selection": "greedy 1-1 match to threading_d0_target_dynamics "
                                       "(the LOST-ckpt 0.952 set) on (len, contact_frac, |da|)",
                          "src_demos": [pool["keys"][j] for j in idx]}
    json.dump(cfg, open(os.path.join(out, "env_cfg.json"), "w"), indent=2)
    print(f"\n-> wrote {out}")


if __name__ == "__main__":
    main()

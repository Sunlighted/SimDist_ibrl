"""Pick a sigma=0.1 demo subset whose TRAJECTORY QUALITY matches the sigma=0.02
set, while KEEPING the sigma=0.1 initial-state coverage.

Why: the sigma0.02 set (threading_d0_target_dynamics, 20 demos) yields a much
better SimDist stage-2 base than the first-N prefix of the sigma0.1 pool
(threading_d0_target_expert_n500), but the two sets differ in FOUR ways at once
(hand-init sigma, collection script, strict-success gating, trajectory style).
Selecting a subset *within* the sigma0.1 pool isolates trajectory style: same
sigma, same expert, same script, same success criterion -- only the choice of
which 20/50 demos changes.

Quality profile (measured, see the audit):
                        len   contact_frac   |da|      eef-init |r|
    sigma0.02 (n=20)    99.2     0.210      0.0697        19.3mm
    sigma0.1  [0:20]    84.5     0.164      0.0887        77.8mm
    sigma0.1  [0:50]    83.1     0.167      0.0871        89.4mm
    sigma0.1  pool500   85.0     0.176      0.0875        94.1mm

Selection = STRATIFIED nearest-to-profile:
  1. sort the pool by eef-init |r| and cut it into `k` equal-count strata
     (k = number of demos wanted). Taking exactly one demo per stratum
     reproduces the pool's eef-init marginal by construction, so the sigma=0.1
     coverage is preserved and is NOT an optimization target.
  2. inside each stratum pick the demo minimizing the z-scored distance to the
     sigma0.02 profile on (len, contact_frac, |da|).

`--mode antimatch` inverts step 2 (pick the FURTHEST demo) to build a
worst-quality control at identical coverage.

Contact phase = #steps whose needle->tripod distance is within +25mm of that
episode's own terminal distance (the terminal obs distance is ~94mm for both
sets -- a body-origin offset, not a gap).

Usage:
    python scripts/select_matched_demos.py --n 20 50
    python scripts/select_matched_demos.py --n 20 --mode antimatch --tag anti
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

POOL = "release/data/mimicgen/threading_d0_target_expert_n500"
REF = "release/data/mimicgen/threading_d0_target_dynamics"
OUT_FMT = "release/data/mimicgen/threading_d0_target_expert_matched_n{n}"

NEEDLE = slice(9, 12)
TRIPOD = slice(23, 26)
EEF = slice(0, 3)


def features(path, n=None):
    """Per-demo (len, contact_frac, jerk, eef_init_r) + the raw arrays we need."""
    with h5py.File(os.path.join(_REPO, path, "dataset.hdf5"), "r") as f:
        g = f["data"]
        keys = sorted(g.keys(), key=lambda k: int(k.split("_")[1]))
        if n:
            keys = keys[:n]
        S = [np.asarray(g[k]["obs"]["state"]) for k in keys]
        A = [np.asarray(g[k]["actions"]) for k in keys]
    L = np.array([len(a) for a in A], dtype=float)
    contact = np.array([
        float((np.linalg.norm(s[:len(a), NEEDLE] - s[:len(a), TRIPOD], axis=1)
               < np.linalg.norm(s[len(a) - 1, NEEDLE] - s[len(a) - 1, TRIPOD]) + 0.025).sum())
        for s, a in zip(S, A)])
    jerk = np.array([float(np.abs(np.diff(a, axis=0)).mean()) if len(a) > 1 else 0.0 for a in A])
    e0 = np.stack([s[0, EEF] for s in S])
    eefr = np.linalg.norm(e0 - e0.mean(0), axis=1)
    return dict(keys=keys, len=L, contact_frac=contact / L, jerk=jerk,
                eefr=eefr, e0=e0, n0=np.stack([s[0, NEEDLE] for s in S]))


def select(pool, target, k, mode):
    """One demo per eef-init stratum, nearest (or furthest) to `target`."""
    X = np.stack([pool["len"], pool["contact_frac"], pool["jerk"]], axis=1)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    t = (target - mu) / sd
    d = np.linalg.norm(Z - t, axis=1)

    order = np.argsort(pool["eefr"])           # ascending eef-init |r|
    strata = np.array_split(order, k)          # k equal-count strata
    picked = []
    for st in strata:
        j = st[np.argmax(d[st])] if mode == "antimatch" else st[np.argmin(d[st])]
        picked.append(int(j))
    return np.array(sorted(picked))


def describe(tag, f, idx=None):
    s = slice(None) if idx is None else idx
    print(f"  {tag:<34} n={len(np.atleast_1d(f['len'][s])):>3}  "
          f"len {f['len'][s].mean():6.1f}  contact_frac {f['contact_frac'][s].mean():.3f}  "
          f"|da| {f['jerk'][s].mean():.4f}  "
          f"eef-init |r| {f['eefr'][s].mean()*1000:5.1f}mm "
          f"(p95 {np.percentile(f['eefr'][s],95)*1000:5.1f}, max {f['eefr'][s].max()*1000:5.1f})  "
          f"eef-init std/dim {np.round(f['e0'][s].std(0)*1000,1)}")


def write_subset(pool_dir, keys, idx, out_dir, note):
    os.makedirs(out_dir, exist_ok=True)
    src = h5py.File(os.path.join(_REPO, pool_dir, "dataset.hdf5"), "r")
    with h5py.File(os.path.join(out_dir, "dataset.hdf5"), "w") as hf:
        out = hf.create_group("data")
        for new_i, j in enumerate(idx):
            gs = src[f"data/{keys[j]}"]
            gd = out.create_group(f"demo_{new_i}")
            for k in gs["obs"].keys():
                gd.create_dataset(f"obs/{k}", data=gs[f"obs/{k}"][:])
            for k in gs.keys():
                if k != "obs":
                    gd.create_dataset(k, data=gs[k][:])
            for ak, av in gs.attrs.items():
                gd.attrs[ak] = av
            gd.attrs["src_demo"] = keys[j]
        out.attrs["num_demos"] = len(idx)
    src.close()
    cfg = json.load(open(os.path.join(_REPO, pool_dir, "env_cfg.json")))
    cfg["composition"] = {"num_demos_total": len(idx), "selected_from": pool_dir,
                          "selection": note, "src_demos": [keys[j] for j in idx]}
    json.dump(cfg, open(os.path.join(out_dir, "env_cfg.json"), "w"), indent=2)
    print(f"  -> wrote {out_dir}  ({len(idx)} demos)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default=POOL,
                   help="demo pool to select from (default: the sigma=0.1 expert_n500 pool). "
                        "Pass another single-policy pool to check whether the SAME selection "
                        "method transfers across pools.")
    p.add_argument("--out_fmt", default=OUT_FMT,
                   help="output dir template, must contain {n}")
    p.add_argument("--n", type=int, nargs="+", default=[20, 50])
    p.add_argument("--mode", choices=["match", "antimatch"], default="match")
    p.add_argument("--tag", default="")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    ref = features(REF, 20)
    pool = features(args.pool)
    target = np.array([ref["len"].mean(), ref["contact_frac"].mean(), ref["jerk"].mean()])

    print("reference / baselines")
    describe("sigma0.02 (target_dynamics, n=20)", ref)
    describe(f"pool ALL ({os.path.basename(args.pool)})", pool)
    describe("pool prefix [0:20]", pool, np.arange(20))
    describe("pool prefix [0:50]", pool, np.arange(50))
    print(f"\nprofile target (len, contact_frac, |da|) = {np.round(target,4)}   mode={args.mode}\n")

    for n in args.n:
        idx = select(pool, target, n, args.mode)
        describe(f"SELECTED n={n}", pool, idx)
        overlap = len(set(idx.tolist()) & set(range(n)))
        print(f"      overlap with the prefix[0:{n}] the old run used: {overlap}/{n} demos")
        if not args.dry_run:
            suffix = f"_{args.tag}" if args.tag else ""
            out = os.path.join(_REPO, args.out_fmt.format(n=n) + suffix)
            write_subset(args.pool, pool["keys"], idx,
                         out, f"{args.mode}-sigma0.02-profile, eef-init stratified")
        print()


if __name__ == "__main__":
    main()

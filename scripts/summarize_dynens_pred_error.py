"""Aggregate diag_dynens_pred_error.py per-state dumps into the head-count table.

Every number is a mean over states; the CI is a bootstrap over EPISODES (states
within an episode are correlated, so resampling states would be too narrow).
Deltas are PAIRED (all models were scored on the identical (s, candidates)).

  python scripts/summarize_dynens_pred_error.py exps/diag_dynens_pred_error/target_perstate.npz
"""
import sys

import numpy as np

METRICS = [  # key, label, reducer
    ("Jbase_spearman", "rank ρ (base)", "mean"),
    ("Jall_spearman", "rank ρ (all)", "mean"),
    ("Jbase_regret", "regret/σ (base)", "mean"),
    ("Jall_regret", "regret/σ (all)", "mean"),
    ("Jbase_noise", "std ε̃ / signal (base)", "ratio:Jbase_signal"),
    ("Jall_noise", "std ε̃ / signal (all)", "ratio:Jall_signal"),
    ("Jall_ebar", "std_s ε̄ / signal (all)", "std_ratio:Jall_signal"),
    ("t1_mm", "needle mm t=1", "mean"),
    ("t5_mm", "needle mm t=5", "mean"),
    ("t5_mm_common", "  common (rms) t=5", "rms"),
    ("t5_mm_spread", "  spread (rms) t=5", "rms"),
    ("t1_zrel", "latent rel t=1", "mean"),
    ("t5_zrel", "latent rel t=5", "mean"),
]


def reduce(d, name, key, how, idx):
    x = d[f"{name}/{key}"][idx]
    if how == "mean":
        return x.mean()
    if how == "rms":
        return np.sqrt((x ** 2).mean())
    kind, den = how.split(":")
    s = d[f"{name}/{den}"][idx].mean()
    return (x.mean() if kind == "ratio" else x.std()) / s


def main():
    path = sys.argv[1]
    ref = sys.argv[2] if len(sys.argv) > 2 else None
    nboot = 2000
    d = np.load(path)
    ep = d["episode"]
    eps = np.unique(ep)
    rows = {ep_: np.where(ep == ep_)[0] for ep_ in eps}
    names = []
    for k in d.files:
        if "/" in k and k.split("/")[0] not in names:
            names.append(k.split("/")[0])
    rng = np.random.default_rng(0)
    boots = [np.concatenate([rows[e] for e in rng.choice(eps, len(eps))]) for _ in range(nboot)]
    full = np.arange(len(ep))
    near = d["near"].astype(bool)
    print(f"# {path}: {len(ep)} states / {len(eps)} episodes; CI = 95% episode bootstrap")
    for key, label, how in METRICS:
        print(f"\n## {label}  [{key}]")
        base = None
        if ref:
            base = np.array([reduce(d, ref, key, how, b) for b in boots])
        for n in names:
            v = reduce(d, n, key, how, full)
            bs = np.array([reduce(d, n, key, how, b) for b in boots])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            vn = reduce(d, n, key, how, full[near]) if how == "mean" else np.nan
            line = f"  {n:14s} {v:8.3f}  [{lo:7.3f},{hi:7.3f}]   near-tertile {vn:7.3f}"
            if base is not None and n != ref:
                dl = bs - base
                line += (f"   Δ vs {ref}: {v - reduce(d, ref, key, how, full):+.3f} "
                         f"[{np.percentile(dl, 2.5):+.3f},{np.percentile(dl, 97.5):+.3f}]")
            print(line)


if __name__ == "__main__":
    main()

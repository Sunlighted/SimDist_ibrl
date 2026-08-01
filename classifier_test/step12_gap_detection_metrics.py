"""Step 12: quantify gap detection as BINARY CLASSIFICATION on in-distribution data.

Frame the per-transition question as: "is there a real dynamics gap here?"
  label y = 1  if  GT_gap > tau_gap   (a real spike / contact gap)
          = 0  otherwise              (flat free-space, ~no gap)

For each detector variant we score every transition and measure how well the
score separates gap vs no-gap:
  - AUROC, AP : threshold-FREE ranking quality (fair across variants whose
                absolute offset differs, e.g. mean-2std is shifted down).
  - best-F1   : sweep the detector threshold, report the operating point that
                maximizes F1 (precision / recall / accuracy there). Each variant
                gets its OWN best threshold -> fair operating-point comparison.

All on IN-DISTRIBUTION rollouts (source IBRL policy in source env), which is
where we actually want to read off a gap. GT_gap thresholds are swept over a
sensible range chosen from the gap distribution (flat ~0.005 vs spikes >0.03).
"""

import argparse
import os

import numpy as np
import torch
from scipy.stats import rankdata

from cf_utils import REPO, TARGET_SHIFT
from step10_eval_ensemble import load_ensemble, member_scores, rollout


def auroc(y, s):
    """Rank-based AUROC (Mann-Whitney). y in {0,1}, s = scores."""
    y = np.asarray(y); s = np.asarray(s)
    n_pos = y.sum(); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def average_precision(y, s):
    """Average precision = area under precision-recall (step interpolation)."""
    y = np.asarray(y); s = np.asarray(s)
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1)
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")
    rec = tp / n_pos
    # AP = sum over positives of precision at that recall step
    return float((prec * y).sum() / n_pos)


def best_f1(y, s, n_grid=200):
    """Sweep threshold over score quantiles; return best-F1 operating point."""
    y = np.asarray(y); s = np.asarray(s)
    n_pos = y.sum()
    if n_pos == 0:
        return dict(thr=float("nan"), f1=0, prec=0, rec=0, acc=float((y == 0).mean()))
    thrs = np.quantile(s, np.linspace(0, 1, n_grid))
    best = dict(f1=-1)
    for thr in np.unique(thrs):
        pred = s >= thr
        tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0)); fn = np.sum(~pred & (y == 1))
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best["f1"]:
            acc = np.mean(pred == (y == 1))
            best = dict(thr=float(thr), f1=float(f1), prec=float(prec),
                        rec=float(rec), acc=float(acc))
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weight", default=f"{REPO}/exps/mimicgen/threading_ibrl_source_seed1/model0.pt")
    p.add_argument("--ens_dir", default=f"{os.path.dirname(__file__)}/disc_ens")
    p.add_argument("--num_traj", type=int, default=16)
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--gap_thresholds", default="0.015,0.02,0.025,0.03,0.04")
    p.add_argument("--fixed_gap_thr", type=float, default=0.025,
                   help="the gap threshold used for the detailed best-F1 table")
    args = p.parse_args()
    dev = "cuda"

    import train_rl
    from env.robosuite_wrapper import PixelRobosuite
    from env.domain_shift import DomainShiftConfig

    agent, _, env_params = train_rl.load_model(args.weight, dev)
    src_params = dict(env_params); src_params["domain_shift"] = DomainShiftConfig(mode="none")
    src_env = PixelRobosuite(**src_params)
    tgt_params = dict(env_params); tgt_params["domain_shift"] = DomainShiftConfig(**TARGET_SHIFT)
    tgt_env = PixelRobosuite(**tgt_params); tgt_env.reset()
    members = load_ensemble(args.ens_dir, dev)

    trajs = rollout(agent, src_env, tgt_env, args.num_traj, args.seed, dev)
    succ = [t["success"] for t in trajs]
    S = np.concatenate([t["s"] for t in trajs])
    A = np.concatenate([t["a"] for t in trajs])
    NS = np.concatenate([t["ssrc"] for t in trajs])
    gap = np.concatenate([t["gap"] for t in trajs])
    sc = member_scores(members, S, A, NS, dev)           # (M, N)
    mean, std = sc.mean(0), sc.std(0)

    variants = {
        "single(m0)": sc[0],
        "ens mean": mean,
        "mean-0.5std": mean - 0.5 * std,
        "mean-1std": mean - 1.0 * std,
        "mean-2std": mean - 2.0 * std,
    }

    print(f"IN-DIST source RL: {len(trajs)} trajs ({sum(succ)} success), "
          f"{len(gap)} transitions")
    print(f"GT gap: median={np.median(gap):.4f} p80={np.quantile(gap,0.8):.4f} "
          f"p90={np.quantile(gap,0.9):.4f} p95={np.quantile(gap,0.95):.4f} max={gap.max():.4f}")

    # ---- threshold-free ranking (AUROC / AP) across GT gap thresholds ----
    gap_thrs = [float(x) for x in args.gap_thresholds.split(",")]
    print(f"\n=== AUROC (threshold-free ranking: detect GT_gap > tau) ===")
    header = "  tau_gap   pos%   " + "  ".join(f"{k:>11s}" for k in variants)
    print(header)
    for tg in gap_thrs:
        y = (gap > tg).astype(int)
        cells = "  ".join(f"{auroc(y, v):>11.3f}" for v in variants.values())
        print(f"  {tg:.3f}  {100*y.mean():5.1f}%  {cells}")
    print(f"\n=== AP / average precision (same) ===")
    print(header)
    for tg in gap_thrs:
        y = (gap > tg).astype(int)
        cells = "  ".join(f"{average_precision(y, v):>11.3f}" for v in variants.values())
        print(f"  {tg:.3f}  {100*y.mean():5.1f}%  {cells}")

    # ---- detailed best-F1 operating point at the chosen GT gap threshold ----
    tg = args.fixed_gap_thr
    y = (gap > tg).astype(int)
    print(f"\n=== best-F1 operating point  (GT gap > {tg} -> positive, "
          f"{100*y.mean():.1f}% positive) ===")
    print(f"  {'variant':12s} {'AUROC':>7s} {'AP':>6s} {'thr*':>7s} {'F1':>6s} "
          f"{'prec':>6s} {'recall':>7s} {'acc':>6s}")
    for k, v in variants.items():
        b = best_f1(y, v)
        print(f"  {k:12s} {auroc(y,v):>7.3f} {average_precision(y,v):>6.3f} "
              f"{b['thr']:>7.3f} {b['f1']:>6.3f} {b['prec']:>6.3f} "
              f"{b['rec']:>7.3f} {b['acc']:>6.3f}")

    # also report a baseline: random / always-positive
    print(f"  {'(baseline)':12s} always-positive F1={2*y.mean()/(1+y.mean()):.3f} "
          f"acc={y.mean():.3f} ; chance AUROC=0.500")

    # ---- WITHIN-trajectory AUROC (controls for cross-traj baseline offset) ----
    # Pooled global-threshold AUROC can be near chance if each trajectory's score
    # baseline differs. Per-traj AUROC measures whether, WITHIN one episode, the
    # score ranks gap>tau transitions above the flat ones. Averaged over trajs
    # that contain both classes.
    print(f"\n=== WITHIN-trajectory mean AUROC (GT gap > {tg}); "
          f"pooled vs per-traj reveals cross-traj offset ===")
    offs = np.cumsum([0] + [len(t["gap"]) for t in trajs])
    for k, v in variants.items():
        aucs = []
        for ti in range(len(trajs)):
            sl = slice(offs[ti], offs[ti + 1])
            yt = (gap[sl] > tg).astype(int)
            if yt.sum() == 0 or yt.sum() == len(yt):
                continue
            aucs.append(auroc(yt, v[sl]))
        m = np.nanmean(aucs) if aucs else float("nan")
        print(f"  {k:12s} per-traj AUROC mean={m:.3f}  (over {len(aucs)} trajs "
              f"with both classes)  | pooled={auroc(y, v):.3f}")


if __name__ == "__main__":
    main()

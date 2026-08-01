"""Per-trajectory match metrics between a predicted gap curve and the GT gap.

The question: WITHIN one trajectory, does the predicted-gap curve light up at the
same timesteps as the ground-truth gap? Three complementary numbers:
  - rho  : Spearman rank correlation (offset/scale-free; "right ordering")
  - cos  : cosine of the mean-subtracted curves ("right shape")
  - prk  : peak recall@q — of the top-q% GT-gap steps, the fraction that land in
           the top-q% predicted steps ("are the spikes in the right place")
Averaged (nan-safe) over trajectories that contain variation.
"""

import numpy as np
from scipy.stats import spearmanr


def rolling(x, k, mode="mean"):
    """Centered window aggregation along a 1-D trajectory."""
    if k <= 1:
        return np.asarray(x, float)
    x = np.asarray(x, float); n = len(x); pad = k // 2
    out = np.empty(n)
    for i in range(n):
        w = x[max(0, i - pad):min(n, i + pad + 1)]
        out[i] = w.mean() if mode == "mean" else w.max()
    return out


def traj_match(pred, gt, q=0.2):
    pred = np.asarray(pred, float); gt = np.asarray(gt, float)
    if len(gt) < 4 or gt.std() < 1e-9 or pred.std() < 1e-9:
        return None
    rho = spearmanr(pred, gt)[0]
    p = pred - pred.mean(); g = gt - gt.mean()
    cos = float(p @ g / (np.linalg.norm(p) * np.linalg.norm(g) + 1e-12))
    k = max(1, int(round(len(gt) * q)))
    gt_top = set(np.argsort(gt)[-k:].tolist())
    pr_top = set(np.argsort(pred)[-k:].tolist())
    prk = len(gt_top & pr_top) / len(gt_top)
    return dict(rho=rho, cos=cos, prk=prk)


def mean_match(preds, gts, q=0.2):
    """preds/gts: lists of per-trajectory 1-D arrays. Returns mean metrics."""
    ms = [traj_match(p, g, q) for p, g in zip(preds, gts)]
    ms = [m for m in ms if m is not None]
    if not ms:
        return dict(rho=float("nan"), cos=float("nan"), prk=float("nan"), n=0)
    return dict(rho=float(np.nanmean([m["rho"] for m in ms])),
                cos=float(np.nanmean([m["cos"] for m in ms])),
                prk=float(np.nanmean([m["prk"] for m in ms])),
                n=len(ms))

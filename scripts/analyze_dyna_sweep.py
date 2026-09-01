"""Read the Dyna trust-region sweep straight out of the train.log files.

The read-out is the PAIRED delta (75-100k steady state) - (round-0 score): stage-2
BC is untouched by every knob in the sweep, so round-0 is identical per seed and
the delta removes the 20-demo sampling noise (sigma~0.092) that dominates the
absolute score. That pairing is what turned the baseline's ambiguous 0.132 into
t=-3.72 (p=0.001) over 20 runs.

Usage: python scripts/analyze_dyna_sweep.py [exps/simdist_threading_3arm]
"""
import glob
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "exps/simdist_threading_3arm"
STEADY_FROM = 75000


def parse(path):
    """-> (round0, steady_mean, n_points, last_step) or None."""
    r0, steady, last = None, [], 0
    with open(path) as f:
        for line in f:
            if "score/score " not in line and "score/score:" not in line:
                continue
            parts = line.split(":")
            # exact key -- dynens logs score/score_best, score/score_c0, ...
            if len(parts) < 3 or parts[1].strip() != "score/score":
                continue
            try:
                step, val = int(parts[0]), float(parts[2])
            except ValueError:
                continue
            last = max(last, step)
            if step == 0:
                r0 = val
            if step >= STEADY_FROM:
                steady.append(val)
    if r0 is None or not steady:
        return None
    return r0, float(np.mean(steady)), len(steady), last


cells = defaultdict(dict)
for d in sorted(glob.glob(os.path.join(ROOT, "*_seed*"))):
    log = os.path.join(d, "train.log")
    if not os.path.isfile(log):
        continue
    name = os.path.basename(d)
    cell, seed = name.rsplit("_seed", 1)
    got = parse(log)
    if got:
        cells[cell][int(seed)] = got

base = cells.get("dyna_h5", {})
print(f"{'cell':22s} {'n':>2s} {'round0':>7s} {'steady':>7s} {'delta':>7s} "
      f"{'sd':>6s} {'vs base (paired)':>18s}")
print("-" * 78)
for cell in sorted(cells):
    rows = cells[cell]
    d = np.array([v[1] - v[0] for v in rows.values()])
    r0 = np.array([v[0] for v in rows.values()])
    st = np.array([v[1] for v in rows.values()])
    sd = d.std(ddof=1) if len(d) > 1 else float("nan")
    cmp_s = ""
    if cell != "dyna_h5" and base:
        shared = sorted(set(rows) & set(base))
        if shared:
            diff = np.array([rows[s][1] - base[s][1] for s in shared])
            m = diff.mean()
            if len(diff) > 1 and diff.std(ddof=1) > 0:
                t = m / (diff.std(ddof=1) / np.sqrt(len(diff)))
                cmp_s = f"{m:+.3f} (t={t:+.1f}, n={len(diff)})"
            else:
                cmp_s = f"{m:+.3f} (n={len(diff)})"
    partial = [s for s, v in rows.items() if v[3] < 100000]
    flag = f"  [partial: {sorted(partial)}]" if partial else ""
    print(f"{cell:22s} {len(rows):2d} {r0.mean():7.3f} {st.mean():7.3f} "
          f"{d.mean():+7.3f} {sd:6.3f} {cmp_s:>18s}{flag}")

"""Compare stage-1 runs on WINDOWED means, not last points (iron law #4).

Reads `score/score` (TARGET) and `score/source_score` (SOURCE) out of each run's
train.log and reports the mean over the LAST `--window` eval points, with the
spread, because in-training evals here are n=50 and swing hard point-to-point
(0.20 -> 0.76 -> 0.34 on adjacent points has been observed).

The source-minus-target GAP is the memorisation signature §45 is about: a high
source score with a flat target score means the source distribution was
memorised, and that gap -- not the target score alone -- is what augmentation and
data scale are supposed to move.

Usage:
  python scripts/cmp_stage1_runs.py exps/mimicgen/pixel_phaseA_off_seed1 \
                                    exps/mimicgen/pixel_phaseA_aug_seed1
"""
import argparse, os, re, sys

def series(run, key):
    log = os.path.join(run, "train.log")
    if not os.path.exists(log):
        return []
    pat = re.compile(rf"^\s*(\d+):\s*{re.escape(key)}\s*:\s*([-0-9.]+)")
    out = []
    with open(log, errors="ignore") as f:
        for ln in f:
            m = pat.match(ln.replace("\r", "\n").split("\n")[-1] if "\r" in ln else ln)
            if m:
                out.append((int(m.group(1)), float(m.group(2))))
    return out

def stat(vals):
    if not vals:
        return None
    n = len(vals)
    mu = sum(vals) / n
    sd = (sum((v - mu) ** 2 for v in vals) / max(n - 1, 1)) ** 0.5
    return mu, sd, n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--window", type=int, default=5, help="last N eval points")
    a = ap.parse_args()
    # ★ Align on the LAST COMMON step. Comparing a run at 50k against one at 100k
    # is not a comparison -- source_score is still CLIMBING early, so the
    # source-target gap of the younger run looks smaller purely because it has not
    # memorised yet. That reads as a win. (Iron law #7: suspect the reporting tool.)
    ends = {}
    for r in a.runs:
        t = series(r, "score/score")
        if t:
            ends[r] = t[-1][0]
    if len(ends) > 1 and len(set(ends.values())) > 1:
        common = min(ends.values())
        print(f"!! runs are at DIFFERENT steps {sorted(set(ends.values()))} -- "
              f"truncating ALL runs to the last common step {common}.")
        print("   (an early run's gap looks small only because source has not "
              "saturated yet; that is not a win.)\n")
        a._common = common
    else:
        a._common = None
    print(f"windowed over the LAST {a.window} eval points (iron law #4: read the "
          f"window, not the point)\n")
    print(f"{'run':<44} {'step':>8} {'target':>16} {'source':>16} {'gap':>7}")
    print("-" * 96)
    rows = []
    for r in a.runs:
        tgt, src = series(r, "score/score"), series(r, "score/source_score")
        if getattr(a, "_common", None):
            tgt = [(k, v) for k, v in tgt if k <= a._common]
            src = [(k, v) for k, v in src if k <= a._common]
        if not tgt:
            print(f"{os.path.basename(r):<44} {'-':>8}   no score/score in train.log yet")
            continue
        st = stat([v for _, v in tgt[-a.window:]])
        ss = stat([v for _, v in src[-a.window:]]) if src else None
        gap = (ss[0] - st[0]) if ss else float("nan")
        rows.append((r, st[0], ss[0] if ss else None, gap))
        s_str = f"{ss[0]:.3f} +-{ss[1]:.3f}" if ss else "n/a"
        print(f"{os.path.basename(r):<44} {tgt[-1][0]:>8} "
              f"{st[0]:.3f} +-{st[1]:.3f} (n={st[2]}) {s_str:>16} {gap:>7.3f}")
    if len(rows) == 2:
        d = rows[1][1] - rows[0][1]
        print(f"\ntarget delta ({os.path.basename(rows[1][0])} - "
              f"{os.path.basename(rows[0][0])}) = {d:+.3f}")
        print("  iron law #1: a delta under 0.05 is UNREADABLE here (single-draw "
              "sigma 0.092, replicate +-0.027, seed spread 0.035).")
        print("  iron law #2: anything within +-0.1 needs 300-500 episodes from an "
              "INDEPENDENT re-eval, not these n=50 in-training points.")
        if rows[0][2] is not None and rows[1][2] is not None:
            dgap = rows[1][3] - rows[0][3]
            dsrc = rows[1][2] - rows[0][2]
            dtgt = rows[1][1] - rows[0][1]
            print(f"source-target GAP: {rows[0][3]:.3f} -> {rows[1][3]:.3f} ({dgap:+.3f})"
                  f"   [source {dsrc:+.3f}, target {dtgt:+.3f}]")
            # ★ A NARROWER GAP IS NOT A WIN BY ITSELF. The gap can shrink because
            # the target ROSE (generalisation -- what we want) or because the
            # SOURCE FELL (just less memorisation, no transfer). Augmentation
            # lowers source almost by construction, so reporting "gap narrowed"
            # without decomposing it would score a null result as a success.
            if dgap < -0.02 and dtgt < 0.05:
                print("  ⚠️ the gap narrowed but TARGET DID NOT MOVE "
                      f"({dtgt:+.3f} < 0.05) -- this is source degradation, "
                      "NOT generalisation. Not a win.")
            elif dtgt >= 0.05:
                print(f"  target rose {dtgt:+.3f} (>= the 0.05 floor) -- "
                      "confirm with an INDEPENDENT 300-500 episode re-eval (iron law #14).")
            else:
                print("  no readable movement in either direction.")

if __name__ == "__main__":
    main()

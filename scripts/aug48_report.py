"""§48 report: the 3x5 r18@224+aug ladder, BOTH measures, PAIRED-within-seed deltas.

Run with no arguments once the 15 adapt runs and their n=500 evals have landed:

    python scripts/aug48_report.py
    python scripts/aug48_report.py --md        # markdown, paste into §48

Why this exists rather than eyeballing cmp_adapt_runs.py 15 times:

★ §47a -- the independent n=500 re-eval does NOT remove TopkSaver's selection
  bias. `model0.pt` is the MAX over in-training eval points, and the premium
  SCALES WITH THE ARM'S CURVE VARIANCE (measured: window SD 0.099 -> best was
  +0.245 above its window; 0.054 -> +0.095; 0.030 -> +0.065). So this prints each
  arm's WINDOW SD next to every delta, and flags a delta whose two arms have
  materially different SDs as "read the window". That is exactly the confound
  that flipped §47's ensemble delta between measures.
★ Deltas are computed PAIRED WITHIN SEED and then averaged, with the SD taken
  over the 5 paired deltas. Differencing cross-seed means instead throws away
  the blocking and reinflates §38's 0.092 draw variance.
★ The published comparators are printed DE-SELECTED: the published dynens_h5
  numbers are `score_best`, a max over 4 pessimism coefficients (~+0.03-0.05 of
  bias `plain_h3` cannot take), so the honest ensemble comparator is ~+0.20-0.22,
  not +0.249. And the published anchor +0.081 was measured on `dynens_h5_slow`.
"""
import argparse, glob, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.cmp_adapt_runs import curve  # single source of truth for the parse

RUNGS = [("A", "plain_h3 (1 head, H=3)"),
         ("B", "dynens_h5 (5 heads, H=5)"),
         ("C", "dynens_h5 + anchor lambda=1")]
SEEDS = [1, 2, 3, 4, 5]
OFFSET = {s: (s - 1) * 20 for s in SEEDS}
EVAL_DIR = "/scratch/yifanr/simdist/eval_staging"
LO, HI = 75000, 100000


def stats(v):
    n = len(v)
    if not n:
        return None
    mu = sum(v) / n
    sd = (sum((x - mu) ** 2 for x in v) / max(n - 1, 1)) ** 0.5
    return mu, sd, n


def window(run_dir):
    """(mean, sd, n, reached, standard?) over the 75-100k protocol window."""
    c = curve(run_dir)
    if not c:
        return None
    reached = c[-1][0]
    inwin = [(t, s, p) for t, s, p in c if LO <= t <= HI]
    if not inwin:                       # NON-STANDARD -- never silently equated
        lo = int(reached * 0.75)
        inwin = [(t, s, p) for t, s, p in c if lo <= t <= reached]
        st = stats([s for _, s, _ in inwin])
        return (*st, reached, False) if st else None
    st = stats([s for _, s, _ in inwin])
    best = max(s for _, s, _ in c)
    return (*st, reached, True, best)


def independent(tag):
    """success rate from the n=500 eval_simdist artifact, newest file wins."""
    hits = sorted(glob.glob(os.path.join(EVAL_DIR, f"{tag}_*.txt")),
                  key=os.path.getmtime)
    for f in reversed(hits):
        txt = open(f, errors="ignore").read()
        # eval.py prints "success rate: 63.80% (319/500)". Take the COUNTS, not
        # the percent -- a bare \\D*([0-9.]+) grabs "63.80" and silently reports a
        # success rate of 63.8 instead of 0.638.
        m = re.findall(r"success rate:\s*[0-9.]+%\s*\((\d+)/(\d+)\)", txt)
        env = re.findall(r"^\[EVAL ENV\].*$", txt, re.M)
        if m:
            k, n = int(m[-1][0]), int(m[-1][1])
            # iron law #15: two numbers compare only if [EVAL ENV] matches
            return k / n, (env[-1] if env else "MISSING [EVAL ENV]"), f
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="markdown for docs/simdist.md")
    a = ap.parse_args()

    W, I, ENVS = {}, {}, set()
    for rung, _ in RUNGS:
        for s in SEEDS:
            d = f"exps/mimicgen/aug48_{rung}_s{s}"
            W[(rung, s)] = window(d) if os.path.isdir(d) else None
            v, env, _f = independent(f"aug48_{rung}_s{s}")
            I[(rung, s)] = v
            if env:
                ENVS.add(env.strip())

    bar = "|" if a.md else " "
    print("§48 -- r18@224 + full augmentation, 5 ADAPT seeds, 1 stage-1 per arch")
    print(f"window = {LO}-{HI} transitions; independent = n=500 of model0.pt\n")
    if a.md:
        print("| rung | seed | offset | window mean ±SD (n) | reached | indep n=500 |")
        print("|---|---:|---:|---:|---:|---:|")
    for rung, label in RUNGS:
        for s in SEEDS:
            w = W[(rung, s)]
            ws = (f"{w[0]:.3f} ±{w[1]:.3f} (n={w[2]})"
                  + ("" if w[4] else " ⚠️NON-STANDARD")) if w else "— not run"
            reach = f"{w[3]}" if w else "—"
            iv = f"{I[(rung,s)]:.3f}" if I[(rung, s)] is not None else "—"
            print(f"{bar} {rung} {bar} {s} {bar} {OFFSET[s]} {bar} {ws} "
                  f"{bar} {reach} {bar} {iv} {bar}")
        # per-rung aggregate
        wm = stats([W[(rung, s)][0] for s in SEEDS if W[(rung, s)]])
        im = stats([I[(rung, s)] for s in SEEDS if I[(rung, s)] is not None])
        print(f"  -> rung {rung} ({label}): window "
              + (f"{wm[0]:.3f} ±{wm[1]:.3f} over {wm[2]} seeds" if wm else "n/a")
              + " | indep " + (f"{im[0]:.3f} ±{im[1]:.3f} (n={im[2]})" if im else "n/a"))
        print()

    print("--- PAIRED-WITHIN-SEED deltas (iron law: never difference cross-seed means) ---")
    for name, hi, lo, published in [
            ("ensemble (B-A)", "B", "A", "+0.249 published, but ~+0.20-0.22 de-selected "
             "(published dynens_h5 is score_best, a max over 4 pessimism coefs)"),
            ("anchor   (C-B)", "C", "B", "+0.081 published -- and measured on "
             "dynens_h5_SLOW (0.898->0.979), not on dynens_h5")]:
        for measure, src in (("window", W), ("indep", I)):
            pairs = []
            for s in SEEDS:
                x, y = src[(hi, s)], src[(lo, s)]
                xv = x[0] if (measure == "window" and x) else x
                yv = y[0] if (measure == "window" and y) else y
                if xv is not None and yv is not None:
                    pairs.append((s, xv - yv))
            st = stats([d for _, d in pairs])
            if not st:
                print(f"{name} [{measure}]: no complete seed pairs yet")
                continue
            se = st[1] / (st[2] ** 0.5)
            per = "  ".join(f"s{s}:{d:+.3f}" for s, d in pairs)
            print(f"{name} [{measure:>6}]: {st[0]:+.3f} ±{st[1]:.3f} over n={st[2]} "
                  f"paired seeds (SE {se:.3f}, {abs(st[0])/se if se else 0:.1f} sigma)")
            print(f"                      {per}")
        # ★ §47a exchangeability check on the window SDs
        sd_hi = stats([W[(hi, s)][1] for s in SEEDS if W[(hi, s)]])
        sd_lo = stats([W[(lo, s)][1] for s in SEEDS if W[(lo, s)]])
        if sd_hi and sd_lo:
            print(f"   window SDs: {hi}={sd_hi[0]:.3f}  {lo}={sd_lo[0]:.3f}"
                  + ("   ⚠️ MATERIALLY DIFFERENT -> the independent numbers are NOT "
                     "exchangeable (§47a); READ THE WINDOW"
                     if abs(sd_hi[0] - sd_lo[0]) > 0.02 else
                     "   (comparable -> the selection premium largely cancels)"))
        print(f"   published comparator: {published}\n")

    print("--- provenance ---")
    print("iron law #15: comparable only if the [EVAL ENV] line is identical.")
    for e in sorted(ENVS):
        print("  " + e)
    if len(ENVS) > 1:
        print("  ⚠️ MORE THAN ONE [EVAL ENV] -> these numbers are NOT comparable.")
    print("\n§46j: 150k vs 800k stage-1 steps is worth ~+0.076. It cancels WITHIN "
          "this ladder,\nso absolute levels are not comparable to the published "
          "state ladder -- only deltas are.")


if __name__ == "__main__":
    main()

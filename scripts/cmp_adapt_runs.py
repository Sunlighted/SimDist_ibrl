"""Read stage-2/3 adapt curves as WINDOWED means over a transition range.

The published protocol (docs/on-policy-adaptation.md) is: 100k target
transitions, an eval every 1k of 100 games, and the reported number is the mean
over the **75-100k steady-state window** -- roughly 25 points x 100 games, which
is how it beats the 20-demo draw noise (sigma 0.092). A single point is n=100 and
swings hard: this project has logged 0.35 -> 0.54 -> 0.35 on ADJACENT points.

Prints the protocol window when the run reaches it, and otherwise the last
available window CLEARLY LABELLED as non-standard -- a 60-73k mean is not
comparable to a published 75-100k mean and must never be presented as if it were.
"""
import argparse, os, re, sys

PAT = re.compile(r"\[adapt @ (\d+) transitions\]\s+target_score=([0-9.]+)"
                 r"(?:\s+policy_only=([0-9.]+))?")


def curve(run):
    log = os.path.join(run, "train.log")
    if not os.path.exists(log):
        return []
    out = []
    with open(log, errors="ignore") as f:
        for ln in f.read().replace("\r", "\n").split("\n"):
            m = PAT.search(ln)
            if m:
                out.append((int(m.group(1)), float(m.group(2)),
                            float(m.group(3)) if m.group(3) else float("nan")))
    return out


def stat(v):
    n = len(v)
    if n == 0:
        return None
    mu = sum(v) / n
    sd = (sum((x - mu) ** 2 for x in v) / max(n - 1, 1)) ** 0.5
    return mu, sd, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--lo", type=int, default=75000)
    ap.add_argument("--hi", type=int, default=100000)
    a = ap.parse_args()
    print(f"protocol window = {a.lo}-{a.hi} transitions "
          f"(docs/on-policy-adaptation.md); a single point is n=100 and swings "
          f"~0.2 (iron law #4)\n")
    # ★ Align MULTIPLE runs on the last COMMON transition count. An adapt curve
    # climbs (pixel went 0.11 -> 0.19 -> 0.22 -> 0.33 over 30k), so putting a
    # run at 73k beside one at 40k and calling it a comparison rewards whichever
    # simply ran longer. (Same trap this project already hit in the stage-1
    # comparator -- iron law #7: suspect the reporting tool.)
    curves = {r: curve(r) for r in a.runs}
    reached = {r: c[-1][0] for r, c in curves.items() if c}
    common = min(reached.values()) if len(reached) > 1 else None
    if common is not None and len(set(reached.values())) > 1:
        print(f"!! runs reached different budgets {sorted(set(reached.values()))} "
              f"-- ALSO reporting a matched window ending at {common}.\n")
    print(f"{'run':<40} {'reached':>8} {'window':>14} {'target':>17} {'policy_only':>15}")
    print("-" * 100)
    for r in a.runs:
        c = curves[r]
        if not c:
            print(f"{os.path.basename(r):<40} {'-':>8}   no adapt points yet")
            continue
        end = c[-1][0]
        inwin = [(t, s, p) for t, s, p in c if a.lo <= t <= a.hi]
        if inwin:
            lo, hi, tag = a.lo, a.hi, ""
        else:                       # fall back to the last 25% of what exists
            lo, hi = int(end * 0.75), end
            inwin = [(t, s, p) for t, s, p in c if lo <= t <= hi]
            tag = "  ⚠️ NON-STANDARD"
        ts = stat([s for _, s, _ in inwin])
        ps = stat([p for _, _, p in inwin if p == p])
        p_str = f"{ps[0]:.3f} +-{ps[1]:.3f}" if ps else "n/a"
        print(f"{os.path.basename(r):<40} {end:>8} {f'{lo}-{hi}':>14} "
              f"{ts[0]:.3f} +-{ts[1]:.3f} (n={ts[2]}) {p_str:>15}{tag}")
        if tag:
            print(f"{'':<40} run has NOT reached {a.lo}; this mean is NOT "
                  f"comparable to a published {a.lo}-{a.hi} number.")
    if common is not None and len(set(reached.values())) > 1:
        lo = int(common * 0.75)
        print(f"\n--- MATCHED window {lo}-{common} (the only fair side-by-side) ---")
        for r in a.runs:
            inw = [(t, sc, pp) for t, sc, pp in curves[r] if lo <= t <= common]
            if not inw:
                continue
            ts = stat([x for _, x, _ in inw])
            ps = stat([x for _, _, x in inw if x == x])
            mult = (ts[0] / ps[0]) if ps and ps[0] > 0 else float("nan")
            print(f"{os.path.basename(r):<40} {'':>8} {f'{lo}-{common}':>14} "
                  f"{ts[0]:.3f} +-{ts[1]:.3f} (n={ts[2]}) "
                  f"{ps[0]:.3f} +-{ps[1]:.3f}   planner x{mult:.2f}")
    print("\nheadline numbers must come from an INDEPENDENT re-eval of the saved "
          "ckpt (iron law #14), not from these in-training points.")


if __name__ == "__main__":
    main()

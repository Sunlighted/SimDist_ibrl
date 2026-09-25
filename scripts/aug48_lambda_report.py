"""§48p: the bc_anchor_coef (= KL coefficient) curve on the SHIFT-ONLY stage-1.

lambda=0 -> aug48k_B_s*, lambda=1 -> aug48k_C_s*, others -> aug48sw_lam<tag>_s*.
Every delta is PAIRED WITHIN DRAW against lambda=0, and only COMPLETE windows
(reached >= 100k) are used -- aug48_report.py's fix for the partial-window artifact.
Reuses aug48_report's parsers, which reproduce all six §47 numbers exactly.
"""
import os, sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "r", os.path.join(os.path.dirname(os.path.abspath(__file__)), "aug48_report.py"))
R = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(R)

LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0]
SEEDS = [1, 2, 3]
def dirtag(lam, s):
    if lam == 0.0: return f"aug48k_B_s{s}"
    if lam == 1.0: return f"aug48k_C_s{s}"
    return f"aug48sw_lam{str(lam).replace('.', 'p').rstrip('0').rstrip('p') if lam != int(lam) else int(lam)}_s{s}"

W = {}; I = {}
for lam in LAMBDAS:
    for s in SEEDS:
        t = dirtag(lam, s); w = R.window(f"exps/mimicgen/{t}")
        W[(lam, s)] = w[0] if (w and w[4]) else None
        I[(lam, s)] = R.independent(t)[0]

print("§48p -- anchor lambda (KL coef) curve, SHIFT-ONLY stage-1, draws seeds 1-3")
print(f"{'lambda':>7} | " + " | ".join(f"s{s} win / ind" for s in SEEDS) + " | paired delta vs lambda=0 (window)")
for lam in LAMBDAS:
    cells = []
    for s in SEEDS:
        w, i = W[(lam, s)], I[(lam, s)]
        cells.append(f"{w:.3f}/{i:.3f}" if (w is not None and i is not None)
                     else (f"{w:.3f}/  -  " if w is not None else "  running  "))
    d = [W[(lam, s)] - W[(0.0, s)] for s in SEEDS
         if W[(lam, s)] is not None and W[(0.0, s)] is not None]
    ds = (f"{st.mean(d):+.3f}" + (f" ±{st.stdev(d):.3f}" if len(d) > 1 else "")
          + f"  n={len(d)}  [" + " ".join(f"{x:+.3f}" for x in d) + "]") if d and lam else ("(reference)" if lam == 0 else "-")
    print(f"{lam:>7} | " + " | ".join(cells) + f" | {ds}")
print("\nPre-registered bar (§48p): a lambda 'works' only if its paired delta is"
      " >= +0.15 on ALL 3 draws. Anything that passes must be rerun at 5 draws.")

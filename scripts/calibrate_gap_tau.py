"""Calibrate gated_agent's gap->confidence mapping from a counterfactual
dynamics-gap dataset (classifier_test/pairs_demo.npz, built by
classifier_test/step1_build_pairs.py).

Picks:
  * soft   gap_tau   = contact-phase median gap / ln2   (c=0.5 at that gap)
  * binary threshold = geometric mean(free p90, contact median gap)

"contact phase" = needle_ring_dist(s) < 0.15 (the only phase where the
Threading dynamics gap manifests). Prints the resulting free/contact
confidence statistics so you can sanity-check the separation.

Run:  python scripts/calibrate_gap_tau.py [--pairs classifier_test/pairs_demo.npz]
"""

import argparse
import os

import numpy as np


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default=os.path.join(repo, "classifier_test/pairs_demo.npz"))
    p.add_argument("--contact_dist", type=float, default=0.15)
    args = p.parse_args()

    d = np.load(args.pairs)
    gap, dist = d["gap"], d["dist"]
    contact = dist < args.contact_dist
    free = ~contact
    g_c_med = float(np.median(gap[contact]))
    g_f_med = float(np.median(gap[free]))
    f_p90 = float(np.percentile(gap[free], 90))

    qs = [10, 50, 90, 95, 99]
    def show(name, g):
        ps = np.percentile(g, qs)
        print(f"  {name:16s} n={len(g):5d} mean={g.mean():.4f} "
              + " ".join(f"p{q}={v:.4f}" for q, v in zip(qs, ps)))
    print(f"pairs: {args.pairs}  (N={len(gap)})")
    show("ALL", gap)
    show("free(>=%.2f)" % args.contact_dist, gap[free])
    show("contact(<%.2f)" % args.contact_dist, gap[contact])

    tau = g_c_med / np.log(2)
    thr = float(np.sqrt(f_p90 * g_c_med))
    print(f"\nfree median={g_f_med:.4f}  contact median={g_c_med:.4f}  free p90={f_p90:.4f}")
    print(f"\n>>> RECOMMENDED:")
    print(f"    gap_c_mode: soft    gap_tau:       {tau:.4f}")
    print(f"    gap_c_mode: binary  gap_threshold: {thr:.4f}")

    print("\nsoft (c=exp(-gap/tau)) region stats:")
    for label, g in [("free", gap[free]), ("contact", gap[contact])]:
        c = np.exp(-g / tau)
        print(f"  {label:8s}: mean c={c.mean():.3f} c@p50={np.exp(-np.median(g)/tau):.3f} "
              f"c@p90={np.exp(-np.percentile(g,90)/tau):.3f} frac(c<0.5)={np.mean(c<0.5):.2f}")
    print("binary (c=1[gap<thr]) region stats:")
    print(f"  free trusted (c=1):    {np.mean(gap[free] < thr):.2f}")
    print(f"  contact gated (c=0):   {np.mean(gap[contact] >= thr):.2f}")


if __name__ == "__main__":
    main()

"""Re-plot + sharpen the coverage diagnostic from the saved npz (no env needed).

Adds the decision-relevant cut: among CONTACT-phase transitions (where the gap
lives and where the gating actually has to choose), are the HIGH-gap ones still
covered by the target-demo support? Clips the heavy OOD tail for readability.
"""
import os, sys
import numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = np.load("coverage_diag/coverage_data.npz", allow_pickle=True)
gaps, dist, nring, demo_nn = D["gaps"], D["demo_dist"], D["needle_ring"], D["demo_nn"]
N = len(gaps)
CONTACT = 0.15


def block(name, g, d):
    if len(g) < 20:
        print(f"{name}: too few ({len(g)})"); return
    hi = g >= np.quantile(g, 0.75); lo = ~hi
    rho = spearmanr(g, d)[0]
    ref = np.median(d)
    print(f"{name}  (n={len(g)})")
    print(f"   Spearman(gap, demo_dist) = {rho:+.3f}")
    print(f"   median demo_dist: high-gap={np.median(d[hi]):.2f}  low-gap={np.median(d[lo]):.2f}"
          f"  ratio={np.median(d[hi])/max(np.median(d[lo]),1e-9):.2f}x")
    print(f"   coverage(<=median {ref:.2f}): high-gap={np.mean(d[hi]<=ref):.2f}  low-gap={np.mean(d[lo]<=ref):.2f}")


print("=" * 64)
print(f"demo intra-NN dist (z-space): p50={np.median(demo_nn):.3f} p90={np.quantile(demo_nn,0.9):.3f}")
print(f"visited demo_dist: p50={np.median(dist):.2f} p90={np.quantile(dist,0.9):.2f} max={dist.max():.1f}")
cm = nring < CONTACT
print(f"contact-phase fraction (needle-ring<{CONTACT}): {cm.mean():.2f}  ({cm.sum()}/{N})")
print("-" * 64)
block("ALL transitions", gaps, dist)
print("-" * 64)
block("CONTACT-phase only", gaps[cm], dist[cm])
print("=" * 64)

# ---- figure (clip x at p98 for readability; tail noted in title)
xcap = np.quantile(dist, 0.98)
fig, ax = plt.subplots(2, 2, figsize=(13, 10))

# A: gap vs demo-dist (contact-phase points only, the decision-relevant set)
g, d, nr = gaps[cm], dist[cm], nring[cm]
sc = ax[0, 0].scatter(np.clip(d, 0, xcap), g, c=nr, s=8, alpha=0.45, cmap="viridis_r")
ax[0, 0].axvline(np.median(dist), color="k", ls="--", lw=1, label=f"median dist={np.median(dist):.1f}")
ax[0, 0].set_xlabel(f"dist to nearest demo (z-space, clipped @p98={xcap:.0f})")
ax[0, 0].set_ylabel("GT dynamics gap")
ax[0, 0].set_title(f"A. CONTACT-phase: gap vs demo-dist  (Spearman={spearmanr(g,d)[0]:+.3f})")
ax[0, 0].legend(fontsize=8); plt.colorbar(sc, ax=ax[0, 0], label="needle-ring dist")

# B: demo-dist hist, high vs low gap, contact-phase only
hi = g >= np.quantile(g, 0.75)
bins = np.linspace(0, xcap, 45)
ax[0, 1].hist(d[~hi], bins=bins, density=True, alpha=0.5, label="low-gap (contact)")
ax[0, 1].hist(d[hi], bins=bins, density=True, alpha=0.5, label="high-gap (contact)")
ax[0, 1].axvline(np.quantile(demo_nn, 0.9), color="g", ls=":", lw=1.5,
                 label=f"demo intra-NN p90={np.quantile(demo_nn,0.9):.2f}")
ax[0, 1].set_xlabel("dist to nearest demo (z-space)"); ax[0, 1].set_ylabel("density")
ax[0, 1].set_title("B. CONTACT-phase demo-dist: high-gap vs low-gap"); ax[0, 1].legend(fontsize=8)

# C: coverage vs gap percentile — ALL vs CONTACT overlaid
for tag, gg, dd, col in [("all", gaps, dist, "C0"), ("contact", g, d, "C3")]:
    ref = np.median(dist); n_bins = 10
    qs = np.quantile(gg, np.linspace(0, 1, n_bins + 1)); cen, cov = [], []
    for i in range(n_bins):
        m = (gg >= qs[i]) & (gg <= qs[i + 1])
        if m.sum() == 0: continue
        cen.append((i + 0.5) / n_bins * 100); cov.append(np.mean(dd[m] <= ref))
    ax[1, 0].plot(cen, cov, "o-", color=col, label=tag)
ax[1, 0].set_xlabel("gap percentile bin (%)"); ax[1, 0].set_ylabel(f"coverage (dist<=median {np.median(dist):.1f})")
ax[1, 0].set_ylim(0, 1.02); ax[1, 0].grid(alpha=0.3); ax[1, 0].legend(fontsize=8)
ax[1, 0].set_title("C. coverage vs gap percentile\n(falling = high-gap under-covered)")

# D: gap vs needle-ring distance (all), color=demo-dist
ax[1, 1].scatter(nring, gaps, c=np.clip(dist, 0, xcap), s=6, alpha=0.4, cmap="plasma")
ax[1, 1].axvline(CONTACT, color="k", ls="--", lw=1, label=f"contact thresh {CONTACT}")
ax[1, 1].set_xlabel("needle-ring distance (m)"); ax[1, 1].set_ylabel("GT dynamics gap")
ax[1, 1].set_title("D. gap concentrates in contact phase (color=demo-dist)"); ax[1, 1].legend(fontsize=8)

fig.suptitle(f"Coverage of high-gap transitions by target-demo support  (n={N}, source-RL rollout)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("coverage_diag/coverage_v2.png", dpi=120)
print("saved coverage_diag/coverage_v2.png")

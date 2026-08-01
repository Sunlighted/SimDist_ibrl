"""Final coverage verdict + figure, using a ROBUST distance (per-dim z clipped to
+-5 so a few near-constant demo dims — e.g. the barely-moving tripod pose, demo
std ~0.002 — don't dominate the L2). The relative high-gap-vs-low-gap conclusion
is invariant to this; clipping only de-pathologizes the absolute scale.
"""
import sys
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "classifier_test")
import cf_utils, data_utils

CLIP = 5.0
CONTACT = 0.15


def canon(s):
    out = s.copy()
    for a, _ in data_utils.QUAT_BLOCKS:
        flip = out[:, a] < 0
        for qa, qb in data_utils.QUAT_BLOCKS:
            if qa == a:
                out[flip, qa:qb] *= -1
    return out


eps = cf_utils.load_demo_episodes("release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5")
demo = canon(np.concatenate([e["state"] for e in eps], 0))
mu, sd = demo.mean(0), demo.std(0)
sd = np.where(sd < 1e-6, 1.0, sd)
demoz = np.clip((demo - mu) / sd, -CLIP, CLIP)
tree = cKDTree(demoz)
intra, _ = tree.query(demoz, k=2); demo_nn = intra[:, 1]

D = np.load("coverage_diag/coverage_data.npz", allow_pickle=True)
gaps, nring = D["gaps"], D["needle_ring"]
visz = np.clip((canon(D["states"].copy()) - mu) / sd, -CLIP, CLIP)
dist, _ = tree.query(visz, k=1)
N = len(gaps)
cm = nring < CONTACT


def block(name, g, d):
    hi = g >= np.quantile(g, 0.75); lo = ~hi
    rho = spearmanr(g, d)[0]; ref = np.median(dist)
    print(f"{name}  (n={len(g)})")
    print(f"   Spearman(gap, demo_dist) = {rho:+.3f}   [~0/neg = high-gap NOT farther = GOOD]")
    print(f"   median demo_dist: high={np.median(d[hi]):.2f} low={np.median(d[lo]):.2f} "
          f"ratio={np.median(d[hi])/max(np.median(d[lo]),1e-9):.2f}x")
    print(f"   coverage(<=median {ref:.2f}): high={np.mean(d[hi]<=ref):.2f} low={np.mean(d[lo]<=ref):.2f}")
    return rho


print("=" * 66)
print(f"robust z-clip=+-{CLIP}")
print(f"demo intra-NN: p50={np.median(demo_nn):.2f} p90={np.quantile(demo_nn,0.9):.2f}")
print(f"visited demo_dist: p50={np.median(dist):.2f} p90={np.quantile(dist,0.9):.2f}")
print(f"contact-phase fraction: {cm.mean():.2f} ({cm.sum()}/{N})")
print("-" * 66)
block("ALL", gaps, dist)
print("-" * 66)
rho_c = block("CONTACT-phase only (decision-relevant)", gaps[cm], dist[cm])
print("=" * 66)

xcap = np.quantile(dist, 0.99)
fig, ax = plt.subplots(2, 2, figsize=(13, 10))
g, d, nr = gaps[cm], dist[cm], nring[cm]
hi = g >= np.quantile(g, 0.75)

sc = ax[0, 0].scatter(d, g, c=nr, s=9, alpha=0.5, cmap="viridis_r")
ax[0, 0].axvline(np.median(dist), color="k", ls="--", lw=1, label=f"median={np.median(dist):.1f}")
ax[0, 0].set_xlabel("robust dist to nearest demo"); ax[0, 0].set_ylabel("GT dynamics gap")
ax[0, 0].set_title(f"A. CONTACT: gap vs demo-dist (Spearman={rho_c:+.3f})")
ax[0, 0].legend(fontsize=8); plt.colorbar(sc, ax=ax[0, 0], label="needle-ring dist")

bins = np.linspace(0, xcap, 45)
ax[0, 1].hist(d[~hi], bins=bins, density=True, alpha=0.55, label="low-gap (contact)")
ax[0, 1].hist(d[hi], bins=bins, density=True, alpha=0.55, label="high-gap (contact)")
ax[0, 1].axvline(np.quantile(demo_nn, 0.9), color="g", ls=":", lw=1.5,
                 label=f"demo intra-NN p90={np.quantile(demo_nn,0.9):.2f}")
ax[0, 1].set_xlabel("robust dist to nearest demo"); ax[0, 1].set_ylabel("density")
ax[0, 1].set_title("B. CONTACT demo-dist: high-gap vs low-gap"); ax[0, 1].legend(fontsize=8)

for tag, gg, dd, col in [("all", gaps, dist, "C0"), ("contact", g, d, "C3")]:
    ref = np.median(dist); qs = np.quantile(gg, np.linspace(0, 1, 11)); cen, cov = [], []
    for i in range(10):
        m = (gg >= qs[i]) & (gg <= qs[i + 1])
        if m.sum(): cen.append((i + 0.5) * 10); cov.append(np.mean(dd[m] <= ref))
    ax[1, 0].plot(cen, cov, "o-", color=col, label=tag)
ax[1, 0].set_xlabel("gap percentile bin (%)"); ax[1, 0].set_ylabel(f"coverage (<=median {np.median(dist):.1f})")
ax[1, 0].set_ylim(0, 1.02); ax[1, 0].grid(alpha=0.3); ax[1, 0].legend(fontsize=8)
ax[1, 0].set_title("C. coverage vs gap percentile\n(falling = high-gap under-covered)")

ax[1, 1].scatter(nring, gaps, c=np.clip(dist, 0, xcap), s=6, alpha=0.4, cmap="plasma")
ax[1, 1].axvline(CONTACT, color="k", ls="--", lw=1, label=f"contact {CONTACT}")
ax[1, 1].set_xlabel("needle-ring distance (m)"); ax[1, 1].set_ylabel("GT dynamics gap")
ax[1, 1].set_title("D. gap concentrates in contact (color=demo-dist)"); ax[1, 1].legend(fontsize=8)

fig.suptitle(f"Coverage of high-gap transitions by target-demo support (n={N}, robust z)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("coverage_diag/coverage_final.png", dpi=120)
print("saved coverage_diag/coverage_final.png")

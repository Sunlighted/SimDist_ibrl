"""Step 4: validate that the discriminator's per-transition gap score tracks
the ground-truth gap ||s'_tgt - s'_src|| on HELD-OUT demo pairs.

Discriminator gap scores (no ground truth needed at inference):
  - two-sided  : r(s,a,s'_tgt) - r(s,a,s'_src)   [needs both s'; sim2sim only]
  - one-sided  : |r(s,a,s'_src)|                  [only the source step; the
                 sim2real-realistic signal — target s' is unavailable]
where r = log p_tgt(s'|s,a)/p_src(s'|s,a). High score => big dynamics gap.

We report Spearman correlation with the ground-truth gap. High correlation =>
the discriminator localizes the gap (does what we want), not just classifies.
"""

import argparse
import os

import numpy as np
import torch
from scipy.stats import spearmanr

from discriminator import DynamicsDiscriminator
from data_utils import episode_split, state_gap


def load_disc(path, device):
    ck = torch.load(path, map_location=device)
    d = DynamicsDiscriminator(ck["state_dim"], ck["action_dim"], ck["hidden"], ck["dropout"]).to(device)
    d.load_state_dict(ck["state_dict"]); d.eval()
    return d


def to_t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default=f"{os.path.dirname(__file__)}/pairs_demo.npz")
    p.add_argument("--disc", default=f"{os.path.dirname(__file__)}/discriminator_cf.pt")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=f"{os.path.dirname(__file__)}/step4_validation.png")
    args = p.parse_args()

    z = np.load(args.pairs)
    s, a, s_src, s_tgt, ep, dist = z["s"], z["a"], z["s_src"], z["s_tgt"], z["ep_id"], z["dist"]
    gap = state_gap(s_src, s_tgt)

    # same split as training so "val" is genuinely held out
    _, va = episode_split(ep, args.val_ratio, args.seed)
    d = load_disc(args.disc, args.device)

    S, A, SRC, TGT = (to_t(x[va], args.device) for x in (s, a, s_src, s_tgt))
    g = gap[va]; dst = dist[va]
    with torch.no_grad():
        r_tgt = d.log_dynamics_ratio(S, A, TGT).cpu().numpy()
        r_src = d.log_dynamics_ratio(S, A, SRC).cpu().numpy()
    two_sided = r_tgt - r_src           # sim2sim
    one_sided = np.abs(r_src)           # sim2real-realistic

    rho2, _ = spearmanr(two_sided, g)
    rho1, _ = spearmanr(one_sided, g)
    print(f"held-out pairs: {len(g)}")
    print(f"Spearman(disc gap score, ground-truth ||Δs'||):")
    print(f"  two-sided (r_tgt - r_src): rho = {rho2:.3f}")
    print(f"  one-sided (|r_src|)      : rho = {rho1:.3f}")
    # sanity: are the two domains' next states correctly ordered on average?
    print(f"mean r_tgt = {r_tgt.mean():.3f} (want >0), mean r_src = {r_src.mean():.3f} (want <0)")
    # does the score separate high-gap vs low-gap transitions?
    hi = g > np.quantile(g, 0.8)
    print(f"two-sided score: high-gap mean {two_sided[hi].mean():.3f} vs "
          f"low-gap mean {two_sided[~hi].mean():.3f}")

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, score, name, rho in [(axes[0], two_sided, "two-sided (r_tgt - r_src)", rho2),
                                  (axes[1], one_sided, "one-sided |r_src|", rho1)]:
        sc = ax.scatter(g, score, c=dst, s=8, cmap="viridis", alpha=0.6)
        ax.set_xlabel("ground-truth gap ||Δs'||"); ax.set_ylabel("discriminator gap score")
        ax.set_title(f"{name}\nSpearman rho = {rho:.3f}")
        plt.colorbar(sc, ax=ax, label="needle-ring dist")
    fig.tight_layout(); fig.savefig(args.out, dpi=120)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

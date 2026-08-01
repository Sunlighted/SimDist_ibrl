"""Step 11: per-trajectory visualization of the ensemble gap detector on
success=True vs success=False rollouts.

Two behavior policies:
  (1) source IBRL policy in source env  -> mostly success=True (in-dist). The
      detector SHOULD track the real per-step gap (rises at contact/insertion).
  (2) a weak target-only MLP BC in source -> success=False (OOD failures, GT
      gap ~ 0 everywhere). The single discriminator hallucinates a high score;
      the ensemble mean - kappa*std should stay LOW/flat (suppressed).

For each trajectory we overlay (left y = detector scores, shared scale):
  - single member |log ratio|
  - ensemble mean
  - ensemble mean - kappa*std
and (right y) the ground-truth gap and needle-ring distance, so the eye can
check WHERE the scores light up vs the true gap and the contact phase.
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cf_utils import REPO, TARGET_SHIFT
from step10_eval_ensemble import load_ensemble, member_scores, rollout


def plot_set(trajs, members, kappa, dev, title, out, max_traj=8):
    trajs = trajs[:max_traj]
    n = len(trajs)
    ncol = min(4, n); nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    axes = axes.ravel()
    for i, tr in enumerate(trajs):
        ax = axes[i]
        sc = member_scores(members, tr["s"], tr["a"], tr["ssrc"], dev)  # (M, T)
        mean, std = sc.mean(0), sc.std(0)
        single = sc[0]
        combo = mean - kappa * std
        t = np.arange(len(mean))
        ax.plot(t, single, c="C0", lw=0.9, alpha=0.7, label="single |log r|")
        ax.plot(t, mean, c="C1", lw=1.2, label="ens mean")
        ax.plot(t, combo, c="C2", lw=1.6, label=f"mean-{kappa}std")
        ax.axhline(0, c="gray", lw=0.5, ls="--")
        ax.set_ylim(-0.4, 1.2)
        ax2 = ax.twinx()
        ax2.plot(t, tr["gap"], c="C3", lw=1.4, alpha=0.8, label="GT gap")
        ax2.plot(t, tr["dist"], c="C4", lw=0.9, ls=":", alpha=0.6, label="needle-ring dist")
        ax2.set_ylim(0, max(0.16, tr["gap"].max() * 1.1))
        ax.set_title(f"traj{i}  success={tr['success']}", fontsize=9)
        if i == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper right")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=120)
    print(f"saved -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weight", default=f"{REPO}/exps/mimicgen/threading_ibrl_source_seed1/model0.pt")
    p.add_argument("--bad_weight",
                   default=f"{REPO}/exps/mimicgen/threading_target_only_mlp_bc_seed1/model1.pt")
    p.add_argument("--ens_dir", default=f"{os.path.dirname(__file__)}/disc_ens")
    p.add_argument("--kappa", type=float, default=2.0)
    p.add_argument("--num_traj", type=int, default=8)
    p.add_argument("--ood_horizon", type=int, default=200)
    p.add_argument("--seed", type=int, default=300)
    args = p.parse_args()
    dev = "cuda"

    import train_rl
    import train_bc
    from env.robosuite_wrapper import PixelRobosuite
    from env.domain_shift import DomainShiftConfig

    agent, _, env_params = train_rl.load_model(args.weight, dev)
    src_params = dict(env_params); src_params["domain_shift"] = DomainShiftConfig(mode="none")
    src_env = PixelRobosuite(**src_params)
    tgt_params = dict(env_params); tgt_params["domain_shift"] = DomainShiftConfig(**TARGET_SHIFT)
    tgt_env = PixelRobosuite(**tgt_params); tgt_env.reset()

    members = load_ensemble(args.ens_dir, dev)

    # (1) source RL — mostly success=True (in-dist)
    rl_trajs = rollout(agent, src_env, tgt_env, args.num_traj, args.seed, dev)
    print("source RL successes:", [t["success"] for t in rl_trajs])
    plot_set(rl_trajs, members, args.kappa, dev,
             f"(1) source IBRL policy in source  (kappa={args.kappa})",
             f"{os.path.dirname(__file__)}/step11_source_rl.png")

    # (2) weak target-only MLP BC — success=False (OOD failures)
    bad_agent, _, _ = train_bc.load_model(args.bad_weight, dev)
    bad_trajs = rollout(bad_agent, src_env, tgt_env, args.num_traj, args.seed + 900, dev,
                        horizon=args.ood_horizon)
    print("bad BC successes:", [t["success"] for t in bad_trajs])
    plot_set(bad_trajs, members, args.kappa, dev,
             f"(2) weak target-only MLP BC in source — OOD failures  (kappa={args.kappa})",
             f"{os.path.dirname(__file__)}/step11_bad_bc.png")


if __name__ == "__main__":
    main()

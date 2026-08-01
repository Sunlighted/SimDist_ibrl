"""Step 5: generalization test on NEW behavior-policy rollouts.

Roll out a behavior policy (flow BC) in the SOURCE env with a wider init
(joint sigma=0.1) to get trajectories unlike the 20 training demos. For each
transition:
  - s'_src = the actual source rollout step (observed; sim2real-realistic)
  - s'_tgt = counterfactual step of the TARGET sim from the same sim_state
             (ground-truth, sim2sim-only — used ONLY to validate, not to train)
  - ground-truth gap = ||s'_tgt - s'_src||
  - discriminator one-sided gap score |r_src| at the observed transition

Then check, per trajectory, whether the discriminator's gap score lights up at
the same places as the ground-truth gap (and tracks the contact phase) — i.e.
whether the gap detector trained on 16 demo episodes generalizes.
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from cf_utils import (REPO, TARGET_SHIFT, counterfactual_step, get_sim_state,
                      extract_state)
from data_utils import state_gap, needle_ring_dist
from discriminator import DynamicsDiscriminator
from common_utils import ibrl_utils as utils


def load_disc(path, device):
    ck = torch.load(path, map_location=device)
    d = DynamicsDiscriminator(ck["state_dim"], ck["action_dim"], ck["hidden"], ck["dropout"]).to(device)
    d.load_state_dict(ck["state_dict"]); d.eval()
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--behavior_weight", default=f"{REPO}/exps/mimicgen/threading_flow_bc_seed1/model0.pt")
    p.add_argument("--disc", default=f"{os.path.dirname(__file__)}/discriminator_cf.pt")
    p.add_argument("--num_traj", type=int, default=6)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--out", default=f"{os.path.dirname(__file__)}/step5_generalization.png")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = args.device

    import train_bc
    from env.robosuite_wrapper import PixelRobosuite
    from env.domain_shift import DomainShiftConfig

    agent, _, env_params = train_bc.load_model(args.behavior_weight, device)

    # source rollout env (wider init), and a target env for counterfactual
    src_params = dict(env_params)
    src_params["domain_shift"] = DomainShiftConfig(mode="none")
    src_params["hand_init_perturb"] = None
    src_params["hand_init_perturb_joint_sigma"] = args.sigma
    src_env = PixelRobosuite(**src_params)

    tgt_params = dict(env_params)
    tgt_params["domain_shift"] = DomainShiftConfig(**TARGET_SHIFT)
    tgt_env = PixelRobosuite(**tgt_params)
    tgt_env.reset()  # build target sim once

    d = load_disc(args.disc, device)

    trajs = []
    with torch.no_grad(), utils.eval_mode(agent):
        for ti in range(args.num_traj):
            np.random.seed(args.seed + ti)
            obs, _ = src_env.reset()
            S, A, SIM, SSRC, R = [], [], [], [], []
            terminal = False
            while not terminal:
                sim_t = get_sim_state(src_env)
                s_t = obs["state"].cpu().numpy()
                action = agent.act(obs, eval_mode=True)
                obs, reward, terminal, _, _ = src_env.step(action)
                S.append(s_t); A.append(action.cpu().numpy())
                SIM.append(sim_t); SSRC.append(obs["state"].cpu().numpy())
                R.append(float(reward))
            S, A, SIM, SSRC = map(lambda x: np.asarray(x, np.float32), (S, A, SIM, SSRC))
            # counterfactual target step from each sim_state -> ground-truth gap
            STGT = np.stack([counterfactual_step(tgt_env, SIM[t].astype(np.float64), A[t])[0]
                             for t in range(len(S))]).astype(np.float32)
            gap = state_gap(SSRC, STGT)
            dist = needle_ring_dist(S)
            # discriminator one-sided score on the OBSERVED source transition
            with torch.no_grad():
                r_src = d.log_dynamics_ratio(
                    torch.as_tensor(S, device=device),
                    torch.as_tensor(A, device=device),
                    torch.as_tensor(SSRC, device=device)).cpu().numpy()
            score = np.abs(r_src)
            trajs.append(dict(gap=gap, dist=dist, score=score, seed=args.seed + ti,
                              success=bool(np.any(np.asarray(R) > 0))))
            print(f"  traj {ti}: len={len(S)}, success={trajs[-1]['success']}, "
                  f"gap[max={gap.max():.3f} mean={gap.mean():.3f}], "
                  f"rho(score,gap)={spearmanr(score, gap)[0]:.3f}")

    # overall correlation across all transitions
    allg = np.concatenate([t["gap"] for t in trajs])
    alls = np.concatenate([t["score"] for t in trajs])
    alld = np.concatenate([t["dist"] for t in trajs])
    rho_all, _ = spearmanr(alls, allg)
    hi = allg > np.quantile(allg, 0.8)
    print(f"\nOVERALL ({len(allg)} transitions): Spearman(disc score, gt gap) = {rho_all:.3f}")
    print(f"  disc score: high-gap mean {alls[hi].mean():.3f} vs low-gap mean {alls[~hi].mean():.3f}")
    print(f"  gt gap by phase median: contact {np.median(allg[alld<0.15]):.4f} "
          f"free {np.median(allg[alld>=0.15]):.4f}")

    # ---- plot: per-trajectory curves + overall scatter ----
    n = len(trajs)
    fig, axes = plt.subplots(2, max(3, (n + 1) // 2), figsize=(16, 7))
    axes = axes.ravel()
    for i, tr in enumerate(trajs):
        ax = axes[i]
        t = np.arange(len(tr["gap"]))
        # normalize each curve to [0,1] for shape comparison
        g = tr["gap"] / (tr["gap"].max() + 1e-9)
        sc = tr["score"] / (tr["score"].max() + 1e-9)
        ax.plot(t, g, color="C3", lw=1.8, label="gt gap (norm)")
        ax.plot(t, sc, color="C0", lw=1.2, alpha=0.8, label="disc score (norm)")
        ax.plot(t, tr["dist"], color="C1", lw=1.0, ls=":", alpha=0.7, label="needle-ring dist")
        ax.set_title(f"traj{i} (success={tr['success']})", fontsize=9)
        if i == 0:
            ax.legend(fontsize=7)
    # last axis: overall scatter
    ax = axes[-1]
    s = ax.scatter(allg, alls, c=alld, s=8, cmap="viridis", alpha=0.6)
    ax.set_xlabel("gt gap"); ax.set_ylabel("disc score")
    ax.set_title(f"overall rho={rho_all:.3f}", fontsize=9)
    plt.colorbar(s, ax=ax, label="dist")
    fig.tight_layout(); fig.savefig(args.out, dpi=120)
    print(f"saved -> {args.out}")

    out_npz = os.path.splitext(args.out)[0] + ".npz"
    tag = os.path.basename(os.path.dirname(args.behavior_weight))
    np.savez(out_npz, trajs=np.array(trajs, dtype=object), tag=tag)
    print(f"saved raw per-traj data -> {out_npz}")


if __name__ == "__main__":
    main()

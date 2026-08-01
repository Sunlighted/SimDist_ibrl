"""Compare the two counterfactually-trained gap detectors on the SOURCE RL
policy's own source-domain rollouts.

Behavior policy = threading_ibrl_source_seed1/model0.pt (the IBRL agent trained
ON the source domain). Rolling it out in the SOURCE env gives competent,
task-completing trajectories whose states stay close to the demo manifold — the
relatively in-distribution regime (cf. step5's source flow BC, rho +0.425),
NOT the OOD flailing regime.

For each transition we record:
  - s'_src   = observed source rollout next state
  - s'_tgt   = counterfactual TARGET sim step from the same sim_state
               (ground-truth, sim2sim-only — validation, never trained on)
  - GT gap   = ||s'_tgt - s'_src||   (quaternion-aware)

Two detectors scored on the SAME transitions:
  - disc      : pure counterfactual H2O discriminator  |log dyn-ratio(s,a,s'_src)|
                (discriminator_cf.pt) — one-sided, sim2real-realistic
  - res_dgi   : residual CDM + direct_gap  ||Δμ(s,a)||  (cdm_res_directgap.pt)
                — needs only (s,a)

Reports per-detector Spearman(score, GT gap), hi/lo-gap separation, and the
contact-phase localization, so we can see which detector tracks the true gap on
this policy's data.
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from cf_utils import REPO, TARGET_SHIFT, counterfactual_step, get_sim_state
from data_utils import state_gap, needle_ring_dist
from discriminator import DynamicsDiscriminator
from conditional_density import build_model
from common_utils import ibrl_utils as utils


def load_disc(path, dev):
    ck = torch.load(path, map_location=dev)
    d = DynamicsDiscriminator(ck["state_dim"], ck["action_dim"], ck["hidden"], ck["dropout"]).to(dev)
    d.load_state_dict(ck["state_dict"]); d.eval()
    return d


def load_cdm(path, dev):
    ck = torch.load(path, map_location=dev)
    m = build_model(ck["head"], ck["state_dim"], ck["action_dim"], ck["ds_dim"], hidden=ck["hidden"]).to(dev)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weight", default=f"{REPO}/exps/mimicgen/threading_ibrl_source_seed1/model0.pt")
    p.add_argument("--disc", default=f"{os.path.dirname(__file__)}/discriminator_cf.pt")
    p.add_argument("--cdm", default=f"{os.path.dirname(__file__)}/cdm_res_directgap.pt")
    p.add_argument("--num_traj", type=int, default=8)
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--sigma", type=float, default=0.0,
                   help="joint-space init sigma; 0 = checkpoint/demo default init")
    p.add_argument("--out", default=f"{os.path.dirname(__file__)}/step9_compare.png")
    args = p.parse_args()
    dev = "cuda"

    import train_rl
    from env.robosuite_wrapper import PixelRobosuite
    from env.domain_shift import DomainShiftConfig

    agent, _, env_params = train_rl.load_model(args.weight, dev)

    src_params = dict(env_params)
    src_params["domain_shift"] = DomainShiftConfig(mode="none")
    if args.sigma > 0:
        src_params["hand_init_perturb"] = None
        src_params["hand_init_perturb_joint_sigma"] = args.sigma
    src_env = PixelRobosuite(**src_params)

    tgt_params = dict(env_params)
    tgt_params["domain_shift"] = DomainShiftConfig(**TARGET_SHIFT)
    tgt_env = PixelRobosuite(**tgt_params)
    tgt_env.reset()  # build target sim once

    disc = load_disc(args.disc, dev)
    cdm = load_cdm(args.cdm, dev)

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
            STGT = np.stack([counterfactual_step(tgt_env, SIM[t].astype(np.float64), A[t])[0]
                             for t in range(len(S))]).astype(np.float32)
            gap = state_gap(SSRC, STGT)
            dist = needle_ring_dist(S)

            St = torch.as_tensor(S, device=dev)
            At = torch.as_tensor(A, device=dev)
            Nt = torch.as_tensor(SSRC, device=dev)
            with torch.no_grad():
                disc_sc = np.abs(disc.log_dynamics_ratio(St, At, Nt).cpu().numpy())
                res_sc = cdm.dgi(St, At).cpu().numpy()

            trajs.append(dict(gap=gap, dist=dist, disc=disc_sc, res=res_sc,
                              seed=args.seed + ti,
                              success=bool(np.any(np.asarray(R) > 0))))
            print(f"  traj {ti}: len={len(S)}, success={trajs[-1]['success']}, "
                  f"gap[max={gap.max():.3f} med={np.median(gap):.4f}] | "
                  f"rho_disc={spearmanr(disc_sc, gap)[0]:+.3f} "
                  f"rho_res={spearmanr(res_sc, gap)[0]:+.3f}")

    allg = np.concatenate([t["gap"] for t in trajs])
    alld = np.concatenate([t["dist"] for t in trajs])
    succ = np.mean([t["success"] for t in trajs])
    hi = allg > np.quantile(allg, 0.8)
    lo = allg < np.quantile(allg, 0.5)
    print(f"\nOVERALL: {len(allg)} transitions, {len(trajs)} trajs, "
          f"success_rate={succ:.2f}")
    print(f"  GT gap by phase median: contact(<0.15) {np.median(allg[alld<0.15]):.4f} "
          f"free(>=0.15) {np.median(allg[alld>=0.15]):.4f}")
    print(f"\n  {'detector':12s} {'rho(score,gap)':>16s} {'hi-gap mean':>12s} "
          f"{'lo-gap mean':>12s} {'hi/lo ratio':>12s}")
    for k, name in [("disc", "disc(cf)"), ("res", "res_directgap")]:
        alls = np.concatenate([t[k] for t in trajs])
        rho = spearmanr(alls, allg)[0]
        him, lom = alls[hi].mean(), alls[lo].mean()
        print(f"  {name:12s} {rho:>+16.3f} {him:>12.3f} {lom:>12.3f} "
              f"{him/(lom+1e-9):>12.2f}")

    # ---- plot: per-traj normalized curves + two overall scatters ----
    n = len(trajs)
    ncol = max(4, (n + 2 + 1) // 2)
    fig, axes = plt.subplots(2, ncol, figsize=(4 * ncol, 8))
    axes = axes.ravel()
    for i, tr in enumerate(trajs):
        ax = axes[i]
        t = np.arange(len(tr["gap"]))
        g = tr["gap"] / (tr["gap"].max() + 1e-9)
        dsc = tr["disc"] / (tr["disc"].max() + 1e-9)
        rsc = tr["res"] / (tr["res"].max() + 1e-9)
        ax.plot(t, g, color="C3", lw=1.8, label="gt gap")
        ax.plot(t, dsc, color="C0", lw=1.0, alpha=0.8, label="disc(cf)")
        ax.plot(t, rsc, color="C2", lw=1.0, alpha=0.8, label="res_directgap")
        ax.plot(t, tr["dist"], color="C1", lw=0.9, ls=":", alpha=0.6, label="needle-ring dist")
        ax.set_title(f"traj{i} (succ={tr['success']})", fontsize=9)
        if i == 0:
            ax.legend(fontsize=7)
    for k, name, ax in [("disc", "disc(cf)", axes[-2]), ("res", "res_directgap", axes[-1])]:
        alls = np.concatenate([t[k] for t in trajs])
        rho = spearmanr(alls, allg)[0]
        sct = ax.scatter(allg, alls, c=alld, s=8, cmap="viridis", alpha=0.6)
        ax.set_xlabel("gt gap"); ax.set_ylabel(f"{name} score")
        ax.set_title(f"{name}: overall rho={rho:.3f}", fontsize=9)
        plt.colorbar(sct, ax=ax, label="needle-ring dist")
    fig.tight_layout(); fig.savefig(args.out, dpi=120)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()

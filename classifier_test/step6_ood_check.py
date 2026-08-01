"""Step 6: test the hypothesis that the discriminator's early-trajectory false
high scores come from OOD init states (test sigma=0.1 >> the demo sigma=0.02
the discriminator was trained on).

For source-BC rollouts at sigma in {0.02, 0.1}, per transition we record:
  - GT gap ||s'_tgt - s'_src||  (counterfactual)
  - disc score |r_src|
  - OOD distance of s_t to the training demo state set (kNN in normalized space)
  - step index in the episode
Then: (a) early-step disc score vs sigma, (b) on the LOW-gap subset, correlation
of disc score with OOD distance.
"""

import argparse
import os

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cf_utils import REPO, TARGET_SHIFT, counterfactual_step, get_sim_state
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
    p.add_argument("--pairs", default=f"{os.path.dirname(__file__)}/pairs_demo.npz")
    p.add_argument("--sigmas", default="0.02,0.1")
    p.add_argument("--num_traj", type=int, default=4)
    p.add_argument("--seed", type=int, default=200)
    p.add_argument("--early_k", type=int, default=30, help="first-K steps = 'early'")
    p.add_argument("--knn", type=int, default=5)
    p.add_argument("--out", default=f"{os.path.dirname(__file__)}/step6_ood.png")
    args = p.parse_args()
    device = "cuda"

    import train_bc
    from env.robosuite_wrapper import PixelRobosuite
    from env.domain_shift import DomainShiftConfig

    d = load_disc(args.disc, device)
    # in-distribution reference = training demo states, normalized by disc stats
    ref = np.load(args.pairs)["s"]
    s_mean = d.s_mean.cpu().numpy(); s_std = d.s_std.cpu().numpy()
    ref_n = (ref - s_mean) / s_std

    def ood_dist(states):
        q = (states - s_mean) / s_std
        D = cdist(q, ref_n)               # (N, Nref)
        D.sort(axis=1)
        return D[:, :args.knn].mean(1)    # mean dist to k nearest demo states

    agent, _, env_params = train_bc.load_model(args.behavior_weight, device)
    tgt_params = dict(env_params); tgt_params["domain_shift"] = DomainShiftConfig(**TARGET_SHIFT)
    tgt_env = PixelRobosuite(**tgt_params); tgt_env.reset()

    sigmas = [float(x) for x in args.sigmas.split(",")]
    results = {}
    for sg in sigmas:
        sp = dict(env_params)
        sp.update(domain_shift=DomainShiftConfig(mode="none"),
                  hand_init_perturb=None, hand_init_perturb_joint_sigma=sg)
        src_env = PixelRobosuite(**sp)
        ST, GAP, SCORE, OOD, STEP = [], [], [], [], []
        with torch.no_grad(), utils.eval_mode(agent):
            for ti in range(args.num_traj):
                np.random.seed(args.seed + ti)
                obs, _ = src_env.reset()
                terminal = False; step = 0
                while not terminal:
                    sim_t = get_sim_state(src_env)
                    s_t = obs["state"].cpu().numpy()
                    action = agent.act(obs, eval_mode=True)
                    obs, reward, terminal, _, _ = src_env.step(action)
                    s_src = obs["state"].cpu().numpy()
                    s_tgt, _, _ = counterfactual_step(tgt_env, sim_t.astype(np.float64),
                                                      action.cpu().numpy())
                    gap = float(state_gap(s_src[None], s_tgt[None])[0])
                    score = float(abs(d.log_dynamics_ratio(
                        torch.as_tensor(s_t[None], device=device),
                        torch.as_tensor(action.cpu().numpy()[None], device=device),
                        torch.as_tensor(s_src[None], device=device)).item()))
                    ST.append(s_t); GAP.append(gap); SCORE.append(score); STEP.append(step)
                    step += 1
        ST = np.asarray(ST, np.float32)
        GAP, SCORE, STEP = map(np.asarray, (GAP, SCORE, STEP))
        OOD = ood_dist(ST)
        results[sg] = dict(gap=GAP, score=SCORE, step=STEP, ood=OOD)

        early = STEP < args.early_k
        lowgap = GAP < np.quantile(GAP, 0.5)
        rho_ood, _ = spearmanr(SCORE[lowgap], OOD[lowgap])
        print(f"sigma={sg}: n={len(GAP)} | disc score early(<{args.early_k}) "
              f"mean={SCORE[early].mean():.3f} vs late mean={SCORE[~early].mean():.3f} | "
              f"OOD early mean={OOD[early].mean():.2f} vs late={OOD[~early].mean():.2f} | "
              f"low-gap rho(score,OOD)={rho_ood:.3f}")

    # ---- plots ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cols = {sigmas[0]: "C0", sigmas[-1]: "C3"}
    # (1) disc score vs step (binned mean)
    ax = axes[0]
    for sg in sigmas:
        r = results[sg]; bins = np.arange(0, r["step"].max() + 10, 10)
        idx = np.digitize(r["step"], bins)
        xm = [r["step"][idx == i].mean() for i in range(1, len(bins)) if (idx == i).any()]
        ym = [r["score"][idx == i].mean() for i in range(1, len(bins)) if (idx == i).any()]
        ax.plot(xm, ym, color=cols[sg], label=f"sigma={sg}")
    ax.axvline(args.early_k, color="k", ls=":", lw=1, label=f"early<{args.early_k}")
    ax.set_xlabel("step in episode"); ax.set_ylabel("disc score (mean)")
    ax.set_title("disc score vs step"); ax.legend(fontsize=8)
    # (2) OOD distance vs step
    ax = axes[1]
    for sg in sigmas:
        r = results[sg]; bins = np.arange(0, r["step"].max() + 10, 10)
        idx = np.digitize(r["step"], bins)
        xm = [r["step"][idx == i].mean() for i in range(1, len(bins)) if (idx == i).any()]
        ym = [r["ood"][idx == i].mean() for i in range(1, len(bins)) if (idx == i).any()]
        ax.plot(xm, ym, color=cols[sg], label=f"sigma={sg}")
    ax.axvline(args.early_k, color="k", ls=":", lw=1)
    ax.set_xlabel("step in episode"); ax.set_ylabel("OOD dist to demos (mean)")
    ax.set_title("OOD distance vs step"); ax.legend(fontsize=8)
    # (3) low-gap: disc score vs OOD distance
    ax = axes[2]
    for sg in sigmas:
        r = results[sg]; lg = r["gap"] < np.quantile(r["gap"], 0.5)
        ax.scatter(r["ood"][lg], r["score"][lg], s=6, alpha=0.4, color=cols[sg], label=f"sigma={sg}")
    ax.set_xlabel("OOD distance to demos"); ax.set_ylabel("disc score")
    ax.set_title("LOW-gap transitions: disc score vs OOD dist"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.out, dpi=120)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

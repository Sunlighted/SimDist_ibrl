"""Step B: does VARIANCE-AWARE scoring improve gap localization?

The raw two-head DGI = ||mu_tgt - mu_src|| treats every state dim / regime
equally. But the source's own aleatoric spread sigma(s,a) is large in free
space (lots of motion) and small in precise contact configs. Normalizing by it
    DGI_var = ||(mu_tgt - mu_src) / sigma_src||
down-weights the noisy free-space regions (fewer false positives) and amplifies
the same mean-difference where the dynamics are crisp -> should sharpen
localization.

Tested on the conditional-Gaussian (cdm_gaussian.pt) and the residual+direct_gap
model (cdm_res_directgap.pt), raw vs variance-aware, alone and composed with the
Step-A temporal aggregation (mean, k=5). Compared against the Step-A winner
(discriminator ensemble mean) on the SAME source-RL in-dist rollouts.
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from cf_utils import REPO, TARGET_SHIFT
from conditional_density import build_model
from step10_eval_ensemble import load_ensemble, member_scores, rollout
from match_utils import rolling, mean_match


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    n_pos = y.sum(); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def load_cdm(path, dev):
    ck = torch.load(path, map_location=dev)
    m = build_model(ck["head"], ck["state_dim"], ck["action_dim"], ck["ds_dim"], hidden=ck["hidden"]).to(dev)
    m.load_state_dict(ck["state_dict"]); m.eval(); return m, ck["head"]


@torch.no_grad()
def cdm_dgi(m, head, S, A, dev, var_aware):
    St = torch.as_tensor(S, device=dev); At = torch.as_tensor(A, device=dev)
    if head == "gaussian_res":
        delta = m._delta(St, At)                      # mu_tgt - mu_src
        _, logvar = m._base(St, At)
    else:  # gaussian
        mu_t = m._mean(St, At, 1); mu_s = m._mean(St, At, 0)
        delta = mu_t - mu_s
        _, logvar = m.forward(St, At, m.onehot(0, St.size(0), dev))
    if var_aware:
        sigma = (0.5 * logvar).exp().clamp_min(1e-3)
        delta = delta / sigma
    return delta.norm(dim=-1).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weight", default=f"{REPO}/exps/mimicgen/threading_ibrl_source_seed1/model0.pt")
    p.add_argument("--ens_dir", default=f"{os.path.dirname(__file__)}/disc_ens")
    p.add_argument("--cdm_gauss", default=f"{os.path.dirname(__file__)}/cdm_gaussian.pt")
    p.add_argument("--cdm_res", default=f"{os.path.dirname(__file__)}/cdm_res_directgap.pt")
    p.add_argument("--num_traj", type=int, default=16)
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--tau_gap", type=float, default=0.025)
    p.add_argument("--agg_k", type=int, default=5)
    args = p.parse_args()
    dev = "cuda"

    import train_rl
    from env.robosuite_wrapper import PixelRobosuite
    from env.domain_shift import DomainShiftConfig

    agent, _, env_params = train_rl.load_model(args.weight, dev)
    src_params = dict(env_params); src_params["domain_shift"] = DomainShiftConfig(mode="none")
    src_env = PixelRobosuite(**src_params)
    tgt_params = dict(env_params); tgt_params["domain_shift"] = DomainShiftConfig(**TARGET_SHIFT)
    tgt_env = PixelRobosuite(**tgt_params); tgt_env.reset()
    members = load_ensemble(args.ens_dir, dev)
    mg, hg = load_cdm(args.cdm_gauss, dev)
    mr, hr = load_cdm(args.cdm_res, dev)

    trajs = rollout(agent, src_env, tgt_env, args.num_traj, args.seed, dev)
    gts = [t["gap"] for t in trajs]
    allgap = np.concatenate(gts); y = (allgap > args.tau_gap).astype(int)
    print(f"IN-DIST source RL: {len(trajs)} trajs, {len(allgap)} transitions, "
          f"{100*y.mean():.1f}% positive (GT gap>{args.tau_gap})")

    # build per-traj score curves for each method
    methods = {}
    methods["disc ens mean (A ref)"] = [member_scores(members, t["s"], t["a"], t["ssrc"], dev).mean(0)
                                        for t in trajs]
    methods["gauss DGI raw"] = [cdm_dgi(mg, hg, t["s"], t["a"], dev, False) for t in trajs]
    methods["gauss DGI var"] = [cdm_dgi(mg, hg, t["s"], t["a"], dev, True) for t in trajs]
    methods["res DGI raw"] = [cdm_dgi(mr, hr, t["s"], t["a"], dev, False) for t in trajs]
    methods["res DGI var"] = [cdm_dgi(mr, hr, t["s"], t["a"], dev, True) for t in trajs]

    print(f"\n=== STEP B: variance-aware vs raw (per-traj match + pooled AUROC) ===")
    print(f"  {'method':22s} | {'k=1':^25s} | {'k='+str(args.agg_k)+' (A+B)':^25s}")
    print(f"  {'':22s} | {'rho':>7s} {'cos':>7s} {'AUROC':>8s} | "
          f"{'rho':>7s} {'cos':>7s} {'AUROC':>8s}")
    summary = {}
    for name, curves in methods.items():
        mm1 = mean_match(curves, gts)
        au1 = auroc(y, np.concatenate(curves))
        agg = [rolling(c, args.agg_k, "mean") for c in curves]
        mmk = mean_match(agg, gts)
        auk = auroc(y, np.concatenate(agg))
        summary[name] = (curves, agg, mmk)
        print(f"  {name:22s} | {mm1['rho']:>7.3f} {mm1['cos']:>7.3f} {au1:>8.3f} | "
              f"{mmk['rho']:>7.3f} {mmk['cos']:>7.3f} {auk:>8.3f}")

    # ---- visualize: best CDM variant (k) vs disc-A vs GT, first 8 trajs ----
    best = max((k for k in methods if k != "disc ens mean (A ref)"),
               key=lambda kk: summary[kk][2]["rho"])
    print(f"\n  best CDM-based method (by A+B match-rho): {best}")
    disc_agg = summary["disc ens mean (A ref)"][1]
    best_agg = summary[best][1]
    n = min(8, len(trajs)); ncol = 4; nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    axes = axes.ravel()
    for i in range(n):
        ax = axes[i]; t = np.arange(len(gts[i]))
        # normalize each score curve to [0,1] for shape overlay
        def nz(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-9)
        ax.plot(t, nz(disc_agg[i]), c="C0", lw=1.2, label=f"disc-A (k={args.agg_k})")
        ax.plot(t, nz(best_agg[i]), c="C2", lw=1.4, label=f"{best} (k={args.agg_k})")
        ax.set_ylim(-0.05, 1.15)
        ax2 = ax.twinx()
        ax2.plot(t, gts[i], c="C3", lw=1.4, alpha=0.8, label="GT gap")
        ax2.set_ylim(0, max(0.16, gts[i].max() * 1.1))
        ax.set_title(f"traj{i} succ={trajs[i]['success']}", fontsize=9)
        if i == 0:
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper left")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"STEP B: variance-aware ({best}) vs disc-A vs GT gap (norm shape)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"{os.path.dirname(__file__)}/step14_variance_aware.png"
    fig.savefig(out, dpi=120); print(f"saved -> {out}")


if __name__ == "__main__":
    main()

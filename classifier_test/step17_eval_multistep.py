"""Step ③: evaluate the multi-step (window) detector vs the single-step detector
on the SAME source-RL in-dist rollouts.

At deployment the multi-step detector needs ONLY observed data: for time t it
reads the observed window (s_t, a_t..a_{t+k-1}, s_{t+1..t+k}) from the rollout
(no sim). The GT it predicts = the k-step counterfactual divergence (set the
TARGET sim to sim_state_t, apply the same k actions, compare trajectories).

Reports per-traj match (rho/cos/peak-recall) + pooled AUROC, comparing:
  - single-step disc ensemble  (disc_ens/, the Step-A winner)  vs single-step GT
  - multi-step  disc ensemble  (disc_ms/)                      vs multi-step GT
Both also with the Step-A temporal aggregation (mean, k=5) on top.
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from cf_utils import REPO, TARGET_SHIFT, get_sim_state
from data_utils import state_gap, needle_ring_dist
from discriminator import DynamicsDiscriminator
from ms_discriminator import MultiStepDiscriminator
from step15_build_multistep_pairs import counterfactual_rollout
from match_utils import rolling, mean_match
from common_utils import ibrl_utils as utils


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    npos = y.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s)
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def load_ss_ensemble(d_dir, dev):
    ms = []
    for f in sorted(x for x in os.listdir(d_dir) if x.startswith("disc_") and x.endswith(".pt")):
        ck = torch.load(os.path.join(d_dir, f), map_location=dev)
        d = DynamicsDiscriminator(ck["state_dim"], ck["action_dim"], ck["hidden"], ck["dropout"]).to(dev)
        d.load_state_dict(ck["state_dict"]); d.eval(); ms.append(d)
    return ms


def load_ms_ensemble(d_dir, dev):
    ms = []
    for f in sorted(x for x in os.listdir(d_dir) if x.startswith("disc_") and x.endswith(".pt")):
        ck = torch.load(os.path.join(d_dir, f), map_location=dev)
        d = MultiStepDiscriminator(37, 7, ck["k"], ck["hidden"], ck["dropout"]).to(dev)
        d.load_state_dict(ck["state_dict"]); d.eval(); ms.append(d)
    return ms, ck["k"]


@torch.no_grad()
def ss_score(members, s, a, ns, dev):
    St, At, Nt = (torch.as_tensor(x, device=dev) for x in (s, a, ns))
    return np.stack([np.abs(m.log_dynamics_ratio(St, At, Nt).cpu().numpy()) for m in members]).mean(0)


@torch.no_grad()
def ms_score(members, s0, aseq, win, dev):
    S0, AS, W = (torch.as_tensor(x, device=dev) for x in (s0, aseq, win))
    return np.stack([np.abs(m.log_dynamics_ratio(S0, AS, W).cpu().numpy()) for m in members]).mean(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weight", default=f"{REPO}/exps/mimicgen/threading_ibrl_source_seed1/model0.pt")
    p.add_argument("--ss_dir", default=f"{os.path.dirname(__file__)}/disc_ens")
    p.add_argument("--ms_dir", default=f"{os.path.dirname(__file__)}/disc_ms")
    p.add_argument("--num_traj", type=int, default=16)
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--agg_k", type=int, default=5)
    p.add_argument("--tau_quant", type=float, default=0.9, help="GT-gap quantile for the +label")
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

    ss_members = load_ss_ensemble(args.ss_dir, dev)
    ms_members, K = load_ms_ensemble(args.ms_dir, dev)
    print(f"loaded single-step ens={len(ss_members)}, multi-step ens={len(ms_members)} (k={K})")

    trajs = []
    with torch.no_grad(), utils.eval_mode(agent):
        for ti in range(args.num_traj):
            np.random.seed(args.seed + ti)
            obs, _ = src_env.reset()
            S, A, SIM = [], [], []
            terminal = False; succ = False
            while not terminal:
                SIM.append(get_sim_state(src_env)); S.append(obs["state"].cpu().numpy())
                act = agent.act(obs, eval_mode=True)
                obs, r, terminal, _, _ = src_env.step(act); A.append(act.cpu().numpy())
                succ = succ or (r > 0)
            S, A, SIM = np.asarray(S, np.float32), np.asarray(A, np.float32), np.asarray(SIM, np.float64)
            n = len(S)
            if n < K + 2:
                continue
            # single-step: (s_t, a_t, s_{t+1}); GT single-step gap via 1-step counterfactual
            m = n - K  # valid window starts
            ss_gt = np.empty(m); ms_gt = np.empty(m)
            s0 = S[:m]
            aseq = np.stack([A[t:t + K].reshape(-1) for t in range(m)])      # (m, K*7)
            win = np.stack([S[t + 1:t + 1 + K].reshape(-1) for t in range(m)])  # (m, K*37) observed source
            for t in range(m):
                tgt_roll = counterfactual_rollout(tgt_env, SIM[t], A[t:t + K])   # (K,37) target
                ms_gt[t] = state_gap(S[t + K:t + K + 1], tgt_roll[K - 1:K])[0]    # terminal k-step gap
                ss_gt[t] = state_gap(S[t + 1:t + 2], tgt_roll[0:1])[0]            # 1-step gap
            ss_sc = ss_score(ss_members, S[:m], A[:m], S[1:m + 1], dev)
            ms_sc = ms_score(ms_members, s0, aseq, win, dev)
            dist = needle_ring_dist(S[:m])
            trajs.append(dict(ss_gt=ss_gt, ms_gt=ms_gt, ss_sc=ss_sc, ms_sc=ms_sc,
                              dist=dist, success=succ))

    # ---- aggregate metrics ----
    def collect(key):
        return [t[key] for t in trajs]
    ss_gts = collect("ss_gt"); ms_gts = collect("ms_gt")
    ss_scs = collect("ss_sc"); ms_scs = collect("ms_sc")
    all_ss_gt = np.concatenate(ss_gts); all_ms_gt = np.concatenate(ms_gts)
    tau_ss = np.quantile(all_ss_gt, args.tau_quant)
    tau_ms = np.quantile(all_ms_gt, args.tau_quant)

    print(f"\nIN-DIST source RL: {len(trajs)} trajs, {len(all_ss_gt)} windows")
    print(f"  single-step GT gap: median {np.median(all_ss_gt):.4f}, tau(q{args.tau_quant})={tau_ss:.4f}")
    print(f"  multi-step  GT gap: median {np.median(all_ms_gt):.4f}, tau(q{args.tau_quant})={tau_ms:.4f}")

    print(f"\n=== detector vs its GT (per-traj match + pooled AUROC) ===")
    print(f"  {'detector':28s} | {'k=1':^25s} | {'agg k='+str(args.agg_k):^25s}")
    print(f"  {'':28s} | {'rho':>7s} {'cos':>7s} {'AUROC':>8s} | {'rho':>7s} {'cos':>7s} {'AUROC':>8s}")
    rows = [
        ("single-step disc vs ss-GT", ss_scs, ss_gts, all_ss_gt, tau_ss),
        ("multi-step disc vs ms-GT", ms_scs, ms_gts, all_ms_gt, tau_ms),
        ("multi-step disc vs ss-GT", ms_scs, ss_gts, all_ss_gt, tau_ss),
    ]
    for name, scs, gts, allgt, tau in rows:
        mm1 = mean_match(scs, gts); au1 = auroc((allgt > tau).astype(int), np.concatenate(scs))
        agg = [rolling(c, args.agg_k, "mean") for c in scs]
        mmk = mean_match(agg, gts); auk = auroc((allgt > tau).astype(int), np.concatenate(agg))
        print(f"  {name:28s} | {mm1['rho']:>7.3f} {mm1['cos']:>7.3f} {au1:>8.3f} | "
              f"{mmk['rho']:>7.3f} {mmk['cos']:>7.3f} {auk:>8.3f}")

    # ---- visualize first 8 trajs: ms-GT, ms detector, ss detector ----
    def nz(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-9)
    n = min(8, len(trajs)); ncol = 4; nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    axes = axes.ravel()
    for i in range(n):
        ax = axes[i]; t = np.arange(len(ms_gts[i]))
        ax.plot(t, nz(rolling(ms_scs[i], args.agg_k)), c="C2", lw=1.6, label=f"multi-step (k={args.agg_k})")
        ax.plot(t, nz(rolling(ss_scs[i], args.agg_k)), c="C0", lw=1.0, alpha=0.7, label=f"single-step (k={args.agg_k})")
        ax.set_ylim(-0.05, 1.15)
        ax2 = ax.twinx()
        ax2.plot(t, ms_gts[i], c="C3", lw=1.4, alpha=0.8, label="multi-step GT gap")
        ax2.set_ylim(0, max(0.05, ms_gts[i].max() * 1.1))
        mm = mean_match([rolling(ms_scs[i], args.agg_k)], [ms_gts[i]])
        ax.set_title(f"traj{i} rho_ms={mm['rho']:.2f}", fontsize=9)
        if i == 0:
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper left")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("STEP ③: multi-step vs single-step detector vs multi-step GT gap", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"{os.path.dirname(__file__)}/step17_multistep_eval.png"
    fig.savefig(out, dpi=120); print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()

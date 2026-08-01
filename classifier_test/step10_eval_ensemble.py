"""Step 10 (eval): does the discriminator ENSEMBLE score
    g(s,a,s') = mean_k |log ratio_k| - kappa * std_k
suppress OOD false-positives while preserving the in-distribution gap signal?

Two regimes, both rolling out the SOURCE IBRL policy in the SOURCE env:
  (A) IN-DIST   : clean eval rollouts (task-completing, states near the demo
                  manifold). WANT: rho(score, GT gap) stays high (signal kept),
                  ensemble std LOW.
  (B) OOD FLAIL : same policy + large action noise -> drives into novel states
                  with GT gap ~ 0. WANT: score pushed DOWN by the -kappa*std
                  term, ensemble std HIGH (members disagree off-manifold).

GT gap = ||s'_tgt - s'_src|| from a counterfactual TARGET sim step (validation
only). Compares single member 0, ensemble mean, and mean-kappa*std.
"""

import argparse
import os

import numpy as np
import torch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cf_utils import REPO, TARGET_SHIFT, counterfactual_step, get_sim_state
from data_utils import state_gap, needle_ring_dist
from discriminator import DynamicsDiscriminator
from common_utils import ibrl_utils as utils


def load_disc(path, dev):
    ck = torch.load(path, map_location=dev)
    d = DynamicsDiscriminator(ck["state_dim"], ck["action_dim"], ck["hidden"], ck["dropout"]).to(dev)
    d.load_state_dict(ck["state_dict"]); d.eval()
    return d


def load_ensemble(out_dir, dev):
    files = sorted(f for f in os.listdir(out_dir) if f.startswith("disc_") and f.endswith(".pt"))
    members = [load_disc(os.path.join(out_dir, f), dev) for f in files]
    print(f"loaded {len(members)} ensemble members from {out_dir}")
    return members


@torch.no_grad()
def member_scores(members, S, A, NS, dev):
    """Return (M, N) array of |log dyn-ratio| per member per transition."""
    St, At, Nt = (torch.as_tensor(x, device=dev) for x in (S, A, NS))
    return np.stack([np.abs(m.log_dynamics_ratio(St, At, Nt).cpu().numpy()) for m in members])


def rollout(agent, src_env, tgt_env, num_traj, seed, dev, action_noise=0.0, horizon=None):
    """Roll out `agent` in src_env; counterfactual target step for GT gap."""
    out = []
    with torch.no_grad(), utils.eval_mode(agent):
        for ti in range(num_traj):
            np.random.seed(seed + ti)
            obs, _ = src_env.reset()
            S, A, SIM, SSRC, R = [], [], [], [], []
            terminal = False; step = 0
            while not terminal:
                sim_t = get_sim_state(src_env)
                s_t = obs["state"].cpu().numpy()
                act = agent.act(obs, eval_mode=True)
                if action_noise > 0:
                    act = (act.cpu() + action_noise * torch.randn(act.shape[-1])).clamp(-1, 1)
                obs, reward, terminal, _, _ = src_env.step(act)
                S.append(s_t); A.append(act.cpu().numpy())
                SIM.append(sim_t); SSRC.append(obs["state"].cpu().numpy()); R.append(float(reward))
                step += 1
                if horizon is not None and step >= horizon:
                    break
            S, A, SIM, SSRC = map(lambda x: np.asarray(x, np.float32), (S, A, SIM, SSRC))
            STGT = np.stack([counterfactual_step(tgt_env, SIM[t].astype(np.float64), A[t])[0]
                             for t in range(len(S))]).astype(np.float32)
            out.append(dict(s=S, a=A, ssrc=SSRC, gap=state_gap(SSRC, STGT),
                            dist=needle_ring_dist(S), success=bool(np.any(np.asarray(R) > 0))))
    return out


def report(name, trajs, members, kappa, dev):
    S = np.concatenate([t["s"] for t in trajs])
    A = np.concatenate([t["a"] for t in trajs])
    NS = np.concatenate([t["ssrc"] for t in trajs])
    gap = np.concatenate([t["gap"] for t in trajs])
    sc = member_scores(members, S, A, NS, dev)        # (M, N)
    mean, std = sc.mean(0), sc.std(0)
    single = sc[0]
    combo = mean - kappa * std
    succ = np.mean([t["success"] for t in trajs])
    print(f"\n=== {name} ({len(gap)} transitions, {len(trajs)} trajs, "
          f"success={succ:.2f}, GT gap median={np.median(gap):.4f}) ===")
    print(f"  ensemble std: mean={std.mean():.3f}  (epistemic disagreement)")
    for tag, v in [("single(m0)", single), ("ens mean", mean),
                   (f"ens mean-{kappa}std", combo)]:
        rho = spearmanr(v, gap)[0]
        hi = gap > np.quantile(gap, 0.8); lo = gap < np.quantile(gap, 0.5)
        print(f"  {tag:18s} rho={rho:+.3f} | mean_score={v.mean():.3f} "
              f"| hi-gap {v[hi].mean():.3f} / lo-gap {v[lo].mean():.3f} "
              f"(ratio {v[hi].mean()/(v[lo].mean()+1e-9):.2f})")
    hi = gap > np.quantile(gap, 0.8)
    return dict(score_mean=mean.mean(), score_combo=combo.mean(), std=std.mean(),
                mean=mean, std_arr=std, gap=gap, hi_real=mean[hi])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weight", default=f"{REPO}/exps/mimicgen/threading_ibrl_source_seed1/model0.pt")
    p.add_argument("--ood_bc_weight",
                   default=f"{REPO}/exps/mimicgen/threading_target_only_mlp_bc_seed1/model1.pt",
                   help="weak target-only MLP BC -> genuine FAILING trajectories in "
                        "source (OOD, GT gap~0). The hallucination case the ensemble "
                        "must suppress.")
    p.add_argument("--ens_dir", default=f"{os.path.dirname(__file__)}/disc_ens")
    p.add_argument("--kappa", type=float, default=0.5)
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

    # (A) in-dist: the source RL policy (already tightly fits the demos) rolling
    # out cleanly in source. WANT: keep whatever gap signal there is, low std.
    indist = rollout(agent, src_env, tgt_env, args.num_traj, args.seed, dev)
    rA = report("(A) IN-DIST clean source RL rollout", indist, members, args.kappa, dev)

    # (B) genuine OOD: weak target-only MLP BC FAILS in source, wandering into
    # novel states with GT gap ~ 0. This is where the single disc hallucinates
    # high; we want the ensemble std to rise here so mean-K*std pushes it down.
    bc_agent, _, _ = train_bc.load_model(args.ood_bc_weight, dev)
    ood = rollout(bc_agent, src_env, tgt_env, args.num_traj, args.seed + 900, dev,
                  horizon=args.ood_horizon)
    rB = report("(B) OOD target-only MLP BC failures in source (GT gap~0)",
                ood, members, args.kappa, dev)

    print(f"\n=== OOD SUPPRESSION SUMMARY (kappa={args.kappa}) ===")
    print(f"  {'regime':32s} {'ens std':>8s} {'ens mean':>9s} {'mean-Kstd':>10s}")
    for nm, r in [("A in-dist source RL (keep signal)", rA),
                  ("B target-only-BC failures (gap~0)", rB)]:
        print(f"  {nm:32s} {r['std']:>8.3f} {r['score_mean']:>9.3f} {r['score_combo']:>10.3f}")
    print(f"  want: std HIGHER on B than A, so mean-Kstd pushes OOD(B) below in-dist(A).")

    # ---- kappa sweep: separation between in-dist REAL high gap vs OOD false gap ----
    # The decision-relevant metric: real in-dist high-gap transitions should stay
    # ABOVE the OOD false positives. separation = mean(A hi-real) / mean(B all).
    print(f"\n=== KAPPA SWEEP: g = mean - kappa*std ===")
    print(f"  {'kappa':>6s} {'A hi-real':>10s} {'B OOD mean':>11s} {'separation':>11s} "
          f"{'A rho':>7s}")
    a_hi_gap = rA["gap"] > np.quantile(rA["gap"], 0.8)
    for kap in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        a_combo = rA["mean"] - kap * rA["std_arr"]
        b_combo = rB["mean"] - kap * rB["std_arr"]
        a_hi = a_combo[a_hi_gap].mean()
        b_ood = b_combo.mean()
        rho_a = spearmanr(a_combo, rA["gap"])[0]
        print(f"  {kap:>6.1f} {a_hi:>10.3f} {b_ood:>11.3f} {a_hi/(b_ood+1e-9):>11.2f} "
              f"{rho_a:>+7.3f}")

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, r, title in [(axes[0], rA, "A in-dist source RL"),
                         (axes[1], rB, "B target-only-BC failures (OOD, gap~0)")]:
        ax.scatter(r["gap"], r["mean"], s=8, alpha=0.4, c="C0", label="ens mean")
        ax.scatter(r["gap"], r["mean"] - args.kappa * r["std_arr"], s=8, alpha=0.4,
                   c="C2", label=f"mean-{args.kappa}std")
        ax.set_xlabel("GT gap"); ax.set_ylabel("score")
        ax.set_title(f"{title}\nstd mean={r['std']:.3f}"); ax.legend(fontsize=8)
    out = f"{os.path.dirname(__file__)}/step10_ensemble.png"
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()

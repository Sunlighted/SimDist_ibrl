"""Would MORE dynamics heads still help? Decompose the H-step model error into a
bias floor and a variance term that averaging can remove.

Motivation. On Square, eval-time truncation of the 5-head ensemble gives a score
that is still rising linearly at 5 heads (1/2/3/5 -> 0.620/0.680/0.730/0.805,
+0.046/head, R^2 0.99), unlike Threading where docs/findings.md 31 found
saturation at ~3. Re-distilling with 10 heads costs ~2.7 days, so measure first
whether there is anything left for averaging to remove.

Model. Write each head's H-step rolled latent as f_i = z* + b + e_i, where z* is
the encoded REAL terminal state, b is the shared (irreducible) bias and e_i the
per-head deviation. The k-head mean then has

    MSE(k) = ||b||^2 + sigma^2 / k        (e_i uncorrelated)

so fitting MSE(k) = A + B/k over k = 1..K splits the error into a floor A that no
number of heads can touch and a reducible part B/k. The decision number is the
share of the CURRENT error that is still reducible: (B/K) / (A + B/K). Small =>
more heads buy nothing; large => the 10-head distillation is worth its 2.7 days.

Measured ON-POLICY (iron rule 5): states come from the agent actually running with
its planner, and the action sequence rolled is the policy's own closed-loop
rollout -- the same candidate construction the planner seeds CEM with. Evaluating
the model anywhere else measures a distribution the planner never visits.

    python scripts/diag_head_convergence.py --episodes 20 --probe_workers 8
"""
import argparse
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/Chenyang/ibrl")
sys.path.insert(0, "/home/Chenyang/ibrl/scripts")
os.environ.setdefault("MUJOCO_GL", "egl")

from eval_square_oracle_mppi import ProbePool, TARGET_B  # noqa: E402

ERR = sys.stderr


def log(*a):
    print(*a, file=ERR, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="exps/tmp/sq_matched_budget/model0.pt",
                    help="SimDist ckpt whose dir holds a stage-1 MainConfig")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--every", type=int, default=5, help="sample every Nth step")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--probe_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import train_simdist
    from env.domain_shift import DomainShiftConfig

    agent, _env, env_params = train_simdist.load_model(args.weight, "cuda")
    agent.train(False)
    for p in agent.parameters():
        p.requires_grad_(False)
    m = agent.model
    K = m.num_dynamics_heads()
    assert K > 1, "need an ensemble"

    env_params = dict(env_params)
    env_params["domain_shift"] = DomainShiftConfig(
        mode="fixed", mask_hole_offset_in_obs=False, **TARGET_B)
    from env.robosuite_wrapper import PixelRobosuite
    env = PixelRobosuite(**env_params)
    idc = env_params.get("init_distribution", None)
    log(f"[setup] {args.weight} | heads={K} H={args.horizon}")
    log(f"[setup] init_dist enabled={getattr(idc, 'enabled', False)} "
        f"sigma={env_params.get('hand_init_perturb_joint_sigma')} "
        f"shift={TARGET_B}")
    pool = ProbePool(env_params, args.probe_workers)

    H = args.horizon
    per_head, z_star = [], []
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        for t in range(int(env_params.get("episode_length", 400))):
            state = obs["state"].to("cuda").float().view(1, -1)
            if t % args.every == 0:
                # the policy's own closed-loop H-step sequence: exactly the base
                # candidate SimDist's planner seeds CEM with (simdist_planner
                # _policy_rollout), so the states scored here are on-policy.
                z = m.encode(state)
                acts, zz = [], z.clone()
                for _ in range(H):
                    a = m.policy_action(m.decode(zz))
                    acts.append(a)
                    zz = m.next(zz, a)
                seq = torch.stack(acts, 0)                      # [H,1,A]

                # every head rolled SEPARATELY on that same sequence
                fs = []
                for i in range(K):
                    zi = z.clone()
                    for h in range(H):
                        zi = m.next_all(zi, seq[h])[i]
                    fs.append(zi.view(-1))
                sim_state = env.env.sim.get_state().flatten()
                real = pool.rollout(sim_state, seq)              # [H,1,obs]
                z_star.append(m.encode(real[-1].float().view(1, -1)).view(-1).cpu())
                per_head.append(torch.stack(fs, 0).cpu())        # [K,D]

            act = agent.act(obs, eval_mode=True, stddev=0.0, cpu=True)
            obs, _, terminal, _, _ = env.step(act)
            if terminal:
                break
        log(f"  ep {ep+1}/{args.episodes}  points={len(per_head)}")

    pool.close()
    F = torch.stack(per_head, 0).double()      # [M,K,D]
    Z = torch.stack(z_star, 0).double()        # [M,D]
    M, _, D = F.shape
    log(f"\n[data] {M} on-policy decision points, {K} heads, latent dim {D}")

    ks, mses = [], []
    for k in range(1, K + 1):
        subs = list(itertools.combinations(range(K), k))
        acc = 0.0
        for s in subs:
            mu = F[:, list(s)].mean(1)                          # [M,D]
            acc += ((mu - Z) ** 2).sum(-1).mean().item()
        ks.append(k)
        mses.append(acc / len(subs))

    A = np.vstack([np.ones(len(ks)), 1.0 / np.array(ks, float)]).T
    (a_fit, b_fit), *_ = np.linalg.lstsq(A, np.array(mses), rcond=None)
    pred = A @ [a_fit, b_fit]
    r2 = 1 - ((np.array(mses) - pred) ** 2).sum() / \
             ((np.array(mses) - np.mean(mses)) ** 2).sum()

    log(f"\n{'k':>3}{'MSE(k) to real z*':>20}{'fit A+B/k':>14}")
    for k, v, p in zip(ks, mses, pred):
        log(f"{k:>3}{v:>20.5f}{p:>14.5f}")
    log(f"\n  fit  MSE(k) = A + B/k   with A={a_fit:.5f} (bias floor)  "
        f"B={b_fit:.5f}   R2={r2:.4f}")
    if b_fit <= 0:
        log("  B <= 0: no variance term at all -- extra heads cannot help.")
        return
    cur = a_fit + b_fit / K
    log(f"  error at K={K}: {cur:.5f}   of which reducible {b_fit/K:.5f} "
        f"= {100*(b_fit/K)/cur:.1f}%")
    for k2 in (7, 10, 20):
        e = a_fit + b_fit / k2
        log(f"  predicted at {k2:>2} heads: {e:.5f}   "
            f"({100*(1-e/cur):+.1f}% vs K={K})")
    log(f"  floor (infinite heads): {a_fit:.5f}  ({100*(1-a_fit/cur):+.1f}% vs K={K})")
    log("\n  NOTE: an extrapolation, and it predicts LATENT error, not score. "
        "docs/findings.md 30 measured that mm-level prediction error has zero "
        "predictive power for the planner's score -- so read this as 'is there "
        "variance left to average away', not as a score forecast.")


if __name__ == "__main__":
    main()

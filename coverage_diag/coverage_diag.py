"""Coverage diagnostic for the "source-online + target-demo-anchored" idea.

THE QUESTION (go/no-go for the gated method):
  The policy pi_phi is improved by interacting ONLY in SOURCE. Target demos are
  the only window onto the TARGET dynamics, so we can only "anchor" the dynamics
  where pi_phi VISITS states that the demos also cover. The method is feasible
  IFF the high-dynamics-gap transitions pi_phi visits are NEAR the target-demo
  support. If high-gap states are systematically FAR from any demo, then in
  exactly the region where source physics is wrong we have no target signal
  (and pi_demo is OOD there too) -> the idea is structurally limited for this gap.

WHAT THIS SCRIPT DOES:
  1. Roll out the SOURCE RL actor (= pi_phi init) in the SOURCE env, optionally
     also the target-demo BC policy (pi_demo) -> the states pi_phi plausibly
     visits over training.
  2. For every visited transition (s,a,s'_src), measure the GROUND-TRUTH
     per-transition dynamics gap by counterfactually replaying (s,a) in the
     TARGET sim (cf_utils.counterfactual_step) -> gap = ||s'_tgt - s'_src||.
  3. For every visited state s, measure the distance to the nearest TARGET-demo
     state in a per-dim-standardized, quaternion-canonicalized 37-dim space.
  4. Headline (threshold-free): is the demo-distance of HIGH-gap transitions
     systematically larger than LOW-gap? Spearman(gap, demo_dist), median ratio,
     coverage-vs-gap-percentile curve. Visualized in coverage.png.

Usage:
  python coverage_diag/coverage_diag.py                       # defaults
  python coverage_diag/coverage_diag.py --n_episodes 60 --no_bc_demo
"""

import argparse
import os
import sys

import numpy as np
import torch

REPO = "/home/Chenyang/ibrl"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "classifier_test"))
os.chdir(REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import cf_utils
import data_utils
from common_utils import ibrl_utils as utils

QUAT_BLOCKS = data_utils.QUAT_BLOCKS


def canon_quats(s):
    """Sign-canonicalize each quaternion block (first comp >= 0) to kill the
    q/-q double cover that otherwise inflates nearest-neighbour distances."""
    out = s.copy()
    for a, _ in QUAT_BLOCKS:
        flip = out[:, a] < 0
        for (qa, qb) in QUAT_BLOCKS:
            if qa == a:
                out[flip, qa:qb] *= -1.0
    return out


# --------------------------------------------------------------------------- IO
def load_demo_states(hdf5_path):
    eps = cf_utils.load_demo_episodes(hdf5_path)
    states = np.concatenate([e["state"] for e in eps], axis=0).astype(np.float32)
    print(f"target demos: {len(eps)} episodes, {states.shape[0]} states")
    return states


# ---------------------------------------------------------------------- rollout
@torch.no_grad()
def rollout(agent, src_env, tgt_env, n_episodes, stddev, label, max_steps=400):
    """Roll the policy in SOURCE; per transition capture (state, gap, needle-ring
    dist). gap via counterfactual replay of (s,a) in the TARGET sim."""
    states, gaps, nrings, ep_ids = [], [], [], []
    ep = 0
    while ep < n_episodes:
        obs, _ = src_env.reset()
        steps = 0
        while True:
            sim_state = cf_utils.get_sim_state(src_env)
            s_now = cf_utils.extract_state(_obs_raw(src_env))
            with utils.eval_mode(agent):
                action = agent.act(obs, eval_mode=(stddev == 0.0), stddev=stddev)
            obs, reward, terminal, success, _ = src_env.step(action)
            a_np = action.detach().cpu().numpy().reshape(-1)
            s_src = cf_utils.extract_state(_obs_raw(src_env))
            s_tgt, _, _ = cf_utils.counterfactual_step(tgt_env, sim_state, a_np)
            n = min(len(s_src), len(s_tgt))
            gap = float(data_utils.state_gap(s_src[None, :n], s_tgt[None, :n])[0])
            states.append(s_now)
            gaps.append(gap)
            nrings.append(float(data_utils.needle_ring_dist(s_now[None])[0]))
            ep_ids.append(ep)
            steps += 1
            if terminal or steps >= max_steps:
                break
        ep += 1
        if ep % 5 == 0:
            print(f"  [{label}] rolled {ep}/{n_episodes} episodes, {len(states)} transitions")
    return (np.asarray(states, np.float32), np.asarray(gaps, np.float32),
            np.asarray(nrings, np.float32), np.asarray(ep_ids, np.int64))


def _obs_raw(env):
    """37-dim state from the wrapper's underlying robosuite obs (force fresh)."""
    return env.env._get_observations(force_update=True)


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_agent", default="exps/mimicgen/threading_ibrl_source_seed1/model0.pt")
    ap.add_argument("--bc_demo", default="exps/mimicgen/threading_target_mlp_bc_n20_seed1/model0.pt")
    ap.add_argument("--no_bc_demo", action="store_true", help="skip the pi_demo rollout set")
    ap.add_argument("--demo_hdf5", default="release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5")
    ap.add_argument("--n_episodes", type=int, default=40)
    ap.add_argument("--stddev", type=float, default=0.1, help="exploration noise during rollout (0=greedy)")
    ap.add_argument("--out_dir", default="coverage_diag")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- target demo support
    demo_states = load_demo_states(args.demo_hdf5)
    demo_c = canon_quats(demo_states)
    mu = demo_c.mean(0)
    sd = demo_c.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)              # freeze constant dims
    demo_z = (demo_c - mu) / sd
    from scipy.spatial import cKDTree
    tree = cKDTree(demo_z)
    # demo intra-support leave-one-out NN distance (k=2 -> skip self): the natural
    # "spacing" scale of the support, for context on the coverage radius.
    d_intra, _ = tree.query(demo_z, k=2)
    demo_nn = d_intra[:, 1]
    print(f"demo intra-NN dist (z-space): p50={np.median(demo_nn):.3f} p90={np.quantile(demo_nn,0.9):.3f}")

    # ---- envs + source RL agent (pi_phi init)
    import train_rl
    print("loading source agent ...")
    agent, _, _ = train_rl.load_model(args.source_agent, "cuda")
    agent.train(False)
    agent.cfg.act_method = "rl"                     # deploy the bare RL actor = pi_phi init
    src_env = cf_utils.get_env(args.source_agent, domain="source")
    tgt_env = cf_utils.get_env(args.source_agent, domain="target")

    sets = []
    print("=== rollout: source RL actor (pi_phi init) ===")
    sets.append(("source_rl", *rollout(agent, src_env, tgt_env, args.n_episodes, args.stddev, "source_rl")))

    # ---- optional pi_demo (target BC) rollout — brackets where gating pushes pi_phi
    if not args.no_bc_demo and args.bc_demo and os.path.isfile(args.bc_demo):
        try:
            import train_bc
            print("=== rollout: pi_demo (target BC) ===")
            bc = train_bc._load_model(args.bc_demo, src_env, "cuda")

            class _BCWrap:
                cfg = type("c", (), {"act_method": "rl"})()
                training = False
                def __init__(self, p): self.p = p
                def train(self, _=False): pass
                def act(self, obs, eval_mode=True, stddev=0.0):
                    a = self.p.act(obs, eval_mode=True)
                    if isinstance(a, (tuple, list)): a = a[0]
                    if a.dim() == 2 and a.size(0) == 1 and a.size(1) > 7: a = a[:, :7]
                    return a
            sets.append(("pi_demo", *rollout(_BCWrap(bc), src_env, tgt_env,
                                             max(10, args.n_episodes // 2), args.stddev, "pi_demo")))
        except Exception as e:
            print(f"[warn] pi_demo rollout skipped: {e}")

    # ---- pool + nearest-demo distance
    pol = np.concatenate([np.full(len(s[1]), i) for i, s in enumerate(sets)])
    states = np.concatenate([s[1] for s in sets], 0)
    gaps = np.concatenate([s[2] for s in sets], 0)
    nrings = np.concatenate([s[3] for s in sets], 0)
    pol_names = [s[0] for s in sets]

    vis_z = (canon_quats(states) - mu) / sd
    demo_dist, _ = tree.query(vis_z, k=1)           # distance to nearest demo state
    print(f"\nvisited transitions: {len(states)}  (policies: {pol_names})")

    np.savez(os.path.join(args.out_dir, "coverage_data.npz"),
             states=states, gaps=gaps, needle_ring=nrings, demo_dist=demo_dist,
             policy=pol, policy_names=np.array(pol_names), demo_nn=demo_nn)

    summarize_and_plot(gaps, demo_dist, nrings, demo_nn, args.out_dir)


def summarize_and_plot(gaps, demo_dist, nrings, demo_nn, out_dir):
    from scipy.stats import spearmanr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hi = gaps >= np.quantile(gaps, 0.75)            # top-25% gap = "where source is most wrong"
    lo = ~hi
    rho = spearmanr(gaps, demo_dist)[0]
    med_hi, med_lo = np.median(demo_dist[hi]), np.median(demo_dist[lo])
    # relative coverage radius = median visited demo-dist (threshold-light illustration)
    ref = np.median(demo_dist)
    cov_hi = np.mean(demo_dist[hi] <= ref)
    cov_lo = np.mean(demo_dist[lo] <= ref)

    print("\n==================== COVERAGE VERDICT ====================")
    print(f"Spearman(gap, dist_to_nearest_demo) = {rho:+.3f}")
    print(f"   ~0 / negative -> high-gap states are NOT farther from demos (GOOD)")
    print(f"   strongly positive -> high-gap states are FARTHER (idea limited)")
    print(f"median demo-dist  high-gap={med_hi:.3f}  low-gap={med_lo:.3f}  ratio={med_hi/max(med_lo,1e-9):.2f}x")
    print(f"coverage(<= median {ref:.3f})  high-gap={cov_hi:.2f}  low-gap={cov_lo:.2f}")
    print("==========================================================")

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # A: gap vs demo-dist, colored by contact phase (needle-ring dist)
    sc = ax[0, 0].scatter(demo_dist, gaps, c=nrings, s=6, alpha=0.4, cmap="viridis_r")
    ax[0, 0].axvline(ref, color="k", ls="--", lw=1, label=f"median dist={ref:.2f}")
    ax[0, 0].set_xlabel("dist to nearest target demo (z-space)")
    ax[0, 0].set_ylabel("GT dynamics gap  ||s'_tgt - s'_src||")
    ax[0, 0].set_title(f"A. gap vs demo-distance  (Spearman={rho:+.3f})")
    ax[0, 0].legend(loc="upper right", fontsize=8)
    plt.colorbar(sc, ax=ax[0, 0], label="needle-ring dist (contact=low)")

    # B: demo-dist distribution, high-gap vs low-gap
    bins = np.linspace(0, np.quantile(demo_dist, 0.99), 50)
    ax[0, 1].hist(demo_dist[lo], bins=bins, density=True, alpha=0.5, label="low-gap (bottom 75%)")
    ax[0, 1].hist(demo_dist[hi], bins=bins, density=True, alpha=0.5, label="high-gap (top 25%)")
    ax[0, 1].axvline(np.quantile(demo_nn, 0.9), color="g", ls=":", lw=1.5,
                     label=f"demo intra-NN p90={np.quantile(demo_nn,0.9):.2f}")
    ax[0, 1].set_xlabel("dist to nearest target demo (z-space)")
    ax[0, 1].set_ylabel("density")
    ax[0, 1].set_title("B. demo-distance: high-gap vs low-gap")
    ax[0, 1].legend(fontsize=8)

    # C: coverage rate vs gap percentile
    n_bins = 10
    qs = np.quantile(gaps, np.linspace(0, 1, n_bins + 1))
    centers, cov, gmed = [], [], []
    for i in range(n_bins):
        m = (gaps >= qs[i]) & (gaps <= qs[i + 1])
        if m.sum() == 0:
            continue
        centers.append((i + 0.5) / n_bins * 100)
        cov.append(np.mean(demo_dist[m] <= ref))
        gmed.append(np.median(gaps[m]))
    ax[1, 0].plot(centers, cov, "o-", color="C3")
    ax[1, 0].set_xlabel("gap percentile bin (%)")
    ax[1, 0].set_ylabel(f"coverage rate (dist <= {ref:.2f})")
    ax[1, 0].set_ylim(0, 1.02)
    ax[1, 0].set_title("C. coverage vs gap percentile\n(falling curve = high-gap under-covered)")
    ax[1, 0].grid(alpha=0.3)

    # D: gap vs needle-ring distance (sanity: gap lives in contact phase)
    ax[1, 1].scatter(nrings, gaps, c=demo_dist, s=6, alpha=0.4, cmap="plasma")
    ax[1, 1].axvline(0.15, color="k", ls="--", lw=1, label="contact thresh 0.15")
    ax[1, 1].set_xlabel("needle-ring distance (m)")
    ax[1, 1].set_ylabel("GT dynamics gap")
    ax[1, 1].set_title("D. gap vs contact phase (color=demo-dist)")
    ax[1, 1].legend(fontsize=8)

    fig.suptitle("Coverage of high-gap transitions by target-demo support", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(out_dir, "coverage.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

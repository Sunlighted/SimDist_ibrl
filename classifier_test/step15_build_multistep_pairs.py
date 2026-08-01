"""Step ①: build TRUE multi-step counterfactual pairs + verify SNR rises with k.

For each target-demo start t, set the SOURCE sim to sim_state_t and roll it
forward under the demo's ACTION SEQUENCE a_t..a_{t+k-1}. The demo's own recorded
states s_{t+1..t+k} are the TARGET k-step trajectory. Same start + same actions
=> the divergence between the two k-step trajectories is the pure dynamics gap,
COMPOUNDED over k steps (signal grows ~linearly while per-step noise is bounded).

We roll each start once to max_k and read the terminal divergence at every k, so
the diagnostic (does the contact-vs-free separation improve with k?) is cheap.
The full k=main_k window pairs are saved for training the multi-step detector.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from cf_utils import REPO, get_source_env, load_demo_episodes, extract_state
from data_utils import state_gap, needle_ring_dist


def counterfactual_rollout(env, sim_state, actions):
    """Set SOURCE sim to sim_state, apply the action SEQUENCE, return the list of
    resulting 37-dim states (one per action). Independent of episode bookkeeping."""
    raw = env.env
    raw.done = False; raw.timestep = 0
    raw.sim.set_state_from_flattened(np.asarray(sim_state, dtype=np.float64))
    # zero the solver warmstart so the rollout is deterministic / reproducible
    # (set_state does not restore it; carried-over warmstart -> chaotic divergence
    # in stiff contact). See step18/step18b.
    raw.sim.data.qacc_warmstart[:] = 0.0
    raw.sim.forward()
    states = []
    for a in actions:
        raw.done = False  # never let a mid-rollout success terminate the rollout
        obs, _, _, _ = raw.step(np.asarray(a, dtype=np.float64))
        states.append(extract_state(obs))
    return np.asarray(states, np.float32)   # (len(actions), 37)


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    npos = y.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s)
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_hdf5", default=f"{REPO}/release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5")
    p.add_argument("--source_weight", default=f"{REPO}/exps/mimicgen/threading_ibrl_source_seed1/model0.pt")
    p.add_argument("--max_k", type=int, default=8)
    p.add_argument("--main_k", type=int, default=5)
    p.add_argument("--out", default=f"{os.path.dirname(__file__)}/pairs_multistep.npz")
    p.add_argument("--plot", default=f"{os.path.dirname(__file__)}/step15_gap_vs_k.png")
    args = p.parse_args()

    eps = load_demo_episodes(args.target_hdf5)
    print(f"loaded {len(eps)} target demo episodes")
    env = get_source_env(args.source_weight)

    K = args.max_k
    # per-anchor records
    S0, ACTS, SRC_WIN, TGT_WIN, DIST, EP, TS = [], [], [], [], [], [], []
    # terminal gap at each k for the diagnostic
    gap_at_k = {k: [] for k in range(1, K + 1)}
    contact_at_k = {k: [] for k in range(1, K + 1)}

    for ei, ep in enumerate(eps):
        st = ep["state"]; ac = ep["action"]; sim = ep["sim_state"]
        T = st.shape[0] - 1
        for t in range(T - K + 1):  # need K future steps available
            acts = ac[t:t + K]                       # (K, 7)
            src = counterfactual_rollout(env, sim[t], acts)   # (K, 37) source dynamics
            tgt = st[t + 1:t + 1 + K]                 # (K, 37) demo (target dynamics)
            d0 = float(needle_ring_dist(st[t:t + 1])[0])
            for k in range(1, K + 1):
                g = float(state_gap(src[k - 1:k], tgt[k - 1:k])[0])  # terminal at horizon k
                gap_at_k[k].append(g)
                contact_at_k[k].append(d0 < 0.15)
            # save the main_k window
            S0.append(st[t]); ACTS.append(acts[:args.main_k])
            SRC_WIN.append(src[:args.main_k]); TGT_WIN.append(tgt[:args.main_k])
            DIST.append(d0); EP.append(ei); TS.append(t)
        print(f"  ep {ei:2d}: {max(0, T - K + 1)} anchors")

    # ---- diagnostic: does terminal gap separate contact vs free better as k grows? ----
    print(f"\n=== multi-step GT gap structure (terminal divergence at horizon k) ===")
    print(f"  {'k':>3s} {'gap median':>11s} {'contact med':>12s} {'free med':>10s} "
          f"{'ratio c/f':>10s} {'AUROC(contact)':>15s}")
    ks_plot, cmed, fmed, aucs = [], [], [], []
    for k in range(1, K + 1):
        g = np.asarray(gap_at_k[k]); c = np.asarray(contact_at_k[k])
        cm = np.median(g[c]); fm = np.median(g[~c])
        au = auroc(c.astype(int), g)
        print(f"  {k:>3d} {np.median(g):>11.4f} {cm:>12.4f} {fm:>10.4f} "
              f"{cm/(fm+1e-9):>10.2f} {au:>15.3f}")
        ks_plot.append(k); cmed.append(cm); fmed.append(fm); aucs.append(au)

    # ---- save main_k window pairs ----
    out = dict(
        s0=np.asarray(S0, np.float32), acts=np.asarray(ACTS, np.float32),
        src_win=np.asarray(SRC_WIN, np.float32), tgt_win=np.asarray(TGT_WIN, np.float32),
        dist=np.asarray(DIST, np.float32), ep_id=np.asarray(EP, np.int64),
        t_in_ep=np.asarray(TS, np.int64), k=args.main_k,
        gap_k=np.asarray(gap_at_k[args.main_k], np.float32),
        gap_k1=np.asarray(gap_at_k[1], np.float32),
    )
    np.savez(args.out, **out)
    print(f"\nsaved {len(S0)} window pairs (k={args.main_k}) -> {args.out}")

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(ks_plot, cmed, "o-", c="C3", label="contact (dist<0.15)")
    axes[0].plot(ks_plot, fmed, "o-", c="C0", label="free space")
    axes[0].set_xlabel("counterfactual horizon k"); axes[0].set_ylabel("median terminal gap")
    axes[0].set_title("multi-step gap: contact vs free"); axes[0].legend()
    axes[1].plot(ks_plot, aucs, "o-", c="C2")
    axes[1].axhline(0.5, c="gray", ls="--", lw=1)
    axes[1].set_xlabel("counterfactual horizon k")
    axes[1].set_ylabel("AUROC(gap separates contact)")
    axes[1].set_title("separation improves with k? (GT, not detector)")
    axes[1].set_ylim(0.45, 1.0)
    fig.tight_layout(); fig.savefig(args.plot, dpi=120)
    print(f"saved -> {args.plot}")


if __name__ == "__main__":
    main()

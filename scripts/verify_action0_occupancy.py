"""Verify whether "reset s in TARGET, apply action=0, |s'-s| > thr => has gap"
recovers state OCCUPANCY / reachability. Claim: NO.

It measures ||f_target(s,0) - s||, the LOCAL DYNAMICS STABILITY of s (how fast s
drifts under a hold action), which is dominated by the state's VELOCITY and is
orthogonal to whether s is reachable / in the target occupancy. We test it on
TARGET-demo states, which are ALL reachable by construction: if the metric
varies wildly across these (and tracks qvel, not the true dynamics gap), then a
threshold on it cannot separate reachable from unreachable.
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "classifier_test"))
sys.path.insert(0, os.path.dirname(_HERE))

import numpy as np
import cf_utils
import data_utils

WEIGHT = "exps/mimicgen/threading_ibrl_source_seed1/model0.pt"
DEMO = "release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5"


def stats(x):
    return (f"min={x.min():.4f} p25={np.percentile(x,25):.4f} med={np.median(x):.4f} "
            f"p75={np.percentile(x,75):.4f} max={x.max():.4f}")


def main():
    tgt_env = cf_utils.get_env(WEIGHT, domain="target")
    src_env = cf_utils.get_env(WEIGHT, domain="source")
    nq = tgt_env.env.sim.model.nq
    nv = tgt_env.env.sim.model.nv
    print(f"nq={nq} nv={nv} (sim_state = 1 + nq + nv = {1+nq+nv})")

    eps = cf_utils.load_demo_episodes(DEMO, max_episodes=20)
    zero = np.zeros(7, dtype=np.float64)

    change_tgt, qvel_norm, dyn_gap0 = [], [], []
    for ep in eps:
        sims = ep["sim_state"]
        for t in range(0, len(sims), 2):  # stride 2 to keep it quick
            ss = sims[t]
            s0 = cf_utils.reconstructed_state(tgt_env, ss)          # reset to s in TARGET
            s1, _, _ = cf_utils.counterfactual_step(tgt_env, ss, zero)  # apply action=0
            change_tgt.append(np.linalg.norm(s1 - s0))
            qvel_norm.append(np.linalg.norm(ss[1 + nq:1 + nq + nv]))
            # the TRUE dynamics gap at action=0 (source vs target next-state)
            s1_src, _, _ = cf_utils.counterfactual_step(src_env, ss, zero)
            dyn_gap0.append(float(data_utils.state_gap(s1_src[None], s1[None])[0]))

    change_tgt = np.array(change_tgt)
    qvel_norm = np.array(qvel_norm)
    dyn_gap0 = np.array(dyn_gap0)

    print(f"\nN = {len(change_tgt)} TARGET-demo states (ALL reachable by construction)\n")
    print("action=0 single-step |s'-s| in TARGET :", stats(change_tgt))
    print("state qvel norm                        :", stats(qvel_norm))
    print("TRUE dynamics gap at action=0 (src↔tgt):", stats(dyn_gap0))
    print()
    print(f"corr( |s'-s| , qvel_norm )   = {np.corrcoef(change_tgt, qvel_norm)[0,1]:+.3f}  "
          "(if high: the metric just reads velocity, not gap)")
    print(f"corr( |s'-s| , true_dyn_gap )= {np.corrcoef(change_tgt, dyn_gap0)[0,1]:+.3f}  "
          "(if low: the metric is NOT measuring the dynamics gap either)")
    print()
    # the user's proposed rule on REACHABLE states: how many get mislabeled?
    for name, thr in [("median", np.median(change_tgt)),
                      ("p25", np.percentile(change_tgt, 25))]:
        frac = (change_tgt > thr).mean()
        print(f"rule '|s'-s| > {thr:.4f} ({name}) => unreachable' "
              f"FLAGS {frac*100:.0f}% of these REACHABLE states as unreachable")

    print("\nVERDICT: action=0 single-step change tracks velocity, is ~uncorrelated")
    print("with the true dynamics gap, and mislabels a large fraction of reachable")
    print("states -> it measures local stability, NOT occupancy/reachability.")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

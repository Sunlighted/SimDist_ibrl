"""Is the WM dynamics error fixable with data, or is the contact dynamics
itself chaotic (so no deterministic model can predict it)? Measure how much the
REAL env diverges under a tiny action perturbation (0.02), at a free-space state
vs a contact/insertion state. If the TRUE env spreads as much as the WM error,
the limit is physical chaos, not training data.
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.eval_simdist_policy_only as e
import classifier_test.cf_utils as cf

SOURCE = "exps/mimicgen/threading_ibrl_source_seed1/model0.pt"
DEMOS = "release/data/mimicgen/threading_d0_source_sigma0.1/dataset.hdf5"
EPS = 0.02
R = 24
HMAX = 10
HS = [1, 2, 3, 5, 10]


def env_rollout(raw, sim0, aseq):
    raw.done = False; raw.timestep = 0
    raw.sim.set_state_from_flattened(np.asarray(sim0, dtype=np.float64))
    raw.sim.data.qacc_warmstart[:] = 0.0
    raw.sim.forward()
    states = []
    for h in range(len(aseq)):
        obs, _, done, _ = raw.step(np.asarray(aseq[h], dtype=np.float64))
        states.append(cf.extract_state(obs))
        if done:
            while len(states) < len(aseq):
                states.append(states[-1])
            break
    return np.stack(states)


def main():
    env = cf.get_env(SOURCE, domain="source")
    raw = env.env
    eps = cf.load_demo_episodes(DEMOS, max_episodes=40)
    succ = [x for x in eps if x["rewards"].sum() > 0]
    rng = np.random.RandomState(0)

    for label, off in [("free-space (early)", 5), ("contact/insertion", None)]:
        spread = np.zeros(HMAX); n = 0
        for si in range(4):
            ep = succ[si]; T = len(ep["action"])
            t0 = off if off is not None and off < T else max(0, T - 12)
            a_star = ep["action"][t0].astype(np.float32)
            sim0 = ep["sim_state"][t0]
            # R rollouts of the SAME action + tiny noise; measure spread of s_H
            outs = []
            for _ in range(R):
                aseq = np.clip(a_star[None] + EPS * rng.randn(HMAX, 7).astype(np.float32), -1, 1)
                outs.append(env_rollout(raw, sim0, aseq))  # [HMAX,37]
            outs = np.stack(outs)  # [R,HMAX,37]
            # spread = mean over dims of std across the R rollouts, per horizon
            spread += np.linalg.norm(outs.std(0), axis=1)  # [HMAX]
            n += 1
        spread /= n
        print(f"\n[{label}]  TRUE-env divergence (std over {R} rollouts, action noise {EPS}):")
        print("   " + "".join(f"  H={h:<2d}" for h in HS))
        print("   " + "".join(f"  {spread[h-1]:.3f}" for h in HS))
    print("\n(compare to the WM error table: if TRUE-env spread ~ WM error, the")
    print(" wall is physical chaos in contact, not missing training data.)")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

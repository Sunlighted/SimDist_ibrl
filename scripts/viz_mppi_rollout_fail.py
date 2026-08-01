"""From the analyzed insertion state, hand control to value-MPPI in the REAL
source env and render HOW it fails — next to the source policy (which succeeds)
from the SAME start state. Top row = source policy (success), bottom row =
value-MPPI (failure). Plus needle-ring distance vs time.
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.eval_simdist_policy_only as e
import classifier_test.cf_utils as cf
import pyrallis

SIMDIST = "exps/mimicgen/threading_simdist_polexpert_seed1/frozen_cmp.pt"
SOURCE = "exps/mimicgen/threading_ibrl_source_seed1/model0.pt"
DEMOS = "release/data/mimicgen/threading_d0_source_sigma0.1/dataset.hdf5"
STEPS = 85
CAM = "sideview"
RES = 256


def load_source_agent(env):
    import train_rl
    cfg = pyrallis.load(train_rl.MainConfig, open(os.path.join(os.path.dirname(SOURCE), "cfg.yaml")))
    return train_rl.load_frozen_agent(SOURCE, env, cfg, "cuda")


def setup_metrics(raw):
    nid = raw.sim.model.geom_name2id("needle_obj_needle")
    ring = [raw.sim.model.geom_name2id(n) for n in raw.sim.model.geom_names
            if "tripod_obj_ring" in n and "vis" not in n]
    return nid, ring


def needle_ring(raw, nid, ring):
    npos = raw.sim.data.geom_xpos[nid].copy()
    rc = raw.sim.data.geom_xpos[ring].mean(0)
    return npos, rc, float(np.linalg.norm(npos - rc))


def render(raw):
    img = raw.sim.render(width=RES, height=RES, camera_name=CAM, depth=False)
    return img[::-1]  # mujoco renders upside down


def full_rollout(env, raw, controller, seed, nid, ring):
    """A full episode from env.reset() (same seed -> identical start), closed-loop.
    No set_state, so no cold-start contact artifact."""
    np.random.seed(seed)
    obs, _ = env.reset()
    frames, dists, nz = [render(raw)], [], []
    _, _, d0 = needle_ring(raw, nid, ring)
    dists.append(d0); nz.append(raw.sim.data.geom_xpos[nid][2])
    success = False
    for t in range(STEPS):
        a = controller(obs, t)
        obs, r, term, _, _ = env.step(a)
        frames.append(render(raw))
        _, _, d = needle_ring(raw, nid, ring)
        dists.append(d); nz.append(raw.sim.data.geom_xpos[nid][2])
        if r > 0.5:
            success = True
            break
        if term:
            break
    return frames, np.array(dists), np.array(nz), success


def main():
    agent, ws = e.build(SIMDIST, use_planner=1,
                        planner_overrides=dict(use_reward=0))  # value-only MPPI
    m = agent.model
    env = cf.get_env(SOURCE, domain="source")
    src = load_source_agent(env)
    raw = env.env
    nid, ring = setup_metrics(raw)

    def mppi_ctrl(obs, t):
        return agent.act(obs, eval_mode=True)

    def src_ctrl(obs, t):  # source RL policy closed-loop = success reference
        return src.act(obs, eval_mode=True)

    # same seed -> identical start; find one where source succeeds + MPPI fails
    SEED = int(os.environ.get("VIZ_SEED", "2"))
    f_src, d_src, z_src, ok_src = full_rollout(env, raw, src_ctrl, SEED, nid, ring)
    f_mp, d_mp, z_mp, ok_mp = full_rollout(env, raw, mppi_ctrl, SEED, nid, ring)
    print(f"source policy: success={ok_src}, min needle-ring dist={d_src.min():.4f}")
    print(f"value-MPPI:    success={ok_mp}, min needle-ring dist={d_mp.min():.4f}")

    # pick key frames
    def pick(frames, k=6):
        idx = np.linspace(0, len(frames) - 1, k).astype(int)
        return idx

    ks, km = pick(f_src), pick(f_mp)
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.9])
    for j, i in enumerate(ks):
        ax = fig.add_subplot(gs[0, j]); ax.imshow(f_src[i]); ax.axis("off")
        ax.set_title(f"source policy  t={i}", fontsize=9)
    for j, i in enumerate(km):
        ax = fig.add_subplot(gs[1, j]); ax.imshow(f_mp[i]); ax.axis("off")
        ax.set_title(f"value-MPPI  t={i}", fontsize=9)
    # metrics
    axd = fig.add_subplot(gs[2, 0:3])
    axd.plot(d_src, "g.-", label=f"source policy (success={ok_src})")
    axd.plot(d_mp, "r.-", label=f"value-MPPI (success={ok_mp})")
    axd.axhline(0.012, ls="--", c="k", lw=1, label="success thresh 12mm")
    axd.set_xlabel("step"); axd.set_ylabel("needle-ring dist (m)")
    axd.set_title("needle-ring distance"); axd.legend(fontsize=8)
    axz = fig.add_subplot(gs[2, 3:6])
    axz.plot(z_src, "g.-", label="source policy")
    axz.plot(z_mp, "r.-", label="value-MPPI")
    axz.set_xlabel("step"); axz.set_ylabel("needle height z (m)")
    axz.set_title("needle height (does it ram down/overshoot?)"); axz.legend(fontsize=8)
    fig.suptitle(f"Same start (seed={SEED}): source policy threads it (success={ok_src}); "
                 f"value-MPPI fails (success={ok_mp})", fontsize=13)
    fig.tight_layout()
    out = "scripts/viz_mppi_rollout_fail.png"
    fig.savefig(out, dpi=105)
    print(f"saved {out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

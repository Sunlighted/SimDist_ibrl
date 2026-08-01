"""Sanity-check the suspiciously-high TARGET latent-MPPI success: render rollouts
in the TARGET env + log the STRICT success flag, the needle-ring distance, and
the RING displacement (the reward-hack tell — if success comes from shoving the
ring toward the needle, ring_disp ~ initial needle-ring distance)."""
import os
os.environ["MUJOCO_GL"] = "egl"
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.eval_simdist_policy_only as ev
from env.robosuite_wrapper import PixelRobosuite

WEIGHT = "exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1/latest.pt"
N_EP = 6
STEPS = 160
CAM = "sideview"
RES = 256


def main():
    agent, ws = ev.build(WEIGHT, use_planner=1)
    agent.cfg.use_planner = 1
    params = dict(ws.eval_env_params)
    params["hand_init_perturb_joint_sigma"] = 0.1
    params["device"] = "cuda"
    env = PixelRobosuite(**params)
    raw = env.env
    nid = raw.sim.model.geom_name2id("needle_obj_needle")
    ring = [raw.sim.model.geom_name2id(n) for n in raw.sim.model.geom_names
            if "tripod_obj_ring" in n and "vis" not in n]

    def ring_center():
        return raw.sim.data.geom_xpos[ring].mean(0).copy()

    rows = []
    fig, axes = plt.subplots(N_EP, 6, figsize=(15, 2.5 * N_EP))
    for ep in range(N_EP):
        np.random.seed(100 + ep)
        obs, _ = env.reset()
        ring0 = ring_center()
        frames = [raw.sim.render(width=RES, height=RES, camera_name=CAM)[::-1]]
        succ = False
        min_d = 9.0
        max_ring_disp = 0.0
        for t in range(STEPS):
            a = agent.act(obs, eval_mode=True)
            obs, r, term, success, _ = env.step(a)
            frames.append(raw.sim.render(width=RES, height=RES, camera_name=CAM)[::-1])
            d = float(np.linalg.norm(raw.sim.data.geom_xpos[nid] - ring_center()))
            min_d = min(min_d, d)
            max_ring_disp = max(max_ring_disp, float(np.linalg.norm(ring_center() - ring0)))
            if success:
                succ = True
            if term:
                break
        rows.append((ep, succ, min_d, max_ring_disp, len(frames)))
        idx = np.linspace(0, len(frames) - 1, 6).astype(int)
        for j, fi in enumerate(idx):
            axes[ep, j].imshow(frames[fi]); axes[ep, j].axis("off")
            if j == 0:
                axes[ep, j].set_title(f"ep{ep} succ={succ} minD={min_d:.3f} ringΔ={max_ring_disp:.3f}",
                                      fontsize=8, loc="left")
        print(f"ep{ep}: strict_success={succ}  needle-ring min={min_d:.4f}  "
              f"max ring displacement={max_ring_disp:.4f}  (len={len(frames)})")
    fig.suptitle("TARGET latent-MPPI rollouts (latest 690k): real threading or ring-shove hack?", fontsize=12)
    fig.tight_layout()
    fig.savefig("scripts/viz_target_mppi.png", dpi=100)
    ns = sum(r[1] for r in rows)
    print(f"\nstrict success {ns}/{N_EP} | mean ring displacement on success: "
          f"{np.mean([r[3] for r in rows if r[1]]) if ns else float('nan'):.4f}")
    print("saved scripts/viz_target_mppi.png")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

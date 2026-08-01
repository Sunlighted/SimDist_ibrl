"""Visualize 20 TARGET rollouts of an MPPI-ADAPTED SimDist model, laid out as a
grid of success/failure episodes.

The adapted ckpt dirs store an AdaptConfig cfg.yaml (not a MainConfig), so we
build the agent from the STAGE-1 MainConfig, load the adapted weights on top,
and replay the SAME planner recipe the adapt run used (original_mppi=1 => the
broad SimDist recipe via apply_original_recipe). Eval env = TARGET (§12) with
hand_init sigma 0.1, matching the adapt eval.

Output: /home/Chenyang/ibrl/viz_adapt_mppi_20rollouts.png
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
import scripts.eval_simdist_policy_only as ev
from rl.simdist_planner import apply_original_recipe, MppiPlanner
from env.robosuite_wrapper import PixelRobosuite

# --- the MPPI-adapted model to visualize ---
STAGE1 = "exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1/latest.pt"
ADAPTED = os.environ.get("ADAPTED", "exps/mimicgen/threading_simdist_adapt_seed1/model0.pt")
ORIGINAL_MPPI = int(os.environ.get("ORIGINAL_MPPI", "1"))  # adapt_seed1 used 1
N_EP = 20
N_KEYFRAME = 6
MAX_STEPS = 400
RENDER_STRIDE = 6      # render every k steps (subsample for display), bounds cost
CAM = "sideview"
RES = 256
OUT = os.environ.get("OUT", "viz_adapt_mppi_20rollouts.png")


def main():
    # build from the STAGE-1 MainConfig, then load the ADAPTED weights on top
    agent, ws = ev.build(STAGE1, use_planner=1)
    agent.load_state_dict(torch.load(ADAPTED))
    agent = agent.to("cuda")
    agent.cfg.use_planner = 1
    if ORIGINAL_MPPI:
        apply_original_recipe(agent.planner.cfg)
        agent.planner = MppiPlanner(agent.model, agent.planner.cfg, agent.action_dim, device="cuda")
        p = agent.planner.cfg
        print(f"[planner] ORIGINAL recipe: samples={p.num_samples} base={p.num_base_override} "
              f"init_std={p.init_std} iters={p.iterations} temp={p.temperature} H={p.horizon}")

    # TARGET env, sigma 0.1 (matches adapt eval)
    params = dict(ws.eval_env_params)
    params["hand_init_perturb_joint_sigma"] = 0.1
    params["device"] = "cuda"
    env = PixelRobosuite(**params)
    raw = env.env
    nid = raw.sim.model.geom_name2id("needle_obj_needle")
    ring = [raw.sim.model.geom_name2id(n) for n in raw.sim.model.geom_names
            if "tripod_obj_ring" in n and "vis" not in n]

    def render():
        return raw.sim.render(width=RES, height=RES, camera_name=CAM)[::-1]

    def ring_center():
        return raw.sim.data.geom_xpos[ring].mean(0).copy()

    episodes = []  # (ep, success, min_d, ring_disp, n_steps, [frames])
    for ep in range(N_EP):
        np.random.seed(100 + ep)
        obs, _ = env.reset()
        ring0 = ring_center()
        frames = [render()]
        succ, min_d, max_ring_disp = False, 9.0, 0.0
        t = 0
        for t in range(MAX_STEPS):
            a = agent.act(obs, eval_mode=True)
            obs, r, term, success, _ = env.step(a)
            if (t + 1) % RENDER_STRIDE == 0:
                frames.append(render())
            d = float(np.linalg.norm(raw.sim.data.geom_xpos[nid] - ring_center()))
            min_d = min(min_d, d)
            max_ring_disp = max(max_ring_disp, float(np.linalg.norm(ring_center() - ring0)))
            succ = succ or bool(success)
            if term:
                break
        frames.append(render())  # always keep the final frame
        episodes.append((ep, succ, min_d, max_ring_disp, t + 1, frames))
        print(f"ep{ep:2d}: success={succ}  min needle-ring={min_d:.4f}  "
              f"ring_disp={max_ring_disp:.4f}  steps={t + 1}", flush=True)

    # sort: successes first, then failures (each by min distance)
    episodes.sort(key=lambda e: (not e[1], e[2]))
    n_succ = sum(e[1] for e in episodes)

    fig, axes = plt.subplots(N_EP, N_KEYFRAME, figsize=(2.3 * N_KEYFRAME, 2.15 * N_EP))
    for row, (ep, succ, min_d, rdisp, nsteps, frames) in enumerate(episodes):
        idx = np.linspace(0, len(frames) - 1, N_KEYFRAME).astype(int)
        col = "#1a7f37" if succ else "#cf222e"
        for j, fi in enumerate(idx):
            ax = axes[row, j]
            ax.imshow(frames[fi])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(col); s.set_linewidth(2.5)
            if j == 0:
                ax.set_ylabel(
                    f"ep{ep}\n{'SUCCESS' if succ else 'FAIL'}\nminD={min_d*1000:.1f}mm\n"
                    f"ringΔ={rdisp*1000:.1f}mm\n{nsteps} steps",
                    fontsize=7.5, color=col, rotation=0, ha="right", va="center", labelpad=32)
    fig.suptitle(
        f"MPPI-adapted SimDist — 20 TARGET rollouts (σ=0.1)  |  strict success {n_succ}/{N_EP}\n"
        f"{ADAPTED}  (original_mppi={ORIGINAL_MPPI}, broad search)  "
        f"[green=success, red=fail; minD=needle-ring closest approach, ringΔ=ring shove]",
        fontsize=11)
    fig.tight_layout(rect=[0.03, 0, 1, 0.985])
    fig.savefig(OUT, dpi=95, bbox_inches="tight")
    print(f"\nstrict success {n_succ}/{N_EP}")
    print(f"saved {os.path.abspath(OUT)}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

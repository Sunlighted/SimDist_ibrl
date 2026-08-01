"""Failure-mode analysis of the BEST config: 1000-demo BC base + narrow MPPI on
the adapted world model (target score ~0.88). Runs N episodes single-process on
TARGET (sigma 0.1), classifies each FAILURE with diag_failure_modes' buckets
(grasp/transport vs align/insert), and renders keyframes of every failed
episode into a grid.

Buckets (from scripts/diag_failure_modes.py):
  never_picked_up  : needle never lifted           (grasp failure)
  lost_before_ring : lifted but never reached ring (transport failure)
  near_miss        : reached ring vicinity, never inside 12 mm (align failure)
  strict_rejected  : tip entered the ball but strict gate rejected / slipped out

Output: viz_mppi_bc1000_failures.png (repo root) + bucket histogram on stdout.
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
from scripts.diag_failure_modes import SimProbe, classify, LIFT_THRESH
from env.robosuite_wrapper import PixelRobosuite
import common_utils

STAGE1 = "exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1/latest.pt"
ADAPTED = os.environ.get(
    "ADAPTED", "exps/mimicgen/threading_simdist_adapt_ibrlmppi_400k_bc1000_seed1/model.pt")
N_GAME = int(os.environ.get("N_GAME", "120"))
SEED = 1
SIGMA = 0.1
RENDER_STRIDE = 6
N_KEYFRAME = 6
CAM = "sideview"
RES = 256
OUT = "viz_mppi_bc1000_failures.png"


def main():
    common_utils.set_all_seeds(SEED)
    # narrow MPPI = the stage-1 default planner (mixture 0.8 / init_std 0.5)
    agent, ws = ev.build(STAGE1, use_planner=1)
    agent.load_state_dict(torch.load(ADAPTED))
    agent = agent.to("cuda")
    agent.cfg.use_planner = 1
    p = agent.planner.cfg
    print(f"[planner] narrow: samples={p.num_samples} mixture={p.mixture_coef} "
          f"init_std={p.init_std} iters={p.iterations} H={p.horizon}")

    params = dict(ws.eval_env_params)
    params["hand_init_perturb_joint_sigma"] = SIGMA
    params["device"] = "cuda"
    env = PixelRobosuite(**params)
    probe = SimProbe(env)
    raw = env.env

    def render():
        return raw.sim.render(width=RES, height=RES, camera_name=CAM)[::-1]

    buckets = {"success": 0, "never_picked_up": 0, "lost_before_ring": 0,
               "near_miss": 0, "strict_rejected": 0}
    failures = []  # (game, bucket, ep-dict, frames)

    with torch.no_grad(), common_utils.ibrl_utils.eval_mode(agent):
        for g in range(N_GAME):
            np.random.seed(SEED + g)
            obs, _ = env.reset()
            probe.refresh()
            z0 = probe.needle_pos()[2]
            max_z, min_d_nr = z0, probe.dist_needle_ring()
            grasp_hist, frames = [], [render()]
            terminal, success, t = False, False, 0
            while not terminal:
                a = agent.act(obs, eval_mode=True)
                obs, _r, terminal, success, _ = env.step(a)
                t += 1
                if t % RENDER_STRIDE == 0:
                    frames.append(render())
                npos = probe.needle_pos()
                max_z = max(max_z, npos[2])
                min_d_nr = min(min_d_nr, probe.dist_needle_ring())
                grasp_hist.append(probe.grasped())
            frames.append(render())
            ep = {"success": bool(success),
                  "ever_lifted": (max_z - z0) > LIFT_THRESH,
                  "min_d_nr": min_d_nr,
                  "max_lift": max_z - z0,
                  "ever_grasped": bool(any(grasp_hist)),
                  "grasp_frac": float(np.mean(grasp_hist)) if grasp_hist else 0.0,
                  "len": t}
            bucket = classify(ep, probe.radius)
            buckets[bucket] += 1
            mark = "" if success else "  <-- FAIL"
            print(f"game {g:3d}: {bucket:18s} lift={ep['max_lift']*100:+5.1f}cm "
                  f"min|n-ring|={min_d_nr*1000:6.1f}mm grasped={'Y' if ep['ever_grasped'] else 'n'}"
                  f"({ep['grasp_frac']:.2f}) len={t}{mark}", flush=True)
            if not success:
                failures.append((g, bucket, ep, frames))
            del frames

    n_fail = len(failures)
    print("\n================ bucket histogram ================")
    for k, v in buckets.items():
        print(f"  {k:18s}: {v:3d}  ({v/N_GAME*100:.1f}%)")
    grasp_phase = buckets["never_picked_up"] + buckets["lost_before_ring"]
    insert_phase = buckets["near_miss"] + buckets["strict_rejected"]
    if n_fail:
        print(f"\n  failures: {n_fail}/{N_GAME}  |  grasp/transport {grasp_phase} "
              f"({grasp_phase/n_fail*100:.0f}%)  vs  align/insert {insert_phase} "
              f"({insert_phase/n_fail*100:.0f}%)")

    # ---- render failure grid ----
    if n_fail:
        fig, axes = plt.subplots(n_fail, N_KEYFRAME,
                                 figsize=(2.3 * N_KEYFRAME, 2.15 * n_fail), squeeze=False)
        for row, (g, bucket, ep, frames) in enumerate(failures):
            idx = np.linspace(0, len(frames) - 1, N_KEYFRAME).astype(int)
            for j, fi in enumerate(idx):
                ax = axes[row, j]
                ax.imshow(frames[fi])
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_edgecolor("#cf222e"); s.set_linewidth(2.0)
                if j == 0:
                    ax.set_ylabel(
                        f"game{g}\n{bucket}\nminD={ep['min_d_nr']*1000:.1f}mm\n"
                        f"lift={ep['max_lift']*100:+.1f}cm\ngrasp={ep['grasp_frac']:.2f}\n"
                        f"{ep['len']} steps",
                        fontsize=7.5, color="#cf222e", rotation=0,
                        ha="right", va="center", labelpad=36)
        fig.suptitle(
            f"bc1000 + narrow MPPI — {n_fail} FAILURES out of {N_GAME} TARGET episodes "
            f"(success {buckets['success']/N_GAME*100:.1f}%)\n"
            f"grasp/transport {grasp_phase} vs align/insert {insert_phase}   |   {ADAPTED}",
            fontsize=11)
        fig.tight_layout(rect=[0.04, 0, 1, 0.97])
        fig.savefig(OUT, dpi=95, bbox_inches="tight")
        print(f"saved {os.path.abspath(OUT)}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

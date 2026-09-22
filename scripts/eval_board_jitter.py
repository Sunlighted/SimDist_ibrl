"""Does the source expert survive board jitter it never saw in training?

This decides SEQUENCING, not just a knob. The expert scored 0.95 with a FIXED
board, so the 14 board dims of object-state (board_pos/quat,
board_to_robot0_eef_*) were CONSTANT for its whole training -- it may never have
learned to read them. If it collapses under jitter, adding board randomisation
means RETRAINING the expert, then regenerating the 173 GB replay, then
restarting the distill. If it holds, only the last two are needed.

Uses train_rl.load_model + evaluate.run_eval so env_params travel with the
checkpoint (iron rule #17: sigma / init_dist / strict must not silently change
between arms).
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
for q in ("deps/mimicgen", "deps/robomimic", "."):
    sys.path.insert(0, os.path.join("/project/6101830/yifanr/SimDist_ibrl", q))
import numpy as np
import env.peg_insertion  # noqa
import train_rl
from evaluate.eval import run_eval

ap = argparse.ArgumentParser()
ap.add_argument("--weight", default="exps/mimicgen/peg_ibrl_source_v2_seed1/model0.pt")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--seed", type=int, default=99000)
a = ap.parse_args()

agent, _, env_params = train_rl.load_model(a.weight, "cuda")
print(f"[env] from ckpt: episode_length={env_params.get('episode_length')} "
      f"sigma={env_params.get('hand_init_perturb_joint_sigma')} "
      f"use_state={env_params.get('use_state')}")

LEVELS = [(0.000, 0.0), (0.005, 2.0), (0.010, 5.0), (0.020, 10.0)]
print(f"\n{'pos(m)':>8s} {'yaw(deg)':>9s} {'success':>8s}   (n={a.n}, same seeds per row)")
base = None
for pn, yn in LEVELS:
    ep = dict(env_params)
    ep["env_kwargs"] = dict(board_pos_noise=pn, board_yaw_noise=yn)
    scores = run_eval(ep, agent, a.n, a.seed, verbose=False)
    s = float(np.mean(scores))
    if base is None:
        base = s
    print(f"{pn:8.3f} {yn:9.1f} {s:8.3f}   delta vs fixed {s-base:+.3f}")

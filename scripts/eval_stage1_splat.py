"""Re-evaluate a stage-1 checkpoint with the obs distribution it was TRAINED on.

The in-training eval built its env from the config, which had no splat field, so
a photoreal-trained encoder was scored on plain MuJoCo frames -> 0.000 on source
AND target. This re-runs the same checkpoint against BOTH obs distributions so
the mismatch is measured, not assumed:
    splat  = what it trained on   -> the real number
    mujoco = what the run reported -> should reproduce ~0.000
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
for q in ("deps/mimicgen", "deps/robomimic", "."):
    sys.path.insert(0, os.path.join("/project/6101830/yifanr/SimDist_ibrl", q))
import numpy as np, torch
import env.peg_insertion  # noqa
import train_simdist
from env.robosuite_wrapper import PixelRobosuite
from real2sim.splat_obs import (DEFAULT_CALIB, DEFAULT_SEG_DIR,
                                DEFAULT_SCENE_PLY, DEFAULT_GROUND)

ap = argparse.ArgumentParser()
ap.add_argument("--weight", default="exps/mimicgen/peg_pixel_stage1_splat_seed1/model0.pt")
ap.add_argument("--n", type=int, default=50)
a = ap.parse_args()

agent, _, env_params = train_simdist.load_model(a.weight, "cuda")
SPLAT = dict(calib_path=DEFAULT_CALIB, seg_dir=DEFAULT_SEG_DIR,
             scene_ply=DEFAULT_SCENE_PLY, ground_json=DEFAULT_GROUND)

for tag, so in [("SPLAT (what it trained on)", SPLAT), ("MuJoCo (what the run reported)", None)]:
    ep = dict(env_params); ep["splat_obs"] = so
    e = PixelRobosuite(**ep)
    wins = 0
    for k in range(a.n):
        np.random.seed(7000 + k)
        obs = e.reset(); obs = obs[0] if isinstance(obs, tuple) else obs
        for _ in range(int(ep.get("episode_length", 300))):
            with torch.no_grad():
                act = agent.act(obs, eval_mode=True)
            out = e.step(act)
            obs, rew = out[0], out[1]
            if isinstance(obs, tuple): obs = obs[0]
            if rew > 0:
                wins += 1; break
            if len(out) > 2 and out[2]:
                break
    print(f"  {tag:34s} success {wins/a.n:.3f}  ({wins}/{a.n})")
    del e

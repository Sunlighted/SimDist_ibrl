"""Render the pre-insertion states the insert-phase diagnostic restores to.

For a handful of demos (source and/or target), find the same pre-insertion
frame `diag_contact_phases.py` uses (first grasped frame within APPROACH_DIST
of the ring), restore it into the TARGET-shifted env, and save a third-person
render so we can eyeball "what state did we reset to".

Usage:
  python scripts/viz_preinsertion_states.py \
      --demos release/data/mimicgen/threading_d0_source_sigma0.1/dataset.hdf5 \
      --tag source --max_demos 4 --out /tmp/preinsert_viz
"""

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import imageio.v2 as imageio

import common_utils  # noqa
from env.robosuite_wrapper import PixelRobosuite
from scripts.diag_contact_phases import (
    SimProbe, find_preinsertion_frames, restore_state, load_env_and_agent,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str,
                   default="exps/mimicgen/threading_flow_lowdata_target_seed1")
    p.add_argument("--weight", type=str, default="model0.pt")
    p.add_argument("--demos", type=str, required=True)
    p.add_argument("--tag", type=str, default="demo")
    p.add_argument("--max_demos", type=int, default=4)
    p.add_argument("--approach_dist", type=float, default=0.10,
                   help="needle-ring distance (m) of the restored state")
    p.add_argument("--cameras", type=str, default="agentview,sideview")
    p.add_argument("--out", type=str, default="/tmp/preinsert_viz")
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--skip_grasp", action="store_true", default=True)
    p.add_argument("--grasp_games", type=int, default=0)
    p.add_argument("--insert_per_state", type=int, default=1)
    return p.parse_args()


def render(env, cam, res):
    img = env.env.sim.render(camera_name=cam, width=res, height=res)
    return np.flipud(img).copy()  # mujoco offscreen is bottom-left origin


def main():
    args = parse_args()
    common_utils.set_all_seeds(args.seed)
    os.makedirs(args.out, exist_ok=True)
    cams = [c.strip() for c in args.cameras.split(",")]

    agent, env_params = load_env_and_agent(args)
    env = PixelRobosuite(**env_params)
    probe = SimProbe(env)

    frames = find_preinsertion_frames(env, probe, args.demos, args.max_demos,
                                      target_dist=args.approach_dist)

    for dk, sim_state, d0 in frames:
        if sim_state is None:
            continue
        restore_state(env, probe, sim_state)
        tiles = []
        for cam in cams:
            tiles.append(render(env, cam, args.res))
        grid = np.concatenate(tiles, axis=1)  # cameras side by side
        fname = os.path.join(args.out, f"{args.tag}_{dk}_d{d0*1000:.0f}mm.png")
        imageio.imwrite(fname, grid)
        print(f"  saved {fname}  (start dist {d0*1000:.1f}mm, "
              f"grasped={probe.grasped()})")

    print(f"\nDone. Images in {args.out}")


if __name__ == "__main__":
    main()

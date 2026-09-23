"""Record SUCCESSFUL rollouts of a real-cell agent as videos + a filmstrip.

Why not `evaluate/eval.py --record_dir`: that records whatever episodes it runs,
successes and failures mixed, one camera stream per file. Here we keep only the
successes, put agentview and the wrist side by side in one frame, and also drop
a filmstrip PNG so a trajectory can be read without playing anything.

  python scripts/record_realcell_rollouts.py \
      --weight exps/mimicgen/square_realcell_ibrl_seed1/model0.pt \
      --cell release/cfgs/sim2real/square_real_cell_calibrated_wrist.yaml \
      --out figs/sim2real_square/rollouts --num 6
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imageio
import numpy as np
import torch
from PIL import Image

import train_rl
from env.robosuite_wrapper import PixelRobosuite
from env.sim2real import load_sim2real

CAMS = ["agentview", "robot0_eye_in_hand"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weight", required=True)
    p.add_argument("--cell", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num", type=int, default=6, help="SUCCESSFUL episodes to keep")
    p.add_argument("--max_tries", type=int, default=30)
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--fps", type=int, default=20, help="= control_freq, so real time")
    p.add_argument("--strips", type=int, default=6, help="frames per filmstrip row")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    agent, _, env_params = train_rl.load_model(args.weight, "cuda")
    agent.eval()   # q_agent.act asserts `not self.training`
    cell = load_sim2real(args.cell)
    # The agent is STATE-based; the cameras are only for looking at it. Keeping
    # use_state on means the policy sees exactly what it was trained on.
    env = PixelRobosuite(
        env_params["env_name"], "Panda", env_params.get("episode_length", 400),
        image_size=args.res, rl_image_size=args.res, camera_names=CAMS, rl_cameras=CAMS,
        # flip_image=True is the COLLECTION/TRAINING convention (wrapper default);
        # recording with False silently shows an upside-down scene.
        use_state=True, obs_stack=1, state_stack=1, prop_stack=1, flip_image=True,
        device="cuda", hand_init_perturb_joint_sigma=args.sigma, sim2real=cell)

    kept, tries, lens = 0, 0, []
    while kept < args.num and tries < args.max_tries:
        np.random.seed(args.seed + tries)
        torch.manual_seed(args.seed + tries)
        obs, _ = env.reset()
        tries += 1
        frames, done = [], False
        def grab(o):
            return tuple(o[c].cpu().numpy().transpose(1, 2, 0).astype(np.uint8) for c in CAMS)
        while not done:
            frames.append(grab(obs))
            with torch.no_grad():
                act = agent.act(obs, eval_mode=True)
            obs, _, terminal, success, _ = env.step(act)
            done = bool(terminal)
        if not bool(success):
            continue
        frames.append(grab(obs))
        tag = f"ep{kept:02d}_len{len(frames)}"
        # video: the two cameras side by side, one file per episode
        imageio.mimsave(f"{args.out}/{tag}.mp4",
                        [np.concatenate(f, axis=1) for f in frames],
                        fps=args.fps, quality=8)
        # filmstrip: one COLUMN per sampled timestep, agentview over wrist --
        # side-by-side columns make the phases unreadable once several episodes
        # are stacked.
        idx = np.linspace(0, len(frames) - 1, args.strips).astype(int)
        Image.fromarray(np.concatenate(
            [np.concatenate(frames[i], axis=0) for i in idx], axis=1)) \
            .save(f"{args.out}/{tag}_strip.png")
        lens.append(len(frames))
        kept += 1
        print(f"[rec] kept {tag}", flush=True)

    print(f"[rec] {kept} successes out of {tries} episodes, "
          f"mean length {np.mean(lens):.0f} steps -> {args.out}")
    strips = sorted(f for f in os.listdir(args.out) if f.endswith("_strip.png"))
    if strips:
        rows = [np.asarray(Image.open(os.path.join(args.out, f))) for f in strips]
        w = min(r.shape[1] for r in rows)
        Image.fromarray(np.concatenate([r[:, :w] for r in rows], axis=0)) \
            .save(f"{args.out}/all_strips.png")
        print(f"[rec] wrote {args.out}/all_strips.png")


main()

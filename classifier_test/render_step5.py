"""Render the arm motion of step-5 rollouts, with per-frame GT gap / disc score.

Reproduces a step5 trajectory (same seed = base_seed + traj_index), records the
agentview camera, and overlays the ground-truth gap ||s'_tgt - s'_src|| and the
discriminator score at each step so the video lines up with the step5 curves.
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw
import imageio

from cf_utils import (REPO, TARGET_SHIFT, counterfactual_step, get_sim_state)
from data_utils import state_gap, needle_ring_dist
from discriminator import DynamicsDiscriminator
from common_utils import ibrl_utils as utils


def load_disc(path, device):
    ck = torch.load(path, map_location=device)
    d = DynamicsDiscriminator(ck["state_dim"], ck["action_dim"], ck["hidden"], ck["dropout"]).to(device)
    d.load_state_dict(ck["state_dict"]); d.eval()
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--behavior_weight",
                   default=f"{REPO}/exps/mimicgen/threading_target_only_flow_seed1/model0.pt")
    p.add_argument("--disc", default=f"{os.path.dirname(__file__)}/discriminator_cf.pt")
    p.add_argument("--base_seed", type=int, default=100)
    p.add_argument("--trajs", type=str, default="3,4", help="traj indices (seed=base+idx)")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--out_dir", default=os.path.dirname(__file__))
    args = p.parse_args()
    device = "cuda"

    import train_bc
    from env.robosuite_wrapper import PixelRobosuite
    from env.domain_shift import DomainShiftConfig

    agent, _, env_params = train_bc.load_model(args.behavior_weight, device)

    src_params = dict(env_params)
    src_params.update(domain_shift=DomainShiftConfig(mode="none"),
                      hand_init_perturb=None, hand_init_perturb_joint_sigma=args.sigma,
                      camera_names=["agentview"], rl_cameras=["agentview"], image_size=256)
    src_env = PixelRobosuite(**src_params)

    tgt_params = dict(env_params)
    tgt_params["domain_shift"] = DomainShiftConfig(**TARGET_SHIFT)
    tgt_env = PixelRobosuite(**tgt_params)
    tgt_env.reset()

    d = load_disc(args.disc, device)
    pol_tag = os.path.basename(os.path.dirname(args.behavior_weight))

    for idx in [int(x) for x in args.trajs.split(",")]:
        seed = args.base_seed + idx
        np.random.seed(seed)
        obs, hi = src_env.reset()
        frames, gaps, scores = [], [], []
        terminal = False
        with torch.no_grad(), utils.eval_mode(agent):
            while not terminal:
                sim_t = get_sim_state(src_env)
                s_t = obs["state"].cpu().numpy()
                action = agent.act(obs, eval_mode=True)
                obs, reward, terminal, _, hi = src_env.step(action)
                s_src = obs["state"].cpu().numpy()
                s_tgt, _, _ = counterfactual_step(tgt_env, sim_t.astype(np.float64),
                                                  action.cpu().numpy())
                gap = float(state_gap(s_src[None], s_tgt[None])[0])
                score = float(abs(d.log_dynamics_ratio(
                    torch.as_tensor(s_t[None], device=device),
                    torch.as_tensor(action.cpu().numpy()[None], device=device),
                    torch.as_tensor(s_src[None], device=device)).item()))
                gaps.append(gap); scores.append(score)

                img = hi["agentview"].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                frames.append((img, len(gaps) - 1))

        gaps, scores = np.array(gaps), np.array(scores)
        gmax, smax = gaps.max() + 1e-9, scores.max() + 1e-9
        out_frames = []
        for img, t in frames:
            im = Image.fromarray(img); dr = ImageDraw.Draw(im)
            dr.text((4, 2), f"{pol_tag}", fill=(255, 255, 0))
            dr.text((4, 14), f"seed{seed} t={t}", fill=(255, 255, 255))
            dr.text((4, 26), f"GT gap={gaps[t]:.4f}", fill=(255, 80, 80))
            dr.text((4, 38), f"disc ={scores[t]:.3f}", fill=(120, 160, 255))
            # bars
            dr.rectangle([4, 52, 4 + int(120 * gaps[t] / gmax), 58], fill=(255, 80, 80))
            dr.rectangle([4, 60, 4 + int(120 * scores[t] / smax), 66], fill=(120, 160, 255))
            out_frames.append(np.asarray(im))

        out = os.path.join(args.out_dir, f"step5_{pol_tag}_traj{idx}_seed{seed}.mp4")
        imageio.mimsave(out, out_frames, fps=args.fps)
        print(f"traj{idx} seed{seed}: len={len(out_frames)} "
              f"GT gap[max={gaps.max():.4f} mean={gaps.mean():.4f}] -> {out}")


if __name__ == "__main__":
    main()

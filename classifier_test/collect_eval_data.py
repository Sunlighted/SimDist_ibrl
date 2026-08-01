"""Collect (s, a, s') transitions by rolling out a trained RL agent in the
source (no shift) and target (dynamics-only shift) envs.

The transitions are saved to an .npz so eval_classifier.py can score how well
the trained discriminator separates source- from target-domain transitions on
ON-POLICY data (the discriminator was trained on DEMO data, so this also tests
transfer across the policy-induced state-action distribution shift).

By default the SAME agent is used in both envs so the only difference between
the two transition sets is the env dynamics (the cleanest dynamics-gap test).
"""

import argparse
import os
import sys

import numpy as np
import torch

REPO = "/home/Chenyang/ibrl"
sys.path.insert(0, REPO)
# load_model resolves the ckpt's cfg.yaml via a path relative to the repo root.
os.chdir(REPO)

os.environ.setdefault("MUJOCO_GL", "egl")

import train_rl  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from env.domain_shift import DomainShiftConfig  # noqa: E402
from common_utils import ibrl_utils as utils  # noqa: E402

# Dynamics-only target shift (matches threading_d0_target_dynamics/env_cfg.json
# and the "new target" in CLAUDE.md §12).
TARGET_SHIFT = dict(
    mode="fixed", friction_scale=0.5, mass_scale=3.0, solref_scale=1.0,
    solimp_scale=1.0, gripper_kp_scale=0.12, hole_offset=(0.0, 0.0, 0.0),
    mask_hole_offset_in_obs=False, ring_size_scale=0.85, ring_friction_scale=2.0,
)


def rollout(env_params, agent, num_game, seed, state_key="state"):
    s_list, a_list, ns_list = [], [], []
    n_success = 0
    env = PixelRobosuite(**env_params)
    with torch.no_grad(), utils.eval_mode(agent):
        for ep in range(num_game):
            np.random.seed(seed + ep)
            obs, _ = env.reset()
            prev_state = obs[state_key].clone()
            terminal = False
            ep_reward = 0.0
            while not terminal:
                action = agent.act(obs, eval_mode=True)
                obs, reward, terminal, _, _ = env.step(action)
                cur_state = obs[state_key]
                s_list.append(prev_state.cpu().numpy())
                a_list.append(action.cpu().numpy())
                ns_list.append(cur_state.cpu().numpy())
                prev_state = cur_state.clone()
                ep_reward += float(reward)
            n_success += int(ep_reward > 0)
    print(f"    collected {len(s_list)} transitions over {num_game} eps, "
          f"success {n_success}/{num_game}")
    return (np.stack(s_list), np.stack(a_list), np.stack(ns_list))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weight", default=f"{REPO}/exps/mimicgen/threading_ibrl_newtgt_seed1/model0.pt")
    p.add_argument("--num_game", type=int, default=30)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--state_key", default="state")
    p.add_argument("--out", default=f"{os.path.dirname(__file__)}/eval_transitions.npz")
    args = p.parse_args()

    agent, _, env_params = train_rl.load_model(args.weight, "cuda")

    # source env: no shift
    src_params = dict(env_params)
    src_params["domain_shift"] = DomainShiftConfig(mode="none")
    print("[collect] SOURCE env (mode=none)")
    src = rollout(src_params, agent, args.num_game, args.seed, args.state_key)

    # target env: dynamics-only shift
    tgt_params = dict(env_params)
    tgt_params["domain_shift"] = DomainShiftConfig(**TARGET_SHIFT)
    print("[collect] TARGET env (dynamics-only shift)")
    tgt = rollout(tgt_params, agent, args.num_game, args.seed, args.state_key)

    np.savez(args.out,
             src_s=src[0], src_a=src[1], src_ns=src[2],
             tgt_s=tgt[0], tgt_a=tgt[1], tgt_ns=tgt[2],
             weight=args.weight)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

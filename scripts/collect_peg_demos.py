"""Collect scripted-expert demos for PegInsertion_D0 (real2sim Phase C).

The scripted expert lives in scripts/smoke_peg_insertion.py (align tip ->
latched gentle insert; 10/10 on the smoke suite). Output format mirrors
scripts/collect_target_demos.py exactly (data/demo_i/{obs,actions,rewards} +
env_cfg.json sidecar), so train_bc.py, train_rl.py preload and
add_demos_to_replay consume it unchanged.

    python scripts/collect_peg_demos.py --num_demos 1000 \
        --out_dir release/data/peg_insertion/demos_n1000 --seed 0

Storage note: rendering is ON by default (agentview + robot0_eye_in_hand @96)
so the same dataset can serve a future pixel arm; --state_only 1 skips all
rendering (state pipeline only, runs on CPU nodes).
"""

import argparse
import dataclasses
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, ".")

from env.robosuite_wrapper import PixelRobosuite
from scripts.smoke_peg_insertion import scripted_expert_action


def rollout_once(env, seed, max_len):
    """One scripted episode. Seeding mirrors collect_target_demos.rollout_once:
    np.random.seed(seed) right before reset() makes the seed BE the initial
    state (robot joint noise + in-hand peg jitter both draw from np.random)."""
    np.random.seed(seed)
    obs, _ = env.reset()

    ep_obs = defaultdict(list)
    actions, rewards = [], []
    ctl = {}
    terminal, succ, steps = False, False, 0
    while not terminal and steps < max_len:
        for k, v in obs.items():
            ep_obs[k].append(v.cpu().numpy())
        action = torch.from_numpy(scripted_expert_action(env.env, ctl))
        actions.append(action.cpu().numpy())
        obs, reward, terminal, succ, _ = env.step(action)
        rewards.append(float(reward))
        steps += 1

    ok = (bool(succ) or np.sum(rewards) > 0) and terminal and steps <= max_len
    return ok, ep_obs, actions, rewards, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_demos", type=int, default=1000)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=0.1,
                        help="hand_init_perturb_joint_sigma (project standard 0.1)")
    parser.add_argument("--episode_length", type=int, default=300)
    parser.add_argument("--state_only", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_attempt_factor", type=float, default=3.0)
    args = parser.parse_args()

    cameras = ["agentview", "robot0_eye_in_hand"]
    env = PixelRobosuite(
        env_name="PegInsertion_D0",
        robots="Panda",
        episode_length=args.episode_length,
        use_state=True,
        record_sim_state=True,
        state_only_no_render=bool(args.state_only),
        image_size=args.image_size,
        rl_image_size=args.image_size,
        camera_names=cameras,
        rl_cameras=cameras,
        device=args.device,
        hand_init_perturb_joint_sigma=args.sigma,
        end_on_success=True,
    )

    accepted = []
    attempt = 0
    max_attempts = int(args.num_demos * args.max_attempt_factor)
    while len(accepted) < args.num_demos and attempt < max_attempts:
        seed = args.seed + attempt
        ok, ep_obs, actions, rewards, steps = rollout_once(env, seed, args.episode_length)
        attempt += 1
        if ok:
            accepted.append(dict(obs=ep_obs, actions=actions, rewards=rewards, seed=seed))
        if attempt % 50 == 0 or len(accepted) == args.num_demos:
            print(f"[collect] attempts {attempt}, accepted {len(accepted)}"
                  f"/{args.num_demos} (last: ok={ok} steps={steps})", flush=True)

    rate = len(accepted) / max(attempt, 1)
    print(f"[collect] done: {len(accepted)} demos in {attempt} attempts "
          f"(scripted success rate {rate:.3f})")

    # ---- write dataset.hdf5 + env_cfg.json (collect_target_demos.py format) ----
    import h5py

    os.makedirs(args.out_dir, exist_ok=True)
    h5_path = os.path.join(args.out_dir, "dataset.hdf5")
    with h5py.File(h5_path, "w") as hf:
        grp = hf.create_group("data")
        grp.attrs["num_demos"] = len(accepted)
        for i, ep in enumerate(accepted):
            g = grp.create_group(f"demo_{i}")
            for k, arr_list in ep["obs"].items():
                g.create_dataset(f"obs/{k}", data=np.asarray(arr_list))
            g.create_dataset("actions", data=np.asarray(ep["actions"], dtype=np.float32))
            g.create_dataset("rewards", data=np.asarray(ep["rewards"], dtype=np.float32))
            g.attrs["num_samples"] = len(ep["actions"])
            g.attrs["seed"] = int(ep["seed"])

    env_cfg = {
        "env_name": "PegInsertion_D0",
        "domain_shift": dataclasses.asdict(env.domain_shift),
        "hand_init_perturb_joint_sigma": args.sigma,
        "init_distribution": None,
        "rl_image_size": None if args.state_only else args.image_size,
        "use_state": True,
        "source_weight": "scripted_expert (scripts/smoke_peg_insertion.py)",
        "collect_action_noise": None,
        "scripted_success_rate": rate,
    }
    with open(os.path.join(args.out_dir, "env_cfg.json"), "w") as f:
        json.dump(env_cfg, f, indent=2)
    print(f"[collect] wrote {h5_path}")


if __name__ == "__main__":
    main()

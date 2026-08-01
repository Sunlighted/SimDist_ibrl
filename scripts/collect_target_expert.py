"""Collect pure TARGET-domain expert success trajectories (multi-process).

Roll the TARGET RL expert (e.g. threading_ibrl_newtgt) in the TARGET sim and
keep ONLY strict-success episodes. No student, no V-gate takeover — every step
is the expert's own action. Output matches scripts/collect_target_recovery.py's
sidecar HDF5 layout (ep_type="expert", takeover_mask all-False) so
bc/dataset.py reads it directly. hand_init sigma defaults to 0.1.

Parallelism (mirrors scripts/generate_simdist_data.py): `--num_proc N` runs N
self-contained worker processes (each loads its own expert + a state-only,
NO-RENDER TARGET env and collects its share of successes into a shard hdf5);
the parent drives a single tqdm progress bar over a shared counter, then merges
the shards into `<out_dir>/dataset.hdf5`. state_only_no_render kills the per-
worker EGL context so many workers share one GPU safely.

Example:
    python scripts/collect_target_expert.py \
        --expert    exps/mimicgen/threading_ibrl_newtgt_seed1/model0.pt \
        --shift_cfg exps/mimicgen/threading_cotrain_unweighted_n1000_seed1/cfg.yaml \
        --out_dir   release/data/mimicgen/threading_d0_target_expert_n500 \
        --n_collect 500 --num_proc 8 --hand_init_perturb_joint_sigma 0.1 --seed 1
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import dataclasses
import gc
import json
import time
from collections import defaultdict

import numpy as np
import torch
import h5py
from tqdm import tqdm

import common_utils
from common_utils import ibrl_utils as utils
import train_rl
from env.robosuite_wrapper import PixelRobosuite
from env.strict_success import StrictSuccessConfig
# reuse the validated TARGET-shift parsing + sidecar key set + shard merge
from scripts.collect_target_recovery import load_target_shift
from scripts.collect_recovery_data import _NON_IMAGE_KEYS
from scripts.generate_simdist_data import merge_shards


def build_collect_env(base_env_params, shift, args):
    """State-only, NO-RENDER TARGET env from the policy's own env_params.

    state_only_no_render=True drops the offscreen EGL context (use_state mode),
    so >1 worker can share one GPU; record_sim_state=True still works (sim_state
    comes from env.sim.get_state(), independent of rendering)."""
    ep = dict(base_env_params)
    ep.update(
        domain_shift=shift,
        hand_init_perturb=None,
        hand_init_perturb_joint_sigma=args.hand_init_perturb_joint_sigma,
        camera_names=[],
        rl_cameras=[],
        record_sim_state=True,
        state_only_no_render=True,
    )
    if args.strict_success:
        ep["strict_success"] = StrictSuccessConfig(
            enabled=True,
            dwell_steps=args.strict_dwell_steps,
            max_tripod_disp=args.strict_max_tripod_disp,
            max_needle_speed=args.strict_max_needle_speed,
        )
    return PixelRobosuite(**ep), ep


def rollout_expert(env, expert):
    """One episode driven entirely by the expert. Records pre-act obs snapshots
    + EXECUTED actions (mirrors collect_recovery_data.rollout's bookkeeping)."""
    obs, _ = env.reset()
    ep_obs: dict = defaultdict(list)
    actions: list = []
    rewards: list = []
    terminal = False
    while not terminal:
        for k, v in obs.items():
            if k in _NON_IMAGE_KEYS and torch.is_tensor(v):
                ep_obs[k].append(v.detach().cpu().numpy())
        action = expert.act(obs, eval_mode=True)
        actions.append(action.detach().cpu().numpy())
        obs, reward, terminal, _, _ = env.step(action)
        rewards.append(float(reward))
    success = bool(np.sum(rewards) > 0)
    return {
        "obs": ep_obs,
        "actions": actions,
        "rewards": rewards,
        "takeover_mask": [False] * len(actions),
        "success": success,
        "took_over": False,
        "n_takeover_steps": 0,
        "steps": len(actions),
        "type": "expert" if success else "fail",
    }


def _write_demo(group, idx, ep):
    g = group.create_group(f"demo_{idx}")
    for k, arr_list in ep["obs"].items():
        if k not in _NON_IMAGE_KEYS:
            continue
        g.create_dataset(f"obs/{k}", data=np.asarray(arr_list))
    g.create_dataset("actions", data=np.asarray(ep["actions"], dtype=np.float32))
    g.create_dataset("rewards", data=np.asarray(ep["rewards"], dtype=np.float32))
    g.create_dataset("takeover_mask", data=np.asarray(ep["takeover_mask"], dtype=bool))
    g.attrs["num_samples"] = len(ep["actions"])
    g.attrs["ep_type"] = ep["type"]
    g.attrs["took_over"] = ep["took_over"]
    g.attrs["n_takeover_steps"] = ep["n_takeover_steps"]
    g.attrs["success"] = ep["success"]


def write_shard(records, shard_path):
    os.makedirs(os.path.dirname(shard_path) or ".", exist_ok=True)
    with h5py.File(shard_path, "w") as hf:
        g = hf.create_group("data")
        for j, ep in enumerate(records):
            _write_demo(g, j, ep)
        g.attrs["num_demos"] = len(records)


def write_env_cfg(out_dir, env_params, args, shift, total):
    env_cfg = {
        "env_name": env_params["env_name"],
        "domain_shift": dataclasses.asdict(shift),
        "hand_init_perturb_joint_sigma": args.hand_init_perturb_joint_sigma,
        "rl_image_size": env_params.get("rl_image_size"),
        "use_state": env_params.get("use_state"),
        "strict_success": bool(args.strict_success),
        "source": "target_expert",
        "composition": {"num_demos_total": total},
    }
    with open(os.path.join(out_dir, "env_cfg.json"), "w") as f:
        json.dump(env_cfg, f, indent=2)


def _worker(rank, args, shift, n_keep, shard_path, counter):
    os.environ["MUJOCO_GL"] = "egl"
    common_utils.set_all_seeds(args.seed * 100000 + rank)
    expert, throwaway_env, base = train_rl.load_model(args.expert, args.device)
    del throwaway_env
    gc.collect()
    if args.act_method is not None and hasattr(expert, "cfg"):
        expert.cfg.act_method = args.act_method
    env, _env_params = build_collect_env(base, shift, args)

    kept = []
    ep = 0
    with torch.no_grad(), utils.eval_mode(expert):
        while len(kept) < n_keep and ep < args.max_episodes_per_worker:
            np.random.seed(args.seed * 100000 + rank * 1000000 + ep)
            r = rollout_expert(env, expert)
            ep += 1
            if r["success"]:
                kept.append(r)
                with counter.get_lock():
                    counter.value += 1
    write_shard(kept, shard_path)
    print(f"[w{rank}] {len(kept)}/{n_keep} successes in {ep} rollouts "
          f"-> {shard_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--expert", type=str, required=True,
                   help="TARGET RL expert ckpt (rollout policy).")
    p.add_argument("--shift_cfg", type=str, required=True,
                   help="a cfg.yaml whose eval_domain_shift defines the TARGET.")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--n_collect", type=int, default=500,
                   help="total strict-success expert episodes to keep.")
    p.add_argument("--num_proc", type=int, default=8,
                   help="parallel worker processes (each self-contained).")
    p.add_argument("--max_episodes_per_worker", type=int, default=2000,
                   help="per-worker rollout cap while hunting for its share.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--hand_init_perturb_joint_sigma", type=float, default=0.1)
    p.add_argument("--act_method", type=str, default=None,
                   help="override expert act_method (rl/ibrl/...).")
    p.add_argument("--strict_success", type=int, default=1)
    p.add_argument("--strict_dwell_steps", type=int, default=3)
    p.add_argument("--strict_max_tripod_disp", type=float, default=0.04)
    p.add_argument("--strict_max_needle_speed", type=float, default=0.0)
    args = p.parse_args()

    shift = load_target_shift(args.shift_cfg)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "dataset.hdf5")
    print(f"expert={args.expert}")
    print(f"domain=TARGET (mode={shift.mode}, fric={shift.friction_scale}, "
          f"gkp={shift.gripper_kp_scale}, ring_sz={shift.ring_size_scale}, "
          f"ring_fric={shift.ring_friction_scale}); sigma={args.hand_init_perturb_joint_sigma}; "
          f"strict={bool(args.strict_success)}; n_collect={args.n_collect}; "
          f"num_proc={args.num_proc}")

    import torch.multiprocessing as mp
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    per = [args.n_collect // args.num_proc + (1 if i < args.n_collect % args.num_proc else 0)
           for i in range(args.num_proc)]
    shard_paths = [os.path.join(args.out_dir, f"_shard_{i}.hdf5") for i in range(args.num_proc)]
    counter = mp.Value("i", 0)
    procs = [mp.Process(target=_worker, args=(i, args, shift, per[i], shard_paths[i], counter))
             for i in range(args.num_proc) if per[i] > 0]
    for pr in procs:
        pr.start()
    with tqdm(total=args.n_collect, desc="expert", unit="demo", dynamic_ncols=True) as bar:
        while any(pr.is_alive() for pr in procs):
            bar.n = counter.value
            bar.refresh()
            time.sleep(0.5)
        bar.n = counter.value
        bar.refresh()
    for pr in procs:
        pr.join()
    failed = [pr.exitcode for pr in procs if pr.exitcode != 0]
    if failed:
        print(f"[error] {len(failed)} worker(s) exited non-zero: {failed}")

    total, _ = merge_shards(shard_paths, out_path)
    # sidecar (build env_params shell from the shift cfg, no env needed)
    import yaml
    rcfg = yaml.safe_load(open(args.shift_cfg))
    env_params = {"env_name": rcfg.get("task_name", "Threading_D0"),
                  "rl_image_size": rcfg.get("rl_image_size", 96),
                  "use_state": 1}
    write_env_cfg(args.out_dir, env_params, args, shift, total)
    for sp in shard_paths:
        if os.path.exists(sp):
            os.remove(sp)
    print(f"DONE: {total} expert demos -> {out_path}")
    os._exit(0)  # EGL teardown can hang


if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = "egl"
    main()

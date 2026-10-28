"""Collect scripted-expert demos on the Threading Robotiq real cell (TRAIN_PLAN.md stage 1, 2026-10-19).

Expert: scripts/threading_robotiq_expert.py (closed loop on the actual in-hand needle pose; 0.95 on 20 episodes).
Env: PixelRobosuite on the cell (10 Hz, execution layer, damped contact, layout ranges, physics + base DR), state only,
sim_state recorded. Output = collect_peg_demos.py's format (data/demo_i/{obs,actions,rewards} + env_cfg.json), which
train_bc.py and train_rl.py's preload read unchanged. Successes only.

    python scripts/collect_threading_robotiq_demos.py --num_demos 125 --shard 0 --out_dir DIR/shard_00
    python scripts/collect_threading_robotiq_demos.py --merge DIR/shard_* --out_dir DIR
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rollout(env, seed, max_len, noise):
    import torch
    from scripts.threading_robotiq_expert import ThreadingExpert
    np.random.seed(seed)
    obs, _ = env.reset()
    ex = ThreadingExpert(env.env, noise=noise, rng=np.random.default_rng(seed))
    ep_obs, actions, rewards = defaultdict(list), [], []
    succ, term, steps = False, False, 0
    while not term and steps < max_len:
        for k, v in obs.items():
            ep_obs[k].append(v.cpu().numpy())
        a, _ = ex.act()
        actions.append(a.astype(np.float32))
        obs, r, term, succ, _ = env.step(torch.from_numpy(a.astype(np.float32)))
        rewards.append(float(r))
        steps += 1
    ok = bool(succ) or float(np.sum(rewards)) > 0
    return ok, ep_obs, actions, rewards, steps


def write(out_dir, eps, meta):
    import h5py
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "dataset.hdf5")
    with h5py.File(path, "w") as hf:
        g = hf.create_group("data")
        g.attrs["num_demos"] = len(eps)
        for i, ep in enumerate(eps):
            d = g.create_group(f"demo_{i}")
            for k, arr in ep["obs"].items():
                d.create_dataset(f"obs/{k}", data=np.asarray(arr))
            d.create_dataset("actions", data=np.asarray(ep["actions"], dtype=np.float32))
            d.create_dataset("rewards", data=np.asarray(ep["rewards"], dtype=np.float32))
            d.attrs["num_samples"] = len(ep["actions"])
            d.attrs["seed"] = int(ep["seed"])
    with open(os.path.join(out_dir, "env_cfg.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return path


def merge(dirs, out_dir):
    import h5py
    eps, metas = [], []
    dirs = [d for d in dirs if os.path.isdir(d)]      # a shard_* glob also matches the shard_*.log files
    for d in sorted(dirs):
        metas.append(json.load(open(os.path.join(d, "env_cfg.json"))))
        with h5py.File(os.path.join(d, "dataset.hdf5"), "r") as hf:
            for k in sorted(hf["data"].keys(), key=lambda s: int(s.split("_")[1])):
                g = hf["data"][k]
                eps.append(dict(obs={o: g[f"obs/{o}"][:] for o in g["obs"].keys()}, actions=g["actions"][:],
                                rewards=g["rewards"][:], seed=int(g.attrs["seed"])))
    shas = {m["cell_sha256"] for m in metas}
    assert len(shas) == 1, f"shards were made with different cells: {shas}"
    meta = dict(metas[0], shards=[os.path.basename(d) for d in sorted(dirs)],
                scripted_success_rate=float(np.mean([m["scripted_success_rate"] for m in metas])))
    path = write(out_dir, eps, meta)
    print(f"[merge] {len(eps)} demos, {sum(len(e['actions']) for e in eps)} transitions -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_demos", type=int, default=125)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard", type=int, default=0, help="seeds = 100000 * (shard + 1) + attempt")
    ap.add_argument("--noise", type=float, default=0.0, help="expert action noise (diversity)")
    ap.add_argument("--sigma", type=float, default=0.1, help="hand_init_perturb_joint_sigma (project standard)")
    ap.add_argument("--episode_length", type=int, default=400)
    ap.add_argument("--max_attempt_factor", type=float, default=2.0)
    ap.add_argument("--merge", nargs="*", default=None)
    ap.add_argument("--cell", default="", help="cell yaml (default: the Robotiq class's, env/threading_variants.py)")
    args = ap.parse_args()
    if args.merge:
        return merge(args.merge, args.out_dir)
    import env.threading_variants as tv
    from env.robosuite_wrapper import PixelRobosuite
    from env.sim2real import load_sim2real
    cell_path = args.cell or tv.REAL_CELL_ROBOTIQ_YAML
    cell = load_sim2real(cell_path)
    env = PixelRobosuite("Threading_RealCellRobotiq", "Panda", args.episode_length, camera_names=[], rl_cameras=[],
                         use_state=True, state_only_no_render=True, record_sim_state=True, device="cpu",
                         hand_init_perturb_joint_sigma=args.sigma, end_on_success=True, sim2real=cell)
    acc, att = [], 0
    while len(acc) < args.num_demos and att < int(args.num_demos * args.max_attempt_factor):
        seed = 100000 * (args.shard + 1) + att
        ok, obs, actions, rewards, steps = rollout(env, seed, args.episode_length, args.noise)
        att += 1
        if ok:
            acc.append(dict(obs=obs, actions=actions, rewards=rewards, seed=seed))
        if att % 25 == 0:
            print(f"[collect {args.shard}] attempts {att}, accepted {len(acc)} (last ok={ok}, {steps} steps)", flush=True)
    rate = len(acc) / max(att, 1)
    meta = dict(env_name="Threading_RealCellRobotiq", sim2real_cell=os.path.relpath(cell_path),
                cell_sha256=sha256(cell_path), control_freq=float(env.env.control_freq),
                exec_layer=bool(getattr(env.env, "exec_layer", None) is not None),
                hand_init_perturb_joint_sigma=args.sigma, episode_length=args.episode_length, use_state=True,
                source_weight="scripted_expert (scripts/threading_robotiq_expert.py)",
                collect_action_noise=args.noise, scripted_success_rate=rate, shard=args.shard)
    path = write(args.out_dir, acc, meta)
    print(f"[collect {args.shard}] {len(acc)} demos in {att} attempts (rate {rate:.3f}) -> {path}")


if __name__ == "__main__":
    main()

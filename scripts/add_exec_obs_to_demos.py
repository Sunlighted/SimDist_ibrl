"""Add obs/exec_ref (the execution layer's hidden reference-arm state + last action, env/robosuite_wrapper.py
EXEC_OBS_*) to Threading Robotiq demos collected without it (2026-10-25).

Each demo is rebuilt bit-exactly: seeded reset (the demo's attrs["seed"]) + replay of its actions, with every step's
replayed obs/state checked for EXACT equality against the stored one; any mismatch aborts. The output holds the first
--num demos with all their original datasets plus obs/exec_ref, so train_rl.py --exec_obs 1 can preload them.

    python scripts/add_exec_obs_to_demos.py --src DIR/dataset.hdf5 --out_dir DIR_execobs --num 1000 --workers 24
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(job):
    import h5py
    import torch
    from env.robosuite_wrapper import PixelRobosuite, EXEC_OBS_DIM
    from env.sim2real import load_sim2real
    src, ids, meta = job
    env = PixelRobosuite("Threading_RealCellRobotiq", "Panda", meta["episode_length"], camera_names=[], rl_cameras=[],
                         use_state=True, state_only_no_render=True, device="cpu", end_on_success=True,
                         hand_init_perturb_joint_sigma=meta["hand_init_perturb_joint_sigma"],
                         sim2real=load_sim2real(meta["sim2real_cell"]), exec_obs=True)
    n0 = env.observation_shape[0] - EXEC_OBS_DIM
    out = {}
    with h5py.File(src, "r") as f:
        for i in ids:
            g = f[f"data/demo_{i}"]
            states, actions = g["obs/state"][:], g["actions"][:]
            np.random.seed(int(g.attrs["seed"]))
            obs, _ = env.reset()
            ref = []
            for t in range(len(actions)):
                s = obs["state"].numpy()
                if not np.array_equal(s[:n0], states[t]):
                    return None, f"demo {i} step {t}: max |diff| {np.abs(s[:n0] - states[t]).max():.3g}"
                ref.append(obs["state"].numpy()[n0:].copy())
                obs, _, _, _, _ = env.step(torch.from_numpy(actions[t]))
            out[i] = np.stack(ref).astype(np.float32)
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    import multiprocessing as mp
    import h5py
    from env.robosuite_wrapper import EXEC_OBS_KEY
    from scripts.collect_threading_robotiq_demos import sha256
    meta = json.load(open(os.path.join(os.path.dirname(args.src), "env_cfg.json")))
    assert meta["exec_layer"], "demos collected without an execution layer"
    if sha256(meta["sim2real_cell"]) != meta["cell_sha256"]:      # the per-step bit-exact state check is what counts
        print(f"WARNING: {meta['sim2real_cell']} changed since collection (e.g. 10-22 table_impact_force 5 -> 35 N)")
    ids = list(range(args.num))
    with mp.get_context("spawn").Pool(args.workers) as p:
        res = p.map(run, [(args.src, ids[w::args.workers], meta) for w in range(args.workers)])
    errs = [e for _, e in res if e]
    if errs:
        sys.exit("replay mismatch, nothing written:\n  " + "\n  ".join(errs))
    refs = {k: v for r, _ in res for k, v in r.items()}
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "dataset.hdf5")
    with h5py.File(args.src, "r") as fs, h5py.File(path, "w") as fo:
        g = fo.create_group("data")
        g.attrs["num_demos"] = len(ids)
        for i in ids:
            fs.copy(fs[f"data/demo_{i}"], g, name=f"demo_{i}")
            g[f"demo_{i}/obs"].create_dataset(EXEC_OBS_KEY, data=refs[i])
    meta = dict(meta, exec_obs=True, exec_obs_source=f"{os.path.abspath(args.src)} (first {len(ids)} demos, "
                "seed + action replay, bit-exact state check)")
    with open(os.path.join(args.out_dir, "env_cfg.json"), "w") as f:
        json.dump(meta, f, indent=2)
    shutil.copy(os.path.join(os.path.dirname(args.src), "env_cfg.json"), os.path.join(args.out_dir, "env_cfg_src.json"))
    allr = np.concatenate(list(refs.values()))
    print(f"{len(ids)} demos, {len(allr)} steps -> {path}\n  exec_ref |mean| per dim: "
          + " ".join(f"{x:.3f}" for x in np.abs(allr).mean(0)))


if __name__ == "__main__":
    main()

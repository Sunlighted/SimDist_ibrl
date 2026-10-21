"""How deep do the gripper and the needle interpenetrate on the Threading Robotiq cell? (2026-10-21)

MuJoCo contacts carry a signed distance (negative = penetration). Per episode we take the deepest penetration of every
geom pair class and the geoms involved:
  demos   -- the expert demos: reset from the stored seed (same DR) and write the recorded sim_states back;
  policy  -- greedy rollouts of an IBRL / BC ckpt (= evaluate/eval.py), contacts read after every env step.

    python scripts/measure_threading_robotiq_penetration.py demos  --data <dataset.hdf5> --num 200 --out <npz>
    python scripts/measure_threading_robotiq_penetration.py policy --weight <model0.pt> [--mode bc] --num 100 --out <npz>
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pair_class(b1, b2):
    s = {("grip" if b.startswith("gripper0") else "needle" if b.startswith("needle") else
          "tripod" if b.startswith("tripod") else "table" if b in ("table", "world") else "robot") for b in (b1, b2)}
    return "+".join(sorted(s)) if len(s) == 2 else next(iter(s)) + "(self)"


def contacts(sim):
    """{pair class: (deepest penetration [m, >0], 'geomA|geomB')} for the current state."""
    import mujoco
    m, d = sim.model._model, sim.data._data
    out = {}
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist >= 0:
            continue
        g1, g2 = c.geom1, c.geom2
        b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g1]) or ""
        b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g2]) or ""
        k = pair_class(b1, b2)
        if k.endswith("(self)") or "robot" in k:
            continue
        if -c.dist > out.get(k, (0.0, ""))[0]:
            n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g1)
            n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g2)
            out[k] = (float(-c.dist), f"{n1}|{n2}")
    return out


def make_env(env_params=None):
    from env.robosuite_wrapper import PixelRobosuite
    if env_params is not None:
        return PixelRobosuite(**dict(env_params, state_only_no_render=True, hand_init_perturb_joint_sigma=0.1))
    import env.threading_variants as tv
    from env.sim2real import load_sim2real
    return PixelRobosuite("Threading_RealCellRobotiq", "Panda", 400, camera_names=[], rl_cameras=[], use_state=True,
                          state_only_no_render=True, device="cpu", hand_init_perturb_joint_sigma=0.1,
                          end_on_success=True, sim2real=load_sim2real(tv.REAL_CELL_ROBOTIQ_YAML))


def episode_stats(per_step):
    """per_step: list of contacts() dicts -> {class: (max depth, geoms at max, frac of steps with depth > 1 mm)}"""
    res = {}
    for k in {k for s in per_step for k in s}:
        depths = np.array([s.get(k, (0.0, ""))[0] for s in per_step])
        i = int(np.argmax(depths))
        res[k] = (float(depths[i]), per_step[i][k][1], float(np.mean(depths > 1e-3)), i)
    return res


def run_demos(args):
    import h5py
    env = make_env()
    f = h5py.File(args.data, "r")["data"]
    keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))[:args.num]
    out = []
    for k in keys:
        np.random.seed(int(f[k].attrs["seed"]))
        env.reset()
        sim = env.env.sim
        steps = []
        for st in f[k]["obs/sim_state"][:]:
            sim.set_state_from_flattened(st)
            sim.forward()
            steps.append(contacts(sim))
        out.append(dict(key=k, stats=episode_stats(steps)))
    return out


def run_policy(args):
    import torch
    import train_bc
    import train_rl
    from common_utils import ibrl_utils as utils
    agent, _, env_params = (train_rl.load_model if args.mode == "rl" else train_bc.load_model)(args.weight, "cuda")
    env = make_env(env_params)
    out = []
    with torch.no_grad(), utils.eval_mode(agent):
        for s in range(1, args.num + 1):
            np.random.seed(s)
            obs, _ = env.reset()
            steps, term, succ_any, queue = [], False, False, []
            while not term:
                if not queue:
                    chunk = (agent.act_chunk(obs, eval_mode=True) if hasattr(agent, "act_chunk")
                             else agent.act(obs, eval_mode=True).unsqueeze(0))
                    queue = [x for x in chunk]
                a = queue.pop(0)
                a = (a.cpu() if hasattr(a, "cpu") else torch.as_tensor(a)).reshape(-1)
                obs, _, term, succ, _ = env.step(a)
                succ_any |= bool(succ)
                steps.append(contacts(env.env.sim))
            out.append(dict(key=f"seed{s}", success=succ_any, stats=episode_stats(steps)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["demos", "policy"])
    ap.add_argument("--data")
    ap.add_argument("--weight")
    ap.add_argument("--mode", default="rl", choices=["rl", "bc"])
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = run_demos(args) if args.what == "demos" else run_policy(args)
    np.save(args.out, np.array(res, dtype=object))
    classes = sorted({k for r in res for k in r["stats"]})
    print(f"{args.what}: {len(res)} episodes")
    for k in classes:
        d = np.array([r["stats"].get(k, (0.0,))[0] for r in res]) * 1e3
        geoms = defaultdict(int)
        for r in res:
            if k in r["stats"] and r["stats"][k][0] > 2e-3:
                geoms[r["stats"][k][1]] += 1
        top = ", ".join(f"{g} x{n}" for g, n in sorted(geoms.items(), key=lambda x: -x[1])[:3])
        print(f"  {k:14s} max penetration per episode [mm]: median {np.median(d):5.2f}  p90 {np.percentile(d, 90):5.2f}  "
              f"max {d.max():6.2f}  | episodes >2 mm {np.mean(d > 2):.2f} >5 mm {np.mean(d > 5):.2f}  | deepest pairs: {top}")


if __name__ == "__main__":
    main()

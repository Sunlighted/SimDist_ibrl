"""Replay MimicGen demos open-loop in the RL env (PixelRobosuite + sim2real cell +
strict success) and report how many still count as successes.

MimicGen's "success" is the task's own lenient _check_success. The IBRL config
trains and evaluates under STRICT success (dwell steps, max tripod displacement),
so a demo set can be 100% "successful" at generation and still teach a prior
behaviour that the RL reward never pays for. This checks the gap before a BC
prior is trained on the data.

Replays the recorded ACTIONS from the recorded initial sim state. The sim is
deterministic, so a mismatch here is a real property of the demo, not noise --
but the check is only valid if the wrapper builds the SAME geometry as the
generation env (asserted below against the demo's own first state).
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch

import env.threading_variants  # noqa: F401
from env.robosuite_wrapper import PixelRobosuite
from env.strict_success import StrictSuccessConfig
from env.sim2real import Sim2RealConfig, load_sim2real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True)
    ap.add_argument("--cell", default="release/cfgs/sim2real/threading_real_cell.yaml")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--no_cell", action="store_true",
                    help="replay on the ORIGINAL geometry (for demos generated there, "
                         "e.g. the core D0 set) -- the reference point for the strict rate")
    args = ap.parse_args()

    cell = load_sim2real(args.cell)
    # matched geometry only: the demo was generated at the nominal cell, no DR
    geo = None if args.no_cell else Sim2RealConfig(
        enabled=1, table_above_base=cell.table_above_base,
        workspace_offset=cell.workspace_offset)
    strict = StrictSuccessConfig(enabled=True, dwell_steps=3, max_tripod_disp=0.04,
                                 max_needle_speed=0.0)
    env = PixelRobosuite("Threading_D0", "Panda", 400, camera_names=[], rl_cameras=[],
                         use_state=True, device="cpu", state_only_no_render=True,
                         hand_init_perturb_joint_sigma=0.0, strict_success=strict,
                         sim2real=geo)

    f = h5py.File(args.demo, "r")
    keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
    if args.n:
        keys = keys[: args.n]
    ok = lax = 0
    lens = []
    for k in keys:
        g = f["data"][k]
        states, actions = g["states"][()], g["actions"][()]
        env.reset()
        sim = env.env.sim
        assert sim.get_state().flatten().shape == states[0].shape, (
            "state size mismatch: the wrapper env is not the generation env")
        sim.set_state_from_flattened(states[0])
        sim.forward()
        # re-latch controller + strict validator on the injected state
        ctrl = env.env.robots[0].controller
        ctrl.update_initial_joints(sim.data.qpos[env.env.robots[0]._ref_joint_pos_indexes].copy())
        ctrl.reset_goal()
        if env._success_validator is not None:
            env._success_validator.reset(env.env, env._bindings)
        got_lax = False
        r_tot, t = 0.0, 0
        for a in actions:
            _, r, term, _, _ = env.step(torch.from_numpy(a.astype(np.float32)))
            got_lax |= bool(env.env._check_success())
            r_tot += float(r); t += 1
            if term:
                break
        ok += r_tot > 0
        lax += got_lax
        lens.append(t)
    n = len(keys)
    print(f"demos={n}  lenient success={lax}/{n} ({lax/n:.2f})  "
          f"STRICT success={ok}/{n} ({ok/n:.2f})  mean len={np.mean(lens):.0f}")


if __name__ == "__main__":
    main()

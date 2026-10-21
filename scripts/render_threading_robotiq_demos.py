"""Render Threading Robotiq scripted-expert demos (threading_robotiq_expert_n1000) to mp4 + a key-frame contact sheet.

Each demo is reset from its stored seed (reproduces the cell's physics / base / visual DR exactly: initial sim_state
error 4e-17, scripts/check_threading_robotiq_demos.py) and then the RECORDED sim_states are written back frame by
frame -- no re-simulation, so what you see is what was saved. Three views: agentview (the scene camera), a close-up on
the ring along its axis (free camera; shows whether the needle really goes through), and the wrist camera.

    python scripts/render_threading_robotiq_demos.py --data <dir>/dataset.hdf5 --out <dir> --demos demo_0 demo_17
"""
import argparse
import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

H, W = 300, 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--demos", nargs="+", required=True)
    ap.add_argument("--keyframes", type=int, default=6)
    args = ap.parse_args()
    import cv2
    import h5py
    import imageio
    import mujoco
    import env.threading_variants as tv
    from env.robosuite_wrapper import PixelRobosuite, resolve_bindings
    from env.sim2real import load_sim2real
    env = PixelRobosuite("Threading_RealCellRobotiq", "Panda", 400, camera_names=[], rl_cameras=[], use_state=True,
                         state_only_no_render=True, record_sim_state=True, device="cpu",
                         hand_init_perturb_joint_sigma=0.1, end_on_success=True,
                         sim2real=load_sim2real(tv.REAL_CELL_ROBOTIQ_YAML))
    os.makedirs(args.out, exist_ok=True)
    f = h5py.File(args.data, "r")["data"]
    for key in args.demos:
        g = f[key]
        seed, states = int(g.attrs["seed"]), g["obs/sim_state"][:]
        np.random.seed(seed)
        env.reset()
        sim = env.env.sim
        b = resolve_bindings(env.env)
        ring = np.mean(sim.data.geom_xpos[b["ring_geom_ids"]], axis=0)
        normal = sim.data.get_body_xmat("tripod_obj_root")[:, 0]   # only used to pick the azimuth
        rend = mujoco.Renderer(sim.model._model, H, W)
        close = mujoco.MjvCamera()
        close.type = mujoco.mjtCamera.mjCAMERA_FREE
        close.lookat[:] = ring
        close.distance, close.elevation = 0.22, -15.0
        close.azimuth = float(np.degrees(np.arctan2(normal[1], normal[0]))) + 35.0
        frames = []
        for t, s in enumerate(states):
            sim.set_state_from_flattened(s)
            sim.forward()
            views = []
            for cam in ("agentview", close, "robot0_eye_in_hand"):
                rend.update_scene(sim.data._data, camera=cam)
                views.append(rend.render().copy())
            fr = np.concatenate(views, 1)
            cv2.putText(fr, f"{key} seed {seed}  t={t}/{len(states)}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            frames.append(fr)
        imageio.mimwrite(os.path.join(args.out, f"{key}.mp4"), frames, fps=10, quality=7)
        idx = np.linspace(0, len(frames) - 1, args.keyframes).round().astype(int)
        imageio.imwrite(os.path.join(args.out, f"{key}_sheet.jpg"), np.concatenate([frames[i] for i in idx], 0),
                        quality=85)
        print(f"[render] {key}: {len(frames)} frames", flush=True)


if __name__ == "__main__":
    main()

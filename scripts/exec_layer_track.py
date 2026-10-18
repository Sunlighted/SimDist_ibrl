"""The tracking test INSIDE the training env, with the cell's execution layer (env/exec_layer.py), no server.

Same TrackScript as policy_server --scripted track, acting on the env's measured grip site; env.step(action) does the
rest (reference OSC, caps, ramp, joint impedance on the fitted arm). Writes steps.jsonl in the real runs' format, so
calib_20261014/tools/analyze_tracking.py compares it with a real run directly:

    python scripts/exec_layer_track.py --cell <cell with exec_layer> --hz 10 --out /tmp/exec10 \
        --real real2sim/zed_capture/calib_20261015/data/track10_sim_ref_1015_190048
"""
import argparse
import json
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--real", default="", help="a real run to print next to it")
    ap.add_argument("--q0_from_real", action="store_true", help="start at the real run's first q (default: cell home)")
    ap.add_argument("--max_steps", type=int, default=3000)
    args = ap.parse_args()
    os.environ["THREADING_ROBOTIQ_CELL_YAML"] = os.path.abspath(args.cell)
    import dataclasses
    import env.threading_variants as tv  # noqa: F401
    from env.robosuite_wrapper import PixelRobosuite
    from env.sim2real import Sim2RealConfig, load_sim2real
    from real2sim.deploy.policy_server import TrackScript, build_parser

    cell = load_sim2real(args.cell)
    off = {f.name: f.default for f in dataclasses.fields(Sim2RealConfig) if f.name.startswith("dr_")}
    cell = dataclasses.replace(cell, cameras=tuple(dataclasses.replace(c, randomize=False) for c in cell.cameras), **off)
    env = PixelRobosuite("Threading_RealCellRobotiq", "Panda", 10 ** 6, camera_names=[], rl_cameras=[], use_state=True,
                         state_only_no_render=True, device="cpu", hand_init_perturb_joint_sigma=0.0, sim2real=cell)
    np.random.seed(0)
    env.reset()
    e = env.env
    rob = e.robots[0]
    ctrl = e.exec_layer
    assert ctrl is not None, f"{args.cell} has no exec_layer"
    hz = float(e.control_freq)
    # park the objects out of the way (free-space test, as on the robot)
    m, d = e.sim.model, e.sim.data
    for n in ("needle_obj_joint0", "tripod_obj_joint0"):
        a = m.get_joint_qpos_addr(n)[0]
        d.qpos[a:a + 3] = [0.0, 2.0 if "needle" in n else 3.0, -5.0]
    if args.q0_from_real and args.real:
        q0 = json.loads(open(os.path.join(args.real, "steps.jsonl")).readline())["q"]
        d.qpos[rob._ref_joint_pos_indexes] = q0
    e.sim.forward()
    ctrl.reset_goal()
    pa = build_parser().parse_args(["--scripted", "track"])
    script = TrackScript(pa.track_axes.split(","), [float(x) for x in pa.track_amps.split(",")], pa.track_dist,
                         pa.track_hold_s, pa.track_max_move_s)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "steps.jsonl"), "w")
    sid = rob.eef_site_id
    caps = 0
    for k in range(args.max_steps):
        eef = np.array(d.site_xpos[sid])
        q = np.array(d.qpos[rob._ref_joint_pos_indexes])
        action, info = script.act(eef, hz)
        e.timestep = 0
        e.step(np.clip(action, -1, 1))
        caps += bool(ctrl.last_caps)
        # commanded TCP = FK of the joint target this step sent (as eef_target_full on the robot)
        from env.exec_layer import site_pose
        tgt, _ = site_pose(ctrl.shadow.model._model, ctrl.shadow.data._data, sid, ctrl.jidx, ctrl.goal)
        log.write(json.dumps(dict(step=k, q=q.round(5).tolist(), eef_full=eef.round(6).tolist(),
                                  eef_target_full=tgt.round(6).tolist(), action=np.round(action, 4).tolist(),
                                  script=info, hz=hz, warn=(["capped(" + ",".join(ctrl.last_caps) + ")"]
                                                            if ctrl.last_caps else []))) + "\n")
        if script.done:
            break
    log.close()
    print(f"[exec-layer track] {k + 1} steps at {hz:g} Hz, {caps} capped, resyncs {ctrl.resyncs}")
    tool = os.path.join(REPO, "real2sim/zed_capture/calib_20261014/tools/analyze_tracking.py")
    subprocess.run([sys.executable, tool] + ([args.real] if args.real else []) + [args.out])


if __name__ == "__main__":
    main()

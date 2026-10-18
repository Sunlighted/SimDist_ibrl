"""Sim twin of the real tracking test: the deploy execution chain with the real arm replaced by a sim arm.

calib_20261015/SIM_TODO.md §2 ("put the real execution layer into the sim"). Everything between the action and
the joint command is the DEPLOY CODE ITSELF, not a re-implementation:
  real2sim/deploy/policy_server.py Server.step  (TrackScript -> ReferenceArm sim_ref -> box clamp -> _speed_caps)
  robot_client.py                               (per-step joint clip, 100 Hz linear ramp over one period)
Only the robot is swapped: a second copy of the cell's arm, torque-controlled exactly like franka_ros2's
joint_impedance_controller,  tau = K (q_cmd - q) - D dq + gravity/Coriolis compensation,  K / D = the real gains.
Free parameters (what the sim does not know about the FR3): joint armature (reflected rotor inertia) and
frictionloss (stiction), per joint or one scale. Fit them so that analyze_tracking.py gives the same numbers
on the twin as on the real run.

    python scripts/sim_tracking_twin.py --real real2sim/zed_capture/calib_20261015/data/track10_sim_ref_1015_190048 \
        --out /tmp/twin10 [--armature_scale 1] [--frictionloss 0.1]
    -> <out>/steps.jsonl + joints.npz (same format as the real run) and the analyze_tracking table of both.
"""
import argparse
import json
import os
import re
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "real2sim", "deploy"))
import numpy as np  # noqa: E402

KP = np.array([240, 240, 240, 240, 100, 60, 20.])     # franka_ros2 joint_impedance_controller (calib_20261015 README)
KD = np.array([20, 20, 20, 10, 10, 10, 5.])
CMD_HZ = 100.0
MAX_STEP_RAD = 0.03                                   # robot_client.py --max_step_rad default


class SimClock:
    """policy_server reads time.monotonic() for the speed cap's dt: give it the twin's clock."""

    def __init__(self):
        import time as _t
        self._t = _t
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def perf_counter(self):
        return self._t.perf_counter()

    def time(self):
        return self._t.time()


def real_run_info(run):
    rows = [json.loads(ln) for ln in open(os.path.join(run, "steps.jsonl")) if ln.strip()]
    hz = float(rows[0].get("hz", 20.0))
    cell = None
    log = os.path.join(run, "server.log")
    if os.path.exists(log):
        m = re.search(r"(real2sim/zed_capture/calib_\d+)/", open(log).read())
        if m:
            cell = m.group(1)
    return rows, hz, cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="a real tracking-test run dir (steps.jsonl, joints.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", default="", help="default: the calib dir the real server.log names")
    ap.add_argument("--armature_scale", type=float, default=1.0)
    ap.add_argument("--armature", type=float, nargs=7, default=None, help="per joint, overrides the scale")
    ap.add_argument("--frictionloss", type=float, nargs="+", default=[0.1], help="1 value or 7 (N m)")
    ap.add_argument("--damping", type=float, nargs="+", default=None, help="joint viscous damping, 1 or 7")
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--action_map", default="", help="sim_ref / reseed; default: what the real server.log says")
    ap.add_argument("--params", default="", help="json from scripts/fit_arm_twin.py (armature/frictionloss/damping)")
    ap.add_argument("--uncomp_mass", type=float, default=0.30,
                    help="load the real gravity comp did not know (0.30 kg before the 10-16 Desk fix, then 0)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    real_rows, hz, cell_dir = real_run_info(args.real)
    cell = args.cell or (os.path.join(REPO, cell_dir, os.path.basename(cell_dir).replace("calib_", "threading_real_cell_") + ".yaml")
                         if cell_dir else "")
    assert cell and os.path.exists(cell), f"cell not found ({cell}); pass --cell"
    # the reference arm of the REAL run: the Robotiq class reads its model cell from this env var at
    # import, so set it before anything imports env.threading_variants
    os.environ["THREADING_ROBOTIQ_CELL_YAML"] = cell
    import real2sim.deploy.policy_server as ps
    clock = SimClock()
    ps.time = clock

    amap = args.action_map
    if not amap:
        log = os.path.join(args.real, "server.log")
        amap = "reseed" if os.path.exists(log) and "action map" not in open(log).read() else "sim_ref"
    a = ps.build_parser().parse_args(["--scripted", "track", "--task", "Threading_RealCellRobotiq",
                                      "--cell", cell, "--no_color", "--save_dir", args.out, "--save_every", "0",
                                      "--action_map", amap])
    a.z_slow_band = 0.0
    os.makedirs(args.out, exist_ok=True)
    open(os.path.join(args.out, "steps.jsonl"), "w").close()
    srv = ps.Server(a)
    srv.obs_images = lambda blobs, names: {}
    srv.hz = hz
    srv._reset_ref()

    # ---- the "real" arm: another copy of the cell's arm, torque control
    twin = ps.ReferenceArm("Threading_RealCellRobotiq", cell)
    sim, m, d = twin.sim, twin.sim.model, twin.sim.data
    jv, jq = twin.jvel, twin.jidx
    act = np.array(twin.rob._ref_joint_actuator_indexes)
    arm0 = m.dof_armature[jv].copy()
    m.dof_armature[jv] = np.asarray(args.armature) if args.armature else arm0 * args.armature_scale
    fl = np.asarray(args.frictionloss, np.float64)
    m.dof_frictionloss[jv] = fl if fl.size == 7 else np.full(7, float(fl[0]))
    if args.damping is not None:
        dm = np.asarray(args.damping, np.float64)
        m.dof_damping[jv] = dm if dm.size == 7 else np.full(7, float(dm[0]))
    if args.params:
        jp = json.load(open(args.params))
        m.dof_armature[jv] = jp["armature"]
        m.dof_frictionloss[jv] = jp["frictionloss"]
        m.dof_damping[jv] = jp["damping"]
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from fit_arm_twin import Twin as _T
    uncomp = (lambda: _T.uncomp(type("o", (), {"m": m, "d": d, "hand": m.body_name2id("robot0_right_hand"),
                                               "jv": jv})(), args.uncomp_mass))
    q0 = np.asarray(real_rows[0]["q"], np.float64)
    twin.set_state(q0, np.zeros(7), [0.0])
    dt_phys = float(m.opt.timestep)
    sub = int(round(1.0 / CMD_HZ / dt_phys))           # physics steps per 100 Hz command
    per = int(round(CMD_HZ / hz))                      # commands per control period

    def physics(q_cmd, n):
        for _ in range(n):
            q, dq = d.qpos[jq], d.qvel[jv]
            tau = KP * (q_cmd - q) - KD * dq + d.qfrc_bias[jv]
            d.ctrl[act] = tau + (uncomp() if args.uncomp_mass else 0.0)
            sim.step()
            clock.now += dt_phys
            jlog.append((clock.now, d.qpos[jq].copy(), d.qvel[jv].copy()))

    jlog, clog = [], []
    cmd = q0.copy()
    last_target = None
    for k in range(args.max_steps):
        q, dq = d.qpos[jq].copy(), d.qvel[jv].copy()
        res = srv.step({"q": q.tolist(), "dq": dq.tolist(), "fingers": [0.0], "cams": [],
                        "t_mono": clock.now, "ages": {}}, [])
        tgt = np.asarray(res["q_target"])
        base = last_target if last_target is not None else q
        tgt = base + np.clip(tgt - base, -MAX_STEP_RAD, MAX_STEP_RAD)
        start = cmd.copy()
        for i in range(per):                            # robot_client InterpolatingHolder: linear ramp over 1 period
            cmd = start + (tgt - start) * (i + 1) / per
            clog.append((clock.now, cmd.copy()))
            physics(cmd, sub)
        last_target = tgt
        if res.get("done"):
            break
    jt = np.array([x[0] for x in jlog])
    np.savez(os.path.join(args.out, "joints.npz"), joint_t=jt, joint_q=np.array([x[1] for x in jlog]),
             joint_dq=np.array([x[2] for x in jlog]), cmd_t=np.array([x[0] for x in clog]),
             cmd_q=np.array([x[1] for x in clog]), hz=hz)
    json.dump(dict(real=args.real, cell=cell, hz=hz, action_map=amap, uncomp_mass=args.uncomp_mass, armature=m.dof_armature[jv].tolist(),
                   frictionloss=m.dof_frictionloss[jv].tolist(), damping=m.dof_damping[jv].tolist(),
                   steps=k + 1), open(os.path.join(args.out, "twin.json"), "w"), indent=1)
    if not args.quiet:
        print(f"[twin] {amap}, {k + 1} steps at {hz:g} Hz, cell {os.path.relpath(cell, REPO)} | armature "
              f"{np.round(m.dof_armature[jv], 3).tolist()} frictionloss {np.round(m.dof_frictionloss[jv], 3).tolist()}")
        tool = os.path.join(REPO, "real2sim/zed_capture/calib_20261014/tools/analyze_tracking.py")
        subprocess.run([sys.executable, tool, args.real, args.out])


if __name__ == "__main__":
    main()

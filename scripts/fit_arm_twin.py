"""Open-loop system ID of the real FR3 under joint impedance: real 100 Hz commands in, compare joints out.

The tracking tests' joints.npz hold every 100 Hz command the client sent (cmd_t, cmd_q) and every measured joint
state (joint_t, joint_q). Feed the commands, zero-order hold as the controller sees them, into a sim copy of the
cell's arm under the real controller (tau = K (q_cmd - q) - D dq + gravity/Coriolis comp), and fit what the sim
does not know: per-joint armature (reflected rotor inertia), frictionloss (stiction) and viscous damping.
Runs recorded before the 10-16 Desk payload fix had 0.30 kg the compensation did not know about:
--uncomp_mass reproduces that (default 0.30 for runs before 2026-10-16 21:44, see calib_20261015/README).

    python scripts/fit_arm_twin.py <run_dir> [<run_dir> ...] [--fit] [--out params.json]
Without --fit: evaluate the given parameters and print per-joint tracking rms (twin vs real).
"""
import argparse
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import numpy as np  # noqa: E402

KP = np.array([240, 240, 240, 240, 100, 60, 20.])
KD = np.array([20, 20, 20, 10, 10, 10, 5.])
G = 9.81


class Twin:
    def __init__(self, cell):
        os.environ["THREADING_ROBOTIQ_CELL_YAML"] = cell
        from real2sim.deploy.policy_server import ReferenceArm
        self.arm = ReferenceArm("Threading_RealCellRobotiq", cell)
        a = self.arm
        self.m, self.d = a.sim.model, a.sim.data
        self.jv, self.jq = a.jvel, a.jidx
        self.act = np.array(a.rob._ref_joint_actuator_indexes)
        self.hand = self.m.body_name2id("robot0_right_hand")
        self.arm0 = self.m.dof_armature[self.jv].copy()

    def set_params(self, armature, frictionloss, damping):
        self.m.dof_armature[self.jv] = armature
        self.m.dof_frictionloss[self.jv] = frictionloss
        self.m.dof_damping[self.jv] = damping

    def uncomp(self, mass, com=(0, 0, 0.065)):
        """torque the real gravity compensation does NOT provide for `mass` at `com` (hand frame): -J^T(m g z)"""
        import mujoco
        if mass == 0:
            return np.zeros(7)
        mm, dd = self.m._model, self.d._data
        R = dd.xmat[self.hand].reshape(3, 3)
        p = dd.xpos[self.hand] + R @ np.asarray(com)
        jp = np.zeros((3, mm.nv))
        mujoco.mj_jac(mm, dd, jp, None, p, self.hand)
        return -(jp[:, self.jv].T @ np.array([0, 0, mass * G]))

    def run(self, cmd_t, cmd_q, q0, t_end, uncomp_mass, sample_t):
        a, m, d = self.arm, self.m, self.d
        a.set_state(q0, np.zeros(7), [0.0])
        dt = float(m.opt.timestep)
        t = cmd_t[0]
        k = 0
        out = np.empty((len(sample_t), 7))
        si = 0
        n = 0
        extra = np.zeros(7)
        while t < t_end and si < len(sample_t):
            while k + 1 < len(cmd_t) and cmd_t[k + 1] <= t:
                k += 1
            q, dq = d.qpos[self.jq], d.qvel[self.jv]
            tau = KP * (cmd_q[k] - q) - KD * dq + d.qfrc_bias[self.jv]
            if uncomp_mass and n % 5 == 0:
                extra = self.uncomp(uncomp_mass)     # the load acts, the compensation does not know it
            d.ctrl[self.act] = tau + extra
            n += 1
            a.sim.step()
            t += dt
            while si < len(sample_t) and sample_t[si] <= t:
                out[si] = d.qpos[self.jq]
                si += 1
        return out[:si]


def load(run):
    z = np.load(os.path.join(run, "joints.npz"))
    return z["cmd_t"], z["cmd_q"], z["joint_t"], z["joint_q"]


def evaluate(twin, runs, p, uncomp_mass, every=0.01):
    errs = []
    for run in runs:
        ct, cq, jt, jqv = load(run)
        t0 = ct[0]
        st = np.arange(t0 + 0.5, min(ct[-1], jt[-1]), every)
        real = np.stack([np.interp(st, jt, jqv[:, j]) for j in range(7)], 1)
        q0 = np.array([np.interp(t0, jt, jqv[:, j]) for j in range(7)])
        twin.set_params(*p)
        sim = twin.run(ct, cq, q0, st[-1] + 1e-9, uncomp_mass, st)
        errs.append(real[:len(sim)] - sim)
    e = np.concatenate(errs)
    return np.sqrt((e ** 2).mean(0)), e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--cell", default=os.path.join(REPO, "real2sim/zed_capture/calib_20261014/threading_real_cell_20261014.yaml"))
    ap.add_argument("--uncomp_mass", type=float, default=0.30)
    ap.add_argument("--params", default="", help="json {armature, frictionloss, damping} to evaluate / start from")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    twin = Twin(args.cell)
    if args.params:
        j = json.load(open(args.params))
        p = [np.asarray(j[k], np.float64) for k in ("armature", "frictionloss", "damping")]
    else:
        p = [twin.arm0.copy(), np.full(7, 0.1), np.array([0.1] * 5 + [0.01] * 2)]
    rms, _ = evaluate(twin, args.runs, p, args.uncomp_mass)
    print(f"start  rms mrad {np.round(1e3 * rms, 2).tolist()}  sum {1e3 * rms.sum():.2f}")
    if args.fit:
        # coordinate search per joint on (armature, frictionloss, damping) -- the joints couple weakly at these
        # speeds, so a few sweeps converge; each eval re-simulates every run
        grids = {0: [0.25, 0.5, 0.7, 1.0, 1.4, 2.0],                 # armature x nominal
                 1: [0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.5],              # frictionloss N m
                 2: [0.0, 0.1, 0.5, 1.0, 2.0, 4.0]}                   # damping N m s
        best = rms
        for sweep in range(2):
            for j in range(7):
                for which, grid in grids.items():
                    cands = []
                    for v in grid:
                        pp = [x.copy() for x in p]
                        pp[which][j] = twin.arm0[j] * v if which == 0 else v
                        r, _ = evaluate(twin, args.runs, pp, args.uncomp_mass)
                        cands.append((r[j] + 0.2 * r.sum(), v, r, pp))
                    _, v, r, pp = min(cands, key=lambda c: c[0])
                    p, best = pp, r
                print(f"sweep {sweep} j{j + 1}: armature {p[0][j]:.3f} fric {p[1][j]:.2f} damp {p[2][j]:.2f} | rms mrad "
                      f"{np.round(1e3 * best, 2).tolist()}", flush=True)
        if args.out:
            json.dump({"armature": p[0].tolist(), "frictionloss": p[1].tolist(), "damping": p[2].tolist(),
                       "rms_mrad": (1e3 * best).tolist(), "runs": args.runs, "uncomp_mass": args.uncomp_mass},
                      open(args.out, "w"), indent=1)
            print("wrote", args.out)


if __name__ == "__main__":
    main()

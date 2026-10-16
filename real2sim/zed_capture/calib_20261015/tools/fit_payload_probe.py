"""Fit the missing payload from a payload_probe.py recording (2026-10-16).

Two independent estimates per recording:
  torque   : the robot's external-torque estimate tau_ext_hat_filtered at rest = the weight the configured
             load leaves out:  tau_ext = J_p(q)^T (-m g z)  with p = the missing mass's CoM.
             Linear in (m, m c) with c in the robot0_right_hand frame, plus one constant offset per joint
             (torque sensor bias). Sign convention checked by the fit itself: m must come out > 0.
  position : the static PD error, K (q_cmd - q) = J_p^T (m g z) (calib_20261015/tools/fit_payload_sag_1016.py).
Also prints the load the controller was configured with (from the recorded robot_state).

    python fit_payload_probe.py <payload_probe.json>
    python fit_payload_probe.py --selftest        # synthetic data from the sim: recovers a known payload
"""
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
sys.path.insert(0, REPO)
K = np.array([240, 240, 240, 240, 100, 60, 20.])
G = 9.81
ZX = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0.]])


class Kin:
    def __init__(self):
        import env.threading_variants as tv
        from real2sim.deploy.policy_server import ReferenceArm
        self.arm = ReferenceArm("Threading_RealCellRobotiq", tv.REAL_CELL_ROBOTIQ_YAML)
        self.hand = self.arm.sim.model.body_name2id("robot0_right_hand")

    def rows(self, q, njoint=7):
        """-> A (njoint x 4) with  J_p^T (m g z) = A @ [m, m cx, m cy, m cz]  (c in the hand frame)."""
        import mujoco
        arm = self.arm
        m, d = arm.sim.model._model, arm.sim.data._data
        arm.set_state(np.asarray(q, np.float64), None, [0.0])
        mujoco.mj_forward(m, d)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jp, jr, self.hand)
        Jp, Jr = jp[:, arm.jvel][:, :njoint], jr[:, arm.jvel][:, :njoint]
        R = d.xmat[self.hand].reshape(3, 3)
        return np.c_[Jp.T @ (G * np.array([0, 0, 1.0])), Jr.T @ (-G * ZX @ R)]


def find(d, key):
    """first value whose key contains `key` (case-insensitive), depth-first"""
    if isinstance(d, dict):
        for k, v in d.items():
            if key in k.lower():
                return v
        for v in d.values():
            r = find(v, key)
            if r is not None:
                return r
    return None


def vec(v):
    if isinstance(v, dict):
        for k in ("effort", "data"):
            if k in v:
                return np.asarray(v[k], np.float64)
    return None if v is None else np.asarray(v, np.float64)


def fit(Arows, b, bias):
    """least squares on [m, mc] (+ per-joint bias); -> m, c (m), residual rms per joint"""
    n = Arows.shape[0] // 7
    A = Arows
    if bias:
        A = np.c_[Arows, np.tile(np.eye(7), (n, 1))]
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = (b - A @ x).reshape(n, 7)
    return x[0], x[1:4] / x[0], np.sqrt((r ** 2).mean(0)), (x[4:] if bias else None)


def analyse(rec, kin):
    st = rec.get("config_state") or {}
    for key in ("m_ee", "m_load", "m_total", "inertia_ee", "inertia_load"):
        v = find(st, key)
        if v is not None:
            print(f"configured {key}: {json.dumps(v)[:300]}")
    At, bt, Ap, bp = [], [], [], []
    for name, p in rec["poses"].items():
        q = np.mean(p["joint_q"], 0) if p.get("joint_q") else np.asarray(p["q_cmd"])
        A = kin.rows(q)
        taus = [vec(find(s, "tau_ext_hat_filtered")) for s in p.get("states", [])]
        taus = [t for t in taus if t is not None and len(t) == 7]
        if taus:
            At.append(-A)                                  # tau_ext = J^T(-m g z)
            bt.append(np.mean(taus, 0))
        if p.get("joint_q"):
            Ap.append(A)
            bp.append(K * (np.asarray(p["q_cmd"]) - q))
        print(f"  {name:16s} tau_ext {np.round(np.mean(taus, 0), 2).tolist() if taus else None}")
    if At:
        At, bt = np.concatenate(At), np.concatenate(bt)
        for bias in (False, True):
            m, c, rms, bb = fit(At, bt, bias)
            if m < 0:      # libfranka's sign convention is the other one: tau_ext = +J^T(m g z)
                m, c, rms, bb = fit(-At, bt, bias)
                print("  (sign: tau_ext_hat_filtered = +J^T(m g z) on this robot)")
            print(f"\nTORQUE fit{' + per-joint bias' if bias else ''}: m = {m:.3f} kg, CoM (hand frame) "
                  f"{np.round(1e3 * c, 1).tolist()} mm | residual rms {np.round(rms, 2).tolist()} N m"
                  + (f" | bias {np.round(bb, 2).tolist()}" if bias else ""))
    if Ap:
        Ap, bp = np.concatenate(Ap), np.concatenate(bp)
        keep = np.tile(np.r_[np.ones(6), 0.0], len(bp) // 7).astype(bool)      # j7 carries no gravity torque
        x, *_ = np.linalg.lstsq(Ap[keep], bp[keep], rcond=None)
        if abs(x[0]) > 1e-6:
            print(
                f"POSITION fit (static PD error, j1-j6): m = {x[0]:.3f} kg, CoM {np.round(1e3 * x[1:] / x[0], 1).tolist()} mm"
                " (CoM poorly determined: friction)")


def selftest(kin):
    """synthetic: plan from payload_probe.POSES, a known payload, sensor bias + noise -> recover it"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib.util
    spec = importlib.util.spec_from_file_location("pp_consts", os.path.join(os.path.dirname(__file__), "payload_probe.py"))
    src = open(spec.origin).read()
    ns = {}
    exec(src[src.index("HOME = "):src.index("@dataclass")], {"np": np}, ns)   # HOME + POSES only (no rclpy)
    rng = np.random.default_rng(0)
    m_true, c_true = 0.30, np.array([0.010, -0.008, 0.060])
    bias = rng.normal(0, 0.4, 7)
    poses = {}
    for name, dd in ns["POSES"].items():
        q = np.array(ns["HOME"]) + np.array([dd.get(j, 0.0) for j in range(7)])
        tau = -kin.rows(q) @ np.r_[m_true, m_true * c_true] + bias
        poses[name] = {"q_cmd": q.tolist(), "joint_q": [q.tolist()],
                       "states": [{"tau_ext_hat_filtered": (tau + rng.normal(0, 0.05, 7)).tolist()} for _ in range(20)]}
    print(f"selftest: true m {m_true} kg, CoM {1e3 * c_true} mm, bias {np.round(bias, 2).tolist()}")
    analyse({"poses": poses}, kin)


if __name__ == "__main__":
    kin = Kin()
    if sys.argv[1] == "--selftest":
        selftest(kin)
    else:
        analyse(json.load(open(sys.argv[1])), kin)

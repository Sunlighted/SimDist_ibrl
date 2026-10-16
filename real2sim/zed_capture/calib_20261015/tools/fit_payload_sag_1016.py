"""Fit the payload the FR3 gravity compensation is missing, from the static sag in the GELLO demos (2026-10-16).

The joint impedance controller is a pure PD on top of libfranka's gravity compensation:
    tau_ctrl = K (q_cmd - q) - D dq.
At rest tau_ctrl holds whatever weight the compensation does not know about. For a missing point mass m
with centre of mass c (in the robot0_right_hand frame), the torque that holds it is
    tau = J_p^T (m g z) + J_r^T ((R (m c)) x (m-free) g z)        -- linear in (m, m c).
Fit on the demos' near-static frames, joints 1-6 (j7 carries no gravity torque). Kinematics: the cell model.

    python fit_payload_sag_1016.py <raw episode dir>
"""
import glob
import sys

import h5py
import numpy as np

sys.path.insert(0, __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), *[".."] * 4)))
import mujoco  # noqa: E402
import env.threading_variants as tv  # noqa: E402
from real2sim.deploy.policy_server import ReferenceArm  # noqa: E402

K = np.array([240, 240, 240, 240, 100, 60, 20.])
G = 9.81
arm = ReferenceArm("Threading_RealCellRobotiq", tv.REAL_CELL_ROBOTIQ_YAML)
m, d = arm.sim.model._model, arm.sim.data._data
hand = arm.sim.model.body_name2id("robot0_right_hand")
jv = arm.jvel


def rows(q):
    arm.set_state(q, None, [0.0])
    mujoco.mj_forward(m, d)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jp, jr, hand)
    Jp, Jr = jp[:, jv][:, :6], jr[:, jv][:, :6]
    R = d.xmat[hand].reshape(3, 3)
    z = np.array([0, 0, 1.0])
    A_m = Jp.T @ (G * z)                                    # coefficient of m
    # torque (R mc) x (g z) = -g z x (R mc) = -g [z]x R mc
    zx = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0.]])
    A_c = Jr.T @ (-G * zx @ R)                               # coefficient of mc (3)
    return np.c_[A_m, A_c]


A, b, w = [], [], []
files = sorted(glob.glob(sys.argv[1].rstrip("/") + "/**/*.h5", recursive=True))
for fn in files:
    try:
        f = h5py.File(fn, "r")
    except OSError:
        print(f"{fn}: unreadable, skipped")
        continue
    q = f["observation/joint_positions"][:]
    c = f["action/joint_positions"][:]
    dq = f["observation/joint_velocities"][:]
    hz = float(f.attrs.get("sample_rate_hz", 20.0))
    dc = np.r_[np.zeros((1, 7)), np.diff(c, axis=0)] * hz
    still = (np.abs(dq).max(1) < 0.02) & (np.abs(dc).max(1) < 0.02)
    for t in np.nonzero(still)[0]:
        A.append(rows(q[t]))
        b.append((K * (c[t] - q[t]))[:6])
    if len(files) <= 10:
        print(f"{fn.split('/')[-1]}: {still.sum()} still frames")
A, b = np.concatenate(A), np.concatenate(b)
x, *_ = np.linalg.lstsq(A, b, rcond=None)
mass, com = x[0], x[1:] / x[0]
res = b - A @ x
print(f"\nmissing payload: m = {mass:.3f} kg, CoM in robot0_right_hand frame = {np.round(1e3 * com, 1).tolist()} mm")
print(f"per-joint torque rms before / after the fit (N m): {np.round(np.sqrt((b.reshape(-1, 6) ** 2).mean(0)), 2).tolist()}"
      f" -> {np.round(np.sqrt((res.reshape(-1, 6) ** 2).mean(0)), 2).tolist()}")
# predicted sag at the DROID home and at a typical grasp pose, as TCP displacement
for tag, qq in (("DROID home", np.array(arm.cell.init_qpos)),):
    tau = rows(qq) @ x
    dq_sag = -np.r_[tau / K[:6], 0.0]
    p0 = arm.site_pose(qq)[0]
    p1 = arm.site_pose(qq + dq_sag)[0]
    print(f"predicted static sag at {tag}: dq {np.round(1e3 * dq_sag, 1).tolist()} mrad, TCP {np.round(1e3 * (p1 - p0), 1).tolist()} mm")

# CoM poorly constrained by these poses -> fix it on the flange axis and fit the mass only
arm.set_state(np.array(arm.cell.init_qpos), None, [0.0])
Rh = arm.sim.data.body_xmat[hand].reshape(3, 3)
tcp_h = Rh.T @ (arm.eef() - arm.sim.data.body_xpos[hand])
print(f"\n(grip site in the hand frame: {np.round(1e3 * tcp_h, 1).tolist()} mm)")
for zc in (0.0, 0.04, 0.06, 0.08, 0.10, 0.12):
    cvec = np.array([0, 0, zc])
    a = A[:, 0] + A[:, 1:] @ cvec
    mm = float(a @ b / (a @ a))
    r = b - a * mm
    print(f"  CoM on the hand axis at {1e3 * zc:5.0f} mm: m = {mm:.3f} kg, j2 rms {np.sqrt((r.reshape(-1, 6)[:, 1] ** 2).mean()):.2f}, "
          f"all rms {np.sqrt((r ** 2).mean()):.3f} N m")

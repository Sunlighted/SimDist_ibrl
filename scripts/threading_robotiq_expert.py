"""Scripted expert for Threading on the Robotiq real cell (TRAIN_PLAN.md stage 1, 2026-10-19).

Why not MimicGen: its source demos are Panda-hand ones. With the Robotiq the needle is lifted every time but sits
differently in the hand, and MimicGen replays the threading segment as an end-effector path relative to the tripod,
assuming the source's in-hand pose -> the needle ends 2-18 cm from the ring (generation 8-10 % vs 53-60 % for the
Panda cell; 2x2 gripper x rate in calib_20261015 README). This expert closes the loop on the ACTUAL in-hand pose.

Privileged state (sim only): needle body pose, ring centre + normal. Phases:
  above -> descend -> close -> lift -> align (needle on the ring axis, tip 6 cm before the ring) -> insert (along the
  axis until the shaft centre passes the ring centre) -> hold.
Actions are OSC deltas on the REFERENCE arm (the cell's execution layer); a phase only advances once the MEASURED grip
site has converged (or a timeout), so the arm's 0.2-0.3 s lag does not cost precision.

    python scripts/threading_robotiq_expert.py --episodes 20          # success rate, per tripod-yaw bin
"""
import argparse
import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.grasp_script import HANDLE_OFFSET, TIP_DROP_OPEN, _axisangle, _rz  # noqa: E402

POS_SCALE, ROT_SCALE = 0.05, 0.5
TIP_LOCAL = np.array([0.0, -0.08, 0.0])       # needle tip in the needle body frame (shaft -80..+40 mm on y)
SHAFT_CENTRE_LOCAL = np.array([0.0, -0.02, 0.0])


def _rot_between(a, b):
    """minimal rotation taking unit vector a to unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        p = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        v = np.cross(a, p)
        v /= np.linalg.norm(v)
        return 2 * np.outer(v, v) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


class ThreadingExpert:
    # grasp_h 0.020 (2026-10-22, was 0.010): at 0.010 the open fingertips came within ~1 mm of the table, so a BC-RNN
    # imitating it hit the table (> 5 N, cell table_impact_force) in 36-48 % of episodes. 0.020: fingertips ~11 mm
    # clear, 0 table contacts in 40 episodes, strict 300/300 (same as 0.010 and 0.016).
    def __init__(self, e, grasp_h=0.020, above=0.08, lift_clear=0.06, pre=0.06, gain=1.0, amp=0.5, rot_amp=0.4,
                 noise=0.0, rng=None):
        self.e = e
        self.p = dict(grasp_h=grasp_h, above=above, lift_clear=lift_clear, pre=pre)
        self.gain, self.amp, self.rot_amp, self.noise = gain, amp, rot_amp, noise
        self.rng = np.random if rng is None else rng
        m = e.sim.model
        self.rob = e.robots[0]
        self.sid = self.rob.eef_site_id
        self.nb = m.body_name2id("needle_obj_root")
        self.ng = m.geom_name2id("needle_obj_needle")
        self.rg = [m.geom_name2id(f"tripod_obj_ring_{i}") for i in range(e.tripod.num_ring_geoms)]
        self.lf = m.body_name2id("gripper0_left_inner_finger")
        self.rf = m.body_name2id("gripper0_right_inner_finger")
        self.tid = m.geom_name2id("table_collision")
        self.phase, self.k, self.done = "above", 0, False
        self.grasp_R = None
        self.insert_dir = None
        self.frozen = None          # (in-hand transform, needle target rotation), fixed at the start of align
        self.dir_override = None    # insertion direction set by takeover() (else chosen by the tip's side of the ring)
        body = m.geom_bodyid
        self.needle_geoms = {g for g in range(m.ngeom) if body[g] == self.nb}
        self.lf_geoms = {g for g in range(m.ngeom) if body[g] == self.lf}
        self.rf_geoms = {g for g in range(m.ngeom) if body[g] == self.rf}

    # ---------------------------------------------------------------- state
    def meas(self):
        d = self.e.sim.data
        return d.site_xpos[self.sid].copy(), d.site_xmat[self.sid].reshape(3, 3).copy()

    def ref(self):
        L = getattr(self.e, "exec_layer", None)
        if L is None:
            return self.meas()
        T = L.ref_site_pose()
        return T[:3, 3], T[:3, :3]

    def needle(self):
        d = self.e.sim.data
        return d.xpos[self.nb].copy(), d.xmat[self.nb].reshape(3, 3).copy()

    def ring(self):
        d = self.e.sim.data
        P = np.array([d.geom_xpos[g] for g in self.rg])
        c = P.mean(0)
        _, _, vt = np.linalg.svd(P - c)
        return c, vt[2] / np.linalg.norm(vt[2])

    def table_top(self):
        d, m = self.e.sim.data, self.e.sim.model
        return float(d.geom_xpos[self.tid][2] + m.geom_size[self.tid][2])

    def touching(self):
        """(left finger, right finger) in contact with the needle."""
        d = self.e.sim.data
        L = R = False
        for i in range(d.ncon):
            c = d.contact[i]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a in self.needle_geoms:
                    L |= b in self.lf_geoms
                    R |= b in self.rf_geoms
        return L, R

    def takeover(self):
        """Start the phase machine from the CURRENT state instead of the reset (DAgger labels / recovery demos,
        2026-10-24). Needle held (both fingers on it, gripper commanded closed):
          shaft already in the ring (tip at/past the ring plane, ring centre < 10 mm off the shaft line) -> insert,
          continuing from the current depth; tip within 10 cm of the ring -> coarse; else -> lift.
        Insertion direction = the ring normal the tip points along (the reset's tip-side rule flips once the tip is
        past the ring plane). Not held: at the grasp pose -> close; low and away from the handle -> rise (straight up,
        open) so the open fingers do not sweep the needle or tripod; else -> above."""
        self.k, self.done, self.frozen, self.grasp_R, self.ok_streak = 0, False, None, None, 0
        self.dir_override = self.insert_dir = None
        L, R = self.touching()
        ga = float(np.mean(np.atleast_1d(getattr(self.rob.gripper, "current_action", [-1.0]))))
        if L and R and ga > 0.3:
            pn, Rn = self.needle()
            c, n = self.ring()
            tip, axis = pn + Rn @ TIP_LOCAL, Rn[:, 1]
            d_pt = n if float(n @ -axis) > 0 else -n
            v = c - tip
            rad = float(np.linalg.norm(v - (v @ axis) * axis))
            ax_tip = float((tip - c) @ d_pt)
            if ax_tip > -0.003 and rad < 0.010:
                self.phase, self.insert_dir = "insert", d_pt
                self.k = max(0, int(np.ceil((ax_tip + self.p["pre"]) / 0.0015)))
            elif np.linalg.norm(tip - c) < 0.10:
                self.phase, self.dir_override = "coarse", d_pt
            else:
                self.phase = "lift"
        else:
            handle, z_grasp, top = self._grasp_targets()
            pe, _ = self.meas()
            xy = float(np.linalg.norm(pe[:2] - handle[:2]))
            if xy < 0.006 and abs(pe[2] - z_grasp) < 0.006:
                self.phase = "close"
            elif pe[2] < top + 0.06 and xy > 0.01:
                self.phase = "rise"
            else:
                self.phase = "above"
        return self.phase

    # ---------------------------------------------------------------- targets
    def _grasp_targets(self):
        pn, Rn = self.needle()
        handle = pn + Rn @ HANDLE_OFFSET
        if self.grasp_R is None:                       # fixed once: top-down, closing axis square to the cube
            _, R0 = self.meas()
            d = self.e.sim.data
            ax = d.xpos[self.rf] - d.xpos[self.lf]
            cur = np.arctan2(ax[1], ax[0])
            want = np.arctan2(Rn[1, 0], Rn[0, 0])
            # closing axis along the needle body x (square to the shaft), either sign. Mod pi/2 also allowed the two
            # faces PARALLEL to the shaft and picked them for needle yaw < ~-118 deg (14/1000 demos of
            # threading_robotiq_expert_n1000 grasped along the shaft, 2026-10-21).
            dz = (want - cur + np.pi / 2) % np.pi - np.pi / 2
            down = np.array([0, 0, -1.0])
            tilt = _rot_between(R0[:, 2], down)       # straighten the approach axis
            self.grasp_R = _rz(dz) @ tilt @ R0
        top = self.table_top()
        z_grasp = top + self.p["grasp_h"] + TIP_DROP_OPEN
        return handle, z_grasp, top

    def _thread_target(self, depth):
        """EEF pose that puts the needle tip at ring_centre + depth * a (a = insertion direction), with the needle axis
        on the ring axis. The in-hand transform and the needle's target rotation are FROZEN at the start of align:
        re-deriving them every step flips the 'minimal rotation' as soon as the needle turns in the hand on contact
        (a 200 mm target jump in the first version)."""
        pe, Re = self.meas()
        pn, Rn = self.needle()
        c, n = self.ring()
        if self.frozen is None:
            tip = pn + Rn @ TIP_LOCAL
            self.insert_dir = self.dir_override if self.dir_override is not None else (
                n if np.dot(n, c - tip) > 0 else -n)
            R_des = _rot_between(Rn[:, 1], -self.insert_dir) @ Rn      # body +y: tip -> handle
            self.frozen = (Rn.T @ (pe - pn), Rn.T @ Re, R_des)
        e_in_n_p, e_in_n_R, R_des = self.frozen
        tip_des = c + depth * self.insert_dir
        pn_des = tip_des - R_des @ TIP_LOCAL
        return pn_des + R_des @ e_in_n_p, R_des @ e_in_n_R

    # ---------------------------------------------------------------- policy
    def _needle_errors(self, depth):
        """tip error to ring_centre + depth * a (split axial / radial) and the needle-axis misalignment (rad)."""
        pn, Rn = self.needle()
        c, _ = self.ring()
        a = self.insert_dir
        tip = pn + Rn @ TIP_LOCAL
        e = (c + depth * a) - tip
        ax = float(e @ a)
        rad = e - ax * a
        ang = np.arccos(np.clip(float(-Rn[:, 1] @ a), -1.0, 1.0))
        return e, ax, rad, ang, Rn

    def act(self):
        if self.done:
            return np.r_[np.zeros(6), 1.0], {"phase": "done"}
        pe, Re = self.meas()
        pr, Rr = self.ref()
        a = np.zeros(7)
        a[6] = -1.0
        info = {"phase": self.phase, "k": self.k}
        nxt = None
        if self.phase in ("rise", "above", "descend", "close", "lift"):
            handle, z_grasp, top = self._grasp_targets()
            R_t = self.grasp_R
            if self.phase == "rise":           # takeover only: straight up, open, current orientation
                p_t, R_t = np.r_[pe[:2], top + 0.04 + self.p["above"] + TIP_DROP_OPEN], Re
            elif self.phase == "above":
                p_t = np.r_[handle[:2], top + 0.04 + self.p["above"] + TIP_DROP_OPEN]
            elif self.phase == "descend":
                p_t = np.r_[handle[:2], z_grasp]
            elif self.phase == "close":
                p_t, a[6] = np.r_[handle[:2], z_grasp], 1.0
            else:
                c, _ = self.ring()
                p_t, a[6], R_t = np.r_[pe[:2], max(c[2], top) + self.p["lift_clear"]], 1.0, Re
            err = np.linalg.norm(p_t - pe)
            a[:3] = np.clip(self.gain * (p_t - pr) / POS_SCALE, -self.amp, self.amp)
            a[3:6] = np.clip(_axisangle(R_t @ Rr.T) / ROT_SCALE, -self.rot_amp, self.rot_amp)
            done_ph = (self.phase == "close" and self.k >= 8) or (self.phase == "rise" and (err < 0.01 or self.k >= 40)) \
                or (self.phase not in ("close", "rise") and (err < 0.004 or self.k >= 60))
            nxt = {"rise": "above", "above": "descend", "descend": "close", "close": "lift", "lift": "coarse"}[self.phase] if done_ph else None
            info["err_mm"] = round(1e3 * err, 1)
        elif self.phase == "coarse":
            # frozen in-hand transform, reference servo: a smooth large move to 6 cm before the ring
            a[6] = 1.0
            p_t, R_t = self._thread_target(-self.p["pre"])
            a[:3] = np.clip(self.gain * (p_t - pr) / POS_SCALE, -self.amp, self.amp)
            a[3:6] = np.clip(_axisangle(R_t @ Rr.T) / ROT_SCALE, -self.rot_amp, self.rot_amp)
            _, _, rad, ang, _ = self._needle_errors(-self.p["pre"])
            err = np.linalg.norm(p_t - pe)
            if (err < 0.015 and ang < np.radians(10)) or self.k >= 80:
                nxt = "fine"
            info.update(err_mm=round(1e3 * err, 1), rad_mm=round(1e3 * np.linalg.norm(rad), 1),
                        ang_deg=round(np.degrees(ang), 1))
        else:
            # fine / insert: close the loop on the NEEDLE itself -- move the reference by a fraction of the tip error,
            # rotate it by a fraction of the needle-axis error (small angles here: no flip)
            a[6] = 1.0
            if self.phase == "fine":
                depth = -self.p["pre"]
            else:
                depth = min(-self.p["pre"] + 0.0015 * self.k, 0.07)
            e, ax, rad, ang, Rn = self._needle_errors(depth)
            k_p = 0.3
            a[:3] = np.clip(k_p * e / POS_SCALE, -0.3, 0.3)
            R_corr = _rot_between(Rn[:, 1], -self.insert_dir)
            a[3:6] = np.clip(k_p * _axisangle(R_corr) / ROT_SCALE, -0.2, 0.2)
            self.ok_streak = getattr(self, "ok_streak", 0) + 1 if (
                np.linalg.norm(rad) < 0.0015 and abs(ax) < 0.005 and ang < np.radians(1.5)) else 0
            if self.phase == "fine" and (self.ok_streak >= 3 or self.k >= 80):
                nxt = "insert"
            elif self.phase == "insert" and self.k >= 120:
                nxt = "hold"
            elif self.phase == "hold":
                nxt = None
            info.update(rad_mm=round(1e3 * np.linalg.norm(rad), 1), ax_mm=round(1e3 * ax, 1),
                        ang_deg=round(np.degrees(ang), 2))
        if self.noise:
            a[:6] += self.rng.normal(0, self.noise, 6)
        self.k += 1
        if nxt is not None:
            self.phase, self.k = nxt, 0
        return np.clip(a, -1, 1), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    import torch
    import env.threading_variants as tv
    from env.robosuite_wrapper import PixelRobosuite
    from env.sim2real import load_sim2real
    cell = load_sim2real(tv.REAL_CELL_ROBOTIQ_YAML)
    env = PixelRobosuite("Threading_RealCellRobotiq", "Panda", args.max_steps, camera_names=[], rl_cameras=[],
                         use_state=True, state_only_no_render=True, device="cpu", hand_init_perturb_joint_sigma=0.1,
                         sim2real=cell)
    rows = []
    for ep in range(args.episodes):
        np.random.seed(args.seed + ep)
        env.reset()
        e = env.env
        ex = ThreadingExpert(e, noise=args.noise)
        tb = e.sim.model.body_name2id("tripod_obj_root")
        R = e.sim.data.body_xmat[tb].reshape(3, 3)
        tyaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        succ, steps, phases = False, 0, []
        for t in range(args.max_steps):
            a, info = ex.act()
            if not phases or phases[-1][0] != info["phase"]:
                phases.append((info["phase"], t))
            _, r, term, succ, _ = env.step(torch.tensor(a, dtype=torch.float32))
            steps = t + 1
            if args.verbose and t % 10 == 0:
                print(t, info)
            if succ or term:
                break
        rows.append((succ, steps, tyaw, phases[-1][0]))
        print(f"ep {ep:3d}: {'OK  ' if succ else 'FAIL'} steps {steps:3d} tripod yaw {tyaw:6.1f} | last phase "
              f"{phases[-1][0]} | {' '.join(f'{p}@{t}' for p, t in phases)}", flush=True)
    s = np.array([r[0] for r in rows])
    print(f"[expert] success {s.sum()}/{len(s)} = {s.mean():.2f}, mean steps (ok) "
          f"{np.mean([r[1] for r in rows if r[0]]) if s.any() else float('nan'):.0f}")


if __name__ == "__main__":
    main()

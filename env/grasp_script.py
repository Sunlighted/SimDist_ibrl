"""Scripted grasp grid on the Threading Robotiq cell -- the same script on the robot and in the sim (2026-10-17).

Why: the sim grasp is decided by the finger contact model, which nothing measured so far constrains (calib_20261015
README "Object poses"). Five human demos cannot pin it (each contact setting fits a different 2-3 of them). A fixed
script removes the human: per trial, top-down to the needle handle at a grid offset, close, lift 10 cm.

Placement: the operator puts the needle back only APPROXIMATELY (tape outline). Each trial therefore measures it: the
arm hovers above the handle and the server saves the raw wrist frame (close range, ~0.25 mm/px) for an offline pose
fit. The analysis bins trials by the REALISED offset (command - measured pose), and the sim replays every trial with its
own measured pose (scripts/grasp_twin.py).

Determinism: phases advance by TIME (steps), and the servo acts on the REFERENCE grip-site pose (the execution layer's
reference arm; no speed caps on this cell), so the action sequence depends only on the script, the nominal needle pose
and the reference -- the sim replays the logged actions exactly.

Frames: needle pose = (x, y, yaw) of the needle body in the ROBOT BASE frame (as demo_layout/*.json). Handle centre =
body origin + Rz(yaw) @ (0, 0.06, 0) (MimicGen needle: handle geom at +60 mm on the body's y axis), 40 mm cube.
"""
import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

HANDLE_OFFSET = np.array([0.0, 0.06, 0.0])
HANDLE_SIZE = 0.04
POS_SCALE, ROT_SCALE = 0.05, 0.5          # OSC_POSE output_max per unit action
# grip site -> lowest fingertip point, gripper OPEN and settled (driver ~ -0.05; sim, 2026-10-17). A CONSTANT, not
# measured per process: writing only the two Robotiq driver joints leaves the passive linkage wherever it was, so a
# kinematic measurement gave 4.1 mm in one env and 9.6 mm in policy_server's reference arm -- the robot and the sim
# would then descend to different heights and stop running the same script.
TIP_DROP_OPEN = 0.0076


def default_grid() -> List[Tuple[float, float, float]]:
    """(fingertip height above the table, offset along the closing axis, offset across it) in metres: 15 trials."""
    g = [(h, dc, 0.0) for h, dc in itertools.product((0.005, 0.010, 0.015), (-0.006, 0.0, 0.006))]
    g += [(0.010, 0.0, dp) for dp in (-0.008, 0.008, -0.008, 0.008)]
    g += [(0.010, 0.0, 0.0)] * 2
    return g


@dataclass
class GraspGeometry:
    """From the cell model, once: the table top and the base pose in world, the grip site -> lowest fingertip drop
    (gripper open), the closing axis in the grip-site frame, the home grip-site orientation."""
    base_p: np.ndarray
    base_R: np.ndarray
    table_top_base_z: float
    tip_drop: float
    close_axis_site: np.ndarray
    home_R: np.ndarray

    @classmethod
    def from_sim(cls, sim, robot, home_q):
        import mujoco
        m, d = sim.model, sim.data
        mm, dd = m._model, d._data
        jidx = np.asarray(robot._ref_joint_pos_indexes)
        save = dd.qpos.copy()
        dd.qpos[jidx] = home_q
        mujoco.mj_forward(mm, dd)
        bid = m.body_name2id(robot.robot_model.root_body)
        base_p, base_R = dd.xpos[bid].copy(), dd.xmat[bid].reshape(3, 3).copy()
        sid = robot.eef_site_id
        p_site, R_site = dd.site_xpos[sid].copy(), dd.site_xmat[sid].reshape(3, 3).copy()
        low = np.inf
        for g in range(mm.ngeom):
            n = m.geom_id2name(g) or ""
            if n.startswith("gripper0") and "fingertip_collision" in n:
                Rg = dd.geom_xmat[g].reshape(3, 3)
                low = min(low, dd.geom_xpos[g][2] - (np.abs(Rg) @ mm.geom_size[g])[2])
        lf = m.body_name2id("gripper0_left_inner_finger")
        rf = m.body_name2id("gripper0_right_inner_finger")
        ax = dd.xpos[rf] - dd.xpos[lf]
        ax = R_site.T @ (ax / np.linalg.norm(ax))
        tid = m.geom_name2id("table_collision")
        top = dd.geom_xpos[tid][2] + mm.geom_size[tid][2]
        dd.qpos[:] = save
        mujoco.mj_forward(mm, dd)
        return cls(base_p, base_R, float((base_R.T @ (np.array([0, 0, top]) - base_p))[2]),
                   TIP_DROP_OPEN, ax, R_site)


def _axisangle(R):
    from robosuite.utils.transform_utils import mat2quat, quat2axisangle
    return np.asarray(quat2axisangle(mat2quat(R)))


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


@dataclass
class GraspScript:
    geo: GraspGeometry
    needle: Tuple[float, float, float]                  # nominal (x, y, yaw deg), base frame
    grid: List[Tuple[float, float, float]] = field(default_factory=default_grid)
    hz: float = 10.0
    above: float = 0.08            # grip-site clearance above the handle top while approaching / hovering
    lift: float = 0.10
    amp: float = 0.4               # max |action| per axis (~8 cm/s on the 10 Hz reference: human-like)
    rot_amp: float = 0.3
    # phase durations (s); the trial ends in "wait" (robot: until the operator presses Enter; sim: skipped)
    durations: Tuple = (("to_above", 3.0), ("hover", 1.5), ("descend", 2.5), ("close", 1.0), ("lift", 2.0),
                        ("hold_up", 1.0), ("lower", 2.0), ("release", 0.8), ("retreat", 2.0), ("wait", 0.0))
    wait_forever: bool = True

    def __post_init__(self):
        self.trial, self.pi, self.k = 0, 0, 0
        self.done = False
        self.released = False      # robot: set by the operator thread to leave "wait"
        self.steps = [(n, int(round(s * self.hz))) for n, s in self.durations]

    # ------------------------------------------------------------- geometry
    def _to_world(self, pb):
        return self.geo.base_p + self.geo.base_R @ pb

    def handle_base(self, needle=None):
        x, y, yaw = needle if needle is not None else self.needle
        return np.array([x, y, 0.0]) + _rz(np.radians(yaw)) @ HANDLE_OFFSET

    def grasp_R(self):
        """Top-down at the home tilt, turned about world z so the closing axis is square to the cube faces
        (the closest of the four 90-degree choices)."""
        ax_w = self.geo.home_R @ self.geo.close_axis_site
        cur = np.arctan2(ax_w[1], ax_w[0])
        want = np.radians(self.needle[2]) + np.arctan2(*(self.geo.base_R[:2, 0][::-1]))   # needle x axis in world
        d = (want - cur + np.pi / 4) % (np.pi / 2) - np.pi / 4
        return _rz(d) @ self.geo.home_R

    def targets(self, trial):
        """-> dict phase -> (world grip-site position, gripper -1/+1) for this trial."""
        h, dc, dp = self.grid[trial]
        R = self.grasp_R()
        ax = R @ self.geo.close_axis_site
        ax[2] = 0.0
        ax /= np.linalg.norm(ax)
        perp = np.array([-ax[1], ax[0], 0.0])
        c = self._to_world(self.handle_base())
        c[:2] += dc * ax[:2] + dp * perp[:2]
        top = self._to_world(np.array([0, 0, self.geo.table_top_base_z]))[2]
        z_grasp = top + h + self.geo.tip_drop
        z_above = top + HANDLE_SIZE + self.above + self.geo.tip_drop
        P = lambda z: np.array([c[0], c[1], z])
        return {"to_above": (P(z_above), -1), "hover": (P(z_above), -1), "descend": (P(z_grasp), -1),
                "close": (P(z_grasp), 1), "lift": (P(z_grasp + self.lift), 1), "hold_up": (P(z_grasp + self.lift), 1),
                "lower": (P(z_grasp + 0.005), 1), "release": (P(z_grasp + 0.005), -1),
                "retreat": (P(z_above), -1), "wait": (P(z_above), -1)}, R

    # ------------------------------------------------------------- policy
    def act(self, ref_p, ref_R):
        """ref_p / ref_R: the REFERENCE grip site (world). -> (7-d action, info)."""
        a = np.zeros(7)
        a[6] = -1.0
        if self.done:
            return a, {"phase": "done"}
        name, n = self.steps[self.pi]
        if name == "wait" and (self.released or not self.wait_forever) or (name != "wait" and self.k >= n):
            self.pi, self.k = self.pi + 1, 0
            if self.pi >= len(self.steps):
                self.trial, self.pi, self.released = self.trial + 1, 0, False
                if self.trial >= len(self.grid):
                    self.done = True
                    return a, {"phase": "done"}
            return self.act(ref_p, ref_R)
        tg, R = self.targets(self.trial)
        p_t, g = tg[name]
        a[:3] = np.clip((p_t - ref_p) / POS_SCALE, -self.amp, self.amp)
        a[3:6] = np.clip(_axisangle(R @ ref_R.T) / ROT_SCALE, -self.rot_amp, self.rot_amp)
        a[6] = float(g)
        info = {"phase": name, "trial": self.trial, "k": self.k, "grid": list(self.grid[self.trial]),
                "target": np.round(p_t, 5).tolist(), "needle_nominal": list(self.needle),
                "save_raw": bool(name == "hover" and self.k == n - 1)}
        self.k += 1
        return a, info

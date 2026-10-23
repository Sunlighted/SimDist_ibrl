"""OmniReset-style start states for Threading on the Robotiq real cell, generated the way UWLab does it (2026-10-23).

UWLab (~/UWLab/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset): reset_states_cfg.py +
partial_assemblies_cfg.py + mdp/events.py (MultiResetManager) + mdp/terminations.py (check_reset_state_success).
Their recipe, and what it is here:

  ObjectAnywhereEEAnywhere          -> "fresh": the cell's own reset (needle on the table, arm at the perturbed home)
  ObjectRestingEEGrasped            -> "resting_grasped": needle where the reset put it, gripper from the grasp
                                       dataset + noise +-2 cm / +-pi/16 (UWLab's pose_range_b)
  ObjectAnywhereEEGrasped           -> "anywhere_grasped": needle uniform in a box over the workspace, z 0..0.3 m
                                       above the table, orientation uniform in SO(3); grasp from the dataset, no noise
  ObjectPartiallyAssembledEEGrasped -> "partial_grasped": needle from a DISASSEMBLY rollout (UWLab
                                       record_partial_assemblies: start assembled, zero friction, no gravity, small
                                       random force/torque, log poses while still overlapping), 50 % required near
                                       the assembled pose (1.5 x the success threshold) and 50 % not
                                       (assembly_success_prob 0.5, assembly_threshold_scale 1.5); grasp + noise
                                       +-1 cm / +-pi/32

The grasp dataset (UWLab record_grasps) = in-hand transforms of the scripted expert's grasps (needle pose in the grip
site frame + the 6 gripper joints), recorded just after it lifts the needle. The arm is put there by damped least
squares IK (env/exec_layer.ik_site) and written into the sim together with the needle and the gripper joints
("teleport"). Validity = check_reset_state_success: collision analyzer thresholds (robot-needle >= -1.5 mm, where
UWLab has -0.5 mm: this cell's gripper contact holds ~1 mm deep, robot-tripod >= 0, needle-tripod >= -1 mm, robot-table >= 0), gripper approach axis within 60 deg of straight down,
5 consecutive stable steps (needle |v| sum < 0.1 m/s, |w| sum < 1 rad/s) while the gripper holds, needle deviation
below the type's limit (resting 0.01 / anywhere 0.05 / partial 0.025 m), grip site deviation < 0.1 m, needle above the
table.

A state is (reset seed, teleport spec, settle actions): the training loop re-seeds the reset, writes the spec and
replays the settle actions, which is bit-exact (the exec layer's reference arm and the DR draws are not in sim_state,
so states cannot be written back directly).
"""
import copy

import numpy as np
import torch

TIP_LOCAL = np.array([0.0, -0.08, 0.0])          # needle body frame: shaft -80..+40 mm on y, handle cube at +60 mm
SHAFT_CENTRE_LOCAL = np.array([0.0, -0.02, 0.0])  # = the needle_obj_needle geom: what _check_success measures
SHAFT_ENDS = (-0.08, 0.04)
GRASP_NOISE = {"resting_grasped": (0.02, np.pi / 16), "anywhere_grasped": (0.0, 0.0),
               "partial_grasped": (0.01, np.pi / 32)}
MAX_OBJ_DEV = {"resting_grasped": 0.01, "anywhere_grasped": 0.05, "partial_grasped": 0.025}
SETTLE_MAX, STABLE_STEPS = 12, 5


def _quat_wxyz(R):
    from robosuite.utils.transform_utils import mat2quat
    x, y, z, w = mat2quat(R)
    return np.array([w, x, y, z])


def _mat(q_wxyz):
    from robosuite.utils.transform_utils import quat2mat
    w, x, y, z = q_wxyz
    return quat2mat(np.array([x, y, z, w]))


def _rand_rot(rng, max_angle):
    if max_angle <= 0:
        return np.eye(3)
    from robosuite.utils.transform_utils import euler2mat
    return euler2mat(rng.uniform(-max_angle, max_angle, 3))


def _rot_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        p = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        v = np.cross(a, p); v /= np.linalg.norm(v)
        return 2 * np.outer(v, v) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1 - c) / np.linalg.norm(v) ** 2


class OmniResetGenerator:
    def __init__(self, wrapper, seeded_reset, rng):
        self.w, self.reset, self.rng = wrapper, seeded_reset, rng
        e = wrapper.env
        m = e.sim.model
        rob = e.robots[0]
        self.nb = m.body_name2id("needle_obj_root")
        self.tb = m.body_name2id("tripod_obj_root")
        self.sid = rob.eef_site_id
        self.jidx = np.asarray(rob._ref_joint_pos_indexes)
        self.jvel = np.asarray(rob._ref_joint_vel_indexes)
        self.gidx = np.asarray(rob._ref_gripper_joint_pos_indexes)
        self.ring_r = float(e.tripod.ring_size[1])
        self.grasps: list = []

    # ------------------------------------------------------------------ scene queries (main sim, read only)
    @property
    def e(self):
        return self.w.env

    @property
    def rob(self):
        # NOT cached: a hard reset rebuilds the gripper model object (its incremental command lives there)
        return self.w.env.robots[0]

    def _ids(self):
        m = self.e.sim.model
        kinds = []
        for g in range(m.ngeom):
            n = m.geom_id2name(g) or ""
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                kinds.append(None)
            elif n.startswith("gripper0") or n.startswith("robot0"):
                kinds.append("robot")
            elif n.startswith("needle_obj"):
                kinds.append("needle")
            elif n.startswith("tripod_obj"):
                kinds.append("tripod")
            elif n == "table_collision":
                kinds.append("table")
            else:
                kinds.append(None)
        return kinds

    def ring(self, d=None):
        e = self.e
        d = e.sim.data._data if d is None else d
        m = e.sim.model
        P = np.array([d.geom_xpos[m.geom_name2id(f"tripod_obj_ring_{i}")] for i in range(e.tripod.num_ring_geoms)])
        c = P.mean(0)
        n = np.linalg.svd(P - c)[2][2]
        return c, n / np.linalg.norm(n)

    def table_top(self):
        m, d = self.e.sim.model, self.e.sim.data
        t = m.geom_name2id("table_collision")
        return float(d.geom_xpos[t][2] + m.geom_size[t][2])

    def needle_pose(self):
        d = self.e.sim.data
        return d.xpos[self.nb].copy(), d.xmat[self.nb].reshape(3, 3).copy()

    def site_pose(self):
        d = self.e.sim.data
        return d.site_xpos[self.sid].copy(), d.site_xmat[self.sid].reshape(3, 3).copy()

    # ------------------------------------------------------------------ grasp dataset (UWLab record_grasps)
    def collect_grasps(self, n):
        """in-hand transforms of the scripted expert's grasps, recorded 3 steps into its lift"""
        from scripts.threading_robotiq_expert import ThreadingExpert
        tries = 0
        while len(self.grasps) < 4 * n and tries < 3 * n:
            tries += 1
            self.reset(int(self.rng.randint(2**31 - 1)), None)
            ex = ThreadingExpert(self.e)
            pn0 = self.needle_pose()[0]
            for _ in range(150):
                a, info = ex.act()
                _, _, term, _, _ = self.w.step(torch.tensor(a, dtype=torch.float32))
                if term:
                    break
                if info["phase"] == "lift" and info["k"] >= 3:
                    pn, Rn = self.needle_pose()
                    if pn[2] - pn0[2] > 0.01:
                        ps, Rs = self.site_pose()
                        # the handle is a cube centred ON the shaft axis (body y): turning the needle about its
                        # own axis by k * 90 deg is the same grasp on another pair of faces (UWLab's grasp
                        # dataset covers every side of the object)
                        for k in range(4):
                            c, s_ = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
                            Ry = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
                            self.grasps.append(dict(p=Rs.T @ (pn - ps), R=Rs.T @ Rn @ Ry,
                                                    g=np.array(self.e.sim.data.qpos[self.gidx]),
                                                    ga=np.array(self.rob.gripper.current_action, copy=True)))
                    break
        return len(self.grasps)

    # ------------------------------------------------------------------ needle pose samplers
    def _box(self):
        """UWLab samples the object over the table region in front of the robot: here the box spanning the needle's
        reset region and the tripod, +-10 cm"""
        pn = self.needle_pose()[0]
        c, _ = self.ring()
        lo, hi = np.minimum(pn, c)[:2] - 0.10, np.maximum(pn, c)[:2] + 0.10
        return lo, hi

    def sample_anywhere(self):
        lo, hi = self._box()
        from robosuite.utils.transform_utils import euler2mat
        p = np.r_[self.rng.uniform(lo, hi), self.table_top() + self.rng.uniform(0.0, 0.3)]
        return p, euler2mat(self.rng.uniform(-np.pi, np.pi, 3))

    def sample_partial(self):
        """disassembly (UWLab record_partial_assemblies): start assembled (shaft centre on the ring centre, shaft on
        the ring axis, the needle's reset orientation turned minimally), zero gravity / friction, robot contacts off,
        tripod held, random force + torque re-drawn at 120 Hz; log every pose that still overlaps the ring"""
        import mujoco
        e = self.e
        m0 = e.sim.model._model
        m = copy.deepcopy(m0)
        d = mujoco.MjData(m)
        d.qpos[:] = e.sim.data._data.qpos
        mujoco.mj_forward(m, d)
        m.opt.gravity[:] = 0.0
        kinds = self._ids()
        for g, k in enumerate(kinds):
            if k == "robot":
                m.geom_contype[g] = m.geom_conaffinity[g] = 0
            elif k in ("needle", "tripod"):
                m.geom_friction[g] = 0.0
        c, n = self.ring(d)
        base = e.sim.data.body_xpos[e.sim.model.body_name2id(self.rob.robot_model.root_body)]
        a = n if np.dot(n[:2], (c - base)[:2]) > 0 else -n          # tip pointing away from the robot
        _, Rn0 = self.needle_pose()
        R = _rot_between(Rn0[:, 1], -a) @ Rn0                        # body +y (tip -> handle) = -a
        p = c - R @ SHAFT_CENTRE_LOCAL
        ja = m.jnt_qposadr[m.body_jntadr[self.nb]]
        jv = m.jnt_dofadr[m.body_jntadr[self.nb]]
        ta = m.jnt_qposadr[m.body_jntadr[self.tb]]
        tv = m.jnt_dofadr[m.body_jntadr[self.tb]]
        tq = d.qpos[ta:ta + 7].copy()
        d.qpos[ja:ja + 3], d.qpos[ja + 3:ja + 7] = p, _quat_wxyz(R)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        every = max(1, int(round(1 / (120 * m.opt.timestep))))
        near, far = [], []
        f_mag, t_mag = 0.005, 0.0005
        for k in range(6000):
            if k % every == 0:
                d.xfrc_applied[self.nb, :3] = self.rng.uniform(-f_mag, f_mag, 3)
                d.xfrc_applied[self.nb, 3:] = self.rng.uniform(-t_mag, t_mag, 3)
            mujoco.mj_step(m, d)
            d.qpos[ta:ta + 7], d.qvel[tv:tv + 6] = tq, 0.0             # the receptive object is held
            if k % 20:
                continue
            pn, Rn = d.xpos[self.nb].copy(), d.xmat[self.nb].reshape(3, 3).copy()
            # overlap: the shaft segment crosses the ring plane inside the ring
            s0, s1 = pn + Rn @ np.array([0, SHAFT_ENDS[0], 0]), pn + Rn @ np.array([0, SHAFT_ENDS[1], 0])
            h0, h1 = float((s0 - c) @ n), float((s1 - c) @ n)
            if h0 * h1 > 0:
                break                                                  # out of the ring: disassembled
            x = s0 + (s1 - s0) * (h0 / (h0 - h1))
            if np.linalg.norm(x - c) > self.ring_r:
                break
            ctr = np.linalg.norm(pn + Rn @ SHAFT_CENTRE_LOCAL - c)
            (near if ctr < 1.5 * self.ring_r else far).append((pn, Rn))
        return near, far

    # ------------------------------------------------------------------ grasp + IK + teleport + validation
    def _ik(self, ps, Rs):
        """damped least squares to the site pose with a null-space pull toward the cell home: the exec layer's
        reference OSC pulls its redundant DoF to the home joints (env/exec_layer.py), so an arbitrary IK posture
        swings by 1-7 cm in the first steps after the write and drags the needle; the posture the controller itself
        prefers does not"""
        import mujoco
        e = self.e
        mm = e.sim.model._model
        d = mujoco.MjData(mm)
        d.qpos[:] = e.sim.data._data.qpos
        jr = np.array([mm.jnt_range[e.sim.model.joint_name2id(nm)] for nm in self.rob.robot_joints])
        lo, hi = jr[:, 0], jr[:, 1]
        home = np.asarray(self.rob.init_qpos, np.float64)
        jp, jrot = np.zeros((3, mm.nv)), np.zeros((3, mm.nv))
        from env.exec_layer import _quat2axisangle_of
        for attempt in range(3):
            q = home.copy() if attempt == 0 else np.clip(home + self.rng.normal(0, 0.3, 7), lo, hi)
            for it in range(300):
                d.qpos[self.jidx] = q
                mujoco.mj_forward(mm, d)
                e_p = ps - d.site_xpos[self.sid]
                e_r = _quat2axisangle_of(Rs @ d.site_xmat[self.sid].reshape(3, 3).T)
                if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_r) < 1e-3 and it > 50:
                    break
                mujoco.mj_jacSite(mm, d, jp, jrot, self.sid)
                J = np.vstack([jp[:, self.jvel], jrot[:, self.jvel]])
                Jp = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))
                dq = Jp @ np.r_[e_p, e_r] + (np.eye(7) - Jp @ J) @ (0.2 * (home - q))
                q = np.clip(q + dq, lo, hi)
            d.qpos[self.jidx] = q
            mujoco.mj_forward(mm, d)
            ep = np.linalg.norm(d.site_xpos[self.sid] - ps)
            er = np.arccos(np.clip((np.trace(Rs.T @ d.site_xmat[self.sid].reshape(3, 3)) - 1) / 2, -1, 1))
            if ep < 0.002 and er < np.radians(2):
                return q
        return None

    def _min_dists(self):
        d = self.e.sim.data
        kinds = self._ids()
        out = {}
        for i in range(d.ncon):
            c = d.contact[i]
            pair = tuple(sorted((str(kinds[c.geom1]), str(kinds[c.geom2]))))
            out[pair] = min(out.get(pair, np.inf), float(c.dist))
        return out

    def draw(self, typ, inner=30):
        """a collision-free teleport spec on the CURRENT (freshly reset) episode: up to `inner` needle-pose / grasp
        draws (cheap: IK + a kinematic write + the collision check). Returns the spec or None (self.why = why the
        last draw was rejected). The rejected writes leave traces in the sim (solver warm start), so the caller
        re-resets, writes ONLY the returned spec and calls settle(): that pass is what training replays."""
        self.why = "no grasps"
        self.__dict__.pop("_rest", None)                                # the resting pose of THIS reset only
        if not self.grasps:
            return None
        e = self.e
        near_far = None
        if typ == "partial_grasped":
            near_far = self.sample_partial()
        for _ in range(inner):
            if typ == "resting_grasped":
                pn, Rn = self.needle_pose() if not hasattr(self, "_rest") else self._rest
                self._rest = (pn, Rn)
            elif typ == "anywhere_grasped":
                pn, Rn = self.sample_anywhere()
            else:
                pick = near_far[0] if self.rng.rand() < 0.5 else near_far[1]  # assembly_success_prob 0.5
                if not pick:
                    self.why = "disassembly gave no pose of the drawn kind"
                    continue
                pn, Rn = pick[self.rng.randint(len(pick))]
            g = self.grasps[self.rng.randint(len(self.grasps))]
            dp, dr = GRASP_NOISE[typ]
            # site pose = needle pose o inverse(in-hand) o noise (UWLab pose_range_b is in the grasp frame)
            Rs = Rn @ g["R"].T
            ps = pn - Rs @ g["p"]
            Rs = Rs @ _rand_rot(self.rng, dr)
            ps = ps + Rs @ self.rng.uniform(-dp, dp, 3) if dp > 0 else ps
            if Rs[2, 2] > -0.5:                                        # approach axis within 60 deg of down
                self.why = "approach not down"
                continue
            q = self._ik(ps, Rs)
            if q is None:
                self.why = "ik"
                continue
            spec = dict(needle_p=pn, needle_R=Rn, arm_q=q, grip_q=g["g"], grip_action=g["ga"])
            self.w.teleport(spec)
            md = self._min_dists()
            # robot-needle: UWLab allows -0.5 mm; this cell's gripper contact sits ~1 mm deep in a normal hold (p90
            # 1.4 mm in IBRL successes, a noise-free dataset grasp written back measures 0.8-0.9 mm) -> -1.5 mm
            lim = {("needle", "robot"): -0.0015, ("robot", "tripod"): 0.0, ("needle", "tripod"): -0.001,
                   ("robot", "table"): 0.0, ("needle", "table"): -0.001}
            bad = [f"{k[0]}-{k[1]} {1e3 * md[k]:.1f}mm" for k, v in lim.items() if md.get(k, np.inf) < v]
            if bad:
                self.why = "collision " + ",".join(bad)
                continue
            return spec
        return None

    def settle(self, typ):
        """from a freshly written spec: hold until 5 consecutive stable steps (<= SETTLE_MAX) and validate
        (check_reset_state_success). Returns the hold actions or None."""
        e = self.e
        p0, ps0 = self.needle_pose()[0], self.site_pose()[0]
        # exactly-zero rotation leaves the reference OSC's goal orientation where it was (env/exec_layer.py), which
        # turns the wrist back toward it: hold with a 1e-6 rad delta
        hold = torch.tensor(np.r_[np.zeros(3), 1e-6, 0.0, 0.0, 1.0], dtype=torch.float32)
        acts, stable = [], 0
        dv = e.sim.model.jnt_dofadr[e.sim.model.body_jntadr[self.nb]]
        for _ in range(SETTLE_MAX):
            _, _, term, _, _ = self.w.step(hold)
            acts.append(hold)
            if term:
                self.why = "terminal while settling"
                return None
            v = np.array(e.sim.data.qvel[dv:dv + 6])
            stable = stable + 1 if (np.abs(v[:3]).sum() < 0.1 and np.abs(v[3:]).sum() < 1.0) else 0
            if stable >= STABLE_STEPS:
                break
        pn1 = self.needle_pose()[0]
        ok = (stable >= STABLE_STEPS
              and np.linalg.norm(pn1 - p0) < MAX_OBJ_DEV[typ]
              and np.linalg.norm(self.site_pose()[0] - ps0) < 0.1
              and pn1[2] > self.table_top() + 0.005
              and self.site_pose()[1][2, 2] < -0.5
              and not e._check_success())
        self.why = (f"settle: stable {stable} dev {1e3 * np.linalg.norm(pn1 - p0):.0f}mm "
                    f"site dev {1e3 * np.linalg.norm(self.site_pose()[0] - ps0):.0f}mm "
                    f"above table {1e3 * (pn1[2] - self.table_top()):.0f}mm down {self.site_pose()[1][2, 2]:.2f} "
                    f"success {e._check_success()}")
        return acts if ok else None

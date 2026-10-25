"""The real cell's EXECUTION LAYER, in the sim (2026-10-17, calib_20261015/SIM_TODO.md §2).

On the robot, an action does not move the arm directly. The chain (real2sim/deploy/) is:
    policy action -> ReferenceArm: the sim arm's own OSC step from the REFERENCE state (q_ref, v_ref)
                  -> hard caps on the commanded TCP (speed, rotation speed, lead over the measured TCP)
                  -> robot_client: per-step joint clip, 100 Hz linear ramp over one control period
                  -> franka_ros2 joint_impedance_controller: tau = K (q_cmd - q) - D dq (+ libfranka gravity comp)
                  -> the FR3: reflected rotor inertia, stiction, ...
The training sim used to stop after the first arrow (the OSC moves the arm itself, no lag, no cap). Measured
on 10-15 (README): the real arm runs at the 8 cm/s cap and trails its command by 0.1-0.4 s -- a persistent
dynamics gap of exactly the kind findings §27 says sets the ceiling.

`SimRefController` replaces the robot's robosuite controller and reproduces the whole chain:
  * set_goal(action)   steps a SHADOW copy of the model (its own MjModel with the nominal sim parameters, objects
                       parked) with robosuite's OSC from the reference state -- what policy_server's ReferenceArm does;
                       then `cap_pose` (shared with policy_server) and the client's joint clip; then the ramp.
  * run_controller()   the joint impedance law on the MAIN arm, whose armature / frictionloss / damping come from the
                       real-arm fit (scripts/fit_arm_twin.py).
Actions therefore keep the deploy meaning: a delta on the REFERENCE pose, not on the measured one (= the converter's
`ref_inverse` labels). The policy still observes the measured arm, as on the robot. Unknown attributes delegate to
the shadow OSC (control_dim, output_max, eef_name, ...), so robosuite and MimicGen keep working;
`ref_site_pose()` is the reference grip-site pose MimicGen must use (env/threading_mg.py).
"""
import copy
from typing import Optional

import numpy as np

# franka_ros2 joint_impedance_controller gains on the cell (calib_20261015/README.md)
REAL_KP = (240.0, 240.0, 240.0, 240.0, 100.0, 60.0, 20.0)
REAL_KD = (20.0, 20.0, 20.0, 10.0, 10.0, 10.0, 5.0)

DEFAULTS = dict(
    kp=REAL_KP, kd=REAL_KD,
    cmd_hz=100.0,              # robot_client COMMAND_RATE_HZ
    # SPEED LIMITS: None = off. 2026-10-17 (owner): no speed limits -- as close to the GELLO teleop as possible
    # (the human exceeds 8 cm/s on 18 % of steps). policy_server / robot_client take these from the cell too.
    # (The 09-25 deploy values were 0.03 rad/step, 0.08 m/s, 0.4 rad/s, 0.03 m lead.)
    max_step_rad=None,         # robot_client per-step joint clip (rad)
    max_tcp_speed=None,        # policy_server --max_tcp_speed (m/s)
    max_rot_speed=None,        # --max_rot_speed (rad/s)
    max_lead=None,             # --max_lead (m)
    resync_m=0.08,             # --resync_m
    joint_margin=0.05,         # --joint_margin (rad) inside the joint range
    # workspace clamp (policy_server _clamp_box), in the ROBOT BASE frame: the deploy reference arm's world is the
    # nominal cell with the base fixed, so its --z_min / --x_box / --y_box are base-frame bounds in disguise.
    # z_min_base = lowest grip-site target above the base origin (None = no clamp; deploy: --z_min - 0.912)
    z_min_base=None, x_box_base=(-np.inf, np.inf), y_box_base=(-np.inf, np.inf),
    # SIM ONLY (2026-10-22): the floor as a height above the episode's actual table top under the grip site, which
    # replaces z_min_base in the sim. The base DR (height +-7.5 mm, tilt 1 deg) moves the table against the base, so a
    # base-frame floor let the grip site down to 2.7 mm above the table in some episodes. The robot has no base DR
    # and keeps reading z_min_base: set this to z_min_base - (nominal table top above the base) so both agree there.
    z_min_above_table=None,
    # SIM ONLY: also keep the gripper's lowest collision point >= this far above the table top. The site floor alone
    # let a CLOSED gripper (tips 17-21 mm below the site; 10.8 mm when open) press into the table: the floor is
    # raised to (site - lowest gripper point, measured now) + this. None = off.
    gripper_clear_table=None,
    z_slow_band=0.0,           # --z_slow_band: a descent speed limit near z_min (0 = off; was 0.06 for policy runs)
    z_slow_step=0.003,         # --z_slow_step
    armature=None, frictionloss=None, damping=None,   # per joint, the real-arm fit; None = sim nominal
)


def exec_layer_config(d: Optional[dict]) -> Optional[dict]:
    """cell `exec_layer` dict -> full config (None/{} or enabled: 0 = off). Unknown keys raise."""
    if not d:
        return None
    d = dict(d)
    if not d.pop("enabled", 1):
        return None
    unknown = set(d) - set(DEFAULTS)
    assert not unknown, f"exec_layer: unknown keys {sorted(unknown)} (known: {sorted(DEFAULTS)})"
    out = dict(DEFAULTS, **d)
    for k in ("kp", "kd", "armature", "frictionloss", "damping"):
        if out[k] is not None:
            out[k] = np.asarray(out[k], np.float64)
            assert out[k].shape == (7,), f"exec_layer.{k} needs 7 values"
    return out


def _quat2axisangle_of(R):
    from robosuite.utils.transform_utils import mat2quat, quat2axisangle
    return np.asarray(quat2axisangle(mat2quat(R)))


def cap_pose(p_cmd, R_cmd, p_prev, R_prev, p_meas, dt, max_tcp_speed, max_rot_speed, max_lead):
    """THE command caps, shared by policy_server (robot) and SimRefController (sim).

    At most max_tcp_speed / max_rot_speed (over dt) from the previous command, and never more than max_lead
    ahead of the MEASURED TCP; a limit of None is off. -> (p, R, why: list of the caps that bit)."""
    from robosuite.utils.transform_utils import axisangle2quat, quat2mat
    why = []
    if max_tcp_speed is not None:
        dp = p_cmd - p_prev
        vmax = max_tcp_speed * dt
        if np.linalg.norm(dp) > vmax:
            p_cmd = p_prev + dp * (vmax / np.linalg.norm(dp))
            why.append("speed")
    if max_lead is not None:
        lead = p_cmd - p_meas
        if np.linalg.norm(lead) > max_lead:
            p_cmd = p_meas + lead * (max_lead / np.linalg.norm(lead))
            why.append("lead")
    if max_rot_speed is not None:
        rv = _quat2axisangle_of(R_cmd @ R_prev.T)
        rmax = max_rot_speed * dt
        if np.linalg.norm(rv) > rmax:
            R_cmd = quat2mat(axisangle2quat(rv * (rmax / np.linalg.norm(rv)))) @ R_prev
            why.append("rot")
    return p_cmd, R_cmd, why


def clamp_box(p, lo, hi, z_last, z_slow_band, z_slow_step):
    """THE workspace clamp, shared by policy_server (robot) and SimRefController (sim).

    Clamp the commanded TCP into [lo, hi]; inside z_slow_band above lo[2] it may descend at most z_slow_step per
    step from the last COMMANDED height (so it never jumps back up). -> (p_clamped, axes clamped as a string)."""
    lo = np.array(lo, np.float64)
    hi = np.array(hi, np.float64)
    band_top = lo[2] + z_slow_band
    if z_slow_band > 0 and p[2] < band_top:
        lo[2] = max(lo[2], min(z_last, band_top) - z_slow_step)
    p_cl = np.clip(p, lo, hi)
    names = "".join(n for n, c in zip("xyz", np.abs(p_cl - p) > 1e-9) if c)
    return p_cl, names


def site_pose(mj_model, mj_data, sid, jidx, q):
    import mujoco
    mj_data.qpos[jidx] = q
    mujoco.mj_forward(mj_model, mj_data)
    return mj_data.site_xpos[sid].copy(), mj_data.site_xmat[sid].reshape(3, 3).copy()


def ik_site(mj_model, mj_data, sid, jidx, jvel, q, p_star, R_star, lo, hi, iters=50, tol=1e-5, damping=1e-3):
    """damped least squares IK of a site to (p_star, R_star) from q (= policy_server ReferenceArm.ik_pose)."""
    import mujoco
    mj_data.qpos[jidx] = q
    mujoco.mj_forward(mj_model, mj_data)
    jp, jr = np.zeros((3, mj_model.nv)), np.zeros((3, mj_model.nv))
    qi = np.array(q, np.float64)
    for _ in range(iters):
        e_p = p_star - mj_data.site_xpos[sid]
        e_r = _quat2axisangle_of(R_star @ mj_data.site_xmat[sid].reshape(3, 3).T)
        if np.linalg.norm(e_p) < tol and np.linalg.norm(e_r) < 10 * tol:
            break
        mujoco.mj_jacSite(mj_model, mj_data, jp, jr, sid)
        J = np.vstack([jp[:, jvel], jr[:, jvel]])
        qi = np.clip(qi + J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), np.concatenate([e_p, e_r])), lo, hi)
        mj_data.qpos[jidx] = qi
        mujoco.mj_forward(mj_model, mj_data)
    return qi


def apply_arm_params(mj_model, jvel, cfg):
    """the real-arm fit onto the MAIN model's arm dofs (armature / frictionloss / damping)."""
    for k, field in (("armature", "dof_armature"), ("frictionloss", "dof_frictionloss"), ("damping", "dof_damping")):
        if cfg.get(k) is not None:
            getattr(mj_model, field)[jvel] = cfg[k]


def _geom_lowest_z(mm, d, g):
    """lowest world z of one box / mesh / capsule geom."""
    import mujoco
    R, c = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
    t = mm.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = mm.geom_dataid[g]
        V = mm.mesh_vert[mm.mesh_vertadr[mid]:mm.mesh_vertadr[mid] + mm.mesh_vertnum[mid]]
        return float(((R @ V.T).T + c)[:, 2].min())
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        C = np.array([[a, b_, c_] for a in (-1, 1) for b_ in (-1, 1) for c_ in (-1, 1)]) * mm.geom_size[g]
        return float(((R @ C.T).T + c)[:, 2].min())
    return float(c[2] - mm.geom_rbound[g])


class SimRefController:
    """Drop-in for robot.controller (robosuite): the deploy execution chain on a simulated FR3. See module doc."""

    def _gripper_lowest_z(self) -> float:
        """lowest world z of the (measured) gripper's collision geoms."""
        m = self.sim.model
        if getattr(self, "_grip_geoms", None) is None:
            self._grip_geoms = [g for g in range(m.ngeom) if (m.geom_id2name(g) or "").startswith("gripper0")
                                and (m.geom_contype[g] or m.geom_conaffinity[g])]
        mm, d = m._model, self.sim.data
        return min(_geom_lowest_z(mm, d, g) for g in self._grip_geoms)

    def __init__(self, robot, cfg: dict, nominal_model=None):
        from robosuite.controllers import controller_factory
        from robosuite.utils.binding_utils import MjSim
        self.robot, self.cfg = robot, cfg
        self.sim = robot.sim
        self.jidx = np.asarray(robot._ref_joint_pos_indexes)
        self.jvel = np.asarray(robot._ref_joint_vel_indexes)
        self.gidx = np.asarray(robot._ref_gripper_joint_pos_indexes)
        self.gvel = np.asarray(robot._ref_gripper_joint_vel_indexes)
        self.sid = robot.eef_site_id
        self.robot_base_id = self.sim.model.body_name2id(robot.robot_model.root_body)
        m = self.sim.model._model
        # shadow = the reference arm: its OWN model (nominal parameters, like policy_server's ReferenceArm env),
        # same geometry / base pose as this episode
        self.shadow = MjSim(copy.deepcopy(nominal_model if nominal_model is not None else m))
        sm, sd = self.shadow.model._model, self.shadow.data._data
        if nominal_model is not None:
            # geometry / base DR of THIS episode: body poses come from the main model
            sm.body_pos[:] = m.body_pos
            sm.body_quat[:] = m.body_quat
        sd.qpos[:] = self.sim.data._data.qpos
        self.free = []                                  # non-robot free joints: parked in the shadow
        import mujoco
        for j in range(sm.njnt):
            if sm.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.free.append((int(sm.jnt_qposadr[j]), int(sm.jnt_dofadr[j])))
        c = dict(robot.controller_config)
        assert c.get("type") == "OSC_POSE", (
            f"exec_layer: the reference arm is policy_server's OSC_POSE ReferenceArm; this env's controller is "
            f"{c.get('type')} (pass controller_configs=load_controller_config(default_controller='OSC_POSE'))")
        c["sim"] = self.shadow
        self.ref = controller_factory(c["type"], c)
        self.ref.update_base_pose(robot.base_pos, robot.base_ori)
        # OSC's nullspace term pulls the redundant DoF toward `initial_joint`, which robosuite sets to the joints at
        # construction (this episode's NOISY start). The deploy ReferenceArm's is always the cell home (it is reset
        # once, without noise): pin it there, or the same action moves the reference differently (1e-2 rad / step)
        self.home = np.asarray(robot.init_qpos, np.float64)
        self.ref.initial_joint = self.home.copy()
        # likewise OSC.set_goal leaves `goal_ori` UNCHANGED when the rotation delta is exactly zero (scripted moves),
        # and it starts at the orientation of construction: the deploy reference's is the home orientation
        p_h, R_h = site_pose(sm, sd, self.sid, self.jidx, self.home)
        self.ref.initial_ee_pos, self.ref.initial_ee_ori_mat = p_h, R_h
        self.ref.goal_pos, self.ref.goal_ori = p_h.copy(), R_h.copy()
        sd.qpos[:] = self.sim.data._data.qpos
        mujoco.mj_forward(sm, sd)
        self.arm_act = np.asarray(robot._ref_joint_actuator_indexes)
        self.lo_t, self.hi_t = robot.torque_limits
        jr = np.array([self.sim.model.jnt_range[self.sim.model.joint_name2id(n)] for n in robot.robot_joints])
        self.q_lo, self.q_hi = jr[:, 0] + cfg["joint_margin"], jr[:, 1] - cfg["joint_margin"]
        self.kp, self.kd = np.asarray(cfg["kp"]), np.asarray(cfg["kd"])
        dt = float(m.opt.timestep)
        self.n_sub = int(round(1.0 / (robot.control_freq * dt)))
        self.per_cmd = max(1, int(round(1.0 / (cfg["cmd_hz"] * dt))))
        self.n_cmd = max(1, int(round(self.n_sub / self.per_cmd)))
        self.dt_ctrl = self.n_sub * dt
        self.reset_goal()

    # ---------------------------------------------------------------- delegation
    def __getattr__(self, name):           # only for attributes not defined here
        if name in ("ref",):
            raise AttributeError(name)
        return getattr(self.ref, name)

    # ---------------------------------------------------------------- state
    def reset_goal(self):
        q = np.array(self.sim.data.qpos[self.jidx])
        self.q_ref, self.v_ref = q.copy(), np.zeros(7)
        self.start = self.goal = q.copy()
        self.k = 0
        self.last_target = q.copy()
        self.last_cmd_pose = None
        self.last_caps = []
        self.last_cmd_z_base = None
        self.resyncs = 0
        self.clamps = 0

    def update_base_pose(self, base_pos, base_ori):
        self.ref.update_base_pose(base_pos, base_ori)

    def update_initial_joints(self, initial_joints):
        self.ref.update_initial_joints(initial_joints)
        self.ref.initial_joint = self.home.copy()     # nullspace target stays the cell home (see __init__)
        self.reset_goal()

    def _park(self):
        sd = self.shadow.data._data
        for i, (a, v) in enumerate(self.free):
            sd.qpos[a:a + 7] = [0.0, 3.0 * (i + 1), -5.0, 1.0, 0.0, 0.0, 0.0]
            sd.qvel[v:v + 6] = 0.0

    def _shadow_state(self, q, v):
        sd = self.shadow.data._data
        sd.qpos[self.jidx] = q
        sd.qvel[self.jvel] = v
        sd.qpos[self.gidx] = self.sim.data.qpos[self.gidx]
        sd.qvel[self.gvel] = 0.0
        self._park()
        self.shadow.forward()

    def ref_site_pose(self):
        """4x4 pose of the REFERENCE grip site (what an action is a delta on)."""
        sm, sd = self.shadow.model._model, self.shadow.data._data
        p, R = site_pose(sm, sd, self.sid, self.jidx, self.q_ref)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        return T

    def hidden_state(self):
        """the reference arm relative to the measured one, base frame (20): grip-site position (3) and rotation
        vector (3) of ref vs measured, q_ref - q (7), v_ref (7). An action is a delta on the REFERENCE, so without
        this the policy cannot tell what its next action will do (the ref leads the arm by ~14 mm median). The deploy
        ReferenceArm (policy_server) has the same q_ref / v_ref."""
        sm, sd = self.shadow.model._model, self.shadow.data._data
        p_r, R_r = site_pose(sm, sd, self.sid, self.jidx, self.q_ref)
        d = self.sim.data
        p_m, R_m = np.array(d.site_xpos[self.sid]), np.array(d.site_xmat[self.sid]).reshape(3, 3)
        Rb = np.array(d.body_xmat[self.robot_base_id]).reshape(3, 3)
        return np.concatenate([Rb.T @ (p_r - p_m), Rb.T @ _quat2axisangle_of(R_r @ R_m.T),
                               self.q_ref - np.array(d.qpos[self.jidx]), self.v_ref])

    # ---------------------------------------------------------------- robosuite API
    def set_goal(self, action, set_pos=None, set_ori=None):
        cfg = self.cfg
        sm, sd = self.shadow.model._model, self.shadow.data._data
        q_meas = np.array(self.sim.data.qpos[self.jidx])
        p_meas = np.array(self.sim.data.site_xpos[self.sid])
        # base-pose DR is written into the MAIN model after the reset (env/sim2real.py apply_sim2real),
        # i.e. after this shadow was copied: follow the geometry every step (physics parameters stay nominal)
        mm = self.sim.model._model
        sm.body_pos[:] = mm.body_pos
        sm.body_quat[:] = mm.body_quat
        p_ref0, _ = site_pose(sm, sd, self.sid, self.jidx, self.q_ref)
        if np.linalg.norm(p_ref0 - p_meas) > cfg["resync_m"]:
            self.q_ref, self.v_ref = q_meas.copy(), np.zeros(7)       # blocked by contact: start over at the arm
            self.resyncs += 1
        # one OSC control period of the reference, from its own state (policy_server ReferenceArm.step_ref)
        self._shadow_state(self.q_ref, self.v_ref)
        self.ref.update(force=True)
        self.ref.set_goal(np.asarray(action, np.float64), set_pos=set_pos, set_ori=set_ori)
        g_ctrl = self.sim.data.ctrl[self.robot._ref_joint_gripper_actuator_indexes]
        for _ in range(self.n_sub):
            self.shadow.forward()
            tau = np.clip(self.ref.run_controller(), self.lo_t, self.hi_t)
            sd.ctrl[self.arm_act] = tau
            sd.ctrl[self.robot._ref_joint_gripper_actuator_indexes] = g_ctrl
            self.shadow.step()
            self._park()
        self.shadow.forward()
        self.q_ref = np.array(sd.qpos[self.jidx])
        self.v_ref = np.array(sd.qvel[self.jvel])
        # workspace clamp (shared with the robot), base frame
        if cfg["z_min_base"] is not None or cfg["z_min_above_table"] is not None:
            p_r, R_r = site_pose(sm, sd, self.sid, self.jidx, self.q_ref)
            bid = self.robot_base_id
            b, Rb = np.array(self.sim.data.body_xpos[bid]), np.array(self.sim.data.body_xmat[bid]).reshape(3, 3)
            pb = Rb.T @ (p_r - b)
            z_last = self.last_cmd_z_base if self.last_cmd_z_base is not None else (Rb.T @ (p_meas - b))[2]
            z_floor = cfg["z_min_base"]
            if cfg["z_min_above_table"] is not None:      # sim: follow the actual table top (base DR)
                m_, d_ = self.sim.model, self.sim.data
                tid = m_.geom_name2id("table_collision")
                top = float(d_.geom_xpos[tid][2] + m_.geom_size[tid][2])
                z_floor = float((Rb.T @ (np.array([p_r[0], p_r[1], top]) - b))[2]) + cfg["z_min_above_table"]
                if cfg["gripper_clear_table"] is not None:   # the closed gripper reaches below the site
                    drop = float(d_.site_xpos[self.sid][2]) - self._gripper_lowest_z()
                    z_floor = max(z_floor, float((Rb.T @ (np.array([p_r[0], p_r[1], top + drop
                                                                    + cfg["gripper_clear_table"]]) - b))[2]))
            pb_cl, names = clamp_box(pb, (cfg["x_box_base"][0], cfg["y_box_base"][0], z_floor),
                                     (cfg["x_box_base"][1], cfg["y_box_base"][1], np.inf), z_last,
                                     cfg["z_slow_band"], cfg["z_slow_step"])
            if names:
                q_c = ik_site(sm, sd, self.sid, self.jidx, self.jvel, self.q_ref, b + Rb @ pb_cl, R_r,
                              self.q_lo, self.q_hi)
                self.q_ref, self.v_ref = np.array(q_c), np.zeros(7)
                self.clamps += 1
        # caps (shared with the robot), from the previous COMMAND
        p_cmd, R_cmd = site_pose(sm, sd, self.sid, self.jidx, self.q_ref)
        R_meas = np.array(self.sim.data.site_xmat[self.sid]).reshape(3, 3)
        p_prev, R_prev = self.last_cmd_pose if self.last_cmd_pose is not None else (p_meas, R_meas)
        p_c, R_c, why = cap_pose(p_cmd, R_cmd, p_prev, R_prev, p_meas, self.dt_ctrl,
                                 cfg["max_tcp_speed"], cfg["max_rot_speed"], cfg["max_lead"])
        q_cmd = self.q_ref
        if why:
            q_cmd = ik_site(sm, sd, self.sid, self.jidx, self.jvel, self.q_ref, p_c, R_c, self.q_lo, self.q_hi)
            self.q_ref, self.v_ref = np.array(q_cmd), np.zeros(7)        # no wind-up
        self.last_cmd_pose = (p_c, R_c)
        self.last_caps = why
        if cfg["z_min_base"] is not None or cfg["z_min_above_table"] is not None:
            bid = self.robot_base_id
            b, Rb = np.array(self.sim.data.body_xpos[bid]), np.array(self.sim.data.body_xmat[bid]).reshape(3, 3)
            self.last_cmd_z_base = float((Rb.T @ (p_c - b))[2])
        q_cmd = np.clip(q_cmd, self.q_lo, self.q_hi)
        # robot_client: per-step joint clip vs the last target, then a linear ramp over one period
        if cfg["max_step_rad"] is not None:
            q_cmd = self.last_target + np.clip(q_cmd - self.last_target, -cfg["max_step_rad"], cfg["max_step_rad"])
        self.last_target = q_cmd
        self.start = self._cmd_now()
        self.goal = q_cmd
        self.k = 0

    def _cmd_now(self):
        i = min(self.k // self.per_cmd + 1, self.n_cmd)
        return self.start + (self.goal - self.start) * (i / self.n_cmd)

    def run_controller(self):
        cmd = self._cmd_now()
        self.k += 1
        d = self.sim.data
        q, dq = d.qpos[self.jidx], d.qvel[self.jvel]
        # franka_ros2 joint impedance + libfranka's gravity (and Coriolis) compensation
        return self.kp * (cmd - q) - self.kd * dq + d.qfrc_bias[self.jvel]

    @property
    def command(self):
        return self._cmd_now()

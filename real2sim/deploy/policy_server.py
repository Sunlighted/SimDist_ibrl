"""SimDist pixel policy server for the real FR3 cell (runs on the robot HOST, GPU).

The ROS side (robot_client.py, inside the gello ros2 container, no torch/GPU)
streams raw observations here over localhost TCP and gets back joint targets.
Everything that must match training lives on THIS side:

  images  : raw ZED JPEG -> undistort (factory K, D or the cell's real_intrinsics,
            newK=K) -> principal-point square crop at the SIM camera's fovy -> 224
            -> ZED->splat colour map  (real2sim/deploy/preprocess.py)
  prop    : [eef_pos(3), eef_quat_xyzw(4), gripper_qpos(2)] computed by MuJoCo FK
            on the SAME model the policy was trained in (ReferenceArm): real joint
            angles in, grip-site position + hand orientation out, which carries the
            45 deg flange offset for free. Frame = the cell's prop_frame: "base" on
            the Threading Robotiq cell (FK(q), no calibration in it), sim world on
            PegInsertion_D0. Robotiq: the ONE real knuckle angle -> both sim drivers.
  action  : the policy's 7-d output is an OSC_POSE delta (+-5 cm, +-0.5 rad per
            50 ms step, world frame, kp 150) + gripper. It is NOT turned into a pose
            target by IK. Instead a REFERENCE copy of the sim arm (object collisions
            disabled) is stepped once through robosuite's own OSC; the joint
            configuration it reaches in that 50 ms is the target sent to the real joint
            impedance controller. So the action means exactly what it meant in training.
            --action_map sim_ref (default): the reference keeps its OWN q and qvel from
              step to step (a reference trajectory the real arm tracks) and is re-synced
              to the measured q only when the TCPs are > --resync_m apart. The real arm
              lags (tau ~100 ms), so the tracking error, and the impedance torque, grow
              until it moves. (The owner's 09-25 fix, real_policy_server.py.)
            --action_map reseed: the reference restarts at the MEASURED q, dq every step.
              The error then stays ~0.007 rad (~1.7 Nm on j4), below joint friction: on
              10-15 x moved 1-4 % of the command per step (calib_20261015/README.md).

Safety checks here (the client adds its own rate / freshness limits):
  - action clipped to [-1, 1]
  - joint target clipped to the FR3 x Panda joint-limit intersection minus a margin
  - grip-site target below --z_min (sim world, metres) or outside the table box:
    sim_ref -> clamped into the box (the reference stops there); reseed -> refused
  - sim_ref: inside --z_slow_band above z_min the target descends <= --z_slow_step/step
  - hard caps on the commanded TCP, both maps: <= --max_tcp_speed / --max_rot_speed
    (real time) from the last command and <= --max_lead ahead of the measured TCP;
    a capped step pulls the reference back onto the capped target (no wind-up)

Usage (robot host):
  python real2sim/deploy/policy_server.py --ckpt <dir>/model0.pt
  then robot_client.py inside the container (see real2sim/deploy/README.md).
"""
import argparse
import json
import os
import socket
import struct
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

from real2sim.deploy.preprocess import (DEFAULT_COLOR_JSON, Rectifier,  # noqa: E402
                                        apply_color, camera_intrinsics, decode_jpeg_rgb,
                                        load_color, peg_fovy, sim_camera_map)

# FR3 and Panda joint limits (rad); targets are clipped to the intersection.
FR3_LO = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
FR3_HI = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
PANDA_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
Q_LO, Q_HI = np.maximum(FR3_LO, PANDA_LO), np.minimum(FR3_HI, PANDA_HI)


# ------------------------------------------------------------------ wire format
# Each message: 4-byte big-endian header length, JSON header, then raw blobs whose
# byte lengths are listed in header["blobs"] (in order). No pickle.
def send_msg(sock, header, blobs=()):
    header = dict(header, blobs=[len(b) for b in blobs])
    h = json.dumps(header).encode()
    sock.sendall(struct.pack(">I", len(h)) + h + b"".join(blobs))


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)


def recv_msg(sock):
    (n,) = struct.unpack(">I", _recv_exact(sock, 4))
    header = json.loads(_recv_exact(sock, n))
    blobs = [_recv_exact(sock, k) for k in header.get("blobs", [])]
    return header, blobs


# --------------------------------------------------------------------- policy
def load_agent(ckpt, device):
    """Rebuild the SimDistAgent from <ckpt dir>/cfg.yaml WITHOUT importing
    train_simdist (whose import chain drags in train_rl/train_bc, diffusers and the
    compiled `rela` replay extension -- none of which inference needs)."""
    import pyrallis
    import torch
    import yaml
    from rl.simdist_agent import SimDistAgent, SimDistConfig
    from env.robosuite_wrapper import PROP_DIM

    raw = yaml.safe_load(open(os.path.join(os.path.dirname(ckpt), "cfg.yaml")))
    assert not raw["use_state"], "this server is for the PIXEL arm"
    assert raw.get("obs_stack", 1) == 1 and raw.get("prop_stack", 1) == 1, "stacking not supported"
    sd = pyrallis.decode(SimDistConfig, raw["simdist"])
    assert int(getattr(sd.world_model, "history_length", 0) or 0) == 0, "history not supported"
    if getattr(sd.world_model, "aux_state_coef", 0.0) > 0:
        sd.world_model.aux_state_dim = 37            # strict load needs the head
    S = int(raw["rl_image_size"])
    agent = SimDistAgent(False, (3, S, S), (PROP_DIM,), 7, raw["rl_camera"], sd, device=device)
    agent.load_state_dict(torch.load(ckpt, map_location=device))
    agent = agent.to(device).eval()

    class _Cfg:                                      # the few fields the server reads
        task_name = raw["task_name"]
        rl_image_size = S
        rl_camera = raw["rl_camera"]
        sim2real_cell = raw.get("sim2real_cell") or ""
        simdist = sd
    return agent, _Cfg


class ReferenceArm:
    """The training env's arm, used for FK (prop) and as the OSC action model.

    PegInsertion_D0: Panda hand, no cell, world-frame prop (unchanged).
    Threading_RealCellRobotiq: the cell the ckpt was trained on (Robotiq + mount, arena,
    init_qpos), every DR off, prop through the SAME code path as the training env
    (robosuite observables -> env.sim2real.apply_prop_frame), so with the cell's
    prop_frame "base" the prop is FK(q) in the base frame and carries no calibration.
    The real gripper reports ONE knuckle angle; robotiq_real_to_sim maps it."""

    TASKS = ("PegInsertion", "Threading_RealCellRobotiq")

    def __init__(self, task, cell_path=""):
        if not task.startswith(self.TASKS):
            raise NotImplementedError(
                f"ReferenceArm supports {self.TASKS}; {task} needs its gripper/arena ported")
        from env.robosuite_wrapper import PixelRobosuite
        import robosuite.utils.transform_utils as T
        self.T = T
        self.task = task
        self.robotiq = task == "Threading_RealCellRobotiq"
        self.cell = None
        if self.robotiq:
            import dataclasses
            import env.threading_variants as tv
            from env.sim2real import Sim2RealConfig, load_sim2real
            assert cell_path, "the ckpt cfg records no sim2real_cell"
            if os.path.realpath(cell_path) != os.path.realpath(tv.REAL_CELL_ROBOTIQ_YAML):
                print(f"[server] ⚠️ ckpt cell {cell_path} != the class's model cell "
                      f"{tv.REAL_CELL_ROBOTIQ_YAML}: the Robotiq/arena geometry comes from the "
                      f"latter (as it did in training at the time)", flush=True)
            cell = load_sim2real(cell_path)
            off = {f.name: f.default for f in dataclasses.fields(Sim2RealConfig)
                   if f.name.startswith("dr_")}
            self.cell = dataclasses.replace(
                cell, cameras=tuple(dataclasses.replace(c, randomize=False) for c in cell.cameras),
                **off)
            self._map = tv.robotiq_real_to_sim
            free = ("needle", "tripod")
        else:
            import env.peg_insertion  # noqa: F401  registers PegInsertion_D0
            free = ("peg",)
        self.e = PixelRobosuite(env_name=task, robots="Panda", episode_length=10 ** 9,
                                camera_names=[], rl_cameras=[], use_state=True,
                                state_only_no_render=True, device="cpu",
                                hand_init_perturb_joint_sigma=0.0, sim2real=self.cell,
                                # this arm IS the reference of the cell's execution layer: never stack one on it
                                env_kwargs={"exec_layer": False} if self.robotiq else None)
        np.random.seed(0)
        self.e.reset()
        self.raw = self.e.env
        self.sim = self.raw.sim
        # one action = one OSC step of 1/control_freq s, as in training (cell control_freq)
        self.control_freq = float(self.raw.control_freq)
        m = self.sim.model
        self.rob = self.raw.robots[0]
        self.jidx = np.array(self.rob._ref_joint_pos_indexes)
        self.jvel = np.array(self.rob._ref_joint_vel_indexes)
        names = [m.joint_id2name(i) for i in range(m.njnt)]
        if self.robotiq:
            gj = list(tv.ROBOTIQ_DRIVER_JOINTS)
        else:
            gj = [n for n in names if n and "finger_joint" in n]
        self.gidx = [m.get_joint_qpos_addr(n) for n in gj]
        self.gvel = [m.get_joint_qvel_addr(n) for n in gj]
        # (qpos addr, qvel addr, sign, driver index) of the Robotiq's passive joints; driver 0 = left
        # (finger_joint), 1 = right (right_outer_knuckle_joint). See set_state.
        self._robotiq_followers = []
        if self.robotiq:
            for n in names:
                if not n or not n.startswith("gripper0_") or n in gj:
                    continue
                side = 0 if "_left_" in n else 1 if "_right_" in n else None
                sign = -1.0 if "inner_finger" in n else 1.0 if "inner_knuckle" in n else None
                if side is None or sign is None:
                    raise RuntimeError(f"unexpected Robotiq joint {n}: set_state does not know how it follows")
                self._robotiq_followers.append((m.get_joint_qpos_addr(n), m.get_joint_qvel_addr(n), sign, side))
            assert len(self._robotiq_followers) == 4, self._robotiq_followers
        # objects must not push the reference arm around: it models free-space OSC
        for g in range(m.ngeom):
            b = m.body_id2name(int(m.geom_bodyid[g])) or ""
            if b.startswith(tuple(f + "_obj" for f in free)) or b.startswith(("board_obj",)):
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
        self.free = []
        for n in names:
            if n and n.startswith(free) and m.jnt_type[m.joint_name2id(n)] == 0:   # free joint
                a = m.get_joint_qpos_addr(n); v = m.get_joint_qvel_addr(n)
                self.free.append((a[0] if isinstance(a, tuple) else a,
                                  v[0] if isinstance(v, tuple) else v))
        tid = m.geom_name2id("table_collision") if "table_collision" in [
            m.geom_id2name(i) for i in range(m.ngeom)] else None
        self.sim.forward()
        self.table_top_z = (float(self.sim.data.geom_xpos[tid][2] + m.geom_size[tid][2])
                            if tid is not None else None)
        self._park()
        self.sim.forward()
        # robosuite's OSC is numba-jitted: the first step costs ~4 s. Pay it now,
        # not on the first live step (which would trip the client's staleness stop).
        q0 = np.array(self.sim.data.qpos[self.jidx])
        t0 = time.time()
        self.step(np.zeros(7))
        self.set_state(q0, np.zeros(7), [0.0] if self.robotiq else [0.04, 0.04])
        print(f"[server] reference-arm OSC warm-up {time.time() - t0:.1f}s | prop frame "
              f"{self.cell.prop_frame if self.cell else 'world'}", flush=True)

    def _park(self):
        """Free objects: far below the table and at rest."""
        for i, (a, v) in enumerate(self.free):
            self.sim.data.qpos[a:a + 3] = [0.0, 2.0 * i, -5.0]
            self.sim.data.qvel[v:v + 6] = 0.0

    def sim_fingers(self, fingers):
        """What the client sends -> the sim's gripper joint values.
        Panda: [left, right] finger metres (right sign-flipped, as before).
        Robotiq: [knuckle_rad] from /gripper/joint_states, mapped to both drivers."""
        if self.robotiq:
            return self._map(fingers[0])
        return np.array([fingers[0], -fingers[1]])

    def set_state(self, q, dq, fingers):
        d = self.sim.data
        d.qpos[self.jidx] = q
        d.qvel[self.jvel] = dq if dq is not None else 0.0
        g = self.sim_fingers(fingers)
        for a, v, x in zip(self.gidx, self.gvel, g):
            d.qpos[a] = x
            d.qvel[v] = 0.0
        if self.robotiq:
            # 2026-10-18: the Robotiq's 4 passive joints must follow the 2 drivers. Left where the previous step
            # put them, they interpenetrate the links whenever the real knuckle changes (16 mm on closing), and the
            # contact forces kicked the reference arm: on the robot it diverged while lifting a grasped needle
            # (runs/grasp_1018_1200 trial 1, re-synced 7x, targets clamped to the box corners). The sim's own
            # closing settles at inner_knuckle = +driver, inner_finger = -driver (same side); exec_layer's
            # shadow copies all six from the sim, so this also matches training.
            for a, v, s, k in self._robotiq_followers:
                d.qpos[a] = s * g[k]
                d.qvel[v] = 0.0
        self._park()
        self.sim.forward()

    def prop(self, fingers):
        if self.robotiq:
            # the training env's own observables + prop_frame: same function of q by construction
            from env.robosuite_wrapper import PROP_KEYS
            from env.sim2real import apply_prop_frame
            obs = self.raw._get_observations(force_update=True)
            apply_prop_frame(self.sim, self.cell, obs)
            return np.concatenate([np.asarray(obs[k], np.float64) for k in PROP_KEYS]).astype(np.float32)
        d = self.sim.data
        pos = np.array(d.site_xpos[self.rob.eef_site_id])
        quat = self.T.convert_quat(d.get_body_xquat(self.rob.robot_model.eef_name), to="xyzw")
        return np.concatenate([pos, quat, [fingers[0], -fingers[1]]]).astype(np.float32)

    def eef(self):
        return np.array(self.sim.data.site_xpos[self.rob.eef_site_id])

    def site_pose(self, q):
        """-> (grip-site position, rotation) at joint configuration q (sim world)."""
        import mujoco
        m, d = self.sim.model._model, self.sim.data._data
        d.qpos[self.jidx] = q
        mujoco.mj_forward(m, d)
        sid = self.rob.eef_site_id
        return d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()

    def ik_pose(self, q, p_star, R_star, iters=50, tol=1e-5, damping=1e-3):
        """Damped-least-squares IK of the grip site to (p_star, R_star) from q -> q_target."""
        import mujoco
        m, d = self.sim.model._model, self.sim.data._data
        T = self.T
        sid = self.rob.eef_site_id
        d.qpos[self.jidx] = q
        mujoco.mj_forward(m, d)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        qi = np.array(q, np.float64)
        for _ in range(iters):
            e_p = p_star - d.site_xpos[sid]
            e_r = T.quat2axisangle(T.mat2quat(R_star @ d.site_xmat[sid].reshape(3, 3).T))
            if np.linalg.norm(e_p) < tol and np.linalg.norm(e_r) < 10 * tol:
                break
            mujoco.mj_jacSite(m, d, jp, jr, sid)
            J = np.vstack([jp[:, self.jvel], jr[:, self.jvel]])
            qi = np.clip(qi + J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6),
                                                    np.concatenate([e_p, e_r])), Q_LO, Q_HI)
            d.qpos[self.jidx] = qi
            mujoco.mj_forward(m, d)
        return qi

    def ref_reset(self):
        self.q_ref, self.v_ref = None, np.zeros(7)

    def step_ref(self, q_real, action, fingers, resync_m):
        """--action_map sim_ref: step the reference trajectory (its own q AND qvel) by one
        action, exactly as in training. Re-synced to the real arm (at rest) only when the
        TCPs are > resync_m apart, e.g. blocked by contact. -> (q_target, resynced?)"""
        resynced = False
        if getattr(self, "q_ref", None) is None:
            self.q_ref, self.v_ref = np.array(q_real, np.float64), np.zeros(7)
        elif np.linalg.norm(self.site_pose(self.q_ref)[0] - self.site_pose(q_real)[0]) > resync_m:
            self.q_ref, self.v_ref = np.array(q_real, np.float64), np.zeros(7)
            resynced = True
        self.set_state(self.q_ref, self.v_ref, fingers)
        q_next = self.step(action)
        self.q_ref = np.array(q_next)
        self.v_ref = np.array(self.sim.data.qvel[self.jvel])
        return q_next, resynced

    def step(self, action):
        """One 50 ms control step through robosuite's own OSC. -> joint positions reached."""
        self.raw.timestep = 0                       # never trip robosuite's horizon check
        self.raw.step(np.asarray(action, np.float64))
        # site_xpos after mj_step is FK of the qpos BEFORE the last substep's integration
        # (~0.4 mm behind at 12 cm/s): eef() -- the z_min / box check -- must see the target
        self.sim.forward()
        return np.array(self.sim.data.qpos[self.jidx])


class TrackScript:
    """--scripted track: a fixed OSC action pattern instead of the policy (2026-10-15).

    Measures what made the 20 Hz runs hit the table: how far behind the commanded pose the
    real arm is, and how far it keeps going after the command stops. Per axis and amplitude:
    hold -> move at -amp (z: down) until the MEASURED grip site has gone `dist` -> hold ->
    move back until it is home -> hold. Everything else (ReferenceArm OSC, joint target,
    z_min / box refusal, client step clip and 100 Hz ramp) is the deploy path unchanged, so a
    20 Hz and a 10 Hz run differ only in the client's --hz. Gripper stays open."""

    def __init__(self, axes, amps, dist, hold_s, max_move_s):
        self.plan = [(ax, a) for ax in axes for a in amps]
        self.dist, self.hold_s, self.max_move_s = dist, hold_s, max_move_s
        self.i, self.phase, self.k, self.home = 0, "hold0", 0, None
        self.done = False

    def act(self, eef, hz):
        """-> (7-d action, info). `eef` = measured grip site (sim world)."""
        a = np.zeros(7)
        a[6] = -1.0
        if self.done or self.i >= len(self.plan):
            self.done = True
            return a, {"phase": "done"}
        ax, amp = self.plan[self.i]
        j = "xyz".index(ax)
        sign = -1.0 if ax == "z" else 1.0            # z: away = DOWN; x/y: +
        if self.home is None:
            self.home = np.array(eef)
        moved = sign * (eef[j] - self.home[j])
        hold, mmax = int(round(self.hold_s * hz)), int(round(self.max_move_s * hz))
        nxt = {"hold0": "away", "away": "hold1", "hold1": "back", "back": "hold2", "hold2": None}
        go = (self.phase.startswith("hold") and self.k >= hold) or \
             (self.phase == "away" and (moved >= self.dist or self.k >= mmax)) or \
             (self.phase == "back" and (moved <= 0.0 or self.k >= mmax))
        if go:
            self.phase, self.k = nxt[self.phase], 0
            if self.phase is None:                   # next (axis, amplitude)
                self.i, self.phase, self.home = self.i + 1, "hold0", None
                return self.act(eef, hz)
        if self.phase == "away":
            a[j] = sign * amp
        elif self.phase == "back":
            a[j] = -sign * amp
        self.k += 1
        return a, {"phase": self.phase, "axis": ax, "amp": amp, "moved": round(float(moved), 5),
                   "home": np.round(self.home, 5).tolist()}


class Server:
    def __init__(self, a):
        import torch
        self.a = a
        self.torch = torch
        t0 = time.time()
        self.script = None
        if a.scripted:
            assert a.task and a.cell, "--scripted needs --task and --cell (no ckpt is loaded)"

            class _Cfg:
                task_name, sim2real_cell, rl_image_size = a.task, a.cell, 224
            self.agent, self.cfg = None, _Cfg
            rl_cameras = ["agentview", "robot0_eye_in_hand"]
            if a.scripted == "track":
                self.script = TrackScript(a.track_axes.split(","),
                                          [float(x) for x in a.track_amps.split(",")],
                                          a.track_dist, a.track_hold_s, a.track_max_move_s)
            else:
                self.script = "grasp"                 # built once the reference arm exists (below)
        else:
            self.agent, self.cfg = load_agent(a.ckpt, a.device)
            rl_cameras = self.agent.rl_cameras
        self.S = int(self.cfg.rl_image_size)
        self.cammap = sim_camera_map(rl_cameras)    # ROS cam -> sim cam
        self.fovy, self.cell_cam = self._sim_cameras()
        self.arm = ReferenceArm(self.cfg.task_name, self.cfg.sim2real_cell)
        self.hz = self.arm.control_freq
        if self.script == "grasp":
            self.script = self._make_grasp_script()
        self._resolve_limits()
        print(f"[server] reference arm control_freq {self.hz:g} Hz (the client must run at it)", flush=True)
        if self.a.z_min is None and isinstance(self.script, TrackScript) and self.arm.table_top_z is not None:
            # free-space test: never let a target come within 15 cm of the table
            self.a.z_min = self.arm.table_top_z + 0.15
            print(f"[server] --scripted: z_min = table top + 0.15 = {self.a.z_min:.3f}", flush=True)
        if self.a.z_min is None:
            if not self.cfg.task_name.startswith("PegInsertion"):
                raise SystemExit(f"--z_min is required for {self.cfg.task_name} (sim world z of the "
                                 f"grip site, m). Table top here = {self.arm.table_top_z}; measure the "
                                 f"grip-site height at a fingertip touch-off and add a margin.")
            self.a.z_min = 0.995
        self.color = {} if a.no_color else load_color(a.color_json)
        self.rect = {}
        print(f"[server] {a.ckpt or 'SCRIPTED ' + a.scripted} loaded in {time.time() - t0:.1f}s | cameras "
              f"{rl_cameras} @ {self.S} | colour map: "
              f"{sorted(self.color) or 'NONE (raw ZED colours)'}", flush=True)
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)
            self.log = open(os.path.join(a.save_dir, "steps.jsonl"), "a")
        else:
            self.log = None
        self.n = 0
        self._reset_ref()
        print(f"[server] action map {a.action_map}"
              + (f" (resync {a.resync_m:g} m, z slow band {a.z_slow_band:g} m)" if a.action_map == "sim_ref" else "")
              + f" | limits ({self.limits_from}): TCP {a.max_tcp_speed} m/s, rot {a.max_rot_speed} rad/s, lead "
              f"{a.max_lead} m, client joint step {self.max_step_rad} rad",
              flush=True)

    def _make_grasp_script(self):
        from env.grasp_script import GraspGeometry, GraspScript, default_grid
        cell = self.arm.cell
        if self.a.grasp_needle:
            nominal = tuple(float(v) for v in self.a.grasp_needle.split(","))
        else:
            nominal = tuple(float(np.mean(r)) for r in (cell.layout_needle_x, cell.layout_needle_y,
                                                        cell.layout_needle_z_rot_deg))
        geo = GraspGeometry.from_sim(self.arm.sim, self.arm.rob, np.asarray(cell.init_qpos, np.float64))
        grid = default_grid()
        if self.a.grasp_trials:
            grid = [grid[int(i)] for i in self.a.grasp_trials.split(",")]
        sc = GraspScript(geo=geo, needle=nominal, grid=grid, hz=self.hz)
        self.grasp_outcomes = {}
        import threading

        def operator():
            while not sc.done:
                line = sys.stdin.readline()
                if not line:
                    return
                ans = line.strip().lower()
                if sc.steps[sc.pi][0] == "wait":
                    self.grasp_outcomes[sc.trial] = ans
                    if self.log:
                        self.log.write(json.dumps({"operator": ans, "trial": sc.trial, "t": time.time()}) + "\n")
                        self.log.flush()
                    sc.released = True
        threading.Thread(target=operator, daemon=True).start()
        print(f"[server] GRASP grid: {len(grid)} trials, nominal needle {nominal} (base), tip drop "
              f"{geo.tip_drop * 1e3:.1f} mm, table top {geo.table_top_base_z:.4f} (base)", flush=True)
        return sc

    def _resolve_limits(self):
        """Speed limits / slow band / z_min: the command line, else the cell's exec_layer (what the sim trains with),
        else the 09-25 deploy values. Negative = off."""
        a = self.a
        cfg = None
        if self.arm.cell is not None and self.arm.cell.exec_layer:
            from env.exec_layer import exec_layer_config
            cfg = exec_layer_config(self.arm.cell.exec_layer)
        legacy = dict(max_tcp_speed=0.08, max_rot_speed=0.4, max_lead=0.03, z_slow_step=0.003,
                      z_slow_band=0.0 if self.script is not None else 0.06, max_step_rad=0.03)
        self.limits_from = "cell exec_layer" if cfg is not None else "09-25 defaults"
        for k in ("max_tcp_speed", "max_rot_speed", "max_lead", "z_slow_step", "z_slow_band"):
            v = getattr(a, k)
            if v is None:
                v = cfg[k] if cfg is not None else legacy[k]
            elif v < 0:
                v = None
            setattr(a, k, v)
        if a.z_slow_band is None:
            a.z_slow_band = 0.0
        if a.z_slow_step is None:
            a.z_slow_step = 0.003
        # robot_client's per-step joint clip, handed over in the hello reply
        self.max_step_rad = cfg["max_step_rad"] if cfg is not None else legacy["max_step_rad"]
        if a.z_min is None and cfg is not None and cfg["z_min_base"] is not None and not isinstance(self.script, TrackScript):
            base_z = float(self.arm.sim.data.body_xpos[self.arm.sim.model.body_name2id("robot0_base")][2])
            a.z_min = cfg["z_min_base"] + base_z
            print(f"[server] z_min = cell exec_layer z_min_base {cfg['z_min_base']:g} + base {base_z:.3f} = {a.z_min:.3f}",
                  flush=True)

    def _reset_ref(self):
        """New episode / client: the reference re-seeds at the next measured q."""
        self.arm.ref_reset()
        self.last_cmd_pose, self.last_step_t, self.last_cmd_z = None, None, None

    def _sim_cameras(self):
        """sim camera -> (fovy it was trained with, its cell RealCameraConfig or None).
        The fovy comes from the same place the training env got it: the cell yaml
        recorded in the ckpt's cfg, else the task constants (PegInsertion_D0)."""
        sims = list(self.cammap.values())
        if self.cfg.sim2real_cell:
            from env.sim2real import load_sim2real
            cell = load_sim2real(self.cfg.sim2real_cell)
            cams = {c.name: c for c in cell.cameras}
            missing = [c for c in sims if c not in cams or cams[c].resolved_fovy() <= 0]
            if missing:
                raise ValueError(f"{self.cfg.sim2real_cell} sets no fovy for {missing}")
            return ({c: cams[c].resolved_fovy() for c in sims}, {c: cams[c] for c in sims})
        pf = peg_fovy()
        return {c: pf[c] for c in sims}, {c: None for c in sims}

    def hello(self, h):
        hz = float(h.get("hz", 20.0))
        if hz != self.arm.control_freq:
            msg = (f"client runs at {hz:g} Hz but the model's env steps at {self.arm.control_freq:g} Hz "
                   f"(cell control_freq): one action would mean another duration on the robot")
            if self.script is None:
                raise ValueError(msg + " -- start robot_client.py with --hz "
                                 f"{self.arm.control_freq:g}")
            print(f"[server] ⚠️ {msg} (scripted test: allowed)", flush=True)
        self.hz = hz
        self._reset_ref()
        for cam, ci in h["camera_info"].items():
            if cam not in self.cammap:
                continue
            sim = self.cammap[cam]
            K, D = camera_intrinsics(self.cell_cam[sim], ci["K"], ci["D"],
                                     ci["width"], ci["height"])
            self.rect[cam] = Rectifier(K, D, ci["width"], ci["height"], self.S,
                                       self.fovy[sim], reduce=self.a.jpeg_reduce)
            src = getattr(self.cell_cam[sim], "real_intrinsics", "") or "camera_info (factory)"
            print(f"[server] {cam} -> {sim}: fovy {self.fovy[sim]:.3f} deg, crop "
                  f"{self.rect[cam].side * self.rect[cam].reduce} px (full res), "
                  f"fy {K[1, 1]:.1f}, K/D from {src}", flush=True)
        missing = [c for c in self.cammap if c not in self.rect]
        if missing:
            raise ValueError(f"no camera_info for {missing}")
        return {"ok": True, "S": self.S, "task": self.cfg.task_name,
                "color": sorted(self.color), "z_min": self.a.z_min, "max_step_rad": self.max_step_rad}

    def obs_images(self, blobs, names):
        out = {}
        for cam, buf in zip(names, blobs):
            im = self.rect[cam](decode_jpeg_rgb(buf, self.rect[cam].reduce))
            im = apply_color(im, self.color.get(cam))
            out[cam] = im
        return out

    def step(self, h, blobs):
        t0 = time.perf_counter()
        q = np.asarray(h["q"], np.float64)
        dq = np.asarray(h.get("dq") or np.zeros(7), np.float64)
        fingers = np.asarray(h["fingers"], np.float64)
        ims = self.obs_images(blobs, h["cams"])
        self.arm.set_state(q, dq, fingers)
        prop = self.arm.prop(fingers)
        eef = self.arm.eef()
        torch = self.torch
        t1 = time.perf_counter()
        info = None
        if isinstance(self.script, TrackScript):
            action, info = self.script.act(eef, self.hz)
        elif self.script is not None:                      # GraspScript: servo on the REFERENCE grip site
            q_ref = getattr(self.arm, "q_ref", None)
            ref_p, ref_R = self.arm.site_pose(q_ref if q_ref is not None else q)
            action, info = self.script.act(ref_p, ref_R)
            if info.get("save_raw") and self.a.save_dir:
                for cam, buf in zip(h["cams"], blobs):
                    with open(os.path.join(self.a.save_dir, f"hover_trial{info['trial']:02d}_{cam}.jpg"), "wb") as f:
                        f.write(buf)
                with open(os.path.join(self.a.save_dir, f"hover_trial{info['trial']:02d}_state.json"), "w") as f:
                    json.dump({"q": q.tolist(), "fingers": fingers.tolist(), "trial": info["trial"],
                               "grid": info["grid"]}, f)
            if info.get("phase") == "wait" and info.get("k") == 0:
                print(f"\n[grasp] trial {info['trial']} {info['grid']} finished (knuckle {fingers.round(3).tolist()}). "
                      f"Was the needle LIFTED and held? type y / n / s(lipped), put the needle back on its mark, "
                      f"then Enter.", flush=True)
        else:
            obs = {self.cammap[c]: torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1)))
                   for c, im in ims.items()}
            obs["prop"] = torch.from_numpy(prop)
            with torch.no_grad():
                action = self.agent.act(obs, eval_mode=True).numpy().astype(np.float64)
        t2 = time.perf_counter()
        action = np.clip(action, -1.0, 1.0)
        warn = []
        if self.a.action_map == "sim_ref":
            q_ref_before = None if getattr(self.arm, "q_ref", None) is None else np.array(self.arm.q_ref)
            q_ref, resynced = self.arm.step_ref(q, action, fingers, self.a.resync_m)
            if resynced:
                warn.append(f"reference re-synced to the real arm (TCPs > {self.a.resync_m * 100:.0f} cm apart)")
            q_ref = self._clamp_box(q_ref, eef, warn)
        else:
            self.arm.set_state(q, dq, fingers)
            q_ref = self.arm.step(action)
            eef_ref = self.arm.eef()
            if eef_ref[2] < self.a.z_min:
                warn.append(f"z {eef_ref[2]:.3f} < z_min {self.a.z_min:.3f}: step refused")
            if not (self.a.x_box[0] <= eef_ref[0] <= self.a.x_box[1]
                    and self.a.y_box[0] <= eef_ref[1] <= self.a.y_box[1]):
                warn.append(f"xy {eef_ref[:2].round(3).tolist()} outside table box: step refused")
        q_ref = self._speed_caps(q, q_ref, warn)
        q_tgt = np.clip(q_ref, Q_LO + self.a.joint_margin, Q_HI - self.a.joint_margin)
        if np.any(np.abs(q_tgt - q_ref) > 1e-6):
            warn.append("joint target clipped to limits")
        if any("refused" in w for w in warn):
            q_tgt = q.copy()
            self.last_cmd_pose = None                    # the arm holds where it is
        eef_tgt = self.arm.site_pose(q_tgt)[0]
        self.last_cmd_z = float(eef_tgt[2])
        grip_close = bool(action[6] > 0.0)
        t3 = time.perf_counter()
        res = {"q_target": q_tgt.tolist(), "gripper_close": grip_close,
               "action": action.round(4).tolist(), "eef": eef.round(4).tolist(),
               "eef_target": eef_tgt.round(4).tolist(), "warn": warn,
               "script": info, "done": bool(self.script is not None and self.script.done),
               "ms": {"pre": round((t1 - t0) * 1e3, 1), "policy": round((t2 - t1) * 1e3, 1),
                      "osc": round((t3 - t2) * 1e3, 1)}}
        if self.log:
            extra = {}
            if self.a.action_map == "sim_ref":
                extra["q_ref_before"] = None if q_ref_before is None else q_ref_before.round(6).tolist()
            self.log.write(json.dumps(dict(res, **extra, fingers=fingers.round(4).tolist(), step=self.n, q=q.round(5).tolist(),
                                           dq=dq.round(5).tolist(), eef_full=eef.round(6).tolist(),
                                           eef_target_full=eef_tgt.round(6).tolist(),
                                           ages=h.get("ages"), t_mono=h.get("t_mono"), hz=self.hz,
                                           t=time.time())) + "\n")
            self.log.flush()
            if self.a.save_every and self.n % self.a.save_every == 0:
                import cv2
                row = np.concatenate([ims[c] for c in self.cammap if c in ims], 1)
                cv2.imwrite(os.path.join(self.a.save_dir, f"obs_{self.n:05d}.png"), row[..., ::-1])
        self.n += 1
        return res

    def _clamp_box(self, q_ref, eef, warn):
        """sim_ref safety: CLAMP the reference TCP into the box (keeping its rotation) and stop
        the reference there, so it does not keep pushing into the floor / wall. Inside
        --z_slow_band above z_min the target may descend at most --z_slow_step per step (from
        the last COMMANDED height, so it never jumps back up): the lagging arm creeps onto the
        floor instead of ramming it."""
        from env.exec_layer import clamp_box      # the same function the sim's execution layer uses
        a = self.a
        p, R = self.arm.site_pose(q_ref)
        z_last = self.last_cmd_z if self.last_cmd_z is not None else eef[2]
        p_cl, names = clamp_box(p, (a.x_box[0], a.y_box[0], a.z_min), (a.x_box[1], a.y_box[1], np.inf),
                                z_last, a.z_slow_band, a.z_slow_step)
        if not names:
            return q_ref
        warn.append(f"target {names} clamped {p.round(3).tolist()} -> {p_cl.round(3).tolist()}")
        q_ref = self.arm.ik_pose(q_ref, p_cl, R)
        self.arm.q_ref, self.arm.v_ref = np.array(q_ref), np.zeros(7)
        return q_ref

    def _speed_caps(self, q, q_cmd, warn):
        """HARD limits on the commanded TCP, whatever the policy, map or client rate: at most
        --max_tcp_speed / --max_rot_speed (real time since the last step) from the previous
        command, and never more than --max_lead ahead of the MEASURED TCP (bounds the
        impedance pull). A capped step pulls the sim reference back onto it (no wind-up)."""
        from env.exec_layer import cap_pose       # the same function the sim's execution layer uses
        now = time.monotonic()
        dt = float(np.clip(now - self.last_step_t, 0.02, 0.25)) if self.last_step_t else 1.0 / self.hz
        self.last_step_t = now
        p_cmd, R_cmd = self.arm.site_pose(q_cmd)
        p_meas, R_meas = self.arm.site_pose(q)
        p_prev, R_prev = self.last_cmd_pose if self.last_cmd_pose is not None else (p_meas, R_meas)
        p_cmd, R_cmd, why = cap_pose(p_cmd, R_cmd, p_prev, R_prev, p_meas, dt, self.a.max_tcp_speed,
                                     self.a.max_rot_speed, self.a.max_lead)
        if why:
            q_cmd = self.arm.ik_pose(q_cmd, p_cmd, R_cmd)
            if self.a.action_map == "sim_ref" and self.arm.q_ref is not None:
                self.arm.q_ref, self.arm.v_ref = np.array(q_cmd), np.zeros(7)
            warn.append("capped(" + ",".join(why) + ")")
        self.last_cmd_pose = (p_cmd, R_cmd)
        return q_cmd

    def serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.a.host, self.a.port))
        srv.listen(1)
        print(f"[server] listening on {self.a.host}:{self.a.port}", flush=True)
        while True:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[server] client {addr}", flush=True)
            try:
                while True:
                    h, blobs = recv_msg(conn)
                    try:
                        if h["type"] == "hello":
                            send_msg(conn, self.hello(h))
                        elif h["type"] == "step":
                            send_msg(conn, self.step(h, blobs))
                        elif h["type"] == "reset":
                            self.n = 0
                            self._reset_ref()
                            if isinstance(self.script, TrackScript):    # the grasp grid keeps its trial count
                                self.script = TrackScript(self.a.track_axes.split(","),
                                                          [float(x) for x in self.a.track_amps.split(",")],
                                                          self.a.track_dist, self.a.track_hold_s,
                                                          self.a.track_max_move_s)
                            send_msg(conn, {"ok": True})
                        else:
                            send_msg(conn, {"error": f"unknown type {h['type']}"})
                    except Exception as e:                    # report, keep serving
                        import traceback
                        traceback.print_exc()
                        send_msg(conn, {"error": f"{type(e).__name__}: {e}"})
            except ConnectionError:
                print("[server] client disconnected", flush=True)
            finally:
                conn.close()


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="policy (not needed with --scripted)")
    ap.add_argument("--scripted", default="", choices=("", "track", "grasp"),
                    help="track: fixed move/hold pattern instead of a policy (tracking-lag test). grasp: the grasp "
                         "grid of env/grasp_script.py (top-down to the needle handle at grid offsets, close, lift; "
                         "hovers to save a raw wrist frame per trial; asks the operator between trials)")
    ap.add_argument("--grasp_needle", default="",
                    help="--scripted grasp: nominal needle pose 'x,y,yaw_deg' in the robot BASE frame "
                         "(default: the centre of the cell's needle layout range)")
    ap.add_argument("--grasp_trials", default="", help="--scripted grasp: comma list of trial indices to run")
    ap.add_argument("--task", default="", help="--scripted: the task whose arm is the OSC model")
    ap.add_argument("--cell", default="", help="--scripted: its cell yaml")
    ap.add_argument("--track_axes", default="z,x")
    ap.add_argument("--track_amps", default="0.2,0.5,1.0", help="OSC action magnitudes to test")
    ap.add_argument("--track_dist", type=float, default=0.06, help="m moved per leg (measured)")
    ap.add_argument("--track_hold_s", type=float, default=1.5)
    ap.add_argument("--track_max_move_s", type=float, default=4.0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--color_json", default=DEFAULT_COLOR_JSON)
    ap.add_argument("--no_color", action="store_true", help="feed raw ZED colours")
    ap.add_argument("--z_min", type=float, default=None,
                    help="lowest allowed grip-site target, sim world z (m). PegInsertion default "
                         "0.995 (the real demos bottom out at 1.008); REQUIRED for any other task.")
    ap.add_argument("--x_box", type=float, nargs=2, default=(-0.35, 0.35))
    ap.add_argument("--y_box", type=float, nargs=2, default=(-0.35, 0.35))
    ap.add_argument("--joint_margin", type=float, default=0.05)
    ap.add_argument("--action_map", choices=("sim_ref", "reseed"), default="sim_ref",
                    help="sim_ref: the sim arm is a reference trajectory with its own q/qvel, the "
                         "real arm tracks it (09-25 fix). reseed: restart it at the measured q every "
                         "step (stalls below joint friction: x 1-4 %%/step on 10-15)")
    ap.add_argument("--resync_m", type=float, default=0.08,
                    help="sim_ref: re-sync the reference to the real arm past this TCP distance (m)")
    ap.add_argument("--z_slow_band", type=float, default=None,
                    help="sim_ref: within this height above --z_min the target descends at most "
                         "--z_slow_step per step (m). Default 0.06, 0 with --scripted")
    ap.add_argument("--z_slow_step", type=float, default=None, help="default: the cell's exec_layer, else 0.003")
    # The speed limits below default to the cell's exec_layer (env/exec_layer.py) so the sim and the robot apply the
    # same ones; the 10-15 cell has none (owner, 2026-10-17). Without an exec_layer: the 09-25 values. A negative value
    # on the command line = off.
    ap.add_argument("--max_tcp_speed", type=float, default=None,
                    help="HARD cap on the commanded TCP speed (m/s, real time)")
    ap.add_argument("--max_rot_speed", type=float, default=None,
                    help="HARD cap on the commanded TCP rotation speed (rad/s, real time)")
    ap.add_argument("--max_lead", type=float, default=None,
                    help="HARD cap on how far the commanded TCP may lead the measured one (m)")
    ap.add_argument("--jpeg_reduce", type=int, default=2, choices=(1, 2, 4),
                    help="decode JPEGs at 1/r resolution (2: 37 -> ~10 ms/step, same 224 image)")
    ap.add_argument("--save_dir", default="", help="log steps.jsonl + policy-input images")
    ap.add_argument("--save_every", type=int, default=5)
    return ap


def main():
    ap = build_parser()
    a = ap.parse_args()
    if not a.scripted and not a.ckpt:
        ap.error("--ckpt is required unless --scripted")
    Server(a).serve()


if __name__ == "__main__":
    main()

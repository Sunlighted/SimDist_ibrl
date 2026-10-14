"""SimDist pixel policy server for the real FR3 cell (runs on the robot HOST, GPU).

The ROS side (robot_client.py, inside the gello ros2 container, no torch/GPU)
streams raw observations here over localhost TCP and gets back joint targets.
Everything that must match training lives on THIS side:

  images  : raw ZED JPEG -> undistort (factory K, D or the cell's real_intrinsics,
            newK=K) -> principal-point square crop at the SIM camera's fovy -> 224
            -> ZED->splat colour map  (real2sim/deploy/preprocess.py)
  prop    : [eef_pos(3), eef_quat_xyzw(4), gripper_qpos(2)] computed by MuJoCo FK
            on the SAME PegInsertion_D0 model the policy was trained in: real joint
            angles in, sim-world grip-site pose out. This carries the measured base
            placement and the 45 deg flange offset for free.
  action  : the policy's 7-d output is an OSC_POSE delta (+-5 cm, +-0.5 rad per
            50 ms step, world frame, kp 150) + gripper. It is NOT turned into a pose
            target by IK. Instead a REFERENCE copy of the sim arm (object collisions
            disabled) is set to the real joint positions AND velocities and stepped
            once through robosuite's own OSC; the joint configuration it reaches in
            that 50 ms is the target sent to the real joint impedance controller.
            So the action means exactly what it meant in training, OSC lag included.

Safety checks here (the client adds its own rate / freshness limits):
  - action clipped to [-1, 1]
  - joint target clipped to the FR3 x Panda joint-limit intersection minus a margin
  - grip-site target below --z_min (sim world, metres) or outside the table box
    -> the step is refused (target = current joints) and flagged

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
    """The training env's arm, used for FK (prop) and as the OSC action model."""

    def __init__(self, task):
        # ⚠️ PegInsertion_D0 only: robots="Panda" with its 2-finger hand, no sim2real
        # cell (arena, Robotiq + mount, init_qpos). On a RealCell policy the FK/prop
        # and the OSC model here would be the WRONG robot -- port before deploying.
        if not task.startswith("PegInsertion"):
            raise NotImplementedError(
                f"ReferenceArm is PegInsertion-only; {task} needs the cell's gripper/arena")
        import env.peg_insertion  # noqa: F401  registers PegInsertion_D0
        from env.robosuite_wrapper import PixelRobosuite
        import robosuite.utils.transform_utils as T
        self.T = T
        self.e = PixelRobosuite(env_name=task, robots="Panda", episode_length=10 ** 9,
                                camera_names=[], rl_cameras=[], use_state=True,
                                state_only_no_render=True, device="cpu")
        np.random.seed(0)
        self.e.reset()
        self.raw = self.e.env
        self.sim = self.raw.sim
        m = self.sim.model
        self.rob = self.raw.robots[0]
        self.jidx = np.array(self.rob._ref_joint_pos_indexes)
        self.jvel = np.array(self.rob._ref_joint_vel_indexes)
        names = [m.joint_id2name(i) for i in range(m.njnt)]
        self.gidx = [m.get_joint_qpos_addr(n) for n in names if n and "finger_joint" in n]
        self.gvel = [m.get_joint_qvel_addr(n) for n in names if n and "finger_joint" in n]
        # objects must not push the reference arm around: it models free-space OSC
        for g in range(m.ngeom):
            b = m.body_id2name(int(m.geom_bodyid[g])) or ""
            if b.startswith(("peg_obj", "board_obj")):
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
        peg = [n for n in names if n and "peg" in n]
        self.peg_q = self.peg_v = None
        if peg:
            a = m.get_joint_qpos_addr(peg[0]); v = m.get_joint_qvel_addr(peg[0])
            self.peg_q = a[0] if isinstance(a, tuple) else a
            self.peg_v = v[0] if isinstance(v, tuple) else v
        self._park_peg()
        self.sim.forward()
        # robosuite's OSC is numba-jitted: the first step costs ~4 s. Pay it now,
        # not on the first live step (which would trip the client's staleness stop).
        q0 = np.array(self.sim.data.qpos[self.jidx])
        t0 = time.time()
        self.step(np.zeros(7))
        self.set_state(q0, np.zeros(7), [0.04, 0.04])
        print(f"[server] reference-arm OSC warm-up {time.time() - t0:.1f}s", flush=True)

    def _park_peg(self):
        """The peg is a free body: keep it far below the table and at rest."""
        if self.peg_q is not None:
            self.sim.data.qpos[self.peg_q:self.peg_q + 3] = [0.0, 0.0, -5.0]
            self.sim.data.qvel[self.peg_v:self.peg_v + 6] = 0.0

    def set_state(self, q, dq, fingers):
        d = self.sim.data
        d.qpos[self.jidx] = q
        d.qvel[self.jvel] = dq if dq is not None else 0.0
        if len(self.gidx) == 2:
            d.qpos[self.gidx[0]], d.qpos[self.gidx[1]] = fingers[0], -fingers[1]
            d.qvel[self.gvel[0]] = d.qvel[self.gvel[1]] = 0.0
        self._park_peg()
        self.sim.forward()

    def prop(self, fingers):
        d = self.sim.data
        pos = np.array(d.site_xpos[self.rob.eef_site_id])
        quat = self.T.convert_quat(d.get_body_xquat(self.rob.robot_model.eef_name), to="xyzw")
        return np.concatenate([pos, quat, [fingers[0], -fingers[1]]]).astype(np.float32)

    def eef(self):
        return np.array(self.sim.data.site_xpos[self.rob.eef_site_id])

    def step(self, action):
        """One 50 ms control step through robosuite's own OSC. -> joint positions reached."""
        self.raw.timestep = 0                       # never trip robosuite's horizon check
        self.raw.step(np.asarray(action, np.float64))
        return np.array(self.sim.data.qpos[self.jidx])


class Server:
    def __init__(self, a):
        import torch
        self.a = a
        self.torch = torch
        t0 = time.time()
        self.agent, self.cfg = load_agent(a.ckpt, a.device)
        self.S = int(self.cfg.rl_image_size)
        self.cammap = sim_camera_map(self.agent.rl_cameras)    # ROS cam -> sim cam
        self.fovy, self.cell_cam = self._sim_cameras()
        self.arm = ReferenceArm(self.cfg.task_name)
        self.color = {} if a.no_color else load_color(a.color_json)
        self.rect = {}
        print(f"[server] {a.ckpt} loaded in {time.time() - t0:.1f}s | cameras "
              f"{self.agent.rl_cameras} @ {self.S} | colour map: "
              f"{sorted(self.color) or 'NONE (raw ZED colours)'}", flush=True)
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)
            self.log = open(os.path.join(a.save_dir, "steps.jsonl"), "a")
        else:
            self.log = None
        self.n = 0

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
                "color": sorted(self.color), "z_min": self.a.z_min}

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
        obs = {self.cammap[c]: torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1)))
               for c, im in ims.items()}
        obs["prop"] = torch.from_numpy(prop)
        t1 = time.perf_counter()
        with torch.no_grad():
            action = self.agent.act(obs, eval_mode=True).numpy().astype(np.float64)
        t2 = time.perf_counter()
        action = np.clip(action, -1.0, 1.0)
        q_ref = self.arm.step(action)
        eef_tgt = self.arm.eef()
        q_tgt = np.clip(q_ref, Q_LO + self.a.joint_margin, Q_HI - self.a.joint_margin)
        warn = []
        if eef_tgt[2] < self.a.z_min:
            warn.append(f"z {eef_tgt[2]:.3f} < z_min {self.a.z_min:.3f}: step refused")
        if not (self.a.x_box[0] <= eef_tgt[0] <= self.a.x_box[1]
                and self.a.y_box[0] <= eef_tgt[1] <= self.a.y_box[1]):
            warn.append(f"xy {eef_tgt[:2].round(3).tolist()} outside table box: step refused")
        if np.any(np.abs(q_tgt - q_ref) > 1e-6):
            warn.append("joint target clipped to limits")
        if any("refused" in w for w in warn):
            q_tgt = q.copy()
        grip_close = bool(action[6] > 0.0)
        t3 = time.perf_counter()
        res = {"q_target": q_tgt.tolist(), "gripper_close": grip_close,
               "action": action.round(4).tolist(), "eef": eef.round(4).tolist(),
               "eef_target": eef_tgt.round(4).tolist(), "warn": warn,
               "ms": {"pre": round((t1 - t0) * 1e3, 1), "policy": round((t2 - t1) * 1e3, 1),
                      "osc": round((t3 - t2) * 1e3, 1)}}
        if self.log:
            self.log.write(json.dumps(dict(res, step=self.n, q=q.round(4).tolist(),
                                           t=time.time())) + "\n")
            self.log.flush()
            if self.a.save_every and self.n % self.a.save_every == 0:
                import cv2
                row = np.concatenate([ims[c] for c in self.cammap if c in ims], 1)
                cv2.imwrite(os.path.join(self.a.save_dir, f"obs_{self.n:05d}.png"), row[..., ::-1])
        self.n += 1
        return res

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--color_json", default=DEFAULT_COLOR_JSON)
    ap.add_argument("--no_color", action="store_true", help="feed raw ZED colours")
    ap.add_argument("--z_min", type=float, default=0.995,
                    help="lowest allowed grip-site target, sim world z (m). The real demos "
                         "bottom out at 1.008 (grasping the peg off the table).")
    ap.add_argument("--x_box", type=float, nargs=2, default=(-0.35, 0.35))
    ap.add_argument("--y_box", type=float, nargs=2, default=(-0.35, 0.35))
    ap.add_argument("--joint_margin", type=float, default=0.05)
    ap.add_argument("--jpeg_reduce", type=int, default=2, choices=(1, 2, 4),
                    help="decode JPEGs at 1/r resolution (2: 37 -> ~10 ms/step, same 224 image)")
    ap.add_argument("--save_dir", default="", help="log steps.jsonl + policy-input images")
    ap.add_argument("--save_every", type=int, default=5)
    Server(ap.parse_args()).serve()


if __name__ == "__main__":
    main()

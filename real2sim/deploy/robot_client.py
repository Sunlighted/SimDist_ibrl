"""ROS 2 side of the SimDist policy runner (runs INSIDE the gello ros2 container).

Thin by design: no torch, no MuJoCo. It streams the latest raw observations to
policy_server.py (host, GPU) over localhost and executes the joint targets that
come back, using the same path as reset_arm.py: gello_publisher is paused with
SIGSTOP and this node publishes the joint impedance controller's command topic
(/gello/joint_states) at 100 Hz, interpolating between 20 Hz policy targets.

Modes
  default (DRY RUN): subscribes and queries the policy every step, prints actions and
      joint deltas, publishes NOTHING. Use it first, with the arm at the start pose,
      to check latency, the policy's first moves and the server's saved input images.
  --live : actually moves the arm. Requires the impedance controller to be ACTIVE
      (i.e. a teleop session is running) and asks for confirmation first.

Safety (in addition to the server's z / workspace / joint-limit refusals)
  - per-policy-step joint change clipped to --max_step_rad (default: what the server's cell says --
    none on the 10-15 Robotiq cell, the same as the sim's execution layer; 0.03 rad for older servers)
  - stale inputs (> --max_age s) or a server error stop the episode
  - any exit path (end, Ctrl-C, error) holds the last measured pose, deactivates the
    impedance controller (the arm holds), then resumes gello_publisher -- exactly
    reset_arm.py's teardown. Realign the GELLO (or run reset_arm.py) afterwards.

Usage (inside the container, host policy_server.py already listening):
  python3 robot_client.py                       # dry run
  python3 robot_client.py --live --steps 300    # real episode (15 s at 20 Hz)
"""
import json
import math
import os
import signal
import socket
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
import tyro
from controller_manager_msgs.srv import ListControllers, SwitchController
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CameraInfo, CompressedImage, JointState
from std_msgs.msg import Float32

IMPEDANCE_CONTROLLER = "joint_impedance_controller"
ROBOTIQ_KNUCKLE = "robotiq_85_left_knuckle_joint"
COMMAND_RATE_HZ = 100.0
ROBOT_STATE_TIMEOUT_S = 1.0
CAMERA_INFO_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)


@dataclass
class Args:
    live: bool = False
    """Move the arm. Without this the client only prints what it would do."""
    steps: int = 300
    """Policy steps per episode (sim episode_length = 300 = 15 s at 20 Hz)."""
    hz: float = 20.0
    """Policy rate. The policy was trained at 20 Hz; lower = the arm gets longer per target."""
    server: str = "127.0.0.1:5555"
    max_step_rad: Optional[float] = None
    """Largest joint change per policy step (rad). Default: the server's (= the cell exec_layer the sim trains
    with); negative = off."""
    max_age: float = 0.25
    """Refuse to act on camera / joint data older than this (s)."""
    yes: bool = False
    """Skip the confirmation before a live episode."""
    activate: bool = False
    """Live, impedance controller inactive (e.g. after reset_arm found the arm already at the
    start pose): pause the GELLO, hold the MEASURED pose and activate the controller, so the arm
    does not jump to wherever the GELLO is."""
    namespace: str = ""
    """Controller-manager namespace (reset_arm.py reads it from the robot config; usually empty)."""
    robot_joint_topic: str = "/franka/joint_states"
    gripper_state_topic: str = "/franka_gripper/joint_states"
    command_topic: str = "/gello/joint_states"
    gripper_command_topic: str = "/gripper/gripper_client/target_gripper_width_percent"
    scene_topic: str = "/cameras/scene/left/image/compressed"
    wrist_topic: str = "/cameras/wrist/left/image/compressed"
    scene_info_topic: str = "/cameras/scene/left/camera_info"
    wrist_info_topic: str = "/cameras/wrist/left/camera_info"
    joint_prefix: str = "fr3_joint"
    joint_log: str = ""
    """Save EVERY joint-state message and every 100 Hz command to this .npz (tracking test)."""


# ------------------------------------------------------------ wire (policy_server)
def send_msg(sock, header, blobs=()):
    header = dict(header, blobs=[len(b) for b in blobs])
    h = json.dumps(header).encode()
    sock.sendall(struct.pack(">I", len(h)) + h + b"".join(blobs))


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("policy server closed the connection")
        buf += chunk
    return bytes(buf)


def recv_msg(sock):
    (n,) = struct.unpack(">I", _recv_exact(sock, 4))
    header = json.loads(_recv_exact(sock, n))
    blobs = [_recv_exact(sock, k) for k in header.get("blobs", [])]
    return header, blobs


# --------------------------------------------------------------------- ROS node
class RunnerNode(Node):
    def __init__(self, a: Args):
        super().__init__("simdist_policy_runner")
        self.a = a
        self.joint_names = [f"{a.joint_prefix}{i + 1}" for i in range(7)]
        self._lock = Lock()
        self.q = self.dq = None
        self.q_at = 0.0
        self.fingers = None
        self.fingers_at = 0.0
        self.imgs: Dict[str, Tuple[bytes, float]] = {}
        self.info: Dict[str, dict] = {}
        self.gello_cmd: Optional[List[float]] = None
        self.jlog = [] if a.joint_log else None     # (t, q, dq) of every joint-state message
        self.clog = [] if a.joint_log else None     # (t, q) of every published command

        self.create_subscription(JointState, a.robot_joint_topic, self._on_robot, 10)
        self.create_subscription(JointState, a.gripper_state_topic, self._on_gripper, 10)
        self.create_subscription(JointState, a.command_topic, self._on_cmd, 10)
        for cam, topic in (("scene", a.scene_topic), ("wrist", a.wrist_topic)):
            self.create_subscription(CompressedImage, topic,
                                     lambda m, c=cam: self._on_img(c, m), qos_profile_sensor_data)
        for cam, topic in (("scene", a.scene_info_topic), ("wrist", a.wrist_info_topic)):
            self.create_subscription(CameraInfo, topic,
                                     lambda m, c=cam: self._on_info(c, m), CAMERA_INFO_QOS)
        self._command = self.create_publisher(JointState, a.command_topic, 10)
        self._gripper = self.create_publisher(Float32, a.gripper_command_topic, 10)
        prefix = f"/{a.namespace}" if a.namespace else ""
        self._switch = self.create_client(SwitchController, f"{prefix}/controller_manager/switch_controller")
        self._list = self.create_client(ListControllers, f"{prefix}/controller_manager/list_controllers")

    def _pick(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        if not all(n in idx for n in self.joint_names):
            return None, None
        pos = [msg.position[idx[n]] for n in self.joint_names]
        vel = ([msg.velocity[idx[n]] for n in self.joint_names]
               if len(msg.velocity) >= len(msg.position) else [0.0] * 7)
        return pos, vel

    def _on_robot(self, msg):
        pos, vel = self._pick(msg)
        if pos is not None:
            with self._lock:
                self.q, self.dq, self.q_at = pos, vel, time.monotonic()
                if self.jlog is not None:
                    self.jlog.append((self.q_at, pos, vel))

    def _on_gripper(self, msg):
        # Robotiq (/gripper/joint_states): ONE knuckle angle, the server maps it onto the sim
        if ROBOTIQ_KNUCKLE in msg.name:
            with self._lock:
                self.fingers = [msg.position[list(msg.name).index(ROBOTIQ_KNUCKLE)]]
                self.fingers_at = time.monotonic()
        elif len(msg.position) >= 2:
            with self._lock:
                self.fingers, self.fingers_at = [msg.position[0], msg.position[1]], time.monotonic()

    def _on_cmd(self, msg):
        pos, _ = self._pick(msg)
        if pos is not None:
            self.gello_cmd = pos

    def _on_img(self, cam, msg):
        with self._lock:
            self.imgs[cam] = (bytes(msg.data), time.monotonic())

    def _on_info(self, cam, msg):
        self.info[cam] = {"K": list(msg.k), "D": list(msg.d),
                          "width": int(msg.width), "height": int(msg.height)}

    def snapshot(self):
        with self._lock:
            return (self.q, self.dq, self.q_at, self.fingers, self.fingers_at, dict(self.imgs))

    def publish_target(self, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "fr3_link0"
        msg.name = self.joint_names
        msg.position = [float(p) for p in positions]
        self._command.publish(msg)

    def publish_gripper(self, close: bool):
        self._gripper.publish(Float32(data=0.0 if close else 1.0))

    def controller_states(self):
        if not self._list.wait_for_service(timeout_sec=5.0):
            raise SystemExit("list_controllers service not available. Is the robot running?")
        fut = self._list.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        if fut.result() is None:
            raise SystemExit("list_controllers did not answer.")
        return {c.name: c.state for c in fut.result().controller}

    def switch(self, activate, deactivate):
        req = SwitchController.Request()
        req.activate_controllers, req.deactivate_controllers = activate, deactivate
        req.strictness = SwitchController.Request.STRICT
        fut = self._switch.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        if fut.result() is None or not fut.result().ok:
            raise SystemExit(f"switch failed (activate={activate}, deactivate={deactivate}).")


class InterpolatingHolder:
    """Publishes the command topic at 100 Hz, linearly ramping to each new target.

    Doubles as the watchdog feeder (the impedance controller kills ros2_control_node
    after 0.5 s without a command), so it must run for as long as gello_publisher is
    paused.
    """

    def __init__(self, node: RunnerNode):
        self.node = node
        self._lock = Lock()
        self._start = self._goal = None
        self._t0, self._dur = 0.0, 1.0
        self._stop = Event()
        self._thread = Thread(target=self._run, daemon=True)

    def start(self, q):
        with self._lock:
            self._start = self._goal = list(q)
        self._thread.start()

    def ramp_to(self, goal, duration):
        with self._lock:
            self._start = self._current_locked()
            self._goal, self._t0, self._dur = list(goal), time.monotonic(), max(duration, 1e-3)

    def hold(self, q):
        with self._lock:
            self._start = self._goal = list(q)

    def _current_locked(self):
        if self._goal is None:
            return None
        f = min(max((time.monotonic() - self._t0) / self._dur, 0.0), 1.0)
        return [s + (g - s) * f for s, g in zip(self._start, self._goal)]

    def current(self):
        with self._lock:
            return self._current_locked()

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self):
        while not self._stop.is_set():
            q = self.current()
            if q is not None:
                self.node.publish_target(q)
                if self.node.clog is not None:
                    self.node.clog.append((time.monotonic(), q))
            self._stop.wait(1.0 / COMMAND_RATE_HZ)


def gello_publisher_pid() -> Optional[int]:
    me = os.getpid()
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    for line in out.splitlines()[1:]:
        pid_text, _, args = line.strip().partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == me or "/bin/bash" in args:
            continue
        if args.rstrip().endswith("gello_publisher") or "/gello_publisher " in args:
            return pid
    return None


def spin_for(node, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.01)


def main(a: Args):
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = RunnerNode(a)
    holder = InterpolatingHolder(node)
    paused = None
    controller_on = False
    host, port = a.server.split(":")
    sock = socket.create_connection((host, int(port)), timeout=10.0)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        print("waiting for joint states, gripper state, both cameras and camera_info ...")
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            spin_for(node, 0.1)
            q, _, _, fing, _, imgs = node.snapshot()
            if q is not None and fing is not None and len(imgs) == 2 and len(node.info) == 2:
                break
        else:
            q, _, _, fing, _, imgs = node.snapshot()
            raise SystemExit(f"inputs missing after 15 s: joints={q is not None} gripper="
                             f"{fing is not None} images={sorted(imgs)} camera_info={sorted(node.info)}")
        send_msg(sock, {"type": "hello", "camera_info": node.info, "hz": a.hz})
        h, _ = recv_msg(sock)
        if "error" in h:
            raise SystemExit(f"server: {h['error']}")
        print(f"server ready: {h}")
        if a.max_step_rad is None:
            a.max_step_rad = h.get("max_step_rad", 0.03) if "max_step_rad" in h else 0.03
        elif a.max_step_rad < 0:
            a.max_step_rad = None
        print(f"per-step joint clip: {a.max_step_rad} rad")
        send_msg(sock, {"type": "reset"}); recv_msg(sock)

        if a.live:
            states = node.controller_states()
            activate = a.activate and states.get(IMPEDANCE_CONTROLLER) == "inactive"
            if states.get(IMPEDANCE_CONTROLLER) != "active" and not activate:
                raise SystemExit(f"{IMPEDANCE_CONTROLLER} is not active ({states.get(IMPEDANCE_CONTROLLER)}). "
                                 "Start a teleop session (or run reset_arm.py, or pass --activate) first.")
            paused = gello_publisher_pid()
            if paused is None:
                raise SystemExit("gello_publisher not found; it must be paused so it stops commanding the arm.")
            spin_for(node, 0.2)
            # inactive: the GELLO pose is not where the arm is, so hold the measured pose
            held = node.snapshot()[0] if activate else (node.gello_cmd or node.snapshot()[0])
            os.kill(paused, signal.SIGSTOP)            # GELLO stops commanding now
            holder.start(held)                          # ...and we feed the watchdog within 0.5 s
            if activate:
                time.sleep(0.2)                         # the controller has a target the moment it starts
                node.switch(activate=[IMPEDANCE_CONTROLLER], deactivate=[])
                print(f"{IMPEDANCE_CONTROLLER} activated, holding the measured pose")
            controller_on = True
            spin_for(node, 0.3)
            q_now = node.snapshot()[0]
            holder.ramp_to(q_now, 1.0)                  # settle onto the measured pose first
            spin_for(node, 1.0)
            print("\n" + "=" * 70)
            print(f"LIVE: the policy will drive the arm for up to {a.steps} steps "
                  f"({a.steps / a.hz:.0f} s). Keep a hand on the e-stop.")
            print("=" * 70)
            if not a.yes and input("Press g then Enter to go, anything else aborts: ").strip().lower() != "g":
                raise SystemExit("aborted; nothing moved.")

        period = 1.0 / a.hz
        last_target = None
        last_grip = None
        t_next = time.monotonic()
        for k in range(a.steps):
            # always drain callbacks: a step slower than the period used to leave 0 s here forever,
            # so nothing was received any more and every input went stale together
            spin_for(node, max(0.005, t_next - time.monotonic()))
            t_next = max(t_next + period, time.monotonic())   # late: skip the missed ticks, no burst
            now = time.monotonic()
            q, dq, q_at, fing, f_at, imgs = node.snapshot()
            ages = {"joints": now - q_at, "gripper": now - f_at,
                    **{c: now - t for c, (_, t) in imgs.items()}}
            stale = {k2: round(v, 3) for k2, v in ages.items() if v > a.max_age}
            if stale:
                raise SystemExit(f"stale inputs {stale}; stopping.")
            send_msg(sock, {"type": "step", "q": q, "dq": dq, "fingers": fing, "t_mono": now,
                            "ages": {k2: round(v, 4) for k2, v in ages.items()},
                            "cams": ["scene", "wrist"]}, [imgs["scene"][0], imgs["wrist"][0]])
            res, _ = recv_msg(sock)
            if "error" in res:
                raise SystemExit(f"server error: {res['error']}")
            tgt = np.asarray(res["q_target"])
            base = np.asarray(last_target if last_target is not None else q)
            step = tgt - base if a.max_step_rad is None else np.clip(tgt - base, -a.max_step_rad, a.max_step_rad)
            clipped = bool(np.any(np.abs(step - (tgt - base)) > 1e-9))
            tgt = base + step
            lat = (time.monotonic() - now) * 1e3
            print(f"[{k:3d}] a={np.round(res['action'], 2).tolist()} grip={'C' if res['gripper_close'] else 'O'} "
                  f"eef={res['eef']} max|dq_cmd|={np.abs(step).max():.3f}{' CLIPPED' if clipped else ''} "
                  f"lat={lat:.0f}ms {res['ms']} {' '.join(res['warn'])}", flush=True)
            if a.live:
                if time.monotonic() - q_at > ROBOT_STATE_TIMEOUT_S:
                    raise SystemExit("robot joint state stopped: ros2_control_node probably died on a reflex.")
                holder.ramp_to(tgt.tolist(), period)
                if res["gripper_close"] != last_grip:
                    node.publish_gripper(res["gripper_close"])
                    last_grip = res["gripper_close"]
            last_target = tgt
            if res.get("done"):
                print("script finished.")
                break
        print("episode finished.")
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        if controller_on:
            try:
                spin_for(node, 0.05)
                q_meas = node.snapshot()[0]
                if q_meas is not None:
                    holder.hold(q_meas)                 # stop chasing the last policy target
                spin_for(node, 0.3)
                node.switch(activate=[], deactivate=[IMPEDANCE_CONTROLLER])
                print("impedance controller deactivated: the arm is holding.")
            except SystemExit as e:
                print(f"WARNING: could not deactivate the controller: {e}")
        if paused is not None:
            try:
                os.kill(paused, signal.SIGCONT)         # never leave the GELLO frozen
            except ProcessLookupError:
                pass
        holder.stop()
        if node.jlog:
            os.makedirs(os.path.dirname(os.path.abspath(a.joint_log)), exist_ok=True)
            np.savez(a.joint_log,
                     joint_t=np.array([x[0] for x in node.jlog]),
                     joint_q=np.array([x[1] for x in node.jlog]),
                     joint_dq=np.array([x[2] for x in node.jlog]),
                     cmd_t=np.array([x[0] for x in node.clog]) if node.clog else np.zeros(0),
                     cmd_q=np.array([x[1] for x in node.clog]) if node.clog else np.zeros((0, 7)),
                     hz=a.hz)
            print(f"joint log: {len(node.jlog)} states, {len(node.clog or [])} commands -> {a.joint_log}")
        sock.close()
        node.destroy_node()
        rclpy.try_shutdown()
        if controller_on:
            print("Next: realign the GELLO to the arm and reactivate control, e.g. run reset_arm.py.")


if __name__ == "__main__":
    main(tyro.cli(Args))

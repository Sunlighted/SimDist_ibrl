"""Payload probe: hold the FR3 still at ~15 poses and record its torque estimates (2026-10-16).

Why: the arm sits ~6 mm below its command because the gravity compensation misses ~0.3 kg
(calib_20261015/README.md "Static sag"). The Robotiq cannot be taken off to be weighed, so the
payload is identified from the robot's own sensors instead: at rest, the controller's external-torque
estimate (`tau_ext_hat_filtered`, computed with the CONFIGURED end-effector load) is exactly the
weight of whatever the configuration leaves out. Several hand orientations separate mass and CoM.

Runs INSIDE the gello ros2 container next to robot_client.py (it imports it):
    cp .../calib_20261015/tools/payload_probe.py /workspace/simdist_runner/
    python3 /workspace/simdist_runner/payload_probe.py                 # dry run: prints config + plan
    python3 /workspace/simdist_runner/payload_probe.py --live --activate \
        --out /workspace/simdist_runner/payload_probe_$(date +%m%d_%H%M).json

Motion: from the DROID home (refuses to start more than 0.15 rad from it -- run reset_arm.py first),
home -> pose -> home for every pose, joint-space ramps at <= --speed rad/s, hold --hold_s s, record the
last --record_s s. Every pose was checked in the sim cell (2026-10-16): joint limits with 0.05 rad margin,
lowest arm/gripper point >= 14 cm above the table, no new self-contact. The sim has no ZED-M / cables:
watch the cables at the j7 +-1.5 rad poses. Gripper: left as it is (open).
Teardown = robot_client.py's: hold the measured pose, deactivate the impedance controller, resume the
GELLO. Realign the GELLO (or run reset_arm.py) afterwards.

Offline: python calib_20261015/tools/fit_payload_probe.py <out.json>
"""
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
import tyro
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_client import (IMPEDANCE_CONTROLLER, Args as ClientArgs, InterpolatingHolder,  # noqa: E402
                          RunnerNode, gello_publisher_pid, spin_for)

HOME = [0.0, -0.6283, 0.0, -2.5133, 0.0, 1.885, 0.0]
# sim-checked (see docstring); deltas from HOME, joint index 0-based
POSES = {
    "home": {}, "j6+0.6": {5: .6}, "j6-0.6": {5: -.6}, "j5+0.8": {4: .8}, "j5-0.8": {4: -.8},
    "j5+0.8,j6-0.5": {4: .8, 5: -.5}, "j5-0.8,j6-0.5": {4: -.8, 5: -.5},
    "j7+1.5": {6: 1.5}, "j7-1.5": {6: -1.5}, "j7+1.5,j6+0.5": {6: 1.5, 5: .5}, "j7-1.5,j6+0.5": {6: -1.5, 5: .5},
    "j4+0.5": {3: .5}, "j4+0.5,j6+0.5": {3: .5, 5: .5}, "j4-0.4,j5+0.8": {3: -.4, 4: .8},
    "j1+0.6,j5-0.8": {0: .6, 4: -.8},
}


@dataclass
class Args:
    live: bool = False
    """Move the arm. Without it: print the configured load (robot_state) and the plan, move nothing."""
    activate: bool = False
    """Impedance controller inactive (e.g. right after reset_arm.py): hold the measured pose and activate."""
    yes: bool = False
    poses: str = ""
    """Comma list of pose names (default: all)."""
    speed: float = 0.3
    """Joint speed limit of the ramps (rad/s)."""
    hold_s: float = 4.0
    record_s: float = 2.5
    out: str = "/workspace/simdist_runner/payload_probe.json"
    robot_state_topic: str = ""
    """Default: the first topic whose type is franka_msgs/msg/FrankaRobotState."""
    robot_joint_topic: str = "/franka/joint_states"
    namespace: str = ""


def to_plain(msg):
    from rosidl_runtime_py.convert import message_to_ordereddict
    return json.loads(json.dumps(message_to_ordereddict(msg), default=float))


def main(a: Args):
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    ca = ClientArgs(namespace=a.namespace, robot_joint_topic=a.robot_joint_topic)
    node = RunnerNode(ca)
    holder = InterpolatingHolder(node)
    from rosidl_runtime_py.utilities import get_message

    spin_for(node, 1.0)
    topic = a.robot_state_topic
    if not topic:
        cands = [(n, t[0]) for n, t in node.get_topic_names_and_types() if t and t[0].endswith("FrankaRobotState")]
        if not cands:
            raise SystemExit("no FrankaRobotState topic found; pass --robot_state_topic "
                             "(ros2 topic list | grep robot_state)")
        topic = cands[0][0]
        rtype = cands[0][1]
    else:
        rtype = dict(node.get_topic_names_and_types())[topic][0]
    latest = {"state": None, "t": 0.0, "effort": None}
    buf = {"on": False, "states": [], "joints": []}

    def on_state(msg):
        # the state can arrive at up to 1 kHz: throttle BEFORE the (slow) dict conversion
        now = time.monotonic()
        latest["t"] = now
        if latest["state"] is None:
            latest["state"] = to_plain(msg)
        elif buf["on"] and (not buf["states"] or now - buf["states"][-1][0] > 0.05):
            buf["states"].append((now, to_plain(msg)))

    def on_joint(msg):
        if buf["on"] and len(msg.effort) >= 7:
            idx = {n: i for i, n in enumerate(msg.name)}
            names = node.joint_names
            if all(n in idx for n in names):
                buf["joints"].append((time.monotonic(), [msg.position[idx[n]] for n in names],
                                      [msg.velocity[idx[n]] for n in names] if msg.velocity else None,
                                      [msg.effort[idx[n]] for n in names]))

    node.create_subscription(get_message(rtype), topic, on_state, 10)
    node.create_subscription(JointState, a.robot_joint_topic, on_joint, 50)
    spin_for(node, 1.5)
    if latest["state"] is None:
        raise SystemExit(f"nothing on {topic} ({rtype})")
    st = latest["state"]
    cfg = {k: v for k, v in st.items() if any(s in k.lower() for s in ("inertia", "m_ee", "m_load", "m_total",
                                                                     "f_x_c", "ee_t_k", "ne_t_ee", "f_t_ne"))}
    print(f"robot state: {topic} ({rtype})")
    print("configured end effector / load (as the controller models it):")
    print(json.dumps(cfg, indent=1)[:3000])

    q0 = node.snapshot()[0]
    if q0 is None:
        raise SystemExit("no joint states")
    off = float(np.abs(np.array(q0) - HOME).max())
    names = [p for p in (a.poses.split(",") if a.poses else POSES)]
    plan = {n: (np.array(HOME) + np.array([POSES[n].get(j, 0.0) for j in range(7)])).tolist() for n in names}
    print(f"\nnow {np.round(q0, 3).tolist()} -- {off:.3f} rad from the DROID home")
    for n, q in plan.items():
        print(f"  {n:16s} {np.round(q, 3).tolist()}")
    total = sum(2 * (np.abs(np.array(q) - HOME).max() / a.speed + 0.5) + a.hold_s for q in plan.values())
    print(f"{len(plan)} poses, ~{total / 60:.1f} min")
    if not a.live:
        json.dump({"topic": topic, "type": rtype, "config_state": st, "plan": plan}, open(a.out, "w"))
        print(f"dry run: nothing moved. config snapshot -> {a.out}")
        node.destroy_node(); rclpy.try_shutdown()
        return
    if off > 0.15:
        raise SystemExit("start at the DROID home (run reset_arm.py): every move goes through it")

    paused, controller_on = None, False
    out = {"topic": topic, "type": rtype, "config_state": st, "plan": plan, "speed": a.speed,
           "hold_s": a.hold_s, "record_s": a.record_s, "poses": {}}
    try:
        states = node.controller_states()
        activate = a.activate and states.get(IMPEDANCE_CONTROLLER) == "inactive"
        if states.get(IMPEDANCE_CONTROLLER) != "active" and not activate:
            raise SystemExit(f"{IMPEDANCE_CONTROLLER} is {states.get(IMPEDANCE_CONTROLLER)}; pass --activate "
                             "or start a teleop session")
        paused = gello_publisher_pid()
        if paused is None:
            raise SystemExit("gello_publisher not found; it must be paused")
        held = node.snapshot()[0] if activate else (node.gello_cmd or node.snapshot()[0])
        os.kill(paused, signal.SIGSTOP)
        holder.start(held)
        if activate:
            time.sleep(0.2)
            node.switch(activate=[IMPEDANCE_CONTROLLER], deactivate=[])
        controller_on = True
        spin_for(node, 0.3)
        holder.ramp_to(node.snapshot()[0], 1.0)
        spin_for(node, 1.0)
        print("\nLIVE: the arm will visit the poses above. Keep a hand on the e-stop; watch the cables at j7.")
        if not a.yes and input("Press g then Enter to go, anything else aborts: ").strip().lower() != "g":
            raise SystemExit("aborted; nothing moved.")

        def go(q):
            cur = np.array(holder.current())
            dur = float(np.abs(np.array(q) - cur).max()) / a.speed + 0.5
            holder.ramp_to(list(q), dur)
            t_end = time.monotonic() + dur
            while time.monotonic() < t_end:
                spin_for(node, 0.05)
                if time.monotonic() - node.snapshot()[2] > 1.0:
                    raise SystemExit("joint states stopped (reflex?)")

        for n, q in plan.items():
            go(HOME)
            go(q)
            spin_for(node, a.hold_s - a.record_s)
            buf["states"], buf["joints"], buf["on"] = [], [], True
            spin_for(node, a.record_s)
            buf["on"] = False
            J = buf["joints"]
            out["poses"][n] = {"q_cmd": q, "states": [s for _, s in buf["states"]],
                               "joint_q": [j[1] for j in J], "joint_dq": [j[2] for j in J],
                               "joint_effort": [j[3] for j in J]}
            e = np.array([j[3] for j in J]).mean(0) if J else np.full(7, np.nan)
            dq = np.array(q) - np.array([j[1] for j in J]).mean(0) if J else np.full(7, np.nan)
            print(f"  {n:16s} {len(buf['states'])} states, {len(J)} joint msgs | cmd-q mrad "
                  f"{np.round(1e3 * dq, 1).tolist()} | effort {np.round(e, 2).tolist()}", flush=True)
            json.dump(out, open(a.out, "w"))
        go(HOME)
        print(f"done -> {a.out}")
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        if controller_on:
            try:
                spin_for(node, 0.05)
                holder.hold(node.snapshot()[0])
                spin_for(node, 0.3)
                node.switch(activate=[], deactivate=[IMPEDANCE_CONTROLLER])
                print("impedance controller deactivated: the arm is holding.")
            except SystemExit as e:
                print(f"WARNING: could not deactivate the controller: {e}")
        if paused is not None:
            try:
                os.kill(paused, signal.SIGCONT)
            except ProcessLookupError:
                pass
        holder.stop()
        if out["poses"]:
            json.dump(out, open(a.out, "w"))
        node.destroy_node()
        rclpy.try_shutdown()
        if controller_on:
            print("Next: realign the GELLO to the arm and reactivate control, e.g. run reset_arm.py.")


if __name__ == "__main__":
    main(tyro.cli(Args))

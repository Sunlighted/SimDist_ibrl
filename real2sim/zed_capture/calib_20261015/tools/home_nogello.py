"""reset_arm.py for a stack WITHOUT the GELLO (bringup_nogello.py): open the Robotiq, drive the FR3 to the
DROID home through the joint impedance controller, then switch the controller off again (the arm holds).

There is no gello_publisher to pause or hand back to: this node publishes /gello/joint_states itself
(reset_arm.CommandHolder, 100 Hz, inside the controller's 0.5 s watchdog) only while the controller is on.
The controller must be INACTIVE when this starts (it always is between runs in the no-GELLO flow).

    python3 /workspace/simdist_runner/home_nogello.py           # shows the move, asks for r + Enter
    python3 /workspace/simdist_runner/home_nogello.py --yes
"""
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, "/workspace/ros2/src/franka_gello_state_publisher/scripts")
import rclpy  # noqa: E402
import tyro  # noqa: E402
from rclpy.signals import SignalHandlerOptions  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import Float32  # noqa: E402

import reset_arm as ra  # noqa: E402

ROBOTIQ_KNUCKLE = "robotiq_85_left_knuckle_joint"


@dataclass
class Args(ra.Args):
    gripper_state_topic: str = "/gripper/joint_states"
    gripper_command_topic: str = "/gripper/gripper_client/target_gripper_width_percent"
    """Robotiq width command, 1.0 = open (the same topic robot_client.py drives)."""
    gripper_open_knuckle: float = 0.05
    """Knuckle angle (rad) below which the Robotiq counts as open."""


def open_robotiq(node, a: Args) -> None:
    knuckle = {}

    def on_state(msg):
        if ROBOTIQ_KNUCKLE in msg.name:
            knuckle["q"] = msg.position[msg.name.index(ROBOTIQ_KNUCKLE)]
        elif msg.position:
            knuckle["q"] = msg.position[0]

    node.create_subscription(JointState, a.gripper_state_topic, on_state, 10)
    pub = node.create_publisher(Float32, a.gripper_command_topic, 10)
    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline:
        pub.publish(Float32(data=1.0))
        ra.refresh(node, 0.1)
        if knuckle.get("q") is not None and knuckle["q"] < a.gripper_open_knuckle:
            print(f"gripper open (knuckle {knuckle['q']:.3f})")
            return
    print(f"WARNING: gripper not confirmed open (knuckle {knuckle.get('q')}); check the gripper stage.")


def main(a: Args) -> None:
    namespace = ra.read_namespace(a.robot_config)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = ra.ResetNode(a, namespace)
    holder = ra.CommandHolder(node)
    controller_on = False
    try:
        node.wait_for_services()
        if ra.gello_publisher_pid() is not None:
            raise SystemExit("gello_publisher is running: use reset_arm.py with a teleop session, not this.")
        states = node.controller_states()
        if states.get(ra.IMPEDANCE_CONTROLLER) == "active":
            raise SystemExit(f"{ra.IMPEDANCE_CONTROLLER} is ACTIVE: something else is commanding the arm. "
                             "Stop it first (its exit switches the controller off).")
        current = ra.wait_for(node, node.robot, "robot joint state")
        if ra.arrived(current, a.pose, a.tolerance):
            print("already at the start pose.")
            open_robotiq(node, a)
            return
        ra.describe_move(current, a.pose, a.duration)
        if not a.yes and input("Press r then Enter to move (opens the gripper first), anything else aborts: "
                               ).strip().lower() != "r":
            raise SystemExit("aborted; nothing moved.")
        open_robotiq(node, a)

        holder.start(list(a.pose))     # the controller has a target the moment it activates
        if ra.IMPEDANCE_CONTROLLER not in states:
            print(f"loading {ra.IMPEDANCE_CONTROLLER}")
            ra.load_controller(ra.IMPEDANCE_CONTROLLER, namespace)
            if ra.IMPEDANCE_CONTROLLER not in node.controller_states():
                raise SystemExit(f"{ra.IMPEDANCE_CONTROLLER} could not be loaded.")
        time.sleep(0.2)
        node.switch(activate=[ra.IMPEDANCE_CONTROLLER], deactivate=[])
        controller_on = True
        print("moving to the start pose")
        ra.move_to_pose(node, a, False, holder)   # inactive -> active: the MotionGenerator smooths the move
        print("at the start pose")
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        if controller_on:
            try:
                ra.refresh(node, 0.05)
                if node.robot() is not None:
                    holder.set(node.robot())      # stop chasing the target before switching off
                ra.refresh(node, 0.3)
                node.switch(activate=[], deactivate=[ra.IMPEDANCE_CONTROLLER])
                print("impedance controller off: the arm is holding.")
            except SystemExit as e:
                print(f"WARNING: could not deactivate the controller: {e}")
        holder.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main(tyro.cli(Args))

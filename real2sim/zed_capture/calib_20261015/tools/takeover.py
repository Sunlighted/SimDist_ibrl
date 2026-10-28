"""Hand the arm to the GELLO WHERE IT STANDS (no move to home): for "policy until it stalls, then a human
finishes" episodes, with a normal teleop session (GELLO running).

After robot_client.py --live is stopped with Ctrl-C the arm holds (impedance controller off) and gello_publisher
is running again. This shows run_teleop_session.py's own matching table; bring the GELLO within the tolerance
(default 15 deg per joint, --max-start-delta) and press g. Exactly as at a session start, the remaining
robot - GELLO difference becomes gello_publisher's joint_offsets, so the arm does not move at the hand-over;
then the controller comes back on and the arm follows the GELLO.

    python3 /workspace/simdist_runner/takeover.py          # g start, f force, q abort

⚠️ The GELLO also commands the gripper the moment gello_publisher resumes: hold the GELLO trigger CLOSED before
you press Ctrl-C in the client, or the Robotiq opens and drops the needle.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/ros2/src/franka_gello_state_publisher/scripts")
import rclpy  # noqa: E402
import tyro  # noqa: E402
from rclpy.signals import SignalHandlerOptions  # noqa: E402

import run_teleop_session as rts  # noqa: E402
from reset_arm import gello_publisher_pid  # noqa: E402


def main(args: rts.Args) -> None:
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    monitor = rts.ReadinessMonitor()
    try:
        if gello_publisher_pid() is None:
            raise SystemExit("gello_publisher is not running: this needs a teleop session (not bringup_nogello.py).")
        # the controller is off, so nothing moves while the offsets change: match on the RAW GELLO pose
        rts.set_gello_offsets(monitor, [0.0] * args.num_arm_joints)
        if not rts.match_before_control(args, monitor):
            print("\naborted; the controller is still off (the arm holds).")
            return
        offsets = rts.handover_offsets(args, monitor)
        rts.set_gello_offsets(monitor, offsets)          # command = GELLO + offset = where the arm is
        rts.activate_controller(args, monitor, Path("/tmp"))
        print("\nthe arm follows the GELLO now (offsets "
              + " ".join(f"{o:+.3f}" for o in offsets) + " rad)")
    except KeyboardInterrupt:
        print("\ninterrupted; the controller is still off (the arm holds).")
    except RuntimeError as error:
        print(f"\n{error}")
    finally:
        monitor.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main(tyro.cli(rts.Args))

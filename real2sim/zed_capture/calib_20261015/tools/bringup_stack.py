"""Bring up cameras + robot + gripper + GELLO publisher exactly as run_teleop_session.py does,
then HOLD: no reset, no matching gate, no controller activation. The arm stays put with the
impedance controller inactive until a runner (robot_client.py --activate, reset_arm.py) takes it.

Used by run_tracking_test.sh. Ctrl+C / SIGTERM tears everything down (the teleop teardown).
Extra arguments go to run_teleop_session's Args (e.g. --gripper robotiq ...).
"""
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/workspace/ros2/src/franka_gello_state_publisher/scripts")
import rclpy  # noqa: E402
import tyro  # noqa: E402
from rclpy.signals import SignalHandlerOptions  # noqa: E402

import run_teleop_session as rts  # noqa: E402


def main(args: rts.Args) -> None:
    log_dir = Path(os.path.expanduser(args.log_dir)) / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"logs: {log_dir}", flush=True)
    os.environ["CYCLONEDDS_URI"] = f"file://{rts.DDS_CONFIG_TWO_HOST}"

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    monitor = rts.ReadinessMonitor()
    session = rts.Session(monitor)
    before, after, _ = rts.build_stages(args, log_dir)

    def on_term(_s, _f):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGHUP, on_term)
    rts.raise_realtime_limits()
    if args.kill_stale:
        stale = rts.stale_processes()
        if stale:
            print("stopping leftover nodes from a previous session:", flush=True)
            rts.stop_stale(stale)
    try:
        rts.check_gello_port(args)
        rts.check_robot_host(args)
        rts.check_fci(args)
        for stage in before:
            session.run(stage)
        session.run(rts.robot_stage(args, log_dir))
        for stage in after:
            session.run(stage)
            if stage.name == "gripper":
                time.sleep(rts.GRIPPER_SETTLE_S)
                if not stage.alive():
                    raise RuntimeError(f"gripper client exited.\n{stage.log_tail()}")
        dead = session.dead_stages()
        if dead:
            raise RuntimeError(f"{', '.join(dead)} exited during start-up.")
        print("STACK_UP (impedance controller left inactive)", flush=True)
        while not session.dead_stages():
            time.sleep(0.5)
        print(f"{', '.join(session.dead_stages())} exited.", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted.", flush=True)
    except RuntimeError as error:
        print(f"\n{error}", flush=True)
    finally:
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, signal.SIG_IGN)
        session.shutdown()
        rts.cleanup_robot_host(args)
        if rclpy.ok():
            monitor.destroy_node()
            rclpy.shutdown()
        print(f"logs kept in {log_dir}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(rts.Args))

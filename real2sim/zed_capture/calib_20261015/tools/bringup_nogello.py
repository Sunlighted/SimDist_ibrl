"""Bring up cameras + robot + gripper exactly as run_teleop_session.py does, but WITHOUT the GELLO:
no gello_publisher, no serial port, nothing writes /gello/joint_states. Then load the joint impedance
controller INACTIVE and HOLD. The arm stays put until home_nogello.py or
robot_client.py --live --no-gello takes it (both publish the command topic themselves and switch the
controller off again when they finish, so the arm holds between runs).

    python3 /workspace/simdist_runner/bringup_nogello.py --gripper robotiq      # Ctrl+C tears it all down

Extra arguments go to run_teleop_session's Args (cameras, camera mode, robot host, ...).
"""
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/workspace/ros2/src/franka_gello_state_publisher/scripts")
import rclpy  # noqa: E402
import tyro  # noqa: E402
from rclpy.signals import SignalHandlerOptions  # noqa: E402

import run_teleop_session as rts  # noqa: E402
from reset_arm import IMPEDANCE_CONTROLLER, load_controller, read_namespace  # noqa: E402


def main(args: rts.Args) -> None:
    log_dir = Path(os.path.expanduser(args.log_dir)) / datetime.now().strftime("%Y%m%d_%H%M%S_nogello")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"logs: {log_dir}", flush=True)
    os.environ["CYCLONEDDS_URI"] = f"file://{rts.DDS_CONFIG_TWO_HOST}"

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    monitor = rts.ReadinessMonitor()
    session = rts.Session(monitor)
    before, after, _ = rts.build_stages(args, log_dir)
    before = [s for s in before if s.name != "gello"]

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
        # the robot comes up with spawn_controller:=false; load the controller now, INACTIVE (no motion)
        print(f"loading {IMPEDANCE_CONTROLLER} (inactive)", flush=True)
        load_controller(IMPEDANCE_CONTROLLER, read_namespace(args.robot_config))
        dead = session.dead_stages()
        if dead:
            raise RuntimeError(f"{', '.join(dead)} exited during start-up.")
        print("STACK_UP, no GELLO (impedance controller inactive). Next: home_nogello.py, then "
              "robot_client.py --live --no-gello", flush=True)
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

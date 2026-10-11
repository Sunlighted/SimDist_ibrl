"""Mirror the latest FR3 joint state into a JSON file the host can read (camera calibration).

The host has no ROS; /workspace is ~/simdist/gello_software on the host, so the file shows up
at ~/simdist/gello_software/simdist_runner/joint_state_latest.json.

Run inside ros2_devcontainer-gello-ros2-1 while the teleop session is up:
    source /opt/ros/humble/setup.bash && source /workspace/ros2/install/setup.bash
    python3 /workspace/simdist_runner/joint_tap.py            # robot on the NUC (default session)
    python3 /workspace/simdist_runner/joint_tap.py --local    # robot on this machine
"""
import argparse
import json
import os
import time

DDS_DIR = "/workspace/ros2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/franka/joint_states")
    ap.add_argument("--prefix", default="fr3_joint")
    ap.add_argument("--out", default="/workspace/simdist_runner/joint_state_latest.json")
    ap.add_argument("--local", action="store_true", help="robot runs on this machine, not the NUC")
    args = ap.parse_args()

    # Same DDS profile as run_teleop_session.py, otherwise the topic is not discovered.
    if "CYCLONEDDS_URI" not in os.environ:
        xml = "cyclonedds_localhost.xml" if args.local else "cyclonedds_ws.xml"
        os.environ["CYCLONEDDS_URI"] = f"file://{DDS_DIR}/{xml}"
    print("DDS:", os.environ["CYCLONEDDS_URI"])

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState

    names = [f"{args.prefix}{i}" for i in range(1, 8)]
    tmp = args.out + ".tmp"
    last = [0.0]

    def on_msg(msg):
        now = time.time()
        if now - last[0] < 0.02:  # 50 Hz is plenty
            return
        idx = {n: i for i, n in enumerate(msg.name)}
        if not all(n in idx for n in names):
            return
        q = [float(msg.position[idx[n]]) for n in names]
        with open(tmp, "w") as f:
            json.dump({"t": now, "names": names, "q": q}, f)
        os.replace(tmp, args.out)  # atomic: the host never reads half a file
        if last[0] == 0.0:
            print("receiving joints:", [round(x, 3) for x in q])
        last[0] = now

    rclpy.init()
    node = Node("calib_joint_tap")
    node.create_subscription(JointState, args.topic, on_msg, qos_profile_sensor_data)
    print(f"listening on {args.topic} -> {args.out}  (Ctrl-C to stop)")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

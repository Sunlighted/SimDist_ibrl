"""Read-only snapshot for the in-hand needle fit: raw scene + wrist JPEGs (FRAMES of each, full res) plus the
arm's 7 joint angles and the Robotiq knuckle, into OUT_ROOT/grasp_NN/ (NN = next free number).

    python3 /workspace/simdist_runner/grab_inhand.py /workspace/inhand_1027   # host: ~/simdist/gello_software/inhand_1027
"""
import json, os, re, sys, time
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, JointState

FRAMES = 5
root = sys.argv[1]; os.makedirs(root, exist_ok=True)
n = 1 + max([int(d.split("_")[1]) for d in os.listdir(root) if re.fullmatch(r"grasp_\d+", d)] + [0])
out = os.path.join(root, f"grasp_{n:02d}"); os.makedirs(out)
rclpy.init(); node = rclpy.create_node("grab_inhand")
frames = {"scene": [], "wrist": []}; state = {}

def on_img(msg, cam):
    if len(frames[cam]) < FRAMES:
        frames[cam].append(bytes(msg.data))

def on_arm(msg):
    js = {int(m.group(1)): p for name, p in zip(msg.name, msg.position) if (m := re.search(r"joint([1-7])$", name))}
    if len(js) == 7:
        state["q"] = [js[i] for i in range(1, 8)]

def on_grip(msg):
    if msg.position:
        state["knuckle"] = msg.position[msg.name.index("robotiq_85_left_knuckle_joint")
                                        if "robotiq_85_left_knuckle_joint" in msg.name else 0]

for cam in frames:
    node.create_subscription(CompressedImage, f"/cameras/{cam}/left/image/compressed",
                             lambda m, c=cam: on_img(m, c), qos_profile_sensor_data)
node.create_subscription(JointState, "/franka/joint_states", on_arm, 10)
node.create_subscription(JointState, "/gripper/joint_states", on_grip, 10)
end = time.time() + 8
while time.time() < end and not (all(len(v) >= FRAMES for v in frames.values()) and "q" in state and "knuckle" in state):
    rclpy.spin_once(node, timeout_sec=0.05)
for cam, bs in frames.items():
    for i, b in enumerate(bs):
        open(os.path.join(out, f"{cam}_{i}.jpg"), "wb").write(b)
state["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(state, open(os.path.join(out, "state.json"), "w"), indent=1)
print(out, {c: len(v) for c, v in frames.items()}, "q" in state and "q ok", "knuckle", state.get("knuckle"))
if "q" not in state or "knuckle" not in state or not all(frames.values()):
    print("WARNING: something missing -- is the stack up?")
node.destroy_node(); rclpy.shutdown()

"""Save one JPEG from each camera topic (read-only): python3 grab_frames.py OUT_DIR"""
import os, sys, time
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
rclpy.init(); node = rclpy.create_node("grab_frames"); got = {}
for cam in ("scene", "wrist"):
    node.create_subscription(CompressedImage, f"/cameras/{cam}/left/image/compressed",
                             lambda m, c=cam: got.setdefault(c, bytes(m.data)), qos_profile_sensor_data)
end = time.time() + 5
while len(got) < 2 and time.time() < end:
    rclpy.spin_once(node, timeout_sec=0.1)
for c, b in got.items():
    open(os.path.join(out, f"{c}.jpg"), "wb").write(b)
print("saved", sorted(got))

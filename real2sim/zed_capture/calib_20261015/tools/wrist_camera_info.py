"""Stop-gap: publish the wrist ZED-M camera_info when publish_zed_uvc could not read its serial
(the ZED-M HID interface did not enumerate, 2026-10-15). Same message the publisher would latch."""
import sys
from pathlib import Path
import rclpy
sys.path.insert(0, "/workspace/ros2/src/franka_gello_state_publisher/scripts")
from publish_zed_uvc import CAMERA_INFO_QOS, camera_info  # noqa: E402
from sensor_msgs.msg import CameraInfo  # noqa: E402

rclpy.init()
node = rclpy.create_node("wrist_camera_info_stopgap")
info = camera_info(Path("/workspace/ros2/zed_calibration/SN16478414.conf"), "hd720", "left",
                   1280, 720, "wrist_left")
assert info is not None
pub = node.create_publisher(CameraInfo, "/cameras/wrist/left/camera_info", CAMERA_INFO_QOS)
node.create_timer(1.0, lambda: (setattr(info.header, "stamp", node.get_clock().now().to_msg()), pub.publish(info)))
print("publishing /cameras/wrist/left/camera_info", list(info.k)[:6], flush=True)
rclpy.spin(node)

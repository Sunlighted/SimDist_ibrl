"""Is robosuite's robot0_right_hand the SAME FRAME polymetis reports?
2d-y only checked POSITION (0.5 mm).  Orientation is the other half.
"""
import os, sys, json
os.environ["MUJOCO_GL"] = "osmesa"
sys.path.insert(0,"deps/mimicgen"); sys.path.insert(0,"deps/robomimic"); sys.path.insert(0,".")
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import env.peg_insertion  # noqa
import robosuite
from robosuite.controllers import load_controller_config
R2 = "real2sim/zed_capture/scan_run2"
def T(R,t):
    M=np.eye(4); M[:3,:3]=R; M[:3,3]=np.asarray(t).ravel(); return M
def inv(M):
    R,t=M[:3,:3],M[:3,3]; return T(R.T,-R.T@t)

cc = load_controller_config(default_controller="OSC_POSE")
e = robosuite.make("PegInsertion_D0", robots="Panda", gripper_types="Robotiq85Gripper",
                   controller_configs=cc, has_renderer=False, has_offscreen_renderer=False,
                   use_camera_obs=False, horizon=60)
e.reset()
aq = e.robots[0]._ref_joint_pos_indexes
bid = e.sim.model.body_name2id("robot0_base")

for body in ["robot0_right_hand", "robot0_link7", "gripper0_robotiq_85_adapter_link"]:
    try: e.sim.model.body_name2id(body)
    except Exception: print(f"{body}: ABSENT"); continue
    offs, pos = [], []
    for i in range(14):
        st = json.load(open(f"{R2}/pose_{i:03d}/state.json"))
        e.sim.data.qpos[aq] = np.array(st["joints"]); e.sim.data.qvel[:] = 0; e.sim.forward()
        bq = e.sim.data.body_xquat[bid]
        T_sw_base = T(Rot.from_quat([bq[1],bq[2],bq[3],bq[0]]).as_matrix(), e.sim.data.body_xpos[bid])
        h = e.sim.model.body_name2id(body); hq = e.sim.data.body_xquat[h]
        T_base_hand = inv(T_sw_base) @ T(Rot.from_quat([hq[1],hq[2],hq[3],hq[0]]).as_matrix(),
                                         e.sim.data.body_xpos[h])
        T_base_ee = T(Rot.from_quat(st["ee_quat"]).as_matrix(), st["ee_pos"])
        offs.append(T_base_hand[:3,:3].T @ T_base_ee[:3,:3])
        pos.append(np.linalg.norm(T_base_hand[:3,3] - T_base_ee[:3,3])*1000)
    ang = [np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))) for R in offs]
    mean_off = Rot.from_matrix(np.array(offs)).mean()
    spread = [np.degrees(np.arccos(np.clip((np.trace(mean_off.as_matrix().T@R)-1)/2,-1,1))) for R in offs]
    print(f"{body:36s} pos {np.mean(pos):6.1f} mm | rot offset {np.mean(ang):6.2f} deg "
          f"(constant to {max(spread):.3f} deg) axis-angle {np.round(mean_off.as_rotvec(degrees=True),2)}")
    print(f"{'':36s} euler zyx of offset: {np.round(mean_off.as_euler('zyx',degrees=True),2)}")

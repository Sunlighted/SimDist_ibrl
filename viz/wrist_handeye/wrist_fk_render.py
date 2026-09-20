"""Wrist-camera check with the MuJoCo MESH (no splat), through the hand-eye only.

For each scan pose:
  sim arm  <- state.json "joints"            (the real configuration)
  camera   <- T_sw_ee . T_ee_zedm            (FK chain -- what we want to test)
Nothing here is fit to the board, so the arm landing on the real arm IS the test.
Gripper set to Robotiq85 to match the hardware present at scan time.
"""
import os, sys, json
os.environ["MUJOCO_GL"] = "osmesa"
sys.path.insert(0, "deps/mimicgen"); sys.path.insert(0, "deps/robomimic"); sys.path.insert(0, ".")
import numpy as np, cv2
from scipy.spatial.transform import Rotation
import env.peg_insertion  # noqa
import robosuite
from robosuite.controllers import load_controller_config
from robosuite.utils.binding_utils import MjSim, MjRenderContextOffscreen

R2, SP = "real2sim/zed_capture/scan_run2", "real2sim/scan_pipeline"
OUT = "/scratch/yifanr/temp/claude-3144347/-project-6101830-yifanr-SimDist-ibrl/a1b0d322-aaab-41c7-9a4a-93e3923ef768/scratchpad"
W, H = 1280, 720
POSES = ["pose_000", "pose_006", "pose_012"]

def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.asarray(t).ravel(); return M
def mj_quat(R_cv):
    q = Rotation.from_matrix(R_cv @ np.diag([1.0, -1.0, -1.0])).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])

sol  = json.load(open(f"{SP}/ground_run2_solved.json"))
T_ee_cam = np.array(sol["T_ee_zedm"])
intr = json.load(open(f"{R2}/intrinsics.json"))["zedm"]
wn, hn = intr["resolution"]; fy = intr["left"]["fy"]
fovy = float(np.degrees(2*np.arctan(hn/(2*fy))))
ppx = (intr["left"]["cx"] - wn/2) * W/wn          # MuJoCo renders a CENTRED pinhole
ppy = (intr["left"]["cy"] - hn/2) * H/hn
Mpp = np.float32([[1, 0, ppx], [0, 1, ppy]])
print(f"[cam] fovy {fovy:.2f} deg, principal-point shift ({ppx:+.1f},{ppy:+.1f}) px @ {W}")

cc = load_controller_config(default_controller="OSC_POSE")
try:
    e = robosuite.make("PegInsertion_D0", robots="Panda", gripper_types="Robotiq85Gripper",
                       controller_configs=cc, has_renderer=False,
                       has_offscreen_renderer=False, use_camera_obs=False, horizon=60)
    print("[env] gripper = Robotiq85Gripper (matches scan-time hardware)")
except Exception as ex:
    print(f"[env] Robotiq85 unavailable ({ex}); falling back to default"); raise
e.reset()
arm_q = e.robots[0]._ref_joint_pos_indexes

cam0 = '<camera name="zedm" pos="0 0 1" quat="1 0 0 0" fovy="%f"/>' % fovy
sim = MjSim.from_xml_string(e.sim.model.get_xml().replace("</worldbody>", cam0 + "</worldbody>"))
MjRenderContextOffscreen(sim, device_id=-1, max_width=W, max_height=H)
cid = sim.model.camera_name2id("zedm")
rob = [g for g in range(sim.model.ngeom)
       if (sim.model.geom_id2name(g) or "").startswith(("robot0", "gripper0"))]

panels = []
for p in POSES:
    st = json.load(open(f"{R2}/{p}/state.json"))
    e.sim.data.qpos[arm_q] = np.array(st["joints"])
    e.sim.data.qvel[:] = 0; e.sim.forward()

    # robosuite's robot0_right_hand is the polymetis flange rotated +45.03 deg
    # about z (verified constant to 0.000 deg over all 14 poses).  2d-y only
    # checked position, which cannot see this.
    bid = e.sim.model.body_name2id("robot0_right_hand")
    bq = e.sim.data.body_xquat[bid]
    T_sw_hand = T(Rotation.from_quat([bq[1], bq[2], bq[3], bq[0]]).as_matrix(),
                  e.sim.data.body_xpos[bid])
    T_sw_ee = T_sw_hand @ T(Rotation.from_euler("z", 45.03, degrees=True).as_matrix(), [0, 0, 0])
    T_sw_cam = T_sw_ee @ T_ee_cam
    sim.model.cam_pos[cid] = T_sw_cam[:3, 3]
    sim.model.cam_quat[cid] = mj_quat(T_sw_cam[:3, :3])
    sim.data.qpos[:] = e.sim.data.qpos; sim.data.qvel[:] = 0; sim.forward()

    rgb = sim.render(camera_name="zedm", width=W, height=H)[::-1]
    seg = sim.render(camera_name="zedm", width=W, height=H, segmentation=True)[::-1][:, :, 1]
    mask = np.isin(seg, rob).astype(np.uint8)
    rgb  = cv2.warpAffine(rgb[:, :, ::-1], Mpp, (W, H), borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpAffine(mask, Mpp, (W, H), flags=cv2.INTER_NEAREST)

    real = cv2.resize(cv2.imread(f"{R2}/{p}/zedm_left.png"), (W, H))
    ov = real.copy()
    out = cv2.dilate(mask, np.ones((3, 3), np.uint8)) - mask
    ov[mask > 0] = (0.55*ov[mask > 0] + 0.45*np.array([0, 0, 255])).astype(np.uint8)
    ov[out > 0] = (0, 255, 255)
    print(f"[{p}] robot covers {100*mask.mean():.1f}% of the wrist frame, "
          f"cam@{np.round(T_sw_cam[:3,3],3)}")
    band = lambda im, s: cv2.putText(im.copy(), s, (24, 44), cv2.FONT_HERSHEY_SIMPLEX,
                                     1.1, (255, 255, 255), 3, cv2.LINE_AA)
    panels.append(np.concatenate([cv2.resize(band(real, f"{p} REAL"), (600, 338)),
                                  np.full((338, 5, 3), 255, np.uint8),
                                  cv2.resize(band(rgb, "MUJOCO via FK.Rz(45).T_ee_zedm"), (600, 338)),
                                  np.full((338, 5, 3), 255, np.uint8),
                                  cv2.resize(band(ov, "OVERLAY"), (600, 338))], 1))
cv2.imwrite(f"{OUT}/wrist_fk_mesh_fixed.png",
            np.concatenate([np.concatenate([q, np.full((5, q.shape[1], 3), 255, np.uint8)], 0)
                            for q in panels], 0))
print("[done]")

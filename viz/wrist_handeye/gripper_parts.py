import os, sys, json
os.environ["MUJOCO_GL"]="osmesa"
sys.path.insert(0,"deps/mimicgen"); sys.path.insert(0,"deps/robomimic"); sys.path.insert(0,".")
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot
import env.peg_insertion, robosuite
from robosuite.controllers import load_controller_config
from robosuite.utils.binding_utils import MjSim, MjRenderContextOffscreen
R2,SP="real2sim/zed_capture/scan_run2","real2sim/scan_pipeline"
O="/scratch/yifanr/temp/claude-3144347/-project-6101830-yifanr-SimDist-ibrl/a1b0d322-aaab-41c7-9a4a-93e3923ef768/scratchpad"
W,H=1280,720
def T(R,t):
    M=np.eye(4);M[:3,:3]=R;M[:3,3]=np.asarray(t).ravel();return M
def mjq(R):
    q=Rot.from_matrix(R@np.diag([1.,-1.,-1.])).as_quat();return np.array([q[3],q[0],q[1],q[2]])
real=np.load(f"{O}/real_gripper_mask.npy")
P=np.load(f"{O}/gripper_fit_params.npy")
T_ee_cam=np.array(json.load(open(f"{SP}/ground_run2_solved.json"))["T_ee_zedm"])
intr=json.load(open(f"{R2}/intrinsics.json"))["zedm"]; wn,hn=intr["resolution"]
fovy=float(np.degrees(2*np.arctan(hn/(2*intr["left"]["fy"]))))
Mpp=np.float32([[1,0,(intr["left"]["cx"]-wn/2)*W/wn],[0,1,(intr["left"]["cy"]-hn/2)*H/hn]])
cc=load_controller_config(default_controller="OSC_POSE")
e=robosuite.make("PegInsertion_D0",robots="Panda",controller_configs=cc,has_renderer=False,
                 has_offscreen_renderer=False,use_camera_obs=False,horizon=60);e.reset()
aq=e.robots[0]._ref_joint_pos_indexes; gq=e.robots[0]._ref_gripper_joint_pos_indexes
e.sim.data.qpos[aq]=np.array(json.load(open(f"{R2}/pose_000/state.json"))["joints"]); e.sim.forward()
sim=MjSim.from_xml_string(e.sim.model.get_xml().replace("</worldbody>",
    f'<camera name="zedm" pos="0 0 1" fovy="{fovy}"/></worldbody>'))
MjRenderContextOffscreen(sim,device_id=-1,max_width=W,max_height=H)
cid=sim.model.camera_name2id("zedm"); bid=sim.model.body_name2id("robot0_right_hand")
Rz=T(Rot.from_euler("z",45.03,degrees=True).as_matrix(),[0,0,0])
ow,tx,ty,tz,rx,ry,rz=P[1]
sim.data.qpos[:]=e.sim.data.qpos
sim.data.qpos[gq[0]]=np.clip(ow,0,0.04); sim.data.qpos[gq[1]]=-np.clip(ow,0,0.04)
sim.forward()
bq=sim.data.body_xquat[bid]
C=T(Rot.from_quat([bq[1],bq[2],bq[3],bq[0]]).as_matrix(),sim.data.body_xpos[bid])@Rz@ \
  T(Rot.from_rotvec([rx,ry,rz]).as_matrix(),[tx,ty,tz])@T_ee_cam
sim.model.cam_pos[cid]=C[:3,3]; sim.model.cam_quat[cid]=mjq(C[:3,:3]); sim.forward()
seg=cv2.warpAffine(sim.render(camera_name="zedm",width=W,height=H,segmentation=True)[::-1][:,:,1].astype(np.int32),
                   Mpp,(W,H),flags=cv2.INTER_NEAREST)
parts={}
for g in range(sim.model.ngeom):
    nm=sim.model.geom_id2name(g) or ""
    if nm.startswith("gripper0") and "visual" in nm:
        m=(seg==g); 
        if m.sum(): parts[nm]=m
for k,v in parts.items():
    ys,xs=np.nonzero(v)
    print(f"{k:26s} {v.sum():6d} px  x{xs.min()}..{xs.max()} y{ys.min()}..{ys.max()}  "
          f"IoU-with-real {(v&(real>0)).sum()/((v|(real>0)).sum()):.3f}")
vis=np.zeros((H,W,3),np.uint8)
col={"gripper0_hand_visual":(255,120,0),"gripper0_finger1_visual":(0,0,255),"gripper0_finger2_visual":(0,255,0)}
for k,v in parts.items(): vis[v]=col.get(k,(200,200,200))
out=cv2.dilate((real>0).astype(np.uint8),np.ones((3,3),np.uint8))-(real>0).astype(np.uint8)
vis[out>0]=(0,255,255)
ys,xs=np.nonzero(real); cv2.drawMarker(vis,(int(xs[ys==ys.min()].mean()),int(ys.min())),(255,255,255),cv2.MARKER_CROSS,30,2)
cv2.imwrite(f"{O}/gripper_parts.png",vis[400:720,300:1000])
print("yellow outline = REAL finger; blue=finger1 green=finger2 orange=hand")

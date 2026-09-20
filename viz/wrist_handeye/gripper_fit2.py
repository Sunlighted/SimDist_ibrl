"""Fit T_ee_zedm correction to the real finger silhouette -- VISUAL geoms only.

mjvOption.geomgroup[0]=0 hides the collision meshes, which otherwise render on
top of the visual ones and poison the segmentation mask.
"""
import os, sys, json
os.environ["MUJOCO_GL"]="osmesa"
sys.path.insert(0,"deps/mimicgen"); sys.path.insert(0,"deps/robomimic"); sys.path.insert(0,".")
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import minimize
import env.peg_insertion, robosuite
from robosuite.controllers import load_controller_config
from robosuite.utils.binding_utils import MjSim, MjRenderContextOffscreen
R2,SP="real2sim/zed_capture/scan_run2","real2sim/scan_pipeline"
O="/scratch/yifanr/temp/claude-3144347/-project-6101830-yifanr-SimDist-ibrl/a1b0d322-aaab-41c7-9a4a-93e3923ef768/scratchpad"
def T(R,t):
    M=np.eye(4);M[:3,:3]=R;M[:3,3]=np.asarray(t).ravel();return M
def mjq(R):
    q=Rot.from_matrix(R@np.diag([1.,-1.,-1.])).as_quat();return np.array([q[3],q[0],q[1],q[2]])
real_full=np.load(f"{O}/real_gripper_mask.npy")
T_ee_cam=np.array(json.load(open(f"{SP}/ground_run2_solved.json"))["T_ee_zedm"])
intr=json.load(open(f"{R2}/intrinsics.json"))["zedm"]; wn,hn=intr["resolution"]
fovy=float(np.degrees(2*np.arctan(hn/(2*intr["left"]["fy"]))))
cc=load_controller_config(default_controller="OSC_POSE")
e=robosuite.make("PegInsertion_D0",robots="Panda",controller_configs=cc,has_renderer=False,
                 has_offscreen_renderer=False,use_camera_obs=False,horizon=60);e.reset()
aq=e.robots[0]._ref_joint_pos_indexes; gq=e.robots[0]._ref_gripper_joint_pos_indexes
e.sim.data.qpos[aq]=np.array(json.load(open(f"{R2}/pose_000/state.json"))["joints"]); e.sim.forward()
sim=MjSim.from_xml_string(e.sim.model.get_xml().replace("</worldbody>",
    f'<camera name="zedm" pos="0 0 1" fovy="{fovy}"/></worldbody>'))
ctx=MjRenderContextOffscreen(sim,device_id=-1,max_width=1280,max_height=720)
ctx.vopt.geomgroup[:]=0; ctx.vopt.geomgroup[1]=1      # visual meshes only
cid=sim.model.camera_name2id("zedm"); bid=sim.model.body_name2id("robot0_right_hand")
gg=[g for g in range(sim.model.ngeom)
    if (sim.model.geom_id2name(g) or "").startswith("gripper0") and "visual" in (sim.model.geom_id2name(g) or "")]
Rz=T(Rot.from_euler("z",45.03,degrees=True).as_matrix(),[0,0,0])
def render(p,w,h):
    ow,tx,ty,tz,rx,ry,rz=p
    sim.data.qpos[:]=e.sim.data.qpos
    sim.data.qpos[gq[0]]=np.clip(ow,0,0.04); sim.data.qpos[gq[1]]=-np.clip(ow,0,0.04)
    sim.data.qvel[:]=0; sim.forward()
    bq=sim.data.body_xquat[bid]
    C=T(Rot.from_quat([bq[1],bq[2],bq[3],bq[0]]).as_matrix(),sim.data.body_xpos[bid])@Rz@ \
      T(Rot.from_rotvec([rx,ry,rz]).as_matrix(),[tx,ty,tz])@T_ee_cam
    sim.model.cam_pos[cid]=C[:3,3]; sim.model.cam_quat[cid]=mjq(C[:3,:3]); sim.forward()
    seg=sim.render(camera_name="zedm",width=w,height=h,segmentation=True)[::-1][:,:,1]
    Mpp=np.float32([[1,0,(intr["left"]["cx"]-wn/2)*w/wn],[0,1,(intr["left"]["cy"]-hn/2)*h/hn]])
    return cv2.warpAffine(np.isin(seg,gg).astype(np.uint8),Mpp,(w,h),flags=cv2.INTER_NEAREST)
def iou(m,r):
    u=(m|r).sum(); return 0.0 if u==0 else (m&r).sum()/u
def tip(m):
    ys,xs=np.nonzero(m); return (xs[ys==ys.min()].mean(), ys.min())
FW,FH=640,360
real=cv2.resize(real_full,(FW,FH),interpolation=cv2.INTER_NEAREST)
cost=lambda p: 1.0-iou(render(p,FW,FH),real)
p0=np.array([0.04,0,0,0,0,0,0])
print(f"published T_ee_zedm, fully open:  IoU {iou(render(p0,1280,720),real_full):.3f}  "
      f"n {render(p0,1280,720).sum()} (real {real_full.sum()})")
f3=lambda q: cost(np.array([q[0],q[1],q[2],q[3],0,0,0]))
r3=minimize(f3,[0.04,0,0,0],method="Nelder-Mead",options=dict(xatol=2e-4,fatol=1e-4,maxfev=1500,
    initial_simplex=[[0.04,0,0,0],[0.025,0,0,0],[0.04,.008,0,0],[0.04,0,.008,0],[0.04,0,0,.008]]))
p3=np.array([r3.x[0],*r3.x[1:],0,0,0])
r6=minimize(cost,p3,method="Nelder-Mead",options=dict(xatol=2e-4,fatol=1e-4,maxfev=4000))
for tag,p in [("published",p0),("trans-fit",p3),("6dof-fit",r6.x)]:
    m=render(p,1280,720); tx,ty=tip(m); rt=tip(real_full)
    print(f"{tag:10s} IoU {iou(m,real_full):.3f}  n {m.sum():5d}  open {p[0]*1000:4.1f} mm  "
          f"dt {np.round(np.array(p[1:4])*1000,1)} mm  dR {np.round(np.degrees(np.array(p[4:])),2)} deg  "
          f"tip d=({tx-rt[0]:+5.1f},{ty-rt[1]:+5.1f}) px")
np.save(f"{O}/fit2_params.npy",np.stack([p0,p3,r6.x]))
vis=cv2.cvtColor(cv2.resize(cv2.imread(f"{R2}/pose_000/zedm_left.png"),(1280,720)),cv2.COLOR_BGR2BGRA)[:,:,:3]
for p,c in [(p0,(255,80,0)),(r6.x,(0,0,255))]:
    m=render(p,1280,720); o=cv2.dilate(m,np.ones((3,3),np.uint8))-m; vis[o>0]=c
o=cv2.dilate(real_full,np.ones((3,3),np.uint8))-real_full; vis[o>0]=(0,255,255)
cv2.imwrite(f"{O}/gripper_fit_overlay.png",vis[430:720,300:1000])
print("overlay: yellow=REAL  orange=published T_ee_zedm  red=fitted")

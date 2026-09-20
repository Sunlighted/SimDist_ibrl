"""Gripper fully open (owner-confirmed) => opening is no longer a free parameter.
Refit the 3-dof translation correction to T_ee_zedm, then render the side-by-side.
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
W,H=1280,720; OPEN=0.04                      # fully open, both fingers
def T(R,t):
    M=np.eye(4);M[:3,:3]=R;M[:3,3]=np.asarray(t).ravel();return M
def inv(M):
    R,t=M[:3,:3],M[:3,3];return T(R.T,-R.T@t)
def mjq(R):
    q=Rot.from_matrix(R@np.diag([1.,-1.,-1.])).as_quat();return np.array([q[3],q[0],q[1],q[2]])

g=[]
for i in range(14):
    im=cv2.resize(cv2.imread(f"{R2}/pose_{i:03d}/zedm_left.png",cv2.IMREAD_GRAYSCALE),(W,H)).astype(np.float32)
    g.append(im/max(np.median(im),1.0))
g=np.stack(g); std,med=g.std(0),np.median(g,0)
def real_mask(pct):
    c=((std<np.percentile(std,pct))&(med<np.percentile(med,pct))).astype(np.uint8)
    c=cv2.morphologyEx(c,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    c=cv2.morphologyEx(c,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    n,lab,st,_=cv2.connectedComponentsWithStats(c,8)
    bi=max(range(1,n),key=lambda k: st[k][4]*(2.0 if (st[k][1]+st[k][3])>=H-3 else .4) if st[k][4]>=800 else -1)
    return (lab==bi).astype(np.uint8)

T_ee_cam=np.array(json.load(open(f"{SP}/ground_run2_solved.json"))["T_ee_zedm"])
intr=json.load(open(f"{R2}/intrinsics.json"))["zedm"]; wn,hn=intr["resolution"]
fovy=float(np.degrees(2*np.arctan(hn/(2*intr["left"]["fy"]))))
cc=load_controller_config(default_controller="OSC_POSE")
e=robosuite.make("PegInsertion_D0",robots="Panda",controller_configs=cc,has_renderer=False,
                 has_offscreen_renderer=False,use_camera_obs=False,horizon=60);e.reset()
print(f"[env] gripper = {type(e.robots[0].gripper).__name__}, opening pinned FULLY OPEN")
aq=e.robots[0]._ref_joint_pos_indexes; gq=e.robots[0]._ref_gripper_joint_pos_indexes
e.sim.data.qpos[aq]=np.array(json.load(open(f"{R2}/pose_000/state.json"))["joints"]); e.sim.forward()
sim=MjSim.from_xml_string(e.sim.model.get_xml().replace("</worldbody>",
    f'<camera name="zedm" pos="0 0 1" fovy="{fovy}"/></worldbody>'))
ctx=MjRenderContextOffscreen(sim,device_id=-1,max_width=W,max_height=H)
cid=sim.model.camera_name2id("zedm"); bid=sim.model.body_name2id("robot0_right_hand")
gvis=[j for j in range(sim.model.ngeom)
      if (sim.model.geom_id2name(j) or "").startswith("gripper0") and "visual" in (sim.model.geom_id2name(j) or "")]
Rz=T(Rot.from_euler("z",45.03,degrees=True).as_matrix(),[0,0,0])
def setcam(dt):
    sim.data.qpos[:]=e.sim.data.qpos
    sim.data.qpos[gq[0]]=OPEN; sim.data.qpos[gq[1]]=-OPEN; sim.data.qvel[:]=0; sim.forward()
    bq=sim.data.body_xquat[bid]
    C=T(Rot.from_quat([bq[1],bq[2],bq[3],bq[0]]).as_matrix(),sim.data.body_xpos[bid])@Rz@T(np.eye(3),dt)@T_ee_cam
    sim.model.cam_pos[cid]=C[:3,3]; sim.model.cam_quat[cid]=mjq(C[:3,:3]); sim.forward()
def gmask(dt,w=W,h=H):
    setcam(dt)
    ctx.vopt.geomgroup[:]=0; ctx.vopt.geomgroup[1]=1
    seg=sim.render(camera_name="zedm",width=w,height=h,segmentation=True)[::-1][:,:,1]
    Mpp=np.float32([[1,0,(intr["left"]["cx"]-wn/2)*w/wn],[0,1,(intr["left"]["cy"]-hn/2)*h/hn]])
    return cv2.warpAffine(np.isin(seg,gvis).astype(np.uint8),Mpp,(w,h),flags=cv2.INTER_NEAREST)
def rgb(dt):
    setcam(dt)
    ctx.vopt.geomgroup[:]=0; ctx.vopt.geomgroup[1]=1
    im=sim.render(camera_name="zedm",width=W,height=H)[::-1][:,:,::-1]
    Mpp=np.float32([[1,0,(intr["left"]["cx"]-wn/2)],[0,1,(intr["left"]["cy"]-hn/2)*H/hn]])
    return cv2.warpAffine(im,Mpp,(W,H),borderMode=cv2.BORDER_REPLICATE)

sols=[]
for pct in [10,12,14,16,20]:
    rm=real_mask(pct)
    c=lambda d:(lambda m:1-((m&rm).sum()/max((m|rm).sum(),1)))(gmask(d))
    r=minimize(c,[0,0,0],method="Nelder-Mead",options=dict(xatol=1e-4,fatol=1e-4,maxfev=800,
        initial_simplex=[[0,0,0],[.006,0,0],[0,.006,0],[0,0,.006]]))
    sols.append(r.x*1000)
    print(f"pct {pct:2d}: mask {rm.sum():5d} px | IoU {1-r.fun:.3f} | dt {np.round(r.x*1000,2)} mm |dt| {np.linalg.norm(r.x)*1000:.2f}")
S=np.array(sols); DT=S.mean(0)/1000
print(f"\nopening-pinned correction: {np.round(S.mean(0),2)} mm, per-axis SD {np.round(S.std(0),2)} mm, |dt| {np.linalg.norm(S.mean(0)):.2f} mm")
real=real_mask(14)
m0,m1=gmask([0,0,0]),gmask(DT)
J=lambda a,b:(a&b).sum()/(a|b).sum()
for tag,m in [("published",m0),("fitted",m1)]:
    ys,xs=np.nonzero(m); ry,rx=np.nonzero(real)
    print(f"{tag:10s} IoU {J(m,real):.3f}  tip dy {ys.min()-ry.min():+4d} px  "
          f"centroid ({xs.mean()-rx.mean():+5.1f},{ys.mean()-ry.mean():+5.1f}) px  area {m.sum()} vs {real.sum()}")
np.save(f"{O}/DT_open.npy",DT)

edge=lambda m:(cv2.dilate(m,np.ones((3,3),np.uint8))-m)>0
full,zoom=[],[]
for p in ["pose_000","pose_006","pose_012"]:
    e.sim.data.qpos[aq]=np.array(json.load(open(f"{R2}/{p}/state.json"))["joints"]); e.sim.data.qvel[:]=0; e.sim.forward()
    R=cv2.resize(cv2.imread(f"{R2}/{p}/zedm_left.png"),(W,H))
    Msim=rgb(DT); ov=R.copy()
    ov[edge(gmask([0,0,0]))]=(255,150,0); ov[edge(gmask(DT))]=(0,0,255); ov[edge(real)]=(0,255,255)
    band=lambda im,s:cv2.putText(cv2.putText(im.copy(),s,(22,46),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,0),6),
                                 s,(22,46),cv2.FONT_HERSHEY_SIMPLEX,1.1,(255,255,255),2)
    sc=lambda im:cv2.resize(im,(600,338))
    full.append(np.concatenate([sc(band(R,f"{p}  REAL")),np.full((338,5,3),255,np.uint8),
                                sc(band(Msim,"MUJOCO  PandaGripper, fitted")),np.full((338,5,3),255,np.uint8),
                                sc(band(ov,"OVERLAY y=real r=fitted b=published"))],1))
    zoom.append(cv2.resize(ov[470:720,380:900],(1040,500)))
j=lambda L:np.concatenate([np.concatenate([q,np.full((6,q.shape[1],3),255,np.uint8)],0) for q in L],0)
cv2.imwrite(f"{O}/SBS_open_full.png",j(full)); cv2.imwrite(f"{O}/SBS_open_zoom.png",j(zoom))
print("saved")

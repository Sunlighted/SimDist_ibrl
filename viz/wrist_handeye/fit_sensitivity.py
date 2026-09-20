"""Noise floor of the silhouette fit: how much does the solved correction move
when the real-mask segmentation threshold changes?  If the spread is the size of
the correction, the correction is not a measurement.
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
W,H=1280,720
def T(R,t):
    M=np.eye(4);M[:3,:3]=R;M[:3,3]=np.asarray(t).ravel();return M
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
    n,lab,st,ce=cv2.connectedComponentsWithStats(c,8)
    best,bi=-1,-1
    for k in range(1,n):
        x,y,w,h,a=st[k]
        if a<800: continue
        s=a*(2.0 if (y+h)>=H-3 else 0.4)
        if s>best: best,bi=s,k
    return (lab==bi).astype(np.uint8)

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
ctx=MjRenderContextOffscreen(sim,device_id=-1,max_width=W,max_height=H)
ctx.vopt.geomgroup[:]=0; ctx.vopt.geomgroup[1]=1
cid=sim.model.camera_name2id("zedm"); bid=sim.model.body_name2id("robot0_right_hand")
gg=[j for j in range(sim.model.ngeom)
    if (sim.model.geom_id2name(j) or "").startswith("gripper0") and "visual" in (sim.model.geom_id2name(j) or "")]
Rz=T(Rot.from_euler("z",45.03,degrees=True).as_matrix(),[0,0,0])
FW,FH=640,360
Mpp=np.float32([[1,0,(intr["left"]["cx"]-wn/2)*FW/wn],[0,1,(intr["left"]["cy"]-hn/2)*FH/hn]])
def render(p):
    ow,tx,ty,tz=p
    sim.data.qpos[:]=e.sim.data.qpos
    sim.data.qpos[gq[0]]=np.clip(ow,0,0.04); sim.data.qpos[gq[1]]=-np.clip(ow,0,0.04)
    sim.data.qvel[:]=0; sim.forward()
    bq=sim.data.body_xquat[bid]
    C=T(Rot.from_quat([bq[1],bq[2],bq[3],bq[0]]).as_matrix(),sim.data.body_xpos[bid])@Rz@T(np.eye(3),[tx,ty,tz])@T_ee_cam
    sim.model.cam_pos[cid]=C[:3,3]; sim.model.cam_quat[cid]=mjq(C[:3,:3]); sim.forward()
    seg=sim.render(camera_name="zedm",width=FW,height=FH,segmentation=True)[::-1][:,:,1]
    return cv2.warpAffine(np.isin(seg,gg).astype(np.uint8),Mpp,(FW,FH),flags=cv2.INTER_NEAREST)
sols=[]
for pct in [8,10,12,14,16,20]:
    rm=real_mask(pct); rs=cv2.resize(rm,(FW,FH),interpolation=cv2.INTER_NEAREST)
    cost=lambda p:(lambda m:1-((m&rs).sum()/max((m|rs).sum(),1)))(render(p))
    r=minimize(cost,[0.04,0,0,0],method="Nelder-Mead",options=dict(xatol=2e-4,fatol=1e-4,maxfev=1500,
        initial_simplex=[[0.04,0,0,0],[0.025,0,0,0],[0.04,.008,0,0],[0.04,0,.008,0],[0.04,0,0,.008]]))
    sols.append(r.x[1:]*1000)
    print(f"pct {pct:2d}: real mask {rm.sum():5d} px | IoU {1-r.fun:.3f} | open {r.x[0]*1000:4.1f} mm "
          f"| dt {np.round(r.x[1:]*1000,2)} mm  |dt| {np.linalg.norm(r.x[1:])*1000:.2f} mm")
s=np.array(sols)
print(f"\ncorrection across thresholds: mean {np.round(s.mean(0),2)} mm, "
      f"per-axis SD {np.round(s.std(0),2)} mm, |dt| {np.linalg.norm(s.mean(0)):.2f} mm")

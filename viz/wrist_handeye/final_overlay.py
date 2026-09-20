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
g=[]
for i in range(14):
    im=cv2.resize(cv2.imread(f"{R2}/pose_{i:03d}/zedm_left.png",cv2.IMREAD_GRAYSCALE),(W,H)).astype(np.float32)
    g.append(im/max(np.median(im),1.0))
g=np.stack(g); std,med=g.std(0),np.median(g,0)
c=((std<np.percentile(std,16))&(med<np.percentile(med,16))).astype(np.uint8)
c=cv2.morphologyEx(c,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
c=cv2.morphologyEx(c,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
n,lab,st,ce=cv2.connectedComponentsWithStats(c,8)
bi=max(range(1,n),key=lambda k: st[k][4]*(2.0 if (st[k][1]+st[k][3])>=H-3 else 0.4) if st[k][4]>=800 else -1)
real=(lab==bi).astype(np.uint8)
T_ee_cam=np.array(json.load(open(f"{SP}/ground_run2_solved.json"))["T_ee_zedm"])
intr=json.load(open(f"{R2}/intrinsics.json"))["zedm"]; wn,hn=intr["resolution"]
fovy=float(np.degrees(2*np.arctan(hn/(2*intr["left"]["fy"]))))
Mpp=np.float32([[1,0,(intr["left"]["cx"]-wn/2)*W/wn],[0,1,(intr["left"]["cy"]-hn/2)*H/hn]])
cc=load_controller_config(default_controller="OSC_POSE")
e=robosuite.make("PegInsertion_D0",robots="Panda",controller_configs=cc,has_renderer=False,
                 has_offscreen_renderer=False,use_camera_obs=False,horizon=60);e.reset()
aq=e.robots[0]._ref_joint_pos_indexes; gq=e.robots[0]._ref_gripper_joint_pos_indexes
sim=None
def build(joints):
    global sim,ctx,cid,bid,gg
    e.sim.data.qpos[aq]=np.array(joints); e.sim.data.qvel[:]=0; e.sim.forward()
    if sim is None:
        sim=MjSim.from_xml_string(e.sim.model.get_xml().replace("</worldbody>",
            f'<camera name="zedm" pos="0 0 1" fovy="{fovy}"/></worldbody>'))
        ctx=MjRenderContextOffscreen(sim,device_id=-1,max_width=W,max_height=H)
        ctx.vopt.geomgroup[:]=0; ctx.vopt.geomgroup[1]=1
        cid=sim.model.camera_name2id("zedm"); bid=sim.model.body_name2id("robot0_right_hand")
        gg=[j for j in range(sim.model.ngeom)
            if (sim.model.geom_id2name(j) or "").startswith("gripper0") and "visual" in (sim.model.geom_id2name(j) or "")]
Rz=T(Rot.from_euler("z",45.03,degrees=True).as_matrix(),[0,0,0])
def mask(dt,ow=0.040):
    sim.data.qpos[:]=e.sim.data.qpos
    sim.data.qpos[gq[0]]=ow; sim.data.qpos[gq[1]]=-ow; sim.data.qvel[:]=0; sim.forward()
    bq=sim.data.body_xquat[bid]
    C=T(Rot.from_quat([bq[1],bq[2],bq[3],bq[0]]).as_matrix(),sim.data.body_xpos[bid])@Rz@T(np.eye(3),dt)@T_ee_cam
    sim.model.cam_pos[cid]=C[:3,3]; sim.model.cam_quat[cid]=mjq(C[:3,:3]); sim.forward()
    seg=sim.render(camera_name="zedm",width=W,height=H,segmentation=True)[::-1][:,:,1]
    return cv2.warpAffine(np.isin(seg,gg).astype(np.uint8),Mpp,(W,H),flags=cv2.INTER_NEAREST)
DT=np.array([-2.23,1.65,-3.52])/1000.0
out=lambda m:(cv2.dilate(m,np.ones((3,3),np.uint8))-m)>0
rows=[]
for p in ["pose_000","pose_006","pose_012"]:
    build(json.load(open(f"{R2}/{p}/state.json"))["joints"])
    vis=cv2.resize(cv2.imread(f"{R2}/{p}/zedm_left.png"),(W,H))
    vis[out(mask([0,0,0]))]=(255,150,0)      # published  (BGR: orange-blue)
    vis[out(mask(DT))]=(0,0,255)             # fitted     (red)
    vis[out(real)]=(0,255,255)               # real       (yellow)
    rows.append(cv2.resize(vis[470:720,380:900],(1040,500)))
cv2.imwrite(f"{O}/FINAL_gripper_match.png",np.concatenate(
    [np.concatenate([r,np.full((6,1040,3),255,np.uint8)],0) for r in rows],0))
m0,m1=mask([0,0,0]),mask(DT)
j=lambda a,b:(a&b).sum()/(a|b).sum()
print(f"IoU published {j(m0,real):.3f} -> fitted {j(m1,real):.3f}")
for tag,m in [("published",m0),("fitted",m1)]:
    ys,xs=np.nonzero(m); ry,rx=np.nonzero(real)
    print(f"{tag:10s} centroid d=({xs.mean()-rx.mean():+5.1f},{ys.mean()-ry.mean():+5.1f}) px  "
          f"top-edge d={ys.min()-ry.min():+4d} px  area {m.sum()} vs real {real.sum()}")

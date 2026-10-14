import os,sys,json,glob,numpy as np
sys.path.insert(0,os.path.expanduser("~/simdist/SimDist_ibrl"))
os.environ.setdefault("MUJOCO_GL","egl")
import robosuite, mujoco
import env.threading_variants as tv
raw=robosuite.make("Threading_RealCellRobotiq",robots="Panda",has_renderer=False,has_offscreen_renderer=False,use_camera_obs=False)
np.random.seed(0); raw.reset()
M=raw.sim.model._model; Dd=raw.sim.data._data
rob=raw.robots[0]
# close the gripper in the air at reset pose
a=np.zeros(raw.action_dim)
for k in range(60): a[-1]=1.0; raw.step(a)
gidx=rob._ref_gripper_joint_pos_indexes
gq=Dd.qpos[gidx].copy()
print("closed gripper joint qpos",np.round(gq,3), "drivers",[round(float(Dd.qpos[M.jnt_qposadr[mujoco.mj_name2id(M,mujoco.mjtObj.mjOBJ_JOINT,n)]]),3) for n in tv.ROBOTIQ_DRIVER_JOINTS])
aidx=rob._ref_joint_pos_indexes
base=mujoco.mj_name2id(M,mujoco.mjtObj.mjOBJ_BODY,"robot0_base")
gbodies=[b for b in range(M.nbody) if mujoco.mj_id2name(M,mujoco.mjtObj.mjOBJ_BODY,b).startswith("gripper0")]
geoms=[g for g in range(M.ngeom) if M.geom_bodyid[g] in gbodies]
def geom_pts(g):
    t=M.geom_type[g]; s=M.geom_size[g]
    if t==mujoco.mjtGeom.mjGEOM_MESH:
        mid=M.geom_dataid[g]; a0=M.mesh_vertadr[mid]; n=M.mesh_vertnum[mid]; return M.mesh_vert[a0:a0+n]
    if t==mujoco.mjtGeom.mjGEOM_BOX:
        return np.array([[x,y,z] for x in(-1,1) for y in(-1,1) for z in(-1,1)])*s
    if t in (mujoco.mjtGeom.mjGEOM_CAPSULE,mujoco.mjtGeom.mjGEOM_CYLINDER):
        return np.array([[0,0,-s[1]-(s[0] if t==mujoco.mjtGeom.mjGEOM_CAPSULE else 0)],[0,0,s[1]]])
    if t==mujoco.mjtGeom.mjGEOM_SPHERE: return np.array([[0,0,-s[0]],[0,0,s[0]]])
    return np.zeros((1,3))
def lowest(q):
    Dd.qpos[aidx]=q; Dd.qpos[gidx]=gq; mujoco.mj_kinematics(M,Dd)
    Rb=Dd.xmat[base].reshape(3,3); pb=Dd.xpos[base]
    best=(9,None)
    for g in geoms:
        P=geom_pts(g); W=(Dd.geom_xmat[g].reshape(3,3)@P.T).T+Dd.geom_xpos[g]
        Bz=((W-pb)@Rb)
        i=np.argmin(Bz[:,2])
        if Bz[i,2]<best[0]: best=(Bz[i,2],(mujoco.mj_id2name(M,mujoco.mjtObj.mjOBJ_GEOM,g),Bz[i]))
    return best
tid=mujoco.mj_name2id(M,mujoco.mjtObj.mjOBJ_GEOM,"table_collision")
mujoco.mj_kinematics(M,Dd)
print("sim work surface (base z) %.4f"%(Dd.geom_xpos[tid][2]+M.geom_size[tid][2]-Dd.xpos[base][2]))
res=[]
for d in sorted(glob.glob(os.path.expanduser("~/simdist/calib/layout_1014b/shot_00[4-8]"))):
    q=json.load(open(d+"/state.json"))["q"]; z,(gn,p)=lowest(np.array(q))
    res.append(p); print(os.path.basename(d),"lowest gripper point base (%.4f, %.4f, %.4f) geom %s"%(*p,gn))
P=np.array(res)
A=np.c_[P[:,0],P[:,1],np.ones(len(P))]; c,*_=np.linalg.lstsq(A,P[:,2],rcond=None)
r=P[:,2]-A@c
print("z mean %.4f  sd %.4f  range %.4f..%.4f"%(P[:,2].mean(),P[:,2].std(),P[:,2].min(),P[:,2].max()))
print("plane fit: dz/dx %.4f (%.2f deg) dz/dy %.4f (%.2f deg), resid max %.1f mm"%(c[0],np.degrees(np.arctan(c[0])),c[1],np.degrees(np.arctan(c[1])),1000*np.abs(r).max()))
print("plane z at needle (0.55,-0.12): %.4f ; at tripod (0.527,0.176): %.4f"%(c@[0.55,-0.12,1],c@[0.527,0.176,1]))
# flange (robot0_right_hand) -> closed fingertip, along the hand z axis
hb=mujoco.mj_name2id(M,mujoco.mjtObj.mjOBJ_BODY,"robot0_right_hand")
ad=mujoco.mj_name2id(M,mujoco.mjtObj.mjOBJ_BODY,"gripper0_robotiq_85_adapter_link")
l7=mujoco.mj_name2id(M,mujoco.mjtObj.mjOBJ_BODY,"robot0_link7")
Dd.qpos[gidx]=gq; mujoco.mj_kinematics(M,Dd)
Rh=Dd.xmat[hb].reshape(3,3); ph=Dd.xpos[hb]
far=-1
for g in geoms:
    P=geom_pts(g); W=(Dd.geom_xmat[g].reshape(3,3)@P.T).T+Dd.geom_xpos[g]; L=(W-ph)@Rh
    far=max(far,L[:,2].max())
print("right_hand vs link7 along z: %.4f m"%(((Dd.xpos[hb]-Dd.xpos[l7])@Dd.xmat[l7].reshape(3,3))[2]))
print("adapter body origin in hand frame: %s"%np.round((Dd.xpos[ad]-ph)@Rh,4))
print("hand(flange) -> farthest closed fingertip along tool z: %.4f m"%far)

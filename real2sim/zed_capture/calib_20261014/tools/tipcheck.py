exec(open("touch.py").read().split("tid=mujoco")[0])
import cv2
from scipy.spatial.transform import Rotation as Rr
import yaml
cell=yaml.safe_load(open(os.path.expanduser("~/simdist/SimDist_ibrl/real2sim/zed_capture/calib_20261013/threading_real_cell_20261013.yaml")))
cam=next(c for c in cell["cameras"] if c["name"]=="agentview")
q_new=[0.15341,-0.29297,0.85476,-0.40002]
fac=json.load(open(os.path.expanduser("~/simdist/SimDist_ibrl/release/cfgs/sim2real/zed_factory_intrinsics.json")))["scene"]; mm=fac["modes"]["hd720"]
K=np.array([[mm["fx"],0,mm["cx"]],[0,mm["fy"],mm["cy"]],[0,0,1.]]); Dk=np.array(fac["distortion_k1k2p1p2k3"])
def proj(P,q):
    w,x,y,z=q; Rbs=Rr.from_quat([x,y,z,w]).as_matrix(); p=(Rbs.T@(P-np.array(cam["pos"])).T).T
    uv,_=cv2.projectPoints(p,np.zeros(3),np.zeros(3),K,Dk); return uv.reshape(-1,2)
tiles=[]
for d in sorted(glob.glob(os.path.expanduser("~/simdist/calib/layout_1014b/shot_00[4-8]"))):
    q=np.array(json.load(open(d+"/state.json"))["q"])
    Dd.qpos[aidx]=q; Dd.qpos[gidx]=gq; mujoco.mj_kinematics(M,Dd)
    Rb=Dd.xmat[base].reshape(3,3); pb=Dd.xpos[base]; hb=mujoco.mj_name2id(M,mujoco.mjtObj.mjOBJ_BODY,"robot0_right_hand")
    tz=Dd.xmat[hb].reshape(3,3)[:,2]
    pts=[]
    for g in geoms:
        n=mujoco.mj_id2name(M,mujoco.mjtObj.mjOBJ_GEOM,g)
        if "fingertip" not in n: continue
        P=geom_pts(g); W=(Dd.geom_xmat[g].reshape(3,3)@P.T).T+Dd.geom_xpos[g]+0.0065*tz   # real gripper 6.5 mm longer
        pts.append((W-pb)@Rb)
    P=np.concatenate(pts)
    img=cv2.imread(d+"/scene_left.png")
    for q_,col in [(cam["quat"],(0,0,255)),(q_new,(0,255,0))]:
        uv=proj(P,q_)
        for u,v in uv.astype(int): cv2.circle(img,(u,v),1,col,-1)
    c=proj(P.mean(0,keepdims=True),q_new)[0].astype(int)
    crop=img[max(c[1]-60,0):c[1]+60,max(c[0]-80,0):c[0]+80]
    tiles.append(cv2.resize(crop,(480,360),interpolation=cv2.INTER_NEAREST))
    print(os.path.basename(d),"tip centre base",np.round(P.mean(0),4),"min z %.4f"%P[:,2].min())
cv2.imwrite("tipcheck.jpg",np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:]+[np.zeros_like(tiles[0])])]))

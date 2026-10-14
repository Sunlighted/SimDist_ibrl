import json,sys,os,numpy as np,cv2,yaml
sys.path.insert(0,os.path.expanduser("~/simdist/calib"))
import solve_calib as sc
from scipy.spatial.transform import Rotation as R
REPO=os.path.expanduser("~/simdist/SimDist_ibrl")
cell=yaml.safe_load(open(REPO+"/real2sim/zed_capture/calib_20261013/threading_real_cell_20261013.yaml"))
cams={c["name"]:c for c in cell["cameras"]}
def camT(c):
    w,x,y,z=c["quat"]; return sc.T(R.from_quat([x,y,z,w]).as_matrix(),c["pos"])
T_bs=camT(cams["agentview"]); X=camT(cams["robot0_eye_in_hand"])
fac=json.load(open(REPO+"/release/cfgs/sim2real/zed_factory_intrinsics.json"))["scene"]
m=fac["modes"]["hd720"]; Ks=np.array([[m["fx"],0,m["cx"]],[0,m["fy"],m["cy"]],[0,0,1.]]); Ds=np.array(fac["distortion_k1k2p1p2k3"])
wi=json.load(open(REPO+"/real2sim/zed_capture/calib_20261010/data/wrist_intrinsics.json")); Kw,Dw=np.array(wi["K"]),np.array(wi["D"])
ZT=0.0285
def crop_f(fovy,S=224): return (S/2)/np.tan(np.radians(fovy)/2)
def proj_crop(Tbc,pts,fovy,S=224):
    p=(sc.inv(Tbc)@np.c_[pts,np.ones(len(pts))].T)[:3].T
    f=crop_f(fovy,S); c=(S-1)/2
    return np.c_[f*p[:,0]/p[:,2]+c, f*p[:,1]/p[:,2]+c], p[:,2]
def needle_pts(x,y,zrot,z0=ZT):
    # sim NeedleObject body frame: shaft along +y from -0.08 (tip) to 0.04, handle 0.04..0.08 (+-0.02 in x,z)
    loc=[(0,-0.08,-0.005),(0,-0.08,0.005)]
    for hx in(-0.02,0.02):
        for hy in(0.04,0.08):
            for hz in(-0.02,0.02): loc.append((hx,hy,hz))
    loc=np.array(loc); Rz=R.from_euler("z",zrot,degrees=True).as_matrix()
    return (Rz@loc.T).T+[x,y,z0+0.02]

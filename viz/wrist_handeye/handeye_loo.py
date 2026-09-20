"""Leave-one-out: refit T_ee_zedm + T_base_board on 13 poses, predict the 14th."""
import json, numpy as np, cv2
R2, SP = "real2sim/zed_capture/scan_run2", "real2sim/scan_pipeline"
def T(R,t):
    M=np.eye(4); M[:3,:3]=R; M[:3,3]=np.asarray(t).ravel(); return M
def inv(M):
    R,t=M[:3,:3],M[:3,3]; return T(R.T,-R.T@t)
def q2R(q):
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def ang(R): return np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
def avg_T(Ts):
    t=np.mean([X[:3,3] for X in Ts],0)
    U,_,Vt=np.linalg.svd(sum(X[:3,:3] for X in Ts))
    R=U@Vt
    if np.linalg.det(R)<0: U[:,-1]*=-1; R=U@Vt
    return T(R,t)

board={eval(k):np.array(v) for k,v in json.load(open(f"{SP}/board_model.json")).items()}
intr=json.load(open(f"{R2}/intrinsics.json"))["zedm"]["left"]
K=np.array([[intr["fx"],0,intr["cx"]],[0,intr["fy"],intr["cy"]],[0,0,1]]); D=np.zeros(5)
det=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50))

P=[]
for i in range(14):
    p=f"pose_{i:03d}"; st=json.load(open(f"{R2}/{p}/state.json"))
    cs,ids,_=det.detectMarkers(cv2.imread(f"{R2}/{p}/zedm_left.png",cv2.IMREAD_GRAYSCALE))
    obj,imgp=[],[]
    for mid,c in zip(ids.ravel(),cs):
        for j in range(4):
            if (int(mid),j) in board: obj.append(board[(int(mid),j)]); imgp.append(c.reshape(4,2)[j])
    obj,imgp=np.array(obj,np.float64),np.array(imgp,np.float64)
    ok,rv,tv,inl=cv2.solvePnPRansac(obj,imgp,K,D,flags=cv2.SOLVEPNP_ITERATIVE,
                                    reprojectionError=3.0,iterationsCount=300)
    rv,tv=cv2.solvePnPRefineLM(obj[inl.ravel()],imgp[inl.ravel()],K,D,rv,tv)
    P.append(dict(name=p, T_cam_board=T(cv2.Rodrigues(rv)[0],tv),
                  T_base_ee=T(q2R(st["ee_quat"]),st["ee_pos"])))

for meth,mid in [("TSAI",cv2.CALIB_HAND_EYE_TSAI),("PARK",cv2.CALIB_HAND_EYE_PARK),
                 ("DANIILIDIS",cv2.CALIB_HAND_EYE_DANIILIDIS)]:
    errs=[]
    for k in range(14):
        tr=[P[i] for i in range(14) if i!=k]
        Rx,tx=cv2.calibrateHandEye([q["T_base_ee"][:3,:3] for q in tr],[q["T_base_ee"][:3,3] for q in tr],
                                   [q["T_cam_board"][:3,:3] for q in tr],[q["T_cam_board"][:3,3] for q in tr],method=mid)
        T_ee_cam=T(Rx,tx)
        T_base_board=avg_T([q["T_base_ee"]@T_ee_cam@q["T_cam_board"] for q in tr])
        h=P[k]
        pred=inv(T_base_board)@h["T_base_ee"]@T_ee_cam            # board<-cam, predicted
        d=inv(inv(h["T_cam_board"]))@pred
        errs.append((np.linalg.norm(d[:3,3])*1000, ang(d[:3,:3])))
    a=np.array(errs)
    print(f"{meth:11s} LOO held-out:  trans med {np.median(a[:,0]):5.2f} mean {a[:,0].mean():5.2f} max {a[:,0].max():5.2f} mm | "
          f"rot med {np.median(a[:,1]):4.2f} mean {a[:,1].mean():4.2f} max {a[:,1].max():4.2f} deg")

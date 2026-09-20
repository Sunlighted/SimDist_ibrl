"""Project the board into the wrist image from joints+hand-eye ONLY.
green = ArUco corners actually detected in the image
red   = same corners predicted through T_base_board . T_base_ee(joints) . T_ee_zedm
"""
import json, numpy as np, cv2
R2, SP = "real2sim/zed_capture/scan_run2", "real2sim/scan_pipeline"
O="/scratch/yifanr/temp/claude-3144347/-project-6101830-yifanr-SimDist-ibrl/a1b0d322-aaab-41c7-9a4a-93e3923ef768/scratchpad"
def T(R,t):
    M=np.eye(4); M[:3,:3]=R; M[:3,3]=np.asarray(t).ravel(); return M
def inv(M):
    R,t=M[:3,:3],M[:3,3]; return T(R.T,-R.T@t)
def q2R(q):
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
board={eval(k):np.array(v) for k,v in json.load(open(f"{SP}/board_model.json")).items()}
sol=json.load(open(f"{SP}/ground_run2_solved.json"))
intr=json.load(open(f"{R2}/intrinsics.json"))["zedm"]["left"]
K=np.array([[intr["fx"],0,intr["cx"]],[0,intr["fy"],intr["cy"]],[0,0,1]]); D=np.zeros(5)
T_ee_cam=np.array(sol["T_ee_zedm"]); T_board_base=inv(np.array(sol["T_base_board"]))
det=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50))

panels=[]; allerr=[]
for i in [0,6,12]:
    p=f"pose_{i:03d}"; st=json.load(open(f"{R2}/{p}/state.json"))
    img=cv2.imread(f"{R2}/{p}/zedm_left.png")
    cs,ids,_=det.detectMarkers(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    obj,seen=[],[]
    for mid,c in zip(ids.ravel(),cs):
        for j in range(4):
            if (int(mid),j) in board: obj.append(board[(int(mid),j)]); seen.append(c.reshape(4,2)[j])
    obj,seen=np.array(obj,np.float64),np.array(seen,np.float64)
    T_cam_board=inv(T_board_base@T(q2R(st["ee_quat"]),st["ee_pos"])@T_ee_cam)
    rv=cv2.Rodrigues(T_cam_board[:3,:3])[0]
    pred=cv2.projectPoints(obj,rv,T_cam_board[:3,3],K,D)[0].reshape(-1,2)
    err=np.linalg.norm(pred-seen,axis=1); allerr.append(err)
    for a,b in zip(seen,pred):
        cv2.circle(img,tuple(a.astype(int)),9,(0,255,0),2)
        cv2.drawMarker(img,tuple(b.astype(int)),(0,0,255),cv2.MARKER_CROSS,16,2)
        cv2.line(img,tuple(a.astype(int)),tuple(b.astype(int)),(255,0,255),1)
    cv2.putText(img,f"{p}  FK-chain reproj {err.mean():.1f} px (mean)  max {err.max():.1f}",
                (30,60),cv2.FONT_HERSHEY_SIMPLEX,1.6,(0,0,0),7)
    cv2.putText(img,f"{p}  FK-chain reproj {err.mean():.1f} px (mean)  max {err.max():.1f}",
                (30,60),cv2.FONT_HERSHEY_SIMPLEX,1.6,(255,255,255),3)
    print(f"{p}: FK-chain reproj mean {err.mean():.2f} px, max {err.max():.2f} px  (n={len(err)})")
    panels.append(cv2.resize(img,(900,506)))
cv2.imwrite(f"{O}/handeye_reproj.png",np.concatenate(
    [np.concatenate([q,np.full((6,900,3),255,np.uint8)],0) for q in panels],0))
e=np.concatenate(allerr); print(f"\nall 3 poses: mean {e.mean():.2f} px, median {np.median(e):.2f}, max {e.max():.2f}")

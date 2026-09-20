"""Does T_ee_zedm actually predict where the wrist camera is, from FK alone?

Per-pose PnP gives T_board_cam(p) straight from the image (no robot involved).
The hand-eye chain predicts the same thing from joints only:
    T_board_cam_pred(p) = T_board_base . T_base_ee(p) . T_ee_zedm
The disagreement IS the hand-eye error.  Anything solved per-pose from the
board (as render_camera_pov.py uses) cannot test this.
"""
import json, numpy as np, cv2
R2 = "real2sim/zed_capture/scan_run2"
SP = "real2sim/scan_pipeline"
DIC = cv2.aruco.DICT_5X5_50

def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.asarray(t).ravel(); return M
def inv(M):
    R, t = M[:3, :3], M[:3, 3]; return T(R.T, -R.T @ t)
def q2R(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
def ang(R): return np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1)))

board = {eval(k): np.array(v) for k, v in json.load(open(f"{SP}/board_model.json")).items()}
sol   = json.load(open(f"{SP}/ground_run2_solved.json"))
intr  = json.load(open(f"{R2}/intrinsics.json"))["zedm"]["left"]
K  = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]])
D  = np.zeros(5)
T_ee_cam    = np.array(sol["T_ee_zedm"])
T_base_board = np.array(sol["T_base_board"])
T_board_base = inv(T_base_board)

det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DIC))
rows = []
for i in range(14):
    p = f"pose_{i:03d}"
    st = json.load(open(f"{R2}/{p}/state.json"))
    img = cv2.imread(f"{R2}/{p}/zedm_left.png", cv2.IMREAD_GRAYSCALE)
    cs, ids, _ = det.detectMarkers(img)
    if ids is None: print(f"{p}: no markers"); continue
    obj, imgp = [], []
    for mid, c in zip(ids.ravel(), cs):
        for j in range(4):
            if (int(mid), j) in board:
                obj.append(board[(int(mid), j)]); imgp.append(c.reshape(4, 2)[j])
    obj, imgp = np.array(obj, np.float64), np.array(imgp, np.float64)
    ok, rv, tv, inl = cv2.solvePnPRansac(obj, imgp, K, D, flags=cv2.SOLVEPNP_ITERATIVE,
                                         reprojectionError=3.0, iterationsCount=300)
    if not ok: print(f"{p}: pnp failed"); continue
    rv, tv = cv2.solvePnPRefineLM(obj[inl.ravel()], imgp[inl.ravel()], K, D, rv, tv)
    T_cam_board = T(cv2.Rodrigues(rv)[0], tv)           # board -> cam
    T_board_cam_pnp = inv(T_cam_board)

    T_base_ee = T(q2R(st["ee_quat"]), st["ee_pos"])
    T_board_cam_fk = T_board_base @ T_base_ee @ T_ee_cam

    d = inv(T_board_cam_pnp) @ T_board_cam_fk
    mm = np.linalg.norm(d[:3, 3]) * 1000
    dg = ang(d[:3, :3])
    rp = cv2.projectPoints(obj[inl.ravel()], rv, tv, K, D)[0].reshape(-1, 2)
    rows.append((p, len(obj), len(inl), np.linalg.norm(rp - imgp[inl.ravel()], axis=1).mean(), mm, dg))
    print(f"{p}: {len(inl):3d}/{len(obj):3d} corners, reproj {rows[-1][3]:.2f} px | "
          f"FK-vs-PnP  {mm:7.2f} mm  {dg:5.2f} deg")

a = np.array([[r[4], r[5]] for r in rows])
print(f"\nn={len(rows)}  translation  med {np.median(a[:,0]):.2f}  mean {a[:,0].mean():.2f}  max {a[:,0].max():.2f} mm")
print(f"          rotation     med {np.median(a[:,1]):.2f}  mean {a[:,1].mean():.2f}  max {a[:,1].max():.2f} deg")

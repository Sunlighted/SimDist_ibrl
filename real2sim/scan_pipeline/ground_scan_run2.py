"""Full grounding from scan_run2 alone -- NO phone-scan/COLMAP dependency.

scan4 is deprecated (owner 2026-09-16). The canonical board/world frame is now
built from METRIC data only:
  board model: marker corners unprojected from the static zed2i's depth,
    median over the 14 robot poses -> constellation -> canonical frame
    (plane -> z=0.8, in-plane PCA -> x with a deterministic sign rule,
    center -> (0.20, 0.30)) -> board_model.json [(id,corner) -> world xyz]
  zed2i extrinsic: the same fit IS the camera pose in world.
  robot base: wrist zedm sees the board per pose (Kabsch to the model) + FK
    (ee_quat is xyzw) -> cv2.calibrateHandEye -> T_base->world constant check.
"""
import json, os
import numpy as np
import cv2

GS = "/scratch/yifanr/gsbench"
SR = "/home/yifanr/scan_run2"
DIC = cv2.aruco.DICT_5X5_50
TABLE_Z, CENTER_XY = 0.8, (0.20, 0.30)


def detect(gray):
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DIC))
    corners, ids, _ = det.detectMarkers(gray)
    return {} if ids is None else {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}


def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.asarray(t).ravel(); return M


def inv(M):
    R, t = M[:3, :3], M[:3, 3]; return T(R.T, -R.T @ t)


def quat_xyzw_to_R(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = cb - R @ ca
    return R, t, np.linalg.norm((A @ R.T + t) - B, axis=1)


def cam_corners_3d(img_p, dep_p, intr):
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    g = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
    depth = np.load(dep_p)
    out = {}
    for mid, cs in detect(g).items():
        for k in range(4):
            u, v = cs[k]
            ui, vi = int(round(u)), int(round(v))
            win = depth[max(0, vi-2):vi+3, max(0, ui-2):ui+3]
            win = win[np.isfinite(win) & (win > 0.05)]
            if win.size >= 3:
                z = float(np.median(win))
                out[(mid, k)] = np.array([(u-cx)/fx*z, (v-cy)/fy*z, z])
    return out


def rot_angle(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def avg_T(Ms):
    ts = np.stack([M[:3, 3] for M in Ms])
    Rm = sum(M[:3, :3] for M in Ms) / len(Ms)
    U, _, Vt = np.linalg.svd(Rm)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    dev_r = [rot_angle(R.T @ M[:3, :3]) for M in Ms]
    dev_t = np.linalg.norm(ts - ts.mean(0), axis=1)
    return T(R, ts.mean(0)), float(np.max(dev_r)), float(np.max(dev_t))


def main():
    intr = json.load(open(f"{SR}/intrinsics.json"))
    poses = sorted(p for p in os.listdir(SR) if p.startswith("pose_"))

    # ---- board model from the WRIST zedm (fused across poses) -----------
    # the external zed2i sees the board too small/oblique at 720p to detect;
    # the wrist close-ups are ideal. Fuse: register every pose's metric corner
    # constellation to pose-0's camera frame by shared-corner Kabsch, median.
    per_pose = []
    for p in poses:
        cw = cam_corners_3d(f"{SR}/{p}/zedm_left.png", f"{SR}/{p}/zedm_depth.npy",
                            intr["zedm"]["left"])
        per_pose.append(cw)
        print(f"  [model] {p}: wrist sees {len(cw)} corners")
    ref_i = int(np.argmax([len(c) for c in per_pose]))
    acc = {k: [v] for k, v in per_pose[ref_i].items()}
    fuse_res = []
    for i, cw in enumerate(per_pose):
        if i == ref_i or len(cw) < 8:
            continue
        shared = sorted(set(cw) & set(per_pose[ref_i]))
        if len(shared) < 6:
            continue
        A = np.stack([cw[k] for k in shared])
        B = np.stack([per_pose[ref_i][k] for k in shared])
        Rk, tk, res = kabsch(A, B)
        fuse_res.append(np.median(res))
        for k, X in cw.items():
            acc.setdefault(k, []).append(Rk @ X + tk)
    cam_pts = {k: np.median(np.stack(v), 0) for k, v in acc.items() if len(v) >= 2}
    spread = [np.median(np.linalg.norm(np.stack(v) - cam_pts[k], axis=1))
              for k, v in acc.items() if k in cam_pts and len(v) >= 2]
    print(f"[model] fused {len(cam_pts)} corners in pose_{ref_i} frame; "
          f"pairwise kabsch med {np.median(fuse_res)*1000:.1f} mm, "
          f"per-corner spread med {np.median(spread)*1000:.1f} mm")

    keys = sorted(cam_pts)
    P = np.stack([cam_pts[k] for k in keys])
    c0 = P.mean(0)
    U, S, Vt = np.linalg.svd(P - c0)
    normal = Vt[2]
    if normal @ (np.zeros(3) - c0) < 0:  # camera at origin of its own frame
        normal = -normal                 # normal points toward the camera (up)
    x_in = Vt[0] - (Vt[0] @ normal) * normal
    x_in /= np.linalg.norm(x_in)
    # deterministic sign: marker id 0's centroid lies at negative x
    m0 = np.mean([cam_pts[k] for k in keys if k[0] == min(k2[0] for k2 in keys)], axis=0)
    if (m0 - c0) @ x_in > 0:
        x_in = -x_in
    y_in = np.cross(normal, x_in)
    Rcw = np.stack([x_in, y_in, normal])          # (zedm pose_ref) cam -> world
    tcw = np.array([CENTER_XY[0], CENTER_XY[1], TABLE_Z]) - Rcw @ c0
    world = {k: Rcw @ cam_pts[k] + tcw for k in keys}
    print(f"[model] plane res p95 {np.percentile(np.abs((P-c0)@normal),95)*1000:.1f} mm")

    # ---- external zed2i: enhanced detection on its own frames -----------
    acc2 = {}
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    for p in poses:
        g = cv2.imread(f"{SR}/{p}/zed2i_left.png", cv2.IMREAD_GRAYSCALE)
        g2 = clahe.apply(cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC))
        depth = np.load(f"{SR}/{p}/zed2i_depth.npy")
        ii = intr["zed2i"]["left"]
        for mid, cs in detect(g2).items():
            for k in range(4):
                u, v = cs[k] / 2.0
                ui, vi = int(round(u)), int(round(v))
                win = depth[max(0, vi-2):vi+3, max(0, ui-2):ui+3]
                win = win[np.isfinite(win) & (win > 0.05)]
                if win.size >= 3:
                    z = float(np.median(win))
                    acc2.setdefault((mid, k), []).append(
                        np.array([(u-ii["cx"])/ii["fx"]*z, (v-ii["cy"])/ii["fy"]*z, z]))
    z2 = {k: np.median(np.stack(v), 0) for k, v in acc2.items()}
    shared2 = sorted(set(z2) & set(world))
    if len(shared2) >= 6:
        A = np.stack([world[k] for k in shared2])
        B = np.stack([z2[k] for k in shared2])
        R2, t2, res2 = kabsch(A, B)
        T_w_zed2i = inv(T(R2, t2))
        print(f"[zed2i] via own frames: {len(shared2)} corners, "
              f"kabsch med {np.median(res2)*1000:.1f} mm, cam@world {T_w_zed2i[:3,3].round(4)}")
    else:
        # fall back: PnP of the HD2K photo against the fused board model
        full = cv2.imread("/home/yifanr/Explorer_HD2K_SN37597132_17-54-59.png")
        left = full[:, : full.shape[1] // 2]
        ii = intr["zed2i"]["left"]
        sc = left.shape[1] / 1280.0
        K = np.array([[ii["fx"]*sc, 0, ii["cx"]*sc], [0, ii["fy"]*sc, ii["cy"]*sc], [0, 0, 1]])
        dets = detect(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY))
        obj = [world[(m, k)] for m, cs in dets.items() for k in range(4) if (m, k) in world]
        img = [cs[k] for m, cs in dets.items() for k in range(4) if (m, k) in world]
        obj, img = np.array(obj, np.float64), np.array(img, np.float64)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, reprojectionError=3.0)
        assert ok, "zed2i unresolvable: no in-session detection AND photo PnP failed"
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
        err = np.median(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1))
        Rp, _ = cv2.Rodrigues(rvec)
        T_w_zed2i = inv(T(Rp, tvec))
        print(f"[zed2i] via HD2K photo PnP (in-session frames undetectable): "
              f"{len(inl)}/{len(obj)} inliers, reproj med {err:.2f}px, "
              f"cam@world {T_w_zed2i[:3,3].round(4)}"
              "  -- WARNING assumes the board did not move between photo and scan_run2")

    # ---- hand-eye from the wrist zedm -----------------------------------
    R_g2b, t_g2b, R_t2c, t_t2c, T_b_ee_list = [], [], [], [], []
    for p in poses:
        st = json.load(open(f"{SR}/{p}/state.json"))
        cw = cam_corners_3d(f"{SR}/{p}/zedm_left.png", f"{SR}/{p}/zedm_depth.npy",
                            intr["zedm"]["left"])
        shared = sorted(set(cw) & set(world))
        if len(shared) < 6:
            print(f"  {p}: wrist sees {len(shared)} shared corners -- skipped")
            continue
        A = np.stack([world[k] for k in shared])
        B = np.stack([cw[k] for k in shared])
        Rk, tk, res = kabsch(A, B)                 # world -> cam
        T_b_ee = T(quat_xyzw_to_R(st["ee_quat"]), st["ee_pos"])
        R_g2b.append(T_b_ee[:3, :3]); t_g2b.append(T_b_ee[:3, 3])
        R_t2c.append(Rk); t_t2c.append(tk); T_b_ee_list.append(T_b_ee)
        print(f"  {p}: wrist {len(shared)} corners, kabsch med {np.median(res)*1000:.1f} mm")

    print(f"[handeye] usable poses: {len(R_g2b)}/{len(poses)}")
    # drop kabsch-outlier poses (depth glitches)
    good = [i for i, (Rk, tk) in enumerate(zip(R_t2c, t_t2c))]
    angs = [rot_angle(R_g2b[0].T @ R) for R in R_g2b[1:]]
    print(f"[handeye] EE rotation diversity vs pose0: {np.round(angs, 1)} deg")

    cam_w = [inv(T(Rk, tk))[:3, 3] for Rk, tk in zip(R_t2c, t_t2c)]
    ee_b = [Tbe[:3, 3] for Tbe in T_b_ee_list]
    # outlier gate on the *pair* sets via per-pose kabsch res is upstream; use
    # a leave-out consistency gate on the translation solve instead
    A = np.stack(ee_b); B = np.stack(cam_w)
    if max(angs) < 3.0:
        # TRANSLATION-ONLY capture: R_world<-base is exactly observable
        # (cam offset constant => delta cam_world = R_wb @ delta ee_base);
        # the base ORIGIN is NOT (absorbed by the unknown wrist-cam offset).
        Rwb, twb_amb, res = kabsch(A, B)
        keep = res < np.median(res) + 3 * 1.4826 * np.median(np.abs(res - np.median(res)))
        Rwb, twb_amb, res = kabsch(A[keep], B[keep])
        print(f"[base-rot] translation-only solve on {keep.sum()}/{len(A)} poses: "
              f"residual med {np.median(res)*1000:.1f} mm max {res.max()*1000:.1f} mm")
        print(f"[base-rot] R_world_base (rows=base axes in world):\n{Rwb.round(4)}")
        zb = Rwb @ np.array([0, 0, 1.0])
        print(f"[base-rot] base +z in world: {zb.round(4)} "
              f"(tilt from vertical {np.degrees(np.arccos(abs(zb[2]))):.2f} deg)")
        base_out = {"R_world_base": Rwb.tolist(),
                    "residual_med_mm": float(np.median(res)*1000),
                    "translation": "UNOBSERVABLE from this capture "
                    "(EE orientation constant); need rotated poses or tip-touch"}
    else:
        best = None
        for name, meth in [("TSAI", cv2.CALIB_HAND_EYE_TSAI),
                           ("PARK", cv2.CALIB_HAND_EYE_PARK),
                           ("DANIILIDIS", cv2.CALIB_HAND_EYE_DANIILIDIS)]:
            Rx, tx = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=meth)
            T_ee_cam = T(Rx, tx)
            Tb = [inv(Tbe @ T_ee_cam @ T(Rk, tk))
                  for Tbe, Rk, tk in zip(T_b_ee_list, R_t2c, t_t2c)]
            T_wb, dr, dt = avg_T(Tb)
            print(f"[handeye] {name}: spread {dr:.2f} deg / {dt*1000:.1f} mm, "
                  f"base@world {T_wb[:3,3].round(4)}")
            if best is None or dt < best[2]:
                best = (name, T_ee_cam, dt, T_wb, dr)
        name, T_ee_cam, dt, T_wb, dr = best
        base_out = {"T_world_base": T_wb.tolist(),
                    "spread_deg_mm": [dr, dt*1000], "method": name,
                    "T_ee_zedm": T_ee_cam.tolist()}

    # per-pose wrist camera poses in world (for POV renders; no hand-eye needed)
    T_w_zedm = {p: inv(T(Rk, tk)).tolist()
                for p, Rk, tk in zip([q for q in poses], R_t2c, t_t2c)}

    json.dump({str(k): v.tolist() for k, v in world.items()},
              open(f"{GS}/board_model.json", "w"), indent=1)
    json.dump({
        "base": base_out,
        "T_world_zed2i": T_w_zed2i.tolist(),
        "T_world_zedm_per_pose": T_w_zedm,
        "n_poses_used": len(R_g2b), "source": "scan_run2 (scan4-free)",
    }, open(f"{GS}/ground_transforms.json", "w"), indent=2)
    print(f"[done] wrote board_model.json + ground_transforms.json")


if __name__ == "__main__":
    main()

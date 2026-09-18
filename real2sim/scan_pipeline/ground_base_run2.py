"""Robot base <-> board, and external camera <-> board, from scan_run2 (14 poses).

Why redo this: real2sim.md 2d records the base solve as FAILED (13 deg / 337 mm
spread, "wrist ZED only saw the board in 3/8 poses") and ground_scan_run2.py
fell back to a single-photo PnP for the external camera. Both premises are false
for scan_run2: measured here, the ChArUco is detectable with metric depth in
14/14 poses -- 40-68 corners per pose in the WRIST camera and 7-42 in the
external one. So the data for a well-conditioned solve exists and the earlier
failure was in the pipeline, not the capture.

Three outputs, each with its own honest spread metric:
  1. T_board_zed2i  -- external camera, from per-pose depth-Kabsch (spread over
     14 independent solves is the accuracy estimate; no planar-PnP ambiguity).
  2. T_ee_zedm      -- wrist camera mount (hand-eye), needed later for the pixel
     stage anyway.
  3. T_base_board   -- THE blocker. Reported per pose so the spread says whether
     it can be trusted, instead of asserting one number.

`ee_quat` in state.json is xyzw (documented trap in real2sim.md 2d).
"""
import glob
import json
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

GS = "/scratch/yifanr/gsbench"
SR = "/home/yifanr/scan_run2"
DIC = cv2.aruco.DICT_5X5_50
_DET = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DIC))


def T(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = np.asarray(t).ravel()
    return M


def kabsch(A, B):
    """R,t with B ~= R@A + t (rigid, no scale), plus residuals in metres."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = cb - R @ ca
    res = np.linalg.norm((R @ A.T).T + t - B, axis=1)
    return R, t, res


def K_from(ii, width, res):
    """Intrinsics for the ACTUAL image size.

    The trap: scan_run1's manifest stores 1280x720 intrinsics, scan_run2's
    stores NATIVE 2208x1242 ones. Hardcoding `width/1280` (which
    ground_scan_run2.py does) therefore inflates scan_run2's focal length by
    1.725x -- fx 1057.6 -> 1824, i.e. a 92.5 deg camera described as 62 deg.
    PnP silently absorbs that by moving the camera, which is why a 1.5 px
    per-pose fit could still be metres wrong. Always scale from the manifest's
    OWN resolution field.
    """
    s = width / float(res[0])
    return np.array([[ii["fx"] * s, 0, ii["cx"] * s],
                     [0, ii["fy"] * s, ii["cy"] * s], [0, 0, 1]])


def cam_pose_pnp(img_p, ii, world, res):
    """T_cam_board from 2D corners only (PnP), no depth.

    Depth is the weak link: the per-pose depth-Kabsch residual measures 18 mm
    (external) to 33 mm (wrist) against a board model whose own self-consistency
    is sub-millimetre (planarity 0.44 mm rms, marker side 30.88 +- 0.52 mm), so
    the error is ZED stereo depth on a high-contrast planar target, not the
    reference. 2D corners + a known-good 3D model is the well-conditioned route.

    Planar-target PnP is two-fold ambiguous, so enumerate solutions and keep the
    one with the lowest reprojection error after LM refinement.
    """
    g = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
    K = K_from(ii, g.shape[1], res)
    cs, ids, _ = _DET.detectMarkers(g)
    if ids is None:
        return None, 0, np.inf
    keys = [(int(i), k) for i in ids.ravel() for k in range(4) if (int(i), k) in world]
    if len(keys) < 8:
        return None, len(keys), np.inf
    det = {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), cs)}
    obj = np.array([world[k] for k in keys], np.float64)
    img = np.array([det[m][k] for m, k in keys], np.float64)
    best = None
    for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_ITERATIVE):
        try:
            n, rvs, tvs, _ = cv2.solvePnPGeneric(obj, img, K, None, flags=flag)
        except cv2.error:
            continue
        for rv, tv in zip(rvs, tvs):
            rv2, tv2 = cv2.solvePnPRefineLM(obj, img, K, None, rv.copy(), tv.copy())
            pr, _ = cv2.projectPoints(obj, rv2, tv2, K, None)
            e = float(np.median(np.linalg.norm(pr.reshape(-1, 2) - img, axis=1)))
            if best is None or e < best[0]:
                best = (e, T(cv2.Rodrigues(rv2)[0], tv2))
    if best is None:
        return None, len(keys), np.inf
    return best[1], len(keys), best[0]


def cam_corners_3d(img_p, dep_p, ii, res):
    """(marker,corner) -> metric 3D point in CAMERA frame, from the ZED depth."""
    g = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
    dep = np.load(dep_p)
    K = K_from(ii, g.shape[1], res)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    cs, ids, _ = _DET.detectMarkers(g)
    out = {}
    if ids is None:
        return out
    for i, c in zip(ids.ravel(), cs):
        for k, (u, v) in enumerate(c.reshape(4, 2)):
            ui, vi = int(round(u)), int(round(v))
            win = dep[max(0, vi - 2):vi + 3, max(0, ui - 2):ui + 3]
            win = win[np.isfinite(win) & (win > 0.1)]
            if win.size >= 3:
                z = float(np.median(win))
                out[(int(i), k)] = np.array([(u - cx) / fx * z, (v - cy) / fy * z, z])
    return out


def spread(Ts, label):
    """Rotation/translation spread of a set of 4x4s that should all be equal."""
    ts = np.stack([M[:3, 3] for M in Ts])
    Rs = [M[:3, :3] for M in Ts]
    Rm = Rotation.from_matrix(np.stack(Rs)).mean().as_matrix()
    ang = [np.degrees(np.arccos(np.clip((np.trace(Rm.T @ R) - 1) / 2, -1, 1))) for R in Rs]
    print(f"[{label}] n={len(Ts)}  translation sd {np.linalg.norm(ts.std(0))*1000:.1f} mm "
          f"(per-axis {np.round(ts.std(0)*1000,1)})  rotation sd {np.std(ang):.2f} deg "
          f"max dev {np.max(ang):.2f} deg")
    return T(Rm, ts.mean(0)), float(np.linalg.norm(ts.std(0)) * 1000), float(np.std(ang))


def main():
    intr = json.load(open(f"{SR}/intrinsics.json"))
    world = {eval(k): np.array(v) for k, v in json.load(open(f"{GS}/board_model.json")).items()}
    poses = sorted(glob.glob(f"{SR}/pose_*"))
    print(f"[data] {len(poses)} poses, board model {len(world)} corners")

    rows = []
    for p in poses:
        st = json.load(open(f"{p}/state.json"))
        q = np.array(st["ee_quat"])                       # xyzw
        T_base_ee = T(Rotation.from_quat(q).as_matrix(), st["ee_pos"])
        rec = {"name": os.path.basename(p), "T_base_ee": T_base_ee}
        for cam, key in (("zedm", "wrist"), ("zed2i", "ext")):
            # depth-Kabsch (kept for the comparison that motivates PnP)
            c3 = cam_corners_3d(f"{p}/{cam}_left.png", f"{p}/{cam}_depth.npy",
                                intr[cam]["left"], intr[cam]["resolution"])
            sh = sorted(set(c3) & set(world))
            if len(sh) >= 6:
                A = np.stack([world[k] for k in sh])      # board frame
                B = np.stack([c3[k] for k in sh])         # camera frame
                R, t, res = kabsch(A, B)
                rec[key + "_depth"] = T(R, t)
                rec[key + "_res"] = float(np.median(res))
                rec[key + "_n"] = len(sh)
            # 2D PnP (the one we actually use)
            Tp, npts, err = cam_pose_pnp(f"{p}/{cam}_left.png", intr[cam]["left"],
                                         world, intr[cam]["resolution"])
            if Tp is not None:
                rec[key] = Tp                              # T_cam_board (target2cam)
                rec[key + "_px"] = err
                rec[key + "_npnp"] = npts
        rows.append(rec)

    # ---- 0. depth vs PnP, the measurement that decides which to trust ----
    for key, label in (("ext", "zed2i(ext)"), ("wrist", "zedm(wrist)")):
        dres = [r[key + "_res"] * 1000 for r in rows if key + "_res" in r]
        pres = [r[key + "_px"] for r in rows if key + "_px" in r]
        print(f"[fit] {label}: depth-kabsch residual median {np.median(dres):.1f} mm "
              f"| PnP reprojection median {np.median(pres):.2f} px "
              f"({len(pres)}/{len(rows)} poses)")
    d_ext = [np.linalg.inv(r["ext_depth"]) for r in rows if "ext_depth" in r]
    spread(d_ext, "T_board_zed2i via DEPTH")

    # ---- 1. external camera: T_board_zed2i from each pose independently ----
    ext = [np.linalg.inv(r["ext"]) for r in rows if "ext" in r]
    T_board_zed2i, ext_mm, ext_deg = spread(ext, "T_board_zed2i via PnP")
    print(f"      camera @ board {T_board_zed2i[:3,3].round(4)}")
    res_e = [r["ext_px"] for r in rows if "ext_px" in r]

    # ---- 2. hand-eye for the wrist camera ----
    ok = [r for r in rows if "wrist" in r]
    print(f"[wrist] {len(ok)}/{len(rows)} poses usable via PnP")
    R_g2b = [r["T_base_ee"][:3, :3] for r in ok]
    t_g2b = [r["T_base_ee"][:3, 3] for r in ok]
    R_t2c = [r["wrist"][:3, :3] for r in ok]
    t_t2c = [r["wrist"][:3, 3] for r in ok]
    best = None
    for name, method in [("TSAI", cv2.CALIB_HAND_EYE_TSAI),
                         ("PARK", cv2.CALIB_HAND_EYE_PARK),
                         ("HORAUD", cv2.CALIB_HAND_EYE_HORAUD),
                         ("ANDREFF", cv2.CALIB_HAND_EYE_ANDREFF),
                         ("DANIILIDIS", cv2.CALIB_HAND_EYE_DANIILIDIS)]:
        try:
            Rc, tc = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
        except Exception as e:
            print(f"  hand-eye {name}: {e}")
            continue
        T_ee_cam = T(Rc, tc)
        # consistency: T_base_board must be the same from every pose
        Tbb = [r["T_base_ee"] @ T_ee_cam @ r["wrist"] for r in ok]
        _, mm, deg = spread(Tbb, f"T_base_board via {name}")
        if best is None or (mm + 50 * deg) < (best[1] + 50 * best[2]):
            best = (T_ee_cam, mm, deg, name, Tbb)
    T_ee_cam, mm, deg, name, Tbb = best
    print(f"\n[hand-eye] best method {name}: T_base_board spread {mm:.1f} mm / {deg:.2f} deg")
    print(f"           wrist camera in EE frame: {T_ee_cam[:3,3].round(4)} m")
    T_base_board, _, _ = spread(Tbb, "T_base_board FINAL")
    print(f"           board origin in BASE frame: {T_base_board[:3,3].round(4)}")

    out = dict(
        T_board_zed2i=T_board_zed2i.tolist(), zed2i_spread_mm=ext_mm,
        zed2i_spread_deg=ext_deg, zed2i_pnp_reproj_px=float(np.median(res_e)),
        T_ee_zedm=T_ee_cam.tolist(), handeye_method=name,
        T_base_board=T_base_board.tolist(),
        base_board_spread_mm=mm, base_board_spread_deg=deg,
        n_poses=len(ok), source="scan_run2 (14 poses), board_model.json",
    )
    json.dump(out, open(f"{GS}/ground_run2_solved.json", "w"), indent=2)
    print(f"\n[done] wrote {GS}/ground_run2_solved.json")


if __name__ == "__main__":
    main()

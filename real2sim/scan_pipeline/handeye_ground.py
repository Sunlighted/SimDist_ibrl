"""Ground the robot base and the cameras in the board/world frame (scan_run1).

Chain (all rigid transforms, no scale unknowns -- ZED depth is metric):
  world(board)  <-- Kabsch -->  zedm cam  <-- calibrateHandEye -->  EE  <-- FK -->  base
so per pose: T_world->base = T_ee->base . T_cam->ee . T_world->cam, which must be
CONSTANT across the 8 robot poses; its spread is the calibration residual.
The external zed2i is static: T_world->zed2i per pose must also be constant.

"world" here = the sim world of calibrate_scene_scale.py (ChArUco corner plane
at z=0.8, board center at (0.20, 0.30)). Corner world coords come from the
scan4 COLMAP triangulation mapped through scene_world_transform.json.

Outputs ground_transforms.json:
  T_world_base   : robot base pose in world frame (+ per-pose spread)
  T_ee_zedm      : wrist ZED extrinsic (hand-eye X)
  T_world_zed2i  : external ZED pose in world frame (+ per-pose spread)
"""
import json, os, sys
import numpy as np
import cv2

sys.path.insert(0, "/scratch/yifanr/gsbench")
import calibrate_scene_scale as C

GS = "/scratch/yifanr/gsbench"
SR = "/home/yifanr/scan_run1"
DIC = cv2.aruco.DICT_5X5_50


def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.asarray(t).ravel(); return M


def inv(M):
    R, t = M[:3, :3], M[:3, 3]
    return T(R.T, -R.T @ t)


def quat_xyzw_to_R(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def kabsch(A, B):
    """R,t with B ~= R@A + t (A,B: Nx3)."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = cb - R @ ca
    res = np.linalg.norm((A @ R.T + t) - B, axis=1)
    return R, t, res


def cam_corners_3d(img_p, dep_p, intr):
    """(id,k) -> 3D point in camera frame via metric depth."""
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    g = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
    depth = np.load(dep_p)
    out = {}
    for mid, cs in C.detect(g, DIC).items():
        for k in range(4):
            u, v = cs[k]
            ui, vi = int(round(u)), int(round(v))
            win = depth[max(0, vi-2):vi+3, max(0, ui-2):ui+3]
            win = win[np.isfinite(win) & (win > 0.05)]
            if win.size < 3:
                continue
            z = float(np.median(win))
            out[(mid, k)] = np.array([(u-cx)/fx*z, (v-cy)/fy*z, z])
    return out


def rot_angle(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def avg_T(Ms):
    ts = np.stack([M[:3, 3] for M in Ms])
    R0 = Ms[0][:3, :3]
    # average rotation via projection of the mean matrix
    Rm = sum(M[:3, :3] for M in Ms) / len(Ms)
    U, _, Vt = np.linalg.svd(Rm)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    dev_r = [rot_angle(R.T @ M[:3, :3]) for M in Ms]
    dev_t = np.linalg.norm(ts - ts.mean(0), axis=1)
    return T(R, ts.mean(0)), (float(np.max(dev_r)), float(np.max(dev_t)))


def main():
    # ---- world-frame coordinates of every (marker, corner) --------------
    cams = C.read_cameras_bin(f"{C.SPARSE}/cameras.bin")
    imgs = C.read_images_bin(f"{C.SPARSE}/images.bin")
    tri = C.colmap_marker_points(cv2.aruco.getPredefinedDictionary(DIC) and DIC, cams, imgs)
    tf = json.load(open(f"{GS}/scene_world_transform.json"))
    s, Rw, tw = tf["scale"], np.array(tf["R"]), np.array(tf["t"])
    world = {k: s * (Rw @ X) + tw for k, X in tri.items()}
    print(f"[world] {len(world)} marker corners in world frame "
          f"(z mean {np.mean([v[2] for v in world.values()]):.4f})")

    intr = json.load(open(f"{SR}/intrinsics.json"))
    poses = sorted(p for p in os.listdir(SR) if p.startswith("pose_"))

    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    T_w_zed2i_list, kab_res = [], []
    T_b_ee_list = []
    for p in poses:
        st = json.load(open(f"{SR}/{p}/state.json"))
        T_b_ee = T(quat_xyzw_to_R(st["ee_quat"]), st["ee_pos"])

        # wrist cam -> world (target) via 3D-3D
        cw = cam_corners_3d(f"{SR}/{p}/zedm_left.png", f"{SR}/{p}/zedm_depth.npy",
                            intr["zedm"]["left"])
        shared = sorted(set(cw) & set(world))
        if len(shared) >= 6:
            A = np.stack([world[k] for k in shared])   # world
            B = np.stack([cw[k] for k in shared])      # cam
            Rk, tk, res = kabsch(A, B)                 # world -> cam
            kab_res.append(np.median(res))
            R_g2b.append(T_b_ee[:3, :3]); t_g2b.append(T_b_ee[:3, 3])
            R_t2c.append(Rk); t_t2c.append(tk)
            T_b_ee_list.append(T_b_ee)
            print(f"  {p}: zedm {len(shared)} corners, kabsch med res "
                  f"{np.median(res)*1000:.1f} mm")
        else:
            print(f"  {p}: zedm sees only {len(cw)} corners ({len(shared)} shared) -- skipped")

        # external cam -> world (should be constant)
        ce = cam_corners_3d(f"{SR}/{p}/zed2i_left.png", f"{SR}/{p}/zed2i_depth.npy",
                            intr["zed2i"]["left"])
        shared_e = sorted(set(ce) & set(world))
        if len(shared_e) >= 6:
            A = np.stack([world[k] for k in shared_e])
            B = np.stack([ce[k] for k in shared_e])
            Rk, tk, _ = kabsch(A, B)
            T_w_zed2i_list.append(inv(T(Rk, tk)))      # cam pose in world

    assert len(R_g2b) >= 3, f"only {len(R_g2b)} usable wrist views"
    Rx, tx = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c,
                                  method=cv2.CALIB_HAND_EYE_TSAI)
    T_ee_cam = T(Rx, tx)  # cam pose in EE frame
    print(f"[handeye] T_ee_zedm t = {(tx.ravel()*1000).round(1)} mm")

    # per-pose base pose in world; must agree
    Tb = []
    for T_b_ee, Rk, tk in zip(T_b_ee_list, R_t2c, t_t2c):
        T_c_w = inv(T(Rk, tk))            # world in cam... careful: T(Rk,tk): world->cam
        # base->world = (world->base)^-1 ; world->base = T_b_ee . T_ee_cam . T(world->cam)
        T_w_b = T_b_ee @ T_ee_cam @ T(Rk, tk)   # maps world -> base
        Tb.append(inv(T_w_b))             # base pose in world frame
    T_wb, (dr, dt) = avg_T(Tb)
    print(f"[base] robot base in world: t = {T_wb[:3,3].round(4)}, "
          f"spread rot {dr:.2f} deg / trans {dt*1000:.1f} mm over {len(Tb)} poses")

    T_z2, (dr2, dt2) = avg_T(T_w_zed2i_list)
    print(f"[zed2i] external cam in world: t = {T_z2[:3,3].round(4)}, "
          f"spread rot {dr2:.2f} deg / trans {dt2*1000:.1f} mm over {len(T_w_zed2i_list)}")

    json.dump({
        "T_world_base": T_wb.tolist(), "base_spread_deg_mm": [dr, dt*1000],
        "T_ee_zedm": T_ee_cam.tolist(),
        "T_world_zed2i": T_z2.tolist(), "zed2i_spread_deg_mm": [dr2, dt2*1000],
        "kabsch_med_res_mm": float(np.median(kab_res)*1000),
        "n_poses": len(Tb),
        "note": "world = board frame of scene_world_transform.json (z=0.8 plane)",
    }, open(f"{GS}/ground_transforms.json", "w"), indent=2)
    print(f"[done] wrote {GS}/ground_transforms.json")


if __name__ == "__main__":
    main()

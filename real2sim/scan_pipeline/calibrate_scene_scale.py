"""Solve the similarity transform placing the scan4 scene splat in the sim world.

Phase A residual (1) of docs/real2sim.md: scene.ply lives in COLMAP's arbitrary
frame (monocular SfM => unknown scale). Anchors, both fully automatic:
  * SCALE: aruco markers of the ChArUco board seen BOTH by the scan4 video
    (triangulated in COLMAP frame from the SfM poses) and by the ZED captures
    in ~/scan_run1 (metric via the depth maps). Same marker-corner pairs,
    distance ratio = scale. No physical board spec needed.
  * PLANE/AXES: the board lies flat on the table => its corner plane maps to
    the sim tabletop z=TABLE_Z; in-plane axes from PCA of the corner layout.

Output: scene_world_transform.json with X_world = s * R @ X_colmap + t.
"""
import json
import os
import struct
import sys

import cv2
import numpy as np

GS = "/scratch/yifanr/gsbench"
SPARSE = f"{GS}/sfm_scan4/sparse/1"
IMAGES = f"{GS}/sfm_scan4/images"
SCAN_RUN = "/home/yifanr/scan_run1"
TABLE_Z = 0.8
BOARD_XY_TARGET = (0.20, 0.30)  # where the ChArUco center lands on the sim table


# ---------------------------------------------------------------- colmap io
def read_cameras_bin(path):
    cams = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
            nparams = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12}[model]
            params = struct.unpack(f"<{nparams}d", f.read(8 * nparams))
            cams[cid] = dict(model=model, w=w, h=h, params=np.array(params))
    return cams


def read_images_bin(path):
    imgs = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid = struct.unpack("<i", f.read(4))[0]
            q = np.array(struct.unpack("<4d", f.read(32)))  # wxyz, world->cam
            t = np.array(struct.unpack("<3d", f.read(24)))
            cid = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            npts = struct.unpack("<Q", f.read(8))[0]
            f.read(npts * 24)  # skip 2D points
            imgs[iid] = dict(q=q, t=t, cid=cid, name=name.decode())
    return imgs


def qvec2rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


# ---------------------------------------------------------------- detection
DICTS = [
    ("4X4_50", cv2.aruco.DICT_4X4_50), ("4X4_250", cv2.aruco.DICT_4X4_250),
    ("5X5_50", cv2.aruco.DICT_5X5_50), ("5X5_250", cv2.aruco.DICT_5X5_250),
    ("6X6_50", cv2.aruco.DICT_6X6_50), ("6X6_250", cv2.aruco.DICT_6X6_250),
    ("APRIL_36h11", cv2.aruco.DICT_APRILTAG_36h11),
    ("ORIGINAL", cv2.aruco.DICT_ARUCO_ORIGINAL),
]


def detect(gray, dic):
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dic))
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}


def pick_dictionary(sample_paths):
    best, best_n = None, 0
    for name, dic in DICTS:
        n = 0
        for p in sample_paths:
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            n += len(detect(g, dic))
        if n > best_n:
            best, best_n = (name, dic), n
    print(f"[dict] {best[0]} with {best_n} detections over {len(sample_paths)} samples")
    assert best_n >= 8, "no aruco dictionary matched the board"
    return best[1]


# ------------------------------------------------------------------- zed 3d
def zed_metric_points(dic):
    """(marker_id, corner_idx) -> list of metric 3D points across ZED poses.
    Depth is sampled from a small window median around each corner pixel."""
    intr = json.load(open(f"{SCAN_RUN}/intrinsics.json"))["zed2i"]["left"]
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    pts = {}
    for pose in sorted(os.listdir(SCAN_RUN)):
        if not pose.startswith("pose_"):
            continue
        img_p = f"{SCAN_RUN}/{pose}/zed2i_left.png"
        dep_p = f"{SCAN_RUN}/{pose}/zed2i_depth.npy"
        if not (os.path.exists(img_p) and os.path.exists(dep_p)):
            continue
        g = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        depth = np.load(dep_p)
        for mid, cs in detect(g, dic).items():
            for k in range(4):
                u, v = cs[k]
                ui, vi = int(round(u)), int(round(v))
                win = depth[max(0, vi - 2):vi + 3, max(0, ui - 2):ui + 3]
                win = win[np.isfinite(win) & (win > 0.1)]
                if win.size < 3:
                    continue
                z = float(np.median(win))
                pts.setdefault((mid, k), []).append(
                    np.array([(u - cx) / fx * z, (v - cy) / fy * z, z])
                )
    return pts


# ------------------------------------------------------------- colmap 3d
def colmap_marker_points(dic, cams, imgs):
    """Triangulate each (marker_id, corner_idx) from all scan4 views (DLT)."""
    obs = {}
    n_img = 0
    for iid, im in imgs.items():
        p = f"{IMAGES}/{im['name']}"
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        cam = cams[im["cid"]]
        fx, fy, cx, cy = cam["params"][:4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist = np.array([*cam["params"][4:8]]) if cam["model"] == 4 else np.zeros(4)
        dets = detect(g, dic)
        if not dets:
            continue
        n_img += 1
        R = qvec2rot(im["q"])
        P = K @ np.hstack([R, im["t"].reshape(3, 1)])
        for mid, cs in dets.items():
            und = cv2.undistortPoints(cs.reshape(-1, 1, 2), K, dist, P=K).reshape(4, 2)
            for k in range(4):
                obs.setdefault((mid, k), []).append((P, und[k]))
    print(f"[colmap] board detected in {n_img} images")

    tri = {}
    for key, ob in obs.items():
        if len(ob) < 3:
            continue
        A = []
        for P, uv in ob:
            A.append(uv[0] * P[2] - P[0])
            A.append(uv[1] * P[2] - P[1])
        _, _, Vt = np.linalg.svd(np.asarray(A))
        X = Vt[-1]
        X = X[:3] / X[3]
        # reprojection sanity: median error < 3 px
        errs = []
        for P, uv in ob:
            x = P @ np.append(X, 1.0)
            errs.append(np.linalg.norm(x[:2] / x[2] - uv))
        if np.median(errs) < 3.0:
            tri[key] = X
    return tri


def main():
    cams = read_cameras_bin(f"{SPARSE}/cameras.bin")
    imgs = read_images_bin(f"{SPARSE}/images.bin")
    print(f"[colmap] {len(imgs)} posed images, cam params {cams[list(cams)[0]]['params'][:4].round(1)}")

    names = sorted(os.listdir(IMAGES))
    samples = [f"{IMAGES}/{n}" for n in names[:: max(1, len(names) // 12)]][:12]
    dic = pick_dictionary(samples + [f"{SCAN_RUN}/pose_000/zed2i_left.png"])

    zed = zed_metric_points(dic)
    print(f"[zed] {len(zed)} marker corners with metric depth")
    tri = colmap_marker_points(dic, cams, imgs)
    print(f"[colmap] {len(tri)} marker corners triangulated")

    common = sorted(set(zed) & set(tri))
    assert len(common) >= 6, f"only {len(common)} shared corners"
    zpts = {k: np.median(np.stack(zed[k]), 0) for k in common}

    # scale: all pairwise distances over shared corners
    ratios = []
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a, b = common[i], common[j]
            dm = np.linalg.norm(zpts[a] - zpts[b])
            dc = np.linalg.norm(tri[a] - tri[b])
            if dc > 1e-6 and dm > 0.02:  # skip tiny baselines
                ratios.append(dm / dc)
    ratios = np.array(ratios)
    s = float(np.median(ratios))
    spread = float(np.percentile(ratios, 75) - np.percentile(ratios, 25))
    print(f"[scale] s = {s:.4f} m/colmap-unit  (n={len(ratios)}, IQR {spread:.4f}, "
          f"rel {spread / s:.1%})")

    # plane + in-plane axes from the COLMAP corners
    P = np.stack([tri[k] for k in common])
    c0 = P.mean(0)
    U, S, Vt = np.linalg.svd(P - c0)
    normal = Vt[2]
    # camera centers must be on the +normal side (cameras look down at the board)
    cam_centers = np.stack([-qvec2rot(im["q"]).T @ im["t"] for im in imgs.values()])
    if np.median((cam_centers - c0) @ normal) < 0:
        normal = -normal
    x_in = Vt[0] - (Vt[0] @ normal) * normal
    x_in /= np.linalg.norm(x_in)
    y_in = np.cross(normal, x_in)
    # rows of R = world axes expressed in colmap frame => R @ v maps to world
    R = np.stack([x_in, y_in, normal])
    t = np.array([BOARD_XY_TARGET[0], BOARD_XY_TARGET[1], TABLE_Z]) - s * (R @ c0)

    out = dict(scale=s, R=R.tolist(), t=t.tolist(),
               n_corners=len(common), scale_iqr=spread,
               board_center_colmap=c0.tolist(),
               note="X_world = scale * R @ X_colmap + t; board plane -> z=%.2f" % TABLE_Z)
    json.dump(out, open(f"{GS}/scene_world_transform.json", "w"), indent=2)
    print(f"[done] wrote {GS}/scene_world_transform.json")


if __name__ == "__main__":
    main()

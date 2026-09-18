"""scan5 COLMAP frame -> board/world frame, WITHOUT scan4.

Triangulates the 5X5_50 markers in the scan5 reconstruction and solves the
similarity (Umeyama, scale included) onto board_model.json -- the metric
corner constellation measured by the ZEDs in scan_run2. Requires the ChArUco
not to have moved between scan5 and scan_run2 (same session).

Writes scene5_world_transform.json: X_world = s * R @ X_colmap + t.
"""
import glob, json, struct, sys
import numpy as np
import cv2

sys.path.insert(0, "/scratch/yifanr/gsbench")
import calibrate_scene_scale as C  # reuse the colmap parsing + detection

GS = "/scratch/yifanr/gsbench"
DIC = cv2.aruco.DICT_5X5_50


def best_sparse(work):
    best, n_best = None, -1
    for p in glob.glob(f"{work}/sparse/*/images.bin"):
        n = struct.unpack("<Q", open(p, "rb").read(8))[0]
        if n > n_best:
            best, n_best = p.rsplit("/", 1)[0], n
    print(f"[sparse] {best} with {n_best} images")
    assert n_best >= 300, f"best model has only {n_best} images"
    return best


def umeyama(A, B):
    """s, R, t with B ~= s*R@A + t."""
    ca, cb = A.mean(0), B.mean(0)
    Ac, Bc = A - ca, B - cb
    H = Ac.T @ Bc / len(A)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    s = np.trace(np.diag(S) @ D) / (Ac ** 2).sum(axis=None) * len(A)
    t = cb - s * R @ ca
    res = np.linalg.norm((s * (A @ R.T) + t) - B, axis=1)
    return s, R, t, res


def main():
    work = f"{GS}/sfm_scan5"
    C.SPARSE = best_sparse(work)
    C.IMAGES = f"{work}/images"
    cams = C.read_cameras_bin(f"{C.SPARSE}/cameras.bin")
    imgs = C.read_images_bin(f"{C.SPARSE}/images.bin")
    tri = C.colmap_marker_points(DIC, cams, imgs)
    print(f"[scan5] {len(tri)} marker corners triangulated")

    board = {eval(k): np.array(v) for k, v in
             json.load(open(f"{GS}/board_model.json")).items()}
    common = sorted(set(tri) & set(board))
    assert len(common) >= 8, f"only {len(common)} shared corners with board model"
    A = np.stack([tri[k] for k in common])
    B = np.stack([board[k] for k in common])
    s, R, t, res = umeyama(A, B)
    print(f"[umeyama] {len(common)} corners: scale {s:.4f}, residual "
          f"med {np.median(res)*1000:.1f} mm p95 {np.percentile(res,95)*1000:.1f} mm")
    json.dump(dict(scale=float(s), R=R.tolist(), t=t.tolist(),
                   n_corners=len(common), res_med_mm=float(np.median(res)*1000),
                   sparse=C.SPARSE, target="board_model.json (scan_run2, scan4-free)"),
              open(f"{GS}/scene5_world_transform.json", "w"), indent=2)
    print(f"[done] wrote scene5_world_transform.json")


if __name__ == "__main__":
    main()

"""Any COLMAP reconstruction -> board/world frame (generalised calibrate_scan5_to_board).

Same method as calibrate_scan5_to_board.py (triangulate the 5X5_50 ChArUco
markers in the reconstruction, Umeyama onto board_model.json -- the metric corner
constellation measured by the ZEDs), but takes the work dir and output path as
arguments so the merged multi-sweep model can be calibrated with the same code.

Umeyama-onto-metric-corners is preferred over the pairwise distance-ratio scale
of calibrate_scene_scale.py: it solves scale, rotation and translation jointly
and reports a per-corner residual in mm, which is a usable correctness gate
(scan5: 0.98 mm median).

    python calibrate_to_board.py --work sfm_merged --out scene_merged_world_transform.json
"""
import argparse
import glob
import json
import struct
import sys

import cv2
import numpy as np

sys.path.insert(0, "/scratch/yifanr/gsbench")
import calibrate_scene_scale as C  # colmap parsing + marker detection

GS = "/scratch/yifanr/gsbench"
DIC = cv2.aruco.DICT_5X5_50


def best_sparse(work, min_images):
    best, n_best = None, -1
    for p in glob.glob(f"{work}/sparse/*/images.bin") + \
             glob.glob(f"{work}/sparse_plus_*/images.bin"):
        n = struct.unpack("<Q", open(p, "rb").read(8))[0]
        print(f"#   {p}: {n} images")
        if n > n_best:
            best, n_best = p.rsplit("/", 1)[0], n
    print(f"[sparse] {best} with {n_best} images")
    assert best and n_best >= min_images, f"best model has only {n_best} images"
    return best


def umeyama(A, B):
    """s, R, t with B ~= s*R@A + t, plus per-point residuals."""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="e.g. sfm_merged (contains sparse/, images/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_images", type=int, default=300)
    ap.add_argument("--board", default=f"{GS}/board_model.json")
    a = ap.parse_args()

    work = a.work if a.work.startswith("/") else f"{GS}/{a.work}"
    C.SPARSE = best_sparse(work, a.min_images)
    C.IMAGES = f"{work}/images"
    cams = C.read_cameras_bin(f"{C.SPARSE}/cameras.bin")
    imgs = C.read_images_bin(f"{C.SPARSE}/images.bin")
    print(f"[colmap] {len(cams)} camera model(s), {len(imgs)} posed images")
    tri = C.colmap_marker_points(DIC, cams, imgs)
    print(f"[tri] {len(tri)} marker corners triangulated")

    board = {eval(k): np.array(v) for k, v in json.load(open(a.board)).items()}
    common = sorted(set(tri) & set(board))
    assert len(common) >= 8, f"only {len(common)} shared corners with the board model"
    A = np.stack([tri[k] for k in common])
    B = np.stack([board[k] for k in common])
    s, R, t, res = umeyama(A, B)
    print(f"[umeyama] {len(common)} corners: scale {s:.5f} m/unit, residual "
          f"med {np.median(res)*1000:.2f} mm  p95 {np.percentile(res,95)*1000:.2f} mm")
    out = a.out if a.out.startswith("/") else f"{GS}/{a.out}"
    json.dump(dict(scale=float(s), R=R.tolist(), t=t.tolist(),
                   n_corners=len(common), res_med_mm=float(np.median(res) * 1000),
                   res_p95_mm=float(np.percentile(res, 95) * 1000),
                   sparse=C.SPARSE, target=a.board),
              open(out, "w"), indent=2)
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()

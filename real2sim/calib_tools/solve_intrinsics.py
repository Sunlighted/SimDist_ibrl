"""Solve one camera's intrinsics from a record_calib.py --intrinsics run.

Fits K (fx = fy) and D = (k1, k2, p1, p2, k3) to the ChArUco chessboard corners, drops views
that do not fit, and reports what the fit can be trusted for: corner coverage of the image
(the distortion is only constrained where corners were seen) and a k-fold check (fit without a
fold, PnP the held-out views with the fixed intrinsics).

The square size does not enter K or D, so the nominal one is used.

    ~/anaconda3/envs/simdist/bin/python ~/simdist/calib/solve_intrinsics.py ~/simdist/calib/intr_XXXX
Pass several runs to pool them (same camera, not moved in between).
Writes <first run>/intrinsics.json, which record_calib.py --wrist_intrinsics and
solve_calib.py --wrist_intrinsics take.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

SQUARE0, MARKER0 = 0.0424, 0.031
BOARD = cv2.aruco.CharucoBoard((5, 7), SQUARE0, MARKER0,
                               cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50))
OBJ = BOARD.getChessboardCorners().astype(np.float32)
DET = cv2.aruco.CharucoDetector(BOARD)
GRID = (6, 10)


def detect(path):
    cc, ci, _, _ = DET.detectBoard(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    if ci is None:
        return np.zeros(0, int), np.zeros((0, 2), np.float32)
    return ci.ravel().astype(int), cc.reshape(-1, 2).astype(np.float32)


def calibrate(views, size, K0, D0, extra=0):
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | extra
    rms, K, D, rv, tv = cv2.calibrateCamera([OBJ[v["ids"]] for v in views],
                                            [v["pix"] for v in views], size, K0.copy(),
                                            D0.copy().reshape(1, -1), flags=flags)
    errs = []
    for v, r, t in zip(views, rv, tv):
        p, _ = cv2.projectPoints(OBJ[v["ids"]], r, t, K, D)
        errs.append(float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - v["pix"]) ** 2, axis=1)))))
    tilt = [float(np.degrees(np.arccos(abs(cv2.Rodrigues(r)[0][2, 2])))) for r in rv]
    return rms, K, D.ravel(), np.array(errs), np.array(tilt)


def pnp_err(v, K, D):
    ok, r, t = cv2.solvePnP(OBJ[v["ids"]], v["pix"], K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    p, _ = cv2.projectPoints(OBJ[v["ids"]], r, t, K, D)
    return float(np.median(np.linalg.norm(p.reshape(-1, 2) - v["pix"], axis=1)))


def shift(Ka, Da, Kb, Db, p):
    """Pixel p under model a -> ray -> pixel under model b; distance in px."""
    ray = cv2.undistortPoints(np.float64(p).reshape(1, 1, 2), Ka, Da).reshape(2)
    q, _ = cv2.projectPoints(np.r_[ray, 1.0].reshape(1, 3), np.zeros(3), np.zeros(3), Kb, Db)
    return float(np.linalg.norm(q.ravel() - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="intrinsics run dir(s); extrinsics runs work too")
    ap.add_argument("--cam", default="wrist")
    ap.add_argument("--min_corners", type=int, default=8)
    ap.add_argument("--reject", type=float, default=3.0,
                    help="drop views whose rms exceeds this many times the median, then refit")
    ap.add_argument("--fovy", type=float, default=49.014,
                    help="policy crop fovy, to check that its corners are covered")
    args = ap.parse_args()
    cam = args.cam
    sess = json.load(open(os.path.join(args.runs[0], "session.json")))
    c = sess["cameras"][cam]
    W, H = sess["size"]
    K0, D0 = np.array(c["K"]), np.array(c["D"])

    views = []
    for run in args.runs:
        other = json.load(open(os.path.join(run, "session.json")))
        if other["cameras"][cam]["serial"] != c["serial"] or other["size"] != sess["size"]:
            raise SystemExit(f"{run}: different camera or image size")
        tag = os.path.basename(os.path.normpath(run))
        for d in sorted(glob.glob(os.path.join(run, "sample_*"))):
            ids, pix = detect(os.path.join(d, f"{cam}_left.png"))
            name = f"{tag}/{os.path.basename(d)}"
            if len(ids) >= args.min_corners:
                views.append(dict(name=name, ids=ids, pix=pix))
            else:
                print(f"skip {name}: {len(ids)} corners")
    print(f"[{cam}] {len(views)} views, {sum(len(v['ids']) for v in views)} corners")

    rms, K, D, errs, _ = calibrate(views, (W, H), K0, D0)
    bad = errs > args.reject * np.median(errs)
    if bad.any():
        print("dropping outlier views:", ", ".join(
            f"{v['name']} ({e:.2f}px)" for v, e, b in zip(views, errs, bad) if b))
        views = [v for v, b in zip(views, bad) if not b]
        rms, K, D, errs, _ = calibrate(views, (W, H), K0, D0)
    _, _, _, _, tilt = calibrate(views, (W, H), K, D)
    print(f"rms {rms:.3f} px  (per view: median {np.median(errs):.2f}, max {errs.max():.2f})")
    print(f"K: fx {K[0, 0]:.2f} fy {K[1, 1]:.2f} cx {K[0, 2]:.2f} cy {K[1, 2]:.2f}")
    print(f"D: {np.round(D, 5).tolist()}")

    # coverage: the distortion is extrapolated wherever there are no corners
    allp = np.concatenate([v["pix"] for v in views])
    cover = np.zeros(GRID, int)
    for v in views:
        r = np.clip((v["pix"][:, 1] * GRID[0] / H).astype(int), 0, GRID[0] - 1)
        cc = np.clip((v["pix"][:, 0] * GRID[1] / W).astype(int), 0, GRID[1] - 1)
        for rc in set(zip(r, cc)):
            cover[rc] += 1
    print(f"\n== coverage: views with a corner in each cell ({GRID[0]}x{GRID[1]}), "
          f"{100 * np.mean(cover > 0):.0f}% of cells, {100 * np.mean(cover >= 3):.0f}% with >= 3")
    print(cover)
    radius = np.linalg.norm(allp - K[:2, 2], axis=1)
    side = min(2 * K[1, 1] * np.tan(np.radians(args.fovy) / 2), 2 * min(K[0, 2], K[1, 2]) + 1)
    crop_r = side / np.sqrt(2)
    print(f"max corner radius {radius.max():.0f} px; policy crop (fovy {args.fovy}) "
          f"corners at {crop_r:.0f} px -> {100 * np.mean(radius > 0.9 * crop_r):.1f}% of corners "
          f"beyond 90% of it")

    # Focal length is only observable from foreshortening: with near-frontal views every
    # distortion model fits equally well and fx / cy drift (intr_1010_2258: max tilt 25 deg,
    # fx 743-832 across models at the same 0.3 px rms).
    n_steep = int(np.sum(tilt >= 35))
    print(f"\n== board tilt: median {np.median(tilt):.0f} deg, max {tilt.max():.0f} deg, "
          f"{n_steep} views >= 35 deg")
    rms2, K2, D2, _, _ = calibrate(views, (W, H), K0, D0, cv2.CALIB_ZERO_TANGENT_DIST)
    dfx, dcy = K2[0, 0] - K[0, 0], K2[1, 2] - K[1, 2]
    print(f"model check (no tangential terms): rms {rms2:.3f}, fx {dfx:+.1f} px, cy {dcy:+.1f} px")
    unstable = n_steep < 8 or abs(dfx) > 5 or abs(dcy) > 5
    if unstable:
        print("WARNING: focal length / principal point are NOT pinned down. Add >= 8 views with "
              "the board tilted 35-55 deg (about both axes), close enough to fill much of the image.")

    # k-fold: fit without a fold, PnP the held-out views with the fixed intrinsics
    k = min(5, len(views))
    held = []
    for f in range(k):
        tr = [v for i, v in enumerate(views) if i % k != f]
        _, Kf, Df, _, _ = calibrate(tr, (W, H), K0, D0)
        held += [pnp_err(v, Kf, Df) for i, v in enumerate(views) if i % k == f]
    print(f"\n== {k}-fold held-out PnP: median {np.median(held):.2f} px, max {np.max(held):.2f} px")

    print("\n== vs the starting intrinsics (" + c.get("intrinsics_file", "?") + "): pixel shift")
    for r in [100, 200, 300, 400, 500, 600]:
        p = K[:2, 2] + [r, 0]
        if p[0] < W:
            print(f"  r={r}: {shift(K, D, K0, D0, p):6.1f} px")

    out = dict(camera=cam, serial=c["serial"], size=[W, H], mode=sess["mode"], eye=sess["eye"],
               K=K.tolist(), D=D.tolist(), rms_px=float(rms), n_views=len(views),
               views=[v["name"] for v in views], kfold_median_px=float(np.median(held)),
               kfold_max_px=float(np.max(held)), coverage=cover.tolist(),
               max_corner_radius_px=float(radius.max()),
               tilt_max_deg=float(tilt.max()), n_views_tilt_ge35=n_steep,
               model_check_dfx_px=float(dfx), model_check_dcy_px=float(dcy),
               unstable=bool(unstable),
               model="OpenCV pinhole, fx=fy, k1 k2 p1 p2 k3; raw (unrectified) left eye")
    out["runs"] = [os.path.abspath(r) for r in args.runs]
    path = os.path.join(args.runs[0], "intrinsics.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

"""Real-frame preprocessing == the sim camera, geometrically.

For each camera of a cell: project 3D points through the REAL camera (K, D as the
real side resolves them), paint a blob at each raw pixel, run the deploy
Rectifier, and check every blob lands where the SIM pinhole (centred, the cell's
fovy, S x S) puts the same point. Also pins Rectifier to the offline stage-2
converter (scripts/convert_real_teleop.py:undistort_crop) and checks reduce=2.

  python scripts/test_preprocess_crop.py [--cell <yaml>]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.sim2real import load_sim2real  # noqa: E402
from real2sim.deploy.preprocess import (Rectifier, SIM_CAMERA_REALCELL,  # noqa: E402
                                        camera_intrinsics, crop_side)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELL = os.path.join(HERE, "real2sim/zed_capture/calib_20261010/threading_real_cell_20261010.yaml")
SESSION = os.path.join(HERE, "real2sim/zed_capture/calib_20261010/data/run_1010_2327/session.json")
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def blob_image(uv, w, h, sigma=2.5):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    im = np.zeros((h, w), np.float32)
    for u, v in uv:
        im += np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * sigma ** 2))
    return (np.clip(im, 0, 1) * 255).astype(np.uint8)[..., None].repeat(3, 2)


def centroids(im, guesses, r=4):
    g = im[..., 0].astype(np.float64)
    out = []
    for u, v in guesses:
        x0, y0 = int(round(u)), int(round(v))
        ys, xs = np.mgrid[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1]
        wgt = g[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1]
        out.append(((wgt * xs).sum() / wgt.sum(), (wgt * ys).sum() / wgt.sum()))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default=CELL)
    ap.add_argument("--S", type=int, default=224)
    a = ap.parse_args()
    cell = load_sim2real(a.cell)
    sess = json.load(open(SESSION))["cameras"]
    W, H, S = 1280, 720, a.S
    from scripts.convert_real_teleop import undistort_crop
    rng = np.random.default_rng(0)
    for ros, sim in SIM_CAMERA_REALCELL.items():
        cc = [c for c in cell.cameras if c.name == sim][0]
        fovy = cc.resolved_fovy()
        K, D = camera_intrinsics(cc, sess[ros]["K"], sess[ros]["D"], W, H)
        print(f"{ros} -> {sim}: fovy {fovy:.3f}, fy {K[1, 1]:.2f}, "
              f"K/D {'refit ' + os.path.basename(cc.real_intrinsics) if cc.real_intrinsics else 'factory'}")
        # points inside the sim view (keep blobs off the crop border)
        # a 7x7 grid (blobs must not overlap or the centroids bias each other)
        t = np.tan(np.radians(fovy) / 2) * 0.9
        g = np.linspace(-t, t, 7)
        P = np.array([[x, y, 1.0] for x in g for y in g]) * rng.uniform(0.3, 1.0, (49, 1))
        raw, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), K, D)
        raw = raw.reshape(-1, 2)
        f_sim = (S / 2) / np.tan(np.radians(fovy) / 2)
        # MuJoCo: centred pinhole; pixel i covers [i, i+1) -> continuous c = idx + .5
        want = S / 2 + f_sim * P[:, :2] / P[:, 2:] - 0.5
        img = blob_image(raw, W, H)
        side_err = {}
        for red in (1, 2):
            im_in = img if red == 1 else cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
            r = Rectifier(K, D, W, H, S, fovy, reduce=red)
            got = centroids(r(im_in), want)
            e = np.linalg.norm(got - want, axis=1)
            side_err[red] = r.side * red
            check(f"{ros} reduce={red}: blobs land on the sim pinhole", e.max() < 0.1,
                  f"(rms {np.sqrt((e ** 2).mean()):.3f} max {e.max():.3f} px @ {S})")
        check(f"{ros}: crop side = 2 fy tan(fovy/2)",
              abs(side_err[1] - 2 * K[1, 1] * np.tan(np.radians(fovy) / 2)) <= 0.5,
              f"({side_err[1]} px, largest square would be "
              f"{2 * int(min(K[0, 2], W - K[0, 2], K[1, 2], H - K[1, 2]))})")
        noise = cv2.GaussianBlur(rng.integers(0, 256, (H, W, 3), dtype=np.uint8), (0, 0), 2)
        ref, _, _ = undistort_crop(noise, K, D, fovy, out=S)
        d = np.abs(Rectifier(K, D, W, H, S, fovy)(noise).astype(int) - ref.astype(int))
        check(f"{ros}: Rectifier == convert_real_teleop.undistort_crop", d.max() == 0,
              f"(max |diff| {d.max()})")
    # PegInsertion_D0: its fovy WAS the largest square, so the fix must not move it
    for cam, fy, fovy, old in (("zedview", 532.470, 67.010, 704), ("robot0_zedm_wrist", 771.705, 48.977, 703)):
        s = crop_side(fy, fovy)
        check(f"peg {cam}: side unchanged within 1 px", abs(s - old) <= 1, f"({s} vs legacy {old})")
    print("ALL PASS" if not FAIL else f"FAILED: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

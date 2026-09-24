"""Fit a per-camera ZED -> phone-splat colour map from the real GELLO demos.

For sampled frames of each real episode, the splat scene is rendered at the REAL
robot's joint angles (same machinery as scripts/compare_real_sim_demos.py), so
real and sim show the same view up to ~15-30 px / 448 of misalignment. The fit
uses only pixels that are:
  - scene splat in sim (not a MuJoCo-rendered arm/gripper pixel, mask dilated to
    absorb the misalignment),
  - not saturated / crushed in the real frame (the strip-light glare on the
    glossy poster clips to white and carries no colour information),
and both images are Gaussian-blurred first so the residual misalignment averages
out. Model: out = M[:, :3] @ rgb + M[:, 3] (sRGB, 0..1), fitted by IRLS with a
Huber weight so the unmodelled peg / people / glare edges cannot dominate.

Validation: fit on episode 1, report MAE on episode 2 (held out), then refit on
both and write real2sim/deploy/color_correction.json + a before/after figure.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import real2sim.splat_obs  # noqa: F401,E402  import-order fix for gs_mujoco
import cv2  # noqa: E402
from real2sim.deploy.preprocess import Rectifier, SIM_CAMERA, apply_color, decode_jpeg_rgb  # noqa: E402
from compare_real_sim_demos import Real, SimReplay  # noqa: E402

OUT_JSON = os.path.join(_REPO, "real2sim", "deploy", "color_correction.json")
OUT_FIG = os.path.join(_REPO, "viz", "real_vs_sim", "color_correction_fit.png")


def collect(real, rp, rects, S, stride, sigma, dil):
    data = {c: ([], []) for c in SIM_CAMERA}
    frames = {c: [] for c in SIM_CAMERA}
    geoms = rp.e._splat.mj_geoms
    for t in range(0, real.T, stride):
        rp.set(real.q[t], real.fingers[t])
        for cam, simcam in SIM_CAMERA.items():
            rgb = decode_jpeg_rgb(real.imgs[cam][t])
            r = rects[cam](rgb)
            s = rp.render(cam)
            seg = rp.sim.render(camera_name=simcam, width=S, height=S,
                                segmentation=True)[::-1][:, :, 1]
            robot = np.isin(seg, geoms).astype(np.uint8)
            robot = cv2.dilate(robot, np.ones((dil, dil), np.uint8)) > 0
            rb = cv2.GaussianBlur(r.astype(np.float32) / 255, (0, 0), sigma)
            sb = cv2.GaussianBlur(s.astype(np.float32) / 255, (0, 0), sigma)
            ok = (~robot) & (r.max(2) < 245) & (r.max(2) > 12)
            ok[::2, :] = False; ok[:, ::2] = False                 # decorrelate neighbours
            data[cam][0].append(rb[ok]); data[cam][1].append(sb[ok])
            if len(frames[cam]) < 4 and t % (stride * 8) == 0:
                frames[cam].append((t, r, s))
    return ({c: (np.concatenate(a), np.concatenate(b)) for c, (a, b) in data.items()}, frames)


def fit_affine(X, Y, iters=8, delta=0.04):
    A = np.hstack([X, np.ones((len(X), 1), np.float32)])
    w = np.ones(len(X), np.float32)
    for _ in range(iters):
        Aw = A * w[:, None]
        M = np.linalg.lstsq(Aw.T @ A, Aw.T @ Y, rcond=None)[0].T           # 3x4
        r = np.linalg.norm(A @ M.T - Y, axis=1)
        w = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-9)).astype(np.float32)
    return M.astype(np.float32)


def fit_diag(X, Y, iters=8, delta=0.04):
    """Per-channel gain + offset (M diagonal): cannot invent hue shifts. The free
    3x3 fit put +-3 cross-channel terms on the wrist camera (glare + close-range
    misalignment) and posterised the image, so this is the model actually used."""
    M = np.zeros((3, 4), np.float32)
    for c in range(3):
        A = np.stack([X[:, c], np.ones(len(X), np.float32)], 1)
        w = np.ones(len(X), np.float32)
        for _ in range(iters):
            Aw = A * w[:, None]
            g, b = np.linalg.lstsq(Aw.T @ A, Aw.T @ Y[:, c], rcond=None)[0]
            r = np.abs(A @ np.array([g, b]) - Y[:, c])
            w = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-9)).astype(np.float32)
        M[c, c], M[c, 3] = g, b
    return M


def fit_moments(X, Y):
    """Per-channel mean/std matching. Pixel-wise regression on images that are
    misaligned by 15-30 px suffers attenuation bias (noisy regressor => slope
    shrunk toward 0: fitted wrist gain 0.47 would halve the contrast). Matching the
    two masked distributions needs no pixel correspondence, so it is unbiased."""
    M = np.zeros((3, 4), np.float32)
    for c in range(3):
        g = Y[:, c].std() / max(X[:, c].std(), 1e-6)
        M[c, c], M[c, 3] = g, Y[:, c].mean() - g * X[:, c].mean()
    return M


def mae(X, Y, M=None):
    P = X if M is None else np.clip(X @ M[:, :3].T + M[:, 3], 0, 1)
    return float(np.abs(P - Y).mean() * 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", default="/scratch/yifanr/simdist/gello_software/demos/peg")
    ap.add_argument("--S", type=int, default=224)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--dilate", type=int, default=17)
    a = ap.parse_args()

    reals = [Real(p) for p in sorted(glob.glob(f"{a.real_dir}/*.h5"))]
    assert len(reals) >= 2, "need >= 2 episodes (one is held out)"
    rp = SimReplay(a.S)
    rects = {c: Rectifier(reals[0].K[c], reals[0].D[c], 1280, 720, a.S) for c in SIM_CAMERA}

    per_ep = []
    for r in reals:
        d, fr = collect(r, rp, rects, a.S, a.stride, a.sigma, a.dilate)
        per_ep.append((d, fr))
        print(f"[collect] {r.name}: " + ", ".join(f"{c} {len(d[c][0])} px" for c in d), flush=True)

    out = {"cameras": {}, "fit": {
        "model": "out = M[:, :3] @ rgb + M[:, 3], sRGB in [0,1], ZED -> splat",
        "episodes": [r.name for r in reals], "S": a.S, "blur_sigma_px": a.sigma,
        "robot_mask_dilate_px": a.dilate, "stride": a.stride}}
    for cam in SIM_CAMERA:
        tr, te = per_ep[0][0][cam], per_ep[1][0][cam]
        allX = np.concatenate([d[cam][0] for d, _ in per_ep])
        allY = np.concatenate([d[cam][1] for d, _ in per_ep])
        Mf_tr, Md_tr, Mm_tr = fit_affine(*tr), fit_diag(*tr), fit_moments(*tr)
        Mf, Md, Mm = fit_affine(allX, allY), fit_diag(allX, allY), fit_moments(allX, allY)
        stats = dict(heldout_mae_raw=mae(*te), heldout_mae_diag=mae(*te, Md_tr),
                     heldout_mae_full=mae(*te, Mf_tr), heldout_mae_moments=mae(*te, Mm_tr),
                     mean_rgb_real=(allX.mean(0) * 255).round(1).tolist(),
                     mean_rgb_splat=(allY.mean(0) * 255).round(1).tolist(), n_px=int(len(allX)))
        print(f"[{cam}] held-out ({reals[1].name}) MAE raw {stats['heldout_mae_raw']:.1f} | "
              f"moments {stats['heldout_mae_moments']:.1f} | per-channel regr {stats['heldout_mae_diag']:.1f} | full 3x4 {stats['heldout_mae_full']:.1f} /255"
              f" | mean real {stats['mean_rgb_real']} splat {stats['mean_rgb_splat']}", flush=True)
        print(f"   moments gain {np.diag(Mm[:, :3]).round(3).tolist()} offset {Mm[:, 3].round(3).tolist()}"
              f" | regr gain {np.diag(Md[:, :3]).round(3).tolist()}", flush=True)
        out["cameras"][cam] = {"M": Mm.tolist(), "model": "per-channel mean/std matching",
                               "M_regression_unused": Md.tolist(), "M_full_3x4_unused": Mf.tolist(), **stats}
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    print(f"[write] {OUT_JSON}")

    rows = []
    for d, fr in per_ep:
        for cam in SIM_CAMERA:
            M = np.asarray(out["cameras"][cam]["M"], np.float32)
            for t, r, s in fr[cam][:3]:
                tiles = [r, apply_color(r, M), s]
                labs = [f"ZED raw {cam} t={t}", "ZED colour-corrected", "splat @ same joints"]
                tiles = [cv2.putText(np.ascontiguousarray(x.copy()), l, (6, 18),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                         for x, l in zip(tiles, labs)]
                rows.append(np.concatenate(tiles, 1))
    cv2.imwrite(OUT_FIG, np.concatenate(rows, 0)[..., ::-1])
    print(f"[write] {OUT_FIG}")


if __name__ == "__main__":
    main()

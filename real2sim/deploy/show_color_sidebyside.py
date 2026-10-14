"""Big, legible three-way panel: raw ZED | colour-corrected ZED | splat.

All three are the POLICY'S framing (undistort -> principal-point square crop ->
resize), so the only differences are colour and the sim/real content gap itself.
The splat column is rendered at the REAL robot's joint angles for that frame, so
the arm pose matches.

One file per camera: viz/real_vs_sim/color_sidebyside_{scene,wrist}.png
"""
import argparse
import glob
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import real2sim.splat_obs  # noqa: F401,E402  import-order fix for gs_mujoco
import cv2  # noqa: E402
from real2sim.deploy.preprocess import (peg_fovy, DEFAULT_COLOR_JSON, SIM_CAMERA, Rectifier,  # noqa: E402
                                        apply_color, decode_jpeg_rgb, load_color)
from compare_real_sim_demos import Real, SimReplay  # noqa: E402

OUT = os.path.join(_REPO, "viz", "real_vs_sim")


def band(img, text, h=34):
    """Caption strip above the tile, so no label sits on top of the picture."""
    strip = np.full((h, img.shape[1], 3), 245, np.uint8)
    cv2.putText(strip, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    return np.concatenate([strip, img], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", default="/scratch/yifanr/simdist/gello_software/demos/peg")
    ap.add_argument("--episode", default="episode_0001")
    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--frames", type=int, nargs="*", default=None,
                    help="timesteps; default = 4 spread over the episode")
    a = ap.parse_args()

    path = os.path.join(a.real_dir, f"{a.episode}.h5")
    if not os.path.exists(path):
        path = sorted(glob.glob(f"{a.real_dir}/*.h5"))[0]
    real = Real(path)
    S = a.tile
    rp = SimReplay(S)
    color = load_color(DEFAULT_COLOR_JSON)
    rects = {c: Rectifier(real.K[c], real.D[c], 1280, 720, S, peg_fovy()[SIM_CAMERA[c]]) for c in SIM_CAMERA}
    ts = a.frames or list(np.linspace(0, real.T - 1, 4).astype(int))
    print(f"[src] {os.path.basename(path)}  T={real.T}  frames={ts}  tile={S}", flush=True)

    for cam in SIM_CAMERA:
        rows = []
        for t in ts:
            rp.set(real.q[t], real.fingers[t])
            raw = rects[cam](decode_jpeg_rgb(real.imgs[cam][t]))
            fil = apply_color(raw, color.get(cam))
            spl = rp.render(cam)
            d_raw = float(np.abs(raw.astype(np.float32) - spl).mean())
            d_fil = float(np.abs(fil.astype(np.float32) - spl).mean())
            rows.append(np.concatenate([
                band(raw, f"t={t} ({t / 20:.1f}s)  RAW ZED   |diff vs splat| {d_raw:.0f}"),
                band(fil, f"FILTERED ZED (what the policy sees)   {d_fil:.0f}"),
                band(spl, "SPLAT rendered at the same joint angles")], 1))
            print(f"  {cam} t={t}: mean|raw-splat| {d_raw:.1f} -> filtered {d_fil:.1f} /255", flush=True)
        title = np.full((44, rows[0].shape[1], 3), 255, np.uint8)
        cv2.putText(title, f"{cam} camera  ({SIM_CAMERA[cam]} in sim)  -- {os.path.basename(path)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        out = np.concatenate([title] + rows, 0)
        p = os.path.join(OUT, f"color_sidebyside_{cam}.png")
        cv2.imwrite(p, out[..., ::-1])
        print(f"[write] {p}  {out.shape[1]}x{out.shape[0]}", flush=True)


if __name__ == "__main__":
    main()

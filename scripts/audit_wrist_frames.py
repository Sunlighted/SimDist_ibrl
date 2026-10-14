"""Audit the stored camera frames of a photoreal peg replay (CPU only, no replay).

Random frames across shards/trajectories/timesteps, per camera:
  - contact sheet (what the encoder actually trains on)
  - sharpness: variance of the Laplacian (blurry splat close-ups score low)
  - "board visible": fraction of saturated-blue pixels (the board is the only blue thing)
  - "nearly uniform": std of the frame (a frame that is mostly one colour carries no info)
and the same stats split by task phase (peg tip height from the stored state).

    python scripts/audit_wrist_frames.py --root /user_data/.../peg_d0_src_splat224_v2
"""
import argparse
import glob
import os

import cv2
import h5py
import imageio
import numpy as np


def decode(ds, i):
    if ds.attrs.get("codec", "raw") == "jpeg":
        return cv2.cvtColor(cv2.imdecode(np.frombuffer(ds[i].tobytes(), np.uint8), cv2.IMREAD_COLOR),
                            cv2.COLOR_BGR2RGB)
    x = np.asarray(ds[i])
    return x.transpose(1, 2, 0) if x.shape[0] == 3 else x


def stats(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    blue = ((hsv[..., 0] > 100) & (hsv[..., 0] < 135) & (hsv[..., 1] > 90)).mean()
    return cv2.Laplacian(g, cv2.CV_64F).var(), blue, g.std()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--n", type=int, default=400, help="frames per camera for the stats")
    p.add_argument("--sheet", type=int, default=24)
    p.add_argument("--out", default="viz/splat_preview/wrist_audit")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    files = sorted(glob.glob(os.path.join(a.root, "shard_*", "dataset.hdf5")))
    hs = [h5py.File(f, "r") for f in files]
    g0 = hs[0]["data"][next(iter(hs[0]["data"]))]
    cams = sorted(k[:-6] for k in g0["obs"].keys() if k.endswith("_image"))
    print("cameras:", cams, "| state dim", g0["obs/state"].shape[1])
    # peg tip height proxy from the stored state: object block starts at 9
    # (eef_pos3 + quat4 + gripper2); peg_pos is its first 3 entries. Phase by peg z.
    rows = {c: [] for c in cams}
    samples = []
    for _ in range(a.n):
        h = hs[rng.integers(len(hs))]
        keys = list(h["data"].keys())
        g = h["data"][keys[rng.integers(len(keys))]]
        t = int(rng.integers(g["actions"].shape[0]))
        peg_z = float(g["obs/state"][t][9 + 2])
        samples.append((g, t, peg_z))
    z_table = np.percentile([s[2] for s in samples], 10)
    for g, t, peg_z in samples:
        phase = "on table" if peg_z < z_table + 0.01 else "lifted"
        for c in cams:
            rows[c].append((phase,) + stats(decode(g[f"obs/{c}_image"], t)))
    for c in cams:
        r = rows[c]
        for ph in ("on table", "lifted", None):
            sel = [x for x in r if ph is None or x[0] == ph]
            if not sel:
                continue
            sh = np.array([x[1] for x in sel]); bl = np.array([x[2] for x in sel]); sd = np.array([x[3] for x in sel])
            print(f"{c:18s} {('all' if ph is None else ph):9s} n={len(sel):3d}  sharpness median {np.median(sh):7.1f} "
                  f"p10 {np.percentile(sh, 10):6.1f} | board visible (>2% blue) {np.mean(bl > 0.02):.2f} "
                  f"| near-uniform (std<15) {np.mean(sd < 15):.2f}")
    # contact sheets: same random (traj, t) for both cameras, labelled with t
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pick = [samples[i] for i in rng.choice(len(samples), a.sheet, replace=False)]
    for c in cams:
        tiles = []
        for g, t, _ in pick:
            im = decode(g[f"obs/{c}_image"], t).copy()
            cv2.putText(im, f"t={t}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
            tiles.append(im)
        grid = np.concatenate([np.concatenate(tiles[i:i + 6], 1) for i in range(0, len(tiles), 6)], 0)
        imageio.imwrite(f"{a.out}_{c}.jpg", grid)
        print("wrote", f"{a.out}_{c}.jpg")


if __name__ == "__main__":
    main()

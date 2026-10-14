"""End-to-end check of a photoreal peg dataset written by generate_simdist_data.py.

For a few stored frames: decode the JPEG from the hdf5, restore the episode's
model_state + that step's sim_state into a fresh env, and
  A) |stored - fresh composite|         ~ small (JPEG only)
  B) |stored - fresh composite flipped|  >> A   (else the data is upside down)
  C) board IoU: blue pixels of the STORED frame vs MuJoCo's board silhouette
Plus dataset stats: trajectories, lengths, success, bytes/transition.

    python scripts/check_peg_splat_dataset.py --dataset <dir>/dataset.hdf5
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import real2sim.splat_obs  # noqa: F401,E402
from real2sim.splat_obs import SplatCompositor, DEFAULT_CALIB, DEFAULT_SEG_DIR, DEFAULT_SCENE_PLY, DEFAULT_GROUND  # noqa: E402
import cv2  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import imageio  # noqa: E402
import robosuite  # noqa: E402
from robosuite import load_controller_config  # noqa: E402
import env.peg_insertion  # noqa: F401,E402
from env.sim2real import restore_model_state  # noqa: E402


def decode(ds, i):
    if ds.attrs.get("codec", "raw") == "jpeg":
        buf = np.frombuffer(ds[i].tobytes(), np.uint8)
        return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    x = np.asarray(ds[i])
    return x.transpose(1, 2, 0) if x.shape[0] == 3 else x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--eps", type=int, default=3)
    p.add_argument("--out", default="viz/splat_preview/dataset_check.png")
    a = p.parse_args()
    f = h5py.File(a.dataset, "r")
    keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    lens = np.array([f[f"data/{k}/actions"].shape[0] for k in keys])
    succ = np.array([np.asarray(f[f"data/{k}/rewards"]).max() >= 1 for k in keys])
    gb = os.path.getsize(a.dataset) / 1e9
    print(f"trajectories {len(keys)}  mean len {lens.mean():.1f}  success {succ.mean():.3f}  "
          f"transitions {lens.sum()}  {gb:.3f} GB -> {gb * 1e9 / lens.sum():.0f} B/transition")
    g0 = f[f"data/{keys[0]}"]
    print("per-episode keys:", sorted(g0.keys()), "obs:", sorted(g0["obs"].keys()))
    cams = [k[:-len("_image")] for k in g0["obs"].keys() if k.endswith("_image")]
    assert "sim_state" in g0 and "model_state" in g0, "no sim_state/model_state -> cannot replay"

    e = robosuite.make("PegInsertion_D0", robots="Panda",
                       controller_configs=load_controller_config(default_controller="OSC_POSE"),
                       has_offscreen_renderer=True, use_camera_obs=False)
    e.reset()
    sim = e.sim
    comp = SplatCompositor(sim.model, sim.data, calib_path=DEFAULT_CALIB, seg_dir=DEFAULT_SEG_DIR,
                           scene_ply=DEFAULT_SCENE_PLY, ground_json=DEFAULT_GROUND)
    bid = e.obj_body_id["board"]
    board_geoms = [g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == bid]
    rows, A, B, IOU, PER = [], [], [], [], {}
    for k in keys[:a.eps]:
        g = f[f"data/{k}"]
        skipped = restore_model_state(e, g["model_state"][()])
        assert not skipped, f"model_state fields not restored: {skipped}"
        T = g["sim_state"].shape[0]
        for i in (0, T // 2, T - 1):
            sim.set_state_from_flattened(g["sim_state"][i]); sim.forward()
            tiles = []
            for c in cams:
                ds = g[f"obs/{c}_image"]
                stored = decode(ds, i)
                S = stored.shape[0]
                fresh = comp.composite(sim, c, S, S)[::-1]                  # top-down
                a_ = np.abs(stored.astype(float) - fresh).mean()
                b_ = np.abs(stored.astype(float) - fresh[::-1]).mean()
                seg = sim.render(camera_name=c, width=S, height=S, segmentation=True)[::-1][:, :, 1]
                m = np.isin(seg, board_geoms).astype(np.uint8)
                hsv = cv2.cvtColor(stored, cv2.COLOR_RGB2HSV)
                blue = ((hsv[..., 0] > 100) & (hsv[..., 0] < 135) & (hsv[..., 1] > 90)).astype(np.uint8)
                iou = (blue & m).sum() / max((blue | m).sum(), 1)
                A.append(a_); B.append(b_); IOU.append(iou if m.sum() > 200 else np.nan)
                PER.setdefault(c, []).append((a_, b_, iou if m.sum() > 200 else np.nan))
                ed = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
                ov = stored.copy(); ov[ed] = (255, 0, 0)
                cv2.putText(ov, f"{k} t={i} {c.replace('robot0_', '')}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
                cv2.putText(ov, f"A={a_:.1f} B={b_:.1f} IoU={iou:.2f}", (3, S - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
                tiles.append(ov)
            rows.append(np.concatenate(tiles, 1))
    print(f"A |stored - fresh|        mean {np.mean(A):.2f}  max {np.max(A):.2f}")
    print(f"B |stored - fresh flipped| mean {np.mean(B):.2f}  min {np.min(B):.2f}   (must be >> A)")
    print(f"C board IoU (stored frame vs MuJoCo silhouette) mean {np.nanmean(IOU):.2f}  min {np.nanmin(IOU):.2f}")
    for c, v in PER.items():
        v = np.array(v, dtype=float)
        print(f"  [{c}] A mean {v[:, 0].mean():.2f} max {v[:, 0].max():.2f} | B min {v[:, 1].min():.2f} | "
              f"board IoU mean {np.nanmean(v[:, 2]) if np.isfinite(v[:, 2]).any() else float('nan'):.2f}")
    imageio.imwrite(a.out, np.concatenate(rows, 0))
    print("wrote", a.out)


if __name__ == "__main__":
    main()

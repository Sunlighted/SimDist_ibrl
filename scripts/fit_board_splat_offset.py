"""Find the board-splat translation that best overlays the MuJoCo board.

Renders ONLY the board splat (no scene) with MjSplatRenderer for a grid of
extra offsets (dx, dy) added to obj_calib.json's board t (board-body frame),
and scores IoU against the MuJoCo board visual-mesh silhouette (segmentation)
in a few cameras. Board jitter off. Prints the current-t IoU and the best
offset; writes an overlay at the best offset.

    python scripts/fit_board_splat_offset.py
"""
import copy
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import real2sim.splat_obs  # noqa: F401,E402
from real2sim.splat_obs import DEFAULT_CALIB, DEFAULT_SEG_DIR  # noqa: E402
from gs_mujoco import MjSplatRenderer  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import imageio  # noqa: E402
import robosuite  # noqa: E402
from robosuite import load_controller_config  # noqa: E402
import env.peg_insertion  # noqa: F401,E402

CAMS = ["zedview", "birdview", "frontview"]
S = 320


def board_mask_splat(r, sim, cam):
    img = r.render(sim.model, sim.data, [cam], S, S)[0]            # top-down, [0,1]
    bg = np.array(MjSplatRenderer.BG_RGB)[None, None]
    return (np.abs(img - bg).max(-1) > 0.12).astype(np.uint8), (img * 255).astype(np.uint8)


def main():
    e = robosuite.make("PegInsertion_D0", robots="Panda",
                       controller_configs=load_controller_config(default_controller="OSC_POSE"),
                       has_offscreen_renderer=True, use_camera_obs=False)
    e.reset()
    sim = e.sim
    # move the peg + arm out of the way of the board silhouette: we only need the board
    bid = e.obj_body_id["board"]
    board_geoms = [g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == bid]
    mj = {}
    for c in CAMS:
        seg = sim.render(camera_name=c, width=S, height=S, segmentation=True)[::-1][:, :, 1]
        mj[c] = np.isin(seg, board_geoms).astype(np.uint8)
        print(f"{c}: MuJoCo board pixels {mj[c].sum()}")
    cams = [c for c in CAMS if mj[c].sum() > 300]

    base = json.load(open(DEFAULT_CALIB))
    t0 = np.array(base["board_obj"]["t"])

    def score(dx, dy):
        cal = copy.deepcopy(base)
        cal["board_obj"]["t"] = (t0 + np.array([dx, dy, 0.0])).tolist()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(cal, f)
        r = MjSplatRenderer(sim.model, f.name, seg_dir=DEFAULT_SEG_DIR, overrides={"peg_obj": "skip"})
        ious, imgs = [], []
        for c in cams:
            m, img = board_mask_splat(r, sim, c)
            ious.append((m & mj[c]).sum() / max((m | mj[c]).sum(), 1))
            imgs.append(img)
        os.unlink(f.name)
        return float(np.mean(ious)), ious, imgs

    s0, per0, _ = score(0, 0)
    print(f"current t={t0.round(4).tolist()}: mean IoU {s0:.3f} per-cam {np.round(per0, 3).tolist()} ({cams})")
    best = (s0, 0.0, 0.0)
    grid = np.arange(-0.10, 0.1001, 0.01)
    for dx in grid:
        for dy in grid:
            s, _, _ = score(dx, dy)
            if s > best[0]:
                best = (s, dx, dy)
    # refine around the coarse best, 2 mm steps
    for dx in best[1] + np.arange(-0.01, 0.0101, 0.002):
        for dy in best[2] + np.arange(-0.01, 0.0101, 0.002):
            s, _, _ = score(dx, dy)
            if s > best[0]:
                best = (s, dx, dy)
    s, per, imgs = score(best[1], best[2])
    print(f"BEST extra offset dx={best[1]*1000:+.1f} mm dy={best[2]*1000:+.1f} mm -> "
          f"t={(t0 + [best[1], best[2], 0]).round(4).tolist()} mean IoU {s:.3f} per-cam {np.round(per, 3).tolist()}")
    # overlay at current and best
    _, _, imgs0 = score(0, 0)
    rows = []
    for tag, ims in (("current t", imgs0), ("best t", imgs)):
        tiles = []
        for c, im in zip(cams, ims):
            ed = cv2.morphologyEx(mj[c], cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
            o = im.copy(); o[ed] = (255, 0, 0)
            tiles.append(cv2.putText(o, f"{tag} {c}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1))
        rows.append(np.concatenate(tiles, 1))
    out = "viz/splat_preview/board_splat_offset_fit.png"
    imageio.imwrite(out, np.concatenate(rows, 0))
    print("wrote", out)


if __name__ == "__main__":
    main()

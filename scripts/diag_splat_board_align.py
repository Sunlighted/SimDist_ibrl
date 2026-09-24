"""Is the board SPLAT registered to the MuJoCo board? Outline overlay + pixel offset.

For a state with the peg inserted (scripted expert), renders the photoreal
composite and draws the MuJoCo board visual mesh's silhouette on it (red). If the
splat's holes do not sit inside the red outline's holes, the splat board is
offset and the policy sees the peg in the wrong hole. Also estimates the 2-D
offset between the splat board and the MuJoCo board masks (blue-ish pixels).

    python scripts/diag_splat_board_align.py --out viz/splat_preview
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import real2sim.splat_obs  # noqa: F401,E402
from real2sim.splat_obs import SplatCompositor, DEFAULT_CALIB, DEFAULT_SEG_DIR, DEFAULT_SCENE_PLY, DEFAULT_GROUND  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import imageio  # noqa: E402

from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from scripts.smoke_peg_insertion import scripted_expert_action  # noqa: E402

CAMS = ["zedview", "robot0_zedm_wrist"]


def run(board_jitter, seed, S, out):
    kw = dict(board_pos_noise=0.01, board_yaw_noise=5.0) if board_jitter else {}
    env = PixelRobosuite(
        env_name="PegInsertion_D0", robots=["Panda"], episode_length=300,
        use_state=True, device="cuda", image_size=S, rl_image_size=S,
        camera_names=CAMS, rl_cameras=CAMS, hand_init_perturb_joint_sigma=0.1,
        end_on_success=True, env_kwargs=kw or None)
    np.random.seed(seed)
    env.reset()
    raw, sim = env.env, env.env.sim
    ctl, succ = {}, False
    for _ in range(300):
        _, _, done, succ, _ = env.step(torch.from_numpy(scripted_expert_action(raw, ctl)))
        if succ or done:
            break
    comp = SplatCompositor(sim.model, sim.data, calib_path=DEFAULT_CALIB, seg_dir=DEFAULT_SEG_DIR,
                           scene_ply=DEFAULT_SCENE_PLY, ground_json=DEFAULT_GROUND)
    bid = raw.obj_body_id["board"]
    board_geoms = [g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == bid]
    tiles = []
    for cam in CAMS:
        splat = comp.composite(sim, cam, S, S)[::-1]                                  # top-down
        mj = sim.render(camera_name=cam, width=S, height=S)[::-1]
        seg = sim.render(camera_name=cam, width=S, height=S, segmentation=True)[::-1][:, :, 1]
        m = np.isin(seg, board_geoms).astype(np.uint8)
        edges = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        ov = splat.copy(); ov[edges] = (255, 0, 0)
        # splat board pixels ~ saturated blue; MuJoCo board mask from segmentation
        hsv = cv2.cvtColor(splat, cv2.COLOR_RGB2HSV)
        blue = ((hsv[..., 0] > 100) & (hsv[..., 0] < 135) & (hsv[..., 1] > 90)).astype(np.uint8)
        def centroid(x):
            ys, xs = np.nonzero(x)
            return np.array([xs.mean(), ys.mean()]) if len(xs) else np.array([np.nan, np.nan])
        d = centroid(blue) - centroid(m)
        iou = (blue & m).sum() / max((blue | m).sum(), 1)
        txt = f"{cam.replace('robot0_', '')} jitter={int(board_jitter)} dx,dy={d[0]:+.1f},{d[1]:+.1f}px IoU={iou:.2f}"
        print(txt)
        lab = lambda im, t: cv2.putText(im.copy(), t, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1, cv2.LINE_AA)
        tiles += [lab(ov, txt[:34]), lab(np.ascontiguousarray(mj), "MuJoCo only")]
    return np.concatenate(tiles, axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="viz/splat_preview")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--size", type=int, default=448)
    a = p.parse_args()
    rows = [run(False, a.seed, a.size, a.out), run(True, a.seed, a.size, a.out)]
    imageio.imwrite(os.path.join(a.out, "board_align_overlay.png"), np.concatenate(rows, axis=0))
    print("wrote", os.path.join(a.out, "board_align_overlay.png"))


if __name__ == "__main__":
    main()

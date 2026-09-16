"""THE GATE: render the Panda+Robotiq with PolaRiS splats through gsplat, beside
MuJoCo's own rasteriser at the identical camera pose, and score the overlap.

RMS from ICP cannot distinguish a correct alignment from a symmetric alias
(panda_link1's 95.6-degree "fit" bought only 10% over identity). Rendering can.
We emit side-by-side PNGs AND a quantitative silhouette IoU per configuration,
so the verdict does not rest on eyeballing.
"""
import os, sys, json, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gs_mujoco import MjSplatRenderer

OUT = "/scratch/yifanr/gsbench/slice"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--cam", default="agentview")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    import robosuite, imageio
    from robosuite.controllers import load_controller_config
    cc = load_controller_config(default_controller="OSC_POSE")
    env = robosuite.make(env_name="Lift", robots="Panda", gripper_types="Robotiq85Gripper",
                         controller_configs=cc, has_renderer=False,
                         has_offscreen_renderer=True, use_camera_obs=True,
                         camera_names=[a.cam], camera_heights=a.res, camera_widths=a.res,
                         horizon=50)
    env.reset()
    for _ in range(8):                       # move off the home pose
        env.step(np.array([0.3, 0.2, -0.2, 0, 0, 0, 0.0]))
    m, d = env.sim.model, env.sim.data

    ref = env.sim.render(camera_name=a.cam, width=a.res, height=a.res)[::-1]
    imageio.imwrite(f"{OUT}/ref_mujoco.png", ref)

    # MuJoCo renders the whole scene (table, cube, arena); the splat set is the
    # ROBOT only. Compare silhouettes on the robot region: take MuJoCo's own
    # segmentation of robot/gripper geoms as ground truth.
    seg = env.sim.render(camera_name=a.cam, width=a.res, height=a.res,
                         segmentation=True)[::-1][:, :, 1]
    robot_gids = {i for i in range(m.ngeom)
                  if (m.body_id2name(m.geom_bodyid[i]) or "").startswith(("robot0", "gripper0"))}
    ref_mask = np.isin(seg, list(robot_gids))
    imageio.imwrite(f"{OUT}/ref_robotmask.png", (ref_mask * 255).astype(np.uint8))

    CONFIGS = {
        "all_icp":        {},
        "link1_identity": {"panda_link1": "identity"},
        "no_link0":       {"panda_link0": "skip"},
        "link1id_nolink0": {"panda_link1": "identity", "panda_link0": "skip"},
        "all_identity":   {k: "identity" for k in json.load(
                            open("/scratch/yifanr/gsbench/splat_frame_calib.json"))},
    }
    results = {}
    for name, ov in CONFIGS.items():
        r = MjSplatRenderer(m, "/scratch/yifanr/gsbench/splat_frame_calib.json", overrides=ov)
        img = r.render(m, d, [a.cam], a.res, a.res)[0]
        img8 = (img * 255).astype(np.uint8)
        imageio.imwrite(f"{OUT}/splat_{name}.png", img8)
        # silhouette: gsplat renders on black; anything non-black is robot
        m_splat = img.max(-1) > 0.02
        inter = np.logical_and(m_splat, ref_mask).sum()
        union = np.logical_or(m_splat, ref_mask).sum()
        iou = float(inter / max(union, 1))
        results[name] = {"iou": round(iou, 4),
                         "splat_px": int(m_splat.sum()), "ref_px": int(ref_mask.sum()),
                         "n_gauss": r.n_gauss, "n_parts": r.n_parts}
        print(f"[slice] {name:<18s} IoU={iou:.4f}  splat_px={m_splat.sum():7d} "
              f"ref_px={ref_mask.sum():7d}  gauss={r.n_gauss}")
        side = np.concatenate([ref, img8], 1)
        imageio.imwrite(f"{OUT}/side_{name}.png", side)

    # PER-LINK diagnostic: a whole-robot IoU hides which link is wrong, and a
    # link contributing zero pixels in this view is NOT evidence of anything
    # (agentview tested only the gripper on the first pass).
    import json as _j
    calib = _j.load(open("/scratch/yifanr/gsbench/splat_frame_calib.json"))
    print(f"[link] {'token':<42s} {'px':>7s} {'IoU':>7s}")
    per = {}
    for tok in calib:
        if tok == "panda_link0":
            continue
        ov = {k: "skip" for k in calib if k != tok}
        rr = MjSplatRenderer(m, "/scratch/yifanr/gsbench/splat_frame_calib.json", overrides=ov)
        im = rr.render(m, d, [a.cam], a.res, a.res)[0]
        ms = im.max(-1) > 0.02
        bid = m.body_name2id(calib[tok]["mj_body"])
        gset = [i for i in range(m.ngeom) if m.geom_bodyid[i] == bid]
        rmask = np.isin(seg, gset)
        i2 = np.logical_and(ms, rmask).sum(); u2 = np.logical_or(ms, rmask).sum()
        iou2 = float(i2 / max(u2, 1))
        per[tok] = {"splat_px": int(ms.sum()), "ref_px": int(rmask.sum()), "iou": round(iou2, 4)}
        print(f"[link] {tok:<42s} {int(ms.sum()):7d} {iou2:7.4f}  ref_px={int(rmask.sum())}")
    results["_per_link"] = per
    json.dump(results, open(f"{OUT}/slice_results.json", "w"), indent=2)
    print("[slice] RESULT " + json.dumps(results))
    env.close()


if __name__ == "__main__":
    main()

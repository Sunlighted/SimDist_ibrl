"""Gate the PHOTOREAL obs path: right content, right way up.

The generator's hdf5 stores no sim_state, so this does not replay it. Instead it
drives the REAL wrapper (PixelRobosuite, splat_obs set) and compares what the
wrapper hands downstream against a composite built independently here, using the
recipe whose output we already verified by eye in render_object_splats.py.

Two numbers decide it, with the vertical flip as the control:
  A) |wrapper - expected|        should be ~0
  B) |wrapper - expected[::-1]|  should be MUCH larger
If A >= B the dataset is vertically mirrored -- which trains perfectly well and
cannot transfer, so it must never reach a real run.
  C) |wrapper - mujoco_only|     should be LARGE, else the splat never happened.

Orientation bookkeeping: robosuite obs and sim.render() are bottom-up, the
wrapper flips to top-down, and MjSplatRenderer returns top-down already.
"""
import os, sys, argparse, copy
os.environ.setdefault("MUJOCO_GL", "egl")
for p in ("deps/mimicgen", "deps/robomimic", "."):
    sys.path.insert(0, os.path.join("/project/6101830/yifanr/SimDist_ibrl", p))
sys.path.insert(0, "/scratch/yifanr/realsim_survey/_GaussianRenderer/src")
sys.path.insert(0, "/scratch/yifanr/gsbench")
import numpy as np, cv2
import env.peg_insertion  # noqa
from env.robosuite_wrapper import PixelRobosuite

SEG = "/scratch/yifanr/gsbench/objsplats"
SCENE = ("/project/6101830/yifanr/SimDist_ibrl/real2sim/splats/"
         "scene_scan5_norobot_v2.ply")
GROUND = "/scratch/yifanr/gsbench/ground_transforms.json"
OUT = "/scratch/yifanr/gsbench/objsplat_shots"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="zedview")
    ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args()
    cfg = dict(calib_path=f"{SEG}/obj_calib.json", seg_dir=SEG,
               scene_ply=SCENE, ground_json=GROUND)
    e = PixelRobosuite(env_name="PegInsertion_D0", robots="Panda", episode_length=40,
                       camera_names=[a.cam], rl_cameras=[a.cam],
                       image_size=224, rl_image_size=224, use_state=False,
                       device="cpu", splat_obs=cfg)
    import torch
    e.reset()
    rows, A, B, C = [], [], [], []
    for k in range(a.steps):
        e.step(torch.zeros(e.action_dim))
        sim = e.env.sim
        raw = e.env._get_observations() if hasattr(e.env, "_get_observations") \
            else e.env.observation_spec()
        got = e._extract_images(copy.deepcopy(raw))[0][a.cam]      # wrapper output
        got = got.numpy().transpose(1, 2, 0).astype(np.float32)    # CHW -> HWC, top-down

        # expected, built the verified way: mujoco bottom-up -> flip; splat already top-down
        mj = sim.render(camera_name=a.cam, width=224, height=224)[::-1].astype(np.float32)
        sp = (e._splat.r.render(sim.model, sim.data, [a.cam], 224, 224)[0] * 255)
        seg = sim.render(camera_name=a.cam, width=224, height=224,
                         segmentation=True)[::-1][:, :, 1]
        keep = np.isin(seg, e._splat.mj_geoms)[..., None]
        exp = np.where(keep, mj, sp).astype(np.float32)

        A.append(np.abs(got - exp).mean())
        B.append(np.abs(got - exp[::-1]).mean())
        C.append(np.abs(got - mj).mean())
        print(f"  step {k}: |wrapper-expected| {A[-1]:6.2f}   "
              f"|wrapper-expectedFLIP| {B[-1]:6.2f}   |wrapper-mujocoOnly| {C[-1]:6.2f}")
        rows.append(np.concatenate([got, exp, mj], 1).astype(np.uint8))

    A, B, C = np.mean(A), np.mean(B), np.mean(C)
    ok_orient = A < 0.5 * B
    ok_splat = C > 10.0
    print(f"\n  mean A={A:.2f} B={B:.2f} C={C:.2f}")
    print(f"  orientation correct : {'PASS' if ok_orient else 'FAIL -- MIRRORED'}")
    print(f"  splat actually used : {'PASS' if ok_splat else 'FAIL -- looks like plain MuJoCo'}")
    os.makedirs(OUT, exist_ok=True)
    cv2.imwrite(f"{OUT}/check_splat_obs_{a.cam}.png",
                np.concatenate(rows, 0)[:, :, ::-1])
    print(f"  wrote {OUT}/check_splat_obs_{a.cam}.png  (wrapper | expected | mujoco-only)")
    sys.exit(0 if (ok_orient and ok_splat) else 1)


if __name__ == "__main__":
    main()

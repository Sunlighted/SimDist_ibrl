"""Gate C for the scene camera: does the calibrated agentview reproduce the real photo?

For each recorded sample (real scene photo + FR3 joints), the sim is put in the same joint
configuration, rendered from the calibrated agentview, and laid over the real photo after the
real-side preprocessing (undistort -> square crop about the principal point -> resize). The
sim table outline (green) and robot outline (red) are drawn on the real image: if the camera
pose, the fovy and the arena are right, they sit on the real table edge and arm.

Expected mismatch: the sim hand is the Panda gripper, the real one is a Robotiq 2F.

  MUJOCO_GL=egl ~/anaconda3/envs/simdist/bin/python real2sim/zed_capture/calib_20261010/tools/overlay_real_scene.py \\
      --config real2sim/zed_capture/calib_20261010/threading_real_cell_20261010.yaml \\
      --samples real2sim/zed_capture/calib_20261010/data/run_1010_2327
"""
import argparse
import dataclasses
import glob
import json
import os
import sys
import types

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# this file lives in real2sim/zed_capture/calib_20261010/tools/ -> repo root is 4 levels up
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
sys.path.insert(0, ROOT)
# common_utils imports the compiled replay buffer (rela), which rendering never touches;
# stub it so this runs on a workstation without the C++ build.
try:
    import common_utils.build  # noqa: F401
except ImportError:
    stub = types.ModuleType("common_utils.build")
    stub.rela = types.ModuleType("rela")
    sys.modules["common_utils.build"] = stub
import torch
from env.robosuite_wrapper import PixelRobosuite
from env.sim2real import Sim2RealConfig, load_sim2real


def real_view(img, K, D, fovy, res):
    """Real-side preprocessing, same as deploy: undistort (newK = K), crop, resize."""
    und = cv2.undistort(img, K, D, None, K)
    fy, cx, cy = K[1, 1], K[0, 2], K[1, 2]
    side = 2 * fy * np.tan(np.radians(fovy) / 2)
    side = min(side, 2 * min(cx, cy, img.shape[1] - cx, img.shape[0] - cy) + 1)
    x0, y0 = cx - side / 2, cy - side / 2
    # sub-pixel crop via an affine warp, so the crop centre is exactly the principal point
    s = res / side
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]])
    return cv2.warpAffine(und, M, (res, res), flags=cv2.INTER_AREA)


def project_table_top(sim, m, gid, res):
    """Edges of the table box's top face, projected with the sim camera itself.
    Geometric on purpose: a segmentation outline gets cut by anything in front of the table
    (the arm, and an invisible-but-shadow-casting wall_front_visual)."""
    d = sim.data
    cid = m.camera_name2id("agentview")
    cp, R = d.cam_xpos[cid], d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    f = (res / 2) / np.tan(np.radians(float(m.cam_fovy[cid])) / 2)
    gp, gR, sz = d.geom_xpos[gid], d.geom_xmat[gid].reshape(3, 3), m.geom_size[gid]
    c = [gp + gR @ np.array([sx * sz[0], sy * sz[1], sz[2]])
         for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    edges = []
    for a, b in zip(c, c[1:] + c[:1]):
        P = np.linspace(a, b, 400)
        pc = (P - cp) @ R
        ok = pc[:, 2] > 1e-3
        uv = np.c_[f * pc[ok, 0] / pc[ok, 2] + res / 2, f * pc[ok, 1] / pc[ok, 2] + res / 2]
        edges.append(uv)
    return edges, np.mean(c, axis=0)


def outline(mask, img, color):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", default="Threading_RealCell")
    ap.add_argument("--samples", required=True,
                    help="dir with sample_*/{scene_left.png,state.json} and extrinsics.json")
    ap.add_argument("--extrinsics", default="",
                    help="calibration json (default <samples>/extrinsics.json); its T_base_scene "
                         "replaces the config's agentview pose")
    ap.add_argument("--keep_config_pose", action="store_true",
                    help="render with the config's agentview pose instead")
    ap.add_argument("--fovy", type=float, default=None, help="default: the config's agentview fovy")
    ap.add_argument("--res", type=int, default=448)
    ap.add_argument("--only", default="", help="comma-separated sample names")
    ap.add_argument("--out", default="figs/sim2real/overlay_scene")
    args = ap.parse_args()

    E = json.load(open(args.extrinsics or os.path.join(args.samples, "extrinsics.json")))
    K, D = np.array(E["scene_K"]), np.array(E["scene_D"])

    cfg = load_sim2real(args.config)
    for f in dataclasses.fields(Sim2RealConfig):   # nominal scene: every DR inert
        if f.name.startswith("dr_"):
            setattr(cfg, f.name, f.default)
    cam = next(c for c in cfg.cameras if c.name == "agentview")
    if not args.keep_config_pose:
        Tbs = np.array(E["T_base_scene"])
        x, y, z, w = Rotation.from_matrix(Tbs[:3, :3]).as_quat()
        cam.pos, cam.quat, cam.look_dir, cam.look_at = tuple(Tbs[:3, 3]), (w, x, y, z), None, None
    if args.fovy is not None:
        cam.fovy, cam.fy = args.fovy, 0.0
    fovy = cam.resolved_fovy()
    # The real crop cannot exceed the largest square about the principal point; render the
    # sim with the fovy the real crop actually has, or the two images have different scales.
    cx, cy = K[0, 2], K[1, 2]
    max_side = 2 * min(cx, cy, 1280 - cx, 720 - cy) + 1
    max_fovy = float(np.degrees(2 * np.arctan(max_side / 2 / K[1, 1])))
    if fovy > max_fovy:
        print(f"fovy {fovy} exceeds the real image ({max_fovy:.3f}); using {max_fovy:.3f}")
        fovy, cam.fovy, cam.fy = max_fovy, max_fovy, 0.0
    print(f"agentview pos {np.round(cam.pos, 4).tolist()} quat {np.round(cam.quat, 5).tolist()} "
          f"fovy {fovy}")

    env = PixelRobosuite(
        args.task, "Panda", 400, image_size=args.res, rl_image_size=args.res,
        camera_names=["agentview"], rl_cameras=["agentview"], use_state=False, obs_stack=1,
        state_stack=1, prop_stack=1, flip_image=True, device="cpu",
        hand_init_perturb_joint_sigma=0.0, sim2real=cfg)
    np.random.seed(0)
    env.reset()
    env.step(torch.zeros(env.action_dim))
    rs = env.env
    sim, m = rs.sim, rs.sim.model
    names = [m.geom_id2name(g) or "" for g in range(m.ngeom)]
    table_gid = names.index("table_visual")
    print(f"table_visual: half size {np.round(m.geom_size[table_gid][:2], 4).tolist()} m, "
          f"top z {sim.data.geom_xpos[table_gid][2] + m.geom_size[table_gid][2]:.4f} (world)")
    # robot visual meshes are unnamed: select them by body
    robot_ids = [g for g in range(m.ngeom)
                 if (m.body_id2name(int(m.geom_bodyid[g])) or "").startswith(("robot0_", "gripper0_"))]
    jidx = rs.robots[0]._ref_joint_pos_indexes

    os.makedirs(args.out, exist_ok=True)
    only = set(filter(None, args.only.split(",")))
    for d in sorted(glob.glob(os.path.join(args.samples, "sample_*"))):
        name = os.path.basename(d)
        if only and name not in only:
            continue
        q = np.array(json.load(open(os.path.join(d, "state.json")))["q"])
        sim.data.qpos[jidx] = q
        sim.data.qvel[:] = 0
        sim.forward()
        rgb = sim.render(width=args.res, height=args.res, camera_name="agentview")[::-1]
        seg = sim.render(width=args.res, height=args.res, camera_name="agentview",
                         segmentation=True)[::-1][..., 1]
        sim_bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        real = real_view(cv2.imread(os.path.join(d, "scene_left.png")), K, D, fovy, args.res)

        marked = real.copy()
        edges, _ = project_table_top(sim, m, table_gid, args.res)
        for uv in edges:
            cv2.polylines(marked, [np.round(uv).astype(np.int32).reshape(-1, 1, 2)], False,
                          (0, 255, 0), 1, cv2.LINE_AA)
        outline(np.isin(seg, robot_ids), marked, (0, 0, 255))
        blend = cv2.addWeighted(real, 0.5, sim_bgr, 0.5, 0)
        for im, t in ((marked, "real + sim outlines"), (sim_bgr, "sim"), (blend, "50/50")):
            cv2.putText(im, t, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(args.out, f"{name}.jpg"), np.hstack([marked, sim_bgr, blend]),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"wrote {args.out}/{name}.jpg")


if __name__ == "__main__":
    main()

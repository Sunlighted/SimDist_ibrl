"""PegInsertion_D0 rendered on the REAL scanned cell (robot-free scene splat).

Layering:
  * SCENE  = the merged, robot-free splat, trained directly in the metric board
             frame, so its SH coefficients are already expressed in that frame.
  * ARM    = PolaRiS DROID per-link splats, posed by the sim's body poses.
  * OBJECTS + PANDA HAND = MuJoCo raster pixels, composited by segmentation
             (the DROID splat set has no Panda hand -- see real2sim.md 6).

Everything is rendered in the BOARD frame, and the SIM CAMERAS are transformed
into it. That direction is deliberate: moving the scene gaussians instead would
require rotating their spherical harmonics, and rotating only xyz+quats (what
the first version of this script did) silently leaves every view-dependent term
in the old frame. The arm is unaffected because its gaussians are already being
re-posed per body each frame.

Sim->board placement is measured, not guessed: RANSAC the tabletop plane out of
the scene splat, put the sim tabletop (z = 0.8) on it, and align the sim origin
with the real table's centroid. This is a VISUALISATION convention -- the
load-bearing grounding (robot base <-> board) is still open, see real2sim.md 2d.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/project/6101830/yifanr/SimDist_ibrl/deps/mimicgen")
sys.path.insert(0, "/project/6101830/yifanr/SimDist_ibrl/deps/robomimic")
sys.path.insert(0, "/project/6101830/yifanr/SimDist_ibrl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import imageio
import numpy as np
import torch

import env.peg_insertion  # noqa: F401  registers PegInsertion_D0
import robosuite
from robosuite.controllers import load_controller_config

from eval_splat import load_ply_raw
from gs_mujoco import MjSplatRenderer
from gaussian_renderer.core.gaussiandata import GaussianData
from gaussian_renderer.core.batch_rasterization import (
    batch_update_gaussians, batch_env_render,
)

GS = "/scratch/yifanr/gsbench"


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def mat2quat_wxyz(R):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])


def fit_table_plane(P, near_xy, radius=0.7, tol=0.004, iters=6000, seed=0):
    """RANSAC the dominant near-horizontal plane within `radius` of `near_xy`."""
    d = np.linalg.norm(P[:, :2] - np.asarray(near_xy), axis=1)
    cand = P[d < radius]
    assert len(cand) > 500, f"only {len(cand)} points near {near_xy}"
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        i = rng.choice(len(cand), 3, replace=False)
        a, b, c = cand[i]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        if abs(n[2]) < 0.8:          # want a roughly horizontal surface
            continue
        if n[2] < 0:
            n = -n
        inl = np.abs((cand - a) @ n) < tol
        if best is None or inl.sum() > best[0]:
            best = (int(inl.sum()), n, cand[inl])
    n_in, n, inliers = best
    # refit the normal on the inlier set
    c = inliers.mean(0)
    u, s, vt = np.linalg.svd(inliers - c)
    n = vt[2] / np.linalg.norm(vt[2])
    if n[2] < 0:
        n = -n
    print(f"[plane] {n_in} inliers of {len(cand)} near points, normal {n.round(4)}, "
          f"centroid {c.round(4)}")
    return n, c, inliers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, help="scene splat, BOARD frame (metric)")
    ap.add_argument("--out", default=f"{GS}/peg_scene_shots")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--bg_color_json", default=f"{GS}/wall_color.json")
    ap.add_argument("--calib", default=f"{GS}/splat_frame_calib.json")
    ap.add_argument("--board_origin_xy", default="0.056,0.242",
                    help="a point on the ChArUco in board frame, to seed the plane fit")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    np.random.seed(3)
    dev = torch.device("cuda")
    bg = json.load(open(a.bg_color_json))["wall_rgb"]

    # ---- scene splat (board frame) ----
    xyz, sh0, rest, opa, sc, qt = load_ply_raw(a.ply, "channel")
    print(f"[scene] {len(xyz):,} gaussians from {a.ply}")
    n_up, c_tab, inliers = fit_table_plane(
        xyz, [float(x) for x in a.board_origin_xy.split(",")])

    # in-plane axes: board x projected onto the plane (deterministic)
    x_ax = np.array([1.0, 0.0, 0.0]) - n_up * (n_up @ np.array([1.0, 0.0, 0.0]))
    x_ax /= np.linalg.norm(x_ax)
    y_ax = np.cross(n_up, x_ax)
    R_bs = np.stack([x_ax, y_ax, n_up], axis=1)      # board <- sim rotation
    TABLE_Z = 0.8
    p0 = c_tab                                        # sim (0,0,0.8) lands here
    q_bs = mat2quat_wxyz(R_bs)

    def sim2board_pos(p):
        return (R_bs @ (np.asarray(p) - np.array([0.0, 0.0, TABLE_Z])).T).T + p0

    # ---- sim ----
    cc = load_controller_config(default_controller="OSC_POSE")
    e = robosuite.make(
        "PegInsertion_D0", robots="Panda", controller_configs=cc,
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
        camera_names=["frontview", "agentview"], camera_heights=a.res,
        camera_widths=a.res, horizon=120,
    )
    e.reset()
    m, d = e.sim.model, e.sim.data

    calib = json.load(open(a.calib))
    overrides = {"panda_link0": "skip"}
    for tok, c in calib.items():
        try:
            m.body_name2id(c["mj_body"])
        except Exception:
            overrides[tok] = "skip"
    arm = MjSplatRenderer(m, a.calib, overrides=overrides)
    print(f"[arm] {arm.n_gauss:,} gaussians, {arm.n_parts} parts")

    # ---- one combined template: arm (dynamic bodies) + scene (identity body) ----
    T = lambda x: torch.tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)
    at = arm.template
    sh_arm = at.sh.reshape(at.xyz.shape[0], -1)
    sh_scene = T(np.concatenate([sh0[:, None, :], rest], 1).reshape(len(xyz), -1))
    w = max(sh_arm.shape[1], sh_scene.shape[1])
    pad = lambda x: (torch.cat([x, torch.zeros(x.shape[0], w - x.shape[1], device=dev)], 1)
                     if x.shape[1] < w else x)
    ident_body = m.nbody                      # virtual body, pose forced to identity
    template = GaussianData(
        xyz=torch.cat([at.xyz, T(xyz)]),
        rot=torch.cat([at.rot, T(qt)]),
        scale=torch.cat([at.scale, T(sc)]),
        opacity=torch.cat([at.opacity, T(opa)]),
        sh=torch.cat([pad(sh_arm), pad(sh_scene)]).reshape(-1, w // 3, 3).contiguous(),
    )
    p2b = torch.cat([arm.p2b,
                     torch.full((len(xyz),), ident_body, dtype=torch.long, device=dev)])
    dyn = torch.ones(len(template.xyz), dtype=torch.bool, device=dev)

    keep_gids = [g for g in range(m.ngeom)
                 if (m.geom_id2name(g) or "").startswith(("board_obj", "peg_obj", "gripper0"))]

    def shoot(tag):
        # body poses -> board frame, plus the identity body for the static scene
        bp = np.concatenate([sim2board_pos(np.asarray(d.xpos)), np.zeros((1, 3))])
        bq = np.concatenate([
            np.stack([quat_mul(q_bs, q) for q in np.asarray(d.xquat)]),
            np.array([[1.0, 0.0, 0.0, 0.0]]),
        ])
        gsb = batch_update_gaussians(template, T(bp)[None], T(bq)[None],
                                     point_to_body_idx=p2b, dynamic_mask=dyn)
        for cam in ("frontview", "agentview"):
            cid = m.camera_name2id(cam)
            cp = T(sim2board_pos(np.asarray(d.cam_xpos)[cid][None]))[None]
            cx = T((R_bs @ np.asarray(d.cam_xmat)[cid].reshape(3, 3)).reshape(1, 9))[None]
            fovy = np.asarray(m.cam_fovy)[cid][None, None].astype(np.float32)
            bgimg = torch.tensor(bg, dtype=torch.float32, device=dev).expand(
                1, 1, a.res, a.res, 3).contiguous()
            rgb, _ = batch_env_render(gsb, cp, cx, a.res, a.res, fovy, bg_imgs=bgimg)
            splat = (rgb[0, 0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

            ref = e.sim.render(camera_name=cam, width=a.res, height=a.res)[::-1]
            seg = e.sim.render(camera_name=cam, width=a.res, height=a.res,
                               segmentation=True)[::-1][:, :, 1]
            keep = np.isin(seg, keep_gids)[..., None]
            comp = np.where(keep, ref, splat)
            imageio.imwrite(f"{a.out}/{tag}_{cam}_splat.png", splat)
            imageio.imwrite(f"{a.out}/{tag}_{cam}_mujoco.png", ref)
            imageio.imwrite(f"{a.out}/{tag}_{cam}_composite.png", comp)
            imageio.imwrite(f"{a.out}/{tag}_{cam}_strip.png",
                            np.concatenate([ref, splat, comp], 1))
        print(f"[shot] {tag}")

    shoot("reset")
    for _ in range(30):
        e.step(np.array([0.25, 0.35, -0.25, 0, 0, 0, 0.0]))
    shoot("posed")
    print(f"[done] {sorted(os.listdir(a.out))}")
    e.close()


if __name__ == "__main__":
    main()

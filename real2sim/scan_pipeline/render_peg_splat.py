"""Render PegInsertion_D0 through the gsplat path (real2sim Phase D demo).

Layers:
  * SCENE: the scan4 scene splat (scene.ply), placed in the sim world via the
    ChArUco similarity transform solved by calibrate_scene_scale.py (metric
    scale from ZED depth; board plane -> sim tabletop z=0.8).
  * ROBOT ARM: PolaRiS DROID per-link splats posed by the env's body poses.
    The cell will run the PANDA hand (owner decision 2026-09-15) and the DROID
    splat set only covers the Robotiq, so gripper tokens are skipped and the
    hand comes from the MuJoCo raster (a Panda-hand splat is a future scan).
  * OBJECTS (board/peg) + HAND: MuJoCo raster pixels, overlaid by segmentation.

Known gaps: peg/board appearance is untextured; the sim table is REPLACED by
the splat table (they coincide at z=0.8 by construction).
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/project/6101830/yifanr/SimDist_ibrl/deps/mimicgen")
sys.path.insert(0, "/project/6101830/yifanr/SimDist_ibrl/deps/robomimic")
sys.path.insert(0, "/project/6101830/yifanr/SimDist_ibrl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import torch
import imageio
from scipy.spatial.transform import Rotation

import env.peg_insertion  # noqa: F401  registers PegInsertion_D0
import robosuite
from robosuite.controllers import load_controller_config
from gs_mujoco import MjSplatRenderer
from gaussian_renderer.core.util_gau import load_ply

GS = "/scratch/yifanr/gsbench"
OUT = f"{GS}/peg_splat_shots"
RES = 512
CALIB = f"{GS}/splat_frame_calib.json"
SCENE_PLY = f"{GS}/sfm_scan4/scene.ply"
WORLD_TF = f"{GS}/scene_world_transform.json"


def load_scene_world(device):
    """scene.ply -> world frame via the ChArUco similarity transform."""
    tf = json.load(open(WORLD_TF))
    s, R, t = float(tf["scale"]), np.array(tf["R"]), np.array(tf["t"])
    g = load_ply(SCENE_PLY)
    xyz = (s * (g.xyz @ R.T) + t).astype(np.float32)
    qR = Rotation.from_matrix(R)
    q = g.rot[:, [1, 2, 3, 0]]  # wxyz -> xyzw
    q = (qR * Rotation.from_quat(q)).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
    scale = (g.scale * s).astype(np.float32)
    # NOTE: SH view-dependent coefficients are NOT rotated (DC dominates; this
    # is a viz path, not a training path -- fix before stage-1 data gen).
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    return T(xyz), T(q), T(scale), T(g.opacity), T(g.sh)


def append_static(renderer, xyz, rot, scale, opa, sh):
    """Concatenate world-frame gaussians onto the renderer template, mapped to
    body 0 (worldbody: identity transform), padding SH widths to match."""
    from gaussian_renderer.core.gaussiandata import GaussianData
    t = renderer.template
    sh_t = t.sh.reshape(t.xyz.shape[0], -1)
    sh_s = sh.reshape(xyz.shape[0], -1)
    w = max(sh_t.shape[1], sh_s.shape[1])
    pad = lambda x: torch.cat(
        [x, torch.zeros(x.shape[0], w - x.shape[1], device=x.device)], 1
    ) if x.shape[1] < w else x
    renderer.template = GaussianData(
        xyz=torch.cat([t.xyz, xyz]), rot=torch.cat([t.rot, rot]),
        scale=torch.cat([t.scale, scale]), opacity=torch.cat([t.opacity, opa]),
        sh=torch.cat([pad(sh_t), pad(sh_s)]).reshape(-1, w // 3, 3).contiguous(),
    )
    n = xyz.shape[0]
    renderer.p2b = torch.cat(
        [renderer.p2b, torch.zeros(n, dtype=torch.long, device=renderer.device)]
    )
    renderer.dyn_mask = torch.cat(
        [renderer.dyn_mask, torch.ones(n, dtype=torch.bool, device=renderer.device)]
    )
    renderer.n_gauss += n


def main():
    os.makedirs(OUT, exist_ok=True)
    np.random.seed(3)
    cc = load_controller_config(default_controller="OSC_POSE")
    e = robosuite.make(
        "PegInsertion_D0", robots="Panda",
        controller_configs=cc, has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_names=["frontview", "agentview"],
        camera_heights=RES, camera_widths=RES, horizon=80,
    )
    e.reset()
    m = e.sim.model

    # skip calib tokens whose mj body does not exist in this env (Robotiq-only
    # pieces under the Panda hand) + panda_link0 per real2sim.md
    calib = json.load(open(CALIB))
    overrides = {"panda_link0": "skip"}
    for tok, c in calib.items():
        try:
            m.body_name2id(c["mj_body"])
        except Exception:
            overrides[tok] = "skip"
    print(f"[peg-splat] skipping tokens: {sorted(overrides)}")

    r = MjSplatRenderer(m, CALIB, overrides=overrides)
    n_arm = r.n_gauss
    r_arm_only = MjSplatRenderer(m, CALIB, overrides=overrides)
    append_static(r, *load_scene_world(r.device))
    print(f"[peg-splat] {n_arm} arm + {r.n_gauss - n_arm} scene gaussians")

    # geoms whose pixels come from the MuJoCo raster (objects + Panda hand)
    keep_prefix = ("board_obj", "peg_obj", "gripper0")
    keep_gids = [g for g in range(m.ngeom)
                 if (m.geom_id2name(g) or "").startswith(keep_prefix)]

    def shoot(tag):
        for cam in ("frontview", "agentview"):
            ref = e.sim.render(camera_name=cam, width=RES, height=RES)[::-1]
            seg = e.sim.render(camera_name=cam, width=RES, height=RES,
                               segmentation=True)[::-1][:, :, 1]
            keep = np.isin(seg, keep_gids)[..., None]
            splat = r.render(m, e.sim.data, [cam], RES, RES)[0]
            splat8 = (splat * 255).astype(np.uint8)
            arm = r_arm_only.render(m, e.sim.data, [cam], RES, RES)[0]
            arm8 = (arm * 255).astype(np.uint8)
            comp = np.where(keep, ref, splat8)
            imageio.imwrite(f"{OUT}/{tag}_{cam}_mujoco.png", ref)
            imageio.imwrite(f"{OUT}/{tag}_{cam}_splat.png", splat8)
            imageio.imwrite(f"{OUT}/{tag}_{cam}_armsplat.png", arm8)
            imageio.imwrite(f"{OUT}/{tag}_{cam}_composite.png", comp)
            side = np.concatenate([ref, splat8, comp], 1)
            imageio.imwrite(f"{OUT}/{tag}_{cam}_side.png", side)
        print(f"[peg-splat] saved {tag}")

    shoot("reset")
    for _ in range(25):
        e.step(np.array([0.25, 0.35, -0.25, 0, 0, 0, 0.0]))
    shoot("posed")
    print("[peg-splat] done:", sorted(os.listdir(OUT)))
    e.close()


if __name__ == "__main__":
    main()

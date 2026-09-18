"""Render the splatted PegInsertion env from the REAL cameras' calibrated POVs.

FRAME: everything is expressed in the SIM world, anchored by the ROBOT BASE.
The grounding gives the real base pose in the board frame (T_bw_base) and the
sim knows its own base pose (T_sw_base), so
    T_sw_bw = T_sw_base . inv(T_bw_base)
maps every real-world quantity (scene splat, camera poses) into sim world. The
sim arm/board/peg then need no transform at all, and the arm splat -- posed by
the sim's body poses -- lands where the real arm is.

Per camera: [real capture | mujoco | splat | composite] at 1280x720, using the
camera's TRUE fovy computed from its native sensor height (FOV is angular: a
fovy derived from the render height silently changes the field of view).
"""
import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/project/6101830/yifanr/SimDist_ibrl/deps/mimicgen")
sys.path.insert(0, "/project/6101830/yifanr/SimDist_ibrl/deps/robomimic")
sys.path.insert(0, "/project/6101830/yifanr/SimDist_ibrl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import imageio
import cv2
from scipy.spatial.transform import Rotation

import env.peg_insertion  # noqa: F401
import robosuite
from robosuite.controllers import load_controller_config
from robosuite.utils.binding_utils import MjSim, MjRenderContextOffscreen
from gs_mujoco import MjSplatRenderer
from gaussian_renderer.core.util_gau import load_ply
from gaussian_renderer.core.gaussiandata import GaussianData

GS = "/scratch/yifanr/gsbench"
SR = "/home/yifanr/scan_run2"
OUT = f"{GS}/pov_shots"
W, H = 1280, 720
WRIST_POSES = ["pose_000", "pose_006", "pose_012"]


def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.asarray(t).ravel(); return M


def inv(M):
    R, t = M[:3, :3], M[:3, 3]; return T(R.T, -R.T @ t)


def mj_quat_from_cv_R(R_cv):
    """CV camera (+z fwd, +y down) -> MuJoCo camera (-z fwd, +y up), wxyz."""
    q = Rotation.from_matrix(R_cv @ np.diag([1.0, -1.0, -1.0])).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def load_scene_sim(device, T_sw_bw):
    """scene.ply -> sim world: similarity to board frame, then base alignment."""
    tf = json.load(open(f"{GS}/scene5_world_transform.json"))
    s, R, t = float(tf["scale"]), np.array(tf["R"]), np.array(tf["t"])
    Rc, tc = T_sw_bw[:3, :3] @ R, T_sw_bw[:3, :3] @ t + T_sw_bw[:3, 3]
    g = load_ply(f"{GS}/sfm_scan5/scene.ply")
    xyz = (s * (g.xyz @ Rc.T) + tc).astype(np.float32)
    keep = ((np.abs(xyz[:, 0]) < 2.0) & (np.abs(xyz[:, 1]) < 2.0)
            & (xyz[:, 2] > 0.2) & (xyz[:, 2] < 2.4)
            & (g.scale.max(1) * s < 0.10) & (g.opacity.reshape(-1) > 0.35))
    print(f"[scene-prune] keep {keep.sum()}/{len(keep)}")
    q = g.rot[keep][:, [1, 2, 3, 0]]
    q = (Rotation.from_matrix(Rc) * Rotation.from_quat(q)).as_quat()[:, [3, 0, 1, 2]]
    Tt = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
    return (Tt(xyz[keep]), Tt(q.astype(np.float32)),
            Tt((g.scale[keep] * s).astype(np.float32)), Tt(g.opacity[keep]), Tt(g.sh[keep]))


def append_static(r, xyz, rot, scale, opa, sh):
    t = r.template
    sh_t = t.sh.reshape(t.xyz.shape[0], -1)
    sh_s = sh.reshape(xyz.shape[0], -1)
    w = max(sh_t.shape[1], sh_s.shape[1])
    pad = lambda x: torch.cat([x, torch.zeros(x.shape[0], w - x.shape[1],
                                              device=x.device)], 1) if x.shape[1] < w else x
    r.template = GaussianData(
        xyz=torch.cat([t.xyz, xyz]), rot=torch.cat([t.rot, rot]),
        scale=torch.cat([t.scale, scale]), opacity=torch.cat([t.opacity, opa]),
        sh=torch.cat([pad(sh_t), pad(sh_s)]).reshape(-1, w // 3, 3).contiguous())
    n = xyz.shape[0]
    r.p2b = torch.cat([r.p2b, torch.zeros(n, dtype=torch.long, device=r.device)])
    r.dyn_mask = torch.cat([r.dyn_mask, torch.ones(n, dtype=torch.bool, device=r.device)])
    r.n_gauss += n


def main():
    os.makedirs(OUT, exist_ok=True)
    np.random.seed(3)
    intr = json.load(open(f"{SR}/intrinsics.json"))
    ground = json.load(open(f"{GS}/ground_transforms.json"))

    cc = load_controller_config(default_controller="OSC_POSE")
    e = robosuite.make("PegInsertion_D0", robots="Panda", controller_configs=cc,
                       has_renderer=False, has_offscreen_renderer=False,
                       use_camera_obs=False, horizon=60)
    e.reset()

    # base alignment: real base (board frame) -> sim base
    bid = e.sim.model.body_name2id("robot0_base")
    bq = e.sim.data.body_xquat[bid]
    T_sw_base = T(Rotation.from_quat([bq[1], bq[2], bq[3], bq[0]]).as_matrix(),
                  e.sim.data.body_xpos[bid])
    T_bw_base = np.array(ground["base"]["T_world_base"])
    T_sw_bw = T_sw_base @ inv(T_bw_base)
    print(f"[align] sim base {np.round(T_sw_base[:3,3],3)} <- real base "
          f"{np.round(T_bw_base[:3,3],3)} (board frame)")

    cams = []
    for key, Tbw in [("zed2i", np.array(ground["T_world_zed2i"]))] + \
                    [(f"zedm_{p}", np.array(ground["T_world_zedm_per_pose"][p]))
                     for p in WRIST_POSES]:
        ik = "zed2i" if key == "zed2i" else "zedm"
        h_native = intr[ik]["resolution"][1]
        fovy = float(np.degrees(2 * np.arctan(h_native / (2 * intr[ik]["left"]["fy"]))))
        cams.append((key, T_sw_bw @ Tbw, fovy))
        print(f"[camera] {key}: fovy {fovy:.2f} deg, sim-world pos "
              f"{np.round((T_sw_bw @ Tbw)[:3,3], 3)}")

    cam_xml = "".join(
        f'<camera name="{n}" pos="{M[0,3]} {M[1,3]} {M[2,3]}" '
        f'quat="{" ".join(str(x) for x in mj_quat_from_cv_R(M[:3,:3]))}" fovy="{f}"/>'
        for n, M, f in cams)
    sim = MjSim.from_xml_string(
        e.sim.model.get_xml().replace("</worldbody>", cam_xml + "</worldbody>"))
    MjRenderContextOffscreen(sim, device_id=-1, max_width=W, max_height=H)
    sim.data.qpos[:] = e.sim.data.qpos
    sim.data.qvel[:] = 0
    sim.forward()
    m = sim.model

    calib = json.load(open(f"{GS}/splat_frame_calib.json"))
    overrides = {"panda_link0": "skip"}
    for tok, c in calib.items():
        try:
            m.body_name2id(c["mj_body"])
        except Exception:
            overrides[tok] = "skip"
    r = MjSplatRenderer(m, f"{GS}/splat_frame_calib.json", overrides=overrides)
    n_arm = r.n_gauss
    append_static(r, *load_scene_sim(r.device, T_sw_bw))
    print(f"[pov] {n_arm} arm + {r.n_gauss - n_arm} scene gaussians")

    keep_gids = [g for g in range(m.ngeom)
                 if (m.geom_id2name(g) or "").startswith(("board_obj", "peg_obj", "gripper0"))]

    for name, _, _ in cams:
        if name == "zed2i":
            full = cv2.imread("/home/yifanr/Explorer_HD2K_SN37597132_17-54-59.png")
            raw = full[:, : full.shape[1] // 2]
        else:
            raw = cv2.imread(f"{SR}/{name.split('zedm_')[1]}/zedm_left.png")
        real = cv2.resize(raw, (W, H))[:, :, ::-1]      # every real frame -> render size
        ref = sim.render(camera_name=name, width=W, height=H)[::-1]
        seg = sim.render(camera_name=name, width=W, height=H,
                         segmentation=True)[::-1][:, :, 1]
        keep = np.isin(seg, keep_gids)[..., None]
        splat8 = (r.render(m, sim.data, [name], H, W)[0] * 255).astype(np.uint8)
        comp = np.where(keep, ref, splat8)
        for tag, im in (("real", real), ("mujoco", ref), ("splat", splat8),
                        ("composite", comp)):
            imageio.imwrite(f"{OUT}/{name}_{tag}.png", im)
        imageio.imwrite(f"{OUT}/{name}_strip.png",
                        np.concatenate([real, ref, splat8, comp], 1))
        print(f"[pov] saved {name}")
    print("[pov] done")


if __name__ == "__main__":
    main()

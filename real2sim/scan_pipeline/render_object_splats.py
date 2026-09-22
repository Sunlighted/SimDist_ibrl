"""Both SIM cameras, splatted board+peg vs plain MuJoCo, through an episode.

Answers "what will the pixel stage actually see if the objects are photoreal?"
The cameras are the CALIBRATED real ones (env/peg_insertion.py: zedview =
external ZED2i, robot0_zedm_wrist = wrist ZED Mini incl. the Rz(45.03) fix), and
the episode comes from a real demo replayed through sim_state, so the poses are
the task's own, not hand-posed.

Compositing: object SILHOUETTES come from the MuJoCo segmentation and the
APPEARANCE from the splat. That is legitimate here precisely because the splat is
registered TO the scanned STL that MuJoCo renders -- the two silhouettes are the
same geometry, so the mask is not smuggling in information. (gs_mujoco.render
discards alpha, so there is no coverage channel to composite on instead.)
"""
import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/project/6101830/yifanr/SimDist_ibrl/deps/mimicgen")
sys.path.insert(0, "/project/6101830/yifanr/SimDist_ibrl/deps/robomimic")
sys.path.insert(0, "/project/6101830/yifanr/SimDist_ibrl")
sys.path.insert(0, "/scratch/yifanr/gsbench")
import numpy as np
import env.peg_insertion  # noqa
import robosuite
from robosuite.controllers import load_controller_config
from robosuite.utils.binding_utils import MjSim, MjRenderContextOffscreen
from gs_mujoco import MjSplatRenderer
from gaussian_renderer.core.util_gau import load_ply
from gaussian_renderer.core.gaussiandata import GaussianData
import torch
from scipy.spatial.transform import Rotation

SEG = "/scratch/yifanr/gsbench/objsplats"
CAL = f"{SEG}/obj_calib.json"
OUT = "/scratch/yifanr/gsbench/objsplat_shots"
# Phase states pre-extracted with the ibrl venv: the gsbench venv (which owns
# gsplat) has no h5py, and adding a dep to the 3DGS env to read three rows is
# the wrong trade.
STATES = f"{SEG}/demo0_states.npy"
W = H = 512
CAMS = [("zedview", "external ZED2i"), ("robot0_zedm_wrist", "wrist ZED Mini")]
GS = "/scratch/yifanr/gsbench"
# ★ Pair the transform with the COLMAP model it was solved on. Both copies of
# scene5_world_transform.json name sparse=sfm_scan5/sparse/1, which is the model
# behind sfm_scan5/scene.ply -- the pairing that produced the validated
# CORRECT_zedpov renders. (CLAUDE.md §2c quotes scale 0.1089; the file says
# 0.10234 -- the doc figure is the superseded scan4-based calibration.)
# ★ v2, NOT sfm_scan5/scene.ply. The latter is a stale 7k-iteration run (~154k
# gaussians, dark background). v2 is 30k iterations, 1,108,793 gaussians,
# held-out PSNR 34.06, exposure_comp off, bg_rgb (0.83,0.82,0.80) -- which is why
# the old one showed black walls. Same colmap_dir (sfm_scan5/sparse/1), so
# scene5_world_transform.json applies unchanged.
# md5 a32f446437488a3f041168755724d4fd == scene_v2_clean_pruned.ply (NOT the
# _shall / _shnarrow SH-damped variants, which share its byte size).
SCENE_PLY = "/project/6101830/yifanr/SimDist_ibrl/real2sim/splats/scene_scan5_norobot_v2.ply"
SCENE_TF = f"{GS}/scene5_world_transform.json"
GROUND = f"{GS}/ground_transforms.json"


def cv2_dilate(mask, k):
    """Binary dilate without importing cv2 -- it does not load under the gsbench
    venv on a compute node (resolves to a DRAC opencv built for another python)."""
    m = mask.astype(bool)
    out = m.copy()
    for _ in range(k):
        out[1:, :] |= out[:-1, :]; out[:-1, :] |= out[1:, :]
        out[:, 1:] |= out[:, :-1]; out[:, :-1] |= out[:, 1:]
    return out


def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.asarray(t).ravel(); return M


def inv(M):
    R, t = M[:3, :3], M[:3, 3]; return T(R.T, -R.T @ t)


def load_scene_sim(device, T_sw_bw):
    """scene v2 -> sim world.

    ★ v2 IS ALREADY IN METRIC BOARD-FRAME COORDINATES. Do NOT apply
    scene5_world_transform.json to it -- that similarity maps the RAW COLMAP
    frame (old sfm_scan5/scene.ply) to the board frame, and applying it to a ply
    that is already board-framed shrinks the room by 0.1023x, which renders as a
    ~0.2 m blob of white spikes.

    Verified: v2 with no transform has densest z bin 0.73 m and an xy span of
    1.83 x 1.58 m -- i.e. the table surface at ~0.8 m, matching what old7k looks
    like only AFTER the transform. The committed CORRECT_zedpov_strip.png is
    byte-identical to pov_correctK/scene_v2_clean_pruned_strip.png, produced by
    `render_zed_pov.py --ply sfm_scan5/scene_v2_clean_pruned.ply
     --extrinsic zed2i_extrinsic_correctK.json` with NO --ply_world_transform,
    which is the independent confirmation.

    So the only mapping left is board frame -> sim world (T_sw_bw).
    """
    g = load_ply(SCENE_PLY)
    R, t = T_sw_bw[:3, :3], T_sw_bw[:3, 3]
    xyz = (g.xyz @ R.T + t).astype(np.float32)
    keep = ((np.abs(xyz[:, 0]) < 2.0) & (np.abs(xyz[:, 1]) < 2.0)
            & (xyz[:, 2] > 0.2) & (xyz[:, 2] < 2.4)
            & (g.scale.max(1) < 0.10) & (g.opacity.reshape(-1) > 0.35))
    print(f"[scene] keep {int(keep.sum())}/{len(keep)} gaussians after prune")
    q = g.rot[keep][:, [1, 2, 3, 0]]
    q = (Rotation.from_matrix(R) * Rotation.from_quat(q)).as_quat()[:, [3, 0, 1, 2]]
    tt = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
    return (tt(xyz[keep]), tt(q.astype(np.float32)),
            tt(g.scale[keep].astype(np.float32)), tt(g.opacity[keep]), tt(g.sh[keep]))


def append_static(r, xyz, rot, scale, opa, sh):
    t = r.template
    sh_t = t.sh.reshape(t.xyz.shape[0], -1)
    sh_s = sh.reshape(xyz.shape[0], -1)
    w = max(sh_t.shape[1], sh_s.shape[1])
    pad = lambda x: torch.cat([x, torch.zeros(x.shape[0], w - x.shape[1], device=x.device)], 1) \
        if x.shape[1] < w else x
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
    calib = json.load(open(CAL))
    for k, v in calib.items():
        print(f"[calib] {k} -> {v['mj_body']}  verdict {v['verdict']}  "
              f"rms {1000*v['rms_m']:.2f} mm  aspect_err {v.get('aspect_log_err', float('nan')):.3f}")

    cc = load_controller_config(default_controller="OSC_POSE")
    e = robosuite.make("PegInsertion_D0", robots="Panda", controller_configs=cc,
                       has_renderer=False, has_offscreen_renderer=False,
                       use_camera_obs=False, horizon=400)
    e.reset()
    sim = MjSim.from_xml_string(e.sim.model.get_xml())
    ctx = MjRenderContextOffscreen(sim, device_id=-1, max_width=W, max_height=H)
    ctx.vopt.geomgroup[0] = 0; ctx.vopt.geomgroup[1] = 1     # visual meshes only
    m = sim.model

    obj_geoms = [g for g in range(m.ngeom)
                 if (m.geom_id2name(g) or "").startswith(("board_obj", "peg_obj"))]
    rgba0 = np.array(m.geom_rgba)                            # to hide/show objects

    # PEG STAYS MUJOCO. Its splat registered POOR (aspect log-err 0.316) and it is
    # a plain off-white box whose MuJoCo material now matches the video to 1-6%
    # once the strip light is right -- a splat buys nothing for it.
    r = MjSplatRenderer(m, CAL, seg_dir=SEG, overrides={"peg_obj": "skip"})
    n_obj = r.n_gauss
    print(f"[splat] {n_obj} board gaussians (peg splat skipped)")
    # base alignment: real base (board frame) -> sim base, same chain as
    # render_camera_pov.py, so the scene lands where the sim arm actually is.
    bid = m.body_name2id("robot0_base")
    bq = m.body_quat[bid] if False else sim.data.body_xquat[bid]
    T_sw_base = T(Rotation.from_quat([bq[1], bq[2], bq[3], bq[0]]).as_matrix(),
                  sim.data.body_xpos[bid])
    T_bw_base = np.array(json.load(open(GROUND))["base"]["T_world_base"])
    T_sw_bw = T_sw_base @ inv(T_bw_base)
    append_static(r, *load_scene_sim(r.device, T_sw_bw))
    print(f"[splat] + scene -> {r.n_gauss} total ({r.n_gauss - n_obj} scene)")
    # ARM stays MuJoCo: the PolaRiS arm splat is Robotiq-only and the Panda hand
    # has no splat at all (docs/real2sim.md §4 A residual 4).
    robot_geoms = [g for g in range(m.ngeom)
                   if (m.geom_id2name(g) or "").startswith(("robot0", "gripper0"))]
    peg_geoms = [g for g in range(m.ngeom)
                 if (m.geom_id2name(g) or "").startswith("peg_obj")]
    board_geoms = [g for g in range(m.ngeom)
                   if (m.geom_id2name(g) or "").startswith("board_obj")]

    ss = np.load(STATES)   # rows = approach / pre-insert / inserted
    # This job saves RAW PANELS ONLY. cv2 does not import under the gsbench venv on
    # a compute node (it resolves to the DRAC opencv module built for a different
    # python), and the figure is worth re-composing without paying for another GPU
    # job anyway. compose_objsplat_figure.py does the labelling on the login node.
    panels_out, meta = {}, []
    for t, phase in enumerate(["approach", "pre-insert", "inserted"]):
        sim.set_state_from_flattened(ss[t]); sim.forward()
        for cam, tag in CAMS:
            plain = sim.render(camera_name=cam, width=W, height=H)[::-1]
            seg = sim.render(camera_name=cam, width=W, height=H,
                             segmentation=True)[::-1][:, :, 1]
            sp = (r.render(m, sim.data, [cam], H, W)[0] * 255).astype(np.uint8)
            arm = np.isin(seg, robot_geoms)
            peg = np.isin(seg, peg_geoms)
            brd = np.isin(seg, board_geoms)
            # A) splat SCENE, MuJoCo board + peg + arm. The board splat is in `sp`
            #    behind the MuJoCo board, so dilate the board mask a little to stop
            #    it ghosting out past the MuJoCo silhouette.
            brd_d = cv2_dilate(brd, 5)
            keepA = arm | peg | brd_d
            a_img = np.where(keepA[..., None], plain, sp)
            # B) splat SCENE + splat BOARD, MuJoCo peg + arm.
            keepB = arm | peg
            b_img = np.where(keepB[..., None], plain, sp)
            panels_out[f"{t}_{cam}_mujoco"] = plain.astype(np.uint8)
            panels_out[f"{t}_{cam}_mjboard"] = a_img.astype(np.uint8)
            panels_out[f"{t}_{cam}_splatboard"] = b_img.astype(np.uint8)
            meta.append(f"{phase}|{cam}|{tag}|peg={int(peg.sum())}|board={int(brd.sum())}")
            print(f"[render] {phase:11s} {cam:20s} peg {int(peg.sum()):6d} px  board {int(brd.sum()):7d} px")
    np.savez_compressed(f"{OUT}/panels.npz", meta=np.array(meta), **panels_out)
    print(f"[done] {OUT}/panels.npz  ({len(panels_out)} panels)")


if __name__ == "__main__":
    main()

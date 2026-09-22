"""Replace MuJoCo camera obs with splat composites, for stage-1 data generation.

WHAT IS SPLATTED AND WHAT IS NOT
  scene  -> splat (the real cell: poster work surface, walls). This is most of
            the frame and nothing a material value can imitate.
  board  -> splat (moulded cutouts, shading, print detail).
  peg    -> MuJoCo. Its splat registered POOR (aspect log-err 0.316) and it is a
            plain off-white box whose material matches the video to 1-6% once the
            strip light is right. A splat buys nothing.
  arm    -> MuJoCo. The PolaRiS arm splat is Robotiq-only and the Panda hand has
            no splat at all (docs/real2sim.md §4 A residual 4).

COMPOSITING: splat provides the background and the board; MuJoCo provides the arm
and the peg, selected by the MuJoCo segmentation buffer. That is legitimate for
the board because the board splat is registered TO the scanned STL MuJoCo renders
-- the silhouettes are the same geometry. The arm/peg masks are MuJoCo's own.

FRAME: `scene_scan5_norobot_v2.ply` is ALREADY in metric board-frame coordinates.
Do NOT apply scene5_world_transform.json to it -- that maps the RAW COLMAP frame
of the older sfm_scan5/scene.ply, and applying it to v2 shrinks the room 0.1023x.
Verified: v2 untransformed has its densest z bin at 0.73 m with a 1.83 x 1.58 m
xy span, i.e. the table surface at ~0.8 m.

ORIENTATION: robosuite's obs images and sim.render() are both BOTTOM-UP (the
wrapper flips them later), while MjSplatRenderer returns TOP-DOWN. So the
composite is flipped once on the way back into the obs dict. Checked against a
known-good render rather than reasoned about -- see scripts/check_splat_obs.py.
"""
import json
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)                      # real2sim/gs_mujoco.py
sys.path.insert(0, os.path.join(_REPO, "deps", "GaussianRenderer", "src"))

# Repo-relative defaults so another machine needs only the repo + submodules.
DEFAULT_SEG_DIR = os.path.join(_REPO, "real2sim", "splats", "objsplats")
DEFAULT_CALIB = os.path.join(DEFAULT_SEG_DIR, "obj_calib.json")
DEFAULT_SCENE_PLY = os.path.join(_REPO, "real2sim", "splats",
                                 "scene_scan5_norobot_v2.ply")
DEFAULT_GROUND = os.path.join(_REPO, "real2sim", "scan_pipeline",
                              "ground_transforms.json")


class SplatCompositor:
    def __init__(self, mj_model, mj_data, calib_path=DEFAULT_CALIB,
                 seg_dir=DEFAULT_SEG_DIR, scene_ply=DEFAULT_SCENE_PLY,
                 ground_json=DEFAULT_GROUND,
                 skip_tokens=("peg_obj",), device="cuda"):
        import torch
        from scipy.spatial.transform import Rotation
        from gs_mujoco import MjSplatRenderer
        from gaussian_renderer.core.util_gau import load_ply
        from gaussian_renderer.core.gaussiandata import GaussianData
        self._torch, self._GD = torch, GaussianData

        self.r = MjSplatRenderer(mj_model, calib_path, seg_dir=seg_dir,
                                 overrides={t: "skip" for t in skip_tokens},
                                 device=device)
        n_obj = self.r.n_gauss

        # board frame -> sim world, anchored on the robot base (same chain as
        # render_camera_pov.py). mj_data must already be forwarded.
        bid = mj_model.body_name2id("robot0_base")
        bq = mj_data.body_xquat[bid]
        T_sw_base = _T(Rotation.from_quat([bq[1], bq[2], bq[3], bq[0]]).as_matrix(),
                       mj_data.body_xpos[bid])
        T_bw_base = np.array(json.load(open(ground_json))["base"]["T_world_base"])
        T = T_sw_base @ _inv(T_bw_base)

        g = load_ply(scene_ply)
        R, t = T[:3, :3], T[:3, 3]
        xyz = (g.xyz @ R.T + t).astype(np.float32)
        keep = ((np.abs(xyz[:, 0]) < 2.0) & (np.abs(xyz[:, 1]) < 2.0)
                & (xyz[:, 2] > 0.2) & (xyz[:, 2] < 2.4)
                & (g.scale.max(1) < 0.10) & (g.opacity.reshape(-1) > 0.35))
        q = g.rot[keep][:, [1, 2, 3, 0]]
        q = (Rotation.from_matrix(R) * Rotation.from_quat(q)).as_quat()[:, [3, 0, 1, 2]]
        tt = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32,
                                    device=self.r.device)
        self._append_static(tt(xyz[keep]), tt(q.astype(np.float32)),
                            tt(g.scale[keep].astype(np.float32)),
                            tt(g.opacity[keep]), tt(g.sh[keep]))
        print(f"[splat-obs] {n_obj} board + {int(keep.sum())} scene = "
              f"{self.r.n_gauss} gaussians", flush=True)

        name = lambda gi: mj_model.geom_id2name(gi) or ""
        self.mj_geoms = [gi for gi in range(mj_model.ngeom)
                         if name(gi).startswith(("robot0", "gripper0", "peg_obj"))]

    def _append_static(self, xyz, rot, scale, opa, sh):
        torch = self._torch
        t = self.r.template
        sh_t = t.sh.reshape(t.xyz.shape[0], -1)
        sh_s = sh.reshape(xyz.shape[0], -1)
        w = max(sh_t.shape[1], sh_s.shape[1])
        pad = lambda x: (torch.cat([x, torch.zeros(x.shape[0], w - x.shape[1],
                                                   device=x.device)], 1)
                         if x.shape[1] < w else x)
        self.r.template = self._GD(
            xyz=torch.cat([t.xyz, xyz]), rot=torch.cat([t.rot, rot]),
            scale=torch.cat([t.scale, scale]), opacity=torch.cat([t.opacity, opa]),
            sh=torch.cat([pad(sh_t), pad(sh_s)]).reshape(-1, w // 3, 3).contiguous())
        n = xyz.shape[0]
        self.r.p2b = torch.cat([self.r.p2b,
                                torch.zeros(n, dtype=torch.long, device=self.r.device)])
        self.r.dyn_mask = torch.cat([self.r.dyn_mask,
                                     torch.ones(n, dtype=torch.bool, device=self.r.device)])
        self.r.n_gauss += n

    def composite(self, sim, cam_name, h, w):
        """Bottom-up uint8 HxWx3, ready to drop straight into obs[f'{cam}_image']."""
        mj = sim.render(camera_name=cam_name, width=w, height=h)          # bottom-up
        seg = sim.render(camera_name=cam_name, width=w, height=h,
                         segmentation=True)[:, :, 1]                      # bottom-up
        sp = (self.r.render(sim.model, sim.data, [cam_name], h, w)[0]
              * 255).astype(np.uint8)[::-1]                               # top-down -> bottom-up
        keep = np.isin(seg, self.mj_geoms)[..., None]
        return np.where(keep, mj, sp).astype(np.uint8)


def _T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.asarray(t).ravel(); return M


def _inv(M):
    R, t = M[:3, :3], M[:3, 3]; return _T(R.T, -R.T @ t)

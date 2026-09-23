"""Reusable bridge: render a robosuite/MuJoCo scene with _GaussianRenderer's
BATCHED gsplat path, using PolaRiS's per-link robot splats.

Handles the two things the raw library does not do for us:
  1. applies the per-body rigid calibration solved by calibrate_splat_frames.py
     (PolaRiS link frames != robosuite MuJoCo body frames), rotating the gaussian
     ORIENTATIONS as well as their positions -- rotating only xyz leaves every
     splat ellipsoid mis-oriented, which is invisible in a centroid check and
     very visible in a render;
  2. merges the several plys that map to one MuJoCo body (PolaRiS ships two
     plys for each inner_finger).
"""
import glob, json, os, sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation

# vendored as a submodule (deps/GaussianRenderer, MIT, pinned 577efdb) so the
# repo is self-contained -- this used to point at /scratch and did not port.
_RR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "deps", "GaussianRenderer", "src")
sys.path.insert(0, os.path.abspath(_RR))
from gaussian_renderer.core.util_gau import load_ply
from gaussian_renderer.core.gaussiandata import GaussianData
from gaussian_renderer.core.batch_rasterization import (
    batch_update_gaussians, batch_env_render,
)

SEG_DEFAULT = "/scratch/yifanr/realsim_survey/polaris/hf-hub/nvidia_droid/SEGMENTED"


def _tok(path):
    t = os.path.basename(path)[:-4].split("-Defeatured")[0]
    return t.split("-")[0] if t.startswith("panda_link") else t


class MjSplatRenderer:
    def __init__(self, mj_model, calib_path, seg_dir=SEG_DEFAULT,
                 overrides=None, device="cuda"):
        """overrides: {link_token: "identity"|"skip"} to veto a calibration whose
        ICP solution we do not trust (e.g. a symmetric-alias rotation)."""
        self.device = torch.device(device)
        calib = json.load(open(calib_path))
        overrides = overrides or {}

        groups = {}
        for f in sorted(glob.glob(f"{seg_dir}/*.ply")):
            groups.setdefault(_tok(f), []).append(f)

        xyz, rot, scale, opa, sh = [], [], [], [], []
        self.body_ids, starts, ends = [], [], []
        n, max_sh = 0, 0
        loaded = []
        for tok, files in groups.items():
            if tok not in calib or overrides.get(tok) == "skip":
                continue
            c = calib[tok]
            bid = mj_model.body_name2id(c["mj_body"])
            if overrides.get(tok) == "identity":
                R, t = np.eye(3), np.zeros(3)
            else:
                R, t = np.array(c["R"]), np.array(c["t"])
            qR = Rotation.from_matrix(R)
            for f in files:
                g = load_ply(f)
                g.xyz = (g.xyz @ R.T + t).astype(np.float32)
                # rot is wxyz; scipy is xyzw
                q = g.rot[:, [1, 2, 3, 0]]
                q = (qR * Rotation.from_quat(q)).as_quat()[:, [3, 0, 1, 2]]
                g.rot = q.astype(np.float32)
                max_sh = max(max_sh, g.sh.shape[1])
                loaded.append(g)
                starts.append(n); n += len(g.xyz); ends.append(n)
                self.body_ids.append(bid)

        def pad(s):
            return (np.concatenate([s, np.zeros((s.shape[0], max_sh - s.shape[1]), np.float32)], 1)
                    if s.shape[1] < max_sh else s)

        T = lambda a: torch.tensor(np.concatenate(a, 0), dtype=torch.float32, device=self.device)
        self.template = GaussianData(
            xyz=T([g.xyz for g in loaded]), rot=T([g.rot for g in loaded]),
            scale=T([g.scale for g in loaded]), opacity=T([g.opacity for g in loaded]),
            sh=T([pad(g.sh) for g in loaded]).reshape(n, -1, 3).contiguous())
        self.dyn_mask = torch.ones(n, dtype=torch.bool, device=self.device)
        self.p2b = torch.zeros(n, dtype=torch.long, device=self.device)
        for s, e, b in zip(starts, ends, self.body_ids):
            self.p2b[s:e] = b
        self.n_gauss = n
        self.n_parts = len(loaded)

    @staticmethod
    def jitter_cameras(cp, cx, pos_sigma=0.0, rot_sigma_deg=0.0, generator=None):
        """Camera-extrinsic domain randomisation, per env per camera.

        This is the one DR axis the repo does not already have (photometric and
        shift live in world_model.py's aug_* knobs). It is mandatory here and not
        optional elsewhere: the external ZED mounts move between sessions, so a
        camera pose baked into a frozen encoder is an unrepairable error.

        Cheap in the splat path -- batch_env_render takes per-(env,cam)
        extrinsics, so this is a tensor perturbation, not a re-render. NOTE the
        corollary: once the camera moves you can no longer precompute a static
        background image, which is why the throughput benchmark rasterised the
        background every step.
        """
        if pos_sigma <= 0 and rot_sigma_deg <= 0:
            return cp, cx
        E, C = cp.shape[0], cp.shape[1]
        dev = cp.device
        g = {"generator": generator} if generator is not None else {}
        cp = cp + torch.randn(cp.shape, device=dev, **g) * pos_sigma
        if rot_sigma_deg > 0:
            # small-angle: compose with exp(skew(w)), w ~ N(0, sigma) rad
            w = torch.randn((E, C, 3), device=dev, **g) * np.radians(rot_sigma_deg)
            K = torch.zeros((E, C, 3, 3), device=dev)
            K[..., 0, 1], K[..., 0, 2] = -w[..., 2], w[..., 1]
            K[..., 1, 0], K[..., 1, 2] = w[..., 2], -w[..., 0]
            K[..., 2, 0], K[..., 2, 1] = -w[..., 1], w[..., 0]
            I = torch.eye(3, device=dev).expand(E, C, 3, 3)
            dR = I + K + 0.5 * K @ K          # 2nd-order exp; sigma is small
            cx = (dR @ cx.reshape(E, C, 3, 3)).reshape(E, C, 9)
        return cp, cx

    # ★ The v2 scene splat was TRAINED against this background (its meta records
    # bg_rgb = the measured wall colour), so wherever coverage is thin the
    # gaussians assume it shows through. Compositing on the default BLACK instead
    # turns every thinly-reconstructed region into a hard black hole -- a strong
    # artificial feature that is not in the real cell. render_zed_pov.py passes
    # this colour, which is why CORRECT_zedpov_strip.png shows no such holes
    # while our composites did.
    BG_RGB = (0.8313725490196079, 0.8215686274509804, 0.8019607843137255)

    def render(self, mj_model, mj_data, cam_names, height, width,
               cam_pos_sigma=0.0, cam_rot_sigma_deg=0.0, generator=None, bg=None):
        cam_ids = [mj_model.camera_name2id(c) for c in cam_names]
        bp = torch.tensor(np.asarray(mj_data.xpos), dtype=torch.float32,
                          device=self.device).unsqueeze(0)
        bq = torch.tensor(np.asarray(mj_data.xquat), dtype=torch.float32,
                          device=self.device).unsqueeze(0)
        gsb = batch_update_gaussians(self.template, bp, bq,
                                     point_to_body_idx=self.p2b, dynamic_mask=self.dyn_mask)
        cp = torch.tensor(np.asarray(mj_data.cam_xpos)[cam_ids], dtype=torch.float32,
                          device=self.device).unsqueeze(0)
        cx = torch.tensor(np.asarray(mj_data.cam_xmat)[cam_ids], dtype=torch.float32,
                          device=self.device).unsqueeze(0)
        cp, cx = self.jitter_cameras(cp, cx, cam_pos_sigma, cam_rot_sigma_deg, generator)
        fovy = np.asarray(mj_model.cam_fovy)[cam_ids][None, :].astype(np.float32)
        bg = self.BG_RGB if bg is None else bg
        bg_imgs = torch.tensor(bg, dtype=torch.float32, device=self.device) \
            .view(1, 1, 1, 1, 3).expand(1, len(cam_ids), height, width, 3).contiguous()
        rgb, _ = batch_env_render(gsb, cp, cx, height, width, fovy, bg_imgs=bg_imgs)
        return rgb[0].clamp(0, 1).detach().cpu().numpy()

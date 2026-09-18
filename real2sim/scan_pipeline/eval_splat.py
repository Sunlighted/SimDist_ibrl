"""Score any .ply against a COLMAP model's held-out frames.

Needed to compare v1 vs v2 with ONE metric on ONE split, instead of asserting
that the new splat is better. Renders through the same gsplat path the sim uses.

`--sh_layout` exists because v1's writer and util_gau's reader disagree on the
f_rest ordering (coeff-major vs channel-major). Scoring a v1 ply BOTH ways
separates "the SH bug" from "the training changes" -- otherwise the two are
confounded in every number.

    python eval_splat.py --ply sfm_merged/scene_v2.ply \
        --colmap_dir sfm_merged/sparse/0 --image_dir sfm_merged/images \
        --holdout_stride 8 --bg_color_json wall_color.json
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData

from train_3dgs import load_colmap


def load_ply_raw(path, sh_layout="channel"):
    ply = PlyData.read(path)
    el = ply.elements[0]
    N = el["x"].shape[0]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float32)
    sh0 = np.stack([el[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float32)
    rest_names = sorted([p.name for p in el.properties if p.name.startswith("f_rest_")],
                        key=lambda s: int(s.split("_")[-1]))
    rest = (np.stack([el[n] for n in rest_names], 1).astype(np.float32)
            if rest_names else np.zeros((N, 0), np.float32))
    n_coeffs = rest.shape[1] // 3
    if n_coeffs:
        if sh_layout == "channel":       # ply/INRIA convention: (3, n_coeffs)
            rest = rest.reshape(N, 3, n_coeffs).transpose(0, 2, 1)
        else:                             # v1 writer's layout: (n_coeffs, 3)
            rest = rest.reshape(N, n_coeffs, 3)
    else:
        rest = rest.reshape(N, 0, 3)
    opacity = np.asarray(el["opacity"], np.float32)
    scales = np.stack([el[f"scale_{i}"] for i in range(3)], 1).astype(np.float32)
    quats = np.stack([el[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)
    return xyz, sh0, rest, opacity, scales, quats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ply", required=True)
    p.add_argument("--colmap_dir", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--holdout_stride", type=int, default=8)
    p.add_argument("--image_scale", type=float, default=0.5)
    p.add_argument("--bg_color_json", default="")
    p.add_argument("--sh_layout", default="channel", choices=["channel", "coeff"])
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--world_transform", default="",
                   help="REQUIRED for a ply trained with --world_transform: the "
                        "COLMAP poses must be mapped the same way or the cameras and "
                        "the gaussians live in different frames and every number is "
                        "meaningless (renders would be empty/garbage, not just worse).")
    a = p.parse_args()

    dev = torch.device("cuda")
    bg = json.load(open(a.bg_color_json))["wall_rgb"] if a.bg_color_json else [0, 0, 0]
    bg_t = torch.tensor(bg, dtype=torch.float32, device=dev)  # (3,), see train_3dgs_v2

    xyz, sh0, rest, opa, sc, qt = load_ply_raw(a.ply, a.sh_layout)
    T = lambda x: torch.tensor(x, device=dev)
    means, sh0_t, rest_t = T(xyz), T(sh0), T(rest)
    opa_t, sc_t, qt_t = T(opa), T(sc), T(qt)
    sh_degree = int(round(np.sqrt(rest.shape[1] + 1) - 1))
    colours = torch.cat([sh0_t.unsqueeze(1), rest_t], 1)
    print(f"[eval] {a.ply}: {len(xyz):,} gaussians, sh_degree {sh_degree}, "
          f"layout {a.sh_layout}, bg {np.round(bg,3).tolist()}")

    frames, _, _ = load_colmap(a.colmap_dir, a.image_dir)
    if a.world_transform:
        wt = json.load(open(a.world_transform))
        s_w, R_w, t_w = float(wt["scale"]), np.array(wt["R"]), np.array(wt["t"])
        for fr in frames:
            R_c, t_c = fr["W2C"][:3, :3], fr["W2C"][:3, 3]
            Rn = R_c @ R_w.T
            W2C = np.eye(4); W2C[:3, :3] = Rn; W2C[:3, 3] = s_w * t_c - Rn @ t_w
            fr["W2C"] = W2C
        print(f"[eval] world transform applied (scale {s_w:.5f})")
    held = np.arange(len(frames))[:: a.holdout_stride]
    if a.max_frames:
        held = held[: a.max_frames]

    import cv2
    from gsplat.rendering import rasterization

    psnrs, l1s = [], []
    with torch.no_grad():
        for i in held:
            fr = frames[int(i)]
            img = cv2.imread(fr["path"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            if a.image_scale != 1.0:
                img = cv2.resize(img, (int(img.shape[1] * a.image_scale),
                                       int(img.shape[0] * a.image_scale)),
                                 interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
            K = torch.tensor(fr["K"], device=dev, dtype=torch.float32).clone()
            if a.image_scale != 1.0:
                K[:2] *= a.image_scale
            out, _, _ = rasterization(
                means=means, quats=qt_t, scales=sc_t.exp(), opacities=opa_t.sigmoid(),
                colors=colours,
                viewmats=torch.tensor(fr["W2C"], device=dev, dtype=torch.float32)[None],
                Ks=K[None], width=w, height=h, sh_degree=sh_degree, backgrounds=bg_t)
            pred = out[0].clamp(0, 1)
            gt = torch.tensor(img, device=dev)
            psnrs.append(-10 * torch.log10(F.mse_loss(pred, gt) + 1e-10).item())
            l1s.append(F.l1_loss(pred, gt).item())
    print(f"[eval] held-out n={len(psnrs)}  PSNR {np.mean(psnrs):.3f} "
          f"(median {np.median(psnrs):.3f})  L1 {np.mean(l1s):.4f}")
    print(json.dumps({"ply": a.ply, "sh_layout": a.sh_layout, "n": len(psnrs),
                      "psnr_mean": float(np.mean(psnrs)),
                      "psnr_median": float(np.median(psnrs)),
                      "l1_mean": float(np.mean(l1s)),
                      "n_gauss": int(len(xyz))}))


if __name__ == "__main__":
    main()

"""Damp view-dependent SH where the data never constrained it.

The problem this addresses is the one that actually matters for our pipeline.
The junk visible from the ZED pose survives visibility pruning, because those
gaussians ARE rendered by training views -- just from a narrow cone of
directions. 3DGS fits their degree>=1 SH to look right inside that cone and is
free to produce anything outside it, so a camera placed elsewhere sees garbage.

That is exactly our deployment situation: stage-1 renders from SIM camera poses,
which are nowhere on the handheld scan trajectory. It also means our held-out
PSNR flatters us -- held-out frames from a continuous sweep are immediate
neighbours of training frames, i.e. an INTERPOLATION test, while deployment is
EXTRAPOLATION.

Fix: for gaussians whose observed directions are tightly concentrated, drop the
view-dependent terms and keep the DC colour. They then render as diffuse -- a
plausible constant instead of an unconstrained extrapolation. Concentration is
measured as the resultant length R of the unit viewing directions (von
Mises-Fisher style): R -> 1 means all views came from one direction.

    python damp_sh.py --ply in.ply --out out.ply --colmap_dir ... --image_dir ... \
        --mode narrow --min_R 0.9
"""
import argparse
import json
import os
import re

import numpy as np
import torch

from eval_splat import load_ply_raw
from train_3dgs import load_colmap
from train_3dgs_v2 import save_ply_channel_major


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--colmap_dir", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--world_transform", default="")
    ap.add_argument("--exclude_idx_ranges", default="")
    ap.add_argument("--mode", default="narrow", choices=["narrow", "all"])
    ap.add_argument("--min_R", type=float, default=0.9,
                    help="narrow mode: damp gaussians with resultant length above this")
    ap.add_argument("--render_scale", type=float, default=0.25)
    a = ap.parse_args()

    dev = torch.device("cuda")
    frames, _, _ = load_colmap(a.colmap_dir, a.image_dir)
    if a.world_transform:
        wt = json.load(open(a.world_transform))
        s_w, R_w, t_w = float(wt["scale"]), np.array(wt["R"]), np.array(wt["t"])
        for fr in frames:
            R_c, t_c = fr["W2C"][:3, :3], fr["W2C"][:3, 3]
            Rn = R_c @ R_w.T
            M = np.eye(4); M[:3, :3] = Rn; M[:3, 3] = s_w * t_c - Rn @ t_w
            fr["W2C"] = M
    if a.exclude_idx_ranges:
        rr = [tuple(int(x) for x in r.split("-")) for r in a.exclude_idx_ranges.split(",")]
        frames = [fr for fr in frames
                  if not any(lo <= int(re.findall(r"(\d+)", os.path.basename(fr["path"]))[-1]) <= hi
                             for lo, hi in rr)]
    print(f"[sh] {len(frames)} training views")

    xyz, sh0, rest, opa, sc, qt = load_ply_raw(a.ply, "channel")
    N = len(xyz)
    T = lambda x: torch.tensor(x, device=dev)
    means, qt_t, sc_t, opa_t = T(xyz), T(qt), T(sc), T(opa)
    colours = torch.cat([T(sh0).unsqueeze(1), T(rest)], 1)
    sh_deg = int(round(np.sqrt(rest.shape[1] + 1) - 1))

    if a.mode == "all":
        damp = np.ones(N, bool)
        Rlen = None
    else:
        dir_sum = torch.zeros(N, 3, device=dev)
        cnt = torch.zeros(N, device=dev)
        from gsplat.rendering import rasterization
        with torch.no_grad():
            for fr in frames:
                w = int(fr["w"] * a.render_scale); h = int(fr["h"] * a.render_scale)
                K = torch.tensor(fr["K"], device=dev, dtype=torch.float32).clone()
                K[:2] *= a.render_scale
                W2C = torch.tensor(fr["W2C"], device=dev, dtype=torch.float32)
                _, _, info = rasterization(
                    means=means, quats=qt_t, scales=sc_t.exp(), opacities=opa_t.sigmoid(),
                    colors=colours, viewmats=W2C[None], Ks=K[None], width=w, height=h,
                    sh_degree=sh_deg, packed=True)
                gid = torch.unique(info["gaussian_ids"])
                cam_c = -W2C[:3, :3].T @ W2C[:3, 3]
                d = means[gid] - cam_c            # gaussian seen FROM cam_c
                d = d / (d.norm(dim=-1, keepdim=True) + 1e-9)
                dir_sum.index_add_(0, gid, d)
                cnt.index_add_(0, gid, torch.ones_like(gid, dtype=torch.float32))
        Rlen = (dir_sum.norm(dim=-1) / cnt.clamp(min=1)).cpu().numpy()
        Rlen[cnt.cpu().numpy() == 0] = 1.0        # unseen -> treat as maximally narrow
        damp = Rlen >= a.min_R
        print(f"[sh] resultant length R: median {np.median(Rlen):.3f}, "
              f"p10 {np.percentile(Rlen,10):.3f}")
    print(f"[sh] damping {damp.sum():,}/{N:,} ({100*damp.sum()/N:.1f}%) gaussians "
          f"to diffuse (mode={a.mode})")

    rest_flat = rest.reshape(N, -1).copy()        # coeff-major
    rest_flat[damp] = 0.0
    save_ply_channel_major(a.out, xyz, sc, qt, opa, sh0, rest_flat)
    json.dump({"in": a.ply, "out": a.out, "mode": a.mode, "min_R": a.min_R,
               "n_damped": int(damp.sum()), "n": int(N),
               "R_median": None if Rlen is None else float(np.median(Rlen))},
              open(a.out + ".sh.json", "w"), indent=2)
    print(f"[sh] wrote {a.out}")


if __name__ == "__main__":
    main()

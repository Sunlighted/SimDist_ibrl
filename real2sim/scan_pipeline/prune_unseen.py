"""Prune gaussians that too few TRAINING views actually see.

Why this is needed after excluding the robot frames: 3DGS only constrains a
gaussian where some training camera renders it. Dropping every frame that saw
the robot removes the observations of that volume but NOT the gaussians other
frames happened to grow there, so the region fills with unconstrained junk --
visible as dark floaters exactly where the arm used to be. A gaussian seen by
< min_views training cameras is by definition unsupported by the data, so
deleting it is not cosmetic: it removes content the optimiser never fit.

The alternative (a hand-placed 3D box around the robot) needs the base pose and
a radius guess; visibility needs neither and also cleans floaters elsewhere.

    python prune_unseen.py --ply in.ply --colmap_dir ... --image_dir ... \
        --exclude_idx_ranges 0-1044,2592-3672 --min_views 3 --out pruned.ply
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
    ap.add_argument("--min_views", type=int, default=3)
    ap.add_argument("--render_scale", type=float, default=0.25,
                    help="visibility only needs coarse rasterisation")
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
        kept = []
        for fr in frames:
            nums = re.findall(r"(\d+)", os.path.basename(fr["path"]))
            src = int(nums[-1]) if nums else -1
            if not any(lo <= src <= hi for lo, hi in rr):
                kept.append(fr)
        print(f"[prune] {len(frames)} -> {len(kept)} views after exclusion")
        frames = kept

    xyz, sh0, rest, opa, sc, qt = load_ply_raw(a.ply, "channel")
    N = len(xyz)
    T = lambda x: torch.tensor(x, device=dev)
    means, qt_t, sc_t, opa_t = T(xyz), T(qt), T(sc), T(opa)
    colours = torch.cat([T(sh0).unsqueeze(1), T(rest)], 1)
    sh_deg = int(round(np.sqrt(rest.shape[1] + 1) - 1))
    counts = torch.zeros(N, dtype=torch.int32, device=dev)

    from gsplat.rendering import rasterization
    with torch.no_grad():
        for i, fr in enumerate(frames):
            w = int(fr["w"] * a.render_scale); h = int(fr["h"] * a.render_scale)
            K = torch.tensor(fr["K"], device=dev, dtype=torch.float32).clone()
            K[:2] *= a.render_scale
            _, _, info = rasterization(
                means=means, quats=qt_t, scales=sc_t.exp(), opacities=opa_t.sigmoid(),
                colors=colours,
                viewmats=torch.tensor(fr["W2C"], device=dev, dtype=torch.float32)[None],
                Ks=K[None], width=w, height=h, sh_degree=sh_deg, packed=True)
            gid = torch.unique(info["gaussian_ids"])
            counts[gid] += 1
    keep = (counts >= a.min_views).cpu().numpy()
    print(f"[prune] views per gaussian: median {np.median(counts.cpu().numpy()):.0f}, "
          f"seen by 0 views {int((counts==0).sum()):,}, "
          f"< {a.min_views} views {int(N - keep.sum()):,}")
    print(f"[prune] keeping {keep.sum():,}/{N:,} ({100*keep.sum()/N:.1f}%)")

    save_ply_channel_major(
        a.out, xyz[keep], sc[keep], qt[keep], opa[keep], sh0[keep],
        # load_ply_raw returns rest as (N, n_coeffs, 3) = coeff-major when flattened
        rest[keep].reshape(int(keep.sum()), -1))
    json.dump({"in": a.ply, "out": a.out, "n_in": int(N), "n_out": int(keep.sum()),
               "min_views": a.min_views, "n_views": len(frames)},
              open(a.out + ".prune.json", "w"), indent=2)
    print(f"[prune] wrote {a.out}")


if __name__ == "__main__":
    main()

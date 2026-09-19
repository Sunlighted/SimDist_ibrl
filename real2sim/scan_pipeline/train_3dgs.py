"""Minimal 3DGS trainer on a COLMAP sparse reconstruction.

Uses gsplat's rasterization() directly. Reads COLMAP binary model format
(cameras.bin, images.bin, points3D.bin) as produced by `colmap mapper`.
Outputs a standard .ply that GaussianRenderer's load_ply can consume.

Usage:
  python train_3dgs.py --colmap_dir sfm_scan4/sparse/1 --image_dir sfm_scan4/images \
      --output scene.ply --iterations 7000
"""
import argparse, os, struct, sys, time, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path


# ---- COLMAP binary readers (minimal, no dependencies) ----
def read_cameras_binary(path):
    cams = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid = struct.unpack("<i", f.read(4))[0]
            model = struct.unpack("<i", f.read(4))[0]
            w, h = struct.unpack("<QQ", f.read(16))
            # COLMAP camera-model param counts. The old table had OPENCV(4)=4
            # instead of 8, so the file pointer desynced after the FIRST camera.
            # Harmless with one camera (the loop ends, and fx,fy,cx,cy are the
            # first four params either way) -- but the merged 3-camera model then
            # parsed garbage and died with KeyError on cam_id 2.
            nparams = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12,
                       7: 5, 8: 4, 9: 5, 10: 12}.get(model, 4)
            params = struct.unpack(f"<{nparams}d", f.read(8*nparams))
            cams[cid] = {"model": model, "w": w, "h": h, "params": params}
    return cams

def read_images_binary(path):
    imgs = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid = struct.unpack("<i", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            cid = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00": break
                name += c
            np2d = struct.unpack("<Q", f.read(8))[0]
            f.read(np2d * 24)  # skip 2D points
            imgs[iid] = {"qvec": [qw, qx, qy, qz], "tvec": [tx, ty, tz],
                         "cam_id": cid, "name": name.decode()}
    return imgs

def read_points3D_binary(path):
    pts = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            pid = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            err = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)
            pts[pid] = {"xyz": xyz, "rgb": rgb, "error": err}
    return pts


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])


def load_colmap(sparse_dir, image_dir):
    cams = read_cameras_binary(f"{sparse_dir}/cameras.bin")
    imgs = read_images_binary(f"{sparse_dir}/images.bin")
    pts = read_points3D_binary(f"{sparse_dir}/points3D.bin")

    frames = []
    for iid, im in sorted(imgs.items()):
        c = cams[im["cam_id"]]
        R = qvec2rotmat(im["qvec"])
        t = np.array(im["tvec"])
        W2C = np.eye(4)
        W2C[:3, :3] = R; W2C[:3, 3] = t
        p = c["params"]
        if c["model"] in (0, 1, 2, 4):  # SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, OPENCV
            fx, fy = (p[0], p[0]) if c["model"] in (0, 2) else (p[0], p[1])
            cx = p[1] if c["model"] in (0, 2) else p[2]
            cy = p[2] if c["model"] in (0, 2) else p[3]
        else:
            fx = fy = p[0]; cx = c["w"]/2; cy = c["h"]/2
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        img_path = f"{image_dir}/{im['name']}"
        frames.append({"W2C": W2C, "K": K, "w": c["w"], "h": c["h"], "path": img_path})

    xyz = np.array([p["xyz"] for p in pts.values()], dtype=np.float32)
    rgb = np.array([p["rgb"] for p in pts.values()], dtype=np.float32) / 255.0
    return frames, xyz, rgb


def save_ply(path, means, scales, quats, opacities, sh0, sh_rest=None):
    """Save in the standard 3DGS .ply format that GaussianRenderer's load_ply reads."""
    from plyfile import PlyData, PlyElement
    N = len(means)
    names = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3): names.append(f"f_dc_{i}")
    n_rest = sh_rest.shape[1] if sh_rest is not None else 0
    for i in range(n_rest): names.append(f"f_rest_{i}")
    names += ["opacity"]
    for i in range(3): names.append(f"scale_{i}")
    for i in range(4): names.append(f"rot_{i}")

    arr = np.zeros(N, dtype=[(n, "f4") for n in names])
    arr["x"], arr["y"], arr["z"] = means[:, 0], means[:, 1], means[:, 2]
    for i in range(3): arr[f"f_dc_{i}"] = sh0[:, i]
    if sh_rest is not None:
        for i in range(n_rest): arr[f"f_rest_{i}"] = sh_rest[:, i]
    arr["opacity"] = opacities
    for i in range(3): arr[f"scale_{i}"] = scales[:, i]
    for i in range(4): arr[f"rot_{i}"] = quats[:, i]

    el = PlyElement.describe(arr, "vertex")
    PlyData([el]).write(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--colmap_dir", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--output", default="scene.ply")
    p.add_argument("--iterations", type=int, default=7000)
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--lr_means", type=float, default=1.6e-4)
    p.add_argument("--lr_scales", type=float, default=5e-3)
    p.add_argument("--lr_quats", type=float, default=1e-3)
    p.add_argument("--lr_opacities", type=float, default=5e-2)
    p.add_argument("--lr_sh", type=float, default=2.5e-3)
    p.add_argument("--densify_interval", type=int, default=100)
    p.add_argument("--densify_until", type=int, default=5000)
    p.add_argument("--densify_grad_thresh", type=float, default=2e-4)
    p.add_argument("--max_gaussians", type=int, default=500000)
    p.add_argument("--image_scale", type=float, default=0.5, help="downscale factor for training images")
    a = p.parse_args()

    device = torch.device("cuda")
    frames, init_xyz, init_rgb = load_colmap(a.colmap_dir, a.image_dir)
    print(f"[3dgs] {len(frames)} frames, {len(init_xyz)} initial points")

    # Initialize gaussians from COLMAP points
    N = len(init_xyz)
    means = torch.tensor(init_xyz, device=device, requires_grad=True)
    # SH DC from RGB (the C0 = 0.2820948... conversion)
    C0 = 0.28209479177387814
    sh0 = torch.tensor((init_rgb - 0.5) / C0, device=device, dtype=torch.float32, requires_grad=True)
    n_sh_rest = (a.sh_degree + 1)**2 - 1
    sh_rest = torch.zeros(N, n_sh_rest * 3, device=device, requires_grad=True)
    # Scales from local point density
    from scipy.spatial import cKDTree
    tree = cKDTree(init_xyz)
    dists, _ = tree.query(init_xyz, k=4)
    avg_dist = np.mean(dists[:, 1:], axis=1).clip(1e-5)
    scales = torch.tensor(np.log(np.stack([avg_dist]*3, -1)), device=device, dtype=torch.float32, requires_grad=True)
    quats = torch.zeros(N, 4, device=device); quats[:, 0] = 1.0; quats.requires_grad_(True)
    opacities_raw = torch.full((N,), 0.1, device=device, requires_grad=True)  # pre-sigmoid

    params = [
        {"params": [means], "lr": a.lr_means, "name": "means"},
        {"params": [scales], "lr": a.lr_scales, "name": "scales"},
        {"params": [quats], "lr": a.lr_quats, "name": "quats"},
        {"params": [opacities_raw], "lr": a.lr_opacities, "name": "opacities"},
        {"params": [sh0], "lr": a.lr_sh, "name": "sh0"},
        {"params": [sh_rest], "lr": a.lr_sh / 20, "name": "sh_rest"},
    ]
    opt = torch.optim.Adam(params)

    # Preload and downscale images
    import cv2
    gt_images = []
    for fr in frames:
        img = cv2.imread(fr["path"])
        if img is None:
            print(f"WARNING: cannot read {fr['path']}"); continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if a.image_scale != 1.0:
            h, w = int(img.shape[0] * a.image_scale), int(img.shape[1] * a.image_scale)
            img = cv2.resize(img, (w, h))
        gt_images.append(torch.tensor(img, device=device))
    print(f"[3dgs] loaded {len(gt_images)} images at {gt_images[0].shape}")

    # Precompute camera tensors
    viewmats, Ks, whs = [], [], []
    for i, fr in enumerate(frames):
        vm = torch.tensor(fr["W2C"], device=device, dtype=torch.float32)
        K = torch.tensor(fr["K"], device=device, dtype=torch.float32)
        if a.image_scale != 1.0:
            K[:2] *= a.image_scale
        viewmats.append(vm); Ks.append(K)
        h, w = gt_images[i].shape[:2]
        whs.append((w, h))

    from gsplat.rendering import rasterization
    grad_accum = torch.zeros(N, device=device)
    grad_count = torch.zeros(N, device=device)

    t0 = time.time()
    for step in range(1, a.iterations + 1):
        idx = np.random.randint(len(gt_images))
        gt = gt_images[idx]
        w, h = whs[idx]

        colors_sh = torch.cat([sh0.unsqueeze(1),
                                sh_rest.reshape(-1, n_sh_rest, 3)], dim=1)
        renders, alphas, meta = rasterization(
            means=means,
            quats=quats / (quats.norm(dim=-1, keepdim=True) + 1e-8),
            scales=scales.exp(),
            opacities=opacities_raw.sigmoid(),
            colors=colors_sh,
            viewmats=viewmats[idx][None],
            Ks=Ks[idx][None],
            width=w, height=h,
            sh_degree=a.sh_degree,
        )
        rendered = renders[0]  # (H, W, 3)
        loss = F.l1_loss(rendered, gt) + 0.2 * (1 - _ssim(rendered, gt))
        loss.backward()

        # Accumulate gradients for densification
        if step < a.densify_until and means.grad is not None:
            gnorm = means.grad.norm(dim=-1)
            has_grad = gnorm > 0
            grad_accum[has_grad] += gnorm[has_grad]
            grad_count[has_grad] += 1

        opt.step(); opt.zero_grad(set_to_none=True)

        # Densify
        if step % a.densify_interval == 0 and step < a.densify_until and len(means) < a.max_gaussians:
            avg_grad = grad_accum / (grad_count + 1e-8)
            mask = avg_grad > a.densify_grad_thresh
            n_new = mask.sum().item()
            if n_new > 0 and len(means) + n_new < a.max_gaussians:
                # Clone (split would be better but clone is simpler)
                new_means = means.data[mask].clone().requires_grad_(True)
                new_sh0 = sh0.data[mask].clone().requires_grad_(True)
                new_sh_rest = sh_rest.data[mask].clone().requires_grad_(True)
                new_scales = (scales.data[mask] - 0.5).clone().requires_grad_(True)
                new_quats = quats.data[mask].clone().requires_grad_(True)
                new_opa = opacities_raw.data[mask].clone().requires_grad_(True)

                means = torch.cat([means.data, new_means.data]).requires_grad_(True)
                sh0 = torch.cat([sh0.data, new_sh0.data]).requires_grad_(True)
                sh_rest = torch.cat([sh_rest.data, new_sh_rest.data]).requires_grad_(True)
                scales = torch.cat([scales.data, new_scales.data]).requires_grad_(True)
                quats = torch.cat([quats.data, new_quats.data]).requires_grad_(True)
                opacities_raw = torch.cat([opacities_raw.data, new_opa.data]).requires_grad_(True)

                params = [
                    {"params": [means], "lr": a.lr_means},
                    {"params": [scales], "lr": a.lr_scales},
                    {"params": [quats], "lr": a.lr_quats},
                    {"params": [opacities_raw], "lr": a.lr_opacities},
                    {"params": [sh0], "lr": a.lr_sh},
                    {"params": [sh_rest], "lr": a.lr_sh / 20},
                ]
                opt = torch.optim.Adam(params)
                grad_accum = torch.zeros(len(means), device=device)
                grad_count = torch.zeros(len(means), device=device)

            # Prune low-opacity
            alive = opacities_raw.data.sigmoid() > 0.005
            if not alive.all():
                means = means.data[alive].clone().requires_grad_(True)
                sh0 = sh0.data[alive].clone().requires_grad_(True)
                sh_rest = sh_rest.data[alive].clone().requires_grad_(True)
                scales = scales.data[alive].clone().requires_grad_(True)
                quats = quats.data[alive].clone().requires_grad_(True)
                opacities_raw = opacities_raw.data[alive].clone().requires_grad_(True)
                params = [
                    {"params": [means], "lr": a.lr_means},
                    {"params": [scales], "lr": a.lr_scales},
                    {"params": [quats], "lr": a.lr_quats},
                    {"params": [opacities_raw], "lr": a.lr_opacities},
                    {"params": [sh0], "lr": a.lr_sh},
                    {"params": [sh_rest], "lr": a.lr_sh / 20},
                ]
                opt = torch.optim.Adam(params)
                grad_accum = torch.zeros(len(means), device=device)
                grad_count = torch.zeros(len(means), device=device)

        if step % 500 == 0 or step == 1:
            elapsed = time.time() - t0
            psnr = -10 * torch.log10(F.mse_loss(rendered.detach(), gt) + 1e-8).item()
            print(f"[3dgs] step {step:5d}/{a.iterations}  loss={loss.item():.4f}  "
                  f"psnr={psnr:.1f}  n_gauss={len(means):,}  {elapsed:.0f}s")

    # Save
    save_ply(a.output,
             means.detach().cpu().numpy(),
             scales.detach().cpu().numpy(),
             (quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)).detach().cpu().numpy(),
             opacities_raw.detach().cpu().numpy(),   # raw logit, load_ply applies sigmoid
             sh0.detach().cpu().numpy(),
             sh_rest.detach().cpu().numpy())
    print(f"[3dgs] saved {a.output} ({len(means):,} gaussians)")


def _ssim(x, y, window_size=11):
    """Simplified SSIM on HWC tensors."""
    C1, C2 = 0.01**2, 0.03**2
    x = x.permute(2, 0, 1).unsqueeze(0)
    y = y.permute(2, 0, 1).unsqueeze(0)
    k = window_size
    w = torch.ones(1, 1, k, k, device=x.device) / (k * k)
    w = w.expand(3, -1, -1, -1)
    mu_x = F.conv2d(x, w, groups=3, padding=k//2)
    mu_y = F.conv2d(y, w, groups=3, padding=k//2)
    s_x = F.conv2d(x*x, w, groups=3, padding=k//2) - mu_x*mu_x
    s_y = F.conv2d(y*y, w, groups=3, padding=k//2) - mu_y*mu_y
    s_xy = F.conv2d(x*y, w, groups=3, padding=k//2) - mu_x*mu_y
    ssim = ((2*mu_x*mu_y + C1)*(2*s_xy + C2)) / ((mu_x**2+mu_y**2+C1)*(s_x+s_y+C2))
    return ssim.mean()


if __name__ == "__main__":
    main()

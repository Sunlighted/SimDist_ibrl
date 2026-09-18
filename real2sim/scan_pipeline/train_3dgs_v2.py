"""3DGS trainer v2 -- fixes the reasons the v1 splats came out blurry/washed out.

What changed vs train_3dgs.py (v1), and why each one matters here:

 1. DENSIFICATION: v1 hand-rolled a CLONE-ONLY densifier and **rebuilt the Adam
    optimizer every 100 steps** (on every densify and every prune), which throws
    away all momentum/second-moment state. It also never SPLIT large gaussians,
    and split is precisely the operation that turns one blurry blob into sharp
    detail. v2 uses gsplat's DefaultStrategy (clone + split + prune +
    opacity reset, with per-parameter optimizer state correctly carried over).
    Symptom this explains: scan5 finished at 154k gaussians from 498 images
    while its 800k cap was never approached.

 2. POSITION LEARNING RATE: standard 3DGS scales the means LR by the scene
    extent (`1.6e-4 * radius`) because the gradient magnitude is in scene units.
    COLMAP units here are ~0.1 m/unit, so the fixed 1.6e-4 of v1 was ~10x too
    small -- gaussians barely moved from their SfM init. v2 computes the camera
    extent and decays the LR exponentially (1.6e-4 -> 1.6e-6, times extent).

 3. SH ORDERING BUG (silent, affects every v1 ply): v1 saved `f_rest` in
    COEFFICIENT-major order, but `util_gau.load_ply` (and the INRIA convention,
    and every standard viewer) reads it CHANNEL-major. So all degree>=1 SH terms
    were permuted at load time -- base colour right, every view-dependent term
    wrong. v2 writes channel-major.

 4. BACKGROUND COLOUR: rasterizes onto the measured WALL colour instead of
    black, so any region the scan never observed (a white wall gives SfM no
    features and gives 3DGS no photometric gradient, so no gaussian is ever
    created there) reads as wall rather than a black hole. Used during TRAINING
    too, so the optimizer is not fighting a black canvas it must paint over.

 5. PER-IMAGE EXPOSURE COMPENSATION: we now merge three sweeps from two phones
    with independent auto-exposure. Without a per-image affine, 3DGS averages
    conflicting brightness for the same surface into muddy low-contrast
    gaussians. Learned per image, applied only to the training render; the saved
    ply stays exposure-neutral.

 6. SH DEGREE WARM-UP + honest HELD-OUT EVAL (every Nth frame excluded from
    training) so "sharper" is measured, not asserted.

    python train_3dgs_v2.py --colmap_dir sfm_merged/sparse/0 \
        --image_dir sfm_merged/images --output sfm_merged/scene_v2.ply \
        --iterations 30000 --bg_color_json wall_color.json
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from train_3dgs import load_colmap, read_cameras_binary, read_images_binary  # noqa: F401


C0 = 0.28209479177387814


def save_ply_channel_major(path, means, scales, quats, opacities, sh0, sh_rest_coeff_major):
    """Write a standard 3DGS ply.

    `sh_rest_coeff_major` is (N, n_coeffs*3) laid out as coeff-major (the layout
    gsplat's rasterization consumes after .reshape(-1, n_coeffs, 3)). The ply
    format wants CHANNEL-major, so transpose before writing -- see header note 3.
    """
    from plyfile import PlyData, PlyElement

    N = len(means)
    n3 = sh_rest_coeff_major.shape[1]
    n_coeffs = n3 // 3
    sh_rest = (
        sh_rest_coeff_major.reshape(N, n_coeffs, 3).transpose(0, 2, 1).reshape(N, n3)
    )

    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(n3)]
    names += ["opacity"]
    names += [f"scale_{i}" for i in range(3)]
    names += [f"rot_{i}" for i in range(4)]

    arr = np.zeros(N, dtype=[(n, "f4") for n in names])
    arr["x"], arr["y"], arr["z"] = means[:, 0], means[:, 1], means[:, 2]
    for i in range(3):
        arr[f"f_dc_{i}"] = sh0[:, i]
    for i in range(n3):
        arr[f"f_rest_{i}"] = sh_rest[:, i]
    arr["opacity"] = opacities
    for i in range(3):
        arr[f"scale_{i}"] = scales[:, i]
    for i in range(4):
        arr[f"rot_{i}"] = quats[:, i]
    PlyData([PlyElement.describe(arr, "vertex")]).write(path)


def camera_extent(frames):
    """Radius of the camera-centre cloud (standard 3DGS `spatial_lr_scale`)."""
    centres = []
    for fr in frames:
        W2C = fr["W2C"]
        centres.append(-W2C[:3, :3].T @ W2C[:3, 3])
    C = np.stack(centres)
    return float(np.linalg.norm(C - C.mean(0), axis=1).max() * 1.1), C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--colmap_dir", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--output", default="scene_v2.ply")
    p.add_argument("--iterations", type=int, default=30000)
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--image_scale", type=float, default=0.5)
    p.add_argument("--max_long_side", type=int, default=0,
                   help="if >0, scale EACH image so its long side is this many px "
                        "(overrides --image_scale). Matters for the merged model: a "
                        "fixed scale would train the 4K sweep at 1920px and the 1080p "
                        "aux sweeps at 960px, i.e. half the pixel density on the same "
                        "surfaces, which biases detail toward whichever sweep is bigger.")
    p.add_argument("--val_render_dir", default="",
                   help="dump held-out render/GT pairs here (look at sharpness, "
                        "do not trust PSNR alone)")
    p.add_argument("--holdout_stride", type=int, default=8,
                   help="every Nth frame is held out of training for honest eval")
    p.add_argument("--bg_color_json", default="", help="wall_color.json from the scans")
    p.add_argument("--bg_color", default="", help="r,g,b in 0-1, overrides the json")
    p.add_argument("--exposure", type=int, default=1, help="per-image affine exposure comp")
    p.add_argument("--max_gaussians", type=int, default=2_500_000)
    p.add_argument("--absgrad", type=int, default=1)
    p.add_argument("--refine_stop_frac", type=float, default=0.5,
                   help="stop densifying after this fraction of iterations")
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--world_transform", default="",
                   help="similarity json (scale,R,t) from calibrate_to_board.py. When "
                        "given, the COLMAP points AND camera poses are mapped into the "
                        "metric board frame BEFORE training, so the output ply is "
                        "already world-frame and metric. This also removes the need to "
                        "rotate SH coefficients at load time -- rotating only xyz/quats "
                        "(what render_peg_splat.py did) leaves every view-dependent SH "
                        "term expressed in the old frame.")
    a = p.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(0)
    np.random.seed(0)

    # ---- background colour (the wall) ----
    if a.bg_color:
        bg = [float(x) for x in a.bg_color.split(",")]
    elif a.bg_color_json:
        bg = json.load(open(a.bg_color_json))["wall_rgb"]
    else:
        bg = [0.0, 0.0, 0.0]
    # gsplat wants backgrounds as image_dims + (channels,); with UNBATCHED means
    # (N,3) the projection returns means2d (N,2) => image_dims == () => shape (3,).
    # Probed on-GPU: (1,3) and (1,1,3) both assert (logs/probe_bg_5518250.out).
    bg_t = torch.tensor(bg, dtype=torch.float32, device=device)
    print(f"[v2] background = {np.round(bg, 4).tolist()}")

    frames, init_xyz, init_rgb = load_colmap(a.colmap_dir, a.image_dir)

    if a.world_transform:
        wt = json.load(open(a.world_transform))
        s_w, R_w, t_w = float(wt["scale"]), np.array(wt["R"]), np.array(wt["t"])
        init_xyz = (s_w * (init_xyz @ R_w.T) + t_w).astype(np.float32)
        # X_w = s R X_c + t  and  x_cam = R_c X_c + t_c
        #   => s * x_cam = (R_c R^T) X_w + (s t_c - R_c R^T t)
        # Scaling camera space by a constant leaves the perspective projection
        # unchanged, so the bracketed pair is a RIGID viewmat for the world frame.
        for fr in frames:
            R_c, t_c = fr["W2C"][:3, :3], fr["W2C"][:3, 3]
            Rn = R_c @ R_w.T
            tn = s_w * t_c - Rn @ t_w
            W2C = np.eye(4); W2C[:3, :3] = Rn; W2C[:3, 3] = tn
            fr["W2C"] = W2C
        print(f"[v2] world transform applied: scale {s_w:.5f} m/unit, "
              f"residual {wt.get('res_med_mm', float('nan')):.2f} mm -> "
              f"ply will be METRIC in the board frame")

    extent, centres = camera_extent(frames)
    print(f"[v2] {len(frames)} frames, {len(init_xyz)} init points, "
          f"camera extent {extent:.3f} colmap-units")

    # ---- train / held-out split ----
    idx_all = np.arange(len(frames))
    held = idx_all[:: a.holdout_stride] if a.holdout_stride > 1 else np.array([], int)
    train_idx = np.setdiff1d(idx_all, held)
    print(f"[v2] {len(train_idx)} train / {len(held)} held-out frames")

    # ---- gaussian params ----
    N = len(init_xyz)
    from scipy.spatial import cKDTree
    d2, _ = cKDTree(init_xyz).query(init_xyz, k=4)
    avg = np.mean(d2[:, 1:], axis=1).clip(1e-6)

    means = torch.nn.Parameter(torch.tensor(init_xyz, device=device))
    scales = torch.nn.Parameter(
        torch.tensor(np.log(np.stack([avg] * 3, -1)), device=device, dtype=torch.float32))
    quats = torch.nn.Parameter(
        torch.tensor(np.tile([1.0, 0, 0, 0], (N, 1)), device=device, dtype=torch.float32))
    opacities = torch.nn.Parameter(
        torch.full((N,), math.log(0.1 / 0.9), device=device))  # logit(0.1), 3DGS default
    sh0 = torch.nn.Parameter(
        torch.tensor((init_rgb - 0.5) / C0, device=device, dtype=torch.float32))
    n_rest = (a.sh_degree + 1) ** 2 - 1
    shN = torch.nn.Parameter(torch.zeros(N, n_rest * 3, device=device))

    params = {"means": means, "scales": scales, "quats": quats,
              "opacities": opacities, "sh0": sh0, "shN": shN}
    lr = {"means": 1.6e-4 * extent, "scales": 5e-3, "quats": 1e-3,
          "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{"params": [v], "lr": lr[k], "name": k}], eps=1e-15)
                  for k, v in params.items()}
    # exponential decay on POSITION lr only (standard 3DGS)
    sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / a.iterations))

    from gsplat.strategy import DefaultStrategy
    strategy = DefaultStrategy(
        absgrad=bool(a.absgrad),
        grow_grad2d=8e-4 if a.absgrad else 2e-4,   # gsplat's documented pairing
        refine_stop_iter=int(a.iterations * a.refine_stop_frac),
        reset_every=3000,
        refine_every=100,
        verbose=False,
    )
    strategy.check_sanity(params, optimizers)
    # scene_scale is load-bearing: grow_scale3d / prune_scale3d are FRACTIONS of it
    # (split above 1% of scale, prune above 10%). Leaving it at the default 1.0
    # silently means "1 unit" -- fine by luck in COLMAP units, but once
    # --world_transform makes the scene metric it would prune every gaussian
    # bigger than 10 cm, i.e. the legitimately large wall/table ones.
    strategy_state = strategy.initialize_state(scene_scale=extent)

    # ---- images ----
    import cv2
    gt, whs, viewmats, Ks = [], [], [], []
    sizes = {}
    for fr in frames:
        img = cv2.imread(fr["path"])
        assert img is not None, fr["path"]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h0, w0 = img.shape[:2]
        s = (min(1.0, a.max_long_side / max(h0, w0)) if a.max_long_side
             else a.image_scale)
        if s != 1.0:
            img = cv2.resize(img, (int(round(w0 * s)), int(round(h0 * s))),
                             interpolation=cv2.INTER_AREA)
        gt.append(torch.tensor(img, device=device))
        whs.append((img.shape[1], img.shape[0]))
        K = torch.tensor(fr["K"], device=device, dtype=torch.float32).clone()
        if s != 1.0:
            K[:2] *= s
        Ks.append(K)
        viewmats.append(torch.tensor(fr["W2C"], device=device, dtype=torch.float32))
        sizes[(img.shape[1], img.shape[0])] = sizes.get((img.shape[1], img.shape[0]), 0) + 1
    print(f"[v2] image sizes: {sizes}")

    # ---- per-image exposure (log-gain + bias per channel), train frames only ----
    exp_gain = torch.nn.Parameter(torch.zeros(len(frames), 3, device=device))
    exp_bias = torch.nn.Parameter(torch.zeros(len(frames), 3, device=device))
    exp_opt = torch.optim.Adam([exp_gain, exp_bias], lr=1e-3)

    from gsplat.rendering import rasterization

    def render(i, sh_deg):
        w, h = whs[i]
        colours = torch.cat([sh0.unsqueeze(1), shN.reshape(-1, n_rest, 3)], 1)
        # packed=True is REQUIRED with DefaultStrategy in gsplat 1.5.3: the
        # unpacked path returns means2d/radii squeezed to (N,2) for a single
        # camera, and the strategy's unpacked branch does torch.where(sel)[1],
        # which needs a [C, N] mask -> IndexError. Packed mode carries explicit
        # gaussian_ids, so the strategy indexes correctly (and uses less VRAM).
        # Verified on-GPU: logs/probe_st_5518335.out (200 -> 470 gaussians).
        out, alpha, info = rasterization(
            means=means, quats=quats, scales=scales.exp(),
            opacities=opacities.sigmoid(), colors=colours,
            viewmats=viewmats[i][None], Ks=Ks[i][None], width=w, height=h,
            sh_degree=sh_deg, backgrounds=bg_t, absgrad=bool(a.absgrad),
            packed=True,
        )
        return out[0], alpha, info

    @torch.no_grad()
    def evaluate(tag, step):
        if len(held) == 0:
            return float("nan")
        ps = []
        for i in held:
            img, _, _ = render(int(i), a.sh_degree)
            mse = F.mse_loss(img.clamp(0, 1), gt[int(i)])
            ps.append(-10 * torch.log10(mse + 1e-10).item())
        print(f"[v2] {tag} step {step}: HELD-OUT psnr {np.mean(ps):.2f} "
              f"(n={len(ps)}, n_gauss={len(means):,})", flush=True)
        return float(np.mean(ps))

    t0 = time.time()
    for step in range(1, a.iterations + 1):
        # SH degree warm-up: 0 for the first 1k, then +1 every 1k steps
        sh_deg = min(a.sh_degree, step // 1000)
        i = int(np.random.choice(train_idx))
        img, alpha, info = render(i, sh_deg)

        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

        pred = img
        if a.exposure:
            pred = pred * exp_gain[i].exp() + exp_bias[i]
        loss = F.l1_loss(pred, gt[i]) + 0.2 * (1 - _ssim(pred.clamp(0, 1), gt[i]))
        loss.backward()

        for o in optimizers.values():
            o.step(); o.zero_grad(set_to_none=True)
        if a.exposure:
            exp_opt.step(); exp_opt.zero_grad(set_to_none=True)
        sched.step()

        # cap: stop growing (but keep pruning/refining scale) once over budget
        if len(means) > a.max_gaussians and strategy.refine_stop_iter > step:
            print(f"[v2] gaussian cap {a.max_gaussians:,} hit at step {step}; "
                  f"stopping densification")
            strategy.refine_stop_iter = step
        strategy.step_post_backward(params, optimizers, strategy_state, step, info,
                                   packed=True)
        # the strategy may have replaced the tensors in `params`
        means, scales, quats = params["means"], params["scales"], params["quats"]
        opacities, sh0, shN = params["opacities"], params["sh0"], params["shN"]

        if step % 1000 == 0 or step == 1:
            psnr = -10 * torch.log10(
                F.mse_loss(pred.detach().clamp(0, 1), gt[i]) + 1e-10).item()
            print(f"[v2] step {step:6d}/{a.iterations} loss={loss.item():.4f} "
                  f"train_psnr={psnr:.1f} n_gauss={len(means):,} "
                  f"lr_means={optimizers['means'].param_groups[0]['lr']:.2e} "
                  f"{time.time()-t0:.0f}s", flush=True)
        if step % a.eval_every == 0:
            evaluate("eval", step)

    final = evaluate("FINAL", a.iterations)

    if a.val_render_dir:
        os.makedirs(a.val_render_dir, exist_ok=True)
        with torch.no_grad():
            for k, i in enumerate(held[:: max(1, len(held) // 6)][:6]):
                img, _, _ = render(int(i), a.sh_degree)
                pair = torch.cat([gt[int(i)], img.clamp(0, 1)], 1).cpu().numpy()
                cv2.imwrite(f"{a.val_render_dir}/held_{k}_gt_vs_render.png",
                            cv2.cvtColor((pair * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        print(f"[v2] wrote held-out GT|render pairs to {a.val_render_dir}")

    save_ply_channel_major(
        a.output,
        means.detach().cpu().numpy(),
        scales.detach().cpu().numpy(),
        (quats / quats.norm(dim=-1, keepdim=True)).detach().cpu().numpy(),
        opacities.detach().cpu().numpy(),
        sh0.detach().cpu().numpy(),
        shN.detach().cpu().numpy(),
    )
    meta = {"n_gaussians": int(len(means)), "held_out_psnr": final,
            "iterations": a.iterations, "bg_rgb": bg, "extent": extent,
            "image_scale": a.image_scale, "colmap_dir": a.colmap_dir,
            "exposure_comp": bool(a.exposure), "sh_degree": a.sh_degree}
    json.dump(meta, open(a.output + ".json", "w"), indent=2)
    print(f"[v2] saved {a.output} ({len(means):,} gaussians, held-out psnr {final:.2f})")


def _ssim(x, y, k=11):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    x = x.permute(2, 0, 1).unsqueeze(0); y = y.permute(2, 0, 1).unsqueeze(0)
    w = torch.ones(3, 1, k, k, device=x.device) / (k * k)
    mx = F.conv2d(x, w, groups=3, padding=k // 2)
    my = F.conv2d(y, w, groups=3, padding=k // 2)
    sx = F.conv2d(x * x, w, groups=3, padding=k // 2) - mx * mx
    sy = F.conv2d(y * y, w, groups=3, padding=k // 2) - my * my
    sxy = F.conv2d(x * y, w, groups=3, padding=k // 2) - mx * my
    return (((2 * mx * my + C1) * (2 * sxy + C2))
            / ((mx ** 2 + my ** 2 + C1) * (sx + sy + C2))).mean()


if __name__ == "__main__":
    main()

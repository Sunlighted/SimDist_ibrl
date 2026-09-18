"""Render a world-frame splat from the REAL ZED2i viewpoint, beside the real photo.

This is the most legible test of "is the room still blurry": the pose comes from
the PnP solve against the ChArUco (`zed2i_new_extrinsic.json`, 1.06 px median
reprojection, 31 inliers), the intrinsics from the ZED manifest, so the rendered
view and the photograph are the same camera -- any softness left is the splat's,
not a viewpoint mismatch.

Renders the gaussians THROUGH gsplat directly (no sim, no compositing) with the
measured wall colour as background, and writes real | render | abs-diff strips.

    python render_zed_pov.py --ply sfm_merged/scene_v2.ply --out pov_v2
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from eval_splat import load_ply_raw

GS = "/scratch/yifanr/gsbench"


def zed_intrinsics(photo_w, photo_h):
    """ZED manifest intrinsics are for the 1280x720 mode; the Explorer stills are
    HD2K (2208x1242). Same FOV across modes, so scale by the width ratio -- the
    trap here is using the 720p numbers directly on an HD2K image."""
    m = json.load(open("/home/yifanr/scan_run2/intrinsics.json"))["zed2i"]["left"]
    s = photo_w / 1280.0
    return (m["fx"] * s, m["fy"] * s, m["cx"] * s, m["cy"] * s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extrinsic", default=f"{GS}/zed2i_extrinsic_boardframe.json")
    ap.add_argument("--photo", default="/home/yifanr/Explorer_HD2K_SN37597132_17-54-59.png")
    ap.add_argument("--bg_color_json", default=f"{GS}/wall_color.json")
    ap.add_argument("--sh_layout", default="channel")
    ap.add_argument("--scale", type=float, default=0.5, help="render at this fraction of the photo")
    ap.add_argument("--ply_world_transform", default="",
                   help="use when the PLY is still in COLMAP frame (v1 splats): the "
                        "similarity json that maps that colmap frame to the board "
                        "frame. The CAMERA is mapped into the ply's frame rather than "
                        "the ply into the camera's -- moving the gaussians would "
                        "require rotating their SH coefficients too, which is exactly "
                        "the error we are trying not to introduce into a baseline.")
    a = ap.parse_args()

    out = a.out if a.out.startswith("/") else f"{GS}/{a.out}"
    os.makedirs(out, exist_ok=True)
    dev = torch.device("cuda")

    photo = cv2.imread(a.photo)
    assert photo is not None, a.photo
    # ZED Explorer stills are SIDE-BY-SIDE stereo (4416x1242 for HD2K). The
    # extrinsic/intrinsics are the LEFT camera's, so take the left half --
    # otherwise the width doubles, cx lands mid-baseline and the render is blank.
    if photo.shape[1] >= 2 * photo.shape[0]:
        photo = photo[:, : photo.shape[1] // 2]
        print(f"[pov] split stereo -> left {photo.shape[1]}x{photo.shape[0]}")
    H0, W0 = photo.shape[:2]
    fx, fy, cx, cy = zed_intrinsics(W0, H0)
    W, H = int(W0 * a.scale), int(H0 * a.scale)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]) * a.scale
    K[2, 2] = 1.0
    photo_s = cv2.resize(photo, (W, H), interpolation=cv2.INTER_AREA)

    # T_world_cam -> viewmat (world->cam)
    ext = json.load(open(a.extrinsic))
    # accept either key: zed2i_new_extrinsic.json (parallel session) or the
    # board-frame solve written here. NOTE the former reprojects 1344 px against
    # board_model.json, so it is NOT interchangeable -- see the note in the json.
    T_wc = np.array(ext.get("T_world_zed2i", ext.get("T_world_zed2i_new")))
    print(f"[pov] extrinsic {os.path.basename(a.extrinsic)} "
          f"(reproj {ext.get('reproj_med_px', float('nan')):.2f} px)")
    W2C = np.linalg.inv(T_wc)

    if a.ply_world_transform:
        # ply frame -> board frame is X_b = s R X_p + t, and the camera is given in
        # board frame. Composing and then rescaling camera space (projection is
        # invariant to a global camera-space scale) gives a RIGID world->cam for
        # the ply's own frame:  A = R_cw R,  a = (R_cw t + t_cw) / s
        wt = json.load(open(a.ply_world_transform))
        s_w, R_w, t_w = float(wt["scale"]), np.array(wt["R"]), np.array(wt["t"])
        R_cw, t_cw = W2C[:3, :3], W2C[:3, 3]
        W2C = np.eye(4)
        W2C[:3, :3] = R_cw @ R_w
        W2C[:3, 3] = (R_cw @ t_w + t_cw) / s_w
        print(f"[pov] camera mapped into the ply's frame (scale {s_w:.5f})")

    xyz, sh0, rest, opa, sc, qt = load_ply_raw(a.ply, a.sh_layout)
    bg = json.load(open(a.bg_color_json))["wall_rgb"]
    T = lambda x: torch.tensor(x, device=dev)
    colours = torch.cat([T(sh0).unsqueeze(1), T(rest)], 1)
    sh_degree = int(round(np.sqrt(rest.shape[1] + 1) - 1))
    print(f"[pov] {len(xyz):,} gaussians, sh_degree {sh_degree}, "
          f"render {W}x{H}, bg {np.round(bg,3).tolist()}")

    from gsplat.rendering import rasterization
    with torch.no_grad():
        img, alpha, _ = rasterization(
            means=T(xyz), quats=T(qt), scales=T(sc).exp(), opacities=T(opa).sigmoid(),
            colors=colours,
            viewmats=torch.tensor(W2C, device=dev, dtype=torch.float32)[None],
            Ks=torch.tensor(K, device=dev, dtype=torch.float32)[None],
            width=W, height=H, sh_degree=sh_degree,
            backgrounds=torch.tensor(bg, device=dev, dtype=torch.float32),
        )
    render = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
    cover = float((alpha[0, ..., 0] > 0.5).float().mean().item())

    diff = cv2.absdiff(photo_s, render)
    strip = np.concatenate([photo_s, render, diff], 1)
    tag = os.path.basename(a.ply).replace(".ply", "")
    cv2.imwrite(f"{out}/{tag}_real.png", photo_s)
    cv2.imwrite(f"{out}/{tag}_render.png", render)
    cv2.imwrite(f"{out}/{tag}_strip.png", strip)

    # sharpness proxies: high-frequency energy of render vs photo (same view, so
    # comparable). A splat that is merely SMOOTH scores far below the photo.
    def hf(im):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())
    print(f"[pov] alpha coverage {cover*100:.1f}%  lapvar photo {hf(photo_s):.0f} "
          f"render {hf(render):.0f} (ratio {hf(render)/max(hf(photo_s),1e-6):.2f})  "
          f"mean|diff| {diff.mean():.1f}/255")
    print(f"[pov] wrote {out}/{tag}_strip.png")


if __name__ == "__main__":
    main()

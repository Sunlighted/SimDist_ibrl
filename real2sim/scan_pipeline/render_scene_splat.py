"""Render the trained scene splat from a couple of the COLMAP camera poses.
Quick visual check before integrating into the full pipeline."""
import os, sys, struct, json, argparse
os.environ.setdefault("MUJOCO_GL", "disable")
import numpy as np
import torch

sys.path.insert(0, "/scratch/yifanr/realsim_survey/_GaussianRenderer/src")
from gaussian_renderer.core.util_gau import load_ply
from gaussian_renderer.core.batch_rasterization import batch_env_render
from gaussian_renderer.core.gaussiandata import GaussianBatchData


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
            f.read(np2d * 24)
            imgs[iid] = {"qvec": [qw, qx, qy, qz], "tvec": [tx, ty, tz],
                         "cam_id": cid, "name": name.decode()}
    return imgs


def read_cameras_binary(path):
    cams = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid = struct.unpack("<i", f.read(4))[0]
            model = struct.unpack("<i", f.read(4))[0]
            w, h = struct.unpack("<QQ", f.read(16))
            nparams = {0:3,1:4,2:4,3:5,4:4,5:5,6:8,7:12,8:12,9:12}.get(model, 4)
            params = struct.unpack(f"<{nparams}d", f.read(8*nparams))
            cams[cid] = {"model": model, "w": w, "h": h, "params": params}
    return cams


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ply", default="/scratch/yifanr/gsbench/sfm_scan4/scene.ply")
    p.add_argument("--colmap_dir", default="/scratch/yifanr/gsbench/sfm_scan4/sparse/1")
    p.add_argument("--out_dir", default="/scratch/yifanr/gsbench/scene_renders")
    p.add_argument("--n_views", type=int, default=4)
    p.add_argument("--res", type=int, default=512)
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device("cuda")

    g = load_ply(a.ply)
    N = len(g.xyz)
    print(f"loaded {N:,} gaussians from {a.ply}")

    gsb = GaussianBatchData(
        xyz=torch.tensor(g.xyz, device=dev, dtype=torch.float32).unsqueeze(0),
        rot=torch.tensor(g.rot, device=dev, dtype=torch.float32).unsqueeze(0),
        scale=torch.tensor(g.scale, device=dev, dtype=torch.float32).unsqueeze(0),
        opacity=torch.tensor(g.opacity, device=dev, dtype=torch.float32).unsqueeze(0),
        sh=torch.tensor(g.sh, device=dev, dtype=torch.float32).reshape(1, N, -1, 3),
    )

    imgs = read_images_binary(f"{a.colmap_dir}/images.bin")
    cams = read_cameras_binary(f"{a.colmap_dir}/cameras.bin")
    iids = sorted(imgs.keys())
    step = max(1, len(iids) // a.n_views)
    selected = [iids[i * step] for i in range(a.n_views)]

    import imageio
    for iid in selected:
        im = imgs[iid]
        c = cams[im["cam_id"]]
        R = qvec2rotmat(im["qvec"])
        t = np.array(im["tvec"])
        # Camera position in world frame
        cam_pos = -R.T @ t
        cam_xmat = R.T.flatten()
        p_c = c["params"]
        if c["model"] in (0, 2):
            fovy = 2 * np.arctan(c["h"] / (2 * p_c[0])) * 180 / np.pi
        else:
            fy = p_c[1] if c["model"] == 1 else p_c[0]
            fovy = 2 * np.arctan(c["h"] / (2 * fy)) * 180 / np.pi

        cp = torch.tensor(cam_pos, device=dev, dtype=torch.float32).reshape(1, 1, 3)
        cx = torch.tensor(cam_xmat, device=dev, dtype=torch.float32).reshape(1, 1, 9)
        fv = np.array([[fovy]], dtype=np.float32)

        rgb, _ = batch_env_render(gsb, cp, cx, a.res, a.res, fv, y_up=False)
        img = rgb[0, 0].clamp(0, 1).detach().cpu().numpy()
        img8 = (img * 255).astype(np.uint8)
        out = f"{a.out_dir}/scene_{iid:04d}.png"
        imageio.imwrite(out, img8)
        print(f"  rendered {im['name']} -> {out}")

    print(f"done: {len(selected)} views in {a.out_dir}/")


if __name__ == "__main__":
    main()

"""Time _GaussianRenderer's BATCHED splat path on our L40S.

The committed benchmark (gs-playground/benchmark/mtx_batch_minimal.ipynb) reports
2210 img/s, but: ONE camera, 640x480, background EXCLUDED from the timed loop
(precomputed to bg_imgs), no free-floating manipuland, and NO GPU MODEL RECORDED.
This measures the configuration we would actually run:
  - 3 cameras @ 224x224 (our pixel arm)
  - background rasterised every step (mandatory: camera-extrinsic DR means the
    camera moves, so a precomputed background image is invalid)
  - gaussian count swept up to real-reconstruction scale. The shipped assets are
    ~76k gaussians; a PolaRiS-class reconstruction is ~730k (10x). gsplat cost is
    per-gaussian in projection+tile-sort, so this axis dominates.
  - peak VRAM recorded (the thing I failed to capture for Isaac).

Renderer only -- physics is deliberately excluded. Our MuJoCo step is already
measured at 11.82 ms (docs/infra.md:383); this gives the render cost to add.
"""

import argparse, json, os, time
import numpy as np
import torch

import sys
sys.path.insert(0, "/scratch/yifanr/realsim_survey/_GaussianRenderer/src")

from gaussian_renderer.core.util_gau import load_ply
from gaussian_renderer.core.gaussiandata import GaussianData
from gaussian_renderer.core.batch_rasterization import (
    batch_update_gaussians, batch_env_render,
)

ASSETS = "/scratch/yifanr/realsim_survey/gs-playground/benchmark/assets/franka_emika_panda_robotiq/3dgs"


def build_template(inflate_to, device):
    """Concatenate robot links (dynamic) + background (static), then optionally
    pad the STATIC set up to `inflate_to` gaussians by tiling it with jitter --
    a real reconstruction has ~10x the shipped background."""
    import glob
    body_plys = sorted(glob.glob(f"{ASSETS}/franka_224/*.ply")) + \
                sorted(glob.glob(f"{ASSETS}/robotiq_224/*.ply"))
    xyz, rot, scale, opa, sh = [], [], [], [], []
    idx_start, idx_end, body_ids = [], [], []
    n = 0
    max_sh = 0
    loaded = []
    for i, p in enumerate(body_plys):
        g = load_ply(p); loaded.append(g); max_sh = max(max_sh, g.sh.shape[1])
        idx_start.append(n); n += len(g.xyz); idx_end.append(n); body_ids.append(i)
    n_dyn = n
    bg = load_ply(f"{ASSETS}/background.ply"); max_sh = max(max_sh, bg.sh.shape[1])
    statics = [bg]
    n_static = len(bg.xyz)
    if inflate_to and inflate_to > n_dyn + n_static:
        need = inflate_to - n_dyn - n_static
        reps = int(np.ceil(need / len(bg.xyz)))
        for r in range(reps):
            g2 = GaussianData(xyz=bg.xyz + np.float32(0.01 * (r + 1)), rot=bg.rot.copy(),
                              scale=bg.scale.copy(), opacity=bg.opacity.copy(), sh=bg.sh.copy())
            statics.append(g2)
    allg = loaded + statics

    def pad(s):
        if s.shape[1] < max_sh:
            return np.concatenate([s, np.zeros((s.shape[0], max_sh - s.shape[1]), np.float32)], 1)
        return s

    T = lambda a: torch.tensor(np.concatenate(a, 0), dtype=torch.float32, device=device)
    tmpl = GaussianData(
        xyz=T([g.xyz for g in allg]), rot=T([g.rot for g in allg]),
        scale=T([g.scale for g in allg]), opacity=T([g.opacity for g in allg]),
        sh=T([pad(g.sh) for g in allg]).reshape(sum(len(g.xyz) for g in allg), -1, 3).contiguous(),
    )
    N = len(tmpl.xyz)
    dyn_mask = torch.zeros(N, dtype=torch.bool, device=device)
    p2b = torch.zeros(N, dtype=torch.long, device=device)
    for s, e, b in zip(idx_start, idx_end, body_ids):
        dyn_mask[s:e] = True; p2b[s:e] = b
    return tmpl, dyn_mask, p2b, len(body_plys), N, n_dyn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_envs", type=int, default=16)
    ap.add_argument("--n_cams", type=int, default=3)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--inflate_to", type=int, default=0)
    ap.add_argument("--minibatch", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    tmpl, dyn_mask, p2b, n_bodies, N, n_dyn = build_template(a.inflate_to, dev)
    print(f"[gs] gaussians={N} (dynamic={n_dyn}) bodies={n_bodies} "
          f"envs={a.n_envs} cams={a.n_cams} res={a.res}")

    E, C = a.n_envs, a.n_cams
    body_pos = torch.zeros((E, n_bodies, 3), device=dev)
    body_quat = torch.zeros((E, n_bodies, 4), device=dev); body_quat[..., 0] = 1.0
    cam_pos = torch.tensor([[[1.2, 0.0, 0.6], [0.0, 1.2, 0.6], [-1.0, -1.0, 0.8]][:C]],
                           device=dev).repeat(E, 1, 1)
    eye = torch.eye(3, device=dev).reshape(1, 1, 9)
    cam_xmat = eye.repeat(E, C, 1)
    fovy = np.full((E, C), 45.0, dtype=np.float32)
    mb = a.minibatch if a.minibatch > 0 else None

    def step():
        # jitter the robot bodies AND the cameras: camera-extrinsic DR is
        # mandatory for us, which is exactly what invalidates a precomputed
        # background image.
        bp = body_pos + torch.randn_like(body_pos) * 0.002
        cp = cam_pos + torch.randn_like(cam_pos) * 0.002
        gsb = batch_update_gaussians(tmpl, bp, body_quat, point_to_body_idx=p2b,
                                     dynamic_mask=dyn_mask)
        rgb, depth = batch_env_render(gsb, cp, cam_xmat, a.res, a.res, fovy,
                                      bg_imgs=None, minibatch=mb)
        return rgb

    def run(n):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n):
            step()
        torch.cuda.synchronize(); return time.perf_counter() - t0

    run(a.warmup)
    torch.cuda.reset_peak_memory_stats()
    dt = run(a.steps)
    ms = 1000 * dt / a.steps
    envsps = a.steps / dt
    res = {
        "gaussians": N, "n_envs": E, "n_cams": C, "res": f"{a.res}x{a.res}",
        "steps": a.steps, "ms_per_step": round(ms, 2),
        "env_steps_per_s": round(envsps, 2),
        "transitions_per_s": round(envsps * E, 1),
        "images_per_s": round(envsps * E * C, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    print("[gs] RESULT " + json.dumps(res))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()

"""Re-expose the board object splat so it is photometrically consistent with the
scene splat -- i.e. so the composite the policy trains on matches what the ZED sees.

The problem (real2sim/deploy/color_landmarks.py): board_obj.ply and the scene splat
come from two different phone captures with independent auto-exposure / white
balance. In the real ZED the board is the DARKEST blue thing in frame (B=60); in the
splat composite it is among the BRIGHTEST (B=199). That inversion is unfittable by
any monotone per-channel colour map on the real image -- least squares resolves it
with a NEGATIVE blue gain, collapsing blue to a constant. So the fix belongs in the
splat, not in the camera pipeline.

Target = the real board as seen through the deployed colour map, i.e.
`apply_color(real_board_pixels)`, matched in per-channel MEAN and STD (std sets the
board's internal contrast -- the pocket shadows -- which a mean-only match would
flatten). Both cameras contribute; a single ply cannot satisfy both exactly, so the
solve is joint and the per-camera residual is reported.

Closed loop, never open loop (CLAUDE.md iron rule #28): the DC->colour algebra is
only approximate here (alpha blending with the scene behind, overlapping gaussians,
and this renderer reshapes the SH block as [N,16,3] while the loader writes it
channel-major, so only the DC row is a true (r,g,b) triple). So we RENDER, MEASURE,
and solve from measured slopes.

Writes real2sim/splats/objsplats/board_obj.ply and keeps the original as
board_obj_phonescan_orig.ply. Use --dry_run to see the numbers without writing.
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import real2sim.splat_obs as so  # noqa: E402  (import order: fixes gs_mujoco)
import cv2  # noqa: E402
from plyfile import PlyData  # noqa: E402
from real2sim.deploy.preprocess import (DEFAULT_COLOR_JSON, SIM_CAMERA, Rectifier,  # noqa: E402
                                        apply_color, decode_jpeg_rgb, load_color)
from real2sim.deploy.color_landmarks import board_geom_ids, biggest  # noqa: E402
from compare_real_sim_demos import Real, SimReplay  # noqa: E402

SH_C0 = 0.28209479177387814
BOARD = os.path.join(_REPO, "real2sim", "splats", "objsplats", "board_obj.ply")
ORIG = os.path.join(_REPO, "real2sim", "splats", "objsplats", "board_obj_phonescan_orig.ply")
CALIB = os.path.join(_REPO, "real2sim", "splats", "objsplats", "obj_calib.json")


def write_reexposed(src_ply, dst_ply, gain, off):
    """colour' = gain*colour + off, expressed on the SH coefficients.

    gsplat evaluates colour = clamp(SH(dirs) + 0.5, 0) and the degree-0 term is
    SH_C0 * f_dc, so
        0.5 + C0*dc' = g*(0.5 + C0*dc) + o  =>  dc' = g*dc + (0.5*g + o - 0.5)/C0
    and every higher-order coefficient scales by g (they are deviations about DC).
    """
    ply = PlyData.read(src_ply)
    el = ply.elements[0]
    data = el.data.copy()
    rest = sorted([p.name for p in el.properties if p.name.startswith("f_rest_")],
                  key=lambda n: int(n.split("_")[-1]))
    per = len(rest) // 3                      # loader layout is channel-major: R.. G.. B..
    for c in range(3):
        g, o = float(gain[c]), float(off[c])
        data[f"f_dc_{c}"] = (g * data[f"f_dc_{c}"] + (0.5 * g + o - 0.5) / SH_C0).astype(
            data[f"f_dc_{c}"].dtype)
        for n in rest[c * per:(c + 1) * per]:
            data[n] = (g * data[n]).astype(data[n].dtype)
    el.data = data
    PlyData([el], text=False, byte_order=ply.byte_order).write(dst_ply)


def board_stats(rp, cam, tmp_ply, tmp_dir):
    """Render the composite with a candidate board ply -> (mean, std) inside the
    board's geometric mask. Rebuilds the compositor because the splats are uploaded
    to the GPU once at construction."""
    shutil.copy(tmp_ply, os.path.join(tmp_dir, "board_obj.ply"))
    rp.e._splat = None                                   # force a rebuild on next use
    from real2sim.splat_obs import SplatCompositor
    rp.e._splat = SplatCompositor(rp.sim.model, rp.sim.data, calib_path=CALIB,
                                  seg_dir=tmp_dir, device="cuda")
    S = rp.S
    img = rp.render(cam)
    seg = rp.sim.render(camera_name=SIM_CAMERA[cam], width=S, height=S,
                        segmentation=True)[::-1][:, :, 1]
    m = biggest(np.isin(seg, board_geom_ids(rp.sim)), 9)
    px = img[m].astype(np.float32)
    return px.mean(0), px.std(0), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", default="/scratch/yifanr/simdist/gello_software/demos/peg")
    ap.add_argument("--S", type=int, default=224)
    ap.add_argument("--n_frames", type=int, default=10)
    ap.add_argument("--target_cam", default="scene", choices=["scene", "wrist", "joint"],
                    help="which camera's measurement defines the target. Default scene: its "
                         "colour map is the validated one and its board pixels are sharp; the "
                         "wrist's std is inflated by close-range blur and the glare streak, "
                         "which are camera artifacts, not board contrast.")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-expose even if the original is missing (applies on top)")
    a = ap.parse_args()
    tmp = "/tmp/board_reexpose"
    os.makedirs(tmp, exist_ok=True)
    shutil.copy(CALIB, os.path.join(tmp, "obj_calib.json"))

    # ---- target: the real board through the deployed colour map -------------------
    reals = [Real(p) for p in sorted(glob.glob(f"{a.real_dir}/*.h5"))]
    color = load_color(DEFAULT_COLOR_JSON)
    rp = SimReplay(a.S)
    rects = {c: Rectifier(reals[0].K[c], reals[0].D[c], 1280, 720, a.S) for c in SIM_CAMERA}
    tgt = {}
    for cam in SIM_CAMERA:
        mus, sds, ns = [], [], 0
        for real in reals:
            for t in np.linspace(0, real.T - 1, a.n_frames).astype(int):
                rp.set(real.q[t], real.fingers[t])
                seg = rp.sim.render(camera_name=SIM_CAMERA[cam], width=a.S, height=a.S,
                                    segmentation=True)[::-1][:, :, 1]
                m = biggest(np.isin(seg, board_geom_ids(rp.sim)), 9)
                raw = rects[cam](decode_jpeg_rgb(real.imgs[cam][t]))
                m = m & (raw.max(2) < 245) & (raw.max(2) > 5)
                if m.sum() < 80:
                    continue
                px = apply_color(raw, color.get(cam))[m].astype(np.float32)
                mus.append(px.mean(0)); sds.append(px.std(0)); ns += m.sum()
        tgt[cam] = (np.mean(mus, 0), np.mean(sds, 0), ns)
        print(f"[target] {cam}: real board through the deployed map -> mean "
              f"{np.round(tgt[cam][0], 1).tolist()} std {np.round(tgt[cam][1], 1).tolist()} "
              f"({len(mus)} frames)", flush=True)
    if a.target_cam == "joint":
        T_mu = np.mean([tgt[c][0] for c in tgt], 0)
        T_sd = np.mean([tgt[c][1] for c in tgt], 0)
    else:
        T_mu, T_sd, _ = tgt[a.target_cam]
    print(f"[target] using '{a.target_cam}': mean {np.round(T_mu, 1).tolist()} "
          f"std {np.round(T_sd, 1).tolist()}", flush=True)

    # ---- closed loop: measure slopes, solve, verify ------------------------------
    # board_obj_phonescan_orig.ply is deliberately NOT tracked (it is byte-identical
    # to the board_obj.ply of the commit before the re-exposure, so git already has
    # it). On a fresh clone it is therefore absent -- and then BOARD is the ALREADY
    # re-exposed file, so sourcing from it would apply the correction twice. Refuse.
    meta_path = os.path.join(os.path.dirname(BOARD), "board_obj_reexposure.json")
    if not os.path.exists(ORIG) and os.path.exists(meta_path) and not a.force:
        raise SystemExit(
            f"{os.path.basename(BOARD)} has already been re-exposed "
            f"({os.path.basename(meta_path)} exists) and the untracked original "
            f"{os.path.basename(ORIG)} is missing, so re-running would apply the "
            f"correction TWICE.\n"
            f"Recover the original first:\n"
            f"  git log --oneline -- {os.path.relpath(BOARD)}\n"
            f"  git show <commit-before-re-exposure>:{os.path.relpath(BOARD)} > "
            f"{os.path.relpath(ORIG)}\n"
            f"or pass --force to re-expose on top of the current file anyway.")
    src = ORIG if os.path.exists(ORIG) else BOARD
    probe = os.path.join(tmp, "probe.ply")
    rp.set(reals[0].q[0], reals[0].fingers[0])

    def measure(gain, off):
        write_reexposed(src, probe, gain, off)
        out = {c: board_stats(rp, c, probe, tmp) for c in SIM_CAMERA}
        mu = np.mean([out[c][0] for c in out], 0)
        sd = np.mean([out[c][1] for c in out], 0)
        return mu, sd, out

    one, zero = np.ones(3), np.zeros(3)
    mu1, sd1, det1 = measure(one, zero)
    print(f"[probe] identity      mean {np.round(mu1, 1).tolist()} std {np.round(sd1, 1).tolist()}",
          flush=True)
    # std is ~proportional to gain and ~independent of the offset
    gain = np.clip(T_sd / np.maximum(sd1, 1e-3), 0.1, 3.0)
    mu2, sd2, _ = measure(gain, zero)
    print(f"[probe] gain={np.round(gain, 3).tolist()}  mean {np.round(mu2, 1).tolist()} "
          f"std {np.round(sd2, 1).tolist()}", flush=True)
    d = 0.10
    mu3, _, _ = measure(gain, np.full(3, d))
    slope = np.maximum((mu3 - mu2) / d, 1e-3)             # d(mean)/d(offset), 0-255 per unit
    off = (T_mu - mu2) / slope
    print(f"[probe] offset slope {np.round(slope, 1).tolist()} -> offset "
          f"{np.round(off, 4).tolist()}", flush=True)
    mu4, sd4, det4 = measure(gain, off)
    print(f"\n[result] gain {np.round(gain, 4).tolist()}  offset {np.round(off, 4).tolist()}",
          flush=True)
    print(f"  mean over both cameras {np.round(mu4, 1).tolist()} (target {np.round(T_mu, 1).tolist()})"
          f"  std {np.round(sd4, 1).tolist()} (target {np.round(T_sd, 1).tolist()})", flush=True)
    for cam in SIM_CAMERA:
        b, s, n = det4[cam]
        print(f"  {cam:6s} board now {np.round(b, 1).tolist()} vs its own target "
              f"{np.round(tgt[cam][0], 1).tolist()}  "
              f"(was {np.round(det1[cam][0], 1).tolist()}, n={n} px)", flush=True)

    if a.dry_run:
        print("\n[dry run] nothing written", flush=True)
        return
    if not os.path.exists(ORIG):
        shutil.copy(BOARD, ORIG)
        print(f"\n[backup] original phone-scan board splat -> {os.path.basename(ORIG)}", flush=True)
    write_reexposed(src, BOARD, gain, off)
    meta = os.path.join(os.path.dirname(BOARD), "board_obj_reexposure.json")
    json.dump({"gain": gain.tolist(), "offset": off.tolist(), "source": os.path.basename(ORIG),
               "target_cam": a.target_cam, "target_mean": T_mu.tolist(), "target_std": T_sd.tolist(),
               "achieved_mean": mu4.tolist(), "achieved_std": sd4.tolist(),
               "per_camera_target": {c: tgt[c][0].tolist() for c in tgt},
               "per_camera_achieved": {c: det4[c][0].tolist() for c in det4},
               "before": {c: det1[c][0].tolist() for c in det1},
               "why": ("board_obj and the scene splat came from different phone captures; the "
                       "board was brighter than the scene's exposure implied, inverting the blue "
                       "ordering vs the real ZED and making the real->splat colour map "
                       "unfittable. Target = real board through the deployed colour map."),
               "script": "real2sim/scan_pipeline/reexpose_board_splat.py"},
              open(meta, "w"), indent=1)
    print(f"[write] {BOARD}\n[write] {meta}", flush=True)


if __name__ == "__main__":
    main()

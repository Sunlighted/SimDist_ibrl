"""Per-landmark colour audit + landmark-weighted refit of the ZED -> splat map.

Why: the shipped map matches the GLOBAL mean/std of every usable pixel, and those
are dominated by the huge pale poster + backdrop. Saturated landmarks (blue peg
board, green mat) contribute a few percent of the pixels, so a global fit can
leave them visibly off -- exactly the "splat still looks brighter / less
saturated" complaint.

Landmarks, all defined WITHOUT reference to the real frame's colours:
  board     : MuJoCo segmentation of the board body's visual geom (pure geometry),
              eroded by the real/sim misalignment margin.
  green_mat : colour-selected in the SPLAT, largest connected component, eroded.
  poster_dark / poster_bright : the tree canopy and the sky of the printed poster,
              also selected in the splat.
For the splat-selected ones the mask is taken from the splat AND, separately, from
the real frame; both directions are reported, so a conclusion can't be an artifact
of which image drew the mask. Every landmark excludes sim-robot pixels (dilated)
and real pixels that are clipped (>245) or crushed (<12).

Refit: per-channel gain+offset, least squares on landmark MEANS with one equation
per (landmark, frame) and equal weight per landmark -- so the blue board counts as
much as the poster. Validated on a held-out episode.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import real2sim.splat_obs  # noqa: F401,E402  import-order fix for gs_mujoco
import cv2  # noqa: E402
from real2sim.deploy.preprocess import (DEFAULT_COLOR_JSON, SIM_CAMERA, Rectifier,  # noqa: E402
                                        apply_color, decode_jpeg_rgb, load_color)
from compare_real_sim_demos import Real, SimReplay  # noqa: E402

OUT_JSON = os.path.join(_REPO, "real2sim", "deploy", "color_correction.json")
VIZ = os.path.join(_REPO, "viz", "real_vs_sim")
LANDMARKS = ["board", "green_mat", "backdrop", "neutral_dark",
             "poster_dark", "poster_bright"]
# A landmark is USED for fitting only if the mask drawn on the splat and the mask
# drawn on the real frame pick out the same splat colour (within RELIABLE_TOL/255).
# High-frequency texture (the tree canopy) fails this: 15-30 px of real/sim
# misalignment means the two masks are not looking at the same material.
RELIABLE_TOL = 25.0


def board_geom_ids(sim):
    m = sim.model
    ids = []
    for g in range(m.ngeom):
        b = m.body_id2name(int(m.geom_bodyid[g])) or ""
        n = m.geom_id2name(g) or ""
        if b.startswith("board_obj") and "wall" not in n:
            ids.append(g)
    return ids


def sat(rgb_u8):
    """Mean HSV saturation (0-255) of a set of RGB pixels."""
    return float(cv2.cvtColor(rgb_u8.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)[:, 0, 1].mean())


def biggest(mask, erode):
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, bool)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    out = (lab == k).astype(np.uint8)
    if erode > 0:
        out = cv2.erode(out, np.ones((erode, erode), np.uint8))
    return out > 0


def colour_masks(img, erode):
    """Landmark masks from ONE image's colours (used on the splat and on the real)."""
    f = img.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx = f.max(2)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    out = {}
    out["green_mat"] = biggest((g > r + 4) & (g > b + 12) & (mx > 60) & (mx < 250), erode)
    # neutral (low-saturation) extremes: the backdrop curtain and the dark rod/shadow.
    # These are what pin down exposure and contrast, which the coloured landmarks alone
    # cannot separate from a hue shift.
    out["backdrop"] = biggest((s < 40) & (v > 150) & (v < 250), erode)
    out["neutral_dark"] = biggest((s < 70) & (v < 70) & (v > 12), erode)
    # the printed poster: warm (r>b) and either very dark (canopy) or bright (sky)
    warm = (r > b + 10) & (s > 30)
    out["poster_dark"] = biggest(warm & (v < 90), erode)
    out["poster_bright"] = biggest(warm & (v > 170) & (v < 250), erode)
    return out


def collect(real, rp, rects, S, frames, erode, dil):
    """-> rows[(cam, landmark, mask_from, frame)] = (mean_real_raw, mean_splat)"""
    rows = []
    geoms = rp.e._splat.mj_geoms
    bgeoms = board_geom_ids(rp.sim)
    for t in frames:
        rp.set(real.q[t], real.fingers[t])
        for cam, simcam in SIM_CAMERA.items():
            raw = rects[cam](decode_jpeg_rgb(real.imgs[cam][t]))
            spl = rp.render(cam)
            seg = rp.sim.render(camera_name=simcam, width=S, height=S,
                                segmentation=True)[::-1][:, :, 1]
            robot = cv2.dilate(np.isin(seg, geoms).astype(np.uint8),
                               np.ones((dil, dil), np.uint8)) > 0
            usable = (~robot) & (raw.max(2) < 245) & (raw.max(2) > 12)
            cand = {("board", "geometry"): biggest(np.isin(seg, bgeoms), erode)}
            for src, im in (("splat", spl), ("real", raw)):
                for name, msk in colour_masks(im, erode).items():
                    cand[(name, src)] = msk
            for (name, src), msk in cand.items():
                m = msk & usable
                if m.sum() < 150:
                    continue
                rows.append(dict(cam=cam, landmark=name, mask_from=src, frame=int(t),
                                 n=int(m.sum()),
                                 real=raw[m].mean(0).astype(float),
                                 splat=spl[m].mean(0).astype(float),
                                 real_sat=sat(raw[m]), splat_sat=sat(spl[m])))
    return rows


def reliability(rows, cam):
    """-> {landmark: mean |splat(mask from splat) - splat(mask from real)|}. Geometric
    masks have no second opinion and are trusted (0.0)."""
    out = {}
    for name in LANDMARKS:
        g = [r for r in rows if r["cam"] == cam and r["landmark"] == name]
        if any(r["mask_from"] == "geometry" for r in g):
            out[name] = 0.0
            continue
        a = [r["splat"] for r in g if r["mask_from"] == "splat"]
        b = [r["splat"] for r in g if r["mask_from"] == "real"]
        out[name] = (float(np.abs(np.mean(a, 0) - np.mean(b, 0)).mean())
                     if a and b else float("inf"))
    return out


def _samples(rows, cam, keep):
    """One (real, splat) mean pair per (landmark, frame), equal total weight per
    landmark. Prefers the geometric mask, else averages the two colour masks."""
    X, Y, W = [], [], []
    per = {n: sum(1 for r in rows if r["cam"] == cam and r["landmark"] == n
                  and r["mask_from"] in ("geometry", "splat")) for n in keep}
    for r in rows:
        if r["cam"] != cam or r["landmark"] not in keep:
            continue
        if r["mask_from"] == "real" and any(
                x["mask_from"] == "geometry" for x in rows
                if x["cam"] == cam and x["landmark"] == r["landmark"]):
            continue
        X.append(r["real"] / 255)
        Y.append(r["splat"] / 255)
        W.append(1.0 / max(per.get(r["landmark"], 1), 1))
    return np.array(X), np.array(Y), np.array(W)


def fit_landmark(rows, cam, keep):
    """Per-channel gain+offset on landmark means, equal weight per landmark."""
    X, Y, W = _samples(rows, cam, keep)
    if len(X) < 2:
        return None
    M = np.zeros((3, 4), np.float32)
    for c in range(3):
        A = np.stack([X[:, c], np.ones(len(X))], 1)
        g, b = np.linalg.lstsq(A * W[:, None], Y[:, c] * W, rcond=None)[0]
        M[c, c], M[c, 3] = g, b
    return M


def fit_gamma(rows, cam, keep):
    """Per-channel out = gain * in**gamma + offset. An affine map cannot lift the
    near-black board to the splat's bright blue AND leave the bright backdrop in
    place; a gamma can, because it lifts shadows more than highlights."""
    from scipy.optimize import least_squares
    X, Y, W = _samples(rows, cam, keep)
    if len(X) < 3:
        return None
    out = np.zeros((3, 3), np.float32)                       # gain, gamma, offset
    for c in range(3):
        def res(p, c=c):
            g, gam, b = p
            return (np.clip(g * np.clip(X[:, c], 0, 1) ** max(gam, 0.05) + b, 0, 1)
                    - Y[:, c]) * W
        # Bounds matter: with 0.05 <= gain and 0.2 <= gamma the blue channel collapsed to a
        # near-constant 0.64 (fitting 4 landmarks whose blues happen to be similar) -- which
        # would erase the very channel that makes the board visible. Keep every channel a
        # real, well-conditioned curve.
        r = least_squares(res, [1.0, 1.0, 0.0], bounds=([0.4, 0.45, -0.2], [2.5, 2.2, 0.5]))
        out[c] = r.x
    return out


def spread_ratio(M, X, Y):
    """Per-channel (output spread) / (splat spread) over the landmark set."""
    out = []
    for c in range(3):
        if M.shape == (3, 3):
            P = M[c, 0] * np.clip(X[:, c], 0, 1) ** M[c, 1] + M[c, 2]
        else:
            P = X @ M[c, :3] + M[c, 3]
        out.append(float(np.ptp(np.clip(P, 0, 1)) / max(np.ptp(Y[:, c]), 1e-6)))
    return out


def describe(kind, M):
    if M is None:
        return "none"
    if kind == "gamma":
        return (f"gain {M[:, 0].round(3).tolist()} gamma {M[:, 1].round(3).tolist()} "
                f"offset {M[:, 2].round(3).tolist()}")
    return f"gain {np.diag(M[:, :3]).round(3).tolist()} offset {M[:, 3].round(3).tolist()}"


def flattens(M, X, Y, frac=0.6):
    """True if the map compresses a channel's landmark spread to < frac of the splat's.
    A fit can reach a low MAE by collapsing a channel onto the landmark mean; that looks
    good in the table and destroys the contrast the policy needs."""
    for c in range(3):
        if M.shape == (3, 3):
            P = M[c, 0] * np.clip(X[:, c], 0, 1) ** M[c, 1] + M[c, 2]
        else:
            P = X @ M[c, :3] + M[c, 3]
        if np.ptp(np.clip(P, 0, 1)) < frac * np.ptp(Y[:, c]):
            return True
    return False


def apply_gamma(rgb_u8, G):
    x = rgb_u8.reshape(-1, 3).astype(np.float32) / 255.0
    y = np.stack([G[c, 0] * np.clip(x[:, c], 0, 1) ** G[c, 1] + G[c, 2] for c in range(3)], 1)
    return (np.clip(y, 0, 1) * 255 + 0.5).astype(np.uint8).reshape(rgb_u8.shape)


def report(rows, cam, maps):
    sel = [r for r in rows if r["cam"] == cam]
    rel = reliability(sel, cam)
    print(f"\n=== {cam}: per-landmark mean RGB (real -> after each map, vs splat target). "
          f"mask-source disagreement (lower = correspondence is trustworthy): "
          + ", ".join(f"{k} {v:.0f}" for k, v in rel.items()), flush=True)
    hdr = f"{'landmark':13s} {'mask':9s} {'n':>6s} {'splat RGB':>18s} {'sat':>4s} | "
    hdr += " | ".join(f"{k:>24s}" for k in maps)
    print(hdr, flush=True)
    for name in LANDMARKS:
        for src in ("geometry", "splat", "real"):
            g = [r for r in sel if r["landmark"] == name and r["mask_from"] == src]
            if not g:
                continue
            S_ = np.mean([r["splat"] for r in g], 0)
            R_ = np.mean([r["real"] for r in g], 0)
            line = (f"{name:13s} {src:9s} {int(np.mean([r['n'] for r in g])):6d} "
                    f"{np.round(S_, 0).astype(int).tolist()!s:>18s} "
                    f"{np.mean([r['splat_sat'] for r in g]):4.0f} | ")
            cells = []
            for k, M in maps.items():
                if M is None:
                    P = R_
                elif M.shape == (3, 3):
                    P = np.array([M[c, 0] * (R_[c] / 255) ** M[c, 1] + M[c, 2]
                                  for c in range(3)]).clip(0, 1) * 255
                else:
                    P = np.clip((R_ / 255) @ M[:, :3].T + M[:, 3], 0, 1) * 255
                dE = float(np.abs(P - S_).mean())
                cells.append(f"{np.round(P, 0).astype(int).tolist()!s:>18s} d{dE:4.0f}")
            print(line + " | ".join(cells), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", default="/scratch/yifanr/simdist/gello_software/demos/peg")
    ap.add_argument("--S", type=int, default=224)
    ap.add_argument("--n_frames", type=int, default=14)
    ap.add_argument("--erode", type=int, default=9)
    ap.add_argument("--dilate", type=int, default=17)
    ap.add_argument("--write", action="store_true", help="update color_correction.json")
    a = ap.parse_args()

    reals = [Real(p) for p in sorted(glob.glob(f"{a.real_dir}/*.h5"))]
    rp = SimReplay(a.S)
    rects = {c: Rectifier(reals[0].K[c], reals[0].D[c], 1280, 720, a.S) for c in SIM_CAMERA}
    cur = load_color(DEFAULT_COLOR_JSON)

    per_ep = []
    for r in reals:
        fr = np.linspace(0, r.T - 1, a.n_frames).astype(int)
        rows = collect(r, rp, rects, a.S, fr, a.erode, a.dilate)
        per_ep.append(rows)
        got = sorted({(x["landmark"], x["mask_from"]) for x in rows if x["cam"] == "scene"})
        print(f"[collect] {r.name}: {len(rows)} landmark samples; scene has {got}", flush=True)

    allrows = [x for rows in per_ep for x in rows]
    new = {}
    for cam in SIM_CAMERA:
        rel = reliability(allrows, cam)
        keep = [n for n, v in rel.items() if v <= RELIABLE_TOL]
        print(f"\n[{cam}] fitting on {keep}  (dropped "
              f"{[n for n in rel if n not in keep]} for mask-source disagreement)", flush=True)
        A_tr, G_tr = fit_landmark(per_ep[0], cam, keep), fit_gamma(per_ep[0], cam, keep)
        A_all, G_all = fit_landmark(allrows, cam, keep), fit_gamma(allrows, cam, keep)
        maps = {"RAW (no map)": None, "CURRENT global": cur.get(cam),
                "landmark AFFINE": A_tr, "landmark GAMMA": G_tr}
        report(per_ep[1], cam, maps)          # held-out episode
        # held-out error over the RELIABLE landmarks only
        Xh, Yh, Wh = _samples(per_ep[1], cam, keep)
        def err(M):
            if M is None:
                P = Xh
            elif M.shape == (3, 3):
                P = np.stack([M[c, 0] * np.clip(Xh[:, c], 0, 1) ** M[c, 1] + M[c, 2]
                              for c in range(3)], 1).clip(0, 1)
            else:
                P = np.clip(Xh @ M[:, :3].T + M[:, 3], 0, 1)
            return float((np.abs(P - Yh).mean(1) * Wh).sum() / Wh.sum() * 255)
        Xa, Ya, _ = _samples(allrows, cam, keep)
        flat = {"affine": flattens(A_tr, Xh, Yh), "gamma": flattens(G_tr, Xh, Yh)}
        for k, M in (("affine", A_tr), ("gamma", G_tr), ("current", cur.get(cam))):
            if M is not None:
                print(f"    {k:8s} {describe('gamma' if M.shape == (3, 3) else 'affine', M)}"
                      f"  spread/splat per channel "
                      f"{[round(v, 2) for v in spread_ratio(M, Xh, Yh)]}", flush=True)
        print(f"  held-out landmark MAE /255:  raw {err(None):.1f} | current "
              f"{err(cur.get(cam)):.1f} | affine {err(A_tr):.1f}"
              f"{' FLATTENS' if flat['affine'] else ''} | gamma {err(G_tr):.1f}"
              f"{' FLATTENS' if flat['gamma'] else ''}", flush=True)
        # per-channel inversion check: if two reliable landmarks swap order between real and
        # splat in some channel, NO monotone per-channel map can fit both -- the splat's own
        # objects disagree on exposure and colour transfer is the wrong tool.
        names = [n for n in keep]
        for c, ch in enumerate("RGB"):
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    def mean_of(n, k):
                        g = [r[k][c] for r in allrows if r["cam"] == cam and r["landmark"] == n]
                        return float(np.mean(g)) if g else np.nan
                    dr = mean_of(names[i], "real") - mean_of(names[j], "real")
                    ds = mean_of(names[i], "splat") - mean_of(names[j], "splat")
                    if np.isfinite(dr) and np.isfinite(ds) and dr * ds < 0 and min(abs(dr), abs(ds)) > 15:
                        print(f"  !! {ch}: {names[i]} vs {names[j]} INVERTS "
                              f"(real {dr:+.0f}, splat {ds:+.0f}) -- unfittable by any "
                              f"monotone per-channel map", flush=True)
        cand = [(err(A_tr), "affine", A_all)] if not flat["affine"] else []
        if not flat["gamma"]:
            cand.append((err(G_tr), "gamma", G_all))
        cand = [c for c in cand if c[0] < err(cur.get(cam)) - 1.0]
        if cand:
            e, kind, M = min(cand)
            new[cam] = (kind, M)
        else:
            new[cam] = ("keep", cur.get(cam))
            print("  -> no candidate beats the current map by >1.0 MAE without flattening; "
                  "keeping the current map", flush=True)
        kind, M = new[cam]
        if kind == "keep":
            continue
        print(f"  -> using {kind}: " + (
            f"gain {M[:, 0].round(3).tolist()} gamma {M[:, 1].round(3).tolist()} "
            f"offset {M[:, 2].round(3).tolist()}" if kind == "gamma" else
            f"gain {np.diag(M[:, :3]).round(3).tolist()} offset {M[:, 3].round(3).tolist()}"),
            flush=True)

    if a.write:
        d = json.load(open(OUT_JSON))
        for cam in SIM_CAMERA:
            kind, M = new[cam]
            if kind == "keep":
                continue
            d["cameras"][cam]["M_global_moments_superseded"] = d["cameras"][cam]["M"]
            if kind == "gamma":
                d["cameras"][cam].pop("M", None)
                d["cameras"][cam]["G"] = M.tolist()
                d["cameras"][cam]["model"] = ("per-channel out = gain*in**gamma + offset, fitted "
                                              "on LANDMARK means (equal weight per landmark)")
            else:
                d["cameras"][cam]["M"] = M.tolist()
                d["cameras"][cam]["model"] = ("per-channel gain+offset, fitted on LANDMARK means "
                                              "(equal weight per landmark)")
        d["fit"]["landmarks"] = LANDMARKS
        d["fit"]["landmark_note"] = ("global mean/std matching under-corrected the saturated "
                                     "landmarks; see real2sim/deploy/color_landmarks.py")
        json.dump(d, open(OUT_JSON, "w"), indent=1)
        print(f"\n[write] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()

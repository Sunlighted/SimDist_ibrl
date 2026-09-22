"""Turn a raw object 3DGS scan into a splat expressed in its MuJoCo body frame.

The scene scan got metric scale from the ChArUco board (§2c). board.MOV/peg.MOV
do NOT contain the board, so scale comes from the other metric thing we have:
the object's own scanned STL, which is already the sim's visual+collision mesh.
That makes this a SIMILARITY fit (scale + R + t), not the rigid fit
calibrate_splat_frames.py does for the PolaRiS links.

Three stages, each with a printed diagnostic so a bad fit is visible rather than
silent:
  1. CROP. The scan contains the whole poster table; nothing upstream masks.
     RANSAC the dominant plane (the table), keep what sits above it, then keep the
     largest connected cluster. Reported as a retained fraction -- a crop that
     keeps 90% of the scan did not isolate the object.
  2. SCALE. Ratio of sorted PCA extents, splat vs mesh. The three axis ratios are
     printed: they must AGREE (a similarity has one scale). Spread across axes
     means the crop is still carrying table, or the scan is non-rigid.
  3. POSE. Coarse PCA-axis alignment over the 4 sign-consistent rotations, then
     trimmed ICP, lowest RMS wins. Same structure as calibrate_splat_frames.py,
     which measured 2-4 mm on the arm links -- use that as the yardstick.

Scale is BAKED INTO THE PLY (positions and the log-scale attributes), because
MjSplatRenderer's calib applies only R,t -- a scaled R would move the centroids
correctly but leave every ellipsoid the wrong size.
"""
import argparse, json, os, struct, sys
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


def read_stl(path):
    """Binary STL -> (n_tri, 3, 3) vertices. ASCII STL falls back to a text parse."""
    with open(path, "rb") as f:
        head = f.read(84)
        n = struct.unpack("<I", head[80:84])[0]
        body = f.read(n * 50)
    if len(body) == n * 50 and n > 0:
        tri = np.zeros((n, 3, 3), np.float64)
        for i in range(n):
            o = i * 50 + 12          # skip the per-facet normal
            tri[i] = np.frombuffer(body[o:o + 36], "<f4").reshape(3, 3)
        return tri
    v = []
    for line in open(path, "r", errors="ignore"):
        s = line.split()
        if s and s[0] == "vertex":
            v.append([float(x) for x in s[1:4]])
    return np.array(v).reshape(-1, 3, 3)


def sample_mesh(tri, n=40000, seed=0):
    """Area-weighted uniform surface sampling.

    NOT the triangle vertices: peg.stl is a box, i.e. 12 triangles / 8 unique
    corners, and ICP against 8 points cannot constrain a pose -- nor can PCA
    extents be trusted from them. Sampling the SURFACE is what makes both the
    aspect test and the ICP meaningful.
    """
    rng = np.random.default_rng(seed)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    tot = area.sum()
    if tot <= 0:
        return tri.reshape(-1, 3)
    k = rng.choice(len(tri), n, p=area / tot)
    u = rng.random((n, 1)); v = rng.random((n, 1))
    flip = (u + v) > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[k] + u * (b[k] - a[k]) + v * (c[k] - a[k])


def fit_plane(P, iters=800, tol=0.03, rng=None):
    """RANSAC plane. tol is in SCAN units, which are arbitrary -- so scale it to
    the cloud's own extent rather than assuming metres."""
    rng = rng or np.random.default_rng(0)
    span = np.percentile(P, 97, 0) - np.percentile(P, 3, 0)
    t = tol * float(np.linalg.norm(span))
    best = (None, -1)
    for _ in range(iters):
        i = rng.choice(len(P), 3, replace=False)
        a, b, c = P[i]
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm)
        if ln < 1e-12:
            continue
        nrm = nrm / ln
        d = np.abs((P - a) @ nrm)
        k = int((d < t).sum())
        if k > best[1]:
            best = ((nrm, a), k)
    return best[0], best[1], t


def largest_cluster(P, idx, tol):
    """Grid-based connected components -- avoids an O(n^2) pairwise pass."""
    cell = tol
    keys = np.floor(P[idx] / cell).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    lut = {tuple(k): i for i, k in enumerate(uniq)}
    seen = np.zeros(len(uniq), bool)
    best = []
    nbr = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for s in range(len(uniq)):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            c = stack.pop()
            comp.append(c)
            k = uniq[c]
            for d in nbr:
                j = lut.get((k[0] + d[0], k[1] + d[1], k[2] + d[2]))
                if j is not None and not seen[j]:
                    seen[j] = True
                    stack.append(j)
        if len(comp) > len(best):
            best = comp
    keep_cells = np.zeros(len(uniq), bool)
    keep_cells[best] = True
    return idx[keep_cells[inv]]


def pca_axes(X):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    if np.linalg.det(Vt.T) < 0:
        Vt[-1] *= -1
    return Vt                     # rows = principal axes


def extents(X, V):
    Y = (X - X.mean(0)) @ V.T
    return np.percentile(Y, 99, 0) - np.percentile(Y, 1, 0)


def kabsch(P, Q, w=None):
    w = np.ones(len(P)) if w is None else w
    w = w / w.sum()
    pc, qc = (P * w[:, None]).sum(0), (Q * w[:, None]).sum(0)
    H = ((P - pc) * w[:, None]).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def trimmed_icp(src, dst, trim=0.8, iters=60):
    tree = cKDTree(dst)
    R, t = np.eye(3), np.zeros(3)
    cur = src.copy()
    rms = np.inf
    for _ in range(iters):
        d, j = tree.query(cur)
        k = max(int(trim * len(cur)), 12)
        sel = np.argsort(d)[:k]
        dR, dt = kabsch(cur[sel], dst[j[sel]])
        cur = cur @ dR.T + dt
        R, t = dR @ R, dR @ t + dt
        new = float(np.sqrt((d[sel] ** 2).mean()))
        if abs(rms - new) < 1e-9:
            break
        rms = new
    return R, t, rms


def orbit_center(sparse_dir):
    """Where the cameras are all looking.

    An object scan is an orbit, so the object sits at the convergence of the
    optical axes -- a far more reliable cue than a plane fit, which fails exactly
    when the object is small relative to the scene (measured: the peg's whole
    0.21-unit thickness was inside the plane's 0.14-unit RANSAC tolerance, so the
    plane absorbed the object and the crop returned a flat 6.2x2.8 sheet).

    Least squares: minimise sum_i ||(I - d_i d_i^T)(x - c_i)||^2.
    """
    sys.path.insert(0, "/scratch/yifanr/gsbench")
    from train_3dgs import read_images_binary, qvec2rotmat
    imgs = read_images_binary(f"{sparse_dir}/images.bin")
    A = np.zeros((3, 3)); b = np.zeros(3); C = []
    for im in imgs.values():
        R = qvec2rotmat(im["qvec"]); t = np.asarray(im["tvec"], float)
        c = -R.T @ t                      # camera centre in world
        d = R.T @ np.array([0.0, 0.0, 1.0])   # COLMAP looks along +z
        d = d / np.linalg.norm(d)
        Pi = np.eye(3) - np.outer(d, d)
        A += Pi; b += Pi @ c; C.append(c)
    x = np.linalg.solve(A, b)
    C = np.asarray(C)
    rad = np.linalg.norm(C - x, axis=1)
    print(f"[orbit] {len(C)} cameras, centre {np.round(x,4)}, "
          f"radius p5/p50/p95 {np.percentile(rad,5):.3f}/{np.percentile(rad,50):.3f}/{np.percentile(rad,95):.3f}")
    return x, float(np.median(rad))


def pick_crop_by_aspect(P, alive, centre, orbit_r, mesh_aspect, plane_removal=True):
    """Choose the crop whose PCA extents best match the mesh's ASPECT.

    Three composed stages per candidate radius, because no single one works:
      a) SPHERE around the orbit centre -- localises, but a sphere around an
         object standing on a table always grabs a disc of table with it
         (measured on the peg: aspect 1:0.54:0.44 against the mesh's 1:0.27:0.17).
      b) LOCAL plane removal -- now the RANSAC tolerance is scaled to the sphere,
         not to the whole scene. Scene-scaled tolerance was the original bug: the
         peg's entire 0.21-unit thickness sat inside a 0.14-unit tolerance.
      c) largest connected cluster -- drops any table fragments left over.

    Selection is on ASPECT, which is SCALE-FREE, so this is a genuine test rather
    than a fit: scale is only derived afterwards, from the crop this picks.
    """
    rng = np.random.default_rng(0)
    best = None
    for frac in np.linspace(0.05, 0.50, 19):
        r = frac * orbit_r
        sph = np.where(alive & (np.linalg.norm(P - centre, axis=1) < r))[0]
        if len(sph) < 400:
            continue
        if plane_removal:
            pl, n_in, tol = fit_plane(P[sph], iters=400, rng=rng)
            if pl is None:
                continue
            nrm, pt = pl
            signed = (P[sph] - pt) @ nrm
            # support surface is the majority side; the object is the minority
            if (signed > 0).sum() > (signed < 0).sum():
                signed = -signed
            off = sph[signed < -0.5 * tol]
            if len(off) < 300:
                continue
        else:
            span = float(np.linalg.norm(np.percentile(P[sph], 97, 0)
                                        - np.percentile(P[sph], 3, 0)))
            tol, off = 0.03 * span, sph
        idx = largest_cluster(P, off, tol)
        if len(idx) < 250:
            continue
        V = pca_axes(P[idx])
        e = np.sort(extents(P[idx], V))[::-1]
        if e[0] <= 0:
            continue
        asp = e / e[0]
        err = float(np.abs(np.log(asp[1:] / mesh_aspect[1:])).mean())
        if best is None or err < best[0]:
            best = (err, r, idx, asp, len(sph), len(off))
    if best is None:
        raise SystemExit("FATAL: no crop radius survived plane removal + clustering")
    err, r, idx, asp, n_sph, n_off = best
    print(f"[crop] r={r:.4f} ({100*r/orbit_r:.1f}% of orbit): "
          f"sphere {n_sph} -> after-plane {n_off} -> cluster {len(idx)} "
          f"({100*len(idx)/len(P):.1f}% of scan)")
    print(f"       aspect splat {np.round(asp,3)} vs mesh {np.round(mesh_aspect,3)}  "
          f"log-err {err:.3f}")
    if err > 0.25:
        print("  ⚠️ aspect mismatch: no crop reproduces the mesh's shape -- the scan "
              "may not have isolated the object. Do NOT trust the scale below.")
    return idx, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, help="raw 3DGS output of the object scan")
    ap.add_argument("--mesh", required=True, help="the object's STL (metric, = sim's mesh)")
    ap.add_argument("--out_ply", required=True)
    ap.add_argument("--token", required=True, help="calib token = out_ply basename")
    ap.add_argument("--mj_body", required=True)
    ap.add_argument("--calib", required=True, help="calib json to create/update")
    ap.add_argument("--colmap_dir", required=True,
                    help="the sparse model the splat was trained from -- the camera "
                         "orbit is what localises the object")
    ap.add_argument("--opacity_min", type=float, default=0.25)
    ap.add_argument("--color_gate", choices=["none", "blue", "white"], default="none",
                    help="APPEARANCE gate, applied before any geometry. Needed because "
                         "geometry alone cannot separate a flat square BOARD from a flat "
                         "square patch of TABLE -- both give aspect ~[1,1,0.18], so the "
                         "aspect test passed a tabletop crop and ICP happily fitted it. "
                         "Colour is nearly orthogonal to shape, so it breaks that tie. "
                         "Measured on these scans: board blueness p90 +0.435 (table is "
                         "brown => negative); peg saturation p50 0.171 with brightness "
                         "p50 0.736 (poster is saturated brown).")
    ap.add_argument("--gate_thresh", type=float, default=0.05)
    ap.add_argument("--plane_removal", choices=["on", "off"], default="on",
                    help="'on' strips the support surface -- correct for an object "
                         "STANDING on a table (the peg). Use 'off' for an object that "
                         "IS essentially a plane lying flat (the board): plane removal "
                         "then deletes the object itself, measured as 59085 -> 865 "
                         "gaussians and aspect [1,0.83,0.39] against a mesh of "
                         "[1,1.00,0.18].")
    args = ap.parse_args()

    ply = PlyData.read(args.ply)
    el = ply.elements[0]
    P = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float64)
    opa = np.asarray(el["opacity"]).astype(np.float64)
    print(f"[load] {len(P)} gaussians from {args.ply}")

    # ---- 1. crop (camera orbit, aspect-selected) ---------------------------
    alive = 1.0 / (1.0 + np.exp(-opa)) > args.opacity_min
    if args.color_gate != "none":
        dc = np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], 1).astype(np.float64)
        rgb = np.clip(dc * 0.2821 + 0.5, 0, 1)            # 3DGS SH dc term -> rgb
        mx, mn = rgb.max(1), rgb.min(1)
        if args.color_gate == "blue":
            keepc = (rgb[:, 2] - 0.5 * (rgb[:, 0] + rgb[:, 1])) > args.gate_thresh
        else:                                              # "white": bright + desaturated
            sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0.0)
            keepc = (rgb.mean(1) > 0.60) & (sat < 0.30)
        before = int(alive.sum())
        alive = alive & keepc
        print(f"[colour] gate '{args.color_gate}' keeps {int(alive.sum())}/{before} "
              f"alive gaussians ({100*alive.sum()/max(before,1):.1f}%)")
        if alive.sum() < 500:
            raise SystemExit("FATAL: colour gate left <500 gaussians -- threshold too tight")
    tri = read_stl(args.mesh)
    M = sample_mesh(tri).astype(np.float64)
    print(f"[mesh] {len(tri)} triangles -> {len(M)} surface samples "
          f"(vertices alone would be {len(np.unique(tri.reshape(-1,3),axis=0))} points)")
    Vm = pca_axes(M)
    em_sorted = np.sort(extents(M, Vm))[::-1]
    mesh_aspect = em_sorted / em_sorted[0]
    centre, orbit_r = orbit_center(args.colmap_dir)
    keep, aspect_err = pick_crop_by_aspect(P, alive, centre, orbit_r, mesh_aspect,
                                           plane_removal=(args.plane_removal == 'on'))

    # ---- 2. scale ----------------------------------------------------------
    Vs = pca_axes(P[keep])
    es = extents(P[keep], Vs)
    ratios = em_sorted / np.sort(es)[::-1]
    s_scale = float(np.median(ratios))
    print(f"[scale] splat extents {np.round(np.sort(es)[::-1],4)} (scan units)")
    print(f"        mesh  extents {np.round(em_sorted,4)} m")
    print(f"        per-axis ratios {np.round(ratios,4)}  -> scale {s_scale:.5f} "
          f"(spread {100*(ratios.max()-ratios.min())/s_scale:.1f}% of scale)")
    if (ratios.max() - ratios.min()) / s_scale > 0.15:
        print("  ⚠️ axis ratios disagree by >15% -- not a clean similarity; suspect the crop")

    # ---- 3. pose -----------------------------------------------------------
    # Coarse PCA-axis alignment has a 4-fold sign ambiguity (a rotation must keep
    # det=+1, so the third sign is forced). Try all four, ICP each, keep the best
    # rms -- the same structure calibrate_splat_frames.py uses, and the reason the
    # Robotiq base_link's 169 deg axis-flip was catchable at all.
    src0 = (P[keep] - P[keep].mean(0)) * s_scale          # centred, now metric
    mmean = M.mean(0)
    best = None
    for sx in (1, -1):
        for sy in (1, -1):
            R0 = Vm.T @ np.diag([sx, sy, sx * sy]) @ Vs
            R1, t1, rms = trimmed_icp(src0 @ R0.T + mmean, M)
            # x -> (x@R0.T + mmean)@R1.T + t1  ==  x@(R1@R0).T + (R1@mmean + t1)
            cand = (rms, R1 @ R0, R1 @ mmean + t1, (sx, sy))
            if best is None or rms < best[0]:
                best = cand
    rms, R, t, signs = best
    print(f"[pose] trimmed-ICP rms {1000*rms:.2f} mm, PCA signs {signs} "
          f"(calibrate_splat_frames.py got 2.0-4.3 mm on the arm links)")
    if rms > 0.010:
        print("  ⚠️ rms >10 mm -- treat this registration as POOR (cf. the "
              "Robotiq base_link 11.9 mm / 169 deg axis-flip failure)")

    # ---- 4. write the ply with the similarity BAKED IN ---------------------
    names = [p.name for p in el.properties]
    out = {n: np.asarray(el[n])[keep].astype(np.float32) for n in names}
    newP = src0 @ R.T + t
    out["x"], out["y"], out["z"] = newP[:, 0].astype(np.float32), newP[:, 1].astype(np.float32), newP[:, 2].astype(np.float32)
    for i in range(3):
        k = f"scale_{i}"
        if k in out:
            out[k] = (out[k] + np.log(s_scale)).astype(np.float32)   # 3DGS stores LOG scales
    rn = [f"rot_{i}" for i in range(4)]
    if all(k in out for k in rn):
        from scipy.spatial.transform import Rotation
        q = np.stack([out[k] for k in rn], 1)                  # wxyz
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        qq = (Rotation.from_matrix(R) * Rotation.from_quat(q[:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
        for i, k in enumerate(rn):
            out[k] = qq[:, i].astype(np.float32)
    arr = np.empty(len(keep), dtype=[(n, "f4") for n in names])
    for n in names:
        arr[n] = out[n]
    os.makedirs(os.path.dirname(args.out_ply) or ".", exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(args.out_ply)
    print(f"[write] {args.out_ply}  ({len(keep)} gaussians, metric, in {args.mj_body} frame)")

    calib = json.load(open(args.calib)) if os.path.exists(args.calib) else {}
    # ★ verdict must NOT key on ICP rms alone. Trimmed ICP drops the worst 20%, so
    # a crop that is too FAT can bury the mesh inside itself and score a low
    # residual -- measured on the peg: aspect log-err 0.517 (crop is wrong) with
    # rms 1.06 mm (looks excellent). The aspect test is scale-free and independent,
    # so it is the one that can fail the fit; rms alone cannot.
    scale_spread = float((ratios.max() - ratios.min()) / s_scale)
    bad = aspect_err > 0.25 or scale_spread > 0.15
    verdict = ("POOR" if bad else
               "OK" if rms < 0.005 else "MARGINAL" if rms < 0.010 else "POOR")
    calib[args.token] = {"mj_body": args.mj_body, "R": np.eye(3).tolist(),
                         "t": [0.0, 0.0, 0.0], "rms_m": float(rms),
                         "aspect_log_err": aspect_err, "scale_spread": scale_spread,
                         "scale": s_scale, "n_splat": int(len(keep)),
                         "verdict": verdict,
                         "note": ("similarity BAKED INTO the ply; calib R,t are identity"
                                  + ("; CROP FAILED the scale-free aspect test -- the low "
                                     "rms is ICP hiding a fat blob, do not trust this"
                                     if bad else ""))}
    json.dump(calib, open(args.calib, "w"), indent=1)
    print(f"[calib] {args.calib}: {args.token} -> {args.mj_body}  verdict "
          f"{calib[args.token]['verdict']}")


if __name__ == "__main__":
    main()

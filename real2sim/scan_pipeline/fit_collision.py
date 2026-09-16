"""Point cloud -> MuJoCo collision geom.

Real2sim gives us a splat / scanned mesh per object; MuJoCo needs a COLLISION
geom. Per the repair asymmetry (stage 3 refits dynamics, the encoder is frozen),
collision geometry is on the REPAIRABLE axis and should be as crude as the task
allows -- and there is now a throughput argument too: rendering measured 3.63
ms/step vs MuJoCo physics at 11.82 ms, so physics is ~76% of data-generation
cost and convex-decomposing everything is the easiest way to make stage 1 slow.

So: prefer an exact primitive (box/cylinder/sphere) when one explains the cloud,
fall back to a convex hull only when none does.

Scoring. For each candidate we report
  - `outside_p95`: 95th-percentile distance of points OUTSIDE the primitive (m).
    Robust to splat floaters, unlike a max.
  - `fill`: hull_volume / primitive_volume. 1.0 = the primitive is exactly the
    object; small = the primitive wastes space (a cylinder fitted to a flat
    plate scores badly here even though every point is inside it).
We pick the primitive with the best fill subject to outside_p95 <= tol, so a
sloppy-but-enclosing shape can never beat a tight one.
"""
import numpy as np
from scipy.spatial import ConvexHull


def _pca(X):
    c = X.mean(0)
    U, S, Vt = np.linalg.svd(X - c, full_matrices=False)
    R = Vt.T
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    return c, R


def _trim(X, q=99.5):
    """Drop isolated outliers before ANY candidate sees them. Splat clouds carry
    floaters; without this the box (min/max) is punished by them while the
    cylinder/sphere (percentile radii) are not, and the comparison is rigged."""
    c = np.median(X, 0)
    d = np.linalg.norm(X - c, axis=1)
    return X[d <= np.percentile(d, q)]


def fit_box(X, q=99.5):
    c, R = _pca(X)
    L = (X - c) @ R
    # percentile extents, matching the cylinder/sphere radii, so all three
    # candidates are equally robust to residual outliers
    lo = np.percentile(L, 100 - q, axis=0)
    hi = np.percentile(L, q, axis=0)
    half = (hi - lo) / 2
    centre = c + R @ ((hi + lo) / 2)
    d = np.maximum(np.abs(L - (hi + lo) / 2) - half, 0)
    return dict(type="box", size=half, pos=centre, R=R,
                volume=float(8 * np.prod(half)),
                outside=np.linalg.norm(d, axis=1))


def fit_cylinder(X):
    c, R = _pca(X)
    L = (X - c) @ R
    best = None
    for ax in range(3):                     # principal axis is not always [0]
        others = [i for i in range(3) if i != ax]
        rad = np.linalg.norm(L[:, others], axis=1)
        r = np.percentile(rad, 99.5)
        hl = (L[:, ax].max() - L[:, ax].min()) / 2
        mid = (L[:, ax].max() + L[:, ax].min()) / 2
        out_r = np.maximum(rad - r, 0)
        out_a = np.maximum(np.abs(L[:, ax] - mid) - hl, 0)
        out = np.sqrt(out_r ** 2 + out_a ** 2)
        vol = float(np.pi * r ** 2 * 2 * hl)
        Rc = R.copy()
        if ax != 2:                          # MuJoCo cylinder axis is local z
            Rc = R[:, [others[0], others[1], ax]]
            if np.linalg.det(Rc) < 0:
                Rc[:, 0] *= -1
        cand = dict(type="cylinder", size=np.array([r, hl]),
                    pos=c + R[:, ax] * mid, R=Rc, volume=vol, outside=out)
        if best is None or np.percentile(out, 95) < np.percentile(best["outside"], 95):
            best = cand
    return best


def fit_sphere(X):
    c = X.mean(0)
    d = np.linalg.norm(X - c, axis=1)
    r = np.percentile(d, 99.5)
    return dict(type="sphere", size=np.array([r]), pos=c, R=np.eye(3),
                volume=float(4 / 3 * np.pi * r ** 3),
                outside=np.maximum(d - r, 0))


def fit_collision(X, tol=0.003, min_fill=0.75, verbose=False):
    """tol = allowed 95th-pct protrusion (m). 3 mm default: below the ~1.3 mm/edge
    tolerance floor would be over-fitting to scan noise, far above it changes
    which contacts happen.

    min_fill = hull/primitive volume ratio below which we refuse the primitive and
    fall back to a hull. This is the knob that stops an L-shape being called a
    box: the box encloses it with zero protrusion but fills in the notch, which
    invents contacts that do not exist. Raise it for contact-critical parts,
    lower it to buy physics speed on parts that only need to not fall through."""
    X = np.asarray(X, dtype=np.float64)
    X = _trim(X)
    try:
        hull_v = float(ConvexHull(X).volume)
    except Exception:
        hull_v = float(np.prod(X.max(0) - X.min(0)))
    cands = [fit_box(X), fit_cylinder(X), fit_sphere(X)]
    for c in cands:
        c["outside_p95"] = float(np.percentile(c["outside"], 95))
        c["fill"] = float(hull_v / max(c["volume"], 1e-12))
    ok = [c for c in cands
          if c["outside_p95"] <= tol and c["fill"] >= min_fill]
    chosen = max(ok, key=lambda c: c["fill"]) if ok else None
    if chosen is None:
        chosen = dict(type="hull", size=None, pos=X.mean(0), R=np.eye(3),
                      volume=hull_v, outside_p95=0.0, fill=1.0)
    if verbose:
        for c in cands:
            print(f"    {c['type']:9s} outside_p95={c['outside_p95']*1000:7.2f} mm  "
                  f"fill={c['fill']:.3f}  vol={c['volume']*1e6:9.1f} cm^3")
    return chosen, cands, hull_v


def to_mjcf(fit, name, mesh_file=None):
    """Emit a MuJoCo <geom>. Rotation goes out as a wxyz quat."""
    from scipy.spatial.transform import Rotation
    p = " ".join(f"{v:.6g}" for v in fit["pos"])
    if fit["type"] == "hull":
        return (f'<geom name="{name}" type="mesh" mesh="{mesh_file or name}" '
                f'group="0"/>')
    q = Rotation.from_matrix(fit["R"]).as_quat()[[3, 0, 1, 2]]
    qs = " ".join(f"{v:.6g}" for v in q)
    s = " ".join(f"{v:.6g}" for v in np.atleast_1d(fit["size"]))
    return (f'<geom name="{name}" type="{fit["type"]}" size="{s}" '
            f'pos="{p}" quat="{qs}" group="0"/>')

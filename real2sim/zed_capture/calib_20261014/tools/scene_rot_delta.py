"""Small scene-camera bump -> rotation correction of T_base_scene, no board needed.

For a pure camera rotation R the undistorted views are related by H = K R K^-1 for EVERY scene
depth, so static features (board edges, walls, table) pin R. Matches on moving things (arm,
objects) and any translation parallax are rejected by RANSAC; the inlier residual says how
well "pure rotation" explains the bump.

    python scene_rot_delta.py ref.png new.png [--mask_robot]
prints R_ref_new (rotvec, deg) and the corrected T_base_scene for the cell yaml.
"""
import argparse, json, os
import cv2, numpy as np
from scipy.spatial.transform import Rotation as R
REPO = os.path.expanduser("~/simdist/SimDist_ibrl")
fac = json.load(open(REPO + "/release/cfgs/sim2real/zed_factory_intrinsics.json"))["scene"]
m = fac["modes"]["hd720"]
K = np.array([[m["fx"], 0, m["cx"]], [0, m["fy"], m["cy"]], [0, 0, 1.0]])
D = np.array(fac["distortion_k1k2p1p2k3"])


def rays(pts):
    r = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    v = np.c_[r, np.ones(len(r))]
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def kabsch(a, b):
    """R with b ~ R a (unit vectors)."""
    U, _, Vt = np.linalg.svd(b.T @ a)
    S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ S @ Vt


def delta(ref_path, new_path, thr_px=1.0):
    a, b = cv2.imread(ref_path, 0), cv2.imread(new_path, 0)
    sift = cv2.SIFT_create(8000)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    mt = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [p for p, q in mt if p.distance < 0.75 * q.distance]
    pa = np.float64([ka[g.queryIdx].pt for g in good]); pb = np.float64([kb[g.trainIdx].pt for g in good])
    ra, rb = rays(pa), rays(pb)
    f = K[0, 0]
    rng = np.random.default_rng(0)
    best = None
    for _ in range(2000):                      # RANSAC on 2-point rotations
        i = rng.choice(len(ra), 2, replace=False)
        Rm = kabsch(ra[i], rb[i])
        err = f * np.linalg.norm(np.cross(ra @ Rm.T, rb), axis=1)
        inl = err < thr_px
        if best is None or inl.sum() > best.sum():
            best = inl
    Rm = kabsch(ra[best], rb[best])
    err = f * np.linalg.norm(np.cross(ra @ Rm.T, rb), axis=1)
    inl = err < thr_px
    Rm = kabsch(ra[inl], rb[inl]); err = f * np.linalg.norm(np.cross(ra @ Rm.T, rb), axis=1)
    return Rm, len(good), int(inl.sum()), float(np.median(err[inl])), pb[inl]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("new")
    ap.add_argument("--cell", default=REPO + "/real2sim/zed_capture/calib_20261013/threading_real_cell_20261013.yaml")
    a = ap.parse_args()
    Rm, n, ni, med, pin = delta(a.ref, a.new)
    rv = np.degrees(R.from_matrix(Rm).as_rotvec())
    print(f"matches {n}, inliers {ni}, inlier median {med:.2f} px; inliers span x {pin[:,0].min():.0f}..{pin[:,0].max():.0f} y {pin[:,1].min():.0f}..{pin[:,1].max():.0f}")
    print(f"rotation ref->new (camera axes x right, y down, z fwd): [{rv[0]:+.3f} {rv[1]:+.3f} {rv[2]:+.3f}] deg, |{np.linalg.norm(rv):.3f}|")
    import yaml
    cam = next(c for c in yaml.safe_load(open(a.cell))["cameras"] if c["name"] == "agentview")
    w, x, y, z = cam["quat"]
    Rbs = R.from_quat([x, y, z, w]).as_matrix()
    # ray_new = Rm ray_ref  =>  new camera axes in base: Rbs_new = Rbs @ Rm.T
    q = R.from_matrix(Rbs @ Rm.T).as_quat()
    print(f"corrected agentview quat (wxyz): [{q[3]:.5f}, {q[0]:.5f}, {q[1]:.5f}, {q[2]:.5f}]  pos unchanged {cam['pos']}")

"""Wrist hand-eye with the SCENE camera FIXED at the cell's validated pose (fingertip-checked,
unbumped since 10-14 by scene_rot_delta): board in base = T_base_scene @ T_scene_board.
Unknowns: X = T_hand_wrist, S = T_scene_board, square size a."""
import sys, json, os, glob
sys.path.insert(0, os.path.expanduser("~/simdist/SimDist_ibrl/real2sim/zed_capture/calib_20261010/tools"))
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import solve_calib as sc

run, wi_path = sys.argv[1], sys.argv[2]
SCENE_POS = np.array([0.23173, 0.55526, 0.52272]); SCENE_Q = [0.15341, -0.29297, 0.85476, -0.40002]   # wxyz
CELL_WRIST = (np.array([-0.07053, -0.03118, 0.01228]), [0.93013, -0.14733, 0.03582, -0.33447])
JOINT_WRIST = (np.array([-0.07499, -0.03605, 0.01066]), [0.92785, -0.15134, 0.03920, -0.33860])
Tq = lambda p, q: sc.T(Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(), p)
Bs = Tq(SCENE_POS, SCENE_Q)
sess = json.load(open(os.path.join(run, "session.json")))
Ks, Ds = np.array(sess["cameras"]["scene"]["K"]), np.array(sess["cameras"]["scene"]["D"])
wi = json.load(open(wi_path)); Kw, Dw = np.array(wi["K"]), np.array(wi["D"])
fk = sc.FK()
samples, sc_ids, sc_pix = [], [], []
for d in sorted(glob.glob(os.path.join(run, "sample_*"))):
    st = json.load(open(os.path.join(d, "state.json")))
    i, p = sc.detect(os.path.join(d, "scene_left.png")); sc_ids.append(i); sc_pix.append(p)
    i, p = sc.detect(os.path.join(d, "wrist_left.png"))
    if len(i) >= 10:
        samples.append(dict(name=os.path.basename(d), ids=i, pix=p, Tbh=fk.base_hand(st["q"])))
scene = dict(ids=np.concatenate(sc_ids), pix=np.concatenate(sc_pix))
n = len(samples)


def solve(idx, a_fixed=None):
    sub = [samples[i] for i in idx]
    X0, _ = sc.initial(sub, sc.SQUARE0, Kw, Dw)
    S0 = sc.pnp(sc.UNIT[scene["ids"]] * sc.SQUARE0, scene["pix"], Ks, Ds)

    def resid(v):
        X, S = sc.vec2T(v[:6]), sc.vec2T(v[6:12]); a = v[12] if a_fixed is None else a_fixed
        Y = Bs @ S
        r = [(sc.project(sc.inv(X) @ sc.inv(s["Tbh"]) @ Y, sc.UNIT[s["ids"]] * a, Kw, Dw) - s["pix"]).ravel() for s in sub]
        r.append((sc.project(S, sc.UNIT[scene["ids"]] * a, Ks, Ds) - scene["pix"]).ravel())
        return np.concatenate(r)
    v0 = np.r_[sc.T2vec(X0), sc.T2vec(S0), sc.SQUARE0]
    v = least_squares(resid, v0, loss="soft_l1", f_scale=1.5, x_scale="jac").x
    return sc.vec2T(v[:6]), sc.vec2T(v[6:12]), (v[12] if a_fixed is None else a_fixed)


X, Sb, a = solve(range(n))
Y = Bs @ Sb
print(f"{n} wrist views; square {a * 1e3:.2f} mm")
errs = [np.median(sc.view_err(s, X, Y, a, Kw, Dw)) for s in samples]
for s, e in zip(samples, errs):
    print(f"  {s['name']}: {e:5.2f} px")
print(f"wrist per-view median {np.median(errs):.2f} px, max {np.max(errs):.2f}")
se = np.linalg.norm(sc.project(Sb, sc.UNIT[scene['ids']] * a, Ks, Ds) - scene["pix"], axis=1)
print(f"scene reprojection median {np.median(se):.2f} px")
loo, Xs = [], []
for i in range(n):
    Xi, Si, ai = solve([j for j in range(n) if j != i])
    loo.append(np.median(sc.view_err(samples[i], Xi, Bs @ Si, ai, Kw, Dw))); Xs.append(Xi)
print(f"LOO predicted-view median {np.median(loo):.2f} px, max {np.max(loo):.2f}; X moves <= "
      f"{max(np.linalg.norm(Xi[:3, 3] - X[:3, 3]) for Xi in Xs) * 1e3:.2f} mm / {max(sc.rot_deg(Xi, X) for Xi in Xs):.3f} deg")
w, x, y, z = np.roll(Rotation.from_matrix(X[:3, :3]).as_quat(), 1)
print(f"wrist (hand frame): pos {np.round(X[:3, 3], 5).tolist()} quat wxyz {np.round([w, x, y, z], 5).tolist()}")
for tag, (p, q) in (("cell 10-13", CELL_WRIST), ("joint fit 10-15", JOINT_WRIST)):
    M = Tq(p, q)
    dp = (X[:3, 3] - M[:3, 3]) * 1e3
    print(f"  vs {tag}: d {np.round(dp, 1).tolist()} mm (|{np.linalg.norm(dp):.1f}|), in camera axes "
          f"{np.round(M[:3, :3].T @ dp, 1).tolist()}, rot {sc.rot_deg(M, X):.2f} deg")
json.dump(dict(pos=X[:3, 3].tolist(), quat=[w, x, y, z], square_m=a, T_base_board=Y.tolist(),
               wrist_median_px=float(np.median(errs)), loo_median_px=float(np.median(loo))),
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wrist_fixed_scene.json"), "w"), indent=1)

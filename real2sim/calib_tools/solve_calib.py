"""Solve both camera extrinsics from a record_calib.py run.

Unknowns (board = the ChArUco on the table, frame of OpenCV's CharucoBoard, z = 0 plane):
  X = T_hand_wrist   wrist ZED-M in the sim's robot0_right_hand frame  (cell yaml frame: hand)
  Y = T_base_board   board in the sim's robot0_base frame
  S = T_scene_board  board in the scene ZED 2i frame (static, pooled over all samples)
  a = square size    metric scale of the board (robot motion is metric, so it is observable)
=> T_base_scene = Y @ inv(S)                                           (cell yaml frame: base)

All four are refined jointly by minimising reprojection error of the ChArUco chessboard
corners in BOTH cameras. FK is robosuite's own Panda model (same kinematics as FR3), so the
result is in exactly the frames env/sim2real.py attaches cameras to -- no flange/45 deg step.
Images are RAW (unrectified) HD720 left eyes. Scene: factory K, D. Wrist: pass the
intrinsics.json of a separate solve_intrinsics.py run (hand-held board covering the whole
image); without it the wrist K, D are refitted from these views, which only see the board in
part of the image (run_1010: upper middle only), so the distortion is extrapolated elsewhere.

    ~/anaconda3/envs/simdist/bin/python ~/simdist/calib/solve_calib.py ~/simdist/calib/run_XXXX \
        --wrist_intrinsics ~/simdist/calib/intr_XXXX/intrinsics.json
"""
import argparse
import glob
import json
import os

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

PANDA_XML = os.path.expanduser(
    "~/simdist/deploy_simdist/deps/robosuite/robosuite/models/assets/robots/panda/robot.xml")

# The physical board: 5x7 ChArUco, DICT_5X5_50, square ~42.4 mm, marker ~31 mm (fitted to
# real2sim/scan_pipeline/board_model.json). That depth-built model is NOT used: its corners
# scatter by up to 2 mm, which costs several px at wrist range. Chessboard corners depend only
# on the square size, which is solved for, so the marker size never enters the geometry.
SQUARE0, MARKER0 = 0.0424, 0.031
BOARD = cv2.aruco.CharucoBoard((5, 7), SQUARE0, MARKER0,
                               cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50))
UNIT = BOARD.getChessboardCorners().astype(np.float64) / SQUARE0   # (24, 3), square = 1
DET = cv2.aruco.CharucoDetector(BOARD)


def T(R, t):
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, np.ravel(t)
    return M


def inv(M):
    return T(M[:3, :3].T, -M[:3, :3].T @ M[:3, 3])


def vec2T(v):
    return T(Rotation.from_rotvec(v[:3]).as_matrix(), v[3:])


def T2vec(M):
    return np.r_[Rotation.from_matrix(M[:3, :3]).as_rotvec(), M[:3, 3]]


def rot_deg(A, B):
    return np.degrees(Rotation.from_matrix(A[:3, :3].T @ B[:3, :3]).magnitude())


class FK:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(PANDA_XML)
        self.d = mujoco.MjData(self.m)

    def base_hand(self, q):
        self.d.qpos[:7] = q
        mujoco.mj_kinematics(self.m, self.d)
        b, h = self.d.body("base"), self.d.body("right_hand")
        return inv(T(b.xmat.reshape(3, 3), b.xpos)) @ T(h.xmat.reshape(3, 3), h.xpos)


def detect(path):
    """-> (charuco corner ids, raw pixel coords)."""
    cc, ci, _, _ = DET.detectBoard(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    if ci is None:
        return np.zeros(0, int), np.zeros((0, 2))
    return ci.ravel().astype(int), cc.reshape(-1, 2).astype(np.float64)


def pnp(obj, pix, K, D):
    ok, rv, tv = cv2.solvePnP(obj, pix, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    assert ok
    return T(cv2.Rodrigues(rv)[0], tv)


def project(Tcb, obj, K, D):
    p, _ = cv2.projectPoints(obj, cv2.Rodrigues(Tcb[:3, :3])[0], Tcb[:3, 3], K, D)
    return p.reshape(-1, 2)


def refit_intrinsics(samples, a, K, D, size=(1280, 720)):
    """Re-estimate one camera's K, D from the board views (aspect ratio fixed to the guess).
    The ZED-M factory distortion was measured to be wrong for this stream (k1 -0.10 vs ~-0.35,
    own-PnP 2.1 px -> 0.3 px), so the wrist camera needs this; the ZED 2i factory K, D are fine."""
    obj = [(UNIT[s["ids"]] * a).astype(np.float32) for s in samples]
    pix = [s["pix"].astype(np.float32) for s in samples]
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO
    rms, K, D, _, _ = cv2.calibrateCamera(obj, pix, size, K.copy(), D.copy().reshape(1, -1),
                                          flags=flags)
    return K, D.ravel(), rms


def initial(samples, a, Kw, Dw):
    """Closed-form start: OpenCV robot-world/hand-eye on per-view PnP."""
    Rw2c, tw2c, Rb2g, tb2g = [], [], [], []
    for s in samples:
        Tcb = pnp(UNIT[s["ids"]] * a, s["pix"], Kw, Dw)
        Rw2c.append(Tcb[:3, :3]); tw2c.append(Tcb[:3, 3])
        Tgb = inv(s["Tbh"])
        Rb2g.append(Tgb[:3, :3]); tb2g.append(Tgb[:3, 3])
    Rbw, tbw, Rgc, tgc = cv2.calibrateRobotWorldHandEye(
        Rw2c, tw2c, Rb2g, tb2g, method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH)
    return inv(T(Rgc, tgc)), inv(T(Rbw, tbw))      # X = T_hand_cam, Y = T_base_board


def solve(samples, scene, cams, idx):
    """Joint refinement of X, Y, S, square over wrist views idx + all scene detections."""
    Kw, Dw = cams["wrist"]
    Ks, Ds = cams["scene"]
    sub = [samples[i] for i in idx]
    X0, Y0 = initial(sub, SQUARE0, Kw, Dw)
    S0 = pnp(UNIT[scene["ids"]] * SQUARE0, scene["pix"], Ks, Ds)

    def resid(v):
        X, Y, S, a = vec2T(v[:6]), vec2T(v[6:12]), vec2T(v[12:18]), v[18]
        r = [(project(inv(X) @ inv(s["Tbh"]) @ Y, UNIT[s["ids"]] * a, Kw, Dw) - s["pix"]).ravel()
             for s in sub]
        r.append((project(S, UNIT[scene["ids"]] * a, Ks, Ds) - scene["pix"]).ravel())
        return np.concatenate(r)

    v0 = np.r_[T2vec(X0), T2vec(Y0), T2vec(S0), SQUARE0]
    sol = least_squares(resid, v0, loss="soft_l1", f_scale=1.5, x_scale="jac")
    v = sol.x
    return vec2T(v[:6]), vec2T(v[6:12]), vec2T(v[12:18]), float(v[18])


def view_err(s, X, Y, a, K, D):
    return np.linalg.norm(project(inv(X) @ inv(s["Tbh"]) @ Y, UNIT[s["ids"]] * a, K, D)
                          - s["pix"], axis=1)


def json_cam(sess, name):
    c = sess["cameras"][name]
    return np.array(c["K"], np.float64), np.array(c["D"], np.float64)


def yaml_cam(name, frame, M, fovy):
    w, x, y, z = np.roll(Rotation.from_matrix(M[:3, :3]).as_quat(), 1)  # xyzw -> wxyz
    p = M[:3, 3]
    return (f"- name: {name}\n  frame: {frame}\n  pos: [{p[0]:.5f}, {p[1]:.5f}, {p[2]:.5f}]\n"
            f"  quat: [{w:.5f}, {x:.5f}, {y:.5f}, {z:.5f}]   # wxyz, OpenCV optical axes\n"
            f"  fovy: {fovy}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--min_corners", type=int, default=10)
    ap.add_argument("--drop", default="", help="comma-separated sample names to exclude")
    ap.add_argument("--wrist_intrinsics", default="",
                    help="intrinsics.json from solve_intrinsics.py; the wrist K, D are then fixed")
    ap.add_argument("--factory_wrist", action="store_true",
                    help="keep the ZED-M factory intrinsics instead of refitting them")
    ap.add_argument("--dup_deg", type=float, default=0.5,
                    help="skip a sample whose joints are all within this of an earlier one")
    args = ap.parse_args()
    sess = json.load(open(os.path.join(args.run, "session.json")))
    cams = {n: (np.array(c["K"]), np.array(c["D"])) for n, c in sess["cameras"].items()}
    drop = set(filter(None, args.drop.split(",")))
    fk = FK()

    if args.wrist_intrinsics:
        wi = json.load(open(args.wrist_intrinsics))
        if str(wi["serial"]) != str(sess["cameras"]["wrist"]["serial"]) or \
                list(wi["size"]) != list(sess["size"]):
            raise SystemExit(f"{args.wrist_intrinsics} is for serial {wi['serial']} {wi['size']}, "
                             f"this run is {sess['cameras']['wrist']['serial']} {sess['size']}")
        cams["wrist"] = (np.array(wi["K"]), np.array(wi["D"]))
        print(f"wrist intrinsics fixed from {args.wrist_intrinsics} "
              f"(rms {wi['rms_px']:.2f} px, {wi['n_views']} views)")

    samples, sc_ids, sc_pix, qs = [], [], [], []
    for d in sorted(glob.glob(os.path.join(args.run, "sample_*"))):
        name = os.path.basename(d)
        if name in drop:
            continue
        st = json.load(open(os.path.join(d, "state.json")))
        q = np.array(st["q"])
        if any(np.max(np.abs(q - p)) < np.radians(args.dup_deg) for p in qs):
            print(f"skip {name}: same pose as an earlier sample")
            continue
        qs.append(q)
        i, p = detect(os.path.join(d, "scene_left.png"))
        sc_ids.append(i); sc_pix.append(p)
        i, p = detect(os.path.join(d, "wrist_left.png"))
        if len(i) < args.min_corners:
            print(f"skip {name}: wrist sees {len(i)} corners")
            continue
        samples.append(dict(name=name, ids=i, pix=p, Tbh=fk.base_hand(st["q"])))
    scene = dict(ids=np.concatenate(sc_ids), pix=np.concatenate(sc_pix))
    n = len(samples)
    print(f"{n} wrist views, {len(scene['ids'])} scene corner observations")

    W, H = sess["size"]
    cover = np.zeros((6, 10), int)
    for s in samples:
        r = np.clip((s["pix"][:, 1] * 6 / H).astype(int), 0, 5)
        c = np.clip((s["pix"][:, 0] * 10 / W).astype(int), 0, 9)
        for rc in set(zip(r, c)):
            cover[rc] += 1
    print(f"wrist corner coverage (views per cell, 6x10): {100 * np.mean(cover > 0):.0f}% of cells")
    print(cover)

    if not args.factory_wrist and not args.wrist_intrinsics:
        print("WARNING: refitting the wrist intrinsics from the extrinsics views; "
              "prefer --wrist_intrinsics from a solve_intrinsics.py run")
        # alternate: intrinsics need the square size, the square size comes from the joint fit
        a = SQUARE0
        for _ in range(3):
            Kw, Dw, rms = refit_intrinsics(samples, a, *json_cam(sess, "wrist"))
            cams["wrist"] = (Kw, Dw)
            _, _, _, a = solve(samples, scene, cams, list(range(n)))
        print(f"wrist intrinsics refitted (rms {rms:.2f} px): fx {Kw[0, 0]:.1f} fy {Kw[1, 1]:.1f} "
              f"cx {Kw[0, 2]:.1f} cy {Kw[1, 2]:.1f}  D {np.round(Dw, 4).tolist()}")
    Kw, Dw = cams["wrist"]
    Ks, Ds = cams["scene"]
    X, Y, S, a = solve(samples, scene, cams, list(range(n)))
    print(f"\nsquare size: {a * 1000:.2f} mm")

    print("\n== wrist: per-view reprojection with the joint fit (median px)")
    for s in samples:
        own = np.median(np.linalg.norm(
            project(pnp(UNIT[s["ids"]] * a, s["pix"], Kw, Dw), UNIT[s["ids"]] * a, Kw, Dw)
            - s["pix"], axis=1))
        print(f"  {s['name']}: {np.median(view_err(s, X, Y, a, Kw, Dw)):6.2f}   (its own PnP {own:.2f})")
    se = np.linalg.norm(project(S, UNIT[scene["ids"]] * a, Ks, Ds) - scene["pix"], axis=1)
    print(f"== scene: reprojection median {np.median(se):.2f} px, p90 {np.percentile(se, 90):.2f} px")

    print("\n== leave-one-out (refit without view i, predict view i)")
    loo, Xs, Ys, Ss = [], [], [], []
    for i in range(n):
        Xi, Yi, Si, ai = solve(samples, scene, cams, [j for j in range(n) if j != i])
        loo.append(np.median(view_err(samples[i], Xi, Yi, ai, Kw, Dw)))
        Xs.append(Xi); Ys.append(Yi); Ss.append(Si)
    worst = np.argsort(loo)[::-1][:3]
    print(f"  predicted-view error: median {np.median(loo):.2f} px, max {np.max(loo):.2f} px "
          f"(worst: {', '.join(samples[k]['name'] for k in worst)})")
    Bs = Y @ inv(S)
    ds_t = [np.linalg.norm((Yi @ inv(Si))[:3, 3] - Bs[:3, 3]) * 1000 for Yi, Si in zip(Ys, Ss)]
    ds_r = [rot_deg(Yi @ inv(Si), Bs) for Yi, Si in zip(Ys, Ss)]
    dw_t = [np.linalg.norm(Xi[:3, 3] - X[:3, 3]) * 1000 for Xi in Xs]
    dw_r = [rot_deg(Xi, X) for Xi in Xs]
    print(f"  T_base_scene moves by at most {max(ds_t):.2f} mm / {max(ds_r):.3f} deg across folds")
    print(f"  T_hand_wrist moves by at most {max(dw_t):.2f} mm / {max(dw_r):.3f} deg across folds")

    print("\n== T_base_board (board origin corner in robot base frame)")
    print(np.round(Y, 4))
    print("\n== cell yaml (env/sim2real.py RealCameraConfig) -- fovy = what you chose")
    print(yaml_cam("agentview", "base", Bs, 50.0))
    print(yaml_cam("robot0_eye_in_hand", "hand", X, 49.014))

    out = dict(T_base_scene=Bs.tolist(), T_hand_wrist=X.tolist(), T_base_board=Y.tolist(),
               T_scene_board=S.tolist(), square_m=a, n_views=n,
               views=[s["name"] for s in samples],
               loo_median_px=float(np.median(loo)), loo_max_px=float(np.max(loo)),
               loo_max_scene_mm=float(max(ds_t)), loo_max_scene_deg=float(max(ds_r)),
               loo_max_wrist_mm=float(max(dw_t)), loo_max_wrist_deg=float(max(dw_r)),
               scene_reproj_median_px=float(np.median(se)),
               fk_model=PANDA_XML,
               frames="scene: robot0_base, wrist: robot0_right_hand; OpenCV optical axes",
               wrist_K=Kw.tolist(), wrist_D=Dw.tolist(), scene_K=Ks.tolist(), scene_D=Ds.tolist(),
               wrist_coverage=cover.tolist(),
               intrinsics=("scene: factory hd720; wrist: " +
                           (f"fixed from {args.wrist_intrinsics}" if args.wrist_intrinsics else
                            "factory hd720" if args.factory_wrist else "REFITTED from these views")
                           + "; raw left eye"))
    json.dump(out, open(os.path.join(args.run, "extrinsics.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(args.run, 'extrinsics.json')}")


if __name__ == "__main__":
    main()

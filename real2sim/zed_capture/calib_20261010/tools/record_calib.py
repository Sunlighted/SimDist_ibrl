"""Record camera-calibration samples (raw = unrectified HD720 left eyes, lossless PNG).

Two modes:

  extrinsics (default)  both cameras + FR3 joints. The ChArUco is fixed on the table and the
                        arm moves the wrist camera around it. Joints come from joint_tap.py in the
                        ROS container; a sample is refused unless they are fresh and still.
      ~/anaconda3/bin/python3 ~/simdist/calib/record_calib.py --out ~/simdist/calib/run_$(date +%m%d_%H%M) \
          --wrist_intrinsics ~/simdist/calib/intr_XXXX/intrinsics.json

  --intrinsics          one camera only (wrist by default), no robot needed. Hold the board in
                        front of the camera and move it so that every part of the image is
                        covered, 0.2-0.4 m away, and >= 8 samples with the board tilted 35-55 deg
                        (the header counts them): near-frontal views leave fx and cy unconstrained.
      ~/anaconda3/bin/python3 ~/simdist/calib/record_calib.py --intrinsics --out ~/simdist/calib/intr_$(date +%m%d_%H%M)

The live view shades every image cell by how many saved samples have a corner there
(red = none, yellow = 1-2, green = 3+): fill the red cells. The 2026-10-10 run had corners only
in the upper middle of the wrist image, which left the distortion unconstrained elsewhere.

The cameras are opened here directly, so the teleop session must NOT publish them
(run_teleop_session.py --cameras with no names).

Keys (in the window):  SPACE record   d drop last   q quit
"""
import argparse
import collections
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

HOME = os.path.expanduser("~/simdist")
INTRINSICS = f"{HOME}/SimDist_ibrl/release/cfgs/sim2real/zed_factory_intrinsics.json"
JOINTS = f"{HOME}/gello_software/simdist_runner/joint_state_latest.json"
MODEL = {"scene": "ZED 2i", "wrist": "ZED-M"}
W, H = 1280, 720
# Same board as solve_calib.py: 5x7 ChArUco, DICT_5X5_50. The square size only scales the PnP
# readout here; the solver fits it.
SQUARE0, MARKER0 = 0.0424, 0.031
BOARD = cv2.aruco.CharucoBoard((5, 7), SQUARE0, MARKER0,
                               cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50))
OBJ = BOARD.getChessboardCorners().astype(np.float64)
DET = cv2.aruco.CharucoDetector(BOARD)
MIN_CORNERS = 8
GRID = (6, 10)  # coverage cells, rows x cols
STEEP_DEG = 35  # focal length needs foreshortened views; aim for >= 8 samples this steep


def find_device(model):
    for p in sorted(glob.glob("/sys/class/video4linux/video*")):
        if open(os.path.join(p, "name")).read().strip().startswith(model):
            return int(p.rsplit("video", 1)[1])
    sys.exit(f"{model} not found -- is publish_zed_uvc.py still holding it?")


def detect(img, K, D):
    """-> (corner ids, pixels, PnP median px or None, board tilt deg or None)."""
    cc, ci, _, _ = DET.detectBoard(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if ci is None:
        return np.zeros(0, int), np.zeros((0, 2)), None, None
    ids, pix = ci.ravel().astype(int), cc.reshape(-1, 2).astype(np.float64)
    if len(ids) < 6:
        return ids, pix, None, None
    ok, rv, tv = cv2.solvePnP(OBJ[ids], pix, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return ids, pix, None, None
    proj, _ = cv2.projectPoints(OBJ[ids], rv, tv, K, D)
    err = float(np.median(np.linalg.norm(proj.reshape(-1, 2) - pix, axis=1)))
    tilt = float(np.degrees(np.arccos(abs(cv2.Rodrigues(rv)[0][2, 2]))))
    return ids, pix, err, tilt


def cells(pix):
    r = np.clip((pix[:, 1] * GRID[0] / H).astype(int), 0, GRID[0] - 1)
    c = np.clip((pix[:, 0] * GRID[1] / W).astype(int), 0, GRID[1] - 1)
    return set(zip(r.tolist(), c.tolist()))


def draw_coverage(vis, cover):
    over = vis.copy()
    for r in range(GRID[0]):
        for c in range(GRID[1]):
            n = cover[r, c]
            col = (0, 0, 255) if n == 0 else (0, 220, 255) if n < 3 else (0, 200, 0)
            p0 = (c * W // GRID[1], r * H // GRID[0])
            p1 = ((c + 1) * W // GRID[1] - 1, (r + 1) * H // GRID[0] - 1)
            cv2.rectangle(over, p0, p1, col, -1)
            cv2.putText(vis, str(n), (p0[0] + 6, p0[1] + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    cv2.addWeighted(over, 0.18, vis, 0.82, 0, vis)


def read_joints():
    try:
        d = json.load(open(JOINTS))
        return d["t"], np.array(d["q"])
    except Exception:
        return None, None


def load_cam(intr, name, override):
    if override:
        o = json.load(open(override))
        return np.array(o["K"], np.float64), np.array(o["D"], np.float64), override
    m = intr[name]["modes"]["hd720"]
    K = np.array([[m["fx"], 0, m["cx"]], [0, m["fy"], m["cy"]], [0, 0, 1]], np.float64)
    return K, np.array(intr[name]["distortion_k1k2p1p2k3"], np.float64), INTRINSICS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--intrinsics", action="store_true",
                    help="intrinsics mode: one camera, no robot, hand-held board")
    ap.add_argument("--cam", default="wrist", choices=["wrist", "scene"],
                    help="camera for --intrinsics")
    ap.add_argument("--wrist_intrinsics", default="",
                    help="intrinsics.json from solve_intrinsics.py (else ZED-M factory values)")
    ap.add_argument("--no_auto", action="store_true",
                    help="extrinsics mode: disable auto-capture (record only on SPACE)")
    ap.add_argument("--still_px", type=float, default=0.7,
                    help="--intrinsics: max corner motion per frame (px) to count the board as still")
    ap.add_argument("--still_s", type=float, default=0.4,
                    help="--intrinsics: how long the board must stay still before SPACE is accepted")
    ap.add_argument("--auto_still_s", type=float, default=1.5,
                    help="auto-capture after the arm has been still this long")
    ap.add_argument("--auto_min_move_deg", type=float, default=8.0,
                    help="auto-capture only if some joint differs this much from every earlier sample")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names = [args.cam] if args.intrinsics else ["scene", "wrist"]
    auto_on = not args.intrinsics and not args.no_auto

    intr = json.load(open(INTRINSICS))
    cams = {}
    for name in names:
        K, D, src = load_cam(intr, name, args.wrist_intrinsics if name == "wrist" else "")
        dev = find_device(MODEL[name])
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2 * W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        if not cap.isOpened():
            sys.exit(f"cannot open /dev/video{dev}")
        cams[name] = dict(cap=cap, K=K, D=D, src=src, serial=intr[name]["serial"], dev=dev,
                          cover=np.zeros(GRID, int), prev=None, still_since=None)
        print(f"[{name}] /dev/video{dev}  intrinsics: {src}")

    # session metadata: everything the solver needs to interpret the samples
    json.dump({"kind": "intrinsics" if args.intrinsics else "extrinsics",
               "mode": "hd720", "size": [W, H], "eye": "left", "rectified": False,
               "board": {"type": "charuco", "squares": [5, 7], "dict": "DICT_5X5_50",
                         "square_m_nominal": SQUARE0, "marker_m_nominal": MARKER0},
               "cameras": {n: {"serial": c["serial"], "K": c["K"].tolist(), "D": c["D"].tolist(),
                               "intrinsics_file": c["src"]} for n, c in cams.items()},
               "joint_source": None if args.intrinsics else "/franka/joint_states via joint_tap.py",
               "joint_order": "fr3_joint1..7",
               "created": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(args.out, "session.json"), "w"), indent=1)

    hist = collections.deque()  # (t_wall, q) for the stillness check
    samples = sorted(glob.glob(os.path.join(args.out, "sample_*")))
    sample_qs, sample_cells, steep = [], [], []
    for s in samples:  # resume: rebuild coverage and pose history
        st = json.load(open(os.path.join(s, "state.json")))
        steep.append((st.get("tilt_deg") or {}).get(names[-1] if args.intrinsics else "wrist") or 0)
        if st.get("q") is not None:
            sample_qs.append(np.array(st["q"]))
        sc = {}
        for n, c in cams.items():
            p = os.path.join(s, f"{n}_left.png")
            if os.path.exists(p):
                ids, pix, _, _ = detect(cv2.imread(p), c["K"], c["D"])
                sc[n] = cells(pix)
                for rc in sc[n]:
                    c["cover"][rc] += 1
        sample_cells.append(sc)
    still_since = None
    win = "calib recorder (SPACE record, d drop, q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    msg, msg_t = "", 0.0
    while True:
        frames = {}
        for n, c in cams.items():
            ok, f = c["cap"].read()
            if not ok or f.shape[:2] != (H, 2 * W):
                sys.exit(f"[{n}] frame read failed")
            frames[n] = f[:, :W].copy()
        now = time.time()

        # stillness: joints in extrinsics mode, the board itself in intrinsics mode
        if args.intrinsics:
            jt, q, fresh = None, None, True
        else:
            jt, q = read_joints()
            if q is not None:
                hist.append((now, q))
            while hist and now - hist[0][0] > 0.5:
                hist.popleft()
            fresh = jt is not None and now - jt < 0.3
        joint_still = fresh and (args.intrinsics or (
            len(hist) >= 5 and np.max(np.abs(np.array([h[1] for h in hist]) - q)) < 2e-3))

        panels, stats = [], {}
        for n, c in cams.items():
            ids, pix, err, tilt = detect(frames[n], c["K"], c["D"])
            # hand-held board: wait until the corners stop moving (motion blur)
            board_still = True
            if args.intrinsics:
                moved = None
                if c["prev"] is not None and len(ids):
                    common, a, b = np.intersect1d(c["prev"][0], ids, return_indices=True)
                    if len(common) >= 4:
                        moved = np.median(np.linalg.norm(c["prev"][1][a] - pix[b], axis=1))
                c["prev"] = (ids, pix)
                if moved is not None and moved < args.still_px:
                    c["still_since"] = c["still_since"] or now
                else:
                    c["still_since"] = None
                board_still = c["still_since"] is not None and now - c["still_since"] > args.still_s
            new = len(cells(pix) - {rc for rc in zip(*np.nonzero(c["cover"]))}) if len(ids) else 0
            stats[n] = dict(n=len(ids), err=err, tilt=tilt, still=board_still, new=new)
            vis = frames[n].copy()
            draw_coverage(vis, c["cover"])
            if len(ids):
                cv2.aruco.drawDetectedCornersCharuco(vis, pix.reshape(-1, 1, 2).astype(np.float32),
                                                     ids.reshape(-1, 1))
            col = (0, 200, 0) if len(ids) >= MIN_CORNERS else (0, 0, 255)
            txt = f"{n}: {len(ids)} corners"
            if err is not None:
                txt += f"  reproj {err:.2f}px  tilt {tilt:.0f}deg" + (" STEEP" if tilt >= STEEP_DEG else "")
            if new:
                txt += f"  +{new} new cells"
            cv2.putText(vis, txt, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
            cov = 100.0 * np.mean(c["cover"] > 0)
            cv2.putText(vis, f"coverage {cov:.0f}%", (W - 260, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            scale = 1.0 if args.intrinsics else 0.5
            panels.append(cv2.resize(vis, (int(W * scale), int(H * scale))))
        view = np.hstack(panels)

        if args.intrinsics:
            s = stats[args.cam]
            still = s["still"]
            jtxt = "board still - ready" if still else "board MOVING / not seen"
        else:
            still = joint_still
            still_since = (still_since or now) if still else None
            jtxt = "joints: NO DATA (joint_tap.py running?)" if not fresh else \
                ("joints: still - ready" if still else "joints: MOVING")
        new_pose = q is not None and all(
            np.max(np.abs(q - sq)) > np.radians(args.auto_min_move_deg) for sq in sample_qs)
        if auto_on and still:
            if not new_pose:
                jtxt += " (same pose as a saved sample - move on)"
            elif stats["wrist"]["n"] < MIN_CORNERS:
                jtxt += f" (wrist sees < {MIN_CORNERS} corners - not saved)"
            else:
                jtxt += f" (auto-save in {max(0.0, args.auto_still_s - (now - still_since)):.1f}s)"
        n_steep = sum(t >= STEEP_DEG for t in steep)
        cv2.putText(view, f"{jtxt}   samples: {len(samples)}   tilt>={STEEP_DEG}: {n_steep}/8", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0) if still else (0, 0, 255), 2)
        if msg and now - msg_t < 2.5:
            cv2.putText(view, msg, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.imshow(win, view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d") and samples:
            last = samples.pop()
            for f in glob.glob(os.path.join(last, "*")):
                os.remove(f)
            os.rmdir(last)
            if not args.intrinsics:
                sample_qs.pop()
            steep.pop()
            for n, rcs in sample_cells.pop().items():
                for rc in rcs:
                    cams[n]["cover"][rc] -= 1
            msg, msg_t = f"dropped {os.path.basename(last)}", now
            print(msg)
        key_cam = args.cam if args.intrinsics else "wrist"
        auto = (auto_on and still and new_pose and stats["wrist"]["n"] >= MIN_CORNERS
                and now - still_since >= args.auto_still_s)
        if key == ord(" ") or auto:
            if not still:
                msg = "REFUSED: " + ("board moving / not seen" if args.intrinsics
                                     else "joints not fresh/still")
                msg_t = now
                print(msg)
                continue
            if args.intrinsics and stats[key_cam]["n"] < MIN_CORNERS:
                msg, msg_t = f"REFUSED: < {MIN_CORNERS} corners", now
                print(msg)
                continue
            d = os.path.join(args.out, f"sample_{len(samples):03d}")
            os.makedirs(d)
            sc = {}
            for n, c in cams.items():
                cv2.imwrite(os.path.join(d, f"{n}_left.png"), frames[n])  # lossless, raw
                ids, pix, _, _ = detect(frames[n], c["K"], c["D"])
                sc[n] = cells(pix)
                for rc in sc[n]:
                    c["cover"][rc] += 1
            json.dump({"q": None if q is None else q.tolist(), "joint_t": jt, "capture_t": now,
                       "corners": {n: s["n"] for n, s in stats.items()},
                       "pnp_median_px": {n: s["err"] for n, s in stats.items()},
                       "tilt_deg": {n: s["tilt"] for n, s in stats.items()}},
                      open(os.path.join(d, "state.json"), "w"), indent=1)
            samples.append(d)
            sample_cells.append(sc)
            steep.append(stats[key_cam]["tilt"] or 0)
            if q is not None:
                sample_qs.append(q.copy())
            msg = f"saved {os.path.basename(d)}  " + "  ".join(
                f"{n} {s['n']} corners" for n, s in stats.items())
            msg_t = now
            print(msg)

    for c in cams.values():
        c["cap"].release()
    cv2.destroyAllWindows()
    print(f"{len(samples)} samples in {args.out}")


if __name__ == "__main__":
    main()
